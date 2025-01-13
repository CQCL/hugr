//! Static evaluation of Hugrs, i.e. down to a [Value] (given inputs).
//! Some of this logic might be generalizable to cases where we cannot deduce a
//! unique [Value], and thus integrated into [constant folding](super::const_fold),
//! but the API is useful, and it seems likely that some of the transforms
//! will not be worth performing if a single-[Value] is not wanted and/or achievable.

use std::collections::HashMap;

use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::hugr::internal::HugrMutInternals;
use hugr_core::hugr::rewrite::inline_dfg::InlineDFG;
use hugr_core::hugr::rewrite::replace::{NewEdgeKind, NewEdgeSpec, Replacement};
use hugr_core::hugr::views::{DescendantsGraph, ExtractHugr, HierarchyView};
use hugr_core::ops::constant::Sum;
use hugr_core::ops::handle::FuncID;
use hugr_core::ops::{Const, DataflowParent, OpType, Value, DFG};
use hugr_core::{Direction, Hugr, HugrView, Node, PortIndex};

use crate::const_fold::ConstantFoldPass;

pub fn static_eval(mut h: Hugr, entry: Option<Node>) -> Option<Vec<Value>> {
    // TODO: allow inputs to be specified
    let mut cp = ConstantFoldPass::default();
    if let Some(ep) = entry {
        cp = cp.with_module_entry_point(ep);
    };
    let start = entry.unwrap_or(h.root());
    'reanalyse: loop {
        cp.run(&mut h).unwrap();
        eprintln!("****Constant folding produced {}", h.mermaid_string());
        loop {
            let mut need_reanalyse = false;
            let mut need_scan = false;
            for n in h.children(start).collect::<Vec<_>>() {
                match h.get_optype(n) {
                    OpType::Conditional(_) => {
                        need_scan |= conditional_to_dfg(&mut h, n).is_some();
                        // will inline on next iter. PERF: inline it now
                    }
                    OpType::DFG(_) => {
                        h.apply_rewrite(InlineDFG(n.into())).unwrap();
                        need_scan = true;
                    }
                    OpType::TailLoop(_) => {
                        need_reanalyse |= peel_tailloop(&mut h, n).is_some();
                    }
                    OpType::CallIndirect(_) => {
                        let (called, _) = h.single_linked_output(n, 0).unwrap();
                        match h.get_optype(called) {
                            OpType::LoadConstant(_) => {
                                // TODO: Inline called Hugr into DFG
                                need_reanalyse = true;
                            }
                            OpType::LoadFunction(_) => {
                                // TODO: Convert to Call
                                need_reanalyse = true
                            }
                            _ => (),
                        }
                    }
                    OpType::Call(_) => {
                        // Even if no inputs are constants (e.g. they are partial sums with multiple tags),
                        // inlining *could* (maybe) be beneficial.
                        // Note we are only doing this at the top level of the Hugr!
                        need_reanalyse |= inline_call(&mut h, n).is_some();
                    }
                    OpType::CFG(_) => {
                        // TODO: if entry node has in-edges (i.e. from other blocks) -> peel, set precision_improved=True
                        // else if entry node is exit block, elide CFG;
                        // else if entry-node predicate is constant -> move contents of entry node outside CFG, make selected successor be the new entry block, set need_scan=True
                    }
                    _ => (),
                }
            }
            eprintln!(
                "Scan of top level ({} reanalysis, {} rescan) produced {}",
                if need_reanalyse { "needs" } else { "no" },
                if need_scan { "needs" } else { "no" },
                h.mermaid_string()
            );
            h.validate().unwrap();
            if need_reanalyse {
                break;
            };
            if !need_scan {
                break 'reanalyse; // done
            };
        }
    }

    let [_, out] = h.get_io(start).unwrap();
    h.signature(out)
        .unwrap()
        .input_ports()
        .map(|p| {
            let (src_node, _) = h.single_linked_output(out, p)?;
            h.get_optype(src_node).as_load_constant()?;
            let cst = h.get_optype(h.static_source(src_node)?).as_const()?;
            Some(cst.value().clone())
        })
        .collect()
}

fn conditional_to_dfg(h: &mut impl HugrMut, cond: Node) -> Option<()> {
    let (pred, _) = h.single_linked_output(cond, 0).unwrap();
    h.get_optype(pred).as_load_constant()?;
    let cst_node = h.static_source(pred).unwrap();
    let Value::Sum(Sum { tag, values, .. }) = h.get_optype(cst_node).as_const().unwrap().value()
    else {
        panic!("Conditional input was not a Sum")
    };
    let case_node = h.children(cond).nth(*tag).unwrap();
    let signature = h.get_optype(case_node).as_case().unwrap().signature.clone();

    let mut replacement = Hugr::new(h.get_optype(h.get_parent(cond).unwrap()).clone());
    let dfg = replacement.add_node_with_parent(replacement.root(), DFG { signature });
    for (i, v) in values.iter().enumerate() {
        let cst = replacement.add_node_with_parent(replacement.root(), Const::new(v.clone()));
        replacement.connect(cst, 0, dfg, i);
    }
    let mut removal = vec![cond];
    if h.static_targets(pred).map_or(0, Iterator::count) == 1 {
        // Also remove the original (Sum) constant - we could leave this for later DCE?
        removal.push(pred);
        if h.static_targets(cst_node).map_or(0, Iterator::count) == 1 {
            removal.push(cst_node);
        }
    }
    h.apply_rewrite(Replacement {
        removal,
        replacement,
        adoptions: HashMap::from([(dfg, case_node)]),
        mu_inp: make_mu(h, cond, dfg, Direction::Incoming, true) // predicate dealt with above
            .map(|mut e| {
                if let NewEdgeKind::Value { tgt_pos, .. } = &mut e.kind {
                    *tgt_pos = (tgt_pos.index() + values.len() - 1).into();
                }
                e
            })
            .collect(),
        mu_new: vec![],
        mu_out: make_mu(h, cond, dfg, Direction::Outgoing, false).collect(),
    })
    .unwrap();
    Some(())
}

fn make_mu(
    h: &impl HugrView,
    old: Node,
    new: Node,
    dir: Direction,
    skip_first: bool,
) -> impl Iterator<Item = NewEdgeSpec> + '_ {
    let ord_port = h.get_optype(old).other_port(dir);
    h.node_ports(old, dir)
        .skip(skip_first as usize)
        .flat_map(move |pos| {
            h.linked_ports(old, pos).map(move |(node, nodp)| {
                let (src, tgt) = if dir == Direction::Outgoing {
                    (new, node)
                } else {
                    (node, new)
                };
                let kind = if Some(pos) == ord_port {
                    NewEdgeKind::Order
                } else {
                    NewEdgeKind::Value {
                        src_pos: pos.as_outgoing().or_else(|_| nodp.as_outgoing()).unwrap(),
                        tgt_pos: pos.as_incoming().or_else(|_| nodp.as_incoming()).unwrap(),
                    }
                };
                NewEdgeSpec { src, tgt, kind }
            })
        })
}

fn inline_call(h: &mut impl HugrMut, call: Node) -> Option<()> {
    let orig_func = h.static_source(call).unwrap();
    let function = DescendantsGraph::<FuncID<true>>::try_new(&h, orig_func).ok()?;
    // Ideally we'd like the following to preserve uses from within "function" of Consts outside
    // the function, but (see https://github.com/CQCL/hugr/discussions/1642) this probably won't happen at the moment - TODO XXX FIXME
    let mut func = function.extract_hugr();
    let recursive_calls = func
        .static_targets(func.root())
        .unwrap()
        .collect::<Vec<_>>();
    let new_op = OpType::from(DFG {
        signature: func
            .root_type()
            .as_func_defn()
            .unwrap()
            .inner_signature()
            .into_owned(),
    });
    let (in_ports, out_ports) = (new_op.input_count(), new_op.output_count());
    func.replace_op(func.root(), new_op).unwrap();
    func.set_num_ports(func.root(), in_ports as _, out_ports as _);
    let func_copy = h.insert_hugr(h.get_parent(call).unwrap(), func);
    for (rc, p) in recursive_calls.into_iter() {
        let call_node = func_copy.node_map.get(&rc).unwrap();
        h.disconnect(*call_node, p);
        h.connect(orig_func, 0, *call_node, p);
    }
    let func_copy = func_copy.new_root;
    let new_connections = h
        .all_linked_outputs(call)
        .filter(|(n, _)| *n != orig_func)
        .enumerate()
        .map(|(tgt_port, (src, src_port))| (src, src_port, func_copy, tgt_port.into()))
        .chain(h.node_outputs(call).flat_map(|src_port| {
            h.linked_inputs(call, src_port)
                .map(move |(tgt, tgt_port)| (func_copy, src_port, tgt, tgt_port))
        }))
        .collect::<Vec<_>>();
    h.remove_node(call);
    for (src_node, src_port, tgt_node, tgt_port) in new_connections {
        h.connect(src_node, src_port, tgt_node, tgt_port);
    }
    Some(())
}

fn peel_tailloop(h: &mut impl HugrMut, tl: Node) -> Option<()> {
    let (pred, _) = h.single_linked_output(tl, 0).unwrap();
    h.get_optype(pred).as_load_constant()?;
    // TODO: copy body of loop into DFG (dup loop, change container type, output Sum)
    // TODO: change constant into elements
    // TODO: nest existing loop inside conditional testing output of DFG.
    // Some(())
    None
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::BufReader};

    use hugr_core::std_extensions::{arithmetic::int_types::ConstInt, STD_REG};
    use hugr_core::HugrView;
    use hugr_core::{ops::Value, Hugr};
    use itertools::Itertools;

    use super::static_eval;

    #[test]
    fn recursive_fibonacci() {
        let h = Hugr::load_json(
            BufReader::new(File::open("/Users/alanlawrence/fibonacci_hugr.json").unwrap()),
            &STD_REG,
        )
        .unwrap();
        let main = h
            .children(h.root())
            .filter(|n| {
                h.get_optype(*n)
                    .as_func_defn()
                    .is_some_and(|f| f.name == "main")
            })
            .exactly_one()
            .ok()
            .unwrap();
        assert_eq!(
            static_eval(h, Some(main)),
            Some(vec![Value::extension(ConstInt::new_u(5, 8).unwrap())])
        );
    }
}
