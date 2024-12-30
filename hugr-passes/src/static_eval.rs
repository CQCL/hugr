//! Static evaluation of Hugrs, i.e. down to a [Value] (given inputs).
//! Some of this logic might be generalizable to cases where we cannot deduce a
//! unique [Value], and thus integrated into [constant folding](super::const_fold),
//! but the API is useful, and it seems likely that some of the transforms
//! will not be worth performing if a single-[Value] is not wanted and/or achievable.

use std::collections::HashMap;

use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::hugr::rewrite::inline_dfg::InlineDFG;
use hugr_core::hugr::rewrite::replace::{NewEdgeKind, NewEdgeSpec, Replacement};
use hugr_core::ops::constant::Sum;
use hugr_core::ops::{Const, OpType, Value, DFG};
use hugr_core::{Hugr, HugrView};

use super::const_fold::constant_fold_pass;

pub fn static_eval(mut h: Hugr) -> Option<Vec<Value>> {
    // TODO: allow inputs to be specified
    loop {
        constant_fold_pass(&mut h);
        let mut precision_improved = false;
        loop {
            let mut need_scan = false;
            for n in h.children(h.root()).collect::<Vec<_>>() {
                match h.get_optype(n) {
                    OpType::Conditional(_) => {
                        let (pred, _) = h.single_linked_output(n, 0).unwrap();
                        if h.get_optype(pred).is_load_constant() {
                            let cst_node = h.static_source(pred).unwrap();
                            let Some(Value::Sum(Sum { tag, values, .. })) =
                                h.get_optype(cst_node).as_const().map(Const::value)
                            else {
                                panic!("Conditional input was not a Sum")
                            };
                            let case_node = h.children(n).nth(*tag).unwrap();
                            let signature =
                                h.get_optype(case_node).as_case().unwrap().signature.clone();

                            let mut replacement = Hugr::new(h.get_optype(h.root()).clone());
                            let dfg = replacement
                                .add_node_with_parent(replacement.root(), DFG { signature });
                            for (i, v) in values.iter().enumerate() {
                                let cst = replacement.add_node_with_parent(
                                    replacement.root(),
                                    Const::new(v.clone()),
                                );
                                replacement.connect(cst, 0, dfg, i);
                            }
                            let mut removal = vec![n];
                            if h.static_targets(cst_node).map_or(0, Iterator::count) == 1 {
                                // Also remove the original (Sum) constant - we could leave this for later DCE?
                                removal.push(cst_node);
                            }
                            h.apply_rewrite(Replacement {
                                removal,
                                replacement,
                                adoptions: HashMap::from([(dfg, case_node)]),
                                mu_inp: h
                                    .all_linked_outputs(n)
                                    .skip(1)
                                    .enumerate()
                                    .map(|(i, (src, src_pos))| NewEdgeSpec {
                                        src,
                                        tgt: dfg,
                                        kind: NewEdgeKind::Value {
                                            src_pos,
                                            tgt_pos: (i + values.len()).into(),
                                        },
                                    })
                                    .collect(),
                                mu_new: vec![],
                                mu_out: h
                                    .node_outputs(n)
                                    .flat_map(|src_pos| {
                                        h.linked_inputs(n, src_pos).map(move |(tgt, tgt_pos)| {
                                            NewEdgeSpec {
                                                src: dfg,
                                                tgt,
                                                kind: NewEdgeKind::Value { src_pos, tgt_pos },
                                            }
                                        })
                                    })
                                    .collect(),
                            })
                            .unwrap();
                            // TODO: replace conditional with DFG containing that case,
                            // and constant with element constants.
                            need_scan = true; // will inline on next iter. PERF: inline it now
                        }
                    }
                    OpType::DFG(_) => {
                        h.apply_rewrite(InlineDFG(n.into())).unwrap();
                        need_scan = true;
                    }
                    OpType::TailLoop(_) => {
                        let (pred, _) = h.single_linked_output(n, 0).unwrap();
                        if h.get_optype(pred).is_load_constant() {
                            // TODO: copy body of loop into DFG (dup loop, change container type, output Sum)
                            // TODO: change constant into elements
                            // TODO: nest existing loop inside conditional testing output of DFG.
                            precision_improved = true;
                        }
                    }
                    OpType::CallIndirect(_) => {
                        let (called, _) = h.single_linked_output(n, 0).unwrap();
                        match h.get_optype(called) {
                            OpType::LoadConstant(_) => {
                                // TODO: Inline called Hugr into DFG
                                precision_improved = true;
                            }
                            OpType::LoadFunction(_) => {
                                // TODO: Convert to Call
                                precision_improved = true
                            }
                            _ => (),
                        }
                    }
                    OpType::Call(_) => {
                        // Even if no inputs are constants (e.g. they are partial sums with multiple tags),
                        // this *could* (maybe) be beneficial.
                        // Note we are only doing this at the top level of the Hugr!
                        // TODO: Copy body of called-function into DFG, wire in remaining inputs (unchanged).
                        precision_improved = true;
                    }
                    OpType::CFG(_) => {
                        // TODO: if entry node has in-edges (i.e. from other blocks) -> peel, set precision_improved=True
                        // else if entry node is exit block, elide CFG;
                        // else if entry-node predicate is constant -> move contents of entry node outside CFG, make selected successor be the new entry block, set need_scan=True
                    }
                    _ => (),
                }
            }
            if precision_improved || !need_scan {
                break;
            };
        }
        if !precision_improved {
            break;
        };
    }

    let [_, out] = h.get_io(h.root()).unwrap();
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
