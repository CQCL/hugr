//! Implementation of [super::LocalizeEdgesPass]

use std::collections::{BTreeMap, HashMap};

use hugr_core::{
    Direction, HugrView, IncomingPort, Wire,
    builder::{ConditionalBuilder, Dataflow, DataflowSubContainer, HugrBuilder},
    core::HugrNode,
    hugr::hugrmut::HugrMut,
    ops::{DataflowOpTrait, OpType, Tag, TailLoop},
    types::{EdgeKind, Type, TypeRow},
};
use itertools::Itertools as _;

use super::just_types;

#[derive(Debug, Clone)]
// For each parent/container node, a map from the source Wires that need to be added
// as extra inputs to that container, to the Type of each.
pub(super) struct ExtraSourceReqs<N: HugrNode>(BTreeMap<N, BTreeMap<Wire<N>, Type>>);

impl<N: HugrNode> Default for ExtraSourceReqs<N> {
    fn default() -> Self {
        Self(BTreeMap::default())
    }
}

impl<N: HugrNode> ExtraSourceReqs<N> {
    fn insert(&mut self, node: N, source: Wire<N>, ty: Type) -> bool {
        self.0.entry(node).or_default().insert(source, ty).is_none()
    }

    fn get(&self, node: N) -> impl Iterator<Item = (&Wire<N>, &Type)> + '_ {
        self.0.get(&node).into_iter().flat_map(BTreeMap::iter)
    }

    pub fn parent_needs(&self, parent: N, source: Wire<N>) -> bool {
        self.get(parent).any(|(w, _)| *w == source)
    }

    /// Identify all required extra inputs (deals with both Dom and Ext edges).
    /// Every intermediate node in the hierarchy
    /// between the source's parent and the target needs that source.
    pub fn add_edge(
        &mut self,
        hugr: &impl HugrView<Node = N>,
        mut parent: N,
        source: Wire<N>,
        ty: Type,
    ) {
        let source_parent = hugr.get_parent(source.node()).unwrap();
        while source_parent != parent {
            debug_assert!(hugr.get_parent(parent).is_some());
            if !self.insert(parent, source, ty.clone()) {
                break;
            }
            if hugr.get_optype(parent).is_conditional() {
                // One of these we must have just done on the previous iteration
                for case in hugr.children(parent) {
                    // Full recursion unnecessary as we've just added parent:
                    self.insert(case, source, ty.clone());
                }
            }
            // this will eventually panic if source_parent is not an ancestor of target
            let parent_parent = hugr.get_parent(parent).unwrap();

            if hugr.get_optype(parent).is_dataflow_block() {
                assert!(hugr.get_optype(parent_parent).is_cfg());
                // For both Dom edges and Ext edges from outside the CFG, also add to all
                // reaching BBs (for a Dom edge, up to but not including the source BB:
                // all paths eventually come from the source since it dominates the target).
                for pred in hugr.input_neighbours(parent).collect::<Vec<_>>() {
                    self.add_edge(hugr, pred, source, ty.clone());
                }
                if Some(parent) == hugr.children(parent_parent).next() {
                    // We've just added to entry node - so carry on and add to CFG as well
                } else {
                    // Recursive calls on predecessors will have traced back to entry block
                    // (or source_parent itself if a dominating Basic Block)
                    break;
                }
            }
            parent = parent_parent;
        }
    }

    /// Threads the extra connections required throughout the Hugr
    pub(super) fn thread_hugr(&self, hugr: &mut impl HugrMut<Node = N>) {
        self.thread_node(hugr, hugr.entrypoint(), &HashMap::new())
    }

    // keys of `locals` are the *original* sources of the non-local edges, in self.0.
    fn thread_node(
        &self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        locals: &HashMap<Wire<N>, Wire<N>>,
    ) {
        if self.get(node).next().is_none() {
            // No edges incoming into this subtree, but there could still be nonlocal edges internal to it
            for ch in hugr.children(node).collect::<Vec<_>>() {
                self.thread_node(hugr, ch, &HashMap::new())
            }
            return;
        }

        let sources: Vec<(Wire<N>, Type)> = self.get(node).map(|(w, t)| (*w, t.clone())).collect();
        let src_wires: Vec<Wire<N>> = sources.iter().map(|(w, _)| *w).collect();

        // `match` must deal with everything inside the node, and update the signature (per OpType)
        let start_new_port_index = match hugr.optype_mut(node) {
            OpType::DFG(dfg) => {
                let ins = dfg.signature.input.to_mut();
                let start_new_port_index = ins.len();
                ins.extend(just_types(&sources));

                self.thread_dataflow_parent(hugr, node, start_new_port_index, sources);
                start_new_port_index
            }
            OpType::Conditional(cond) => {
                let start_new_port_index = cond.signature().input.len();
                cond.other_inputs.to_mut().extend(just_types(&sources));

                self.thread_conditional(hugr, node, sources);
                start_new_port_index
            }
            OpType::TailLoop(tail_op) => {
                vec_prepend(tail_op.just_inputs.to_mut(), just_types(&sources));
                self.thread_tailloop(hugr, node, sources);
                0
            }
            OpType::CFG(cfg) => {
                vec_prepend(cfg.signature.input.to_mut(), just_types(&sources));
                assert_eq!(
                    self.get(node).collect::<Vec<_>>(),
                    self.get(hugr.children(node).next().unwrap())
                        .collect::<Vec<_>>()
                ); // Entry node
                for bb in hugr.children(node).collect::<Vec<_>>() {
                    if hugr.get_optype(bb).is_dataflow_block() {
                        self.thread_bb(hugr, bb);
                    }
                }
                0
            }
            _ => panic!(
                "All containers handled except Module/FuncDefn or root Case/DFB, which should not have incoming nonlocal edges"
            ),
        };

        let new_dfg_ports = hugr.insert_ports(
            node,
            Direction::Incoming,
            start_new_port_index,
            src_wires.len(),
        );
        let local_srcs = src_wires.into_iter().map(|w| *locals.get(&w).unwrap_or(&w));
        for (w, tgt_port) in local_srcs.zip_eq(new_dfg_ports) {
            assert_eq!(hugr.get_parent(w.node()), hugr.get_parent(node));
            hugr.connect(w.node(), w.source(), node, tgt_port)
        }
    }

    // Add to Input node; assume container type already updated.
    fn thread_dataflow_parent(
        &self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        start_new_port_index: usize,
        srcs: Vec<(Wire<N>, Type)>,
    ) -> HashMap<Wire<N>, Wire<N>> {
        let nlocals = if srcs.is_empty() {
            HashMap::new()
        } else {
            let (srcs, tys): (Vec<_>, Vec<Type>) = srcs.into_iter().unzip();
            let [inp, _] = hugr.get_io(node).unwrap();
            let OpType::Input(in_op) = hugr.optype_mut(inp) else {
                panic!("Expected Input node")
            };
            vec_insert(in_op.types.to_mut(), tys, start_new_port_index);
            let new_outports =
                hugr.insert_ports(inp, Direction::Outgoing, start_new_port_index, srcs.len());

            srcs.into_iter()
                .zip_eq(new_outports)
                .map(|(w, p)| (w, Wire::new(inp, p)))
                .collect()
        };
        for ch in hugr.children(node).collect::<Vec<_>>() {
            for (inp, _) in hugr.in_value_types(ch).collect::<Vec<_>>() {
                if let Some((src_n, src_p)) = hugr.single_linked_output(ch, inp) {
                    if hugr.get_parent(src_n) != Some(node) {
                        hugr.disconnect(ch, inp);
                        let new_p = nlocals.get(&Wire::new(src_n, src_p)).unwrap();
                        hugr.connect(new_p.node(), new_p.source(), ch, inp);
                    }
                }
            }
            self.thread_node(hugr, ch, &nlocals);
        }
        nlocals
    }

    // Add to children (assuming conditional already updated).
    fn thread_conditional(
        &self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        srcs: Vec<(Wire<N>, Type)>,
    ) {
        for case in hugr.children(node).collect::<Vec<_>>() {
            let OpType::Case(case_op) = hugr.optype_mut(case) else {
                continue;
            };
            let ins = case_op.signature.input.to_mut();
            let start_case_port_index = ins.len();
            ins.extend(just_types(&srcs));
            self.thread_dataflow_parent(hugr, case, start_case_port_index, srcs.clone());
        }
    }

    // Add to body of loop (assume TailLoop node itself already updated).
    fn thread_tailloop(
        &self,
        hugr: &mut impl HugrMut<Node = N>,
        node: N,
        srcs: Vec<(Wire<N>, Type)>,
    ) {
        let [_, o] = hugr.get_io(node).unwrap();
        let new_sum_row_prefixes = {
            let mut v = vec![vec![]; 2];
            v[TailLoop::CONTINUE_TAG] = srcs.clone();
            v
        };
        add_control_prefixes(hugr, o, new_sum_row_prefixes);
        self.thread_dataflow_parent(hugr, node, 0, srcs);
    }

    // Add to DataflowBlock *and* inner dataflow sibling subgraph
    fn thread_bb(&self, hugr: &mut impl HugrMut<Node = N>, node: N) {
        let OpType::DataflowBlock(this_dfb) = hugr.optype_mut(node) else {
            panic!("Expected dataflow block")
        };
        let my_inputs: Vec<_> = self.get(node).map(|(w, t)| (*w, t.clone())).collect();
        vec_prepend(this_dfb.inputs.to_mut(), just_types(&my_inputs));
        let locals = self.thread_dataflow_parent(hugr, node, 0, my_inputs);
        let variant_source_prefixes: Vec<Vec<(Wire<N>, Type)>> = hugr
            .output_neighbours(node)
            .map(|succ| {
                // The wires required for each successor block, should be available in the predecessor
                self.get(succ)
                    .map(|(w, ty)| {
                        (
                            if hugr.get_parent(w.node()) == Some(node) {
                                *w
                            } else {
                                *locals.get(w).unwrap()
                            },
                            ty.clone(),
                        )
                    })
                    .collect()
            })
            .collect();
        let OpType::DataflowBlock(this_dfb) = hugr.optype_mut(node) else {
            panic!("It worked earlier!")
        };
        for (source_prefix, sum_row) in variant_source_prefixes
            .iter()
            .zip_eq(this_dfb.sum_rows.iter_mut())
        {
            vec_prepend(sum_row.to_mut(), just_types(source_prefix));
        }
        let [_, output_node] = hugr.get_io(node).unwrap();
        add_control_prefixes(hugr, output_node, variant_source_prefixes);
    }
}

/// `variant_source_prefixes` are extra wires/types to prepend onto each variant
/// (must have one element per variant of control Sum)
fn add_control_prefixes<H: HugrMut>(
    hugr: &mut H,
    output_node: H::Node,
    variant_source_prefixes: Vec<Vec<(Wire<H::Node>, Type)>>,
) {
    debug_assert!(hugr.get_optype(output_node).is_output()); // Just to fail fast
    let parent = hugr.get_parent(output_node).unwrap();
    let mut needed_sources = BTreeMap::new();
    let (cond, new_control_type) = {
        let Some(EdgeKind::Value(control_type)) = hugr
            .get_optype(output_node)
            .port_kind(IncomingPort::from(0))
        else {
            panic!("impossible")
        };
        let Some(sum_type) = control_type.as_sum() else {
            panic!("impossible")
        };

        let mut type_for_source = |source: &(Wire<H::Node>, Type)| {
            let (w, t) = source;
            let replaced = needed_sources.insert(*w, (*w, t.clone()));
            debug_assert!(!replaced.is_some_and(|x| x != (*w, t.clone())));
            t.clone()
        };
        let old_sum_rows: Vec<TypeRow> = sum_type
            .variants()
            .map(|x| x.clone().try_into().unwrap())
            .collect_vec();
        let new_sum_rows: Vec<TypeRow> =
            itertools::zip_eq(variant_source_prefixes.iter(), old_sum_rows.iter())
                .map(|(new_sources, old_tys)| {
                    new_sources
                        .iter()
                        .map(&mut type_for_source)
                        .chain(old_tys.iter().cloned())
                        .collect_vec()
                        .into()
                })
                .collect_vec();

        let new_control_type = Type::new_sum(new_sum_rows.clone());
        let mut cond = ConditionalBuilder::new(
            old_sum_rows.clone(),
            just_types(needed_sources.values()).collect_vec(),
            new_control_type.clone(),
        )
        .unwrap();
        for (i, new_sources) in variant_source_prefixes.into_iter().enumerate() {
            let mut case = cond.case_builder(i).unwrap();
            let case_inputs = case.input_wires().collect_vec();
            let mut args = new_sources
                .into_iter()
                .map(|(s, _ty)| {
                    case_inputs[old_sum_rows[i].len()
                        + needed_sources
                            .iter()
                            .find_position(|(w, _)| **w == s)
                            .unwrap()
                            .0]
                })
                .collect_vec();
            args.extend(&case_inputs[..old_sum_rows[i].len()]);
            let case_outputs = case
                .add_dataflow_op(Tag::new(i, new_sum_rows.clone()), args)
                .unwrap()
                .outputs();
            case.finish_with_outputs(case_outputs).unwrap();
        }
        (cond.finish_hugr().unwrap(), new_control_type)
    };
    let cond_node = hugr.insert_hugr(parent, cond).inserted_entrypoint;
    let (old_output_source_node, old_output_source_port) =
        hugr.single_linked_output(output_node, 0).unwrap();
    debug_assert_eq!(hugr.get_parent(old_output_source_node).unwrap(), parent);
    hugr.connect(old_output_source_node, old_output_source_port, cond_node, 0);
    for (i, &(w, _)) in needed_sources.values().enumerate() {
        hugr.connect(w.node(), w.source(), cond_node, i + 1);
    }
    hugr.disconnect(output_node, IncomingPort::from(0));
    hugr.connect(cond_node, 0, output_node, 0);
    let OpType::Output(output) = hugr.optype_mut(output_node) else {
        panic!("impossible")
    };
    output.types.to_mut()[0] = new_control_type;
}

fn vec_prepend<T>(v: &mut Vec<T>, ts: impl IntoIterator<Item = T>) {
    vec_insert(v, ts, 0)
}

fn vec_insert<T>(v: &mut Vec<T>, ts: impl IntoIterator<Item = T>, index: usize) {
    let mut old_v_iter = std::mem::take(v).into_iter();
    v.extend(old_v_iter.by_ref().take(index).chain(ts));
    v.extend(old_v_iter);
}

#[cfg(test)]
mod test {
    use super::vec_insert;

    #[test]
    fn vec_insert0() {
        let mut v = vec![5, 7, 9];
        vec_insert(&mut v, [1, 2], 0);
        assert_eq!(v, [1, 2, 5, 7, 9]);
    }

    #[test]
    fn vec_insert1() {
        let mut v = vec![5, 7, 9];
        vec_insert(&mut v, [1, 2], 1);
        assert_eq!(v, [5, 1, 2, 7, 9]);
    }
}
