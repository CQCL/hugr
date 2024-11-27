use std::collections::{hash_map::Entry, HashMap};

use crate::{
    extension::ExtensionRegistry,
    ops::{FuncDefn, OpTrait},
    types::{Substitution, TypeArg},
    Node,
};

use super::{HugrMut, HugrView, OpType};

pub(super) fn mono_scan(
    h: &mut impl HugrMut,
    parent: Node,
    subst_into: Option<(&Substitution<'_>, Node)>,
    cache: &mut HashMap<(Node, Vec<TypeArg>), Node>,
    reg: &ExtensionRegistry,
) {
    if subst_into.is_some() {
        // First flatten: move all FuncDefns (which do not refer to the TypeParams
        // being substituted) to be siblings of the enclosing FuncDefn.
        // TODO/PERF: we should do this only the first time we see each polymorphic FuncDefn.
        for ch in h.children(parent).collect::<Vec<_>>() {
            if h.get_optype(ch).is_func_defn() {
                // Lift the FuncDefn out
                let enclosing_poly_func = std::iter::successors(Some(ch), |n| h.get_parent(*n))
                    .find(|n| is_polymorphic_funcdefn(h, *n))
                    .unwrap();
                h.move_after_sibling(ch, enclosing_poly_func);
            }
        }
        // Since one can only call up the hierarchy,
        // that means we've moved FuncDefns before we encounter any Calls to them
    }
    let mut ch_map = HashMap::new();
    for old_ch in h.children(parent).collect::<Vec<_>>() {
        let ch_op = h.get_optype(old_ch);
        if let Some(fd) = ch_op.as_func_defn() {
            assert!(subst_into.is_none());
            if !fd.signature.params().is_empty() {
                continue;
            }
        }
        // Perform substitution, and recurse into containers (mono_scan does nothing if no children)
        let (ch, nsubst) = if let Some((subst, new_parent)) = subst_into {
            let new_ch = h.add_node_with_parent(new_parent, ch_op.clone().substitute(subst));
            ch_map.insert(old_ch, new_ch);
            (new_ch, Some((subst, new_ch)))
        } else {
            (old_ch, None)
        };
        mono_scan(h, old_ch, nsubst, cache, reg);

        let ch_op = h.get_optype(ch);
        let (type_args, instantiation) = match ch_op {
            OpType::Call(c) => (&c.type_args, &c.instantiation),
            OpType::LoadFunction(lf) => (&lf.type_args, &lf.signature),
            _ => continue,
        };
        if type_args.is_empty() {
            continue;
        };
        let fn_inp = ch_op.static_input_port().unwrap();
        let tgt = h.static_source(ch).unwrap();
        let new_tgt = match cache.entry((tgt, type_args.clone())) {
            Entry::Occupied(n) => *n.get(),
            Entry::Vacant(ve) => {
                let type_args = type_args.clone(); // Need to mutate Hugr...
                let name = name_mangle(&h.get_optype(tgt).as_func_defn().unwrap().name, &type_args);
                let mono_tgt = h.add_node_after(
                    tgt,
                    FuncDefn {
                        name,
                        signature: instantiation.clone().into(),
                    },
                );
                ve.insert(mono_tgt);
                mono_scan(
                    h,
                    tgt,
                    Some((&Substitution::new(&type_args, reg), mono_tgt)),
                    cache,
                    reg,
                );
                mono_tgt
            }
        };
        h.disconnect(ch, fn_inp);
        h.connect(new_tgt, h.num_outputs(new_tgt) - 1, ch, fn_inp);
    }
    if subst_into.is_some() {
        // TODO This won't work for 'ext' edges as cannot guarantee predecessors have been scanned.
        // Need to wait until all recursive calls made by parent have completed
        // ---> build ch_map across calls, then do this edge remapping in monomorphize()
        // (also, remove all the polymorphic FuncDefns, as now all at toplevel, and defunct).
        for &ch in ch_map.keys() {
            for inport in h.node_inputs(ch).collect::<Vec<_>>() {
                let srcs = h.linked_outputs(ch, inport).collect::<Vec<_>>();
                h.disconnect(ch, inport);
                for (src, outport) in srcs {
                    h.connect(
                        ch_map.get(&src).copied().unwrap_or(src),
                        outport,
                        ch,
                        inport,
                    );
                }
            }
        }
    }
}

fn is_polymorphic_funcdefn(h: &impl HugrView, n: Node) -> bool {
    h.get_optype(n)
        .as_func_defn()
        .is_some_and(|fd| !fd.signature.params().is_empty())
}

fn name_mangle(name: &str, type_args: &[TypeArg]) -> String {
    todo!()
}
