use std::collections::{hash_map::Entry, HashMap};

use crate::{
    extension::ExtensionRegistry,
    ops::{FuncDefn, OpTrait},
    types::{Signature, Substitution, TypeArg},
    Node,
};

use super::{Hugr, HugrMut, HugrView, OpType};

pub fn monomorphize(mut h: Hugr, reg: &ExtensionRegistry) -> Hugr {
    let root = h.root(); // I.e. "all monomorphic funcs" for Module-Rooted Hugrs...right?
    mono_scan(&mut h, root, None, &mut HashMap::new(), reg);
    h
}

struct Instantiating<'a> {
    poly_func: Node,
    subst: &'a Substitution<'a>,
    target_container: Node,
    node_map: &'a mut HashMap<Node, Node>,
}

type Instantiations = HashMap<(Node, Vec<TypeArg>), Node>;

fn mono_scan(
    h: &mut impl HugrMut,
    parent: Node,
    mut subst_into: Option<&mut Instantiating>,
    cache: &mut Instantiations,
    reg: &ExtensionRegistry,
) {
    if let Some(Instantiating { poly_func, .. }) = subst_into {
        // First flatten: move all FuncDefns (which do not refer to the TypeParams
        // being substituted) to be siblings of the enclosing FuncDefn.
        // TODO/PERF: we should do this only the first time we see each polymorphic FuncDefn.
        for ch in h.children(parent).collect::<Vec<_>>() {
            if h.get_optype(ch).is_func_defn() {
                // Lift the FuncDefn out
                h.move_after_sibling(ch, *poly_func);
            }
        }
        // Since one can only call up the hierarchy,
        // that means we've moved FuncDefns before we encounter any Calls to them
    }
    for old_ch in h.children(parent).collect::<Vec<_>>() {
        let ch_op = h.get_optype(old_ch);
        if let Some(fd) = ch_op.as_func_defn() {
            assert!(subst_into.is_none());
            if !fd.signature.params().is_empty() {
                continue;
            }
        }
        // Perform substitution, and recurse into containers (mono_scan does nothing if no children)
        let ch = if let Some(ref mut inst) = subst_into {
            let new_ch =
                h.add_node_with_parent(inst.target_container, ch_op.clone().substitute(inst.subst));
            inst.node_map.insert(old_ch, new_ch);
            mono_scan(
                h,
                old_ch,
                Some(&mut Instantiating {
                    target_container: new_ch,
                    node_map: inst.node_map,   // &mut ref, so borrow
                    ..**inst
                }),
                cache,
                reg,
            );
            new_ch
        } else {
            mono_scan(h, old_ch, None, cache, reg);
            old_ch
        };

        let ch_op = h.get_optype(ch);
        let (type_args, mono_sig) = match ch_op {
            OpType::Call(c) => (&c.type_args, &c.instantiation),
            OpType::LoadFunction(lf) => (&lf.type_args, &lf.signature),
            _ => continue,
        };
        if type_args.is_empty() {
            continue;
        };
        let fn_inp = ch_op.static_input_port().unwrap();
        let tgt = h.static_source(ch).unwrap();
        let new_tgt = instantiate(h, tgt, type_args.clone(), mono_sig.clone(), cache, reg);
        h.disconnect(ch, fn_inp);
        h.connect(new_tgt, h.num_outputs(new_tgt) - 1, ch, fn_inp);
    }
}

fn instantiate(
    h: &mut impl HugrMut,
    poly_func: Node,
    type_args: Vec<TypeArg>,
    mono_sig: Signature,
    cache: &mut Instantiations,
    reg: &ExtensionRegistry,
) -> Node {
    let ve = match cache.entry((poly_func, type_args.clone())) {
        Entry::Occupied(n) => return *n.get(),
        Entry::Vacant(ve) => ve,
    };

    let name = name_mangle(
        &h.get_optype(poly_func).as_func_defn().unwrap().name,
        &type_args,
    );
    let mono_tgt = h.add_node_after(
        poly_func,
        FuncDefn {
            name,
            signature: mono_sig.into(),
        },
    );
    // Insert BEFORE we scan (in case of recursion), hence we cannot use Entry::or_insert
    ve.insert(mono_tgt);
    // Now make the instantiation
    let mut node_map = HashMap::new();
    mono_scan(
        h,
        poly_func,
        Some(&mut Instantiating {
            poly_func,
            subst: &Substitution::new(&type_args, reg),
            target_container: mono_tgt,
            node_map: &mut node_map,
        }),
        cache,
        reg,
    );
    // Copy edges...we have built a node_map for every node in the function.
    // Note we could avoid building the "large" map (smaller than the Hugr we've just created)
    // by doing this during recursion, but we'd need to be careful with nonlocal edges -
    // 'ext' edges by copying every node before recursing on any of them,
    // 'dom' edges would *also* require recursing in dominator-tree preorder.
    for &ch in node_map.keys() {
        for inport in h.node_inputs(ch).collect::<Vec<_>>() {
            let srcs = h.linked_outputs(ch, inport).collect::<Vec<_>>();
            // Sources could be a mixture of within this polymorphic FuncDefn, and Static edges from outside
            h.disconnect(ch, inport);
            for (src, outport) in srcs {
                h.connect(
                    node_map.get(&src).copied().unwrap_or(src),
                    outport,
                    ch,
                    inport,
                );
            }
        }
    }

    mono_tgt
}

fn name_mangle(name: &str, type_args: &[TypeArg]) -> String {
    todo!()
}
