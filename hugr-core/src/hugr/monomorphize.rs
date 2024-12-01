use std::collections::{hash_map::Entry, HashMap};

use crate::{
    extension::ExtensionRegistry,
    ops::{Call, FuncDefn, OpTrait},
    types::{Signature, Substitution, TypeArg},
    Node,
};

use super::{internal::HugrMutInternals, Hugr, HugrMut, HugrView, OpType};

pub fn monomorphize(mut h: Hugr, reg: &ExtensionRegistry) -> Hugr {
    let root = h.root(); // I.e. "all monomorphic funcs" for Module-Rooted Hugrs...right?
    mono_scan(&mut h, root, None, &mut HashMap::new(), reg);
    h
}

pub fn remove_polyfuncs(mut h: Hugr) -> Hugr {
    let mut pfs_to_delete = Vec::new();
    let mut to_scan = Vec::from_iter(h.children(h.root()));
    while let Some(n) = to_scan.pop() {
        if h.get_optype(n)
            .as_func_defn()
            .is_some_and(|fd| !fd.signature.params().is_empty())
        {
            pfs_to_delete.push(n)
        } else {
            to_scan.extend(h.children(n));
        }
    }
    for n in pfs_to_delete {
        h.remove_subtree(n);
    }
    h
}

struct Instantiating<'a> {
    subst: &'a Substitution<'a>,
    target_container: Node,
    node_map: &'a mut HashMap<Node, Node>,
}

type Instantiations = HashMap<Node, HashMap<Vec<TypeArg>, Node>>;

fn mono_scan(
    h: &mut Hugr,
    parent: Node,
    mut subst_into: Option<&mut Instantiating>,
    cache: &mut Instantiations,
    reg: &ExtensionRegistry,
) {
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
            let mut inst = Instantiating {
                target_container: new_ch,
                node_map: inst.node_map,
                ..**inst
            };
            mono_scan(h, old_ch, Some(&mut inst), cache, reg);
            new_ch
        } else {
            mono_scan(h, old_ch, None, cache, reg);
            old_ch
        };

        // Now instantiate the target of any Call/LoadFunction to a polymorphic function...
        let ch_op = h.get_optype(ch);
        let (type_args, mono_sig) = match ch_op {
            OpType::Call(c) => (&c.type_args, c.instantiation.clone()),
            OpType::LoadFunction(lf) => (&lf.type_args, lf.signature.clone()),
            _ => continue,
        };
        if type_args.is_empty() {
            continue;
        };
        let fn_inp = ch_op.static_input_port().unwrap();
        let tgt = h.static_source(old_ch).unwrap(); // Use old_ch as edges not copied yet
        let new_tgt = instantiate(h, tgt, type_args.clone(), mono_sig.clone(), cache, reg);
        let fn_out = h.get_optype(new_tgt).static_output_port().unwrap();
        h.disconnect(ch, fn_inp);
        h.connect(new_tgt, fn_out, ch, fn_inp);
        *h.op_types.get_mut(ch.pg_index()) =
            Call::try_new(mono_sig.into(), vec![], &reg).unwrap().into();
    }
}

fn instantiate(
    h: &mut Hugr,
    poly_func: Node,
    type_args: Vec<TypeArg>,
    mono_sig: Signature,
    cache: &mut Instantiations,
    reg: &ExtensionRegistry,
) -> Node {
    let for_func = cache.entry(poly_func).or_insert_with(|| {
        // First time we've instantiated poly_func. Lift any nested FuncDefn's out to the same level.
        let outer_name = h.get_optype(poly_func).as_func_defn().unwrap().name.clone();
        let mut to_scan = Vec::from_iter(h.children(poly_func));
        while let Some(n) = to_scan.pop() {
            if let OpType::FuncDefn(fd) = h.op_types.get_mut(n.pg_index()) {
                fd.name = mangle_inner_func(&outer_name, &fd.name);
                h.move_after_sibling(n, poly_func);
            } else {
                to_scan.extend(h.children(n))
            }
        }
        HashMap::new()
    });

    let ve = match for_func.entry(type_args.clone()) {
        Entry::Occupied(n) => return *n.get(),
        Entry::Vacant(ve) => ve,
    };

    let name = mangle_name(
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
    let mut inst = Instantiating {
        subst: &Substitution::new(&type_args, reg),
        target_container: mono_tgt,
        node_map: &mut node_map,
    };
    mono_scan(h, poly_func, Some(&mut inst), cache, reg);
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

fn mangle_name(name: &str, type_args: &[TypeArg]) -> String {
    todo!()
}

fn mangle_inner_func(outer_name: &str, inner_name: &str) -> String {
    todo!()
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rstest::rstest;

    use crate::builder::BuildHandle;
    use crate::builder::{test::simple_dfg_hugr, Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder};
    use crate::extension::{prelude::{PRELUDE, USIZE_T}, ExtensionRegistry, EMPTY_REG};
    use crate::ops::handle::FuncID;
    use crate::std_extensions::collections::{self, list_type, ListOp, ListValue};
    use crate::types::{PolyFuncType, Signature, Type, TypeBound};
    use crate::{Hugr, HugrView};

    use super::{monomorphize, mangle_name};


    #[rstest]
    fn test_null(simple_dfg_hugr: Hugr) {
        let mono = monomorphize(simple_dfg_hugr.clone(), &EMPTY_REG);
        assert_eq!(simple_dfg_hugr, mono);
    }

    fn add_singleton(ctr: &mut impl Container, reg: &ExtensionRegistry) -> BuildHandle<FuncID<true>> {
        let elem_ty = Type::new_var_use(0, TypeBound::Any);
        let mut fb = ctr.define_function("singleton", PolyFuncType::new([TypeBound::Any.into()], Signature::new(
        elem_ty.clone(), list_type(elem_ty.clone())
        ))).unwrap();
        let [elem] = fb.input_wires_arr();
        let empty = fb.add_load_value(ListValue::new_empty(elem_ty.clone()));
        let push = fb.add_dataflow_op(ListOp::push.with_type(elem_ty).to_extension_op(&reg).unwrap(), [empty, elem]).unwrap();
        fb.finish_with_outputs(push.outputs()).unwrap()
    }

    #[test]
    fn test_module() {
        let reg = ExtensionRegistry::try_new([collections::EXTENSION.to_owned(), PRELUDE.to_owned()]).unwrap();
        let mut mb = ModuleBuilder::new();
        let sing = add_singleton(&mut mb, &reg);
        let dub = {
            let elem_ty = Type::new_var_use(0, TypeBound::Copyable);
            let mut fb = mb.define_function("doubled", PolyFuncType::new([TypeBound::Copyable.into()], Signature::new(
                elem_ty.clone(), list_type(elem_ty.clone())
            ))).unwrap();
            let [elem] = fb.input_wires_arr();
            let [sing] = fb.call(sing.handle(), &[elem_ty.clone().into()], [elem], &reg).unwrap().outputs_arr();
            let push = fb.add_dataflow_op(ListOp::push.with_type(elem_ty).to_extension_op(&reg).unwrap(), [sing, elem]).unwrap();
            fb.finish_with_outputs(push.outputs()).unwrap()
        };
        {
            let mut fb = mb.define_function("main", Signature::new(
                USIZE_T,
                vec![
                    list_type(USIZE_T),
                    list_type(list_type(USIZE_T))
                ])).unwrap();
            let [elem] = fb.input_wires_arr();
            let [two] = fb.call(dub.handle(), &[USIZE_T.into()], [elem], &reg).unwrap().outputs_arr();
            let sing = fb.call(sing.handle(), &[USIZE_T.into()], [elem], &reg).unwrap();
            let [two_by_one] = fb.call(dub.handle(), &[list_type(USIZE_T).into()], sing.outputs(), &reg).unwrap().outputs_arr();
            fb.finish_with_outputs([two, two_by_one]).unwrap();
        }
        let hugr = mb.finish_hugr(&reg).unwrap();
        assert_eq!(hugr.nodes().filter(|n| hugr.get_optype(*n).is_func_defn()).count(), 3);
        let mono_hugr = monomorphize(hugr, &reg);
        mono_hugr.validate(&reg).unwrap();
        let funcs = mono_hugr.nodes().filter_map(|n| mono_hugr.get_optype(n).as_func_defn()).collect_vec();
        let expected_mangled_names = [mangle_name("singleton", &[USIZE_T.into()]),
        mangle_name("doubled", &[USIZE_T.into()]), mangle_name("singleton", &[list_type(USIZE_T).into()]),
    mangle_name("doubled", &[list_type(USIZE_T).into()])];

        assert_eq!(funcs.iter().map(|fd|&fd.name).sorted().collect_vec(),
            ["main", "singleton", "doubled"].into_iter().chain(expected_mangled_names.iter().map(String::as_str)).sorted().collect_vec());
        for n in expected_mangled_names {
            let mono_fn = funcs.iter().find(|fd|fd.name == n).unwrap();
            assert!(mono_fn.signature.params().is_empty());
        }
    }

    #[test]
    fn test_flattening() {

    }

    #[test]
    fn test_recursive() {
        
    }

    #[test]
    fn test_root_polyfunc() {

    }
}