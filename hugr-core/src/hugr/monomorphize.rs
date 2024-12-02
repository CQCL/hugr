use std::collections::{hash_map::Entry, HashMap};

use crate::{
    extension::ExtensionRegistry,
    ops::{Call, FuncDefn, OpTrait},
    types::{Signature, Substitution, TypeArg},
    Node,
};

use super::{internal::HugrMutInternals, Hugr, HugrMut, HugrView, OpType};

/// Replaces calls to polymorphic functions with calls to new monomorphic
/// instantiations of the polymorphic ones.
///
/// If the Hugr is [Module](OpType::Module)-rooted,
/// * then the original polymorphic [FuncDefn]s are left untouched (including Calls inside them)
///     - call [remove_polyfuncs] when no other Hugr will be linked in that might instantiate these
/// * else, the originals are removed (they are invisible from outside the Hugr).
pub fn monomorphize(mut h: Hugr, reg: &ExtensionRegistry) -> Hugr {
    let root = h.root();
    // If the root is a polymorphic function, then there are no external calls, so nothing to do
    if !is_polymorphic_funcdefn(h.get_optype(root)) {
        mono_scan(&mut h, root, None, &mut HashMap::new(), reg);
        if !h.get_optype(root).is_module() {
            return remove_polyfuncs(h);
        }
    }
    h
}

/// Removes any polymorphic [FuncDefn]s from the Hugr. Note that if these have
/// calls from *monomorphic* code, this will make the Hugr invalid (call [monomorphize]
/// first).
pub fn remove_polyfuncs(mut h: Hugr) -> Hugr {
    let mut pfs_to_delete = Vec::new();
    let mut to_scan = Vec::from_iter(h.children(h.root()));
    while let Some(n) = to_scan.pop() {
        if is_polymorphic_funcdefn(h.get_optype(n)) {
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

fn is_polymorphic_funcdefn(t: &OpType) -> bool {
    t.as_func_defn()
        .is_some_and(|fd| !fd.signature.params().is_empty())
}

struct Instantiating<'a> {
    subst: &'a Substitution<'a>,
    target_container: Node,
    node_map: &'a mut HashMap<Node, Node>,
}

type Instantiations = HashMap<Node, HashMap<Vec<TypeArg>, Node>>;

/// Scans a subtree for polymorphic calls and monomorphizes them by instantiating the
/// called functions (if instantations not already in `cache`).
/// Optionally copies the subtree into a new location whilst applying a substitution.
/// The subtree should be monomorphic after the substitution (if provided) has been applied.
fn mono_scan(
    h: &mut Hugr,
    parent: Node,
    mut subst_into: Option<&mut Instantiating>,
    cache: &mut Instantiations,
    reg: &ExtensionRegistry,
) {
    for old_ch in h.children(parent).collect::<Vec<_>>() {
        let ch_op = h.get_optype(old_ch);
        debug_assert!(!ch_op.is_func_defn() || subst_into.is_none()); // If substituting, should have flattened already
        if is_polymorphic_funcdefn(ch_op) {
            continue;
        };
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
        h.disconnect(ch, fn_inp); // No-op if copying+substituting
        h.connect(new_tgt, fn_out, ch, fn_inp);

        *h.op_types.get_mut(ch.pg_index()) =
            Call::try_new(mono_sig.into(), vec![], reg).unwrap().into();
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
    for (&old_ch, &new_ch) in node_map.iter() {
        for inport in h.node_inputs(old_ch).collect::<Vec<_>>() {
            // Edges from monomorphized functions to their calls already added during mono_scan()
            // as these depend not just on the original FuncDefn but also the TypeArgs
            if h.linked_outputs(new_ch, inport).next().is_some() {
                continue;
            };
            let srcs = h.linked_outputs(old_ch, inport).collect::<Vec<_>>();
            for (src, outport) in srcs {
                // Sources could be a mixture of within this polymorphic FuncDefn, and Static edges from outside
                h.connect(
                    node_map.get(&src).copied().unwrap_or(src),
                    outport,
                    new_ch,
                    inport,
                );
            }
        }
    }

    mono_tgt
}

fn mangle_name(name: &str, type_args: &[TypeArg]) -> String {
    let s = format!("__{name}_{type_args:?}");
    s.replace(['[', ']', '{', '}', ' '], "_")
}

fn mangle_inner_func(outer_name: &str, inner_name: &str) -> String {
    format!("$_{outer_name}_$_{inner_name}")
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use itertools::Itertools;
    use rstest::rstest;

    use crate::builder::test::simple_dfg_hugr;
    use crate::builder::{
        BuildHandle, Container, Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder,
        HugrBuilder, ModuleBuilder,
    };
    use crate::extension::prelude::{usize_t, ConstUsize, UnpackTuple};
    use crate::extension::{ExtensionRegistry, EMPTY_REG, PRELUDE, PRELUDE_REGISTRY};
    use crate::hugr::monomorphize::mangle_inner_func;
    use crate::ops::handle::FuncID;
    use crate::ops::Tag;
    use crate::std_extensions::arithmetic::int_types::{self, INT_TYPES};
    use crate::types::{PolyFuncType, Signature, Type, TypeBound};
    use crate::{type_row, Hugr, HugrView};

    use super::{mangle_name, monomorphize, remove_polyfuncs};

    #[rstest]
    fn test_null(simple_dfg_hugr: Hugr) {
        let mono = monomorphize(simple_dfg_hugr.clone(), &EMPTY_REG);
        assert_eq!(simple_dfg_hugr, mono);
    }

    fn pair_type(ty: Type) -> Type {
        Type::new_tuple(vec![ty.clone(), ty])
    }

    fn triple_type(ty: Type) -> Type {
        Type::new_tuple(vec![ty.clone(), ty.clone(), ty])
    }

    fn add_double(ctr: &mut impl Container) -> BuildHandle<FuncID<true>> {
        let elem_ty = Type::new_var_use(0, TypeBound::Copyable);
        let pfty = PolyFuncType::new(
            [TypeBound::Copyable.into()],
            Signature::new(elem_ty.clone(), pair_type(elem_ty.clone())),
        );
        let mut fb = ctr.define_function("double", pfty).unwrap();
        let [elem] = fb.input_wires_arr();
        let tag = Tag::new(0, vec![vec![elem_ty; 2].into()]);
        let tag = fb.add_dataflow_op(tag, [elem, elem]).unwrap();
        fb.finish_with_outputs(tag.outputs()).unwrap()
    }

    #[test]
    fn test_module() -> Result<(), Box<dyn std::error::Error>> {
        let mut mb = ModuleBuilder::new();
        let db = add_double(&mut mb);
        let tr = {
            let tv0 = || Type::new_var_use(0, TypeBound::Copyable);
            let pfty = PolyFuncType::new(
                [TypeBound::Copyable.into()],
                Signature::new(tv0(), Type::new_tuple(vec![tv0(); 3])),
            );
            let mut fb = mb.define_function("triple", pfty)?;
            let [elem] = fb.input_wires_arr();
            let pair = fb.call(db.handle(), &[tv0().into()], [elem], &PRELUDE_REGISTRY)?;

            let [elem1, elem2] = fb
                .add_dataflow_op(UnpackTuple(vec![tv0(); 2].into()), pair.outputs())?
                .outputs_arr();
            let tag = Tag::new(0, vec![vec![tv0(); 3].into()]);
            let trip = fb.add_dataflow_op(tag, [elem1, elem2, elem])?;
            fb.finish_with_outputs(trip.outputs())?
        };
        {
            let sig = Signature::new(
                usize_t(),
                vec![triple_type(usize_t()), triple_type(pair_type(usize_t()))],
            );
            let mut fb = mb.define_function("main", sig)?;
            let [elem] = fb.input_wires_arr();
            let [res1] = fb
                .call(tr.handle(), &[usize_t().into()], [elem], &PRELUDE_REGISTRY)?
                .outputs_arr();
            let pair = fb.call(db.handle(), &[usize_t().into()], [elem], &PRELUDE_REGISTRY)?;
            let pty = pair_type(usize_t()).into();
            let [res2] = fb
                .call(tr.handle(), &[pty], pair.outputs(), &PRELUDE_REGISTRY)?
                .outputs_arr();
            fb.finish_with_outputs([res1, res2])?;
        }
        let hugr = mb.finish_hugr(&PRELUDE_REGISTRY)?;
        assert_eq!(
            hugr.nodes()
                .filter(|n| hugr.get_optype(*n).is_func_defn())
                .count(),
            3
        );
        let mono = monomorphize(hugr, &PRELUDE_REGISTRY);
        mono.validate(&PRELUDE_REGISTRY)?;

        let mut funcs = mono
            .nodes()
            .filter_map(|n| mono.get_optype(n).as_func_defn().map(|fd| (&fd.name, fd)))
            .collect::<HashMap<_, _>>();
        let expected_mangled_names = [
            mangle_name("double", &[usize_t().into()]),
            mangle_name("triple", &[usize_t().into()]),
            mangle_name("double", &[pair_type(usize_t()).into()]),
            mangle_name("triple", &[pair_type(usize_t()).into()]),
        ];

        for n in expected_mangled_names.iter() {
            let mono_fn = funcs.remove(n).unwrap();
            assert!((*mono_fn).signature.params().is_empty());
        }

        assert_eq!(
            funcs.keys().sorted().collect_vec(),
            ["double", "main", "triple"].iter().collect_vec()
        );

        assert_eq!(monomorphize(mono.clone(), &PRELUDE_REGISTRY), mono); // Idempotent

        let nopoly = remove_polyfuncs(mono);
        let mut funcs = nopoly
            .nodes()
            .filter_map(|n| nopoly.get_optype(n).as_func_defn().map(|fd| (&fd.name, fd)))
            .collect::<HashMap<_, _>>();

        assert!(funcs.values().all(|fd| (*fd).signature.params().is_empty()));
        for n in expected_mangled_names {
            assert!(funcs.remove(&n).is_some());
        }
        assert_eq!(funcs.keys().collect_vec(), vec![&"main"]);
        Ok(())
    }

    #[test]
    fn test_flattening() -> Result<(), Box<dyn std::error::Error>> {
        //pf1 contains pf2 contains mono_func -> pf1<a> and pf1<b> share pf2's and they share mono_func

        let reg =
            ExtensionRegistry::try_new([int_types::EXTENSION.to_owned(), PRELUDE.to_owned()])?;
        let tv0 = || Type::new_var_use(0, TypeBound::Any);
        let pf_any = |sig: Signature| PolyFuncType::new([TypeBound::Any.into()], sig);
        let ity = || INT_TYPES[3].clone();

        let mut outer = FunctionBuilder::new("mainish", Signature::new(ity(), usize_t()))?;
        let sig = pf_any(Signature::new(tv0(), vec![tv0(), usize_t(), usize_t()]));
        let mut pf1 = outer.define_function("pf1", sig)?;

        let sig = pf_any(Signature::new(tv0(), vec![tv0(), usize_t()]));
        let mut pf2 = pf1.define_function("pf2", sig)?;

        let mono_func = {
            let mut fb = pf2.define_function("get_usz", Signature::new(type_row![], usize_t()))?;
            let cst0 = fb.add_load_value(ConstUsize::new(1));
            fb.finish_with_outputs([cst0])?
        };
        let pf2 = {
            let [inw] = pf2.input_wires_arr();
            let [usz] = pf2.call(mono_func.handle(), &[], [], &reg)?.outputs_arr();
            pf2.finish_with_outputs([inw, usz])?
        };
        // pf1: Two calls to pf2, one depending on pf1's TypeArg, the other not
        let [a, u] = pf1
            .call(pf2.handle(), &[tv0().into()], pf1.input_wires(), &reg)?
            .outputs_arr();
        let [u1, u2] = pf1
            .call(pf2.handle(), &[usize_t().into()], [u], &reg)?
            .outputs_arr();
        let pf1 = pf1.finish_with_outputs([a, u1, u2])?;
        // Outer: two calls to pf1 with different TypeArgs
        let [_, u, _] = outer
            .call(pf1.handle(), &[ity().into()], outer.input_wires(), &reg)?
            .outputs_arr();
        let [_, u, _] = outer
            .call(pf1.handle(), &[usize_t().into()], [u], &reg)?
            .outputs_arr();
        let hugr = outer.finish_hugr_with_outputs([u], &reg)?;

        let mono_hugr = monomorphize(hugr, &reg);
        mono_hugr.validate(&reg)?;
        let funcs = mono_hugr
            .nodes()
            .filter_map(|n| mono_hugr.get_optype(n).as_func_defn().map(|fd| (n, fd)))
            .collect_vec();
        let pf2_name = mangle_inner_func("pf1", "pf2");
        assert_eq!(
            funcs.iter().map(|(_, fd)| &fd.name).sorted().collect_vec(),
            vec![
                &mangle_name("pf1", &[ity().into()]),
                &mangle_name("pf1", &[usize_t().into()]),
                &mangle_name(&pf2_name, &[ity().into()]), // from pf1<int>
                &mangle_name(&pf2_name, &[usize_t().into()]), // from pf1<int> and (2*)pf1<usize_t>
                &mangle_inner_func(&pf2_name, "get_usz"),
                "mainish"
            ]
            .into_iter()
            .sorted()
            .collect_vec()
        );
        for (n, fd) in funcs {
            assert!(fd.signature.params().is_empty());
            assert!(mono_hugr.get_parent(n) == (fd.name != "mainish").then_some(mono_hugr.root()));
        }
        Ok(())
    }

    #[test]
    fn test_not_flattened() {
        //monof2 contains polyf3 (and instantiates) - not moved
        //polyf4 contains polyf5 but not instantiated -> not moved
    }

    #[test]
    fn test_recursive() {
        // make map,
    }
}
