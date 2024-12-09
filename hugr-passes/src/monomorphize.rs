use std::collections::{hash_map::Entry, HashMap};

use hugr_core::{
    extension::ExtensionRegistry,
    ops::{Call, FuncDefn, OpTrait},
    types::{Signature, Substitution, TypeArg},
    Node,
};

use hugr_core::hugr::{hugrmut::HugrMut, internal::HugrMutInternals, Hugr, HugrView, OpType};

/// Replaces calls to polymorphic functions with calls to new monomorphic
/// instantiations of the polymorphic ones. The original polymorphic [FuncDefn]s
/// are left untouched, although with fewer calls (they may still have calls
/// from *other* polymorphic functions still present).
pub fn monomorphize(mut h: Hugr, reg: &ExtensionRegistry) -> Hugr {
    let root = h.root();
    // If the root is a polymorphic function, then there are no external calls, so nothing to do
    if !is_polymorphic_funcdefn(h.get_optype(root)) {
        mono_scan(&mut h, root, None, &mut HashMap::new(), reg);
    }
    h
}

fn is_polymorphic(fd: &FuncDefn) -> bool {
    !fd.signature.params().is_empty()
}

fn is_polymorphic_funcdefn(t: &OpType) -> bool {
    t.as_func_defn().is_some_and(is_polymorphic)
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
                h.add_node_with_parent(inst.target_container, ch_op.substitute(inst.subst));
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

        h.replace_op(ch, Call::try_new(mono_sig.into(), vec![], reg).unwrap())
            .unwrap();
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
            if let OpType::FuncDefn(fd) = h.get_optype(n) {
                let fd = FuncDefn {
                    name: mangle_inner_func(&outer_name, &fd.name),
                    signature: fd.signature.clone(),
                };
                h.replace_op(n, fd).unwrap();
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

    use hugr_core::extension::simple_op::MakeRegisteredOp;
    use hugr_core::types::type_param::TypeParam;
    use itertools::Itertools;

    use hugr_core::builder::{
        Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder,
        HugrBuilder, ModuleBuilder,
    };
    use hugr_core::extension::prelude::{
        array_type, usize_t, ArrayOpDef, ConstUsize, UnpackTuple, UnwrapBuilder, PRELUDE_ID,
    };
    use hugr_core::extension::{ExtensionRegistry, EMPTY_REG, PRELUDE, PRELUDE_REGISTRY};
    use hugr_core::ops::handle::{FuncID, NodeHandle};
    use hugr_core::ops::{DataflowOpTrait, FuncDefn, Tag};
    use hugr_core::std_extensions::arithmetic::int_types::{self, INT_TYPES};
    use hugr_core::types::{
        PolyFuncType, Signature, SumType, Type, TypeArg, TypeBound, TypeEnum, TypeRow,
    };
    use hugr_core::{Hugr, HugrView, Node};

    use super::{is_polymorphic, mangle_inner_func, mangle_name, monomorphize};

    fn pair_type(ty: Type) -> Type {
        Type::new_tuple(vec![ty.clone(), ty])
    }

    fn triple_type(ty: Type) -> Type {
        Type::new_tuple(vec![ty.clone(), ty.clone(), ty])
    }

    fn prelusig(ins: impl Into<TypeRow>, outs: impl Into<TypeRow>) -> Signature {
        Signature::new(ins, outs).with_extension_delta(PRELUDE_ID)
    }

    #[test]
    fn test_null() {
        let dfg_builder =
            DFGBuilder::new(Signature::new(vec![usize_t()], vec![usize_t()])).unwrap();
        let [i1] = dfg_builder.input_wires_arr();
        let hugr = dfg_builder.finish_prelude_hugr_with_outputs([i1]).unwrap();
        let hugr2 = monomorphize(hugr.clone(), &PRELUDE_REGISTRY);
        assert_eq!(hugr, hugr2);
    }

    #[test]
    fn test_recursion_module() -> Result<(), Box<dyn std::error::Error>> {
        let tv0 = || Type::new_var_use(0, TypeBound::Copyable);
        let mut mb = ModuleBuilder::new();
        let db = {
            let pfty = PolyFuncType::new(
                [TypeBound::Copyable.into()],
                Signature::new(tv0(), pair_type(tv0())),
            );
            let mut fb = mb.define_function("double", pfty)?;
            let [elem] = fb.input_wires_arr();
            // A "genuine" impl might:
            //   let tag = Tag::new(0, vec![vec![elem_ty; 2].into()]);
            //   let tag = fb.add_dataflow_op(tag, [elem, elem]).unwrap();
            // ...but since this will never execute, we can test recursion here
            let tag = fb.call(
                &FuncID::<true>::from(fb.container_node()),
                &[tv0().into()],
                [elem],
                &EMPTY_REG,
            )?;
            fb.finish_with_outputs(tag.outputs())?
        };

        let tr = {
            let sig = prelusig(tv0(), Type::new_tuple(vec![tv0(); 3]));
            let mut fb = mb.define_function(
                "triple",
                PolyFuncType::new([TypeBound::Copyable.into()], sig),
            )?;
            let [elem] = fb.input_wires_arr();
            let pair = fb.call(db.handle(), &[tv0().into()], [elem], &PRELUDE_REGISTRY)?;

            let [elem1, elem2] = fb
                .add_dataflow_op(UnpackTuple::new(vec![tv0(); 2].into()), pair.outputs())?
                .outputs_arr();
            let tag = Tag::new(0, vec![vec![tv0(); 3].into()]);
            let trip = fb.add_dataflow_op(tag, [elem1, elem2, elem])?;
            fb.finish_with_outputs(trip.outputs())?
        };
        {
            let outs = vec![triple_type(usize_t()), triple_type(pair_type(usize_t()))];
            let mut fb = mb.define_function("main", prelusig(usize_t(), outs))?;
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
        assert_eq!(hugr.static_targets(db.node()).unwrap().count(), 3); // from double (recursive), triple, main
        let mono = monomorphize(hugr, &PRELUDE_REGISTRY);
        mono.validate(&PRELUDE_REGISTRY)?;

        let mut funcs = list_funcs(&mono);
        let expected_mangled_names = [
            mangle_name("double", &[usize_t().into()]),
            mangle_name("triple", &[usize_t().into()]),
            mangle_name("double", &[pair_type(usize_t()).into()]),
            mangle_name("triple", &[pair_type(usize_t()).into()]),
        ];

        for n in expected_mangled_names.iter() {
            assert!(!is_polymorphic(funcs.remove(n).unwrap().1));
        }

        assert_eq!(
            funcs.into_keys().sorted().collect_vec(),
            ["double", "main", "triple"]
        );

        // Original double/triple retained, but call to double from main now lost
        assert!(is_polymorphic(
            mono.get_optype(db.node()).as_func_defn().unwrap()
        ));
        assert!(is_polymorphic(
            mono.get_optype(tr.node()).as_func_defn().unwrap()
        ));
        assert_eq!(
            mono.static_targets(db.node())
                .unwrap()
                .map(|(call, _port)| mono.get_parent(call))
                .collect_vec(),
            vec![Some(db.node()), Some(tr.node())]
        );

        assert_eq!(monomorphize(mono.clone(), &PRELUDE_REGISTRY), mono); // Idempotent
        Ok(())
    }

    #[test]
    fn test_flattening_multiargs_nats() -> Result<(), Box<dyn std::error::Error>> {
        //pf1 contains pf2 contains mono_func -> pf1<a> and pf1<b> share pf2's and they share mono_func

        let reg =
            ExtensionRegistry::try_new([int_types::EXTENSION.to_owned(), PRELUDE.to_owned()])?;
        let tv = |i| Type::new_var_use(i, TypeBound::Copyable);
        let sv = |i| TypeArg::new_var_use(i, TypeParam::max_nat());
        let sa = |n| TypeArg::BoundedNat { n };

        let n: u64 = 5;
        let mut outer = FunctionBuilder::new(
            "mainish",
            prelusig(
                array_type(sa(n), array_type(sa(2), usize_t())),
                vec![usize_t(); 2],
            ),
        )?;

        let arr2u = || array_type(sa(2), usize_t());
        let pf1t = PolyFuncType::new(
            [TypeParam::max_nat()],
            prelusig(array_type(sv(0), arr2u()), usize_t()),
        );
        let mut pf1 = outer.define_function("pf1", pf1t)?;

        let pf2t = PolyFuncType::new(
            [TypeParam::max_nat(), TypeBound::Copyable.into()],
            prelusig(vec![array_type(sv(0), tv(1))], tv(1)),
        );
        let mut pf2 = pf1.define_function("pf2", pf2t)?;

        let mono_func = {
            let mut fb = pf2.define_function("get_usz", prelusig(vec![], usize_t()))?;
            let cst0 = fb.add_load_value(ConstUsize::new(1));
            fb.finish_with_outputs([cst0])?
        };
        let pf2 = {
            let [inw] = pf2.input_wires_arr();
            let [idx] = pf2.call(mono_func.handle(), &[], [], &reg)?.outputs_arr();
            let op_def = PRELUDE.get_op("get").unwrap();
            let op =
                hugr_core::ops::ExtensionOp::new(op_def.clone(), vec![sv(0), tv(1).into()], &reg)?;
            let [get] = pf2.add_dataflow_op(op, [inw, idx])?.outputs_arr();
            let [got] = pf2.build_unwrap_sum(&reg, 1, SumType::new([vec![], vec![tv(1)]]), get)?;
            pf2.finish_with_outputs([got])?
        };
        // pf1: Two calls to pf2, one depending on pf1's TypeArg, the other not
        let inner = pf1.call(
            pf2.handle(),
            &[sv(0), arr2u().into()],
            pf1.input_wires(),
            &reg,
        )?;
        let elem = pf1.call(
            pf2.handle(),
            &[TypeArg::BoundedNat { n: 2 }, usize_t().into()],
            inner.outputs(),
            &reg,
        )?;
        let pf1 = pf1.finish_with_outputs(elem.outputs())?;
        // Outer: two calls to pf1 with different TypeArgs
        let [e1] = outer
            .call(pf1.handle(), &[sa(n)], outer.input_wires(), &reg)?
            .outputs_arr();
        let popleft = ArrayOpDef::pop_left.to_concrete(arr2u(), n);
        let ar2 = outer.add_dataflow_op(popleft.clone(), outer.input_wires())?;
        let sig = popleft.to_extension_op().unwrap().signature();
        let TypeEnum::Sum(st) = sig.output().get(0).unwrap().as_type_enum() else {
            panic!()
        };
        let [_, ar2_unwrapped] = outer
            .build_unwrap_sum(&reg, 1, st.clone(), ar2.out_wire(0))
            .unwrap();
        let [e2] = outer
            .call(pf1.handle(), &[sa(n - 1)], [ar2_unwrapped], &reg)?
            .outputs_arr();
        let hugr = outer.finish_hugr_with_outputs([e1, e2], &reg)?;

        let mono_hugr = monomorphize(hugr, &reg);
        mono_hugr.validate(&reg)?;
        let funcs = list_funcs(&mono_hugr);
        let pf2_name = mangle_inner_func("pf1", "pf2");
        assert_eq!(
            funcs.keys().copied().sorted().collect_vec(),
            vec![
                &mangle_name("pf1", &[TypeArg::BoundedNat { n: 5 }]),
                &mangle_name("pf1", &[TypeArg::BoundedNat { n: 4 }]),
                &mangle_name(&pf2_name, &[TypeArg::BoundedNat { n: 5 }, arr2u().into()]), // from pf1<5>
                &mangle_name(&pf2_name, &[TypeArg::BoundedNat { n: 4 }, arr2u().into()]), // from pf1<4>
                &mangle_name(&pf2_name, &[TypeArg::BoundedNat { n: 2 }, usize_t().into()]), // from both pf1<4> and <5>
                &mangle_inner_func(&pf2_name, "get_usz"),
                "mainish",
                "pf1",
                &pf2_name,
            ]
            .into_iter()
            .sorted()
            .collect_vec()
        );
        for (n, fd) in funcs.into_values() {
            assert_eq!(
                is_polymorphic(fd),
                ["pf1", &pf2_name].contains(&fd.name.as_str())
            );
            assert_eq!(
                mono_hugr.get_parent(n),
                (fd.name != "mainish").then_some(mono_hugr.root())
            );
        }
        Ok(())
    }

    fn list_funcs(h: &Hugr) -> HashMap<&String, (Node, &FuncDefn)> {
        h.nodes()
            .filter_map(|n| h.get_optype(n).as_func_defn().map(|fd| (&fd.name, (n, fd))))
            .collect::<HashMap<_, _>>()
    }

    #[test]
    fn test_no_flatten_out_of_mono_func() -> Result<(), Box<dyn std::error::Error>> {
        let ity = || INT_TYPES[4].clone();
        let reg =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), int_types::EXTENSION.to_owned()])?;
        let sig = Signature::new_endo(vec![usize_t(), ity()]);
        let mut dfg = DFGBuilder::new(sig.clone())?;
        let mut monof = dfg.define_function("id2", sig)?;
        let polyf = monof.define_function(
            "id",
            PolyFuncType::new(
                [TypeBound::Any.into()],
                Signature::new_endo(Type::new_var_use(0, TypeBound::Any)),
            ),
        )?;
        let outs = polyf.input_wires();
        let polyf = polyf.finish_with_outputs(outs)?;
        let [a, b] = monof.input_wires_arr();
        let [a] = monof
            .call(polyf.handle(), &[usize_t().into()], [a], &reg)?
            .outputs_arr();
        let [b] = monof
            .call(polyf.handle(), &[ity().into()], [b], &reg)?
            .outputs_arr();
        let monof = monof.finish_with_outputs([a, b])?;
        let c = dfg.call(monof.handle(), &[], dfg.input_wires(), &reg)?;
        let hugr = dfg.finish_hugr_with_outputs(c.outputs(), &reg)?;
        let mono_hugr = monomorphize(hugr, &reg);

        let mut funcs = list_funcs(&mono_hugr);
        for (n, fd) in funcs.values() {
            assert_eq!(is_polymorphic(fd), *n == polyf.node());
        }
        #[allow(clippy::unnecessary_to_owned)] // It is necessary
        let (m, _) = funcs.remove(&"id2".to_string()).unwrap();
        assert_eq!(m, monof.node());
        assert_eq!(mono_hugr.get_parent(m), Some(mono_hugr.root()));
        for t in [usize_t(), ity()] {
            let (n, _) = funcs.remove(&mangle_name("id", &[t.into()])).unwrap();
            assert_eq!(mono_hugr.get_parent(n), Some(m)); // Not lifted to top
        }
        Ok(())
    }
}
