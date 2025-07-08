use std::{
    collections::{HashMap, hash_map::Entry},
    convert::Infallible,
    fmt::Write,
};

use hugr_core::{
    Node,
    ops::{Call, FuncDefn, LoadFunction, OpTrait},
    types::{Signature, Substitution, TypeArg},
};

use hugr_core::hugr::{HugrView, OpType, hugrmut::HugrMut};
use itertools::Itertools as _;

use crate::ComposablePass;
use crate::composable::{ValidatePassError, validate_if_test};

/// Replaces calls to polymorphic functions with calls to new monomorphic
/// instantiations of the polymorphic ones.
///
/// If the Hugr is [Module](OpType::Module)-rooted,
/// * then the original polymorphic [`FuncDefn`]s are left untouched (including Calls inside them)
///     - [`crate::remove_dead_funcs`] can be used when no other Hugr will be linked in that might instantiate these
/// * else, the originals are removed (they are invisible from outside the Hugr); however, note
///   that this behaviour is expected to change in a future release to match Module-rooted Hugrs.
///
/// If the Hugr is [`FuncDefn`](OpType::FuncDefn)-rooted with polymorphic
/// signature then the HUGR will not be modified.
///
/// Monomorphic copies of polymorphic functions will be added to the HUGR as
/// children of the root node.  We make best effort to ensure that names (derived
/// from parent function names and concrete type args) of new functions are unique
/// whenever the names of their parents are unique, but this is not guaranteed.
pub fn monomorphize(
    hugr: &mut impl HugrMut<Node = Node>,
) -> Result<(), ValidatePassError<Node, Infallible>> {
    validate_if_test(MonomorphizePass, hugr)
}

fn is_polymorphic(fd: &FuncDefn) -> bool {
    !fd.signature().params().is_empty()
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
/// called functions (if instantiations not already in `cache`).
/// Optionally copies the subtree into a new location whilst applying a substitution.
/// The subtree should be monomorphic after the substitution (if provided) has been applied.
fn mono_scan(
    h: &mut impl HugrMut<Node = Node>,
    parent: Node,
    mut subst_into: Option<&mut Instantiating>,
    cache: &mut Instantiations,
) {
    for old_ch in h.children(parent).collect_vec() {
        let ch_op = h.get_optype(old_ch);
        debug_assert!(!ch_op.is_func_defn() || subst_into.is_none()); // If substituting, should have flattened already
        if is_polymorphic_funcdefn(ch_op) {
            continue;
        }
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
            mono_scan(h, old_ch, Some(&mut inst), cache);
            new_ch
        } else {
            mono_scan(h, old_ch, None, cache);
            old_ch
        };

        // Now instantiate the target of any Call/LoadFunction to a polymorphic function...
        let ch_op = h.get_optype(ch);
        let (type_args, mono_sig, new_op) = match ch_op {
            OpType::Call(c) => {
                let mono_sig = c.instantiation.clone();
                (
                    &c.type_args,
                    mono_sig.clone(),
                    OpType::from(Call::try_new(mono_sig.into(), []).unwrap()),
                )
            }
            OpType::LoadFunction(lf) => {
                let mono_sig = lf.instantiation.clone();
                (
                    &lf.type_args,
                    mono_sig.clone(),
                    LoadFunction::try_new(mono_sig.into(), []).unwrap().into(),
                )
            }
            _ => continue,
        };
        if type_args.is_empty() {
            continue;
        }
        let fn_inp = ch_op.static_input_port().unwrap();
        let tgt = h.static_source(old_ch).unwrap(); // Use old_ch as edges not copied yet
        let new_tgt = instantiate(h, tgt, type_args.clone(), mono_sig.clone(), cache);
        let fn_out = {
            let func = h.get_optype(new_tgt).as_func_defn().unwrap();
            debug_assert_eq!(func.signature(), &mono_sig.into());
            h.get_optype(new_tgt).static_output_port().unwrap()
        };
        h.disconnect(ch, fn_inp); // No-op if copying+substituting
        h.connect(new_tgt, fn_out, ch, fn_inp);

        h.replace_op(ch, new_op);
    }
}

fn instantiate(
    h: &mut impl HugrMut<Node = Node>,
    poly_func: Node,
    type_args: Vec<TypeArg>,
    mono_sig: Signature,
    cache: &mut Instantiations,
) -> Node {
    let for_func = cache.entry(poly_func).or_default();

    let ve = match for_func.entry(type_args.clone()) {
        Entry::Occupied(n) => return *n.get(),
        Entry::Vacant(ve) => ve,
    };

    let name = mangle_name(
        h.get_optype(poly_func).as_func_defn().unwrap().func_name(),
        &type_args,
    );
    let mono_tgt = h.add_node_after(poly_func, FuncDefn::new(name, mono_sig));
    // Insert BEFORE we scan (in case of recursion), hence we cannot use Entry::or_insert
    ve.insert(mono_tgt);
    // Now make the instantiation
    let mut node_map = HashMap::new();
    let mut inst = Instantiating {
        subst: &Substitution::new(&type_args),
        target_container: mono_tgt,
        node_map: &mut node_map,
    };
    mono_scan(h, poly_func, Some(&mut inst), cache);
    // Copy edges...we have built a node_map for every node in the function.
    // Note we could avoid building the "large" map (smaller than the Hugr we've just created)
    // by doing this during recursion, but we'd need to be careful with nonlocal edges -
    // 'ext' edges by copying every node before recursing on any of them,
    // 'dom' edges would *also* require recursing in dominator-tree preorder.
    for (&old_ch, &new_ch) in &node_map {
        for in_port in h.node_inputs(old_ch).collect::<Vec<_>>() {
            // Edges from monomorphized functions to their calls already added during mono_scan()
            // as these depend not just on the original FuncDefn but also the TypeArgs
            if h.linked_outputs(new_ch, in_port).next().is_some() {
                continue;
            }
            let srcs = h.linked_outputs(old_ch, in_port).collect::<Vec<_>>();
            for (src, outport) in srcs {
                // Sources could be a mixture of within this polymorphic FuncDefn, and Static edges from outside
                h.connect(
                    node_map.get(&src).copied().unwrap_or(src),
                    outport,
                    new_ch,
                    in_port,
                );
            }
        }
    }

    mono_tgt
}

/// Replaces calls to polymorphic functions with calls to new monomorphic
/// instantiations of the polymorphic ones.
///
/// The original polymorphic [`FuncDefn`]s are left untouched (including Calls inside them).
/// Call [`crate::remove_dead_funcs`] to remove them.
///
/// If the Hugr is [`FuncDefn`](OpType::FuncDefn)-rooted with polymorphic
/// signature then the HUGR will not be modified.
///
/// Monomorphic copies of polymorphic functions will be added to the HUGR as
/// children of the root node.  We make best effort to ensure that names (derived
/// from parent function names and concrete type args) of new functions are unique
/// whenever the names of their parents are unique, but this is not guaranteed.
#[derive(Debug, Clone)]
pub struct MonomorphizePass;

impl<H: HugrMut<Node = Node>> ComposablePass<H> for MonomorphizePass {
    type Error = Infallible;
    type Result = ();

    fn run(&self, h: &mut H) -> Result<(), Self::Error> {
        let root = h.entrypoint();
        // If the root is a polymorphic function, then there are no external calls, so nothing to do
        if !is_polymorphic_funcdefn(h.get_optype(root)) {
            mono_scan(h, root, None, &mut HashMap::new());
        }
        Ok(())
    }
}

/// Helper to create mangled representations of lists of [TypeArg]s.
struct TypeArgsSeq<'a>(&'a [TypeArg]);

impl std::fmt::Display for TypeArgsSeq<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for arg in self.0 {
            f.write_char('$')?;
            write_type_arg_str(arg, f)?;
        }
        Ok(())
    }
}

fn escape_dollar(str: impl AsRef<str>) -> String {
    str.as_ref().replace('$', "\\$")
}

fn write_type_arg_str(arg: &TypeArg, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match arg {
        TypeArg::Runtime(ty) => f.write_fmt(format_args!("t({})", escape_dollar(ty.to_string()))),
        TypeArg::BoundedNat(n) => f.write_fmt(format_args!("n({n})")),
        TypeArg::String(arg) => f.write_fmt(format_args!("s({})", escape_dollar(arg))),
        TypeArg::List(elems) => f.write_fmt(format_args!("list({})", TypeArgsSeq(elems))),
        TypeArg::Tuple(elems) => f.write_fmt(format_args!("tuple({})", TypeArgsSeq(elems))),
        // We are monomorphizing. We will never monomorphize to a signature
        // containing a variable.
        TypeArg::Variable(_) => panic!("type_arg_str variable: {arg}"),
        _ => panic!("unknown type arg: {arg}"),
    }
}

/// Produce a mangled name for a monomorphized instance of a function.
///
/// Best effort to generate unique names. The strategy is to pick out '$' as
/// a special character.
///
///  - New name of the form `{func_name}$$arg0$arg1$arg2` constructed.
///  - Existing `$` in the function name or type args string
///    representation replaced with `r"\$"`.
///  - `Display` impl of `Type` used to generate the string
///    representation of a `TypeArg::Type`. Other `TypeArgs` use `Display`
///    of inner type.
///  - For all `TypeArg` Constructors a short prefix (e.g. `t` for type)
///    is used as "t({arg})" for the string representation of that arg.
pub fn mangle_name(name: &str, type_args: impl AsRef<[TypeArg]>) -> String {
    let name = escape_dollar(name);
    format!("${name}${}", TypeArgsSeq(type_args.as_ref()))
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::iter;

    use hugr_core::extension::simple_op::MakeRegisteredOp as _;
    use hugr_core::hugr::hugrmut::HugrMut;
    use hugr_core::std_extensions::arithmetic::int_types::INT_TYPES;
    use hugr_core::std_extensions::collections;
    use hugr_core::std_extensions::collections::array::ArrayKind;
    use hugr_core::std_extensions::collections::value_array::{VArrayOpDef, ValueArray};
    use hugr_core::types::type_param::TypeParam;
    use itertools::Itertools;

    use hugr_core::builder::{
        Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder,
        HugrBuilder, ModuleBuilder,
    };
    use hugr_core::extension::prelude::{ConstUsize, UnpackTuple, UnwrapBuilder, usize_t};
    use hugr_core::ops::handle::{FuncID, NodeHandle};
    use hugr_core::ops::{CallIndirect, DataflowOpTrait as _, FuncDefn, Tag};
    use hugr_core::types::{PolyFuncType, Signature, SumType, Type, TypeArg, TypeBound, TypeEnum};
    use hugr_core::{Hugr, HugrView, Node};
    use rstest::rstest;

    use crate::{monomorphize, remove_dead_funcs};

    use super::{is_polymorphic, mangle_name};

    fn pair_type(ty: Type) -> Type {
        Type::new_tuple(vec![ty.clone(), ty])
    }

    fn triple_type(ty: Type) -> Type {
        Type::new_tuple(vec![ty.clone(), ty.clone(), ty])
    }

    #[test]
    fn test_null() {
        let dfg_builder =
            DFGBuilder::new(Signature::new(vec![usize_t()], vec![usize_t()])).unwrap();
        let [i1] = dfg_builder.input_wires_arr();
        let hugr = dfg_builder.finish_hugr_with_outputs([i1]).unwrap();
        let mut hugr2 = hugr.clone();
        monomorphize(&mut hugr2).unwrap();
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
            )?;
            fb.finish_with_outputs(tag.outputs())?
        };

        let tr = {
            let sig = Signature::new(tv0(), Type::new_tuple(vec![tv0(); 3]));
            let mut fb = mb.define_function(
                "triple",
                PolyFuncType::new([TypeBound::Copyable.into()], sig),
            )?;
            let [elem] = fb.input_wires_arr();
            let pair = fb.call(db.handle(), &[tv0().into()], [elem])?;

            let [elem1, elem2] = fb
                .add_dataflow_op(UnpackTuple::new(vec![tv0(); 2].into()), pair.outputs())?
                .outputs_arr();
            let tag = Tag::new(0, vec![vec![tv0(); 3].into()]);
            let trip = fb.add_dataflow_op(tag, [elem1, elem2, elem])?;
            fb.finish_with_outputs(trip.outputs())?
        };
        let mn = {
            let outs = vec![triple_type(usize_t()), triple_type(pair_type(usize_t()))];
            let mut fb = mb.define_function("main", Signature::new(usize_t(), outs))?;
            let [elem] = fb.input_wires_arr();
            let [res1] = fb
                .call(tr.handle(), &[usize_t().into()], [elem])?
                .outputs_arr();
            let pair = fb.call(db.handle(), &[usize_t().into()], [elem])?;
            let pty = pair_type(usize_t()).into();
            let [res2] = fb.call(tr.handle(), &[pty], pair.outputs())?.outputs_arr();
            fb.finish_with_outputs([res1, res2])?
        };
        let mut hugr = mb.finish_hugr()?;
        assert_eq!(
            hugr.entry_descendants()
                .filter(|n| hugr.get_optype(*n).is_func_defn())
                .count(),
            3
        );
        monomorphize(&mut hugr)?;
        let mono = hugr;
        mono.validate()?;

        let mut funcs = list_funcs(&mono);
        let expected_mangled_names = [
            mangle_name("double", &[usize_t().into()]),
            mangle_name("triple", &[usize_t().into()]),
            mangle_name("double", &[pair_type(usize_t()).into()]),
            mangle_name("triple", &[pair_type(usize_t()).into()]),
        ];

        for n in &expected_mangled_names {
            assert!(!is_polymorphic(funcs.remove(n).unwrap().1));
        }

        assert_eq!(
            funcs.into_keys().sorted().collect_vec(),
            ["double", "main", "triple"]
        );
        let mut mono2 = mono.clone();
        monomorphize(&mut mono2)?;

        assert_eq!(mono2, mono); // Idempotent

        let mut nopoly = mono;
        remove_dead_funcs(&mut nopoly, [mn.node()])?;
        let mut funcs = list_funcs(&nopoly);

        assert!(funcs.values().all(|(_, fd)| !is_polymorphic(fd)));
        for n in expected_mangled_names {
            assert!(funcs.remove(&n).is_some());
        }
        assert_eq!(funcs.keys().collect_vec(), vec![&"main"]);
        Ok(())
    }

    #[test]
    fn test_multiargs_nats() {
        //pf1 contains pf2 contains mono_func -> pf1<a> and pf1<b> share pf2's and they share mono_func

        let tv = |i| Type::new_var_use(i, TypeBound::Copyable);
        let sv = |i| TypeArg::new_var_use(i, TypeParam::max_nat_type());
        let sa = |n| TypeArg::BoundedNat(n);
        let n: u64 = 5;
        let mut outer = FunctionBuilder::new(
            "mainish",
            Signature::new(
                ValueArray::ty_parametric(
                    sa(n),
                    ValueArray::ty_parametric(sa(2), usize_t()).unwrap(),
                )
                .unwrap(),
                vec![usize_t(); 2],
            ),
        )
        .unwrap();

        let arr2u = || ValueArray::ty_parametric(sa(2), usize_t()).unwrap();

        let mut mb = outer.module_root_builder();

        let mono_func = {
            let mut fb = mb
                .define_function("get_usz", Signature::new(vec![], usize_t()))
                .unwrap();
            let cst0 = fb.add_load_value(ConstUsize::new(1));
            fb.finish_with_outputs([cst0]).unwrap()
        };

        let pf2 = {
            let pf2t = PolyFuncType::new(
                [TypeParam::max_nat_type(), TypeBound::Copyable.into()],
                Signature::new(ValueArray::ty_parametric(sv(0), tv(1)).unwrap(), tv(1)),
            );
            let mut pf2 = mb.define_function("pf2", pf2t).unwrap();
            let [inw] = pf2.input_wires_arr();
            let [idx] = pf2.call(mono_func.handle(), &[], []).unwrap().outputs_arr();
            let op_def = collections::value_array::EXTENSION.get_op("get").unwrap();
            let op = hugr_core::ops::ExtensionOp::new(op_def.clone(), vec![sv(0), tv(1).into()])
                .unwrap();
            let [get, _] = pf2.add_dataflow_op(op, [inw, idx]).unwrap().outputs_arr();
            let [got] = pf2
                .build_unwrap_sum(1, SumType::new([vec![], vec![tv(1)]]), get)
                .unwrap();
            pf2.finish_with_outputs([got]).unwrap()
        };

        let pf1t = PolyFuncType::new(
            [TypeParam::max_nat_type()],
            Signature::new(
                ValueArray::ty_parametric(sv(0), arr2u()).unwrap(),
                usize_t(),
            ),
        );
        let mut pf1 = mb.define_function("pf1", pf1t).unwrap();

        // pf1: Two calls to pf2, one depending on pf1's TypeArg, the other not
        let inner = pf1
            .call(pf2.handle(), &[sv(0), arr2u().into()], pf1.input_wires())
            .unwrap();
        let elem = pf1
            .call(
                pf2.handle(),
                &[TypeArg::BoundedNat(2), usize_t().into()],
                inner.outputs(),
            )
            .unwrap();
        let pf1 = pf1.finish_with_outputs(elem.outputs()).unwrap();

        // Outer: two calls to pf1 with different TypeArgs
        let [e1] = outer
            .call(pf1.handle(), &[sa(n)], outer.input_wires())
            .unwrap()
            .outputs_arr();
        let popleft = VArrayOpDef::pop_left.to_concrete(arr2u(), n);
        let ar2 = outer
            .add_dataflow_op(popleft.clone(), outer.input_wires())
            .unwrap();
        let sig = popleft.to_extension_op().unwrap().signature().into_owned();
        let TypeEnum::Sum(st) = sig.output().get(0).unwrap().as_type_enum() else {
            panic!()
        };
        let [_, ar2_unwrapped] = outer
            .build_unwrap_sum(1, st.clone(), ar2.out_wire(0))
            .unwrap();
        let [e2] = outer
            .call(pf1.handle(), &[sa(n - 1)], [ar2_unwrapped])
            .unwrap()
            .outputs_arr();
        let outer_func = outer.container_node();
        let mut hugr = outer.finish_hugr_with_outputs([e1, e2]).unwrap();
        hugr.set_entrypoint(hugr.module_root()); // We want to act on everything, not just `main`

        monomorphize(&mut hugr).unwrap();
        let mono_hugr = hugr;
        mono_hugr.validate().unwrap();
        let funcs = list_funcs(&mono_hugr);
        assert_eq!(
            funcs.keys().copied().sorted().collect_vec(),
            vec![
                &mangle_name("pf1", &[TypeArg::BoundedNat(5)]),
                &mangle_name("pf1", &[TypeArg::BoundedNat(4)]),
                &mangle_name("pf2", &[TypeArg::BoundedNat(5), arr2u().into()]), // from pf1<5>
                &mangle_name("pf2", &[TypeArg::BoundedNat(4), arr2u().into()]), // from pf1<4>
                &mangle_name("pf2", &[TypeArg::BoundedNat(2), usize_t().into()]), // from both pf1<4> and <5>
                "get_usz",
                "pf2",
                "mainish",
                "pf1"
            ]
            .into_iter()
            .sorted()
            .collect_vec()
        );
        #[allow(clippy::unnecessary_to_owned)] // it is necessary
        let (n, fd) = *funcs.get(&"mainish".to_string()).unwrap();
        assert_eq!(n, outer_func);
        assert_eq!(fd.func_name(), "mainish"); // just a sanity check on list_funcs
    }

    fn list_funcs(h: &Hugr) -> HashMap<&String, (Node, &FuncDefn)> {
        h.entry_descendants()
            .filter_map(|n| {
                h.get_optype(n)
                    .as_func_defn()
                    .map(|fd| (fd.func_name(), (n, fd)))
            })
            .collect::<HashMap<_, _>>()
    }

    #[test]
    fn load_function() {
        let mut hugr = {
            let mut module_builder = ModuleBuilder::new();
            let foo = {
                let builder = module_builder
                    .define_function(
                        "foo",
                        PolyFuncType::new(
                            [TypeBound::Linear.into()],
                            Signature::new_endo(Type::new_var_use(0, TypeBound::Linear)),
                        ),
                    )
                    .unwrap();
                let inputs = builder.input_wires();
                builder.finish_with_outputs(inputs).unwrap()
            };

            let _main = {
                let mut builder = module_builder
                    .define_function("main", Signature::new_endo(Type::UNIT))
                    .unwrap();
                let func_ptr = builder
                    .load_func(foo.handle(), &[Type::UNIT.into()])
                    .unwrap();
                let [r] = {
                    let signature = Signature::new_endo(Type::UNIT);
                    builder
                        .add_dataflow_op(
                            CallIndirect { signature },
                            iter::once(func_ptr).chain(builder.input_wires()),
                        )
                        .unwrap()
                        .outputs_arr()
                };

                builder.finish_with_outputs([r]).unwrap()
            };
            module_builder.finish_hugr().unwrap()
        };

        monomorphize(&mut hugr).unwrap();
        remove_dead_funcs(&mut hugr, []).unwrap();

        let funcs = list_funcs(&hugr);
        assert!(funcs.values().all(|(_, fd)| !is_polymorphic(fd)));
    }

    #[rstest]
    #[case::bounded_nat(vec![0.into()], "$foo$$n(0)")]
    #[case::type_unit(vec![Type::UNIT.into()], "$foo$$t(Unit)")]
    #[case::type_int(vec![INT_TYPES[2].clone().into()], "$foo$$t(int(2))")]
    #[case::string(vec!["arg".into()], "$foo$$s(arg)")]
    #[case::dollar_string(vec!["$arg".into()], "$foo$$s(\\$arg)")]
    #[case::sequence(vec![vec![0.into(), Type::UNIT.into()].into()], "$foo$$list($n(0)$t(Unit))")]
    #[case::sequence(vec![TypeArg::Tuple(vec![0.into(),Type::UNIT.into()])], "$foo$$tuple($n(0)$t(Unit))")]
    #[should_panic]
    #[case::typeargvariable(vec![TypeArg::new_var_use(1, TypeParam::StringType)],
                            "$foo$$v(1)")]
    #[case::multiple(vec![0.into(), "arg".into()], "$foo$$n(0)$s(arg)")]
    fn test_mangle_name(#[case] args: Vec<TypeArg>, #[case] expected: String) {
        assert_eq!(mangle_name("foo", &args), expected);
    }
}
