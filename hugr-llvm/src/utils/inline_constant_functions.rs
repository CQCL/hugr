use hugr_core::{
    extension::ExtensionRegistry,
    hugr::hugrmut::HugrMut,
    ops::{FuncDefn, LoadFunction, Value},
    types::PolyFuncType,
    HugrView, Node, NodeIndex as _,
};

use anyhow::{anyhow, bail, Result};

fn const_fn_name(konst_n: Node) -> String {
    format!("const_fun_{}", konst_n.index())
}

pub fn inline_constant_functions(
    hugr: &mut impl HugrMut,
    registry: &ExtensionRegistry,
) -> Result<()> {
    while inline_constant_functions_impl(hugr, registry)? {}
    Ok(())
}

fn inline_constant_functions_impl(
    hugr: &mut impl HugrMut,
    registry: &ExtensionRegistry,
) -> Result<bool> {
    let mut const_funs = vec![];

    for n in hugr.nodes() {
        let konst_hugr = {
            let Some(konst) = hugr.get_optype(n).as_const() else {
                continue;
            };
            let Value::Function { hugr } = konst.value() else {
                continue;
            };
            let optype = hugr.get_optype(hugr.root());
            if !optype.is_dfg() && !optype.is_func_defn() {
                bail!(
                    "Constant function has unsupported root: {:?}",
                    hugr.get_optype(hugr.root())
                )
            }
            hugr.clone()
        };
        let mut lcs = vec![];
        for load_constant in hugr.output_neighbours(n) {
            if !hugr.get_optype(load_constant).is_load_constant() {
                bail!(
                    "Constant function has non-LoadConstant output-neighbour: {load_constant} {:?}",
                    hugr.get_optype(load_constant)
                )
            }
            lcs.push(load_constant);
        }
        const_funs.push((n, konst_hugr.as_ref().clone(), lcs));
    }

    let mut any_changes = false;

    for (konst_n, func_hugr, load_constant_ns) in const_funs {
        if !load_constant_ns.is_empty() {
            let polysignature: PolyFuncType = func_hugr
                .inner_function_type()
                .ok_or(anyhow!(
                    "Constant function hugr has no inner_func_type: {}",
                    konst_n.index()
                ))?
                .into();
            let func_defn = FuncDefn {
                name: const_fn_name(konst_n),
                signature: polysignature.clone(),
            };
            let func_node = hugr.add_node_with_parent(hugr.root(), func_defn);
            hugr.insert_hugr(func_node, func_hugr);

            for lcn in load_constant_ns {
                hugr.replace_op(
                    lcn,
                    LoadFunction::try_new(polysignature.clone(), [], registry)?,
                )?;
            }
            any_changes = true;
        }
        hugr.remove_node(konst_n);
    }
    Ok(any_changes)
}

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{
            Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder,
            ModuleBuilder,
        },
        extension::{prelude::qb_t, PRELUDE_REGISTRY},
        ops::{CallIndirect, Const, Value},
        types::Signature,
        Hugr, HugrView, Wire,
    };

    use super::inline_constant_functions;

    fn build_const(go: impl FnOnce(&mut DFGBuilder<Hugr>) -> Wire) -> Const {
        Value::function({
            let mut builder = DFGBuilder::new(Signature::new_endo(qb_t())).unwrap();
            let r = go(&mut builder);
            builder
                .finish_hugr_with_outputs([r], &PRELUDE_REGISTRY)
                .unwrap()
        })
        .unwrap()
        .into()
    }

    #[test]
    fn simple() {
        let qb_sig: Signature = Signature::new_endo(qb_t());
        let mut hugr = {
            let mut builder = ModuleBuilder::new();
            let const_node = builder.add_constant(build_const(|builder| {
                let [r] = builder.input_wires_arr();
                r
            }));
            {
                let mut builder = builder.define_function("main", qb_sig.clone()).unwrap();
                let [i] = builder.input_wires_arr();
                let fun = builder.load_const(&const_node);
                let [r] = builder
                    .add_dataflow_op(
                        CallIndirect {
                            signature: qb_sig.clone(),
                        },
                        [fun, i],
                    )
                    .unwrap()
                    .outputs_arr();
                builder.finish_with_outputs([r]).unwrap();
            };
            builder.finish_hugr(&PRELUDE_REGISTRY).unwrap()
        };

        inline_constant_functions(&mut hugr, &PRELUDE_REGISTRY).unwrap();

        for n in hugr.nodes() {
            if let Some(konst) = hugr.get_optype(n).as_const() {
                assert!(!matches!(konst.value(), Value::Function { .. }))
            }
        }
    }

    #[test]
    fn nested() {
        let qb_sig: Signature = Signature::new_endo(qb_t());
        let mut hugr = {
            let mut builder = ModuleBuilder::new();
            let const_node = builder.add_constant(build_const(|builder| {
                let [i] = builder.input_wires_arr();
                let func = builder.add_load_const(build_const(|builder| {
                    let [r] = builder.input_wires_arr();
                    r
                }));
                let [r] = builder
                    .add_dataflow_op(
                        CallIndirect {
                            signature: qb_sig.clone(),
                        },
                        [func, i],
                    )
                    .unwrap()
                    .outputs_arr();
                r
            }));
            {
                let mut builder = builder.define_function("main", qb_sig.clone()).unwrap();
                let [i] = builder.input_wires_arr();
                let fun = builder.load_const(&const_node);
                let [r] = builder
                    .add_dataflow_op(
                        CallIndirect {
                            signature: qb_sig.clone(),
                        },
                        [fun, i],
                    )
                    .unwrap()
                    .outputs_arr();
                builder.finish_with_outputs([r]).unwrap();
            };
            builder.finish_hugr(&PRELUDE_REGISTRY).unwrap()
        };

        inline_constant_functions(&mut hugr, &PRELUDE_REGISTRY).unwrap();

        for n in hugr.nodes() {
            if let Some(konst) = hugr.get_optype(n).as_const() {
                assert!(!matches!(konst.value(), Value::Function { .. }))
            }
        }
    }
}
