use std::hash::Hasher as _;

use hugr_core::{
    HugrView, Node,
    extension::{
        prelude::{option_type, usize_t},
        simple_op::HasConcrete as _,
    },
    ops::{ExtensionOp, constant::TryHash},
    std_extensions::collections::static_array::{
        self, StaticArrayOp, StaticArrayOpDef, StaticArrayValue,
    },
};
use inkwell::{
    AddressSpace, IntPredicate,
    builder::Builder,
    context::Context,
    types::{BasicType, BasicTypeEnum, StructType},
    values::{ArrayValue, BasicValue, BasicValueEnum, IntValue, PointerValue},
};
use itertools::Itertools as _;

use crate::{
    CodegenExtension, CodegenExtsBuilder,
    emit::{EmitFuncContext, EmitOpArgs, emit_value},
    types::{HugrType, TypingSession},
};

use anyhow::{Result, bail};

#[derive(Debug, Clone, derive_more::From)]
/// A [`CodegenExtension`] that lowers the
/// [`hugr_core::std_extensions::collections::static_array`].
///
/// All behaviour is delegated to `SACG`.
pub struct StaticArrayCodegenExtension<SACG>(SACG);

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    /// Add a [`StaticArrayCodegenExtension`] to the given [`CodegenExtsBuilder`] using `ccg`
    /// as the implementation.
    pub fn add_static_array_extensions(self, ccg: impl StaticArrayCodegen + 'static) -> Self {
        self.add_extension(StaticArrayCodegenExtension::from(ccg))
    }

    /// Add a [`StaticArrayCodegenExtension`] to the given [`CodegenExtsBuilder`] using
    /// [`DefaultStaticArrayCodegen`] as the implementation.
    #[must_use]
    pub fn add_default_static_array_extensions(self) -> Self {
        self.add_static_array_extensions(DefaultStaticArrayCodegen)
    }
}

// This is not provided by inkwell, it seems like it should be
fn value_is_const<'c>(value: impl BasicValue<'c>) -> bool {
    match value.as_basic_value_enum() {
        BasicValueEnum::ArrayValue(v) => v.is_const(),
        BasicValueEnum::IntValue(v) => v.is_const(),
        BasicValueEnum::FloatValue(v) => v.is_const(),
        BasicValueEnum::PointerValue(v) => v.is_const(),
        BasicValueEnum::StructValue(v) => v.is_const(),
        BasicValueEnum::VectorValue(v) => v.is_const(),
        BasicValueEnum::ScalableVectorValue(v) => v.is_const(),
    }
}

// This is not provided by inkwell, it seems like it should be
fn const_array<'c>(
    ty: impl BasicType<'c>,
    values: impl IntoIterator<Item = impl BasicValue<'c>>,
) -> ArrayValue<'c> {
    match ty.as_basic_type_enum() {
        BasicTypeEnum::ArrayType(t) => t.const_array(
            values
                .into_iter()
                .map(|x| x.as_basic_value_enum().into_array_value())
                .collect_vec()
                .as_slice(),
        ),
        BasicTypeEnum::FloatType(t) => t.const_array(
            values
                .into_iter()
                .map(|x| x.as_basic_value_enum().into_float_value())
                .collect_vec()
                .as_slice(),
        ),
        BasicTypeEnum::IntType(t) => t.const_array(
            values
                .into_iter()
                .map(|x| x.as_basic_value_enum().into_int_value())
                .collect_vec()
                .as_slice(),
        ),
        BasicTypeEnum::PointerType(t) => t.const_array(
            values
                .into_iter()
                .map(|x| x.as_basic_value_enum().into_pointer_value())
                .collect_vec()
                .as_slice(),
        ),
        BasicTypeEnum::StructType(t) => t.const_array(
            values
                .into_iter()
                .map(|x| x.as_basic_value_enum().into_struct_value())
                .collect_vec()
                .as_slice(),
        ),
        BasicTypeEnum::VectorType(t) => t.const_array(
            values
                .into_iter()
                .map(|x| x.as_basic_value_enum().into_vector_value())
                .collect_vec()
                .as_slice(),
        ),
        BasicTypeEnum::ScalableVectorType(t) => t.const_array(
            values
                .into_iter()
                .map(|x| x.as_basic_value_enum().into_scalable_vector_value())
                .collect_vec()
                .as_slice(),
        ),
    }
}

fn static_array_struct_type<'c>(
    context: &'c Context,
    index_type: impl BasicType<'c>,
    element_type: impl BasicType<'c>,
    len: u32,
) -> StructType<'c> {
    context.struct_type(
        &[
            index_type.as_basic_type_enum(),
            element_type.array_type(len).into(),
        ],
        false,
    )
}

fn build_read_len<'c>(
    context: &'c Context,
    builder: &Builder<'c>,
    struct_ty: StructType<'c>,
    mut ptr: PointerValue<'c>,
) -> Result<IntValue<'c>> {
    let canonical_ptr_ty = struct_ty.ptr_type(AddressSpace::default());
    if ptr.get_type() != canonical_ptr_ty {
        ptr = builder.build_pointer_cast(ptr, canonical_ptr_ty, "")?;
    }
    let i32_ty = context.i32_type();
    let indices = [i32_ty.const_zero(), i32_ty.const_zero()];
    let len_ptr = unsafe { builder.build_in_bounds_gep(ptr, &indices, "") }?;
    Ok(builder.build_load(len_ptr, "")?.into_int_value())
}

/// A helper trait for customising the lowering of [`hugr_core::std_extensions::collections::static_array`]
/// types, [`hugr_core::ops::constant::CustomConst`]s, and ops.
pub trait StaticArrayCodegen: Clone {
    /// Return the llvm type of
    /// [`hugr_core::std_extensions::collections::static_array::STATIC_ARRAY_TYPENAME`].
    ///
    /// By default a static array of llvm type `t` and length `l` is stored in a
    /// global of type `struct { i64, [t * l] }``
    ///
    /// The `i64` stores the length of the array.
    ///
    /// However a `static_array` `HugrType` is represented by an llvm pointer type
    /// `struct {i64, [t * 0]}`;  i.e. the array is zero length. This gives all
    /// static arrays of the same element type a uniform llvm type.
    ///
    /// It is legal to index past the end of an array (it is only undefined behaviour
    /// to index past the allocation).
    fn static_array_type<'c>(
        &self,
        session: TypingSession<'c, '_>,
        element_type: &HugrType,
    ) -> Result<BasicTypeEnum<'c>> {
        let index_type = session.llvm_type(&usize_t())?;
        let element_type = session.llvm_type(element_type)?;
        Ok(
            static_array_struct_type(session.iw_context(), index_type, element_type, 0)
                .ptr_type(AddressSpace::default())
                .into(),
        )
    }

    /// Emit a
    /// [`hugr_core::std_extensions::collections::static_array::StaticArrayValue`].
    ///
    /// Note that the type of the return value must match the type returned by
    /// [`Self::static_array_type`].
    ///
    /// By default a global is created and we return a pointer to it.
    fn static_array_value<'c, H: HugrView<Node = Node>>(
        &self,
        context: &mut EmitFuncContext<'c, '_, H>,
        value: &StaticArrayValue,
    ) -> Result<BasicValueEnum<'c>> {
        let element_type = value.get_element_type();
        let llvm_element_type = context.llvm_type(element_type)?;
        let index_type = context.llvm_type(&usize_t())?.into_int_type();
        let array_elements = value.get_contents().iter().map(|v| {
            let value = emit_value(context, v)?;
            if !value_is_const(value) {
                anyhow::bail!("Static array value must be constant. HUGR value '{v:?}' was codegened as non-const");
            }
            Ok(value)
        }).collect::<Result<Vec<_>>>()?;
        let len = array_elements.len();
        let struct_ty = static_array_struct_type(
            context.iw_context(),
            index_type,
            llvm_element_type,
            len as u32,
        );
        let array_value = struct_ty.const_named_struct(&[
            index_type.const_int(len as u64, false).into(),
            const_array(llvm_element_type, array_elements).into(),
        ]);

        let gv = {
            let module = context.get_current_module();
            let hash = {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                let _ = value.try_hash(&mut hasher);
                hasher.finish() as u32 // a bit shorter than u64
            };
            let prefix = format!("sa.{}.{hash:x}.", value.name);
            (0..)
                .find_map(|i| {
                    let sym = format!("{prefix}{i}");
                    if let Some(global) = module.get_global(&sym) {
                        // Note this comparison may be expensive for large
                        // values.  We could avoid it(and therefore avoid
                        // creating array_value in this branch) if we had
                        // https://github.com/CQCL/hugr/issues/2004
                        if global.get_initializer().is_some_and(|x| x == array_value) {
                            Some(global)
                        } else {
                            None
                        }
                    } else {
                        let global = module.add_global(struct_ty, None, &sym);
                        global.set_constant(true);
                        global.set_initializer(&array_value);
                        Some(global)
                    }
                })
                .unwrap()
        };
        let canonical_type = self
            .static_array_type(context.typing_session(), value.get_element_type())?
            .into_pointer_type();
        Ok(gv.as_pointer_value().const_cast(canonical_type).into())
    }

    /// Emit a [`hugr_core::std_extensions::collections::static_array::StaticArrayOp`].
    fn static_array_op<'c, H: HugrView<Node = Node>>(
        &self,
        context: &mut EmitFuncContext<'c, '_, H>,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
        op: StaticArrayOp,
    ) -> Result<()> {
        match op.def {
            StaticArrayOpDef::get => {
                let ptr = args.inputs[0].into_pointer_value();
                let index = args.inputs[1].into_int_value();
                let index_ty = index.get_type();
                let element_llvm_ty = context.llvm_type(&op.elem_ty)?;
                let struct_ty =
                    static_array_struct_type(context.iw_context(), index_ty, element_llvm_ty, 0);

                let len = build_read_len(context.iw_context(), context.builder(), struct_ty, ptr)?;

                let result_sum_ty = option_type(op.elem_ty);
                let rmb = context.new_row_mail_box([&result_sum_ty.clone().into()], "")?;
                let result_llvm_sum_ty = context.llvm_sum_type(result_sum_ty)?;

                let exit_block = context.build_positioned_new_block(
                    "static_array_get_exit",
                    context.builder().get_insert_block(),
                    |context, bb| {
                        args.outputs
                            .finish(context.builder(), rmb.read_vec(context.builder(), [])?)?;
                        anyhow::Ok(bb)
                    },
                )?;

                let fail_block = context.build_positioned_new_block(
                    "static_array_get_out_of_bounds",
                    Some(exit_block),
                    |context, bb| {
                        rmb.write(
                            context.builder(),
                            [result_llvm_sum_ty
                                .build_tag(context.builder(), 0, vec![])?
                                .into()],
                        )?;
                        context.builder().build_unconditional_branch(exit_block)?;
                        anyhow::Ok(bb)
                    },
                )?;

                let success_block = context.build_positioned_new_block(
                    "static_array_get_in_bounds",
                    Some(exit_block),
                    |context, bb| {
                        let i32_ty = context.iw_context().i32_type();
                        let indices = [i32_ty.const_zero(), i32_ty.const_int(1, false), index];
                        let element_ptr =
                            unsafe { context.builder().build_in_bounds_gep(ptr, &indices, "") }?;
                        let element = context.builder().build_load(element_ptr, "")?;
                        rmb.write(
                            context.builder(),
                            [result_llvm_sum_ty
                                .build_tag(context.builder(), 1, vec![element])?
                                .into()],
                        )?;
                        context.builder().build_unconditional_branch(exit_block)?;
                        anyhow::Ok(bb)
                    },
                )?;

                let inbounds =
                    context
                        .builder()
                        .build_int_compare(IntPredicate::ULT, index, len, "")?;
                context
                    .builder()
                    .build_conditional_branch(inbounds, success_block, fail_block)?;

                context.builder().position_at_end(exit_block);
                Ok(())
            }
            StaticArrayOpDef::len => {
                let ptr = args.inputs[0].into_pointer_value();
                let element_llvm_ty = context.llvm_type(&op.elem_ty)?;
                let index_ty = args.outputs.get_types().next().unwrap().into_int_type();
                let struct_ty =
                    static_array_struct_type(context.iw_context(), index_ty, element_llvm_ty, 0);
                let len = build_read_len(context.iw_context(), context.builder(), struct_ty, ptr)?;
                args.outputs.finish(context.builder(), [len.into()])
            }
            op => bail!("StaticArrayCodegen: Unsupported op: {op:?}"),
        }
    }
}

#[derive(Debug, Clone)]
/// An implementation of [`StaticArrayCodegen`] that uses all default
/// implementations.
pub struct DefaultStaticArrayCodegen;

impl StaticArrayCodegen for DefaultStaticArrayCodegen {}

impl<SAC: StaticArrayCodegen + 'static> CodegenExtension for StaticArrayCodegenExtension<SAC> {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a,
    {
        builder
            .custom_type(
                (
                    static_array::EXTENSION_ID,
                    static_array::STATIC_ARRAY_TYPENAME,
                ),
                {
                    let sac = self.0.clone();
                    move |ts, custom_type| {
                        let element_type = custom_type.args()[0]
                            .as_runtime()
                            .expect("Type argument for static array must be a type");
                        sac.static_array_type(ts, &element_type)
                    }
                },
            )
            .custom_const::<StaticArrayValue>({
                let sac = self.0.clone();
                move |context, sav| sac.static_array_value(context, sav)
            })
            .simple_extension_op::<StaticArrayOpDef>({
                let sac = self.0.clone();
                move |context, args, op| {
                    let op = op.instantiate(args.node().args())?;
                    sac.static_array_op(context, args, op)
                }
            })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use float_types::float64_type;
    use hugr_core::builder::DataflowHugr;
    use hugr_core::extension::prelude::ConstUsize;
    use hugr_core::ops::OpType;
    use hugr_core::ops::Value;
    use hugr_core::ops::constant::CustomConst;
    use hugr_core::std_extensions::arithmetic::float_types::{self, ConstF64};
    use rstest::rstest;

    use hugr_core::extension::simple_op::MakeRegisteredOp;
    use hugr_core::extension::{ExtensionRegistry, prelude::bool_t};
    use hugr_core::{builder::SubContainer as _, type_row};
    use static_array::StaticArrayOpBuilder as _;

    use crate::check_emission;
    use crate::test::single_op_hugr;
    use crate::{
        emit::test::SimpleHugrConfig,
        test::{TestContext, exec_ctx, llvm_ctx},
    };
    use hugr_core::builder::{Dataflow as _, DataflowSubContainer as _};

    #[rstest]
    #[case(0, StaticArrayOpDef::get, usize_t())]
    #[case(1, StaticArrayOpDef::get, bool_t())]
    #[case(2, StaticArrayOpDef::len, usize_t())]
    #[case(3, StaticArrayOpDef::len, bool_t())]
    fn static_array_op_codegen(
        #[case] _i: i32,
        #[with(_i)] mut llvm_ctx: TestContext,
        #[case] op: StaticArrayOpDef,
        #[case] ty: HugrType,
    ) {
        let op = op.instantiate(&[ty.clone().into()]).unwrap();
        let op = OpType::from(op.to_extension_op().unwrap());
        llvm_ctx.add_extensions(|ceb| {
            ceb.add_default_static_array_extensions()
                .add_default_prelude_extensions()
        });
        let hugr = single_op_hugr(op);
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    #[case(0, StaticArrayValue::try_new("a", usize_t(), (0..10).map(|x| ConstUsize::new(x).into())).unwrap())]
    #[case(1, StaticArrayValue::try_new("b", float64_type(), (0..10).map(|x| ConstF64::new(f64::from(x)).into())).unwrap())]
    #[case(2, StaticArrayValue::try_new("c", bool_t(), (0..10).map(|x| Value::from_bool(x % 2 == 0))).unwrap())]
    #[case(3, StaticArrayValue::try_new("d", option_type(usize_t()).into(), (0..10).map(|x| Value::some([ConstUsize::new(x)]))).unwrap())]
    fn static_array_const_codegen(
        #[case] _i: i32,
        #[with(_i)] mut llvm_ctx: TestContext,
        #[case] value: StaticArrayValue,
    ) {
        llvm_ctx.add_extensions(|ceb| {
            ceb.add_default_static_array_extensions()
                .add_default_prelude_extensions()
                .add_float_extensions()
        });

        let hugr = SimpleHugrConfig::new()
            .with_outs(value.get_type())
            .with_extensions(ExtensionRegistry::new(vec![
                static_array::EXTENSION.to_owned(),
                float_types::EXTENSION.to_owned(),
            ]))
            .finish(|mut builder| {
                let a = builder.add_load_value(value);
                builder.finish_hugr_with_outputs([a]).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }

    #[rstest]
    #[case(0, 0, 999)]
    #[case(1, 1, 998)]
    #[case(2, 1000, u64::MAX)]
    fn static_array_exec(
        #[case] _i: i32,
        #[with(_i)] mut exec_ctx: TestContext,
        #[case] index: u64,
        #[case] expected: u64,
    ) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(ExtensionRegistry::new(vec![
                static_array::EXTENSION.to_owned(),
            ]))
            .finish(|mut builder| {
                let arr = builder.add_load_value(
                    StaticArrayValue::try_new(
                        "exec_arr",
                        usize_t(),
                        (0..1000)
                            .map(|x| ConstUsize::new(999 - x).into())
                            .collect_vec(),
                    )
                    .unwrap(),
                );
                let index = builder.add_load_value(ConstUsize::new(index));
                let get_r = builder.add_static_array_get(usize_t(), arr, index).unwrap();
                let [out] = {
                    let mut cond = builder
                        .conditional_builder(
                            ([type_row!(), usize_t().into()], get_r),
                            [],
                            usize_t().into(),
                        )
                        .unwrap();
                    {
                        let mut oob_case = cond.case_builder(0).unwrap();
                        let err = oob_case.add_load_value(ConstUsize::new(u64::MAX));
                        oob_case.finish_with_outputs([err]).unwrap();
                    }
                    {
                        let inbounds_case = cond.case_builder(1).unwrap();
                        let [out] = inbounds_case.input_wires_arr();
                        inbounds_case.finish_with_outputs([out]).unwrap();
                    }
                    cond.finish_sub_container().unwrap().outputs_arr()
                };
                builder.finish_hugr_with_outputs([out]).unwrap()
            });

        exec_ctx.add_extensions(|ceb| {
            ceb.add_default_static_array_extensions()
                .add_default_prelude_extensions()
                .add_float_extensions()
        });
        assert_eq!(expected, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    fn len_0_array(mut exec_ctx: TestContext) {
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(ExtensionRegistry::new(vec![
                static_array::EXTENSION.to_owned(),
            ]))
            .finish(|mut builder| {
                let arr = builder
                    .add_load_value(StaticArrayValue::try_new("empty", usize_t(), vec![]).unwrap());
                let len = builder.add_static_array_len(usize_t(), arr).unwrap();
                builder.finish_hugr_with_outputs([len]).unwrap()
            });

        exec_ctx.add_extensions(|ceb| {
            ceb.add_default_static_array_extensions()
                .add_default_prelude_extensions()
        });
        assert_eq!(0, exec_ctx.exec_hugr_u64(hugr, "main"));
    }

    #[rstest]
    fn emit_static_array_of_static_array(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(|ceb| {
            ceb.add_default_static_array_extensions()
                .add_default_prelude_extensions()
        });
        let hugr = SimpleHugrConfig::new()
            .with_outs(usize_t())
            .with_extensions(ExtensionRegistry::new(vec![
                static_array::EXTENSION.to_owned(),
            ]))
            .finish(|mut builder| {
                let inner_arrs: Vec<Value> = (0..10)
                    .map(|i| {
                        StaticArrayValue::try_new(
                            "inner",
                            usize_t(),
                            vec![Value::from(ConstUsize::new(i)); i as usize],
                        )
                        .unwrap()
                        .into()
                    })
                    .collect_vec();
                let inner_arr_ty = inner_arrs[0].get_type();
                let outer_arr = builder.add_load_value(
                    StaticArrayValue::try_new("outer", inner_arr_ty.clone(), inner_arrs).unwrap(),
                );
                let len = builder
                    .add_static_array_len(inner_arr_ty, outer_arr)
                    .unwrap();
                builder.finish_hugr_with_outputs([len]).unwrap()
            });
        check_emission!(hugr, llvm_ctx);
    }
}
