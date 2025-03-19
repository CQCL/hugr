use hugr_core::{extension::simple_op::HasConcrete as _, ops::ExtensionOp, std_extensions::collections::static_array::{self, StaticArrayOp, StaticArrayOpDef, StaticArrayValue}, HugrView, Node};
use inkwell::{types::{BasicType, BasicTypeEnum}, values::{ArrayValue, BasicValue, BasicValueEnum}, AddressSpace};
use itertools::Itertools as _;

use crate::{emit::{emit_value, EmitFuncContext, EmitOpArgs}, types::{HugrType, TypingSession}, CodegenExtension, CodegenExtsBuilder};

use anyhow::Result;

#[derive(Debug, Clone, derive_more::From)]
struct StaticArrayCodegenExtension<SACG>(SACG);

impl<'a, H: HugrView<Node = Node> + 'a> CodegenExtsBuilder<'a, H> {
    /// Add a [ArrayCodegenExtension] to the given [CodegenExtsBuilder] using `ccg`
    /// as the implementation.
    pub fn add_static_array_extensions(self, ccg: impl StaticArrayCodegen + 'static) -> Self {
        self.add_extension(StaticArrayCodegenExtension::from(ccg))
    }

    /// Add a [ArrayCodegenExtension] to the given [CodegenExtsBuilder] using
    /// [DefaultArrayCodegen] as the implementation.
    pub fn add_default_static_array_extensions(self) -> Self {
        self.add_static_array_extensions(DefaultStaticArrayCodegen)
    }
}

fn value_is_const<'c>(value: impl BasicValue<'c>) -> bool {
    match value.as_basic_value_enum() {
        BasicValueEnum::ArrayValue(v) => v.is_const(),
        BasicValueEnum::IntValue(v) => v.is_const(),
        BasicValueEnum::FloatValue(v) => v.is_const(),
        BasicValueEnum::PointerValue(v) => v.is_const(),
        BasicValueEnum::StructValue(v) => v.is_const(),
        BasicValueEnum::VectorValue(v) => v.is_const(),
    }
}

fn const_array<'c>(ty: impl BasicType<'c>, values: impl IntoIterator<Item = impl BasicValue<'c>>) -> ArrayValue<'c> {
    match ty.as_basic_type_enum() {
        BasicTypeEnum::ArrayType(t) => t.const_array(values.into_iter().map(|x| x.as_basic_value_enum().into_array_value()).collect_vec().as_slice()),
        BasicTypeEnum::FloatType(t) => t.const_array(values.into_iter().map(|x| x.as_basic_value_enum().into_float_value()).collect_vec().as_slice()),
        BasicTypeEnum::IntType(t) => t.const_array(values.into_iter().map(|x| x.as_basic_value_enum().into_int_value()).collect_vec().as_slice()),
        BasicTypeEnum::PointerType(t) => t.const_array(values.into_iter().map(|x| x.as_basic_value_enum().into_pointer_value()).collect_vec().as_slice()),
        BasicTypeEnum::StructType(t) => t.const_array(values.into_iter().map(|x| x.as_basic_value_enum().into_struct_value()).collect_vec().as_slice()),
        BasicTypeEnum::VectorType(t) => t.const_array(values.into_iter().map(|x| x.as_basic_value_enum().into_vector_value()).collect_vec().as_slice()),
    }
}

pub trait StaticArrayCodegen: Clone {
    fn static_array_type<'c>(&self, session: TypingSession<'c, '_>, element_type: &HugrType) -> Result<BasicTypeEnum<'c>> {
        Ok(session.llvm_type(element_type)?.ptr_type(AddressSpace::default()).as_basic_type_enum())
    }

    fn static_array_value<'c, H: HugrView<Node=Node>>(&self, context: &mut EmitFuncContext<'c, '_, H>, value: &StaticArrayValue) -> Result<BasicValueEnum<'c>> {
        let element_type = value.get_element_type();
        let llvm_element_type = context.llvm_type(element_type)?;
        let array_type = llvm_element_type.array_type(value.value.get_contents().len() as u32);

        let array_elements = value.value.get_contents().iter().map(|v| {
            let value = emit_value(context, v)?;
            if !value_is_const(value) {
                anyhow::bail!("Static array value must be constant. HUGR value '{v:?}' was codegened as non-const");
            }
            Ok(value)
        }).collect::<Result<Vec<_>>>()?;
        let array_value = const_array(llvm_element_type, array_elements);
        // llvm_element_type.const_array(value.value.get_contents().iter().map(|v| context.emit_const(element_type, v)).collect::<Result<Vec<_>>>()?.as_slice()).

        // let element_size = element_type.size_of();
        // let element_size = element_size.const_cast(context.i32_type());
        // let array_size = value.len() as u32;
        // let array_size = context.i32_type().const_int(array_size as u64, false);
        // let array_type = self.static_array_type(context.ts, element_type)?;
        // let array_type = array_type.into_pointer_type();
        // let array = context.builder.build_array_alloca(array_type, array_size, "static_array");
        // for (i, element) in value.iter().enumerate() {
        //     let element = context.emit_const(element_type, element)?;
        //     let gep = context.builder.build_gep(array, &[context.i32_type().const_int(i as u64, false)], "element");
        //     context.builder.build_store(gep, element);
        // }
        // Ok(array.into())
    }

    fn static_array_op<'c, H: HugrView<Node=Node>>(&self, context: &mut EmitFuncContext<'c, '_, H>, args: EmitOpArgs<'c, '_, ExtensionOp, H>, op: StaticArrayOp) -> Result<()> {
        todo!()
    }
}

#[derive(Debug,Clone)]
pub struct DefaultStaticArrayCodegen;

impl StaticArrayCodegen for DefaultStaticArrayCodegen {
}

impl<SAC: StaticArrayCodegen + 'static> CodegenExtension for StaticArrayCodegenExtension<SAC> {
    fn add_extension<'a, H: HugrView<Node = Node> + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H> where Self: 'a {
        builder.custom_type((static_array::EXTENSION_ID, static_array::STATIC_ARRAY_TYPENAME), {
            let sac = self.0.clone();
            move |ts, custom_type| {
                let element_type = custom_type
                    .args()[0].as_type().expect("Type argument for static array must be a type");
                sac.static_array_type(ts, &element_type)
            }
        }).custom_const::<StaticArrayValue>({
            let sac = self.0.clone();
            move |context, sav| sac.static_array_value(context, sav)
        }).simple_extension_op::<StaticArrayOpDef>({
            let sac = self.0.clone();
            move |context, args, op| {
                let op = op.instantiate(args.node().args())?;
                sac.static_array_op(context, args, op)
            }
        })
    }
}
