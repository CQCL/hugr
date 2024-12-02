//! Provides an interface for extending `hugr-llvm` to emit [CustomType]s,
//! [CustomConst]s, and [ExtensionOp]s.
//!
//! [CustomType]: hugr_core::types::CustomType
//! [CustomConst]: hugr_core::ops::constant::CustomConst
//! [ExtensionOp]: hugr_core::ops::ExtensionOp
use std::rc::Rc;

use self::extension_op::{ExtensionOpFn, ExtensionOpMap};
use hugr_core::{
    extension::{simple_op::MakeOpDef, ExtensionId},
    ops::{constant::CustomConst, ExtensionOp, OpName},
    HugrView,
};

use strum::IntoEnumIterator;
use types::CustomTypeKey;

use self::load_constant::{LoadConstantFn, LoadConstantsMap};
use self::types::LLVMCustomTypeFn;
use anyhow::Result;

use crate::{
    emit::{func::EmitFuncContext, EmitOpArgs},
    types::TypeConverter,
};

pub mod extension_op;
pub mod load_constant;
pub mod types;

/// A helper to register codegen extensions.
///
/// Types that implement this trait can be registered with a [CodegenExtsBuilder]
/// via [CodegenExtsBuilder::add_extension].
///
/// See [crate::extension::PreludeCodegenExtension] for an example.
pub trait CodegenExtension {
    /// Implementers should add each of their handlers to `builder` and return the
    /// resulting [CodegenExtsBuilder].
    fn add_extension<'a, H: HugrView + 'a>(
        self,
        builder: CodegenExtsBuilder<'a, H>,
    ) -> CodegenExtsBuilder<'a, H>
    where
        Self: 'a;
}

/// A container for a collection of codegen callbacks as they are being
/// assembled.
///
/// The callbacks are registered against several keys:
///  - [CustomType]s, with [CodegenExtsBuilder::custom_type]
///  - [CustomConst]s, with [CodegenExtsBuilder::custom_const]
///  - [ExtensionOp]s, with [CodegenExtsBuilder::extension_op]
///
/// Each callback may hold references older than `'a`.
///
/// Registering any callback silently replaces any other callback registered for
/// that same key.
///
/// [CustomType]: hugr_core::types::CustomType
#[derive(Default)]
pub struct CodegenExtsBuilder<'a, H> {
    load_constant_handlers: LoadConstantsMap<'a, H>,
    extension_op_handlers: ExtensionOpMap<'a, H>,
    type_converter: TypeConverter<'a>,
}

impl<'a, H: HugrView + 'a> CodegenExtsBuilder<'a, H> {
    /// Forwards to [CodegenExtension::add_extension].
    ///
    /// ```
    /// use hugr_llvm::{extension::{PreludeCodegenExtension, DefaultPreludeCodegen}, CodegenExtsBuilder};
    /// let ext = PreludeCodegenExtension::from(DefaultPreludeCodegen);
    /// CodegenExtsBuilder::<hugr_core::Hugr>::default().add_extension(ext);
    /// ```
    pub fn add_extension(self, ext: impl CodegenExtension + 'a) -> Self {
        ext.add_extension(self)
    }

    /// Register a callback to map a [CustomType] to a [BasicTypeEnum].
    ///
    /// [CustomType]: hugr_core::types::CustomType
    /// [BasicTypeEnum]: inkwell::types::BasicTypeEnum
    pub fn custom_type(
        mut self,
        custom_type: CustomTypeKey,
        handler: impl LLVMCustomTypeFn<'a>,
    ) -> Self {
        self.type_converter.custom_type(custom_type, handler);
        self
    }

    /// Register a callback to emit a [ExtensionOp], keyed by fully
    /// qualified [OpName].
    pub fn extension_op(
        mut self,
        extension: ExtensionId,
        op: OpName,
        handler: impl ExtensionOpFn<'a, H>,
    ) -> Self {
        self.extension_op_handlers
            .extension_op(extension, op, handler);
        self
    }

    /// Register callbacks to emit [ExtensionOp]s that match the
    /// definitions generated by `Op`s impl of [strum::IntoEnumIterator]>
    pub fn simple_extension_op<Op: MakeOpDef + IntoEnumIterator>(
        mut self,
        handler: impl 'a
            + for<'c> Fn(
                &mut EmitFuncContext<'c, 'a, H>,
                EmitOpArgs<'c, '_, ExtensionOp, H>,
                Op,
            ) -> Result<()>,
    ) -> Self {
        self.extension_op_handlers
            .simple_extension_op::<Op>(handler);
        self
    }

    /// Register a callback to materialise a constant implemented by `CC`.
    pub fn custom_const<CC: CustomConst>(
        mut self,
        handler: impl LoadConstantFn<'a, H, CC>,
    ) -> Self {
        self.load_constant_handlers.custom_const(handler);
        self
    }

    /// Consume `self` to return collections of callbacks for each of the
    /// supported keys.`
    pub fn finish(self) -> CodegenExtsMap<'a, H> {
        CodegenExtsMap {
            load_constant_handlers: Rc::new(self.load_constant_handlers),
            extension_op_handlers: Rc::new(self.extension_op_handlers),
            type_converter: Rc::new(self.type_converter),
        }
    }
}

/// The result of [CodegenExtsBuilder::finish]. Users are expected to
/// deconstruct this type, and consume the fields independently.
/// We expect to add further collections at a later date, and so this type is
/// marked `non_exhaustive`
#[derive(Default)]
#[non_exhaustive]
pub struct CodegenExtsMap<'a, H> {
    pub load_constant_handlers: Rc<LoadConstantsMap<'a, H>>,
    pub extension_op_handlers: Rc<ExtensionOpMap<'a, H>>,
    pub type_converter: Rc<TypeConverter<'a>>,
}

#[cfg(test)]
mod test {
    use hugr_core::{
        extension::prelude::{string_type, ConstString, PRELUDE_ID, PRINT_OP_ID, STRING_TYPE_NAME},
        Hugr,
    };
    use inkwell::{
        context::Context,
        types::BasicType,
        values::{BasicMetadataValueEnum, BasicValue},
    };
    use itertools::Itertools as _;

    use crate::{emit::libc::emit_libc_printf, CodegenExtsBuilder};

    #[test]
    fn types_with_lifetimes() {
        let n = "name_with_lifetime".to_string();

        let cem = CodegenExtsBuilder::<Hugr>::default()
            .custom_type((PRELUDE_ID, STRING_TYPE_NAME), |session, _| {
                let ctx = session.iw_context();
                Ok(ctx
                    .get_struct_type(n.as_ref())
                    .unwrap_or_else(|| ctx.opaque_struct_type(n.as_ref()))
                    .as_basic_type_enum())
            })
            .finish();

        let ctx = Context::create();

        let ty = cem
            .type_converter
            .session(&ctx)
            .llvm_type(&string_type())
            .unwrap()
            .into_struct_type();
        let ty_n = ty.get_name().unwrap().to_str().unwrap();
        assert_eq!(ty_n, n);
    }

    #[test]
    fn custom_const_lifetime_of_context() {
        let ctx = Context::create();

        let _ = CodegenExtsBuilder::<Hugr>::default()
            .custom_const::<ConstString>(|_, konst| {
                Ok(ctx
                    .const_string(konst.value().as_bytes(), true)
                    .as_basic_value_enum())
            })
            .finish();
    }

    #[test]
    fn extension_op_lifetime() {
        let ctx = Context::create();

        let _ = CodegenExtsBuilder::<Hugr>::default()
            .extension_op(PRELUDE_ID, PRINT_OP_ID, |context, args| {
                let mut print_args: Vec<BasicMetadataValueEnum> =
                    vec![ctx.const_string("%s".as_bytes(), true).into()];
                print_args.extend(args.inputs.into_iter().map_into::<BasicMetadataValueEnum>());
                emit_libc_printf(context, &print_args)?;
                args.outputs.finish(context.builder(), [])
            })
            .finish();
    }
}
