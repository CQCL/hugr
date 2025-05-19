use std::rc::Rc;

use anyhow::Result;
use delegate::delegate;
use hugr_core::extension::ExtensionId;
use hugr_core::types::{SumType, Type, TypeName};
use inkwell::types::FunctionType;
use inkwell::{context::Context, types::BasicTypeEnum};

use crate::custom::types::{LLVMCustomTypeFn, LLVMTypeMapping};
pub use crate::sum::LLVMSumType;
use crate::utils::type_map::TypeMap;

/// A type alias for a hugr function type. We use this to disambiguate from
/// the LLVM [`FunctionType`].
pub type HugrFuncType = hugr_core::types::Signature;

/// A type alias for a hugr type. We use this to disambiguate from LLVM types.
pub type HugrType = Type;

/// A type alias for a hugr sum type.
pub type HugrSumType = SumType;

/// A type that holds [Rc] shared pointers to everything needed to convert from
/// a hugr [`HugrType`] to an LLVM [Type](inkwell::types).
#[derive(Clone)]
pub struct TypingSession<'c, 'a> {
    iw_context: &'c Context,
    type_converter: Rc<TypeConverter<'a>>,
}

impl<'c, 'a> TypingSession<'c, 'a> {
    delegate! {
        to self.type_converter.clone() {
            /// Convert a [HugrType] into an LLVM [Type](BasicTypeEnum).
            pub fn llvm_type(&self, [self.clone()], hugr_type: &HugrType) -> Result<BasicTypeEnum<'c>>;
            /// Convert a [HugrFuncType] into an LLVM [FunctionType].
            pub fn llvm_func_type(&self, [self.clone()], hugr_type: &HugrFuncType) -> Result<FunctionType<'c>>;
            /// Convert a hugr [HugrSumType] into an LLVM [LLVMSumType].
            pub fn llvm_sum_type(&self, [self.clone()], hugr_type: HugrSumType) -> Result<LLVMSumType<'c>>;
        }
    }

    /// Creates a new `TypingSession`.
    #[must_use]
    pub fn new(iw_context: &'c Context, type_converter: Rc<TypeConverter<'a>>) -> Self {
        Self {
            iw_context,
            type_converter,
        }
    }

    /// Returns a reference to the inner [Context].
    #[must_use]
    pub fn iw_context(&self) -> &'c Context {
        self.iw_context
    }
}

#[derive(Default)]
pub struct TypeConverter<'a>(TypeMap<'a, LLVMTypeMapping<'a>>);

impl<'a> TypeConverter<'a> {
    pub(super) fn custom_type(
        &mut self,
        custom_type: (ExtensionId, TypeName),
        handler: impl LLVMCustomTypeFn<'a>,
    ) {
        self.0.set_callback(custom_type, handler);
    }

    pub fn llvm_type<'c>(
        self: Rc<Self>,
        context: TypingSession<'c, 'a>,
        hugr_type: &HugrType,
    ) -> Result<BasicTypeEnum<'c>> {
        self.0.map_type(hugr_type, context)
    }

    pub fn llvm_func_type<'c>(
        self: Rc<Self>,
        context: TypingSession<'c, 'a>,
        hugr_type: &HugrFuncType,
    ) -> Result<FunctionType<'c>> {
        self.0.map_function_type(hugr_type, context)
    }

    pub fn llvm_sum_type<'c>(
        self: Rc<Self>,
        context: TypingSession<'c, 'a>,
        hugr_type: HugrSumType,
    ) -> Result<LLVMSumType<'c>> {
        self.0.map_sum_type(&hugr_type, context)
    }

    #[must_use]
    pub fn session<'c>(self: Rc<Self>, iw_context: &'c Context) -> TypingSession<'c, 'a> {
        TypingSession::new(iw_context, self)
    }
}

#[cfg(test)]
#[allow(drop_bounds)]
pub mod test {

    use hugr_core::{
        std_extensions::arithmetic::int_types::INT_TYPES,
        type_row,
        types::{SumType, Type},
    };

    use insta::assert_snapshot;
    use rstest::rstest;

    use crate::{test::*, types::HugrFuncType};

    #[rstest]
    #[case(0,HugrFuncType::new(type_row!(Type::new_unit_sum(2)), type_row!()))]
    #[case(1, HugrFuncType::new(Type::new_unit_sum(1), Type::new_unit_sum(3)))]
    #[case(2,HugrFuncType::new(vec![], vec![Type::new_unit_sum(1), Type::new_unit_sum(1)]))]
    fn func_types(#[case] _id: i32, #[with(_id)] llvm_ctx: TestContext, #[case] ft: HugrFuncType) {
        assert_snapshot!(
            "func_type_to_llvm",
            llvm_ctx.get_typing_session().llvm_func_type(&ft).unwrap(),
            &ft.to_string()
        );
    }

    #[rstest]
    #[case(0, SumType::new_unary(0))]
    #[case(1, SumType::new_unary(1))]
    #[case(2,SumType::new([vec![Type::new_unit_sum(4), Type::new_unit_sum(1)], vec![Type::new_unit_sum(2), Type::new_unit_sum(3)]]))]
    #[case(3, SumType::new_unary(2))]
    fn sum_types(#[case] _id: i32, #[with(_id)] llvm_ctx: TestContext, #[case] st: SumType) {
        assert_snapshot!(
            "sum_type_to_llvm",
            llvm_ctx
                .get_typing_session()
                .llvm_sum_type(st.clone())
                .unwrap(),
            &st.to_string()
        );
    }

    #[rstest]
    #[case(0, INT_TYPES[0].clone())]
    #[case(1, INT_TYPES[3].clone())]
    #[case(2, INT_TYPES[4].clone())]
    #[case(3, INT_TYPES[5].clone())]
    #[case(4, INT_TYPES[6].clone())]
    #[case(5, Type::new_sum([vec![INT_TYPES[2].clone()]]))]
    #[case(6, Type::new_sum([vec![INT_TYPES[6].clone(),Type::new_unit_sum(1)], vec![Type::new_unit_sum(2), INT_TYPES[2].clone()]]))]
    #[case(7, Type::new_function(HugrFuncType::new(type_row!(Type::new_unit_sum(2)), Type::new_unit_sum(3))))]
    fn ext_types(#[case] _id: i32, #[with(_id)] mut llvm_ctx: TestContext, #[case] t: Type) {
        use crate::CodegenExtsBuilder;

        llvm_ctx.add_extensions(CodegenExtsBuilder::add_default_int_extensions);
        assert_snapshot!(
            "type_to_llvm",
            llvm_ctx.get_typing_session().llvm_type(&t).unwrap(),
            &t.to_string()
        );
    }
}
