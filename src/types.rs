use std::rc::Rc;

use anyhow::{anyhow, Result};
use delegate::delegate;
use hugr::types::{Signature, SumType, Type};
use inkwell::types::FunctionType;
use inkwell::AddressSpace;
use inkwell::{
    context::Context,
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum},
};

pub use crate::sum::LLVMSumType;

use super::custom::CodegenExtsMap;

/// A type alias for a hugr function type. We use this to disambiguate from
/// the LLVM [FunctionType].
pub type HugrFuncType = hugr::types::Signature;

/// A type alias for a hugr type. We use this to disambiguate from LLVM types.
pub type HugrType = Type;

/// A type alias for a hugr sum type.
pub type HugrSumType = SumType;

/// This type is mostly vestigal, it does very little but hold a &[Context].
///
/// I had thought it would grow some configuration that it would use while
/// constructing a [TypingSession].
#[derive(Copy, Clone, Debug)]
pub struct TypeConverter<'c> {
    context: &'c Context,
}

/// A type that holds [Rc] shared pointers to everything needed to convert from
/// a hugr [HugrType] to an LLVM [Type](inkwell::types).
pub struct TypingSession<'c, H> {
    tc: Rc<TypeConverter<'c>>,
    extensions: Rc<CodegenExtsMap<'c, H>>,
}

impl<'c, H> TypingSession<'c, H> {
    delegate! {
        to self.tc {
            /// Returns a reference to the inner [Context].
            pub fn iw_context(&self) -> &'c Context;
        }
    }

    /// Creates a new `TypingSession`.
    pub fn new(tc: Rc<TypeConverter<'c>>, extensions: Rc<CodegenExtsMap<'c, H>>) -> Self {
        TypingSession { tc, extensions }
    }

    /// Convert a [HugrType] into an LLVM [Type](BasicTypeEnum).
    pub fn llvm_type(&self, hugr_type: &HugrType) -> Result<BasicTypeEnum<'c>> {
        use hugr::types::TypeEnum;
        match hugr_type.as_type_enum() {
            TypeEnum::Extension(ref custom_type) => self.extensions.llvm_type(self, custom_type),
            TypeEnum::Sum(sum) => self.llvm_sum_type(sum.clone()).map(Into::into),
            TypeEnum::Function(func_ty) => {
                let func_ty: Signature = func_ty.as_ref().clone().try_into()?;
                Ok(self
                    .llvm_func_type(&func_ty)?
                    .ptr_type(AddressSpace::default()) // Note: deprecated in LLVM >= 15
                    .into())
            }
            x => Err(anyhow!("Invalid type: {:?}", x)),
        }
    }

    /// Convert a [HugrFuncType] into an LLVM [FunctionType].
    pub fn llvm_func_type(
        &self,
        hugr_type: &HugrFuncType,
    ) -> Result<inkwell::types::FunctionType<'c>> {
        let args = hugr_type
            .input()
            .iter()
            .map(|x| self.llvm_type(x).map(Into::<BasicMetadataTypeEnum>::into))
            .collect::<Result<Vec<_>>>()?;
        let res_unpacked = hugr_type
            .output()
            .iter()
            .map(|x| self.llvm_type(x))
            .collect::<Result<Vec<_>>>()?;

        Ok(match res_unpacked.as_slice() {
            &[] => self.iw_context().void_type().fn_type(&args, false),
            [res] => res.fn_type(&args, false),
            ress => self
                .iw_context()
                .struct_type(ress, false)
                .fn_type(&args, false),
        })
    }

    /// Convert a hugr [HugrSumType] into an LLVM [LLVMSumType].
    pub fn llvm_sum_type(&self, sum_type: HugrSumType) -> Result<LLVMSumType<'c>> {
        LLVMSumType::try_new(self, sum_type)
    }
}

impl<'c> TypeConverter<'c> {
    /// Create a new `TypeConverter`.
    pub fn new(context: &'c Context) -> Rc<Self> {
        Self { context }.into()
    }

    /// Returns a reference to the inner [Context].
    pub fn iw_context(&self) -> &'c Context {
        self.context
    }

    /// Creates a new [TypingSession].
    pub fn session<H>(self: Rc<Self>, exts: Rc<CodegenExtsMap<'c, H>>) -> TypingSession<'c, H> {
        TypingSession::new(self, exts)
    }

    /// Convert a [HugrType] into an LLVM [Type](BasicTypeEnum).
    pub fn llvm_type<H>(
        self: Rc<Self>,
        extensions: Rc<CodegenExtsMap<'c, H>>,
        hugr_type: &HugrType,
    ) -> Result<BasicTypeEnum<'c>> {
        self.session(extensions).llvm_type(hugr_type)
    }

    /// Convert a [HugrFuncType] into an LLVM [FunctionType].
    pub fn llvm_func_type<H>(
        self: Rc<Self>,
        extensions: Rc<CodegenExtsMap<'c, H>>,
        hugr_type: &HugrFuncType,
    ) -> Result<FunctionType<'c>> {
        self.session(extensions).llvm_func_type(hugr_type)
    }

    /// Convert a hugr [HugrSumType] into an LLVM [LLVMSumType].
    pub fn llvm_sum_type<H>(
        self: Rc<Self>,
        extensions: Rc<CodegenExtsMap<'c, H>>,
        sum_type: HugrSumType,
    ) -> Result<LLVMSumType<'c>> {
        self.session(extensions).llvm_sum_type(sum_type)
    }
}

#[cfg(test)]
#[allow(drop_bounds)]
pub mod test {

    use hugr::{
        std_extensions::arithmetic::int_types::INT_TYPES,
        type_row,
        types::{SumType, Type},
    };

    use insta::assert_snapshot;
    use rstest::rstest;

    use crate::{custom::int::add_int_extensions, test::*, types::HugrFuncType};

    #[rstest]
    #[case(0,HugrFuncType::new(type_row!(Type::new_unit_sum(2)), type_row!()))]
    #[case(1, HugrFuncType::new(Type::new_unit_sum(1), Type::new_unit_sum(3)))]
    #[case(2,HugrFuncType::new(vec![], vec![Type::new_unit_sum(1), Type::new_unit_sum(1)]))]
    fn func_types(#[case] _id: i32, #[with(_id)] llvm_ctx: TestContext, #[case] ft: HugrFuncType) {
        assert_snapshot!(
            "func_type_to_llvm",
            llvm_ctx.get_typing_session().llvm_func_type(&ft).unwrap(),
            &ft.to_string()
        )
    }

    #[rstest]
    #[case(0, SumType::new_unary(0))]
    #[case(1, SumType::new_unary(1))]
    #[case(2,SumType::new([vec![Type::new_unit_sum(0), Type::new_unit_sum(1)], vec![Type::new_unit_sum(2), Type::new_unit_sum(3)]]))]
    #[case(3, SumType::new_unary(2))]
    fn sum_types(#[case] _id: i32, #[with(_id)] llvm_ctx: TestContext, #[case] st: SumType) {
        assert_snapshot!(
            "sum_type_to_llvm",
            llvm_ctx
                .get_typing_session()
                .llvm_sum_type(st.clone())
                .unwrap(),
            &st.to_string()
        )
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
        llvm_ctx.add_extensions(add_int_extensions);
        assert_snapshot!(
            "type_to_llvm",
            llvm_ctx.get_typing_session().llvm_type(&t).unwrap(),
            &t.to_string()
        );
    }
}
