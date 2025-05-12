use std::marker::PhantomData;

use itertools::Itertools as _;

use hugr_core::types::CustomType;

use anyhow::Result;
use inkwell::types::{BasicMetadataTypeEnum, BasicType as _, BasicTypeEnum, FunctionType};

pub use crate::utils::type_map::CustomTypeKey;

use crate::{
    sum::LLVMSumType,
    types::{HugrFuncType, HugrSumType, TypingSession},
    utils::type_map::TypeMapping,
};

pub trait LLVMCustomTypeFn<'a>:
    for<'c> Fn(TypingSession<'c, 'a>, &CustomType) -> Result<BasicTypeEnum<'c>> + 'a
{
}

impl<
    'a,
    F: for<'c> Fn(TypingSession<'c, 'a>, &CustomType) -> Result<BasicTypeEnum<'c>> + 'a + ?Sized,
> LLVMCustomTypeFn<'a> for F
{
}

#[derive(Default, Clone)]
pub struct LLVMTypeMapping<'a>(PhantomData<&'a ()>);

impl<'a> TypeMapping for LLVMTypeMapping<'a> {
    type InV<'c> = TypingSession<'c, 'a>;

    type OutV<'c> = BasicTypeEnum<'c>;

    type SumOutV<'c> = LLVMSumType<'c>;

    type FuncOutV<'c> = FunctionType<'c>;

    fn sum_into_out<'c>(&self, sum: Self::SumOutV<'c>) -> Self::OutV<'c> {
        sum.as_basic_type_enum()
    }

    fn func_into_out<'c>(&self, sum: Self::FuncOutV<'c>) -> Self::OutV<'c> {
        sum.ptr_type(Default::default()).as_basic_type_enum()
    }

    fn map_sum_type<'c>(
        &self,
        _sum_type: &HugrSumType,
        context: TypingSession<'c, 'a>,
        variants: impl IntoIterator<Item = Vec<Self::OutV<'c>>>,
    ) -> Result<Self::SumOutV<'c>> {
        LLVMSumType::try_new(context.iw_context(), variants.into_iter().collect_vec())
    }

    fn map_function_type<'c>(
        &self,
        _: &HugrFuncType,
        context: TypingSession<'c, 'a>,
        inputs: impl IntoIterator<Item = Self::OutV<'c>>,
        outputs: impl IntoIterator<Item = Self::OutV<'c>>,
    ) -> Result<Self::FuncOutV<'c>> {
        let iw_context = context.iw_context();
        let inputs: Vec<BasicMetadataTypeEnum<'c>> = inputs.into_iter().map_into().collect_vec();
        let outputs = outputs.into_iter().collect_vec();
        Ok(match outputs.as_slice() {
            &[] => iw_context.void_type().fn_type(&inputs, false),
            [res] => res.fn_type(&inputs, false),
            ress => iw_context.struct_type(ress, false).fn_type(&inputs, false),
        })
    }
}
