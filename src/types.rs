use std::fmt::Display;
use std::iter;
use std::rc::Rc;

use anyhow::{anyhow, Result};
use delegate::delegate;
use hugr::types::{SumType, Type};
use hugr::{types::TypeRow, HugrView};
use inkwell::builder::Builder;
use inkwell::types::{self as iw, AnyType, AsTypeRef, IntType};
use inkwell::values::{BasicValue, BasicValueEnum, IntValue, StructValue};
use inkwell::AddressSpace;
use inkwell::{
    context::Context,
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, StructType},
};
use itertools::{zip_eq, Itertools};
use llvm_sys_140::prelude::LLVMTypeRef;

use super::custom::CodegenExtsMap;

/// A type alias for a hugr function type. We use this to disambiguate from
/// the LLVM [iw::FunctionType].
pub type HugrFuncType = hugr::types::FunctionType;

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
pub struct TypingSession<'c, H: HugrView> {
    tc: Rc<TypeConverter<'c>>,
    extensions: Rc<CodegenExtsMap<'c, H>>,
}

impl<'c, H: HugrView> TypingSession<'c, H> {
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
            TypeEnum::Function(ref func_ty) => Ok(self
                .llvm_func_type(func_ty)?
                .ptr_type(AddressSpace::default()) // Note: deprecated in LLVM >= 15
                .into()),

            x => Err(anyhow!("Invalid type: {:?}", x)),
        }
    }

    /// Convert a [HugrFuncType] into an LLVM [iw::FunctionType].
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
    pub fn session<H: HugrView>(
        self: Rc<Self>,
        exts: Rc<CodegenExtsMap<'c, H>>,
    ) -> TypingSession<'c, H> {
        TypingSession::new(self, exts)
    }

    /// Convert a [HugrType] into an LLVM [Type](BasicTypeEnum).
    pub fn llvm_type<H: HugrView>(
        self: Rc<Self>,
        extensions: Rc<CodegenExtsMap<'c, H>>,
        hugr_type: &HugrType,
    ) -> Result<BasicTypeEnum<'c>> {
        self.session(extensions).llvm_type(hugr_type)
    }

    /// Convert a [HugrFuncType] into an LLVM [iw::FunctionType].
    pub fn llvm_func_type<H: HugrView>(
        self: Rc<Self>,
        extensions: Rc<CodegenExtsMap<'c, H>>,
        hugr_type: &HugrFuncType,
    ) -> Result<iw::FunctionType<'c>> {
        self.session(extensions).llvm_func_type(hugr_type)
    }

    /// Convert a hugr [HugrSumType] into an LLVM [LLVMSumType].
    pub fn llvm_sum_type<H: HugrView>(
        self: Rc<Self>,
        extensions: Rc<CodegenExtsMap<'c, H>>,
        sum_type: HugrSumType,
    ) -> Result<LLVMSumType<'c>> {
        self.session(extensions).llvm_sum_type(sum_type)
    }
}

/// The opaque representation of a hugr [SumType].
///
/// Using the public methods of this type one emit "tag"s,"untag"s, and
/// "get_tag"s while not exposing the underlying LLVM representation.
///
/// We offer impls of [BasicType] and parent traits.
#[derive(Debug)]
pub struct LLVMSumType<'c>(StructType<'c>, SumType);

impl<'c> LLVMSumType<'c> {
    /// Attempt to create a new `LLVMSumType` from a [HugrSumType].
    pub fn try_new<H: HugrView>(session: &TypingSession<'c, H>, sum_type: SumType) -> Result<Self> {
        let mut i = 0;
        let (sum_type_ref, session_ref) = (&sum_type, &session);
        let variants = iter::from_fn(move || {
            let r = sum_type_ref.get_variant(i).map(|tr| {
                tr.iter()
                    .map(|t| session_ref.llvm_type(t))
                    .collect::<Result<Vec<_>>>()
            });
            i += 1;
            r
        })
        .collect::<Result<Vec<_>>>()?;
        assert!(variants.len() < u32::MAX as usize);
        let htf = Self::sum_type_has_tag_field(&sum_type);
        let types = htf
            .then_some(session.iw_context().i32_type().as_basic_type_enum())
            .into_iter()
            .chain(
                variants
                    .iter()
                    .map(|lty_vec| session.iw_context().struct_type(lty_vec, false).into()),
            )
            .collect_vec();
        Ok(Self(
            session.iw_context().struct_type(&types, false),
            sum_type,
        ))
    }

    /// Returns an LLVM constant value of `undef`.
    pub fn get_undef(&self) -> impl BasicValue<'c> {
        self.0.get_undef()
    }

    /// Returns an LLVM constant value of `poison`.
    pub fn get_poison(&self) -> impl BasicValue<'c> {
        self.0.get_poison()
    }

    /// Emit instructions to read the tag of a value of type `LLVMSumType`.
    ///
    /// The type of the value is that returned by [LLVMSumType::get_tag_type].
    pub fn build_get_tag(
        &self,
        builder: &Builder<'c>,
        v: impl BasicValue<'c>,
    ) -> Result<IntValue<'c>> {
        let struct_value: StructValue<'c> = v
            .as_basic_value_enum()
            .try_into()
            .map_err(|_| anyhow!("Not a struct type"))?;
        if self.has_tag_field() {
            Ok(builder
                .build_extract_value(struct_value, 0, "")?
                .into_int_value())
        } else {
            Ok(self.get_tag_type().const_int(0, false))
        }
    }

    /// Emit instructions to read the inner values of a value of type
    /// `LLVMSumType`, on the assumption that it's tag is `tag`.
    ///
    /// If it's tag is not `tag`, the returned values will be poison.
    pub fn build_untag(
        &self,
        builder: &Builder<'c>,
        tag: u32,
        v: impl BasicValue<'c>,
    ) -> Result<Vec<BasicValueEnum<'c>>> {
        debug_assert!((tag as usize) < self.1.num_variants());
        debug_assert!(v.as_basic_value_enum().get_type() == self.0.as_basic_type_enum());

        let v: StructValue<'c> = builder
            .build_extract_value(
                v.as_basic_value_enum().into_struct_value(),
                self.get_variant_index(tag),
                "",
            )?
            .into_struct_value();
        let r = (0..v.get_type().count_fields())
            .map(|i| Ok(builder.build_extract_value(v, i, "")?.as_basic_value_enum()))
            .collect::<Result<Vec<_>>>()?;
        debug_assert_eq!(r.len(), self.num_fields(tag).unwrap());
        Ok(r)
    }

    /// Emit instructions to build a value of type `LLVMSumType`, being of variant `tag`.
    pub fn build_tag(
        &self,
        builder: &Builder<'c>,
        tag: u32,
        vs: Vec<BasicValueEnum<'c>>,
    ) -> Result<BasicValueEnum<'c>> {
        let expected_num_fields = self.num_fields(tag)?;
        if expected_num_fields != vs.len() {
            Err(anyhow!("LLVMSumType::build: wrong number of fields: expected: {expected_num_fields} actual: {}", vs.len()))?
        }
        let variant_index = self.get_variant_index(tag);
        let row_t = self
            .0
            .get_field_type_at_index(variant_index)
            .ok_or(anyhow!("LLVMSumType::build: no field type at index"))
            .and_then(|row_t| {
                if !row_t.is_struct_type() {
                    Err(anyhow!("LLVMSumType::build"))?
                }
                Ok(row_t.into_struct_type())
            })?;
        debug_assert!(zip_eq(vs.iter(), row_t.get_field_types().into_iter())
            .all(|(lhs, rhs)| lhs.as_basic_value_enum().get_type() == rhs));
        let mut row_v = row_t.get_undef();
        for (i, val) in vs.into_iter().enumerate() {
            row_v = builder
                .build_insert_value(row_v, val, i as u32, "")?
                .into_struct_value();
        }
        let mut sum_v = self.get_poison().as_basic_value_enum().into_struct_value();
        if self.has_tag_field() {
            sum_v = builder
                .build_insert_value(
                    sum_v,
                    self.get_tag_type().const_int(tag as u64, false),
                    0u32,
                    "",
                )?
                .into_struct_value();
        }
        Ok(builder
            .build_insert_value(sum_v, row_v, variant_index, "")?
            .as_basic_value_enum())
    }

    /// Get the type of the value that would be returned by `build_get_tag`.
    pub fn get_tag_type(&self) -> IntType<'c> {
        self.0.get_context().i32_type()
    }

    fn sum_type_has_tag_field(st: &SumType) -> bool {
        st.num_variants() >= 2
    }

    fn has_tag_field(&self) -> bool {
        Self::sum_type_has_tag_field(&self.1)
    }

    fn get_variant_index(&self, tag: u32) -> u32 {
        tag + (if self.has_tag_field() { 1 } else { 0 })
    }

    fn num_fields(&self, tag: u32) -> Result<usize> {
        self.1
            .get_variant(tag as usize)
            .ok_or(anyhow!("Bad variant index"))
            .map(TypeRow::len)
    }
}

impl<'c> From<LLVMSumType<'c>> for BasicTypeEnum<'c> {
    fn from(value: LLVMSumType<'c>) -> Self {
        value.0.as_basic_type_enum()
    }
}

impl<'c> Display for LLVMSumType<'c> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

unsafe impl<'c> AsTypeRef for LLVMSumType<'c> {
    fn as_type_ref(&self) -> LLVMTypeRef {
        self.0.as_type_ref()
    }
}

unsafe impl<'c> AnyType<'c> for LLVMSumType<'c> {}

unsafe impl<'c> BasicType<'c> for LLVMSumType<'c> {}

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
    #[case(5, Type::new_sum([vec![INT_TYPES[2].clone()].into()]))]
    #[case(6, Type::new_sum([vec![INT_TYPES[6].clone(),Type::new_unit_sum(1)].into(), vec![Type::new_unit_sum(2), INT_TYPES[2].clone()].into()]))]
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
