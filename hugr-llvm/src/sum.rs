use crate::types::{HugrSumType, TypingSession};

use anyhow::{anyhow, Result};
use delegate::delegate;
use hugr_core::types::TypeRow;
use inkwell::{
    builder::Builder,
    context::Context,
    types::{AnyType, AsTypeRef, BasicType, BasicTypeEnum, IntType, StructType},
    values::{AnyValue, AsValueRef, BasicValue, BasicValueEnum, IntValue, StructValue},
};
use itertools::{zip_eq, Itertools};

fn get_variant_typerow(sum_type: &HugrSumType, tag: u32) -> Result<TypeRow> {
    sum_type
        .get_variant(tag as usize)
        .ok_or(anyhow!("Bad variant index {tag} in {sum_type}"))
        .and_then(|tr| Ok(TypeRow::try_from(tr.clone())?))
}

fn sum_type_has_tag_field(st: &HugrSumType) -> bool {
    st.num_variants() >= 2
}

/// The opaque representation of a [HugrSumType].
///
/// Using the public methods of this type one emit "tag"s,"untag"s, and
/// "get_tag"s while not exposing the underlying LLVM representation.
///
/// We offer impls of [BasicType] and parent traits.
#[derive(Debug, Clone)]
pub struct LLVMSumType<'c>(StructType<'c>, HugrSumType);

impl<'c> LLVMSumType<'c> {
    pub fn try_new2(
        context: &'c Context,
        variants: Vec<Vec<BasicTypeEnum<'c>>>,
        sum_type: HugrSumType,
    ) -> Result<Self> {
        let has_tag_field = sum_type_has_tag_field(&sum_type);
        let types = has_tag_field
            .then_some(context.i32_type().as_basic_type_enum())
            .into_iter()
            .chain(
                variants
                    .iter()
                    .map(|lty_vec| context.struct_type(lty_vec, false).into()),
            )
            .collect_vec();
        Ok(Self(context.struct_type(&types, false), sum_type.clone()))
    }
    /// Attempt to create a new `LLVMSumType` from a [HugrSumType].
    pub fn try_new(session: &TypingSession<'c, '_>, sum_type: HugrSumType) -> Result<Self> {
        assert!(sum_type.num_variants() < u32::MAX as usize);
        let variants = (0..sum_type.num_variants())
            .map(|i| {
                let tr = get_variant_typerow(&sum_type, i as u32)?;
                tr.iter()
                    .map(|t| session.llvm_type(t))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Self::try_new2(session.iw_context(), variants, sum_type)
    }

    /// Returns an LLVM constant value of `undef`.
    pub fn get_undef(&self) -> impl BasicValue<'c> {
        self.0.get_undef()
    }

    /// Returns an LLVM constant value of `poison`.
    pub fn get_poison(&self) -> impl BasicValue<'c> {
        self.0.get_poison()
    }

    /// Emit instructions to build a value of type `LLVMSumType`, being of variant `tag`.
    pub fn build_tag(
        &self,
        builder: &Builder<'c>,
        tag: usize,
        vs: Vec<BasicValueEnum<'c>>,
    ) -> Result<BasicValueEnum<'c>> {
        let expected_num_fields = self.variant_num_fields(tag)?;
        if expected_num_fields != vs.len() {
            Err(anyhow!("LLVMSumType::build: wrong number of fields: expected: {expected_num_fields} actual: {}", vs.len()))?
        }
        let variant_field_index = self.get_variant_field_index(tag);
        let row_t = self
            .0
            .get_field_type_at_index(variant_field_index as u32)
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
            .build_insert_value(sum_v, row_v, variant_field_index as u32, "")?
            .as_basic_value_enum())
    }

    /// Get the type of the value that would be returned by `build_get_tag`.
    pub fn get_tag_type(&self) -> IntType<'c> {
        self.0.get_context().i32_type()
    }

    fn has_tag_field(&self) -> bool {
        sum_type_has_tag_field(&self.1)
    }

    fn get_variant_field_index(&self, tag: usize) -> usize {
        tag + (if self.has_tag_field() { 1 } else { 0 })
    }

    fn variant_num_fields(&self, tag: usize) -> Result<usize> {
        self.get_variant(tag).map(|x| x.len())
    }

    pub fn get_variant(&self, tag: usize) -> Result<TypeRow> {
        let tr = self
            .1
            .get_variant(tag)
            .ok_or(anyhow!("Bad variant index {tag} in {}", self.1))?
            .to_owned();
        tr.try_into()
            .map_err(|rv| anyhow!("Row variable in {}: {rv}", self.1))
    }

    delegate! {
        to self.1 {
            pub(self) fn num_variants(&self) -> usize;
        }
    }
}

impl<'c> From<LLVMSumType<'c>> for BasicTypeEnum<'c> {
    fn from(value: LLVMSumType<'c>) -> Self {
        value.0.as_basic_type_enum()
    }
}

impl std::fmt::Display for LLVMSumType<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

unsafe impl AsTypeRef for LLVMSumType<'_> {
    fn as_type_ref(&self) -> inkwell::llvm_sys::prelude::LLVMTypeRef {
        self.0.as_type_ref()
    }
}

unsafe impl<'c> AnyType<'c> for LLVMSumType<'c> {}

unsafe impl<'c> BasicType<'c> for LLVMSumType<'c> {}

/// A Value equivalent of [LLVMSumType]. Represents a [HugrSumType] Value on the
/// wire, offering functions for deconstructing such Values.
#[derive(Debug)]
pub struct LLVMSumValue<'c>(StructValue<'c>, LLVMSumType<'c>);

impl<'c> From<LLVMSumValue<'c>> for BasicValueEnum<'c> {
    fn from(value: LLVMSumValue<'c>) -> Self {
        value.0.as_basic_value_enum()
    }
}

unsafe impl AsValueRef for LLVMSumValue<'_> {
    fn as_value_ref(&self) -> inkwell::llvm_sys::prelude::LLVMValueRef {
        self.0.as_value_ref()
    }
}

unsafe impl<'c> AnyValue<'c> for LLVMSumValue<'c> {}

unsafe impl<'c> BasicValue<'c> for LLVMSumValue<'c> {}

impl<'c> LLVMSumValue<'c> {
    pub fn try_new(value: impl BasicValue<'c>, sum_type: LLVMSumType<'c>) -> Result<Self> {
        let value: StructValue<'c> = value
            .as_basic_value_enum()
            .try_into()
            .map_err(|_| anyhow!("Not a StructValue"))?;
        let (v_t, st_t) = (
            value.get_type().as_basic_type_enum(),
            sum_type.as_basic_type_enum(),
        );
        if v_t != st_t {
            Err(anyhow!(
                "LLVMSumValue::new: type of value does not match sum_type: {v_t} != {st_t}"
            ))?
        }
        Ok(Self(value, sum_type))
    }

    /// Emit instructions to read the tag of a value of type `LLVMSumType`.
    ///
    /// The type of the value is that returned by [LLVMSumType::get_tag_type].
    pub fn build_get_tag(&self, builder: &Builder<'c>) -> Result<IntValue<'c>> {
        if self.1.has_tag_field() {
            Ok(builder.build_extract_value(self.0, 0, "")?.into_int_value())
        } else {
            Ok(self.1.get_tag_type().const_int(0, false))
        }
    }

    /// Emit instructions to read the inner values of a value of type
    /// `LLVMSumType`, on the assumption that it's tag is `tag`.
    ///
    /// If it's tag is not `tag`, the returned values will be poison.
    pub fn build_untag(
        &self,
        builder: &Builder<'c>,
        tag: usize,
    ) -> Result<Vec<BasicValueEnum<'c>>> {
        debug_assert!(tag < self.1 .1.num_variants());

        let v = builder
            .build_extract_value(self.0, self.1.get_variant_field_index(tag) as u32, "")?
            .into_struct_value();
        let r = (0..v.get_type().count_fields())
            .map(|i| Ok(builder.build_extract_value(v, i, "")?))
            .collect::<Result<Vec<_>>>()?;
        debug_assert_eq!(r.len(), self.1.variant_num_fields(tag).unwrap());
        Ok(r)
    }

    pub fn build_destructure(
        &self,
        builder: &Builder<'c>,
        mut handler: impl FnMut(&Builder<'c>, usize, Vec<BasicValueEnum<'c>>) -> Result<()>,
    ) -> Result<()> {
        let orig_bb = builder
            .get_insert_block()
            .ok_or(anyhow!("No current insertion point"))?;
        let context = orig_bb.get_context();
        let mut last_bb = orig_bb;
        let tag_ty = self.1.get_tag_type();

        let mut cases = vec![];

        for var_i in 0..self.1.num_variants() {
            let bb = context.insert_basic_block_after(last_bb, "");
            last_bb = bb;
            cases.push((tag_ty.const_int(var_i as u64, false), bb));

            builder.position_at_end(bb);
            let inputs = self.build_untag(builder, var_i)?;
            handler(builder, var_i, inputs)?;
        }

        builder.position_at_end(orig_bb);
        let tag = self.build_get_tag(builder)?;
        builder.build_switch(tag, cases[0].1, &cases[1..])?;

        Ok(())
    }

    delegate! {
        to self.1 {
            /// Get the type of the value that would be returned by `build_get_tag`.
            pub fn get_tag_type(&self) -> IntType<'c>;
        }
    }
}
