mod layout;

use std::{iter, slice};

use crate::types::{HugrSumType, TypingSession};

use anyhow::{Result, anyhow, bail, ensure};
use delegate::delegate;
use hugr_core::types::TypeRow;
use inkwell::{
    builder::Builder,
    context::Context,
    types::{AnyType, AsTypeRef, BasicType, BasicTypeEnum, IntType, StructType},
    values::{AnyValue, AsValueRef, BasicValue, BasicValueEnum, IntValue, StructValue},
};
use itertools::{Itertools as _, izip};

/// An elidable type is one that holds no information, for example `{}`, the
/// empty struct.
///
/// Currently the following types are elidable:
///   * Empty structs, which may be packed, unpacked, named, or unnamed
///   * Empty arrays of any type.
pub fn elidable_type<'c>(ty: impl BasicType<'c>) -> bool {
    let ty = ty.as_basic_type_enum();
    match ty {
        BasicTypeEnum::ArrayType(array_type) => array_type.is_empty(),
        BasicTypeEnum::StructType(struct_type) => struct_type.count_fields() == 0,
        _ => false,
    }
}

fn get_variant_typerow(sum_type: &HugrSumType, tag: u32) -> Result<TypeRow> {
    sum_type
        .get_variant(tag as usize)
        .ok_or(anyhow!("Bad variant index {tag} in {sum_type}"))
        .and_then(|tr| Ok(TypeRow::try_from(tr.clone())?))
}

/// Returns an `undef` value for any [`BasicType`].
fn basic_type_undef<'c>(t: impl BasicType<'c>) -> BasicValueEnum<'c> {
    let t = t.as_basic_type_enum();
    match t {
        BasicTypeEnum::ArrayType(t) => t.get_undef().as_basic_value_enum(),
        BasicTypeEnum::FloatType(t) => t.get_undef().as_basic_value_enum(),
        BasicTypeEnum::IntType(t) => t.get_undef().as_basic_value_enum(),
        BasicTypeEnum::PointerType(t) => t.get_undef().as_basic_value_enum(),
        BasicTypeEnum::StructType(t) => t.get_undef().as_basic_value_enum(),
        BasicTypeEnum::VectorType(t) => t.get_undef().as_basic_value_enum(),
        BasicTypeEnum::ScalableVectorType(t) => t.get_undef().as_basic_value_enum(),
    }
}

/// Returns an `poison` value for any [`BasicType`].
fn basic_type_poison<'c>(t: impl BasicType<'c>) -> BasicValueEnum<'c> {
    let t = t.as_basic_type_enum();
    match t {
        BasicTypeEnum::ArrayType(t) => t.get_poison().as_basic_value_enum(),
        BasicTypeEnum::FloatType(t) => t.get_poison().as_basic_value_enum(),
        BasicTypeEnum::IntType(t) => t.get_poison().as_basic_value_enum(),
        BasicTypeEnum::PointerType(t) => t.get_poison().as_basic_value_enum(),
        BasicTypeEnum::StructType(t) => t.get_poison().as_basic_value_enum(),
        BasicTypeEnum::VectorType(t) => t.get_poison().as_basic_value_enum(),
        BasicTypeEnum::ScalableVectorType(t) => t.get_poison().as_basic_value_enum(),
    }
}

#[derive(Debug, Clone, derive_more::Display)]
/// The opaque representation of a [`HugrSumType`].
///
/// Provides an `impl`s of `BasicType`, allowing interoperation with other
/// inkwell tools.
///
/// To obtain an [`LLVMSumType`] corresponding to a [`HugrSumType`] use
/// [`LLVMSumType::try_new`] or [`LLVMSumType::try_from_hugr_type`].
///
/// Any such [`LLVMSumType`] has a fixed underlying LLVM type, which can be
/// obtained by [`BasicType::as_basic_type_enum`] or [`LLVMSumType::value_type`].
/// Note this type is unspecified, and we go to some effort to ensure that it is
/// minimal and efficient. Users should not expect this type to remain the same
/// across versions.
///
/// Unit types such as empty structs(`{}`) are elided from the LLVM type where
/// possible. See [`elidable_type`] for the specification of which types are
/// elided.
///
/// Each [`LLVMSumType`] has an associated [`IntType`] tag type, which can be
/// obtained via [`LLVMSumType::tag_type`].
///
/// The value type [`LLVMSumValue`] represents values of this type. To obtain an
/// [`LLVMSumValue`] use [`LLVMSumType::build_tag`] or [`LLVMSumType::value`].
pub struct LLVMSumType<'c>(LLVMSumTypeEnum<'c>);

impl<'c> LLVMSumType<'c> {
    delegate! {
        to self.0 {
            /// The underlying LLVM type.
            #[must_use] pub fn value_type(&self) -> BasicTypeEnum<'c>;
            /// The type of the value that would be returned by [LLVMSumValue::build_get_tag].
            #[must_use] pub fn tag_type(&self) -> IntType<'c>;
            /// The number of variants in the represented [HugrSumType].
            #[must_use] pub fn num_variants(&self) -> usize;
            /// The number of fields in the `tag`th variant of the represented [HugrSumType].
            /// Panics if `tag` is out of bounds.
            #[must_use] pub fn num_fields_for_variant(&self, tag: usize) -> usize;
            /// The LLVM types representing the fields in the `tag` variant of the represented [HugrSumType].
            /// Panics if `tag` is out of bounds.
            #[must_use] pub fn fields_for_variant(&self, tag: usize) -> &[BasicTypeEnum<'c>];
        }
    }

    /// Constructs a new [`LLVMSumType`] from a [`HugrSumType`], using `session` to
    /// determine the types of the fields.
    ///
    /// Returns an error if the type of any field cannot be converted by
    /// `session`, or if `sum_type` has no variants.
    pub fn try_from_hugr_type(
        session: &TypingSession<'c, '_>,
        sum_type: HugrSumType,
    ) -> Result<Self> {
        let variants = (0..sum_type.num_variants())
            .map(|i| {
                let tr = get_variant_typerow(&sum_type, i as u32)?;
                tr.iter()
                    .map(|t| session.llvm_type(t))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Self::try_new(session.iw_context(), variants)
    }

    /// Constructs a new [`LLVMSumType`] from a `Vec` of variants.
    /// Each variant is a `Vec` of LLVM types each corresponding to a field in the sum.
    ///
    /// Returns an error if `variant_types` is empty;
    pub fn try_new(
        context: &'c Context,
        variant_types: impl Into<Vec<Vec<BasicTypeEnum<'c>>>>,
    ) -> Result<Self> {
        Ok(Self(LLVMSumTypeEnum::try_new(
            context,
            variant_types.into(),
        )?))
    }

    /// Returns an constant `undef` value of the underlying LLVM type.
    #[must_use]
    pub fn get_undef(&self) -> impl BasicValue<'c> + use<'c> {
        basic_type_undef(self.0.value_type())
    }

    /// Returns an constant `poison` value of the underlying LLVM type.
    #[must_use]
    pub fn get_poison(&self) -> impl BasicValue<'c> + use<'c> {
        basic_type_poison(self.0.value_type())
    }

    /// Emits instructions to construct an [`LLVMSumValue`] of this type. The
    /// value will represent the `tag`th variant.
    pub fn build_tag(
        &self,
        builder: &Builder<'c>,
        tag: usize,
        vs: Vec<BasicValueEnum<'c>>,
    ) -> Result<LLVMSumValue<'c>> {
        self.value(self.0.build_tag(builder, tag, vs)?)
    }

    /// Returns an [`LLVMSumValue`] of this type.
    ///
    /// Returns an error if `value.get_type() != self.value_type()`.
    pub fn value(&self, value: impl BasicValue<'c>) -> Result<LLVMSumValue<'c>> {
        LLVMSumValue::try_new(value, self.clone())
    }
}

/// The internal representation of a [`HugrSumType`].
///
/// This type is not public, so that it can be changed without breaking users.
#[derive(Debug, Clone)]
enum LLVMSumTypeEnum<'c> {
    /// A Sum type with no variants. It's representation is unspecified.
    ///
    /// Values of this type can only be constructed by [`get_poison`].
    Void { tag_type: IntType<'c> },
    /// A Sum type with a single variant and all-elidable fields.
    /// Represented by `{}`
    /// Values of this type contain no information, so they never need to be
    /// stored. One can always use `undef` to materialize a value of this type.
    /// Represented by an empty struct.
    Unit {
        /// The LLVM types of the fields. One entry for each field in the Hugr
        /// variant. Each field must be elidable.
        field_types: Vec<BasicTypeEnum<'c>>,
        /// The LLVM type of the tag. Always `i1` for now.
        /// We store it here so because otherwise we would need a &[Context] to
        /// construct it.
        tag_type: IntType<'c>,
        /// The underlying LLVM type. Always `{}` for now.
        value_type: StructType<'c>,
    },
    /// A Sum type with more than one variant and all elidable fields.
    /// Values of this type contain information only in their tag.
    /// Represented by the value of their tag.
    NoFields {
        /// The LLVM types of the fields. One entry for each variant, with that
        /// entry containing one entry per Hugr field in the variant. Each field
        /// must be elidable.
        variant_types: Vec<Vec<BasicTypeEnum<'c>>>,
        /// The underlying LLVM type. For now it is the smallest integer type
        /// large enough to index the variants.
        value_type: IntType<'c>,
    },
    /// A Sum type with a single variant and exactly one non-elidable field.
    /// Values of this type contain information only in the value of their
    /// non-elidable field.
    /// Represented by the value of their non-elidable field.
    SingleVariantSingleField {
        /// The LLVM types of the fields. One entry for each Hugr field in the single
        /// variant.
        field_types: Vec<BasicTypeEnum<'c>>,
        /// The index into `variant_types` of the non-elidable field.
        field_index: usize,
        /// The LLVM type of the tag. Always `i1` for now.
        /// We store it here so because otherwise we would need a &[Context] to
        /// construct it.
        tag_type: IntType<'c>,
    },
    /// A Sum type with a single variant and more than one non-elidable field.
    /// Values of this type contain information in the values of their
    /// non-elidable fields.
    /// Represented by a struct containing each non-elidable field.
    SingleVariantMultiField {
        /// The LLVM types of the fields. One entry for each Hugr field in the
        /// single variant.
        field_types: Vec<BasicTypeEnum<'c>>,
        /// For each field, an index into the fields of `value_type`
        field_indices: Vec<Option<usize>>,
        /// The LLVM type of the tag. Always `i1` for now.
        /// We store it here so because otherwise we would need a &[Context] to
        /// construct it.
        tag_type: IntType<'c>,
        /// The underlying LLVM type. Has one field for each non-elidable field
        /// in the single variant.
        value_type: StructType<'c>,
    },
    /// A Sum type with multiple variants and at least one non-elidable field.
    /// Values of this type contain information in their tag and in the values
    /// of their non-elidable fields.
    /// Represented by a struct containing a tag and fields enough to store the
    /// non-elidable fields of any one variant.
    MultiVariant {
        /// The LLVM types of the fields. One entry for each variant, with that
        /// entry containing one entry per Hugr field in the variant.
        variant_types: Vec<Vec<BasicTypeEnum<'c>>>,
        /// For each field in each variant, an index into the fields of `value_type`.
        field_indices: Vec<Vec<Option<usize>>>,
        /// The underlying LLVM type. The first field is of `tag_type`. The
        /// remaining fields are minimal such that any one variant can be
        /// injectively mapped into those fields.
        value_type: StructType<'c>,
    },
}

/// Returns the smallest width for an integer type to be able to represent values smaller than `num_variants
fn tag_width_for_num_variants(num_variants: usize) -> u32 {
    debug_assert!(num_variants >= 1);
    if num_variants == 1 {
        return 1;
    }
    (num_variants - 1).ilog2() + 1
}

impl<'c> LLVMSumTypeEnum<'c> {
    /// Constructs a new [`LLVMSumTypeEnum`] from a `Vec` of variants.
    /// Each variant is a `Vec` of LLVM types each corresponding to a field in the sum.
    pub fn try_new(
        context: &'c Context,
        variant_types: Vec<Vec<BasicTypeEnum<'c>>>,
    ) -> Result<Self> {
        let result = match variant_types.len() {
            0 => Self::Void {
                tag_type: context.bool_type(),
            },
            1 => {
                let variant_types = variant_types.into_iter().exactly_one().unwrap();
                let (fields, field_indices) =
                    layout::layout_variants(slice::from_ref(&variant_types));
                let field_indices = field_indices.into_iter().exactly_one().unwrap();
                match fields.len() {
                    0 => Self::Unit {
                        field_types: variant_types,
                        tag_type: context.bool_type(),
                        value_type: context.struct_type(&[], false),
                    },
                    1 => {
                        let field_index = field_indices
                            .into_iter()
                            .enumerate()
                            .filter_map(|(i, f_i)| f_i.is_some().then_some(i))
                            .exactly_one()
                            .unwrap();
                        Self::SingleVariantSingleField {
                            field_types: variant_types,
                            field_index,
                            tag_type: context.bool_type(),
                        }
                    }
                    _num_fields => Self::SingleVariantMultiField {
                        field_types: variant_types,
                        field_indices,
                        tag_type: context.bool_type(),
                        value_type: context.struct_type(&fields, false),
                    },
                }
            }
            num_variants => {
                let (mut fields, field_indices) = layout::layout_variants(&variant_types);
                let tag_type =
                    context.custom_width_int_type(tag_width_for_num_variants(num_variants));
                if fields.is_empty() {
                    Self::NoFields {
                        variant_types,
                        value_type: tag_type,
                    }
                } else {
                    // prefix the tag fields
                    fields.insert(0, tag_type.into());
                    let value_type = context.struct_type(&fields, false);
                    Self::MultiVariant {
                        variant_types,
                        field_indices,
                        value_type,
                    }
                }
            }
        };
        Ok(result)
    }

    /// Emit instructions to build a value of type `LLVMSumType`, being of variant `tag`.
    ///
    /// Returns an error if:
    ///   * `tag` is out of bounds
    ///   * `vs` does not have a length equal to the length of the `tag`th
    ///     variant of the represented Hugr type.
    ///   * Any entry of `vs` does not have the expected type.
    pub fn build_tag(
        &self,
        builder: &Builder<'c>,
        tag: usize,
        vs: Vec<BasicValueEnum<'c>>,
    ) -> Result<BasicValueEnum<'c>> {
        ensure!(tag < self.num_variants());
        ensure!(vs.len() == self.num_fields_for_variant(tag));
        ensure!(iter::zip(&vs, self.fields_for_variant(tag)).all(|(x, y)| &x.get_type() == y));
        let value = match self {
            Self::Void { .. } => bail!("Can't tag an empty sum"),
            Self::Unit { value_type, .. } => value_type.get_undef().as_basic_value_enum(),
            Self::NoFields { value_type, .. } => value_type
                .const_int(tag as u64, false)
                .as_basic_value_enum(),
            Self::SingleVariantSingleField { field_index, .. } => vs[*field_index],
            Self::SingleVariantMultiField {
                value_type,
                field_indices,
                ..
            } => {
                let mut value = value_type.get_poison();
                for (mb_i, v) in itertools::zip_eq(field_indices, vs) {
                    if let Some(i) = mb_i {
                        value = builder
                            .build_insert_value(value, v, *i as u32, "")?
                            .into_struct_value();
                    }
                }
                value.as_basic_value_enum()
            }
            Self::MultiVariant {
                field_indices,
                variant_types,
                value_type,
            } => {
                let variant_field_types = &variant_types[tag];
                let variant_field_indices = &field_indices[tag];
                let mut value = builder
                    .build_insert_value(
                        value_type.get_poison(),
                        self.tag_type().const_int(tag as u64, false),
                        0,
                        "",
                    )?
                    .into_struct_value();
                for (t, mb_i, v) in izip!(variant_field_types, variant_field_indices, vs) {
                    ensure!(&v.get_type() == t);
                    if let Some(i) = mb_i {
                        value = builder
                            .build_insert_value(value, v, *i as u32 + 1, "")?
                            .into_struct_value();
                    }
                }
                value.as_basic_value_enum()
            }
        };
        debug_assert_eq!(value.get_type(), self.value_type());
        Ok(value)
    }

    /// Get the type of the value that would be returned by `build_get_tag`.
    pub fn tag_type(&self) -> IntType<'c> {
        match self {
            Self::Void { tag_type, .. } => *tag_type,
            Self::Unit { tag_type, .. } => *tag_type,
            Self::NoFields { value_type, .. } => *value_type,
            Self::SingleVariantSingleField { tag_type, .. } => *tag_type,
            Self::SingleVariantMultiField { tag_type, .. } => *tag_type,
            Self::MultiVariant { value_type, .. } => value_type
                .get_field_type_at_index(0)
                .unwrap()
                .into_int_type(),
        }
    }

    /// The underlying LLVM type.
    pub fn value_type(&self) -> BasicTypeEnum<'c> {
        match self {
            Self::Void { tag_type, .. } => (*tag_type).into(),
            Self::Unit { value_type, .. } => (*value_type).into(),
            Self::NoFields { value_type, .. } => (*value_type).into(),
            Self::SingleVariantSingleField {
                field_index,
                field_types: variant_types,
                ..
            } => variant_types[*field_index],
            Self::SingleVariantMultiField { value_type, .. }
            | Self::MultiVariant { value_type, .. } => (*value_type).into(),
        }
    }

    /// The number of variants in the represented [`HugrSumType`].
    pub fn num_variants(&self) -> usize {
        match self {
            Self::Void { .. } => 0,
            Self::Unit { .. }
            | Self::SingleVariantSingleField { .. }
            | Self::SingleVariantMultiField { .. } => 1,
            Self::NoFields { variant_types, .. } | Self::MultiVariant { variant_types, .. } => {
                variant_types.len()
            }
        }
    }

    /// The number of fields in the `tag`th variant of the represented [`HugrSumType`].
    /// Panics if `tag` is out of bounds.
    pub(self) fn num_fields_for_variant(&self, tag: usize) -> usize {
        self.fields_for_variant(tag).len()
    }

    /// The LLVM types representing the fields in the `tag` variant of the
    /// represented [`HugrSumType`].  Panics if `tag` is out of bounds.
    pub(self) fn fields_for_variant(&self, tag: usize) -> &[BasicTypeEnum<'c>] {
        assert!(tag < self.num_variants());
        match self {
            Self::Void { .. } => unreachable!("Void has no valid tag"),
            Self::SingleVariantSingleField { field_types, .. }
            | Self::SingleVariantMultiField { field_types, .. }
            | Self::Unit { field_types, .. } => &field_types[..],
            Self::MultiVariant { variant_types, .. } | Self::NoFields { variant_types, .. } => {
                &variant_types[tag]
            }
        }
    }
}

impl<'c> From<LLVMSumTypeEnum<'c>> for BasicTypeEnum<'c> {
    fn from(value: LLVMSumTypeEnum<'c>) -> Self {
        value.value_type()
    }
}

impl std::fmt::Display for LLVMSumTypeEnum<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value_type().fmt(f)
    }
}

unsafe impl AsTypeRef for LLVMSumType<'_> {
    fn as_type_ref(&self) -> inkwell::llvm_sys::prelude::LLVMTypeRef {
        BasicTypeEnum::from(self.0.clone()).as_type_ref()
    }
}

unsafe impl<'c> AnyType<'c> for LLVMSumType<'c> {}

unsafe impl<'c> BasicType<'c> for LLVMSumType<'c> {}

/// A Value equivalent of [`LLVMSumType`]. Represents a [`HugrSumType`] Value on the
/// wire, offering functions for inspecting and deconstructing such Values.
#[derive(Debug)]
pub struct LLVMSumValue<'c>(BasicValueEnum<'c>, LLVMSumType<'c>);

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
        let value = value.as_basic_value_enum();
        ensure!(
            !matches!(sum_type.0, LLVMSumTypeEnum::Void { .. }),
            "Cannot construct LLVMSumValue of a Void sum"
        );
        ensure!(
            value.get_type() == sum_type.value_type(),
            "Cannot construct LLVMSumValue of type {sum_type} from value of type {}",
            value.get_type()
        );
        Ok(Self(value, sum_type))
    }

    #[must_use]
    pub fn get_type(&self) -> LLVMSumType<'c> {
        self.1.clone()
    }

    /// Emit instructions to read the tag of a value of type `LLVMSumType`.
    ///
    /// The type of the value is that returned by [`LLVMSumType::tag_type`].
    pub fn build_get_tag(&self, builder: &Builder<'c>) -> Result<IntValue<'c>> {
        let result = match self.get_type().0 {
            LLVMSumTypeEnum::Void { .. } => bail!("Cannot get tag of void sum"),
            LLVMSumTypeEnum::Unit { tag_type, .. }
            | LLVMSumTypeEnum::SingleVariantSingleField { tag_type, .. }
            | LLVMSumTypeEnum::SingleVariantMultiField { tag_type, .. } => {
                anyhow::Ok(tag_type.const_int(0, false))
            }
            LLVMSumTypeEnum::NoFields { .. } => Ok(self.0.into_int_value()),
            LLVMSumTypeEnum::MultiVariant { .. } => {
                let value: StructValue = self.0.into_struct_value();
                Ok(builder.build_extract_value(value, 0, "")?.into_int_value())
            }
        }?;
        debug_assert_eq!(result.get_type(), self.tag_type());
        Ok(result)
    }

    /// Emit instructions to read the inner values of a value of type
    /// `LLVMSumType`, on the assumption that it's tag is `tag`.
    ///
    /// If it's tag is not `tag`, the returned values are unspecified.
    pub fn build_untag(
        &self,
        builder: &Builder<'c>,
        tag: usize,
    ) -> Result<Vec<BasicValueEnum<'c>>> {
        ensure!(tag < self.num_variants(), "Bad tag {tag} in {}", self.1);
        let results =
            match self.get_type().0 {
                LLVMSumTypeEnum::Void { .. } => bail!("Cannot untag void sum"),
                LLVMSumTypeEnum::Unit {
                    field_types: variant_types,
                    ..
                } => anyhow::Ok(
                    variant_types
                        .into_iter()
                        .map(basic_type_undef)
                        .collect_vec(),
                ),
                LLVMSumTypeEnum::NoFields { variant_types, .. } => Ok(variant_types[tag]
                    .iter()
                    .copied()
                    .map(basic_type_undef)
                    .collect()),
                LLVMSumTypeEnum::SingleVariantSingleField {
                    field_types: variant_types,
                    field_index,
                    ..
                } => Ok(variant_types
                    .iter()
                    .enumerate()
                    .map(|(i, t)| {
                        if i == field_index {
                            self.0
                        } else {
                            basic_type_undef(*t)
                        }
                    })
                    .collect()),
                LLVMSumTypeEnum::SingleVariantMultiField {
                    field_types: variant_types,
                    field_indices,
                    ..
                } => itertools::zip_eq(variant_types, field_indices)
                    .map(|(t, mb_i)| {
                        if let Some(i) = mb_i {
                            Ok(builder.build_extract_value(
                                self.0.into_struct_value(),
                                i as u32,
                                "",
                            )?)
                        } else {
                            Ok(basic_type_undef(t))
                        }
                    })
                    .collect(),
                LLVMSumTypeEnum::MultiVariant {
                    variant_types,
                    field_indices,
                    ..
                } => {
                    let value = self.0.into_struct_value();
                    itertools::zip_eq(&variant_types[tag], &field_indices[tag])
                        .map(|(ty, mb_i)| {
                            if let Some(i) = mb_i {
                                Ok(builder.build_extract_value(value, *i as u32 + 1, "")?)
                            } else {
                                Ok(basic_type_undef(*ty))
                            }
                        })
                        .collect()
                }
            }?;
        #[cfg(debug_assertions)]
        {
            let result_types = results
                .iter()
                .map(inkwell::values::BasicValueEnum::get_type)
                .collect_vec();
            assert_eq!(&result_types, self.get_type().fields_for_variant(tag));
        }
        Ok(results)
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
        let tag_ty = self.tag_type();

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
            #[must_use] pub fn tag_type(&self) -> IntType<'c>;
            /// The number of variants in the represented [HugrSumType].
            #[must_use] pub fn num_variants(&self) -> usize;
        }
    }
}

#[cfg(test)]
mod test {
    use hugr_core::extension::prelude::{bool_t, usize_t};
    use insta::assert_snapshot;
    use rstest::{Context, rstest};

    use crate::{
        test::{TestContext, llvm_ctx},
        types::HugrType,
    };

    use super::*;

    #[rstest]
    #[case(1, 1)]
    #[case(2, 1)]
    #[case(3, 2)]
    #[case(4, 2)]
    #[case(5, 3)]
    #[case(8, 3)]
    #[case(9, 4)]
    fn tag_width(#[case] num_variants: usize, #[case] expected: u32) {
        assert_eq!(tag_width_for_num_variants(num_variants), expected);
    }

    #[rstest]
    fn sum_types(mut llvm_ctx: TestContext) {
        llvm_ctx.add_extensions(
            super::super::custom::CodegenExtsBuilder::add_default_prelude_extensions,
        );
        let ts = llvm_ctx.get_typing_session();
        let iwc = ts.iw_context();
        let empty_struct = iwc.struct_type(&[], false).as_basic_type_enum();
        let i1 = iwc.bool_type().as_basic_type_enum();
        let i2 = iwc.custom_width_int_type(2).as_basic_type_enum();
        let i64 = iwc.i64_type().as_basic_type_enum();

        {
            // no-variants -> i1
            let hugr_type = HugrType::new_unit_sum(0);
            assert_eq!(ts.llvm_type(&hugr_type).unwrap(), i1);
        }

        {
            // one-variant-no-fields -> empty_struct
            let hugr_type = HugrType::UNIT;
            assert_eq!(ts.llvm_type(&hugr_type).unwrap(), empty_struct.clone());
        }

        {
            // one-variant-elidable-fields -> empty_struct
            let hugr_type = HugrType::new_tuple(vec![HugrType::UNIT, HugrType::UNIT]);
            assert_eq!(ts.llvm_type(&hugr_type).unwrap(), empty_struct.clone());
        }

        {
            // multi-variant-no-fields -> bare tag
            let hugr_type = bool_t();
            assert_eq!(ts.llvm_type(&hugr_type).unwrap(), i1);
        }

        {
            // multi-variant-elidable-fields -> bare tag
            let hugr_type = HugrType::new_sum(vec![vec![HugrType::UNIT]; 3]);
            assert_eq!(ts.llvm_type(&hugr_type).unwrap(), i2);
        }

        {
            // one-variant-one-field -> bare field
            let hugr_type = HugrType::new_tuple(vec![usize_t()]);
            assert_eq!(ts.llvm_type(&hugr_type).unwrap(), i64);
        }

        {
            // one-variant-one-non-elidable-field -> bare field
            let hugr_type = HugrType::new_tuple(vec![HugrType::UNIT, usize_t()]);
            assert_eq!(ts.llvm_type(&hugr_type).unwrap(), i64);
        }

        {
            // one-variant-multi-field -> struct-of-fields
            let hugr_type = HugrType::new_tuple(vec![usize_t(), bool_t(), HugrType::UNIT]);
            let llvm_type = iwc.struct_type(&[i64, i1], false).into();
            assert_eq!(ts.llvm_type(&hugr_type).unwrap(), llvm_type);
        }

        {
            // multi-variant-multi-field -> struct-of-fields-with-tag
            let hugr_type1 =
                HugrType::new_sum([vec![bool_t(), HugrType::UNIT, usize_t()], vec![usize_t()]]);
            let hugr_type2 = HugrType::new_sum([vec![usize_t(), bool_t()], vec![usize_t()]]);
            let llvm_type = iwc.struct_type(&[i1, i64, i1], false).into();
            assert_eq!(ts.llvm_type(&hugr_type1).unwrap(), llvm_type);
            assert_eq!(ts.llvm_type(&hugr_type2).unwrap(), llvm_type);
        }
    }

    #[rstest]
    #[case::unit(HugrSumType::new_unary(1), 0)]
    #[case::unit_elided_fields(HugrSumType::new([HugrType::UNIT]), 0)]
    #[case::nofields(HugrSumType::new_unary(4), 2)]
    #[case::nofields_elided_fields(HugrSumType::new([vec![HugrType::UNIT], vec![]]), 0)]
    #[case::one_variant_one_field(HugrSumType::new([bool_t()]), 0)]
    #[case::one_variant_one_field_elided_fields(HugrSumType::new([vec![HugrType::UNIT,bool_t()]]), 0)]
    #[case::one_variant_two_fields(HugrSumType::new([vec![bool_t(),bool_t()]]), 0)]
    #[case::one_variant_two_fields_elided_fields(HugrSumType::new([vec![bool_t(),HugrType::UNIT,bool_t()]]), 0)]
    #[case::two_variant_one_field(HugrSumType::new([vec![bool_t()],vec![]]), 1)]
    #[case::two_variant_one_field_elided_fields(HugrSumType::new([vec![bool_t()],vec![HugrType::UNIT]]), 1)]
    fn build_untag_tag(
        #[context] rstest_ctx: Context,
        llvm_ctx: TestContext,
        #[case] sum: HugrSumType,
        #[case] tag: usize,
    ) {
        let module = {
            let ts = llvm_ctx.get_typing_session();
            let iwc = llvm_ctx.iw_context();
            let module = iwc.create_module("");
            let llvm_ty = ts.llvm_sum_type(sum.clone()).unwrap();
            let func_ty = llvm_ty.fn_type(&[llvm_ty.as_basic_type_enum().into()], false);
            let func = module.add_function("untag_tag", func_ty, None);
            let bb = iwc.append_basic_block(func, "");
            let builder = iwc.create_builder();
            builder.position_at_end(bb);
            let value = llvm_ty.value(func.get_nth_param(0).unwrap()).unwrap();
            let _tag = value.build_get_tag(&builder).unwrap();
            let fields = value.build_untag(&builder, tag).unwrap();
            let new_value = llvm_ty.build_tag(&builder, tag, fields).unwrap();
            let _ = builder.build_return(Some(&new_value));
            module.verify().unwrap();
            module
        };

        let mut insta_settings = insta::Settings::clone_current();
        insta_settings.set_snapshot_suffix(rstest_ctx.description.unwrap());
        insta_settings.bind(|| {
            assert_snapshot!(module.to_string());
        });
    }
}
