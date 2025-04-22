//! A version of the standard fixed-length array extension where arrays of copyable types
//! are copyable themselves.
//!
//! Supports all regular array operations apart from `clone` and `discard`.

use std::sync::Arc;

use delegate::delegate;
use lazy_static::lazy_static;

use crate::extension::resolution::{ExtensionResolutionError, WeakExtensionRegistry};
use crate::extension::simple_op::MakeOpDef;
use crate::extension::{ExtensionId, ExtensionSet, SignatureError, TypeDef, TypeDefBound};
use crate::ops::constant::{CustomConst, ValueName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{CustomCheckFailure, Type, TypeBound, TypeName};
use crate::Extension;

use super::array::{
    Array, ArrayKind, GenericArrayConvert, GenericArrayConvertDef, GenericArrayOp,
    GenericArrayOpDef, GenericArrayRepeat, GenericArrayRepeatDef, GenericArrayScan,
    GenericArrayScanDef, GenericArrayValue, FROM, INTO,
};

/// Reported unique name of the value array type.
pub const VALUE_ARRAY_TYPENAME: TypeName = TypeName::new_inline("value_array");
/// Reported unique name of the value array value.
pub const VALUE_ARRAY_VALUENAME: TypeName = TypeName::new_inline("value_array");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_static_unchecked("collections.value_array");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

/// A fixed-length collection of values.
///
/// A value array inherits its linearity from its elements.
#[derive(Clone, Copy, Debug, derive_more::Display, Eq, PartialEq, Default)]
pub struct ValueArray;

impl ArrayKind for ValueArray {
    const EXTENSION_ID: ExtensionId = EXTENSION_ID;
    const TYPE_NAME: TypeName = VALUE_ARRAY_TYPENAME;
    const VALUE_NAME: ValueName = VALUE_ARRAY_VALUENAME;

    fn extension() -> &'static Arc<Extension> {
        &EXTENSION
    }

    fn type_def() -> &'static TypeDef {
        EXTENSION.get_type(&VALUE_ARRAY_TYPENAME).unwrap()
    }
}

/// Value array operation definitions.
pub type VArrayOpDef = GenericArrayOpDef<ValueArray>;
/// Value array repeat operation definition.
pub type VArrayRepeatDef = GenericArrayRepeatDef<ValueArray>;
/// Value array scan operation definition.
pub type VArrayScanDef = GenericArrayScanDef<ValueArray>;
/// Value array to default array conversion operation definition.
pub type VArrayToArrayDef = GenericArrayConvertDef<ValueArray, INTO, Array>;
/// Value array from default array conversion operation definition.
pub type VArrayFromArrayDef = GenericArrayConvertDef<ValueArray, FROM, Array>;

/// Value array operations.
pub type VArrayOp = GenericArrayOp<ValueArray>;
/// The value array repeat operation.
pub type VArrayRepeat = GenericArrayRepeat<ValueArray>;
/// The value array scan operation.
pub type VArrayScan = GenericArrayScan<ValueArray>;
/// The value array to default array conversion operation.
pub type VArrayToArray = GenericArrayConvert<ValueArray, INTO, Array>;
/// The value array from default array conversion operation.
pub type VArrayFromArray = GenericArrayConvert<ValueArray, FROM, Array>;

/// A value array extension value.
pub type VArrayValue = GenericArrayValue<ValueArray>;

lazy_static! {
    /// Extension for value array operations.
    pub static ref EXTENSION: Arc<Extension> = {
        Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
            extension.add_type(
                    VALUE_ARRAY_TYPENAME,
                    vec![ TypeParam::max_nat(), TypeBound::Any.into()],
                    "Fixed-length value array".into(),
                    // Value arrays are copyable iff their elements are
                    TypeDefBound::from_params(vec![1]),
                    extension_ref,
                )
                .unwrap();

            VArrayOpDef::load_all_ops(extension, extension_ref).unwrap();
            VArrayRepeatDef::new().add_to_extension(extension, extension_ref).unwrap();
            VArrayScanDef::new().add_to_extension(extension, extension_ref).unwrap();
            VArrayToArrayDef::new().add_to_extension(extension, extension_ref).unwrap();
            VArrayFromArrayDef::new().add_to_extension(extension, extension_ref).unwrap();
        })
    };
}

#[typetag::serde(name = "VArrayValue")]
impl CustomConst for VArrayValue {
    delegate! {
        to self {
            fn name(&self) -> ValueName;
            fn extension_reqs(&self) -> ExtensionSet;
            fn validate(&self) -> Result<(), CustomCheckFailure>;
            fn update_extensions(
                &mut self,
                extensions: &WeakExtensionRegistry,
            ) -> Result<(), ExtensionResolutionError>;
            fn get_type(&self) -> Type;
        }
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }
}

/// Gets the [TypeDef] for value arrays. Note that instantiations are more easily
/// created via [value_array_type] and [value_array_type_parametric]
pub fn value_array_type_def() -> &'static TypeDef {
    ValueArray::type_def()
}

/// Instantiate a new value array type given a size argument and element type.
///
/// This method is equivalent to [`value_array_type_parametric`], but uses concrete
/// arguments types to ensure no errors are possible.
pub fn value_array_type(size: u64, element_ty: Type) -> Type {
    ValueArray::ty(size, element_ty)
}

/// Instantiate a new value array type given the size and element type parameters.
///
/// This is a generic version of [`value_array_type`].
pub fn value_array_type_parametric(
    size: impl Into<TypeArg>,
    element_ty: impl Into<TypeArg>,
) -> Result<Type, SignatureError> {
    ValueArray::ty_parametric(size, element_ty)
}
