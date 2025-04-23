//! Fixed-length array type and operations extension.

mod array_clone;
mod array_conversion;
mod array_discard;
mod array_kind;
mod array_op;
mod array_repeat;
mod array_scan;
mod array_value;

use std::sync::Arc;

use delegate::delegate;
use lazy_static::lazy_static;

use crate::extension::resolution::{ExtensionResolutionError, WeakExtensionRegistry};
use crate::extension::simple_op::{MakeOpDef, MakeRegisteredOp};
use crate::extension::{ExtensionId, ExtensionSet, SignatureError, TypeDef, TypeDefBound};
use crate::ops::constant::{CustomConst, ValueName};
use crate::ops::{ExtensionOp, OpName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{CustomCheckFailure, Type, TypeBound, TypeName};
use crate::Extension;

pub use array_clone::{GenericArrayClone, GenericArrayCloneDef, ARRAY_CLONE_OP_ID};
pub use array_conversion::{Direction, GenericArrayConvert, GenericArrayConvertDef, FROM, INTO};
pub use array_discard::{GenericArrayDiscard, GenericArrayDiscardDef, ARRAY_DISCARD_OP_ID};
pub use array_kind::ArrayKind;
pub use array_op::{GenericArrayOp, GenericArrayOpDef};
pub use array_repeat::{GenericArrayRepeat, GenericArrayRepeatDef, ARRAY_REPEAT_OP_ID};
pub use array_scan::{GenericArrayScan, GenericArrayScanDef, ARRAY_SCAN_OP_ID};
pub use array_value::GenericArrayValue;

/// Reported unique name of the array type.
pub const ARRAY_TYPENAME: TypeName = TypeName::new_inline("array");
/// Reported unique name of the array value.
pub const ARRAY_VALUENAME: TypeName = TypeName::new_inline("array");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("collections.array");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

/// A linear, fixed-length collection of values.
///
/// Arrays are linear, even if their elements are copyable.
#[derive(Clone, Copy, Debug, derive_more::Display, Eq, PartialEq, Default)]
pub struct Array;

impl ArrayKind for Array {
    const EXTENSION_ID: ExtensionId = EXTENSION_ID;
    const TYPE_NAME: TypeName = ARRAY_TYPENAME;
    const VALUE_NAME: ValueName = ARRAY_VALUENAME;

    fn extension() -> &'static Arc<Extension> {
        &EXTENSION
    }

    fn type_def() -> &'static TypeDef {
        EXTENSION.get_type(&ARRAY_TYPENAME).unwrap()
    }
}

/// Array operation definitions.
pub type ArrayOpDef = GenericArrayOpDef<Array>;
/// Array clone operation definition.
pub type ArrayCloneDef = GenericArrayCloneDef<Array>;
/// Array discard operation definition.
pub type ArrayDiscardDef = GenericArrayDiscardDef<Array>;
/// Array repeat operation definition.
pub type ArrayRepeatDef = GenericArrayRepeatDef<Array>;
/// Array scan operation definition.
pub type ArrayScanDef = GenericArrayScanDef<Array>;

/// Array operations.
pub type ArrayOp = GenericArrayOp<Array>;
/// The array clone operation.
pub type ArrayClone = GenericArrayClone<Array>;
/// The array discard operation.
pub type ArrayDiscard = GenericArrayDiscard<Array>;
/// The array repeat operation.
pub type ArrayRepeat = GenericArrayRepeat<Array>;
/// The array scan operation.
pub type ArrayScan = GenericArrayScan<Array>;

/// An array extension value.
pub type ArrayValue = GenericArrayValue<Array>;

lazy_static! {
    /// Extension for array operations.
    pub static ref EXTENSION: Arc<Extension> = {
        Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
            extension.add_type(
                    ARRAY_TYPENAME,
                    vec![ TypeParam::max_nat(), TypeBound::Any.into()],
                    "Fixed-length array".into(),
                    // Default array is linear, even if the elements are copyable
                    TypeDefBound::any(),
                    extension_ref,
                )
                .unwrap();

            ArrayOpDef::load_all_ops(extension, extension_ref).unwrap();
            ArrayCloneDef::new().add_to_extension(extension, extension_ref).unwrap();
            ArrayDiscardDef::new().add_to_extension(extension, extension_ref).unwrap();
            ArrayRepeatDef::new().add_to_extension(extension, extension_ref).unwrap();
            ArrayScanDef::new().add_to_extension(extension, extension_ref).unwrap();
        })
    };
}

impl ArrayValue {
    /// Name of the constructor for creating constant arrays.
    #[cfg_attr(not(feature = "model_unstable"), allow(dead_code))]
    pub(crate) const CTR_NAME: &'static str = "collections.array.const";
}

#[typetag::serde(name = "ArrayValue")]
impl CustomConst for ArrayValue {
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

/// Gets the [TypeDef] for arrays. Note that instantiations are more easily
/// created via [array_type] and [array_type_parametric]
pub fn array_type_def() -> &'static TypeDef {
    Array::type_def()
}

/// Instantiate a new array type given a size argument and element type.
///
/// This method is equivalent to [`array_type_parametric`], but uses concrete
/// arguments types to ensure no errors are possible.
pub fn array_type(size: u64, element_ty: Type) -> Type {
    Array::ty(size, element_ty)
}

/// Instantiate a new array type given the size and element type parameters.
///
/// This is a generic version of [`array_type`].
pub fn array_type_parametric(
    size: impl Into<TypeArg>,
    element_ty: impl Into<TypeArg>,
) -> Result<Type, SignatureError> {
    Array::ty_parametric(size, element_ty)
}

/// Name of the operation in the prelude for creating new arrays.
pub const NEW_ARRAY_OP_ID: OpName = OpName::new_inline("new_array");

/// Initialize a new array op of element type `element_ty` of length `size`
pub fn new_array_op(element_ty: Type, size: u64) -> ExtensionOp {
    let op = ArrayOpDef::new_array.to_concrete(element_ty, size);
    op.to_extension_op().unwrap()
}

#[cfg(test)]
mod test {
    use crate::builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr};
    use crate::extension::prelude::qb_t;

    use super::{array_type, new_array_op};

    #[test]
    /// Test building a HUGR involving a new_array operation.
    fn test_new_array() {
        let mut b =
            DFGBuilder::new(inout_sig(vec![qb_t(), qb_t()], array_type(2, qb_t()))).unwrap();

        let [q1, q2] = b.input_wires_arr();

        let op = new_array_op(qb_t(), 2);

        let out = b.add_dataflow_op(op, [q1, q2]).unwrap();

        b.finish_hugr_with_outputs(out.outputs()).unwrap();
    }
}
