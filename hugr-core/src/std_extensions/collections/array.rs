//! Fixed-length array type and operations extension.

mod array_op;
mod array_repeat;
mod array_scan;

use std::sync::Arc;

use lazy_static::lazy_static;

use crate::extension::simple_op::{MakeOpDef, MakeRegisteredOp};
use crate::extension::{ExtensionId, SignatureError, TypeDef, TypeDefBound};
use crate::ops::{ExtensionOp, OpName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{Type, TypeBound, TypeName};
use crate::Extension;

pub use array_op::{ArrayOp, ArrayOpDef, ArrayOpDefIter};
pub use array_repeat::{ArrayRepeat, ArrayRepeatDef, ARRAY_REPEAT_OP_ID};
pub use array_scan::{ArrayScan, ArrayScanDef, ARRAY_SCAN_OP_ID};

/// Reported unique name of the array type.
pub const ARRAY_TYPENAME: TypeName = TypeName::new_inline("array");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("collections.array");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

lazy_static! {
    /// Extension for list operations.
    pub static ref EXTENSION: Arc<Extension> = {
        Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
            extension.add_type(
                    ARRAY_TYPENAME,
                    vec![ TypeParam::max_nat(), TypeBound::Any.into()],
                    "Fixed-length array".into(),
                    TypeDefBound::from_params(vec![1] ),
                    extension_ref,
                )
                .unwrap();

            array_op::ArrayOpDef::load_all_ops(extension, extension_ref).unwrap();
            array_repeat::ArrayRepeatDef.add_to_extension(extension, extension_ref).unwrap();
            array_scan::ArrayScanDef.add_to_extension(extension, extension_ref).unwrap();
        })
    };
}

fn array_type_def() -> &'static TypeDef {
    EXTENSION.get_type(&ARRAY_TYPENAME).unwrap()
}

/// Instantiate a new array type given a size argument and element type.
///
/// This method is equivalent to [`array_type_parametric`], but uses concrete
/// arguments types to ensure no errors are possible.
pub fn array_type(size: u64, element_ty: Type) -> Type {
    instantiate_array(array_type_def(), size, element_ty).expect("array parameters are valid")
}

/// Instantiate a new array type given the size and element type parameters.
///
/// This is a generic version of [`array_type`].
pub fn array_type_parametric(
    size: impl Into<TypeArg>,
    element_ty: impl Into<TypeArg>,
) -> Result<Type, SignatureError> {
    instantiate_array(array_type_def(), size, element_ty)
}

fn instantiate_array(
    array_def: &TypeDef,
    size: impl Into<TypeArg>,
    element_ty: impl Into<TypeArg>,
) -> Result<Type, SignatureError> {
    array_def
        .instantiate(vec![size.into(), element_ty.into()])
        .map(Into::into)
}

/// Name of the operation in the prelude for creating new arrays.
pub const NEW_ARRAY_OP_ID: OpName = OpName::new_inline("new_array");

/// Initialize a new array op of element type `element_ty` of length `size`
pub fn new_array_op(element_ty: Type, size: u64) -> ExtensionOp {
    let op = array_op::ArrayOpDef::new_array.to_concrete(element_ty, size);
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
