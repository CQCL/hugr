//! Fixed-length array type and operations extension.

mod array_op;
mod array_repeat;
mod array_scan;
pub mod builder;

use std::sync::Arc;

use itertools::Itertools as _;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

use crate::extension::resolution::{
    resolve_type_extensions, resolve_value_extensions, ExtensionResolutionError,
    WeakExtensionRegistry,
};
use crate::extension::simple_op::{MakeOpDef, MakeRegisteredOp};
use crate::extension::{ExtensionId, ExtensionSet, SignatureError, TypeDef, TypeDefBound};
use crate::ops::constant::{maybe_hash_values, CustomConst, TryHash, ValueName};
use crate::ops::{ExtensionOp, OpName, Value};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{CustomCheckFailure, CustomType, Type, TypeBound, TypeName};
use crate::Extension;

pub use array_op::{ArrayOp, ArrayOpDef, ArrayOpDefIter};
pub use array_repeat::{ArrayRepeat, ArrayRepeatDef, ARRAY_REPEAT_OP_ID};
pub use array_scan::{ArrayScan, ArrayScanDef, ARRAY_SCAN_OP_ID};
pub use builder::ArrayOpBuilder;

/// Reported unique name of the array type.
pub const ARRAY_TYPENAME: TypeName = TypeName::new_inline("array");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("collections.array");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Statically sized array of values, all of the same type.
pub struct ArrayValue {
    values: Vec<Value>,
    typ: Type,
}

impl ArrayValue {
    /// Name of the constructor for creating constant arrays.
    #[cfg_attr(not(feature = "model_unstable"), allow(dead_code))]
    pub(crate) const CTR_NAME: &'static str = "collections.array.const";

    /// Create a new [CustomConst] for an array of values of type `typ`.
    /// That all values are of type `typ` is not checked here.
    pub fn new(typ: Type, contents: impl IntoIterator<Item = Value>) -> Self {
        Self {
            values: contents.into_iter().collect_vec(),
            typ,
        }
    }

    /// Create a new [CustomConst] for an empty array of values of type `typ`.
    pub fn new_empty(typ: Type) -> Self {
        Self {
            values: vec![],
            typ,
        }
    }

    /// Returns the type of the `[ArrayValue]` as a `[CustomType]`.`
    pub fn custom_type(&self) -> CustomType {
        array_custom_type(self.values.len() as u64, self.typ.clone())
    }

    /// Returns the type of values inside the `[ArrayValue]`.
    pub fn get_element_type(&self) -> &Type {
        &self.typ
    }

    /// Returns the values contained inside the `[ArrayValue]`.
    pub fn get_contents(&self) -> &[Value] {
        &self.values
    }
}

impl TryHash for ArrayValue {
    fn try_hash(&self, mut st: &mut dyn Hasher) -> bool {
        maybe_hash_values(&self.values, &mut st) && {
            self.typ.hash(&mut st);
            true
        }
    }
}

#[typetag::serde]
impl CustomConst for ArrayValue {
    fn name(&self) -> ValueName {
        ValueName::new_inline("array")
    }

    fn get_type(&self) -> Type {
        self.custom_type().into()
    }

    fn validate(&self) -> Result<(), CustomCheckFailure> {
        let typ = self.custom_type();

        EXTENSION
            .get_type(&ARRAY_TYPENAME)
            .unwrap()
            .check_custom(&typ)
            .map_err(|_| {
                CustomCheckFailure::Message(format!(
                    "Custom typ {typ} is not a valid instantiation of array."
                ))
            })?;

        // constant can only hold classic type.
        let ty = match typ.args() {
            [TypeArg::BoundedNat { n }, TypeArg::Type { ty }]
                if *n as usize == self.values.len() =>
            {
                ty
            }
            _ => {
                return Err(CustomCheckFailure::Message(format!(
                    "Invalid array type arguments: {:?}",
                    typ.args()
                )))
            }
        };

        // check all values are instances of the element type
        for v in &self.values {
            if v.get_type() != *ty {
                return Err(CustomCheckFailure::Message(format!(
                    "Array element {v:?} is not of expected type {ty}"
                )));
            }
        }

        Ok(())
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::union_over(self.values.iter().map(Value::extension_reqs))
            .union(EXTENSION_ID.into())
    }

    fn update_extensions(
        &mut self,
        extensions: &WeakExtensionRegistry,
    ) -> Result<(), ExtensionResolutionError> {
        for val in &mut self.values {
            resolve_value_extensions(val, extensions)?;
        }
        resolve_type_extensions(&mut self.typ, extensions)
    }
}

lazy_static! {
    /// Extension for array operations.
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

/// Gets the [TypeDef] for arrays. Note that instantiations are more easily
/// created via [array_type] and [array_type_parametric]
pub fn array_type_def() -> &'static TypeDef {
    EXTENSION.get_type(&ARRAY_TYPENAME).unwrap()
}

/// Instantiate a new array type given a size argument and element type.
///
/// This method is equivalent to [`array_type_parametric`], but uses concrete
/// arguments types to ensure no errors are possible.
pub fn array_type(size: u64, element_ty: Type) -> Type {
    array_custom_type(size, element_ty).into()
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

fn array_custom_type(size: impl Into<TypeArg>, element_ty: impl Into<TypeArg>) -> CustomType {
    instantiate_array_custom(array_type_def(), size, element_ty)
        .expect("array parameters are valid")
}

fn instantiate_array_custom(
    array_def: &TypeDef,
    size: impl Into<TypeArg>,
    element_ty: impl Into<TypeArg>,
) -> Result<CustomType, SignatureError> {
    array_def.instantiate(vec![size.into(), element_ty.into()])
}

fn instantiate_array(
    array_def: &TypeDef,
    size: impl Into<TypeArg>,
    element_ty: impl Into<TypeArg>,
) -> Result<Type, SignatureError> {
    instantiate_array_custom(array_def, size, element_ty).map(Into::into)
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
    use crate::extension::prelude::{qb_t, usize_t, ConstUsize};
    use crate::ops::constant::CustomConst;
    use crate::std_extensions::arithmetic::float_types::ConstF64;

    use super::{array_type, new_array_op, ArrayValue};

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

    #[test]
    fn test_array_value() {
        let array_value = ArrayValue {
            values: vec![ConstUsize::new(3).into()],
            typ: usize_t(),
        };

        array_value.validate().unwrap();

        let wrong_array_value = ArrayValue {
            values: vec![ConstF64::new(1.2).into()],
            typ: usize_t(),
        };
        assert!(wrong_array_value.validate().is_err());
    }
}
