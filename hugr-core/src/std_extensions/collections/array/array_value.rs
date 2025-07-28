use itertools::Itertools as _;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use crate::extension::resolution::{
    ExtensionResolutionError, WeakExtensionRegistry, resolve_type_extensions,
    resolve_value_extensions,
};
use crate::ops::Value;
use crate::ops::constant::{TryHash, ValueName, maybe_hash_values};
use crate::types::type_param::TypeArg;
use crate::types::{CustomCheckFailure, CustomType, Type};

use super::array_kind::ArrayKind;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Statically sized array of values, all of the same type.
pub struct GenericArrayValue<AK: ArrayKind> {
    values: Vec<Value>,
    typ: Type,
    #[serde(skip)]
    _kind: PhantomData<AK>,
}

impl<AK: ArrayKind> GenericArrayValue<AK> {
    /// Create a new [`CustomConst`] for an array of values of type `typ`.
    /// That all values are of type `typ` is not checked here.
    ///
    /// [`CustomConst`]: crate::ops::constant::CustomConst
    pub fn new(typ: Type, contents: impl IntoIterator<Item = Value>) -> Self {
        Self {
            values: contents.into_iter().collect_vec(),
            typ,
            _kind: PhantomData,
        }
    }

    /// Create a new [`CustomConst`] for an empty array of values of type `typ`.
    ///
    /// [`CustomConst`]: crate::ops::constant::CustomConst
    #[must_use]
    pub fn new_empty(typ: Type) -> Self {
        Self {
            values: vec![],
            typ,
            _kind: PhantomData,
        }
    }

    /// Returns the type of the `[GenericArrayValue]` as a `[CustomType]`.`
    #[must_use]
    pub fn custom_type(&self) -> CustomType {
        AK::custom_ty(self.values.len() as u64, self.typ.clone())
    }

    /// Returns the type of the `[GenericArrayValue]`.
    #[must_use]
    pub fn get_type(&self) -> Type {
        self.custom_type().into()
    }

    /// Returns the type of values inside the `[ArrayValue]`.
    #[must_use]
    pub fn get_element_type(&self) -> &Type {
        &self.typ
    }

    /// Returns the values contained inside the `[ArrayValue]`.
    #[must_use]
    pub fn get_contents(&self) -> &[Value] {
        &self.values
    }

    /// Returns the name of the value.
    #[must_use]
    pub fn name(&self) -> ValueName {
        AK::VALUE_NAME
    }

    /// Validates the array value.
    pub fn validate(&self) -> Result<(), CustomCheckFailure> {
        let typ = self.custom_type();

        AK::extension()
            .get_type(&AK::TYPE_NAME)
            .unwrap()
            .check_custom(&typ)
            .map_err(|_| {
                CustomCheckFailure::Message(format!(
                    "Custom typ {typ} is not a valid instantiation of array."
                ))
            })?;

        // constant can only hold classic type.
        let ty = match typ.args() {
            [TypeArg::BoundedNat(n), TypeArg::Runtime(ty)] if *n as usize == self.values.len() => {
                ty
            }
            _ => {
                return Err(CustomCheckFailure::Message(format!(
                    "Invalid array type arguments: {:?}",
                    typ.args()
                )));
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

    /// Update the extensions associated with the internal values.
    pub fn update_extensions(
        &mut self,
        extensions: &WeakExtensionRegistry,
    ) -> Result<(), ExtensionResolutionError> {
        for val in &mut self.values {
            resolve_value_extensions(val, extensions)?;
        }
        resolve_type_extensions(&mut self.typ, extensions)
    }
}

impl<AK: ArrayKind> TryHash for GenericArrayValue<AK> {
    fn try_hash(&self, mut st: &mut dyn Hasher) -> bool {
        maybe_hash_values(&self.values, &mut st) && {
            self.typ.hash(&mut st);
            true
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::extension::prelude::{ConstUsize, usize_t};
    use crate::std_extensions::arithmetic::float_types::ConstF64;

    use crate::std_extensions::collections::array::Array;
    use crate::std_extensions::collections::borrow_array::BorrowArray;
    use crate::std_extensions::collections::value_array::ValueArray;

    use super::*;

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    fn test_array_value<AK: ArrayKind>(#[case] _kind: AK) {
        let array_value = GenericArrayValue::<AK>::new(usize_t(), vec![ConstUsize::new(3).into()]);
        array_value.validate().unwrap();

        let wrong_array_value =
            GenericArrayValue::<AK>::new(usize_t(), vec![ConstF64::new(1.2).into()]);
        assert!(wrong_array_value.validate().is_err());
    }
}
