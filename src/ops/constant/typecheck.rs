//! Simple type checking - takes a hugr and some extra info and checks whether
//! the types at the sources of each wire match those of the targets

use lazy_static::lazy_static;

use std::collections::HashSet;

use thiserror::Error;

// For static typechecking
use crate::ops::ConstValue;
use crate::types::{ClassicType, Container, HashableType, PrimType, TypeRow};

use crate::ops::constant::{HugrIntValueStore, HugrIntWidthStore, HUGR_MAX_INT_WIDTH};

/// An error in fitting an integer constant into its size
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ConstIntError {
    /// The value exceeds the max value of its `I<n>` type
    /// E.g. checking 300 against I8
    #[error("Const int {1} too large for type I{0}")]
    IntTooLarge(HugrIntWidthStore, HugrIntValueStore),
    /// Width (n) of an `I<n>` type doesn't fit into a HugrIntWidthStore
    #[error("Int type too large: I{0}")]
    IntWidthTooLarge(HugrIntWidthStore),
    /// The width of an integer type wasn't a power of 2
    #[error("The int type I{0} is invalid, because {0} is not a power of 2")]
    IntWidthInvalid(HugrIntWidthStore),
}

/// Struct for custom type check fails.
#[derive(Clone, Debug, PartialEq, Error)]
#[error("Error when checking custom type.")]
pub struct CustomCheckFail(String);

impl CustomCheckFail {
    /// Creates a new [`CustomCheckFail`].
    pub fn new(message: String) -> Self {
        Self(message)
    }
}

/// Errors that arise from typechecking constants
#[derive(Clone, Debug, PartialEq, Error)]
pub enum ConstTypeError {
    /// This case hasn't been implemented. Possibly because we don't have value
    /// constructors to check against it
    #[error("Unimplemented: there are no constants of type {0}")]
    Unimplemented(ClassicType),
    /// There was some problem fitting a const int into its declared size
    #[error("Error with int constant")]
    Int(#[from] ConstIntError),
    /// Expected width (packed with const int) doesn't match type
    #[error("Type mismatch for int: expected I{0}, but found I{1}")]
    IntWidthMismatch(HugrIntWidthStore, HugrIntWidthStore),
    /// Found a Var type constructor when we're checking a const val
    #[error("Type of a const value can't be Var")]
    ConstCantBeVar,
    /// The length of the tuple value doesn't match the length of the tuple type
    #[error("Tuple of wrong length")]
    TupleWrongLength,
    /// Tag for a sum value exceeded the number of variants
    #[error("Tag of Sum value is invalid")]
    InvalidSumTag,
    /// A mismatch between the type expected and the value.
    #[error("Value {1:?} does not match expected type {0}")]
    ValueCheckFail(ClassicType, ConstValue),
    /// Error when checking a custom value.
    #[error("Custom value type check error: {0:?}")]
    CustomCheckFail(#[from] CustomCheckFail),
}

lazy_static! {
    static ref VALID_WIDTHS: HashSet<HugrIntWidthStore> =
        HashSet::from_iter((0..8).map(|a| HugrIntWidthStore::pow(2, a)));
}

/// Per the spec, valid widths for integers are 2^n for all n in [0,7]
pub(crate) fn check_int_fits_in_width(
    value: HugrIntValueStore,
    width: HugrIntWidthStore,
) -> Result<(), ConstIntError> {
    if width > HUGR_MAX_INT_WIDTH {
        return Err(ConstIntError::IntWidthTooLarge(width));
    }

    if VALID_WIDTHS.contains(&width) {
        let max_value = if width == HUGR_MAX_INT_WIDTH {
            HugrIntValueStore::MAX
        } else {
            HugrIntValueStore::pow(2, width as u32) - 1
        };
        if value <= max_value {
            Ok(())
        } else {
            Err(ConstIntError::IntTooLarge(width, value))
        }
    } else {
        Err(ConstIntError::IntWidthInvalid(width))
    }
}

fn map_vals<T: PrimType, T2: PrimType>(
    container: Container<T>,
    f: &impl Fn(T) -> T2,
) -> Container<T2> {
    fn map_row<T: PrimType, T2: PrimType>(
        row: TypeRow<T>,
        f: &impl Fn(T) -> T2,
    ) -> Box<TypeRow<T2>> {
        Box::new(TypeRow::from(
            row.into_owned().into_iter().map(f).collect::<Vec<T2>>(),
        ))
    }
    match container {
        Container::List(elem) => Container::List(Box::new(f(*elem))),
        Container::Map(kv) => {
            let (k, v) = *kv;
            Container::Map(Box::new((k, f(v))))
        }
        Container::Tuple(elems) => Container::Tuple(map_row(*elems, f)),
        Container::Sum(variants) => Container::Sum(map_row(*variants, f)),
        Container::Array(elem, sz) => Container::Array(Box::new(f(*elem)), sz),
        Container::Alias(s) => Container::Alias(s),
        Container::Opaque(custom) => Container::Opaque(custom),
    }
}

/// Typecheck a constant value
pub(super) fn typecheck_const(typ: &ClassicType, val: &ConstValue) -> Result<(), ConstTypeError> {
    match (typ, val) {
        (ClassicType::Hashable(HashableType::Int(exp_width)), ConstValue::Int(value)) => {
            check_int_fits_in_width(*value, *exp_width).map_err(ConstTypeError::Int)
        }
        (ClassicType::F64, ConstValue::F64(_)) => Ok(()),
        (ty @ ClassicType::Container(c), tm) => match (c, tm) {
            (Container::Tuple(row), ConstValue::Tuple(xs)) => {
                if row.len() != xs.len() {
                    return Err(ConstTypeError::TupleWrongLength);
                }
                for (ty, tm) in row.iter().zip(xs.iter()) {
                    typecheck_const(ty, tm)?
                }
                Ok(())
            }
            (Container::Tuple(_), _) => Err(ConstTypeError::ValueCheckFail(ty.clone(), tm.clone())),
            (Container::Sum(row), ConstValue::Sum(tag, val)) => {
                if let Some(ty) = row.get(*tag) {
                    typecheck_const(ty, val.as_ref())
                } else {
                    Err(ConstTypeError::InvalidSumTag)
                }
            }
            (Container::Sum(_), _) => Err(ConstTypeError::ValueCheckFail(ty.clone(), tm.clone())),
            (Container::Opaque(ty), ConstValue::Opaque((val,))) => {
                val.check_custom_type(ty).map_err(ConstTypeError::from)
            }
            _ => Err(ConstTypeError::Unimplemented(ty.clone())),
        },
        (ClassicType::Hashable(HashableType::Container(c)), tm) => {
            // Here we deliberately build malformed Container-of-Hashable types
            // (rather than Hashable-of-Container) in order to reuse logic above
            typecheck_const(
                &ClassicType::Container(map_vals(c.clone(), &ClassicType::Hashable)),
                tm,
            )
        }
        (ty @ ClassicType::Graph(_), _) => Err(ConstTypeError::Unimplemented(ty.clone())),
        (ty @ ClassicType::Hashable(HashableType::String), _) => {
            Err(ConstTypeError::Unimplemented(ty.clone()))
        }
        (ClassicType::Hashable(HashableType::Variable(_)), _) => {
            Err(ConstTypeError::ConstCantBeVar)
        }
        (ty, _) => Err(ConstTypeError::ValueCheckFail(ty.clone(), val.clone())),
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::{classic_row, types::ClassicType};

    use super::*;

    #[test]
    fn test_typecheck_const() {
        const INT: ClassicType = ClassicType::int::<64>();
        typecheck_const(&INT, &ConstValue::Int(3)).unwrap();
        typecheck_const(&ClassicType::F64, &ConstValue::F64(17.4)).unwrap();
        assert_eq!(
            typecheck_const(&ClassicType::F64, &ConstValue::Int(5)),
            Err(ConstTypeError::ValueCheckFail(
                ClassicType::F64,
                ConstValue::Int(5)
            ))
        );
        let tuple_ty = ClassicType::new_tuple(classic_row![INT, ClassicType::F64,]);
        typecheck_const(
            &tuple_ty,
            &ConstValue::Tuple(vec![ConstValue::Int(7), ConstValue::F64(5.1)]),
        )
        .unwrap();
        assert_matches!(
            typecheck_const(
                &tuple_ty,
                &ConstValue::Tuple(vec![ConstValue::F64(4.8), ConstValue::Int(2)])
            ),
            Err(ConstTypeError::ValueCheckFail(_, _))
        );
        assert_eq!(
            typecheck_const(
                &tuple_ty,
                &ConstValue::Tuple(vec![
                    ConstValue::Int(5),
                    ConstValue::F64(3.3),
                    ConstValue::Int(2)
                ])
            ),
            Err(ConstTypeError::TupleWrongLength)
        );
    }
}
