//! Representation of values (shared between [Const] and in future [TypeArg])
//!
//! [Const]: crate::ops::Const
//! [TypeArg]: crate::types::type_param::TypeArg

use thiserror::Error;

use crate::ops::constant::{HugrIntWidthStore, HUGR_MAX_INT_WIDTH};
use crate::types::{Container, CustomType, HashableType, PrimType, SimpleType};
use crate::{
    ops::constant::{ConstValue, HugrIntValueStore},
    types::TypeRow,
};

/// A constant value/instance of a [HashableType]. Note there is no
/// equivalent of [HashableType::Variable]; we can't have instances of that.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HashableValue {
    /// A string, i.e. corresponding to [HashableType::String]
    String(String),
    /// An integer, i.e. an instance of all [HashableType::Int]s of sufficient width
    Int(HugrIntValueStore),
    /// A container of other hashable values
    Container(ContainerValue<HashableValue>),
}

/// Trait for classes which represent values of some kind of [PrimType]
pub trait ValueOfType: Clone {
    /// The exact type whose values the type implementing [ValueOfType] represents
    type T: PrimType;

    /// Checks that a value can be an instance of the specified type.
    fn check_type(&self, ty: &Self::T) -> Result<(), ConstTypeError>;

    /// Unique name of the constant/value.
    fn name(&self) -> String;

    /// When there is an error fitting a [ContainerValue] of these values
    /// into a [Container] (type), produce a [ConstTypeError::ValueCheckFail] for that.
    fn container_error(typ: Container<Self::T>, vals: ContainerValue<Self>) -> ConstTypeError;
}

impl ValueOfType for HashableValue {
    type T = HashableType;

    fn name(&self) -> String {
        match self {
            HashableValue::String(s) => format!("const:string:\"{}\"", s),
            HashableValue::Int(v) => format!("const:int:{}", v),
            HashableValue::Container(c) => c.desc(),
        }
    }

    fn check_type(&self, ty: &HashableType) -> Result<(), ConstTypeError> {
        if let HashableType::Container(Container::Alias(s)) = ty {
            return Err(ConstTypeError::NoAliases(s.to_string()));
        };
        match self {
            HashableValue::String(_) => {
                if let HashableType::String = ty {
                    return Ok(());
                };
            }
            HashableValue::Int(value) => {
                if let HashableType::Int(width) = ty {
                    return check_int_fits_in_width(*value, *width).map_err(ConstTypeError::Int);
                };
            }
            HashableValue::Container(vals) => {
                if let HashableType::Container(c_ty) = ty {
                    return vals.check_container(c_ty);
                };
            }
        }
        Err(ConstTypeError::ValueCheckFail(
            ty.clone().into(),
            ConstValue::Hashable(self.clone()),
        ))
    }

    fn container_error(
        typ: Container<HashableType>,
        vals: ContainerValue<HashableValue>,
    ) -> ConstTypeError {
        ConstTypeError::ValueCheckFail(
            HashableType::Container(typ).into(),
            ConstValue::Hashable(HashableValue::Container(vals)),
        )
    }
}

/// A value that is a container of other values, e.g. a tuple or sum;
/// thus, corresponding to [Container]. Note there is no member
/// corresponding to [Container::Alias]; such types must have been
/// resolved to concrete types in order to create instances (values),
/// nor to [Container::Opaque], which is left to classes for broader
/// sets of values (see e.g. [ConstValue::Opaque])
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ContainerValue<T> {
    /// A [Container::Array] or [Container::Tuple] or [Container::List]
    Sequence(Vec<T>),
    /// A [Container::Map]
    Map(Vec<(HashableValue, T)>), // TODO try to make this an actual map?
    /// A [Container::Sum] - for any Sum type where this value meets
    /// the type of the variant indicated by the tag
    Sum(usize, Box<T>), // Tag and value
}

impl<Elem: ValueOfType> ContainerValue<Elem> {
    pub(crate) fn desc(&self) -> String {
        match self {
            ContainerValue::Sequence(vals) => {
                let names: Vec<_> = vals.iter().map(ValueOfType::name).collect();
                format!("const:seq:{{{}}}", names.join(", "))
            }
            ContainerValue::Map(_) => "a map".to_string(),
            ContainerValue::Sum(tag, val) => format!("const:sum:{{tag:{tag}, val:{}}}", val.name()),
        }
    }
    pub(crate) fn check_container(&self, ty: &Container<Elem::T>) -> Result<(), ConstTypeError> {
        match (self, ty) {
            (ContainerValue::Sequence(elems), Container::List(elem_ty)) => {
                for elem in elems {
                    elem.check_type(elem_ty)?;
                }
                Ok(())
            }
            (ContainerValue::Sequence(elems), Container::Tuple(tup_tys)) => {
                if elems.len() != tup_tys.len() {
                    return Err(ConstTypeError::TupleWrongLength);
                }
                for (elem, ty) in elems.iter().zip(tup_tys.iter()) {
                    elem.check_type(ty)?;
                }
                Ok(())
            }
            (ContainerValue::Sequence(elems), Container::Array(elem_ty, sz)) => {
                if elems.len() != *sz {
                    return Err(ConstTypeError::TupleWrongLength);
                }
                for elem in elems {
                    elem.check_type(elem_ty)?;
                }
                Ok(())
            }
            (ContainerValue::Map(mappings), Container::Map(kv)) => {
                let (key_ty, val_ty) = &**kv;
                for (key, val) in mappings {
                    key.check_type(key_ty)?;
                    val.check_type(val_ty)?;
                }
                Ok(())
            }
            (ContainerValue::Sum(tag, value), Container::Sum(variants)) => {
                value.check_type(variants.get(*tag).ok_or(ConstTypeError::InvalidSumTag)?)
            }
            (_, Container::Alias(s)) => Err(ConstTypeError::NoAliases(s.to_string())),
            (_, _) => Err(ValueOfType::container_error(ty.clone(), self.clone())),
        }
    }

    pub(crate) fn map_vals<T2: ValueOfType>(&self, f: &impl Fn(Elem) -> T2) -> ContainerValue<T2> {
        match self {
            ContainerValue::Sequence(vals) => {
                ContainerValue::Sequence(vals.iter().cloned().map(f).collect())
            }
            ContainerValue::Map(_) => todo!(),
            ContainerValue::Sum(tag, value) => {
                ContainerValue::Sum(*tag, Box::new(f((**value).clone())))
            }
        }
    }
}

pub(crate) fn map_container_type<T: PrimType, T2: PrimType>(
    container: &Container<T>,
    f: &impl Fn(T) -> T2,
) -> Container<T2> {
    fn map_row<T: PrimType, T2: PrimType>(
        row: &TypeRow<T>,
        f: &impl Fn(T) -> T2,
    ) -> Box<TypeRow<T2>> {
        Box::new(TypeRow::from(
            (*row)
                .to_owned()
                .into_owned()
                .into_iter()
                .map(f)
                .collect::<Vec<T2>>(),
        ))
    }
    match container {
        Container::List(elem) => Container::List(Box::new(f(*(elem).clone()))),
        Container::Map(kv) => {
            let (k, v) = (**kv).clone();
            Container::Map(Box::new((k, f(v))))
        }
        Container::Tuple(elems) => Container::Tuple(map_row(elems, f)),
        Container::Sum(variants) => Container::Sum(map_row(variants, f)),
        Container::Array(elem, sz) => Container::Array(Box::new(f((**elem).clone())), *sz),
        Container::Alias(s) => Container::Alias(s.clone()),
        Container::Opaque(custom) => Container::Opaque(custom.clone()),
    }
}

use lazy_static::lazy_static;

use std::collections::HashSet;

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

lazy_static! {
    static ref VALID_WIDTHS: HashSet<HugrIntWidthStore> =
        HashSet::from_iter((0..8).map(|a| HugrIntWidthStore::pow(2, a)));
}

/// Per the spec, valid widths for integers are 2^n for all n in [0,7]
fn check_int_fits_in_width(
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

/// Struct for custom type check fails.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CustomCheckFail {
    /// The value had a specific type that was not what was expected
    #[error("Expected type: {0} but value was of type: {1}")]
    TypeMismatch(CustomType, CustomType),
    /// Any other message
    #[error("{0}")]
    Message(String),
}

/// Errors that arise from typechecking constants
#[derive(Clone, Debug, PartialEq, Error)]
pub enum ConstTypeError {
    /// There was some problem fitting a const int into its declared size
    #[error("Error with int constant")]
    Int(#[from] ConstIntError),
    /// Found a Var type constructor when we're checking a const val
    #[error("Type of a const value can't be Var")]
    ConstCantBeVar,
    /// Type we were checking against was an Alias.
    /// This should have been resolved to an actual type.
    #[error("Type of a const value can't be an Alias {0}")]
    NoAliases(String),
    /// The length of the tuple value doesn't match the length of the tuple type
    #[error("Tuple of wrong length")]
    TupleWrongLength,
    /// Tag for a sum value exceeded the number of variants
    #[error("Tag of Sum value is invalid")]
    InvalidSumTag,
    /// A mismatch between the type expected and the value.
    #[error("Value {1:?} does not match expected type {0}")]
    ValueCheckFail(SimpleType, ConstValue),
    /// Error when checking a custom value.
    #[error("Error when checking custom type: {0:?}")]
    CustomCheckFail(#[from] CustomCheckFail),
}

#[cfg(test)]
mod test {

    use super::*;
    use cool_asserts::assert_matches;

    #[test]
    fn test_biggest_int() {
        assert_matches!(check_int_fits_in_width(u128::MAX, 128), Ok(_))
    }

    #[test]
    fn test_odd_widths_invalid() {
        assert_matches!(
            check_int_fits_in_width(0, 3),
            Err(ConstIntError::IntWidthInvalid(_))
        );
    }

    #[test]
    fn test_zero_width_invalid() {
        assert_matches!(
            check_int_fits_in_width(0, 0),
            Err(ConstIntError::IntWidthInvalid(_))
        );
    }

    #[test]
    fn test_width_too_large() {
        assert_matches!(
            check_int_fits_in_width(0, 130),
            Err(ConstIntError::IntWidthTooLarge(_))
        );
    }
}
