//! Representation of values (shared between [Const] and in future [TypeArg])
//!
//! [Const]: crate::ops::Const
//! [TypeArg]: crate::types::type_param::TypeArg

use thiserror::Error;

use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::type_row::TypeRowElem;
use crate::types::{ClassicType, Container, CustomType, HashableType};
use crate::{
    ops::constant::{
        typecheck::{check_int_fits_in_width, ConstIntError},
        ConstValue, HugrIntValueStore,
    },
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

/// Trait for classes which represent values of some kind of type
/// - either [TypeParam] or an impl of [PrimType].
///
/// [PrimType]: crate::types::PrimType
pub trait ValueOfType: Clone {
    /// The exact type whose values the type implementing [ValueOfType] represents
    type T: TypeRowElem;

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
            ClassicType::Hashable(ty.clone()),
            ConstValue::Hashable(self.clone()),
        ))
    }

    fn container_error(
        typ: Container<HashableType>,
        vals: ContainerValue<HashableValue>,
    ) -> ConstTypeError {
        ConstTypeError::ValueCheckFail(
            ClassicType::Hashable(HashableType::Container(typ)),
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

pub(crate) fn map_container_type<T: TypeRowElem, T2: TypeRowElem>(
    container: &Container<T>,
    f: &impl Fn(T) -> T2,
) -> Container<T2> {
    fn map_row<T: TypeRowElem, T2: TypeRowElem>(
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

/// Struct for custom type check fails.
#[derive(Clone, Debug, PartialEq, Error)]
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
    ValueCheckFail(ClassicType, ConstValue),
    /// A mismatch between an expected TypeParam and an actual TypeArg
    #[error("TypeArg {1:?} does not match expected TypeParam {0:?}")]
    TypeArgCheckFail(TypeParam, TypeArg),
    /// Error when checking a custom value.
    #[error("Error when checking custom type: {0:?}")]
    CustomCheckFail(#[from] CustomCheckFail),
}
