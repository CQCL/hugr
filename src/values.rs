//! Representation of values (shared between [Const] and in future [TypeArg])
//!
//! [Const]: crate::ops::Const
//! [TypeArg]: crate::types::type_param::TypeArg

use thiserror::Error;

use crate::ops::constant::{
    typecheck::{check_int_fits_in_width, ConstIntError},
    HugrIntValueStore,
};
use crate::types::simple::HashableElem;
use crate::types::{Container, CustomType, PrimType};

/// A constant value/instance of a [HashableType]. Note there is no
/// equivalent of [HashableType::Variable]; we can't have instances of that.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HashableLeaf {
    /// A string, i.e. corresponding to [HashableType::String]
    String(String),
    /// An integer, i.e. an instance of all [HashableType::Int]s of sufficient width
    Int(HugrIntValueStore),
}
pub type HashableValue = ContainerValue<HashableLeaf>;

/// Trait for classes which represent values of some kind of [PrimType]
pub trait ValueOfType: Clone {
    /// The exact type whose values the type implementing [ValueOfType] represents
    type T: std::fmt::Debug; // TODO: unclear what the bound here should be

    /// Checks that a value can be an instance of the specified type.
    fn check_type(&self, ty: &Self::T) -> Result<(), ValueError<Self>>;

    /// Unique name of the constant/value.
    fn name(&self) -> String;
}

impl ValueOfType for HashableLeaf {
    type T = HashableElem;

    fn name(&self) -> String {
        match self {
            HashableLeaf::String(s) => format!("const:string:\"{}\"", s),
            HashableLeaf::Int(v) => format!("const:int:{}", v),
        }
    }

    fn check_type(&self, ty: &HashableElem) -> Result<(), ValueError<HashableLeaf>> {
        match self {
            HashableLeaf::String(_) => {
                if let HashableElem::String = ty {
                    return Ok(());
                };
            }
            HashableLeaf::Int(value) => {
                if let HashableElem::Int(width) = ty {
                    return check_int_fits_in_width(*value, *width).map_err(ValueError::Int);
                };
            }
        }
        Err(ValueError::ValueCheckFail(ty.clone(), self.clone()))
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
    Single(T),
    /// A [Container::Array] or [Container::Tuple] or [Container::List]
    Sequence(Vec<ContainerValue<T>>),
    /// A [Container::Map]
    Map(Vec<(HashableValue, ContainerValue<T>)>), // TODO try to make this an actual map?
    /// A [Container::Sum] - for any Sum type where this value meets
    /// the type of the variant indicated by the tag
    Sum(usize, Box<ContainerValue<T>>), // Tag and value
}

impl<Leaf: ValueOfType> ValueOfType for ContainerValue<Leaf>
where
    Leaf::T: PrimType, // possibly also some other trait (ElemValue?) - check opaque
    HashableLeaf: Into<Leaf>,
    HashableElem: Into<Leaf::T>,
{
    type T = Container<Leaf::T>;

    fn name(&self) -> String {
        match self {
            ContainerValue::Single(e) => e.name(),
            ContainerValue::Sequence(vals) => {
                let names: Vec<_> = vals.iter().map(ValueOfType::name).collect();
                format!("const:seq:{{{}}}", names.join(", "))
            }
            ContainerValue::Map(_) => "a map".to_string(),
            ContainerValue::Sum(tag, val) => format!("const:sum:{{tag:{tag}, val:{}}}", val.name()),
        }
    }

    fn check_type(&self, ty: &Container<Leaf::T>) -> Result<(), ValueError<Self>> {
        match (self, ty) {
            (ContainerValue::Sequence(elems), Container::List(elem_ty)) => {
                for elem in elems {
                    elem.check_type(&**elem_ty)?;
                }
                Ok(())
            }
            (ContainerValue::Sequence(elems), Container::Tuple(tup_tys)) => {
                if elems.len() != tup_tys.len() {
                    return Err(ValueError::TupleWrongLength);
                }
                for (elem, ty) in elems.iter().zip(tup_tys.iter()) {
                    elem.check_type(ty)?;
                }
                Ok(())
            }
            (ContainerValue::Sequence(elems), Container::Array(elem_ty, sz)) => {
                if elems.len() != *sz {
                    return Err(ValueError::TupleWrongLength);
                }
                for elem in elems {
                    elem.check_type(elem_ty)?;
                }
                Ok(())
            }
            (ContainerValue::Map(mappings), Container::Map(kv)) => {
                let (key_ty, val_ty) = &**kv;
                for (key, val) in mappings {
                    key.check_type(key_ty)
                        .map_err(|e| e.map(|ty| ty.map_into(), |val| val.map_into()))?;
                    val.check_type(val_ty)?;
                }
                Ok(())
            }
            (ContainerValue::Sum(tag, value), Container::Sum(variants)) => {
                value.check_type(variants.get(*tag).ok_or(ValueError::InvalidSumTag)?)
            }
            (_, Container::Alias(s)) => Err(ValueError::NoAliases(s.to_string())),
            (_, _) => Err(ValueError::ValueCheckFail(ty.clone(), self.clone())),
        }
    }

    /*pub(crate) fn map_vals<T2: ValueOfType>(&self, f: &impl Fn(Elem) -> T2) -> ContainerValue<T2> {
        match self {
            ContainerValue::Sequence(vals) => {
                ContainerValue::Sequence(vals.iter().cloned().map(f).collect())
            }
            ContainerValue::Map(_) => todo!(),
            ContainerValue::Sum(tag, value) => {
                ContainerValue::Sum(*tag, Box::new(f((**value).clone())))
            }
        }
    }*/
}

/*pub(crate) fn map_container_type<T: PrimType, T2: PrimType>(
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
}*/

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

/// Errors that arise from typechecking values against types
#[derive(Clone, Debug, PartialEq, Error)]
pub enum ValueError<V: ValueOfType> {
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
    #[error("Value {1:?} does not match expected type {0:?}")]
    ValueCheckFail(V::T, V),
    /// Error when checking a custom value.
    #[error("Error when checking custom type: {0:?}")]
    CustomCheckFail(#[from] CustomCheckFail),
}

impl<V: ValueOfType> ValueError<V> {
    pub(crate) fn map_into<V2: ValueOfType>(self) -> ValueError<V2>
    where
        V: Into<V2>,
        V::T: Into<V2::T>,
    {
        self.map(V::T::into, V::into)
    }

    pub(crate) fn map<V2: ValueOfType>(
        self,
        ty_fn: impl Fn(V::T) -> V2::T,
        val_fn: impl Fn(V) -> V2,
    ) -> ValueError<V2> {
        match self {
            ValueError::Int(i) => ValueError::Int(i),
            ValueError::ConstCantBeVar => ValueError::ConstCantBeVar,
            ValueError::NoAliases(s) => ValueError::NoAliases(s),
            ValueError::TupleWrongLength => ValueError::TupleWrongLength,
            ValueError::InvalidSumTag => ValueError::InvalidSumTag,
            ValueError::ValueCheckFail(ty, val) => {
                ValueError::ValueCheckFail(ty_fn(ty), val_fn(val))
            }
            ValueError::CustomCheckFail(c) => ValueError::CustomCheckFail(c),
        }
    }
}
