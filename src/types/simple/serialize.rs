use super::Type;

use super::Container;

use super::HashableType;
use super::PrimType;
use super::TypeTag;

use itertools::Itertools;
use smol_str::SmolStr;

use super::super::custom::CustomType;

use super::TypeRow;

use super::Type;

use super::super::AbstractSignature;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(crate) enum SerSimpleType {
    Q,
    I,
    S,
    G {
        signature: Box<AbstractSignature>,
    },
    Tuple {
        row: Vec<SerSimpleType>,
        c: TypeTag,
    },
    Sum {
        row: Vec<SerSimpleType>,
        c: TypeTag,
    },
    Array {
        inner: Box<SerSimpleType>,
        len: usize,
        c: TypeTag,
    },
    Opaque {
        custom: CustomType,
        c: TypeTag,
    },
    Alias {
        name: SmolStr,
        c: TypeTag,
    },
    Var {
        name: SmolStr,
    },
}

trait SerializableType: PrimType {
    const TAG: TypeTag;
}

impl SerializableType for Type {
    const TAG: TypeTag = TypeTag::Copyable;
}

impl SerializableType for Type {
    const TAG: TypeTag = TypeTag::Simple;
}

impl SerializableType for HashableType {
    const TAG: TypeTag = TypeTag::Eq;
}

impl<T: SerializableType> From<Container<T>> for SerSimpleType
where
    SerSimpleType: From<T>,
    Type: From<T>,
{
    fn from(value: Container<T>) -> Self {
        match value {
            Container::Sum(inner) => SerSimpleType::Sum {
                row: inner.into_owned().into_iter().map_into().collect(),
                c: T::TAG, // We could inspect inner.containing_tag(), but this should have been done already
            },
            Container::Tuple(inner) => SerSimpleType::Tuple {
                row: inner.into_owned().into_iter().map_into().collect(),
                c: T::TAG,
            },
            Container::Array(inner, len) => SerSimpleType::Array {
                inner: Box::new((*inner).into()),
                len,
                c: T::TAG,
            },
            Container::Alias(name) => SerSimpleType::Alias { name, c: T::TAG },
            Container::Opaque(custom) => SerSimpleType::Opaque { custom, c: T::TAG },
        }
    }
}

impl From<HashableType> for SerSimpleType {
    fn from(value: HashableType) -> Self {
        match value {
            HashableType::Variable(s) => SerSimpleType::Var { name: s },
            HashableType::USize => SerSimpleType::I,
            HashableType::String => SerSimpleType::S,
            HashableType::Container(c) => c.into(),
        }
    }
}

impl From<Type> for SerSimpleType {
    fn from(value: Type) -> Self {
        match value {
            Type::Graph(inner) => SerSimpleType::G {
                signature: Box::new(*inner),
            },
            Type::Container(c) => c.into(),
            Type::Hashable(h) => h.into(),
        }
    }
}

impl From<Type> for SerSimpleType {
    fn from(value: Type) -> Self {
        match value {
            Type::Classic(c) => c.into(),
            Type::Qubit => SerSimpleType::Q,
            Type::Qontainer(c) => c.into(),
        }
    }
}

fn try_convert_list<T: TryInto<T2>, T2: TypeRowElem>(
    values: Vec<T>,
) -> Result<TypeRow<T2>, T::Error> {
    let vals = values
        .into_iter()
        .map(T::try_into)
        .collect::<Result<Vec<T2>, T::Error>>()?;
    Ok(TypeRow::from(vals))
}

macro_rules! handle_container {
   ($tag:ident, $variant:ident($($r:expr),*)) => {
        match $tag {
            TypeTag::Simple => (Container::<Type>::$variant($($r),*)).into(),
            TypeTag::Copyable => (Container::<Type>::$variant($($r),*)).into(),
            TypeTag::Eq => (Container::<HashableType>::$variant($($r),*)).into()
        }
    }
}

impl From<SerSimpleType> for Type {
    fn from(value: SerSimpleType) -> Self {
        match value {
            SerSimpleType::Q => Type::Qubit,
            SerSimpleType::I => HashableType::USize.into(),
            SerSimpleType::S => HashableType::String.into(),
            SerSimpleType::G { signature } => Type::Graph(Box::new(*signature)).into(),
            SerSimpleType::Tuple { row: inner, c } => {
                handle_container!(c, Tuple(Box::new(try_convert_list(inner).unwrap())))
            }
            SerSimpleType::Sum { row: inner, c } => {
                handle_container!(c, Sum(Box::new(try_convert_list(inner).unwrap())))
            }
            SerSimpleType::Array { inner, len, c } => {
                handle_container!(c, Array(Box::new((*inner).try_into().unwrap()), len))
            }
            SerSimpleType::Alias { name: s, c } => handle_container!(c, Alias(s)),
            SerSimpleType::Opaque { custom, c } => {
                handle_container!(c, Opaque(custom))
            }
            SerSimpleType::Var { name: s } => {
                Type::Hashable(HashableType::Variable(s)).into()
            }
        }
    }
}

impl TryFrom<SerSimpleType> for Type {
    type Error = String;

    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        let s: Type = value.into();
        if let Type::Classic(c) = s {
            Ok(c)
        } else {
            Err(format!("Not a Type: {}", s))
        }
    }
}

impl TryFrom<SerSimpleType> for HashableType {
    type Error = String;
    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        match value.try_into()? {
            Type::Hashable(h) => Ok(h),
            ty => Err(format!("Classic type is not hashable: {}", ty)),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::hugr::serialize::test::ser_roundtrip;
    use crate::types::custom::test::CLASSIC_T;
    use crate::types::{Type, Container, HashableType, Type};

    #[test]
    fn serialize_types_roundtrip() {
        // A Simple tuple
        let t = Type::new_tuple(vec![
            Type::Qubit,
            Type::from(HashableType::USize),
        ]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Classic sum
        let t = Type::new_sum(vec![
            Type::Classic(Type::Hashable(HashableType::USize)),
            Type::Classic(CLASSIC_T),
        ]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Hashable list
        let t = Type::Classic(Type::Hashable(HashableType::Container(
            Container::Array(Box::new(HashableType::USize), 3),
        )));
        assert_eq!(ser_roundtrip(&t), t);
    }

    #[test]
    fn serialize_types_current_behaviour() {
        // This list should be represented as a HashableType::Container.
        let malformed = Type::Qontainer(Container::Array(
            Box::new(Type::Classic(Type::Hashable(
                HashableType::USize,
            ))),
            6,
        ));
        // If this behaviour changes, i.e. to return the well-formed version, that'd be fine.
        // Just to document current serialization behaviour that we leave it untouched.
        assert_eq!(ser_roundtrip(&malformed), malformed);
    }
}
