use super::ClassicType;

use super::Container;

use super::HashableType;
use super::PrimType;
use super::TypeTag;

use itertools::Itertools;
use smol_str::SmolStr;

use super::super::custom::CustomType;

use super::TypeRow;

use super::SimpleType;

use super::super::signature::AbstractSignature;

use crate::types::type_row::TypeRowElem;

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

impl SerializableType for ClassicType {
    const TAG: TypeTag = TypeTag::Classic;
}

impl SerializableType for SimpleType {
    const TAG: TypeTag = TypeTag::Simple;
}

impl SerializableType for HashableType {
    const TAG: TypeTag = TypeTag::Hashable;
}

impl<T: SerializableType> From<Container<T>> for SerSimpleType
where
    SerSimpleType: From<T>,
    SimpleType: From<T>,
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

impl From<ClassicType> for SerSimpleType {
    fn from(value: ClassicType) -> Self {
        match value {
            ClassicType::Graph(inner) => SerSimpleType::G {
                signature: Box::new(*inner),
            },
            ClassicType::Container(c) => c.into(),
            ClassicType::Hashable(h) => h.into(),
        }
    }
}

impl From<SimpleType> for SerSimpleType {
    fn from(value: SimpleType) -> Self {
        match value {
            SimpleType::Classic(c) => c.into(),
            SimpleType::Qubit => SerSimpleType::Q,
            SimpleType::Qontainer(c) => c.into(),
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
            TypeTag::Simple => (Container::<SimpleType>::$variant($($r),*)).into(),
            TypeTag::Classic => (Container::<ClassicType>::$variant($($r),*)).into(),
            TypeTag::Hashable => (Container::<HashableType>::$variant($($r),*)).into()
        }
    }
}

impl From<SerSimpleType> for SimpleType {
    fn from(value: SerSimpleType) -> Self {
        match value {
            SerSimpleType::Q => SimpleType::Qubit,
            SerSimpleType::I => HashableType::USize.into(),
            SerSimpleType::S => HashableType::String.into(),
            SerSimpleType::G { signature } => ClassicType::Graph(Box::new(*signature)).into(),
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
                ClassicType::Hashable(HashableType::Variable(s)).into()
            }
        }
    }
}

impl TryFrom<SerSimpleType> for ClassicType {
    type Error = String;

    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        let s: SimpleType = value.into();
        if let SimpleType::Classic(c) = s {
            Ok(c)
        } else {
            Err(format!("Not a ClassicType: {}", s))
        }
    }
}

impl TryFrom<SerSimpleType> for HashableType {
    type Error = String;
    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        match value.try_into()? {
            ClassicType::Hashable(h) => Ok(h),
            ty => Err(format!("Classic type is not hashable: {}", ty)),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::hugr::serialize::test::ser_roundtrip;
    use crate::types::custom::test::CLASSIC_T;
    use crate::types::{ClassicType, Container, HashableType, SimpleType};

    #[test]
    fn serialize_types_roundtrip() {
        // A Simple tuple
        let t = SimpleType::new_tuple(vec![
            SimpleType::Qubit,
            SimpleType::from(HashableType::USize),
        ]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Classic sum
        let t = SimpleType::new_sum(vec![
            SimpleType::Classic(ClassicType::Hashable(HashableType::USize)),
            SimpleType::Classic(CLASSIC_T),
        ]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Hashable list
        let t = SimpleType::Classic(ClassicType::Hashable(HashableType::Container(
            Container::Array(Box::new(HashableType::USize), 3),
        )));
        assert_eq!(ser_roundtrip(&t), t);
    }

    #[test]
    fn serialize_types_current_behaviour() {
        // This list should be represented as a HashableType::Container.
        let malformed = SimpleType::Qontainer(Container::Array(
            Box::new(SimpleType::Classic(ClassicType::Hashable(
                HashableType::USize,
            ))),
            6,
        ));
        // If this behaviour changes, i.e. to return the well-formed version, that'd be fine.
        // Just to document current serialization behaviour that we leave it untouched.
        assert_eq!(ser_roundtrip(&malformed), malformed);
    }
}
