use std::convert::Infallible;

use super::{ClassicElem, HashableElem, SimpleElem, SimpleType};

use super::Container;

use super::PrimType;
use super::TypeTag;

use smol_str::SmolStr;

use super::super::custom::CustomType;

use super::TypeRow;

use super::super::AbstractSignature;

use crate::ops::constant::HugrIntWidthStore;
use crate::types::type_row::TypeRowElem;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(crate) enum SerSimpleType {
    Q,
    I {
        width: HugrIntWidthStore,
    },
    F,
    S,
    G {
        signature: Box<AbstractSignature>,
    },
    List {
        inner: Box<SerSimpleType>,
        c: TypeTag,
    },
    Map {
        k: Box<SerSimpleType>,
        v: Box<SerSimpleType>,
        c: TypeTag,
    },
    Tuple {
        row: Box<TypeRow<SerSimpleType>>,
        c: TypeTag,
    },
    Sum {
        row: Box<TypeRow<SerSimpleType>>,
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
impl TypeRowElem for SerSimpleType {}

pub(super) trait SerializableType:
    PrimType + Into<SerSimpleType> + TryFrom<SimpleElem>
{
    // ALAN Note that we write the tag out quite a bit (in every container-y SerSimpleType),
    // but we never read it in. Now the tag we write will always be that of the declared
    // primitive/leaf type (of the outermost Container). Previous behaviour would have
    // written out the smallest tag containing all the leaves within this sub-Container
    // - which we would get by dynamic inspection of Tagged:tag(), without needing this. ???
    const TAG: TypeTag;
    fn from_ser(val: SerSimpleType) -> Result<Self, <Self as TryFrom<SimpleElem>>::Error> {
        let v: SimpleType = val.try_into().unwrap(); // Cannot fail for SimpleElem
                                                     // Believe we should only call this from above for single values. So make a separate trait??
        let SimpleType::Single(e) = v else {panic!("Found container {:?}", v)};
        e.try_into()
    }

    fn err_to_string(err: <Self as TryFrom<SimpleElem>>::Error) -> String;
}

impl SerializableType for ClassicElem {
    const TAG: TypeTag = TypeTag::Classic;
    fn err_to_string(err: String) -> String {
        err
    }
}

impl SerializableType for SimpleElem {
    const TAG: TypeTag = TypeTag::Simple;
    fn err_to_string(err: Infallible) -> String {
        panic!("Infallible!")
    }
}

impl SerializableType for HashableElem {
    const TAG: TypeTag = TypeTag::Hashable;
    fn err_to_string(err: String) -> String {
        err
    }
}

impl<T: SerializableType> From<Container<T>> for SerSimpleType {
    fn from(value: Container<T>) -> Self {
        match value {
            Container::Single(elem) => elem.into(),
            Container::Sum(inner) => SerSimpleType::Sum {
                row: Box::new(inner.map_into()),
                c: T::TAG, // We could inspect inner.containing_tag(), but this should have been done already
            },
            Container::List(inner) => SerSimpleType::List {
                inner: Box::new((*inner).into()),
                c: T::TAG, // We could inspect inner.tag(), but this should have been done already
            },
            Container::Tuple(inner) => SerSimpleType::Tuple {
                row: Box::new(inner.map_into()),
                c: T::TAG,
            },
            Container::Map(inner) => SerSimpleType::Map {
                k: {
                    let h: Container<HashableElem> = inner.0;
                    let k: SerSimpleType = h.into();
                    Box::new(k)
                },
                v: Box::new(inner.1.into()),
                c: T::TAG,
            },
            Container::Array(inner, len) => SerSimpleType::Array {
                inner: box_convert(*inner),
                len,
                c: T::TAG,
            },
            Container::Alias(name) => SerSimpleType::Alias { name, c: T::TAG },
            Container::Opaque(custom) => SerSimpleType::Opaque { custom, c: T::TAG },
        }
    }
}

impl From<HashableElem> for SerSimpleType {
    fn from(value: HashableElem) -> Self {
        match value {
            HashableElem::Variable(s) => SerSimpleType::Var { name: s },
            HashableElem::Int(w) => SerSimpleType::I { width: w },
            HashableElem::String => SerSimpleType::S,
        }
    }
}

impl From<ClassicElem> for SerSimpleType {
    fn from(value: ClassicElem) -> Self {
        match value {
            ClassicElem::F64 => SerSimpleType::F,
            ClassicElem::Graph(inner) => SerSimpleType::G {
                signature: Box::new(*inner),
            },
            ClassicElem::Hashable(h) => h.into(),
        }
    }
}

impl From<SimpleElem> for SerSimpleType {
    fn from(value: SimpleElem) -> Self {
        match value {
            SimpleElem::Classic(c) => c.into(),
            SimpleElem::Qubit => SerSimpleType::Q,
        }
    }
}

pub(crate) fn box_convert_try<T, F>(value: T) -> Result<Box<F>, <T as TryInto<F>>::Error>
where
    T: TryInto<F>,
{
    Ok(Box::new((value).try_into()?))
}

pub(crate) fn box_convert<T, F>(value: T) -> Box<F>
where
    T: Into<F>,
{
    Box::new((value).into())
}

impl<T: SerializableType> TryFrom<SerSimpleType> for Container<T> {
    type Error = String;
    fn try_from(value: SerSimpleType) -> Result<Self, String> {
        let elem: SimpleElem = match value {
            SerSimpleType::Q => SimpleElem::Qubit,
            SerSimpleType::I { width } => HashableElem::Int(width).into(),
            SerSimpleType::F => ClassicElem::F64.into(),
            SerSimpleType::S => HashableElem::String.into(),
            SerSimpleType::G { signature } => ClassicElem::Graph(Box::new(*signature)).into(),
            SerSimpleType::Tuple { row: inner, c } => {
                return Ok(Container::Tuple(Box::new(inner.try_convert_elems()?)))
            }
            SerSimpleType::Sum { row: inner, c } => {
                return Ok(Container::Sum(Box::new(inner.try_convert_elems()?)))
            }
            SerSimpleType::List { inner, c } => {
                return Ok(Container::List(box_convert_try(*inner)?))
            }
            SerSimpleType::Map { k, v, c } => {
                return Ok(Container::Map(Box::new((
                    (*k).try_into()?,
                    (*v).try_into()?,
                ))))
            }
            SerSimpleType::Array { inner, len, c } => {
                return Ok(Container::Array(box_convert_try(*inner)?, len))
            }

            SerSimpleType::Alias { name: s, c } => {
                return if T::TAG.union(c) == T::TAG {
                    Ok(Container::Alias(s))
                } else {
                    Err(format!("Alias of too-broad tag {} expected {}", c, T::TAG))
                }
            }
            SerSimpleType::Opaque { custom, c } => {
                return if T::TAG.union(c) == T::TAG {
                    Ok(Container::Opaque(custom))
                } else {
                    Err(format!("Opaque of too-broad tag {} expected {}", c, T::TAG))
                }
            }
            SerSimpleType::Var { name: s } => HashableElem::Variable(s).into(),
        };
        Ok(Container::Single(
            elem.try_into().map_err(T::err_to_string)?,
        ))
    }
}

/*impl TryFrom<SerSimpleType> for ClassicElem {
    type Error = String;

    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        let s: SimpleType = value.into();
        s.try_convert_elems(|e| match e {
            SimpleElem::Classic(c) => Ok(c),
            _ => Err(format!("Not classic: {}", e)),
        })
    }
}

impl TryFrom<SerSimpleType> for HashableType {
    type Error = String;
    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        let c: ClassicType = value.try_into()?;
        c.try_convert_elems(|c| match c {
            ClassicElem::Hashable(h) => Ok(h),
            _ => Err(format!("Not hashable: {}", c)),
        })
    }
}*/

#[cfg(test)]
mod test {
    use crate::hugr::serialize::test::ser_roundtrip;
    use crate::types::simple::{ClassicElem, HashableElem, SimpleElem};
    use crate::types::SimpleType;

    #[test]
    fn serialize_types_roundtrip() {
        // A Simple tuple
        let t = SimpleType::new_tuple(vec![
            SimpleType::Single(SimpleElem::Qubit),
            SimpleType::Single(ClassicElem::F64.into()),
        ]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Classic sum
        let t = SimpleType::new_sum(vec![
            SimpleType::Single(HashableElem::Int(4).into()),
            SimpleType::Single(ClassicElem::F64.into()),
        ]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Hashable list
        let t = SimpleType::List(Box::new(SimpleType::Single(HashableElem::Int(8).into())));
        assert_eq!(ser_roundtrip(&t), t);
    }
}
