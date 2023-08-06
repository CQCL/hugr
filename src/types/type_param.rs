//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::resource::TypeDef

use crate::ops::constant::typecheck::check_int_fits_in_width;
use crate::ops::constant::HugrIntValueStore;
use crate::values::{ValueError, ValueOfType};

use super::simple::{HashableElem, Tagged};
use super::{simple::Container, ClassicType, HashableType, SimpleType, TypeTag};

/// A parameter declared by an OpDef. Specifies a value
/// that must be provided by each operation node.
// TODO any other 'leaf' types? We specifically do not want float.
// bool should eventually be a Sum type (Container).
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeParam {
    /// Argument is a [TypeArg::Type] - classic or linear
    Type,
    /// Argument is a [TypeArg::ClassicType] - hashable or otherwise
    ClassicType,
    /// Argument is a [TypeArg::HashableType]
    HashableType,
    /// Node must provide a [TypeArg::List] (of whatever length)
    /// TODO add missing [Container]-equivalents: Tuple+Sum (no Array,
    /// but keep List even when that's dropped from Container)
    List(Box<TypeParam>),
    /// Argument is a value of the specified type.
    /// TODO It'd be better to use HashableLeaf with a better notion of Container.
    Value(HashableType),
}

/// A statically-known argument value to an operation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeArg {
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::Type]
    Type(SimpleType),
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::ClassicType],
    /// it'll get one of these (rather than embedding inside a Type)
    ClassicType(ClassicType),
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::HashableType],
    /// this is the value.
    HashableType(HashableType),
    /// Where the (Type/Op)Def declares a [TypeParam::Value] of type [HashableType::Int], a constant value thereof
    Int(HugrIntValueStore),
    /// Where the (Type/Op)Def declares a [TypeParam::Value] of type [HashableType::String], here it is
    String(String),
    /// Where the (Type/Op)Def declares a [TypeParam::List]`<T>` - all elements will implicitly
    /// be of the same variety of TypeArg, i.e. `T`s.
    List(Vec<TypeArg>),
    /// Where the TypeDef declares a [TypeParam::Value] of [Container::Opaque]
    CustomValue(serde_yaml::Value),
}

pub type TypeArgError = ValueError<TypeArg>;

impl TypeArg {
    /// Report [`TypeTag`] if param is a type
    pub fn tag_of_type(&self) -> Option<TypeTag> {
        match self {
            TypeArg::Type(s) => Some(s.tag()),
            TypeArg::ClassicType(c) => Some(c.tag()),
            _ => None,
        }
    }
}

impl ValueOfType for TypeArg {
    type T = TypeParam;

    fn name(&self) -> String {
        todo!()
    }
    /// Checks a [TypeArg] is as expected for a [TypeParam]
    fn check_type(self: &TypeArg, param: &TypeParam) -> Result<(), TypeArgError> {
        match (self, param) {
            (TypeArg::Type(_), TypeParam::Type) => Ok(()),
            (TypeArg::ClassicType(_), TypeParam::ClassicType) => Ok(()),
            (TypeArg::HashableType(_), TypeParam::HashableType) => Ok(()),
            (TypeArg::List(items), TypeParam::List(ty)) => {
                for item in items {
                    item.check_type(ty.as_ref())?;
                }
                Ok(())
            }
            (TypeArg::Int(v), TypeParam::Value(HashableType::Single(HashableElem::Int(width)))) => {
                check_int_fits_in_width(*v, *width).map_err(ValueError::Int)
            }
            (TypeArg::String(_), TypeParam::Value(HashableType::Single(HashableElem::String))) => {
                Ok(())
            }
            (TypeArg::CustomValue(_), TypeParam::Value(HashableType::Opaque(_))) => Ok(()), // TODO more checks here, e.g. storing CustomType in the value
            (arg, TypeParam::Value(Container::List(elem))) => {
                // Do we just fail here, as the LHS value must include types, and the RHS clearly does not?
                // (This is because we have not yet properly separated TypeArg into Leaf and container-varant,
                //  and are still *stealing* HashableType = Container<HashableElem>)
                arg.check_type(&TypeParam::List(Box::new(TypeParam::Value(**elem))))
            }
            (TypeArg::List(items), TypeParam::Value(Container::Array(elem, sz))) => {
                if items.len() != *sz {
                    return Err(ValueError::WrongNumber("array elements", items.len(), *sz));
                }
                let elem_ty = TypeParam::Value(**elem);
                for item in items {
                    item.check_type(&elem_ty)?;
                }
                Ok(())
            }
            (TypeArg::List(items), TypeParam::Value(Container::Tuple(tys))) => {
                if items.len() != tys.len() {
                    return Err(ValueError::WrongNumber(
                        "tuple elements",
                        items.len(),
                        tys.len(),
                    ));
                }
                for (i, t) in items.iter().zip(tys.iter()) {
                    i.check_type(&TypeParam::Value(*t))?;
                }
                Ok(())
            }
            (_, TypeParam::Value(Container::Alias(n))) => Err(ValueError::NoAliases(n.to_string())),
            _ => Err(ValueError::ValueCheckFail(param.clone(), self.clone())),
        }
    }
}
