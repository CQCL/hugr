//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::extension::TypeDef

use std::num::NonZeroU64;

use thiserror::Error;

use crate::extension::ExtensionRegistry;
use crate::extension::ExtensionSet;
use crate::extension::SignatureError;

use super::CustomType;
use super::Type;
use super::TypeBound;

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
/// The upper non-inclusive bound of a [`TypeParam::BoundedNat`]
// A None inner value implies the maximum bound: u64::MAX + 1 (all u64 values valid)
pub struct UpperBound(Option<NonZeroU64>);
impl UpperBound {
    fn valid_value(&self, val: u64) -> bool {
        match (val, self.0) {
            (0, _) | (_, None) => true,
            (val, Some(inner)) if NonZeroU64::new(val).unwrap() < inner => true,
            _ => false,
        }
    }
}

/// A parameter declared by an OpDef. Specifies a value
/// that must be provided by each operation node.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeParam {
    /// Argument is a [TypeArg::Type].
    Type(TypeBound),
    /// Argument is a [TypeArg::BoundedNat] that is less than the upper bound.
    BoundedNat(UpperBound),
    /// Argument is a [TypeArg::Opaque], defined by a [CustomType].
    Opaque(CustomType),
    /// Argument is a [TypeArg::Sequence]. A list of indeterminate size containing parameters.
    List(Box<TypeParam>),
    /// Argument is a [TypeArg::Sequence]. A tuple of parameters.
    Tuple(Vec<TypeParam>),
    /// Argument is a [TypeArg::Extensions]. A set of [ExtensionId]s.
    ///
    /// [ExtensionId]: crate::extension::ExtensionId
    Extensions,
}

impl TypeParam {
    /// [`TypeParam::BoundedNat`] with the maximum bound (`u64::MAX` + 1)
    pub const fn max_nat() -> Self {
        Self::BoundedNat(UpperBound(None))
    }

    /// [`TypeParam::BoundedNat`] with the stated upper bound (non-exclusive)
    pub const fn bounded_nat(upper_bound: NonZeroU64) -> Self {
        Self::BoundedNat(UpperBound(Some(upper_bound)))
    }

    fn contains(&self, other: &TypeParam) -> bool {
        match (self, other) {
            (TypeParam::Type(b1), TypeParam::Type(b2)) => b1.contains(*b2),
            (TypeParam::BoundedNat(b1), TypeParam::BoundedNat(b2)) => match (b1.0, b2.0) {
                (None, _) => true,
                (Some(b1), Some(b2)) if b1 >= b2 => true,
                _ => false,
            },
            (TypeParam::Opaque(c1), TypeParam::Opaque(c2)) => c1 == c2,
            (TypeParam::List(e1), TypeParam::List(e2)) => e1.contains(e2),
            (TypeParam::Tuple(es1), TypeParam::Tuple(es2)) => {
                es1.len() == es2.len() && es1.iter().zip(es2).all(|(e1, e2)| e1.contains(e2))
            }
            (TypeParam::Extensions, TypeParam::Extensions) => true,
            _ => false,
        }
    }
}

/// A statically-known argument value to an operation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
#[serde(tag = "tya")]
pub enum TypeArg {
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::Type]
    Type {
        #[allow(missing_docs)]
        ty: Type,
    },
    /// Instance of [TypeParam::BoundedNat]. 64-bit unsigned integer.
    BoundedNat {
        #[allow(missing_docs)]
        n: u64,
    },
    ///Instance of [TypeParam::Opaque] An opaque value, stored as serialized blob.
    Opaque {
        #[allow(missing_docs)]
        arg: CustomTypeArg,
    },
    /// Instance of [TypeParam::List] or [TypeParam::Tuple], defined by a
    /// sequence of elements.
    Sequence {
        #[allow(missing_docs)]
        elems: Vec<TypeArg>,
    },
    /// Instance of [TypeParam::Extensions], providing the extension ids.
    Extensions {
        #[allow(missing_docs)]
        es: ExtensionSet,
    },
    /// Variable (used in type schemes only) - see [TypeArg::use_var]
    Variable {
        #[allow(missing_docs)]
        v: TypeArgVariable,
    },
}

/// Variable in a TypeArg, that is not a [TypeArg::Type] or [TypeArg::Extensions],
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct TypeArgVariable {
    idx: usize,
    cached_decl: TypeParam,
}

impl TypeArg {
    /// Makes a TypeArg representing the type variable with the specified (DeBruijn) index
    /// and declared [TypeParam].
    pub fn use_var(idx: usize, decl: TypeParam) -> Self {
        match decl {
            TypeParam::Type(b) => TypeArg::Type {
                ty: Type::new_variable(idx, b),
            },
            TypeParam::Extensions => TypeArg::Extensions {
                es: ExtensionSet::type_var(idx),
            },
            _ => TypeArg::Variable {
                v: TypeArgVariable {
                    idx,
                    cached_decl: decl,
                },
            },
        }
    }

    pub(crate) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        type_vars: &[TypeParam],
    ) -> Result<(), SignatureError> {
        match self {
            TypeArg::Type { ty } => ty.validate(extension_registry, type_vars),
            TypeArg::BoundedNat { .. } => Ok(()),
            TypeArg::Opaque { arg: custarg } => {
                // We could also add a facility to Extension to validate that the constant *value*
                // here is a valid instance of the type.
                // The type must be equal to that declared (in a TypeParam) by the instantiated TypeDef,
                // so cannot contain variables declared by the instantiator (providing the TypeArgs)
                custarg.typ.validate(extension_registry, &[])
            }
            TypeArg::Sequence { elems } => elems
                .iter()
                .try_for_each(|a| a.validate(extension_registry, type_vars)),
            TypeArg::Extensions { es: _ } => Ok(()),
            TypeArg::Variable {
                v: TypeArgVariable { idx, cached_decl },
            } => {
                if type_vars.get(*idx) == Some(cached_decl) {
                    Ok(())
                } else {
                    Err(SignatureError::TypeVarDoesNotMatchDeclaration {
                        used: cached_decl.clone(),
                        decl: type_vars.get(*idx).cloned(),
                    })
                }
            }
        }
    }

    pub(super) fn substitute(&self, exts: &ExtensionRegistry, args: &[TypeArg]) -> Self {
        match self {
            TypeArg::Type { ty } => TypeArg::Type {
                ty: ty.substitute(exts, args),
            },
            TypeArg::BoundedNat { .. } => self.clone(), // We do not allow variables as bounds on BoundedNat's
            TypeArg::Opaque {
                arg: CustomTypeArg { typ, .. },
            } => {
                // The type must be equal to that declared (in a TypeParam) by the instantiated TypeDef,
                // so cannot contain variables declared by the instantiator (providing the TypeArgs)
                debug_assert_eq!(&typ.substitute(exts, args), typ);
                self.clone()
            }
            TypeArg::Sequence { elems } => TypeArg::Sequence {
                elems: elems.iter().map(|ta| ta.substitute(exts, args)).collect(),
            },
            TypeArg::Extensions { es } => TypeArg::Extensions {
                es: es.substitute(args),
            },
            TypeArg::Variable {
                v: TypeArgVariable { idx, .. },
            } => args
                .get(*idx)
                .expect("validate + check_type_args should rule this out")
                .clone(),
        }
    }
}

/// A serialized representation of a value of a [CustomType]
/// restricted to equatable types.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CustomTypeArg {
    /// The type of the constant.
    /// (Exact matches only - the constant is exactly this type.)
    pub typ: CustomType,
    /// Serialized representation.
    pub value: serde_yaml::Value,
}

impl CustomTypeArg {
    /// Create a new CustomTypeArg. Enforces that the type must be checkable for
    /// equality.
    pub fn new(typ: CustomType, value: serde_yaml::Value) -> Result<Self, &'static str> {
        if typ.bound() == TypeBound::Eq {
            Ok(Self { typ, value })
        } else {
            Err("Only TypeBound::Eq CustomTypes can be used as TypeArgs")
        }
    }
}

/// Checks a [TypeArg] is as expected for a [TypeParam]
pub fn check_type_arg(arg: &TypeArg, param: &TypeParam) -> Result<(), TypeArgError> {
    match (arg, param) {
        (
            TypeArg::Variable {
                v: TypeArgVariable { cached_decl, .. },
            },
            _,
        ) if param.contains(cached_decl) => Ok(()),
        (TypeArg::Type { ty }, TypeParam::Type(bound))
            if bound.contains(ty.least_upper_bound()) =>
        {
            Ok(())
        }
        (TypeArg::Sequence { elems }, TypeParam::List(param)) => {
            elems.iter().try_for_each(|arg| check_type_arg(arg, param))
        }
        (TypeArg::Sequence { elems: items }, TypeParam::Tuple(types)) => {
            if items.len() != types.len() {
                Err(TypeArgError::WrongNumberTuple(items.len(), types.len()))
            } else {
                items
                    .iter()
                    .zip(types.iter())
                    .try_for_each(|(arg, param)| check_type_arg(arg, param))
            }
        }
        (TypeArg::BoundedNat { n: val }, TypeParam::BoundedNat(bound))
            if bound.valid_value(*val) =>
        {
            Ok(())
        }

        (TypeArg::Opaque { arg }, TypeParam::Opaque(param))
            if param.bound() == TypeBound::Eq && &arg.typ == param =>
        {
            Ok(())
        }
        (TypeArg::Extensions { .. }, TypeParam::Extensions) => Ok(()),
        _ => Err(TypeArgError::TypeMismatch {
            arg: arg.clone(),
            param: param.clone(),
        }),
    }
}

/// Check a list of type arguments match a list of required type parameters
pub fn check_type_args(args: &[TypeArg], params: &[TypeParam]) -> Result<(), TypeArgError> {
    if args.len() != params.len() {
        return Err(TypeArgError::WrongNumberArgs(args.len(), params.len()));
    }
    for (a, p) in args.iter().zip(params.iter()) {
        check_type_arg(a, p)?;
    }
    Ok(())
}

/// Errors that can occur fitting a [TypeArg] into a [TypeParam]
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TypeArgError {
    #[allow(missing_docs)]
    /// For now, general case of a type arg not fitting a param.
    /// We'll have more cases when we allow general Containers.
    // TODO It may become possible to combine this with ConstTypeError.
    #[error("Type argument {arg:?} does not fit declared parameter {param:?}")]
    TypeMismatch { param: TypeParam, arg: TypeArg },
    /// Wrong number of type arguments (actual vs expected).
    // For now this only happens at the top level (TypeArgs of op/type vs TypeParams of Op/TypeDef).
    // However in the future it may be applicable to e.g. contents of Tuples too.
    #[error("Wrong number of type arguments: {0} vs expected {1} declared type parameters")]
    WrongNumberArgs(usize, usize),

    /// Wrong number of type arguments in tuple (actual vs expected).
    #[error("Wrong number of type arguments to tuple parameter: {0} vs expected {1} declared type parameters")]
    WrongNumberTuple(usize, usize),
    /// Opaque value type check error.
    #[error("Opaque type argument does not fit declared parameter type: {0:?}")]
    OpaqueTypeMismatch(#[from] crate::types::CustomCheckFailure),
    /// Invalid value
    #[error("Invalid value of type argument")]
    InvalidValue(TypeArg),
}
