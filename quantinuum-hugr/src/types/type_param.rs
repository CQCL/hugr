//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::extension::TypeDef

use itertools::Itertools;
use std::num::NonZeroU64;
use thiserror::Error;

use crate::extension::ExtensionRegistry;
use crate::extension::ExtensionSet;
use crate::extension::SignatureError;

use super::{check_typevar_decl, CustomType, Substitution, Type, TypeBound};

/// The upper non-inclusive bound of a [`TypeParam::BoundedNat`]
// A None inner value implies the maximum bound: u64::MAX + 1 (all u64 values valid)
#[derive(
    Clone, Debug, PartialEq, Eq, derive_more::Display, serde::Deserialize, serde::Serialize,
)]
#[display(fmt = "{}", "_0.map(|i|i.to_string()).unwrap_or(\"-\".to_string())")]
pub struct UpperBound(Option<NonZeroU64>);
impl UpperBound {
    fn valid_value(&self, val: u64) -> bool {
        match (val, self.0) {
            (0, _) | (_, None) => true,
            (val, Some(inner)) if NonZeroU64::new(val).unwrap() < inner => true,
            _ => false,
        }
    }
    fn contains(&self, other: &UpperBound) -> bool {
        match (self.0, other.0) {
            (None, _) => true,
            (Some(b1), Some(b2)) if b1 >= b2 => true,
            _ => false,
        }
    }

    /// Returns the value of the upper bound.
    pub fn value(&self) -> &Option<NonZeroU64> {
        &self.0
    }
}

/// A *kind* of [TypeArg]. Thus, a parameter declared by a [PolyFuncType] (e.g. [OpDef]),
/// specifying a value that may (resp. must) be provided to instantiate it.
///
/// [PolyFuncType]: super::PolyFuncType
/// [OpDef]: crate::extension::OpDef
#[derive(
    Clone, Debug, PartialEq, Eq, derive_more::Display, serde::Deserialize, serde::Serialize,
)]
#[non_exhaustive]
#[serde(tag = "tp")]
pub enum TypeParam {
    /// Argument is a [TypeArg::Type].
    Type {
        /// Bound for the type parameter.
        b: TypeBound,
    },
    /// Argument is a [TypeArg::BoundedNat] that is less than the upper bound.
    BoundedNat {
        /// Upper bound for the Nat parameter.
        bound: UpperBound,
    },
    /// Argument is a [TypeArg::Opaque], defined by a [CustomType].
    Opaque {
        /// The [CustomType] defining the parameter.
        ty: CustomType,
    },
    /// Argument is a [TypeArg::Sequence]. A list of indeterminate size containing parameters.
    List {
        /// The [TypeParam] contained in the list.
        param: Box<TypeParam>,
    },
    /// Argument is a [TypeArg::Sequence]. A tuple of parameters.
    #[display(fmt = "Tuple({})", "params.iter().map(|t|t.to_string()).join(\", \")")]
    Tuple {
        /// The [TypeParam]s contained in the tuple.
        params: Vec<TypeParam>,
    },
    /// Argument is a [TypeArg::Extensions]. A set of [ExtensionId]s.
    ///
    /// [ExtensionId]: crate::extension::ExtensionId
    Extensions,
}

impl TypeParam {
    /// [`TypeParam::BoundedNat`] with the maximum bound (`u64::MAX` + 1)
    pub const fn max_nat() -> Self {
        Self::BoundedNat {
            bound: UpperBound(None),
        }
    }

    /// [`TypeParam::BoundedNat`] with the stated upper bound (non-exclusive)
    pub const fn bounded_nat(upper_bound: NonZeroU64) -> Self {
        Self::BoundedNat {
            bound: UpperBound(Some(upper_bound)),
        }
    }

    fn contains(&self, other: &TypeParam) -> bool {
        match (self, other) {
            (TypeParam::Type { b: b1 }, TypeParam::Type { b: b2 }) => b1.contains(*b2),
            (TypeParam::BoundedNat { bound: b1 }, TypeParam::BoundedNat { bound: b2 }) => {
                b1.contains(b2)
            }
            (TypeParam::Opaque { ty: c1 }, TypeParam::Opaque { ty: c2 }) => c1 == c2,
            (TypeParam::List { param: e1 }, TypeParam::List { param: e2 }) => e1.contains(e2),
            (TypeParam::Tuple { params: es1 }, TypeParam::Tuple { params: es2 }) => {
                es1.len() == es2.len() && es1.iter().zip(es2).all(|(e1, e2)| e1.contains(e2))
            }
            (TypeParam::Extensions, TypeParam::Extensions) => true,
            _ => false,
        }
    }
}

impl From<TypeBound> for TypeParam {
    fn from(bound: TypeBound) -> Self {
        Self::Type { b: bound }
    }
}

impl From<Type> for TypeArg {
    fn from(ty: Type) -> Self {
        Self::Type { ty }
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
    /// Variable (used in type schemes only), that is not a [TypeArg::Type]
    /// or [TypeArg::Extensions] - see [TypeArg::new_var_use]
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
    /// Makes a TypeArg representing a use (occurrence) of the type variable
    /// with the specified DeBruijn index. For use within type schemes only:
    /// `bound` must match that with which the variable was declared.
    pub fn new_var_use(idx: usize, decl: TypeParam) -> Self {
        match decl {
            TypeParam::Type { b } => TypeArg::Type {
                ty: Type::new_var_use(idx, b),
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

    /// Much as [Type::validate], also checks that the type of any [TypeArg::Opaque]
    /// is valid and closed.
    pub(crate) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        match self {
            TypeArg::Type { ty } => ty.validate(extension_registry, var_decls),
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
                .try_for_each(|a| a.validate(extension_registry, var_decls)),
            TypeArg::Extensions { es: _ } => Ok(()),
            TypeArg::Variable {
                v: TypeArgVariable { idx, cached_decl },
            } => check_typevar_decl(var_decls, *idx, cached_decl),
        }
    }

    pub(crate) fn substitute(&self, t: &impl Substitution) -> Self {
        match self {
            TypeArg::Type { ty } => TypeArg::Type {
                ty: ty.substitute(t),
            },
            TypeArg::BoundedNat { .. } => self.clone(), // We do not allow variables as bounds on BoundedNat's
            TypeArg::Opaque {
                arg: CustomTypeArg { typ, .. },
            } => {
                // The type must be equal to that declared (in a TypeParam) by the instantiated TypeDef,
                // so cannot contain variables declared by the instantiator (providing the TypeArgs)
                debug_assert_eq!(&typ.substitute(t), typ);
                self.clone()
            }
            TypeArg::Sequence { elems } => TypeArg::Sequence {
                elems: elems.iter().map(|ta| ta.substitute(t)).collect(),
            },
            TypeArg::Extensions { es } => TypeArg::Extensions {
                es: es.substitute(t),
            },
            TypeArg::Variable {
                v: TypeArgVariable { idx, cached_decl },
            } => t.apply_var(*idx, cached_decl),
        }
    }
}

impl TypeArgVariable {
    /// Return the index.
    pub fn index(&self) -> usize {
        self.idx
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
        (TypeArg::Type { ty }, TypeParam::Type { b: bound })
            if bound.contains(ty.least_upper_bound()) =>
        {
            Ok(())
        }
        (TypeArg::Sequence { elems }, TypeParam::List { param }) => {
            elems.iter().try_for_each(|arg| check_type_arg(arg, param))
        }
        (TypeArg::Sequence { elems: items }, TypeParam::Tuple { params: types }) => {
            if items.len() != types.len() {
                Err(TypeArgError::WrongNumberTuple(items.len(), types.len()))
            } else {
                items
                    .iter()
                    .zip(types.iter())
                    .try_for_each(|(arg, param)| check_type_arg(arg, param))
            }
        }
        (TypeArg::BoundedNat { n: val }, TypeParam::BoundedNat { bound })
            if bound.valid_value(*val) =>
        {
            Ok(())
        }

        (TypeArg::Opaque { arg }, TypeParam::Opaque { ty: param })
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
