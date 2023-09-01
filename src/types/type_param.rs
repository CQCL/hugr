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
pub enum TypeArg {
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::Type]
    Type(Type),
    /// Instance of [TypeParam::BoundedNat]. 64-bit unsigned integer.
    BoundedNat(u64),
    ///Instance of [TypeParam::Opaque] An opaque value, stored as serialized blob.
    Opaque(CustomTypeArg),
    /// Instance of [TypeParam::List] or [TypeParam::Tuple], defined by a
    /// sequence of arguments.
    Sequence(Vec<TypeArg>),
    /// Instance of [TypeParam::Extensions], providing the extension ids.
    Extensions(ExtensionSet),
    /// Type variable (used in type schemes only), with cache of the declaration.
    // Note that if the type variable is declared as a TypeParam::Type, we can represent
    // the typevar as *either* a TypeArg::Variable *or* a TypeArg::Type(Type::Variable),
    // and these two will behave equivalently.
    // This probably means we should not declare TypeArg as 'Eq'...
    Variable(usize, TypeParam),
}

impl TypeArg {
    pub(super) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        type_vars: &[TypeParam],
    ) -> Result<(), SignatureError> {
        match self {
            TypeArg::Type(ty) => ty.validate(extension_registry, type_vars),
            TypeArg::BoundedNat(_) => Ok(()),
            TypeArg::Opaque(custarg) => {
                // We could also add a facility to Extension to validate that the constant *value*
                // here is a valid instance of the type.
                // Moreover, passing the type_vars here means the constant could itself have a
                // type polymorphic in those vars - e.g. empty lists. User beware?!
                // TODO are we going deeper than we *need* here - would it be ok (not too restrictive) to insist,
                // no type vars here? Or,
                // TODO are we going deeper than *is correct* here, i.e. allowing dependent type(params) or something?
                custarg.typ.validate(extension_registry, type_vars)
            }
            TypeArg::Sequence(args) => args
                .iter()
                .try_for_each(|a| a.validate(extension_registry, type_vars)),
            TypeArg::Extensions(_) => Ok(()),
            TypeArg::Variable(idx, cache) => {
                if type_vars.get(*idx) == Some(cache) {
                    Ok(())
                } else {
                    Err(SignatureError::TypeVarDoesNotMatchDeclaration {
                        used: cache.clone(),
                        decl: type_vars.get(*idx).cloned(),
                    })
                }
            }
        }
    }

    pub(super) fn substitute(&self, args: &[TypeArg]) -> Self {
        match self {
            TypeArg::Type(t) => TypeArg::Type(t.substitute(args)),
            TypeArg::BoundedNat(_) => self.clone(), // We do not allow variables as bounds on BoundedNat's
            TypeArg::Opaque(CustomTypeArg { typ, value }) => {
                // Allow substitution (e.g. empty list)...note TODO above:
                //       is there demand for this, is it actually correct?
                TypeArg::Opaque(CustomTypeArg {
                    typ: typ.substitute(args),
                    value: value.clone(),
                })
            }
            TypeArg::Sequence(elems) => {
                TypeArg::Sequence(elems.iter().map(|ta| ta.substitute(args)).collect())
            }
            TypeArg::Extensions(_) => self.clone(), // TODO extension variables
            // Caller should already have checked arg against bound (cached here):
            TypeArg::Variable(idx, _) => args.get(*idx).unwrap().clone(),
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
        (TypeArg::Variable(_, bound), _) if param.contains(bound) => Ok(()),
        (TypeArg::Type(t), TypeParam::Type(bound)) if bound.contains(t.least_upper_bound()) => {
            Ok(())
        }
        (TypeArg::Sequence(items), TypeParam::List(param)) => {
            items.iter().try_for_each(|arg| check_type_arg(arg, param))
        }
        (TypeArg::Sequence(items), TypeParam::Tuple(types)) => {
            if items.len() != types.len() {
                Err(TypeArgError::WrongNumberTuple(items.len(), types.len()))
            } else {
                items
                    .iter()
                    .zip(types.iter())
                    .try_for_each(|(arg, param)| check_type_arg(arg, param))
            }
        }
        (TypeArg::BoundedNat(val), TypeParam::BoundedNat(bound)) if bound.valid_value(*val) => {
            Ok(())
        }

        (TypeArg::Opaque(arg), TypeParam::Opaque(param))
            if param.bound() == TypeBound::Eq && &arg.typ == param =>
        {
            Ok(())
        }
        (TypeArg::Extensions(_), TypeParam::Extensions) => Ok(()),
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
