//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::extension::TypeDef

use itertools::Itertools;
use ordered_float::OrderedFloat;
#[cfg(test)]
use proptest_derive::Arbitrary;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;
use std::sync::Arc;
use thiserror::Error;

use super::row_var::MaybeRV;
use super::{
    NoRV, RowVariable, Substitution, Transformable, Type, TypeBase, TypeBound, TypeTransformer,
    check_typevar_decl,
};
use crate::extension::SignatureError;

/// The upper non-inclusive bound of a [`TypeParam::BoundedNat`]
// A None inner value implies the maximum bound: u64::MAX + 1 (all u64 values valid)
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, derive_more::Display, serde::Deserialize, serde::Serialize,
)]
#[display("{}", _0.map(|i|i.to_string()).unwrap_or("-".to_string()))]
#[cfg_attr(test, derive(Arbitrary))]
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
    #[must_use]
    pub fn value(&self) -> &Option<NonZeroU64> {
        &self.0
    }
}

pub type TypeArg = Term;
pub type TypeParam = Term;

/// A *kind* of [`TypeArg`]. Thus, a parameter declared by a [`PolyFuncType`] or [`PolyFuncTypeRV`],
/// specifying a value that must be provided statically in order to instantiate it.
///
/// [`PolyFuncType`]: super::PolyFuncType
/// [`PolyFuncTypeRV`]: super::PolyFuncTypeRV
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, derive_more::Display, serde::Deserialize, serde::Serialize,
)]
#[non_exhaustive]
#[serde(tag = "t")]
pub enum Term {
    /// Argument is a [`TypeArg::Type`].
    #[display("Type{}", match b {
        TypeBound::Any => String::new(),
        _ => format!("[{b}]")
    })]
    RuntimeType {
        /// Bound for the type parameter.
        b: TypeBound,
    },
    /// Argument is a [`TypeArg::BoundedNat`] that is less than the upper bound.
    #[display("{}", match bound.value() {
        Some(v) => format!("BoundedNat[{v}]"),
        None => "Nat".to_string()
    })]
    BoundedNatType {
        /// Upper bound for the Nat parameter.
        bound: UpperBound,
    },
    /// Argument is a [`TypeArg::String`].
    StringType,
    /// Argument is a [`TypeArg::Bytes`].
    BytesType,
    /// Argument is a [`TypeArg::Float`].
    FloatType,
    /// Argument is a [`TypeArg::List`]. A list of indeterminate size containing
    /// parameters all of the (same) specified element type.
    #[display("ListType[{param}]")]
    ListType {
        /// The [`TypeParam`] describing each element of the list.
        param: Box<TypeParam>,
    },
    /// Argument is a [`TypeArg::Tuple`]. A tuple of parameters.
    #[display("TupleType[{}]", params.iter().map(std::string::ToString::to_string).join(", "))]
    TupleType {
        /// The [`TypeParam`]s contained in the tuple.
        params: Vec<TypeParam>,
    },
    /// Where the (Type/Op)Def declares that an argument is a [`TypeParam::Type`]
    #[display("{ty}")]
    Type {
        /// The concrete type for the parameter.
        ty: Type,
    },
    /// Instance of [`TypeParam::BoundedNat`]. 64-bit unsigned integer.
    #[display("{n}")]
    BoundedNat {
        /// The integer value for the parameter.
        n: u64,
    },
    ///Instance of [`TypeParam::String`]. UTF-8 encoded string argument.
    #[display("\"{arg}\"")]
    String {
        /// The string value for the parameter.
        arg: String,
    },
    /// Instance of [`TypeParam::Bytes`]. Byte string.
    #[display("bytes")]
    Bytes {
        /// The value of the bytes parameter.
        #[serde(with = "base64")]
        value: Arc<[u8]>,
    },
    /// Instance of [`TypeParam::Float`]. 64-bit floating point number.
    #[display("{}", value.into_inner())]
    Float {
        /// The value of the float parameter.
        value: OrderedFloat<f64>,
    },
    /// Instance of [`TypeParam::List`] defined by a sequence of elements of the same type.
    #[display("[{}]", {
        use itertools::Itertools as _;
        elems.iter().map(|t|t.to_string()).join(",")
    })]
    List {
        /// List of elements
        elems: Vec<TypeArg>,
    },
    /// Instance of [`TypeParam::Tuple`] defined by a sequence of elements of varying type.
    #[display("({})", {
        use itertools::Itertools as _;
        elems.iter().map(std::string::ToString::to_string).join(",")
    })]
    Tuple {
        /// List of elements
        elems: Vec<TypeArg>,
    },
    /// Variable (used in type schemes or inside polymorphic functions),
    /// but not a [`TypeArg::Type`] (not even a row variable i.e. [`TypeParam::List`] of type)
    /// - see [`TypeArg::new_var_use`]
    #[display("{v}")]
    Variable {
        #[allow(missing_docs)]
        #[serde(flatten)]
        v: TypeArgVariable,
    },
}

impl TypeParam {
    /// [`TypeParam::BoundedNatType`] with the maximum bound (`u64::MAX` + 1)
    #[must_use]
    pub const fn max_nat_type() -> Self {
        Self::BoundedNatType {
            bound: UpperBound(None),
        }
    }

    /// [`TypeParam::BoundedNatType`] with the stated upper bound (non-exclusive)
    #[must_use]
    pub const fn bounded_nat_type(upper_bound: NonZeroU64) -> Self {
        Self::BoundedNatType {
            bound: UpperBound(Some(upper_bound)),
        }
    }

    /// Make a new [`TypeParam::ListType`] (an arbitrary-length homogeneous list)
    pub fn new_list_type(elem: impl Into<TypeParam>) -> Self {
        Self::ListType {
            param: Box::new(elem.into()),
        }
    }

    fn contains(&self, other: &TypeParam) -> bool {
        match (self, other) {
            (TypeParam::RuntimeType { b: b1 }, TypeParam::RuntimeType { b: b2 }) => {
                b1.contains(*b2)
            }
            (TypeParam::BoundedNatType { bound: b1 }, TypeParam::BoundedNatType { bound: b2 }) => {
                b1.contains(b2)
            }
            (TypeParam::StringType, TypeParam::StringType) => true,
            (TypeParam::ListType { param: e1 }, TypeParam::ListType { param: e2 }) => {
                e1.contains(e2)
            }
            (TypeParam::TupleType { params: es1 }, TypeParam::TupleType { params: es2 }) => {
                es1.len() == es2.len() && es1.iter().zip(es2).all(|(e1, e2)| e1.contains(e2))
            }
            _ => false,
        }
    }
}

impl From<TypeBound> for TypeParam {
    fn from(bound: TypeBound) -> Self {
        Self::RuntimeType { b: bound }
    }
}

impl From<UpperBound> for TypeParam {
    fn from(bound: UpperBound) -> Self {
        Self::BoundedNatType { bound }
    }
}

impl<RV: MaybeRV> From<TypeBase<RV>> for TypeArg {
    fn from(value: TypeBase<RV>) -> Self {
        match value.try_into_type() {
            Ok(ty) => TypeArg::Type { ty },
            Err(RowVariable(idx, bound)) => {
                TypeArg::new_var_use(idx, TypeParam::new_list_type(bound))
            }
        }
    }
}

impl From<u64> for TypeArg {
    fn from(n: u64) -> Self {
        Self::BoundedNat { n }
    }
}

impl From<String> for TypeArg {
    fn from(arg: String) -> Self {
        TypeArg::String { arg }
    }
}

impl From<&str> for TypeArg {
    fn from(arg: &str) -> Self {
        TypeArg::String {
            arg: arg.to_string(),
        }
    }
}

impl From<Vec<TypeArg>> for TypeArg {
    fn from(elems: Vec<TypeArg>) -> Self {
        Self::List { elems }
    }
}

/// Variable in a `TypeArg`, that is not a single [`TypeArg::Type`] (i.e. not a [`Type::new_var_use`]
/// - it might be a [`Type::new_row_var_use`]).
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize, derive_more::Display,
)]
#[display("#{idx}")]
pub struct TypeArgVariable {
    idx: usize,
    cached_decl: Box<TypeParam>,
}

impl TypeArg {
    /// [`Type::UNIT`] as a [`TypeArg::Type`]
    pub const UNIT: Self = Self::Type { ty: Type::UNIT };

    /// Makes a `TypeArg` representing a use (occurrence) of the type variable
    /// with the specified index.
    /// `decl` must be exactly that with which the variable was declared.
    #[must_use]
    pub fn new_var_use(idx: usize, decl: TypeParam) -> Self {
        match decl {
            // Note a TypeParam::List of TypeParam::Type *cannot* be represented
            // as a TypeArg::Type because the latter stores a Type<false> i.e. only a single type,
            // not a RowVariable.
            TypeParam::RuntimeType { b } => Type::new_var_use(idx, b).into(),
            _ => TypeArg::Variable {
                v: TypeArgVariable {
                    idx,
                    cached_decl: Box::new(decl),
                },
            },
        }
    }

    /// Returns an integer if the `TypeArg` is an instance of `BoundedNat`.
    #[must_use]
    pub fn as_nat(&self) -> Option<u64> {
        match self {
            TypeArg::BoundedNat { n } => Some(*n),
            _ => None,
        }
    }

    /// Returns a type if the `TypeArg` is an instance of Type.
    #[must_use]
    pub fn as_type(&self) -> Option<TypeBase<NoRV>> {
        match self {
            TypeArg::Type { ty } => Some(ty.clone()),
            _ => None,
        }
    }

    /// Returns a string if the `TypeArg` is an instance of String.
    #[must_use]
    pub fn as_string(&self) -> Option<String> {
        match self {
            TypeArg::String { arg } => Some(arg.clone()),
            _ => None,
        }
    }

    /// Much as [`Type::validate`], also checks that the type of any [`TypeArg::Opaque`]
    /// is valid and closed.
    pub(crate) fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        match self {
            Term::Type { ty } => ty.validate(var_decls),
            Term::List { elems } => {
                // TODO: Full validation would check that the type of the elements agrees
                elems.iter().try_for_each(|a| a.validate(var_decls))
            }
            Term::Tuple { elems } => elems.iter().try_for_each(|a| a.validate(var_decls)),
            Term::BoundedNat { .. }
            | Term::String { .. }
            | Term::Float { .. }
            | Term::Bytes { .. } => Ok(()),
            Term::Variable {
                v: TypeArgVariable { idx, cached_decl },
            } => {
                assert!(
                    !matches!(&**cached_decl, TypeParam::RuntimeType { .. }),
                    "Malformed TypeArg::Variable {cached_decl} - should be inconstructible"
                );

                check_typevar_decl(var_decls, *idx, cached_decl)
            }
            Term::RuntimeType { b } => todo!(),
            Term::BoundedNatType { bound } => todo!(),
            Term::StringType => todo!(),
            Term::BytesType => todo!(),
            Term::FloatType => todo!(),
            Term::ListType { param } => todo!(),
            Term::TupleType { params } => todo!(),
        }
    }

    pub(crate) fn substitute(&self, t: &Substitution) -> Self {
        match self {
            Term::Type { ty } => {
                // RowVariables are represented as Term::Variable
                ty.substitute1(t).into()
            }
            Term::BoundedNat { .. }
            | Term::String { .. }
            | Term::Bytes { .. }
            | Term::Float { .. } => self.clone(),
            Term::List { elems } => {
                let mut are_types = elems.iter().map(|ta| match ta {
                    Term::Type { .. } => true,
                    Term::Variable { v } => v.bound_if_row_var().is_some(),
                    _ => false,
                });
                let elems = match are_types.next() {
                    Some(true) => {
                        assert!(are_types.all(|b| b)); // If one is a Type, so must the rest be
                        // So, anything that doesn't produce a Type, was a row variable => multiple Types
                        elems
                            .iter()
                            .flat_map(|ta| match ta.substitute(t) {
                                ty @ Term::Type { .. } => vec![ty],
                                Term::List { elems } => elems,
                                _ => panic!("Expected Type or row of Types"),
                            })
                            .collect()
                    }
                    _ => {
                        // not types, no need to flatten (and mustn't, in case of nested Sequences)
                        elems.iter().map(|ta| ta.substitute(t)).collect()
                    }
                };
                Term::List { elems }
            }
            Term::Tuple { elems } => Term::Tuple {
                elems: elems.iter().map(|elem| elem.substitute(t)).collect(),
            },
            Term::Variable {
                v: TypeArgVariable { idx, cached_decl },
            } => t.apply_var(*idx, cached_decl),
            Term::RuntimeType { b } => todo!(),
            Term::BoundedNatType { bound } => todo!(),
            Term::StringType => todo!(),
            Term::BytesType => todo!(),
            Term::FloatType => todo!(),
            Term::ListType { param } => todo!(),
            Term::TupleType { params } => todo!(),
        }
    }
}

impl Transformable for Term {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        match self {
            Term::Type { ty } => ty.transform(tr),
            Term::List { elems } => elems.transform(tr),
            Term::Tuple { elems } => elems.transform(tr),
            Term::BoundedNat { .. }
            | Term::String { .. }
            | Term::Variable { .. }
            | Term::Float { .. }
            | Term::Bytes { .. } => Ok(false),
            Term::RuntimeType { b } => todo!(),
            Term::BoundedNatType { bound } => todo!(),
            Term::StringType => todo!(),
            Term::BytesType => todo!(),
            Term::FloatType => todo!(),
            Term::ListType { param } => todo!(),
            Term::TupleType { params } => todo!(),
        }
    }
}

impl TypeArgVariable {
    /// Return the index.
    #[must_use]
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Determines whether this represents a row variable; if so, returns
    /// the [`TypeBound`] of the individual types it might stand for.
    #[must_use]
    pub fn bound_if_row_var(&self) -> Option<TypeBound> {
        if let TypeParam::ListType { param } = &*self.cached_decl {
            if let TypeParam::RuntimeType { b } = **param {
                return Some(b);
            }
        }
        None
    }
}

/// Checks a [`TypeArg`] is as expected for a [`TypeParam`]
pub fn check_type_arg(arg: &TypeArg, param: &TypeParam) -> Result<(), TypeArgError> {
    match (arg, param) {
        (
            TypeArg::Variable {
                v: TypeArgVariable { cached_decl, .. },
            },
            _,
        ) if param.contains(cached_decl) => Ok(()),
        (TypeArg::Type { ty }, TypeParam::RuntimeType { b: bound })
            if bound.contains(ty.least_upper_bound()) =>
        {
            Ok(())
        }
        (TypeArg::List { elems }, TypeParam::ListType { param }) => {
            elems.iter().try_for_each(|arg| {
                // Also allow elements that are RowVars if fitting into a List of Types
                if let (TypeArg::Variable { v }, TypeParam::RuntimeType { b: param_bound }) =
                    (arg, &**param)
                {
                    if v.bound_if_row_var()
                        .is_some_and(|arg_bound| param_bound.contains(arg_bound))
                    {
                        return Ok(());
                    }
                }
                check_type_arg(arg, param)
            })
        }
        (TypeArg::Tuple { elems: items }, TypeParam::TupleType { params: types }) => {
            if items.len() != types.len() {
                return Err(TypeArgError::WrongNumberTuple(items.len(), types.len()));
            }

            items
                .iter()
                .zip(types.iter())
                .try_for_each(|(arg, param)| check_type_arg(arg, param))
        }
        (TypeArg::BoundedNat { n: val }, TypeParam::BoundedNatType { bound })
            if bound.valid_value(*val) =>
        {
            Ok(())
        }

        (TypeArg::String { .. }, TypeParam::StringType) => Ok(()),
        (TypeArg::Bytes { .. }, TypeParam::BytesType) => Ok(()),
        (TypeArg::Float { .. }, TypeParam::FloatType) => Ok(()),
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

/// Errors that can occur fitting a [`TypeArg`] into a [`TypeParam`]
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum TypeArgError {
    #[allow(missing_docs)]
    /// For now, general case of a type arg not fitting a param.
    /// We'll have more cases when we allow general Containers.
    // TODO It may become possible to combine this with ConstTypeError.
    #[error("Type argument {arg} does not fit declared parameter {param}")]
    TypeMismatch { param: TypeParam, arg: TypeArg },
    /// Wrong number of type arguments (actual vs expected).
    // For now this only happens at the top level (TypeArgs of op/type vs TypeParams of Op/TypeDef).
    // However in the future it may be applicable to e.g. contents of Tuples too.
    #[error("Wrong number of type arguments: {0} vs expected {1} declared type parameters")]
    WrongNumberArgs(usize, usize),

    /// Wrong number of type arguments in tuple (actual vs expected).
    #[error(
        "Wrong number of type arguments to tuple parameter: {0} vs expected {1} declared type parameters"
    )]
    WrongNumberTuple(usize, usize),
    /// Opaque value type check error.
    #[error("Opaque type argument does not fit declared parameter type: {0}")]
    OpaqueTypeMismatch(#[from] crate::types::CustomCheckFailure),
    /// Invalid value
    #[error("Invalid value of type argument")]
    InvalidValue(TypeArg),
}

/// Helper for to serialize and deserialize the byte string in `TypeArg::Bytes` via base64.
mod base64 {
    use std::sync::Arc;

    use base64::Engine as _;
    use base64::prelude::BASE64_STANDARD;
    use serde::{Deserialize, Serialize};
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Arc<[u8]>, s: S) -> Result<S::Ok, S::Error> {
        let base64 = BASE64_STANDARD.encode(v);
        base64.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Arc<[u8]>, D::Error> {
        let base64 = String::deserialize(d)?;
        BASE64_STANDARD
            .decode(base64.as_bytes())
            .map(|v| v.into())
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::{Substitution, TypeArg, TypeParam, check_type_arg};
    use crate::extension::prelude::{bool_t, usize_t};
    use crate::types::{TypeBound, TypeRV, type_param::TypeArgError};

    #[test]
    fn type_arg_fits_param() {
        let rowvar = TypeRV::new_row_var_use;
        fn check(arg: impl Into<TypeArg>, param: &TypeParam) -> Result<(), TypeArgError> {
            check_type_arg(&arg.into(), param)
        }
        fn check_seq<T: Clone + Into<TypeArg>>(
            args: &[T],
            param: &TypeParam,
        ) -> Result<(), TypeArgError> {
            let arg = args.iter().cloned().map_into().collect_vec().into();
            check_type_arg(&arg, param)
        }
        // Simple cases: a TypeArg::Type is a TypeParam::Type but singleton sequences are lists
        check(usize_t(), &TypeBound::Copyable.into()).unwrap();
        let seq_param = TypeParam::new_list_type(TypeBound::Copyable);
        check(usize_t(), &seq_param).unwrap_err();
        check_seq(&[usize_t()], &TypeBound::Any.into()).unwrap_err();

        // Into a list of type, we can fit a single row var
        check(rowvar(0, TypeBound::Copyable), &seq_param).unwrap();
        // or a list of (types or row vars)
        check(vec![], &seq_param).unwrap();
        check_seq(&[rowvar(0, TypeBound::Copyable)], &seq_param).unwrap();
        check_seq(
            &[
                rowvar(1, TypeBound::Any),
                usize_t().into(),
                rowvar(0, TypeBound::Copyable),
            ],
            &TypeParam::new_list_type(TypeBound::Any),
        )
        .unwrap();
        // Next one fails because a list of Eq is required
        check_seq(
            &[
                rowvar(1, TypeBound::Any),
                usize_t().into(),
                rowvar(0, TypeBound::Copyable),
            ],
            &seq_param,
        )
        .unwrap_err();
        // seq of seq of types is not allowed
        check(
            vec![usize_t().into(), vec![usize_t().into()].into()],
            &seq_param,
        )
        .unwrap_err();

        // Similar for nats (but no equivalent of fancy row vars)
        check(5, &TypeParam::max_nat_type()).unwrap();
        check_seq(&[5], &TypeParam::max_nat_type()).unwrap_err();
        let list_of_nat = TypeParam::new_list_type(TypeParam::max_nat_type());
        check(5, &list_of_nat).unwrap_err();
        check_seq(&[5], &list_of_nat).unwrap();
        check(TypeArg::new_var_use(0, list_of_nat.clone()), &list_of_nat).unwrap();
        // But no equivalent of row vars - can't append a nat onto a list-in-a-var:
        check(
            vec![5.into(), TypeArg::new_var_use(0, list_of_nat.clone())],
            &list_of_nat,
        )
        .unwrap_err();

        // TypeParam::Tuples require a TypeArg::Tuple of the same number of elems
        let usize_and_ty = TypeParam::TupleType {
            params: vec![TypeParam::max_nat_type(), TypeBound::Copyable.into()],
        };
        check(
            TypeArg::Tuple {
                elems: vec![5.into(), usize_t().into()],
            },
            &usize_and_ty,
        )
        .unwrap();
        check(
            TypeArg::Tuple {
                elems: vec![usize_t().into(), 5.into()],
            },
            &usize_and_ty,
        )
        .unwrap_err(); // Wrong way around
        let two_types = TypeParam::TupleType {
            params: vec![TypeBound::Any.into(), TypeBound::Any.into()],
        };
        check(TypeArg::new_var_use(0, two_types.clone()), &two_types).unwrap();
        // not a Row Var which could have any number of elems
        check(TypeArg::new_var_use(0, seq_param), &two_types).unwrap_err();
    }

    #[test]
    fn type_arg_subst_row() {
        let row_param = TypeParam::new_list_type(TypeBound::Copyable);
        let row_arg: TypeArg = vec![bool_t().into(), TypeArg::UNIT].into();
        check_type_arg(&row_arg, &row_param).unwrap();

        // Now say a row variable referring to *that* row was used
        // to instantiate an outer "row parameter" (list of type).
        let outer_param = TypeParam::new_list_type(TypeBound::Any);
        let outer_arg = TypeArg::List {
            elems: vec![
                TypeRV::new_row_var_use(0, TypeBound::Copyable).into(),
                usize_t().into(),
            ],
        };
        check_type_arg(&outer_arg, &outer_param).unwrap();

        let outer_arg2 = outer_arg.substitute(&Substitution(&[row_arg]));
        assert_eq!(
            outer_arg2,
            vec![bool_t().into(), TypeArg::UNIT, usize_t().into()].into()
        );

        // Of course this is still valid (as substitution is guaranteed to preserve validity)
        check_type_arg(&outer_arg2, &outer_param).unwrap();
    }

    #[test]
    fn subst_list_list() {
        let outer_param = TypeParam::new_list_type(TypeParam::new_list_type(TypeBound::Any));
        let row_var_decl = TypeParam::new_list_type(TypeBound::Copyable);
        let row_var_use = TypeArg::new_var_use(0, row_var_decl.clone());
        let good_arg = TypeArg::List {
            elems: vec![
                // The row variables here refer to `row_var_decl` above
                vec![usize_t().into()].into(),
                row_var_use.clone(),
                vec![row_var_use, usize_t().into()].into(),
            ],
        };
        check_type_arg(&good_arg, &outer_param).unwrap();

        // Outer list cannot include single types:
        let TypeArg::List { mut elems } = good_arg.clone() else {
            panic!()
        };
        elems.push(usize_t().into());
        assert_eq!(
            check_type_arg(&TypeArg::List { elems }, &outer_param),
            Err(TypeArgError::TypeMismatch {
                arg: usize_t().into(),
                // The error reports the type expected for each element of the list:
                param: TypeParam::new_list_type(TypeBound::Any)
            })
        );

        // Now substitute a list of two types for that row-variable
        let row_var_arg = vec![usize_t().into(), bool_t().into()].into();
        check_type_arg(&row_var_arg, &row_var_decl).unwrap();
        let subst_arg = good_arg.substitute(&Substitution(&[row_var_arg.clone()]));
        check_type_arg(&subst_arg, &outer_param).unwrap(); // invariance of substitution
        assert_eq!(
            subst_arg,
            TypeArg::List {
                elems: vec![
                    vec![usize_t().into()].into(),
                    row_var_arg,
                    vec![usize_t().into(), bool_t().into(), usize_t().into()].into()
                ]
            }
        );
    }

    #[test]
    fn bytes_json_roundtrip() {
        let bytes_arg = TypeArg::Bytes {
            value: vec![0, 1, 2, 3, 255, 254, 253, 252].into(),
        };
        let serialized = serde_json::to_string(&bytes_arg).unwrap();
        let deserialized: TypeArg = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, bytes_arg);
    }

    mod proptest {

        use proptest::prelude::*;

        use super::super::{TypeArg, TypeArgVariable, TypeParam, UpperBound};
        use crate::proptest::RecursionDepth;
        use crate::types::{Type, TypeBound};

        impl Arbitrary for TypeArgVariable {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                (any::<usize>(), any_with::<TypeParam>(depth))
                    .prop_map(|(idx, cached_decl)| Self {
                        idx,
                        cached_decl: Box::new(cached_decl),
                    })
                    .boxed()
            }
        }

        impl Arbitrary for TypeParam {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                use prop::collection::vec;
                use prop::strategy::Union;
                let mut strat = Union::new([
                    Just(Self::StringType).boxed(),
                    Just(Self::BytesType).boxed(),
                    Just(Self::FloatType).boxed(),
                    Just(Self::StringType).boxed(),
                    any::<TypeBound>()
                        .prop_map(|b| Self::RuntimeType { b })
                        .boxed(),
                    any::<UpperBound>()
                        .prop_map(|bound| Self::BoundedNatType { bound })
                        .boxed(),
                    any::<u64>().prop_map(|n| Self::BoundedNat { n }).boxed(),
                    any::<String>().prop_map(|arg| Self::String { arg }).boxed(),
                    any::<Vec<u8>>()
                        .prop_map(|bytes| Self::Bytes {
                            value: bytes.into(),
                        })
                        .boxed(),
                    any::<f64>()
                        .prop_map(|value| Self::Float {
                            value: value.into(),
                        })
                        .boxed(),
                    any_with::<Type>(depth)
                        .prop_map(|ty| Self::Type { ty })
                        .boxed(),
                    // TODO this is a bit dodgy, TypeArgVariables are supposed
                    // to be constructed from TypeArg::new_var_use. We are only
                    // using this instance for serialization now, but if we want
                    // to generate valid TypeArgs this will need to change.
                    any_with::<TypeArgVariable>(depth)
                        .prop_map(|v| Self::Variable { v })
                        .boxed(),
                ]);
                if !depth.leaf() {
                    // we descend here because we these constructors contain Terms
                    strat = strat
                        .or(any_with::<Self>(depth.descend())
                            .prop_map(|x| Self::ListType { param: Box::new(x) })
                            .boxed())
                        .or(vec(any_with::<Self>(depth.descend()), 0..3)
                            .prop_map(|params| Self::TupleType { params })
                            .boxed())
                        .or(vec(any_with::<Self>(depth.descend()), 0..3)
                            .prop_map(|elems| Self::List { elems })
                            .boxed());
                }

                strat.boxed()
            }
        }
    }
}
