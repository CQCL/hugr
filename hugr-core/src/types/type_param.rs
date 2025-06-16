//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::extension::TypeDef

use itertools::Itertools;
use ordered_float::OrderedFloat;
#[cfg(test)]
use proptest_derive::Arbitrary;
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

/// A [`Term`] that is a static argument to an operation or constructor.
pub type TypeArg = Term;

/// A [`Term`] that is the static type of an operation or constructor parameter.
pub type TypeParam = Term;

/// A term in the language of static parameters in HUGR.
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, derive_more::Display, serde::Deserialize, serde::Serialize,
)]
#[non_exhaustive]
#[serde(
    from = "crate::types::serialize::TermSer",
    into = "crate::types::serialize::TermSer"
)]
pub enum Term {
    /// The type of runtime types.
    #[display("Type{}", match _0 {
        TypeBound::Any => String::new(),
        _ => format!("[{_0}]")
    })]
    RuntimeType(TypeBound),
    /// The type of static data.
    StaticType,
    /// The type of static natural numbers up to a given bound.
    #[display("{}", match _0.value() {
        Some(v) => format!("BoundedNat[{v}]"),
        None => "Nat".to_string()
    })]
    BoundedNatType(UpperBound),
    /// The type of static strings. See [`Term::String`].
    StringType,
    /// The type of static byte strings. See [`Term::Bytes`].
    BytesType,
    /// The type of static floating point numbers. See [`Term::Float`].
    FloatType,
    /// The type of static lists of indeterminate size containing terms of the
    /// specified static type.
    #[display("ListType[{_0}]")]
    ListType(Box<Term>),
    /// The type of static tuples.
    #[display("TupleType[{}]", _0.iter().map(std::string::ToString::to_string).join(", "))]
    TupleType(Vec<Term>),
    /// A runtime type as a term. Instance of [`Term::RuntimeType`].
    #[display("{_0}")]
    Runtime(Type),
    /// A 64bit unsigned integer literal. Instance of [`Term::BoundedNatType`].
    #[display("{_0}")]
    BoundedNat(u64),
    /// UTF-8 encoded string literal. Instance of [`Term::StringType`].
    #[display("\"{_0}\"")]
    String(String),
    /// Byte string literal. Instance of [`Term::BytesType`].
    #[display("bytes")]
    Bytes(Arc<[u8]>),
    /// A 64-bit floating point number. Instance of [`Term::FloatType`].
    #[display("{}", _0.into_inner())]
    Float(OrderedFloat<f64>),
    /// A list of static terms. Instance of [`Term::ListType`].
    #[display("[{}]", {
        use itertools::Itertools as _;
        _0.iter().map(|t|t.to_string()).join(",")
    })]
    List(Vec<Term>),
    /// A tuple of static terms. Instance of [`Term::TupleType`].
    #[display("({})", {
        use itertools::Itertools as _;
        _0.iter().map(std::string::ToString::to_string).join(",")
    })]
    Tuple(Vec<Term>),
    /// Variable (used in type schemes or inside polymorphic functions),
    /// but not a runtime type (not even a row variable i.e. list of runtime types)
    /// - see [`Term::new_var_use`]
    #[display("{_0}")]
    Variable(TermVar),
}

impl Term {
    /// Creates a [`Term::BoundedNatType`] with the maximum bound (`u64::MAX` + 1).
    #[must_use]
    pub const fn max_nat_type() -> Self {
        Self::BoundedNatType(UpperBound(None))
    }

    /// Creates a [`Term::BoundedNatType`] with the stated upper bound (non-exclusive).
    #[must_use]
    pub const fn bounded_nat_type(upper_bound: NonZeroU64) -> Self {
        Self::BoundedNatType(UpperBound(Some(upper_bound)))
    }

    /// Creates a new [`Term::List`] given a sequence of its items.
    pub fn new_list(items: impl IntoIterator<Item = Term>) -> Self {
        Self::List(items.into_iter().collect())
    }

    /// Creates a new [`Term::ListType`] given the type of its elements.
    pub fn new_list_type(elem: impl Into<Term>) -> Self {
        Self::ListType(Box::new(elem.into()))
    }

    /// Creates a new [`Term::TupleType`] given the types of its elements.
    pub fn new_tuple_type(item_types: impl IntoIterator<Item = Term>) -> Self {
        Self::TupleType(item_types.into_iter().collect())
    }

    /// Checks if this term is a supertype of another.
    fn is_supertype(&self, other: &Term) -> bool {
        match (self, other) {
            (Term::RuntimeType(b1), Term::RuntimeType(b2)) => b1.contains(*b2),
            (Term::BoundedNatType(b1), Term::BoundedNatType(b2)) => b1.contains(b2),
            (Term::StringType, Term::StringType) => true,
            (Term::StaticType, Term::StaticType) => true,
            (Term::ListType(e1), Term::ListType(e2)) => e1.is_supertype(e2),
            (Term::TupleType(es1), Term::TupleType(es2)) => {
                es1.len() == es2.len() && es1.iter().zip(es2).all(|(e1, e2)| e1.is_supertype(e2))
            }
            (Term::BytesType, Term::BytesType) => true,
            (Term::FloatType, Term::FloatType) => true,
            (
                Term::Runtime(_)
                | Term::BoundedNat(_)
                | Term::String(_)
                | Term::Bytes(_)
                | Term::Float(_)
                | Term::List(_)
                | Term::Tuple(_),
                _,
            ) => {
                // This is not a type at all, so it's not a supertype of anything.
                false
            }
            // ALAN not clear if we need this one? Tests would need complexifying
            // (Term::Variable(v1), Term::Variable(v2)) => v1 == v2,
            _ => false,
        }
    }
}

impl From<TypeBound> for Term {
    fn from(bound: TypeBound) -> Self {
        Self::RuntimeType(bound)
    }
}

impl From<UpperBound> for Term {
    fn from(bound: UpperBound) -> Self {
        Self::BoundedNatType(bound)
    }
}

impl<RV: MaybeRV> From<TypeBase<RV>> for Term {
    fn from(value: TypeBase<RV>) -> Self {
        match value.try_into_type() {
            Ok(ty) => Term::Runtime(ty),
            Err(RowVariable(idx, bound)) => Term::new_var_use(idx, TypeParam::new_list_type(bound)),
        }
    }
}

impl From<u64> for Term {
    fn from(n: u64) -> Self {
        Self::BoundedNat(n)
    }
}

impl From<String> for Term {
    fn from(arg: String) -> Self {
        Term::String(arg)
    }
}

impl From<&str> for Term {
    fn from(arg: &str) -> Self {
        Term::String(arg.to_string())
    }
}

impl From<Vec<Term>> for Term {
    fn from(elems: Vec<Term>) -> Self {
        Self::new_list(elems)
    }
}

/// Variable in a [`Term`], that is not a single runtime type (i.e. not a [`Type::new_var_use`]
/// - it might be a [`Type::new_row_var_use`]).
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize, derive_more::Display,
)]
#[display("#{idx}")]
pub struct TermVar {
    idx: usize,
    cached_decl: Box<Term>,
}

impl Term {
    /// [`Type::UNIT`] as a [`Term::Runtime`]
    pub const UNIT: Self = Self::Runtime(Type::UNIT);

    /// Makes a `TypeArg` representing a use (occurrence) of the type variable
    /// with the specified index.
    /// `decl` must be exactly that with which the variable was declared.
    #[must_use]
    pub fn new_var_use(idx: usize, decl: Term) -> Self {
        match decl {
            // Note a TypeParam::List of TypeParam::Type *cannot* be represented
            // as a TypeArg::Type because the latter stores a Type<false> i.e. only a single type,
            // not a RowVariable.
            Term::RuntimeType(b) => Type::new_var_use(idx, b).into(),
            _ => Term::Variable(TermVar {
                idx,
                cached_decl: Box::new(decl),
            }),
        }
    }

    /// Returns an integer if the [`Term`] is a natural number literal.
    #[must_use]
    pub fn as_nat(&self) -> Option<u64> {
        match self {
            TypeArg::BoundedNat(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns a [`Type`] if the [`Term`] is a runtime type.
    #[must_use]
    pub fn as_runtime(&self) -> Option<TypeBase<NoRV>> {
        match self {
            TypeArg::Runtime(ty) => Some(ty.clone()),
            _ => None,
        }
    }

    /// Returns a string if the [`Term`] is a string literal.
    #[must_use]
    pub fn as_string(&self) -> Option<String> {
        match self {
            TypeArg::String(arg) => Some(arg.clone()),
            _ => None,
        }
    }

    /// Check that this is a valid bound on (type for) a parameter
    pub(crate) fn validate_param(&self) -> Result<(), SignatureError> {
        match self {
            Term::RuntimeType(_)
            | Term::StaticType
            | Term::BoundedNatType(_)
            | Term::StringType
            | Term::BytesType
            | Term::FloatType => Ok(()),
            Term::ListType(term) => term.validate_param(),
            Term::TupleType(terms) => terms.iter().try_for_each(Term::validate_param),
            Term::Runtime(_)
            | Term::BoundedNat(_)
            | Term::String(_)
            | Term::Bytes(_)
            | Term::Float(_)
            | Term::List(_)
            | Term::Tuple(_)
            | Term::Variable(_) => Err(SignatureError::InvalidTypeParam(self.clone())),
        }
    }

    /// Much as [`Type::validate`], also checks that the type of any [`TypeArg::Opaque`]
    /// is valid and closed.
    pub(crate) fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        match self {
            Term::Runtime(ty) => ty.validate(var_decls),
            Term::List(elems) => {
                // TODO: Full validation would check that the type of the elements agrees
                elems.iter().try_for_each(|a| a.validate(var_decls))
            }
            Term::Tuple(elems) => elems.iter().try_for_each(|a| a.validate(var_decls)),
            Term::BoundedNat(_) | Term::String { .. } | Term::Float(_) | Term::Bytes(_) => Ok(()),
            Term::Variable(TermVar { idx, cached_decl }) => {
                assert!(
                    !matches!(&**cached_decl, TypeParam::RuntimeType { .. }),
                    "Malformed TypeArg::Variable {cached_decl} - should be inconstructible"
                );

                check_typevar_decl(var_decls, *idx, cached_decl)
            }
            Term::RuntimeType(_) => Ok(()),
            Term::BoundedNatType { .. } => Ok(()),
            Term::StringType => Ok(()),
            Term::BytesType => Ok(()),
            Term::FloatType => Ok(()),
            Term::ListType(item_type) => item_type.validate(var_decls),
            Term::TupleType(params) => params.iter().try_for_each(|p| p.validate(var_decls)),
            Term::StaticType => Ok(()),
        }
    }

    pub(crate) fn substitute(&self, t: &Substitution) -> Self {
        match self {
            Term::Runtime(ty) => {
                // RowVariables are represented as Term::Variable
                ty.substitute1(t).into()
            }
            Term::BoundedNat(_) | Term::String { .. } | Term::Bytes(_) | Term::Float(_) => {
                self.clone()
            }
            Term::List(elems) => {
                let mut are_types = elems.iter().map(|ta| match ta {
                    Term::Runtime { .. } => true,
                    Term::Variable(v) => v.bound_if_row_var().is_some(),
                    _ => false,
                });
                let elems = match are_types.next() {
                    Some(true) => {
                        assert!(are_types.all(|b| b)); // If one is a Type, so must the rest be
                        // So, anything that doesn't produce a Type, was a row variable => multiple Types
                        elems
                            .iter()
                            .flat_map(|ta| match ta.substitute(t) {
                                ty @ Term::Runtime { .. } => vec![ty],
                                Term::List(elems) => elems,
                                _ => panic!("Expected Type or row of Types"),
                            })
                            .collect()
                    }
                    _ => {
                        // not types, no need to flatten (and mustn't, in case of nested Sequences)
                        elems.iter().map(|ta| ta.substitute(t)).collect()
                    }
                };
                Term::List(elems)
            }
            Term::Tuple(elems) => {
                Term::Tuple(elems.iter().map(|elem| elem.substitute(t)).collect())
            }
            Term::Variable(TermVar { idx, cached_decl }) => t.apply_var(*idx, cached_decl),
            Term::RuntimeType { .. } => self.clone(),
            Term::BoundedNatType { .. } => self.clone(),
            Term::StringType => self.clone(),
            Term::BytesType => self.clone(),
            Term::FloatType => self.clone(),
            Term::ListType(item_type) => Term::new_list_type(item_type.substitute(t)),
            Term::TupleType(params) => {
                Term::TupleType(params.iter().map(|p| p.substitute(t)).collect())
            }
            Term::StaticType => self.clone(),
        }
    }
}

impl Transformable for Term {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        match self {
            Term::Runtime(ty) => ty.transform(tr),
            Term::List(elems) => elems.transform(tr),
            Term::Tuple(elems) => elems.transform(tr),
            Term::BoundedNat(_)
            | Term::String(_)
            | Term::Variable(_)
            | Term::Float(_)
            | Term::Bytes(_) => Ok(false),
            Term::RuntimeType { .. } => Ok(false),
            Term::BoundedNatType { .. } => Ok(false),
            Term::StringType => Ok(false),
            Term::BytesType => Ok(false),
            Term::FloatType => Ok(false),
            Term::ListType(item_type) => item_type.transform(tr),
            Term::TupleType(item_types) => item_types.transform(tr),
            Term::StaticType => Ok(false),
        }
    }
}

impl TermVar {
    /// Return the index.
    #[must_use]
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Determines whether this represents a row variable; if so, returns
    /// the [`TypeBound`] of the individual types it might stand for.
    #[must_use]
    pub fn bound_if_row_var(&self) -> Option<TypeBound> {
        if let Term::ListType(item_type) = &*self.cached_decl {
            if let Term::RuntimeType(b) = **item_type {
                return Some(b);
            }
        }
        None
    }
}

/// Checks that a [`Term`] is valid for a given type.
pub fn check_term_type(term: &Term, type_: &Term) -> Result<(), TermTypeError> {
    match (term, type_) {
        (Term::Variable(TermVar { cached_decl, .. }), _) if type_.is_supertype(cached_decl) => {
            Ok(())
        }
        (Term::Runtime(ty), Term::RuntimeType(bound)) if bound.contains(ty.least_upper_bound()) => {
            Ok(())
        }
        (Term::List(elems), Term::ListType(item_type)) => {
            elems.iter().try_for_each(|term| {
                // Also allow elements that are RowVars if fitting into a List of Types
                if let (Term::Variable(v), Term::RuntimeType(param_bound)) = (term, &**item_type) {
                    if v.bound_if_row_var()
                        .is_some_and(|arg_bound| param_bound.contains(arg_bound))
                    {
                        return Ok(());
                    }
                }
                check_term_type(term, item_type)
            })
        }
        (Term::Tuple(items), Term::TupleType(item_types)) => {
            if items.len() != item_types.len() {
                return Err(TermTypeError::WrongNumberTuple(
                    items.len(),
                    item_types.len(),
                ));
            }

            items
                .iter()
                .zip(item_types.iter())
                .try_for_each(|(term, type_)| check_term_type(term, type_))
        }
        (Term::BoundedNat(val), Term::BoundedNatType(bound)) if bound.valid_value(*val) => Ok(()),
        (Term::String { .. }, Term::StringType) => Ok(()),
        (Term::Bytes(_), Term::BytesType) => Ok(()),
        (Term::Float(_), Term::FloatType) => Ok(()),

        // Static types
        (Term::StaticType, Term::StaticType) => Ok(()),
        (Term::StringType, Term::StaticType) => Ok(()),
        (Term::BytesType, Term::StaticType) => Ok(()),
        (Term::BoundedNatType { .. }, Term::StaticType) => Ok(()),
        (Term::FloatType, Term::StaticType) => Ok(()),
        (Term::ListType { .. }, Term::StaticType) => Ok(()),
        (Term::TupleType(_), Term::StaticType) => Ok(()),

        _ => Err(TermTypeError::TypeMismatch {
            term: term.clone(),
            type_: type_.clone(),
        }),
    }
}

/// Check a list of [`Term`]s is valid for a list of types.
pub fn check_term_types(terms: &[Term], types: &[Term]) -> Result<(), TermTypeError> {
    if terms.len() != types.len() {
        return Err(TermTypeError::WrongNumberArgs(terms.len(), types.len()));
    }
    for (term, type_) in terms.iter().zip(types.iter()) {
        check_term_type(term, type_)?;
    }
    Ok(())
}

/// Errors that can occur when checking that a [`Term`] has an expected type.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum TermTypeError {
    #[allow(missing_docs)]
    /// For now, general case of a term not fitting a type.
    /// We'll have more cases when we allow general Containers.
    // TODO It may become possible to combine this with ConstTypeError.
    #[error("Term {term} does not fit declared type {type_}")]
    TypeMismatch { term: Term, type_: Term },
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

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::{Substitution, TypeArg, TypeParam, check_term_type};
    use crate::extension::prelude::{bool_t, usize_t};
    use crate::types::Term;
    use crate::types::{TypeBound, TypeRV, type_param::TermTypeError};

    #[test]
    fn type_arg_fits_param() {
        let rowvar = TypeRV::new_row_var_use;
        fn check(arg: impl Into<TypeArg>, param: &TypeParam) -> Result<(), TermTypeError> {
            check_term_type(&arg.into(), param)
        }
        fn check_seq<T: Clone + Into<TypeArg>>(
            args: &[T],
            param: &TypeParam,
        ) -> Result<(), TermTypeError> {
            let arg = args.iter().cloned().map_into().collect_vec().into();
            check_term_type(&arg, param)
        }
        // Simple cases: a Term::Type is a Term::RuntimeType but singleton sequences are lists
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

        // `Term::TupleType` requires a `Term::Tuple` of the same number of elems
        let usize_and_ty =
            TypeParam::TupleType(vec![TypeParam::max_nat_type(), TypeBound::Copyable.into()]);
        check(
            TypeArg::Tuple(vec![5.into(), usize_t().into()]),
            &usize_and_ty,
        )
        .unwrap();
        check(
            TypeArg::Tuple(vec![usize_t().into(), 5.into()]),
            &usize_and_ty,
        )
        .unwrap_err(); // Wrong way around
        let two_types = TypeParam::TupleType(vec![TypeBound::Any.into(), TypeBound::Any.into()]);
        check(TypeArg::new_var_use(0, two_types.clone()), &two_types).unwrap();
        // not a Row Var which could have any number of elems
        check(TypeArg::new_var_use(0, seq_param), &two_types).unwrap_err();
    }

    #[test]
    fn type_arg_subst_row() {
        let row_param = Term::new_list_type(TypeBound::Copyable);
        let row_arg: Term = vec![bool_t().into(), Term::UNIT].into();
        check_term_type(&row_arg, &row_param).unwrap();

        // Now say a row variable referring to *that* row was used
        // to instantiate an outer "row parameter" (list of type).
        let outer_param = Term::new_list_type(TypeBound::Any);
        let outer_arg = Term::new_list([
            TypeRV::new_row_var_use(0, TypeBound::Copyable).into(),
            usize_t().into(),
        ]);
        check_term_type(&outer_arg, &outer_param).unwrap();

        let outer_arg2 = outer_arg.substitute(&Substitution(&[row_arg]));
        assert_eq!(
            outer_arg2,
            vec![bool_t().into(), Term::UNIT, usize_t().into()].into()
        );

        // Of course this is still valid (as substitution is guaranteed to preserve validity)
        check_term_type(&outer_arg2, &outer_param).unwrap();
    }

    #[test]
    fn subst_list_list() {
        let outer_param = Term::new_list_type(Term::new_list_type(TypeBound::Any));
        let row_var_decl = Term::new_list_type(TypeBound::Copyable);
        let row_var_use = Term::new_var_use(0, row_var_decl.clone());
        let good_arg = Term::new_list([
            // The row variables here refer to `row_var_decl` above
            vec![usize_t().into()].into(),
            row_var_use.clone(),
            vec![row_var_use, usize_t().into()].into(),
        ]);
        check_term_type(&good_arg, &outer_param).unwrap();

        // Outer list cannot include single types:
        let Term::List(mut elems) = good_arg.clone() else {
            panic!()
        };
        elems.push(usize_t().into());
        assert_eq!(
            check_term_type(&Term::new_list(elems), &outer_param),
            Err(TermTypeError::TypeMismatch {
                term: usize_t().into(),
                // The error reports the type expected for each element of the list:
                type_: TypeParam::new_list_type(TypeBound::Any)
            })
        );

        // Now substitute a list of two types for that row-variable
        let row_var_arg = vec![usize_t().into(), bool_t().into()].into();
        check_term_type(&row_var_arg, &row_var_decl).unwrap();
        let subst_arg = good_arg.substitute(&Substitution(&[row_var_arg.clone()]));
        check_term_type(&subst_arg, &outer_param).unwrap(); // invariance of substitution
        assert_eq!(
            subst_arg,
            Term::new_list([
                Term::new_list([usize_t().into()]),
                row_var_arg,
                Term::new_list([usize_t().into(), bool_t().into(), usize_t().into()])
            ])
        );
    }

    #[test]
    fn bytes_json_roundtrip() {
        let bytes_arg = Term::Bytes(vec![0, 1, 2, 3, 255, 254, 253, 252].into());
        let serialized = serde_json::to_string(&bytes_arg).unwrap();
        let deserialized: Term = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, bytes_arg);
    }

    mod proptest {

        use proptest::prelude::*;

        use super::super::{TermVar, UpperBound};
        use crate::proptest::RecursionDepth;
        use crate::types::{Term, Type, TypeBound};

        impl Arbitrary for TermVar {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                (any::<usize>(), any_with::<Term>(depth))
                    .prop_map(|(idx, cached_decl)| Self {
                        idx,
                        cached_decl: Box::new(cached_decl),
                    })
                    .boxed()
            }
        }

        impl Arbitrary for Term {
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
                    any::<TypeBound>().prop_map(Self::from).boxed(),
                    any::<UpperBound>().prop_map(Self::from).boxed(),
                    any::<u64>().prop_map(Self::from).boxed(),
                    any::<String>().prop_map(Self::from).boxed(),
                    any::<Vec<u8>>()
                        .prop_map(|bytes| Self::Bytes(bytes.into()))
                        .boxed(),
                    any::<f64>()
                        .prop_map(|value| Self::Float(value.into()))
                        .boxed(),
                    any_with::<Type>(depth).prop_map(Self::from).boxed(),
                ]);
                if !depth.leaf() {
                    // we descend here because we these constructors contain Terms
                    strat = strat
                        .or(
                            // TODO this is a bit dodgy, TypeArgVariables are supposed
                            // to be constructed from TypeArg::new_var_use. We are only
                            // using this instance for serialization now, but if we want
                            // to generate valid TypeArgs this will need to change.
                            any_with::<TermVar>(depth.descend())
                                .prop_map(Self::Variable)
                                .boxed(),
                        )
                        .or(any_with::<Self>(depth.descend())
                            .prop_map(Self::new_list_type)
                            .boxed())
                        .or(vec(any_with::<Self>(depth.descend()), 0..3)
                            .prop_map(Self::new_tuple_type)
                            .boxed())
                        .or(vec(any_with::<Self>(depth.descend()), 0..3)
                            .prop_map(Term::new_list)
                            .boxed());
                }

                strat.boxed()
            }
        }

        proptest! {
            #[test]
            fn term_contains_itself(term: Term) {
                assert!(term.is_supertype(&term));
            }
        }
    }
}
