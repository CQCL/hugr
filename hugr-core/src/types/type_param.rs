//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::extension::TypeDef

use fxhash::FxHasher;
use itertools::Itertools;
use ordered_float::OrderedFloat;
#[cfg(test)]
use proptest_derive::Arbitrary;
use servo_arc::ThinArc;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU64;
use std::sync::Arc;
use thiserror::Error;

use super::row_var::MaybeRV;
use super::{
    NoRV, RowVariable, Substitution, Transformable, Type, TypeBase, TypeBound, TypeTransformer,
    check_typevar_decl,
};
use crate::extension::SignatureError;

/// The upper non-inclusive bound of a bounded natural.
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
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(
    from = "crate::types::serialize::TermSer",
    into = "crate::types::serialize::TermSer"
)]
pub struct Term(ThinArc<TermHeader, Term>);

/// Ensure that Term is the size of a pointer.
const _: () = assert!(std::mem::size_of::<Term>() == std::mem::size_of::<usize>());

/// Ensure that Option<Term> is the size of a pointer.
const _: () = assert!(std::mem::size_of::<Option<Term>>() == std::mem::size_of::<usize>());

impl Term {
    /// Create a new term.
    pub fn new(term: TermEnum) -> Self {
        let (data, terms) = TermData::split(term);
        Self::new_internal(data, terms)
    }

    fn new_internal(data: TermData, terms: &[Term]) -> Self {
        let header = TermHeader::new(data, terms);
        Self(ThinArc::from_header_and_iter(header, terms.iter().cloned()))
    }

    /// Returns a [`TermEnum`] to allow pattern matching this term.
    pub fn get(&self) -> TermEnum {
        match &self.0.header.data {
            TermData::RuntimeType(bound) => TermEnum::RuntimeType(*bound),
            TermData::StaticType => TermEnum::StaticType,
            TermData::BoundedNatType(bound) => TermEnum::BoundedNatType(bound.clone()),
            TermData::StringType => TermEnum::StringType,
            TermData::BytesType => TermEnum::BytesType,
            TermData::FloatType => TermEnum::FloatType,
            TermData::ListType(item_type) => TermEnum::ListType(item_type),
            TermData::TupleType => TermEnum::TupleType(self.0.slice()),
            TermData::Runtime(ty) => TermEnum::Runtime(ty),
            TermData::BoundedNat(value) => TermEnum::BoundedNat(*value),
            TermData::String(value) => TermEnum::String(value),
            TermData::Bytes(value) => TermEnum::Bytes(value),
            TermData::Float(value) => TermEnum::Float(*value),
            TermData::List => TermEnum::List(self.0.slice()),
            TermData::Tuple => TermEnum::Tuple(self.0.slice()),
            TermData::Variable(v) => TermEnum::Variable(v),
        }
    }
}

impl std::fmt::Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.get().fmt(f)
    }
}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.header.hash);
    }
}

/// Internal data structure for [`Term`] that owns all term information except
/// for the list of children.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TermData {
    RuntimeType(TypeBound),
    StaticType,
    BoundedNatType(UpperBound),
    StringType,
    BytesType,
    FloatType,
    ListType(Term),
    TupleType,
    Runtime(Box<Type>),
    BoundedNat(u64),
    String(String),
    Bytes(Arc<[u8]>),
    Float(OrderedFloat<f64>),
    List,
    Tuple,
    Variable(TermVar),
}

impl TermData {
    pub fn split<'a>(term: TermEnum<'a>) -> (Self, &'a [Term]) {
        match term {
            TermEnum::RuntimeType(bound) => (TermData::RuntimeType(bound), &[]),
            TermEnum::StaticType => (TermData::StaticType, &[]),
            TermEnum::BoundedNatType(bound) => (TermData::BoundedNatType(bound.clone()), &[]),
            TermEnum::StringType => (TermData::StringType, &[]),
            TermEnum::BytesType => (TermData::BytesType, &[]),
            TermEnum::FloatType => (TermData::FloatType, &[]),
            TermEnum::ListType(item_type) => (TermData::ListType(item_type.clone()), &[]),
            TermEnum::TupleType(item_types) => (TermData::TupleType, item_types),
            TermEnum::Runtime(ty) => (TermData::Runtime(Box::new(ty.clone())), &[]),
            TermEnum::BoundedNat(value) => (TermData::BoundedNat(value), &[]),
            TermEnum::String(value) => (TermData::String(value.to_string()), &[]),
            TermEnum::Bytes(value) => (TermData::Bytes(value.clone()), &[]),
            TermEnum::Float(value) => (TermData::Float(value), &[]),
            TermEnum::List(elems) => (TermData::List, elems),
            TermEnum::Tuple(elems) => (TermData::Tuple, elems),
            TermEnum::Variable(v) => (TermData::Variable(v.clone()), &[]),
        }
    }
}

/// Internal data structure for [`Term`] that contains the [`TermData`] together
/// with derived information. We require that the derived information can be
/// determined without additional context.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TermHeader {
    /// The cached hash value.
    hash: u64,
    /// The data of the term, excluding the list of child terms.
    data: TermData,
}

impl TermHeader {
    pub fn new(data: TermData, terms: &[Term]) -> Self {
        let hash = {
            let mut hasher = FxHasher::default();
            data.hash(&mut hasher);
            terms.hash(&mut hasher);
            hasher.finish()
        };

        Self { hash, data }
    }
}

/// The cases of [`Term`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, derive_more::Display)]
#[non_exhaustive]
pub enum TermEnum<'a> {
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
    /// The type of static strings.
    StringType,
    /// The type of static byte strings.
    BytesType,
    /// The type of static floating point numbers.
    FloatType,
    /// The type of static lists of indeterminate size containing terms of the
    /// specified static type.
    #[display("ListType[{_0}]")]
    ListType(&'a Term),
    /// The type of static tuples.
    #[display("TupleType[{}]", _0.iter().map(std::string::ToString::to_string).join(", "))]
    TupleType(&'a [Term]),
    /// A runtime type as a term.
    #[display("{_0}")]
    Runtime(&'a Type),
    /// A 64bit unsigned integer literal.
    #[display("{_0}")]
    BoundedNat(u64),
    /// UTF-8 encoded string literal.
    #[display("\"{_0}\"")]
    String(&'a str),
    /// Byte string literal.
    #[display("bytes")]
    Bytes(&'a Arc<[u8]>),
    /// A 64-bit floating point number.
    #[display("{}", _0.into_inner())]
    Float(OrderedFloat<f64>),
    /// A list of static terms.
    #[display("[{}]", {
        use itertools::Itertools as _;
        _0.iter().map(|t|t.to_string()).join(",")
    })]
    List(&'a [Term]),
    /// A tuple of static terms.
    #[display("({})", {
        use itertools::Itertools as _;
        _0.iter().map(std::string::ToString::to_string).join(",")
    })]
    Tuple(&'a [Term]),
    /// Variable (used in type schemes or inside polymorphic functions),
    /// but not a runtime type (not even a row variable i.e. list of runtime types)
    /// - see [`Term::new_var_use`]
    #[display("{_0}")]
    Variable(&'a TermVar),
}

impl Term {
    /// Creates a (bounded) natural type with the maximum bound (`u64::MAX` + 1).
    #[must_use]
    pub fn max_nat_type() -> Self {
        Self::new(TermEnum::BoundedNatType(UpperBound(None)))
    }

    /// Creates a bounded natural type with the stated upper bound (non-exclusive).
    #[must_use]
    pub fn bounded_nat_type(upper_bound: NonZeroU64) -> Self {
        Self::new(TermEnum::BoundedNatType(UpperBound(Some(upper_bound))))
    }

    /// Creates a new list type given the type of its elements.
    pub fn new_list_type(elem: impl Into<Term>) -> Self {
        Self::new(TermEnum::ListType(&elem.into()))
    }

    /// Creates a new tuple type given the types of its elements.
    pub fn new_tuple_type(item_types: impl IntoIterator<Item = Term>) -> Self {
        let item_types: Vec<_> = item_types.into_iter().collect();
        Self::new(TermEnum::TupleType(&item_types))
    }

    /// Creates a new list given an iterator of items.
    pub fn new_list(elems: impl IntoIterator<Item = Term>) -> Self {
        let elems: Vec<_> = elems.into_iter().collect();
        Self::new(TermEnum::List(&elems))
    }

    /// Creates a new tuple given an iterator of items.
    pub fn new_tuple(elems: impl IntoIterator<Item = Term>) -> Self {
        let elems: Vec<_> = elems.into_iter().collect();
        Self::new(TermEnum::Tuple(&elems))
    }

    /// Creates a new bytes literal.
    pub fn new_bytes(value: impl Into<Arc<[u8]>>) -> Self {
        Self::new_internal(TermData::Bytes(value.into()), &[])
    }

    /// Creates a new string literal.
    pub fn new_string(value: impl ToString) -> Self {
        Self::new_internal(TermData::String(value.to_string()), &[])
    }

    /// Checks if this term is a supertype of another.
    ///
    /// The subtyping relation applies primarily to terms that represent static
    /// types. For consistency the relation is extended to a partial order on
    /// all terms; in particular it is reflexive so that every term (even if it
    /// is not a static type) is considered a subtype of itself.
    fn is_supertype(&self, other: &Term) -> bool {
        match (self.get(), other.get()) {
            (TermEnum::RuntimeType(b1), TermEnum::RuntimeType(b2)) => b1.contains(b2),
            (TermEnum::BoundedNatType(b1), TermEnum::BoundedNatType(b2)) => b1.contains(&b2),
            (TermEnum::StringType, TermEnum::StringType) => true,
            (TermEnum::StaticType, TermEnum::StaticType) => true,
            (TermEnum::ListType(e1), TermEnum::ListType(e2)) => e1.is_supertype(e2),
            (TermEnum::TupleType(es1), TermEnum::TupleType(es2)) => {
                es1.len() == es2.len() && es1.iter().zip(es2).all(|(e1, e2)| e1.is_supertype(e2))
            }
            (TermEnum::BytesType, TermEnum::BytesType) => true,
            (TermEnum::FloatType, TermEnum::FloatType) => true,
            (TermEnum::Runtime(t1), TermEnum::Runtime(t2)) => t1 == t2,
            (TermEnum::BoundedNat(n1), TermEnum::BoundedNat(n2)) => n1 >= n2,
            (TermEnum::String(s1), TermEnum::String(s2)) => s1 == s2,
            (TermEnum::Bytes(v1), TermEnum::Bytes(v2)) => v1 == v2,
            (TermEnum::Float(f1), TermEnum::Float(f2)) => f1 == f2,
            (TermEnum::Variable(v1), TermEnum::Variable(v2)) => v1 == v2,
            (TermEnum::List(es1), TermEnum::List(es2)) => {
                es1.len() == es2.len() && es1.iter().zip(es2).all(|(e1, e2)| e1.is_supertype(e2))
            }
            (TermEnum::Tuple(es1), TermEnum::Tuple(es2)) => {
                es1.len() == es2.len() && es1.iter().zip(es2).all(|(e1, e2)| e1.is_supertype(e2))
            }
            (_, _) => false,
        }
    }
}

impl From<TypeBound> for Term {
    fn from(bound: TypeBound) -> Self {
        Self::new(TermEnum::RuntimeType(bound))
    }
}

impl From<UpperBound> for Term {
    fn from(bound: UpperBound) -> Self {
        Self::new(TermEnum::BoundedNatType(bound))
    }
}

impl<RV: MaybeRV> From<TypeBase<RV>> for Term {
    fn from(value: TypeBase<RV>) -> Self {
        match value.try_into_type() {
            Ok(ty) => Term::new(TermEnum::Runtime(&ty)),
            Err(RowVariable(idx, bound)) => Term::new_var_use(idx, TypeParam::new_list_type(bound)),
        }
    }
}

impl From<u64> for Term {
    fn from(n: u64) -> Self {
        Self::new(TermEnum::BoundedNat(n))
    }
}

impl From<String> for Term {
    fn from(arg: String) -> Self {
        Self::new(TermEnum::String(&arg))
    }
}

impl From<&str> for Term {
    fn from(arg: &str) -> Self {
        Self::new(TermEnum::String(arg))
    }
}

impl From<Vec<Term>> for Term {
    fn from(elems: Vec<Term>) -> Self {
        Self::new_list(elems)
    }
}

impl From<Arc<[u8]>> for Term {
    fn from(value: Arc<[u8]>) -> Self {
        Term::new_bytes(value)
    }
}

impl From<f64> for Term {
    fn from(value: f64) -> Self {
        Term::new(TermEnum::Float(value.into()))
    }
}

impl From<hugr_model::v0::Literal> for Term {
    fn from(value: hugr_model::v0::Literal) -> Self {
        use hugr_model::v0::Literal;
        match value {
            Literal::Str(value) => Term::new_string(value),
            Literal::Nat(value) => Term::from(value),
            Literal::Bytes(value) => Term::new_bytes(value),
            Literal::Float(value) => Term::from(value.into_inner()),
        }
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
    cached_decl: Term,
}

impl Term {
    /// [`Type::UNIT`] as a term.
    pub fn new_unit() -> Self {
        Type::UNIT.into()
    }

    /// Creates a new term representing a local variable, given its index and type.
    /// The type must be exactly that with which the variable was declared.
    #[must_use]
    pub fn new_var_use(idx: usize, decl: Term) -> Self {
        match decl.get() {
            // Note a TypeParam::List of TypeParam::Type *cannot* be represented
            // as a TypeArg::Type because the latter stores a Type<false> i.e. only a single type,
            // not a RowVariable.
            TermEnum::RuntimeType(b) => Type::new_var_use(idx, b).into(),
            _ => Term::new(TermEnum::Variable(&TermVar {
                idx,
                cached_decl: decl,
            })),
        }
    }

    /// Returns an integer if the [`Term`] is a natural number literal.
    #[must_use]
    pub fn as_nat(&self) -> Option<u64> {
        match self.get() {
            TermEnum::BoundedNat(n) => Some(n),
            _ => None,
        }
    }

    /// Returns a [`Type`] if the [`Term`] is a runtime type.
    #[must_use]
    pub fn as_runtime(&self) -> Option<TypeBase<NoRV>> {
        match self.get() {
            TermEnum::Runtime(ty) => Some(ty.clone()),
            _ => None,
        }
    }

    /// Returns a string if the [`Term`] is a string literal.
    #[must_use]
    pub fn as_string(&self) -> Option<String> {
        match self.get() {
            TermEnum::String(arg) => Some(arg.to_string()),
            _ => None,
        }
    }

    /// See [`Type::validate`].
    pub(crate) fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        match self.get() {
            TermEnum::Runtime(ty) => ty.validate(var_decls),
            TermEnum::List(elems) => {
                // TODO: Full validation would check that the type of the elements agrees
                elems.iter().try_for_each(|a| a.validate(var_decls))
            }
            TermEnum::Tuple(elems) => elems.iter().try_for_each(|a| a.validate(var_decls)),
            TermEnum::BoundedNat { .. }
            | TermEnum::String(_)
            | TermEnum::Float(_)
            | TermEnum::Bytes(_) => Ok(()),
            TermEnum::Variable(TermVar { idx, cached_decl }) => {
                assert!(
                    !matches!(cached_decl.get(), TermEnum::RuntimeType { .. }),
                    "Malformed TypeArg::Variable {cached_decl} - should be inconstructible"
                );

                check_typevar_decl(var_decls, *idx, cached_decl)
            }
            TermEnum::RuntimeType(_) => Ok(()),
            TermEnum::BoundedNatType(_) => Ok(()),
            TermEnum::StringType => Ok(()),
            TermEnum::BytesType => Ok(()),
            TermEnum::FloatType => Ok(()),
            TermEnum::ListType(item_type) => item_type.validate(var_decls),
            TermEnum::TupleType(item_types) => {
                item_types.iter().try_for_each(|p| p.validate(var_decls))
            }
            TermEnum::StaticType => Ok(()),
        }
    }

    pub(crate) fn substitute(&self, t: &Substitution) -> Self {
        match self.get() {
            TermEnum::Runtime(ty) => {
                // RowVariables are represented as Term::Variable
                ty.substitute1(t).into()
            }
            TermEnum::BoundedNat(_)
            | TermEnum::String(_)
            | TermEnum::Bytes(_)
            | TermEnum::Float(_) => self.clone(),
            TermEnum::List(elems) => {
                let mut are_types = elems.iter().map(|ta| match ta.get() {
                    TermEnum::Runtime(_) => true,
                    TermEnum::Variable(v) => v.bound_if_row_var().is_some(),
                    _ => false,
                });
                let elems: Vec<_> = match are_types.next() {
                    Some(true) => {
                        assert!(are_types.all(|b| b)); // If one is a Type, so must the rest be
                        // So, anything that doesn't produce a Type, was a row variable => multiple Types
                        elems
                            .iter()
                            .flat_map(|ta| {
                                let ta = ta.substitute(t);
                                match ta.get() {
                                    TermEnum::Runtime(_) => vec![ta],
                                    TermEnum::List(elems) => elems.to_vec(),
                                    _ => panic!("Expected Type or row of Types"),
                                }
                            })
                            .collect()
                    }
                    _ => {
                        // not types, no need to flatten (and mustn't, in case of nested Sequences)
                        elems.iter().map(|ta| ta.substitute(t)).collect()
                    }
                };
                Term::new_list(elems)
            }
            TermEnum::Tuple(elems) => Term::new_tuple(elems.iter().map(|elem| elem.substitute(t))),
            TermEnum::Variable(TermVar { idx, cached_decl }) => t.apply_var(*idx, cached_decl),
            TermEnum::RuntimeType(_) => self.clone(),
            TermEnum::BoundedNatType(_) => self.clone(),
            TermEnum::StringType => self.clone(),
            TermEnum::BytesType => self.clone(),
            TermEnum::FloatType => self.clone(),
            TermEnum::ListType(item_type) => Term::new_list_type(item_type.substitute(t)),
            TermEnum::TupleType(item_types) => {
                Term::new_tuple_type(item_types.iter().map(|p| p.substitute(t)))
            }
            TermEnum::StaticType => self.clone(),
        }
    }
}

impl Transformable for Term {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        match self.get() {
            TermEnum::Runtime(ty) => {
                let mut ty = ty.clone();
                if ty.transform(tr)? {
                    *self = Term::from(ty);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            TermEnum::List(elems) => {
                let mut elems = elems.to_vec();
                if elems.transform(tr)? {
                    *self = Term::new_list(elems);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            TermEnum::Tuple(elems) => {
                let mut elems = elems.to_vec();
                if elems.transform(tr)? {
                    *self = Term::new_tuple(elems);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            TermEnum::BoundedNat(_)
            | TermEnum::String(_)
            | TermEnum::Variable(_)
            | TermEnum::Float(_)
            | TermEnum::Bytes(_) => Ok(false),
            TermEnum::RuntimeType(_) => Ok(false),
            TermEnum::BoundedNatType(_) => Ok(false),
            TermEnum::StringType => Ok(false),
            TermEnum::BytesType => Ok(false),
            TermEnum::FloatType => Ok(false),
            TermEnum::ListType(item_type) => {
                let mut item_type = item_type.clone();
                if item_type.transform(tr)? {
                    *self = Term::new_list_type(item_type);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            TermEnum::TupleType(item_types) => {
                let mut item_types = item_types.to_vec();
                if item_types.transform(tr)? {
                    *self = Term::new_tuple_type(item_types);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            TermEnum::StaticType => Ok(false),
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
        if let TermEnum::ListType(item_type) = self.cached_decl.get() {
            if let TermEnum::RuntimeType(b) = item_type.get() {
                return Some(b);
            }
        }
        None
    }
}

/// Checks that a [`Term`] is valid for a given type.
pub fn check_term_type(term: &Term, type_: &Term) -> Result<(), TermTypeError> {
    match (term.get(), type_.get()) {
        (TermEnum::Variable(TermVar { cached_decl, .. }), _) if type_.is_supertype(cached_decl) => {
            Ok(())
        }
        (TermEnum::Runtime(ty), TermEnum::RuntimeType(bound))
            if bound.contains(ty.least_upper_bound()) =>
        {
            Ok(())
        }
        (TermEnum::List(elems), TermEnum::ListType(item_type)) => {
            elems.iter().try_for_each(|term| {
                // Also allow elements that are RowVars if fitting into a List of Types
                if let (TermEnum::Variable(v), TermEnum::RuntimeType(param_bound)) =
                    (term.get(), item_type.get())
                {
                    if v.bound_if_row_var()
                        .is_some_and(|arg_bound| param_bound.contains(arg_bound))
                    {
                        return Ok(());
                    }
                }
                check_term_type(term, item_type)
            })
        }
        (TermEnum::Tuple(items), TermEnum::TupleType(item_types)) => {
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
        (TermEnum::BoundedNat(value), TermEnum::BoundedNatType(bound))
            if bound.valid_value(value) =>
        {
            Ok(())
        }
        (TermEnum::String(_), TermEnum::StringType) => Ok(()),
        (TermEnum::Bytes(_), TermEnum::BytesType) => Ok(()),
        (TermEnum::Float(_), TermEnum::FloatType) => Ok(()),

        // Static types
        (TermEnum::StaticType, TermEnum::StaticType) => Ok(()),
        (TermEnum::StringType, TermEnum::StaticType) => Ok(()),
        (TermEnum::BytesType, TermEnum::StaticType) => Ok(()),
        (TermEnum::BoundedNatType(_), TermEnum::StaticType) => Ok(()),
        (TermEnum::FloatType, TermEnum::StaticType) => Ok(()),
        (TermEnum::ListType(_), TermEnum::StaticType) => Ok(()),
        (TermEnum::TupleType(_), TermEnum::StaticType) => Ok(()),
        (TermEnum::RuntimeType(_), TermEnum::StaticType) => Ok(()),

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
    use crate::types::type_param::TermEnum;
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
            Term::new_tuple_type([TypeParam::max_nat_type(), TypeBound::Copyable.into()]);
        check(
            TypeArg::new_tuple([5.into(), usize_t().into()]),
            &usize_and_ty,
        )
        .unwrap();
        check(Term::new_tuple([usize_t().into(), 5.into()]), &usize_and_ty).unwrap_err(); // Wrong way around
        let two_types = Term::new_tuple_type([TypeBound::Any.into(), TypeBound::Any.into()]);
        check(TypeArg::new_var_use(0, two_types.clone()), &two_types).unwrap();
        // not a Row Var which could have any number of elems
        check(TypeArg::new_var_use(0, seq_param), &two_types).unwrap_err();
    }

    #[test]
    fn type_arg_subst_row() {
        let row_param = Term::new_list_type(TypeBound::Copyable);
        let row_arg = Term::new_list([bool_t().into(), TypeArg::new_unit()]);
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
            Term::new_list([bool_t().into(), Term::new_unit(), usize_t().into()])
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
        let TermEnum::List(elems) = good_arg.get() else {
            panic!()
        };
        let mut elems = elems.to_vec();
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
        let bytes_arg = Term::new_bytes([0, 1, 2, 3, 255, 254, 253, 252]);
        let serialized = serde_json::to_string(&bytes_arg).unwrap();
        let deserialized: Term = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, bytes_arg);
    }

    mod proptest {

        use proptest::prelude::*;

        use super::super::{TermVar, UpperBound};
        use crate::proptest::RecursionDepth;
        use crate::types::type_param::TermEnum;
        use crate::types::{Term, Type, TypeBound};

        impl Arbitrary for TermVar {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                (any::<usize>(), any_with::<Term>(depth))
                    .prop_map(|(idx, cached_decl)| Self { idx, cached_decl })
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
                    Just(Self::new(TermEnum::StringType)).boxed(),
                    Just(Self::new(TermEnum::BytesType)).boxed(),
                    Just(Self::new(TermEnum::FloatType)).boxed(),
                    Just(Self::new(TermEnum::StringType)).boxed(),
                    any::<TypeBound>().prop_map(Self::from).boxed(),
                    any::<UpperBound>().prop_map(Self::from).boxed(),
                    any::<u64>().prop_map(Self::from).boxed(),
                    any::<String>().prop_map(Self::from).boxed(),
                    any::<Vec<u8>>().prop_map(Self::new_bytes).boxed(),
                    any::<f64>().prop_map(Self::from).boxed(),
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
                                .prop_map(|v| Self::new(TermEnum::Variable(&v)))
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
