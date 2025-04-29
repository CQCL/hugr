//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::extension::TypeDef

use itertools::Itertools;
use ordered_float::OrderedFloat;
#[cfg(test)]
use proptest_derive::Arbitrary;
use smallvec::{SmallVec, smallvec};
use std::iter::FusedIterator;
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

/// A *kind* of [`TypeArg`]. Thus, a parameter declared by a [`PolyFuncType`] or [`PolyFuncTypeRV`],
/// specifying a value that must be provided statically in order to instantiate it.
///
/// [`PolyFuncType`]: super::PolyFuncType
/// [`PolyFuncTypeRV`]: super::PolyFuncTypeRV
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, derive_more::Display, serde::Deserialize, serde::Serialize,
)]
#[non_exhaustive]
#[serde(tag = "tp")]
pub enum TypeParam {
    /// Argument is a [`TypeArg::Type`].
    #[display("Type{}", match b {
        TypeBound::Any => String::new(),
        _ => format!("[{b}]")
    })]
    Type {
        /// Bound for the type parameter.
        b: TypeBound,
    },
    /// Argument is a [`TypeArg::BoundedNat`] that is less than the upper bound.
    #[display("{}", match bound.value() {
        Some(v) => format!("BoundedNat[{v}]"),
        None => "Nat".to_string()
    })]
    BoundedNat {
        /// Upper bound for the Nat parameter.
        bound: UpperBound,
    },
    /// Argument is a [`TypeArg::String`].
    String,
    /// Argument is a [`TypeArg::Bytes`].
    Bytes,
    /// Argument is a [`TypeArg::Float`].
    Float,
    /// Argument is a [`TypeArg::List`]. A list of indeterminate size containing
    /// parameters all of the (same) specified element type.
    #[display("List[{param}]")]
    List {
        /// The [`TypeParam`] describing each element of the list.
        param: Box<TypeParam>,
    },
    /// Argument is a [`TypeArg::Tuple`]. A tuple of parameters.
    #[display("Tuple[{}]", params.iter().map(std::string::ToString::to_string).join(", "))]
    Tuple {
        /// The [`TypeParam`]s contained in the tuple.
        params: Vec<TypeParam>,
    },
}

impl TypeParam {
    /// [`TypeParam::BoundedNat`] with the maximum bound (`u64::MAX` + 1)
    #[must_use]
    pub const fn max_nat() -> Self {
        Self::BoundedNat {
            bound: UpperBound(None),
        }
    }

    /// [`TypeParam::BoundedNat`] with the stated upper bound (non-exclusive)
    #[must_use]
    pub const fn bounded_nat(upper_bound: NonZeroU64) -> Self {
        Self::BoundedNat {
            bound: UpperBound(Some(upper_bound)),
        }
    }

    /// Make a new [`TypeParam::List`] (an arbitrary-length homogeneous list)
    pub fn new_list(elem: impl Into<TypeParam>) -> Self {
        Self::List {
            param: Box::new(elem.into()),
        }
    }

    /// Make a new [`TypeParam::Tuple`].
    pub fn new_tuple(elems: impl IntoIterator<Item = TypeParam>) -> Self {
        Self::Tuple {
            params: elems.into_iter().collect(),
        }
    }

    fn contains(&self, other: &TypeParam) -> bool {
        match (self, other) {
            (TypeParam::Type { b: b1 }, TypeParam::Type { b: b2 }) => b1.contains(*b2),
            (TypeParam::BoundedNat { bound: b1 }, TypeParam::BoundedNat { bound: b2 }) => {
                b1.contains(b2)
            }
            (TypeParam::String, TypeParam::String) => true,
            (TypeParam::List { param: e1 }, TypeParam::List { param: e2 }) => e1.contains(e2),
            (TypeParam::Tuple { params: es1 }, TypeParam::Tuple { params: es2 }) => {
                es1.len() == es2.len() && es1.iter().zip(es2).all(|(e1, e2)| e1.contains(e2))
            }
            _ => false,
        }
    }
}

impl From<TypeBound> for TypeParam {
    fn from(bound: TypeBound) -> Self {
        Self::Type { b: bound }
    }
}

impl From<UpperBound> for TypeParam {
    fn from(bound: UpperBound) -> Self {
        Self::BoundedNat { bound }
    }
}

/// A statically-known argument value to an operation.
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize, derive_more::Display,
)]
#[non_exhaustive]
#[serde(tag = "tya")]
pub enum TypeArg {
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
    /// Instance of [`TypeParam::List`] defined by a sequence of concatenated lists of the same type.
    #[display("[{}]", {
        use itertools::Itertools as _;
        lists.iter().map(|t| format!("... {}", t)).join(",")
    })]
    ListConcat {
        /// The lists to concat.
        lists: Vec<TypeArg>,
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
    /// Instance of [`TypeParam::Tuple`] defined by a sequence of concatenated tuples.
    #[display("({})", {
          use itertools::Itertools as _;
          tuples.iter().map(|tuple| format!("... {}", tuple)).join(",")
      })]
    TupleConcat {
        /// The tuples to concat.
        tuples: Vec<TypeArg>,
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

impl<RV: MaybeRV> From<TypeBase<RV>> for TypeArg {
    fn from(value: TypeBase<RV>) -> Self {
        match value.try_into_type() {
            Ok(ty) => TypeArg::Type { ty },
            Err(RowVariable(idx, bound)) => TypeArg::new_var_use(idx, TypeParam::new_list(bound)),
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
    cached_decl: TypeParam,
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
            TypeParam::Type { b } => Type::new_var_use(idx, b).into(),
            _ => TypeArg::Variable {
                v: TypeArgVariable {
                    idx,
                    cached_decl: decl,
                },
            },
        }
    }

    /// Creates a new string literal.
    #[inline]
    pub fn new_string(str: impl ToString) -> Self {
        Self::String {
            arg: str.to_string(),
        }
    }

    /// Creates a new list from its items.
    #[inline]
    pub fn new_list(items: impl IntoIterator<Item = Self>) -> Self {
        Self::List {
            elems: items.into_iter().collect(),
        }
    }

    /// Creates a new concatenated list.
    #[inline]
    pub fn new_list_concat(lists: impl IntoIterator<Item = Self>) -> Self {
        Self::ListConcat {
            lists: lists.into_iter().collect(),
        }
    }

    /// Creates a new tuple from its items.
    #[inline]
    pub fn new_tuple(items: impl IntoIterator<Item = Self>) -> Self {
        Self::Tuple {
            elems: items.into_iter().collect(),
        }
    }

    /// Creates a new concatenated tuple.
    #[inline]
    pub fn new_tuple_concat(tuples: impl IntoIterator<Item = Self>) -> Self {
        Self::TupleConcat {
            tuples: tuples.into_iter().collect(),
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
            TypeArg::Type { ty } => ty.validate(var_decls),
            TypeArg::List { elems } => {
                // TODO: Full validation would check that the type of the elements agrees
                elems.iter().try_for_each(|a| a.validate(var_decls))
            }
            TypeArg::ListConcat { lists } => {
                // TODO: Full validation would check that each of the lists is indeed a
                // list or list variable of the correct types.
                lists.iter().try_for_each(|a| a.validate(var_decls))
            }
            TypeArg::Tuple { elems } => elems.iter().try_for_each(|a| a.validate(var_decls)),
            TypeArg::TupleConcat { tuples } => {
                tuples.iter().try_for_each(|a| a.validate(var_decls))
            }
            TypeArg::BoundedNat { .. }
            | TypeArg::String { .. }
            | TypeArg::Float { .. }
            | TypeArg::Bytes { .. } => Ok(()),
            TypeArg::Variable {
                v: TypeArgVariable { idx, cached_decl },
            } => {
                assert!(
                    !matches!(cached_decl, TypeParam::Type { .. }),
                    "Malformed TypeArg::Variable {cached_decl} - should be inconstructible"
                );

                check_typevar_decl(var_decls, *idx, cached_decl)
            }
        }
    }

    pub(crate) fn substitute(&self, t: &Substitution) -> Self {
        match self {
            TypeArg::Type { ty } => {
                // RowVariables are represented as TypeArg::Variable
                ty.substitute1(t).into()
            }
            TypeArg::BoundedNat { .. }
            | TypeArg::String { .. }
            | TypeArg::Bytes { .. }
            | TypeArg::Float { .. } => self.clone(), // We do not allow variables as bounds on BoundedNat's
            TypeArg::List { elems } => {
                // NOTE: This implements a hack allowing substitutions to
                // replace `TypeArg::Variable`s representing "row variables"
                // with a list that is to be spliced into the containing list.
                // We won't need this code anymore once we stop conflating types
                // with lists of types.

                fn is_type(type_arg: &TypeArg) -> bool {
                    match type_arg {
                        TypeArg::Type { .. } => true,
                        TypeArg::Variable { v } => v.bound_if_row_var().is_some(),
                        _ => false,
                    }
                }

                let are_types = elems.first().map(is_type).unwrap_or(false);

                Self::new_list_from_parts(elems.iter().map(|elem| match elem.substitute(t) {
                    list @ TypeArg::List { .. } if are_types => SeqPart::Splice(list),
                    list @ TypeArg::ListConcat { .. } if are_types => SeqPart::Splice(list),
                    elem => SeqPart::Item(elem),
                }))
            }
            TypeArg::ListConcat { lists } => {
                // When a substitution instantiates spliced list variables, we
                // may be able to merge the concatenated lists.
                Self::new_list_from_parts(
                    lists.iter().map(|list| SeqPart::Splice(list.substitute(t))),
                )
            }
            TypeArg::Tuple { elems } => TypeArg::Tuple {
                elems: elems.iter().map(|elem| elem.substitute(t)).collect(),
            },
            TypeArg::TupleConcat { tuples } => {
                // When a substitution instantiates spliced tuple variables,
                // we may be able to merge the concatenated tuples.
                Self::new_tuple_from_parts(
                    tuples
                        .iter()
                        .map(|tuple| SeqPart::Splice(tuple.substitute(t))),
                )
            }
            TypeArg::Variable {
                v: TypeArgVariable { idx, cached_decl },
            } => t.apply_var(*idx, cached_decl),
        }
    }

    /// Helper method for [`TypeArg::new_list_from_parts`] and [`TypeArg::new_tuple_from_parts`].
    fn new_seq_from_parts(
        parts: impl IntoIterator<Item = SeqPart<Self>>,
        make_items: impl Fn(Vec<Self>) -> Self,
        make_concat: impl Fn(Vec<Self>) -> Self,
    ) -> Self {
        let mut items = Vec::new();
        let mut seqs = Vec::new();

        for part in parts {
            match part {
                SeqPart::Item(item) => items.push(item),
                SeqPart::Splice(seq) => {
                    if !items.is_empty() {
                        seqs.push(make_items(std::mem::take(&mut items)));
                    }
                    seqs.push(seq);
                }
            }
        }

        if seqs.is_empty() {
            make_items(items)
        } else if items.is_empty() {
            make_concat(seqs)
        } else {
            seqs.push(make_items(items));
            make_concat(seqs)
        }
    }

    /// Creates a new list from a sequence of [`SeqPart`]s.
    pub fn new_list_from_parts(parts: impl IntoIterator<Item = SeqPart<Self>>) -> Self {
        Self::new_seq_from_parts(
            parts.into_iter().flat_map(ListPartIter::new),
            |elems| TypeArg::List { elems },
            |lists| TypeArg::ListConcat { lists },
        )
    }

    /// Iterates over the [`SeqPart`]s of a list.
    ///
    /// # Examples
    ///
    /// The parts of a closed list are the items of that list wrapped in [`SeqPart::Item`]:
    ///
    /// ```
    /// # use hugr_core::types::type_param::{TypeArg, SeqPart};
    /// # let a = TypeArg::new_string("a");
    /// # let b = TypeArg::new_string("b");
    /// let type_arg = TypeArg::new_list([a.clone(), b.clone()]);
    ///
    /// assert_eq!(
    ///     type_arg.into_list_parts().collect::<Vec<_>>(),
    ///     vec![SeqPart::Item(a), SeqPart::Item(b)]
    /// );
    /// ```
    ///
    /// Parts of a concatenated list that are not closed lists are wrapped in [`SeqPart::Splice`]:
    ///
    /// ```
    /// # use hugr_core::types::type_param::{TypeParam, TypeArg, SeqPart};
    /// # let a = TypeArg::new_string("a");
    /// # let b = TypeArg::new_string("b");
    /// # let c = TypeArg::new_string("c");
    /// let var = TypeArg::new_var_use(0, TypeParam::new_list(TypeParam::String));
    /// let type_arg = TypeArg::new_list_concat([
    ///     TypeArg::new_list([a.clone(), b.clone()]),
    ///     var.clone(),
    ///     TypeArg::new_list([c.clone()])
    ///  ]);
    ///
    /// assert_eq!(
    ///     type_arg.into_list_parts().collect::<Vec<_>>(),
    ///     vec![SeqPart::Item(a), SeqPart::Item(b), SeqPart::Splice(var), SeqPart::Item(c)]
    /// );
    /// ```
    ///
    /// Nested concatenations are traversed recursively:
    ///
    /// ```
    /// # use hugr_core::types::type_param::{TypeArg, SeqPart};
    /// # let a = TypeArg::new_string("a");
    /// # let b = TypeArg::new_string("b");
    /// # let c = TypeArg::new_string("c");
    /// let type_arg = TypeArg::new_list_concat([
    ///     TypeArg::new_list_concat([
    ///         TypeArg::new_list([a.clone()]),
    ///         TypeArg::new_list([b.clone()])
    ///     ]),
    ///     TypeArg::new_list([]),
    ///     TypeArg::new_list([c.clone()])
    /// ]);
    ///
    /// assert_eq!(
    ///     type_arg.into_list_parts().collect::<Vec<_>>(),
    ///     vec![SeqPart::Item(a), SeqPart::Item(b), SeqPart::Item(c)]
    /// );
    /// ```
    ///
    /// When invoked on a type argument that is not a list, a single
    /// [`SeqPart::Splice`] is returned that wraps the type argument.
    /// This is the expected behaviour for type variables that stand for lists.
    /// This behaviour also allows this method not to fail on ill-typed type arguments.
    /// ```
    /// # use hugr_core::types::type_param::{TypeArg, SeqPart};
    /// let type_arg = TypeArg::new_string("not a list");
    /// assert_eq!(
    ///     type_arg.clone().into_list_parts().collect::<Vec<_>>(),
    ///     vec![SeqPart::Splice(type_arg)]
    /// );
    /// ```
    #[inline]
    pub fn into_list_parts(self) -> ListPartIter {
        ListPartIter::new(SeqPart::Splice(self))
    }

    /// Creates a new tuple from a sequence of [`SeqPart`]s.
    ///
    /// Analogous to [`TypeArg::new_list_from_parts`].
    pub fn new_tuple_from_parts(parts: impl IntoIterator<Item = SeqPart<Self>>) -> Self {
        Self::new_seq_from_parts(
            parts.into_iter().flat_map(TuplePartIter::new),
            |elems| TypeArg::Tuple { elems },
            |tuples| TypeArg::TupleConcat { tuples },
        )
    }

    /// Iterates over the [`SeqPart`]s of a tuple.
    ///
    /// Analogous to [`TypeArg::into_list_parts`].
    #[inline]
    pub fn into_tuple_parts(self) -> TuplePartIter {
        TuplePartIter::new(SeqPart::Splice(self))
    }
}

impl Transformable for TypeArg {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        match self {
            TypeArg::Type { ty } => ty.transform(tr),
            TypeArg::List { elems } => elems.transform(tr),
            TypeArg::ListConcat { lists } => lists.transform(tr),
            TypeArg::Tuple { elems } => elems.transform(tr),
            TypeArg::TupleConcat { tuples } => tuples.transform(tr),
            TypeArg::BoundedNat { .. }
            | TypeArg::String { .. }
            | TypeArg::Variable { .. }
            | TypeArg::Float { .. }
            | TypeArg::Bytes { .. } => Ok(false),
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
        if let TypeParam::List { param } = &self.cached_decl {
            if let TypeParam::Type { b } = **param {
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
        (TypeArg::Type { ty }, TypeParam::Type { b: bound })
            if bound.contains(ty.least_upper_bound()) =>
        {
            Ok(())
        }
        (TypeArg::List { elems }, TypeParam::List { param }) => {
            elems.iter().try_for_each(|arg| {
                // Also allow elements that are RowVars if fitting into a List of Types
                if let (TypeArg::Variable { v }, TypeParam::Type { b: param_bound }) =
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
        (TypeArg::ListConcat { lists }, TypeParam::List { .. }) => lists
            .iter()
            .try_for_each(|list| check_type_arg(list, param)),
        (TypeArg::Tuple { elems: items }, TypeParam::Tuple { params: types }) => {
            if items.len() != types.len() {
                return Err(TypeArgError::WrongNumberTuple(items.len(), types.len()));
            }

            items
                .iter()
                .zip(types.iter())
                .try_for_each(|(arg, param)| check_type_arg(arg, param))
        }
        (TypeArg::BoundedNat { n: val }, TypeParam::BoundedNat { bound })
            if bound.valid_value(*val) =>
        {
            Ok(())
        }
        (TypeArg::String { .. }, TypeParam::String) => Ok(()),
        (TypeArg::Bytes { .. }, TypeParam::Bytes) => Ok(()),
        (TypeArg::Float { .. }, TypeParam::Float) => Ok(()),
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

/// Part of a sequence.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeqPart<T> {
    /// An individual item in the sequence.
    Item(T),
    /// A subsequence that is spliced into the parent sequence.
    Splice(T),
}

/// Iterator created by [`TypeArg::into_list_parts`].
#[derive(Debug, Clone)]
pub struct ListPartIter {
    parts: SmallVec<[SeqPart<TypeArg>; 1]>,
}

impl ListPartIter {
    #[inline]
    fn new(part: SeqPart<TypeArg>) -> Self {
        Self {
            parts: smallvec![part],
        }
    }
}

impl Iterator for ListPartIter {
    type Item = SeqPart<TypeArg>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.parts.pop()? {
                SeqPart::Splice(TypeArg::List { elems }) => self
                    .parts
                    .extend(elems.into_iter().rev().map(SeqPart::Item)),
                SeqPart::Splice(TypeArg::ListConcat { lists }) => self
                    .parts
                    .extend(lists.into_iter().rev().map(SeqPart::Splice)),
                part => return Some(part),
            }
        }
    }
}

impl FusedIterator for ListPartIter {}

/// Iterator created by [`TypeArg::into_tuple_parts`].
#[derive(Debug, Clone)]
pub struct TuplePartIter {
    parts: SmallVec<[SeqPart<TypeArg>; 1]>,
}

impl TuplePartIter {
    #[inline]
    fn new(part: SeqPart<TypeArg>) -> Self {
        Self {
            parts: smallvec![part],
        }
    }
}

impl Iterator for TuplePartIter {
    type Item = SeqPart<TypeArg>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.parts.pop()? {
                SeqPart::Splice(TypeArg::Tuple { elems }) => self
                    .parts
                    .extend(elems.into_iter().rev().map(SeqPart::Item)),
                SeqPart::Splice(TypeArg::TupleConcat { tuples }) => self
                    .parts
                    .extend(tuples.into_iter().rev().map(SeqPart::Splice)),
                part => return Some(part),
            }
        }
    }
}

impl FusedIterator for TuplePartIter {}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::{Substitution, TypeArg, TypeParam, check_type_arg};
    use crate::extension::prelude::{bool_t, usize_t};
    use crate::types::type_param::SeqPart;
    use crate::types::{TypeBound, TypeRV, type_param::TypeArgError};

    #[test]
    fn new_list_from_parts_items() {
        let a = TypeArg::new_string("a");
        let b = TypeArg::new_string("b");

        let parts = [SeqPart::Item(a.clone()), SeqPart::Item(b.clone())];
        let items = [a, b];

        assert_eq!(
            TypeArg::new_list_from_parts(parts.clone()),
            TypeArg::new_list(items.clone())
        );

        assert_eq!(
            TypeArg::new_tuple_from_parts(parts),
            TypeArg::new_tuple(items)
        );
    }

    #[test]
    fn new_list_from_parts_flatten() {
        let a = TypeArg::new_string("a");
        let b = TypeArg::new_string("b");
        let c = TypeArg::new_string("c");
        let d = TypeArg::new_string("d");
        let var = TypeArg::new_var_use(0, TypeParam::new_list(TypeParam::String));
        let parts = [
            SeqPart::Splice(TypeArg::new_list([a.clone(), b.clone()])),
            SeqPart::Splice(TypeArg::new_list_concat([TypeArg::new_list([c.clone()])])),
            SeqPart::Item(d.clone()),
            SeqPart::Splice(var.clone()),
        ];
        assert_eq!(
            TypeArg::new_list_from_parts(parts),
            TypeArg::new_list_concat([TypeArg::new_list([a, b, c, d]), var])
        );
    }

    #[test]
    fn new_tuple_from_parts_flatten() {
        let a = TypeArg::new_string("a");
        let b = TypeArg::new_string("b");
        let c = TypeArg::new_string("c");
        let d = TypeArg::new_string("d");
        let var = TypeArg::new_var_use(0, TypeParam::new_tuple([TypeParam::String]));
        let parts = [
            SeqPart::Splice(TypeArg::new_tuple([a.clone(), b.clone()])),
            SeqPart::Splice(TypeArg::new_tuple_concat([TypeArg::new_tuple([c.clone()])])),
            SeqPart::Item(d.clone()),
            SeqPart::Splice(var.clone()),
        ];
        assert_eq!(
            TypeArg::new_tuple_from_parts(parts),
            TypeArg::new_tuple_concat([TypeArg::new_tuple([a, b, c, d]), var])
        );
    }

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
        let seq_param = TypeParam::new_list(TypeBound::Copyable);
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
            &TypeParam::new_list(TypeBound::Any),
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
        check(5, &TypeParam::max_nat()).unwrap();
        check_seq(&[5], &TypeParam::max_nat()).unwrap_err();
        let list_of_nat = TypeParam::new_list(TypeParam::max_nat());
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
        let usize_and_ty = TypeParam::Tuple {
            params: vec![TypeParam::max_nat(), TypeBound::Copyable.into()],
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
        let two_types = TypeParam::Tuple {
            params: vec![TypeBound::Any.into(), TypeBound::Any.into()],
        };
        check(TypeArg::new_var_use(0, two_types.clone()), &two_types).unwrap();
        // not a Row Var which could have any number of elems
        check(TypeArg::new_var_use(0, seq_param), &two_types).unwrap_err();
    }

    #[test]
    fn type_arg_subst_row() {
        let row_param = TypeParam::new_list(TypeBound::Copyable);
        let row_arg: TypeArg = vec![bool_t().into(), TypeArg::UNIT].into();
        check_type_arg(&row_arg, &row_param).unwrap();

        // Now say a row variable referring to *that* row was used
        // to instantiate an outer "row parameter" (list of type).
        let outer_param = TypeParam::new_list(TypeBound::Any);
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
        let outer_param = TypeParam::new_list(TypeParam::new_list(TypeBound::Any));
        let row_var_decl = TypeParam::new_list(TypeBound::Copyable);
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
                param: TypeParam::new_list(TypeBound::Any)
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
                    .prop_map(|(idx, cached_decl)| Self { idx, cached_decl })
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
                    Just(Self::String).boxed(),
                    Just(Self::Bytes).boxed(),
                    Just(Self::Float).boxed(),
                    Just(Self::String).boxed(),
                    any::<TypeBound>().prop_map(|b| Self::Type { b }).boxed(),
                    any::<UpperBound>()
                        .prop_map(|bound| Self::BoundedNat { bound })
                        .boxed(),
                ]);
                if !depth.leaf() {
                    // we descend here because we these constructors contain TypeParams
                    strat = strat
                        .or(any_with::<Self>(depth.descend())
                            .prop_map(|x| Self::List { param: Box::new(x) })
                            .boxed())
                        .or(vec(any_with::<Self>(depth.descend()), 0..3)
                            .prop_map(|params| Self::Tuple { params })
                            .boxed());
                }

                strat.boxed()
            }
        }

        impl Arbitrary for TypeArg {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                use prop::collection::vec;
                use prop::strategy::Union;
                let mut strat = Union::new([
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
                    // We descend here because this constructor contains TypeArg>
                    strat = strat.or(vec(any_with::<Self>(depth.descend()), 0..3)
                        .prop_map(|elems| Self::List { elems })
                        .boxed());
                }
                strat.boxed()
            }
        }
    }
}
