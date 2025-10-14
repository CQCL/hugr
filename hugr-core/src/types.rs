//! General wire types used in the compiler

mod check;
pub mod custom;
mod poly_func;
mod row_var;
pub(crate) mod serialize;
mod signature;
pub mod type_param;
pub mod type_row;
pub(crate) use row_var::MaybeRV;
pub use row_var::{NoRV, RowVariable};

use crate::extension::resolution::{
    ExtensionCollectionError, WeakExtensionRegistry, collect_type_exts,
};
pub use crate::ops::constant::{ConstTypeError, CustomCheckFailure};
use crate::types::type_param::check_term_type;
use crate::utils::display_list_with_separator;
pub use check::SumTypeError;
pub use custom::CustomType;
pub use poly_func::{PolyFuncType, PolyFuncTypeRV};
pub use signature::{FuncTypeBase, FuncValueType, Signature};
use smol_str::SmolStr;
pub use type_param::{Term, TypeArg};
pub use type_row::{TypeRow, TypeRowRV};

pub(crate) use poly_func::PolyFuncTypeBase;

use itertools::FoldWhile::{Continue, Done};
use itertools::{Either, Itertools as _};
#[cfg(test)]
use proptest_derive::Arbitrary;
use serde::{Deserialize, Serialize};

use crate::extension::{ExtensionRegistry, ExtensionSet, SignatureError};
use crate::ops::AliasDecl;

use self::type_param::TypeParam;
use self::type_row::TypeRowBase;

/// A unique identifier for a type.
pub type TypeName = SmolStr;

/// Slice of a [`TypeName`] type identifier.
pub type TypeNameRef = str;

/// The kinds of edges in a HUGR, excluding Hierarchy.
#[derive(
    Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize, derive_more::Display,
)]
#[non_exhaustive]
pub enum EdgeKind {
    /// Control edges of a CFG region.
    ControlFlow,
    /// Data edges of a DDG region, also known as "wires".
    Value(Type),
    /// A reference to a static constant value - must be a Copyable type
    Const(Type),
    /// A reference to a function i.e. [`FuncDecl`] or [`FuncDefn`].
    ///
    /// [`FuncDecl`]: crate::ops::FuncDecl
    /// [`FuncDefn`]: crate::ops::FuncDefn
    Function(PolyFuncType),
    /// Explicitly enforce an ordering between nodes in a DDG.
    StateOrder,
}

impl EdgeKind {
    /// Returns whether the type might contain linear data.
    #[must_use]
    pub fn is_linear(&self) -> bool {
        matches!(self, EdgeKind::Value(t) if !t.copyable())
    }

    /// Whether this `EdgeKind` represents a Static edge (in the spec)
    /// - i.e. the value is statically known
    #[must_use]
    pub fn is_static(&self) -> bool {
        matches!(self, EdgeKind::Const(_) | EdgeKind::Function(_))
    }

    /// Returns `true` if the edge kind is [`ControlFlow`].
    ///
    /// [`ControlFlow`]: EdgeKind::ControlFlow
    #[must_use]
    pub fn is_control_flow(&self) -> bool {
        matches!(self, Self::ControlFlow)
    }

    /// Returns `true` if the edge kind is [`Value`].
    ///
    /// [`Value`]: EdgeKind::Value
    #[must_use]
    pub fn is_value(&self) -> bool {
        matches!(self, Self::Value(..))
    }

    /// Returns `true` if the edge kind is [`Const`].
    ///
    /// [`Const`]: EdgeKind::Const
    #[must_use]
    pub fn is_const(&self) -> bool {
        matches!(self, Self::Const(..))
    }

    /// Returns `true` if the edge kind is [`Function`].
    ///
    /// [`Function`]: EdgeKind::Function
    #[must_use]
    pub fn is_function(&self) -> bool {
        matches!(self, Self::Function(..))
    }

    /// Returns `true` if the edge kind is [`StateOrder`].
    ///
    /// [`StateOrder`]: EdgeKind::StateOrder
    #[must_use]
    pub fn is_state_order(&self) -> bool {
        matches!(self, Self::StateOrder)
    }
}

#[derive(
    Copy, Default, Clone, PartialEq, Eq, Hash, Debug, derive_more::Display, Serialize, Deserialize,
)]
#[cfg_attr(test, derive(Arbitrary))]
/// Bounds on the valid operations on a type in a HUGR program.
pub enum TypeBound {
    /// The type can be copied in the program.
    #[serde(rename = "C", alias = "E")] // alias to read in legacy Eq variants
    Copyable,
    /// No bound on the type.
    ///
    /// It cannot be copied nor discarded.
    #[serde(rename = "A")]
    #[default]
    Linear,
}

impl TypeBound {
    /// Returns the smallest `TypeTag` containing both the receiver and argument.
    /// (This will be one of the receiver or the argument.)
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        if self.contains(other) {
            self
        } else {
            debug_assert!(other.contains(self));
            other
        }
    }

    /// Report if this bound contains another.
    #[must_use]
    pub const fn contains(&self, other: TypeBound) -> bool {
        use TypeBound::{Copyable, Linear};
        matches!((self, other), (Linear, _) | (_, Copyable))
    }
}

/// Calculate the least upper bound for an iterator of bounds
pub(crate) fn least_upper_bound(mut tags: impl Iterator<Item = TypeBound>) -> TypeBound {
    tags.fold_while(TypeBound::Copyable, |acc, new| {
        if acc == TypeBound::Linear || new == TypeBound::Linear {
            Done(TypeBound::Linear)
        } else {
            Continue(acc.union(new))
        }
    })
    .into_inner()
}

#[derive(Clone, Debug, Eq, Serialize, Deserialize)]
#[serde(tag = "s")]
#[non_exhaustive]
/// Representation of a Sum type.
/// Either store the types of the variants, or in the special (but common) case
/// of a `UnitSum` (sum over empty tuples), store only the size of the Sum.
pub enum SumType {
    /// Special case of a Sum over unit types.
    #[allow(missing_docs)]
    Unit { size: u8 },
    /// General case of a Sum type.
    #[allow(missing_docs)]
    General { rows: Vec<TypeRowRV> },
}

impl std::hash::Hash for SumType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.variants().for_each(|v| v.hash(state));
    }
}

impl PartialEq for SumType {
    fn eq(&self, other: &Self) -> bool {
        self.num_variants() == other.num_variants() && self.variants().eq(other.variants())
    }
}

impl std::fmt::Display for SumType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.num_variants() == 0 {
            return write!(f, "⊥");
        }

        match self {
            SumType::Unit { size: 1 } => write!(f, "Unit"),
            SumType::Unit { size: 2 } => write!(f, "Bool"),
            SumType::Unit { size } => {
                display_list_with_separator(itertools::repeat_n("[]", *size as usize), f, "+")
            }
            SumType::General { rows } => match rows.len() {
                1 if rows[0].is_empty() => write!(f, "Unit"),
                2 if rows[0].is_empty() && rows[1].is_empty() => write!(f, "Bool"),
                _ => display_list_with_separator(rows.iter(), f, "+"),
            },
        }
    }
}

impl SumType {
    /// Initialize a new sum type.
    pub fn new<V>(variants: impl IntoIterator<Item = V>) -> Self
    where
        V: Into<TypeRowRV>,
    {
        let rows = variants.into_iter().map(Into::into).collect_vec();

        let len: usize = rows.len();
        if u8::try_from(len).is_ok() && rows.iter().all(TypeRowRV::is_empty) {
            Self::new_unary(len as u8)
        } else {
            Self::General { rows }
        }
    }

    /// New `UnitSum` with empty Tuple variants.
    #[must_use]
    pub const fn new_unary(size: u8) -> Self {
        Self::Unit { size }
    }

    /// New tuple (single row of variants).
    pub fn new_tuple(types: impl Into<TypeRow>) -> Self {
        Self::new([types.into()])
    }

    /// New option type (either an empty option, or a row of types).
    pub fn new_option(types: impl Into<TypeRow>) -> Self {
        Self::new([vec![].into(), types.into()])
    }

    /// Report the tag'th variant, if it exists.
    #[must_use]
    pub fn get_variant(&self, tag: usize) -> Option<&TypeRowRV> {
        match self {
            SumType::Unit { size } if tag < (*size as usize) => Some(TypeRV::EMPTY_TYPEROW_REF),
            SumType::General { rows } => rows.get(tag),
            _ => None,
        }
    }

    /// Returns the number of variants in the sum type.
    #[must_use]
    pub fn num_variants(&self) -> usize {
        match self {
            SumType::Unit { size } => *size as usize,
            SumType::General { rows } => rows.len(),
        }
    }

    /// Returns variant row if there is only one variant.
    #[must_use]
    pub fn as_tuple(&self) -> Option<&TypeRowRV> {
        match self {
            SumType::Unit { size } if *size == 1 => Some(TypeRV::EMPTY_TYPEROW_REF),
            SumType::General { rows } if rows.len() == 1 => Some(&rows[0]),
            _ => None,
        }
    }

    /// If the sum matches the convention of `Option[row]`, return the row.
    #[must_use]
    pub fn as_option(&self) -> Option<&TypeRowRV> {
        match self {
            SumType::Unit { size } if *size == 2 => Some(TypeRV::EMPTY_TYPEROW_REF),
            SumType::General { rows } if rows.len() == 2 && rows[0].is_empty() => Some(&rows[1]),
            _ => None,
        }
    }

    /// If a sum is an option of a single type, return the type.
    #[must_use]
    pub fn as_unary_option(&self) -> Option<&TypeRV> {
        self.as_option()
            .and_then(|row| row.iter().exactly_one().ok())
    }

    /// Returns an iterator over the variants.
    pub fn variants(&self) -> impl Iterator<Item = &TypeRowRV> {
        match self {
            SumType::Unit { size } => Either::Left(itertools::repeat_n(
                TypeRV::EMPTY_TYPEROW_REF,
                *size as usize,
            )),
            SumType::General { rows } => Either::Right(rows.iter()),
        }
    }
}

impl Transformable for SumType {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        match self {
            SumType::Unit { .. } => Ok(false),
            SumType::General { rows } => rows.transform(tr),
        }
    }
}

impl<RV: MaybeRV> From<SumType> for TypeBase<RV> {
    fn from(sum: SumType) -> Self {
        match sum {
            SumType::Unit { size } => TypeBase::new_unit_sum(size),
            SumType::General { rows } => TypeBase::new_sum(rows),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, derive_more::Display)]
/// Core types
pub enum TypeEnum<RV: MaybeRV> {
    /// An extension type.
    //
    // TODO optimise with `Box<CustomType>`?
    // or some static version of this?
    Extension(CustomType),
    /// An alias of a type.
    #[display("Alias({})", _0.name())]
    Alias(AliasDecl),
    /// A function type.
    #[display("{_0}")]
    Function(Box<FuncValueType>),
    /// A type variable, defined by an index into a list of type parameters.
    //
    // We cache the TypeBound here (checked in validation)
    #[display("#{_0}")]
    Variable(usize, TypeBound),
    /// `RowVariable`. Of course, this requires that `RV` has instances, [`NoRV`] doesn't.
    #[display("RowVar({_0})")]
    RowVar(RV),
    /// Sum of types.
    #[display("{_0}")]
    Sum(SumType),
}

impl<RV: MaybeRV> TypeEnum<RV> {
    /// The smallest type bound that covers the whole type.
    fn least_upper_bound(&self) -> TypeBound {
        match self {
            TypeEnum::Extension(c) => c.bound(),
            TypeEnum::Alias(a) => a.bound,
            TypeEnum::Function(_) => TypeBound::Copyable,
            TypeEnum::Variable(_, b) => *b,
            TypeEnum::RowVar(b) => b.bound(),
            TypeEnum::Sum(SumType::Unit { size: _ }) => TypeBound::Copyable,
            TypeEnum::Sum(SumType::General { rows }) => least_upper_bound(
                rows.iter()
                    .flat_map(TypeRowRV::iter)
                    .map(TypeRV::least_upper_bound),
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, derive_more::Display, serde::Serialize, serde::Deserialize)]
#[display("{_0}")]
#[serde(
    into = "serialize::SerSimpleType",
    try_from = "serialize::SerSimpleType"
)]
/// A HUGR type - the valid types of [`EdgeKind::Value`] and [`EdgeKind::Const`] edges.
///
/// Such an edge is valid if the ports on either end agree on the [Type].
/// Types have an optional [`TypeBound`] which places limits on the valid
/// operations on a type.
///
/// Examples:
/// ```
/// # use hugr::types::{Type, TypeBound};
/// # use hugr::type_row;
///
/// let sum = Type::new_sum([type_row![], type_row![]]);
/// assert_eq!(sum.least_upper_bound(), TypeBound::Copyable);
/// ```
///
/// ```
/// # use hugr::types::{Type, TypeBound, Signature};
///
/// let func_type: Type = Type::new_function(Signature::new_endo(vec![]));
/// assert_eq!(func_type.least_upper_bound(), TypeBound::Copyable);
/// ```
pub struct TypeBase<RV: MaybeRV>(TypeEnum<RV>, TypeBound);

/// The type of a single value, that can be sent down a wire
pub type Type = TypeBase<NoRV>;

/// One or more types - either a single type, or a row variable
/// standing for multiple types.
pub type TypeRV = TypeBase<RowVariable>;

impl<RV1: MaybeRV, RV2: MaybeRV> PartialEq<TypeEnum<RV1>> for TypeEnum<RV2> {
    fn eq(&self, other: &TypeEnum<RV1>) -> bool {
        match (self, other) {
            (TypeEnum::Extension(e1), TypeEnum::Extension(e2)) => e1 == e2,
            (TypeEnum::Alias(a1), TypeEnum::Alias(a2)) => a1 == a2,
            (TypeEnum::Function(f1), TypeEnum::Function(f2)) => f1 == f2,
            (TypeEnum::Variable(i1, b1), TypeEnum::Variable(i2, b2)) => i1 == i2 && b1 == b2,
            (TypeEnum::RowVar(v1), TypeEnum::RowVar(v2)) => v1.as_rv() == v2.as_rv(),
            (TypeEnum::Sum(s1), TypeEnum::Sum(s2)) => s1 == s2,
            _ => false,
        }
    }
}

impl<RV1: MaybeRV, RV2: MaybeRV> PartialEq<TypeBase<RV1>> for TypeBase<RV2> {
    fn eq(&self, other: &TypeBase<RV1>) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<RV: MaybeRV> TypeBase<RV> {
    /// An empty `TypeRow` or `TypeRowRV`. Provided here for convenience
    pub const EMPTY_TYPEROW: TypeRowBase<RV> = TypeRowBase::<RV>::new();
    /// Unit type (empty tuple).
    pub const UNIT: Self = Self(
        TypeEnum::Sum(SumType::Unit { size: 1 }),
        TypeBound::Copyable,
    );

    const EMPTY_TYPEROW_REF: &'static TypeRowBase<RV> = &Self::EMPTY_TYPEROW;

    /// Initialize a new function type.
    pub fn new_function(fun_ty: impl Into<FuncValueType>) -> Self {
        Self::new(TypeEnum::Function(Box::new(fun_ty.into())))
    }

    /// Initialize a new tuple type by providing the elements.
    #[inline(always)]
    pub fn new_tuple(types: impl Into<TypeRowRV>) -> Self {
        let row = types.into();
        match row.len() {
            0 => Self::UNIT,
            _ => Self::new_sum([row]),
        }
    }

    /// Initialize a new sum type by providing the possible variant types.
    #[inline(always)]
    pub fn new_sum<R>(variants: impl IntoIterator<Item = R>) -> Self
    where
        R: Into<TypeRowRV>,
    {
        Self::new(TypeEnum::Sum(SumType::new(variants)))
    }

    /// Initialize a new custom type.
    // TODO remove? Extensions/TypeDefs should just provide `Type` directly
    #[must_use]
    pub const fn new_extension(opaque: CustomType) -> Self {
        let bound = opaque.bound();
        TypeBase(TypeEnum::Extension(opaque), bound)
    }

    /// Initialize a new alias.
    #[must_use]
    pub fn new_alias(alias: AliasDecl) -> Self {
        Self::new(TypeEnum::Alias(alias))
    }

    pub(crate) fn new(type_e: TypeEnum<RV>) -> Self {
        let bound = type_e.least_upper_bound();
        Self(type_e, bound)
    }

    /// New `UnitSum` with empty Tuple variants
    #[must_use]
    pub const fn new_unit_sum(size: u8) -> Self {
        // should be the only way to avoid going through SumType::new
        Self(TypeEnum::Sum(SumType::new_unary(size)), TypeBound::Copyable)
    }

    /// New use (occurrence) of the type variable with specified index.
    /// `bound` must be exactly that with which the variable was declared
    /// (i.e. as a [`Term::RuntimeType`]`(bound)`), which may be narrower
    /// than required for the use.
    #[must_use]
    pub const fn new_var_use(idx: usize, bound: TypeBound) -> Self {
        Self(TypeEnum::Variable(idx, bound), bound)
    }

    /// Report the least upper [`TypeBound`]
    #[inline(always)]
    pub const fn least_upper_bound(&self) -> TypeBound {
        self.1
    }

    /// Report the component `TypeEnum`.
    #[inline(always)]
    pub const fn as_type_enum(&self) -> &TypeEnum<RV> {
        &self.0
    }

    /// Report a mutable reference to the component `TypeEnum`.
    #[inline(always)]
    pub fn as_type_enum_mut(&mut self) -> &mut TypeEnum<RV> {
        &mut self.0
    }

    /// Returns the inner [`SumType`] if the type is a sum.
    pub fn as_sum(&self) -> Option<&SumType> {
        match &self.0 {
            TypeEnum::Sum(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the inner [`CustomType`] if the type is from an extension.
    pub fn as_extension(&self) -> Option<&CustomType> {
        match &self.0 {
            TypeEnum::Extension(ct) => Some(ct),
            _ => None,
        }
    }

    /// Report if the type is copyable - i.e.the least upper bound of the type
    /// is contained by the copyable bound.
    pub const fn copyable(&self) -> bool {
        TypeBound::Copyable.contains(self.least_upper_bound())
    }

    /// Checks all variables used in the type are in the provided list
    /// of bound variables, rejecting any [`RowVariable`]s if `allow_row_vars` is False;
    /// and that for each [`CustomType`] the corresponding
    /// [`TypeDef`] is in the [`ExtensionRegistry`] and the type arguments
    /// [validate] and fit into the def's declared parameters.
    ///
    /// [RowVariable]: TypeEnum::RowVariable
    /// [validate]: crate::types::type_param::TypeArg::validate
    /// [TypeDef]: crate::extension::TypeDef
    pub(crate) fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        // There is no need to check the components against the bound,
        // that is guaranteed by construction (even for deserialization)
        match &self.0 {
            TypeEnum::Sum(SumType::General { rows }) => {
                rows.iter().try_for_each(|row| row.validate(var_decls))
            }
            TypeEnum::Sum(SumType::Unit { .. }) => Ok(()), // No leaves there
            TypeEnum::Alias(_) => Ok(()),
            TypeEnum::Extension(custy) => custy.validate(var_decls),
            // Function values may be passed around without knowing their arity
            // (i.e. with row vars) as long as they are not called:
            TypeEnum::Function(ft) => ft.validate(var_decls),
            TypeEnum::Variable(idx, bound) => check_typevar_decl(var_decls, *idx, &(*bound).into()),
            TypeEnum::RowVar(rv) => rv.validate(var_decls),
        }
    }

    /// Applies a substitution to a type.
    /// This may result in a row of types, if this [Type] is not really a single type but actually a row variable
    /// Invariants may be confirmed by validation:
    /// * If [`Type::validate`]`(false)` returns successfully, this method will return a Vec containing exactly one type
    /// * If [`Type::validate`]`(false)` fails, but `(true)` succeeds, this method may (depending on structure of self)
    ///   return a Vec containing any number of [Type]s. These may (or not) pass [`Type::validate`]
    fn substitute(&self, t: &Substitution) -> Vec<Self> {
        match &self.0 {
            TypeEnum::RowVar(rv) => rv.substitute(t),
            TypeEnum::Alias(_) | TypeEnum::Sum(SumType::Unit { .. }) => vec![self.clone()],
            TypeEnum::Variable(idx, bound) => {
                let TypeArg::Runtime(ty) = t.apply_var(*idx, &((*bound).into())) else {
                    panic!("Variable was not a type - try validate() first")
                };
                vec![ty.into_()]
            }
            TypeEnum::Extension(cty) => vec![TypeBase::new_extension(cty.substitute(t))],
            TypeEnum::Function(bf) => vec![TypeBase::new_function(bf.substitute(t))],
            TypeEnum::Sum(SumType::General { rows }) => {
                vec![TypeBase::new_sum(rows.iter().map(|r| r.substitute(t)))]
            }
        }
    }

    /// Returns a registry with the concrete extensions used by this type.
    ///
    /// This includes the extensions of custom types that may be nested
    /// inside other types.
    pub fn used_extensions(&self) -> Result<ExtensionRegistry, ExtensionCollectionError> {
        let mut used = WeakExtensionRegistry::default();
        let mut missing = ExtensionSet::new();

        collect_type_exts(self, &mut used, &mut missing);

        if missing.is_empty() {
            Ok(used.try_into().expect("all extensions are present"))
        } else {
            Err(ExtensionCollectionError::dropped_type(self, missing))
        }
    }
}

impl<RV: MaybeRV> Transformable for TypeBase<RV> {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        match &mut self.0 {
            TypeEnum::Alias(_) | TypeEnum::RowVar(_) | TypeEnum::Variable(..) => Ok(false),
            TypeEnum::Extension(custom_type) => {
                if let Some(nt) = tr.apply_custom(custom_type)? {
                    *self = nt.into_();
                    Ok(true)
                } else {
                    let args_changed = custom_type.args_mut().transform(tr)?;
                    if args_changed {
                        *self = Self::new_extension(
                            custom_type
                                .get_type_def(&custom_type.get_extension()?)?
                                .instantiate(custom_type.args())?,
                        );
                    }
                    Ok(args_changed)
                }
            }
            TypeEnum::Function(fty) => fty.transform(tr),
            TypeEnum::Sum(sum_type) => {
                let ch = sum_type.transform(tr)?;
                self.1 = self.0.least_upper_bound();
                Ok(ch)
            }
        }
    }
}

impl Type {
    fn substitute1(&self, s: &Substitution) -> Self {
        let v = self.substitute(s);
        let [r] = v.try_into().unwrap(); // No row vars, so every Type<false> produces exactly one
        r
    }
}

impl TypeRV {
    /// Tells if this Type is a row variable, i.e. could stand for any number >=0 of Types
    #[must_use]
    pub fn is_row_var(&self) -> bool {
        matches!(self.0, TypeEnum::RowVar(_))
    }

    /// New use (occurrence) of the row variable with specified index.
    /// `bound` must match that with which the variable was declared
    /// (i.e. as a list of runtime types of that bound).
    /// For use in [OpDef], not [FuncDefn], type schemes only.
    ///
    /// [OpDef]: crate::extension::OpDef
    /// [FuncDefn]: crate::ops::FuncDefn
    #[must_use]
    pub const fn new_row_var_use(idx: usize, bound: TypeBound) -> Self {
        Self(TypeEnum::RowVar(RowVariable(idx, bound)), bound)
    }
}

// ====== Conversions ======
impl<RV: MaybeRV> TypeBase<RV> {
    /// (Fallibly) converts a `TypeBase` (parameterized, so may or may not be able
    /// to contain [`RowVariable`]s) into a [Type] that definitely does not.
    pub fn try_into_type(self) -> Result<Type, RowVariable> {
        Ok(TypeBase(
            match self.0 {
                TypeEnum::Extension(e) => TypeEnum::Extension(e),
                TypeEnum::Alias(a) => TypeEnum::Alias(a),
                TypeEnum::Function(f) => TypeEnum::Function(f),
                TypeEnum::Variable(idx, bound) => TypeEnum::Variable(idx, bound),
                TypeEnum::RowVar(rv) => Err(rv.as_rv().clone())?,
                TypeEnum::Sum(s) => TypeEnum::Sum(s),
            },
            self.1,
        ))
    }
}

impl TryFrom<TypeRV> for Type {
    type Error = RowVariable;
    fn try_from(value: TypeRV) -> Result<Self, RowVariable> {
        value.try_into_type()
    }
}

impl<RV1: MaybeRV> TypeBase<RV1> {
    /// A swiss-army-knife for any safe conversion of the type argument `RV1`
    /// to/from [`NoRV`]/RowVariable/rust-type-variable.
    fn into_<RV2: MaybeRV>(self) -> TypeBase<RV2>
    where
        RV1: Into<RV2>,
    {
        TypeBase(
            match self.0 {
                TypeEnum::Extension(e) => TypeEnum::Extension(e),
                TypeEnum::Alias(a) => TypeEnum::Alias(a),
                TypeEnum::Function(f) => TypeEnum::Function(f),
                TypeEnum::Variable(idx, bound) => TypeEnum::Variable(idx, bound),
                TypeEnum::RowVar(rv) => TypeEnum::RowVar(rv.into()),
                TypeEnum::Sum(s) => TypeEnum::Sum(s),
            },
            self.1,
        )
    }
}

impl From<Type> for TypeRV {
    fn from(value: Type) -> Self {
        value.into_()
    }
}

/// Details a replacement of type variables with a finite list of known values.
/// (Variables out of the range of the list will result in a panic)
#[derive(Clone, Debug, derive_more::Display)]
#[display("[{}]", _0.iter().map(std::string::ToString::to_string).join(", "))]
pub struct Substitution<'a>(&'a [TypeArg]);

impl<'a> Substitution<'a> {
    /// Create a new Substitution given the replacement values (indexed
    /// as the variables they replace). `exts` must contain the [`TypeDef`]
    /// for every custom [Type] (to which the Substitution is applied)
    /// containing a type-variable.
    ///
    /// [`TypeDef`]: crate::extension::TypeDef
    #[must_use]
    pub fn new(items: &'a [TypeArg]) -> Self {
        Self(items)
    }

    pub(crate) fn apply_var(&self, idx: usize, decl: &TypeParam) -> TypeArg {
        let arg = self
            .0
            .get(idx)
            .expect("Undeclared type variable - call validate() ?");
        debug_assert_eq!(check_term_type(arg, decl), Ok(()));
        arg.clone()
    }

    fn apply_rowvar(&self, idx: usize, bound: TypeBound) -> Vec<TypeRV> {
        let arg = self
            .0
            .get(idx)
            .expect("Undeclared type variable - call validate() ?");
        debug_assert!(check_term_type(arg, &TypeParam::new_list_type(bound)).is_ok());
        match arg {
            TypeArg::List(elems) => elems
                .iter()
                .map(|ta| {
                    match ta {
                        Term::Runtime(ty) => return ty.clone().into(),
                        Term::Variable(v) => {
                            if let Some(b) = v.bound_if_row_var() {
                                return TypeRV::new_row_var_use(v.index(), b);
                            }
                        }
                        _ => (),
                    }
                    panic!("Not a list of types - call validate() ?")
                })
                .collect(),
            Term::Runtime(ty) if matches!(ty.0, TypeEnum::RowVar(_)) => {
                // Standalone "Type" can be used iff its actually a Row Variable not an actual (single) Type
                vec![ty.clone().into()]
            }
            _ => panic!("Not a type or list of types - call validate() ?"),
        }
    }
}

/// A transformation that can be applied to a [Type] or [`TypeArg`].
///
/// More general in some ways than a Substitution: can fail with a
/// [`Self::Err`],  may change [`TypeBound::Copyable`] to [`TypeBound::Linear`],
/// and applies to arbitrary extension types rather than type variables.
pub trait TypeTransformer {
    /// Error returned when a [`CustomType`] cannot be transformed, or a type
    /// containing it (e.g. if changing a runtime type from copyable to
    /// linear invalidates a parameterized type).
    type Err: std::error::Error + From<SignatureError>;

    /// Applies the transformation to an extension type.
    ///
    /// Note that if the [`CustomType`] has type arguments, these will *not*
    /// have been transformed first (this might not produce a valid type
    /// due to changes in [`TypeBound`]).
    ///
    /// Returns a type to use instead, or None to indicate no change
    ///   (in which case, the `TypeArgs` will be transformed instead.
    ///    To prevent transforming the arguments, return `t.clone().into()`.)
    fn apply_custom(&self, t: &CustomType) -> Result<Option<Type>, Self::Err>;

    // Note: in future releases more methods may be added here to transform other types.
    // By defaulting such trait methods to Ok(None), backwards compatibility will be preserved.
}

/// Trait for things that can be transformed by applying a [`TypeTransformer`].
/// (A destructive / in-place mutation.)
pub trait Transformable {
    /// Applies a [`TypeTransformer`] to this instance.
    ///
    /// Returns true if any part may have changed, or false for definitely no change.
    ///
    /// If an Err occurs, `self` may be left in an inconsistent state (e.g. partially
    /// transformed).
    fn transform<T: TypeTransformer>(&mut self, t: &T) -> Result<bool, T::Err>;
}

impl<E: Transformable> Transformable for [E] {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        let mut any_change = false;
        for item in self {
            any_change |= item.transform(tr)?;
        }
        Ok(any_change)
    }
}

pub(crate) fn check_typevar_decl(
    decls: &[TypeParam],
    idx: usize,
    cached_decl: &TypeParam,
) -> Result<(), SignatureError> {
    match decls.get(idx) {
        None => Err(SignatureError::FreeTypeVar {
            idx,
            num_decls: decls.len(),
        }),
        Some(actual) => {
            // The cache here just mirrors the declaration. The typevar can be used
            // anywhere expecting a kind *containing* the decl - see `check_type_arg`.
            if actual == cached_decl {
                Ok(())
            } else {
                Err(SignatureError::TypeVarDoesNotMatchDeclaration {
                    cached: Box::new(cached_decl.clone()),
                    actual: Box::new(actual.clone()),
                })
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::hash::{Hash, Hasher};
    use std::sync::Weak;

    use super::*;
    use crate::extension::TypeDefBound;
    use crate::extension::prelude::{option_type, qb_t, usize_t};
    use crate::std_extensions::collections::array::{array_type, array_type_parametric};
    use crate::std_extensions::collections::list::list_type;
    use crate::types::type_param::TermTypeError;
    use crate::{Extension, hugr::IdentList, type_row};

    #[test]
    fn construct() {
        let t: Type = Type::new_tuple(vec![
            usize_t(),
            Type::new_function(Signature::new_endo(vec![])),
            Type::new_extension(CustomType::new(
                "my_custom",
                [],
                "my_extension".try_into().unwrap(),
                TypeBound::Copyable,
                // Dummy extension reference.
                &Weak::default(),
            )),
            Type::new_alias(AliasDecl::new("my_alias", TypeBound::Copyable)),
        ]);
        assert_eq!(
            &t.to_string(),
            "[usize, [] -> [], my_custom, Alias(my_alias)]"
        );
    }

    #[rstest::rstest]
    fn sum_construct() {
        let pred1 = Type::new_sum([type_row![], type_row![]]);
        let pred2 = TypeRV::new_unit_sum(2);

        assert_eq!(pred1, pred2);

        let pred_direct = SumType::Unit { size: 2 };
        assert_eq!(pred1, Type::from(pred_direct));
    }

    #[test]
    fn as_sum() {
        let t = Type::new_unit_sum(0);
        assert!(t.as_sum().is_some());
    }

    #[test]
    fn as_option() {
        let opt = option_type(usize_t());

        assert_eq!(opt.as_unary_option().unwrap().clone(), usize_t());
        assert_eq!(
            Type::new_unit_sum(2).as_sum().unwrap().as_unary_option(),
            None
        );

        assert_eq!(
            Type::new_tuple(vec![usize_t()])
                .as_sum()
                .unwrap()
                .as_option(),
            None
        );
    }

    #[test]
    fn as_extension() {
        assert_eq!(
            Type::new_extension(usize_t().as_extension().unwrap().clone()),
            usize_t()
        );
        assert_eq!(Type::new_unit_sum(0).as_extension(), None);
    }

    #[test]
    fn sum_variants() {
        let variants: Vec<TypeRowRV> = vec![
            TypeRV::UNIT.into(),
            vec![TypeRV::new_row_var_use(0, TypeBound::Linear)].into(),
        ];
        let t = SumType::new(variants.clone());
        assert_eq!(variants, t.variants().cloned().collect_vec());

        let empty_rows = vec![TypeRV::EMPTY_TYPEROW; 3];
        let sum_unary = SumType::new_unary(3);
        let sum_general = SumType::General {
            rows: empty_rows.clone(),
        };
        assert_eq!(&empty_rows, &sum_unary.variants().cloned().collect_vec());
        assert_eq!(sum_general, sum_unary);

        let mut hasher_general = std::hash::DefaultHasher::new();
        sum_general.hash(&mut hasher_general);
        let mut hasher_unary = std::hash::DefaultHasher::new();
        sum_unary.hash(&mut hasher_unary);
        assert_eq!(hasher_general.finish(), hasher_unary.finish());
    }

    pub(super) struct FnTransformer<T>(pub(super) T);
    impl<T: Fn(&CustomType) -> Option<Type>> TypeTransformer for FnTransformer<T> {
        type Err = SignatureError;

        fn apply_custom(&self, t: &CustomType) -> Result<Option<Type>, Self::Err> {
            Ok((self.0)(t))
        }
    }
    #[test]
    fn transform() {
        const LIN: SmolStr = SmolStr::new_inline("MyLinear");
        let e = Extension::new_test_arc(IdentList::new("TestExt").unwrap(), |e, w| {
            e.add_type(LIN, vec![], String::new(), TypeDefBound::any(), w)
                .unwrap();
        });
        let lin = e.get_type(&LIN).unwrap().instantiate([]).unwrap();

        let lin_to_usize = FnTransformer(|ct: &CustomType| (*ct == lin).then_some(usize_t()));
        let mut t = Type::new_extension(lin.clone());
        assert_eq!(t.transform(&lin_to_usize), Ok(true));
        assert_eq!(t, usize_t());

        for coln in [
            list_type,
            |t| array_type(10, t),
            |t| {
                array_type_parametric(
                    TypeArg::new_var_use(0, TypeParam::bounded_nat_type(3.try_into().unwrap())),
                    t,
                )
                .unwrap()
            },
        ] {
            let mut t = coln(lin.clone().into());
            assert_eq!(t.transform(&lin_to_usize), Ok(true));
            let expected = coln(usize_t());
            assert_eq!(t, expected);
            assert_eq!(t.transform(&lin_to_usize), Ok(false));
            assert_eq!(t, expected);
        }
    }

    #[test]
    fn transform_copyable_to_linear() {
        const CPY: SmolStr = SmolStr::new_inline("MyCopyable");
        const COLN: SmolStr = SmolStr::new_inline("ColnOfCopyableElems");
        let e = Extension::new_test_arc(IdentList::new("TestExt").unwrap(), |e, w| {
            e.add_type(CPY, vec![], String::new(), TypeDefBound::copyable(), w)
                .unwrap();
            e.add_type(
                COLN,
                vec![TypeParam::new_list_type(TypeBound::Copyable)],
                String::new(),
                TypeDefBound::copyable(),
                w,
            )
            .unwrap();
        });

        let cpy = e.get_type(&CPY).unwrap().instantiate([]).unwrap();
        let mk_opt = |t: Type| Type::new_sum([type_row![], TypeRow::from(t)]);

        let cpy_to_qb = FnTransformer(|ct: &CustomType| (ct == &cpy).then_some(qb_t()));

        let mut t = mk_opt(cpy.clone().into());
        assert_eq!(t.transform(&cpy_to_qb), Ok(true));
        assert_eq!(t, mk_opt(qb_t()));

        let coln = e.get_type(&COLN).unwrap();
        let c_of_cpy = coln
            .instantiate([Term::new_list([Type::from(cpy.clone()).into()])])
            .unwrap();

        let mut t = Type::new_extension(c_of_cpy.clone());
        assert_eq!(
            t.transform(&cpy_to_qb),
            Err(SignatureError::from(TermTypeError::TypeMismatch {
                type_: Box::new(TypeBound::Copyable.into()),
                term: Box::new(qb_t().into())
            }))
        );

        let mut t = Type::new_extension(
            coln.instantiate([Term::new_list([mk_opt(Type::from(cpy.clone())).into()])])
                .unwrap(),
        );
        assert_eq!(
            t.transform(&cpy_to_qb),
            Err(SignatureError::from(TermTypeError::TypeMismatch {
                type_: Box::new(TypeBound::Copyable.into()),
                term: Box::new(mk_opt(qb_t()).into())
            }))
        );

        // Finally, check handling Coln<Cpy> overrides handling of Cpy
        let cpy_to_qb2 = FnTransformer(|ct: &CustomType| {
            assert_ne!(ct, &cpy);
            (ct == &c_of_cpy).then_some(usize_t())
        });
        let mut t = Type::new_extension(
            coln.instantiate([Term::new_list(vec![Type::from(c_of_cpy.clone()).into(); 2])])
                .unwrap(),
        );
        assert_eq!(t.transform(&cpy_to_qb2), Ok(true));
        assert_eq!(
            t,
            Type::new_extension(
                coln.instantiate([Term::new_list([usize_t().into(), usize_t().into()])])
                    .unwrap()
            )
        );
    }

    mod proptest {

        use crate::proptest::RecursionDepth;

        use super::{AliasDecl, MaybeRV, TypeBase, TypeBound, TypeEnum};
        use crate::types::{CustomType, FuncValueType, SumType, TypeRowRV};
        use proptest::prelude::*;

        impl Arbitrary for super::SumType {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                use proptest::collection::vec;
                if depth.leaf() {
                    any::<u8>().prop_map(Self::new_unary).boxed()
                } else {
                    vec(any_with::<TypeRowRV>(depth), 0..3)
                        .prop_map(SumType::new)
                        .boxed()
                }
            }
        }

        impl<RV: MaybeRV> Arbitrary for TypeBase<RV> {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                // We descend here, because a TypeEnum may contain a Type
                let depth = depth.descend();
                prop_oneof![
                    1 => any::<AliasDecl>().prop_map(TypeBase::new_alias),
                    1 => any_with::<CustomType>(depth.into()).prop_map(TypeBase::new_extension),
                    1 => any_with::<FuncValueType>(depth).prop_map(TypeBase::new_function),
                    1 => any_with::<SumType>(depth).prop_map(TypeBase::from),
                    1 => (any::<usize>(), any::<TypeBound>()).prop_map(|(i,b)| TypeBase::new_var_use(i,b)),
                    // proptest_derive::Arbitrary's weight attribute requires a constant,
                    // rather than this expression, hence the manual impl:
                    RV::weight() => RV::arb().prop_map(|rv| TypeBase::new(TypeEnum::RowVar(rv)))
                ]
                    .boxed()
            }
        }
    }
}

#[cfg(test)]
pub(super) mod proptest_utils {
    use proptest::collection::vec;
    use proptest::prelude::{Strategy, any_with};

    use super::serialize::{TermSer, TypeArgSer, TypeParamSer};
    use super::type_param::Term;

    use crate::proptest::RecursionDepth;
    use crate::types::serialize::ArrayOrTermSer;

    fn term_is_serde_type_arg(t: &Term) -> bool {
        let TermSer::TypeArg(arg) = TermSer::from(t.clone()) else {
            return false;
        };
        match arg {
            TypeArgSer::List { elems: terms }
            | TypeArgSer::ListConcat { lists: terms }
            | TypeArgSer::Tuple { elems: terms }
            | TypeArgSer::TupleConcat { tuples: terms } => terms.iter().all(term_is_serde_type_arg),
            TypeArgSer::Variable { v } => term_is_serde_type_param(&v.cached_decl),
            TypeArgSer::Type { ty } => {
                if let Some(cty) = ty.as_extension() {
                    cty.args().iter().all(term_is_serde_type_arg)
                } else {
                    true
                }
            } // Do we need to inspect inside function types? sum types?
            TypeArgSer::BoundedNat { .. }
            | TypeArgSer::String { .. }
            | TypeArgSer::Bytes { .. }
            | TypeArgSer::Float { .. } => true,
        }
    }

    fn term_is_serde_type_param(t: &Term) -> bool {
        let TermSer::TypeParam(parm) = TermSer::from(t.clone()) else {
            return false;
        };
        match parm {
            TypeParamSer::Type { .. }
            | TypeParamSer::BoundedNat { .. }
            | TypeParamSer::String
            | TypeParamSer::Bytes
            | TypeParamSer::Float
            | TypeParamSer::StaticType
            | TypeParamSer::ConstType { .. } => true,
            TypeParamSer::List { param } => term_is_serde_type_param(&param),
            TypeParamSer::Tuple { params } => {
                match &params {
                    ArrayOrTermSer::Array(terms) => terms.iter().all(term_is_serde_type_param),
                    ArrayOrTermSer::Term(b) => match &**b {
                        Term::List(_) => panic!("Should be represented as ArrayOrTermSer::Array"),
                        // This might be well-typed, but does not fit the (TODO: update) JSON schema
                        Term::Variable(_) => false,
                        // Similarly, but not produced by our `impl Arbitrary`:
                        Term::ListConcat(_) => todo!("Update schema"),

                        // The others do not fit the JSON schema, and are not well-typed,
                        // but can be produced by our impl of Arbitrary, so we must filter out:
                        _ => false,
                    },
                }
            }
        }
    }

    pub fn any_serde_type_arg(depth: RecursionDepth) -> impl Strategy<Value = Term> {
        any_with::<Term>(depth).prop_filter("Term was not a TypeArg", term_is_serde_type_arg)
    }

    pub fn any_serde_type_arg_vec() -> impl Strategy<Value = Vec<Term>> {
        vec(any_serde_type_arg(RecursionDepth::default()), 1..3)
    }

    pub fn any_serde_type_param(depth: RecursionDepth) -> impl Strategy<Value = Term> {
        any_with::<Term>(depth).prop_filter("Term was not a TypeParam", term_is_serde_type_param)
    }
}
