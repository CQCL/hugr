//! General wire types used in the compiler

mod check;
pub mod custom;
mod poly_func;
mod serialize;
mod signature;
pub mod type_param;
pub mod type_row;

pub use crate::ops::constant::{ConstTypeError, CustomCheckFailure};
use crate::types::type_param::check_type_arg;
use crate::utils::display_list_with_separator;
pub use check::SumTypeError;
pub use custom::CustomType;
pub use poly_func::{TypeScheme, TypeSchemeRV};
pub use signature::{FunctionType, FunctionTypeRV};
use smol_str::SmolStr;
pub use type_param::TypeArg;
pub use type_row::{TypeRow, TypeRowRV};

use itertools::FoldWhile::{Continue, Done};
use itertools::{repeat_n, Itertools};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use {crate::proptest::RecursionDepth, ::proptest::prelude::*, proptest_derive::Arbitrary};

use crate::extension::{ExtensionRegistry, SignatureError};
use crate::ops::AliasDecl;

use self::type_param::TypeParam;
use self::type_row::TypeRowBase;

/// A unique identifier for a type.
pub type TypeName = SmolStr;

/// Slice of a [`TypeName`] type identifier.
pub type TypeNameRef = str;

/// The kinds of edges in a HUGR, excluding Hierarchy.
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum EdgeKind {
    /// Control edges of a CFG region.
    ControlFlow,
    /// Data edges of a DDG region, also known as "wires".
    Value(Type),
    /// A reference to a static constant value - must be a Copyable type
    Const(Type),
    /// A reference to a function i.e. [FuncDecl] or [FuncDefn]
    ///
    /// [FuncDecl]: crate::ops::FuncDecl
    /// [FuncDefn]: crate::ops::FuncDefn
    Function(TypeScheme),
    /// Explicitly enforce an ordering between nodes in a DDG.
    StateOrder,
}

impl EdgeKind {
    /// Returns whether the type might contain linear data.
    pub fn is_linear(&self) -> bool {
        matches!(self, EdgeKind::Value(t) if !t.copyable())
    }

    /// Whether this EdgeKind represents a Static edge (in the spec)
    /// - i.e. the value is statically known
    pub fn is_static(&self) -> bool {
        matches!(self, EdgeKind::Const(_) | EdgeKind::Function(_))
    }
}

#[derive(
    Copy, Default, Clone, PartialEq, Eq, Hash, Debug, derive_more::Display, Serialize, Deserialize,
)]
#[cfg_attr(test, derive(Arbitrary))]
/// Bounds on the valid operations on a type in a HUGR program.
pub enum TypeBound {
    /// The equality operation is valid on this type.
    #[serde(rename = "E")]
    Eq,
    /// The type can be copied in the program.
    #[serde(rename = "C")]
    Copyable,
    /// No bound on the type.
    #[serde(rename = "A")]
    #[default]
    Any,
}

impl TypeBound {
    /// Returns the smallest TypeTag containing both the receiver and argument.
    /// (This will be one of the receiver or the argument.)
    pub fn union(self, other: Self) -> Self {
        if self.contains(other) {
            self
        } else {
            debug_assert!(other.contains(self));
            other
        }
    }

    /// Report if this bound contains another.
    pub const fn contains(&self, other: TypeBound) -> bool {
        use TypeBound::*;
        matches!((self, other), (Any, _) | (_, Eq) | (Copyable, Copyable))
    }
}

/// Calculate the least upper bound for an iterator of bounds
pub(crate) fn least_upper_bound(mut tags: impl Iterator<Item = TypeBound>) -> TypeBound {
    tags.fold_while(TypeBound::Eq, |acc, new| {
        if acc == TypeBound::Any || new == TypeBound::Any {
            Done(TypeBound::Any)
        } else {
            Continue(acc.union(new))
        }
    })
    .into_inner()
}

#[derive(Clone, PartialEq, Debug, Eq, Serialize, Deserialize)]
#[serde(tag = "s")]
#[non_exhaustive]
/// Representation of a Sum type.
/// Either store the types of the variants, or in the special (but common) case
/// of a UnitSum (sum over empty tuples), store only the size of the Sum.
pub enum SumType {
    /// Special case of a Sum over unit types.
    #[allow(missing_docs)]
    Unit { size: u8 },
    /// General case of a Sum type.
    #[allow(missing_docs)]
    General { rows: Vec<TypeRowRV> },
}

impl std::fmt::Display for SumType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.num_variants() == 0 {
            return write!(f, "⊥");
        }

        match self {
            SumType::Unit { size } => {
                display_list_with_separator(repeat_n("[]", *size as usize), f, "+")
            }
            SumType::General { rows } => display_list_with_separator(rows.iter(), f, "+"),
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
        if len <= (u8::MAX as usize) && rows.iter().all(TypeRowRV::is_empty) {
            Self::new_unary(len as u8)
        } else {
            Self::General { rows }
        }
    }

    /// New UnitSum with empty Tuple variants
    pub const fn new_unary(size: u8) -> Self {
        Self::Unit { size }
    }

    /// Report the tag'th variant, if it exists.
    pub fn get_variant(&self, tag: usize) -> Option<&TypeRowRV> {
        match self {
            SumType::Unit { size } if tag < (*size as usize) => Some(TypeRV::EMPTY_TYPEROW_REF),
            SumType::General { rows } => rows.get(tag),
            _ => None,
        }
    }

    /// Returns the number of variants in the sum type.
    pub fn num_variants(&self) -> usize {
        match self {
            SumType::Unit { size } => *size as usize,
            SumType::General { rows } => rows.len(),
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

/// Variable index, and cache of inner TypeBound - matches a [TypeParam::List] of [TypeParam::Type]
/// of this bound (checked in validation)
// Note that I believe the serde derives here are not used except as markers
// so that other types containing this can also #derive-serde the same way.
#[derive(
    Clone, Debug, Eq, PartialEq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[display(fmt = "{}", "_0")]
pub struct RowVariable(usize, TypeBound);

trait MaybeRV: Clone + std::fmt::Debug + std::fmt::Display + From<NoRV> + Eq + PartialEq + 'static {
    fn as_rv(&self) -> &RowVariable;
    fn try_from_rv(rv: RowVariable) -> Result<Self, SignatureError>;
    fn bound(&self) -> TypeBound;
    fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError>;
    fn substitute(&self, s: &Substitution) -> Vec<TypeBase<Self>>;
}

// Note that I believe the serde derives here are not used except as markers
// so that other types containing this can also #derive-serde the same way.
#[derive(
    Clone, Debug, Eq, PartialEq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
pub enum NoRV {}

impl From<NoRV> for RowVariable {
    fn from(value: NoRV) -> Self {
        match value { }
    }
}

impl MaybeRV for RowVariable {
    fn as_rv(&self) -> &RowVariable {
        self
    }

    fn bound(&self) -> TypeBound {
        self.1
    }

    fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        check_typevar_decl(var_decls, self.0, &TypeParam::new_list(self.1))
    }

    fn substitute(&self, s: &Substitution) -> Vec<TypeBase<Self>> {
        s.apply_rowvar(self.0, self.1)
    }

    fn try_from_rv(rv: RowVariable) -> Result<Self, SignatureError> {
        Ok(rv)
    }
}

impl MaybeRV for NoRV {
    fn as_rv(&self) -> &RowVariable {
        match *self {}
    }

    fn bound(&self) -> TypeBound {
        match *self {}
    }

    fn validate(&self, _var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        match *self {}
    }

    fn substitute(&self, _s: &Substitution) -> Vec<TypeBase<Self>> {
        match *self {}
    }

    fn try_from_rv(rv: RowVariable) -> Result<Self, SignatureError> {
        Err(SignatureError::RowVarWhereTypeExpected { idx: rv.0 })
    }
}

#[derive(Clone, Debug, Eq, derive_more::Display)]
#[cfg_attr(test, derive(Arbitrary), proptest(params = "RecursionDepth"))]
/// Core types
pub enum TypeEnum<RV: MaybeRV> {
    // TODO optimise with Box<CustomType> ?
    // or some static version of this?
    #[allow(missing_docs)]
    Extension(
        #[cfg_attr(test, proptest(strategy = "any_with::<CustomType>(params.into())"))] CustomType,
    ),
    #[allow(missing_docs)]
    #[display(fmt = "Alias({})", "_0.name()")]
    Alias(AliasDecl),
    #[allow(missing_docs)]
    #[display(fmt = "Function({})", "_0")]
    Function(
        #[cfg_attr(
            test,
            proptest(strategy = "any_with::<FunctionTypeRV>(params).prop_map(Box::new)")
        )]
        Box<FunctionTypeRV>,
    ),
    // Index into TypeParams, and cache of TypeBound (checked in validation)
    #[allow(missing_docs)]
    #[display(fmt = "Variable({})", _0)]
    Variable(usize, TypeBound),
    /// RowVariable. Of course, this requires that `RV` has instances, [NoRV] doesn't.
    #[display(fmt = "RowVar({})", _0)]
    RowVar(RV),
    #[allow(missing_docs)]
    #[display(fmt = "{}", "_0")]
    Sum(#[cfg_attr(test, proptest(strategy = "any_with::<SumType>(params)"))] SumType),
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
            TypeEnum::Sum(SumType::Unit { size: _ }) => TypeBound::Eq,
            TypeEnum::Sum(SumType::General { rows }) => least_upper_bound(
                rows.iter()
                    .flat_map(TypeRowRV::iter)
                    .map(TypeRV::least_upper_bound),
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize)]
#[display(fmt = "{}", "_0")]
#[serde(
    into = "serialize::SerSimpleType",
    try_from = "serialize::SerSimpleType"
)]
/// A HUGR type - the valid types of [EdgeKind::Value] and [EdgeKind::Const] edges.
/// Such an edge is valid if the ports on either end agree on the [Type].
/// Types have an optional [TypeBound] which places limits on the valid
/// operations on a type.
///
/// Examples:
/// ```
/// # use hugr::types::{Type, TypeBound};
/// # use hugr::type_row;
///
/// let sum: Type = Type::new_sum([type_row![], type_row![]]);
/// assert_eq!(sum.least_upper_bound(), TypeBound::Eq);
/// ```
///
/// ```
/// # use hugr::types::{Type, TypeBound, FunctionType};
///
/// let func_type: Type = Type::new_function(FunctionType::new_endo(vec![]));
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
    pub const UNIT: Self = Self(TypeEnum::Sum(SumType::Unit { size: 1 }), TypeBound::Eq);

    const EMPTY_TYPEROW_REF: &'static TypeRowBase<RV> = &Self::EMPTY_TYPEROW;

    /// Initialize a new function type.
    pub fn new_function(fun_ty: impl Into<FunctionTypeRV>) -> Self {
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
    pub const fn new_extension(opaque: CustomType) -> Self {
        let bound = opaque.bound();
        TypeBase(TypeEnum::Extension(opaque), bound)
    }

    /// Initialize a new alias.
    pub fn new_alias(alias: AliasDecl) -> Self {
        Self::new(TypeEnum::Alias(alias))
    }

    fn new(type_e: TypeEnum<RV>) -> Self {
        let bound = type_e.least_upper_bound();
        Self(type_e, bound)
    }

    /// New UnitSum with empty Tuple variants
    pub const fn new_unit_sum(size: u8) -> Self {
        // should be the only way to avoid going through SumType::new
        Self(TypeEnum::Sum(SumType::new_unary(size)), TypeBound::Eq)
    }

    /// New use (occurrence) of the type variable with specified index.
    /// `bound` must be exactly that with which the variable was declared
    /// (i.e. as a [TypeParam::Type]`(bound)`), which may be narrower
    /// than required for the use.
    pub const fn new_var_use(idx: usize, bound: TypeBound) -> Self {
        Self(TypeEnum::Variable(idx, bound), bound)
    }

    /// Report the least upper [TypeBound]
    #[inline(always)]
    pub const fn least_upper_bound(&self) -> TypeBound {
        self.1
    }

    /// Report the component TypeEnum.
    #[inline(always)]
    pub const fn as_type_enum(&self) -> &TypeEnum<RV> {
        &self.0
    }

    /// Report if the type is copyable - i.e.the least upper bound of the type
    /// is contained by the copyable bound.
    pub const fn copyable(&self) -> bool {
        TypeBound::Copyable.contains(self.least_upper_bound())
    }

    /// Checks all variables used in the type are in the provided list
    /// of bound variables, rejecting any [RowVariable]s if `allow_row_vars` is False;
    /// and that for each [CustomType] the corresponding
    /// [TypeDef] is in the [ExtensionRegistry] and the type arguments
    /// [validate] and fit into the def's declared parameters.
    ///
    /// [RowVariable]: TypeEnum::RowVariable
    /// [validate]: crate::types::type_param::TypeArg::validate
    /// [TypeDef]: crate::extension::TypeDef
    pub(crate) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        // There is no need to check the components against the bound,
        // that is guaranteed by construction (even for deserialization)
        match &self.0 {
            TypeEnum::Sum(SumType::General { rows }) => rows
                .iter()
                .try_for_each(|row| row.validate(extension_registry, var_decls)),
            TypeEnum::Sum(SumType::Unit { .. }) => Ok(()), // No leaves there
            TypeEnum::Alias(_) => Ok(()),
            TypeEnum::Extension(custy) => custy.validate(extension_registry, var_decls),
            // Function values may be passed around without knowing their arity
            // (i.e. with row vars) as long as they are not called:
            TypeEnum::Function(ft) => ft.validate(extension_registry, var_decls),
            TypeEnum::Variable(idx, bound) => check_typevar_decl(var_decls, *idx, &(*bound).into()),
            TypeEnum::RowVar(rv) => rv.validate(var_decls),
        }
    }

    /// Applies a substitution to a type.
    /// This may result in a row of types, if this [Type] is not really a single type but actually a row variable
    /// Invariants may be confirmed by validation:
    /// * If [Type::validate]`(false)` returns successfully, this method will return a Vec containing exactly one type
    /// * If [Type::validate]`(false)` fails, but `(true)` succeeds, this method may (depending on structure of self)
    ///   return a Vec containing any number of [Type]s. These may (or not) pass [Type::validate]
    fn substitute(&self, t: &Substitution) -> Vec<Self> {
        match &self.0 {
            TypeEnum::RowVar(rv) => rv.substitute(t),
            TypeEnum::Alias(_) | TypeEnum::Sum(SumType::Unit { .. }) => vec![self.clone()],
            TypeEnum::Variable(idx, bound) => {
                let TypeArg::Type { ty } = t.apply_var(*idx, &((*bound).into())) else {
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
    pub fn is_row_var(&self) -> bool {
        matches!(self.0, TypeEnum::RowVar(_))
    }

    /// New use (occurrence) of the row variable with specified index.
    /// `bound` must match that with which the variable was declared
    /// (i.e. as a [TypeParam::List]` of a `[TypeParam::Type]` of that bound).
    /// For use in [OpDef], not [FuncDefn], type schemes only.
    ///
    /// [OpDef]: crate::extension::OpDef
    /// [FuncDefn]: crate::ops::FuncDefn
    pub const fn new_row_var_use(idx: usize, bound: TypeBound) -> Self {
        Self(TypeEnum::RowVar(RowVariable(idx, bound)), bound)
    }
}

// ====== Conversions ======
impl TypeRV {
    fn try_into_<RV: MaybeRV>(self) -> Result<TypeBase<RV>, SignatureError> {
        Ok(TypeBase(
            match self.0 {
                TypeEnum::Extension(e) => TypeEnum::Extension(e),
                TypeEnum::Alias(a) => TypeEnum::Alias(a),
                TypeEnum::Function(f) => TypeEnum::Function(f),
                TypeEnum::Variable(idx, bound) => TypeEnum::Variable(idx, bound),
                TypeEnum::RowVar(rv) => TypeEnum::RowVar(RV::try_from_rv(rv)?),
                TypeEnum::Sum(s) => TypeEnum::Sum(s),
            },
            self.1,
        ))
    }
}

impl<RV1: MaybeRV> TypeBase<RV1> {
    /// A swiss-army-knife for any safe conversion of the type argument `RV1`
    /// to/from [NoRV]/RowVariable/rust-type-variable.
    fn into_<RV2: MaybeRV>(self) -> TypeBase<RV2> where RV1: Into<RV2> {
        TypeBase(
            match self.0 {
                TypeEnum::Extension(e) => TypeEnum::Extension(e),
                TypeEnum::Alias(a) => TypeEnum::Alias(a),
                TypeEnum::Function(f) => TypeEnum::Function(f),
                TypeEnum::Variable(idx, bound) => TypeEnum::Variable(idx, bound),
                TypeEnum::RowVar(rv) => TypeEnum::RowVar(rv.into()),
                TypeEnum::Sum(s) => TypeEnum::Sum(s),
            }, self.1
        )
    }
}

impl From<Type> for TypeRV {
    fn from(value: Type) -> Self {
        value.into_()
    }
}

impl TryFrom<TypeRV> for Type {
    type Error = SignatureError;
    fn try_from(value: TypeRV) -> Result<Self, Self::Error> {
        value.try_into_()
    }
}

/// Details a replacement of type variables with a finite list of known values.
/// (Variables out of the range of the list will result in a panic)
pub(crate) struct Substitution<'a>(&'a [TypeArg], &'a ExtensionRegistry);

impl<'a> Substitution<'a> {
    pub(crate) fn apply_var(&self, idx: usize, decl: &TypeParam) -> TypeArg {
        let arg = self
            .0
            .get(idx)
            .expect("Undeclared type variable - call validate() ?");
        debug_assert_eq!(check_type_arg(arg, decl), Ok(()));
        arg.clone()
    }

    fn apply_rowvar(&self, idx: usize, bound: TypeBound) -> Vec<TypeRV> {
        let arg = self
            .0
            .get(idx)
            .expect("Undeclared type variable - call validate() ?");
        debug_assert!(check_type_arg(arg, &TypeParam::new_list(bound)).is_ok());
        match arg {
            TypeArg::Sequence { elems } => elems
                .iter()
                .map(|ta| {
                    match ta {
                        TypeArg::Type { ty } => return ty.clone().into(),
                        TypeArg::Variable { v } => {
                            if let Some(b) = v.bound_if_row_var() {
                                return TypeRV::new_row_var_use(v.index(), b);
                            }
                        }
                        _ => (),
                    }
                    panic!("Not a list of types - call validate() ?")
                })
                .collect(),
            TypeArg::Type { ty } if matches!(ty.0, TypeEnum::RowVar(_)) => {
                // Standalone "Type" can be used iff its actually a Row Variable not an actual (single) Type
                vec![ty.clone().into()]
            }
            _ => panic!("Not a type or list of types - call validate() ?"),
        }
    }

    fn extension_registry(&self) -> &ExtensionRegistry {
        self.1
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
                    cached: cached_decl.clone(),
                    actual: actual.clone(),
                })
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test {

    use super::*;
    use crate::extension::prelude::USIZE_T;
    use crate::type_row;

    #[test]
    fn construct() {
        let t: Type = Type::new_tuple(vec![
            USIZE_T,
            Type::new_function(FunctionType::new_endo(vec![])),
            Type::new_extension(CustomType::new(
                "my_custom",
                [],
                "my_extension".try_into().unwrap(),
                TypeBound::Copyable,
            )),
            Type::new_alias(AliasDecl::new("my_alias", TypeBound::Eq)),
        ]);
        assert_eq!(
            &t.to_string(),
            "[usize, Function([[]][]), my_custom, Alias(my_alias)]"
        );
    }

    #[rstest::rstest]
    fn sum_construct() {
        let pred1 = Type::new_sum([type_row![], type_row![]]);
        let pred2 = TypeRV::new_unit_sum(2);

        assert_eq!(pred1, pred2);

        let pred_direct = SumType::Unit { size: 2 };
        // Pick <false> arbitrarily
        assert_eq!(pred1, Type::from(pred_direct));
    }

    mod proptest {

        use crate::proptest::RecursionDepth;

        use crate::types::{SumType, TypeEnum, TypeRow};
        use ::proptest::prelude::*;

        impl Arbitrary for super::SumType {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                use proptest::collection::vec;
                if depth.leaf() {
                    any::<u8>().prop_map(Self::new_unary).boxed()
                } else {
                    vec(any_with::<TypeRow>(depth), 0..3)
                        .prop_map(SumType::new)
                        .boxed()
                }
            }
        }

        impl<const RV: bool> Arbitrary for super::TypeBase<RV> {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                // We descend here, because a TypeEnum may contain a Type
                any_with::<TypeEnum>(depth.descend())
                    .prop_filter("Type<false> cannot be a Row Variable", |t| {
                        RV || !matches!(t, TypeEnum::RowVariable(_, _))
                    })
                    .prop_map(Self::new)
                    .boxed()
            }
        }
    }
}
