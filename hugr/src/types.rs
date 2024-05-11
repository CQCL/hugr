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
pub use poly_func::PolyFuncType;
pub use signature::FunctionType;
use smol_str::SmolStr;
pub use type_param::TypeArg;
pub use type_row::TypeRow;

use itertools::FoldWhile::{Continue, Done};
use itertools::{repeat_n, Itertools};
use serde::{Deserialize, Serialize};

use crate::extension::{ExtensionRegistry, SignatureError};
use crate::ops::AliasDecl;
use crate::type_row;

use self::type_param::TypeParam;

/// A unique identifier for a type.
pub type TypeName = SmolStr;

/// Slice of a [`TypeName`] type identifier.
pub type TypeNameRef = str;

/// The kinds of edges in a HUGR, excluding Hierarchy.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
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
    Function(PolyFuncType),
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
    General { rows: Vec<TypeRow<true>> },
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
        V: Into<TypeRow<true>>,
    {
        let rows = variants.into_iter().map(Into::into).collect_vec();

        let len: usize = rows.len();
        if len <= (u8::MAX as usize) && rows.iter().all(TypeRow::is_empty) {
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
    pub fn get_variant(&self, tag: usize) -> Option<&TypeRow<true>> {
        match self {
            SumType::Unit { size } if tag < (*size as usize) => Some(Type::EMPTY_TYPEROW_REF),
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

impl <const RV:bool> From<SumType> for Type<RV> {
    fn from(sum: SumType) -> Self {
        match sum {
            SumType::Unit { size } => Type::new_unit_sum(size),
            SumType::General { rows } => Type::new_sum(rows),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, derive_more::Display)]
/// Core types
pub enum TypeEnum<const ROWVARS:bool=false> {
    // TODO optimise with Box<CustomType> ?
    // or some static version of this?
    #[allow(missing_docs)]
    Extension(CustomType),
    #[allow(missing_docs)]
    #[display(fmt = "Alias({})", "_0.name()")]
    Alias(AliasDecl),
    #[allow(missing_docs)]
    #[display(fmt = "Function({})", "_0")]
    Function(Box<FunctionType<true>>),
    // Index into TypeParams, and cache of TypeBound (checked in validation)
    #[allow(missing_docs)]
    #[display(fmt = "Variable({})", _0)]
    Variable(usize, TypeBound),
    /// Variable index, and cache of inner TypeBound - matches a [TypeParam::List] of [TypeParam::Type]
    /// of this bound (checked in validation). Should only exist for `Type<true>` and `TypeEnum<true>`.
    #[display(fmt = "RowVar({})", _0)]
    RowVariable(usize, TypeBound),
    #[allow(missing_docs)]
    #[display(fmt = "{}", "_0")]
    Sum(SumType),
}

/*impl <const RV:bool> PartialEq<TypeEnum> for TypeEnum<RV> {
    fn eq(&self, other: &TypeEnum) -> bool {
        match (self, other) {
            (TypeEnum::Extension(e1), TypeEnum::Extension(e2)) => e1 == e2,
            (TypeEnum::Alias(a1), TypeEnum::Alias(a2)) => a1 == a2,
            (TypeEnum::Function(f1), TypeEnum::Function(f2)) => f1==f2,
            (TypeEnum::Variable(i1, b1), TypeEnum::Variable(i2, b2)) => i1==i2 && b1==b2,
            (TypeEnum::RowVariable(i1, b1), TypeEnum::RowVariable(i2,  b2)) => i1==i2 && b1==b2,
            (TypeEnum::Sum(s1), TypeEnum::Sum(s2)) => s1 == s2,
            _ => false
        }
    }
}*/

impl <const RV:bool> TypeEnum<RV> {
    /// The smallest type bound that covers the whole type.
    fn least_upper_bound(&self) -> TypeBound {
        match self {
            TypeEnum::Extension(c) => c.bound(),
            TypeEnum::Alias(a) => a.bound,
            TypeEnum::Function(_) => TypeBound::Copyable,
            TypeEnum::Variable(_, b) | TypeEnum::RowVariable(_, b) => *b,
            TypeEnum::Sum(SumType::Unit { size: _ }) => TypeBound::Eq,
            TypeEnum::Sum(SumType::General { rows }) => least_upper_bound(
                rows.iter()
                    .flat_map(TypeRow::iter)
                    .map(Type::least_upper_bound),
            ),
        }
    }
}

#[derive(
    Clone, Debug, PartialEq, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[display(fmt = "{}", "_0")]
#[serde(into = "serialize::SerSimpleType", try_from = "serialize::SerSimpleType")]
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
/// let sum = Type::new_sum([type_row![], type_row![]]);
/// assert_eq!(sum.least_upper_bound(), TypeBound::Eq);
/// ```
///
/// ```
/// # use hugr::types::{Type, TypeBound, FunctionType};
///
/// let func_type = Type::new_function(FunctionType::new_endo(vec![]));
/// assert_eq!(func_type.least_upper_bound(), TypeBound::Copyable);
/// ```
pub struct Type<const ROWVARS:bool=false>(TypeEnum<ROWVARS>, TypeBound);

/*impl<const RV:bool> PartialEq<Type> for Type<RV> {
    fn eq(&self, other: &Type) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}*/

impl<const RV:bool> Type<RV> {
    /// An empty `TypeRow`. Provided here for convenience
    pub const EMPTY_TYPEROW: TypeRow<RV> = type_row![];
    /// Unit type (empty tuple).
    pub const UNIT: Self = Self(TypeEnum::Sum(SumType::Unit { size: 1 }), TypeBound::Eq);

    const EMPTY_TYPEROW_REF: &'static TypeRow<RV> = &Self::EMPTY_TYPEROW;

    /// Initialize a new function type.
    pub fn new_function(fun_ty: impl Into<FunctionType<true>>) -> Self {
        Self::new(TypeEnum::Function(Box::new(fun_ty.into())))
    }

    /// Initialize a new tuple type by providing the elements.
    #[inline(always)]
    pub fn new_tuple(types: impl Into<TypeRow<true>>) -> Self {
        let row = types.into();
        match row.len() {
            0 => Self::UNIT,
            _ => Self::new_sum([row]),
        }
    }

    /// Initialize a new sum type by providing the possible variant types.
    #[inline(always)]
    pub fn new_sum(variants: impl IntoIterator<Item = TypeRow<true>>) -> Self where {
        Self::new(TypeEnum::Sum(SumType::new(variants)))
    }

    /// Initialize a new custom type.
    // TODO remove? Extensions/TypeDefs should just provide `Type` directly
    pub const fn new_extension(opaque: CustomType) -> Self {
        let bound = opaque.bound();
        Type(TypeEnum::Extension(opaque), bound)
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
    /// For use in type schemes only: `bound` must match that with which the
    /// variable was declared (i.e. as a [TypeParam::Type]`(bound)`).
    pub const fn new_var_use(idx: usize, bound: TypeBound) -> Self {
        Self(TypeEnum::Variable(idx, bound), bound)
    }

    /// Report the least upper TypeBound, if there is one.
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

    /// Checks that this [Type] represents a single Type, not a row variable,
    /// that all variables used within are in the provided list of bound variables,
    /// and that for each [CustomType], the corresponding
    /// [TypeDef] is in the [ExtensionRegistry] and the type arguments
    /// [validate] and fit into the def's declared parameters.
    ///
    /// [validate]: crate::types::type_param::TypeArg::validate
    /// [TypeDef]: crate::extension::TypeDef
    pub(crate) fn validate_1type(
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
            // FunctionTypes used as types of values may have unknown arity (e.g. if they are not called)
            TypeEnum::Function(ft) => ft.validate(extension_registry, var_decls),
            TypeEnum::Variable(idx, bound) => check_typevar_decl(var_decls, *idx, &(*bound).into()),
            TypeEnum::RowVariable(idx, _) => {
                Err(SignatureError::RowTypeVarOutsideRow { idx: *idx })
            }
        }
    }

    /// Checks that this [Type] is valid (as per [Type::validate_1type]), but also
    /// allows row variables that may become multiple types
    fn validate_in_row(
        &self,
        extension_registry: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        if let TypeEnum::RowVariable(idx, bound) = self.0 {
            let t = TypeParam::List {
                param: Box::new(bound.into()),
            };
            check_typevar_decl(var_decls, idx, &t)
        } else {
            self.validate_1type(extension_registry, var_decls)
        }
    }

    /// Applies a substitution to a type.
    /// This may result in a row of types, if this [Type] is not really a single type but actually a row variable
    /// (of course this can only occur for a `Type<true``). For a `Type<false>`, will always return exactly one element.
    fn subst_vec(&self, s: &Substitution) -> Vec<Self> {
        match &self.0 {
            TypeEnum::RowVariable(idx, bound) => 
                if RV {s.apply_rowvar(idx, bound)} // ALAN Argh, type error here as Type<true> != Type<RV> even inside "if RV"
                else {panic!("Row Variable outside Row - should not have validated?")},
            TypeEnum::Alias(_) | TypeEnum::Sum(SumType::Unit { .. }) => vec![self.clone().into()],
            TypeEnum::Variable(idx, bound) => {
                let TypeArg::Type { ty } = s.apply_var(*idx, &((*bound).into())) else {
                    panic!("Variable was not a type - try validate() first")
                };
                vec![ty] // ALAN argh, can't convert parametrically from Type<false> to Type<RV>
            }
            TypeEnum::Extension(cty) => vec![Type::new_extension(cty.substitute(s))],
            TypeEnum::Function(bf) => vec![Type::new_function(bf.substitute(s))],
            TypeEnum::Sum(SumType::General { rows }) => {
                vec![Type::new_sum(rows.iter().map(|r| r.substitute(s)))]
            }
        }
    }
}

impl TryFrom<Type<true>> for Type<false> {
    type Error = ConvertError;
    fn try_from(value: Type<true>) -> Result<Self, Self::Error> {
        Ok(Self(match value.0 {
            TypeEnum::Extension(e) => TypeEnum::Extension(e),
            TypeEnum::Alias(a) => TypeEnum::Alias(a),
            TypeEnum::Function(ft) => TypeEnum::Function(ft),
            TypeEnum::Variable(i, b) => TypeEnum::Variable(i,b),
            TypeEnum::RowVariable(_, _) => return Err(ConvertError),
            TypeEnum::Sum(st) => TypeEnum::Sum(st)
        }, value.1))
    }
}

impl Type<false> {
    fn substitute(&self, s: &Substitution) -> Self {
        let v = self.subst_vec(s);
        let [r] = v.try_into().unwrap(); // No row vars, so every Type<false> produces exactly one
        r
    }
}

impl Type<true> {
    /// New use (occurrence) of the row variable with specified index.
    /// `bound` must match that with which the variable was declared
    /// (i.e. as a [TypeParam::List]` of a `[TypeParam::Type]` of that bound).
    /// For use in [OpDef], not [FuncDefn], type schemes only.
    ///
    /// [OpDef]: crate::extension::OpDef
    /// [FuncDefn]: crate::ops::FuncDefn
    pub const fn new_row_var(idx: usize, bound: TypeBound) -> Self {
        Self(TypeEnum::RowVariable(idx, bound), bound)
    }

    fn substitute(&self, s: &Substitution) -> Vec<Self> {
        self.subst_vec(s)
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

    fn apply_rowvar(&self, idx: usize, bound: TypeBound) -> Vec<Type<true>> {
        let arg = self
            .0
            .get(idx)
            .expect("Undeclared type variable - call validate() ?");
        match arg {
            TypeArg::Sequence { elems } => elems
                .iter()
                .map(|ta| match ta {
                    TypeArg::Type { ty } => ty.clone().into(),
                    _ => panic!("Not a list of types - did validate() ?"),
                })
                .collect(),
            TypeArg::Type { ty } => {
                debug_assert_eq!(check_type_arg(arg, &TypeParam::Type { b: bound }), Ok(()));
                vec![ty.clone().into()]
            }
            _ => panic!("Not a type or list of types - did validate() ?"),
        }
    }

    fn extension_registry(&self) -> &ExtensionRegistry {
        self.1
    }
}

impl From<Type<false>> for Type<true> {
    fn from(value: Type<false>) -> Self {
        Self(match value.0 {
            TypeEnum::Alias(a) => TypeEnum::Alias(a),
            TypeEnum::Extension(e) => TypeEnum::Extension(e),
            TypeEnum::Function(ft) => TypeEnum::Function(ft),
            TypeEnum::Variable(idx, bound) => TypeEnum::Variable(idx, bound),
            TypeEnum::RowVariable(_, _) => panic!("Type<false> should not contain row variables"),
            TypeEnum::Sum(st) => TypeEnum::Sum(st),
        }, value.1)
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

    #[test]
    fn sum_construct() {
        let pred1 = Type::new_sum([Type::EMPTY_TYPEROW, Type::EMPTY_TYPEROW]);
        let pred2 = Type::new_unit_sum(2);

        assert_eq!(pred1, pred2);

        let pred_direct = SumType::Unit { size: 2 };
        assert_eq!(pred1, pred_direct.into())
    }
}
