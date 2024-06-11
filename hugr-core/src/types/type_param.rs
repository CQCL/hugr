//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::extension::TypeDef

use itertools::Itertools;
#[cfg(test)]
use proptest_derive::Arbitrary;
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
    /// Argument is a [TypeArg::Sequence]. A list of indeterminate size containing
    /// parameters all of the (same) specified element type.
    List {
        /// The [TypeParam] describing each element of the list.
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

    /// Make a new `TypeParam::List` (an arbitrary-length homogenous list)
    pub fn new_list(elem: impl Into<TypeParam>) -> Self {
        Self::List {
            param: Box::new(elem.into()),
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

impl From<UpperBound> for TypeParam {
    fn from(bound: UpperBound) -> Self {
        Self::BoundedNat { bound }
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
        #[serde(flatten)]
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
    /// Variable (used in type schemes or inside polymorphic functions),
    /// but not a [TypeArg::Type] (not even a row variable i.e. [TypeParam::List] of type)
    /// nor [TypeArg::Extensions] - see [TypeArg::new_var_use]
    Variable {
        #[allow(missing_docs)]
        #[serde(flatten)]
        v: TypeArgVariable,
    },
}

impl<const RV: bool> From<Type<RV>> for TypeArg {
    fn from(ty: Type<RV>) -> Self {
        match ty.try_into_no_rv() {
            Ok(ty) => Self::Type { ty },
            Err((idx, bound)) => TypeArg::new_var_use(idx, TypeParam::new_list(bound)),
        }
    }
}

impl From<u64> for TypeArg {
    fn from(n: u64) -> Self {
        Self::BoundedNat { n }
    }
}

impl From<CustomTypeArg> for TypeArg {
    fn from(arg: CustomTypeArg) -> Self {
        Self::Opaque { arg }
    }
}

impl From<Vec<TypeArg>> for TypeArg {
    fn from(elems: Vec<TypeArg>) -> Self {
        Self::Sequence { elems }
    }
}

impl From<ExtensionSet> for TypeArg {
    fn from(es: ExtensionSet) -> Self {
        Self::Extensions { es }
    }
}

/// Variable in a TypeArg, that is neither a [TypeArg::Extensions]
/// nor a single [TypeArg::Type] (i.e. not a [Type::new_var_use]
/// - it might be a [Type::new_row_var_use]).
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct TypeArgVariable {
    idx: usize,
    cached_decl: TypeParam,
}

impl TypeArg {
    /// [Type::UNIT] as a [TypeArg::Type]
    pub const UNIT: Self = Self::Type { ty: Type::UNIT };

    /// Makes a TypeArg representing a use (occurrence) of the type variable
    /// with the specified index.
    /// `decl` must be exactly that with which the variable was declared.
    pub fn new_var_use(idx: usize, decl: TypeParam) -> Self {
        match decl {
            // Note a TypeParam::List of TypeParam::Type *cannot* be represented
            // as a TypeArg::Type because the latter stores a Type<false> i.e. only a single type,
            // not a RowVariable.
            TypeParam::Type { b } => Type::<false>::new_var_use(idx, b).into(),
            // Prevent TypeArg::Variable(idx, TypeParam::Extensions)
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
            } => {
                assert!(
                    !matches!(cached_decl, TypeParam::Type { .. }),
                    "Malformed TypeArg::Variable {} - should be inconstructible",
                    cached_decl
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
            TypeArg::BoundedNat { .. } => self.clone(), // We do not allow variables as bounds on BoundedNat's
            TypeArg::Opaque {
                arg: CustomTypeArg { typ, .. },
            } => {
                // The type must be equal to that declared (in a TypeParam) by the instantiated TypeDef,
                // so cannot contain variables declared by the instantiator (providing the TypeArgs)
                debug_assert_eq!(&typ.substitute(t), typ);
                self.clone()
            }
            TypeArg::Sequence { elems } => {
                let mut are_types = elems.iter().map(|ta| match ta {
                    TypeArg::Type { .. } => true,
                    TypeArg::Variable { v } => v.bound_if_row_var().is_some(),
                    _ => false,
                });
                let elems = match are_types.next() {
                    Some(true) => {
                        assert!(are_types.all(|b| b)); // If one is a Type, so must the rest be
                                                       // So, anything that doesn't produce a Type, was a row variable => multiple Types
                        elems
                            .iter()
                            .flat_map(|ta| match ta.substitute(t) {
                                ty @ TypeArg::Type { .. } => vec![ty],
                                TypeArg::Sequence { elems } => elems,
                                _ => panic!("Expected Type or row of Types"),
                            })
                            .collect()
                    }
                    _ => {
                        // not types, no need to flatten (and mustn't, in case of nested Sequences)
                        elems.iter().map(|ta| ta.substitute(t)).collect()
                    }
                };
                TypeArg::Sequence { elems }
            }
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

    /// Determines whether this represents a row variable; if so, returns
    /// the [TypeBound] of the individual types it might stand for.
    pub fn bound_if_row_var(&self) -> Option<TypeBound> {
        if let TypeParam::List { param } = &self.cached_decl {
            if let TypeParam::Type { b } = **param {
                return Some(b);
            }
        }
        None
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
            elems.iter().try_for_each(|arg| {
                // Also allow elements that are RowVars if fitting into a List of Types
                if let (
                    TypeArg::Variable {
                        v:
                            TypeArgVariable {
                                idx: _,
                                cached_decl: TypeParam::List { param: arg_var },
                            },
                    },
                    TypeParam::Type { b: param_bound },
                ) = (arg, &**param)
                {
                    if let TypeParam::Type { b: var_bound } = **arg_var {
                        if param_bound.contains(var_bound) {
                            return Ok(());
                        }
                    }
                }
                check_type_arg(arg, param)
            })
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
#[non_exhaustive]
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

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::{check_type_arg, Substitution, TypeArg, TypeParam};
    use crate::extension::prelude::{BOOL_T, PRELUDE_REGISTRY, USIZE_T};
    use crate::types::{type_param::TypeArgError, Type, TypeBound};

    #[test]
    fn type_arg_fits_param() {
        let rowvar = Type::new_row_var_use;
        fn check(arg: impl Into<TypeArg>, parm: &TypeParam) -> Result<(), TypeArgError> {
            check_type_arg(&arg.into(), parm)
        }
        fn check_seq<T: Clone + Into<TypeArg>>(
            args: &[T],
            parm: &TypeParam,
        ) -> Result<(), TypeArgError> {
            let arg = args.iter().cloned().map_into().collect_vec().into();
            check_type_arg(&arg, parm)
        }
        // Simple cases: a TypeArg::Type is a TypeParam::Type but singleton sequences are lists
        check(USIZE_T, &TypeBound::Eq.into()).unwrap();
        let seq_param = TypeParam::new_list(TypeBound::Eq);
        check(USIZE_T, &seq_param).unwrap_err();
        check_seq(&[USIZE_T], &TypeBound::Any.into()).unwrap_err();

        // Into a list of type, we can fit a single row var
        check(rowvar(0, TypeBound::Eq), &seq_param).unwrap();
        // or a list of (types or row vars)
        check(vec![], &seq_param).unwrap();
        check_seq(&[rowvar(0, TypeBound::Eq)], &seq_param).unwrap();
        check_seq(
            &[
                rowvar(1, TypeBound::Any),
                USIZE_T.into(),
                rowvar(0, TypeBound::Eq),
            ],
            &TypeParam::new_list(TypeBound::Any),
        )
        .unwrap();
        // Next one fails because a list of Eq is required
        check_seq(
            &[
                rowvar(1, TypeBound::Any),
                USIZE_T.into(),
                rowvar(0, TypeBound::Eq),
            ],
            &seq_param,
        )
        .unwrap_err();
        // seq of seq of types is not allowed
        check(
            vec![USIZE_T.into(), vec![USIZE_T.into()].into()],
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

        // TypeParam::Tuples require a TypeArg::Seq of the same number of elems
        let usize_and_ty = TypeParam::Tuple {
            params: vec![TypeParam::max_nat(), TypeBound::Eq.into()],
        };
        check(vec![5.into(), USIZE_T.into()], &usize_and_ty).unwrap();
        check(vec![USIZE_T.into(), 5.into()], &usize_and_ty).unwrap_err(); // Wrong way around
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
        // The <false> here is arbitrary but we have to specify it:
        let row_arg: TypeArg = vec![BOOL_T.into(), TypeArg::UNIT].into();
        check_type_arg(&row_arg, &row_param).unwrap();

        // Now say a row variable referring to *that* row was used
        // to instantiate an outer "row parameter" (list of type).
        let outer_param = TypeParam::new_list(TypeBound::Any);
        let outer_arg = TypeArg::Sequence {
            elems: vec![
                Type::new_row_var_use(0, TypeBound::Copyable).into(),
                USIZE_T.into(),
            ],
        };
        check_type_arg(&outer_arg, &outer_param).unwrap();

        let outer_arg2 = outer_arg.substitute(&Substitution(&[row_arg], &PRELUDE_REGISTRY));
        assert_eq!(
            outer_arg2,
            vec![BOOL_T.into(), TypeArg::UNIT, USIZE_T.into()].into()
        );

        // Of course this is still valid (as substitution is guaranteed to preserve validity)
        check_type_arg(&outer_arg2, &outer_param).unwrap();
    }

    #[test]
    fn subst_list_list() {
        let outer_param = TypeParam::new_list(TypeParam::new_list(TypeBound::Any));
        let row_var_decl = TypeParam::new_list(TypeBound::Copyable);
        let row_var_use = TypeArg::new_var_use(0, row_var_decl.clone());
        let good_arg = TypeArg::Sequence {
            elems: vec![
                // The row variables here refer to `row_var_decl` above
                vec![USIZE_T.into()].into(),
                row_var_use.clone(),
                vec![row_var_use, USIZE_T.into()].into(),
            ],
        };
        check_type_arg(&good_arg, &outer_param).unwrap();

        // Outer list cannot include single types:
        let TypeArg::Sequence { mut elems } = good_arg.clone() else {
            panic!()
        };
        elems.push(USIZE_T.into());
        assert_eq!(
            check_type_arg(&TypeArg::Sequence { elems }, &outer_param),
            Err(TypeArgError::TypeMismatch {
                arg: USIZE_T.into(),
                // The error reports the type expected for each element of the list:
                param: TypeParam::new_list(TypeBound::Any)
            })
        );

        // Now substitute a list of two types for that row-variable
        let row_var_arg = vec![USIZE_T.into(), BOOL_T.into()].into();
        check_type_arg(&row_var_arg, &row_var_decl).unwrap();
        let subst_arg =
            good_arg.substitute(&Substitution(&[row_var_arg.clone()], &PRELUDE_REGISTRY));
        check_type_arg(&subst_arg, &outer_param).unwrap(); // invariance of substitution
        assert_eq!(
            subst_arg,
            TypeArg::Sequence {
                elems: vec![
                    vec![USIZE_T.into()].into(),
                    row_var_arg,
                    vec![USIZE_T.into(), BOOL_T.into(), USIZE_T.into()].into()
                ]
            }
        );
    }

    mod proptest {

        use proptest::prelude::*;

        use super::super::{CustomTypeArg, TypeArg, TypeArgVariable, TypeParam, UpperBound};
        use crate::extension::ExtensionSet;
        use crate::proptest::{any_serde_yaml_value, RecursionDepth};
        use crate::types::{CustomType, Type, TypeBound};

        impl Arbitrary for CustomTypeArg {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                (
                    any_with::<CustomType>(
                        <CustomType as Arbitrary>::Parameters::new(depth).with_bound(TypeBound::Eq),
                    ),
                    any_serde_yaml_value(),
                )
                    .prop_map(|(ct, value)| CustomTypeArg::new(ct, value.clone()).unwrap())
                    .boxed()
            }
        }

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
                    Just(Self::Extensions).boxed(),
                    any::<TypeBound>().prop_map(|b| Self::Type { b }).boxed(),
                    any::<UpperBound>()
                        .prop_map(|bound| Self::BoundedNat { bound })
                        .boxed(),
                    any_with::<CustomType>(depth.into())
                        .prop_map(|ty| Self::Opaque { ty })
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
                            .boxed())
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
                    any::<ExtensionSet>()
                        .prop_map(|es| Self::Extensions { es })
                        .boxed(),
                    any_with::<Type>(depth)
                        .prop_map(|ty| Self::Type { ty })
                        .boxed(),
                    any_with::<CustomTypeArg>(depth)
                        .prop_map(|arg| Self::Opaque { arg })
                        .boxed(),
                    // TODO this is a bit dodgy, TypeArgVariables are supposed
                    // to be constructed from TypeArg::new_var_use. We are only
                    // using this instance for serialisation now, but if we want
                    // to generate valid TypeArgs this will need to change.
                    any_with::<TypeArgVariable>(depth)
                        .prop_map(|v| Self::Variable { v })
                        .boxed(),
                ]);
                if !depth.leaf() {
                    // We descend here because this constructor contains TypeArg>
                    strat = strat.or(vec(any_with::<Self>(depth.descend()), 0..3)
                        .prop_map(|elems| Self::Sequence { elems })
                        .boxed());
                }
                strat.boxed()
            }
        }
    }
}
