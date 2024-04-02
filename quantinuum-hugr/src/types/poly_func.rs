//! Polymorphic Function Types

use crate::{
    extension::{ExtensionRegistry, SignatureError},
    types::type_param::check_type_arg,
};
use itertools::Itertools;

use super::{
    signature::{FuncTypeBase, TypeRowElem},
    type_param::{check_type_args, TypeArg, TypeParam},
    type_row::RowVarOrType,
    Type,
};
use super::{Substitution, TypeBound};

/// A polymorphic function type, e.g. of a [Graph], or perhaps an [OpDef].
/// (Nodes/operations in the Hugr are not polymorphic.)
///
/// [Graph]: crate::ops::constant::Const::Function
/// [OpDef]: crate::extension::OpDef
#[derive(
    Clone, PartialEq, Debug, Default, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[display(
    fmt = "forall {}. {}",
    "params.iter().map(ToString::to_string).join(\" \")",
    "body"
)]
pub struct PolyFuncBase<T>
where
    T: TypeRowElem,
    [T]: ToOwned<Owned = Vec<T>>,
{
    /// The declared type parameters, i.e., these must be instantiated with
    /// the same number of [TypeArg]s before the function can be called. Note that within
    /// the [Self::body], variable (DeBruijn) index 0 is element 0 of this array, i.e. the
    /// variables are bound from right to left.
    ///
    /// [TypeArg]: super::type_param::TypeArg
    params: Vec<TypeParam>,
    /// Template for the function. May contain variables up to length of [Self::params]
    body: FuncTypeBase<T>,
}
pub type PolyFuncType = PolyFuncBase<RowVarOrType>;
pub type PolyFixedFunc = PolyFuncBase<Type>;

impl<T: TypeRowElem> From<FuncTypeBase<T>> for PolyFuncBase<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn from(body: FuncTypeBase<T>) -> Self {
        Self {
            params: vec![],
            body,
        }
    }
}

impl From<PolyFixedFunc> for PolyFuncType {
    fn from(value: PolyFixedFunc) -> Self {
        Self {
            params: value.params,
            body: value.body.into(),
        }
    }
}

impl<T: TypeRowElem> PolyFuncBase<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    /// The type parameters, aka binders, over which this type is polymorphic
    pub fn params(&self) -> &[TypeParam] {
        &self.params
    }

    /// The body of the type, a function type.
    pub fn body(&self) -> &FuncTypeBase<T> {
        &self.body
    }

    /// Create a new PolyFuncBase given the kinds of the variables it declares
    /// and the underlying [FunctionType].
    pub fn new(params: impl Into<Vec<TypeParam>>, body: impl Into<FuncTypeBase<T>>) -> Self {
        Self {
            params: params.into(),
            body: body.into(),
        }
    }

    /// Validates this instance, checking that the types in the body are
    /// wellformed with respect to the registry, and that all type variables
    /// are declared (perhaps in an enclosing scope, kinds passed in).
    pub fn validate(
        &self,
        reg: &ExtensionRegistry,
        external_var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        // TODO https://github.com/CQCL/hugr/issues/624 validate TypeParams declared here, too
        let mut v; // Declared here so live until end of scope
        let all_var_decls = if self.params.is_empty() {
            external_var_decls
        } else {
            // Type vars declared here go at lowest indices (as per DeBruijn)
            v = self.params.clone();
            v.extend_from_slice(external_var_decls);
            v.as_slice()
        };
        self.body.validate(reg, all_var_decls)
    }

    pub(super) fn substitute(&self, t: &impl Substitution) -> Self {
        if self.params.is_empty() {
            // Avoid using complex code for simple Monomorphic case
            return self.body.substitute(t).into();
        }
        PolyFuncBase {
            params: self.params.clone(),
            body: self.body.substitute(&InsideBinders {
                num_binders: self.params.len(),
                underlying: t,
            }),
        }
    }

    /// (Perhaps-partially) instantiates this [PolyFuncBase] into another with fewer binders.
    /// Note that indices into `args` correspond to the same index within [Self::params],
    /// so we instantiate the lowest-index [Self::params] first, even though these
    /// would be considered "innermost" / "closest" according to DeBruijn numbering.
    pub(crate) fn instantiate_poly(
        &self,
        args: &[TypeArg],
        exts: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let remaining = self.params.get(args.len()..).unwrap_or_default();
        let mut v;
        let args = if remaining.is_empty() {
            args // instantiate below will fail if there were too many
        } else {
            // Partial application - renumber remaining params (still bound) downward
            v = args.to_vec();
            v.extend(
                remaining
                    .iter()
                    .enumerate()
                    .map(|(i, decl)| TypeArg::new_var_use(i, decl.clone())),
            );
            v.as_slice()
        };
        Ok(Self {
            params: remaining.to_vec(),
            body: self.instantiate(args, exts)?,
        })
    }

    /// Instantiates an outer [PolyFuncBase], i.e. with no free variables
    /// (as ensured by [Self::validate]), into a monomorphic type.
    ///
    /// # Errors
    /// If there is not exactly one [TypeArg] for each binder ([Self::params]),
    /// or an arg does not fit into its corresponding [TypeParam]
    pub(crate) fn instantiate(
        &self,
        args: &[TypeArg],
        ext_reg: &ExtensionRegistry,
    ) -> Result<FuncTypeBase<T>, SignatureError> {
        // Check that args are applicable, and that we have a value for each binder,
        // i.e. each possible free variable within the body.
        check_type_args(args, &self.params)?;
        Ok(self.body.substitute(&SubstValues(args, ext_reg)))
    }
}

/// A [Substitution] with a finite list of known values.
/// (Variables out of the range of the list will result in a panic)
struct SubstValues<'a>(&'a [TypeArg], &'a ExtensionRegistry);

impl<'a> Substitution for SubstValues<'a> {
    fn apply_var(&self, idx: usize, decl: &TypeParam) -> TypeArg {
        let arg = self
            .0
            .get(idx)
            .expect("Undeclared type variable - call validate() ?");
        debug_assert_eq!(check_type_arg(arg, decl), Ok(()));
        arg.clone()
    }

    fn apply_rowvar(&self, idx: usize, _bound: TypeBound) -> Vec<RowVarOrType> {
        let arg = self
            .0
            .get(idx)
            .expect("Undeclared type variable - call validate() ?");
        match arg {
            TypeArg::Sequence { elems } => elems
                .iter()
                .map(|ta| match ta {
                    TypeArg::Type { ty } => ty.clone(),
                    _ => panic!("Not a list of types - did validate() ?"),
                })
                .map(RowVarOrType::T)
                .collect(),
            _ => panic!("Not a type or list of types - did validate() ?"),
        }
    }

    fn extension_registry(&self) -> &ExtensionRegistry {
        self.1
    }
}

/// A [Substitution] that renumbers any type variable to another (of the same kind)
/// with a index increased by a fixed `usize``.
struct Renumber<'a> {
    offset: usize,
    exts: &'a ExtensionRegistry,
}

impl<'a> Substitution for Renumber<'a> {
    fn apply_var(&self, idx: usize, decl: &TypeParam) -> TypeArg {
        TypeArg::new_var_use(idx + self.offset, decl.clone())
    }

    fn extension_registry(&self) -> &ExtensionRegistry {
        self.exts
    }
}

/// Given a [Substitution] defined outside a binder (i.e. [PolyFuncBase]),
/// applies that transformer to types inside the binder (i.e. arguments/results of said function)
struct InsideBinders<'a> {
    /// The number of binders we have entered since (beneath where) we started to apply
    /// [Self::underlying]).
    /// That is, the lowest `num_binders` variable indices refer to locals bound since then.
    num_binders: usize,
    /// Substitution that was being applied outside those binders (i.e. in outer scope)
    underlying: &'a dyn Substitution,
}

impl<'a> Substitution for InsideBinders<'a> {
    fn apply_var(&self, idx: usize, decl: &TypeParam) -> TypeArg {
        // Convert variable index into outer scope
        match idx.checked_sub(self.num_binders) {
            None => TypeArg::new_var_use(idx, decl.clone()), // Bound locally, unknown to `underlying`
            Some(idx_in_outer_scope) => {
                let result_in_outer_scope = self.underlying.apply_var(idx_in_outer_scope, decl);
                // Transform returned value into the current scope, i.e. avoid the variables newly bound
                result_in_outer_scope.substitute(&Renumber {
                    offset: self.num_binders,
                    exts: self.extension_registry(),
                })
            }
        }
    }

    fn extension_registry(&self) -> &ExtensionRegistry {
        self.underlying.extension_registry()
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::num::NonZeroU64;

    use lazy_static::lazy_static;
    use smol_str::SmolStr;

    use crate::extension::prelude::{array_type, BOOL_T, PRELUDE_ID, USIZE_CUSTOM_T, USIZE_T};
    use crate::extension::{
        ExtensionId, ExtensionRegistry, SignatureError, TypeDefBound, PRELUDE, PRELUDE_REGISTRY,
    };
    use crate::std_extensions::collections::{EXTENSION, LIST_TYPENAME};
    use crate::types::signature::FuncTypeBase;
    use crate::types::type_param::{TypeArg, TypeArgError, TypeParam};
    use crate::types::type_row::RowVarOrType;
    use crate::types::{CustomType, FunctionType, PolyFuncType, Type, TypeBound};
    use crate::Extension;

    use super::{PolyFuncBase, TypeRowElem};

    lazy_static! {
        static ref REGISTRY: ExtensionRegistry =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), EXTENSION.to_owned()]).unwrap();
    }

    impl<T: TypeRowElem> PolyFuncBase<T>
    where
        [T]: ToOwned<Owned = Vec<T>>,
    {
        fn new_validated(
            params: impl Into<Vec<TypeParam>>,
            body: FuncTypeBase<T>,
            extension_registry: &ExtensionRegistry,
        ) -> Result<Self, SignatureError> {
            let res = Self::new(params, body);
            res.validate(extension_registry, &[])?;
            Ok(res)
        }
    }

    #[test]
    fn test_opaque() -> Result<(), SignatureError> {
        let list_def = EXTENSION.get_type(&LIST_TYPENAME).unwrap();
        let tyvar = TypeArg::new_var_use(0, TypeBound::Any.into());
        let list_of_var = Type::new_extension(list_def.instantiate([tyvar.clone()])?);
        let list_len = PolyFuncBase::new_validated(
            [TypeBound::Any.into()],
            FunctionType::new(vec![list_of_var], vec![USIZE_T]),
            &REGISTRY,
        )?;

        let t = list_len.instantiate(&[TypeArg::Type { ty: USIZE_T }], &REGISTRY)?;
        assert_eq!(
            t,
            FunctionType::new(
                vec![Type::new_extension(
                    list_def
                        .instantiate([TypeArg::Type { ty: USIZE_T }])
                        .unwrap()
                )],
                vec![USIZE_T]
            )
        );

        Ok(())
    }

    fn id_fn(t: Type) -> FunctionType {
        FunctionType::new(vec![t.clone()], vec![t])
    }

    #[test]
    fn test_mismatched_args() -> Result<(), SignatureError> {
        let ar_def = PRELUDE.get_type("array").unwrap();
        let typarams = [TypeParam::max_nat(), TypeBound::Any.into()];
        let [tyvar, szvar] =
            [0, 1].map(|i| TypeArg::new_var_use(i, typarams.get(i).unwrap().clone()));

        // Valid schema...
        let good_array = Type::new_extension(ar_def.instantiate([tyvar.clone(), szvar.clone()])?);
        let good_ts =
            PolyFuncBase::new_validated(typarams.clone(), id_fn(good_array), &PRELUDE_REGISTRY)?;

        // Sanity check (good args)
        good_ts.instantiate(
            &[TypeArg::BoundedNat { n: 5 }, TypeArg::Type { ty: USIZE_T }],
            &PRELUDE_REGISTRY,
        )?;

        let wrong_args = good_ts.instantiate(
            &[TypeArg::Type { ty: USIZE_T }, TypeArg::BoundedNat { n: 5 }],
            &PRELUDE_REGISTRY,
        );
        assert_eq!(
            wrong_args,
            Err(SignatureError::TypeArgMismatch(
                TypeArgError::TypeMismatch {
                    param: typarams[0].clone(),
                    arg: TypeArg::Type { ty: USIZE_T }
                }
            ))
        );

        // (Try to) make a schema with bad args
        let arg_err = SignatureError::TypeArgMismatch(TypeArgError::TypeMismatch {
            param: typarams[0].clone(),
            arg: szvar.clone(),
        });
        assert_eq!(
            ar_def.instantiate([szvar.clone(), tyvar.clone()]),
            Err(arg_err.clone())
        );
        // ok, so that doesn't work - well, it shouldn't! So let's say we just have this signature (with bad args)...
        let bad_array = Type::new_extension(CustomType::new(
            "array",
            [szvar, tyvar],
            PRELUDE_ID,
            TypeBound::Any,
        ));
        let bad_ts =
            PolyFuncBase::new_validated(typarams.clone(), id_fn(bad_array), &PRELUDE_REGISTRY);
        assert_eq!(bad_ts.err(), Some(arg_err));

        Ok(())
    }

    #[test]
    fn test_misused_variables() -> Result<(), SignatureError> {
        // Variables in args have different bounds from variable declaration
        let tv = TypeArg::new_var_use(0, TypeBound::Copyable.into());
        let list_def = EXTENSION.get_type(&LIST_TYPENAME).unwrap();
        let body_type = id_fn(Type::new_extension(list_def.instantiate([tv])?));
        for decl in [
            TypeParam::Extensions,
            TypeParam::List {
                param: Box::new(TypeParam::max_nat()),
            },
            TypeParam::Opaque { ty: USIZE_CUSTOM_T },
            TypeParam::Tuple {
                params: vec![TypeBound::Any.into(), TypeParam::max_nat()],
            },
        ] {
            let invalid_ts =
                PolyFuncBase::new_validated([decl.clone()], body_type.clone(), &REGISTRY);
            assert_eq!(
                invalid_ts.err(),
                Some(SignatureError::TypeVarDoesNotMatchDeclaration {
                    cached: TypeBound::Copyable.into(),
                    actual: decl
                })
            );
        }
        // Variable not declared at all
        let invalid_ts = PolyFuncBase::new_validated([], body_type, &REGISTRY);
        assert_eq!(
            invalid_ts.err(),
            Some(SignatureError::FreeTypeVar {
                idx: 0,
                num_decls: 0
            })
        );

        Ok(())
    }

    fn decl_accepts_rejects_var(
        bound: TypeParam,
        accepted: &[TypeParam],
        rejected: &[TypeParam],
    ) -> Result<(), SignatureError> {
        const EXT_ID: ExtensionId = ExtensionId::new_unchecked("my_ext");
        const TYPE_NAME: SmolStr = SmolStr::new_inline("MyType");

        let mut e = Extension::new(EXT_ID);
        e.add_type(
            TYPE_NAME,
            vec![bound.clone()],
            "".into(),
            TypeDefBound::Explicit(TypeBound::Any),
        )
        .unwrap();

        let reg = ExtensionRegistry::try_new([e]).unwrap();

        let make_scheme = |tp: TypeParam| {
            PolyFuncBase::new_validated(
                [tp.clone()],
                id_fn(Type::new_extension(CustomType::new(
                    TYPE_NAME,
                    [TypeArg::new_var_use(0, tp)],
                    EXT_ID,
                    TypeBound::Any,
                ))),
                &reg,
            )
        };
        for decl in accepted {
            make_scheme(decl.clone())?;
        }
        for decl in rejected {
            assert_eq!(
                make_scheme(decl.clone()).err(),
                Some(SignatureError::TypeArgMismatch(
                    TypeArgError::TypeMismatch {
                        param: bound.clone(),
                        arg: TypeArg::new_var_use(0, decl.clone())
                    }
                ))
            );
        }
        Ok(())
    }

    #[test]
    fn test_bound_covariance() -> Result<(), SignatureError> {
        decl_accepts_rejects_var(
            TypeBound::Copyable.into(),
            &[TypeBound::Copyable.into(), TypeBound::Eq.into()],
            &[TypeBound::Any.into()],
        )?;

        let list_of_tys = |b: TypeBound| TypeParam::List {
            param: Box::new(b.into()),
        };
        decl_accepts_rejects_var(
            list_of_tys(TypeBound::Copyable),
            &[list_of_tys(TypeBound::Copyable), list_of_tys(TypeBound::Eq)],
            &[list_of_tys(TypeBound::Any)],
        )?;

        decl_accepts_rejects_var(
            TypeParam::max_nat(),
            &[TypeParam::bounded_nat(NonZeroU64::new(5).unwrap())],
            &[],
        )?;
        decl_accepts_rejects_var(
            TypeParam::bounded_nat(NonZeroU64::new(10).unwrap()),
            &[TypeParam::bounded_nat(NonZeroU64::new(5).unwrap())],
            &[TypeParam::max_nat()],
        )?;
        Ok(())
    }

    fn new_pf1(param: TypeParam, input: Type, output: Type) -> PolyFuncType {
        PolyFuncType {
            params: vec![param],
            body: FunctionType::new(vec![input], vec![output]),
        }
    }

    const USIZE_TA: TypeArg = TypeArg::Type { ty: USIZE_T };

    #[test]
    fn partial_instantiate() -> Result<(), SignatureError> {
        // forall A,N.(Array<A,N> -> A)
        let array_max = PolyFuncBase::new_validated(
            vec![TypeBound::Any.into(), TypeParam::max_nat()],
            FunctionType::new(
                vec![array_type(
                    TypeArg::new_var_use(1, TypeParam::max_nat()),
                    Type::new_var_use(0, TypeBound::Any),
                )],
                vec![Type::new_var_use(0, TypeBound::Any)],
            ),
            &PRELUDE_REGISTRY,
        )?;

        let concrete = FunctionType::new(
            vec![array_type(TypeArg::BoundedNat { n: 3 }, USIZE_T)],
            vec![USIZE_T],
        );
        let actual = array_max
            .instantiate_poly(&[USIZE_TA, TypeArg::BoundedNat { n: 3 }], &PRELUDE_REGISTRY)?;

        assert_eq!(actual, concrete.into());

        // forall N.(Array<usize,N> -> usize)
        let partial = PolyFuncBase::new_validated(
            vec![TypeParam::max_nat()],
            FunctionType::new(
                vec![array_type(
                    TypeArg::new_var_use(0, TypeParam::max_nat()),
                    USIZE_T,
                )],
                vec![USIZE_T],
            ),
            &PRELUDE_REGISTRY,
        )?;
        let res = array_max.instantiate_poly(&[USIZE_TA], &PRELUDE_REGISTRY)?;
        assert_eq!(res, partial);

        Ok(())
    }

    fn list_of_tup(t1: Type, t2: Type) -> Type {
        let list_def = EXTENSION.get_type(LIST_TYPENAME.as_str()).unwrap();
        Type::new_extension(
            list_def
                .instantiate([TypeArg::Type {
                    ty: Type::new_tuple(vec![t1, t2]),
                }])
                .unwrap(),
        )
    }

    // forall A. A -> (forall C. C -> List(Tuple(C, A))
    pub(crate) fn nested_func() -> PolyFuncType {
        PolyFuncBase::new_validated(
            vec![TypeBound::Any.into()],
            FunctionType::new(
                vec![Type::new_var_use(0, TypeBound::Any)],
                vec![Type::new_function(new_pf1(
                    TypeBound::Copyable.into(),
                    Type::new_var_use(0, TypeBound::Copyable),
                    list_of_tup(
                        Type::new_var_use(0, TypeBound::Copyable),
                        Type::new_var_use(1, TypeBound::Any), // The outer variable (renumbered)
                    ),
                ))],
            ),
            &REGISTRY,
        )
        .unwrap()
    }

    #[test]
    fn test_instantiate_nested() -> Result<(), SignatureError> {
        let outer = nested_func();

        let arg = array_type(TypeArg::BoundedNat { n: 5 }, USIZE_T);
        // `arg` -> (forall C. C -> List(Tuple(C, `arg`)))
        let outer_applied = FunctionType::new(
            vec![arg.clone()], // This had index 0, but is replaced
            vec![Type::new_function(new_pf1(
                TypeBound::Copyable.into(),
                // We are checking that the substitution has been applied to the right var
                // - NOT to the inner_var which has index 0 here
                Type::new_var_use(0, TypeBound::Copyable),
                list_of_tup(
                    Type::new_var_use(0, TypeBound::Copyable),
                    arg.clone(), // This had index 1, but is replaced
                ),
            ))],
        );

        let res = outer.instantiate(&[TypeArg::Type { ty: arg }], &REGISTRY)?;
        assert_eq!(res, outer_applied);
        Ok(())
    }

    #[test]
    fn free_var_under_binder() {
        let outer = nested_func();

        // Now substitute in a free var from further outside
        const FREE: usize = 3;
        const TP_EQ: TypeParam = TypeParam::Type { b: TypeBound::Eq };
        let res = outer
            .instantiate(&[TypeArg::new_var_use(FREE, TP_EQ)], &REGISTRY)
            .unwrap();
        assert_eq!(
            res,
            // F -> forall C. (C -> List(Tuple(C, F)))
            FunctionType::new(
                vec![Type::new_var_use(FREE, TypeBound::Eq)],
                vec![Type::new_function(new_pf1(
                    TypeBound::Copyable.into(),
                    Type::new_var_use(0, TypeBound::Copyable), // unchanged
                    list_of_tup(
                        Type::new_var_use(0, TypeBound::Copyable),
                        // Next is the free variable that we substituted in (hence Eq)
                        // - renumbered because of the intervening forall (Copyable)
                        Type::new_var_use(FREE + 1, TypeBound::Eq)
                    )
                ))]
            )
        );

        // Also try substituting in a type containing both free and bound vars
        let rhs = |i| {
            Type::new_function(new_pf1(
                TP_EQ,
                Type::new_var_use(0, TypeBound::Eq),
                array_type(
                    TypeArg::new_var_use(i, TypeParam::max_nat()),
                    Type::new_var_use(0, TypeBound::Eq),
                ),
            ))
        };

        let res = outer
            .instantiate(&[TypeArg::Type { ty: rhs(FREE) }], &REGISTRY)
            .unwrap();
        assert_eq!(
            res,
            FunctionType::new(
                vec![rhs(FREE)], // Input: forall TEQ. (TEQ -> Array(TEQ, FREE))
                // Output: forall C. C -> List(Tuple(C, Input))
                vec![Type::new_function(new_pf1(
                    TypeBound::Copyable.into(),
                    Type::new_var_use(0, TypeBound::Copyable),
                    list_of_tup(
                        Type::new_var_use(0, TypeBound::Copyable), // not renumbered...
                        rhs(FREE + 1)                              // renumbered
                    )
                ))]
            )
        )
    }

    #[test]
    fn row_variables() {
        const TP: TypeParam = TypeParam::Type { b: TypeBound::Any };
        // Mismatched TypeBound (Copyable vs Any)
        PolyFuncBase::new_validated(
            [TypeParam::List {
                param: Box::new(TypeParam::Type {
                    b: TypeBound::Copyable,
                }),
            }],
            FunctionType::new(
                vec![USIZE_T, Type::new_var_use(0, TypeBound::Any)],
                vec![Type::new_sum(vec![RowVarOrType::RV(0, TypeBound::Any)])],
            ),
            &PRELUDE_REGISTRY,
        )
        .unwrap_err();

        let pf = PolyFuncBase::new_validated(
            [TypeParam::List {
                param: Box::new(TP),
            }],
            FunctionType::new(
                vec![USIZE_T.into(), RowVarOrType::RV(0, TypeBound::Any)],
                vec![Type::new_tuple(RowVarOrType::RV(0, TypeBound::Any))],
            ),
            &PRELUDE_REGISTRY,
        )
        .unwrap();

        fn seq2() -> Vec<TypeArg> {
            vec![USIZE_T.into(), BOOL_T.into()]
        }
        pf.instantiate(&[TypeArg::Type { ty: USIZE_T }], &PRELUDE_REGISTRY)
            .unwrap_err();
        pf.instantiate(
            &[TypeArg::Sequence {
                elems: vec![USIZE_T.into(), TypeArg::Sequence { elems: seq2() }],
            }],
            &PRELUDE_REGISTRY,
        )
        .unwrap_err();

        let t2 = pf
            .instantiate(&[TypeArg::Sequence { elems: seq2() }], &PRELUDE_REGISTRY)
            .unwrap();
        assert_eq!(
            t2,
            FunctionType::new(
                vec![USIZE_T, USIZE_T, BOOL_T],
                vec![Type::new_tuple(vec![
                    RowVarOrType::T(USIZE_T),
                    BOOL_T.into()
                ])]
            )
        );
    }

    #[test]
    fn row_variables_inner() {
        let inner_fty = Type::new_function(FunctionType::new_endo(vec![RowVarOrType::RV(
            0,
            TypeBound::Copyable,
        )]));
        let pf = PolyFuncBase::new_validated(
            [TypeParam::List {
                param: Box::new(TypeParam::Type {
                    b: TypeBound::Copyable,
                }),
            }],
            FunctionType::new(vec![USIZE_T, inner_fty.clone()], vec![inner_fty]),
            &PRELUDE_REGISTRY,
        )
        .unwrap();

        let inner3 = Type::new_function(FunctionType::new_endo(vec![USIZE_T, BOOL_T, USIZE_T]));
        let t3 = pf
            .instantiate(
                &[TypeArg::Sequence {
                    elems: vec![USIZE_T.into(), BOOL_T.into(), USIZE_T.into()],
                }],
                &PRELUDE_REGISTRY,
            )
            .unwrap();
        assert_eq!(
            t3,
            FunctionType::new(vec![USIZE_T, inner3.clone()], vec![inner3])
        );
    }
}
