//! Polymorphic Function Types

use crate::extension::{ExtensionRegistry, SignatureError};
use itertools::Itertools;
#[cfg(test)]
use {
    crate::proptest::RecursionDepth,
    ::proptest::{collection::vec, prelude::*},
    proptest_derive::Arbitrary,
};

use super::signature::{FuncTypeBase, FunctionType};
use super::type_param::{check_type_args, TypeArg, TypeParam};
use super::Substitution;

/// A polymorphic type scheme, i.e. of a [FuncDecl], [FuncDefn] or [OpDef].
/// (Nodes/operations in the Hugr are not polymorphic.)
///
/// [FuncDecl]: crate::ops::module::FuncDecl
/// [FuncDefn]: crate::ops::module::FuncDefn
/// [OpDef]: crate::extension::OpDef
#[derive(
    Clone, PartialEq, Debug, Default, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(test, derive(Arbitrary), proptest(params = "RecursionDepth"))]
#[display(
    fmt = "forall {}. {}",
    "params.iter().map(ToString::to_string).join(\" \")",
    "body"
)]
pub struct PolyFuncType<const ROWVARS: bool = true> {
    /// The declared type parameters, i.e., these must be instantiated with
    /// the same number of [TypeArg]s before the function can be called. This
    /// defines the indices used by variables inside the body.
    #[cfg_attr(test, proptest(strategy = "vec(any_with::<TypeParam>(params), 0..3)"))]
    params: Vec<TypeParam>,
    /// Template for the function. May contain variables up to length of [Self::params]
    #[cfg_attr(test, proptest(strategy = "any_with::<FunctionType>(params)"))]
    body: FuncTypeBase<ROWVARS>,
}

impl<const RV: bool> From<FuncTypeBase<RV>> for PolyFuncType<RV> {
    fn from(body: FuncTypeBase<RV>) -> Self {
        Self {
            params: vec![],
            body
        }
    }
}

impl From<PolyFuncType<false>> for PolyFuncType<true> {
    fn from(value: PolyFuncType<false>) -> Self {
        Self {
            params: value.params,
            body: value.body.into_()
        }
    }
}

impl<const RV: bool> TryFrom<PolyFuncType<RV>> for FuncTypeBase<RV> {
    /// If the PolyFuncType is not monomorphic, fail with its binders
    type Error = Vec<TypeParam>;

    fn try_from(value: PolyFuncType<RV>) -> Result<Self, Self::Error> {
        if value.params.is_empty() {
            Ok(value.body)
        } else {
            Err(value.params)
        }
    }
}

impl<const RV: bool> PolyFuncType<RV> {
    /// The type parameters, aka binders, over which this type is polymorphic
    pub fn params(&self) -> &[TypeParam] {
        &self.params
    }

    /// The body of the type, a function type.
    pub fn body(&self) -> &FuncTypeBase<RV> {
        &self.body
    }

    /// Create a new PolyFuncType given the kinds of the variables it declares
    /// and the underlying function type.
    pub fn new(params: impl Into<Vec<TypeParam>>, body: impl Into<FuncTypeBase<RV>>) -> Self {
        Self {
            params: params.into(),
            body: body.into()
        }
    }

    /// Instantiates an outer [PolyFuncType], i.e. with no free variables
    /// (as ensured by [Self::validate]), into a monomorphic type.
    ///
    /// # Errors
    /// If there is not exactly one [TypeArg] for each binder ([Self::params]),
    /// or an arg does not fit into its corresponding [TypeParam]
    pub(crate) fn instantiate(
        &self,
        args: &[TypeArg],
        ext_reg: &ExtensionRegistry,
    ) -> Result<FuncTypeBase<RV>, SignatureError> {
        // Check that args are applicable, and that we have a value for each binder,
        // i.e. each possible free variable within the body.
        check_type_args(args, &self.params)?;
        Ok(self.body.substitute(&Substitution(args, ext_reg)))
    }

    /// Validates this instance, checking that the types in the body are
    /// wellformed with respect to the registry, and the type variables declared.
    /// Allows both inputs and outputs to contain [RowVariable]s
    ///
    /// [RowVariable]: [crate::types::TypeEnum::RowVariable]
    pub fn validate(&self, reg: &ExtensionRegistry) -> Result<(), SignatureError> {
        // TODO https://github.com/CQCL/hugr/issues/624 validate TypeParams declared here, too
        self.body.validate(reg, &self.params)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::num::NonZeroU64;

    use cool_asserts::assert_matches;
    use lazy_static::lazy_static;

    use crate::extension::prelude::{BOOL_T, PRELUDE_ID, USIZE_CUSTOM_T, USIZE_T};
    use crate::extension::{
        ExtensionId, ExtensionRegistry, SignatureError, TypeDefBound, EMPTY_REG, PRELUDE,
        PRELUDE_REGISTRY,
    };
    use crate::std_extensions::collections::{EXTENSION, LIST_TYPENAME};
    use crate::types::type_param::{TypeArg, TypeArgError, TypeParam};
    use crate::types::{CustomType, FunctionType, Type, TypeBound, TypeName};
    use crate::Extension;

    use super::PolyFuncType;

    lazy_static! {
        static ref REGISTRY: ExtensionRegistry =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), EXTENSION.to_owned()]).unwrap();
    }

    impl PolyFuncType {
        fn new_validated(
            params: impl Into<Vec<TypeParam>>,
            body: FunctionType,
            extension_registry: &ExtensionRegistry,
        ) -> Result<Self, SignatureError> {
            let res = Self::new(params, body);
            res.validate(extension_registry)?;
            Ok(res)
        }
    }

    #[test]
    fn test_opaque() -> Result<(), SignatureError> {
        let list_def = EXTENSION.get_type(&LIST_TYPENAME).unwrap();
        let tyvar = TypeArg::new_var_use(0, TypeBound::Any.into());
        let list_of_var = Type::new_extension(list_def.instantiate([tyvar.clone()])?);
        let list_len = PolyFuncType::new_validated(
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

    #[test]
    fn test_mismatched_args() -> Result<(), SignatureError> {
        let ar_def = PRELUDE.get_type("array").unwrap();
        let typarams = [TypeParam::max_nat(), TypeBound::Any.into()];
        let [tyvar, szvar] =
            [0, 1].map(|i| TypeArg::new_var_use(i, typarams.get(i).unwrap().clone()));

        // Valid schema...
        let good_array = Type::new_extension(ar_def.instantiate([tyvar.clone(), szvar.clone()])?);
        let good_ts = PolyFuncType::new_validated(
            typarams.clone(),
            FunctionType::new_endo(good_array),
            &PRELUDE_REGISTRY,
        )?;

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
        let bad_ts = PolyFuncType::new_validated(
            typarams.clone(),
            FunctionType::new_endo(bad_array),
            &PRELUDE_REGISTRY,
        );
        assert_eq!(bad_ts.err(), Some(arg_err));

        Ok(())
    }

    #[test]
    fn test_misused_variables() -> Result<(), SignatureError> {
        // Variables in args have different bounds from variable declaration
        let tv = TypeArg::new_var_use(0, TypeBound::Copyable.into());
        let list_def = EXTENSION.get_type(&LIST_TYPENAME).unwrap();
        let body_type = FunctionType::new_endo(Type::new_extension(list_def.instantiate([tv])?));
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
                PolyFuncType::new_validated([decl.clone()], body_type.clone(), &REGISTRY);
            assert_eq!(
                invalid_ts.err(),
                Some(SignatureError::TypeVarDoesNotMatchDeclaration {
                    cached: TypeBound::Copyable.into(),
                    actual: decl
                })
            );
        }
        // Variable not declared at all
        let invalid_ts = PolyFuncType::new_validated([], body_type, &REGISTRY);
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
        const TYPE_NAME: TypeName = TypeName::new_inline("MyType");

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
            PolyFuncType::new_validated(
                [tp.clone()],
                FunctionType::new_endo(Type::new_extension(CustomType::new(
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

    const TP_ANY: TypeParam = TypeParam::Type { b: TypeBound::Any };
    #[test]
    fn row_variables_bad_schema() {
        // Mismatched TypeBound (Copyable vs Any)
        let decl = TypeParam::List {
            param: Box::new(TP_ANY),
        };
        let e = PolyFuncType::new_validated(
            [decl.clone()],
            FunctionType::new(
                vec![USIZE_T],
                vec![Type::new_row_var_use(0, TypeBound::Copyable)],
            ),
            &PRELUDE_REGISTRY,
        )
        .unwrap_err();
        assert_matches!(e, SignatureError::TypeVarDoesNotMatchDeclaration { actual, cached } => {
            assert_eq!(actual, decl);
            assert_eq!(cached, TypeParam::List {param: Box::new(TypeParam::Type {b: TypeBound::Copyable})});
        });
        // Declared as row variable, used as type variable
        let e = PolyFuncType::new_validated(
            [decl.clone()],
            FunctionType::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
            &EMPTY_REG,
        )
        .unwrap_err();
        assert_matches!(e, SignatureError::TypeVarDoesNotMatchDeclaration { actual, cached } => {
            assert_eq!(actual, decl);
            assert_eq!(cached, TP_ANY);
        });
    }

    #[test]
    fn row_variables() {
        let rty = Type::new_row_var_use(0, TypeBound::Any);
        let pf = PolyFuncType::new_validated(
            [TypeParam::new_list(TP_ANY)],
            FunctionType::new(vec![USIZE_T, rty.clone()], vec![Type::new_tuple(rty)]),
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
                vec![Type::new_tuple(vec![USIZE_T, BOOL_T])]
            )
        );
    }

    #[test]
    fn row_variables_inner() {
        let inner_fty = Type::new_function(FunctionType::new_endo(vec![Type::new_row_var_use(
            0,
            TypeBound::Copyable,
        )]));
        let pf = PolyFuncType::new_validated(
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
