//! Polymorphic Function Types

use itertools::Itertools;

use crate::extension::{ExtensionRegistry, SignatureError};
#[cfg(test)]
use {
    crate::proptest::RecursionDepth,
    ::proptest::{collection::vec, prelude::*},
    proptest_derive::Arbitrary,
};

use super::type_param::{check_type_args, TypeArg, TypeParam};
use super::Substitution;
use super::{signature::FuncTypeBase, MaybeRV, NoRV, RowVariable};

/// A polymorphic type scheme, i.e. of a [FuncDecl], [FuncDefn] or [OpDef].
/// (Nodes/operations in the Hugr are not polymorphic.)
///
/// [FuncDecl]: crate::ops::module::FuncDecl
/// [FuncDefn]: crate::ops::module::FuncDefn
/// [OpDef]: crate::extension::OpDef
#[derive(
    Clone, PartialEq, Debug, Eq, Hash, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(test, derive(Arbitrary), proptest(params = "RecursionDepth"))]
#[display("{}{body}", self.display_params())]
pub struct PolyFuncTypeBase<RV: MaybeRV> {
    /// The declared type parameters, i.e., these must be instantiated with
    /// the same number of [TypeArg]s before the function can be called. This
    /// defines the indices used by variables inside the body.
    #[cfg_attr(test, proptest(strategy = "vec(any_with::<TypeParam>(params), 0..3)"))]
    params: Vec<TypeParam>,
    /// Template for the function. May contain variables up to length of [Self::params]
    #[cfg_attr(test, proptest(strategy = "any_with::<FuncTypeBase<RV>>(params)"))]
    body: FuncTypeBase<RV>,
}

/// The polymorphic type of a [Call]-able function ([FuncDecl] or [FuncDefn]).
/// Number of inputs and outputs fixed.
///
/// [Call]: crate::ops::Call
/// [FuncDefn]: crate::ops::FuncDefn
/// [FuncDecl]: crate::ops::FuncDecl
pub type PolyFuncType = PolyFuncTypeBase<NoRV>;

/// The polymorphic type of an [OpDef], whose number of input and outputs
/// may vary according to how [RowVariable]s therein are instantiated.
///
/// [OpDef]: crate::extension::OpDef
pub type PolyFuncTypeRV = PolyFuncTypeBase<RowVariable>;

// deriving Default leads to an impl that only applies for RV: Default
impl<RV: MaybeRV> Default for PolyFuncTypeBase<RV> {
    fn default() -> Self {
        Self {
            params: Default::default(),
            body: Default::default(),
        }
    }
}

impl<RV: MaybeRV> From<FuncTypeBase<RV>> for PolyFuncTypeBase<RV> {
    fn from(body: FuncTypeBase<RV>) -> Self {
        Self {
            params: vec![],
            body,
        }
    }
}

impl From<PolyFuncType> for PolyFuncTypeRV {
    fn from(value: PolyFuncType) -> Self {
        Self {
            params: value.params,
            body: value.body.into(),
        }
    }
}

impl<RV: MaybeRV> TryFrom<PolyFuncTypeBase<RV>> for FuncTypeBase<RV> {
    /// If the PolyFuncTypeBase is not monomorphic, fail with its binders
    type Error = Vec<TypeParam>;

    fn try_from(value: PolyFuncTypeBase<RV>) -> Result<Self, Self::Error> {
        if value.params.is_empty() {
            Ok(value.body)
        } else {
            Err(value.params)
        }
    }
}

impl<RV: MaybeRV> PolyFuncTypeBase<RV> {
    /// The type parameters, aka binders, over which this type is polymorphic
    pub fn params(&self) -> &[TypeParam] {
        &self.params
    }

    /// The body of the type, a function type.
    pub fn body(&self) -> &FuncTypeBase<RV> {
        &self.body
    }

    /// Create a new PolyFuncTypeBase given the kinds of the variables it declares
    /// and the underlying [FuncTypeBase].
    pub fn new(params: impl Into<Vec<TypeParam>>, body: impl Into<FuncTypeBase<RV>>) -> Self {
        Self {
            params: params.into(),
            body: body.into(),
        }
    }

    /// Instantiates an outer [PolyFuncTypeBase], i.e. with no free variables
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
    pub fn validate(&self, reg: &ExtensionRegistry) -> Result<(), SignatureError> {
        // TODO https://github.com/CQCL/hugr/issues/624 validate TypeParams declared here, too
        self.body.validate(reg, &self.params)
    }

    /// Helper function for the Display implementation
    fn display_params(&self) -> String {
        if self.params.is_empty() {
            return String::new();
        }
        format!(
            "forall {}. ",
            self.params.iter().map(ToString::to_string).join(" ")
        )
    }

    /// Returns a mutable reference to the body of the function type.
    pub fn body_mut(&mut self) -> &mut FuncTypeBase<RV> {
        &mut self.body
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use cool_asserts::assert_matches;
    use lazy_static::lazy_static;

    use crate::extension::prelude::{bool_t, usize_t, PRELUDE_ID};
    use crate::extension::{
        ExtensionId, ExtensionRegistry, SignatureError, TypeDefBound, EMPTY_REG, PRELUDE,
        PRELUDE_REGISTRY,
    };
    use crate::std_extensions::collections::{EXTENSION, LIST_TYPENAME};
    use crate::types::signature::FuncTypeBase;
    use crate::types::type_param::{TypeArg, TypeArgError, TypeParam};
    use crate::types::{
        CustomType, FuncValueType, MaybeRV, Signature, Type, TypeBound, TypeName, TypeRV,
    };
    use crate::Extension;

    use super::PolyFuncTypeBase;

    lazy_static! {
        static ref REGISTRY: ExtensionRegistry =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), EXTENSION.to_owned()]).unwrap();
    }

    impl<RV: MaybeRV> PolyFuncTypeBase<RV> {
        fn new_validated(
            params: impl Into<Vec<TypeParam>>,
            body: FuncTypeBase<RV>,
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
        let list_len = PolyFuncTypeBase::new_validated(
            [TypeBound::Any.into()],
            Signature::new(vec![list_of_var], vec![usize_t()]),
            &REGISTRY,
        )?;

        let t = list_len.instantiate(&[TypeArg::Type { ty: usize_t() }], &REGISTRY)?;
        assert_eq!(
            t,
            Signature::new(
                vec![Type::new_extension(
                    list_def
                        .instantiate([TypeArg::Type { ty: usize_t() }])
                        .unwrap()
                )],
                vec![usize_t()]
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
        let good_ts = PolyFuncTypeBase::new_validated(
            typarams.clone(),
            Signature::new_endo(good_array),
            &PRELUDE_REGISTRY,
        )?;

        // Sanity check (good args)
        good_ts.instantiate(
            &[
                TypeArg::BoundedNat { n: 5 },
                TypeArg::Type { ty: usize_t() },
            ],
            &PRELUDE_REGISTRY,
        )?;

        let wrong_args = good_ts.instantiate(
            &[
                TypeArg::Type { ty: usize_t() },
                TypeArg::BoundedNat { n: 5 },
            ],
            &PRELUDE_REGISTRY,
        );
        assert_eq!(
            wrong_args,
            Err(SignatureError::TypeArgMismatch(
                TypeArgError::TypeMismatch {
                    param: typarams[0].clone(),
                    arg: TypeArg::Type { ty: usize_t() }
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
            &Arc::downgrade(&PRELUDE),
        ));
        let bad_ts = PolyFuncTypeBase::new_validated(
            typarams.clone(),
            Signature::new_endo(bad_array),
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
        let body_type = Signature::new_endo(Type::new_extension(list_def.instantiate([tv])?));
        for decl in [
            TypeParam::Extensions,
            TypeParam::List {
                param: Box::new(TypeParam::max_nat()),
            },
            TypeParam::String,
            TypeParam::Tuple {
                params: vec![TypeBound::Any.into(), TypeParam::max_nat()],
            },
        ] {
            let invalid_ts =
                PolyFuncTypeBase::new_validated([decl.clone()], body_type.clone(), &REGISTRY);
            assert_eq!(
                invalid_ts.err(),
                Some(SignatureError::TypeVarDoesNotMatchDeclaration {
                    cached: TypeBound::Copyable.into(),
                    actual: decl
                })
            );
        }
        // Variable not declared at all
        let invalid_ts = PolyFuncTypeBase::new_validated([], body_type, &REGISTRY);
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

        let ext = Extension::new_test_arc(EXT_ID, |ext, extension_ref| {
            ext.add_type(
                TYPE_NAME,
                vec![bound.clone()],
                "".into(),
                TypeDefBound::any(),
                extension_ref,
            )
            .unwrap();
        });

        let reg = ExtensionRegistry::try_new([ext.clone()]).unwrap();

        let make_scheme = |tp: TypeParam| {
            PolyFuncTypeBase::new_validated(
                [tp.clone()],
                Signature::new_endo(Type::new_extension(CustomType::new(
                    TYPE_NAME,
                    [TypeArg::new_var_use(0, tp)],
                    EXT_ID,
                    TypeBound::Any,
                    &Arc::downgrade(&ext),
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
            &[TypeBound::Copyable.into()],
            &[TypeBound::Any.into()],
        )?;

        let list_of_tys = |b: TypeBound| TypeParam::List {
            param: Box::new(b.into()),
        };
        decl_accepts_rejects_var(
            list_of_tys(TypeBound::Copyable),
            &[list_of_tys(TypeBound::Copyable)],
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
        let e = PolyFuncTypeBase::new_validated(
            [decl.clone()],
            FuncValueType::new(
                vec![usize_t()],
                vec![TypeRV::new_row_var_use(0, TypeBound::Copyable)],
            ),
            &PRELUDE_REGISTRY,
        )
        .unwrap_err();
        assert_matches!(e, SignatureError::TypeVarDoesNotMatchDeclaration { actual, cached } => {
            assert_eq!(actual, decl);
            assert_eq!(cached, TypeParam::List {param: Box::new(TypeParam::Type {b: TypeBound::Copyable})});
        });
        // Declared as row variable, used as type variable
        let e = PolyFuncTypeBase::new_validated(
            [decl.clone()],
            Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
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
        let rty = TypeRV::new_row_var_use(0, TypeBound::Any);
        let pf = PolyFuncTypeBase::new_validated(
            [TypeParam::new_list(TP_ANY)],
            FuncValueType::new(
                vec![usize_t().into(), rty.clone()],
                vec![TypeRV::new_tuple(rty)],
            ),
            &PRELUDE_REGISTRY,
        )
        .unwrap();

        fn seq2() -> Vec<TypeArg> {
            vec![usize_t().into(), bool_t().into()]
        }
        pf.instantiate(&[TypeArg::Type { ty: usize_t() }], &PRELUDE_REGISTRY)
            .unwrap_err();
        pf.instantiate(
            &[TypeArg::Sequence {
                elems: vec![usize_t().into(), TypeArg::Sequence { elems: seq2() }],
            }],
            &PRELUDE_REGISTRY,
        )
        .unwrap_err();

        let t2 = pf
            .instantiate(&[TypeArg::Sequence { elems: seq2() }], &PRELUDE_REGISTRY)
            .unwrap();
        assert_eq!(
            t2,
            Signature::new(
                vec![usize_t(), usize_t(), bool_t()],
                vec![Type::new_tuple(vec![usize_t(), bool_t()])]
            )
        );
    }

    #[test]
    fn row_variables_inner() {
        let inner_fty = Type::new_function(FuncValueType::new_endo(TypeRV::new_row_var_use(
            0,
            TypeBound::Copyable,
        )));
        let pf = PolyFuncTypeBase::new_validated(
            [TypeParam::List {
                param: Box::new(TypeParam::Type {
                    b: TypeBound::Copyable,
                }),
            }],
            Signature::new(vec![usize_t(), inner_fty.clone()], vec![inner_fty]),
            &PRELUDE_REGISTRY,
        )
        .unwrap();

        let inner3 = Type::new_function(Signature::new_endo(vec![usize_t(), bool_t(), usize_t()]));
        let t3 = pf
            .instantiate(
                &[TypeArg::Sequence {
                    elems: vec![usize_t().into(), bool_t().into(), usize_t().into()],
                }],
                &PRELUDE_REGISTRY,
            )
            .unwrap();
        assert_eq!(
            t3,
            Signature::new(vec![usize_t(), inner3.clone()], vec![inner3])
        );
    }
}
