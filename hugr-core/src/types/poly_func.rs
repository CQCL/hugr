//! Polymorphic Function Types

use std::borrow::Cow;

use itertools::Itertools;

use crate::extension::SignatureError;
#[cfg(test)]
use {
    super::proptest_utils::any_serde_type_param,
    crate::proptest::RecursionDepth,
    ::proptest::{collection::vec, prelude::*},
    proptest_derive::Arbitrary,
};

use super::Substitution;
use super::type_param::{TypeArg, TypeParam, check_term_types};
use super::{MaybeRV, NoRV, RowVariable, signature::FuncTypeBase};

/// A polymorphic type scheme, i.e. of a [`FuncDecl`], [`FuncDefn`] or [`OpDef`].
/// (Nodes/operations in the Hugr are not polymorphic.)
///
/// [`FuncDecl`]: crate::ops::module::FuncDecl
/// [`FuncDefn`]: crate::ops::module::FuncDefn
/// [`OpDef`]: crate::extension::OpDef
#[derive(
    Clone, PartialEq, Debug, Eq, Hash, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(test, derive(Arbitrary), proptest(params = "RecursionDepth"))]
#[display("{}{body}", self.display_params())]
pub struct PolyFuncTypeBase<RV: MaybeRV> {
    /// The declared type parameters, i.e., these must be instantiated with
    /// the same number of [`TypeArg`]s before the function can be called. This
    /// defines the indices used by variables inside the body.
    #[cfg_attr(test, proptest(strategy = "vec(any_serde_type_param(params), 0..3)"))]
    params: Vec<TypeParam>,
    /// Template for the function. May contain variables up to length of [`Self::params`]
    #[cfg_attr(test, proptest(strategy = "any_with::<FuncTypeBase<RV>>(params)"))]
    body: FuncTypeBase<RV>,
}

/// The polymorphic type of a [`Call`]-able function ([`FuncDecl`] or [`FuncDefn`]).
/// Number of inputs and outputs fixed.
///
/// [`Call`]: crate::ops::Call
/// [`FuncDefn`]: crate::ops::FuncDefn
/// [`FuncDecl`]: crate::ops::FuncDecl
pub type PolyFuncType = PolyFuncTypeBase<NoRV>;

/// The polymorphic type of an [`OpDef`], whose number of input and outputs
/// may vary according to how [`RowVariable`]s therein are instantiated.
///
/// [`OpDef`]: crate::extension::OpDef
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
    /// If the `PolyFuncTypeBase` is not monomorphic, fail with its binders
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

    /// Create a new `PolyFuncTypeBase` given the kinds of the variables it declares
    /// and the underlying [`FuncTypeBase`].
    pub fn new(params: impl Into<Vec<TypeParam>>, body: impl Into<FuncTypeBase<RV>>) -> Self {
        Self {
            params: params.into(),
            body: body.into(),
        }
    }

    /// Instantiates an outer [`PolyFuncTypeBase`], i.e. with no free variables
    /// (as ensured by [`Self::validate`]), into a monomorphic type.
    ///
    /// # Errors
    /// If there is not exactly one [`TypeArg`] for each binder ([`Self::params`]),
    /// or an arg does not fit into its corresponding [`TypeParam`]
    pub fn instantiate(&self, args: &[TypeArg]) -> Result<FuncTypeBase<RV>, SignatureError> {
        // Check that args are applicable, and that we have a value for each binder,
        // i.e. each possible free variable within the body.
        check_term_types(args, &self.params)?;
        Ok(self.body.substitute(&Substitution(args)))
    }

    /// Validates this instance, checking that the types in the body are
    /// wellformed with respect to the registry, and the type variables declared.
    pub fn validate(&self) -> Result<(), SignatureError> {
        self.body.validate(&self.params)
    }

    /// Helper function for the Display implementation
    fn display_params(&self) -> Cow<'static, str> {
        if self.params.is_empty() {
            return Cow::Borrowed("");
        }
        let params_list = self
            .params
            .iter()
            .enumerate()
            .map(|(i, param)| format!("(#{i} : {param})"))
            .join(" ");
        Cow::Owned(format!("∀ {params_list}. ",))
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

    use crate::Extension;
    use crate::extension::prelude::{bool_t, usize_t};
    use crate::extension::{ExtensionId, ExtensionRegistry, SignatureError, TypeDefBound};
    use crate::std_extensions::collections::array::{self, array_type_parametric};
    use crate::std_extensions::collections::list;
    use crate::types::signature::FuncTypeBase;
    use crate::types::type_param::{TermTypeError, TypeArg, TypeParam};
    use crate::types::{
        CustomType, FuncValueType, MaybeRV, Signature, Term, Type, TypeBound, TypeName, TypeRV,
    };

    use super::PolyFuncTypeBase;

    impl<RV: MaybeRV> PolyFuncTypeBase<RV> {
        fn new_validated(
            params: impl Into<Vec<TypeParam>>,
            body: FuncTypeBase<RV>,
        ) -> Result<Self, SignatureError> {
            let res = Self::new(params, body);
            res.validate()?;
            Ok(res)
        }
    }

    #[test]
    fn test_opaque() -> Result<(), SignatureError> {
        let list_def = list::EXTENSION.get_type(&list::LIST_TYPENAME).unwrap();
        let tyvar = TypeArg::new_var_use(0, TypeBound::Linear.into());
        let list_of_var = Type::new_extension(list_def.instantiate([tyvar.clone()])?);
        let list_len = PolyFuncTypeBase::new_validated(
            [TypeBound::Linear.into()],
            Signature::new(vec![list_of_var], vec![usize_t()]),
        )?;

        let t = list_len.instantiate(&[usize_t().into()])?;
        assert_eq!(
            t,
            Signature::new(
                vec![Type::new_extension(
                    list_def.instantiate([usize_t().into()]).unwrap()
                )],
                vec![usize_t()]
            )
        );

        Ok(())
    }

    #[test]
    fn test_mismatched_args() -> Result<(), SignatureError> {
        let size_var = TypeArg::new_var_use(0, TypeParam::max_nat_type());
        let ty_var = TypeArg::new_var_use(1, TypeBound::Linear.into());
        let type_params = [TypeParam::max_nat_type(), TypeBound::Linear.into()];

        // Valid schema...
        let good_array = array_type_parametric(size_var.clone(), ty_var.clone())?;
        let good_ts =
            PolyFuncTypeBase::new_validated(type_params.clone(), Signature::new_endo(good_array))?;

        // Sanity check (good args)
        good_ts.instantiate(&[5u64.into(), usize_t().into()])?;

        let wrong_args = good_ts.instantiate(&[usize_t().into(), 5u64.into()]);
        assert_eq!(
            wrong_args,
            Err(SignatureError::TypeArgMismatch(
                TermTypeError::TypeMismatch {
                    type_: Box::new(type_params[0].clone()),
                    term: Box::new(usize_t().into()),
                }
            ))
        );

        // (Try to) make a schema with the args in the wrong order
        let arg_err = SignatureError::TypeArgMismatch(TermTypeError::TypeMismatch {
            type_: Box::new(type_params[0].clone()),
            term: Box::new(ty_var.clone()),
        });
        assert_eq!(
            array_type_parametric(ty_var.clone(), size_var.clone()),
            Err(arg_err.clone())
        );
        // ok, so that doesn't work - well, it shouldn't! So let's say we just have this signature (with bad args)...
        let bad_array = Type::new_extension(CustomType::new(
            "array",
            [ty_var, size_var],
            array::EXTENSION_ID,
            TypeBound::Linear,
            &Arc::downgrade(&array::EXTENSION),
        ));
        let bad_ts =
            PolyFuncTypeBase::new_validated(type_params.clone(), Signature::new_endo(bad_array));
        assert_eq!(bad_ts.err(), Some(arg_err));

        Ok(())
    }

    #[test]
    fn test_misused_variables() -> Result<(), SignatureError> {
        // Variables in args have different bounds from variable declaration
        let tv = TypeArg::new_var_use(0, TypeBound::Copyable.into());
        let list_def = list::EXTENSION.get_type(&list::LIST_TYPENAME).unwrap();
        let body_type = Signature::new_endo(Type::new_extension(list_def.instantiate([tv])?));
        for decl in [
            Term::new_list_type(Term::max_nat_type()),
            Term::StringType,
            Term::new_tuple_type([TypeBound::Linear.into(), Term::max_nat_type()]),
        ] {
            let invalid_ts = PolyFuncTypeBase::new_validated([decl.clone()], body_type.clone());
            assert_eq!(
                invalid_ts.err(),
                Some(SignatureError::TypeVarDoesNotMatchDeclaration {
                    cached: Box::new(TypeBound::Copyable.into()),
                    actual: Box::new(decl)
                })
            );
        }
        // Variable not declared at all
        let invalid_ts = PolyFuncTypeBase::new_validated([], body_type);
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
                String::new(),
                TypeDefBound::any(),
                extension_ref,
            )
            .unwrap();
        });

        let reg = ExtensionRegistry::new([ext.clone()]);
        reg.validate().unwrap();

        let make_scheme = |tp: TypeParam| {
            PolyFuncTypeBase::new_validated(
                [tp.clone()],
                Signature::new_endo(Type::new_extension(CustomType::new(
                    TYPE_NAME,
                    [TypeArg::new_var_use(0, tp)],
                    EXT_ID,
                    TypeBound::Linear,
                    &Arc::downgrade(&ext),
                ))),
            )
        };
        for decl in accepted {
            make_scheme(decl.clone())?;
        }
        for decl in rejected {
            assert_eq!(
                make_scheme(decl.clone()).err(),
                Some(SignatureError::TypeArgMismatch(
                    TermTypeError::TypeMismatch {
                        type_: Box::new(bound.clone()),
                        term: Box::new(TypeArg::new_var_use(0, decl.clone()))
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
            &[TypeBound::Linear.into()],
        )?;

        decl_accepts_rejects_var(
            Term::new_list_type(TypeBound::Copyable),
            &[Term::new_list_type(TypeBound::Copyable)],
            &[Term::new_list_type(TypeBound::Linear)],
        )?;

        decl_accepts_rejects_var(
            TypeParam::max_nat_type(),
            &[TypeParam::bounded_nat_type(NonZeroU64::new(5).unwrap())],
            &[],
        )?;
        decl_accepts_rejects_var(
            TypeParam::bounded_nat_type(NonZeroU64::new(10).unwrap()),
            &[TypeParam::bounded_nat_type(NonZeroU64::new(5).unwrap())],
            &[TypeParam::max_nat_type()],
        )?;
        Ok(())
    }

    const TP_ANY: TypeParam = TypeParam::RuntimeType(TypeBound::Linear);
    #[test]
    fn row_variables_bad_schema() {
        // Mismatched TypeBound (Copyable vs Any)
        let decl = Term::new_list_type(TP_ANY);
        let e = PolyFuncTypeBase::new_validated(
            [decl.clone()],
            FuncValueType::new(
                vec![usize_t()],
                vec![TypeRV::new_row_var_use(0, TypeBound::Copyable)],
            ),
        )
        .unwrap_err();
        assert_matches!(e, SignatureError::TypeVarDoesNotMatchDeclaration { actual, cached } => {
            assert_eq!(*actual, decl);
            assert_eq!(*cached, TypeParam::new_list_type(TypeBound::Copyable));
        });
        // Declared as row variable, used as type variable
        let e = PolyFuncTypeBase::new_validated(
            [decl.clone()],
            Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Linear)]),
        )
        .unwrap_err();
        assert_matches!(e, SignatureError::TypeVarDoesNotMatchDeclaration { actual, cached } => {
            assert_eq!(*actual, decl);
            assert_eq!(*cached, TP_ANY);
        });
    }

    #[test]
    fn row_variables() {
        let rty = TypeRV::new_row_var_use(0, TypeBound::Linear);
        let pf = PolyFuncTypeBase::new_validated(
            [TypeParam::new_list_type(TP_ANY)],
            FuncValueType::new(
                vec![usize_t().into(), rty.clone()],
                vec![TypeRV::new_tuple(rty)],
            ),
        )
        .unwrap();

        fn seq2() -> Vec<TypeArg> {
            vec![usize_t().into(), bool_t().into()]
        }
        pf.instantiate(&[usize_t().into()]).unwrap_err();
        pf.instantiate(&[Term::new_list([usize_t().into(), Term::new_list(seq2())])])
            .unwrap_err();

        let t2 = pf.instantiate(&[Term::new_list(seq2())]).unwrap();
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
            [Term::new_list_type(TypeBound::Copyable)],
            Signature::new(vec![usize_t(), inner_fty.clone()], vec![inner_fty]),
        )
        .unwrap();

        let inner3 = Type::new_function(Signature::new_endo(vec![usize_t(), bool_t(), usize_t()]));
        let t3 = pf
            .instantiate(&[Term::new_list([
                usize_t().into(),
                bool_t().into(),
                usize_t().into(),
            ])])
            .unwrap();
        assert_eq!(
            t3,
            Signature::new(vec![usize_t(), inner3.clone()], vec![inner3])
        );
    }
}
