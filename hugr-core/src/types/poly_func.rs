//! Polymorphic Function Types

use itertools::Itertools;
use std::borrow::Cow;

use crate::extension::SignatureError;
#[cfg(test)]
use {
    crate::proptest::RecursionDepth,
    ::proptest::{collection::vec, prelude::*},
};

use super::{
    FuncValueType, Signature, Substitution, Term,
    type_param::{TypeParam, check_term_types},
};

/// A polymorphic object.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    Default,
    serde::Serialize,
    serde::Deserialize,
    derive_more::Display,
)]
#[display("{}{body}", self.display_params())]
pub struct Polymorphic<T> {
    params: Vec<Term>,
    body: T,
}

impl<T> From<T> for Polymorphic<T> {
    fn from(body: T) -> Self {
        Self::new_mono(body)
    }
}

impl<T> Polymorphic<T> {
    /// Create a new polymorphic `T` given the types of its parameters.
    pub fn new(params: impl IntoIterator<Item = Term>, body: T) -> Self {
        Self {
            params: params.into_iter().collect(),
            body: body.into(),
        }
    }

    /// Create a new monomorphic
    pub fn new_mono(body: T) -> Self {
        Self {
            params: Default::default(),
            body,
        }
    }

    /// Returns the `T` if there are no parameters.
    pub fn into_mono(self) -> Option<T> {
        if self.params.is_empty() {
            Some(self.body)
        } else {
            None
        }
    }

    /// Returns the types of the parameters.
    pub fn params(&self) -> &[TypeParam] {
        &self.params
    }

    /// Returns the body of the polymorphic object.
    pub fn body(&self) -> &T {
        &self.body
    }

    /// Returns a mutable reference to the body of the polymorphic object.
    pub fn body_mut(&mut self) -> &mut T {
        &mut self.body
    }

    /// Convert the body of the polymorphic object.
    pub fn map_into<S>(self) -> Polymorphic<S>
    where
        S: From<T>,
    {
        Polymorphic {
            params: self.params,
            body: self.body.into(),
        }
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
}

impl Polymorphic<Signature> {
    /// Instantiates this polymorphic signature by providing arguments for each parameter.
    ///
    /// # Errors
    ///
    /// - If there is not exactly one argument for each parameter.
    /// - If an argument does not have the correct type.
    pub fn instantiate(&self, args: &[Term]) -> Result<Signature, SignatureError> {
        check_term_types(args, &self.params)?;
        Ok(self.body.substitute(&Substitution(args)))
    }

    /// Validates this instance, checking that the types in the body are
    /// wellformed with respect to the registry, and the type variables declared.
    pub fn validate(&self) -> Result<(), SignatureError> {
        self.body.validate(&self.params)
    }
}

impl Polymorphic<FuncValueType> {
    /// Instantiates this polymorphic function type by providing arguments for each parameter.
    ///
    /// # Errors
    ///
    /// - If there is not exactly one argument for each parameter.
    /// - If an argument does not have the correct type.
    pub fn instantiate(&self, args: &[Term]) -> Result<FuncValueType, SignatureError> {
        check_term_types(args, &self.params)?;
        Ok(self.body.substitute(&Substitution(args)))
    }

    /// Validates this instance, checking that the types in the body are
    /// wellformed with respect to the registry, and the type variables declared.
    pub fn validate(&self) -> Result<(), SignatureError> {
        self.body.validate(&self.params)
    }
}

#[cfg(test)]
impl<T> Arbitrary for Polymorphic<T>
where
    T: Arbitrary<Parameters = RecursionDepth> + 'static,
{
    type Parameters = RecursionDepth;
    type Strategy = BoxedStrategy<Polymorphic<T>>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        (vec(any_with::<Term>(args), 0..3), any_with::<T>(args))
            .prop_map(|(params, body)| Self::new(params, body))
            .boxed()
    }
}

/// The polymorphic type of a [`Call`]-able function ([`FuncDecl`] or [`FuncDefn`]).
/// Number of inputs and outputs fixed.
///
/// [`Call`]: crate::ops::Call
/// [`FuncDefn`]: crate::ops::FuncDefn
/// [`FuncDecl`]: crate::ops::FuncDecl
pub type PolyFuncType = Polymorphic<Signature>;

/// The polymorphic type of an [`OpDef`], whose number of input and outputs
/// may vary according to how [`RowVariable`]s therein are instantiated.
///
/// [`OpDef`]: crate::extension::OpDef
pub type PolyFuncTypeRV = Polymorphic<FuncValueType>;

#[cfg(test)]
pub(crate) mod test {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use cool_asserts::assert_matches;
    use lazy_static::lazy_static;

    use crate::Extension;
    use crate::extension::prelude::{bool_t, usize_t};
    use crate::extension::{ExtensionId, ExtensionRegistry, PRELUDE, SignatureError, TypeDefBound};
    use crate::std_extensions::collections::array::{self, array_type_parametric};
    use crate::std_extensions::collections::list;
    use crate::types::type_param::{TermTypeError, TypeArg, TypeParam};
    use crate::types::{
        CustomType, FuncValueType, Signature, Term, Type, TypeBound, TypeName, TypeRV,
    };

    use super::{PolyFuncType, PolyFuncTypeRV};

    lazy_static! {
        static ref REGISTRY: ExtensionRegistry =
            ExtensionRegistry::new([PRELUDE.to_owned(), list::EXTENSION.to_owned()]);
    }

    impl PolyFuncType {
        fn new_validated(
            params: impl IntoIterator<Item = Term>,
            body: Signature,
        ) -> Result<Self, SignatureError> {
            let res = Self::new(params, body);
            res.validate()?;
            Ok(res)
        }
    }

    impl PolyFuncTypeRV {
        fn new_validated(
            params: impl IntoIterator<Item = Term>,
            body: FuncValueType,
        ) -> Result<Self, SignatureError> {
            let res = Self::new(params, body);
            res.validate()?;
            Ok(res)
        }
    }

    #[test]
    fn test_opaque() -> Result<(), SignatureError> {
        let list_def = list::EXTENSION.get_type(&list::LIST_TYPENAME).unwrap();
        let tyvar = TypeArg::new_var_use(0, TypeBound::Any.into());
        let list_of_var = Type::new_extension(list_def.instantiate([tyvar.clone()])?);
        let list_len = PolyFuncType::new_validated(
            [TypeBound::Any.into()],
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
        let ty_var = TypeArg::new_var_use(1, TypeBound::Any.into());
        let type_params = [TypeParam::max_nat_type(), TypeBound::Any.into()];

        // Valid schema...
        let good_array = array_type_parametric(size_var.clone(), ty_var.clone())?;
        let good_ts =
            PolyFuncType::new_validated(type_params.clone(), Signature::new_endo(good_array))?;

        // Sanity check (good args)
        good_ts.instantiate(&[5u64.into(), usize_t().into()])?;

        let wrong_args = good_ts.instantiate(&[usize_t().into(), 5u64.into()]);
        assert_eq!(
            wrong_args,
            Err(SignatureError::TypeArgMismatch(
                TermTypeError::TypeMismatch {
                    type_: type_params[0].clone(),
                    term: usize_t().into(),
                }
            ))
        );

        // (Try to) make a schema with the args in the wrong order
        let arg_err = SignatureError::TypeArgMismatch(TermTypeError::TypeMismatch {
            type_: type_params[0].clone(),
            term: ty_var.clone(),
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
            TypeBound::Any,
            &Arc::downgrade(&array::EXTENSION),
        ));
        let bad_ts =
            PolyFuncType::new_validated(type_params.clone(), Signature::new_endo(bad_array));
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
            Term::new_tuple_type([TypeBound::Any.into(), Term::max_nat_type()]),
        ] {
            let invalid_ts = PolyFuncType::new_validated([decl.clone()], body_type.clone());
            assert_eq!(
                invalid_ts.err(),
                Some(SignatureError::TypeVarDoesNotMatchDeclaration {
                    cached: TypeBound::Copyable.into(),
                    actual: decl
                })
            );
        }
        // Variable not declared at all
        let invalid_ts = PolyFuncType::new_validated([], body_type);
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
            PolyFuncType::new_validated(
                [tp.clone()],
                Signature::new_endo(Type::new_extension(CustomType::new(
                    TYPE_NAME,
                    [TypeArg::new_var_use(0, tp)],
                    EXT_ID,
                    TypeBound::Any,
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
                        type_: bound.clone(),
                        term: TypeArg::new_var_use(0, decl.clone())
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

        decl_accepts_rejects_var(
            Term::new_list_type(TypeBound::Copyable),
            &[Term::new_list_type(TypeBound::Copyable)],
            &[Term::new_list_type(TypeBound::Any)],
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

    const TP_ANY: TypeParam = TypeParam::RuntimeType(TypeBound::Any);
    #[test]
    fn row_variables_bad_schema() {
        // Mismatched TypeBound (Copyable vs Any)
        let decl = Term::new_list_type(TP_ANY);
        let e = PolyFuncTypeRV::new_validated(
            [decl.clone()],
            FuncValueType::new(
                vec![usize_t()],
                vec![TypeRV::new_row_var_use(0, TypeBound::Copyable)],
            ),
        )
        .unwrap_err();
        assert_matches!(e, SignatureError::TypeVarDoesNotMatchDeclaration { actual, cached } => {
            assert_eq!(actual, decl);
            assert_eq!(cached, TypeParam::new_list_type(TypeBound::Copyable));
        });
        // Declared as row variable, used as type variable
        let e = PolyFuncType::new_validated(
            [decl.clone()],
            Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
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
        let pf = PolyFuncTypeRV::new_validated(
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
        let pf = PolyFuncType::new_validated(
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
