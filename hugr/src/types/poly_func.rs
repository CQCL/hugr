//! Polymorphic Function Types

use crate::extension::{ExtensionRegistry, SignatureError};
use itertools::Itertools;

use super::signature::{FuncTypeBase, TypeRowElem};
use super::type_param::{check_type_args, TypeArg, TypeParam};
use super::{RowVarOrType, Substitution, Type};

/// A polymorphic type scheme, i.e. of a [FuncDecl], [FuncDefn] or [OpDef].
/// (Nodes/operations in the Hugr and runtime [Graph] values are monomorphic.)
///
/// [FuncDecl]: crate::ops::module::FuncDecl
/// [FuncDefn]: crate::ops::module::FuncDefn
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
where T: 'static + Sized + Clone, [T]: ToOwned<Owned = Vec<T>> {
    /// The declared type parameters, i.e., these must be instantiated with
    /// the same number of [TypeArg]s before the function can be called. This
    /// defines the indices used by variables inside the body.
    params: Vec<TypeParam>,
    /// Template for the function. May contain variables up to length of [Self::params]
    body: FuncTypeBase<T>,
}

/// A polymorphic type scheme which may include row variables, hence the
/// actual number of inputs/outputs may vary according to instantiation.
pub type PolyFuncVarLen = PolyFuncBase<RowVarOrType>;

/// A type scheme that is polymorphic only over types, so fixed arity.
pub type PolyFuncType = PolyFuncBase<Type>;

impl<T: 'static + Sized + Clone> From<FuncTypeBase<T>> for PolyFuncBase<T>
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

impl<T: 'static + Sized + Clone> TryFrom<PolyFuncBase<T>> for FuncTypeBase<T>
where [T]: ToOwned<Owned = Vec<T>> {
    /// If conversion fails, return the binders (which prevent conversion)
    type Error = Vec<TypeParam>;

    fn try_from(value: PolyFuncBase<T>) -> Result<Self, Self::Error> {
        if value.params.is_empty() {Ok(value.body)}
        else {Err(value.params)}
    }
}

impl From<PolyFuncType> for PolyFuncVarLen {
    fn from(value: PolyFuncType) -> Self {
        Self {
            params: value.params,
            body: value.body.into(),
        }
    }
}

impl<T: 'static + Sized + Clone> PolyFuncBase<T>
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
    /// and the underlying [FuncTypeVarLen].
    pub fn new(params: impl Into<Vec<TypeParam>>, body: impl Into<FuncTypeBase<T>>) -> Self {
        Self {
            params: params.into(),
            body: body.into(),
        }
    }
}

#[allow(private_bounds)]  // TypeRowElem is pub(super) and these are pub(crate)
impl <T: TypeRowElem + Clone> PolyFuncBase<T> {

    /// Validates this instance, checking that the types in the body are
    /// wellformed with respect to the registry, and that all type variables
    /// are declared (perhaps in an enclosing scope, kinds passed in).
    pub(crate) fn validate(
        &self,
        reg: &ExtensionRegistry
    ) -> Result<(), SignatureError> {
        // TODO https://github.com/CQCL/hugr/issues/624 validate TypeParams declared here, too
        self.body.validate(reg, &self.params)
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
        Ok(self.body.substitute(&Substitution(args, ext_reg)))
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::num::NonZeroU64;

    use lazy_static::lazy_static;
    use smol_str::SmolStr;

    use crate::extension::prelude::{PRELUDE_ID, USIZE_CUSTOM_T, USIZE_T};
    use crate::extension::{
        ExtensionId, ExtensionRegistry, SignatureError, TypeDefBound, PRELUDE, PRELUDE_REGISTRY,
    };
    use crate::std_extensions::collections::{EXTENSION, LIST_TYPENAME};
    use crate::types::type_param::{TypeArg, TypeArgError, TypeParam};
    use crate::types::{CustomType, FunctionType, Type, TypeBound};
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
            PolyFuncType::new_validated(typarams.clone(), id_fn(good_array), &PRELUDE_REGISTRY)?;

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
            PolyFuncType::new_validated(typarams.clone(), id_fn(bad_array), &PRELUDE_REGISTRY);
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
            PolyFuncType::new_validated(
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
}
