//! Polymorphic Function Types

use crate::extension::{ExtensionRegistry, SignatureError};
use itertools::Itertools;

use super::{
    type_param::{check_type_args, TypeArg, TypeParam},
    FunctionType, Substitution,
};

/// A polymorphic function type, e.g. of a [Graph], or perhaps an [OpDef].
/// (Nodes/operations in the Hugr are not polymorphic.)
///
/// [Graph]: crate::values::PrimValue::Function
/// [OpDef]: crate::extension::OpDef
#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[display(
    fmt = "forall {}. {}",
    "params.iter().map(ToString::to_string).join(\" \")",
    "body"
)]
pub struct PolyFuncType {
    /// The declared type parameters, i.e., these must be instantiated with
    /// the same number of [TypeArg]s before the function can be called.
    ///
    /// [TypeArg]: super::type_param::TypeArg
    pub(crate) params: Vec<TypeParam>,
    /// Template for the function. May contain variables up to length of [Self::params]
    body: Box<FunctionType>,
}

impl From<FunctionType> for PolyFuncType {
    fn from(body: FunctionType) -> Self {
        Self {
            params: vec![],
            body: Box::new(body),
        }
    }
}

impl PolyFuncType {
    /// Create a new PolyFuncType and validates it. (This will only succeed
    /// for outermost PolyFuncTypes i.e. with no free type-variables.)
    /// The [ExtensionRegistry] should be the same (or a subset) of that which will later
    /// be used to validate the Hugr; at this point we only need the types.
    ///
    /// #Errors
    /// Validates that all types in the schema are well-formed and all variables in the body
    /// are declared with [TypeParam]s that guarantee they will fit.
    pub fn new_validated(
        params: impl Into<Vec<TypeParam>>,
        body: FunctionType,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let params = params.into();
        body.validate(extension_registry, &params)?;
        Ok(Self {
            params,
            body: Box::new(body),
        })
    }

    pub(super) fn validate(
        &self,
        reg: &ExtensionRegistry,
        external_type_vars: &[TypeParam],
    ) -> Result<(), SignatureError> {
        // TODO should we add a mechanism to validate a TypeParam?
        let mut v = vec![];
        let all_type_vars = if self.params.is_empty() {
            external_type_vars
        } else {
            // Type vars declared here go at lowest indices (as per DeBruijn)
            v.extend(self.params.iter().cloned());
            v.extend_from_slice(external_type_vars);
            v.as_slice()
        };
        self.body.validate(reg, all_type_vars)
    }

    pub(super) fn substitute(&self, exts: &ExtensionRegistry, sub: &Substitution) -> Self {
        Self {
            body: Box::new(
                self.body
                    .substitute(exts, &sub.enter_scope(self.params.len(), exts)),
            ),
            params: self.params.clone(),
        }
    }

    pub(crate) fn instantiate(
        &self,
        args: &[TypeArg],
        exts: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let (fixed, remaining) = self.params.split_at(args.len());
        check_type_args(args, fixed)?;
        let sub: Substitution = if remaining.is_empty() {
            args.into()
        } else {
            // Partial application - renumber remaining params (still bound) downward
            let mut rhs = args.to_vec();
            rhs.extend(
                remaining
                    .iter()
                    .enumerate()
                    .map(|(i, decl)| TypeArg::use_var(i, decl.clone())),
            );
            rhs.into()
        };
        Ok(Self {
            params: remaining.to_vec(),
            body: Box::new(self.body.substitute(exts, &sub)),
        })
    }

    pub(crate) fn instantiate_all(
        &self,
        args: &[TypeArg],
        extension_registry: &ExtensionRegistry,
    ) -> Result<FunctionType, SignatureError> {
        check_type_args(args, &self.params)?;
        let pf = self.instantiate(args, extension_registry)?;
        // Since we checked args vs params, we know there are the right number
        assert!(pf.params.is_empty());
        Ok(*pf.body)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::num::NonZeroU64;

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

    #[test]
    fn test_opaque() -> Result<(), SignatureError> {
        let list_def = EXTENSION.get_type(&LIST_TYPENAME).unwrap();
        let tyvar = TypeArg::use_var(0, TypeParam::Type(TypeBound::Any));
        let list_of_var = Type::new_extension(list_def.instantiate([tyvar.clone()])?);
        let reg: ExtensionRegistry = [PRELUDE.to_owned(), EXTENSION.to_owned()].into();
        let list_len = PolyFuncType::new_validated(
            [TypeParam::Type(TypeBound::Any)],
            FunctionType::new(vec![list_of_var], vec![USIZE_T]),
            &reg,
        )?;

        let t = list_len.instantiate_all(&[TypeArg::Type { ty: USIZE_T }], &reg)?;
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
        let typarams = [TypeParam::Type(TypeBound::Any), TypeParam::max_nat()];
        let [tyvar, szvar] = [0, 1].map(|i| TypeArg::use_var(i, typarams.get(i).unwrap().clone()));

        // Valid schema...
        let good_array = Type::new_extension(ar_def.instantiate([tyvar.clone(), szvar.clone()])?);
        let good_ts =
            PolyFuncType::new_validated(typarams.clone(), id_fn(good_array), &PRELUDE_REGISTRY)?;

        // Sanity check (good args)
        good_ts.instantiate_all(
            &[TypeArg::Type { ty: USIZE_T }, TypeArg::BoundedNat { n: 5 }],
            &PRELUDE_REGISTRY,
        )?;

        let wrong_args = good_ts.instantiate_all(
            &[TypeArg::BoundedNat { n: 5 }, TypeArg::Type { ty: USIZE_T }],
            &PRELUDE_REGISTRY,
        );
        assert_eq!(
            wrong_args,
            Err(SignatureError::TypeArgMismatch(
                TypeArgError::TypeMismatch {
                    param: typarams[0].clone(),
                    arg: TypeArg::BoundedNat { n: 5 }
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
        let tv = TypeArg::use_var(0, TypeParam::Type(TypeBound::Copyable));
        let list_def = EXTENSION.get_type(&LIST_TYPENAME).unwrap();
        let body_type = id_fn(Type::new_extension(list_def.instantiate([tv])?));
        let reg = [EXTENSION.to_owned()].into();
        for decl in [
            TypeParam::Extensions,
            TypeParam::List(Box::new(TypeParam::max_nat())),
            TypeParam::Opaque(USIZE_CUSTOM_T),
            TypeParam::Tuple(vec![TypeParam::Type(TypeBound::Any), TypeParam::max_nat()]),
        ] {
            let invalid_ts = PolyFuncType::new_validated([decl.clone()], body_type.clone(), &reg);
            assert_eq!(
                invalid_ts.err(),
                Some(SignatureError::TypeVarDoesNotMatchDeclaration {
                    used: TypeParam::Type(TypeBound::Copyable),
                    decl
                })
            );
        }
        // Variable not declared at all
        let invalid_ts = PolyFuncType::new_validated([], body_type, &reg);
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

        let reg: ExtensionRegistry = [e].into();

        let make_scheme = |tp: TypeParam| {
            PolyFuncType::new_validated(
                [tp.clone()],
                id_fn(Type::new_extension(CustomType::new(
                    TYPE_NAME,
                    [TypeArg::use_var(0, tp)],
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
                        arg: TypeArg::use_var(0, decl.clone())
                    }
                ))
            );
        }
        Ok(())
    }

    #[test]
    fn test_bound_covariance() -> Result<(), SignatureError> {
        decl_accepts_rejects_var(
            TypeParam::Type(TypeBound::Copyable),
            &[
                TypeParam::Type(TypeBound::Copyable),
                TypeParam::Type(TypeBound::Eq),
            ],
            &[TypeParam::Type(TypeBound::Any)],
        )?;

        let list_of_tys = |b| TypeParam::List(Box::new(TypeParam::Type(b)));
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
            body: Box::new(FunctionType::new(vec![input], vec![output])),
        }
    }

    // The standard library new_array does not allow passing in a variable for size.
    pub(crate) fn new_array(ty: Type, s: TypeArg) -> Type {
        let array_def = PRELUDE.get_type("array").unwrap();
        Type::new_extension(
            array_def
                .instantiate(vec![TypeArg::Type { ty }, s])
                .unwrap(),
        )
    }

    #[test]
    fn substitute_under_binder() {
        let list_def = EXTENSION.get_type(LIST_TYPENAME.as_str()).unwrap();
        let list_of_tup = |t1, t2| {
            Type::new_extension(
                list_def
                    .instantiate([TypeArg::Type {
                        ty: Type::new_tuple(vec![t1, t2]),
                    }])
                    .unwrap(),
            )
        };
        let reg = [EXTENSION.to_owned()].into();
        let pf = PolyFuncType::new_validated(
            vec![TypeParam::Type(TypeBound::Any)],
            FunctionType::new(
                vec![Type::new_variable(0, TypeBound::Any)],
                vec![Type::new_function(new_pf1(
                    TypeParam::Type(TypeBound::Copyable),
                    Type::new_variable(0, TypeBound::Copyable),
                    list_of_tup(
                        Type::new_variable(0, TypeBound::Copyable),
                        Type::new_variable(1, TypeBound::Any), // The outer variable (renumbered)
                    ),
                ))],
            ),
            &reg,
        )
        .unwrap();

        // Now substitute in a free var from further outside
        let reg = [EXTENSION.to_owned(), PRELUDE.to_owned()].into();
        const FREE: usize = 3;
        const TP_EQ: TypeParam = TypeParam::Type(TypeBound::Eq);
        let res = pf
            .instantiate_all(&[TypeArg::use_var(FREE, TP_EQ)], &reg)
            .unwrap();
        assert_eq!(
            res,
            FunctionType::new(
                vec![Type::new_variable(FREE, TypeBound::Eq)],
                vec![Type::new_function(new_pf1(
                    TypeParam::Type(TypeBound::Copyable),
                    Type::new_variable(0, TypeBound::Copyable), // unchanged
                    list_of_tup(
                        Type::new_variable(0, TypeBound::Copyable),
                        // Next is the free variable that we substituted in (hence Eq)
                        // - renumbered because of the intervening forall (Copyable)
                        Type::new_variable(FREE + 1, TypeBound::Eq)
                    )
                ))]
            )
        );

        // Also try substituting in a type containing both free and bound vars
        let rhs = |i| {
            Type::new_function(new_pf1(
                TP_EQ,
                Type::new_variable(0, TypeBound::Eq),
                new_array(
                    Type::new_variable(0, TypeBound::Eq),
                    TypeArg::use_var(i, TypeParam::max_nat()),
                ),
            ))
        };

        let res = pf
            .instantiate_all(&[TypeArg::Type { ty: rhs(FREE) }], &reg)
            .unwrap();
        assert_eq!(
            res,
            FunctionType::new(
                vec![rhs(FREE)],
                vec![Type::new_function(new_pf1(
                    TypeParam::Type(TypeBound::Copyable),
                    Type::new_variable(0, TypeBound::Copyable),
                    list_of_tup(
                        Type::new_variable(0, TypeBound::Copyable), // not renumbered...
                        rhs(FREE + 1)                               // renumbered
                    )
                ))]
            )
        )
    }

    const USIZE_TA: TypeArg = TypeArg::Type { ty: USIZE_T };

    #[test]
    fn test_instantiate() -> Result<(), SignatureError> {
        let array_max = PolyFuncType::new_validated(
            vec![TypeParam::Type(TypeBound::Any), TypeParam::max_nat()],
            FunctionType::new(
                vec![new_array(
                    Type::new_variable(0, TypeBound::Any),
                    TypeArg::use_var(1, TypeParam::max_nat()),
                )],
                vec![Type::new_variable(0, TypeBound::Any)],
            ),
            &PRELUDE_REGISTRY,
        )?;

        let concrete = FunctionType::new(
            vec![new_array(USIZE_T, TypeArg::BoundedNat { n: 3 })],
            vec![USIZE_T],
        );
        let actual =
            array_max.instantiate(&[USIZE_TA, TypeArg::BoundedNat { n: 3 }], &PRELUDE_REGISTRY)?;

        assert_eq!(actual, concrete.into());

        let partial = PolyFuncType::new_validated(
            vec![TypeParam::max_nat()],
            FunctionType::new(
                vec![new_array(
                    USIZE_T,
                    TypeArg::use_var(0, TypeParam::max_nat()),
                )],
                vec![USIZE_T],
            ),
            &PRELUDE_REGISTRY,
        )?;
        let res = array_max.instantiate(&[USIZE_TA], &PRELUDE_REGISTRY)?;
        assert_eq!(res, partial);

        Ok(())
    }

    #[test]
    fn test_type_apply_nested() -> Result<(), SignatureError> {
        let inner_var = Type::new_variable(0, TypeBound::Any);
        let inner = PolyFuncType {
            params: vec![TypeParam::Type(TypeBound::Any)],
            body: Box::new(FunctionType::new(
                vec![new_array(
                    inner_var.clone(),
                    TypeArg::use_var(1, TypeParam::max_nat()),
                )],
                vec![inner_var.clone()],
            )),
        };
        let outer = PolyFuncType::new_validated(
            vec![TypeParam::max_nat()],
            FunctionType::new(vec![], vec![Type::new_function(inner)]),
            &PRELUDE_REGISTRY,
        )?;

        let outer_applied = FunctionType::new(
            vec![],
            vec![Type::new_function(PolyFuncType::new_validated(
                vec![TypeParam::Type(TypeBound::Any)],
                FunctionType::new(
                    // We are checking that the substitution has been applied to the right var
                    // - NOT to the inner_var which has index 0 here
                    vec![new_array(inner_var.clone(), TypeArg::BoundedNat { n: 5 })],
                    vec![inner_var.clone()],
                ),
                &PRELUDE_REGISTRY,
            )?)],
        );

        let res = outer.instantiate(&[TypeArg::BoundedNat { n: 5 }], &PRELUDE_REGISTRY)?;
        assert_eq!(res, outer_applied.into());
        Ok(())
    }
}
