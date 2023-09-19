//! Definition of the leaf operations.

use smol_str::SmolStr;

use super::custom::ExternalOp;
use super::{OpName, OpTag, OpTrait, StaticTag};

use crate::extension::{ExtensionRegistry, SignatureError};
use crate::types::type_param::{check_type_args, TypeArg};
use crate::types::{PolyFuncType, Substitution};
use crate::{
    extension::{ExtensionId, ExtensionSet},
    types::{EdgeKind, FunctionType, SignatureDescription, Type, TypeRow},
};

/// Dataflow operations with no children.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(tag = "lop")]
pub enum LeafOp {
    /// A user-defined operation that can be downcasted by the extensions that
    /// define it.
    CustomOp(Box<ExternalOp>),

    /// A no-op operation.
    Noop {
        /// The type of edges connecting the Noop.
        ty: Type,
    },
    /// An operation that packs all its inputs into a tuple.
    MakeTuple {
        ///Tuple element types.
        tys: TypeRow,
    },
    /// An operation that unpacks a tuple into its components.
    UnpackTuple {
        ///Tuple element types.
        tys: TypeRow,
    },
    /// An operation that creates a tagged sum value from one of its variants.
    Tag {
        /// The variant to create.
        tag: usize,
        /// The variants of the sum type.
        variants: TypeRow,
    },
    /// A node which adds a extension req to the types of the wires it is passed
    /// It has no effect on the values passed along the edge
    Lift {
        /// The types of the edges
        type_row: TypeRow,
        /// The extensions which we're adding to the inputs
        new_extension: ExtensionId,
    },
    /// Fixes some [TypeParam]s of a polymorphic type by providing [TypeArg]s
    ///
    /// [TypeParam]: crate::types::type_param::TypeParam
    TypeApply {
        /// The type and args, plus a cache of the resulting type
        ta: TypeApplication,
    },
}

/// Records details of an application of a [PolyFuncType] to some [TypeArg]s
/// and the result (a less-, but still potentially-, polymorphic type).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TypeApplication {
    input: PolyFuncType,
    args: Vec<TypeArg>,
    output: PolyFuncType, // cached
}

impl TypeApplication {
    /// Checks that the specified args are correct for the [TypeParam]s of the polymorphic input.
    /// Note the extension registry is required here to recompute [Type::least_upper_bound]s.
    ///
    /// [TypeParam]: crate::types::type_param::TypeParam
    pub fn try_new(
        input: PolyFuncType,
        args: impl Into<Vec<TypeArg>>,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let args = args.into();
        // Should we require >=1 `arg`s here? Or that input declares >=1 params?
        // At the moment we allow an identity TypeApply on a monomorphic function type.
        let (fixed, remaining) = input.params.split_at(args.len());
        check_type_args(&args, fixed)?;
        let body = input.body.substitute(
            extension_registry,
            &Substitution::new(&args, remaining.len()),
        );
        let params = Vec::from(remaining);
        Ok(Self {
            input,
            args,
            output: PolyFuncType {
                params,
                body: Box::new(body),
            },
        })
    }

    /// Turns this into a [LeafOp].
    /// (Would impl [Into] but that ends up with the client needing annotations.)
    pub fn to_leaf(self) -> LeafOp {
        LeafOp::TypeApply { ta: self }
    }

    pub(crate) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), SignatureError> {
        let other = Self::try_new(self.input.clone(), self.args.clone(), extension_registry)?;
        if other.output == self.output {
            Ok(())
        } else {
            Err(SignatureError::CachedTypeIncorrect {
                stored: self.output.clone(),
                expected: other.output.clone(),
            })
        }
    }
}

impl Default for LeafOp {
    fn default() -> Self {
        Self::Noop { ty: Type::UNIT }
    }
}
impl OpName for LeafOp {
    /// The name of the operation.
    fn name(&self) -> SmolStr {
        match self {
            LeafOp::CustomOp(ext) => return ext.name(),
            LeafOp::Noop { ty: _ } => "Noop",
            LeafOp::MakeTuple { tys: _ } => "MakeTuple",
            LeafOp::UnpackTuple { tys: _ } => "UnpackTuple",
            LeafOp::Tag { .. } => "Tag",
            LeafOp::Lift { .. } => "Lift",
            LeafOp::TypeApply { .. } => "TypeApply",
        }
        .into()
    }
}

impl StaticTag for LeafOp {
    const TAG: OpTag = OpTag::Leaf;
}

impl OpTrait for LeafOp {
    /// A human-readable description of the operation.
    fn description(&self) -> &str {
        match self {
            LeafOp::CustomOp(ext) => ext.description(),
            LeafOp::Noop { ty: _ } => "Noop gate",
            LeafOp::MakeTuple { tys: _ } => "MakeTuple operation",
            LeafOp::UnpackTuple { tys: _ } => "UnpackTuple operation",
            LeafOp::Tag { .. } => "Tag Sum operation",
            LeafOp::Lift { .. } => "Add a extension requirement to an edge",
            LeafOp::TypeApply { .. } => {
                "Instantiate (perhaps partially) a polymorphic type with some type arguments"
            }
        }
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    /// The signature of the operation.
    fn signature(&self) -> FunctionType {
        // Static signatures. The `TypeRow`s in the `FunctionType` use a
        // copy-on-write strategy, so we can avoid unnecessary allocations.

        match self {
            LeafOp::Noop { ty: typ } => FunctionType::new(vec![typ.clone()], vec![typ.clone()]),
            LeafOp::CustomOp(ext) => ext.signature(),
            LeafOp::MakeTuple { tys: types } => {
                FunctionType::new(types.clone(), vec![Type::new_tuple(types.clone())])
            }
            LeafOp::UnpackTuple { tys: types } => {
                FunctionType::new(vec![Type::new_tuple(types.clone())], types.clone())
            }
            LeafOp::Tag { tag, variants } => FunctionType::new(
                vec![variants.get(*tag).expect("Not a valid tag").clone()],
                vec![Type::new_sum(variants.clone())],
            ),
            LeafOp::Lift {
                type_row,
                new_extension,
            } => FunctionType::new(type_row.clone(), type_row.clone())
                .with_extension_delta(&ExtensionSet::singleton(new_extension)),
            LeafOp::TypeApply { ta } => FunctionType::new(
                vec![Type::new_function(ta.input.clone())],
                vec![Type::new_function(ta.output.clone())],
            ),
        }
    }

    /// Optional description of the ports in the signature.
    fn signature_desc(&self) -> SignatureDescription {
        match self {
            LeafOp::CustomOp(ext) => ext.signature_desc(),
            // TODO: More port descriptions
            _ => Default::default(),
        }
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
}

#[cfg(test)]
mod test {
    use crate::extension::prelude::USIZE_T;
    use crate::extension::{SignatureError, PRELUDE, PRELUDE_REGISTRY};
    use crate::ops::OpTrait;
    use crate::types::type_param::{TypeArg, TypeParam};
    use crate::types::{FunctionType, PolyFuncType, Type, TypeBound};

    use super::TypeApplication;

    const USIZE_TA: TypeArg = TypeArg::Type { ty: USIZE_T };

    // The standard library new_array does not allow passing in a variable for size.
    fn new_array(ty: Type, s: TypeArg) -> Type {
        let array_def = PRELUDE.get_type("array").unwrap();
        Type::new_extension(
            array_def
                .instantiate_concrete(vec![TypeArg::Type { ty }, s])
                .unwrap(),
        )
    }

    #[test]
    fn test_type_apply() -> Result<(), SignatureError> {
        let sig_fn =
            |i, o| FunctionType::new(vec![Type::new_function(i)], vec![Type::new_function(o)]);

        let array_max = PolyFuncType {
            params: vec![TypeParam::Type(TypeBound::Any), TypeParam::max_nat()],
            body: Box::new(FunctionType::new(
                vec![new_array(
                    Type::new_variable(0, TypeBound::Any),
                    TypeArg::use_var(1, TypeParam::max_nat()),
                )],
                vec![Type::new_variable(0, TypeBound::Any)],
            )),
        };

        let concrete = FunctionType::new(
            vec![new_array(USIZE_T, TypeArg::BoundedNat { n: 3 })],
            vec![USIZE_T],
        );
        let ta = TypeApplication::try_new(
            array_max.clone(),
            [USIZE_TA, TypeArg::BoundedNat { n: 3 }],
            &PRELUDE_REGISTRY,
        )?;
        assert_eq!(
            ta.to_leaf().signature(),
            sig_fn(array_max.clone(), concrete.into())
        );

        let partial = PolyFuncType {
            params: vec![TypeParam::max_nat()],
            body: Box::new(FunctionType::new(
                vec![new_array(
                    USIZE_T,
                    TypeArg::use_var(0, TypeParam::max_nat()),
                )],
                vec![USIZE_T],
            )),
        };
        let ta = TypeApplication::try_new(array_max.clone(), [USIZE_TA], &PRELUDE_REGISTRY)?;
        assert_eq!(ta.to_leaf().signature(), sig_fn(array_max, partial));

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
        let outer = PolyFuncType {
            params: vec![TypeParam::max_nat()],
            body: Box::new(FunctionType::new(vec![], vec![Type::new_function(inner)])),
        };

        let outer_applied = FunctionType::new(
            vec![],
            vec![Type::new_function(PolyFuncType {
                params: vec![TypeParam::Type(TypeBound::Any)],
                body: Box::new(FunctionType::new(
                    // We are checking that the substitution has been applied to the right var
                    // - NOT to the inner_var which has index 0 here
                    vec![new_array(inner_var.clone(), TypeArg::BoundedNat { n: 5 })],
                    vec![inner_var.clone()],
                )),
            })],
        );

        let do_apply = TypeApplication::try_new(
            outer.clone(),
            [TypeArg::BoundedNat { n: 5 }],
            &PRELUDE_REGISTRY,
        )?;
        assert_eq!(
            do_apply.to_leaf().signature(),
            FunctionType::new(
                vec![Type::new_function(outer)],
                vec![Type::new_function(outer_applied)]
            )
        );
        Ok(())
    }
}
