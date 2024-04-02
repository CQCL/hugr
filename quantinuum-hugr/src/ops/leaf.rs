//! Definition of the leaf operations.

use smol_str::SmolStr;

use super::custom::{ExtensionOp, ExternalOp};
use super::dataflow::DataflowOpTrait;
use super::{OpName, OpTag};

use crate::extension::{ExtensionId, ExtensionRegistry, ExtensionSet, SignatureError};
use crate::types::type_param::TypeArg;
use crate::types::{EdgeKind, FunctionType, PolyFuncVarLen, Type, TypeRow};

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
        variants: Vec<TypeRow>,
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

impl LeafOp {
    /// If instance of [ExtensionOp] return a reference to it.
    pub fn as_extension_op(&self) -> Option<&ExtensionOp> {
        let LeafOp::CustomOp(ext) = self else {
            return None;
        };

        match ext.as_ref() {
            ExternalOp::Extension(e) => Some(e),
            ExternalOp::Opaque(_) => None,
        }
    }
}

/// Records details of an application of a [PolyFuncType] to some [TypeArg]s
/// and the result (a less-, but still potentially-, polymorphic type).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TypeApplication {
    input: PolyFuncVarLen,
    args: Vec<TypeArg>,
    output: PolyFuncVarLen, // cached
}

impl TypeApplication {
    /// Checks that the specified args are correct for the [TypeParam]s of the polymorphic input.
    /// Note the extension registry is required here to recompute [Type::least_upper_bound]s.
    ///
    /// [TypeParam]: crate::types::type_param::TypeParam
    pub fn try_new(
        input: PolyFuncVarLen,
        args: impl Into<Vec<TypeArg>>,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let args = args.into();
        // Should we require >=1 `arg`s here? Or that input declares >=1 params?
        // At the moment we allow an identity TypeApply on a monomorphic function type.
        let output = input.instantiate_poly(&args, extension_registry)?;
        Ok(Self {
            input,
            args,
            output,
        })
    }

    pub(crate) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), SignatureError> {
        let other = Self::try_new(self.input.clone(), self.args.clone(), extension_registry)?;
        if other.output == self.output {
            Ok(())
        } else {
            Err(SignatureError::TypeApplyIncorrectCache {
                cached: self.output.clone(),
                expected: other.output.clone(),
            })
        }
    }

    /// Returns the type of the input function.
    pub fn input(&self) -> &PolyFuncVarLen {
        &self.input
    }

    /// Returns the args applied to the input function.
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Returns the type of the output function.
    pub fn output(&self) -> &PolyFuncVarLen {
        &self.output
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

// impl StaticTag for LeafOp {
// }

impl DataflowOpTrait for LeafOp {
    const TAG: OpTag = OpTag::Leaf;
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

    /// The signature of the operation.
    fn signature(&self) -> FunctionType {
        // Static signatures. The `TypeRow`s in the `FuncTypeVarLen` use a
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
                variants.get(*tag).expect("Not a valid tag").clone(),
                vec![Type::new_sum(variants.clone())],
            ),
            LeafOp::Lift {
                type_row,
                new_extension,
            } => FunctionType::new(type_row.clone(), type_row.clone())
                .with_extension_delta(ExtensionSet::singleton(new_extension)),
            LeafOp::TypeApply { ta } => FunctionType::new(
                vec![Type::new_function(ta.input.clone())],
                vec![Type::new_function(ta.output.clone())],
            ),
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
    use crate::builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr};
    use crate::extension::prelude::BOOL_T;
    use crate::extension::{prelude::USIZE_T, PRELUDE};
    use crate::extension::{ExtensionRegistry, SignatureError};
    use crate::hugr::ValidationError;
    use crate::ops::handle::NodeHandle;
    use crate::std_extensions::collections::EXTENSION;
    use crate::types::test::nested_func;
    use crate::types::{FunctionType, Type, TypeArg};

    use super::{LeafOp, TypeApplication};

    const USIZE_TA: TypeArg = TypeArg::Type { ty: USIZE_T };

    #[test]
    fn hugr_with_type_apply() -> Result<(), Box<dyn std::error::Error>> {
        let reg = ExtensionRegistry::try_new([PRELUDE.to_owned(), EXTENSION.to_owned()]).unwrap();
        let pf_in = nested_func();
        let pf_out = pf_in.instantiate(&[USIZE_TA], &reg)?;
        let mut dfg = DFGBuilder::new(FunctionType::new(
            vec![Type::new_function(pf_in.clone())],
            vec![Type::new_function(pf_out)],
        ))?;
        let ta = dfg.add_dataflow_op(
            LeafOp::TypeApply {
                ta: TypeApplication::try_new(pf_in, [USIZE_TA], &reg).unwrap(),
            },
            dfg.input_wires(),
        )?;
        dfg.finish_hugr_with_outputs(ta.outputs(), &reg)?;
        Ok(())
    }

    #[test]
    fn bad_type_apply() -> Result<(), Box<dyn std::error::Error>> {
        let reg = ExtensionRegistry::try_new([PRELUDE.to_owned(), EXTENSION.to_owned()]).unwrap();
        let pf = nested_func();
        let pf_usz = pf.instantiate_poly(&[USIZE_TA], &reg)?;
        let pf_bool = pf.instantiate_poly(&[TypeArg::Type { ty: BOOL_T }], &reg)?;
        let mut dfg = DFGBuilder::new(FunctionType::new(
            vec![Type::new_function(pf.clone())],
            vec![Type::new_function(pf_usz.clone())],
        ))?;
        let ta = dfg.add_dataflow_op(
            LeafOp::TypeApply {
                ta: TypeApplication {
                    input: pf,
                    args: vec![TypeArg::Type { ty: BOOL_T }],
                    output: pf_usz.clone(),
                },
            },
            dfg.input_wires(),
        )?;
        let res = dfg.finish_hugr_with_outputs(ta.outputs(), &reg);
        assert_eq!(
            res.unwrap_err(),
            BuildError::InvalidHUGR(ValidationError::SignatureError {
                node: ta.node(),
                cause: SignatureError::TypeApplyIncorrectCache {
                    cached: pf_usz,
                    expected: pf_bool
                }
            })
        );
        Ok(())
    }
}
