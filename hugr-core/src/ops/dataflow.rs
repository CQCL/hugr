//! Dataflow operations.

use super::{impl_op_name, OpTag, OpTrait};

use crate::extension::{ExtensionRegistry, ExtensionSet, SignatureError};
use crate::ops::StaticTag;
use crate::types::{EdgeKind, FunctionType, PolyFuncType, Signature, Type, TypeArg, TypeRow};
use crate::IncomingPort;

#[cfg(test)]
use ::proptest_derive::Arbitrary;

pub(crate) trait DataflowOpTrait {
    const TAG: OpTag;
    fn description(&self) -> &str;
    fn signature(&self) -> Signature;

    /// The edge kind for the non-dataflow or constant inputs of the operation,
    /// not described by the signature.
    ///
    /// If not None, a single extra output multiport of that kind will be
    /// present.
    #[inline]
    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
    /// The edge kind for the non-dataflow outputs of the operation, not
    /// described by the signature.
    ///
    /// If not None, a single extra output multiport of that kind will be
    /// present.
    #[inline]
    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    /// The edge kind for a single constant input of the operation, not
    /// described by the dataflow signature.
    ///
    /// If not None, an extra input port of that kind will be present after the
    /// dataflow input ports and before any [`DataflowOpTrait::other_input`] ports.
    #[inline]
    fn static_input(&self) -> Option<EdgeKind> {
        None
    }
}

/// Helpers to construct input and output nodes
pub trait IOTrait {
    /// Construct a new I/O node from a type row with no extension requirements
    fn new(types: impl Into<TypeRow>) -> Self;
}

/// An input node.
/// The outputs of this node are the inputs to the function.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct Input {
    /// Input value types
    pub types: TypeRow,
}

impl_op_name!(Input);

impl IOTrait for Input {
    fn new(types: impl Into<TypeRow>) -> Self {
        Input {
            types: types.into(),
        }
    }
}

/// An output node. The inputs are the outputs of the function.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct Output {
    /// Output value types
    pub types: TypeRow,
}

impl_op_name!(Output);

impl IOTrait for Output {
    fn new(types: impl Into<TypeRow>) -> Self {
        Output {
            types: types.into(),
        }
    }
}

impl DataflowOpTrait for Input {
    const TAG: OpTag = OpTag::Input;

    fn description(&self) -> &str {
        "The input node for this dataflow subgraph"
    }

    fn other_input(&self) -> Option<EdgeKind> {
        None
    }

    fn signature(&self) -> Signature {
        FunctionType::new(TypeRow::new(), self.types.clone())
    }
}
impl DataflowOpTrait for Output {
    const TAG: OpTag = OpTag::Output;

    fn description(&self) -> &str {
        "The output node for this dataflow subgraph"
    }

    // Note: We know what the input extensions should be, so we *could* give an
    // instantiated Signature instead
    fn signature(&self) -> Signature {
        FunctionType::new(self.types.clone(), TypeRow::new())
    }

    fn other_output(&self) -> Option<EdgeKind> {
        None
    }
}

impl<T: DataflowOpTrait> OpTrait for T {
    fn description(&self) -> &str {
        DataflowOpTrait::description(self)
    }
    fn tag(&self) -> OpTag {
        T::TAG
    }
    fn dataflow_signature(&self) -> Option<Signature> {
        Some(DataflowOpTrait::signature(self))
    }
    fn extension_delta(&self) -> ExtensionSet {
        DataflowOpTrait::signature(self).extension_reqs.clone()
    }
    fn other_input(&self) -> Option<EdgeKind> {
        DataflowOpTrait::other_input(self)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        DataflowOpTrait::other_output(self)
    }

    fn static_input(&self) -> Option<EdgeKind> {
        DataflowOpTrait::static_input(self)
    }
}
impl<T: DataflowOpTrait> StaticTag for T {
    const TAG: OpTag = T::TAG;
}

/// Call a function directly.
///
/// The first ports correspond to the signature of the function being called.
/// The port immediately following those those is connected to the def/declare
/// block with a [`EdgeKind::Function`] edge.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct Call {
    /// Signature of function being called
    func_sig: PolyFuncType<false>,
    type_args: Vec<TypeArg>,
    instantiation: Signature, // Cache, so we can fail in try_new() not in signature()
}
impl_op_name!(Call);

impl DataflowOpTrait for Call {
    const TAG: OpTag = OpTag::FnCall;

    fn description(&self) -> &str {
        "Call a function directly"
    }

    fn signature(&self) -> Signature {
        self.instantiation.clone()
    }

    fn static_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Function(self.called_function_type().clone()))
    }
}
impl Call {
    /// Try to make a new Call. Returns an error if the `type_args`` do not fit the [TypeParam]s
    /// declared by the function.
    ///
    /// [TypeParam]: crate::types::type_param::TypeParam
    pub fn try_new(
        func_sig: PolyFuncType<false>,
        type_args: impl Into<Vec<TypeArg>>,
        exts: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let type_args = type_args.into();
        let instantiation = func_sig.instantiate(&type_args, exts)?;
        Ok(Self {
            func_sig,
            type_args,
            instantiation,
        })
    }

    #[inline]
    /// Return the signature of the function called by this op.
    pub fn called_function_type(&self) -> &PolyFuncType<false> {
        &self.func_sig
    }

    /// The IncomingPort which links to the function being called.
    ///
    /// This matches [`OpType::static_input_port`].
    ///
    /// ```
    /// # use hugr::ops::dataflow::Call;
    /// # use hugr::ops::OpType;
    /// # use hugr::types::FunctionType;
    /// # use hugr::extension::prelude::QB_T;
    /// # use hugr::extension::PRELUDE_REGISTRY;
    /// let signature = FunctionType::new(vec![QB_T, QB_T], vec![QB_T, QB_T]);
    /// let call = Call::try_new(signature.into(), &[], &PRELUDE_REGISTRY).unwrap();
    /// let op = OpType::Call(call.clone());
    /// assert_eq!(op.static_input_port(), Some(call.called_function_port()));
    /// ```
    ///
    /// [`OpType::static_input_port`]: crate::ops::OpType::static_input_port
    #[inline]
    pub fn called_function_port(&self) -> IncomingPort {
        self.instantiation.input_count().into()
    }

    pub(crate) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), SignatureError> {
        let other = Self::try_new(
            self.func_sig.clone(),
            self.type_args.clone(),
            extension_registry,
        )?;
        if other.instantiation == self.instantiation {
            Ok(())
        } else {
            Err(SignatureError::CallIncorrectlyAppliesType {
                cached: self.instantiation.clone(),
                expected: other.instantiation.clone(),
            })
        }
    }
}

/// Call a function indirectly. Like call, but the function input is a value
/// (runtime, not static) dataflow edge, and thus does not need any type-args.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct CallIndirect {
    /// Signature of function being called
    pub signature: Signature,
}
impl_op_name!(CallIndirect);

impl DataflowOpTrait for CallIndirect {
    const TAG: OpTag = OpTag::FnCall;

    fn description(&self) -> &str {
        "Call a function indirectly"
    }

    fn signature(&self) -> Signature {
        let mut s = self.signature.clone();
        s.input
            .to_mut()
            .insert(0, Type::new_function(self.signature.clone().into_()));
        s
    }
}

/// Load a static constant in to the local dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct LoadConstant {
    /// Constant type
    pub datatype: Type,
}
impl_op_name!(LoadConstant);
impl DataflowOpTrait for LoadConstant {
    const TAG: OpTag = OpTag::LoadConst;

    fn description(&self) -> &str {
        "Load a static constant in to the local dataflow graph"
    }

    fn signature(&self) -> Signature {
        FunctionType::new(TypeRow::new(), vec![self.datatype.clone()])
    }

    fn static_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Const(self.constant_type().clone()))
    }
}
impl LoadConstant {
    #[inline]
    /// The type of the constant loaded by this op.
    pub fn constant_type(&self) -> &Type {
        &self.datatype
    }

    /// The IncomingPort which links to the loaded constant.
    ///
    /// This matches [`OpType::static_input_port`].
    ///
    /// ```
    /// # use hugr::ops::dataflow::LoadConstant;
    /// # use hugr::ops::OpType;
    /// # use hugr::types::Type;
    /// let datatype = Type::UNIT;
    /// let load_constant = LoadConstant { datatype };
    /// let op = OpType::LoadConstant(load_constant.clone());
    /// assert_eq!(op.static_input_port(), Some(load_constant.constant_port()));
    /// ```
    ///
    /// [`OpType::static_input_port`]: crate::ops::OpType::static_input_port
    #[inline]
    pub fn constant_port(&self) -> IncomingPort {
        0.into()
    }
}

/// Load a static function in to the local dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct LoadFunction {
    /// Signature of the function
    func_sig: PolyFuncType<false>,
    type_args: Vec<TypeArg>,
    signature: Signature, // Cache, so we can fail in try_new() not in signature()
}
impl_op_name!(LoadFunction);
impl DataflowOpTrait for LoadFunction {
    const TAG: OpTag = OpTag::LoadFunc;

    fn description(&self) -> &str {
        "Load a static function in to the local dataflow graph"
    }

    fn signature(&self) -> Signature {
        self.signature.clone()
    }

    fn static_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Function(self.func_sig.clone()))
    }
}
impl LoadFunction {
    /// Try to make a new LoadFunction op. Returns an error if the `type_args`` do not fit
    /// the [TypeParam]s declared by the function.
    ///
    /// [TypeParam]: crate::types::type_param::TypeParam
    pub fn try_new(
        func_sig: PolyFuncType<false>,
        type_args: impl Into<Vec<TypeArg>>,
        exts: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let type_args = type_args.into();
        let instantiation = func_sig.instantiate(&type_args, exts)?.into_();
        let signature = FunctionType::new(TypeRow::new(), vec![Type::new_function(instantiation)]);
        Ok(Self {
            func_sig,
            type_args,
            signature,
        })
    }

    #[inline]
    /// Return the type of the function loaded by this op.
    pub fn function_type(&self) -> &PolyFuncType<false> {
        &self.func_sig
    }

    /// The IncomingPort which links to the loaded function.
    ///
    /// This matches [`OpType::static_input_port`].
    ///
    /// [`OpType::static_input_port`]: crate::ops::OpType::static_input_port
    #[inline]
    pub fn function_port(&self) -> IncomingPort {
        0.into()
    }

    pub(crate) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), SignatureError> {
        let other = Self::try_new(
            self.func_sig.clone(),
            self.type_args.clone(),
            extension_registry,
        )?;
        if other.signature == self.signature {
            Ok(())
        } else {
            Err(SignatureError::LoadFunctionIncorrectlyAppliesType {
                cached: self.signature.clone(),
                expected: other.signature.clone(),
            })
        }
    }
}

/// Operations that is the parent of a dataflow graph.
pub trait DataflowParent {
    /// Signature of the inner dataflow graph.
    fn inner_signature(&self) -> Signature;
}

/// A simply nested dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct DFG {
    /// Signature of DFG node
    pub signature: Signature,
}

impl_op_name!(DFG);

impl DataflowParent for DFG {
    fn inner_signature(&self) -> Signature {
        self.signature.clone()
    }
}

impl DataflowOpTrait for DFG {
    const TAG: OpTag = OpTag::Dfg;

    fn description(&self) -> &str {
        "A simply nested dataflow graph"
    }

    fn signature(&self) -> Signature {
        self.inner_signature()
    }
}
