//! Dataflow operations.

use std::borrow::Cow;

use super::{OpTag, OpTrait, impl_op_name};

use crate::extension::SignatureError;
use crate::ops::StaticTag;
use crate::types::{EdgeKind, PolyFuncType, Signature, Substitution, Type, TypeArg, TypeRow};
use crate::{IncomingPort, type_row};

#[cfg(test)]
use {crate::types::proptest_utils::any_serde_type_arg_vec, proptest_derive::Arbitrary};

/// Trait implemented by all dataflow operations.
pub trait DataflowOpTrait: Sized {
    /// Tag identifying the operation.
    const TAG: OpTag;

    /// A human-readable description of the operation.
    fn description(&self) -> &str;

    /// The signature of the operation.
    fn signature(&self) -> Cow<'_, Signature>;

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

    /// Apply a type-level substitution to this `OpType`, i.e. replace
    /// [type variables](TypeArg::new_var_use) with new types.
    fn substitute(&self, _subst: &Substitution) -> Self;
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

    fn description(&self) -> &'static str {
        "The input node for this dataflow subgraph"
    }

    fn other_input(&self) -> Option<EdgeKind> {
        None
    }

    fn signature(&self) -> Cow<'_, Signature> {
        // TODO: Store a cached signature
        Cow::Owned(Signature::new(TypeRow::new(), self.types.clone()))
    }

    fn substitute(&self, subst: &Substitution) -> Self {
        Self {
            types: self.types.substitute(subst),
        }
    }
}
impl DataflowOpTrait for Output {
    const TAG: OpTag = OpTag::Output;

    fn description(&self) -> &'static str {
        "The output node for this dataflow subgraph"
    }

    // Note: We know what the input extensions should be, so we *could* give an
    // instantiated Signature instead
    fn signature(&self) -> Cow<'_, Signature> {
        // TODO: Store a cached signature
        Cow::Owned(Signature::new(self.types.clone(), TypeRow::new()))
    }

    fn other_output(&self) -> Option<EdgeKind> {
        None
    }

    fn substitute(&self, subst: &Substitution) -> Self {
        Self {
            types: self.types.substitute(subst),
        }
    }
}

impl<T: DataflowOpTrait + Clone> OpTrait for T {
    fn description(&self) -> &str {
        DataflowOpTrait::description(self)
    }

    fn tag(&self) -> OpTag {
        T::TAG
    }

    fn dataflow_signature(&self) -> Option<Cow<'_, Signature>> {
        Some(DataflowOpTrait::signature(self))
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

    fn substitute(&self, subst: &crate::types::Substitution) -> Self {
        DataflowOpTrait::substitute(self, subst)
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
    /// Signature of function being called.
    pub func_sig: PolyFuncType,
    /// The type arguments that instantiate `func_sig`.
    #[cfg_attr(test, proptest(strategy = "any_serde_type_arg_vec()"))]
    pub type_args: Vec<TypeArg>,
    /// The instantiation of `func_sig`.
    pub instantiation: Signature, // Cache, so we can fail in try_new() not in signature()
}
impl_op_name!(Call);

impl DataflowOpTrait for Call {
    const TAG: OpTag = OpTag::FnCall;

    fn description(&self) -> &'static str {
        "Call a function directly"
    }

    fn signature(&self) -> Cow<'_, Signature> {
        Cow::Borrowed(&self.instantiation)
    }

    fn static_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Function(self.called_function_type().clone()))
    }

    fn substitute(&self, subst: &Substitution) -> Self {
        let type_args = self
            .type_args
            .iter()
            .map(|ta| ta.substitute(subst))
            .collect::<Vec<_>>();
        let instantiation = self.instantiation.substitute(subst);
        debug_assert_eq!(
            self.func_sig.instantiate(&type_args).as_ref(),
            Ok(&instantiation)
        );
        Self {
            type_args,
            instantiation,
            func_sig: self.func_sig.clone(),
        }
    }
}
impl Call {
    /// Try to make a new Call. Returns an error if the `type_args`` do not fit the [TypeParam]s
    /// declared by the function.
    ///
    /// [TypeParam]: crate::types::type_param::TypeParam
    pub fn try_new(
        func_sig: PolyFuncType,
        type_args: impl Into<Vec<TypeArg>>,
    ) -> Result<Self, SignatureError> {
        let type_args: Vec<_> = type_args.into();
        let instantiation = func_sig.instantiate(&type_args)?;
        Ok(Self {
            func_sig,
            type_args,
            instantiation,
        })
    }

    #[inline]
    /// Return the signature of the function called by this op.
    #[must_use]
    pub fn called_function_type(&self) -> &PolyFuncType {
        &self.func_sig
    }

    /// The `IncomingPort` which links to the function being called.
    ///
    /// This matches [`OpType::static_input_port`].
    ///
    /// ```
    /// # use hugr::ops::dataflow::Call;
    /// # use hugr::ops::OpType;
    /// # use hugr::types::Signature;
    /// # use hugr::extension::prelude::qb_t;
    /// # use hugr::extension::PRELUDE_REGISTRY;
    /// let signature = Signature::new(vec![qb_t(), qb_t()], vec![qb_t(), qb_t()]);
    /// let call = Call::try_new(signature.into(), &[]).unwrap();
    /// let op = OpType::Call(call.clone());
    /// assert_eq!(op.static_input_port(), Some(call.called_function_port()));
    /// ```
    ///
    /// [`OpType::static_input_port`]: crate::ops::OpType::static_input_port
    #[inline]
    #[must_use]
    pub fn called_function_port(&self) -> IncomingPort {
        self.instantiation.input_count().into()
    }

    pub(crate) fn validate(&self) -> Result<(), SignatureError> {
        let other = Self::try_new(self.func_sig.clone(), self.type_args.clone())?;
        if other.instantiation == self.instantiation {
            Ok(())
        } else {
            Err(SignatureError::CallIncorrectlyAppliesType {
                cached: Box::new(self.instantiation.clone()),
                expected: Box::new(other.instantiation.clone()),
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
    const TAG: OpTag = OpTag::DataflowChild;

    fn description(&self) -> &'static str {
        "Call a function indirectly"
    }

    fn signature(&self) -> Cow<'_, Signature> {
        // TODO: Store a cached signature
        let mut s = self.signature.clone();
        s.input
            .to_mut()
            .insert(0, Type::new_function(self.signature.clone()));
        Cow::Owned(s)
    }

    fn substitute(&self, subst: &Substitution) -> Self {
        Self {
            signature: self.signature.substitute(subst),
        }
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

    fn description(&self) -> &'static str {
        "Load a static constant in to the local dataflow graph"
    }

    fn signature(&self) -> Cow<'_, Signature> {
        // TODO: Store a cached signature
        Cow::Owned(Signature::new(TypeRow::new(), vec![self.datatype.clone()]))
    }

    fn static_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Const(self.constant_type().clone()))
    }

    fn substitute(&self, _subst: &Substitution) -> Self {
        // Constants cannot refer to TypeArgs, so neither can loading them
        self.clone()
    }
}

impl LoadConstant {
    #[inline]
    /// The type of the constant loaded by this op.
    #[must_use]
    pub fn constant_type(&self) -> &Type {
        &self.datatype
    }

    /// The `IncomingPort` which links to the loaded constant.
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
    #[must_use]
    pub fn constant_port(&self) -> IncomingPort {
        0.into()
    }
}

/// Load a static function in to the local dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct LoadFunction {
    /// Signature of the function
    pub func_sig: PolyFuncType,
    /// The type arguments that instantiate `func_sig`.
    #[cfg_attr(test, proptest(strategy = "any_serde_type_arg_vec()"))]
    pub type_args: Vec<TypeArg>,
    /// The instantiation of `func_sig`.
    pub instantiation: Signature, // Cache, so we can fail in try_new() not in signature()
}
impl_op_name!(LoadFunction);
impl DataflowOpTrait for LoadFunction {
    const TAG: OpTag = OpTag::LoadFunc;

    fn description(&self) -> &'static str {
        "Load a static function in to the local dataflow graph"
    }

    fn signature(&self) -> Cow<'_, Signature> {
        Cow::Owned(Signature::new(
            type_row![],
            Type::new_function(self.instantiation.clone()),
        ))
    }

    fn static_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Function(self.func_sig.clone()))
    }

    fn substitute(&self, subst: &Substitution) -> Self {
        let type_args = self
            .type_args
            .iter()
            .map(|ta| ta.substitute(subst))
            .collect::<Vec<_>>();
        let instantiation = self.instantiation.substitute(subst);
        debug_assert_eq!(
            self.func_sig.instantiate(&type_args).as_ref(),
            Ok(&instantiation)
        );
        Self {
            func_sig: self.func_sig.clone(),
            type_args,
            instantiation,
        }
    }
}
impl LoadFunction {
    /// Try to make a new LoadFunction op. Returns an error if the `type_args`` do not fit
    /// the [TypeParam]s declared by the function.
    ///
    /// [TypeParam]: crate::types::type_param::TypeParam
    pub fn try_new(
        func_sig: PolyFuncType,
        type_args: impl Into<Vec<TypeArg>>,
    ) -> Result<Self, SignatureError> {
        let type_args: Vec<_> = type_args.into();
        let instantiation = func_sig.instantiate(&type_args)?;
        Ok(Self {
            func_sig,
            type_args,
            instantiation,
        })
    }

    #[inline]
    /// Return the type of the function loaded by this op.
    #[must_use]
    pub fn function_type(&self) -> &PolyFuncType {
        &self.func_sig
    }

    /// The `IncomingPort` which links to the loaded function.
    ///
    /// This matches [`OpType::static_input_port`].
    ///
    /// [`OpType::static_input_port`]: crate::ops::OpType::static_input_port
    #[inline]
    #[must_use]
    pub fn function_port(&self) -> IncomingPort {
        0.into()
    }

    pub(crate) fn validate(&self) -> Result<(), SignatureError> {
        let other = Self::try_new(self.func_sig.clone(), self.type_args.clone())?;
        if other.instantiation == self.instantiation {
            Ok(())
        } else {
            Err(SignatureError::LoadFunctionIncorrectlyAppliesType {
                cached: Box::new(self.instantiation.clone()),
                expected: Box::new(other.instantiation.clone()),
            })
        }
    }
}

/// An operation that is the parent of a dataflow graph.
///
/// The children region contains an input and an output node matching the
/// signature returned by [`DataflowParent::inner_signature`].
pub trait DataflowParent {
    /// Signature of the inner dataflow graph.
    fn inner_signature(&self) -> Cow<'_, Signature>;
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
    fn inner_signature(&self) -> Cow<'_, Signature> {
        Cow::Borrowed(&self.signature)
    }
}

impl DataflowOpTrait for DFG {
    const TAG: OpTag = OpTag::Dfg;

    fn description(&self) -> &'static str {
        "A simply nested dataflow graph"
    }

    fn signature(&self) -> Cow<'_, Signature> {
        self.inner_signature()
    }

    fn substitute(&self, subst: &Substitution) -> Self {
        Self {
            signature: self.signature.substitute(subst),
        }
    }
}
