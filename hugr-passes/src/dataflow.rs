//! Dataflow analysis of Hugrs.

mod datalog;
pub use datalog::Machine;
mod value_row;

mod results;
pub use results::{AnalysisResults, TailLoopTermination};

mod partial_value;
pub use partial_value::{AbstractValue, AsConcrete, LoadedFunction, PartialSum, PartialValue, Sum};

use hugr_core::Hugr;
use hugr_core::ops::constant::OpaqueValue;
use hugr_core::ops::{ExtensionOp, Value};

/// Clients of the dataflow framework (particular analyses, such as constant folding)
/// must implement this trait (including providing an appropriate domain type `V`).
pub trait DFContext<V>: ConstLoader<V> {
    /// Given lattice values for each input, update lattice values for the (dataflow) outputs.
    /// For extension ops only, excluding [`MakeTuple`] and [`UnpackTuple`] which are handled automatically.
    /// `_outs` is an array with one element per dataflow output, each initialized to [`PartialValue::Top`]
    /// which is the correct value to leave if nothing can be deduced about that output.
    /// (The default does nothing, i.e. leaves `Top` for all outputs.)
    ///
    /// [`MakeTuple`]: hugr_core::extension::prelude::MakeTuple
    /// [`UnpackTuple`]: hugr_core::extension::prelude::UnpackTuple
    fn interpret_leaf_op(
        &mut self,
        _node: Self::Node,
        _e: &ExtensionOp,
        _ins: &[PartialValue<V, Self::Node>],
        _outs: &mut [PartialValue<V, Self::Node>],
    ) {
    }
}

/// A location where a [Value] could be find in a Hugr. That is,
/// (perhaps deeply nested within [`Value::Sum`]s) within a node `N`
/// that is a [Const](hugr_core::ops::Const).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstLocation<'a, N> {
    /// The specified-index'th field of the [`Value::Sum`] constant identified by the RHS
    Field(usize, &'a ConstLocation<'a, N>),
    /// The entire ([`Const::value`](hugr_core::ops::Const::value)) of the node.
    Node(N),
}

impl<N> From<N> for ConstLocation<'_, N> {
    fn from(value: N) -> Self {
        ConstLocation::Node(value)
    }
}

/// Trait for loading [`PartialValue`]s from constant [Value]s in a Hugr.
///
/// Implementors will likely want to override either/both of [`Self::value_from_opaque`]
/// and [`Self::value_from_const_hugr`]: the defaults
/// are "correct" but maximally conservative (minimally informative).
pub trait ConstLoader<V> {
    /// The type of nodes in the Hugr.
    type Node;

    /// Produces an abstract value from an [`OpaqueValue`], if possible.
    /// The default just returns `None`, which will be interpreted as [`PartialValue::Top`].
    fn value_from_opaque(&self, _loc: ConstLocation<Self::Node>, _val: &OpaqueValue) -> Option<V> {
        None
    }

    /// Produces an abstract value from a Hugr in a [`Value::Function`], if possible.
    /// The default just returns `None`, which will be interpreted as [`PartialValue::Top`].
    fn value_from_const_hugr(&self, _loc: ConstLocation<Self::Node>, _h: &Hugr) -> Option<V> {
        None
    }
}

/// Produces a [`PartialValue`] from a constant.
///
/// Traverses [Sum](Value::Sum) constants to their leaves ([`Value::Extension`] and [`Value::Function`]),
/// converts these using [`ConstLoader::value_from_opaque`] and [`ConstLoader::value_from_const_hugr`],
/// and builds nested [`PartialValue::new_variant`] to represent the structure.
pub fn partial_from_const<'a, V, CL: ConstLoader<V>>(
    cl: &CL,
    loc: impl Into<ConstLocation<'a, CL::Node>>,
    cst: &Value,
) -> PartialValue<V, CL::Node>
where
    CL::Node: 'a,
{
    let loc = loc.into();
    match cst {
        Value::Sum(hugr_core::ops::constant::Sum { tag, values, .. }) => {
            let elems = values
                .iter()
                .enumerate()
                .map(|(idx, elem)| partial_from_const(cl, ConstLocation::Field(idx, &loc), elem));
            PartialValue::new_variant(*tag, elems)
        }
        Value::Extension { e } => cl
            .value_from_opaque(loc, e)
            .map_or(PartialValue::Top, PartialValue::from),
        Value::Function { hugr } => cl
            .value_from_const_hugr(loc, hugr)
            .map_or(PartialValue::Top, PartialValue::from),
    }
}

/// A row of inputs to a node contains bottom (can't happen, the node
/// can't execute) if any element [`contains_bottom`](PartialValue::contains_bottom).
pub fn row_contains_bottom<'a, V: 'a, N: 'a>(
    elements: impl IntoIterator<Item = &'a PartialValue<V, N>>,
) -> bool {
    elements.into_iter().any(PartialValue::contains_bottom)
}

#[cfg(test)]
mod test;
