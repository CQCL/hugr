#![warn(missing_docs)]
//! Dataflow analysis of Hugrs.

mod datalog;

mod machine;
use hugr_core::ops::constant::CustomConst;
pub use machine::{AnalysisResults, Machine, TailLoopTermination};

mod partial_value;
pub use partial_value::{AbstractValue, PartialSum, PartialValue, Sum};

use hugr_core::ops::{ExtensionOp, Value};
use hugr_core::{Hugr, Node};

/// Clients of the dataflow framework (particular analyses, such as constant folding)
/// must implement this trait (including providing an appropriate domain type `V`).
pub trait DFContext<V> {
    /// Given lattice values for each input, update lattice values for the (dataflow) outputs.
    /// For extension ops only, excluding [MakeTuple] and [UnpackTuple].
    /// `_outs` is an array with one element per dataflow output, each initialized to [PartialValue::Top]
    /// which is the correct value to leave if nothing can be deduced about that output.
    /// (The default does nothing, i.e. leaves `Top` for all outputs.)
    ///
    /// [MakeTuple]: hugr_core::extension::prelude::MakeTuple
    /// [UnpackTuple]: hugr_core::extension::prelude::UnpackTuple
    fn interpret_leaf_op(
        &self,
        _node: Node,
        _e: &ExtensionOp,
        _ins: &[PartialValue<V>],
        _outs: &mut [PartialValue<V>],
    ) {
    }

    /// Produces an abstract value from a constant. The default impl
    /// traverses the constant [Value] to its leaves ([Value::Extension] and [Value::Function]),
    /// converts these using [Self::value_from_custom_const] and [Self::value_from_const_hugr],
    /// and builds nested [PartialValue::new_variant] to represent the structure.
    fn value_from_const(&self, n: Node, cst: &Value) -> PartialValue<V> {
        traverse_value(self, n, &mut Vec::new(), cst)
    }

    /// Produces an abstract value from a [CustomConst], if possible.
    /// The default just returns `None`, which will be interpreted as [PartialValue::Top].
    fn value_from_custom_const(
        &self,
        _node: Node,
        _fields: &[usize],
        _cc: &dyn CustomConst,
    ) -> Option<V> {
        None
    }

    /// Produces an abstract value from a Hugr in a [Value::Function], if possible.
    /// The default just returns `None`, which will be interpreted as [PartialValue::Top].
    fn value_from_const_hugr(&self, _node: Node, _fields: &[usize], _h: &Hugr) -> Option<V> {
        None
    }
}

fn traverse_value<V>(
    s: &(impl DFContext<V> + ?Sized),
    n: Node,
    fields: &mut Vec<usize>,
    cst: &Value,
) -> PartialValue<V> {
    match cst {
        Value::Sum(hugr_core::ops::constant::Sum { tag, values, .. }) => {
            let elems = values.iter().enumerate().map(|(idx, elem)| {
                fields.push(idx);
                let r = traverse_value(s, n, fields, elem);
                fields.pop();
                r
            });
            PartialValue::new_variant(*tag, elems)
        }
        Value::Extension { e } => s
            .value_from_custom_const(n, fields, e.value())
            .map(PartialValue::from)
            .unwrap_or(PartialValue::Top),
        Value::Function { hugr } => s
            .value_from_const_hugr(n, fields, hugr)
            .map(PartialValue::from)
            .unwrap_or(PartialValue::Top),
    }
}

#[cfg(test)]
mod test;
