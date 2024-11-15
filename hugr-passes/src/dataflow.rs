#![warn(missing_docs)]
//! Dataflow analysis of Hugrs.

mod datalog;
pub use datalog::Machine;
mod value_row;

mod results;
pub use results::{AnalysisResults, TailLoopTermination};

mod partial_value;
pub use partial_value::{AbstractValue, PartialSum, PartialValue, Sum, NoDefaultConversionToSum};

use hugr_core::ops::constant::OpaqueValue;
use hugr_core::ops::{ExtensionOp, Value};
use hugr_core::types::TypeArg;
use hugr_core::{Hugr, HugrView, Node};

/// Clients of the dataflow framework (particular analyses, such as constant folding)
/// must implement this trait (including providing an appropriate domain type `V`).
pub trait DFContext<V>: ConstLoader<V> + std::ops::Deref<Target = Self::View> {
    /// Type of view contained within this context. (Ideally we'd constrain
    /// by `std::ops::Deref<Target: impl HugrView>` but that's not stable yet.)
    type View: HugrView;

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
}

/// Trait for loading [PartialValue]s from constant [Value]s in a Hugr.
/// Implementors will likely want to override some/all of [Self::value_from_opaque],
/// [Self::value_from_const_hugr], and [Self::value_from_function]: the defaults
/// are "correct" but maximally conservative (minimally informative).
pub trait ConstLoader<V> {
    /// Produces a [PartialValue] from a constant. The default impl (expected
    /// to be appropriate in most cases) uses [partial_from_const].
    fn value_from_const(&self, n: Node, cst: &Value) -> PartialValue<V> {
        partial_from_const(self, n, &mut Vec::new(), cst)
    }

    /// Produces an abstract value from an [OpaqueValue], if possible.
    /// The default just returns `None`, which will be interpreted as [PartialValue::Top].
    fn value_from_opaque(&self, _node: Node, _fields: &[usize], _val: &OpaqueValue) -> Option<V> {
        None
    }

    /// Produces an abstract value from a Hugr in a [Value::Function], if possible.
    /// The default just returns `None`, which will be interpreted as [PartialValue::Top].
    fn value_from_const_hugr(&self, _node: Node, _fields: &[usize], _h: &Hugr) -> Option<V> {
        None
    }

    /// Produces an abstract value from a [FuncDefn] or [FuncDecl] node
    /// (that has been loaded via a [LoadFunction]), if possible.
    /// The default just returns `None`, which will be interpreted as [PartialValue::Top].
    ///
    /// [FuncDefn]: hugr_core::ops::FuncDefn
    /// [FuncDecl]: hugr_core::ops::FuncDecl
    /// [LoadFunction]: hugr_core::ops::LoadFunction
    fn value_from_function(&self, _node: Node, _type_args: &[TypeArg]) -> Option<V> {
        None
    }
}

/// Converts a constant [Value] by traversing [Sum](Value::Sum) constants
/// to their leaves ([Value::Extension] and [Value::Function]),
/// converting these using [ConstLoader::value_from_opaque] and [ConstLoader::value_from_const_hugr],
/// and building nested [PartialValue::new_variant]s to represent the structure.
pub fn partial_from_const<V>(
    s: &(impl ConstLoader<V> + ?Sized),
    n: Node,
    fields: &mut Vec<usize>,
    cst: &Value,
) -> PartialValue<V> {
    match cst {
        Value::Sum(hugr_core::ops::constant::Sum { tag, values, .. }) => {
            let elems = values.iter().enumerate().map(|(idx, elem)| {
                fields.push(idx);
                let r = partial_from_const(s, n, fields, elem);
                fields.pop();
                r
            });
            PartialValue::new_variant(*tag, elems)
        }
        Value::Extension { e } => s
            .value_from_opaque(n, fields, e)
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
