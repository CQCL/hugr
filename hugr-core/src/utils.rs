//! General utilities.

use std::fmt::{self, Debug, Display};

use itertools::Itertools;

use crate::{Hugr, HugrView, IncomingPort, Node, ops::Value};

/// Write a comma separated list of of some types.
/// Like `debug_list`, but using the Display instance rather than Debug,
/// and not adding surrounding square brackets.
pub fn display_list<T>(ts: impl IntoIterator<Item = T>, f: &mut fmt::Formatter) -> fmt::Result
where
    T: Display,
{
    display_list_with_separator(ts, f, ", ")
}

/// Write a separated list of of some types, using a custom separator.
/// Like `debug_list`, but using the Display instance rather than Debug,
/// and not adding surrounding square brackets.
pub fn display_list_with_separator<T>(
    ts: impl IntoIterator<Item = T>,
    f: &mut fmt::Formatter,
    sep: &str,
) -> fmt::Result
where
    T: Display,
{
    let mut first = true;
    for t in ts {
        if !first {
            f.write_str(sep)?;
        }
        t.fmt(f)?;
        if first {
            first = false;
        }
    }
    Ok(())
}

/// Collect a vector into an array.
///
/// This is useful for deconstructing a vectors content.
///
/// # Example
///
/// ```ignore
/// let iter = 0..3;
/// let [a, b, c] = crate::utils::collect_array(iter);
/// assert_eq!(b, 1);
/// ```
///
/// # Panics
///
/// If the length of the slice is not equal to `N`.
///
/// See also [`try_collect_array`] for a non-panicking version.
#[inline]
#[track_caller]
pub fn collect_array<const N: usize, T: Debug>(arr: impl IntoIterator<Item = T>) -> [T; N] {
    match try_collect_array(arr) {
        Ok(v) => v,
        Err(v) => panic!("Expected {N} elements, got {v:?}"),
    }
}

/// Collect a vector into an array.
///
/// This is useful for deconstructing a vectors content.
///
/// # Example
///
/// ```ignore
/// let iter = 0..3;
/// let [a, b, c] = crate::utils::try_collect_array(iter)
///     .unwrap_or_else(|v| panic!("Expected 3 elements, got {:?}", v));
/// assert_eq!(b, 1);
/// ```
///
/// See also [`collect_array`].
#[inline]
#[track_caller]
pub fn try_collect_array<const N: usize, T>(
    arr: impl IntoIterator<Item = T>,
) -> Result<[T; N], Vec<T>> {
    arr.into_iter().collect_vec().try_into()
}

/// Helper method to skip serialization of default values in serde.
///
/// ```ignore
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct MyStruct {
///     #[serde(skip_serializing_if = "crate::utils::is_default")]
///     field: i32,
/// }
/// ```
///
/// From <https://github.com/serde-rs/serde/issues/818>.
#[allow(dead_code)]
pub(crate) fn is_default<T: Default + PartialEq>(t: &T) -> bool {
    *t == Default::default()
}

/// An empty type.
///
/// # Example
///
/// ```ignore
/// fn foo(never: Never) -> ! {
///     match never {}
/// }
/// ```
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Never {}

#[cfg(test)]
pub(crate) mod test_quantum_extension {
    use std::sync::{Arc, LazyLock};

    use crate::ops::{OpName, OpNameRef};
    use crate::std_extensions::arithmetic::float_ops;
    use crate::std_extensions::logic;
    use crate::types::FuncValueType;
    use crate::{
        Extension,
        extension::{
            ExtensionId, ExtensionRegistry, PRELUDE,
            prelude::{bool_t, qb_t},
        },
        ops::ExtensionOp,
        std_extensions::arithmetic::float_types,
        type_row,
        types::{PolyFuncTypeRV, Signature},
    };

    fn one_qb_func() -> PolyFuncTypeRV {
        FuncValueType::new_endo(qb_t()).into()
    }

    fn two_qb_func() -> PolyFuncTypeRV {
        FuncValueType::new_endo(vec![qb_t(), qb_t()]).into()
    }
    /// The extension identifier.
    pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("test.quantum");
    fn extension() -> Arc<Extension> {
        Extension::new_test_arc(EXTENSION_ID, |extension, extension_ref| {
            extension
                .add_op(
                    OpName::new_inline("H"),
                    "Hadamard".into(),
                    one_qb_func(),
                    extension_ref,
                )
                .unwrap();
            extension
                .add_op(
                    OpName::new_inline("RzF64"),
                    "Rotation specified by float".into(),
                    Signature::new(vec![qb_t(), float_types::float64_type()], vec![qb_t()]),
                    extension_ref,
                )
                .unwrap();

            extension
                .add_op(
                    OpName::new_inline("CX"),
                    "CX".into(),
                    two_qb_func(),
                    extension_ref,
                )
                .unwrap();

            extension
                .add_op(
                    OpName::new_inline("Measure"),
                    "Measure a qubit, returning the qubit and the measurement result.".into(),
                    Signature::new(vec![qb_t()], vec![qb_t(), bool_t()]),
                    extension_ref,
                )
                .unwrap();

            extension
                .add_op(
                    OpName::new_inline("QAlloc"),
                    "Allocate a new qubit.".into(),
                    Signature::new(type_row![], vec![qb_t()]),
                    extension_ref,
                )
                .unwrap();

            extension
                .add_op(
                    OpName::new_inline("QDiscard"),
                    "Discard a qubit.".into(),
                    Signature::new(vec![qb_t()], type_row![]),
                    extension_ref,
                )
                .unwrap();
        })
    }

    /// Quantum extension definition.
    pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(extension);

    /// A registry with all necessary extensions to run tests internally, including the test quantum extension.
    pub static REG: LazyLock<ExtensionRegistry> = LazyLock::new(|| {
        ExtensionRegistry::new([
            EXTENSION.clone(),
            PRELUDE.clone(),
            float_types::EXTENSION.clone(),
            float_ops::EXTENSION.clone(),
            logic::EXTENSION.clone(),
        ])
    });

    fn get_gate(gate_name: &OpNameRef) -> ExtensionOp {
        EXTENSION.instantiate_extension_op(gate_name, []).unwrap()
    }
    pub(crate) fn h_gate() -> ExtensionOp {
        get_gate("H")
    }

    pub(crate) fn cx_gate() -> ExtensionOp {
        get_gate("CX")
    }

    pub(crate) fn measure() -> ExtensionOp {
        get_gate("Measure")
    }

    pub(crate) fn rz_f64() -> ExtensionOp {
        get_gate("RzF64")
    }

    pub(crate) fn q_alloc() -> ExtensionOp {
        get_gate("QAlloc")
    }

    pub(crate) fn q_discard() -> ExtensionOp {
        get_gate("QDiscard")
    }
}

/// Sort folding inputs with [`IncomingPort`] as key
fn sort_by_in_port(consts: &[(IncomingPort, Value)]) -> Vec<&(IncomingPort, Value)> {
    let mut v: Vec<_> = consts.iter().collect();
    v.sort_by_key(|(i, _)| i);
    v
}

/// Sort some input constants by port and just return the constants.
#[must_use]
pub fn sorted_consts(consts: &[(IncomingPort, Value)]) -> Vec<&Value> {
    sort_by_in_port(consts)
        .into_iter()
        .map(|(_, c)| c)
        .collect()
}

/// Calculate the depth of a node in the hierarchy.
pub fn depth(h: &Hugr, n: Node) -> u32 {
    match h.get_parent(n) {
        Some(p) => 1 + depth(h, p),
        None => 0,
    }
}

#[allow(dead_code)]
// Test only utils
#[cfg(test)]
pub(crate) mod test {
    #[allow(unused_imports)]
    use crate::HugrView;
    use crate::{
        Hugr,
        ops::{OpType, Value},
    };

    /// Check that a hugr just loads and returns a single expected constant.
    pub(crate) fn assert_fully_folded(h: &Hugr, expected_value: &Value) {
        assert_fully_folded_with(h, |v| v == expected_value);
    }

    /// Check that a hugr just loads and returns a single constant, and validate
    /// that constant using `check_value`.
    ///
    /// [`CustomConst::equals_const`] is not required to be implemented. Use this
    /// function for Values containing such a `CustomConst`.
    pub(crate) fn assert_fully_folded_with(h: &Hugr, check_value: impl Fn(&Value) -> bool) {
        let mut node_count = 0;

        for node in h.children(h.entrypoint()) {
            let op = h.get_optype(node);
            match op {
                OpType::Input(_) | OpType::Output(_) | OpType::LoadConstant(_) => node_count += 1,
                OpType::Const(c) if check_value(c.value()) => node_count += 1,
                _ => panic!("unexpected op: {}\n{}", op, h.mermaid_string()),
            }
        }

        assert_eq!(node_count, 4);
    }
}
