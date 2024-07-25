//! General utilities.

use std::fmt::{self, Debug, Display};

use itertools::Itertools;

use crate::{ops::Value, Hugr, HugrView, IncomingPort, Node};

/// Write a comma separated list of of some types.
/// Like debug_list, but using the Display instance rather than Debug,
/// and not adding surrounding square brackets.
pub fn display_list<T>(ts: impl IntoIterator<Item = T>, f: &mut fmt::Formatter) -> fmt::Result
where
    T: Display,
{
    display_list_with_separator(ts, f, ", ")
}

/// Write a separated list of of some types, using a custom separator.
/// Like debug_list, but using the Display instance rather than Debug,
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
    for t in ts.into_iter() {
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
pub fn collect_array<const N: usize, T: Debug>(arr: impl IntoIterator<Item = T>) -> [T; N] {
    try_collect_array(arr).unwrap_or_else(|v| panic!("Expected {} elements, got {:?}", N, v))
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
/// From https://github.com/serde-rs/serde/issues/818.
#[allow(dead_code)]
pub(crate) fn is_default<T: Default + PartialEq>(t: &T) -> bool {
    *t == Default::default()
}

#[cfg(test)]
pub(crate) mod test_quantum_extension {
    use crate::ops::{OpName, OpNameRef};
    use crate::types::FuncValueType;
    use crate::{
        extension::{
            prelude::{BOOL_T, QB_T},
            ExtensionId, ExtensionRegistry, PRELUDE,
        },
        ops::CustomOp,
        std_extensions::arithmetic::float_types,
        type_row,
        types::{PolyFuncTypeRV, Signature},
        Extension,
    };

    use lazy_static::lazy_static;

    fn one_qb_func() -> PolyFuncTypeRV {
        FuncValueType::new_endo(QB_T).into()
    }

    fn two_qb_func() -> PolyFuncTypeRV {
        FuncValueType::new_endo(type_row![QB_T, QB_T]).into()
    }
    /// The extension identifier.
    pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("test.quantum");
    fn extension() -> Extension {
        let mut extension = Extension::new(EXTENSION_ID, semver::Version::parse("0.1.0").unwrap());

        extension
            .add_op(OpName::new_inline("H"), "Hadamard".into(), one_qb_func())
            .unwrap();
        extension
            .add_op(
                OpName::new_inline("RzF64"),
                "Rotation specified by float".into(),
                Signature::new(type_row![QB_T, float_types::FLOAT64_TYPE], type_row![QB_T]),
            )
            .unwrap();

        extension
            .add_op(OpName::new_inline("CX"), "CX".into(), two_qb_func())
            .unwrap();

        extension
            .add_op(
                OpName::new_inline("Measure"),
                "Measure a qubit, returning the qubit and the measurement result.".into(),
                Signature::new(type_row![QB_T], type_row![QB_T, BOOL_T]),
            )
            .unwrap();

        extension
            .add_op(
                OpName::new_inline("QAlloc"),
                "Allocate a new qubit.".into(),
                Signature::new(type_row![], type_row![QB_T]),
            )
            .unwrap();

        extension
            .add_op(
                OpName::new_inline("QDiscard"),
                "Discard a qubit.".into(),
                Signature::new(type_row![QB_T], type_row![]),
            )
            .unwrap();

        extension
    }

    lazy_static! {
        /// Quantum extension definition.
        pub static ref EXTENSION: Extension = extension();
        static ref REG: ExtensionRegistry = ExtensionRegistry::try_new([EXTENSION.to_owned(), PRELUDE.to_owned(), float_types::EXTENSION.to_owned()]).unwrap();

    }

    fn get_gate(gate_name: &OpNameRef) -> CustomOp {
        EXTENSION
            .instantiate_extension_op(gate_name, [], &REG)
            .unwrap()
            .into()
    }
    pub(crate) fn h_gate() -> CustomOp {
        get_gate("H")
    }

    pub(crate) fn cx_gate() -> CustomOp {
        get_gate("CX")
    }

    pub(crate) fn measure() -> CustomOp {
        get_gate("Measure")
    }

    pub(crate) fn rz_f64() -> CustomOp {
        get_gate("RzF64")
    }

    pub(crate) fn q_alloc() -> CustomOp {
        get_gate("QAlloc")
    }

    pub(crate) fn q_discard() -> CustomOp {
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
        ops::{OpType, Value},
        Hugr,
    };

    /// Open a browser page to render a dot string graph.
    ///
    /// This can be used directly on the output of `Hugr::dot_string`
    #[cfg(not(ci_run))]
    pub(crate) fn viz_dotstr(dotstr: impl AsRef<str>) {
        let mut base: String = "https://dreampuf.github.io/GraphvizOnline/#".into();
        base.push_str(&urlencoding::encode(dotstr.as_ref()));
        webbrowser::open(&base).unwrap();
    }

    /// Open a browser page to render a HugrView's dot string graph.
    #[cfg(not(ci_run))]
    pub(crate) fn viz_hugr(hugr: &impl HugrView) {
        viz_dotstr(hugr.dot_string());
    }

    /// Check that a hugr just loads and returns a single expected constant.
    pub(crate) fn assert_fully_folded(h: &Hugr, expected_value: &Value) {
        assert_fully_folded_with(h, |v| v == expected_value)
    }

    /// Check that a hugr just loads and returns a single constant, and validate
    /// that constant using `check_value`.
    ///
    /// [CustomConst::equals_const] is not required to be implemented. Use this
    /// function for Values containing such a `CustomConst`.
    pub(crate) fn assert_fully_folded_with(h: &Hugr, check_value: impl Fn(&Value) -> bool) {
        let mut node_count = 0;

        for node in h.children(h.root()) {
            let op = h.get_optype(node);
            match op {
                OpType::Input(_) | OpType::Output(_) | OpType::LoadConstant(_) => node_count += 1,
                OpType::Const(c) if check_value(c.value()) => node_count += 1,
                _ => panic!("unexpected op: {:?}\n{}", op, h.mermaid_string()),
            }
        }

        assert_eq!(node_count, 4);
    }
}
