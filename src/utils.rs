use std::fmt::{self, Debug, Display};

use itertools::Itertools;

/// Write a comma separated list of of some types.
/// Like debug_list, but using the Display instance rather than Debug,
/// and not adding surrounding square brackets.
pub fn display_list<T>(ts: &[T], f: &mut fmt::Formatter) -> fmt::Result
where
    T: Display,
{
    let mut first = true;
    for t in ts.iter() {
        if !first {
            f.write_str(", ")?;
        }
        t.fmt(f)?;
        if first {
            first = false;
        }
    }
    Ok(())
}

/// Collect a vector into an array.
pub fn collect_array<const N: usize, T: Debug>(arr: &[T]) -> [&T; N] {
    arr.iter().collect_vec().try_into().unwrap()
}

#[cfg(test)]
pub(crate) mod test_quantum_extension {
    use smol_str::SmolStr;

    use crate::{
        extension::{
            prelude::{BOOL_T, QB_T},
            ExtensionId, ExtensionRegistry, PRELUDE,
        },
        ops::LeafOp,
        std_extensions::arithmetic::float_types,
        type_row,
        types::{FunctionType, PolyFuncType},
        Extension,
    };

    use lazy_static::lazy_static;

    fn one_qb_func() -> PolyFuncType {
        FunctionType::new_endo(type_row![QB_T]).into()
    }

    fn two_qb_func() -> PolyFuncType {
        FunctionType::new_endo(type_row![QB_T, QB_T]).into()
    }
    /// The extension identifier.
    pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("test.quantum");
    fn extension() -> Extension {
        let mut extension = Extension::new(EXTENSION_ID);

        extension
            .add_op(SmolStr::new_inline("H"), "Hadamard".into(), one_qb_func())
            .unwrap();
        extension
            .add_op(
                SmolStr::new_inline("RzF64"),
                "Rotation specified by float".into(),
                FunctionType::new(type_row![QB_T, float_types::FLOAT64_TYPE], type_row![QB_T]),
            )
            .unwrap();

        extension
            .add_op(SmolStr::new_inline("CX"), "CX".into(), two_qb_func())
            .unwrap();

        extension
            .add_op(
                SmolStr::new_inline("Measure"),
                "Measure a qubit, returning the qubit and the measurement result.".into(),
                FunctionType::new(type_row![QB_T], type_row![QB_T, BOOL_T]),
            )
            .unwrap();

        extension
    }

    lazy_static! {
        /// Quantum extension definition.
        pub static ref EXTENSION: Extension = extension();
        static ref REG: ExtensionRegistry = ExtensionRegistry::try_new([EXTENSION.to_owned(), PRELUDE.to_owned(), float_types::extension()]).unwrap();

    }
    fn get_gate(gate_name: &str) -> LeafOp {
        EXTENSION
            .instantiate_extension_op(gate_name, [], &REG)
            .unwrap()
            .into()
    }
    pub(crate) fn h_gate() -> LeafOp {
        get_gate("H")
    }

    pub(crate) fn cx_gate() -> LeafOp {
        get_gate("CX")
    }

    pub(crate) fn measure() -> LeafOp {
        get_gate("Measure")
    }
}

#[allow(dead_code)]
// Test only utils
#[cfg(test)]
pub(crate) mod test {
    #[allow(unused_imports)]
    use crate::HugrView;

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
}
