//! Basic HUGR quantum operations

use smol_str::SmolStr;

use crate::extension::prelude::{BOOL_T, QB_T};
use crate::extension::{ExtensionSet, SignatureError};
use crate::type_row;
use crate::types::type_param::TypeArg;
use crate::types::FunctionType;
use crate::Extension;

use lazy_static::lazy_static;

/// The extension identifier.
pub const EXTENSION_ID: SmolStr = SmolStr::new_inline("quantum");
fn one_qb_func(_: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    Ok(FunctionType {
        input: type_row![QB_T],
        output: type_row![QB_T],
        extension_reqs: ExtensionSet::new(),
    })
}

fn two_qb_func(_: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    Ok(FunctionType {
        input: type_row![QB_T, QB_T],
        output: type_row![QB_T, QB_T],
        extension_reqs: ExtensionSet::new(),
    })
}

fn extension() -> Extension {
    let mut extension = Extension::new(EXTENSION_ID);

    extension
        .add_op_custom_sig_simple(
            SmolStr::new_inline("H"),
            "Hadamard".into(),
            vec![],
            one_qb_func,
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(SmolStr::new_inline("CX"), "CX".into(), vec![], two_qb_func)
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            SmolStr::new_inline("Measure"),
            "Measure a qubit, returning the qubit and the measurement result.".into(),
            vec![],
            |_arg_values: &[TypeArg]| {
                Ok(FunctionType {
                    input: type_row![QB_T],
                    output: type_row![QB_T, BOOL_T],
                    // TODO add logic as an extension delta when inference is
                    // done?
                    // https://github.com/CQCL-DEV/hugr/issues/425
                    extension_reqs: ExtensionSet::new(),
                })
            },
        )
        .unwrap();

    extension
}

lazy_static! {
    /// Quantum extension definition.
    pub static ref EXTENSION: Extension = extension();
}

#[cfg(test)]
pub(crate) mod test {
    use crate::ops::LeafOp;

    use super::EXTENSION;

    fn get_gate(gate_name: &str) -> LeafOp {
        EXTENSION
            .instantiate_extension_op(gate_name, [])
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
