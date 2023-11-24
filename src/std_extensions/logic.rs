//! Basic logical operations.

use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::{
    extension::{
        prelude::BOOL_T, simple_op::OpEnum, ExtensionId, SignatureError, SignatureFromArgs,
        SignatureFunc,
    },
    ops, type_row,
    types::{
        type_param::{TypeArg, TypeParam},
        FunctionType,
    },
    Extension,
};
use lazy_static::lazy_static;
use std::str::FromStr;
use thiserror::Error;
/// Name of extension false value.
pub const FALSE_NAME: &str = "FALSE";
/// Name of extension true value.
pub const TRUE_NAME: &str = "TRUE";

/// Logic extension operations.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs)]
pub enum LogicOp {
    And,
    Or,
    Not,
}

/// Error in trying to load logic operation.
#[derive(Debug, Error)]
#[error("Not a logic extension operation.")]
pub struct NotLogicOp;

impl OpEnum for LogicOp {
    const EXTENSION_ID: ExtensionId = EXTENSION_ID;
    type LoadError = NotLogicOp;
    type Description = &'static str;

    fn from_extension_name(op_name: &str) -> Result<Self, NotLogicOp> {
        Self::from_str(op_name).map_err(|_| NotLogicOp)
    }

    fn signature(&self) -> SignatureFunc {
        match self {
            LogicOp::Or | LogicOp::And => logic_op_sig().into(),
            LogicOp::Not => FunctionType::new_endo(type_row![BOOL_T]).into(),
        }
    }

    fn description(&self) -> &'static str {
        match self {
            LogicOp::And => "logical 'and'",
            LogicOp::Or => "logical 'or'",
            LogicOp::Not => "logical 'not'",
        }
    }
}
/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("logic");

fn logic_op_sig() -> impl SignatureFromArgs {
    struct LogicOpCustom;

    const MAX: &[TypeParam; 1] = &[TypeParam::max_nat()];
    impl SignatureFromArgs for LogicOpCustom {
        fn compute_signature(
            &self,
            arg_values: &[TypeArg],
        ) -> Result<crate::types::PolyFuncType, SignatureError> {
            // get the number of input booleans.
            let [TypeArg::BoundedNat { n }] = *arg_values else {
                return Err(SignatureError::InvalidTypeArgs);
            };
            let var_arg_row = vec![BOOL_T; n as usize];
            Ok(FunctionType::new(var_arg_row, vec![BOOL_T]).into())
        }

        fn static_params(&self) -> &[TypeParam] {
            MAX
        }
    }
    LogicOpCustom
}
/// Extension for basic logical operations.
fn extension() -> Extension {
    let mut extension = Extension::new(EXTENSION_ID);
    LogicOp::load_all_ops(&mut extension).unwrap();

    extension
        .add_value(FALSE_NAME, ops::Const::unit_sum(0, 2))
        .unwrap();
    extension
        .add_value(TRUE_NAME, ops::Const::unit_sum(1, 2))
        .unwrap();
    extension
}

lazy_static! {
    /// Reference to the logic Extension.
    pub static ref EXTENSION: Extension = extension();
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        extension::{prelude::BOOL_T, simple_op::OpEnum, EMPTY_REG},
        ops::OpType,
        types::type_param::TypeArg,
        Extension,
    };

    use super::{extension, LogicOp, EXTENSION, FALSE_NAME, TRUE_NAME};

    #[test]
    fn test_logic_extension() {
        let r: Extension = extension();
        assert_eq!(r.name() as &str, "logic");
        assert_eq!(r.operations().count(), 3);

        for op in LogicOp::all_variants() {
            assert_eq!(
                LogicOp::try_from_op_def(r.get_op(op.name()).unwrap()).unwrap(),
                op
            );
        }
    }

    #[test]
    fn test_values() {
        let r: Extension = extension();
        let false_val = r.get_value(FALSE_NAME).unwrap();
        let true_val = r.get_value(TRUE_NAME).unwrap();

        for v in [false_val, true_val] {
            let simpl = v.typed_value().const_type();
            assert_eq!(simpl, &BOOL_T);
        }
    }

    /// Generate a logic extension and "and" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn and_op() -> OpType {
        LogicOp::And
            .to_optype(&EXTENSION, &[TypeArg::BoundedNat { n: 2 }], &EMPTY_REG)
            .unwrap()
    }

    /// Generate a logic extension and "or" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn or_op() -> OpType {
        LogicOp::Or
            .to_optype(&EXTENSION, &[TypeArg::BoundedNat { n: 2 }], &EMPTY_REG)
            .unwrap()
    }

    /// Generate a logic extension and "not" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn not_op() -> OpType {
        LogicOp::Not.to_optype(&EXTENSION, &[], &EMPTY_REG).unwrap()
    }
}
