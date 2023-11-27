//! Basic logical operations.

use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::{
    extension::{
        prelude::BOOL_T, simple_op::OpEnum, ExtensionId, OpDef, SignatureError, SignatureFromArgs,
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
#[derive(Clone, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs)]
pub enum LogicOp {
    And(u64),
    Or(u64),
    Not,
}

/// Error in trying to load logic operation.
#[derive(Debug, Error)]
pub enum LogicOpLoadError {
    #[error("Not a logic extension operation.")]
    NotLogicOp,
    #[error("Type args invalid: {0}.")]
    InvalidArgs(#[from] SignatureError),
}

impl OpEnum for LogicOp {
    type LoadError = LogicOpLoadError;
    type Description = &'static str;

    fn from_op_def(op_def: &OpDef, args: &[TypeArg]) -> Result<Self, Self::LoadError> {
        let mut out = Self::from_str(op_def.name()).map_err(|_| LogicOpLoadError::NotLogicOp)?;
        match &mut out {
            LogicOp::And(i) | LogicOp::Or(i) => {
                let [TypeArg::BoundedNat { n }] = *args else {
                    return Err(SignatureError::InvalidTypeArgs.into());
                };
                *i = n;
            }
            LogicOp::Not => (),
        }

        Ok(out)
    }

    fn def_signature(&self) -> SignatureFunc {
        match self {
            LogicOp::Or(_) | LogicOp::And(_) => logic_op_sig().into(),
            LogicOp::Not => FunctionType::new_endo(type_row![BOOL_T]).into(),
        }
    }

    fn description(&self) -> &'static str {
        match self {
            LogicOp::And(_) => "logical 'and'",
            LogicOp::Or(_) => "logical 'or'",
            LogicOp::Not => "logical 'not'",
        }
    }
    fn type_args(&self) -> Vec<TypeArg> {
        match self {
            LogicOp::And(n) | LogicOp::Or(n) => vec![TypeArg::BoundedNat { n: *n }],
            LogicOp::Not => vec![],
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
    use super::{extension, LogicOp, EXTENSION, EXTENSION_ID, FALSE_NAME, TRUE_NAME};
    use crate::{
        extension::{
            prelude::BOOL_T,
            simple_op::{OpEnum, OpEnumName},
            ExtensionRegistry,
        },
        ops::OpType,
        types::TypeArg,
        Extension,
    };
    use lazy_static::lazy_static;
    lazy_static! {
        pub(crate) static ref LOGIC_REG: ExtensionRegistry =
            ExtensionRegistry::try_new([EXTENSION.to_owned()]).unwrap();
    }
    #[test]
    fn test_logic_extension() {
        let r: Extension = extension();
        assert_eq!(r.name() as &str, "logic");
        assert_eq!(r.operations().count(), 3);

        for op in LogicOp::all_variants() {
            assert_eq!(
                LogicOp::from_op_def(
                    r.get_op(op.name()).unwrap(),
                    // `all_variants` will set default type arg values.
                    &[TypeArg::BoundedNat { n: 0 }]
                )
                .unwrap(),
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
        LogicOp::And(2)
            .to_registered(EXTENSION_ID.to_owned(), &LOGIC_REG)
            .to_optype()
            .unwrap()
    }

    /// Generate a logic extension and "or" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn or_op() -> OpType {
        LogicOp::Or(2)
            .to_registered(EXTENSION_ID.to_owned(), &LOGIC_REG)
            .to_optype()
            .unwrap()
    }

    /// Generate a logic extension and "not" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn not_op() -> OpType {
        LogicOp::Not
            .to_registered(EXTENSION_ID.to_owned(), &LOGIC_REG)
            .to_optype()
            .unwrap()
    }
}
