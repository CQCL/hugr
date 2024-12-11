//! Basic logical operations.

use std::sync::{Arc, Weak};

use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::extension::{ConstFold, ConstFoldResult};
use crate::ops::constant::ValueName;
use crate::ops::Value;
use crate::types::Signature;
use crate::{
    extension::{
        prelude::bool_t,
        simple_op::{try_from_name, MakeOpDef, MakeRegisteredOp, OpLoadError},
        ExtensionId, ExtensionRegistry, OpDef, SignatureFunc,
    },
    ops,
    types::type_param::TypeArg,
    utils::sorted_consts,
    Extension, IncomingPort,
};
use lazy_static::lazy_static;
/// Name of extension false value.
pub const FALSE_NAME: ValueName = ValueName::new_inline("FALSE");
/// Name of extension true value.
pub const TRUE_NAME: ValueName = ValueName::new_inline("TRUE");

impl ConstFold for LogicOp {
    fn fold(&self, _type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        match self {
            Self::And => {
                let inps = read_inputs(consts)?;
                let res = inps.iter().all(|x| *x);
                // We can only fold to true if we have a const for all our inputs.
                (!res || inps.len() as u64 == 2)
                    .then_some(vec![(0.into(), ops::Value::from_bool(res))])
            }
            Self::Or => {
                let inps = read_inputs(consts)?;
                let res = inps.iter().any(|x| *x);
                // We can only fold to false if we have a const for all our inputs
                (res || inps.len() as u64 == 2)
                    .then_some(vec![(0.into(), ops::Value::from_bool(res))])
            }
            Self::Eq => {
                let inps = read_inputs(consts)?;
                let res = inps.iter().copied().reduce(|a, b| a == b)?;
                // If we have only some inputs, we can still fold to false, but not to true
                (!res || inps.len() as u64 == 2)
                    .then_some(vec![(0.into(), ops::Value::from_bool(res))])
            }
            Self::Not => {
                let inps = read_inputs(consts)?;
                let res = inps.iter().all(|x| !*x);
                (!res || inps.len() as u64 == 1)
                    .then_some(vec![(0.into(), ops::Value::from_bool(res))])
            }
        }
    }
}
/// Logic extension operation definitions.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum LogicOp {
    And,
    Or,
    Eq,
    Not,
}

impl MakeOpDef for LogicOp {
    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        match self {
            LogicOp::And | LogicOp::Or | LogicOp::Eq => {
                Signature::new(vec![bool_t(); 2], vec![bool_t()])
            }
            LogicOp::Not => Signature::new_endo(vec![bool_t()]),
        }
        .into()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn description(&self) -> String {
        match self {
            LogicOp::And => "logical 'and'",
            LogicOp::Or => "logical 'or'",
            LogicOp::Eq => "test if bools are equal",
            LogicOp::Not => "logical 'not'",
        }
        .to_string()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        def.set_constant_folder(*self);
    }
}

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("logic");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

/// Extension for basic logical operations.
fn extension() -> Arc<Extension> {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        LogicOp::load_all_ops(extension, extension_ref).unwrap();

        extension
            .add_value(FALSE_NAME, ops::Value::false_val())
            .unwrap();
        extension
            .add_value(TRUE_NAME, ops::Value::true_val())
            .unwrap();
    })
}

lazy_static! {
    /// Reference to the logic Extension.
    pub static ref EXTENSION: Arc<Extension> = extension();
    /// Registry required to validate logic extension.
    pub static ref LOGIC_REG: ExtensionRegistry = ExtensionRegistry::new([EXTENSION.clone()]);
}

impl MakeRegisteredOp for LogicOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &LOGIC_REG
    }
}

fn read_inputs(consts: &[(IncomingPort, ops::Value)]) -> Option<Vec<bool>> {
    let true_val = ops::Value::true_val();
    let false_val = ops::Value::false_val();
    let inps: Option<Vec<bool>> = sorted_consts(consts)
        .into_iter()
        .map(|c| {
            if c == &true_val {
                Some(true)
            } else if c == &false_val {
                Some(false)
            } else {
                None
            }
        })
        .collect();
    let inps = inps?;
    Some(inps)
}

#[cfg(test)]
pub(crate) mod test {
    use std::sync::Arc;

    use super::{extension, LogicOp, FALSE_NAME, TRUE_NAME};
    use crate::{
        extension::{
            prelude::bool_t,
            simple_op::{MakeOpDef, MakeRegisteredOp},
        },
        ops::{NamedOp, Value},
        Extension,
    };

    use rstest::rstest;
    use strum::IntoEnumIterator;

    #[test]
    fn test_logic_extension() {
        let r: Arc<Extension> = extension();
        assert_eq!(r.name() as &str, "logic");
        assert_eq!(r.operations().count(), 4);

        for op in LogicOp::iter() {
            assert_eq!(
                LogicOp::from_def(r.get_op(&op.name()).unwrap(),).unwrap(),
                op
            );
        }
    }

    #[test]
    fn test_conversions() {
        for o in LogicOp::iter() {
            let ext_op = o.to_extension_op().unwrap();
            assert_eq!(LogicOp::from_op(&ext_op).unwrap(), o);
        }
    }

    #[test]
    fn test_values() {
        let r: Arc<Extension> = extension();
        let false_val = r.get_value(&FALSE_NAME).unwrap();
        let true_val = r.get_value(&TRUE_NAME).unwrap();

        for v in [false_val, true_val] {
            let simpl = v.typed_value().get_type();
            assert_eq!(simpl, bool_t());
        }
    }

    /// Generate a logic extension "and" operation over [`crate::prelude::bool_t()`]
    pub(crate) fn and_op() -> LogicOp {
        LogicOp::And
    }

    /// Generate a logic extension "or" operation over [`crate::prelude::bool_t()`]
    pub(crate) fn or_op() -> LogicOp {
        LogicOp::Or
    }

    #[rstest]
    #[case(LogicOp::And, [true, true], true)]
    #[case(LogicOp::And, [true, false], false)]
    #[case(LogicOp::Or, [false, true], true)]
    #[case(LogicOp::Or, [false, false], false)]
    #[case(LogicOp::Eq, [true, false], false)]
    #[case(LogicOp::Eq, [false, false], true)]
    #[case(LogicOp::Not, [false], true)]
    #[case(LogicOp::Not, [true], false)]
    fn const_fold(
        #[case] op: LogicOp,
        #[case] ins: impl IntoIterator<Item = bool>,
        #[case] out: bool,
    ) {
        use itertools::Itertools;

        use crate::extension::ConstFold;
        let in_vals = ins
            .into_iter()
            .enumerate()
            .map(|(i, b)| (i.into(), Value::from_bool(b)))
            .collect_vec();
        assert_eq!(
            Some(vec![(0.into(), Value::from_bool(out))]),
            op.fold(&[(in_vals.len() as u64).into()], &in_vals)
        );
    }

    #[rstest]
    #[case(LogicOp::And, [Some(true), None], None)]
    #[case(LogicOp::And, [Some(false), None], Some(false))]
    #[case(LogicOp::Or, [None, Some(false)], None)]
    #[case(LogicOp::Or, [None, Some(true)], Some(true))]
    #[case(LogicOp::Eq, [None, Some(true)], None)]
    #[case(LogicOp::Not, [None], None)]
    fn partial_const_fold(
        #[case] op: LogicOp,
        #[case] ins: impl IntoIterator<Item = Option<bool>>,
        #[case] mb_out: Option<bool>,
    ) {
        use itertools::Itertools;

        use crate::extension::ConstFold;
        let in_vals0 = ins.into_iter().enumerate().collect_vec();
        let num_args = in_vals0.len() as u64;
        let in_vals = in_vals0
            .into_iter()
            .filter_map(|(i, mb_b)| mb_b.map(|b| (i.into(), Value::from_bool(b))))
            .collect_vec();
        assert_eq!(
            mb_out.map(|out| vec![(0.into(), Value::from_bool(out))]),
            op.fold(&[num_args.into()], &in_vals)
        );
    }
}
