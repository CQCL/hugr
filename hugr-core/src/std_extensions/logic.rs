//! Basic logical operations.

use std::sync::{Arc, Weak};

use strum::{EnumIter, EnumString, IntoStaticStr};

use crate::Extension;
use crate::extension::{
    ConstFolder, ExtensionId, FoldVal, OpDef, SignatureFunc,
    prelude::bool_t,
    simple_op::{MakeOpDef, MakeRegisteredOp, OpLoadError, try_from_name},
};
use crate::ops::{OpName, constant::ValueName};
use crate::types::{Signature, type_param::TypeArg};

use lazy_static::lazy_static;
/// Name of extension false value.
pub const FALSE_NAME: ValueName = ValueName::new_inline("FALSE");
/// Name of extension true value.
pub const TRUE_NAME: ValueName = ValueName::new_inline("TRUE");

impl ConstFolder for LogicOp {
    fn fold(&self, _type_args: &[TypeArg], inputs: &[FoldVal], outputs: &mut [FoldVal]) {
        let inps = read_known_inputs(inputs);
        let out = match self {
            Self::And => {
                let res = inps.iter().all(|x| *x);
                // We can only fold to true if we have a const for all our inputs.
                (!res || inps.len() as u64 == 2).then_some(res)
            }
            Self::Or => {
                let res = inps.iter().any(|x| *x);
                // We can only fold to false if we have a const for all our inputs
                let r = (res || inps.len() == 2).then_some(res);
                r
            }
            Self::Eq => {
                debug_assert_eq!(inputs.len(), 2);
                (inps.len() == 2).then(|| inps[0] == inps[1])
            }
            Self::Not => {
                debug_assert_eq!(inputs.len(), 1);
                inps.first().map(|b| !*b)
            }
            Self::Xor => {
                debug_assert_eq!(inputs.len(), 2);
                (inps.len() == 2).then(|| inps[0] ^ inps[1])
            }
        };
        debug_assert_eq!(outputs, &[FoldVal::Unknown]);
        if let Some(res) = out {
            outputs[0] = FoldVal::from_bool(res);
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
    Xor,
}

impl MakeOpDef for LogicOp {
    fn opdef_id(&self) -> OpName {
        <&'static str>::from(self).into()
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        match self {
            LogicOp::And | LogicOp::Or | LogicOp::Eq | LogicOp::Xor => {
                Signature::new(vec![bool_t(); 2], bool_t())
            }
            LogicOp::Not => Signature::new_endo(bool_t()),
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
            LogicOp::Xor => "logical 'xor'",
        }
        .to_string()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.clone()
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
    })
}

lazy_static! {
    /// Reference to the logic Extension.
    pub static ref EXTENSION: Arc<Extension> = extension();
}

impl MakeRegisteredOp for LogicOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

fn read_known_inputs(consts: &[FoldVal]) -> Vec<bool> {
    let true_val = FoldVal::true_val();
    let false_val = FoldVal::false_val();
    let mut res = Vec::new();
    for c in consts {
        if c == &true_val {
            res.push(true)
        } else if c == &false_val {
            res.push(false)
        } else if c != &FoldVal::Unknown {
            // Preserving legacy behaviour, but if any input is not true/false,
            // bail completely.
            return vec![];
        }
    }
    res
}

#[cfg(test)]
pub(crate) mod test {
    use std::sync::Arc;

    use super::{LogicOp, extension};
    use crate::{
        Extension,
        extension::simple_op::{MakeOpDef, MakeRegisteredOp},
        extension::{ConstFolder, FoldVal},
    };

    use itertools::Itertools;
    use rstest::rstest;
    use strum::IntoEnumIterator;

    #[test]
    fn test_logic_extension() {
        let r: Arc<Extension> = extension();
        assert_eq!(r.name() as &str, "logic");
        assert_eq!(r.operations().count(), 5);

        for op in LogicOp::iter() {
            assert_eq!(
                LogicOp::from_def(r.get_op(op.into()).unwrap(),).unwrap(),
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
    #[case(LogicOp::Xor, [true, false], true)]
    #[case(LogicOp::Xor, [true, true], false)]
    fn const_fold(
        #[case] op: LogicOp,
        #[case] ins: impl IntoIterator<Item = bool>,
        #[case] out: bool,
    ) {
        use itertools::Itertools;

        let in_vals = ins.into_iter().map(FoldVal::from_bool).collect_vec();
        let type_args = [(in_vals.len() as u64).into()];
        let mut outs = [FoldVal::Unknown];
        op.fold(&type_args, &in_vals, &mut outs);
        assert_eq!(outs, [FoldVal::from_bool(out)]);
    }

    #[rstest]
    #[case(LogicOp::And, [Some(true), None], None)]
    #[case(LogicOp::And, [Some(false), None], Some(false))]
    #[case(LogicOp::Or, [None, Some(false)], None)]
    #[case(LogicOp::Or, [None, Some(true)], Some(true))]
    #[case(LogicOp::Eq, [None, Some(true)], None)]
    #[case(LogicOp::Not, [None], None)]
    #[case(LogicOp::Xor, [None, Some(true)], None)]
    fn partial_const_fold(
        #[case] op: LogicOp,
        #[case] ins: impl IntoIterator<Item = Option<bool>>,
        #[case] mb_out: Option<bool>,
    ) {
        let in_vals = ins
            .into_iter()
            .map(|mb_b| mb_b.map_or(FoldVal::Unknown, FoldVal::from_bool))
            .collect_vec();
        let type_args = [(in_vals.len() as u64).into()];
        let mut outs = [FoldVal::Unknown];
        op.fold(&type_args, &in_vals, &mut outs);
        assert_eq!(outs, [mb_out.map_or(FoldVal::Unknown, FoldVal::from_bool)]);
    }
}
