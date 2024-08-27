//! Folding definitions for list operations.

use crate::extension::{ConstFold, OpDef};
use crate::ops;
use crate::types::type_param::TypeArg;
use crate::utils::sorted_consts;

use super::{ListOp, ListValue};

pub(super) fn set_fold(op: &ListOp, def: &mut OpDef) {
    match op {
        ListOp::pop => def.set_constant_folder(PopFold),
        ListOp::push => def.set_constant_folder(PushFold),
    }
}

pub struct PopFold;

impl ConstFold for PopFold {
    fn fold(
        &self,
        _type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, ops::Value)],
    ) -> crate::extension::ConstFoldResult {
        let [list]: [&ops::Value; 1] = sorted_consts(consts).try_into().ok()?;
        let list: &ListValue = list.get_custom_value().expect("Should be list value.");
        let mut list = list.clone();
        let elem = list.0.pop()?; // empty list fails to evaluate "pop"

        Some(vec![(0.into(), list.into()), (1.into(), elem)])
    }
}

pub struct PushFold;

impl ConstFold for PushFold {
    fn fold(
        &self,
        _type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, ops::Value)],
    ) -> crate::extension::ConstFoldResult {
        let [list, elem]: [&ops::Value; 2] = sorted_consts(consts).try_into().ok()?;
        let list: &ListValue = list.get_custom_value().expect("Should be list value.");
        let mut list = list.clone();
        list.0.push(elem.clone());

        Some(vec![(0.into(), list.into())])
    }
}
