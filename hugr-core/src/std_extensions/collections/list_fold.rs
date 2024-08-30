//! Folding definitions for list operations.

use crate::extension::prelude::{
    const_err, const_none, const_ok, const_ok_tuple, const_some, ConstUsize,
};
use crate::extension::{ConstFold, ConstFoldResult, OpDef};
use crate::ops::Value;
use crate::types::type_param::TypeArg;
use crate::types::Type;
use crate::utils::sorted_consts;
use crate::IncomingPort;

use super::{ListOp, ListValue};

pub(super) fn set_fold(op: &ListOp, def: &mut OpDef) {
    match op {
        ListOp::pop => def.set_constant_folder(PopFold),
        ListOp::push => def.set_constant_folder(PushFold),
        ListOp::get => def.set_constant_folder(GetFold),
        ListOp::set => def.set_constant_folder(SetFold),
        ListOp::insert => def.set_constant_folder(InsertFold),
        ListOp::length => def.set_constant_folder(LengthFold),
    }
}

pub struct PopFold;

impl ConstFold for PopFold {
    fn fold(
        &self,
        _type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, Value)],
    ) -> ConstFoldResult {
        let [list]: [&Value; 1] = sorted_consts(consts).try_into().ok()?;
        let list: &ListValue = list.get_custom_value().expect("Should be list value.");
        let mut list = list.clone();

        match list.0.pop() {
            Some(elem) => Some(vec![(0.into(), list.into()), (1.into(), const_some(elem))]),
            None => {
                let elem_type = list.1.clone();
                Some(vec![
                    (0.into(), list.into()),
                    (1.into(), const_none(elem_type)),
                ])
            }
        }
    }
}

pub struct PushFold;

impl ConstFold for PushFold {
    fn fold(
        &self,
        _type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, Value)],
    ) -> ConstFoldResult {
        let [list, elem]: [&Value; 2] = sorted_consts(consts).try_into().ok()?;
        let list: &ListValue = list.get_custom_value().expect("Should be list value.");
        let mut list = list.clone();
        list.0.push(elem.clone());

        Some(vec![(0.into(), list.into())])
    }
}

pub struct GetFold;

impl ConstFold for GetFold {
    fn fold(&self, _type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        let [list, index]: [&Value; 2] = sorted_consts(consts).try_into().ok()?;
        let list: &ListValue = list.get_custom_value().expect("Should be list value.");
        let index: &ConstUsize = index.get_custom_value().expect("Should be int value.");
        let idx = index.value() as usize;

        match list.0.get(idx) {
            Some(elem) => Some(vec![(0.into(), const_some(elem.clone()))]),
            None => Some(vec![(0.into(), const_none(list.1.clone()))]),
        }
    }
}

pub struct SetFold;

impl ConstFold for SetFold {
    fn fold(&self, _type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        let [list, idx, elem]: [&Value; 3] = sorted_consts(consts).try_into().ok()?;
        let list: &ListValue = list.get_custom_value().expect("Should be list value.");

        let idx: &ConstUsize = idx.get_custom_value().expect("Should be int value.");
        let idx = idx.value() as usize;

        let mut list = list.clone();
        let mut elem = elem.clone();
        let res_elem: Value = match list.0.get_mut(idx) {
            Some(old_elem) => {
                std::mem::swap(old_elem, &mut elem);
                const_ok(elem, list.1.clone())
            }
            None => const_err(elem, list.1.clone()),
        };
        Some(vec![(0.into(), list.into()), (1.into(), res_elem)])
    }
}

pub struct InsertFold;

impl ConstFold for InsertFold {
    fn fold(&self, _type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        let [list, idx, elem]: [&Value; 3] = sorted_consts(consts).try_into().ok()?;
        let list: &ListValue = list.get_custom_value().expect("Should be list value.");

        let idx: &ConstUsize = idx.get_custom_value().expect("Should be int value.");
        let idx = idx.value() as usize;

        let mut list = list.clone();
        let elem = elem.clone();
        let res_elem: Value = if list.0.len() > idx {
            list.0.insert(idx, elem);
            const_ok_tuple([], list.1.clone())
        } else {
            const_err(elem, Type::UNIT)
        };
        Some(vec![(0.into(), list.into()), (1.into(), res_elem)])
    }
}

pub struct LengthFold;

impl ConstFold for LengthFold {
    fn fold(&self, _type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        let [list]: [&Value; 1] = sorted_consts(consts).try_into().ok()?;
        let list: &ListValue = list.get_custom_value().expect("Should be list value.");
        let len = list.0.len();

        Some(vec![(0.into(), ConstUsize::new(len as u64).into())])
    }
}
