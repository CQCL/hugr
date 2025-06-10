//! Folding definitions for list operations.

use crate::extension::prelude::{
    ConstUsize, const_fail, const_none, const_ok, const_ok_tuple, const_some,
};
use crate::extension::{ConstFolder, FoldVal, OpDef};
use crate::ops::Value;
use crate::types::Type;
use crate::types::type_param::TypeArg;

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

impl ConstFolder for PopFold {
    fn fold(&self, _type_args: &[TypeArg], inputs: &[FoldVal], outputs: &mut [FoldVal]) {
        let [fv] = inputs else {
            panic!("Expected one input")
        };
        let list: &ListValue = fv.get_custom_value().expect("Should be list value.");
        let mut list = list.clone();

        let elem_type = list.1.clone();
        outputs[1] = list
            .0
            .pop()
            .map_or(const_none(elem_type), const_some)
            .into();
        outputs[0] = list.into();
    }
}

pub struct PushFold;

impl ConstFolder for PushFold {
    fn fold(&self, _type_args: &[TypeArg], inputs: &[FoldVal], outputs: &mut [FoldVal]) {
        let [list, elem] = inputs else {
            panic!("Expected two inputs")
        };
        if let Some(list) = list.get_custom_value::<ListValue>() {
            // We have to convert `elem` to a Value to store it in the list (TODO)
            // So e.g. a LoadedFunction would mean we can't constant-fold.
            if let Ok(elem) = elem.clone().try_into() {
                let mut list = list.clone();
                list.0.push(elem);
                outputs[0] = list.into();
            }
        }
    }
}

pub struct GetFold;

impl ConstFolder for GetFold {
    fn fold(&self, _type_args: &[TypeArg], inputs: &[FoldVal], outputs: &mut [FoldVal]) {
        let [list, index] = inputs else {
            panic!("Expected two inputs")
        };
        if let Some(list) = list.get_custom_value::<ListValue>() {
            if let Some(index) = index.get_custom_value::<ConstUsize>() {
                let idx = index.value() as usize;

                outputs[0] = match list.0.get(idx) {
                    Some(elem) => const_some(elem.clone()),
                    None => const_none(list.1.clone()),
                }
                .into();
            }
        }
    }
}

pub struct SetFold;

impl ConstFolder for SetFold {
    fn fold(&self, _type_args: &[TypeArg], inputs: &[FoldVal], outputs: &mut [FoldVal]) {
        let [list, idx, elem] = inputs else {
            panic!("Expected 3 inputs")
        };
        if let Some(list) = list.get_custom_value::<ListValue>() {
            if let Some(idx) = idx.get_custom_value::<ConstUsize>() {
                if let Ok(mut elem) = Value::try_from(elem.clone()) {
                    let idx = idx.value() as usize;

                    let mut list = list.clone();
                    let res_elem: Value = match list.0.get_mut(idx) {
                        Some(old_elem) => {
                            std::mem::swap(old_elem, &mut elem);
                            const_ok(elem, list.1.clone())
                        }
                        None => const_fail(elem, list.1.clone()),
                    };
                    outputs[0] = list.into();
                    outputs[1] = res_elem.into();
                }
            }
        }
    }
}

pub struct InsertFold;

impl ConstFolder for InsertFold {
    fn fold(&self, _type_args: &[TypeArg], inputs: &[FoldVal], outputs: &mut [FoldVal]) {
        let [list, idx, elem] = inputs else {
            panic!("Expected 3 inputs")
        };
        if let Some(list) = list.get_custom_value::<ListValue>() {
            if let Some(idx) = idx.get_custom_value::<ConstUsize>() {
                if let Ok(elem) = Value::try_from(elem.clone()) {
                    let idx = idx.value() as usize;

                    let mut list = list.clone();
                    let res_elem: Value = if list.0.len() > idx {
                        list.0.insert(idx, elem);
                        const_ok_tuple([], list.1.clone())
                    } else {
                        const_fail(elem, Type::UNIT)
                    };
                    outputs[0] = list.into();
                    outputs[1] = res_elem.into();
                }
            }
        }
    }
}

pub struct LengthFold;

impl ConstFolder for LengthFold {
    fn fold(&self, _type_args: &[TypeArg], inputs: &[FoldVal], outputs: &mut [FoldVal]) {
        let [list] = inputs else {
            panic!("Expected one input")
        };
        if let Some(list) = list.get_custom_value::<ListValue>() {
            outputs[0] = ConstUsize::new(list.0.len() as u64).into();
        }
    }
}
