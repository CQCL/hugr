use crate::{
    extension::{
        prelude::{sum_with_error, ConstError},
        ConstFold, ConstFoldResult, OpDef,
    },
    ops::{constant::get_single_input_value, Value},
    std_extensions::arithmetic::int_types::{get_log_width, ConstInt, INT_TYPES},
    types::TypeArg,
    IncomingPort,
};

use super::IntOpDef;

struct IWidenUFolder;
impl ConstFold for IWidenUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        let [arg0, arg1] = type_args else {
            return None;
        };
        let logwidth0: u8 = get_log_width(arg0).ok()?;
        let logwidth1: u8 = get_log_width(arg1).ok()?;
        let n0: &ConstInt = get_single_input_value(consts)?;
        if logwidth0 > logwidth1 || n0.log_width() != logwidth0 {
            None
        } else {
            let n1 = ConstInt::new_u(logwidth1, n0.value_u()).ok()?;
            Some(vec![(0.into(), n1.into())])
        }
    }
}

struct IWidenSFolder;
impl ConstFold for IWidenSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        let [arg0, arg1] = type_args else {
            return None;
        };
        let logwidth0: u8 = get_log_width(arg0).ok()?;
        let logwidth1: u8 = get_log_width(arg1).ok()?;
        let n0: &ConstInt = get_single_input_value(consts)?;
        if logwidth0 > logwidth1 || n0.log_width() != logwidth0 {
            None
        } else {
            let n1 = ConstInt::new_s(logwidth1, n0.value_s()).ok()?;
            Some(vec![(0.into(), n1.into())])
        }
    }
}

struct INarrowUFolder;
impl ConstFold for INarrowUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        let [arg0, arg1] = type_args else {
            return None;
        };
        let logwidth0: u8 = get_log_width(arg0).ok()?;
        let logwidth1: u8 = get_log_width(arg1).ok()?;
        let n0: &ConstInt = get_single_input_value(consts)?;

        let int_out_type = INT_TYPES[logwidth1 as usize].to_owned();
        let sum_type = sum_with_error(int_out_type.clone());
        let err_value = || {
            let err_val = ConstError {
                signal: 0,
                message: "Integer too large to narrow".to_string(),
            };
            Value::sum(1, [err_val.into()], sum_type.clone())
                .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
        };
        let n0val: u64 = n0.value_u();
        let out_const: Value = if n0val >> (1 << logwidth1) != 0 {
            err_value()
        } else {
            Value::extension(ConstInt::new_u(logwidth1, n0val).unwrap())
        };
        if logwidth0 < logwidth1 || n0.log_width() != logwidth0 {
            None
        } else {
            Some(vec![(0.into(), out_const)])
        }
    }
}

struct INarrowSFolder;
impl ConstFold for INarrowSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        let [arg0, arg1] = type_args else {
            return None;
        };
        let logwidth0: u8 = get_log_width(arg0).ok()?;
        let logwidth1: u8 = get_log_width(arg1).ok()?;
        let n0: &ConstInt = get_single_input_value(consts)?;

        let int_out_type = INT_TYPES[logwidth1 as usize].to_owned();
        let sum_type = sum_with_error(int_out_type.clone());
        let err_value = || {
            let err_val = ConstError {
                signal: 0,
                message: "Integer too large to narrow".to_string(),
            };
            Value::sum(1, [err_val.into()], sum_type.clone())
                .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
        };
        let n0val: i64 = n0.value_s();
        let ub = 1i64 << ((1 << logwidth1) - 1);
        let out_const: Value = if n0val >= ub || n0val < -ub {
            err_value()
        } else {
            Value::extension(ConstInt::new_s(logwidth1, n0val).unwrap())
        };
        if logwidth0 < logwidth1 || n0.log_width() != logwidth0 {
            None
        } else {
            Some(vec![(0.into(), out_const)])
        }
    }
}

pub(super) fn set_fold(op: &IntOpDef, def: &mut OpDef) {
    match op {
        IntOpDef::iwiden_u => def.set_constant_folder(IWidenUFolder),
        IntOpDef::iwiden_s => def.set_constant_folder(IWidenSFolder),
        IntOpDef::inarrow_u => def.set_constant_folder(INarrowUFolder),
        IntOpDef::inarrow_s => def.set_constant_folder(INarrowSFolder),

        // IntOpDef::itobool => todo!(),
        // IntOpDef::ifrombool => todo!(),
        // IntOpDef::ieq => todo!(),
        // IntOpDef::ine => todo!(),
        // IntOpDef::ilt_u => todo!(),
        // IntOpDef::ilt_s => todo!(),
        // IntOpDef::igt_u => todo!(),
        // IntOpDef::igt_s => todo!(),
        // IntOpDef::ile_u => todo!(),
        // IntOpDef::ile_s => todo!(),
        // IntOpDef::ige_u => todo!(),
        // IntOpDef::ige_s => todo!(),
        // IntOpDef::imax_u => todo!(),
        // IntOpDef::imax_s => todo!(),
        // IntOpDef::imin_u => todo!(),
        // IntOpDef::imin_s => todo!(),
        // IntOpDef::iadd => todo!(),
        // IntOpDef::isub => todo!(),
        // IntOpDef::ineg => todo!(),
        // IntOpDef::imul => todo!(),
        // IntOpDef::idivmod_checked_u => todo!(),
        // IntOpDef::idivmod_u => todo!(),
        // IntOpDef::idivmod_checked_s => todo!(),
        // IntOpDef::idivmod_s => todo!(),
        // IntOpDef::idiv_checked_u => todo!(),
        // IntOpDef::idiv_u => todo!(),
        // IntOpDef::imod_checked_u => todo!(),
        // IntOpDef::imod_u => todo!(),
        // IntOpDef::idiv_checked_s => todo!(),
        // IntOpDef::idiv_s => todo!(),
        // IntOpDef::imod_checked_s => todo!(),
        // IntOpDef::imod_s => todo!(),
        // IntOpDef::iabs => todo!(),
        // IntOpDef::iand => todo!(),
        // IntOpDef::ior => todo!(),
        // IntOpDef::ixor => todo!(),
        // IntOpDef::inot => todo!(),
        // IntOpDef::ishl => todo!(),
        // IntOpDef::ishr => todo!(),
        // IntOpDef::irotl => todo!(),
        // IntOpDef::irotr => todo!(),
        // IntOpDef::itostring_u => todo!(),
        // IntOpDef::itostring_s => todo!(),
        _ => {}
    }
}
