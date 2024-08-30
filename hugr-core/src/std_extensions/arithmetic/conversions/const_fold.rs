use crate::ops::constant::get_single_input_value;
use crate::ops::Value;
use crate::std_extensions::arithmetic::int_types::INT_TYPES;
use crate::{
    extension::{
        prelude::{const_ok, ConstError, ERROR_TYPE},
        ConstFold, ConstFoldResult, OpDef,
    },
    ops,
    std_extensions::arithmetic::{
        float_types::ConstF64,
        int_types::{get_log_width, ConstInt},
    },
    types::ConstTypeError,
    IncomingPort,
};

use super::ConvertOpDef;

pub(super) fn set_fold(op: &ConvertOpDef, def: &mut OpDef) {
    use ConvertOpDef::*;

    match op {
        trunc_u => def.set_constant_folder(TruncU),
        trunc_s => def.set_constant_folder(TruncS),
        convert_u => def.set_constant_folder(ConvertU),
        convert_s => def.set_constant_folder(ConvertS),
    }
}

fn fold_trunc(
    type_args: &[crate::types::TypeArg],
    consts: &[(IncomingPort, Value)],
    convert: impl Fn(f64, u8) -> Result<Value, ConstTypeError>,
) -> ConstFoldResult {
    let f: &ConstF64 = get_single_input_value(consts)?;
    let f = f.value();
    let [arg] = type_args else {
        return None;
    };
    let log_width = get_log_width(arg).ok()?;
    let int_type = INT_TYPES[log_width as usize].to_owned();
    let err_value = || {
        ConstError {
            signal: 0,
            message: "Can't truncate non-finite float".to_string(),
        }
        .as_either(int_type.clone())
    };
    let out_const: ops::Value = if !f.is_finite() {
        err_value()
    } else {
        let cv = convert(f, log_width);
        if let Ok(cv) = cv {
            const_ok(cv, ERROR_TYPE)
        } else {
            err_value()
        }
    };

    Some(vec![(0.into(), out_const)])
}

struct TruncU;

impl ConstFold for TruncU {
    fn fold(
        &self,
        type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        fold_trunc(type_args, consts, |f, log_width| {
            ConstInt::new_u(log_width, f.trunc() as u64).map(Into::into)
        })
    }
}

struct TruncS;

impl ConstFold for TruncS {
    fn fold(
        &self,
        type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        fold_trunc(type_args, consts, |f, log_width| {
            ConstInt::new_s(log_width, f.trunc() as i64).map(Into::into)
        })
    }
}

struct ConvertU;

impl ConstFold for ConvertU {
    fn fold(
        &self,
        _type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        let u: &ConstInt = crate::ops::constant::get_single_input_value(consts)?;
        let f = u.value_u() as f64;
        Some(vec![(0.into(), ConstF64::new(f).into())])
    }
}

struct ConvertS;

impl ConstFold for ConvertS {
    fn fold(
        &self,
        _type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        let u: &ConstInt = get_single_input_value(consts)?;
        let f = u.value_s() as f64;
        Some(vec![(0.into(), ConstF64::new(f).into())])
    }
}
