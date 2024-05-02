use crate::ops::Value;
use crate::{
    extension::{
        prelude::{sum_with_error, ConstError},
        ConstFold, ConstFoldResult, OpDef,
    },
    ops,
    ops::constant::CustomConst,
    std_extensions::arithmetic::{
        float_types::ConstF64,
        int_types::{get_log_width, ConstInt, INT_TYPES},
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

fn get_input<T: CustomConst>(consts: &[(IncomingPort, ops::Value)]) -> Option<&T> {
    let [(_, c)] = consts else {
        return None;
    };
    c.get_custom_value()
}

fn fold_trunc(
    type_args: &[crate::types::TypeArg],
    consts: &[(IncomingPort, Value)],
    convert: impl Fn(f64, u8) -> Result<Value, ConstTypeError>,
) -> ConstFoldResult {
    let f: &ConstF64 = get_input(consts)?;
    let f = f.value();
    let [arg] = type_args else {
        return None;
    };
    let log_width = get_log_width(arg).ok()?;
    let int_type = INT_TYPES[log_width as usize].to_owned();
    let sum_type = sum_with_error(int_type.clone());
    let err_value = || {
        let err_val = ConstError {
            signal: 0,
            message: "Can't truncate non-finite float".to_string(),
        };
        Value::sum(1, [err_val.into()], sum_type.clone())
            .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
    };
    let out_const: ops::Value = if !f.is_finite() {
        err_value()
    } else {
        let cv = convert(f, log_width);
        if let Ok(cv) = cv {
            Value::sum(0, [cv], sum_type).unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
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
        let u: &ConstInt = get_input(consts)?;
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
        let u: &ConstInt = get_input(consts)?;
        let f = u.value_s() as f64;
        Some(vec![(0.into(), ConstF64::new(f).into())])
    }
}
