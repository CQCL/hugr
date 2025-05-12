use crate::extension::prelude::{ConstString, ConstUsize};
use crate::ops::Value;
use crate::ops::constant::get_single_input_value;
use crate::std_extensions::arithmetic::int_types::INT_TYPES;
use crate::{
    IncomingPort,
    extension::{
        ConstFold, ConstFoldResult, OpDef,
        prelude::{ConstError, const_ok, error_type},
    },
    ops,
    std_extensions::arithmetic::{
        float_types::ConstF64,
        int_types::{ConstInt, get_log_width},
    },
    types::ConstTypeError,
};

use super::ConvertOpDef;

pub(super) fn set_fold(op: &ConvertOpDef, def: &mut OpDef) {
    use ConvertOpDef::*;

    match op {
        trunc_u => def.set_constant_folder(TruncU),
        trunc_s => def.set_constant_folder(TruncS),
        convert_u => def.set_constant_folder(ConvertU),
        convert_s => def.set_constant_folder(ConvertS),
        itobool => def.set_constant_folder(IToBool),
        ifrombool => def.set_constant_folder(IFromBool),
        itostring_u => def.set_constant_folder(IToStringU),
        itostring_s => def.set_constant_folder(IToStringS),
        itousize => def.set_constant_folder(IToUsize),
        ifromusize => def.set_constant_folder(IFromUsize),
        bytecast_float64_to_int64 | bytecast_int64_to_float64 => (), // We don't have constant folders for bytecasting yet
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
    let int_type = INT_TYPES[log_width as usize].clone();
    let err_value = || {
        ConstError {
            signal: 0,
            message: "Can't truncate non-finite float".to_string(),
        }
        .as_either(int_type.clone())
    };
    let out_const: ops::Value = if f.is_finite() {
        let cv = convert(f, log_width);
        if let Ok(cv) = cv {
            const_ok(cv, error_type())
        } else {
            err_value()
        }
    } else {
        err_value()
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

struct IToBool;

impl ConstFold for IToBool {
    fn fold(
        &self,
        type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        if !type_args.is_empty() {
            return None;
        }
        let n0: &ConstInt = get_single_input_value(consts)?;
        if n0.log_width() != 0 {
            None
        } else {
            Some(vec![(0.into(), Value::from_bool(n0.value_u() == 1))])
        }
    }
}

struct IFromBool;

impl ConstFold for IFromBool {
    fn fold(
        &self,
        type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        if !type_args.is_empty() {
            return None;
        }
        let [(_, b0)] = consts else {
            return None;
        };
        Some(vec![(
            0.into(),
            Value::extension(
                ConstInt::new_u(0, u64::from(b0.clone() == Value::true_val())).unwrap(),
            ),
        )])
    }
}

struct IToStringU;

impl ConstFold for IToStringU {
    fn fold(
        &self,
        type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        let [arg] = type_args else {
            return None;
        };
        let logwidth: u8 = get_log_width(arg).ok()?;
        let n0: &ConstInt = get_single_input_value(consts)?;
        if n0.log_width() == logwidth {
            Some(vec![(
                0.into(),
                Value::extension(ConstString::new(n0.value_u().to_string())),
            )])
        } else {
            None
        }
    }
}

struct IToStringS;

impl ConstFold for IToStringS {
    fn fold(
        &self,
        type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        let [arg] = type_args else {
            return None;
        };
        let logwidth: u8 = get_log_width(arg).ok()?;
        let n0: &ConstInt = get_single_input_value(consts)?;
        if n0.log_width() == logwidth {
            Some(vec![(
                0.into(),
                Value::extension(ConstString::new(n0.value_s().to_string())),
            )])
        } else {
            None
        }
    }
}

struct IToUsize;

impl ConstFold for IToUsize {
    fn fold(
        &self,
        type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        if !type_args.is_empty() {
            return None;
        }
        let n0: &ConstInt = get_single_input_value(consts)?;
        if n0.log_width() == 6 {
            Some(vec![(
                0.into(),
                Value::extension(ConstUsize::new(n0.value_u())),
            )])
        } else {
            None
        }
    }
}

struct IFromUsize;

impl ConstFold for IFromUsize {
    fn fold(
        &self,
        type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        if !type_args.is_empty() {
            return None;
        }
        let n0: &ConstUsize = get_single_input_value(consts)?;
        Some(vec![(
            0.into(),
            Value::extension(ConstInt::new_u(6, n0.value()).unwrap()),
        )])
    }
}
