use crate::{
    extension::{
        prelude::{sum_with_error, ConstError},
        ConstFold, ConstFoldResult, OpDef,
    },
    ops,
    std_extensions::arithmetic::{
        float_types::ConstF64,
        int_types::{get_log_width, ConstIntS, ConstIntU, INT_TYPES},
    },
    types::ConstTypeError,
    values::{CustomConst, Value},
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

fn get_input<T: CustomConst>(consts: &[(IncomingPort, ops::Const)]) -> Option<&T> {
    let [(_, c)] = consts else {
        return None;
    };
    c.get_custom_value()
}

fn fold_trunc(
    type_args: &[crate::types::TypeArg],
    consts: &[(IncomingPort, ops::Const)],
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
        let sum_val = Value::Sum {
            tag: 1,
            value: Box::new(err_val.into()),
        };

        ops::Const::new(sum_val, sum_type.clone()).unwrap()
    };
    let out_const: ops::Const = if !f.is_finite() {
        err_value()
    } else {
        let cv = convert(f, log_width);
        if let Ok(cv) = cv {
            let sum_val = Value::Sum {
                tag: 0,
                value: Box::new(cv),
            };

            ops::Const::new(sum_val, sum_type).unwrap()
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
        consts: &[(IncomingPort, ops::Const)],
    ) -> ConstFoldResult {
        fold_trunc(type_args, consts, |f, log_width| {
            ConstIntU::new(log_width, f.trunc() as u64).map(Into::into)
        })
    }
}

struct TruncS;

impl ConstFold for TruncS {
    fn fold(
        &self,
        type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Const)],
    ) -> ConstFoldResult {
        fold_trunc(type_args, consts, |f, log_width| {
            ConstIntS::new(log_width, f.trunc() as i64).map(Into::into)
        })
    }
}

struct ConvertU;

impl ConstFold for ConvertU {
    fn fold(
        &self,
        _type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Const)],
    ) -> ConstFoldResult {
        let u: &ConstIntU = get_input(consts)?;
        let f = u.value() as f64;
        Some(vec![(0.into(), ConstF64::new(f).into())])
    }
}

struct ConvertS;

impl ConstFold for ConvertS {
    fn fold(
        &self,
        _type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Const)],
    ) -> ConstFoldResult {
        let u: &ConstIntS = get_input(consts)?;
        let f = u.value() as f64;
        Some(vec![(0.into(), ConstF64::new(f).into())])
    }
}
