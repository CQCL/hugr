use std::{
    cmp::{max, min},
    sync::LazyLock,
};

use crate::{
    IncomingPort,
    extension::{
        ConstFoldResult, Folder, OpDef,
        prelude::{ConstError, sum_with_error},
    },
    ops::{
        Value,
        constant::{get_pair_of_input_values, get_single_input_value},
    },
    std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES, get_log_width},
    types::{Type, TypeArg},
};

use super::IntOpDef;

static INARROW_ERROR_VALUE: LazyLock<Value> = LazyLock::new(|| {
    ConstError {
        signal: 0,
        message: "Integer too large to narrow".to_string(),
    }
    .into()
});

fn bitmask_from_width(width: u64) -> u64 {
    debug_assert!(width <= 64);
    if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn bitmask_from_logwidth(logwidth: u8) -> u64 {
    bitmask_from_width(1u64 << logwidth)
}

// return q, r s.t. n = qm + r, 0 <= r < m
fn divmod_s(n: i64, m: u64) -> (i64, u64) {
    // This is quite hairy.
    if n >= 0 {
        let n_u = n as u64;
        ((n_u / m) as i64, n_u % m)
    } else if n != i64::MIN {
        // -2^63 < n < 0
        let n_u = (-n) as u64;
        let q = (n_u / m) as i64;
        let r = n_u % m;
        if r == 0 { (-q, 0) } else { (-q - 1, m - r) }
    } else if m == 1 {
        // n = -2^63, m = 1
        (n, 0)
    } else if m < (1u64 << 63) {
        // n = -2^63, 1 < m < 2^63
        let m_s = m as i64;
        let q = n / m_s;
        let r = n % m_s;
        if r == 0 {
            (q, 0)
        } else {
            (q - 1, (m_s - r) as u64)
        }
    } else {
        // n = -2^63, m >= 2^63
        (-1, m - (1u64 << 63))
    }
}

pub(super) fn set_fold(op: &IntOpDef, def: &mut OpDef) {
    def.set_constant_folder(match op {
        IntOpDef::iwiden_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
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
                },
            ),
        },
        IntOpDef::iwiden_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
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
                },
            ),
        },
        IntOpDef::inarrow_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let n0: &ConstInt = get_single_input_value(consts)?;
                    (logwidth0 >= logwidth1 && n0.log_width() == logwidth0).then_some(())?;

                    let int_out_type = INT_TYPES[logwidth1 as usize].clone();
                    let sum_type = sum_with_error(int_out_type.clone());

                    let mk_out_const = |i, mb_v: Result<Value, _>| {
                        mb_v.and_then(|v| Value::sum(i, [v], sum_type))
                            .unwrap_or_else(|e| panic!("Invalid computed sum, {e}"))
                    };
                    let n0val: u64 = n0.value_u();
                    let out_const: Value = if n0val >> (1 << logwidth1) != 0 {
                        mk_out_const(0, Ok(INARROW_ERROR_VALUE.clone()))
                    } else {
                        mk_out_const(1, ConstInt::new_u(logwidth1, n0val).map(Into::into))
                    };
                    Some(vec![(0.into(), out_const)])
                },
            ),
        },
        IntOpDef::inarrow_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let n0: &ConstInt = get_single_input_value(consts)?;
                    (logwidth0 >= logwidth1 && n0.log_width() == logwidth0).then_some(())?;

                    let int_out_type = INT_TYPES[logwidth1 as usize].clone();
                    let sum_type = sum_with_error(int_out_type.clone());
                    let mk_out_const = |i, mb_v: Result<Value, _>| {
                        mb_v.and_then(|v| Value::sum(i, [v], sum_type))
                            .unwrap_or_else(|e| panic!("Invalid computed sum, {e}"))
                    };
                    let n0val: i64 = n0.value_s();
                    let ub = 1i64 << ((1 << logwidth1) - 1);
                    let out_const: Value = if n0val >= ub || n0val < -ub {
                        mk_out_const(0, Ok(INARROW_ERROR_VALUE.clone()))
                    } else {
                        mk_out_const(1, ConstInt::new_s(logwidth1, n0val).map(Into::into))
                    };
                    Some(vec![(0.into(), out_const)])
                },
            ),
        },
        IntOpDef::ieq => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::from_bool(n0.value_u() == n1.value_u()),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ine => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::from_bool(n0.value_u() != n1.value_u()),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ilt_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::from_bool(n0.value_u() < n1.value_u()),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ilt_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::from_bool(n0.value_s() < n1.value_s()),
                        )])
                    }
                },
            ),
        },
        IntOpDef::igt_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::from_bool(n0.value_u() > n1.value_u()),
                        )])
                    }
                },
            ),
        },
        IntOpDef::igt_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::from_bool(n0.value_s() > n1.value_s()),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ile_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::from_bool(n0.value_u() <= n1.value_u()),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ile_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::from_bool(n0.value_s() <= n1.value_s()),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ige_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::from_bool(n0.value_u() >= n1.value_u()),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ige_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::from_bool(n0.value_s() >= n1.value_s()),
                        )])
                    }
                },
            ),
        },
        IntOpDef::imax_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(logwidth, max(n0.value_u(), n1.value_u())).unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::imax_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_s(logwidth, max(n0.value_s(), n1.value_s())).unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::imin_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(logwidth, min(n0.value_u(), n1.value_u())).unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::imin_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_s(logwidth, min(n0.value_s(), n1.value_s())).unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::iadd => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(
                                    logwidth,
                                    n0.value_u().overflowing_add(n1.value_u()).0
                                        & bitmask_from_logwidth(logwidth),
                                )
                                .unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::isub => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(
                                    logwidth,
                                    n0.value_u().overflowing_sub(n1.value_u()).0
                                        & bitmask_from_logwidth(logwidth),
                                )
                                .unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ineg => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let n0: &ConstInt = get_single_input_value(consts)?;
                    if n0.log_width() == logwidth {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(
                                    logwidth,
                                    n0.value_u().overflowing_neg().0
                                        & bitmask_from_logwidth(logwidth),
                                )
                                .unwrap(),
                            ),
                        )])
                    } else {
                        None
                    }
                },
            ),
        },
        IntOpDef::imul => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(
                                    logwidth,
                                    n0.value_u().overflowing_mul(n1.value_u()).0
                                        & bitmask_from_logwidth(logwidth),
                                )
                                .unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ipow => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(
                                    logwidth,
                                    n0.value_u()
                                        .overflowing_pow(
                                            n1.value_u().try_into().unwrap_or(u32::MAX),
                                        )
                                        .0
                                        & bitmask_from_logwidth(logwidth),
                                )
                                .unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::idivmod_checked_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 {
                        None
                    } else {
                        let q_type = INT_TYPES[logwidth0 as usize].clone();
                        let r_type = q_type.clone();
                        let qr_type: Type = Type::new_tuple(vec![q_type, r_type]);
                        let err_value = || {
                            ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            }
                            .as_either(qr_type)
                        };
                        let nval = n.value_u();
                        let mval = m.value_u();
                        let out_const: Value = if mval == 0 {
                            err_value()
                        } else {
                            let qval = nval / mval;
                            let rval = nval % mval;
                            Value::tuple(vec![
                                Value::extension(ConstInt::new_u(logwidth0, qval).unwrap()),
                                Value::extension(ConstInt::new_u(logwidth0, rval).unwrap()),
                            ])
                        };
                        Some(vec![(0.into(), out_const)])
                    }
                },
            ),
        },
        IntOpDef::idivmod_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_u();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 || mval == 0 {
                        None
                    } else {
                        let qval = nval / mval;
                        let rval = nval % mval;
                        let q = Value::extension(ConstInt::new_u(logwidth0, qval).unwrap());
                        let r = Value::extension(ConstInt::new_u(logwidth0, rval).unwrap());
                        Some(vec![(0.into(), q), (1.into(), r)])
                    }
                },
            ),
        },
        IntOpDef::idivmod_checked_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 {
                        None
                    } else {
                        let q_type = INT_TYPES[logwidth0 as usize].clone();
                        let r_type = INT_TYPES[logwidth0 as usize].clone();
                        let qr_type: Type = Type::new_tuple(vec![q_type, r_type]);
                        let err_value = || {
                            ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            }
                            .as_either(qr_type)
                        };
                        let nval = n.value_s();
                        let mval = m.value_u();
                        let out_const: Value = if mval == 0 {
                            err_value()
                        } else {
                            let (qval, rval) = divmod_s(nval, mval);
                            Value::tuple(vec![
                                Value::extension(ConstInt::new_s(logwidth0, qval).unwrap()),
                                Value::extension(ConstInt::new_u(logwidth0, rval).unwrap()),
                            ])
                        };
                        Some(vec![(0.into(), out_const)])
                    }
                },
            ),
        },
        IntOpDef::idivmod_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_s();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 || mval == 0 {
                        None
                    } else {
                        let (qval, rval) = divmod_s(nval, mval);
                        let q = Value::extension(ConstInt::new_s(logwidth0, qval).unwrap());
                        let r = Value::extension(ConstInt::new_u(logwidth0, rval).unwrap());
                        Some(vec![(0.into(), q), (1.into(), r)])
                    }
                },
            ),
        },
        IntOpDef::idiv_checked_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 {
                        None
                    } else {
                        let int_out_type = INT_TYPES[logwidth0 as usize].clone();
                        let err_value = || {
                            ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            }
                            .as_either(int_out_type.clone())
                        };
                        let nval = n.value_u();
                        let mval = m.value_u();
                        let out_const: Value = if mval == 0 {
                            err_value()
                        } else {
                            Value::extension(ConstInt::new_u(logwidth0, nval / mval).unwrap())
                        };
                        Some(vec![(0.into(), out_const)])
                    }
                },
            ),
        },
        IntOpDef::idiv_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_u();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 || mval == 0 {
                        None
                    } else {
                        let q = Value::extension(ConstInt::new_u(logwidth0, nval / mval).unwrap());
                        Some(vec![(0.into(), q)])
                    }
                },
            ),
        },
        IntOpDef::imod_checked_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 {
                        None
                    } else {
                        let int_out_type = INT_TYPES[logwidth0 as usize].clone();
                        let err_value = || {
                            ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            }
                            .as_either(int_out_type.clone())
                        };
                        let nval = n.value_u();
                        let mval = m.value_u();
                        let out_const: Value = if mval == 0 {
                            err_value()
                        } else {
                            Value::extension(ConstInt::new_u(logwidth0, nval % mval).unwrap())
                        };
                        Some(vec![(0.into(), out_const)])
                    }
                },
            ),
        },
        IntOpDef::imod_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_u();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 || mval == 0 {
                        None
                    } else {
                        let r = Value::extension(ConstInt::new_u(logwidth0, nval % mval).unwrap());
                        Some(vec![(0.into(), r)])
                    }
                },
            ),
        },
        IntOpDef::idiv_checked_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 {
                        None
                    } else {
                        let int_out_type = INT_TYPES[logwidth0 as usize].clone();
                        let err_value = || {
                            ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            }
                            .as_either(int_out_type.clone())
                        };
                        let nval = n.value_s();
                        let mval = m.value_u();
                        let out_const: Value = if mval == 0 {
                            err_value()
                        } else {
                            let (qval, _) = divmod_s(nval, mval);
                            Value::extension(ConstInt::new_s(logwidth0, qval).unwrap())
                        };
                        Some(vec![(0.into(), out_const)])
                    }
                },
            ),
        },
        IntOpDef::idiv_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_s();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 || mval == 0 {
                        None
                    } else {
                        let (qval, _) = divmod_s(nval, mval);
                        let q = Value::extension(ConstInt::new_s(logwidth0, qval).unwrap());
                        Some(vec![(0.into(), q)])
                    }
                },
            ),
        },
        IntOpDef::imod_checked_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 {
                        None
                    } else {
                        let int_out_type = INT_TYPES[logwidth0 as usize].clone();
                        let err_value = || {
                            ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            }
                            .as_either(int_out_type.clone())
                        };
                        let nval = n.value_s();
                        let mval = m.value_u();
                        let out_const: Value = if mval == 0 {
                            err_value()
                        } else {
                            let (_, rval) = divmod_s(nval, mval);
                            Value::extension(ConstInt::new_u(logwidth0, rval).unwrap())
                        };
                        Some(vec![(0.into(), out_const)])
                    }
                },
            ),
        },
        IntOpDef::imod_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_s();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth0 || mval == 0 {
                        None
                    } else {
                        let (_, rval) = divmod_s(nval, mval);
                        let r = Value::extension(ConstInt::new_u(logwidth0, rval).unwrap());
                        Some(vec![(0.into(), r)])
                    }
                },
            ),
        },
        IntOpDef::iabs => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let n0: &ConstInt = get_single_input_value(consts)?;
                    let n0val = n0.value_s();
                    if n0.log_width() == logwidth {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                if n0val == i64::MIN {
                                    debug_assert!(logwidth == 6);
                                    ConstInt::new_u(6, 1u64 << 63)
                                } else {
                                    ConstInt::new_s(logwidth, n0val.abs())
                                }
                                .unwrap(),
                            ),
                        )])
                    } else {
                        None
                    }
                },
            ),
        },
        IntOpDef::iand => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(logwidth, n0.value_u() & n1.value_u()).unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ior => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(logwidth, n0.value_u() | n1.value_u()).unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ixor => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth || n1.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(logwidth, n0.value_u() ^ n1.value_u()).unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::inot => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let n0: &ConstInt = get_single_input_value(consts)?;
                    if n0.log_width() == logwidth {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(
                                    logwidth,
                                    bitmask_from_logwidth(logwidth) & !n0.value_u(),
                                )
                                .unwrap(),
                            ),
                        )])
                    } else {
                        None
                    }
                },
            ),
        },
        IntOpDef::ishl => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth0 || n1.log_width() != logwidth0 {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(
                                    logwidth0,
                                    (n0.value_u() << n1.value_u())
                                        & bitmask_from_logwidth(logwidth0),
                                )
                                .unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::ishr => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth0 || n1.log_width() != logwidth0 {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(logwidth0, n0.value_u() >> n1.value_u()).unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::irotl => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth0 || n1.log_width() != logwidth0 {
                        None
                    } else {
                        let n = n0.value_u();
                        let w = 1 << logwidth0;
                        let k = n1.value_u() % w; // equivalent rotation amount
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(
                                    logwidth0,
                                    ((n << k) & bitmask_from_width(w)) | (n >> (w - k)),
                                )
                                .unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::irotr => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;

                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth0 || n1.log_width() != logwidth0 {
                        None
                    } else {
                        let n = n0.value_u();
                        let w = 1 << logwidth0;
                        let k = n1.value_u() % w; // equivalent rotation amount
                        Some(vec![(
                            0.into(),
                            Value::extension(
                                ConstInt::new_u(
                                    logwidth0,
                                    ((n << (w - k)) & bitmask_from_width(w)) | (n >> k),
                                )
                                .unwrap(),
                            ),
                        )])
                    }
                },
            ),
        },
        IntOpDef::is_to_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let n0: &ConstInt = get_single_input_value(consts)?;
                    if n0.log_width() == logwidth {
                        assert!(
                            (n0.value_s() >= 0),
                            "Cannot convert negative integer {} to unsigned.",
                            n0.value_s()
                        );
                        Some(vec![(0.into(), Value::extension(n0.clone()))])
                    } else {
                        None
                    }
                },
            ),
        },
        IntOpDef::iu_to_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let n0: &ConstInt = get_single_input_value(consts)?;
                    if n0.log_width() == logwidth {
                        assert!(
                            (n0.value_s() >= 0),
                            "Unsigned integer {} is too large to be converted to signed.",
                            n0.value_u()
                        );
                        Some(vec![(0.into(), Value::extension(n0.clone()))])
                    } else {
                        None
                    }
                },
            ),
        },
    });
}
