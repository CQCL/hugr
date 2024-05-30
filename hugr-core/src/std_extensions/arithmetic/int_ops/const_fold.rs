use std::cmp::{max, min};

use crate::{
    extension::{
        prelude::{sum_with_error, ConstError, ConstString},
        ConstFoldResult, Folder, OpDef,
    },
    ops::{
        constant::{get_pair_of_input_values, get_single_input_value},
        Value,
    },
    std_extensions::arithmetic::int_types::{get_log_width, ConstInt, INT_TYPES},
    types::{SumType, Type, TypeArg},
    IncomingPort,
};

use super::IntOpDef;

use lazy_static::lazy_static;

lazy_static! {
    static ref INARROW_ERROR_VALUE: Value = ConstError {
        signal: 0,
        message: "Integer too large to narrow".to_string(),
    }
    .into();
}

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
        if r == 0 {
            (-q, 0)
        } else {
            (-q - 1, m - r)
        }
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

                    let int_out_type = INT_TYPES[logwidth1 as usize].to_owned();
                    let sum_type = sum_with_error(int_out_type.clone());

                    let mk_out_const = |i, mb_v: Result<Value, _>| {
                        mb_v.and_then(|v| Value::sum(i, [v], sum_type))
                            .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
                    };
                    let n0val: u64 = n0.value_u();
                    let out_const: Value = if n0val >> (1 << logwidth1) != 0 {
                        mk_out_const(1, Ok(INARROW_ERROR_VALUE.clone()))
                    } else {
                        mk_out_const(0, ConstInt::new_u(logwidth1, n0val).map(Into::into))
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

                    let int_out_type = INT_TYPES[logwidth1 as usize].to_owned();
                    let sum_type = sum_with_error(int_out_type.clone());
                    let mk_out_const = |i, mb_v: Result<Value, _>| {
                        mb_v.and_then(|v| Value::sum(i, [v], sum_type))
                            .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
                    };
                    let n0val: i64 = n0.value_s();
                    let ub = 1i64 << ((1 << logwidth1) - 1);
                    let out_const: Value = if n0val >= ub || n0val < -ub {
                        mk_out_const(1, Ok(INARROW_ERROR_VALUE.clone()))
                    } else {
                        mk_out_const(0, ConstInt::new_s(logwidth1, n0val).map(Into::into))
                    };
                    Some(vec![(0.into(), out_const)])
                },
            ),
        },
        IntOpDef::itobool => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    if !type_args.is_empty() {
                        return None;
                    }
                    let n0: &ConstInt = get_single_input_value(consts)?;
                    if n0.log_width() != 0 {
                        None
                    } else {
                        Some(vec![(0.into(), Value::from_bool(n0.value_u() == 1))])
                    }
                },
            ),
        },
        IntOpDef::ifrombool => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    if !type_args.is_empty() {
                        return None;
                    }
                    let [(_, b0)] = consts else {
                        return None;
                    };
                    Some(vec![(
                        0.into(),
                        Value::extension(
                            ConstInt::new_u(
                                0,
                                if b0.clone() == Value::true_val() {
                                    1
                                } else {
                                    0
                                },
                            )
                            .unwrap(),
                        ),
                    )])
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
                    if n0.log_width() != logwidth {
                        None
                    } else {
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
        IntOpDef::idivmod_checked_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 {
                        None
                    } else {
                        let q_type = INT_TYPES[logwidth0 as usize].to_owned();
                        let r_type = INT_TYPES[logwidth1 as usize].to_owned();
                        let qr_type: Type = Type::new_tuple(vec![q_type, r_type]);
                        let sum_type: SumType = sum_with_error(qr_type);
                        let err_value = || {
                            let err_val = ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            };
                            Value::sum(1, [err_val.into()], sum_type.clone())
                                .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
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
                                Value::extension(ConstInt::new_u(logwidth1, rval).unwrap()),
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
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_u();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 || mval == 0 {
                        None
                    } else {
                        let qval = nval / mval;
                        let rval = nval % mval;
                        let q = Value::extension(ConstInt::new_u(logwidth0, qval).unwrap());
                        let r = Value::extension(ConstInt::new_u(logwidth1, rval).unwrap());
                        Some(vec![(0.into(), q), (1.into(), r)])
                    }
                },
            ),
        },
        IntOpDef::idivmod_checked_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 {
                        None
                    } else {
                        let q_type = INT_TYPES[logwidth0 as usize].to_owned();
                        let r_type = INT_TYPES[logwidth1 as usize].to_owned();
                        let qr_type: Type = Type::new_tuple(vec![q_type, r_type]);
                        let sum_type: SumType = sum_with_error(qr_type);
                        let err_value = || {
                            let err_val = ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            };
                            Value::sum(1, [err_val.into()], sum_type.clone())
                                .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
                        };
                        let nval = n.value_s();
                        let mval = m.value_u();
                        let out_const: Value = if mval == 0 {
                            err_value()
                        } else {
                            let (qval, rval) = divmod_s(nval, mval);
                            Value::tuple(vec![
                                Value::extension(ConstInt::new_s(logwidth0, qval).unwrap()),
                                Value::extension(ConstInt::new_u(logwidth1, rval).unwrap()),
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
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_s();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 || mval == 0 {
                        None
                    } else {
                        let (qval, rval) = divmod_s(nval, mval);
                        let q = Value::extension(ConstInt::new_s(logwidth0, qval).unwrap());
                        let r = Value::extension(ConstInt::new_u(logwidth1, rval).unwrap());
                        Some(vec![(0.into(), q), (1.into(), r)])
                    }
                },
            ),
        },
        IntOpDef::idiv_checked_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 {
                        None
                    } else {
                        let int_out_type = INT_TYPES[logwidth0 as usize].to_owned();
                        let sum_type = sum_with_error(int_out_type.clone());
                        let err_value = || {
                            let err_val = ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            };
                            Value::sum(1, [err_val.into()], sum_type.clone())
                                .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
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
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_u();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 || mval == 0 {
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
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 {
                        None
                    } else {
                        let int_out_type = INT_TYPES[logwidth1 as usize].to_owned();
                        let sum_type = sum_with_error(int_out_type.clone());
                        let err_value = || {
                            let err_val = ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            };
                            Value::sum(1, [err_val.into()], sum_type.clone())
                                .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
                        };
                        let nval = n.value_u();
                        let mval = m.value_u();
                        let out_const: Value = if mval == 0 {
                            err_value()
                        } else {
                            Value::extension(ConstInt::new_u(logwidth1, nval % mval).unwrap())
                        };
                        Some(vec![(0.into(), out_const)])
                    }
                },
            ),
        },
        IntOpDef::imod_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_u();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 || mval == 0 {
                        None
                    } else {
                        let r = Value::extension(ConstInt::new_u(logwidth1, nval % mval).unwrap());
                        Some(vec![(0.into(), r)])
                    }
                },
            ),
        },
        IntOpDef::idiv_checked_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 {
                        None
                    } else {
                        let int_out_type = INT_TYPES[logwidth0 as usize].to_owned();
                        let sum_type = sum_with_error(int_out_type.clone());
                        let err_value = || {
                            let err_val = ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            };
                            Value::sum(1, [err_val.into()], sum_type.clone())
                                .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
                        };
                        let nval = n.value_s();
                        let mval = m.value_u();
                        let out_const: Value = if mval == 0 {
                            err_value()
                        } else {
                            let (qval, _) = divmod_s(nval, mval);
                            Value::extension(ConstInt::new_s(logwidth1, qval).unwrap())
                        };
                        Some(vec![(0.into(), out_const)])
                    }
                },
            ),
        },
        IntOpDef::idiv_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_s();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 || mval == 0 {
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
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 {
                        None
                    } else {
                        let int_out_type = INT_TYPES[logwidth1 as usize].to_owned();
                        let sum_type = sum_with_error(int_out_type.clone());
                        let err_value = || {
                            let err_val = ConstError {
                                signal: 0,
                                message: "Division by zero".to_string(),
                            };
                            Value::sum(1, [err_val.into()], sum_type.clone())
                                .unwrap_or_else(|e| panic!("Invalid computed sum, {}", e))
                        };
                        let nval = n.value_s();
                        let mval = m.value_u();
                        let out_const: Value = if mval == 0 {
                            err_value()
                        } else {
                            let (_, rval) = divmod_s(nval, mval);
                            Value::extension(ConstInt::new_u(logwidth1, rval).unwrap())
                        };
                        Some(vec![(0.into(), out_const)])
                    }
                },
            ),
        },
        IntOpDef::imod_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n, m): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    let nval = n.value_s();
                    let mval = m.value_u();
                    if n.log_width() != logwidth0 || m.log_width() != logwidth1 || mval == 0 {
                        None
                    } else {
                        let (_, rval) = divmod_s(nval, mval);
                        let r = Value::extension(ConstInt::new_u(logwidth1, rval).unwrap());
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
                    if n0.log_width() != logwidth {
                        None
                    } else {
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
                    if n0.log_width() != logwidth {
                        None
                    } else {
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
                    }
                },
            ),
        },
        IntOpDef::ishl => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth0 || n1.log_width() != logwidth1 {
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
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth0 || n1.log_width() != logwidth1 {
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
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth0 || n1.log_width() != logwidth1 {
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
                    let [arg0, arg1] = type_args else {
                        return None;
                    };
                    let logwidth0: u8 = get_log_width(arg0).ok()?;
                    let logwidth1: u8 = get_log_width(arg1).ok()?;
                    let (n0, n1): (&ConstInt, &ConstInt) = get_pair_of_input_values(consts)?;
                    if n0.log_width() != logwidth0 || n1.log_width() != logwidth1 {
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
        IntOpDef::itostring_u => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let n0: &ConstInt = get_single_input_value(consts)?;
                    if n0.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(ConstString::new(n0.value_u().to_string())),
                        )])
                    }
                },
            ),
        },
        IntOpDef::itostring_s => Folder {
            folder: Box::new(
                |type_args: &[TypeArg], consts: &[(IncomingPort, Value)]| -> ConstFoldResult {
                    let [arg] = type_args else {
                        return None;
                    };
                    let logwidth: u8 = get_log_width(arg).ok()?;
                    let n0: &ConstInt = get_single_input_value(consts)?;
                    if n0.log_width() != logwidth {
                        None
                    } else {
                        Some(vec![(
                            0.into(),
                            Value::extension(ConstString::new(n0.value_s().to_string())),
                        )])
                    }
                },
            ),
        },
    });
}
