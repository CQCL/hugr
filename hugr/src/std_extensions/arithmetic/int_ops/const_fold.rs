use std::cmp::{max, min};

use crate::{
    extension::{
        prelude::{sum_with_error, ConstError, ConstString},
        ConstFold, ConstFoldResult, OpDef,
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

struct IToBoolFolder;
impl ConstFold for IToBoolFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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

struct IFromBoolFolder;
impl ConstFold for IFromBoolFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IEqFolder;
impl ConstFold for IEqFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct INeFolder;
impl ConstFold for INeFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct ILtUFolder;
impl ConstFold for ILtUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct ILtSFolder;
impl ConstFold for ILtSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IGtUFolder;
impl ConstFold for IGtUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IGtSFolder;
impl ConstFold for IGtSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct ILeUFolder;
impl ConstFold for ILeUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct ILeSFolder;
impl ConstFold for ILeSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IGeUFolder;
impl ConstFold for IGeUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IGeSFolder;
impl ConstFold for IGeSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IMaxUFolder;
impl ConstFold for IMaxUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IMaxSFolder;
impl ConstFold for IMaxSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IMinUFolder;
impl ConstFold for IMinUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IMinSFolder;
impl ConstFold for IMinSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
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

struct IAddFolder;
impl ConstFold for IAddFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct ISubFolder;
impl ConstFold for ISubFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct INegFolder;
impl ConstFold for INegFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
                        n0.value_u().overflowing_neg().0 & bitmask_from_logwidth(logwidth),
                    )
                    .unwrap(),
                ),
            )])
        }
    }
}

struct IMulFolder;
impl ConstFold for IMulFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IDivModCheckedUFolder;
impl ConstFold for IDivModCheckedUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IDivModUFolder;
impl ConstFold for IDivModUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
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

struct IDivModCheckedSFolder;
impl ConstFold for IDivModCheckedSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IDivModSFolder;
impl ConstFold for IDivModSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IDivCheckedUFolder;
impl ConstFold for IDivCheckedUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IDivUFolder;
impl ConstFold for IDivUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IModCheckedUFolder;
impl ConstFold for IModCheckedUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IModUFolder;
impl ConstFold for IModUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IDivCheckedSFolder;
impl ConstFold for IDivCheckedSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IDivSFolder;
impl ConstFold for IDivSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IModCheckedSFolder;
impl ConstFold for IModCheckedSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IModSFolder;
impl ConstFold for IModSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IAbsFolder;
impl ConstFold for IAbsFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IAndFolder;
impl ConstFold for IAndFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
                Value::extension(ConstInt::new_u(logwidth, n0.value_u() & n1.value_u()).unwrap()),
            )])
        }
    }
}

struct IOrFolder;
impl ConstFold for IOrFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
                Value::extension(ConstInt::new_u(logwidth, n0.value_u() | n1.value_u()).unwrap()),
            )])
        }
    }
}

struct IXorFolder;
impl ConstFold for IXorFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
                Value::extension(ConstInt::new_u(logwidth, n0.value_u() ^ n1.value_u()).unwrap()),
            )])
        }
    }
}

struct INotFolder;
impl ConstFold for INotFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
                Value::extension(ConstInt::new_u(logwidth, !n0.value_u()).unwrap()),
            )])
        }
    }
}

struct IShlFolder;
impl ConstFold for IShlFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
                        (n0.value_u() << n1.value_u()) & bitmask_from_logwidth(logwidth0),
                    )
                    .unwrap(),
                ),
            )])
        }
    }
}

struct IShrFolder;
impl ConstFold for IShrFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
                Value::extension(ConstInt::new_u(logwidth0, n0.value_u() >> n1.value_u()).unwrap()),
            )])
        }
    }
}

struct IRotlFolder;
impl ConstFold for IRotlFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IRotrFolder;
impl ConstFold for IRotrFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IToStringUFolder;
impl ConstFold for IToStringUFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

struct IToStringSFolder;
impl ConstFold for IToStringSFolder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
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
    }
}

pub(super) fn set_fold(op: &IntOpDef, def: &mut OpDef) {
    match op {
        IntOpDef::iwiden_u => def.set_constant_folder(IWidenUFolder),
        IntOpDef::iwiden_s => def.set_constant_folder(IWidenSFolder),
        IntOpDef::inarrow_u => def.set_constant_folder(INarrowUFolder),
        IntOpDef::inarrow_s => def.set_constant_folder(INarrowSFolder),
        IntOpDef::itobool => def.set_constant_folder(IToBoolFolder),
        IntOpDef::ifrombool => def.set_constant_folder(IFromBoolFolder),
        IntOpDef::ieq => def.set_constant_folder(IEqFolder),
        IntOpDef::ine => def.set_constant_folder(INeFolder),
        IntOpDef::ilt_u => def.set_constant_folder(ILtUFolder),
        IntOpDef::ilt_s => def.set_constant_folder(ILtSFolder),
        IntOpDef::igt_u => def.set_constant_folder(IGtUFolder),
        IntOpDef::igt_s => def.set_constant_folder(IGtSFolder),
        IntOpDef::ile_u => def.set_constant_folder(ILeUFolder),
        IntOpDef::ile_s => def.set_constant_folder(ILeSFolder),
        IntOpDef::ige_u => def.set_constant_folder(IGeUFolder),
        IntOpDef::ige_s => def.set_constant_folder(IGeSFolder),
        IntOpDef::imax_u => def.set_constant_folder(IMaxUFolder),
        IntOpDef::imax_s => def.set_constant_folder(IMaxSFolder),
        IntOpDef::imin_u => def.set_constant_folder(IMinUFolder),
        IntOpDef::imin_s => def.set_constant_folder(IMinSFolder),
        IntOpDef::iadd => def.set_constant_folder(IAddFolder),
        IntOpDef::isub => def.set_constant_folder(ISubFolder),
        IntOpDef::ineg => def.set_constant_folder(INegFolder),
        IntOpDef::imul => def.set_constant_folder(IMulFolder),
        IntOpDef::idivmod_checked_u => def.set_constant_folder(IDivModCheckedUFolder),
        IntOpDef::idivmod_u => def.set_constant_folder(IDivModUFolder),
        IntOpDef::idivmod_checked_s => def.set_constant_folder(IDivModCheckedSFolder),
        IntOpDef::idivmod_s => def.set_constant_folder(IDivModSFolder),
        IntOpDef::idiv_checked_u => def.set_constant_folder(IDivCheckedUFolder),
        IntOpDef::idiv_u => def.set_constant_folder(IDivUFolder),
        IntOpDef::imod_checked_u => def.set_constant_folder(IModCheckedUFolder),
        IntOpDef::imod_u => def.set_constant_folder(IModUFolder),
        IntOpDef::idiv_checked_s => def.set_constant_folder(IDivCheckedSFolder),
        IntOpDef::idiv_s => def.set_constant_folder(IDivSFolder),
        IntOpDef::imod_checked_s => def.set_constant_folder(IModCheckedSFolder),
        IntOpDef::imod_s => def.set_constant_folder(IModSFolder),
        IntOpDef::iabs => def.set_constant_folder(IAbsFolder),
        IntOpDef::iand => def.set_constant_folder(IAndFolder),
        IntOpDef::ior => def.set_constant_folder(IOrFolder),
        IntOpDef::ixor => def.set_constant_folder(IXorFolder),
        IntOpDef::inot => def.set_constant_folder(INotFolder),
        IntOpDef::ishl => def.set_constant_folder(IShlFolder),
        IntOpDef::ishr => def.set_constant_folder(IShrFolder),
        IntOpDef::irotl => def.set_constant_folder(IRotlFolder),
        IntOpDef::irotr => def.set_constant_folder(IRotrFolder),
        IntOpDef::itostring_u => def.set_constant_folder(IToStringUFolder),
        IntOpDef::itostring_s => def.set_constant_folder(IToStringSFolder),
    }
}
