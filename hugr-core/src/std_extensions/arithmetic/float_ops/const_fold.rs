use crate::{
    IncomingPort,
    extension::{ConstFold, ConstFoldResult, OpDef, prelude::ConstString},
    ops,
    std_extensions::arithmetic::float_types::ConstF64,
    utils::sorted_consts,
};

use super::FloatOps;

pub(super) fn set_fold(op: &FloatOps, def: &mut OpDef) {
    use FloatOps::*;

    match op {
        fmax | fmin | fadd | fsub | fmul | fdiv | fpow => {
            def.set_constant_folder(BinaryFold::from_op(op));
        }
        feq | fne | flt | fgt | fle | fge => def.set_constant_folder(CmpFold::from_op(*op)),
        fneg | fabs | ffloor | fceil | fround => def.set_constant_folder(UnaryFold::from_op(op)),
        ftostring => def.set_constant_folder(ToStringFold::from_op(op)),
    }
}

/// Extract float values from constants in port order.
fn get_floats<const N: usize>(consts: &[(IncomingPort, ops::Value)]) -> Option<[f64; N]> {
    let consts: [&ops::Value; N] = sorted_consts(consts).try_into().ok()?;

    Some(consts.map(|c| {
        let const_f64: &ConstF64 = c
            .get_custom_value()
            .expect("This function assumes all incoming constants are floats.");
        const_f64.value()
    }))
}

/// Fold binary operations
struct BinaryFold(Box<dyn Fn(f64, f64) -> f64 + Send + Sync>);
impl BinaryFold {
    fn from_op(op: &FloatOps) -> Self {
        use FloatOps::*;
        Self(Box::new(match op {
            fmax => f64::max,
            fmin => f64::min,
            fadd => std::ops::Add::add,
            fsub => std::ops::Sub::sub,
            fmul => std::ops::Mul::mul,
            fdiv => std::ops::Div::div,
            fpow => f64::powf,
            _ => panic!("not binary op"),
        }))
    }
}
impl ConstFold for BinaryFold {
    fn fold(
        &self,
        _type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        let [f1, f2] = get_floats(consts)?;
        let x: f64 = (self.0)(f1, f2);
        if !x.is_finite() {
            return None;
        }
        let res = ConstF64::new(x);
        Some(vec![(0.into(), res.into())])
    }
}

/// Fold comparisons.
struct CmpFold(Box<dyn Fn(f64, f64) -> bool + Send + Sync>);
impl CmpFold {
    fn from_op(op: FloatOps) -> Self {
        use FloatOps::*;
        Self(Box::new(move |x, y| {
            (match op {
                feq => f64::eq,
                fne => f64::ne,
                flt => f64::lt,
                fgt => f64::gt,
                fle => f64::le,
                fge => f64::ge,
                _ => panic!("not cmp op"),
            })(&x, &y)
        }))
    }
}

impl ConstFold for CmpFold {
    fn fold(
        &self,
        _type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        let [f1, f2] = get_floats(consts)?;

        let res = ops::Value::from_bool((self.0)(f1, f2));

        Some(vec![(0.into(), res)])
    }
}

/// Fold unary operations
struct UnaryFold(Box<dyn Fn(f64) -> f64 + Send + Sync>);
impl UnaryFold {
    fn from_op(op: &FloatOps) -> Self {
        use FloatOps::*;
        Self(Box::new(match op {
            fneg => std::ops::Neg::neg,
            fabs => f64::abs,
            ffloor => f64::floor,
            fceil => f64::ceil,
            fround => f64::round,
            _ => panic!("not unary op."),
        }))
    }
}

impl ConstFold for UnaryFold {
    fn fold(
        &self,
        _type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        let [f1] = get_floats(consts)?;
        let x: f64 = (self.0)(f1);
        if !x.is_finite() {
            return None;
        }
        let res = ConstF64::new(x);
        Some(vec![(0.into(), res.into())])
    }
}

/// Fold string-conversion operations
struct ToStringFold(Box<dyn Fn(f64) -> String + Send + Sync>);
impl ToStringFold {
    fn from_op(_op: &FloatOps) -> Self {
        Self(Box::new(|x| x.to_string()))
    }
}
impl ConstFold for ToStringFold {
    fn fold(
        &self,
        _type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Value)],
    ) -> ConstFoldResult {
        let [f] = get_floats(consts)?;
        let res = ConstString::new((self.0)(f));
        Some(vec![(0.into(), res.into())])
    }
}
