use crate::{
    algorithm::const_fold::sorted_consts,
    extension::{ConstFold, ConstFoldResult, OpDef},
    ops,
    std_extensions::arithmetic::float_types::ConstF64,
    IncomingPort,
};

use super::FloatOps;

pub(super) fn set_fold(op: &FloatOps, def: &mut OpDef) {
    use FloatOps::*;

    match op {
        fmax | fmin | fadd | fsub | fmul | fdiv => def.set_constant_folder(BinaryFold::from_op(op)),
        feq | fne | flt | fgt | fle | fge => def.set_constant_folder(CmpFold::from_op(*op)),
        fneg | fabs | ffloor | fceil => def.set_constant_folder(UnaryFold::from_op(op)),
    }
}

/// Extract float values from constants in port order.
fn get_floats<const N: usize>(consts: &[(IncomingPort, ops::Const)]) -> Option<[f64; N]> {
    let consts: [&ops::Const; N] = sorted_consts(consts).try_into().ok()?;

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
            _ => panic!("not binary op"),
        }))
    }
}
impl ConstFold for BinaryFold {
    fn fold(
        &self,
        _type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Const)],
    ) -> ConstFoldResult {
        let [f1, f2] = get_floats(consts)?;

        let res = ConstF64::new((self.0)(f1, f2));
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
                fne => f64::lt,
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
        consts: &[(IncomingPort, ops::Const)],
    ) -> ConstFoldResult {
        let [f1, f2] = get_floats(consts)?;

        let res = ops::Const::from_bool((self.0)(f1, f2));

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
            _ => panic!("not unary op."),
        }))
    }
}

impl ConstFold for UnaryFold {
    fn fold(
        &self,
        _type_args: &[crate::types::TypeArg],
        consts: &[(IncomingPort, ops::Const)],
    ) -> ConstFoldResult {
        let [f1] = get_floats(consts)?;
        let res = ConstF64::new((self.0)(f1));
        Some(vec![(0.into(), res.into())])
    }
}
