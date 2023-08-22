#![allow(missing_docs)]
//! This is an experiment, it is probably already outdated.

use std::ops::{Add, Div, Mul, Neg, Sub};

use cgmath::num_traits::ToPrimitive;
use num_rational::Rational64;
use smol_str::SmolStr;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::types::type_param::TypeArg;
use crate::types::{CustomCheckFailure, CustomType, FunctionType, Type, TypeBound, TypeRow};
use crate::values::CustomConst;
use crate::{ops, Extension};

pub const PI_NAME: &str = "PI";
pub const ANGLE_T_NAME: &str = "angle";
pub const QUAT_T_NAME: &str = "quat";
pub const EXTENSION_ID: SmolStr = SmolStr::new_inline("rotations");

pub const ANGLE_T: Type = Type::new_extension(CustomType::new_simple(
    SmolStr::new_inline(ANGLE_T_NAME),
    EXTENSION_ID,
    TypeBound::Copyable,
));

pub const QUAT_T: Type = Type::new_extension(CustomType::new_simple(
    SmolStr::new_inline(QUAT_T_NAME),
    EXTENSION_ID,
    TypeBound::Copyable,
));
/// The extension with all the operations and types defined in this extension.
pub fn extension() -> Extension {
    let mut extension = Extension::new(EXTENSION_ID);

    RotationType::Angle.add_to_extension(&mut extension);
    RotationType::Quaternion.add_to_extension(&mut extension);

    extension
        .add_op_custom_sig_simple(
            "AngleAdd".into(),
            "".into(),
            vec![],
            |_arg_values: &[TypeArg]| {
                let t: TypeRow =
                    vec![Type::new_extension(RotationType::Angle.custom_type())].into();
                Ok(FunctionType::new(t.clone(), t))
            },
        )
        .unwrap();

    let pi_val = RotationValue::PI;

    extension
        .add_value(PI_NAME, ops::Const::new(pi_val.into(), ANGLE_T).unwrap())
        .unwrap();
    extension
}

/// Custom types defined by this extension.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RotationType {
    Angle,
    Quaternion,
}

impl RotationType {
    pub const fn name(&self) -> SmolStr {
        match self {
            RotationType::Angle => SmolStr::new_inline(ANGLE_T_NAME),
            RotationType::Quaternion => SmolStr::new_inline(QUAT_T_NAME),
        }
    }

    pub const fn description(&self) -> &str {
        match self {
            RotationType::Angle => "Floating point angle",
            RotationType::Quaternion => "Quaternion specifying rotation.",
        }
    }

    pub fn custom_type(self) -> CustomType {
        CustomType::new(self.name(), [], EXTENSION_ID, TypeBound::Copyable)
    }

    fn add_to_extension(self, extension: &mut Extension) {
        extension
            .add_type(
                self.name(),
                vec![],
                self.description().to_string(),
                TypeBound::Copyable.into(),
            )
            .unwrap();
    }
}

impl From<RotationType> for CustomType {
    fn from(ty: RotationType) -> Self {
        ty.custom_type()
    }
}

/// Constant values for [`RotationType`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RotationValue {
    Angle(AngleValue),
    Quaternion(cgmath::Quaternion<f64>),
}

impl RotationValue {
    const PI: Self = Self::Angle(AngleValue::PI);
    fn rotation_type(&self) -> RotationType {
        match self {
            RotationValue::Angle(_) => RotationType::Angle,
            RotationValue::Quaternion(_) => RotationType::Quaternion,
        }
    }
}

#[typetag::serde]
impl CustomConst for RotationValue {
    fn name(&self) -> SmolStr {
        match self {
            RotationValue::Angle(val) => format!("AngleConstant({})", val.radians()),
            RotationValue::Quaternion(val) => format!("QuatConstant({:?})", val),
        }
        .into()
    }

    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        let self_typ = self.rotation_type();

        if &self_typ.custom_type() == typ {
            Ok(())
        } else {
            Err(CustomCheckFailure::Message(
                "Rotation constant type mismatch.".into(),
            ))
        }
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::values::downcast_equal_consts(self, other)
    }
}

//
// TODO:
//
// operations:
//
//     AngleAdd,
//     AngleMul,
//     AngleNeg,
//     QuatMul,
//     RxF64,
//     RzF64,
//     TK1,
//     Rotation,
//     ToRotation,
//
//
//
// signatures:
//
//             LeafOp::AngleAdd | LeafOp::AngleMul => FunctionType::new_linear([Type::Angle]),
//             LeafOp::QuatMul => FunctionType::new_linear([Type::Quat64]),
//             LeafOp::AngleNeg => FunctionType::new_linear([Type::Angle]),
//             LeafOp::RxF64 | LeafOp::RzF64 => {
//                 FunctionType::new_df([Type::Qubit], [Type::Angle])
//             }
//             LeafOp::TK1 => FunctionType::new_df(vec![Type::Qubit], vec![Type::Angle; 3]),
//             LeafOp::Rotation => FunctionType::new_df([Type::Qubit], [Type::Quat64]),
//             LeafOp::ToRotation => FunctionType::new_df(
//                 [
//                     Type::Angle,
//                     Type::F64,
//                     Type::F64,
//                     Type::F64,
//                 ],
//                 [Type::Quat64],
//             ),

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "pyo3", pyclass(name = "Rational"))]
pub struct Rational(pub Rational64);

impl From<Rational64> for Rational {
    fn from(r: Rational64) -> Self {
        Self(r)
    }
}

// angle is contained value * pi in radians
#[derive(Clone, PartialEq, Debug, Copy, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "pyo3", derive(FromPyObject))]
pub enum AngleValue {
    F64(f64),
    Rational(Rational),
}

impl AngleValue {
    const PI: Self = AngleValue::Rational(Rational(Rational64::new_raw(1, 1)));
    fn binary_op<F: FnOnce(f64, f64) -> f64, G: FnOnce(Rational64, Rational64) -> Rational64>(
        self,
        rhs: Self,
        opf: F,
        opr: G,
    ) -> Self {
        match (self, rhs) {
            (AngleValue::F64(x), AngleValue::F64(y)) => AngleValue::F64(opf(x, y)),
            (AngleValue::F64(x), AngleValue::Rational(y))
            | (AngleValue::Rational(y), AngleValue::F64(x)) => {
                AngleValue::F64(opf(x, y.0.to_f64().unwrap()))
            }
            (AngleValue::Rational(x), AngleValue::Rational(y)) => {
                AngleValue::Rational(Rational(opr(x.0, y.0)))
            }
        }
    }

    fn unary_op<F: FnOnce(f64) -> f64, G: FnOnce(Rational64) -> Rational64>(
        self,
        opf: F,
        opr: G,
    ) -> Self {
        match self {
            AngleValue::F64(x) => AngleValue::F64(opf(x)),
            AngleValue::Rational(x) => AngleValue::Rational(Rational(opr(x.0))),
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            AngleValue::F64(x) => *x,
            AngleValue::Rational(x) => x.0.to_f64().expect("Floating point conversion error."),
        }
    }

    pub fn radians(&self) -> f64 {
        self.to_f64() * std::f64::consts::PI
    }
}

impl Add for AngleValue {
    type Output = AngleValue;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, |x, y| x + y, |x, y| x + y)
    }
}

impl Sub for AngleValue {
    type Output = AngleValue;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, |x, y| x - y, |x, y| x - y)
    }
}

impl Mul for AngleValue {
    type Output = AngleValue;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, |x, y| x * y, |x, y| x * y)
    }
}

impl Div for AngleValue {
    type Output = AngleValue;

    fn div(self, rhs: Self) -> Self::Output {
        self.binary_op(rhs, |x, y| x / y, |x, y| x / y)
    }
}

impl Neg for AngleValue {
    type Output = AngleValue;

    fn neg(self) -> Self::Output {
        self.unary_op(|x| -x, |x| -x)
    }
}

impl Add for &AngleValue {
    type Output = AngleValue;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary_op(*rhs, |x, y| x + y, |x, y| x + y)
    }
}

impl Sub for &AngleValue {
    type Output = AngleValue;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binary_op(*rhs, |x, y| x - y, |x, y| x - y)
    }
}

impl Mul for &AngleValue {
    type Output = AngleValue;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binary_op(*rhs, |x, y| x * y, |x, y| x * y)
    }
}

impl Div for &AngleValue {
    type Output = AngleValue;

    fn div(self, rhs: Self) -> Self::Output {
        self.binary_op(*rhs, |x, y| x / y, |x, y| x / y)
    }
}

impl Neg for &AngleValue {
    type Output = AngleValue;

    fn neg(self) -> Self::Output {
        self.unary_op(|x| -x, |x| -x)
    }
}

#[cfg(test)]
mod test {

    use rstest::{fixture, rstest};

    use super::{AngleValue, RotationValue, ANGLE_T, ANGLE_T_NAME, PI_NAME};
    use crate::{
        extension::SignatureError,
        types::{CustomType, Type, TypeBound},
        values::CustomConst,
        Extension,
    };

    #[fixture]
    fn extension() -> Extension {
        super::extension()
    }

    #[rstest]
    fn test_types(extension: Extension) {
        let angle = extension.get_type(ANGLE_T_NAME).unwrap();

        let custom = angle.instantiate_concrete([]).unwrap();

        angle.check_custom(&custom).unwrap();

        let false_custom = CustomType::new(
            custom.name().clone(),
            vec![],
            "wrong_extension",
            TypeBound::Copyable,
        );
        assert_eq!(
            angle.check_custom(&false_custom),
            Err(SignatureError::ExtensionMismatch(
                "rotations".into(),
                "wrong_extension".into(),
            ))
        );

        assert_eq!(Type::new_extension(custom), ANGLE_T);
    }

    #[rstest]
    fn test_type_check(extension: Extension) {
        let custom_type = extension
            .get_type(ANGLE_T_NAME)
            .unwrap()
            .instantiate_concrete([])
            .unwrap();

        let custom_value = RotationValue::Angle(AngleValue::F64(0.0));

        // correct type
        custom_value.check_custom_type(&custom_type).unwrap();

        let wrong_custom_type = extension
            .get_type("quat")
            .unwrap()
            .instantiate_concrete([])
            .unwrap();
        let res = custom_value.check_custom_type(&wrong_custom_type);
        assert!(res.is_err());
    }

    #[rstest]
    fn test_constant(extension: Extension) {
        let pi_val = extension.get_value(PI_NAME).unwrap();

        ANGLE_T.check_type(pi_val.typed_value().value()).unwrap();
    }
}
