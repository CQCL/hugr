#![allow(missing_docs)]
//! This is an experiment, it is probably already outdated.

use std::ops::{Add, Div, Mul, Neg, Sub};

use cgmath::num_traits::ToPrimitive;
use num_rational::Rational64;
use smol_str::SmolStr;
use std::collections::HashMap;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use crate::ops::constant::CustomConst;
use crate::resource::{CustomSignatureFunc, OpDef, ResourceSet, SignatureError, TypeDef};
use crate::types::{type_param::TypeArg, ClassicType, CustomType, SimpleType, TypeRow};
use crate::Resource;

pub const fn resource_id() -> SmolStr {
    SmolStr::new_inline("rotations")
}

/// The resource with all the operations and types defined in this extension.
pub fn resource() -> Resource {
    let mut resource = Resource::new(resource_id());

    resource.add_type(Type::Angle.type_def());
    resource.add_type(Type::Quaternion.type_def());

    resource
        .add_op(OpDef::new_with_custom_sig(
            "AngleAdd".into(),
            "".into(),
            vec![],
            HashMap::default(),
            AngleAdd,
        ))
        .unwrap();
    resource
}

/// Custom types defined by this extension.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Type {
    Angle,
    Quaternion,
}

impl Type {
    pub const fn name(&self) -> SmolStr {
        match self {
            Type::Angle => SmolStr::new_inline("angle"),
            Type::Quaternion => SmolStr::new_inline("quat"),
        }
    }

    pub fn custom_type(self) -> CustomType {
        CustomType::new(self.name(), [])
    }

    pub fn type_def(self) -> TypeDef {
        TypeDef {
            name: self.name(),
            args: vec![],
        }
    }
}

impl From<Type> for CustomType {
    fn from(ty: Type) -> Self {
        ty.custom_type()
    }
}

/// Constant values for [`Type`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Constant {
    Angle(AngleValue),
    Quaternion(cgmath::Quaternion<f64>),
}

#[typetag::serde]
impl CustomConst for Constant {
    fn name(&self) -> SmolStr {
        match self {
            Constant::Angle(val) => format!("AngleConstant({})", val.radians()),
            Constant::Quaternion(val) => format!("QuatConstant({:?})", val),
        }
        .into()
    }

    fn const_type(&self) -> ClassicType {
        let t: Type = match self {
            Constant::Angle(_) => Type::Angle,
            Constant::Quaternion(_) => Type::Quaternion,
        };
        t.custom_type().into()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AngleAdd;

/// When we have a YAML type-scheme interpreter, we'll be able to use that;
/// there is no need for a binary compute_signature for a case this simple.
impl CustomSignatureFunc for AngleAdd {
    fn compute_signature(
        &self,
        _name: &SmolStr,
        _arg_values: &[TypeArg],
        _misc: &HashMap<String, serde_yaml::Value>,
    ) -> Result<(TypeRow<SimpleType>, TypeRow<SimpleType>, ResourceSet), SignatureError> {
        let t: TypeRow<SimpleType> = vec![SimpleType::Classic(
            Into::<CustomType>::into(Type::Angle).into(),
        )]
        .into();
        Ok((t.clone(), t, ResourceSet::default()))
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
//             LeafOp::AngleAdd | LeafOp::AngleMul => Signature::new_linear([Type::Angle]),
//             LeafOp::QuatMul => Signature::new_linear([Type::Quat64]),
//             LeafOp::AngleNeg => Signature::new_linear([Type::Angle]),
//             LeafOp::RxF64 | LeafOp::RzF64 => {
//                 Signature::new_df([Type::Qubit], [Type::Angle])
//             }
//             LeafOp::TK1 => Signature::new_df(vec![Type::Qubit], vec![Type::Angle; 3]),
//             LeafOp::Rotation => Signature::new_df([Type::Qubit], [Type::Quat64]),
//             LeafOp::ToRotation => Signature::new_df(
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
