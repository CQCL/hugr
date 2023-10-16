//! Basic HUGR quantum operations

use std::num::NonZeroU64;

use smol_str::SmolStr;

use crate::extension::prelude::{BOOL_T, ERROR_TYPE, QB_T};
use crate::extension::{ExtensionId, SignatureError};
use crate::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
use crate::type_row;
use crate::types::type_param::{TypeArg, TypeArgError, TypeParam};
use crate::types::{ConstTypeError, CustomCheckFailure, CustomType, FunctionType, Type, TypeBound};
use crate::utils::collect_array;
use crate::values::CustomConst;
use crate::Extension;

use lazy_static::lazy_static;

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("quantum");

/// Identifier for the angle type.
const ANGLE_TYPE_ID: SmolStr = SmolStr::new_inline("angle");

fn angle_custom_type(precision_arg: TypeArg) -> CustomType {
    CustomType::new(ANGLE_TYPE_ID, [precision_arg], EXTENSION_ID, TypeBound::Eq)
}

/// Angle type of a given precision (specified by the TypeArg).
///
/// This type is capable of representing angles that are multiples of 2π / 2^N where N is the
/// precision.
pub(super) fn angle_type(precision_arg: TypeArg) -> Type {
    Type::new_extension(angle_custom_type(precision_arg))
}

/// The smallest forbidden precision.
pub const PRECISION_BOUND: u8 = 54;

const fn is_valid_precision(n: u8) -> bool {
    n < PRECISION_BOUND
}

/// Type parameter for the precision of an angle.
// SAFETY: unsafe block should be ok as the value is definitely not zero.
#[allow(clippy::assertions_on_constants)]
pub const PRECISION_TYPE_PARAM: TypeParam = TypeParam::bounded_nat(unsafe {
    assert!(PRECISION_BOUND > 0);
    NonZeroU64::new_unchecked(PRECISION_BOUND as u64)
});

/// Get the precision of the specified type argument or error if the argument is invalid.
pub(super) fn get_precision(arg: &TypeArg) -> Result<u8, TypeArgError> {
    match arg {
        TypeArg::BoundedNat { n } if is_valid_precision(*n as u8) => Ok(*n as u8),
        _ => Err(TypeArgError::TypeMismatch {
            arg: arg.clone(),
            param: PRECISION_TYPE_PARAM,
        }),
    }
}

pub(super) const fn type_arg(precision: u8) -> TypeArg {
    TypeArg::BoundedNat {
        n: precision as u64,
    }
}

/// An angle
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConstAngle {
    precision: u8,
    value: u64,
}

impl ConstAngle {
    /// Create a new [`ConstAngle`]
    pub fn new(precision: u8, value: u64) -> Result<Self, ConstTypeError> {
        if !is_valid_precision(precision) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message("Invalid angle precision.".to_owned()),
            ));
        }
        if value >= (1u64 << precision) {
            return Err(ConstTypeError::CustomCheckFail(
                crate::types::CustomCheckFailure::Message(
                    "Invalid unsigned integer value.".to_owned(),
                ),
            ));
        }
        Ok(Self { precision, value })
    }

    /// Returns the value of the constant
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Returns the precision of the constant
    pub fn precision(&self) -> u8 {
        self.precision
    }
}

#[typetag::serde]
impl CustomConst for ConstAngle {
    fn name(&self) -> SmolStr {
        format!("a(2π*{}/{})", self.value, 1u64 << self.precision).into()
    }
    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        if typ.clone() == angle_custom_type(type_arg(self.precision)) {
            Ok(())
        } else {
            Err(CustomCheckFailure::Message(
                "Angle constant type mismatch.".into(),
            ))
        }
    }
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::values::downcast_equal_consts(self, other)
    }
}

fn atrunc_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let m: u8 = get_precision(arg0)?;
    let n: u8 = get_precision(arg1)?;
    if m < n {
        return Err(SignatureError::InvalidTypeArgs);
    }
    Ok(FunctionType::new(
        vec![angle_type(arg0.clone())],
        vec![Type::new_sum(vec![angle_type(arg1.clone()), ERROR_TYPE])],
    ))
}

fn awiden_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let m: u8 = get_precision(arg0)?;
    let n: u8 = get_precision(arg1)?;
    if m > n {
        return Err(SignatureError::InvalidTypeArgs);
    }
    Ok(FunctionType::new(
        vec![angle_type(arg0.clone())],
        vec![angle_type(arg1.clone())],
    ))
}

fn abinop_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg] = collect_array(arg_values);
    Ok(FunctionType::new(
        vec![angle_type(arg.clone()); 2],
        vec![angle_type(arg.clone())],
    ))
}

fn aunop_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg] = collect_array(arg_values);
    Ok(FunctionType::new(
        vec![angle_type(arg.clone())],
        vec![angle_type(arg.clone())],
    ))
}

fn one_qb_func(_: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    Ok(FunctionType::new(type_row![QB_T], type_row![QB_T]))
}

fn two_qb_func(_: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    Ok(FunctionType::new(
        type_row![QB_T, QB_T],
        type_row![QB_T, QB_T],
    ))
}

fn extension() -> Extension {
    let mut extension = Extension::new(EXTENSION_ID);

    extension
        .add_type(
            ANGLE_TYPE_ID,
            vec![PRECISION_TYPE_PARAM],
            "angle value with a given precision".to_owned(),
            TypeBound::Eq.into(),
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            "atrunc".into(),
            "truncate an angle to a lower-precision one with the same value, rounding down in \
            [0, 2π) if necessary"
                .to_owned(),
            vec![PRECISION_TYPE_PARAM, PRECISION_TYPE_PARAM],
            atrunc_sig,
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            "awiden".into(),
            "widen an angle to a higher-precision one with the same value".to_owned(),
            vec![PRECISION_TYPE_PARAM, PRECISION_TYPE_PARAM],
            awiden_sig,
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            "aadd".into(),
            "addition of angles".to_owned(),
            vec![PRECISION_TYPE_PARAM],
            abinop_sig,
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            "asub".into(),
            "subtraction of the second angle from the first".to_owned(),
            vec![PRECISION_TYPE_PARAM],
            abinop_sig,
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            "aneg".into(),
            "negation of an angle".to_owned(),
            vec![PRECISION_TYPE_PARAM],
            aunop_sig,
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            SmolStr::new_inline("H"),
            "Hadamard".into(),
            vec![],
            one_qb_func,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            SmolStr::new_inline("RzF64"),
            "Rotation specified by float".into(),
            vec![],
            |_: &[_]| {
                Ok(FunctionType::new(
                    type_row![QB_T, FLOAT64_TYPE],
                    type_row![QB_T],
                ))
            },
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(SmolStr::new_inline("CX"), "CX".into(), vec![], two_qb_func)
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            SmolStr::new_inline("Measure"),
            "Measure a qubit, returning the qubit and the measurement result.".into(),
            vec![],
            |_arg_values: &[TypeArg]| {
                Ok(FunctionType::new(type_row![QB_T], type_row![QB_T, BOOL_T]))
                // TODO add logic as an extension delta when inference is
                // done?
                // https://github.com/CQCL-DEV/hugr/issues/425
            },
        )
        .unwrap();

    extension
}

lazy_static! {
    /// Quantum extension definition.
    pub static ref EXTENSION: Extension = extension();
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{extension::EMPTY_REG, ops::LeafOp};

    use super::EXTENSION;

    fn get_gate(gate_name: &str) -> LeafOp {
        EXTENSION
            .instantiate_extension_op(gate_name, [], &EMPTY_REG)
            .unwrap()
            .into()
    }

    pub(crate) fn h_gate() -> LeafOp {
        get_gate("H")
    }

    pub(crate) fn cx_gate() -> LeafOp {
        get_gate("CX")
    }

    pub(crate) fn measure() -> LeafOp {
        get_gate("Measure")
    }
}
