//! Basic floating-point operations.

use std::sync::{Arc, LazyLock, Weak};

use strum::{EnumIter, EnumString, IntoStaticStr};

use super::float_types::float64_type;
use crate::{
    Extension,
    extension::{
        ExtensionId, OpDef, SignatureFunc,
        prelude::{bool_t, string_type},
        simple_op::{MakeOpDef, MakeRegisteredOp, OpLoadError},
    },
    ops::OpName,
    types::Signature,
};
mod const_fold;
/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.float");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

/// Integer extension operation definitions.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs, non_camel_case_types)]
#[non_exhaustive]
pub enum FloatOps {
    feq,
    fne,
    flt,
    fgt,
    fle,
    fge,
    fmax,
    fmin,
    fadd,
    fsub,
    fneg,
    fabs,
    fmul,
    fdiv,
    fpow,
    ffloor,
    fceil,
    fround,
    ftostring,
}

impl MakeOpDef for FloatOps {
    fn opdef_id(&self) -> OpName {
        <&Self as Into<&'static str>>::into(self).into()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        use FloatOps::*;

        match self {
            feq | fne | flt | fgt | fle | fge => {
                Signature::new(vec![float64_type(); 2], vec![bool_t()])
            }
            fmax | fmin | fadd | fsub | fmul | fdiv | fpow => {
                Signature::new(vec![float64_type(); 2], vec![float64_type()])
            }
            fneg | fabs | ffloor | fceil | fround => Signature::new_endo(vec![float64_type()]),
            ftostring => Signature::new(vec![float64_type()], string_type()),
        }
        .into()
    }

    fn description(&self) -> String {
        use FloatOps::*;
        match self {
            feq => "equality test",
            fne => "inequality test",
            flt => "\"less than\"",
            fgt => "\"greater than\"",
            fle => "\"less than or equal\"",
            fge => "\"greater than or equal\"",
            fmax => "maximum",
            fmin => "minimum",
            fadd => "addition",
            fsub => "subtraction",
            fneg => "negation",
            fabs => "absolute value",
            fmul => "multiplication",
            fdiv => "division",
            fpow => "exponentiation",
            ffloor => "floor",
            fceil => "ceiling",
            fround => "round",
            ftostring => "string representation",
        }
        .to_string()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        const_fold::set_fold(self, def);
    }
}

/// Extension for basic float operations.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(|| {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        FloatOps::load_all_ops(extension, extension_ref).unwrap();
    })
});

impl MakeRegisteredOp for FloatOps {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

#[cfg(test)]
mod test {
    use cgmath::AbsDiffEq;
    use rstest::rstest;

    use super::*;

    #[test]
    fn test_float_ops_extension() {
        let r = &EXTENSION;
        assert_eq!(r.name() as &str, "arithmetic.float");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.as_str().starts_with('f'));
        }
    }

    #[rstest]
    #[case::fadd(FloatOps::fadd, &[0.1, 0.2], &[0.30000000000000004])]
    #[case::fsub(FloatOps::fsub, &[1., 2.], &[-1.])]
    #[case::fmul(FloatOps::fmul, &[2., 3.], &[6.])]
    #[case::fdiv(FloatOps::fdiv, &[7., 2.], &[3.5])]
    #[case::fpow(FloatOps::fpow, &[0.5, 3.], &[0.125])]
    #[case::ffloor(FloatOps::ffloor, &[42.42], &[42.])]
    #[case::fceil(FloatOps::fceil, &[42.42], &[43.])]
    #[case::fround(FloatOps::fround, &[42.42], &[42.])]
    fn float_fold(#[case] op: FloatOps, #[case] inputs: &[f64], #[case] outputs: &[f64]) {
        use crate::ops::Value;
        use crate::std_extensions::arithmetic::float_types::ConstF64;

        let consts: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(i, &x)| (i.into(), Value::extension(ConstF64::new(x))))
            .collect();

        let res = op
            .to_extension_op()
            .unwrap()
            .constant_fold(&consts)
            .unwrap();

        for (i, expected) in outputs.iter().enumerate() {
            let res_val: f64 = res
                .get(i)
                .unwrap()
                .1
                .get_custom_value::<ConstF64>()
                .expect("This function assumes all incoming constants are floats.")
                .value();

            assert!(
                res_val.abs_diff_eq(expected, f64::EPSILON),
                "expected {expected:?}, got {res_val:?}"
            );
        }
    }
}
