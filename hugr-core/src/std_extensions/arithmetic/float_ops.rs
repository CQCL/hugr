//! Basic floating-point operations.

use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use super::float_types::FLOAT64_TYPE;
use crate::{
    extension::{
        prelude::{BOOL_T, STRING_TYPE},
        simple_op::{MakeOpDef, MakeRegisteredOp, OpLoadError},
        ExtensionId, ExtensionRegistry, ExtensionSet, OpDef, SignatureFunc, PRELUDE,
    },
    type_row,
    types::Signature,
    Extension,
};
use lazy_static::lazy_static;
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
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn signature(&self) -> SignatureFunc {
        use FloatOps::*;

        match self {
            feq | fne | flt | fgt | fle | fge => {
                Signature::new(type_row![FLOAT64_TYPE; 2], type_row![BOOL_T])
            }
            fmax | fmin | fadd | fsub | fmul | fdiv | fpow => {
                Signature::new(type_row![FLOAT64_TYPE; 2], type_row![FLOAT64_TYPE])
            }
            fneg | fabs | ffloor | fceil | fround => Signature::new_endo(type_row![FLOAT64_TYPE]),
            ftostring => Signature::new(type_row![FLOAT64_TYPE], STRING_TYPE),
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
        const_fold::set_fold(self, def)
    }
}

lazy_static! {
    /// Extension for basic float operations.
    pub static ref EXTENSION: Extension = {
        let mut extension = Extension::new(
            EXTENSION_ID,
            VERSION).with_reqs(
            ExtensionSet::singleton(&super::int_types::EXTENSION_ID),
        );

        FloatOps::load_all_ops(&mut extension).unwrap();

        extension
    };

    /// Registry of extensions required to validate float operations.
    pub static ref FLOAT_OPS_REGISTRY: ExtensionRegistry  = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        super::float_types::EXTENSION.to_owned(),
        EXTENSION.to_owned(),
    ])
    .unwrap();
}

impl MakeRegisteredOp for FloatOps {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &FLOAT_OPS_REGISTRY
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
                "expected {:?}, got {:?}",
                expected,
                res_val
            );
        }
    }
}
