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
    ffloor,
    fceil,
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
            fmax | fmin | fadd | fsub | fmul | fdiv => {
                Signature::new(type_row![FLOAT64_TYPE; 2], type_row![FLOAT64_TYPE])
            }
            fneg | fabs | ffloor | fceil => Signature::new_endo(type_row![FLOAT64_TYPE]),
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
            ffloor => "floor",
            fceil => "ceiling",
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
        let mut extension = Extension::new_with_reqs(
            EXTENSION_ID,
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
}
