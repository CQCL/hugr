//! Conversions between integer and floating-point values.

use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::ops::OpName;
use crate::{
    extension::{
        prelude::sum_with_error,
        simple_op::{MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError},
        ExtensionId, ExtensionRegistry, ExtensionSet, OpDef, SignatureError, SignatureFunc,
        PRELUDE,
    },
    ops::{custom::ExtensionOp, NamedOp},
    type_row,
    types::{FunTypeVarArgs, PolyFuncType, Type, TypeArg},
    Extension,
};

use super::int_types::int_tv;
use super::{float_types::FLOAT64_TYPE, int_types::LOG_WIDTH_TYPE_PARAM};
use lazy_static::lazy_static;
mod const_fold;
/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.conversions");

/// Extension for conversions between floats and integers.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs, non_camel_case_types)]
#[non_exhaustive]
pub enum ConvertOpDef {
    trunc_u,
    trunc_s,
    convert_u,
    convert_s,
}

impl MakeOpDef for ConvertOpDef {
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        crate::extension::simple_op::try_from_name(op_def.name())
    }

    fn signature(&self) -> SignatureFunc {
        use ConvertOpDef::*;
        PolyFuncType::new(
            vec![LOG_WIDTH_TYPE_PARAM],
            match self {
                trunc_s | trunc_u => FunTypeVarArgs::new(
                    type_row![FLOAT64_TYPE],
                    Type::<true>::from(sum_with_error(int_tv(0))),
                ),
                convert_s | convert_u => {
                    FunTypeVarArgs::new(vec![int_tv(0)], type_row![FLOAT64_TYPE])
                }
            },
        )
        .into()
    }

    fn description(&self) -> String {
        use ConvertOpDef::*;
        match self {
            trunc_u => "float to unsigned int",
            trunc_s => "float to signed int",
            convert_u => "unsigned int to float",
            convert_s => "signed int to float",
        }
        .to_string()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        const_fold::set_fold(self, def)
    }
}

impl ConvertOpDef {
    /// Initialise a conversion op with an integer log width type argument.
    pub fn with_log_width(self, log_width: u8) -> ConvertOpType {
        ConvertOpType {
            def: self,
            log_width,
        }
    }
}
/// Concrete convert operation with integer log width set.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertOpType {
    def: ConvertOpDef,
    log_width: u8,
}

impl NamedOp for ConvertOpType {
    fn name(&self) -> OpName {
        self.def.name()
    }
}

impl MakeExtensionOp for ConvertOpType {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError> {
        let def = ConvertOpDef::from_def(ext_op.def())?;
        let log_width: u64 = match *ext_op.args() {
            [TypeArg::BoundedNat { n }] => n,
            _ => return Err(SignatureError::InvalidTypeArgs.into()),
        };
        Ok(Self {
            def,
            log_width: u8::try_from(log_width).unwrap(),
        })
    }

    fn type_args(&self) -> Vec<crate::types::TypeArg> {
        vec![TypeArg::BoundedNat {
            n: self.log_width as u64,
        }]
    }
}

lazy_static! {
    /// Extension for conversions between integers and floats.
    pub static ref EXTENSION: Extension = {
        let mut extension = Extension::new_with_reqs(
            EXTENSION_ID,
            ExtensionSet::from_iter(vec![
                super::int_types::EXTENSION_ID,
                super::float_types::EXTENSION_ID,
            ]),
        );

        ConvertOpDef::load_all_ops(&mut extension).unwrap();

        extension
    };

    /// Registry of extensions required to validate integer operations.
    pub static ref CONVERT_OPS_REGISTRY: ExtensionRegistry  = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        super::int_types::EXTENSION.to_owned(),
        super::float_types::EXTENSION.to_owned(),
        EXTENSION.to_owned(),
    ])
    .unwrap();
}

impl MakeRegisteredOp for ConvertOpType {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &CONVERT_OPS_REGISTRY
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_conversions_extension() {
        let r = &EXTENSION;
        assert_eq!(r.name() as &str, "arithmetic.conversions");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.as_str().starts_with("convert") || name.as_str().starts_with("trunc"));
        }
    }
}
