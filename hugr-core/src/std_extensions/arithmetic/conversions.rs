//! Conversions between integer and floating-point values.

use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::extension::prelude::{BOOL_T, STRING_TYPE};
use crate::extension::simple_op::{HasConcrete, HasDef};
use crate::ops::OpName;
use crate::std_extensions::arithmetic::int_ops::int_polytype;
use crate::std_extensions::arithmetic::int_types::int_type;
use crate::{
    extension::{
        prelude::sum_with_error,
        simple_op::{MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError},
        ExtensionId, ExtensionRegistry, ExtensionSet, OpDef, SignatureError, SignatureFunc,
        PRELUDE,
    },
    ops::{custom::ExtensionOp, NamedOp},
    type_row,
    types::{TypeArg, TypeRV},
    Extension,
};

use super::float_types::FLOAT64_TYPE;
use super::int_types::{get_log_width, int_tv};
use lazy_static::lazy_static;
mod const_fold;
/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.conversions");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

/// Extension for conversions between floats and integers.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs, non_camel_case_types)]
#[non_exhaustive]
pub enum ConvertOpDef {
    trunc_u,
    trunc_s,
    convert_u,
    convert_s,
    itobool,
    ifrombool,
    itostring_u,
    itostring_s,
}

impl MakeOpDef for ConvertOpDef {
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn signature(&self) -> SignatureFunc {
        use ConvertOpDef::*;
        match self {
            trunc_s | trunc_u => int_polytype(
                1,
                type_row![FLOAT64_TYPE],
                TypeRV::from(sum_with_error(int_tv(0))),
            ),
            convert_s | convert_u => int_polytype(1, vec![int_tv(0)], type_row![FLOAT64_TYPE]),
            itobool => int_polytype(0, vec![int_type(0)], vec![BOOL_T]),
            ifrombool => int_polytype(0, vec![BOOL_T], vec![int_type(0)]),
            itostring_u | itostring_s => int_polytype(1, vec![int_tv(0)], vec![STRING_TYPE]),
        }
        .into()
    }

    fn description(&self) -> String {
        use ConvertOpDef::*;
        match self {
            trunc_u => "float to unsigned int",
            trunc_s => "float to signed int",
            convert_u => "unsigned int to float",
            convert_s => "signed int to float",
            itobool => "convert a 1-bit integer to bool (1 is true, 0 is false)",
            ifrombool => "convert from bool into a 1-bit integer (1 is true, 0 is false)",
            itostring_s => "convert a signed integer to its string representation",
            itostring_u => "convert an unsigned integer to its string representation",
        }
        .to_string()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        const_fold::set_fold(self, def)
    }
}

impl ConvertOpDef {
    /// Initialize a [ConvertOpType] from a [ConvertOpDef] which requires no
    /// integer widths set.
    pub fn without_log_width(self) -> ConvertOpType {
        ConvertOpType {
            def: self,
            log_widths: vec![],
        }
    }
    /// Initialize a [ConvertOpType] from a [ConvertOpDef] which requires one
    /// integer width set.
    pub fn with_log_width(self, log_width: u8) -> ConvertOpType {
        ConvertOpType {
            def: self,
            log_widths: vec![log_width],
        }
    }
}
/// Concrete convert operation with integer log width set.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertOpType {
    /// The kind of conversion op.
    pub def: ConvertOpDef,
    /// The integer width parameters of the conversion op. These are interpreted
    /// differently, depending on `def`. The integer types in the inputs and
    /// outputs of the op will have [int_type]s of these widths.
    pub log_widths: Vec<u8>,
}

impl NamedOp for ConvertOpType {
    fn name(&self) -> OpName {
        self.def.name()
    }
}

impl MakeExtensionOp for ConvertOpType {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError> {
        let def = ConvertOpDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        self.log_widths.iter().map(|&n| (n as u64).into()).collect()
    }
}

lazy_static! {
    /// Extension for conversions between integers and floats.
    pub static ref EXTENSION: Extension = {
        let mut extension = Extension::new(
            EXTENSION_ID,
            VERSION).with_reqs(
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

impl HasConcrete for ConvertOpDef {
    type Concrete = ConvertOpType;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        let log_widths: Vec<u8> = type_args
            .iter()
            .map(|a| get_log_width(a).map_err(|_| SignatureError::InvalidTypeArgs))
            .collect::<Result<_, _>>()?;
        Ok(ConvertOpType {
            def: *self,
            log_widths,
        })
    }
}

impl HasDef for ConvertOpType {
    type Def = ConvertOpDef;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_conversions_extension() {
        let r = &EXTENSION;
        assert_eq!(r.name() as &str, "arithmetic.conversions");
        assert_eq!(r.types().count(), 0);
    }

    #[test]
    fn test_conversions() {
        // Initialization with an invalid number of type arguments should fail.
        assert!(
            ConvertOpDef::itobool
                .with_log_width(1)
                .to_extension_op()
                .is_none(),
            "type arguments invalid"
        );

        // This should work
        let o = ConvertOpDef::itobool.without_log_width();
        let ext_op: ExtensionOp = o.clone().to_extension_op().unwrap();

        assert_eq!(ConvertOpType::from_op(&ext_op).unwrap(), o);
        assert_eq!(
            ConvertOpDef::from_op(&ext_op).unwrap(),
            ConvertOpDef::itobool
        );
    }
}
