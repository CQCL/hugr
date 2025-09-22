//! Conversions between integer and floating-point values.

use std::sync::{Arc, LazyLock, Weak};

use strum::{EnumIter, EnumString, IntoStaticStr};

use crate::Extension;
use crate::extension::prelude::sum_with_error;
use crate::extension::prelude::{bool_t, string_type, usize_t};
use crate::extension::simple_op::{HasConcrete, HasDef};
use crate::extension::simple_op::{MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError};
use crate::extension::{ExtensionId, OpDef, SignatureError, SignatureFunc};
use crate::ops::{ExtensionOp, OpName};
use crate::std_extensions::arithmetic::int_ops::int_polytype;
use crate::std_extensions::arithmetic::int_types::int_type;
use crate::types::{TypeArg, TypeRV};

use super::float_types::float64_type;
use super::int_types::{get_log_width, int_tv};
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
    itousize,
    ifromusize,
    bytecast_int64_to_float64,
    bytecast_float64_to_int64,
}

impl MakeOpDef for ConvertOpDef {
    fn opdef_id(&self) -> OpName {
        <&'static str>::from(self).into()
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
        use ConvertOpDef::*;
        match self {
            trunc_s | trunc_u => int_polytype(
                1,
                vec![float64_type()],
                TypeRV::from(sum_with_error(int_tv(0))),
            ),
            convert_s | convert_u => int_polytype(1, vec![int_tv(0)], vec![float64_type()]),
            itobool => int_polytype(0, vec![int_type(0)], vec![bool_t()]),
            ifrombool => int_polytype(0, vec![bool_t()], vec![int_type(0)]),
            itostring_u | itostring_s => int_polytype(1, vec![int_tv(0)], vec![string_type()]),
            itousize => int_polytype(0, vec![int_type(6)], vec![usize_t()]),
            ifromusize => int_polytype(0, vec![usize_t()], vec![int_type(6)]),
            bytecast_int64_to_float64 => int_polytype(0, vec![int_type(6)], vec![float64_type()]),
            bytecast_float64_to_int64 => int_polytype(0, vec![float64_type()], vec![int_type(6)]),
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
            itousize => "convert a 64b unsigned integer to its usize representation",
            ifromusize => "convert a usize to a 64b unsigned integer",
            bytecast_int64_to_float64 => {
                "reinterpret an int64 as a float64 based on its bytes, with the same endianness"
            }
            bytecast_float64_to_int64 => {
                "reinterpret an float64 as an int based on its bytes, with the same endianness"
            }
        }
        .to_string()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        const_fold::set_fold(self, def);
    }
}

impl ConvertOpDef {
    /// Initialize a [`ConvertOpType`] from a [`ConvertOpDef`] which requires no
    /// integer widths set.
    #[must_use]
    pub fn without_log_width(self) -> ConvertOpType {
        ConvertOpType {
            def: self,
            log_width: None,
        }
    }
    /// Initialize a [`ConvertOpType`] from a [`ConvertOpDef`] which requires one
    /// integer width set.
    #[must_use]
    pub fn with_log_width(self, log_width: u8) -> ConvertOpType {
        ConvertOpType {
            def: self,
            log_width: Some(log_width),
        }
    }
}
/// Concrete convert operation with integer log width set.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertOpType {
    /// The kind of conversion op.
    def: ConvertOpDef,
    /// The integer width parameter of the conversion op, if any. This is interpreted
    /// differently, depending on `def`. The integer types in the inputs and
    /// outputs of the op will have [`int_type`]s of this width.
    log_width: Option<u8>,
}

impl ConvertOpType {
    /// Returns the generic [`ConvertOpDef`] of this [`ConvertOpType`].
    #[must_use]
    pub fn def(&self) -> &ConvertOpDef {
        &self.def
    }

    /// Returns the integer width parameters of this [`ConvertOpType`], if any.
    #[must_use]
    pub fn log_widths(&self) -> &[u8] {
        self.log_width.as_slice()
    }
}

impl MakeExtensionOp for ConvertOpType {
    fn op_id(&self) -> OpName {
        self.def.opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError> {
        let def = ConvertOpDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        self.log_width
            .iter()
            .map(|&n| u64::from(n).into())
            .collect()
    }
}

/// Extension for conversions between integers and floats.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(|| {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        ConvertOpDef::load_all_ops(extension, extension_ref).unwrap();
    })
});

impl MakeRegisteredOp for ConvertOpType {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

impl HasConcrete for ConvertOpDef {
    type Concrete = ConvertOpType;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        let log_width = match type_args {
            [] => None,
            [arg] => Some(get_log_width(arg).map_err(|_| SignatureError::InvalidTypeArgs)?),
            _ => return Err(SignatureError::InvalidTypeArgs.into()),
        };
        Ok(ConvertOpType {
            def: *self,
            log_width,
        })
    }
}

impl HasDef for ConvertOpType {
    type Def = ConvertOpDef;
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::IncomingPort;
    use crate::extension::prelude::ConstUsize;
    use crate::ops::Value;
    use crate::std_extensions::arithmetic::int_types::ConstInt;

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

    #[rstest]
    #[case::itobool_false(ConvertOpDef::itobool.without_log_width(), &[ConstInt::new_u(0, 0).unwrap().into()], &[Value::false_val()])]
    #[case::itobool_true(ConvertOpDef::itobool.without_log_width(), &[ConstInt::new_u(0, 1).unwrap().into()], &[Value::true_val()])]
    #[case::ifrombool_false(ConvertOpDef::ifrombool.without_log_width(), &[Value::false_val()], &[ConstInt::new_u(0, 0).unwrap().into()])]
    #[case::ifrombool_true(ConvertOpDef::ifrombool.without_log_width(), &[Value::true_val()], &[ConstInt::new_u(0, 1).unwrap().into()])]
    #[case::itousize(ConvertOpDef::itousize.without_log_width(), &[ConstInt::new_u(6, 42).unwrap().into()], &[ConstUsize::new(42).into()])]
    #[case::ifromusize(ConvertOpDef::ifromusize.without_log_width(), &[ConstUsize::new(42).into()], &[ConstInt::new_u(6, 42).unwrap().into()])]
    fn convert_fold(
        #[case] op: ConvertOpType,
        #[case] inputs: &[Value],
        #[case] outputs: &[Value],
    ) {
        use crate::ops::Value;

        let consts: Vec<(IncomingPort, Value)> = inputs
            .iter()
            .enumerate()
            .map(|(i, v)| (i.into(), v.clone()))
            .collect();

        let res = op
            .to_extension_op()
            .unwrap()
            .constant_fold(&consts)
            .unwrap();

        for (i, expected) in outputs.iter().enumerate() {
            let res_val: &Value = &res.get(i).unwrap().1;

            assert_eq!(res_val, expected);
        }
    }
}
