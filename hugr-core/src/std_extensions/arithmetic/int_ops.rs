//! Basic integer operations.

use super::int_types::{get_log_width, int_tv, LOG_WIDTH_TYPE_PARAM};
use crate::extension::prelude::{sum_with_error, BOOL_T, STRING_TYPE};
use crate::extension::simple_op::{MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError};
use crate::extension::{
    CustomValidator, ExtensionRegistry, OpDef, SignatureFunc, ValidateJustArgs, PRELUDE,
};
use crate::ops::custom::ExtensionOp;
use crate::ops::{NamedOp, OpName};
use crate::std_extensions::arithmetic::int_types::int_type;
use crate::type_row;
use crate::types::{FuncValueType, PolyFuncTypeRV, TypeRowRV};
use crate::utils::collect_array;

use crate::{
    extension::{ExtensionId, ExtensionSet, SignatureError},
    types::{type_param::TypeArg, Type},
    Extension,
};

use lazy_static::lazy_static;
use strum_macros::{EnumIter, EnumString, IntoStaticStr};

mod const_fold;

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.int");

struct IOValidator {
    // whether the first type argument should be greater than or equal to the second
    f_ge_s: bool,
}

impl ValidateJustArgs for IOValidator {
    fn validate(&self, arg_values: &[TypeArg]) -> Result<(), SignatureError> {
        let [arg0, arg1] = collect_array(arg_values);
        let i: u8 = get_log_width(arg0)?;
        let o: u8 = get_log_width(arg1)?;
        let cmp = if self.f_ge_s { i >= o } else { i <= o };
        if !cmp {
            return Err(SignatureError::InvalidTypeArgs);
        }
        Ok(())
    }
}
/// Integer extension operation definitions.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs, non_camel_case_types)]
#[non_exhaustive]
pub enum IntOpDef {
    iwiden_u,
    iwiden_s,
    inarrow_u,
    inarrow_s,
    itobool,
    ifrombool,
    ieq,
    ine,
    ilt_u,
    ilt_s,
    igt_u,
    igt_s,
    ile_u,
    ile_s,
    ige_u,
    ige_s,
    imax_u,
    imax_s,
    imin_u,
    imin_s,
    iadd,
    isub,
    ineg,
    imul,
    idivmod_checked_u,
    idivmod_u,
    idivmod_checked_s,
    idivmod_s,
    idiv_checked_u,
    idiv_u,
    imod_checked_u,
    imod_u,
    idiv_checked_s,
    idiv_s,
    imod_checked_s,
    imod_s,
    iabs,
    iand,
    ior,
    ixor,
    inot,
    ishl,
    ishr,
    irotl,
    irotr,
    itostring_u,
    itostring_s,
}

impl MakeOpDef for IntOpDef {
    fn from_def(op_def: &OpDef) -> Result<Self, crate::extension::simple_op::OpLoadError> {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn signature(&self) -> SignatureFunc {
        use IntOpDef::*;
        match self {
            iwiden_s | iwiden_u => CustomValidator::new_with_validator(
                int_polytype(2, vec![int_tv(0)], vec![int_tv(1)]),
                IOValidator { f_ge_s: false },
            )
            .into(),
            inarrow_s | inarrow_u => CustomValidator::new_with_validator(
                int_polytype(2, int_tv(0), sum_ty_with_err(int_tv(1))),
                IOValidator { f_ge_s: true },
            )
            .into(),
            itobool => int_polytype(0, vec![int_type(0)], type_row![BOOL_T]).into(),
            ifrombool => int_polytype(0, type_row![BOOL_T], vec![int_type(0)]).into(),
            ieq | ine | ilt_u | ilt_s | igt_u | igt_s | ile_u | ile_s | ige_u | ige_s => {
                int_polytype(1, vec![int_tv(0); 2], type_row![BOOL_T]).into()
            }
            imax_u | imax_s | imin_u | imin_s | iadd | isub | imul | iand | ior | ixor => {
                ibinop_sig().into()
            }
            ineg | iabs | inot => iunop_sig().into(),
            //TODO inline
            idivmod_checked_u | idivmod_checked_s => {
                let intpair: TypeRowRV = vec![int_tv(0), int_tv(1)].into();
                int_polytype(
                    2,
                    intpair.clone(),
                    sum_ty_with_err(Type::new_tuple(intpair)),
                )
            }
            .into(),
            idivmod_u | idivmod_s => {
                let intpair: TypeRowRV = vec![int_tv(0), int_tv(1)].into();
                int_polytype(2, intpair.clone(), intpair.clone())
            }
            .into(),
            idiv_u | idiv_s => int_polytype(2, vec![int_tv(0), int_tv(1)], vec![int_tv(0)]).into(),
            idiv_checked_u | idiv_checked_s => {
                int_polytype(2, vec![int_tv(0), int_tv(1)], sum_ty_with_err(int_tv(0))).into()
            }
            imod_checked_u | imod_checked_s => int_polytype(
                2,
                vec![int_tv(0), int_tv(1).clone()],
                sum_ty_with_err(int_tv(1)),
            )
            .into(),
            imod_u | imod_s => {
                int_polytype(2, vec![int_tv(0), int_tv(1).clone()], vec![int_tv(1)]).into()
            }
            ishl | ishr | irotl | irotr => {
                int_polytype(2, vec![int_tv(0), int_tv(1)], vec![int_tv(0)]).into()
            }
            itostring_u | itostring_s => PolyFuncTypeRV::new(
                vec![LOG_WIDTH_TYPE_PARAM],
                FuncValueType::new(vec![int_tv(0)], vec![STRING_TYPE]),
            )
            .into(),
        }
    }

    fn description(&self) -> String {
        use IntOpDef::*;

        match self {
            iwiden_u => "widen an unsigned integer to a wider one with the same value",
            iwiden_s => "widen a signed integer to a wider one with the same value",
            inarrow_u => "narrow an unsigned integer to a narrower one with the same value if possible",
            inarrow_s => "narrow a signed integer to a narrower one with the same value if possible",
            itobool => "convert to bool (1 is true, 0 is false)",
            ifrombool => "convert from bool (1 is true, 0 is false)",
            ieq => "equality test",
            ine => "inequality test",
            ilt_u => "\"less than\" as unsigned integers",
            ilt_s => "\"less than\" as signed integers",
            igt_u =>"\"greater than\" as unsigned integers",
            igt_s => "\"greater than\" as signed integers",
            ile_u => "\"less than or equal\" as unsigned integers",
            ile_s => "\"less than or equal\" as signed integers",
            ige_u => "\"greater than or equal\" as unsigned integers",
            ige_s => "\"greater than or equal\" as signed integers",
            imax_u => "maximum of unsigned integers",
            imax_s => "maximum of signed integers",
            imin_u => "minimum of unsigned integers",
            imin_s => "minimum of signed integers",
            iadd => "addition modulo 2^N (signed and unsigned versions are the same op)",
            isub => "subtraction modulo 2^N (signed and unsigned versions are the same op)",
            ineg => "negation modulo 2^N (signed and unsigned versions are the same op)",
            imul => "multiplication modulo 2^N (signed and unsigned versions are the same op)",
            idivmod_checked_u => "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^M, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 is an error)",
            idivmod_u => "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^M, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 will call panic)",
            idivmod_checked_s => "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^M, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 is an error)",
            idivmod_s => "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^M, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 will call panic)",
            idiv_checked_u => "as idivmod_checked_u but discarding the second output",
            idiv_u => "as idivmod_u but discarding the second output",
            imod_checked_u => "as idivmod_checked_u but discarding the first output",
            imod_u => "as idivmod_u but discarding the first output",
            idiv_checked_s => "as idivmod_checked_s but discarding the second output",
            idiv_s => "as idivmod_s but discarding the second output",
            imod_checked_s => "as idivmod_checked_s but discarding the first output",
            imod_s => "as idivmod_s but discarding the first output",
            iabs => "convert signed to unsigned by taking absolute value",
            iand => "bitwise AND",
            ior => "bitwise OR",
            ixor => "bitwise XOR",
            inot => "bitwise NOT",
            ishl => "shift first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits dropped, rightmost bits set to zero",
            ishr => "shift first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits dropped, leftmost bits set to zero)",
            irotl => "rotate first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits replace rightmost bits)",
            irotr => "rotate first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits replace leftmost bits)",
            itostring_s => "convert a signed integer to its string representation",
            itostring_u => "convert an unsigned integer to its string representation",
        }.into()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        const_fold::set_fold(self, def)
    }
}
fn int_polytype(
    n_vars: usize,
    input: impl Into<TypeRowRV>,
    output: impl Into<TypeRowRV>,
) -> PolyFuncTypeRV {
    PolyFuncTypeRV::new(
        vec![LOG_WIDTH_TYPE_PARAM; n_vars],
        FuncValueType::new(input, output),
    )
}

fn ibinop_sig() -> PolyFuncTypeRV {
    let int_type_var = int_tv(0);

    int_polytype(1, vec![int_type_var.clone(); 2], vec![int_type_var])
}

fn iunop_sig() -> PolyFuncTypeRV {
    let int_type_var = int_tv(0);
    int_polytype(1, vec![int_type_var.clone()], vec![int_type_var])
}

lazy_static! {
    /// Extension for basic integer operations.
    pub static ref EXTENSION: Extension = {
        let mut extension = Extension::new_with_reqs(
            EXTENSION_ID,
            ExtensionSet::singleton(&super::int_types::EXTENSION_ID),
        );

        IntOpDef::load_all_ops(&mut extension).unwrap();

        extension
    };

    /// Registry of extensions required to validate integer operations.
    pub static ref INT_OPS_REGISTRY: ExtensionRegistry  = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        super::int_types::EXTENSION.to_owned(),
        EXTENSION.to_owned(),
    ])
    .unwrap();
}

/// Concrete integer operation with integer widths set.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ConcreteIntOp {
    /// The kind of int op.
    pub def: IntOpDef,
    /// The width parameters of the int op. These are interpreted differently,
    /// depending on `def`. The types of inputs and outputs of the op will have
    /// [int_type]s of these widths.
    pub log_widths: Vec<u8>,
}

impl NamedOp for ConcreteIntOp {
    fn name(&self) -> OpName {
        self.def.name()
    }
}
impl MakeExtensionOp for ConcreteIntOp {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError> {
        let def = IntOpDef::from_def(ext_op.def())?;
        let args = ext_op.args();
        let log_widths: Vec<u8> = args
            .iter()
            .map(|a| get_log_width(a).map_err(|_| SignatureError::InvalidTypeArgs))
            .collect::<Result<_, _>>()?;
        Ok(Self { def, log_widths })
    }

    fn type_args(&self) -> Vec<TypeArg> {
        self.log_widths.iter().map(|&n| (n as u64).into()).collect()
    }
}

impl MakeRegisteredOp for ConcreteIntOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &INT_OPS_REGISTRY
    }
}

impl IntOpDef {
    /// Initialize a [ConcreteIntOp] from a [IntOpDef] which requires no
    /// integer widths set.
    pub fn without_log_width(self) -> ConcreteIntOp {
        ConcreteIntOp {
            def: self,
            log_widths: vec![],
        }
    }
    /// Initialize a [ConcreteIntOp] from a [IntOpDef] which requires one
    /// integer width set.
    pub fn with_log_width(self, log_width: u8) -> ConcreteIntOp {
        ConcreteIntOp {
            def: self,
            log_widths: vec![log_width],
        }
    }
    /// Initialize a [ConcreteIntOp] from a [IntOpDef] which requires two
    /// integer widths set.
    pub fn with_two_log_widths(self, first_log_width: u8, second_log_width: u8) -> ConcreteIntOp {
        ConcreteIntOp {
            def: self,
            log_widths: vec![first_log_width, second_log_width],
        }
    }
}

fn sum_ty_with_err(t: Type) -> Type {
    sum_with_error(t).into()
}

#[cfg(test)]
mod test {
    use crate::{
        ops::dataflow::DataflowOpTrait, std_extensions::arithmetic::int_types::int_type,
        types::FunctionType,
    };

    use super::*;

    #[test]
    fn test_int_ops_extension() {
        assert_eq!(EXTENSION.name() as &str, "arithmetic.int");
        assert_eq!(EXTENSION.types().count(), 0);
        assert_eq!(EXTENSION.operations().count(), 47);
        for (name, _) in EXTENSION.operations() {
            assert!(name.starts_with('i'));
        }
    }

    #[test]
    fn test_binary_signatures() {
        assert_eq!(
            IntOpDef::iwiden_s
                .with_two_log_widths(3, 4)
                .to_extension_op()
                .unwrap()
                .signature(),
            FunctionType::new(int_type(3), int_type(4)).with_extension_delta(EXTENSION_ID)
        );
        assert_eq!(
            IntOpDef::iwiden_s
                .with_two_log_widths(3, 3)
                .to_extension_op()
                .unwrap()
                .signature(),
            FunctionType::new_endo(int_type(3)).with_extension_delta(EXTENSION_ID)
        );
        assert_eq!(
            IntOpDef::inarrow_s
                .with_two_log_widths(3, 3)
                .to_extension_op()
                .unwrap()
                .signature(),
            FunctionType::new(int_type(3), sum_ty_with_err(int_type(3)))
                .with_extension_delta(EXTENSION_ID)
        );
        assert!(
            IntOpDef::iwiden_u
                .with_two_log_widths(4, 3)
                .to_extension_op()
                .is_none(),
            "type arguments invalid"
        );

        assert_eq!(
            IntOpDef::inarrow_s
                .with_two_log_widths(2, 1)
                .to_extension_op()
                .unwrap()
                .signature(),
            FunctionType::new(int_type(2), sum_ty_with_err(int_type(1)))
                .with_extension_delta(EXTENSION_ID)
        );

        assert!(IntOpDef::inarrow_u
            .with_two_log_widths(1, 2)
            .to_extension_op()
            .is_none());
    }

    #[test]
    fn test_conversions() {
        let o = IntOpDef::itobool.without_log_width();
        assert!(
            IntOpDef::itobool
                .with_two_log_widths(1, 2)
                .to_extension_op()
                .is_none(),
            "type arguments invalid"
        );
        let ext_op = o.clone().to_extension_op().unwrap();

        assert_eq!(ConcreteIntOp::from_extension_op(&ext_op).unwrap(), o);
    }
}
