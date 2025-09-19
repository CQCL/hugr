//! Basic integer operations.

use std::sync::{Arc, LazyLock, Weak};

use super::int_types::{LOG_WIDTH_TYPE_PARAM, get_log_width, int_tv};
use crate::extension::prelude::{bool_t, sum_with_error};
use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{CustomValidator, OpDef, SignatureFunc, ValidateJustArgs};
use crate::ops::OpName;
use crate::ops::custom::ExtensionOp;
use crate::types::{FuncValueType, PolyFuncTypeRV, TypeRowRV};
use crate::utils::collect_array;

use crate::{
    Extension,
    extension::{ExtensionId, SignatureError},
    types::{Type, type_param::TypeArg},
};

use strum::{EnumIter, EnumString, IntoStaticStr};

mod const_fold;

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.int");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

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
    ipow,
    iabs,
    iand,
    ior,
    ixor,
    inot,
    ishl,
    ishr,
    irotl,
    irotr,
    iu_to_s,
    is_to_u,
}

impl MakeOpDef for IntOpDef {
    fn opdef_id(&self) -> OpName {
        <&Self as Into<&'static str>>::into(self).into()
    }
    fn from_def(op_def: &OpDef) -> Result<Self, crate::extension::simple_op::OpLoadError> {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        use IntOpDef::*;
        let tv0 = int_tv(0);
        match self {
            iwiden_s | iwiden_u => CustomValidator::new(
                int_polytype(2, vec![tv0], vec![int_tv(1)]),
                IOValidator { f_ge_s: false },
            )
            .into(),
            inarrow_s | inarrow_u => CustomValidator::new(
                int_polytype(2, tv0, sum_ty_with_err(int_tv(1))),
                IOValidator { f_ge_s: true },
            )
            .into(),
            ieq | ine | ilt_u | ilt_s | igt_u | igt_s | ile_u | ile_s | ige_u | ige_s => {
                int_polytype(1, vec![tv0; 2], vec![bool_t()]).into()
            }
            imax_u | imax_s | imin_u | imin_s | iadd | isub | imul | iand | ior | ixor | ipow => {
                ibinop_sig().into()
            }
            ineg | iabs | inot | iu_to_s | is_to_u => iunop_sig().into(),
            idivmod_checked_u | idivmod_checked_s => {
                let intpair: TypeRowRV = vec![tv0; 2].into();
                int_polytype(
                    1,
                    intpair.clone(),
                    sum_ty_with_err(Type::new_tuple(intpair)),
                )
            }
            .into(),
            idivmod_u | idivmod_s => {
                let intpair: TypeRowRV = vec![tv0; 2].into();
                int_polytype(1, intpair.clone(), intpair.clone())
            }
            .into(),
            idiv_u | idiv_s => int_polytype(1, vec![tv0.clone(); 2], vec![tv0]).into(),
            idiv_checked_u | idiv_checked_s => {
                int_polytype(1, vec![tv0.clone(); 2], sum_ty_with_err(tv0)).into()
            }
            imod_checked_u | imod_checked_s => {
                int_polytype(1, vec![tv0.clone(); 2], sum_ty_with_err(tv0)).into()
            }
            imod_u | imod_s => int_polytype(1, vec![tv0.clone(); 2], vec![tv0]).into(),
            ishl | ishr | irotl | irotr => int_polytype(1, vec![tv0.clone(); 2], vec![tv0]).into(),
        }
    }

    fn description(&self) -> String {
        use IntOpDef::*;

        match self {
            iwiden_u => "widen an unsigned integer to a wider one with the same value",
            iwiden_s => "widen a signed integer to a wider one with the same value",
            inarrow_u => "narrow an unsigned integer to a narrower one with the same value if possible",
            inarrow_s => "narrow a signed integer to a narrower one with the same value if possible",
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
            idivmod_checked_u => "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^N, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 is an error)",
            idivmod_u => "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^N, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 will call panic)",
            idivmod_checked_s => "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^N, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 is an error)",
            idivmod_s => "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^N, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 will call panic)",
            idiv_checked_u => "as idivmod_checked_u but discarding the second output",
            idiv_u => "as idivmod_u but discarding the second output",
            imod_checked_u => "as idivmod_checked_u but discarding the first output",
            imod_u => "as idivmod_u but discarding the first output",
            idiv_checked_s => "as idivmod_checked_s but discarding the second output",
            idiv_s => "as idivmod_s but discarding the second output",
            imod_checked_s => "as idivmod_checked_s but discarding the first output",
            imod_s => "as idivmod_s but discarding the first output",
            ipow => "raise first input to the power of second input, the exponent is treated as an unsigned integer",
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
            is_to_u => "convert signed to unsigned by taking absolute value",
            iu_to_s => "convert unsigned to signed by taking absolute value",
        }.into()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        const_fold::set_fold(self, def);
    }
}

/// Returns a polytype composed by a function type, and a number of integer width type parameters.
pub(in crate::std_extensions::arithmetic) fn int_polytype(
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

/// Extension for basic integer operations.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(|| {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        IntOpDef::load_all_ops(extension, extension_ref).unwrap();
    })
});

impl HasConcrete for IntOpDef {
    type Concrete = ConcreteIntOp;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        let log_widths: Vec<u8> = type_args
            .iter()
            .map(|a| get_log_width(a).map_err(|_| SignatureError::InvalidTypeArgs))
            .collect::<Result<_, _>>()?;
        Ok(ConcreteIntOp {
            def: *self,
            log_widths,
        })
    }
}

impl HasDef for ConcreteIntOp {
    type Def = IntOpDef;
}

/// Concrete integer operation with integer widths set.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ConcreteIntOp {
    /// The kind of int op.
    pub def: IntOpDef,
    /// The width parameters of the int op. These are interpreted differently,
    /// depending on `def`. The types of inputs and outputs of the op will have
    /// [`int_type`]s of these widths.
    ///
    /// [`int_type`]: crate::std_extensions::arithmetic::int_types::int_type
    pub log_widths: Vec<u8>,
}

impl MakeExtensionOp for ConcreteIntOp {
    fn op_id(&self) -> OpName {
        self.def.opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError> {
        let def = IntOpDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        self.log_widths
            .iter()
            .map(|&n| u64::from(n).into())
            .collect()
    }
}

impl MakeRegisteredOp for ConcreteIntOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

impl IntOpDef {
    /// Initialize a [`ConcreteIntOp`] from a [`IntOpDef`] which requires no
    /// integer widths set.
    #[must_use]
    pub fn without_log_width(self) -> ConcreteIntOp {
        ConcreteIntOp {
            def: self,
            log_widths: vec![],
        }
    }
    /// Initialize a [`ConcreteIntOp`] from a [`IntOpDef`] which requires one
    /// integer width set.
    #[must_use]
    pub fn with_log_width(self, log_width: u8) -> ConcreteIntOp {
        ConcreteIntOp {
            def: self,
            log_widths: vec![log_width],
        }
    }
    /// Initialize a [`ConcreteIntOp`] from a [`IntOpDef`] which requires two
    /// integer widths set.
    #[must_use]
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
    use rstest::rstest;

    use crate::{
        ops::dataflow::DataflowOpTrait, std_extensions::arithmetic::int_types::int_type,
        types::Signature,
    };

    use super::*;

    #[test]
    fn test_int_ops_extension() {
        assert_eq!(EXTENSION.name() as &str, "arithmetic.int");
        assert_eq!(EXTENSION.types().count(), 0);
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
                .signature()
                .as_ref(),
            &Signature::new(int_type(3), int_type(4))
        );
        assert_eq!(
            IntOpDef::iwiden_s
                .with_two_log_widths(3, 3)
                .to_extension_op()
                .unwrap()
                .signature()
                .as_ref(),
            &Signature::new_endo(int_type(3))
        );
        assert_eq!(
            IntOpDef::inarrow_s
                .with_two_log_widths(3, 3)
                .to_extension_op()
                .unwrap()
                .signature()
                .as_ref(),
            &Signature::new(int_type(3), sum_ty_with_err(int_type(3)))
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
                .signature()
                .as_ref(),
            &Signature::new(int_type(2), sum_ty_with_err(int_type(1)))
        );

        assert!(
            IntOpDef::inarrow_u
                .with_two_log_widths(1, 2)
                .to_extension_op()
                .is_none()
        );
    }

    #[rstest]
    #[case::iadd(IntOpDef::iadd.with_log_width(5), &[1, 2], &[3], 5)]
    #[case::isub(IntOpDef::isub.with_log_width(5), &[5, 2], &[3], 5)]
    #[case::imul(IntOpDef::imul.with_log_width(5), &[2, 8], &[16], 5)]
    #[case::idiv(IntOpDef::idiv_u.with_log_width(5), &[37, 8], &[4], 5)]
    #[case::imod(IntOpDef::imod_u.with_log_width(5), &[43, 8], &[3], 5)]
    #[case::ipow(IntOpDef::ipow.with_log_width(5), &[2, 8], &[256], 5)]
    #[case::iu_to_s(IntOpDef::iu_to_s.with_log_width(5), &[42], &[42], 5)]
    #[case::is_to_u(IntOpDef::is_to_u.with_log_width(5), &[42], &[42], 5)]
    #[should_panic(expected = "too large to be converted to signed")]
    #[case::iu_to_s_panic(IntOpDef::iu_to_s.with_log_width(5), &[u64::from(u32::MAX)], &[], 5)]
    #[should_panic(expected = "Cannot convert negative integer")]
    #[case::is_to_u_panic(IntOpDef::is_to_u.with_log_width(5), &[u64::from(0u32.wrapping_sub(42))], &[], 5)]
    fn int_fold(
        #[case] op: ConcreteIntOp,
        #[case] inputs: &[u64],
        #[case] outputs: &[u64],
        #[case] log_width: u8,
    ) {
        use crate::ops::Value;
        use crate::std_extensions::arithmetic::int_types::ConstInt;

        let consts: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                (
                    i.into(),
                    Value::extension(ConstInt::new_u(log_width, x).unwrap()),
                )
            })
            .collect();

        let res = op
            .to_extension_op()
            .unwrap()
            .constant_fold(&consts)
            .unwrap();

        for (i, &expected) in outputs.iter().enumerate() {
            let res_val: u64 = res
                .get(i)
                .unwrap()
                .1
                .get_custom_value::<ConstInt>()
                .expect("This function assumes all incoming constants are floats.")
                .value_u();

            assert_eq!(res_val, expected);
        }
    }
}
