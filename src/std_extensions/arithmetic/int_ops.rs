//! Basic integer operations.

use super::int_types::{get_log_width, int_type, int_type_var, LOG_WIDTH_TYPE_PARAM};
use crate::extension::prelude::{sum_with_error, BOOL_T};
use crate::extension::{ExtensionRegistry, PRELUDE};
use crate::type_row;
use crate::types::{FunctionType, PolyFuncType};
use crate::utils::collect_array;
use crate::{
    extension::{ExtensionId, ExtensionSet, SignatureError},
    types::{type_param::TypeArg, Type, TypeRow},
    Extension,
};

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.int");

fn iwiden_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let m: u8 = get_log_width(arg0)?;
    let n: u8 = get_log_width(arg1)?;
    if m > n {
        return Err(SignatureError::InvalidTypeArgs);
    }
    Ok(FunctionType::new(
        vec![int_type(arg0.clone())],
        vec![int_type(arg1.clone())],
    ))
}

fn inarrow_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let m: u8 = get_log_width(arg0)?;
    let n: u8 = get_log_width(arg1)?;
    if m < n {
        return Err(SignatureError::InvalidTypeArgs);
    }
    Ok(FunctionType::new(
        vec![int_type(arg0.clone())],
        vec![sum_with_error(int_type(arg1.clone()))],
    ))
}

fn int_polytype(
    n_vars: usize,
    input: impl Into<TypeRow>,
    output: impl Into<TypeRow>,
    temp_reg: &ExtensionRegistry,
) -> Result<PolyFuncType, SignatureError> {
    PolyFuncType::new_validated(
        vec![LOG_WIDTH_TYPE_PARAM; n_vars],
        FunctionType::new(input, output),
        temp_reg,
    )
}

fn itob_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    int_polytype(1, vec![int_type_var(0)], type_row![BOOL_T], temp_reg)
}

fn btoi_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    int_polytype(1, type_row![BOOL_T], vec![int_type_var(0)], temp_reg)
}

fn icmp_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    int_polytype(1, vec![int_type_var(0); 2], type_row![BOOL_T], temp_reg)
}

fn ibinop_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    let int_type_var = int_type_var(0);

    int_polytype(
        1,
        vec![int_type_var.clone(); 2],
        vec![int_type_var],
        temp_reg,
    )
}

fn iunop_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    let int_type_var = int_type_var(0);
    int_polytype(1, vec![int_type_var.clone()], vec![int_type_var], temp_reg)
}

fn idivmod_checked_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    let intpair: TypeRow = vec![int_type_var(0), int_type_var(1)].into();
    int_polytype(
        2,
        intpair.clone(),
        vec![sum_with_error(Type::new_tuple(intpair))],
        temp_reg,
    )
}

fn idivmod_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    let intpair: TypeRow = vec![int_type_var(0), int_type_var(1)].into();
    int_polytype(2, intpair.clone(), vec![Type::new_tuple(intpair)], temp_reg)
}

fn idiv_checked_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    int_polytype(
        2,
        vec![int_type_var(1)],
        vec![sum_with_error(int_type_var(0))],
        temp_reg,
    )
}

fn idiv_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    int_polytype(2, vec![int_type_var(1)], vec![int_type_var(0)], temp_reg)
}

fn imod_checked_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    int_polytype(
        2,
        vec![int_type_var(0), int_type_var(1).clone()],
        vec![sum_with_error(int_type_var(1))],
        temp_reg,
    )
}

fn imod_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    int_polytype(
        2,
        vec![int_type_var(0), int_type_var(1).clone()],
        vec![int_type_var(1)],
        temp_reg,
    )
}

fn ish_sig(temp_reg: &ExtensionRegistry) -> Result<PolyFuncType, SignatureError> {
    int_polytype(2, vec![int_type_var(1)], vec![int_type_var(0)], temp_reg)
}

/// Extension for basic integer operations.
pub fn extension() -> Extension {
    let mut extension = Extension::new_with_reqs(
        EXTENSION_ID,
        ExtensionSet::singleton(&super::int_types::EXTENSION_ID),
    );

    let temp_reg: ExtensionRegistry =
        [super::int_types::EXTENSION.to_owned(), PRELUDE.to_owned()].into();

    extension
        .add_op_custom_sig_simple(
            "iwiden_u".into(),
            "widen an unsigned integer to a wider one with the same value".to_owned(),
            vec![LOG_WIDTH_TYPE_PARAM, LOG_WIDTH_TYPE_PARAM],
            iwiden_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "iwiden_s".into(),
            "widen a signed integer to a wider one with the same value".to_owned(),
            vec![LOG_WIDTH_TYPE_PARAM, LOG_WIDTH_TYPE_PARAM],
            iwiden_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "inarrow_u".into(),
            "narrow an unsigned integer to a narrower one with the same value if possible"
                .to_owned(),
            vec![LOG_WIDTH_TYPE_PARAM, LOG_WIDTH_TYPE_PARAM],
            inarrow_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "inarrow_s".into(),
            "narrow a signed integer to a narrower one with the same value if possible".to_owned(),
            vec![LOG_WIDTH_TYPE_PARAM, LOG_WIDTH_TYPE_PARAM],
            inarrow_sig,
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "itobool".into(),
            "convert to bool (1 is true, 0 is false)".to_owned(),
            itob_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ifrombool".into(),
            "convert from bool (1 is true, 0 is false)".to_owned(),
            btoi_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ieq".into(),
            "equality test".to_owned(),
            icmp_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ine".into(),
            "inequality test".to_owned(),
            icmp_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ilt_u".into(),
            "\"less than\" as unsigned integers".to_owned(),
            icmp_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ilt_s".into(),
            "\"less than\" as signed integers".to_owned(),
            icmp_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "igt_u".into(),
            "\"greater than\" as unsigned integers".to_owned(),
            icmp_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "igt_s".into(),
            "\"greater than\" as signed integers".to_owned(),
            icmp_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ile_u".into(),
            "\"less than or equal\" as unsigned integers".to_owned(),
            icmp_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ile_s".into(),
            "\"less than or equal\" as signed integers".to_owned(),
            icmp_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ige_u".into(),
            "\"greater than or equal\" as unsigned integers".to_owned(),
            icmp_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ige_s".into(),
            "\"greater than or equal\" as signed integers".to_owned(),
            icmp_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "imax_u".into(),
            "maximum of unsigned integers".to_owned(),
            ibinop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "imax_s".into(),
            "maximum of signed integers".to_owned(),
            ibinop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "imin_u".into(),
            "minimum of unsigned integers".to_owned(),
            ibinop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "imin_s".into(),
            "minimum of signed integers".to_owned(),
            ibinop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "iadd".into(),
            "addition modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            ibinop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "isub".into(),
            "subtraction modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            ibinop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ineg".into(),
            "negation modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            iunop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "imul".into(),
            "multiplication modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            ibinop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "idivmod_checked_u".into(),
            "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^M, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 is an error)"
                .to_owned(),
            idivmod_checked_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "idivmod_u".into(),
            "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^M, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 will call panic)"
                .to_owned(),
            idivmod_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "idivmod_checked_s".into(),
            "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^M, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 is an error)"
                .to_owned(),
            idivmod_checked_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "idivmod_s".into(),
            "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^M, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 will call panic)"
                .to_owned(),
            idivmod_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "idiv_checked_u".into(),
            "as idivmod_checked_u but discarding the second output".to_owned(),
            idiv_checked_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "idiv_u".into(),
            "as idivmod_u but discarding the second output".to_owned(),
            idiv_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "imod_checked_u".into(),
            "as idivmod_checked_u but discarding the first output".to_owned(),
            imod_checked_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "imod_u".into(),
            "as idivmod_u but discarding the first output".to_owned(),
            imod_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "idiv_checked_s".into(),
            "as idivmod_checked_s but discarding the second output".to_owned(),
            idiv_checked_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "idiv_s".into(),
            "as idivmod_s but discarding the second output".to_owned(),
            idiv_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "imod_checked_s".into(),
            "as idivmod_checked_s but discarding the first output".to_owned(),
            imod_checked_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "imod_s".into(),
            "as idivmod_s but discarding the first output".to_owned(),
            imod_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "iabs".into(),
            "convert signed to unsigned by taking absolute value".to_owned(),
            iunop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "iand".into(),
            "bitwise AND".to_owned(),
            ibinop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ior".into(),
            "bitwise OR".to_owned(),
            ibinop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ixor".into(),
            "bitwise XOR".to_owned(),
            ibinop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "inot".into(),
            "bitwise NOT".to_owned(),
            iunop_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ishl".into(),
            "shift first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits dropped, rightmost bits set to zero"
                .to_owned(),
            ish_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "ishr".into(),
            "shift first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits dropped, leftmost bits set to zero)"
                .to_owned(),
            ish_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "irotl".into(),
            "rotate first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits replace rightmost bits)"
                .to_owned(),
            ish_sig(&temp_reg).unwrap(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "irotr".into(),
            "rotate first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits replace leftmost bits)"
                .to_owned(),
            ish_sig(  &temp_reg).unwrap(),
        )
        .unwrap();

    extension
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_int_ops_extension() {
        let r = extension();
        assert_eq!(r.name() as &str, "arithmetic.int");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.starts_with('i'));
        }
    }

    const fn ta(n: u64) -> TypeArg {
        TypeArg::BoundedNat { n }
    }
    #[test]
    fn test_binary_signatures() {
        let sig = iwiden_sig(&[ta(3), ta(4)]).unwrap();
        assert_eq!(
            sig,
            FunctionType::new(vec![int_type(ta(3))], vec![int_type(ta(4))],)
        );

        iwiden_sig(&[ta(4), ta(3)]).unwrap_err();

        let sig = inarrow_sig(&[ta(2), ta(1)]).unwrap();
        assert_eq!(
            sig,
            FunctionType::new(vec![int_type(ta(2))], vec![sum_with_error(int_type(ta(1)))],)
        );

        inarrow_sig(&[ta(1), ta(2)]).unwrap_err();
    }
}
