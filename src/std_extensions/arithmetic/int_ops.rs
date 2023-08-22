//! Basic integer operations.

use smol_str::SmolStr;

use super::int_types::{get_width, int_type};
use crate::extension::prelude::{BOOL_T, ERROR_TYPE};
use crate::type_row;
use crate::types::type_param::TypeParam;
use crate::types::FunctionType;
use crate::utils::collect_array;
use crate::{
    extension::{ExtensionSet, SignatureError},
    types::{type_param::TypeArg, Type, TypeRow},
    Extension,
};

/// The extension identifier.
pub const EXTENSION_ID: SmolStr = SmolStr::new_inline("arithmetic.int");

fn iwiden_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let m: u8 = get_width(arg0)?;
    let n: u8 = get_width(arg1)?;
    if m > n {
        return Err(SignatureError::InvalidTypeArgs);
    }
    Ok(FunctionType::new(vec![int_type(m)], vec![int_type(n)]))
}

fn inarrow_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let m: u8 = get_width(arg0)?;
    let n: u8 = get_width(arg1)?;
    if m < n {
        return Err(SignatureError::InvalidTypeArgs);
    }
    Ok(FunctionType::new(
        vec![int_type(m)],
        vec![Type::new_sum(vec![int_type(n), ERROR_TYPE])],
    ))
}

fn itob_sig(_arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    Ok(FunctionType::new(vec![int_type(1)], type_row![BOOL_T]))
}

fn btoi_sig(_arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    Ok(FunctionType::new(type_row![BOOL_T], vec![int_type(1)]))
}

fn icmp_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg] = collect_array(arg_values);
    let n: u8 = get_width(arg)?;
    Ok(FunctionType::new(vec![int_type(n); 2], type_row![BOOL_T]))
}

fn ibinop_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg] = collect_array(arg_values);
    let n: u8 = get_width(arg)?;
    Ok(FunctionType::new(vec![int_type(n); 2], vec![int_type(n)]))
}

fn iunop_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg] = collect_array(arg_values);
    let n: u8 = get_width(arg)?;
    Ok(FunctionType::new(vec![int_type(n)], vec![int_type(n)]))
}

fn idivmod_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    let intpair: TypeRow = vec![int_type(n), int_type(m)].into();
    Ok(FunctionType::new(
        intpair.clone(),
        vec![Type::new_sum(vec![Type::new_tuple(intpair), ERROR_TYPE])],
    ))
}

fn idiv_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    Ok(FunctionType::new(
        vec![int_type(n), int_type(m)],
        vec![Type::new_sum(vec![int_type(n), ERROR_TYPE])],
    ))
}

fn imod_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    Ok(FunctionType::new(
        vec![int_type(n), int_type(m)],
        vec![Type::new_sum(vec![int_type(m), ERROR_TYPE])],
    ))
}

fn ish_sig(arg_values: &[TypeArg]) -> Result<FunctionType, SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    Ok(FunctionType::new(
        vec![int_type(n), int_type(m)],
        vec![int_type(n)],
    ))
}

/// Extension for basic integer operations.
pub fn extension() -> Extension {
    let mut extension = Extension::new_with_reqs(
        EXTENSION_ID,
        ExtensionSet::singleton(&super::int_types::EXTENSION_ID),
    );

    extension
        .add_op_custom_sig_simple(
            "iwiden_u".into(),
            "widen an unsigned integer to a wider one with the same value".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            iwiden_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "iwiden_s".into(),
            "widen a signed integer to a wider one with the same value".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            iwiden_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "inarrow_u".into(),
            "narrow an unsigned integer to a narrower one with the same value if possible"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            inarrow_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "inarrow_s".into(),
            "narrow a signed integer to a narrower one with the same value if possible".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            inarrow_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "itobool".into(),
            "convert to bool (1 is true, 0 is false)".to_owned(),
            vec![],
            itob_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ifrombool".into(),
            "convert from bool (1 is true, 0 is false)".to_owned(),
            vec![],
            btoi_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ieq".into(),
            "equality test".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ine".into(),
            "inequality test".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ilt_u".into(),
            "\"less than\" as unsigned integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ilt_s".into(),
            "\"less than\" as signed integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "igt_u".into(),
            "\"greater than\" as unsigned integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "igt_s".into(),
            "\"greater than\" as signed integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ile_u".into(),
            "\"less than or equal\" as unsigned integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ile_s".into(),
            "\"less than or equal\" as signed integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ige_u".into(),
            "\"greater than or equal\" as unsigned integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ige_s".into(),
            "\"greater than or equal\" as signed integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "imax_u".into(),
            "maximum of unsigned integers".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "imax_s".into(),
            "maximum of signed integers".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "imin_u".into(),
            "minimum of unsigned integers".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "imin_s".into(),
            "minimum of signed integers".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "iadd".into(),
            "addition modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "isub".into(),
            "subtraction modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ineg".into(),
            "negation modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![TypeParam::USize],
            iunop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "imul".into(),
            "multiplication modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "idivmod_u".into(),
            "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^M, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 is an error)"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            idivmod_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "idivmod_s".into(),
            "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^M, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 is an error)"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            idivmod_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "idiv_u".into(),
            "as idivmod_u but discarding the second output".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            idiv_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "imod_u".into(),
            "as idivmod_u but discarding the first output".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            idiv_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "idiv_s".into(),
            "as idivmod_s but discarding the second output".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            imod_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "imod_s".into(),
            "as idivmod_s but discarding the first output".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            imod_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "iabs".into(),
            "convert signed to unsigned by taking absolute value".to_owned(),
            vec![TypeParam::USize],
            iunop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "iand".into(),
            "bitwise AND".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ior".into(),
            "bitwise OR".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ixor".into(),
            "bitwise XOR".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "inot".into(),
            "bitwise NOT".to_owned(),
            vec![TypeParam::USize],
            iunop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ishl".into(),
            "shift first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits dropped, rightmost bits set to zero"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            ish_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "ishr".into(),
            "shift first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits dropped, leftmost bits set to zero)"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            ish_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "irotl".into(),
            "rotate first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits replace rightmost bits)"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            ish_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "irotr".into(),
            "rotate first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits replace leftmost bits)"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            ish_sig,
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
        assert_eq!(r.name(), "arithmetic.int");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.starts_with('i'));
        }
    }
}
