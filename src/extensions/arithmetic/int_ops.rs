//! Basic integer operations.

use smol_str::SmolStr;

use super::super::logic::bool_type;
use super::int_types::{get_width, int_type};
use crate::resource::prelude::ERROR_TYPE;
use crate::types::type_param::TypeParam;
use crate::utils::collect_array;
use crate::{
    resource::{ResourceSet, SignatureError},
    types::{type_param::TypeArg, Type, TypeRow},
    Resource,
};

/// The resource identifier.
pub const RESOURCE_ID: SmolStr = SmolStr::new_inline("arithmetic.int");

fn iwiden_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let m: u8 = get_width(arg0)?;
    let n: u8 = get_width(arg1)?;
    if m > n {
        return Err(SignatureError::InvalidTypeArgs);
    }
    Ok((
        vec![int_type(m)].into(),
        vec![int_type(n)].into(),
        ResourceSet::default(),
    ))
}

fn inarrow_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let m: u8 = get_width(arg0)?;
    let n: u8 = get_width(arg1)?;
    if m < n {
        return Err(SignatureError::InvalidTypeArgs);
    }
    Ok((
        vec![int_type(m)].into(),
        vec![Type::new_sum(vec![int_type(n), ERROR_TYPE])].into(),
        ResourceSet::default(),
    ))
}

fn itob_sig(_arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    Ok((
        vec![int_type(1)].into(),
        vec![bool_type()].into(),
        ResourceSet::default(),
    ))
}

fn btoi_sig(_arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    Ok((
        vec![bool_type()].into(),
        vec![int_type(1)].into(),
        ResourceSet::default(),
    ))
}

fn icmp_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg] = collect_array(arg_values);
    let n: u8 = get_width(arg)?;
    Ok((
        vec![int_type(n); 2].into(),
        vec![bool_type()].into(),
        ResourceSet::default(),
    ))
}

fn ibinop_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg] = collect_array(arg_values);
    let n: u8 = get_width(arg)?;
    Ok((
        vec![int_type(n); 2].into(),
        vec![int_type(n)].into(),
        ResourceSet::default(),
    ))
}

fn iunop_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg] = collect_array(arg_values);
    let n: u8 = get_width(arg)?;
    Ok((
        vec![int_type(n)].into(),
        vec![int_type(n)].into(),
        ResourceSet::default(),
    ))
}

fn idivmod_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    let intpair: TypeRow = vec![int_type(n), int_type(m)].into();
    Ok((
        intpair.clone(),
        vec![Type::new_sum(vec![Type::new_tuple(intpair), ERROR_TYPE])].into(),
        ResourceSet::default(),
    ))
}

fn idiv_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    Ok((
        vec![int_type(n), int_type(m)].into(),
        vec![Type::new_sum(vec![int_type(n), ERROR_TYPE])].into(),
        ResourceSet::default(),
    ))
}

fn imod_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    Ok((
        vec![int_type(n), int_type(m)].into(),
        vec![Type::new_sum(vec![int_type(m), ERROR_TYPE])].into(),
        ResourceSet::default(),
    ))
}

fn ish_sig(arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError> {
    let [arg0, arg1] = collect_array(arg_values);
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    Ok((
        vec![int_type(n), int_type(m)].into(),
        vec![int_type(n)].into(),
        ResourceSet::default(),
    ))
}

/// Resource for basic integer operations.
pub fn resource() -> Resource {
    let mut resource = Resource::new_with_reqs(
        RESOURCE_ID,
        ResourceSet::singleton(&super::int_types::RESOURCE_ID),
    );

    resource
        .add_op_custom_sig_simple(
            "iwiden_u".into(),
            "widen an unsigned integer to a wider one with the same value".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            iwiden_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "iwiden_s".into(),
            "widen a signed integer to a wider one with the same value".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            iwiden_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "inarrow_u".into(),
            "narrow an unsigned integer to a narrower one with the same value if possible"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            inarrow_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "inarrow_s".into(),
            "narrow a signed integer to a narrower one with the same value if possible".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            inarrow_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "itobool".into(),
            "convert to bool (1 is true, 0 is false)".to_owned(),
            vec![],
            itob_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ifrombool".into(),
            "convert from bool (1 is true, 0 is false)".to_owned(),
            vec![],
            btoi_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ieq".into(),
            "equality test".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ine".into(),
            "inequality test".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ilt_u".into(),
            "\"less than\" as unsigned integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ilt_s".into(),
            "\"less than\" as signed integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "igt_u".into(),
            "\"greater than\" as unsigned integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "igt_s".into(),
            "\"greater than\" as signed integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ile_u".into(),
            "\"less than or equal\" as unsigned integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ile_s".into(),
            "\"less than or equal\" as signed integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ige_u".into(),
            "\"greater than or equal\" as unsigned integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ige_s".into(),
            "\"greater than or equal\" as signed integers".to_owned(),
            vec![TypeParam::USize],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imax_u".into(),
            "maximum of unsigned integers".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imax_s".into(),
            "maximum of signed integers".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imin_u".into(),
            "minimum of unsigned integers".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imin_s".into(),
            "minimum of signed integers".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "iadd".into(),
            "addition modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "isub".into(),
            "subtraction modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ineg".into(),
            "negation modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![TypeParam::USize],
            iunop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imul".into(),
            "multiplication modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "idivmod_u".into(),
            "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^M, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 is an error)"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            idivmod_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "idivmod_s".into(),
            "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^M, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 is an error)"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            idivmod_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "idiv_u".into(),
            "as idivmod_u but discarding the second output".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            idiv_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imod_u".into(),
            "as idivmod_u but discarding the first output".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            idiv_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "idiv_s".into(),
            "as idivmod_s but discarding the second output".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            imod_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imod_s".into(),
            "as idivmod_s but discarding the first output".to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            imod_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "iabs".into(),
            "convert signed to unsigned by taking absolute value".to_owned(),
            vec![TypeParam::USize],
            iunop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "iand".into(),
            "bitwise AND".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ior".into(),
            "bitwise OR".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ixor".into(),
            "bitwise XOR".to_owned(),
            vec![TypeParam::USize],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "inot".into(),
            "bitwise NOT".to_owned(),
            vec![TypeParam::USize],
            iunop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ishl".into(),
            "shift first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits dropped, rightmost bits set to zero"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            ish_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ishr".into(),
            "shift first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits dropped, leftmost bits set to zero)"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            ish_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "irotl".into(),
            "rotate first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits replace rightmost bits)"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            ish_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "irotr".into(),
            "rotate first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits replace leftmost bits)"
                .to_owned(),
            vec![TypeParam::USize, TypeParam::USize],
            ish_sig,
        )
        .unwrap();

    resource
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_int_ops_resource() {
        let r = resource();
        assert_eq!(r.name(), "arithmetic.int");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.starts_with('i'));
        }
    }
}
