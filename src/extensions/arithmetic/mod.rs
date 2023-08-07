//! Basic arithmetic operations.

use itertools::Itertools;
use smol_str::SmolStr;

use crate::{
    resource::{ResourceSet, SignatureError},
    types::{
        type_param::{TypeArg, TypeArgError, TypeParam},
        CustomType, HashableType, SimpleRow, SimpleType, TypeRow, TypeTag,
    },
    Resource,
};

use super::logic::bool_type;

/// The resource identifier.
pub const fn resource_id() -> SmolStr {
    SmolStr::new_inline("Arithmetic")
}

const INT_PARAM: TypeParam = TypeParam::Value(HashableType::Int(8));

const INT_TYPE_ID: SmolStr = SmolStr::new_inline("int");
const FLOAT64_TYPE_ID: SmolStr = SmolStr::new_inline("float64");

fn int_type(n: u8) -> SimpleType {
    CustomType::new(
        INT_TYPE_ID,
        [TypeArg::Int(n as u128)],
        resource_id(),
        TypeTag::Classic,
    )
    .into()
}

fn float64_type() -> SimpleType {
    CustomType::new(FLOAT64_TYPE_ID, [], resource_id(), TypeTag::Classic).into()
}

fn get_width(arg: &TypeArg) -> Result<u8, SignatureError> {
    let n: u8 = match arg {
        TypeArg::Int(n) => *n as u8,
        _ => {
            return Err(TypeArgError::TypeMismatch(arg.clone(), INT_PARAM).into());
        }
    };
    if (n != 1)
        && (n != 2)
        && (n != 4)
        && (n != 8)
        && (n != 16)
        && (n != 32)
        && (n != 64)
        && (n != 128)
    {
        return Err(TypeArgError::InvalidValue(arg.clone()).into());
    }
    Ok(n)
}

fn iwiden_sig(
    arg_values: &[TypeArg],
) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let (arg0, arg1) = arg_values.iter().collect_tuple().unwrap();
    let m: u8 = get_width(arg0)?;
    let n: u8 = get_width(arg1)?;
    if m > n {
        return Err(SignatureError::InvalidTypeArgs());
    }
    Ok((
        vec![int_type(m)].into(),
        vec![int_type(n)].into(),
        ResourceSet::default(),
    ))
}

fn inarrow_sig(
    arg_values: &[TypeArg],
) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let (arg0, arg1) = arg_values.iter().collect_tuple().unwrap();
    let m: u8 = get_width(arg0)?;
    let n: u8 = get_width(arg1)?;
    if m < n {
        return Err(SignatureError::InvalidTypeArgs());
    }
    Ok((
        vec![int_type(m)].into(),
        vec![SimpleType::new_sum(vec![
            int_type(n),
            HashableType::OpError.into(),
        ])]
        .into(),
        ResourceSet::default(),
    ))
}

fn itob_sig(
    _arg_values: &[TypeArg],
) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    Ok((
        vec![int_type(1)].into(),
        vec![bool_type()].into(),
        ResourceSet::default(),
    ))
}

fn btoi_sig(
    _arg_values: &[TypeArg],
) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    Ok((
        vec![bool_type()].into(),
        vec![int_type(1)].into(),
        ResourceSet::default(),
    ))
}

fn icmp_sig(arg_values: &[TypeArg]) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let arg = arg_values.iter().exactly_one().unwrap();
    let n: u8 = get_width(arg)?;
    Ok((
        vec![int_type(n); 2].into(),
        vec![bool_type()].into(),
        ResourceSet::default(),
    ))
}

fn ibinop_sig(
    arg_values: &[TypeArg],
) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let arg = arg_values.iter().exactly_one().unwrap();
    let n: u8 = get_width(arg)?;
    Ok((
        vec![int_type(n); 2].into(),
        vec![int_type(n)].into(),
        ResourceSet::default(),
    ))
}

fn iunop_sig(
    arg_values: &[TypeArg],
) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let arg = arg_values.iter().exactly_one().unwrap();
    let n: u8 = get_width(arg)?;
    Ok((
        vec![int_type(n)].into(),
        vec![int_type(n)].into(),
        ResourceSet::default(),
    ))
}

fn idivmod_sig(
    arg_values: &[TypeArg],
) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let (arg0, arg1) = arg_values.iter().collect_tuple().unwrap();
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    let intpair: TypeRow<SimpleType> = vec![int_type(n), int_type(m)].into();
    Ok((
        intpair.clone(),
        vec![SimpleType::new_sum(vec![
            SimpleType::new_tuple(intpair),
            HashableType::OpError.into(),
        ])]
        .into(),
        ResourceSet::default(),
    ))
}

fn idiv_sig(arg_values: &[TypeArg]) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let (arg0, arg1) = arg_values.iter().collect_tuple().unwrap();
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    Ok((
        vec![int_type(n), int_type(m)].into(),
        vec![SimpleType::new_sum(vec![
            int_type(n),
            HashableType::OpError.into(),
        ])]
        .into(),
        ResourceSet::default(),
    ))
}

fn imod_sig(arg_values: &[TypeArg]) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let (arg0, arg1) = arg_values.iter().collect_tuple().unwrap();
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    Ok((
        vec![int_type(n), int_type(m)].into(),
        vec![SimpleType::new_sum(vec![
            int_type(m),
            HashableType::OpError.into(),
        ])]
        .into(),
        ResourceSet::default(),
    ))
}

fn ish_sig(arg_values: &[TypeArg]) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let (arg0, arg1) = arg_values.iter().collect_tuple().unwrap();
    let n: u8 = get_width(arg0)?;
    let m: u8 = get_width(arg1)?;
    Ok((
        vec![int_type(n), int_type(m)].into(),
        vec![int_type(n)].into(),
        ResourceSet::default(),
    ))
}

fn fcmp_sig(
    _arg_values: &[TypeArg],
) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    Ok((
        vec![float64_type(); 2].into(),
        vec![bool_type()].into(),
        ResourceSet::default(),
    ))
}

fn fbinop_sig(
    _arg_values: &[TypeArg],
) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    Ok((
        vec![float64_type(); 2].into(),
        vec![float64_type()].into(),
        ResourceSet::default(),
    ))
}

fn funop_sig(
    _arg_values: &[TypeArg],
) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    Ok((
        vec![float64_type()].into(),
        vec![float64_type()].into(),
        ResourceSet::default(),
    ))
}

fn ftoi_sig(arg_values: &[TypeArg]) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let arg = arg_values.iter().exactly_one().unwrap();
    let n: u8 = get_width(arg)?;
    Ok((
        vec![float64_type()].into(),
        vec![SimpleType::new_sum(vec![
            int_type(n),
            HashableType::OpError.into(),
        ])]
        .into(),
        ResourceSet::default(),
    ))
}

fn itof_sig(arg_values: &[TypeArg]) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
    let arg = arg_values.iter().exactly_one().unwrap();
    let n: u8 = get_width(arg)?;
    Ok((
        vec![int_type(n)].into(),
        vec![float64_type()].into(),
        ResourceSet::default(),
    ))
}

/// Resource for basic arithmetic operations.
pub fn resource() -> Resource {
    let mut resource = Resource::new(resource_id());

    // Add types.
    resource
        .add_type(
            "int".into(),
            vec![INT_PARAM],
            "integral value of a given bit width".to_owned(),
            TypeTag::Classic.into(),
        )
        .unwrap();
    resource
        .add_type(
            "float64".into(),
            vec![],
            "64-bit IEEE 754-2019 floating-point value".to_owned(),
            TypeTag::Classic.into(),
        )
        .unwrap();

    // TODO Add consts. https://github.com/CQCL-DEV/hugr/issues/332

    // Add operations.
    resource
        .add_op_custom_sig_simple(
            "iwiden_u".into(),
            "widen an unsigned integer to a wider one with the same value".to_owned(),
            vec![INT_PARAM, INT_PARAM],
            iwiden_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "iwiden_s".into(),
            "widen a signed integer to a wider one with the same value".to_owned(),
            vec![INT_PARAM, INT_PARAM],
            iwiden_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "inarrow_u".into(),
            "narrow an unsigned integer to a narrower one with the same value if possible"
                .to_owned(),
            vec![INT_PARAM, INT_PARAM],
            inarrow_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "inarrow_s".into(),
            "narrow a signed integer to a narrower one with the same value if possible".to_owned(),
            vec![INT_PARAM, INT_PARAM],
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
            vec![INT_PARAM],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ine".into(),
            "inequality test".to_owned(),
            vec![INT_PARAM],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ilt_u".into(),
            "\"less than\" as unsigned integers".to_owned(),
            vec![INT_PARAM],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ilt_s".into(),
            "\"less than\" as signed integers".to_owned(),
            vec![INT_PARAM],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "igt_u".into(),
            "\"greater than\" as unsigned integers".to_owned(),
            vec![INT_PARAM],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "igt_s".into(),
            "\"greater than\" as signed integers".to_owned(),
            vec![INT_PARAM],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ile_u".into(),
            "\"less than or equal\" as unsigned integers".to_owned(),
            vec![INT_PARAM],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ile_s".into(),
            "\"less than or equal\" as signed integers".to_owned(),
            vec![INT_PARAM],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ige_u".into(),
            "\"greater than or equal\" as unsigned integers".to_owned(),
            vec![INT_PARAM],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ige_s".into(),
            "\"greater than or equal\" as signed integers".to_owned(),
            vec![INT_PARAM],
            icmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imax_u".into(),
            "maximum of unsigned integers".to_owned(),
            vec![INT_PARAM],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imax_s".into(),
            "maximum of signed integers".to_owned(),
            vec![INT_PARAM],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imin_u".into(),
            "minimum of unsigned integers".to_owned(),
            vec![INT_PARAM],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imin_s".into(),
            "minimum of signed integers".to_owned(),
            vec![INT_PARAM],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "iadd".into(),
            "addition modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![INT_PARAM],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "isub".into(),
            "subtraction modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![INT_PARAM],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ineg".into(),
            "negation modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![INT_PARAM],
            iunop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imul".into(),
            "multiplication modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            vec![INT_PARAM],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "idivmod_u".into(),
            "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^M, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 is an error)"
                .to_owned(),
            vec![INT_PARAM, INT_PARAM],
            idivmod_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "idivmod_s".into(),
            "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^M, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 is an error)"
                .to_owned(),
            vec![INT_PARAM, INT_PARAM],
            idivmod_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "idiv_u".into(),
            "as idivmod_u but discarding the second output".to_owned(),
            vec![INT_PARAM, INT_PARAM],
            idiv_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imod_u".into(),
            "as idivmod_u but discarding the first output".to_owned(),
            vec![INT_PARAM, INT_PARAM],
            idiv_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "idiv_s".into(),
            "as idivmod_s but discarding the second output".to_owned(),
            vec![INT_PARAM, INT_PARAM],
            imod_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "imod_s".into(),
            "as idivmod_s but discarding the first output".to_owned(),
            vec![INT_PARAM, INT_PARAM],
            imod_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "iabs".into(),
            "convert signed to unsigned by taking absolute value".to_owned(),
            vec![INT_PARAM],
            iunop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "iand".into(),
            "bitwise AND".to_owned(),
            vec![INT_PARAM],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ior".into(),
            "bitwise OR".to_owned(),
            vec![INT_PARAM],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ixor".into(),
            "bitwise XOR".to_owned(),
            vec![INT_PARAM],
            ibinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "inot".into(),
            "bitwise NOT".to_owned(),
            vec![INT_PARAM],
            iunop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ishl".into(),
            "shift first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits dropped, rightmost bits set to zero"
                .to_owned(),
            vec![INT_PARAM, INT_PARAM],
            ish_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "ishr".into(),
            "shift first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits dropped, leftmost bits set to zero)"
                .to_owned(),
            vec![INT_PARAM, INT_PARAM],
            ish_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "irotl".into(),
            "rotate first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits replace rightmost bits)"
                .to_owned(),
            vec![INT_PARAM, INT_PARAM],
            ish_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "irotr".into(),
            "rotate first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits replace leftmost bits)"
                .to_owned(),
            vec![INT_PARAM, INT_PARAM],
            ish_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple("feq".into(), "equality test".to_owned(), vec![], fcmp_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple("fne".into(), "inequality test".to_owned(), vec![], fcmp_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple("flt".into(), "\"less than\"".to_owned(), vec![], fcmp_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "fgt".into(),
            "\"greater than\"".to_owned(),
            vec![],
            fcmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "fle".into(),
            "\"less than or equal\"".to_owned(),
            vec![],
            fcmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "fge".into(),
            "\"greater than or equal\"".to_owned(),
            vec![],
            fcmp_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple("fmax".into(), "maximum".to_owned(), vec![], fbinop_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple("fmin".into(), "minimum".to_owned(), vec![], fbinop_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple("fadd".into(), "addition".to_owned(), vec![], fbinop_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple("fsub".into(), "subtraction".to_owned(), vec![], fbinop_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple("fneg".into(), "negation".to_owned(), vec![], funop_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "fabs".into(),
            "absolute value".to_owned(),
            vec![],
            funop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "fmul".into(),
            "multiplication".to_owned(),
            vec![],
            fbinop_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple("fdiv".into(), "division".to_owned(), vec![], fbinop_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple("ffloor".into(), "floor".to_owned(), vec![], funop_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple("fceil".into(), "ceiling".to_owned(), vec![], funop_sig)
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "trunc_u".into(),
            "float to unsigned int".to_owned(),
            vec![INT_PARAM],
            ftoi_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "trunc_s".into(),
            "float to signed int".to_owned(),
            vec![INT_PARAM],
            ftoi_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "convert_u".into(),
            "unsigned int to float".to_owned(),
            vec![INT_PARAM],
            itof_sig,
        )
        .unwrap();
    resource
        .add_op_custom_sig_simple(
            "convert_s".into(),
            "signed int to float".to_owned(),
            vec![INT_PARAM],
            itof_sig,
        )
        .unwrap();

    resource
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_arithmetic_resource() {
        let r = resource();
        assert_eq!(r.name(), "Arithmetic");
        assert_eq!(r.types().count(), 2);
        for (name, _) in r.operations() {
            assert!(
                name.starts_with('i')
                    || name.starts_with('f')
                    || name.starts_with("convert")
                    || name.starts_with("trunc")
            );
        }
    }
}
