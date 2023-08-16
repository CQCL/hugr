//! Basic floating-point operations.

use smol_str::SmolStr;

use crate::{
    extension::{ExtensionSet, SignatureError},
    types::{type_param::TypeArg, TypeRow},
    Extension,
};

use super::super::logic::bool_type;
use super::float_types::float64_type;

/// The resource identifier.
pub const RESOURCE_ID: SmolStr = SmolStr::new_inline("arithmetic.float");

fn fcmp_sig(_arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ExtensionSet), SignatureError> {
    Ok((
        vec![float64_type(); 2].into(),
        vec![bool_type()].into(),
        ExtensionSet::default(),
    ))
}

fn fbinop_sig(_arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ExtensionSet), SignatureError> {
    Ok((
        vec![float64_type(); 2].into(),
        vec![float64_type()].into(),
        ExtensionSet::default(),
    ))
}

fn funop_sig(_arg_values: &[TypeArg]) -> Result<(TypeRow, TypeRow, ExtensionSet), SignatureError> {
    Ok((
        vec![float64_type()].into(),
        vec![float64_type()].into(),
        ExtensionSet::default(),
    ))
}

/// Resource for basic arithmetic operations.
pub fn extension() -> Extension {
    let mut extension = Extension::new_with_reqs(
        RESOURCE_ID,
        ExtensionSet::singleton(&super::float_types::RESOURCE_ID),
    );

    extension
        .add_op_custom_sig_simple("feq".into(), "equality test".to_owned(), vec![], fcmp_sig)
        .unwrap();
    extension
        .add_op_custom_sig_simple("fne".into(), "inequality test".to_owned(), vec![], fcmp_sig)
        .unwrap();
    extension
        .add_op_custom_sig_simple("flt".into(), "\"less than\"".to_owned(), vec![], fcmp_sig)
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "fgt".into(),
            "\"greater than\"".to_owned(),
            vec![],
            fcmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "fle".into(),
            "\"less than or equal\"".to_owned(),
            vec![],
            fcmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "fge".into(),
            "\"greater than or equal\"".to_owned(),
            vec![],
            fcmp_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple("fmax".into(), "maximum".to_owned(), vec![], fbinop_sig)
        .unwrap();
    extension
        .add_op_custom_sig_simple("fmin".into(), "minimum".to_owned(), vec![], fbinop_sig)
        .unwrap();
    extension
        .add_op_custom_sig_simple("fadd".into(), "addition".to_owned(), vec![], fbinop_sig)
        .unwrap();
    extension
        .add_op_custom_sig_simple("fsub".into(), "subtraction".to_owned(), vec![], fbinop_sig)
        .unwrap();
    extension
        .add_op_custom_sig_simple("fneg".into(), "negation".to_owned(), vec![], funop_sig)
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "fabs".into(),
            "absolute value".to_owned(),
            vec![],
            funop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple(
            "fmul".into(),
            "multiplication".to_owned(),
            vec![],
            fbinop_sig,
        )
        .unwrap();
    extension
        .add_op_custom_sig_simple("fdiv".into(), "division".to_owned(), vec![], fbinop_sig)
        .unwrap();
    extension
        .add_op_custom_sig_simple("ffloor".into(), "floor".to_owned(), vec![], funop_sig)
        .unwrap();
    extension
        .add_op_custom_sig_simple("fceil".into(), "ceiling".to_owned(), vec![], funop_sig)
        .unwrap();

    extension
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_float_ops_extension() {
        let r = extension();
        assert_eq!(r.name(), "arithmetic.float");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.starts_with('f'));
        }
    }
}
