//! Basic floating-point operations.

use smol_str::SmolStr;

use crate::{
    resource::{ResourceSet, SignatureError},
    types::{type_param::TypeArg, SimpleRow},
    Resource,
};

use super::super::logic::bool_type;
use super::float_types::float64_type;

/// The resource identifier.
pub const RESOURCE_ID: SmolStr = SmolStr::new_inline("arithmetic.float");

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

/// Resource for basic arithmetic operations.
pub fn resource() -> Resource {
    let mut resource = Resource::new_with_reqs(
        RESOURCE_ID,
        ResourceSet::singleton(&super::float_types::RESOURCE_ID),
    );

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
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_float_ops_resource() {
        let r = resource();
        assert_eq!(r.name(), "arithmetic.float");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.starts_with('f'));
        }
    }
}
