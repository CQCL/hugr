//! Basic floating-point operations.

use crate::{
    extension::{ExtensionId, ExtensionSet},
    type_row,
    types::{FunctionType, PolyFuncType},
    Extension,
};

use super::float_types::FLOAT64_TYPE;

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.float");

/// Extension for basic arithmetic operations.
pub fn extension() -> Extension {
    let mut extension = Extension::new_with_reqs(
        EXTENSION_ID,
        ExtensionSet::singleton(&super::float_types::EXTENSION_ID),
    );

    let fcmp_sig: PolyFuncType = FunctionType::new(
        type_row![FLOAT64_TYPE; 2],
        type_row![crate::extension::prelude::BOOL_T],
    )
    .into();
    let fbinop_sig: PolyFuncType =
        FunctionType::new(type_row![FLOAT64_TYPE; 2], type_row![FLOAT64_TYPE]).into();
    let funop_sig: PolyFuncType =
        FunctionType::new(type_row![FLOAT64_TYPE], type_row![FLOAT64_TYPE]).into();
    extension
        .add_op_type_scheme_simple("feq".into(), "equality test".to_owned(), fcmp_sig.clone())
        .unwrap();
    extension
        .add_op_type_scheme_simple("fne".into(), "inequality test".to_owned(), fcmp_sig.clone())
        .unwrap();
    extension
        .add_op_type_scheme_simple("flt".into(), "\"less than\"".to_owned(), fcmp_sig.clone())
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "fgt".into(),
            "\"greater than\"".to_owned(),
            fcmp_sig.clone(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "fle".into(),
            "\"less than or equal\"".to_owned(),
            fcmp_sig.clone(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "fge".into(),
            "\"greater than or equal\"".to_owned(),
            fcmp_sig,
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple("fmax".into(), "maximum".to_owned(), fbinop_sig.clone())
        .unwrap();
    extension
        .add_op_type_scheme_simple("fmin".into(), "minimum".to_owned(), fbinop_sig.clone())
        .unwrap();
    extension
        .add_op_type_scheme_simple("fadd".into(), "addition".to_owned(), fbinop_sig.clone())
        .unwrap();
    extension
        .add_op_type_scheme_simple("fsub".into(), "subtraction".to_owned(), fbinop_sig.clone())
        .unwrap();
    extension
        .add_op_type_scheme_simple("fneg".into(), "negation".to_owned(), funop_sig.clone())
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "fabs".into(),
            "absolute value".to_owned(),
            funop_sig.clone(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            "fmul".into(),
            "multiplication".to_owned(),
            fbinop_sig.clone(),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple("fdiv".into(), "division".to_owned(), fbinop_sig)
        .unwrap();
    extension
        .add_op_type_scheme_simple("ffloor".into(), "floor".to_owned(), funop_sig.clone())
        .unwrap();
    extension
        .add_op_type_scheme_simple("fceil".into(), "ceiling".to_owned(), funop_sig)
        .unwrap();

    extension
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_float_ops_extension() {
        let r = extension();
        assert_eq!(r.name() as &str, "arithmetic.float");
        assert_eq!(r.types().count(), 0);
        for (name, _) in r.operations() {
            assert!(name.starts_with('f'));
        }
    }
}
