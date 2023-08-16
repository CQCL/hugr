//! Basic logical operations.

use itertools::Itertools;
use smol_str::SmolStr;

use crate::{
    extension::ExtensionSet,
    ops,
    types::{
        type_param::{TypeArg, TypeArgError, TypeParam},
        Type,
    },
    Extension,
};

/// Name of resource false value.
pub const FALSE_NAME: &str = "FALSE";
/// Name of resource true value.
pub const TRUE_NAME: &str = "TRUE";

/// The resource identifier.
pub const RESOURCE_ID: SmolStr = SmolStr::new_inline("logic");

/// Construct a boolean type.
pub fn bool_type() -> Type {
    Type::new_simple_predicate(2)
}

/// Resource for basic logical operations.
pub fn extension() -> Extension {
    const H_INT: TypeParam = TypeParam::USize;
    let mut extension = Extension::new(RESOURCE_ID);

    extension
        .add_op_custom_sig_simple(
            "Not".into(),
            "logical 'not'".into(),
            vec![],
            |_arg_values: &[TypeArg]| {
                Ok((
                    vec![bool_type()].into(),
                    vec![bool_type()].into(),
                    ExtensionSet::default(),
                ))
            },
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            "And".into(),
            "logical 'and'".into(),
            vec![H_INT],
            |arg_values: &[TypeArg]| {
                let a = arg_values.iter().exactly_one().unwrap();
                let n: u64 = match a {
                    TypeArg::USize(n) => *n,
                    _ => {
                        return Err(TypeArgError::TypeMismatch {
                            arg: a.clone(),
                            param: H_INT,
                        }
                        .into());
                    }
                };
                Ok((
                    vec![bool_type(); n as usize].into(),
                    vec![bool_type()].into(),
                    ExtensionSet::default(),
                ))
            },
        )
        .unwrap();

    extension
        .add_op_custom_sig_simple(
            "Or".into(),
            "logical 'or'".into(),
            vec![H_INT],
            |arg_values: &[TypeArg]| {
                let a = arg_values.iter().exactly_one().unwrap();
                let n: u64 = match a {
                    TypeArg::USize(n) => *n,
                    _ => {
                        return Err(TypeArgError::TypeMismatch {
                            arg: a.clone(),
                            param: H_INT,
                        }
                        .into());
                    }
                };
                Ok((
                    vec![bool_type(); n as usize].into(),
                    vec![bool_type()].into(),
                    ExtensionSet::default(),
                ))
            },
        )
        .unwrap();

    extension
        .add_value(FALSE_NAME, ops::Const::simple_predicate(0, 2))
        .unwrap();
    extension
        .add_value(TRUE_NAME, ops::Const::simple_predicate(1, 2))
        .unwrap();
    extension
}

#[cfg(test)]
mod test {
    use crate::Extension;

    use super::{bool_type, extension, FALSE_NAME, TRUE_NAME};

    #[test]
    fn test_logic_extension() {
        let r: Extension = extension();
        assert_eq!(r.name(), "logic");
        assert_eq!(r.operations().count(), 3);
    }

    #[test]
    fn test_values() {
        let r: Extension = extension();
        let false_val = r.get_value(FALSE_NAME).unwrap();
        let true_val = r.get_value(TRUE_NAME).unwrap();

        for v in [false_val, true_val] {
            let simpl = v.typed_value().const_type();
            assert_eq!(simpl, &bool_type());
        }
    }
}
