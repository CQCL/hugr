//! Basic logical operations.

use smol_str::SmolStr;

use crate::{
    extension::{prelude::BOOL_T, ExtensionId, SignatureError, SignatureFromArgs},
    ops, type_row,
    types::{
        type_param::{TypeArg, TypeParam},
        FunctionType,
    },
    Extension,
};
use lazy_static::lazy_static;

/// Name of extension false value.
pub const FALSE_NAME: &str = "FALSE";
/// Name of extension true value.
pub const TRUE_NAME: &str = "TRUE";

/// Name of the "not" operation.
pub const NOT_NAME: &str = "Not";
/// Name of the "and" operation.
pub const AND_NAME: &str = "And";
/// Name of the "or" operation.
pub const OR_NAME: &str = "Or";
/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("logic");

fn logic_op_sig() -> impl SignatureFromArgs {
    struct LogicOpCustom;

    const MAX: &[TypeParam; 1] = &[TypeParam::max_nat()];
    impl SignatureFromArgs for LogicOpCustom {
        fn compute_signature(
            &self,
            arg_values: &[TypeArg],
        ) -> Result<crate::types::PolyFuncType, SignatureError> {
            // get the number of input booleans.
            let [TypeArg::BoundedNat { n }] = *arg_values else {
                return Err(SignatureError::InvalidTypeArgs);
            };
            let var_arg_row = vec![BOOL_T; n as usize];
            Ok(FunctionType::new(var_arg_row, vec![BOOL_T]).into())
        }

        fn static_params(&self) -> &[TypeParam] {
            MAX
        }
    }
    LogicOpCustom
}
/// Extension for basic logical operations.
fn extension() -> Extension {
    let mut extension = Extension::new(EXTENSION_ID);

    extension
        .add_op(
            SmolStr::new_inline(NOT_NAME),
            "logical 'not'".into(),
            FunctionType::new(type_row![BOOL_T], type_row![BOOL_T]),
        )
        .unwrap();

    extension
        .add_op(
            SmolStr::new_inline(AND_NAME),
            "logical 'and'".into(),
            logic_op_sig(),
        )
        .unwrap();

    extension
        .add_op(
            SmolStr::new_inline(OR_NAME),
            "logical 'or'".into(),
            logic_op_sig(),
        )
        .unwrap();

    extension
        .add_value(FALSE_NAME, ops::Const::unit_sum(0, 2))
        .unwrap();
    extension
        .add_value(TRUE_NAME, ops::Const::unit_sum(1, 2))
        .unwrap();
    extension
}

lazy_static! {
    /// Reference to the logic Extension.
    pub static ref EXTENSION: Extension = extension();
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        extension::{prelude::BOOL_T, EMPTY_REG},
        ops::LeafOp,
        types::type_param::TypeArg,
        Extension,
    };

    use super::{extension, AND_NAME, EXTENSION, FALSE_NAME, NOT_NAME, OR_NAME, TRUE_NAME};

    #[test]
    fn test_logic_extension() {
        let r: Extension = extension();
        assert_eq!(r.name() as &str, "logic");
        assert_eq!(r.operations().count(), 3);
    }

    #[test]
    fn test_values() {
        let r: Extension = extension();
        let false_val = r.get_value(FALSE_NAME).unwrap();
        let true_val = r.get_value(TRUE_NAME).unwrap();

        for v in [false_val, true_val] {
            let simpl = v.typed_value().const_type();
            assert_eq!(simpl, &BOOL_T);
        }
    }

    /// Generate a logic extension and "and" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn and_op() -> LeafOp {
        EXTENSION
            .instantiate_extension_op(AND_NAME, [TypeArg::BoundedNat { n: 2 }], &EMPTY_REG)
            .unwrap()
            .into()
    }

    /// Generate a logic extension and "or" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn or_op() -> LeafOp {
        EXTENSION
            .instantiate_extension_op(OR_NAME, [TypeArg::BoundedNat { n: 2 }], &EMPTY_REG)
            .unwrap()
            .into()
    }

    /// Generate a logic extension and "not" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn not_op() -> LeafOp {
        EXTENSION
            .instantiate_extension_op(NOT_NAME, [], &EMPTY_REG)
            .unwrap()
            .into()
    }
}
