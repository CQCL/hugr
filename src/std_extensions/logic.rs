//! Basic logical operations.

use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::{
    extension::{
        prelude::BOOL_T,
        simple_op::{try_from_name, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError},
        ExtensionId, ExtensionRegistry, OpDef, SignatureError, SignatureFromArgs, SignatureFunc,
    },
    ops::{self, custom::ExtensionOp, OpName},
    type_row,
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

/// Logic extension operation definitions.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs)]
pub enum NaryLogic {
    And,
    Or,
}

impl MakeOpDef for NaryLogic {
    fn signature(&self) -> SignatureFunc {
        logic_op_sig().into()
    }

    fn description(&self) -> String {
        match self {
            NaryLogic::And => "logical 'and'",
            NaryLogic::Or => "logical 'or'",
        }
        .to_string()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        try_from_name(op_def.name())
    }
}

/// Make a [NaryLogic] operation concrete by setting the type argument.
pub struct ConcreteLogicOp(pub NaryLogic, u64);

impl OpName for ConcreteLogicOp {
    fn name(&self) -> smol_str::SmolStr {
        self.0.name()
    }
}
impl MakeExtensionOp for ConcreteLogicOp {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError> {
        let def: NaryLogic = NaryLogic::from_def(ext_op.def())?;
        let [TypeArg::BoundedNat { n }] = *ext_op.args() else {
            return Err(SignatureError::InvalidTypeArgs.into());
        };
        Ok(Self(def, n))
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![TypeArg::BoundedNat { n: self.1 }]
    }
}

/// Not operation.
#[derive(Debug, Copy, Clone)]
pub struct NotOp;
impl OpName for NotOp {
    fn name(&self) -> smol_str::SmolStr {
        "Not".into()
    }
}
impl MakeOpDef for NotOp {
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        if op_def.name() == &NotOp.name() {
            Ok(NotOp)
        } else {
            Err(OpLoadError::NotMember(op_def.name().to_string()))
        }
    }

    fn signature(&self) -> SignatureFunc {
        FunctionType::new_endo(type_row![BOOL_T]).into()
    }
    fn description(&self) -> String {
        "logical 'not'".into()
    }
}
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
    NaryLogic::load_all_ops(&mut extension).unwrap();
    NotOp.add_to_extension(&mut extension).unwrap();

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
    /// Registry required to validate logic extension.
    pub static ref LOGIC_REG: ExtensionRegistry =
        ExtensionRegistry::try_new([EXTENSION.to_owned()]).unwrap();
}

impl MakeRegisteredOp for ConcreteLogicOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &LOGIC_REG
    }
}

impl MakeRegisteredOp for NotOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &LOGIC_REG
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::{extension, ConcreteLogicOp, NaryLogic, FALSE_NAME, TRUE_NAME};
    use crate::{
        extension::{prelude::BOOL_T, simple_op::MakeOpDef},
        ops::OpName,
        Extension,
    };

    use strum::IntoEnumIterator;

    #[test]
    fn test_logic_extension() {
        let r: Extension = extension();
        assert_eq!(r.name() as &str, "logic");
        assert_eq!(r.operations().count(), 3);

        for op in NaryLogic::iter() {
            assert_eq!(
                NaryLogic::from_def(r.get_op(&op.name()).unwrap(),).unwrap(),
                op
            );
        }
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

    /// Generate a logic extension "and" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn and_op() -> ConcreteLogicOp {
        ConcreteLogicOp(NaryLogic::And, 2)
    }

    /// Generate a logic extension "or" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn or_op() -> ConcreteLogicOp {
        ConcreteLogicOp(NaryLogic::Or, 2)
    }
}
