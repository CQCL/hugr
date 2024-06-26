//! Basic logical operations.

use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::extension::{ConstFold, ConstFoldResult};
use crate::ops::constant::ValueName;
use crate::ops::{OpName, Value};
use crate::types::{FunctionType, FunctionTypeRV};
use crate::{
    extension::{
        prelude::BOOL_T,
        simple_op::{try_from_name, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError},
        ExtensionId, ExtensionRegistry, OpDef, SignatureError, SignatureFromArgs, SignatureFunc,
    },
    ops::{self, custom::ExtensionOp, NamedOp},
    type_row,
    types::type_param::{TypeArg, TypeParam},
    utils::sorted_consts,
    Extension, IncomingPort,
};
use lazy_static::lazy_static;
/// Name of extension false value.
pub const FALSE_NAME: ValueName = ValueName::new_inline("FALSE");
/// Name of extension true value.
pub const TRUE_NAME: ValueName = ValueName::new_inline("TRUE");

impl ConstFold for NaryLogic {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        let [TypeArg::BoundedNat { n: num_args }] = *type_args else {
            panic!("impossible by validation");
        };
        match self {
            Self::And => {
                let inps = read_inputs(consts)?;
                let res = inps.iter().all(|x| *x);
                // We can only fold to true if we have a const for all our inputs.
                (!res || inps.len() as u64 == num_args)
                    .then_some(vec![(0.into(), ops::Value::from_bool(res))])
            }
            Self::Or => {
                let inps = read_inputs(consts)?;
                let res = inps.iter().any(|x| *x);
                // We can only fold to false if we have a const for all our inputs
                (res || inps.len() as u64 == num_args)
                    .then_some(vec![(0.into(), ops::Value::from_bool(res))])
            }
        }
    }
}
/// Logic extension operation definitions.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs)]
#[non_exhaustive]
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

    fn post_opdef(&self, def: &mut OpDef) {
        def.set_constant_folder(*self);
    }
}

/// Make a [NaryLogic] operation concrete by setting the type argument.
#[derive(Debug, Clone, PartialEq)]
pub struct ConcreteLogicOp(pub NaryLogic, u64);

impl NaryLogic {
    /// Initialise a [ConcreteLogicOp] by setting the number of inputs to this
    /// logic operation.
    pub fn with_n_inputs(self, n: u64) -> ConcreteLogicOp {
        ConcreteLogicOp(self, n)
    }
}
impl NamedOp for ConcreteLogicOp {
    fn name(&self) -> OpName {
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
impl NamedOp for NotOp {
    fn name(&self) -> OpName {
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

    fn post_opdef(&self, def: &mut OpDef) {
        def.set_constant_folder(|consts: &_| {
            let inps = read_inputs(consts)?;
            if inps.len() != 1 {
                None
            } else {
                Some(vec![(0.into(), ops::Value::from_bool(!inps[0]))])
            }
        })
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
            Ok(FunctionTypeRV::new(var_arg_row, vec![BOOL_T]).into())
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
        .add_value(FALSE_NAME, ops::Value::false_val())
        .unwrap();
    extension
        .add_value(TRUE_NAME, ops::Value::true_val())
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

fn read_inputs(consts: &[(IncomingPort, ops::Value)]) -> Option<Vec<bool>> {
    let true_val = ops::Value::true_val();
    let false_val = ops::Value::false_val();
    let inps: Option<Vec<bool>> = sorted_consts(consts)
        .into_iter()
        .map(|c| {
            if c == &true_val {
                Some(true)
            } else if c == &false_val {
                Some(false)
            } else {
                None
            }
        })
        .collect();
    let inps = inps?;
    Some(inps)
}

#[cfg(test)]
pub(crate) mod test {
    use super::{extension, ConcreteLogicOp, NaryLogic, NotOp, FALSE_NAME, TRUE_NAME};
    use crate::{
        extension::{
            prelude::BOOL_T,
            simple_op::{MakeExtensionOp, MakeOpDef, MakeRegisteredOp},
        },
        ops::{NamedOp, Value},
        Extension,
    };

    use rstest::rstest;
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
    fn test_conversions() {
        for def in [NaryLogic::And, NaryLogic::Or] {
            let o = def.with_n_inputs(3);
            let ext_op = o.clone().to_extension_op().unwrap();
            assert_eq!(ConcreteLogicOp::from_extension_op(&ext_op).unwrap(), o);
        }

        NotOp::from_extension_op(&NotOp.to_extension_op().unwrap()).unwrap();
    }

    #[test]
    fn test_values() {
        let r: Extension = extension();
        let false_val = r.get_value(&FALSE_NAME).unwrap();
        let true_val = r.get_value(&TRUE_NAME).unwrap();

        for v in [false_val, true_val] {
            let simpl = v.typed_value().get_type();
            assert_eq!(simpl, BOOL_T);
        }
    }

    /// Generate a logic extension "and" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn and_op() -> ConcreteLogicOp {
        NaryLogic::And.with_n_inputs(2)
    }

    /// Generate a logic extension "or" operation over [`crate::prelude::BOOL_T`]
    pub(crate) fn or_op() -> ConcreteLogicOp {
        NaryLogic::Or.with_n_inputs(2)
    }

    #[rstest]
    #[case(NaryLogic::And, [], true)]
    #[case(NaryLogic::And, [true, true, true], true)]
    #[case(NaryLogic::And, [true, false, true], false)]
    #[case(NaryLogic::Or, [], false)]
    #[case(NaryLogic::Or, [false, false, true], true)]
    #[case(NaryLogic::Or, [false, false, false], false)]
    fn nary_const_fold(
        #[case] op: NaryLogic,
        #[case] ins: impl IntoIterator<Item = bool>,
        #[case] out: bool,
    ) {
        use itertools::Itertools;

        use crate::extension::ConstFold;
        let in_vals = ins
            .into_iter()
            .enumerate()
            .map(|(i, b)| (i.into(), Value::from_bool(b)))
            .collect_vec();
        assert_eq!(
            Some(vec![(0.into(), Value::from_bool(out))]),
            op.fold(&[(in_vals.len() as u64).into()], &in_vals)
        );
    }

    #[rstest]
    #[case(NaryLogic::And, [Some(true), None], None)]
    #[case(NaryLogic::And, [Some(false), None], Some(false))]
    #[case(NaryLogic::Or, [None, Some(false)], None)]
    #[case(NaryLogic::Or, [None, Some(true)], Some(true))]
    fn nary_partial_const_fold(
        #[case] op: NaryLogic,
        #[case] ins: impl IntoIterator<Item = Option<bool>>,
        #[case] mb_out: Option<bool>,
    ) {
        use itertools::Itertools;

        use crate::extension::ConstFold;
        let in_vals0 = ins.into_iter().enumerate().collect_vec();
        let num_args = in_vals0.len() as u64;
        let in_vals = in_vals0
            .into_iter()
            .filter_map(|(i, mb_b)| mb_b.map(|b| (i.into(), Value::from_bool(b))))
            .collect_vec();
        assert_eq!(
            mb_out.map(|out| vec![(0.into(), Value::from_bool(out))]),
            op.fold(&[num_args.into()], &in_vals)
        );
    }
}
