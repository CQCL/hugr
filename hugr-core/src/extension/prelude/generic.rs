use std::str::FromStr;
use std::sync::{Arc, Weak};

use crate::extension::OpDef;
use crate::extension::SignatureFunc;
use crate::extension::prelude::usize_custom_t;
use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{ConstFold, ExtensionId};
use crate::ops::ExtensionOp;
use crate::ops::OpName;
use crate::type_row;
use crate::types::FuncValueType;

use crate::types::Type;

use crate::extension::SignatureError;

use crate::types::PolyFuncTypeRV;

use crate::Extension;
use crate::types::type_param::TypeArg;

use super::PRELUDE;
use super::{ConstUsize, PRELUDE_ID};
use crate::types::type_param::TypeParam;

/// Name of the operation for loading generic `BoundedNat` parameters.
pub static LOAD_NAT_OP_ID: OpName = OpName::new_inline("load_nat");

/// Definition of the load nat operation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct LoadNatDef;

impl FromStr for LoadNatDef {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == Self.op_id() { Ok(Self) } else { Err(()) }
    }
}

impl ConstFold for LoadNatDef {
    fn fold(
        &self,
        type_args: &[TypeArg],
        _consts: &[(crate::IncomingPort, crate::ops::Value)],
    ) -> crate::extension::ConstFoldResult {
        let [arg] = type_args else {
            return None;
        };
        let nat = arg.as_nat();
        if let Some(n) = nat {
            let n_const = ConstUsize::new(n);
            Some(vec![(0.into(), n_const.into())])
        } else {
            None
        }
    }
}

impl MakeOpDef for LoadNatDef {
    fn opdef_id(&self) -> OpName {
        LOAD_NAT_OP_ID.clone()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        let usize_t: Type = usize_custom_t(_extension_ref).into();
        let params = vec![TypeParam::max_nat_type()];
        PolyFuncTypeRV::new(params, FuncValueType::new(type_row![], vec![usize_t])).into()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&PRELUDE)
    }

    fn extension(&self) -> ExtensionId {
        PRELUDE_ID
    }

    fn description(&self) -> String {
        "Loads a generic bounded nat parameter into a usize runtime value.".into()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        def.set_constant_folder(*self);
    }
}

/// Concrete load nat operation.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadNat {
    nat: TypeArg,
}

impl LoadNat {
    /// Creates a new [`LoadNat`] operation.
    #[must_use]
    pub fn new(nat: TypeArg) -> Self {
        LoadNat { nat }
    }

    /// Returns the nat type argument that should be loaded.
    #[must_use]
    pub fn get_nat(self) -> TypeArg {
        self.nat
    }
}

impl MakeExtensionOp for LoadNat {
    fn op_id(&self) -> OpName {
        LoadNatDef.opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = LoadNatDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![self.nat.clone()]
    }
}

impl MakeRegisteredOp for LoadNat {
    fn extension_id(&self) -> ExtensionId {
        PRELUDE_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&PRELUDE)
    }
}

impl HasDef for LoadNat {
    type Def = LoadNatDef;
}

impl HasConcrete for LoadNatDef {
    type Concrete = LoadNat;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match type_args {
            [n] => Ok(LoadNat::new(n.clone())),
            _ => Err(SignatureError::InvalidTypeArgs.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        HugrView, OutgoingPort,
        builder::{DFGBuilder, Dataflow, DataflowHugr, inout_sig},
        extension::prelude::{ConstUsize, usize_t},
        ops::{OpType, constant},
        type_row,
        types::Term,
    };

    use super::LoadNat;

    #[test]
    fn test_load_nat() {
        let mut b = DFGBuilder::new(inout_sig(type_row![], vec![usize_t()])).unwrap();

        let arg = Term::from(4u64);
        let op = LoadNat::new(arg);

        let out = b.add_dataflow_op(op.clone(), []).unwrap();

        let result = b.finish_hugr_with_outputs(out.outputs()).unwrap();

        let exp_optype: OpType = op.into();

        for child in result.children(result.entrypoint()) {
            let node_optype = result.get_optype(child);
            // The only node in the HUGR besides Input and Output should be LoadNat.
            if !node_optype.is_input() && !node_optype.is_output() {
                assert_eq!(node_optype, &exp_optype);
            }
        }
    }

    #[test]
    fn test_load_nat_fold() {
        let arg = Term::from(5u64);
        let op = LoadNat::new(arg);

        let optype: OpType = op.into();

        if let OpType::ExtensionOp(ext_op) = optype {
            let result = ext_op.constant_fold(&[]);
            let exp_port: OutgoingPort = 0.into();
            let exp_val: constant::Value = ConstUsize::new(5).into();
            assert_eq!(result, Some(vec![(exp_port, exp_val)]));
        } else {
            panic!()
        }
    }
}
