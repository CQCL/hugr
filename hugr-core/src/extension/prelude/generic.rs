use std::str::FromStr;
use std::sync::{Arc, Weak};

use crate::extension::prelude::usize_custom_t;
use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::ExtensionId;
use crate::extension::OpDef;
use crate::extension::SignatureFunc;
use crate::ops::ExtensionOp;
use crate::ops::NamedOp;
use crate::ops::OpName;
use crate::type_row;
use crate::types::FuncValueType;

use crate::types::Type;

use crate::extension::SignatureError;

use crate::types::PolyFuncTypeRV;

use crate::types::type_param::TypeArg;
use crate::Extension;

use super::PRELUDE_ID;
use super::{PRELUDE, PRELUDE_REGISTRY};
use crate::types::type_param::TypeParam;

/// Name of the operation for loading generic BoundedNat parameters.
pub const LOAD_NAT_OP_ID: OpName = OpName::new_inline("load_nat");

/// Definition of the load nat operation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct LoadNatDef;

impl NamedOp for LoadNatDef {
    fn name(&self) -> OpName {
        LOAD_NAT_OP_ID
    }
}

impl FromStr for LoadNatDef {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == LoadNatDef.name() {
            Ok(Self)
        } else {
            Err(())
        }
    }
}

impl MakeOpDef for LoadNatDef {
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        let usize_t: Type = usize_custom_t(_extension_ref).into();
        let params = vec![TypeParam::max_nat()];
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
}

/// Concrete load nat operation.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadNat {
    nat: TypeArg,
}

impl LoadNat {
    fn new(nat: TypeArg) -> Self {
        LoadNat { nat }
    }
}

impl NamedOp for LoadNat {
    fn name(&self) -> OpName {
        LOAD_NAT_OP_ID
    }
}

impl MakeExtensionOp for LoadNat {
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

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &PRELUDE_REGISTRY
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
