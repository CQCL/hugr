//! Definition of the array repeat operation.

use std::str::FromStr;
use std::sync::{Arc, Weak};

use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{ExtensionId, OpDef, SignatureError, SignatureFunc, TypeDef};
use crate::ops::{ExtensionOp, NamedOp, OpName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{FuncValueType, PolyFuncTypeRV, Signature, Type, TypeBound};
use crate::Extension;

use super::{array_type_def, instantiate_array, ARRAY_TYPENAME};

/// Name of the operation to repeat a value multiple times
pub const ARRAY_REPEAT_OP_ID: OpName = OpName::new_inline("repeat");

/// Definition of the array repeat op.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ArrayRepeatDef;

impl NamedOp for ArrayRepeatDef {
    fn name(&self) -> OpName {
        ARRAY_REPEAT_OP_ID
    }
}

impl FromStr for ArrayRepeatDef {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == ArrayRepeatDef.name() {
            Ok(Self)
        } else {
            Err(())
        }
    }
}

impl ArrayRepeatDef {
    /// To avoid recursion when defining the extension, take the type definition as an argument.
    fn signature_from_def(&self, array_def: &TypeDef) -> SignatureFunc {
        let params = vec![TypeParam::max_nat(), TypeBound::Any.into()];
        let n = TypeArg::new_var_use(0, TypeParam::max_nat());
        let t = Type::new_var_use(1, TypeBound::Any);
        let func = Type::new_function(Signature::new(vec![], vec![t.clone()]));
        let array_ty = instantiate_array(array_def, n, t).expect("Array type instantiation failed");
        PolyFuncTypeRV::new(params, FuncValueType::new(vec![func], array_ty)).into()
    }
}

impl MakeOpDef for ArrayRepeatDef {
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        self.signature_from_def(array_type_def())
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&super::EXTENSION)
    }

    fn extension(&self) -> ExtensionId {
        super::EXTENSION_ID
    }

    fn description(&self) -> String {
        "Creates a new array whose elements are initialised by calling \
        the given function n times"
            .into()
    }

    /// Add an operation implemented as a [MakeOpDef], which can provide the data
    /// required to define an [OpDef], to an extension.
    //
    // This method is re-defined here since we need to pass the array type def while
    // computing the signature, to avoid recursive loops initializing the extension.
    fn add_to_extension(
        &self,
        extension: &mut Extension,
        extension_ref: &Weak<Extension>,
    ) -> Result<(), crate::extension::ExtensionBuildError> {
        let sig = self.signature_from_def(extension.get_type(&ARRAY_TYPENAME).unwrap());
        let def = extension.add_op(self.name(), self.description(), sig, extension_ref)?;

        self.post_opdef(def);

        Ok(())
    }
}

/// Definition of the array repeat op.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayRepeat {
    /// The element type of the resulting array.
    pub elem_ty: Type,
    /// Size of the array.
    pub size: u64,
}

impl ArrayRepeat {
    /// Creates a new array repeat op.
    pub fn new(elem_ty: Type, size: u64) -> Self {
        ArrayRepeat { elem_ty, size }
    }
}

impl NamedOp for ArrayRepeat {
    fn name(&self) -> OpName {
        ARRAY_REPEAT_OP_ID
    }
}

impl MakeExtensionOp for ArrayRepeat {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = ArrayRepeatDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![
            TypeArg::BoundedNat { n: self.size },
            self.elem_ty.clone().into(),
        ]
    }
}

impl MakeRegisteredOp for ArrayRepeat {
    fn extension_id(&self) -> ExtensionId {
        super::EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&super::EXTENSION)
    }
}

impl HasDef for ArrayRepeat {
    type Def = ArrayRepeatDef;
}

impl HasConcrete for ArrayRepeatDef {
    type Concrete = ArrayRepeat;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match type_args {
            [TypeArg::BoundedNat { n }, TypeArg::Type { ty }] => {
                Ok(ArrayRepeat::new(ty.clone(), *n))
            }
            _ => Err(SignatureError::InvalidTypeArgs.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::std_extensions::collections::array::array_type;
    use crate::{
        extension::prelude::qb_t,
        ops::{OpTrait, OpType},
        types::Signature,
    };

    use super::*;

    #[test]
    fn test_repeat_def() {
        let op = ArrayRepeat::new(qb_t(), 2);
        let optype: OpType = op.clone().into();
        let new_op: ArrayRepeat = optype.cast().unwrap();
        assert_eq!(new_op, op);
    }

    #[test]
    fn test_repeat() {
        let size = 2;
        let element_ty = qb_t();
        let op = ArrayRepeat::new(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![Type::new_function(Signature::new(vec![], vec![qb_t()]))].into(),
                &vec![array_type(size, element_ty.clone())].into(),
            )
        );
    }
}
