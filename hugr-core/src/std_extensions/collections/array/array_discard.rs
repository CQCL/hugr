//! Definition of the array discard operation.

use std::marker::PhantomData;
use std::str::FromStr;
use std::sync::{Arc, Weak};

use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{ExtensionId, OpDef, SignatureError, SignatureFunc, TypeDef};
use crate::ops::{ExtensionOp, OpName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{FuncValueType, PolyFuncTypeRV, Type, TypeBound};
use crate::{Extension, type_row};

use super::array_kind::ArrayKind;

/// Name of the operation to discard an array
pub const ARRAY_DISCARD_OP_ID: OpName = OpName::new_inline("discard");

/// Definition of the array discard op. Generic over the concrete array implementation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct GenericArrayDiscardDef<AK: ArrayKind>(PhantomData<AK>);

impl<AK: ArrayKind> GenericArrayDiscardDef<AK> {
    /// Creates a new array discard operation definition.
    #[must_use]
    pub fn new() -> Self {
        GenericArrayDiscardDef(PhantomData)
    }
}

impl<AK: ArrayKind> Default for GenericArrayDiscardDef<AK> {
    fn default() -> Self {
        Self::new()
    }
}

impl<AK: ArrayKind> FromStr for GenericArrayDiscardDef<AK> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == ARRAY_DISCARD_OP_ID {
            Ok(GenericArrayDiscardDef::new())
        } else {
            Err(())
        }
    }
}

impl<AK: ArrayKind> GenericArrayDiscardDef<AK> {
    /// To avoid recursion when defining the extension, take the type definition as an argument.
    fn signature_from_def(&self, array_def: &TypeDef) -> SignatureFunc {
        let params = vec![TypeParam::max_nat_type(), TypeBound::Copyable.into()];
        let size = TypeArg::new_var_use(0, TypeParam::max_nat_type());
        let element_ty = Type::new_var_use(1, TypeBound::Copyable);
        let array_ty = AK::instantiate_ty(array_def, size, element_ty)
            .expect("Array type instantiation failed");
        PolyFuncTypeRV::new(params, FuncValueType::new(array_ty, type_row![])).into()
    }
}

impl<AK: ArrayKind> MakeOpDef for GenericArrayDiscardDef<AK> {
    fn opdef_id(&self) -> OpName {
        ARRAY_DISCARD_OP_ID
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        self.signature_from_def(AK::type_def())
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(AK::extension())
    }

    fn extension(&self) -> ExtensionId {
        AK::EXTENSION_ID
    }

    fn description(&self) -> String {
        "Discards an array with copyable elements".into()
    }

    /// Add an operation implemented as a [`MakeOpDef`], which can provide the data
    /// required to define an [`OpDef`], to an extension.
    //
    // This method is re-defined here since we need to pass the array type def while
    // computing the signature, to avoid recursive loops initializing the extension.
    fn add_to_extension(
        &self,
        extension: &mut Extension,
        extension_ref: &Weak<Extension>,
    ) -> Result<(), crate::extension::ExtensionBuildError> {
        let sig = self.signature_from_def(extension.get_type(&AK::TYPE_NAME).unwrap());
        let def = extension.add_op(self.opdef_id(), self.description(), sig, extension_ref)?;
        self.post_opdef(def);
        Ok(())
    }
}

/// Definition of the array discard op. Generic over the concrete array implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct GenericArrayDiscard<AK: ArrayKind> {
    /// The element type of the array.
    pub elem_ty: Type,
    /// Size of the array.
    pub size: u64,
    _kind: PhantomData<AK>,
}

impl<AK: ArrayKind> GenericArrayDiscard<AK> {
    /// Creates a new array discard op.
    #[must_use]
    pub fn new(elem_ty: Type, size: u64) -> Option<Self> {
        elem_ty.copyable().then_some(GenericArrayDiscard {
            elem_ty,
            size,
            _kind: PhantomData,
        })
    }
}

impl<AK: ArrayKind> MakeExtensionOp for GenericArrayDiscard<AK> {
    fn op_id(&self) -> OpName {
        GenericArrayDiscardDef::<AK>::default().opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = GenericArrayDiscardDef::<AK>::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![self.size.into(), self.elem_ty.clone().into()]
    }
}

impl<AK: ArrayKind> MakeRegisteredOp for GenericArrayDiscard<AK> {
    fn extension_id(&self) -> ExtensionId {
        AK::EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(AK::extension())
    }
}

impl<AK: ArrayKind> HasDef for GenericArrayDiscard<AK> {
    type Def = GenericArrayDiscardDef<AK>;
}

impl<AK: ArrayKind> HasConcrete for GenericArrayDiscardDef<AK> {
    type Concrete = GenericArrayDiscard<AK>;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match type_args {
            [TypeArg::BoundedNat(n), TypeArg::Runtime(ty)] if ty.copyable() => {
                Ok(GenericArrayDiscard::new(ty.clone(), *n).unwrap())
            }
            _ => Err(SignatureError::InvalidTypeArgs.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::extension::prelude::bool_t;
    use crate::std_extensions::collections::array::Array;
    use crate::std_extensions::collections::borrow_array::BorrowArray;
    use crate::{
        extension::prelude::qb_t,
        ops::{OpTrait, OpType},
    };

    use super::*;

    #[rstest]
    #[case(Array)]
    #[case(BorrowArray)]
    fn test_discard_def<AK: ArrayKind>(#[case] _kind: AK) {
        let op = GenericArrayDiscard::<AK>::new(bool_t(), 2).unwrap();
        let optype: OpType = op.clone().into();
        let new_op: GenericArrayDiscard<AK> = optype.cast().unwrap();
        assert_eq!(new_op, op);

        assert_eq!(GenericArrayDiscard::<AK>::new(qb_t(), 2), None);
    }

    #[rstest]
    #[case(Array)]
    #[case(BorrowArray)]
    fn test_discard<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 2;
        let element_ty = bool_t();
        let op = GenericArrayDiscard::<AK>::new(element_ty.clone(), size).unwrap();
        let optype: OpType = op.into();
        let sig = optype.dataflow_signature().unwrap();
        assert_eq!(
            sig.io(),
            (
                &vec![AK::ty(size, element_ty.clone())].into(),
                &vec![].into(),
            )
        );
    }
}
