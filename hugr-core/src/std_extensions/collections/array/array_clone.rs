//! Definition of the array clone operation.

use std::marker::PhantomData;
use std::str::FromStr;
use std::sync::{Arc, Weak};

use crate::Extension;
use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{ExtensionId, OpDef, SignatureError, SignatureFunc, TypeDef};
use crate::ops::{ExtensionOp, NamedOp, OpName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{FuncValueType, PolyFuncTypeRV, Type, TypeBound};

use super::array_kind::ArrayKind;

/// Name of the operation to clone an array
pub const ARRAY_CLONE_OP_ID: OpName = OpName::new_inline("clone");

/// Definition of the array clone operation. Generic over the concrete array implementation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct GenericArrayCloneDef<AK: ArrayKind>(PhantomData<AK>);

impl<AK: ArrayKind> GenericArrayCloneDef<AK> {
    /// Creates a new clone operation definition.
    #[must_use]
    pub fn new() -> Self {
        GenericArrayCloneDef(PhantomData)
    }
}

impl<AK: ArrayKind> Default for GenericArrayCloneDef<AK> {
    fn default() -> Self {
        Self::new()
    }
}

impl<AK: ArrayKind> FromStr for GenericArrayCloneDef<AK> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == ARRAY_CLONE_OP_ID {
            Ok(GenericArrayCloneDef::new())
        } else {
            Err(())
        }
    }
}

impl<AK: ArrayKind> GenericArrayCloneDef<AK> {
    /// To avoid recursion when defining the extension, take the type definition as an argument.
    fn signature_from_def(&self, array_def: &TypeDef) -> SignatureFunc {
        let params = vec![TypeParam::max_nat_type(), TypeBound::Copyable.into()];
        let size = TypeArg::new_var_use(0, TypeParam::max_nat_type());
        let element_ty = Type::new_var_use(1, TypeBound::Copyable);
        let array_ty = AK::instantiate_ty(array_def, size, element_ty)
            .expect("Array type instantiation failed");
        PolyFuncTypeRV::new(
            params,
            FuncValueType::new(array_ty.clone(), vec![array_ty; 2]),
        )
        .into()
    }
}

impl<AK: ArrayKind> MakeOpDef for GenericArrayCloneDef<AK> {
    fn opdef_id(&self) -> OpName {
        ARRAY_CLONE_OP_ID
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
        "Clones an array with copyable elements".into()
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

/// Definition of the array clone op. Generic over the concrete array implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct GenericArrayClone<AK: ArrayKind> {
    /// The element type of the array.
    pub elem_ty: Type,
    /// Size of the array.
    pub size: u64,
    _kind: PhantomData<AK>,
}

impl<AK: ArrayKind> GenericArrayClone<AK> {
    /// Creates a new array clone op.
    ///
    /// # Errors
    ///
    /// If the provided element type is not copyable.
    pub fn new(elem_ty: Type, size: u64) -> Result<Self, OpLoadError> {
        elem_ty
            .copyable()
            .then_some(GenericArrayClone {
                elem_ty,
                size,
                _kind: PhantomData,
            })
            .ok_or(SignatureError::InvalidTypeArgs.into())
    }
}

impl<AK: ArrayKind> NamedOp for GenericArrayClone<AK> {
    fn name(&self) -> OpName {
        ARRAY_CLONE_OP_ID
    }
}

impl<AK: ArrayKind> MakeExtensionOp for GenericArrayClone<AK> {
    fn op_id(&self) -> OpName {
        GenericArrayCloneDef::<AK>::default().opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = GenericArrayCloneDef::<AK>::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![self.size.into(), self.elem_ty.clone().into()]
    }
}

impl<AK: ArrayKind> MakeRegisteredOp for GenericArrayClone<AK> {
    fn extension_id(&self) -> ExtensionId {
        AK::EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(AK::extension())
    }
}

impl<AK: ArrayKind> HasDef for GenericArrayClone<AK> {
    type Def = GenericArrayCloneDef<AK>;
}

impl<AK: ArrayKind> HasConcrete for GenericArrayCloneDef<AK> {
    type Concrete = GenericArrayClone<AK>;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match type_args {
            [TypeArg::BoundedNat(n), TypeArg::Runtime(ty)] if ty.copyable() => {
                Ok(GenericArrayClone::new(ty.clone(), *n).unwrap())
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
    fn test_clone_def<AK: ArrayKind>(#[case] _kind: AK) {
        let op = GenericArrayClone::<AK>::new(bool_t(), 2).unwrap();
        let optype: OpType = op.clone().into();
        let new_op: GenericArrayClone<AK> = optype.cast().unwrap();
        assert_eq!(new_op, op);

        assert_eq!(
            GenericArrayClone::<AK>::new(qb_t(), 2),
            Err(OpLoadError::InvalidArgs(SignatureError::InvalidTypeArgs))
        );
    }

    #[rstest]
    #[case(Array)]
    #[case(BorrowArray)]
    fn test_clone<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 2;
        let element_ty = bool_t();
        let op = GenericArrayClone::<AK>::new(element_ty.clone(), size).unwrap();
        let optype: OpType = op.into();
        let sig = optype.dataflow_signature().unwrap();
        assert_eq!(
            sig.io(),
            (
                &vec![AK::ty(size, element_ty.clone())].into(),
                &vec![AK::ty(size, element_ty.clone()); 2].into(),
            )
        );
    }
}
