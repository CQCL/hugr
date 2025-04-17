//! Definition of the array repeat operation.

use std::marker::PhantomData;
use std::str::FromStr;
use std::sync::{Arc, Weak};

use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{ExtensionId, ExtensionSet, OpDef, SignatureError, SignatureFunc, TypeDef};
use crate::ops::{ExtensionOp, NamedOp, OpName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{FuncValueType, PolyFuncTypeRV, Signature, Type, TypeBound};
use crate::Extension;

use super::array_kind::ArrayKind;

/// Name of the operation to repeat a value multiple times
pub const ARRAY_REPEAT_OP_ID: OpName = OpName::new_inline("repeat");

/// Definition of the array repeat op. Generic over the concrete array implementation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct GenericArrayRepeatDef<AK: ArrayKind>(PhantomData<AK>);

impl<AK: ArrayKind> GenericArrayRepeatDef<AK> {
    /// Creates a new array repeat operation definition.
    pub fn new() -> Self {
        GenericArrayRepeatDef(PhantomData)
    }
}

impl<AK: ArrayKind> Default for GenericArrayRepeatDef<AK> {
    fn default() -> Self {
        Self::new()
    }
}

impl<AK: ArrayKind> NamedOp for GenericArrayRepeatDef<AK> {
    fn name(&self) -> OpName {
        ARRAY_REPEAT_OP_ID
    }
}

impl<AK: ArrayKind> FromStr for GenericArrayRepeatDef<AK> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == ARRAY_REPEAT_OP_ID {
            Ok(GenericArrayRepeatDef::new())
        } else {
            Err(())
        }
    }
}

impl<AK: ArrayKind> GenericArrayRepeatDef<AK> {
    /// To avoid recursion when defining the extension, take the type definition as an argument.
    fn signature_from_def(&self, array_def: &TypeDef) -> SignatureFunc {
        let params = vec![
            TypeParam::max_nat(),
            TypeBound::Any.into(),
            TypeParam::Extensions,
        ];
        let n = TypeArg::new_var_use(0, TypeParam::max_nat());
        let t = Type::new_var_use(1, TypeBound::Any);
        let es = ExtensionSet::type_var(2);
        let func =
            Type::new_function(Signature::new(vec![], vec![t.clone()]).with_extension_delta(es));
        let array_ty =
            AK::instantiate_ty(array_def, n, t).expect("Array type instantiation failed");
        PolyFuncTypeRV::new(params, FuncValueType::new(vec![func], array_ty)).into()
    }
}

impl<AK: ArrayKind> MakeOpDef for GenericArrayRepeatDef<AK> {
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
        let sig = self.signature_from_def(extension.get_type(&AK::TYPE_NAME).unwrap());
        let def = extension.add_op(self.name(), self.description(), sig, extension_ref)?;

        self.post_opdef(def);

        Ok(())
    }
}

/// Definition of the array repeat op. Generic over the concrete array implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct GenericArrayRepeat<AK: ArrayKind> {
    /// The element type of the resulting array.
    pub elem_ty: Type,
    /// Size of the array.
    pub size: u64,
    /// The extensions required by the function that generates the array elements.
    pub extension_reqs: ExtensionSet,
    _kind: PhantomData<AK>,
}

impl<AK: ArrayKind> GenericArrayRepeat<AK> {
    /// Creates a new array repeat op.
    pub fn new(elem_ty: Type, size: u64, extension_reqs: ExtensionSet) -> Self {
        GenericArrayRepeat {
            elem_ty,
            size,
            extension_reqs,
            _kind: PhantomData,
        }
    }
}

impl<AK: ArrayKind> NamedOp for GenericArrayRepeat<AK> {
    fn name(&self) -> OpName {
        ARRAY_REPEAT_OP_ID
    }
}

impl<AK: ArrayKind> MakeExtensionOp for GenericArrayRepeat<AK> {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = GenericArrayRepeatDef::<AK>::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![
            TypeArg::BoundedNat { n: self.size },
            self.elem_ty.clone().into(),
            TypeArg::Extensions {
                es: self.extension_reqs.clone(),
            },
        ]
    }
}

impl<AK: ArrayKind> MakeRegisteredOp for GenericArrayRepeat<AK> {
    fn extension_id(&self) -> ExtensionId {
        AK::EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(AK::extension())
    }
}

impl<AK: ArrayKind> HasDef for GenericArrayRepeat<AK> {
    type Def = GenericArrayRepeatDef<AK>;
}

impl<AK: ArrayKind> HasConcrete for GenericArrayRepeatDef<AK> {
    type Concrete = GenericArrayRepeat<AK>;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match type_args {
            [TypeArg::BoundedNat { n }, TypeArg::Type { ty }, TypeArg::Extensions { es }] => {
                Ok(GenericArrayRepeat::new(ty.clone(), *n, es.clone()))
            }
            _ => Err(SignatureError::InvalidTypeArgs.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::std_extensions::collections::array::{Array, EXTENSION_ID};
    use crate::std_extensions::collections::value_array::ValueArray;
    use crate::{
        extension::prelude::qb_t,
        ops::{OpTrait, OpType},
        types::Signature,
    };

    use super::*;

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    fn test_repeat_def<AK: ArrayKind>(#[case] _kind: AK) {
        let op = GenericArrayRepeat::<AK>::new(qb_t(), 2, ExtensionSet::singleton(EXTENSION_ID));
        let optype: OpType = op.clone().into();
        let new_op: GenericArrayRepeat<AK> = optype.cast().unwrap();
        assert_eq!(new_op, op);
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    fn test_repeat<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 2;
        let element_ty = qb_t();
        let es = ExtensionSet::singleton(EXTENSION_ID);
        let op = GenericArrayRepeat::<AK>::new(element_ty.clone(), size, es.clone());

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![Type::new_function(
                    Signature::new(vec![], vec![qb_t()]).with_extension_delta(es)
                )]
                .into(),
                &vec![AK::ty(size, element_ty.clone())].into(),
            )
        );
    }
}
