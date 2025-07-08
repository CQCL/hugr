//! Operations for converting between the different array extensions

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

/// Array conversion direction.
///
/// Either the current array type [INTO] the other one, or the current array type [FROM] the
/// other one.
pub type Direction = bool;

/// Array conversion direction to turn the current array type [INTO] the other one.
pub const INTO: Direction = true;

/// Array conversion direction to obtain the current array type [FROM] the other one.
pub const FROM: Direction = false;

/// Definition of array conversion operations.
///
/// Generic over the concrete array implementation of the extension containing the operation, as
/// well as over another array implementation that should be converted between. Also generic over
/// the conversion [Direction].
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct GenericArrayConvertDef<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind>(
    PhantomData<AK>,
    PhantomData<OtherAK>,
);

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind>
    GenericArrayConvertDef<AK, DIR, OtherAK>
{
    /// Creates a new array conversion definition.
    #[must_use]
    pub fn new() -> Self {
        GenericArrayConvertDef(PhantomData, PhantomData)
    }
}

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind> Default
    for GenericArrayConvertDef<AK, DIR, OtherAK>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind> FromStr
    for GenericArrayConvertDef<AK, DIR, OtherAK>
{
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let def = GenericArrayConvertDef::new();
        if s == def.opdef_id() {
            Ok(def)
        } else {
            Err(())
        }
    }
}

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind>
    GenericArrayConvertDef<AK, DIR, OtherAK>
{
    /// To avoid recursion when defining the extension, take the type definition as an argument.
    fn signature_from_def(&self, array_def: &TypeDef) -> SignatureFunc {
        let params = vec![TypeParam::max_nat_type(), TypeBound::Linear.into()];
        let size = TypeArg::new_var_use(0, TypeParam::max_nat_type());
        let element_ty = Type::new_var_use(1, TypeBound::Linear);

        let this_ty = AK::instantiate_ty(array_def, size.clone(), element_ty.clone())
            .expect("Array type instantiation failed");
        let other_ty =
            OtherAK::ty_parametric(size, element_ty).expect("Array type instantiation failed");

        let sig = match DIR {
            INTO => FuncValueType::new(this_ty, other_ty),
            FROM => FuncValueType::new(other_ty, this_ty),
        };
        PolyFuncTypeRV::new(params, sig).into()
    }
}

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind> MakeOpDef
    for GenericArrayConvertDef<AK, DIR, OtherAK>
{
    fn opdef_id(&self) -> OpName {
        match DIR {
            INTO => format!("to_{}", OtherAK::TYPE_NAME).into(),
            FROM => format!("from_{}", OtherAK::TYPE_NAME).into(),
        }
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
        match DIR {
            INTO => format!("Turns `{}` into `{}`", AK::TYPE_NAME, OtherAK::TYPE_NAME),
            FROM => format!("Turns `{}` into `{}`", OtherAK::TYPE_NAME, AK::TYPE_NAME),
        }
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

/// Definition of the array conversion op.
///
/// Generic over the concrete array implementation of the extension containing the operation, as
/// well as over another array implementation that should be converted between. Also generic over
/// the conversion [Direction].
#[derive(Clone, Debug, PartialEq)]
pub struct GenericArrayConvert<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind> {
    /// The element type of the array.
    pub elem_ty: Type,
    /// Size of the array.
    pub size: u64,
    _kind: PhantomData<AK>,
    _other_kind: PhantomData<OtherAK>,
}

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind>
    GenericArrayConvert<AK, DIR, OtherAK>
{
    /// Creates a new array conversion op.
    #[must_use]
    pub fn new(elem_ty: Type, size: u64) -> Self {
        GenericArrayConvert {
            elem_ty,
            size,
            _kind: PhantomData,
            _other_kind: PhantomData,
        }
    }
}

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind> NamedOp
    for GenericArrayConvert<AK, DIR, OtherAK>
{
    fn name(&self) -> OpName {
        match DIR {
            INTO => format!("to_{}", OtherAK::TYPE_NAME).into(),
            FROM => format!("from_{}", OtherAK::TYPE_NAME).into(),
        }
    }
}

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind> MakeExtensionOp
    for GenericArrayConvert<AK, DIR, OtherAK>
{
    fn op_id(&self) -> OpName {
        GenericArrayConvertDef::<AK, DIR, OtherAK>::new().opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = GenericArrayConvertDef::<AK, DIR, OtherAK>::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![TypeArg::BoundedNat(self.size), self.elem_ty.clone().into()]
    }
}

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind> MakeRegisteredOp
    for GenericArrayConvert<AK, DIR, OtherAK>
{
    fn extension_id(&self) -> ExtensionId {
        AK::EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(AK::extension())
    }
}

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind> HasDef
    for GenericArrayConvert<AK, DIR, OtherAK>
{
    type Def = GenericArrayConvertDef<AK, DIR, OtherAK>;
}

impl<AK: ArrayKind, const DIR: Direction, OtherAK: ArrayKind> HasConcrete
    for GenericArrayConvertDef<AK, DIR, OtherAK>
{
    type Concrete = GenericArrayConvert<AK, DIR, OtherAK>;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match type_args {
            [TypeArg::BoundedNat(n), TypeArg::Runtime(ty)] => {
                Ok(GenericArrayConvert::new(ty.clone(), *n))
            }
            _ => Err(SignatureError::InvalidTypeArgs.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::extension::prelude::bool_t;
    use crate::ops::{OpTrait, OpType};
    use crate::std_extensions::collections::array::Array;
    use crate::std_extensions::collections::borrow_array::BorrowArray;
    use crate::std_extensions::collections::value_array::ValueArray;

    use super::*;

    #[rstest]
    #[case(ValueArray, Array)]
    #[case(BorrowArray, Array)]
    fn test_convert_from_def<AK: ArrayKind, OtherAK: ArrayKind>(
        #[case] _kind: AK,
        #[case] _other_kind: OtherAK,
    ) {
        let op = GenericArrayConvert::<AK, FROM, OtherAK>::new(bool_t(), 2);
        let optype: OpType = op.clone().into();
        let new_op: GenericArrayConvert<AK, FROM, OtherAK> = optype.cast().unwrap();
        assert_eq!(new_op, op);
    }

    #[rstest]
    #[case(ValueArray, Array)]
    #[case(BorrowArray, Array)]
    fn test_convert_into_def<AK: ArrayKind, OtherAK: ArrayKind>(
        #[case] _kind: AK,
        #[case] _other_kind: OtherAK,
    ) {
        let op = GenericArrayConvert::<AK, INTO, OtherAK>::new(bool_t(), 2);
        let optype: OpType = op.clone().into();
        let new_op: GenericArrayConvert<AK, INTO, OtherAK> = optype.cast().unwrap();
        assert_eq!(new_op, op);
    }

    #[rstest]
    #[case(ValueArray, Array)]
    #[case(BorrowArray, Array)]
    fn test_convert_from<AK: ArrayKind, OtherAK: ArrayKind>(
        #[case] _kind: AK,
        #[case] _other_kind: OtherAK,
    ) {
        let size = 2;
        let element_ty = bool_t();
        let op = GenericArrayConvert::<AK, FROM, OtherAK>::new(element_ty.clone(), size);
        let optype: OpType = op.into();
        let sig = optype.dataflow_signature().unwrap();
        assert_eq!(
            sig.io(),
            (
                &vec![OtherAK::ty(size, element_ty.clone())].into(),
                &vec![AK::ty(size, element_ty.clone())].into(),
            )
        );
    }

    #[rstest]
    #[case(ValueArray, Array)]
    #[case(BorrowArray, Array)]
    fn test_convert_into<AK: ArrayKind, OtherAK: ArrayKind>(
        #[case] _kind: AK,
        #[case] _other_kind: OtherAK,
    ) {
        let size = 2;
        let element_ty = bool_t();
        let op = GenericArrayConvert::<AK, INTO, OtherAK>::new(element_ty.clone(), size);
        let optype: OpType = op.into();
        let sig = optype.dataflow_signature().unwrap();
        assert_eq!(
            sig.io(),
            (
                &vec![AK::ty(size, element_ty.clone())].into(),
                &vec![OtherAK::ty(size, element_ty.clone())].into(),
            )
        );
    }
}
