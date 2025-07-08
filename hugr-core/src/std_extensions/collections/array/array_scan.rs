//! Array scanning operation

use std::marker::PhantomData;
use std::str::FromStr;
use std::sync::{Arc, Weak};

use itertools::Itertools;

use crate::Extension;
use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{ExtensionId, OpDef, SignatureError, SignatureFunc, TypeDef};
use crate::ops::{ExtensionOp, OpName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{FuncTypeBase, PolyFuncTypeRV, RowVariable, Type, TypeBound, TypeRV};

use super::array_kind::ArrayKind;

/// Name of the operation for the combined map/fold operation
pub const ARRAY_SCAN_OP_ID: OpName = OpName::new_inline("scan");

/// Definition of the array scan op. Generic over the concrete array implementation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct GenericArrayScanDef<AK: ArrayKind>(PhantomData<AK>);

impl<AK: ArrayKind> GenericArrayScanDef<AK> {
    /// Creates a new array scan operation definition.
    #[must_use]
    pub fn new() -> Self {
        GenericArrayScanDef(PhantomData)
    }
}

impl<AK: ArrayKind> Default for GenericArrayScanDef<AK> {
    fn default() -> Self {
        Self::new()
    }
}

impl<AK: ArrayKind> FromStr for GenericArrayScanDef<AK> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == ARRAY_SCAN_OP_ID {
            Ok(Self::new())
        } else {
            Err(())
        }
    }
}

impl<AK: ArrayKind> GenericArrayScanDef<AK> {
    /// To avoid recursion when defining the extension, take the type definition
    /// and a reference to the extension as an argument.
    fn signature_from_def(&self, array_def: &TypeDef) -> SignatureFunc {
        // array<N, T1>, (T1, *A -> T2, *A), *A, -> array<N, T2>, *A
        let params = vec![
            TypeParam::max_nat_type(),
            TypeBound::Linear.into(),
            TypeBound::Linear.into(),
            TypeParam::new_list_type(TypeBound::Linear),
        ];
        let n = TypeArg::new_var_use(0, TypeParam::max_nat_type());
        let t1 = Type::new_var_use(1, TypeBound::Linear);
        let t2 = Type::new_var_use(2, TypeBound::Linear);
        let s = TypeRV::new_row_var_use(3, TypeBound::Linear);
        PolyFuncTypeRV::new(
            params,
            FuncTypeBase::<RowVariable>::new(
                vec![
                    AK::instantiate_ty(array_def, n.clone(), t1.clone())
                        .expect("Array type instantiation failed")
                        .into(),
                    Type::new_function(FuncTypeBase::<RowVariable>::new(
                        vec![t1.into(), s.clone()],
                        vec![t2.clone().into(), s.clone()],
                    ))
                    .into(),
                    s.clone(),
                ],
                vec![
                    AK::instantiate_ty(array_def, n, t2)
                        .expect("Array type instantiation failed")
                        .into(),
                    s,
                ],
            ),
        )
        .into()
    }
}

impl<AK: ArrayKind> MakeOpDef for GenericArrayScanDef<AK> {
    fn opdef_id(&self) -> OpName {
        ARRAY_SCAN_OP_ID
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
        "A combination of map and foldl. Applies a function to each element \
        of the array with an accumulator that is passed through from start to \
        finish. Returns the resulting array and the final state of the \
        accumulator."
            .into()
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
        let def = extension.add_op(self.op_id(), self.description(), sig, extension_ref)?;

        self.post_opdef(def);

        Ok(())
    }
}

/// Definition of the array scan op. Generic over the concrete array implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct GenericArrayScan<AK: ArrayKind> {
    /// The element type of the input array.
    pub src_ty: Type,
    /// The target element type of the output array.
    pub tgt_ty: Type,
    /// The accumulator types.
    pub acc_tys: Vec<Type>,
    /// Size of the array.
    pub size: u64,
    _kind: PhantomData<AK>,
}

impl<AK: ArrayKind> GenericArrayScan<AK> {
    /// Creates a new array scan op.
    #[must_use]
    pub fn new(src_ty: Type, tgt_ty: Type, acc_tys: Vec<Type>, size: u64) -> Self {
        GenericArrayScan {
            src_ty,
            tgt_ty,
            acc_tys,
            size,
            _kind: PhantomData,
        }
    }
}

impl<AK: ArrayKind> MakeExtensionOp for GenericArrayScan<AK> {
    fn op_id(&self) -> OpName {
        GenericArrayScanDef::<AK>::default().opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = GenericArrayScanDef::<AK>::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![
            self.size.into(),
            self.src_ty.clone().into(),
            self.tgt_ty.clone().into(),
            TypeArg::new_list(self.acc_tys.clone().into_iter().map_into()),
        ]
    }
}

impl<AK: ArrayKind> MakeRegisteredOp for GenericArrayScan<AK> {
    fn extension_id(&self) -> ExtensionId {
        AK::EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(AK::extension())
    }
}

impl<AK: ArrayKind> HasDef for GenericArrayScan<AK> {
    type Def = GenericArrayScanDef<AK>;
}

impl<AK: ArrayKind> HasConcrete for GenericArrayScanDef<AK> {
    type Concrete = GenericArrayScan<AK>;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match type_args {
            [
                TypeArg::BoundedNat(n),
                TypeArg::Runtime(src_ty),
                TypeArg::Runtime(tgt_ty),
                TypeArg::List(acc_tys),
            ] => {
                let acc_tys: Result<_, OpLoadError> = acc_tys
                    .iter()
                    .map(|acc_ty| match acc_ty {
                        TypeArg::Runtime(ty) => Ok(ty.clone()),
                        _ => Err(SignatureError::InvalidTypeArgs.into()),
                    })
                    .collect();
                Ok(GenericArrayScan::new(
                    src_ty.clone(),
                    tgt_ty.clone(),
                    acc_tys?,
                    *n,
                ))
            }
            _ => Err(SignatureError::InvalidTypeArgs.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::extension::prelude::usize_t;
    use crate::std_extensions::collections::array::Array;
    use crate::std_extensions::collections::borrow_array::BorrowArray;
    use crate::std_extensions::collections::value_array::ValueArray;
    use crate::{
        extension::prelude::{bool_t, qb_t},
        ops::{OpTrait, OpType},
        types::Signature,
    };

    use super::*;

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    fn test_scan_def<AK: ArrayKind>(#[case] _kind: AK) {
        let op = GenericArrayScan::<AK>::new(bool_t(), qb_t(), vec![usize_t()], 2);
        let optype: OpType = op.clone().into();
        let new_op: GenericArrayScan<AK> = optype.cast().unwrap();
        assert_eq!(new_op, op);
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    fn test_scan_map<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 2;
        let src_ty = qb_t();
        let tgt_ty = bool_t();

        let op = GenericArrayScan::<AK>::new(src_ty.clone(), tgt_ty.clone(), vec![], size);
        let optype: OpType = op.into();
        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![
                    AK::ty(size, src_ty.clone()),
                    Type::new_function(Signature::new(vec![src_ty], vec![tgt_ty.clone()]))
                ]
                .into(),
                &vec![AK::ty(size, tgt_ty)].into(),
            )
        );
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    fn test_scan_accs<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 2;
        let src_ty = qb_t();
        let tgt_ty = bool_t();
        let acc_ty1 = usize_t();
        let acc_ty2 = qb_t();

        let op = GenericArrayScan::<AK>::new(
            src_ty.clone(),
            tgt_ty.clone(),
            vec![acc_ty1.clone(), acc_ty2.clone()],
            size,
        );
        let optype: OpType = op.into();
        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![
                    AK::ty(size, src_ty.clone()),
                    Type::new_function(Signature::new(
                        vec![src_ty, acc_ty1.clone(), acc_ty2.clone()],
                        vec![tgt_ty.clone(), acc_ty1.clone(), acc_ty2.clone()]
                    )),
                    acc_ty1.clone(),
                    acc_ty2.clone()
                ]
                .into(),
                &vec![AK::ty(size, tgt_ty), acc_ty1, acc_ty2].into(),
            )
        );
    }
}
