//! Array scanning operation

use std::str::FromStr;
use std::sync::{Arc, Weak};

use itertools::Itertools;

use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{ExtensionId, ExtensionSet, OpDef, SignatureError, SignatureFunc, TypeDef};
use crate::ops::{ExtensionOp, NamedOp, OpName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{FuncTypeBase, PolyFuncTypeRV, RowVariable, Type, TypeBound, TypeRV};
use crate::Extension;

use super::{array_type_def, instantiate_array, ARRAY_REGISTRY, ARRAY_TYPENAME};

/// Name of the operation for the combined map/fold operation
pub const ARRAY_SCAN_OP_ID: OpName = OpName::new_inline("scan");

/// Definition of the array scan op.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ArrayScanDef;

impl NamedOp for ArrayScanDef {
    fn name(&self) -> OpName {
        ARRAY_SCAN_OP_ID
    }
}

impl FromStr for ArrayScanDef {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == ArrayScanDef.name() {
            Ok(Self)
        } else {
            Err(())
        }
    }
}

impl ArrayScanDef {
    /// To avoid recursion when defining the extension, take the type definition
    /// and a reference to the extension as an argument.
    fn signature_from_def(&self, array_def: &TypeDef) -> SignatureFunc {
        // array<N, T1>, (T1, *A -> T2, *A), *A, -> array<N, T2>, *A
        let params = vec![
            TypeParam::max_nat(),
            TypeBound::Any.into(),
            TypeBound::Any.into(),
            TypeParam::new_list(TypeBound::Any),
            TypeParam::Extensions,
        ];
        let n = TypeArg::new_var_use(0, TypeParam::max_nat());
        let t1 = Type::new_var_use(1, TypeBound::Any);
        let t2 = Type::new_var_use(2, TypeBound::Any);
        let s = TypeRV::new_row_var_use(3, TypeBound::Any);
        let es = ExtensionSet::type_var(4);
        PolyFuncTypeRV::new(
            params,
            FuncTypeBase::<RowVariable>::new(
                vec![
                    instantiate_array(array_def, n.clone(), t1.clone())
                        .expect("Array type instantiation failed")
                        .into(),
                    Type::new_function(
                        FuncTypeBase::<RowVariable>::new(
                            vec![t1.into(), s.clone()],
                            vec![t2.clone().into(), s.clone()],
                        )
                        .with_extension_delta(es),
                    )
                    .into(),
                    s.clone(),
                ],
                vec![
                    instantiate_array(array_def, n, t2)
                        .expect("Array type instantiation failed")
                        .into(),
                    s,
                ],
            ),
        )
        .into()
    }
}

impl MakeOpDef for ArrayScanDef {
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
        "A combination of map and foldl. Applies a function to each element \
        of the array with an accumulator that is passed through from start to \
        finish. Returns the resulting array and the final state of the \
        accumulator."
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

/// Definition of the array scan op.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayScan {
    /// The element type of the input array.
    pub src_ty: Type,
    /// The target element type of the output array.
    pub tgt_ty: Type,
    /// The accumulator types.
    pub acc_tys: Vec<Type>,
    /// Size of the array.
    pub size: u64,
    /// The extensions required by the scan function.
    pub extension_reqs: ExtensionSet,
}

impl ArrayScan {
    /// Creates a new array scan op.
    pub fn new(
        src_ty: Type,
        tgt_ty: Type,
        acc_tys: Vec<Type>,
        size: u64,
        extension_reqs: ExtensionSet,
    ) -> Self {
        ArrayScan {
            src_ty,
            tgt_ty,
            acc_tys,
            size,
            extension_reqs,
        }
    }
}

impl NamedOp for ArrayScan {
    fn name(&self) -> OpName {
        ARRAY_SCAN_OP_ID
    }
}

impl MakeExtensionOp for ArrayScan {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = ArrayScanDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![
            TypeArg::BoundedNat { n: self.size },
            self.src_ty.clone().into(),
            self.tgt_ty.clone().into(),
            TypeArg::Sequence {
                elems: self.acc_tys.clone().into_iter().map_into().collect(),
            },
            TypeArg::Extensions {
                es: self.extension_reqs.clone(),
            },
        ]
    }
}

impl MakeRegisteredOp for ArrayScan {
    fn extension_id(&self) -> ExtensionId {
        super::EXTENSION_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &ARRAY_REGISTRY
    }
}

impl HasDef for ArrayScan {
    type Def = ArrayScanDef;
}

impl HasConcrete for ArrayScanDef {
    type Concrete = ArrayScan;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match type_args {
            [TypeArg::BoundedNat { n }, TypeArg::Type { ty: src_ty }, TypeArg::Type { ty: tgt_ty }, TypeArg::Sequence { elems: acc_tys }, TypeArg::Extensions { es }] =>
            {
                let acc_tys: Result<_, OpLoadError> = acc_tys
                    .iter()
                    .map(|acc_ty| match acc_ty {
                        TypeArg::Type { ty } => Ok(ty.clone()),
                        _ => Err(SignatureError::InvalidTypeArgs.into()),
                    })
                    .collect();
                Ok(ArrayScan::new(
                    src_ty.clone(),
                    tgt_ty.clone(),
                    acc_tys?,
                    *n,
                    es.clone(),
                ))
            }
            _ => Err(SignatureError::InvalidTypeArgs.into()),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::extension::prelude::usize_t;
    use crate::std_extensions::collections::array::{array_type, EXTENSION_ID};
    use crate::{
        extension::prelude::{bool_t, qb_t},
        ops::{OpTrait, OpType},
        types::Signature,
    };

    use super::*;

    #[test]
    fn test_scan_def() {
        let op = ArrayScan::new(
            bool_t(),
            qb_t(),
            vec![usize_t()],
            2,
            ExtensionSet::singleton(EXTENSION_ID),
        );
        let optype: OpType = op.clone().into();
        let new_op: ArrayScan = optype.cast().unwrap();
        assert_eq!(new_op, op);
    }

    #[test]
    fn test_scan_map() {
        let size = 2;
        let src_ty = qb_t();
        let tgt_ty = bool_t();
        let es = ExtensionSet::singleton(EXTENSION_ID);

        let op = ArrayScan::new(src_ty.clone(), tgt_ty.clone(), vec![], size, es.clone());
        let optype: OpType = op.into();
        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![
                    array_type(size, src_ty.clone()),
                    Type::new_function(
                        Signature::new(vec![src_ty], vec![tgt_ty.clone()]).with_extension_delta(es)
                    )
                ]
                .into(),
                &vec![array_type(size, tgt_ty)].into(),
            )
        );
    }

    #[test]
    fn test_scan_accs() {
        let size = 2;
        let src_ty = qb_t();
        let tgt_ty = bool_t();
        let acc_ty1 = usize_t();
        let acc_ty2 = qb_t();
        let es = ExtensionSet::singleton(EXTENSION_ID);

        let op = ArrayScan::new(
            src_ty.clone(),
            tgt_ty.clone(),
            vec![acc_ty1.clone(), acc_ty2.clone()],
            size,
            es.clone(),
        );
        let optype: OpType = op.into();
        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![
                    array_type(size, src_ty.clone()),
                    Type::new_function(
                        Signature::new(
                            vec![src_ty, acc_ty1.clone(), acc_ty2.clone()],
                            vec![tgt_ty.clone(), acc_ty1.clone(), acc_ty2.clone()]
                        )
                        .with_extension_delta(es)
                    ),
                    acc_ty1.clone(),
                    acc_ty2.clone()
                ]
                .into(),
                &vec![array_type(size, tgt_ty), acc_ty1, acc_ty2].into(),
            )
        );
    }
}
