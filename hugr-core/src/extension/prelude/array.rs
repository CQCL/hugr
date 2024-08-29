use strum_macros::EnumIter;
use strum_macros::EnumString;
use strum_macros::IntoStaticStr;

use crate::extension::simple_op::MakeExtensionOp;
use crate::extension::simple_op::MakeOpDef;
use crate::extension::simple_op::MakeRegisteredOp;
use crate::extension::simple_op::OpLoadError;
use crate::extension::ExtensionId;
use crate::extension::OpDef;
use crate::extension::SignatureFromArgs;
use crate::extension::SignatureFunc;
use crate::extension::TypeDef;
use crate::ops::ExtensionOp;
use crate::ops::NamedOp;
use crate::ops::OpName;
use crate::types::FuncValueType;

use crate::types::TypeBound;

use crate::types::Type;

use crate::extension::SignatureError;

use crate::types::PolyFuncTypeRV;

use crate::types::type_param::TypeArg;

use super::PRELUDE_ID;
use super::{PRELUDE, PRELUDE_REGISTRY};
use crate::types::type_param::TypeParam;

/// Array operation definitions.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(non_camel_case_types, missing_docs)]
#[non_exhaustive]
pub enum ArrayOpDef {
    new_array,
}
const MAX: &[TypeParam; 1] = &[TypeParam::max_nat()];

impl SignatureFromArgs for ArrayOpDef {
    fn compute_signature(&self, arg_values: &[TypeArg]) -> Result<PolyFuncTypeRV, SignatureError> {
        let [TypeArg::BoundedNat { n }] = *arg_values else {
            return Err(SignatureError::InvalidTypeArgs);
        };
        let elem_ty_var = Type::new_var_use(0, TypeBound::Any);
        let array_ty = array_type(TypeArg::BoundedNat { n }, elem_ty_var.clone());
        let poly_func_ty = match self {
            ArrayOpDef::new_array => PolyFuncTypeRV::new(
                vec![TypeBound::Any.into()],
                FuncValueType::new(vec![elem_ty_var.clone(); n as usize], array_ty),
            ),
        };
        Ok(poly_func_ty)
    }

    fn static_params(&self) -> &[TypeParam] {
        MAX
    }
}

impl ArrayOpDef {
    /// Instantiate a new array operation with the given element type and array size.
    pub fn to_concrete(self, elem_ty: Type, size: u64) -> ArrayOp {
        ArrayOp {
            def: self,
            elem_ty,
            size,
        }
    }
}

impl MakeOpDef for ArrayOpDef {
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension())
    }

    fn signature(&self) -> SignatureFunc {
        (*self).into() // implements SignatureFromArgs
    }

    fn extension(&self) -> ExtensionId {
        PRELUDE_ID
    }

    fn description(&self) -> String {
        match self {
            ArrayOpDef::new_array => "Create a new array from elements",
        }
        .into()
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Concrete array operation.
pub struct ArrayOp {
    /// The operation definition.
    pub def: ArrayOpDef,
    /// The element type of the array.
    pub elem_ty: Type,
    /// The size of the array.
    pub size: u64,
}

impl NamedOp for ArrayOp {
    fn name(&self) -> OpName {
        self.def.name()
    }
}

impl MakeExtensionOp for ArrayOp {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = ArrayOpDef::from_def(ext_op.def())?;
        let [TypeArg::BoundedNat { n }, TypeArg::Type { ty }] = ext_op.args() else {
            return Err(SignatureError::InvalidTypeArgs.into());
        };

        Ok(def.to_concrete(ty.clone(), *n))
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![
            TypeArg::BoundedNat { n: self.size },
            TypeArg::Type {
                ty: self.elem_ty.clone(),
            },
        ]
    }
}

impl MakeRegisteredOp for ArrayOp {
    fn extension_id(&self) -> ExtensionId {
        PRELUDE_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &PRELUDE_REGISTRY
    }
}

/// Name of the array type in the prelude.
pub const ARRAY_TYPE_NAME: &str = "array";

fn array_type_def() -> &'static TypeDef {
    PRELUDE.get_type(ARRAY_TYPE_NAME).unwrap()
}
/// Initialize a new array of element type `element_ty` of length `size`
pub fn array_type(size: impl Into<TypeArg>, element_ty: Type) -> Type {
    let array_def = array_type_def();
    array_def
        .instantiate(vec![size.into(), element_ty.into()])
        .unwrap()
        .into()
}

/// Name of the operation in the prelude for creating new arrays.
pub const NEW_ARRAY_OP_ID: OpName = OpName::new_inline("new_array");

/// Initialize a new array op of element type `element_ty` of length `size`
pub fn new_array_op(element_ty: Type, size: u64) -> ExtensionOp {
    let op = ArrayOpDef::new_array.to_concrete(element_ty, size);
    op.to_extension_op().unwrap()
}

#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;

    use crate::{
        builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::QB_T,
        ops::OpType,
    };

    use super::*;

    #[test]
    fn test_array_ops() {
        for def in ArrayOpDef::iter() {
            let op = def.to_concrete(QB_T, 2);
            let ext_op = op.clone().to_extension_op().unwrap();
            let optype: OpType = ext_op.into();
            let new_op: ArrayOp = optype.cast().unwrap();
            assert_eq!(new_op, op);
        }
    }

    #[test]
    /// Test building a HUGR involving a new_array operation.
    fn test_new_array() {
        let mut b = DFGBuilder::new(inout_sig(
            vec![QB_T, QB_T],
            array_type(TypeArg::BoundedNat { n: 2 }, QB_T),
        ))
        .unwrap();

        let [q1, q2] = b.input_wires_arr();

        let op = new_array_op(QB_T, 2);

        let out = b.add_dataflow_op(op, [q1, q2]).unwrap();

        b.finish_prelude_hugr_with_outputs(out.outputs()).unwrap();
    }
}
