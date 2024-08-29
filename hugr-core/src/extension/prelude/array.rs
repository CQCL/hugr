use crate::extension::SignatureFromArgs;
use crate::ops::ExtensionOp;
use crate::ops::OpName;
use crate::types::FuncValueType;

use crate::types::TypeBound;

use crate::types::Type;

use crate::extension::SignatureError;

use crate::types::PolyFuncTypeRV;

use crate::types::type_param::TypeArg;

use super::{PRELUDE, PRELUDE_REGISTRY};
use crate::types::type_param::TypeParam;
pub(super) struct ArrayOpCustom;

pub(crate) const MAX: &[TypeParam; 1] = &[TypeParam::max_nat()];

impl SignatureFromArgs for ArrayOpCustom {
    fn compute_signature(&self, arg_values: &[TypeArg]) -> Result<PolyFuncTypeRV, SignatureError> {
        let [TypeArg::BoundedNat { n }] = *arg_values else {
            return Err(SignatureError::InvalidTypeArgs);
        };
        let elem_ty_var = Type::new_var_use(0, TypeBound::Any);

        let var_arg_row = vec![elem_ty_var.clone(); n as usize];
        let other_row = vec![array_type(TypeArg::BoundedNat { n }, elem_ty_var.clone())];

        Ok(PolyFuncTypeRV::new(
            vec![TypeBound::Any.into()],
            FuncValueType::new(var_arg_row, other_row),
        ))
    }

    fn static_params(&self) -> &[TypeParam] {
        MAX
    }
}

/// Initialize a new array of element type `element_ty` of length `size`
pub fn array_type(size: impl Into<TypeArg>, element_ty: Type) -> Type {
    let array_def = PRELUDE.get_type("array").unwrap();
    let custom_t = array_def
        .instantiate(vec![size.into(), element_ty.into()])
        .unwrap();
    Type::new_extension(custom_t)
}

/// Name of the operation in the prelude for creating new arrays.
pub const NEW_ARRAY_OP_ID: OpName = OpName::new_inline("new_array");

/// Initialize a new array op of element type `element_ty` of length `size`
pub fn new_array_op(element_ty: Type, size: u64) -> ExtensionOp {
    PRELUDE
        .instantiate_extension_op(
            &NEW_ARRAY_OP_ID,
            vec![
                TypeArg::BoundedNat { n: size },
                TypeArg::Type { ty: element_ty },
            ],
            &PRELUDE_REGISTRY,
        )
        .unwrap()
}

#[cfg(test)]
mod tests {
    use crate::{
        builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::QB_T,
    };

    use super::*;

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
