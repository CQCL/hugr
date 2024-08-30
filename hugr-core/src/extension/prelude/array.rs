use strum_macros::EnumIter;
use strum_macros::EnumString;
use strum_macros::IntoStaticStr;

use crate::extension::prelude::either_type;
use crate::extension::prelude::option_type;
use crate::extension::prelude::USIZE_T;
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
use crate::type_row;
use crate::types::FuncValueType;

use crate::types::TypeBound;

use crate::types::Type;

use crate::extension::SignatureError;

use crate::types::PolyFuncTypeRV;

use crate::types::type_param::TypeArg;
use crate::Extension;

use super::PRELUDE_ID;
use super::{PRELUDE, PRELUDE_REGISTRY};
use crate::types::type_param::TypeParam;

/// Array operation definitions.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(non_camel_case_types, missing_docs)]
#[non_exhaustive]
pub enum ArrayOpDef {
    new_array,
    get,
    set,
    swap,
    pop_left,
    pop_right,
    discard_empty,
}
const MAX: &[TypeParam; 1] = &[TypeParam::max_nat()];

impl SignatureFromArgs for ArrayOpDef {
    fn compute_signature(&self, arg_values: &[TypeArg]) -> Result<PolyFuncTypeRV, SignatureError> {
        let [TypeArg::BoundedNat { n }] = *arg_values else {
            return Err(SignatureError::InvalidTypeArgs);
        };
        let elem_ty_var = Type::new_var_use(0, TypeBound::Any);
        let array_ty = array_type(TypeArg::BoundedNat { n }, elem_ty_var.clone());
        let params = vec![TypeBound::Any.into()];
        let poly_func_ty = match self {
            ArrayOpDef::new_array => PolyFuncTypeRV::new(
                params,
                FuncValueType::new(vec![elem_ty_var.clone(); n as usize], array_ty),
            ),
            ArrayOpDef::pop_left | ArrayOpDef::pop_right => {
                let popped_array_ty =
                    array_type(TypeArg::BoundedNat { n: n - 1 }, elem_ty_var.clone());
                PolyFuncTypeRV::new(
                    params,
                    FuncValueType::new(array_ty, vec![elem_ty_var, popped_array_ty]),
                )
            }
            _ => unreachable!("Other operations should not need custom computation."),
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
        if self == ArrayOpDef::discard_empty {
            debug_assert_eq!(
                size, 0,
                "discard_empty should only be called on empty arrays"
            );
        }
        ArrayOp {
            def: self,
            elem_ty,
            size,
        }
    }

    /// To avoid recursion when defining the extension, take the type definition as an argument.
    fn signature_from_def(&self, array_def: &TypeDef) -> SignatureFunc {
        use ArrayOpDef::*;
        if let new_array | pop_left | pop_right = self {
            // implements SignatureFromArgs
            // signature computed dynamically, so can rely on type definition in extension.
            (*self).into()
        } else {
            let size_var = TypeArg::new_var_use(0, TypeParam::max_nat());
            let elem_ty_var = Type::new_var_use(1, TypeBound::Any);
            let array_ty = instantiate(array_def, size_var.clone(), elem_ty_var.clone());
            let standard_params = vec![TypeParam::max_nat(), TypeBound::Any.into()];

            match self {
                get => {
                    let params = vec![TypeParam::max_nat(), TypeBound::Copyable.into()];
                    let copy_elem_ty = Type::new_var_use(1, TypeBound::Copyable);
                    let copy_array_ty = instantiate(array_def, size_var, copy_elem_ty.clone());
                    let option_type: Type = option_type(copy_elem_ty).into();
                    PolyFuncTypeRV::new(
                        params,
                        FuncValueType::new(vec![copy_array_ty, USIZE_T], option_type),
                    )
                }
                set => {
                    let result_row = vec![elem_ty_var.clone(), array_ty.clone()];
                    let result_type: Type = either_type(result_row.clone(), result_row).into();
                    PolyFuncTypeRV::new(
                        standard_params,
                        FuncValueType::new(
                            vec![array_ty.clone(), USIZE_T, elem_ty_var],
                            result_type,
                        ),
                    )
                }
                swap => {
                    let result_type: Type = either_type(array_ty.clone(), array_ty.clone()).into();
                    PolyFuncTypeRV::new(
                        standard_params,
                        FuncValueType::new(vec![array_ty, USIZE_T, USIZE_T], result_type),
                    )
                }
                discard_empty => PolyFuncTypeRV::new(
                    vec![TypeBound::Any.into()],
                    FuncValueType::new(
                        instantiate(array_def, 0, Type::new_var_use(0, TypeBound::Any)),
                        type_row![],
                    ),
                ),
                new_array | pop_left | pop_right => unreachable!(),
            }
            .into()
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
        self.signature_from_def(array_type_def())
    }

    fn extension(&self) -> ExtensionId {
        PRELUDE_ID
    }

    fn description(&self) -> String {
        match self {
            ArrayOpDef::new_array => "Create a new array from elements",
            ArrayOpDef::get => "Get an element from an array",
            ArrayOpDef::set => "Set an element in an array",
            ArrayOpDef::swap => "Swap two elements in an array",
            ArrayOpDef::pop_left => "Pop an element from the left of an array",
            ArrayOpDef::pop_right => "Pop an element from the right of an array",
            ArrayOpDef::discard_empty => "Discard an empty array",
        }
        .into()
    }

    /// Add an operation implemented as an [MakeOpDef], which can provide the data
    /// required to define an [OpDef], to an extension.
    //
    // This method is re-defined here since we need to pass the list type def while computing the signature,
    // to avoid recursive loops initializing the extension.
    fn add_to_extension(
        &self,
        extension: &mut Extension,
    ) -> Result<(), crate::extension::ExtensionBuildError> {
        let sig = self.signature_from_def(extension.get_type(ARRAY_TYPE_NAME).unwrap());
        let def = extension.add_op(self.name(), self.description(), sig)?;

        self.post_opdef(def);

        Ok(())
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

        let (ty, size) = match (def, ext_op.args()) {
            (ArrayOpDef::discard_empty, [TypeArg::Type { ty }]) => (ty.clone(), 0),
            (_, [TypeArg::BoundedNat { n }, TypeArg::Type { ty }]) => (ty.clone(), *n),
            _ => return Err(SignatureError::InvalidTypeArgs.into()),
        };

        Ok(def.to_concrete(ty.clone(), size))
    }

    fn type_args(&self) -> Vec<TypeArg> {
        use ArrayOpDef::*;
        let ty_arg = TypeArg::Type {
            ty: self.elem_ty.clone(),
        };
        match self.def {
            discard_empty => {
                debug_assert_eq!(
                    self.size, 0,
                    "discard_empty should only be called on empty arrays"
                );
                vec![ty_arg]
            }
            new_array | pop_left | pop_right | get | set | swap => {
                vec![TypeArg::BoundedNat { n: self.size }, ty_arg]
            }
        }
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
    instantiate(array_type_def(), size, element_ty)
}

fn instantiate(
    array_def: &TypeDef,
    size: impl Into<TypeArg>,
    element_ty: crate::types::TypeBase<crate::types::NoRV>,
) -> crate::types::TypeBase<crate::types::NoRV> {
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
        extension::prelude::{BOOL_T, QB_T},
        ops::OpType,
    };

    use super::*;

    #[test]
    fn test_array_ops() {
        for def in ArrayOpDef::iter() {
            dbg!(def);
            let ty = if def == ArrayOpDef::get { BOOL_T } else { QB_T };
            let size = if def == ArrayOpDef::discard_empty {
                0
            } else {
                2
            };
            let op = def.to_concrete(ty, size);
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
