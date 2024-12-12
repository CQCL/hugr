//! Definitions of `ArrayOp` and `ArrayOpDef`.

use std::sync::{Arc, Weak};

use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::extension::prelude::{either_type, option_type, usize_custom_t};
use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{
    ExtensionId, OpDef, SignatureError, SignatureFromArgs, SignatureFunc, TypeDef,
};
use crate::ops::{ExtensionOp, NamedOp, OpName};
use crate::std_extensions::collections::array::instantiate_array;
use crate::type_row;
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{FuncValueType, PolyFuncTypeRV, Type, TypeBound};
use crate::Extension;

use super::{array_type, array_type_def, ARRAY_TYPENAME};

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

/// Static parameters for array operations. Includes array size. Type is part of the type scheme.
const STATIC_SIZE_PARAM: &[TypeParam; 1] = &[TypeParam::max_nat()];

impl SignatureFromArgs for ArrayOpDef {
    fn compute_signature(&self, arg_values: &[TypeArg]) -> Result<PolyFuncTypeRV, SignatureError> {
        let [TypeArg::BoundedNat { n }] = *arg_values else {
            return Err(SignatureError::InvalidTypeArgs);
        };
        let elem_ty_var = Type::new_var_use(0, TypeBound::Any);
        let array_ty = array_type(n, elem_ty_var.clone());
        let params = vec![TypeBound::Any.into()];
        let poly_func_ty = match self {
            ArrayOpDef::new_array => PolyFuncTypeRV::new(
                params,
                FuncValueType::new(vec![elem_ty_var.clone(); n as usize], array_ty),
            ),
            ArrayOpDef::pop_left | ArrayOpDef::pop_right => {
                let popped_array_ty = array_type(n - 1, elem_ty_var.clone());
                PolyFuncTypeRV::new(
                    params,
                    FuncValueType::new(
                        array_ty,
                        Type::from(option_type(vec![elem_ty_var, popped_array_ty])),
                    ),
                )
            }
            _ => unreachable!(
                "Operation {} should not need custom computation.",
                self.name()
            ),
        };
        Ok(poly_func_ty)
    }

    fn static_params(&self) -> &[TypeParam] {
        STATIC_SIZE_PARAM
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
    fn signature_from_def(
        &self,
        array_def: &TypeDef,
        extension_ref: &Weak<Extension>,
    ) -> SignatureFunc {
        use ArrayOpDef::*;
        if let new_array | pop_left | pop_right = self {
            // implements SignatureFromArgs
            // signature computed dynamically, so can rely on type definition in extension.
            (*self).into()
        } else {
            let size_var = TypeArg::new_var_use(0, TypeParam::max_nat());
            let elem_ty_var = Type::new_var_use(1, TypeBound::Any);
            let array_ty = instantiate_array(array_def, size_var.clone(), elem_ty_var.clone())
                .expect("Array type instantiation failed");
            let standard_params = vec![TypeParam::max_nat(), TypeBound::Any.into()];

            // Construct the usize type using the passed extension reference.
            //
            // If we tried to use `usize_t()` directly it would try to access
            // the `PRELUDE` lazy static recursively, causing a deadlock.
            let usize_t: Type = usize_custom_t(extension_ref).into();

            match self {
                get => {
                    let params = vec![TypeParam::max_nat(), TypeBound::Copyable.into()];
                    let copy_elem_ty = Type::new_var_use(1, TypeBound::Copyable);
                    let copy_array_ty =
                        instantiate_array(array_def, size_var, copy_elem_ty.clone())
                            .expect("Array type instantiation failed");
                    let option_type: Type = option_type(copy_elem_ty).into();
                    PolyFuncTypeRV::new(
                        params,
                        FuncValueType::new(vec![copy_array_ty, usize_t], option_type),
                    )
                }
                set => {
                    let result_row = vec![elem_ty_var.clone(), array_ty.clone()];
                    let result_type: Type = either_type(result_row.clone(), result_row).into();
                    PolyFuncTypeRV::new(
                        standard_params,
                        FuncValueType::new(
                            vec![array_ty.clone(), usize_t, elem_ty_var],
                            result_type,
                        ),
                    )
                }
                swap => {
                    let result_type: Type = either_type(array_ty.clone(), array_ty.clone()).into();
                    PolyFuncTypeRV::new(
                        standard_params,
                        FuncValueType::new(vec![array_ty, usize_t.clone(), usize_t], result_type),
                    )
                }
                discard_empty => PolyFuncTypeRV::new(
                    vec![TypeBound::Any.into()],
                    FuncValueType::new(
                        instantiate_array(array_def, 0, Type::new_var_use(0, TypeBound::Any))
                            .expect("Array type instantiation failed"),
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
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, extension_ref: &Weak<Extension>) -> SignatureFunc {
        self.signature_from_def(array_type_def(), extension_ref)
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&super::EXTENSION)
    }

    fn extension(&self) -> ExtensionId {
        super::EXTENSION_ID
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
        extension_ref: &Weak<Extension>,
    ) -> Result<(), crate::extension::ExtensionBuildError> {
        let sig =
            self.signature_from_def(extension.get_type(&ARRAY_TYPENAME).unwrap(), extension_ref);
        let def = extension.add_op(self.name(), self.description(), sig, extension_ref)?;

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
        def.instantiate(ext_op.args())
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
        super::EXTENSION_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &super::ARRAY_REGISTRY
    }
}

impl HasDef for ArrayOp {
    type Def = ArrayOpDef;
}

impl HasConcrete for ArrayOpDef {
    type Concrete = ArrayOp;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        let (ty, size) = match (self, type_args) {
            (ArrayOpDef::discard_empty, [TypeArg::Type { ty }]) => (ty.clone(), 0),
            (_, [TypeArg::BoundedNat { n }, TypeArg::Type { ty }]) => (ty.clone(), *n),
            _ => return Err(SignatureError::InvalidTypeArgs.into()),
        };

        Ok(self.to_concrete(ty.clone(), size))
    }
}

#[cfg(test)]
mod tests {
    use std::arch::aarch64::float32x2_t;

    use strum::IntoEnumIterator;

    use crate::extension::prelude::usize_t;
    use crate::std_extensions::arithmetic::float_types::float64_type;
    use crate::std_extensions::collections::array::new_array_op;
    use crate::{
        builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::{bool_t, qb_t},
        ops::{OpTrait, OpType},
    };

    use super::*;

    #[test]
    fn test_array_ops() {
        for def in ArrayOpDef::iter() {
            let ty = if def == ArrayOpDef::get {
                bool_t()
            } else {
                qb_t()
            };
            let size = if def == ArrayOpDef::discard_empty {
                0
            } else {
                2
            };
            let op = def.to_concrete(ty, size);
            let optype: OpType = op.clone().into();
            let new_op: ArrayOp = optype.cast().unwrap();
            assert_eq!(new_op, op);
        }
    }

    #[test]
    /// Test building a HUGR involving a new_array operation.
    fn test_new_array() {
        let mut b =
            DFGBuilder::new(inout_sig(vec![qb_t(), qb_t()], array_type(2, qb_t()))).unwrap();

        let [q1, q2] = b.input_wires_arr();

        let op = new_array_op(qb_t(), 2);

        let out = b.add_dataflow_op(op, [q1, q2]).unwrap();

        b.finish_hugr_with_outputs(out.outputs()).unwrap();
    }

    #[test]
    fn test_get() {
        let size = 2;
        let element_ty = bool_t();
        let op = ArrayOpDef::get.to_concrete(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![array_type(size, element_ty.clone()), usize_t()].into(),
                &vec![option_type(element_ty.clone()).into()].into()
            )
        );
    }

    #[test]
    fn test_set() {
        let size = 2;
        let element_ty = bool_t();
        let op = ArrayOpDef::set.to_concrete(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();
        let array_ty = array_type(size, element_ty.clone());
        let result_row = vec![element_ty.clone(), array_ty.clone()];
        assert_eq!(
            sig.io(),
            (
                &vec![array_ty.clone(), usize_t(), element_ty.clone()].into(),
                &vec![either_type(result_row.clone(), result_row).into()].into()
            )
        );
    }

    #[test]
    fn test_swap() {
        let size = 2;
        let element_ty = bool_t();
        let op = ArrayOpDef::swap.to_concrete(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();
        let array_ty = array_type(size, element_ty.clone());
        assert_eq!(
            sig.io(),
            (
                &vec![array_ty.clone(), usize_t(), usize_t()].into(),
                &vec![either_type(array_ty.clone(), array_ty).into()].into()
            )
        );
    }

    #[test]
    fn test_pops() {
        let size = 2;
        let element_ty = bool_t();
        for op in [ArrayOpDef::pop_left, ArrayOpDef::pop_right].iter() {
            let op = op.to_concrete(element_ty.clone(), size);

            let optype: OpType = op.into();

            let sig = optype.dataflow_signature().unwrap();
            assert_eq!(
                sig.io(),
                (
                    &vec![array_type(size, element_ty.clone())].into(),
                    &vec![option_type(vec![
                        element_ty.clone(),
                        array_type(size - 1, element_ty.clone())
                    ])
                    .into()]
                    .into()
                )
            );
        }
    }

    #[test]
    fn test_discard_empty() {
        let size = 0;
        let element_ty = bool_t();
        let op = ArrayOpDef::discard_empty.to_concrete(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![array_type(size, element_ty.clone())].into(),
                &type_row![]
            )
        );
    }

    #[test]
    /// Initialize an array operation where the element type is not from the prelude.
    fn test_non_prelude_op() {
        let size = 2;
        let element_ty = float64_type();
        let op = ArrayOpDef::get.to_concrete(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![array_type(size, element_ty.clone()), usize_t()].into(),
                &vec![option_type(element_ty.clone()).into()].into()
            )
        );
    }
}
