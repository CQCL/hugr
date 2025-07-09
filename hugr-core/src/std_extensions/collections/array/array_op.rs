//! Definitions of `ArrayOp` and `ArrayOpDef`.

use std::marker::PhantomData;
use std::sync::{Arc, Weak};

use strum::{EnumIter, EnumString, IntoStaticStr};

use crate::Extension;
use crate::extension::prelude::{either_type, option_type, usize_t};
use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{
    ExtensionId, OpDef, SignatureError, SignatureFromArgs, SignatureFunc, TypeDef,
};
use crate::ops::{ExtensionOp, OpName};
use crate::type_row;
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{FuncValueType, PolyFuncTypeRV, Term, Type, TypeBound};
use crate::utils::Never;

use super::array_kind::ArrayKind;

/// Array operation definitions. Generic over the concrete array implementation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, IntoStaticStr, EnumIter, EnumString)]
#[allow(non_camel_case_types)]
#[non_exhaustive]
pub enum GenericArrayOpDef<AK: ArrayKind> {
    /// Makes a new array, given distinct inputs equal to its length:
    /// `new_array<SIZE><elemty>: (elemty)^SIZE -> array<SIZE, elemty>`
    /// where `SIZE` must be statically known (not a variable)
    new_array,
    /// Copies an element out of the array ([`TypeBound::Copyable`] elements only):
    /// `get<size,elemty>: array<size, elemty>, index -> option<elemty>, array`
    get,
    /// Exchanges an element of the array with an external value:
    /// `set<size, elemty>: array<size, elemty>, index, elemty -> either(elemty, array | elemty, array)`
    /// tagged for failure/success respectively
    set,
    /// Exchanges the elements at two indices within the array:
    /// `swap<size, elemty>: array<size, elemty>, index, index -> either(array, array)`
    /// tagged for failure/success respectively
    swap,
    /// Separates the leftmost element from the rest of the array:
    /// `pop_left<SIZE><elemty>: array<SIZE, elemty> -> Option<elemty, array<SIZE-1, elemty>>`
    /// where `SIZE` must be known statically (not a variable).
    /// `None` is returned if the input array was size 0.
    pop_left,
    /// Separates the rightmost element from the rest of the array.
    /// `pop_right<SIZE><elemty>: array<SIZE, elemty> -> Option<elemty, array<SIZE-1, elemty>>`
    /// where `SIZE` must be known statically (not a variable).
    /// `None` is returned if the input array was size 0.
    pop_right,
    /// Allows discarding a 0-element array of linear type.
    /// `discard_empty<elemty>: array<0, elemty> -> ` (no outputs)
    discard_empty,
    /// Not an actual operation definition, but an unhabitable variant that
    /// references `AK` to ensure that the type parameter is used.
    #[strum(disabled)]
    _phantom(PhantomData<AK>, Never),
    /// Unpacks an array into its individual elements:
    /// `unpack<SIZE><elemty>: array<SIZE, elemty> -> (elemty)^SIZE`
    /// where `SIZE` must be statically known (not a variable)
    unpack,
}

/// Static parameters for array operations. Includes array size. Type is part of the type scheme.
const STATIC_SIZE_PARAM: &[TypeParam; 1] = &[TypeParam::max_nat_type()];

impl<AK: ArrayKind> SignatureFromArgs for GenericArrayOpDef<AK> {
    fn compute_signature(&self, arg_values: &[TypeArg]) -> Result<PolyFuncTypeRV, SignatureError> {
        let [TypeArg::BoundedNat(n)] = *arg_values else {
            return Err(SignatureError::InvalidTypeArgs);
        };
        let elem_ty_var = Type::new_var_use(0, TypeBound::Linear);
        let array_ty = AK::ty(n, elem_ty_var.clone());
        let params = vec![TypeBound::Linear.into()];
        let poly_func_ty = match self {
            GenericArrayOpDef::new_array => PolyFuncTypeRV::new(
                params,
                FuncValueType::new(vec![elem_ty_var.clone(); n as usize], array_ty),
            ),
            GenericArrayOpDef::unpack => PolyFuncTypeRV::new(
                params,
                FuncValueType::new(array_ty, vec![elem_ty_var.clone(); n as usize]),
            ),
            GenericArrayOpDef::pop_left | GenericArrayOpDef::pop_right => {
                let popped_array_ty = AK::ty(n - 1, elem_ty_var.clone());
                PolyFuncTypeRV::new(
                    params,
                    FuncValueType::new(
                        array_ty,
                        Type::from(option_type(vec![elem_ty_var, popped_array_ty])),
                    ),
                )
            }
            GenericArrayOpDef::_phantom(_, never) => match *never {},
            _ => unreachable!(
                "Operation {} should not need custom computation.",
                self.opdef_id()
            ),
        };
        Ok(poly_func_ty)
    }

    fn static_params(&self) -> &[TypeParam] {
        STATIC_SIZE_PARAM
    }
}

impl<AK: ArrayKind> GenericArrayOpDef<AK> {
    /// Instantiate a new array operation with the given element type and array size.
    #[must_use]
    pub fn to_concrete(self, elem_ty: Type, size: u64) -> GenericArrayOp<AK> {
        if self == GenericArrayOpDef::discard_empty {
            debug_assert_eq!(
                size, 0,
                "discard_empty should only be called on empty arrays"
            );
        }
        GenericArrayOp {
            def: self,
            elem_ty,
            size,
        }
    }

    /// To avoid recursion when defining the extension, take the type definition as an argument.
    fn signature_from_def(
        &self,
        array_def: &TypeDef,
        _extension_ref: &Weak<Extension>,
    ) -> SignatureFunc {
        use GenericArrayOpDef::{
            _phantom, discard_empty, get, new_array, pop_left, pop_right, set, swap, unpack,
        };
        if let new_array | unpack | pop_left | pop_right = self {
            // implements SignatureFromArgs
            // signature computed dynamically, so can rely on type definition in extension.
            (*self).into()
        } else {
            let size_var = TypeArg::new_var_use(0, TypeParam::max_nat_type());
            let elem_ty_var = Type::new_var_use(1, TypeBound::Linear);
            let array_ty = AK::instantiate_ty(array_def, size_var.clone(), elem_ty_var.clone())
                .expect("Array type instantiation failed");
            let standard_params = vec![TypeParam::max_nat_type(), TypeBound::Linear.into()];

            // We can assume that the prelude has ben loaded at this point,
            // since it doesn't depend on the array extension.
            let usize_t: Type = usize_t();

            match self {
                get => {
                    let params = vec![TypeParam::max_nat_type(), TypeBound::Copyable.into()];
                    let copy_elem_ty = Type::new_var_use(1, TypeBound::Copyable);
                    let copy_array_ty =
                        AK::instantiate_ty(array_def, size_var, copy_elem_ty.clone())
                            .expect("Array type instantiation failed");
                    let option_type: Type = option_type(copy_elem_ty).into();
                    PolyFuncTypeRV::new(
                        params,
                        FuncValueType::new(
                            vec![copy_array_ty.clone(), usize_t],
                            vec![option_type, copy_array_ty],
                        ),
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
                    vec![TypeBound::Linear.into()],
                    FuncValueType::new(
                        AK::instantiate_ty(array_def, 0, Type::new_var_use(0, TypeBound::Linear))
                            .expect("Array type instantiation failed"),
                        type_row![],
                    ),
                ),
                _phantom(_, never) => match *never {},
                new_array | unpack | pop_left | pop_right => unreachable!(),
            }
            .into()
        }
    }
}

impl<AK: ArrayKind> MakeOpDef for GenericArrayOpDef<AK> {
    fn opdef_id(&self) -> OpName {
        <&'static str>::from(self).into()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, extension_ref: &Weak<Extension>) -> SignatureFunc {
        self.signature_from_def(AK::type_def(), extension_ref)
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(AK::extension())
    }

    fn extension(&self) -> ExtensionId {
        AK::EXTENSION_ID
    }

    fn description(&self) -> String {
        match self {
            GenericArrayOpDef::new_array => "Create a new array from elements",
            GenericArrayOpDef::unpack => "Unpack an array into its elements",
            GenericArrayOpDef::get => "Get an element from an array",
            GenericArrayOpDef::set => "Set an element in an array",
            GenericArrayOpDef::swap => "Swap two elements in an array",
            GenericArrayOpDef::pop_left => "Pop an element from the left of an array",
            GenericArrayOpDef::pop_right => "Pop an element from the right of an array",
            GenericArrayOpDef::discard_empty => "Discard an empty array",
            GenericArrayOpDef::_phantom(_, never) => match *never {},
        }
        .into()
    }

    /// Add an operation implemented as an [`MakeOpDef`], which can provide the data
    /// required to define an [`OpDef`], to an extension.
    //
    // This method is re-defined here since we need to pass the list type def while computing the signature,
    // to avoid recursive loops initializing the extension.
    fn add_to_extension(
        &self,
        extension: &mut Extension,
        extension_ref: &Weak<Extension>,
    ) -> Result<(), crate::extension::ExtensionBuildError> {
        let sig =
            self.signature_from_def(extension.get_type(&AK::TYPE_NAME).unwrap(), extension_ref);
        let def = extension.add_op(self.opdef_id(), self.description(), sig, extension_ref)?;

        self.post_opdef(def);

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Concrete array operation. Generic over the actual array implementation.
pub struct GenericArrayOp<AK: ArrayKind> {
    /// The operation definition.
    pub def: GenericArrayOpDef<AK>,
    /// The element type of the array.
    pub elem_ty: Type,
    /// The size of the array.
    pub size: u64,
}

impl<AK: ArrayKind> MakeExtensionOp for GenericArrayOp<AK> {
    fn op_id(&self) -> OpName {
        self.def.opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = GenericArrayOpDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<Term> {
        use GenericArrayOpDef::{
            _phantom, discard_empty, get, new_array, pop_left, pop_right, set, swap, unpack,
        };
        let ty_arg = self.elem_ty.clone().into();
        match self.def {
            discard_empty => {
                debug_assert_eq!(
                    self.size, 0,
                    "discard_empty should only be called on empty arrays"
                );
                vec![ty_arg]
            }
            new_array | unpack | pop_left | pop_right | get | set | swap => {
                vec![self.size.into(), ty_arg]
            }
            _phantom(_, never) => match never {},
        }
    }
}

impl<AK: ArrayKind> MakeRegisteredOp for GenericArrayOp<AK> {
    fn extension_id(&self) -> ExtensionId {
        AK::EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(AK::extension())
    }
}

impl<AK: ArrayKind> HasDef for GenericArrayOp<AK> {
    type Def = GenericArrayOpDef<AK>;
}

impl<AK: ArrayKind> HasConcrete for GenericArrayOpDef<AK> {
    type Concrete = GenericArrayOp<AK>;

    fn instantiate(&self, type_args: &[Term]) -> Result<Self::Concrete, OpLoadError> {
        let (ty, size) = match (self, type_args) {
            (GenericArrayOpDef::discard_empty, [Term::Runtime(ty)]) => (ty.clone(), 0),
            (_, [Term::BoundedNat(n), Term::Runtime(ty)]) => (ty.clone(), *n),
            _ => return Err(SignatureError::InvalidTypeArgs.into()),
        };

        Ok(self.to_concrete(ty.clone(), size))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use strum::IntoEnumIterator;

    use crate::extension::prelude::usize_t;
    use crate::std_extensions::arithmetic::float_types::float64_type;
    use crate::std_extensions::collections::array::Array;
    use crate::std_extensions::collections::borrow_array::BorrowArray;
    use crate::std_extensions::collections::value_array::ValueArray;
    use crate::{
        builder::{DFGBuilder, Dataflow, DataflowHugr, inout_sig},
        extension::prelude::{bool_t, qb_t},
        ops::{OpTrait, OpType},
    };

    use super::*;

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    fn test_array_ops<AK: ArrayKind>(#[case] _kind: AK) {
        for def in GenericArrayOpDef::<AK>::iter() {
            let ty = if def == GenericArrayOpDef::get {
                bool_t()
            } else {
                qb_t()
            };
            let size = if def == GenericArrayOpDef::discard_empty {
                0
            } else {
                2
            };
            let op = def.to_concrete(ty, size);
            let optype: OpType = op.clone().into();
            let new_op: GenericArrayOp<AK> = optype.cast().unwrap();
            assert_eq!(new_op, op);
        }
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    /// Test building a HUGR involving a new_array operation.
    fn test_new_array<AK: ArrayKind>(#[case] _kind: AK) {
        let mut b = DFGBuilder::new(inout_sig(vec![qb_t(), qb_t()], AK::ty(2, qb_t()))).unwrap();

        let [q1, q2] = b.input_wires_arr();

        let op = GenericArrayOpDef::<AK>::new_array.to_concrete(qb_t(), 2);

        let out = b.add_dataflow_op(op, [q1, q2]).unwrap();

        b.finish_hugr_with_outputs(out.outputs()).unwrap();
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    /// Test building a HUGR involving an unpack operation.
    fn test_unpack<AK: ArrayKind>(#[case] _kind: AK) {
        let mut b = DFGBuilder::new(inout_sig(AK::ty(2, qb_t()), vec![qb_t(), qb_t()])).unwrap();

        let [array] = b.input_wires_arr();

        let op = GenericArrayOpDef::<AK>::unpack.to_concrete(qb_t(), 2);

        let out = b.add_dataflow_op(op, [array]).unwrap();

        b.finish_hugr_with_outputs(out.outputs()).unwrap();
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    fn test_get<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 2;
        let element_ty = bool_t();
        let op = GenericArrayOpDef::<AK>::get.to_concrete(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![AK::ty(size, element_ty.clone()), usize_t()].into(),
                &vec![
                    option_type(element_ty.clone()).into(),
                    AK::ty(size, element_ty.clone())
                ]
                .into()
            )
        );
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    fn test_set<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 2;
        let element_ty = bool_t();
        let op = GenericArrayOpDef::<AK>::set.to_concrete(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();
        let array_ty = AK::ty(size, element_ty.clone());
        let result_row = vec![element_ty.clone(), array_ty.clone()];
        assert_eq!(
            sig.io(),
            (
                &vec![array_ty.clone(), usize_t(), element_ty.clone()].into(),
                &vec![either_type(result_row.clone(), result_row).into()].into()
            )
        );
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    fn test_swap<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 2;
        let element_ty = bool_t();
        let op = GenericArrayOpDef::<AK>::swap.to_concrete(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();
        let array_ty = AK::ty(size, element_ty.clone());
        assert_eq!(
            sig.io(),
            (
                &vec![array_ty.clone(), usize_t(), usize_t()].into(),
                &vec![either_type(array_ty.clone(), array_ty).into()].into()
            )
        );
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    fn test_pops<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 2;
        let element_ty = bool_t();
        for op in &[
            GenericArrayOpDef::<AK>::pop_left,
            GenericArrayOpDef::<AK>::pop_right,
        ] {
            let op = op.to_concrete(element_ty.clone(), size);

            let optype: OpType = op.into();

            let sig = optype.dataflow_signature().unwrap();
            assert_eq!(
                sig.io(),
                (
                    &vec![AK::ty(size, element_ty.clone())].into(),
                    &vec![
                        option_type(vec![
                            element_ty.clone(),
                            AK::ty(size - 1, element_ty.clone())
                        ])
                        .into()
                    ]
                    .into()
                )
            );
        }
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    fn test_discard_empty<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 0;
        let element_ty = bool_t();
        let op = GenericArrayOpDef::<AK>::discard_empty.to_concrete(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (&vec![AK::ty(size, element_ty.clone())].into(), &type_row![])
        );
    }

    #[rstest]
    #[case(Array)]
    #[case(ValueArray)]
    #[case(BorrowArray)]
    /// Initialize an array operation where the element type is not from the prelude.
    fn test_non_prelude_op<AK: ArrayKind>(#[case] _kind: AK) {
        let size = 2;
        let element_ty = float64_type();
        let op = GenericArrayOpDef::<AK>::get.to_concrete(element_ty.clone(), size);

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![AK::ty(size, element_ty.clone()), usize_t()].into(),
                &vec![
                    option_type(element_ty.clone()).into(),
                    AK::ty(size, element_ty.clone())
                ]
                .into()
            )
        );
    }
}
