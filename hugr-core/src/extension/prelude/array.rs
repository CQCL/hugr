use std::str::FromStr;
use std::sync::{Arc, Weak};

use itertools::Itertools;
use strum_macros::EnumIter;
use strum_macros::EnumString;
use strum_macros::IntoStaticStr;

use crate::extension::prelude::option_type;
use crate::extension::prelude::{either_type, usize_custom_t};
use crate::extension::simple_op::{
    HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::ExtensionId;
use crate::extension::ExtensionSet;
use crate::extension::OpDef;
use crate::extension::SignatureFromArgs;
use crate::extension::SignatureFunc;
use crate::extension::TypeDef;
use crate::ops::ExtensionOp;
use crate::ops::NamedOp;
use crate::ops::OpName;
use crate::type_row;
use crate::types::FuncTypeBase;
use crate::types::FuncValueType;

use crate::types::RowVariable;
use crate::types::Signature;
use crate::types::TypeBound;

use crate::types::Type;

use crate::extension::SignatureError;

use crate::types::PolyFuncTypeRV;

use crate::types::type_param::TypeArg;
use crate::types::TypeRV;
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

/// Static parameters for array operations. Includes array size. Type is part of the type scheme.
const STATIC_SIZE_PARAM: &[TypeParam; 1] = &[TypeParam::max_nat()];

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
            let array_ty = instantiate(array_def, size_var.clone(), elem_ty_var.clone());
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
                    let copy_array_ty = instantiate(array_def, size_var, copy_elem_ty.clone());
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
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, extension_ref: &Weak<Extension>) -> SignatureFunc {
        self.signature_from_def(array_type_def(), extension_ref)
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&PRELUDE)
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
        extension_ref: &Weak<Extension>,
    ) -> Result<(), crate::extension::ExtensionBuildError> {
        let sig =
            self.signature_from_def(extension.get_type(ARRAY_TYPE_NAME).unwrap(), extension_ref);
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
        PRELUDE_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &PRELUDE_REGISTRY
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

/// Name of the operation to repeat a value multiple times
pub const ARRAY_REPEAT_OP_ID: OpName = OpName::new_inline("repeat");

/// Definition of the array repeat op.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct ArrayRepeatDef;

impl NamedOp for ArrayRepeatDef {
    fn name(&self) -> OpName {
        ARRAY_REPEAT_OP_ID
    }
}

impl FromStr for ArrayRepeatDef {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == ArrayRepeatDef.name() {
            Ok(Self)
        } else {
            Err(())
        }
    }
}

impl ArrayRepeatDef {
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
        let array_ty = instantiate(array_def, n, t);
        PolyFuncTypeRV::new(params, FuncValueType::new(vec![func], array_ty)).into()
    }
}

impl MakeOpDef for ArrayRepeatDef {
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
        Arc::downgrade(&PRELUDE)
    }

    fn extension(&self) -> ExtensionId {
        PRELUDE_ID
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
        let sig = self.signature_from_def(extension.get_type(ARRAY_TYPE_NAME).unwrap());
        let def = extension.add_op(self.name(), self.description(), sig, extension_ref)?;

        self.post_opdef(def);

        Ok(())
    }
}

/// Definition of the array repeat op.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayRepeat {
    /// The element type of the resulting array.
    pub elem_ty: Type,
    /// Size of the array.
    pub size: u64,
    /// The extensions required by the function that generates the array elements.
    pub extension_reqs: ExtensionSet,
}

impl ArrayRepeat {
    /// Creates a new array repeat op.
    pub fn new(elem_ty: Type, size: u64, extension_reqs: ExtensionSet) -> Self {
        ArrayRepeat {
            elem_ty,
            size,
            extension_reqs,
        }
    }
}

impl NamedOp for ArrayRepeat {
    fn name(&self) -> OpName {
        ARRAY_REPEAT_OP_ID
    }
}

impl MakeExtensionOp for ArrayRepeat {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = ArrayRepeatDef::from_def(ext_op.def())?;
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

impl MakeRegisteredOp for ArrayRepeat {
    fn extension_id(&self) -> ExtensionId {
        PRELUDE_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &PRELUDE_REGISTRY
    }
}

impl HasDef for ArrayRepeat {
    type Def = ArrayRepeatDef;
}

impl HasConcrete for ArrayRepeatDef {
    type Concrete = ArrayRepeat;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match type_args {
            [TypeArg::BoundedNat { n }, TypeArg::Type { ty }, TypeArg::Extensions { es }] => {
                Ok(ArrayRepeat::new(ty.clone(), *n, es.clone()))
            }
            _ => Err(SignatureError::InvalidTypeArgs.into()),
        }
    }
}

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
                    instantiate(array_def, n.clone(), t1.clone()).into(),
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
                vec![instantiate(array_def, n, t2).into(), s],
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
        Arc::downgrade(&PRELUDE)
    }

    fn extension(&self) -> ExtensionId {
        PRELUDE_ID
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
        let sig = self.signature_from_def(extension.get_type(ARRAY_TYPE_NAME).unwrap());
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
        PRELUDE_ID
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &PRELUDE_REGISTRY
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
    use strum::IntoEnumIterator;

    use crate::extension::prelude::usize_t;
    use crate::{
        builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::{bool_t, qb_t},
        ops::{OpTrait, OpType},
        types::Signature,
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
        let mut b = DFGBuilder::new(inout_sig(
            vec![qb_t(), qb_t()],
            array_type(TypeArg::BoundedNat { n: 2 }, qb_t()),
        ))
        .unwrap();

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
    fn test_repeat_def() {
        let op = ArrayRepeat::new(qb_t(), 2, ExtensionSet::singleton(PRELUDE_ID));
        let optype: OpType = op.clone().into();
        let new_op: ArrayRepeat = optype.cast().unwrap();
        assert_eq!(new_op, op);
    }

    #[test]
    fn test_repeat() {
        let size = 2;
        let element_ty = qb_t();
        let es = ExtensionSet::singleton(PRELUDE_ID);
        let op = ArrayRepeat::new(element_ty.clone(), size, es.clone());

        let optype: OpType = op.into();

        let sig = optype.dataflow_signature().unwrap();

        assert_eq!(
            sig.io(),
            (
                &vec![Type::new_function(
                    Signature::new(vec![], vec![qb_t()]).with_extension_delta(es)
                )]
                .into(),
                &vec![array_type(size, element_ty.clone())].into(),
            )
        );
    }

    #[test]
    fn test_scan_def() {
        let op = ArrayScan::new(
            bool_t(),
            qb_t(),
            vec![usize_t()],
            2,
            ExtensionSet::singleton(PRELUDE_ID),
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
        let es = ExtensionSet::singleton(PRELUDE_ID);

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
        let es = ExtensionSet::singleton(PRELUDE_ID);

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
