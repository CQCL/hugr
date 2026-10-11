//! Shared mutable storage with explicitly managed linear pointer handles.
//!
//! Pointers can hold linear values. Use [`PtrOpDef::Dup`] to share a cell and
//! [`PtrOpDef::Free`] to release each handle, recovering the value when the last
//! handle is released. [`PtrOpDef::Swap`] and [`PtrOpDef::Map`] update the cell
//! without copying or discarding its contents.
//!
//! Pointer handles are threaded through operations to establish their order:
//! passing the pointer returned by one operation into the next makes the next
//! operation wait for the first to finish. After [`PtrOpDef::Dup`], operations on
//! the separate handles have unspecified order unless another dependency orders
//! them. Sharing a cell does not itself establish an execution order.

use std::sync::{Arc, LazyLock, Weak};

use strum::{EnumIter, EnumString, IntoStaticStr};

use crate::Wire;
use crate::builder::{BuildError, Dataflow};
use crate::extension::TypeDefBound;
use crate::extension::prelude::{bool_t, option_type};
use crate::ops::OpName;
use crate::types::{
    CustomType, FuncValueType, PolyFuncType, PolyFuncTypeRV, Signature, Type, TypeBound, TypeName,
    TypeRow, TypeRowRV,
};
use crate::{
    Extension,
    extension::{
        ExtensionId, OpDef, SignatureError, SignatureFunc,
        simple_op::{
            HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
        },
    },
    ops::custom::ExtensionOp,
    types::type_param::{TypeArg, TypeParam},
};
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[non_exhaustive]
/// Pointer operation definitions.
pub enum PtrOpDef {
    /// Create a new pointer.
    New,
    /// Copy the stored value and return the pointer. Requires a copyable value.
    Read,
    /// Replace a copyable stored value and return the pointer.
    Write,
    /// Exchange the stored value with the input and return the pointer and old value.
    Swap,
    /// Return two handles to the same cell, without copying its contents.
    Dup,
    /// Compare cell identity and return both handles in input order.
    /// Does not read or compare the stored values.
    Eq,
    /// Release a handle, returning the stored value only for the last handle.
    Free,
    /// Apply a function to the stored value and extra inputs, replacing the value.
    Map,
}

impl PtrOpDef {
    /// Create a concrete pointer operation with the given value type.
    /// For [`Self::Map`], the extra input and output rows are empty.
    /// Use [`PtrOp::map`] to supply nonempty rows.
    #[must_use]
    pub fn with_type(self, ty: Type) -> PtrOp {
        PtrOp::new(self, ty)
    }
}

impl MakeOpDef for PtrOpDef {
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
        let [
            (linear_params, linear_ptr, linear_t),
            (cp_params, cp_ptr, cp_t),
        ] = [TypeBound::Linear, TypeBound::Copyable].map(|bound| {
            let params = [TypeParam::TypeKind(bound)];
            let inner_t = Type::new_var_use(0, bound);
            let ptr_t: Type = ptr_custom_type(inner_t.clone(), extension_ref).into();
            (params, ptr_t, inner_t)
        });
        SignatureFunc::from(match self {
            PtrOpDef::New => {
                PolyFuncType::new(linear_params, Signature::new([linear_t], [linear_ptr]))
            }
            PtrOpDef::Read => {
                PolyFuncType::new(cp_params, Signature::new([cp_ptr.clone()], [cp_ptr, cp_t]))
            }
            PtrOpDef::Write => {
                PolyFuncType::new(cp_params, Signature::new([cp_ptr.clone(), cp_t], [cp_ptr]))
            }
            PtrOpDef::Swap => PolyFuncType::new(
                linear_params,
                Signature::new(
                    [linear_ptr.clone(), linear_t.clone()],
                    [linear_ptr, linear_t],
                ),
            ),
            PtrOpDef::Dup => PolyFuncType::new(
                linear_params,
                Signature::new([linear_ptr.clone()], [linear_ptr.clone(), linear_ptr]),
            ),
            PtrOpDef::Eq => PolyFuncType::new(
                linear_params,
                Signature::new(
                    [linear_ptr.clone(), linear_ptr.clone()],
                    [linear_ptr.clone(), linear_ptr, bool_t()],
                ),
            ),
            PtrOpDef::Free => PolyFuncType::new(
                linear_params,
                Signature::new([linear_ptr], [option_type([linear_t]).into()]),
            ),
            PtrOpDef::Map => {
                let row_param =
                    TypeParam::ListKind(Box::new(TypeParam::TypeKind(TypeBound::Linear)));
                let params = [linear_params[0].clone(), row_param.clone(), row_param];

                let t = Type::new_var_use(0, TypeBound::Linear);
                let [input_row, output_row] =
                    [1, 2].map(|i| TypeRowRV::new_var_use(i, TypeBound::Linear));
                let func_t = Type::new_function(FuncValueType::new(
                    TypeRowRV::from(vec![t.clone()]).concat(input_row.clone()),
                    TypeRowRV::from(vec![t]).concat(output_row.clone()),
                ));
                return PolyFuncTypeRV::new(
                    params,
                    FuncValueType::new(
                        TypeRowRV::from(vec![linear_ptr.clone(), func_t]).concat(input_row),
                        TypeRowRV::from(vec![linear_ptr]).concat(output_row),
                    ),
                )
                .into();
            }
        })
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn description(&self) -> String {
        match self {
            PtrOpDef::New => "Create a new pointer from a value.".into(),
            PtrOpDef::Read => "Copy the stored value and return the pointer.".into(),
            PtrOpDef::Write => "Replace a copyable stored value and return the pointer.".into(),
            PtrOpDef::Swap => "Exchange the stored value, returning the pointer and old value.".into(),
            PtrOpDef::Dup => "Return two handles to the same cell without copying the stored value.".into(),
            PtrOpDef::Eq => "Compare cell identity without reading the stored values, returning both handles in input order and a boolean.".into(),
            PtrOpDef::Free => "Release a handle, returning Some(value) for the last handle and None otherwise.".into(),
            PtrOpDef::Map => "Apply a function to the stored value and extra inputs, replacing the stored value and returning the pointer and extra outputs.".into(),
        }
    }
}

/// Name of pointer extension.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("ptr");
/// Name of pointer type.
pub const PTR_TYPE_ID: TypeName = TypeName::new_inline("ptr");
const TYPE_PARAMS: [TypeParam; 1] = [TypeParam::TypeKind(TypeBound::Linear)];
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 2, 0);

/// Extension for pointer operations.
fn extension() -> Arc<Extension> {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        extension
            .add_type(
                PTR_TYPE_ID,
                TYPE_PARAMS.into(),
                "Linear handle to a shared mutable cell.".into(),
                TypeDefBound::Explicit {
                    bound: TypeBound::Linear,
                },
                extension_ref,
            )
            .unwrap();
        PtrOpDef::load_all_ops(extension, extension_ref).unwrap();
    })
}

/// Reference to the pointer Extension.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(extension);

/// Construct a pointer type without initializing the extension recursively.
fn ptr_custom_type(ty: impl Into<Type>, extension_ref: &Weak<Extension>) -> CustomType {
    let ty = ty.into();
    CustomType::new(
        PTR_TYPE_ID,
        [ty.into()],
        EXTENSION_ID,
        VERSION,
        TypeBound::Linear,
        extension_ref,
    )
}

/// A linear handle to a shared mutable cell containing the given type.
pub fn ptr_type(ty: impl Into<Type>) -> Type {
    ptr_custom_type(ty, &Arc::<Extension>::downgrade(&EXTENSION)).into()
}

#[derive(Clone, Debug, PartialEq)]
/// A concrete pointer operation.
pub struct PtrOp {
    /// The operation definition.
    pub def: PtrOpDef,
    /// Type of the value being pointed to.
    pub ty: Type,
    /// Extra input and output rows for [`PtrOpDef::Map`].
    /// Other operations use empty rows.
    pub map_signature: Signature,
}

impl PtrOp {
    /// Create a map operation with the given extra input and output rows.
    /// The callback also takes and returns the stored value as its first value.
    pub fn map(ty: Type, inputs: impl Into<TypeRow>, outputs: impl Into<TypeRow>) -> Self {
        Self {
            def: PtrOpDef::Map,
            ty,
            map_signature: Signature::new(inputs, outputs),
        }
    }

    fn new(op: PtrOpDef, ty: Type) -> Self {
        Self {
            def: op,
            ty,
            map_signature: Signature::new_endo(vec![]),
        }
    }
}

impl MakeExtensionOp for PtrOp {
    fn op_id(&self) -> OpName {
        self.def.opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError> {
        let def = PtrOpDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        let mut args = vec![self.ty.clone().into()];
        if self.def == PtrOpDef::Map {
            args.extend([
                self.map_signature.input().clone().into(),
                self.map_signature.output().clone().into(),
            ]);
        }
        args
    }
}

impl MakeRegisteredOp for PtrOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> Arc<Extension> {
        EXTENSION.clone()
    }
}

/// An extension trait for [Dataflow] providing methods to add pointer
/// operations.
pub trait PtrOpBuilder: Dataflow {
    /// Add a "ptr.New" op.
    fn add_new_ptr(&mut self, val_wire: Wire) -> Result<Wire, BuildError> {
        let ty = self.get_wire_type(val_wire)?;
        let handle = self.add_dataflow_op(PtrOpDef::New.with_type(ty), [val_wire])?;

        Ok(handle.out_wire(0))
    }

    /// Copy a stored value, returning the pointer and the copied value.
    fn add_read_ptr(&mut self, ptr_wire: Wire, ty: Type) -> Result<(Wire, Wire), BuildError> {
        let handle = self.add_dataflow_op(PtrOpDef::Read.with_type(ty.clone()), [ptr_wire])?;
        Ok((handle.out_wire(0), handle.out_wire(1)))
    }

    /// Replace a copyable stored value, returning the pointer.
    fn add_write_ptr(&mut self, ptr_wire: Wire, val_wire: Wire) -> Result<Wire, BuildError> {
        let ty = self.get_wire_type(val_wire)?;

        let handle = self.add_dataflow_op(PtrOpDef::Write.with_type(ty), [ptr_wire, val_wire])?;
        Ok(handle.out_wire(0))
    }

    /// Exchange the stored value, returning the pointer and the previous value.
    fn add_swap_ptr(&mut self, ptr_wire: Wire, val_wire: Wire) -> Result<(Wire, Wire), BuildError> {
        let ty = self.get_wire_type(val_wire)?;
        let handle = self.add_dataflow_op(PtrOpDef::Swap.with_type(ty), [ptr_wire, val_wire])?;
        Ok((handle.out_wire(0), handle.out_wire(1)))
    }

    /// Return two handles to the same cell containing values of type `ty`.
    fn add_dup_ptr(&mut self, ptr_wire: Wire, ty: Type) -> Result<(Wire, Wire), BuildError> {
        let handle = self.add_dataflow_op(PtrOpDef::Dup.with_type(ty), [ptr_wire])?;
        Ok((handle.out_wire(0), handle.out_wire(1)))
    }

    /// Compare the identity of two cells with the same stored type `ty`.
    /// Return both linear handles in input order, followed by the comparison result.
    /// The stored values are neither read nor compared.
    fn add_eq_ptr(
        &mut self,
        lhs: Wire,
        rhs: Wire,
        ty: Type,
    ) -> Result<(Wire, Wire, Wire), BuildError> {
        let handle = self.add_dataflow_op(PtrOpDef::Eq.with_type(ty), [lhs, rhs])?;
        Ok((handle.out_wire(0), handle.out_wire(1), handle.out_wire(2)))
    }

    /// Release a handle to a cell containing values of type `ty`.
    /// The returned wire has type `option<ty>` and contains the stored value only
    /// when this was the last handle.
    fn add_free_ptr(&mut self, ptr_wire: Wire, ty: Type) -> Result<Wire, BuildError> {
        let handle = self.add_dataflow_op(PtrOpDef::Free.with_type(ty), [ptr_wire])?;
        Ok(handle.out_wire(0))
    }

    /// Apply `func_wire` to the stored value and the extra `inputs`.
    ///
    /// The callback takes and returns the stored type `ty` first. Its remaining
    /// input types are inferred from `inputs`, and its remaining output types
    /// are given by `output_types`. Return the pointer and the extra outputs in
    /// callback order. Both extra rows may be empty or contain linear values.
    fn add_map_ptr(
        &mut self,
        ptr_wire: Wire,
        func_wire: Wire,
        ty: Type,
        inputs: impl IntoIterator<Item = Wire>,
        output_types: impl Into<TypeRow>,
    ) -> Result<(Wire, Vec<Wire>), BuildError> {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        let input_types = inputs
            .iter()
            .map(|&wire| self.get_wire_type(wire))
            .collect::<Result<Vec<_>, _>>()?;
        let op = PtrOp::map(ty, input_types, output_types);
        let handle = self.add_dataflow_op(op, [ptr_wire, func_wire].into_iter().chain(inputs))?;
        Ok((handle.out_wire(0), handle.outputs().skip(1).collect()))
    }
}

impl<D: Dataflow> PtrOpBuilder for D {}

impl HasConcrete for PtrOpDef {
    type Concrete = PtrOp;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        let op = match (self, type_args) {
            (Self::Map, [ty, inputs, outputs]) => PtrOp::map(
                Type::try_from(ty.clone())?,
                TypeRow::try_from(inputs.clone())?,
                TypeRow::try_from(outputs.clone())?,
            ),
            (def, [ty]) if *def != Self::Map => def.with_type(Type::try_from(ty.clone())?),
            _ => return Err(SignatureError::InvalidTypeArgs.into()),
        };
        op.clone().to_extension_op()?;
        Ok(op)
    }
}

impl HasDef for PtrOp {
    type Def = PtrOpDef;
}

#[cfg(test)]
pub(crate) mod test {
    use crate::builder::DFGBuilder;
    use crate::extension::prelude::{bool_t, qb_t};
    use crate::ops::ExtensionOp;
    use crate::{HugrView, PortIndex};
    use crate::{
        builder::{Dataflow, DataflowHugr},
        std_extensions::arithmetic::int_types::INT_TYPES,
    };
    use cool_asserts::assert_matches;
    use std::sync::Arc;
    use strum::IntoEnumIterator;

    use super::*;
    use crate::std_extensions::arithmetic::float_types::float64_type;
    fn get_opdef(op: impl Into<&'static str>) -> Option<&'static Arc<OpDef>> {
        EXTENSION.get_op(op.into())
    }

    #[test]
    fn create_extension() {
        assert_eq!(EXTENSION.name(), &EXTENSION_ID);

        for o in PtrOpDef::iter() {
            assert_eq!(PtrOpDef::from_def(get_opdef(o).unwrap()), Ok(o));
        }
    }

    #[test]
    fn test_ops() {
        let ops = [
            PtrOp::new(PtrOpDef::New, bool_t().clone()),
            PtrOp::new(PtrOpDef::Read, float64_type()),
            PtrOp::new(PtrOpDef::Write, INT_TYPES[5].clone()),
        ];
        for op in ops {
            let op_t: ExtensionOp = op.clone().to_extension_op().unwrap();
            let def_op = PtrOpDef::from_op(&op_t).unwrap();
            assert_eq!(op.def, def_op);
            let new_op = PtrOp::from_op(&op_t).unwrap();
            assert_eq!(new_op, op);
        }
    }

    #[test]
    fn test_build() {
        let in_row = vec![bool_t(), float64_type()];

        let hugr = {
            let mut builder = DFGBuilder::new(Signature::new(in_row.clone(), vec![])).unwrap();

            let in_wires: [Wire; 2] = builder.input_wires_arr();
            for (ty, w) in in_row.into_iter().zip(in_wires.iter()) {
                let new_ptr = builder.add_new_ptr(*w).unwrap();
                let (ptr, read) = builder.add_read_ptr(new_ptr, ty.clone()).unwrap();
                let ptr = builder.add_write_ptr(ptr, read).unwrap();
                builder
                    .add_dataflow_op(PtrOpDef::Free.with_type(ty), [ptr])
                    .unwrap();
            }

            builder.finish_hugr_with_outputs([]).unwrap()
        };
        assert_matches!(hugr.validate(), Ok(()));
    }
    #[test]
    fn signatures_and_bounds() {
        use crate::ops::DataflowOpTrait;
        for ty in [bool_t(), qb_t()] {
            let ptr = ptr_type(ty.clone());
            assert_eq!(ptr.least_upper_bound(), TypeBound::Linear);
            let declared: Type = EXTENSION
                .get_type(&PTR_TYPE_ID)
                .unwrap()
                .instantiate(vec![ty.clone().into()])
                .unwrap()
                .into();
            assert_eq!(ptr, declared);
            let cases = [
                (PtrOpDef::New, vec![ty.clone()], vec![ptr.clone()]),
                (
                    PtrOpDef::Eq,
                    vec![ptr.clone(), ptr.clone()],
                    vec![ptr.clone(), ptr.clone(), bool_t()],
                ),
                (
                    PtrOpDef::Swap,
                    vec![ptr.clone(), ty.clone()],
                    vec![ptr.clone(), ty.clone()],
                ),
                (
                    PtrOpDef::Dup,
                    vec![ptr.clone()],
                    vec![ptr.clone(), ptr.clone()],
                ),
                (
                    PtrOpDef::Free,
                    vec![ptr.clone()],
                    vec![option_type([ty.clone()]).into()],
                ),
            ];
            for (def, inputs, outputs) in cases {
                let op = def.with_type(ty.clone());
                let ext = op.clone().to_extension_op().unwrap();
                assert_eq!(
                    ext.signature().into_owned(),
                    Signature::new(inputs, outputs)
                );
                assert_eq!(PtrOp::from_op(&ext).unwrap(), op);
            }
            for (def, inputs, outputs) in [
                (
                    PtrOpDef::Read,
                    vec![ptr.clone()],
                    vec![ptr.clone(), ty.clone()],
                ),
                (
                    PtrOpDef::Write,
                    vec![ptr.clone(), ty.clone()],
                    vec![ptr.clone()],
                ),
            ] {
                let result = def.with_type(ty.clone()).to_extension_op();
                if ty == qb_t() {
                    assert!(result.is_err());
                    assert!(def.instantiate(&[ty.clone().into()]).is_err());
                } else {
                    assert_eq!(
                        result.unwrap().signature().into_owned(),
                        Signature::new(inputs, outputs)
                    );
                }
            }
        }
    }

    #[test]
    fn map_rows_and_roundtrip() {
        use crate::ops::DataflowOpTrait;
        for (inputs, outputs) in [
            (vec![], vec![]),
            (vec![bool_t(), qb_t()], vec![qb_t()]),
            (vec![], vec![bool_t(), qb_t()]),
            (vec![qb_t()], vec![]),
        ] {
            let ty = qb_t();
            let op = PtrOp::map(ty.clone(), inputs.clone(), outputs.clone());
            let mut callback_inputs = vec![ty.clone()];
            callback_inputs.extend(inputs.clone());
            let mut callback_outputs = vec![ty.clone()];
            callback_outputs.extend(outputs.clone());
            let callback =
                Type::new_function(FuncValueType::new(callback_inputs, callback_outputs));
            let mut expected_inputs = vec![ptr_type(ty.clone()), callback];
            expected_inputs.extend(inputs);
            let mut expected_outputs = vec![ptr_type(ty)];
            expected_outputs.extend(outputs.clone());
            let expected = Signature::new(expected_inputs, expected_outputs);
            let ext = op.clone().to_extension_op().unwrap();
            assert_eq!(ext.signature().into_owned(), expected);
            assert_eq!(PtrOp::from_op(&ext).unwrap(), op);
            let mut builder = DFGBuilder::new(expected).unwrap();
            let wires = builder.input_wires().collect::<Vec<_>>();
            let (ptr, results) = builder
                .add_map_ptr(
                    wires[0],
                    wires[1],
                    qb_t(),
                    wires[2..].iter().copied(),
                    outputs,
                )
                .unwrap();
            assert_eq!(builder.get_wire_type(ptr).unwrap(), ptr_type(qb_t()));
            let outputs = std::iter::once(ptr).chain(results);
            builder
                .finish_hugr_with_outputs(outputs)
                .unwrap()
                .validate()
                .unwrap();
        }
        assert!(PtrOpDef::Map.instantiate(&[qb_t().into()]).is_err());
        assert!(
            PtrOpDef::Map
                .instantiate(&[
                    qb_t().into(),
                    bool_t().into(),
                    TypeArg::new_list::<Type>([]),
                ])
                .is_err()
        );
    }

    #[test]
    fn linear_pointer_lifecycle() {
        let option: Type = option_type([qb_t()]).into();
        let mut builder = DFGBuilder::new(Signature::new(
            [qb_t(), qb_t()],
            [option.clone(), option, qb_t()],
        ))
        .unwrap();
        let [value, replacement] = builder.input_wires_arr();
        let ptr = builder.add_new_ptr(value).unwrap();
        let (ptr, other) = builder.add_dup_ptr(ptr, qb_t()).unwrap();
        let (ptr, old_value) = builder.add_swap_ptr(ptr, replacement).unwrap();
        let first = builder.add_free_ptr(ptr, qb_t()).unwrap();
        let last = builder.add_free_ptr(other, qb_t()).unwrap();
        builder
            .finish_hugr_with_outputs([first, last, old_value])
            .unwrap()
            .validate()
            .unwrap();
    }

    #[test]
    fn eq_preserves_linear_handles() {
        let ptr = ptr_type(qb_t());
        let signature = Signature::new(
            [ptr.clone(), ptr.clone()],
            [ptr.clone(), ptr.clone(), bool_t()],
        );
        let mut builder = DFGBuilder::new(signature).unwrap();
        let [lhs, rhs] = builder.input_wires_arr();
        let (lhs, rhs, equal) = builder.add_eq_ptr(lhs, rhs, qb_t()).unwrap();
        assert_eq!(lhs.source().index(), 0);
        assert_eq!(rhs.source().index(), 1);
        assert_eq!(equal.source().index(), 2);
        builder
            .finish_hugr_with_outputs([lhs, rhs, equal])
            .unwrap()
            .validate()
            .unwrap();

        for copy in [false, true] {
            let outputs = if copy {
                vec![ptr.clone(), ptr.clone(), ptr.clone(), bool_t()]
            } else {
                vec![ptr.clone(), bool_t()]
            };
            let mut builder =
                DFGBuilder::new(Signature::new([ptr.clone(), ptr.clone()], outputs)).unwrap();
            let [lhs, rhs] = builder.input_wires_arr();
            let (lhs, rhs, equal) = builder.add_eq_ptr(lhs, rhs, qb_t()).unwrap();
            let outputs = if copy {
                vec![lhs, rhs, rhs, equal]
            } else {
                vec![lhs, equal]
            };
            assert!(builder.finish_hugr_with_outputs(outputs).is_err());
        }
        assert!(PtrOpDef::Eq.instantiate(&[]).is_err());
        assert!(
            PtrOpDef::Eq
                .instantiate(&[qb_t().into(), bool_t().into()])
                .is_err()
        );
        let mut builder = DFGBuilder::new(Signature::new(
            [ptr_type(qb_t()), ptr_type(bool_t())],
            [ptr_type(qb_t()), ptr_type(qb_t()), bool_t()],
        ))
        .unwrap();
        let [lhs, rhs] = builder.input_wires_arr();
        let (lhs, rhs, equal) = builder.add_eq_ptr(lhs, rhs, qb_t()).unwrap();
        assert!(builder.finish_hugr_with_outputs([lhs, rhs, equal]).is_err());
    }

    #[test]
    fn implicit_pointer_copy_or_drop_is_rejected() {
        let ptr = ptr_type(bool_t());
        for outputs in [vec![], vec![ptr.clone(), ptr.clone()]] {
            let builder = DFGBuilder::new(Signature::new([ptr.clone()], outputs.clone())).unwrap();
            let [input] = builder.input_wires_arr();
            assert!(
                builder
                    .finish_hugr_with_outputs(vec![input; outputs.len()])
                    .is_err()
            );
        }
    }
}
