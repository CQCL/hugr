//! Pointer type and operations.

use std::sync::{Arc, LazyLock, Weak};

use strum::{EnumIter, EnumString, IntoStaticStr};

use crate::Wire;
use crate::builder::{BuildError, Dataflow};
use crate::extension::TypeDefBound;
use crate::ops::OpName;
use crate::types::{CustomType, PolyFuncType, Signature, Type, TypeBound, TypeName};
use crate::{
    Extension,
    extension::{
        ExtensionId, OpDef, SignatureError, SignatureFunc,
        simple_op::{
            HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
        },
    },
    ops::custom::ExtensionOp,
    type_row,
    types::type_param::{TypeArg, TypeParam},
};
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs)]
#[non_exhaustive]
/// Pointer operation definitions.
pub enum PtrOpDef {
    /// Create a new pointer.
    New,
    /// Read a value from a pointer.
    Read,
    /// Write a value to a pointer.
    Write,
}

impl PtrOpDef {
    /// Create a new concrete pointer operation with the given value type.
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
        let ptr_t: Type =
            ptr_custom_type(Type::new_var_use(0, TypeBound::Copyable), extension_ref).into();
        let inner_t = Type::new_var_use(0, TypeBound::Copyable);
        let body = match self {
            PtrOpDef::New => Signature::new(inner_t, ptr_t),
            PtrOpDef::Read => Signature::new(ptr_t, inner_t),
            PtrOpDef::Write => Signature::new(vec![ptr_t, inner_t], type_row![]),
        };

        PolyFuncType::new(TYPE_PARAMS, body).into()
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
            PtrOpDef::Read => "Read a value from a pointer.".into(),
            PtrOpDef::Write => "Write a value to a pointer, overwriting existing value.".into(),
        }
    }
}

/// Name of pointer extension.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("ptr");
/// Name of pointer type.
pub const PTR_TYPE_ID: TypeName = TypeName::new_inline("ptr");
const TYPE_PARAMS: [TypeParam; 1] = [TypeParam::RuntimeType(TypeBound::Copyable)];
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

/// Extension for pointer operations.
fn extension() -> Arc<Extension> {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        extension
            .add_type(
                PTR_TYPE_ID,
                TYPE_PARAMS.into(),
                "Standard extension pointer type.".into(),
                TypeDefBound::copyable(),
                extension_ref,
            )
            .unwrap();
        PtrOpDef::load_all_ops(extension, extension_ref).unwrap();
    })
}

/// Reference to the pointer Extension.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(extension);

/// Integer type of a given bit width (specified by the `TypeArg`).  Depending on
/// the operation, the semantic interpretation may be unsigned integer, signed
/// integer or bit string.
fn ptr_custom_type(ty: impl Into<Type>, extension_ref: &Weak<Extension>) -> CustomType {
    let ty = ty.into();
    CustomType::new(
        PTR_TYPE_ID,
        [ty.into()],
        EXTENSION_ID,
        TypeBound::Copyable,
        extension_ref,
    )
}

/// Integer type of a given bit width (specified by the `TypeArg`).
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
}

impl PtrOp {
    fn new(op: PtrOpDef, ty: Type) -> Self {
        Self { def: op, ty }
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
        vec![self.ty.clone().into()]
    }
}

impl MakeRegisteredOp for PtrOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
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

    /// Add a "ptr.Read" op.
    fn add_read_ptr(&mut self, ptr_wire: Wire, ty: Type) -> Result<Wire, BuildError> {
        let handle = self.add_dataflow_op(PtrOpDef::Read.with_type(ty.clone()), [ptr_wire])?;
        Ok(handle.out_wire(0))
    }

    /// Add a "ptr.Write" op.
    fn add_write_ptr(&mut self, ptr_wire: Wire, val_wire: Wire) -> Result<(), BuildError> {
        let ty = self.get_wire_type(val_wire)?;

        let handle = self.add_dataflow_op(PtrOpDef::Write.with_type(ty), [ptr_wire, val_wire])?;
        debug_assert_eq!(handle.outputs().len(), 0);
        Ok(())
    }
}

impl<D: Dataflow> PtrOpBuilder for D {}

impl HasConcrete for PtrOpDef {
    type Concrete = PtrOp;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        let ty = match type_args {
            [TypeArg::Runtime(ty)] => ty.clone(),
            _ => return Err(SignatureError::InvalidTypeArgs.into()),
        };

        Ok(self.with_type(ty))
    }
}

impl HasDef for PtrOp {
    type Def = PtrOpDef;
}

#[cfg(test)]
pub(crate) mod test {
    use crate::HugrView;
    use crate::builder::DFGBuilder;
    use crate::extension::prelude::bool_t;
    use crate::ops::ExtensionOp;
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
            let mut builder = DFGBuilder::new(Signature::new(in_row.clone(), type_row![])).unwrap();

            let in_wires: [Wire; 2] = builder.input_wires_arr();
            for (ty, w) in in_row.into_iter().zip(in_wires.iter()) {
                let new_ptr = builder.add_new_ptr(*w).unwrap();
                let read = builder.add_read_ptr(new_ptr, ty).unwrap();
                builder.add_write_ptr(new_ptr, read).unwrap();
            }

            builder.finish_hugr_with_outputs([]).unwrap()
        };
        assert_matches!(hugr.validate(), Ok(()));
    }
}
