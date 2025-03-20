//! An extension for working with static arrays.
//!
//! The extension `collections.static_arrays` models globally available constant
//! arrays of [TypeBound::Copyable] values.
//!
//! The type `static_array<T>` is parameterised by its element type. Note that
//! unlike `collections.array.array` the length of a static array is not tracked
//! in type args.
//!
//! The [CustomConst] [StaticArrayValue] is the only manner by which a value of
//! `static_array` type can be obtained.
//!
//! Operations provided:
//!  * `get<T: Copyable>: [static_array<T>, prelude.usize] -> [[] + [T]]`
//!  * `len<T: Copyable>: [static_array<T>] -> [prelude.usize]`
use std::{
    hash::{self, Hash as _}, iter,
    sync::{self, Arc},
};

use crate::{
    builder::{BuildError, Dataflow},
    extension::{
        prelude::{option_type, usize_t},
        resolution::{resolve_type_extensions, resolve_value_extensions, ExtensionResolutionError, WeakExtensionRegistry},
        simple_op::{
            try_from_name, HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp,
            OpLoadError,
        },
        ExtensionId, ExtensionSet, OpDef, SignatureError, SignatureFunc, TypeDef,
    },
    ops::{
        constant::{maybe_hash_values, CustomConst, TryHash, ValueName},
        ExtensionOp, NamedOp, OpName, Value,
    },
    types::{
        type_param::{TypeArgError, TypeParam},
        ConstTypeError, CustomCheckFailure, CustomType, PolyFuncType, Signature, Type, TypeArg,
        TypeBound, TypeName,
    },
    Extension, Wire,
};

use lazy_static::lazy_static;

/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_static_unchecked("collections.static_array");
/// Reported unique name of the array type.
pub const STATIC_ARRAY_TYPENAME: TypeName = TypeName::new_inline("static_array");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, derive_more::From)]
/// Statically sized array of values, all of the same [TypeBound::Copyable]
/// type.
pub struct StaticArrayValue {
    values: Vec<Value>,
    typ: Type,
    /// The name of the `StaticArrayValue`.
    pub name: String,
}

impl StaticArrayValue {
    /// Returns the type of values inside the `[StaticArrayValue]`.
    pub fn get_element_type(&self) -> &Type {
        &self.typ
    }

    /// Returns the values contained inside the `[StaticArrayValue]`.
    pub fn get_contents(&self) -> &[Value] {
        &self.values
    }

    /// Create a new [CustomConst] for an array of values of type `typ`.
    /// That all values are of type `typ` is not checked here.
    pub fn try_new(
        name: impl ToString,
        typ: Type,
        contents: impl IntoIterator<Item = Value>,
    ) -> Result<Self, ConstTypeError> {
        if !TypeBound::Copyable.contains(typ.least_upper_bound()) {
            return Err(CustomCheckFailure::Message(format!(
                "Failed to construct a StaticArrayValue with non-Copyable type: {typ}"
            ))
            .into());
        }
        Ok(Self {
            typ,
            values: contents.into_iter().collect(),
            name: name.to_string(),
        })
    }

    /// Create a new [CustomConst] for an empty array of values of type `typ`.
    pub fn try_new_empty(name: impl ToString, typ: Type) -> Result<Self, ConstTypeError> {
        Self::try_new(name, typ, iter::empty())
    }

    /// Returns the type of the `[StaticArrayValue]` as a `[CustomType]`.`
    pub fn custom_type(&self) -> CustomType {
        static_array_custom_type(self.get_element_type().clone())
    }
}

impl TryHash for StaticArrayValue {
    fn try_hash(&self, mut st: &mut dyn hash::Hasher) -> bool {
        maybe_hash_values(&self.values, &mut st) && {
            self.name.hash(&mut st);
            self.typ.hash(&mut st);
            true
        }
    }
}

#[typetag::serde]
impl CustomConst for StaticArrayValue {
    fn name(&self) -> ValueName {
        ValueName::new_inline("const_array")
    }

    fn get_type(&self) -> Type {
        self.custom_type().into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::union_over(self.values.iter().map(Value::extension_reqs))
            .union(EXTENSION_ID.into())
    }

    fn update_extensions(
        &mut self,
        extensions: &WeakExtensionRegistry,
    ) -> Result<(), ExtensionResolutionError> {
        for val in &mut self.values {
            resolve_value_extensions(val, extensions)?;
        }
        resolve_type_extensions(&mut self.typ, extensions)
    }
}

lazy_static! {
    /// Extension for array operations.
    pub static ref EXTENSION: Arc<Extension> = {
        use TypeBound::Copyable;
        Extension::new_arc(EXTENSION_ID.to_owned(), VERSION, |extension, extension_ref| {
            extension.add_type(
                    STATIC_ARRAY_TYPENAME,
                    vec![Copyable.into()],
                    "Fixed-length constant array".into(),
                    Copyable.into(),
                    extension_ref,
                )
                .unwrap();

            StaticArrayOpDef::load_all_ops(extension, extension_ref).unwrap();
        })
    };
}

fn instantiate_const_static_array_custom_type(
    def: &TypeDef,
    element_ty: impl Into<TypeArg>,
) -> CustomType {
    def.instantiate([element_ty.into()])
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Instantiate a new `static_array` [CustomType] given an element type.
pub fn static_array_custom_type(element_ty: impl Into<TypeArg>) -> CustomType {
    instantiate_const_static_array_custom_type(
        EXTENSION.get_type(&STATIC_ARRAY_TYPENAME).unwrap(),
        element_ty,
    )
}

/// Instantiate a new `static_array` [Type] given an element type.
pub fn static_array_type(element_ty: impl Into<TypeArg>) -> Type {
    static_array_custom_type(element_ty).into()
}

#[derive(
    Clone,
    Copy,
    Debug,
    Hash,
    PartialEq,
    Eq,
    strum::EnumIter,
    strum::IntoStaticStr,
    strum::EnumString,
)]
#[allow(non_camel_case_types, missing_docs)]
#[non_exhaustive]
pub enum StaticArrayOpDef {
    get,
    len,
}

impl StaticArrayOpDef {
    fn signature_from_def(&self, def: &TypeDef, _: &sync::Weak<Extension>) -> SignatureFunc {
        use TypeBound::*;
        let t_param = TypeParam::from(Copyable);
        let elem_ty = Type::new_var_use(0, Copyable);
        let array_ty: Type =
            instantiate_const_static_array_custom_type(def, elem_ty.clone()).into();
        match self {
            Self::get => PolyFuncType::new(
                [t_param],
                Signature::new(vec![array_ty, usize_t()], Type::from(option_type(elem_ty))),
            )
            .into(),
            Self::len => PolyFuncType::new([t_param], Signature::new(array_ty, usize_t())).into(),
        }
    }
}

impl MakeOpDef for StaticArrayOpDef {
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, extension_ref: &sync::Weak<Extension>) -> SignatureFunc {
        self.signature_from_def(
            EXTENSION.get_type(&STATIC_ARRAY_TYPENAME).unwrap(),
            extension_ref,
        )
    }

    fn extension_ref(&self) -> sync::Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn description(&self) -> String {
        match self {
            Self::get => "Get an element from a static array",
            Self::len => "Get the length of a static array",
        }
        .into()
    }

    // This method is re-defined here since we need to pass the static array
    // type def while computing the signature, to avoid recursive loops
    // initializing the extension.
    fn add_to_extension(
        &self,
        extension: &mut Extension,
        extension_ref: &sync::Weak<Extension>,
    ) -> Result<(), crate::extension::ExtensionBuildError> {
        let sig = self.signature_from_def(
            extension.get_type(&STATIC_ARRAY_TYPENAME).unwrap(),
            extension_ref,
        );
        let def = extension.add_op(self.name(), self.description(), sig, extension_ref)?;

        self.post_opdef(def);

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Concrete array operation.
pub struct StaticArrayOp {
    /// The operation definition.
    pub def: StaticArrayOpDef,
    /// The element type of the array.
    pub elem_ty: Type,
}

impl NamedOp for StaticArrayOp {
    fn name(&self) -> OpName {
        self.def.name()
    }
}

impl MakeExtensionOp for StaticArrayOp {
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = StaticArrayOpDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![self.elem_ty.clone().into()]
    }
}

impl HasDef for StaticArrayOp {
    type Def = StaticArrayOpDef;
}

impl HasConcrete for StaticArrayOpDef {
    type Concrete = StaticArrayOp;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        use TypeBound::Copyable;
        match type_args {
            [arg] => {
                let elem_ty = arg
                    .as_type()
                    .filter(|t| Copyable.contains(t.least_upper_bound()))
                    .ok_or(SignatureError::TypeArgMismatch(
                        TypeArgError::TypeMismatch {
                            param: Copyable.into(),
                            arg: arg.clone(),
                        },
                    ))?;

                Ok(StaticArrayOp {
                    def: *self,
                    elem_ty,
                })
            }
            _ => Err(
                SignatureError::TypeArgMismatch(TypeArgError::WrongNumberArgs(type_args.len(), 1))
                    .into(),
            ),
        }
    }
}

impl MakeRegisteredOp for StaticArrayOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn extension_ref(&self) -> sync::Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

/// A trait for building static array operations in a dataflow graph.
pub trait StaticArrayOpBuilder: Dataflow {
    /// Adds a `get` operation to retrieve an element from a static array.
    ///
    /// # Arguments
    ///
    /// + `elem_ty` - The type of the elements in the array.
    /// + `array` - The wire carrying the array.
    /// + `index` - The wire carrying the index.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the wire for the retrieved element or a `BuildError`.
    fn add_static_array_get(
        &mut self,
        elem_ty: Type,
        array: Wire,
        index: Wire,
    ) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(
                StaticArrayOp {
                    def: StaticArrayOpDef::get,
                    elem_ty,
                }
                .to_extension_op()
                .unwrap(),
                [array, index],
            )?
            .out_wire(0))
    }

    /// Adds a `len` operation to get the length of a static array.
    ///
    /// # Arguments
    ///
    /// + `elem_ty` - The type of the elements in the array.
    /// + `array` - The wire representing the array.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the wire for the length of the array or a `BuildError`.
    fn add_static_array_len(&mut self, elem_ty: Type, array: Wire) -> Result<Wire, BuildError> {
        Ok(self
            .add_dataflow_op(
                StaticArrayOp {
                    def: StaticArrayOpDef::len,
                    elem_ty,
                }
                .to_extension_op()
                .unwrap(),
                [array],
            )?
            .out_wire(0))
    }
}

impl<T: Dataflow> StaticArrayOpBuilder for T {}

#[cfg(test)]
mod test {
    use crate::{
        builder::{DFGBuilder, DataflowHugr as _},
        extension::prelude::{qb_t, ConstUsize, PRELUDE_ID},
        type_row,
    };

    use super::*;

    #[test]
    fn const_static_array_copyable() {
        let _good = StaticArrayValue::try_new_empty("good", Type::UNIT).unwrap();
        let _bad = StaticArrayValue::try_new_empty("good", qb_t()).unwrap_err();
    }

    #[test]
    fn all_ops() {
        let _ = {
            let mut builder = DFGBuilder::new(
                Signature::new(type_row![], Type::from(option_type(usize_t())))
                    .with_extension_delta(ExtensionSet::from_iter([PRELUDE_ID, EXTENSION_ID])),
            )
            .unwrap();
            let array = builder.add_load_value(
                StaticArrayValue::try_new(
                    "t",
                    usize_t(),
                    (1..999).map(|x| ConstUsize::new(x).into()),
                )
                .unwrap(),
            );
            let _ = builder.add_static_array_len(usize_t(), array).unwrap();
            let index = builder.add_load_value(ConstUsize::new(777));
            let x = builder
                .add_static_array_get(usize_t(), array, index)
                .unwrap();
            builder.finish_hugr_with_outputs([x]).unwrap()
        };
    }
}
