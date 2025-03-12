use std::{hash, sync::{self, Arc}};

use crate::{extension::{prelude::usize_t, resolution::{ExtensionResolutionError, WeakExtensionRegistry}, simple_op::{try_from_name, MakeOpDef, OpLoadError}, ExtensionId, ExtensionSet, OpDef, SignatureFunc, TypeDefBound}, ops::{constant::{CustomConst, TryHash, ValueName}, Value}, types::{type_param::TypeParam, CustomType, PolyFuncType, Signature, Type, TypeArg, TypeBound, TypeName}, Extension};

use super::array::ArrayValue;

use delegate::delegate;
use lazy_static::lazy_static;

/// Reported unique name of the array type.
pub const CONST_ARRAY_TYPENAME: TypeName = TypeName::new_inline("const_array");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("collections.const_array");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, derive_more::From)]
/// Statically sized array of values, all of the same type.
pub struct ConstArrayValue(ArrayValue);

impl ConstArrayValue {
    delegate! {
        to self.0 {
            /// Returns the type of the `[ConstArrayValue]` as a `[CustomType]`.`
            pub fn get_element_type(&self) -> &Type;
        }

    }

    /// Create a new [CustomConst] for an array of values of type `typ`.
    /// That all values are of type `typ` is not checked here.
    pub fn new(typ: Type, contents: impl IntoIterator<Item = Value>) -> Self {
        Self(ArrayValue::new(typ, contents))
    }

    /// Create a new [CustomConst] for an empty array of values of type `typ`.
    pub fn new_empty(typ: Type) -> Self {
        Self(ArrayValue::new_empty(typ))
    }

    /// Returns the type of the `[ConstArrayValue]` as a `[CustomType]`.`
    pub fn custom_type(&self) -> CustomType {
        todo!()
    }
}

impl TryHash for ConstArrayValue {
    fn try_hash(&self, st: &mut dyn hash::Hasher) -> bool {
        self.0.try_hash(st)
    }
}

#[typetag::serde]
impl CustomConst for ConstArrayValue {
    fn name(&self) -> ValueName {
        ValueName::new_inline("const_array")
    }

    fn get_type(&self) -> Type {
        self.custom_type().into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    delegate! {
        to self.0 {
            fn update_extensions(
                &mut self,
                extensions: &WeakExtensionRegistry,
            ) -> Result<(), ExtensionResolutionError>;

            fn extension_reqs(&self) -> ExtensionSet;
        }
    }

}

lazy_static! {
    /// Extension for array operations.
    pub static ref EXTENSION: Arc<Extension> = {
        Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
            extension.add_type(
                    CONST_ARRAY_TYPENAME,
                    vec![TypeBound::Copyable.into()],
                    "Fixed-length constant array".into(),
                    TypeDefBound::from_params(vec![0] ),
                    extension_ref,
                )
                .unwrap();

            ConstArrayOpDef::load_all_ops(extension, extension_ref).unwrap();
        })
    };
}

fn instantiate_const_array_custom_type(extension: &Extension, element_ty: impl Into<TypeArg>) -> CustomType {
    extension.get_type(&CONST_ARRAY_TYPENAME).unwrap().instantiate([element_ty.into()]).expect("const array parameters are valid")
}

pub fn const_array_custom_type(element_ty: impl Into<TypeArg>) -> CustomType {
    instantiate_const_array_custom_type(&EXTENSION, element_ty)
}

/// Instantiate a new const_array type given an element type.
pub fn const_array_type(element_ty: impl Into<TypeArg>) -> Type {
    const_array_custom_type(element_ty).into()
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, strum::EnumIter, strum::IntoStaticStr, strum::EnumString)]
#[allow(non_camel_case_types, missing_docs)]
#[non_exhaustive]
pub enum ConstArrayOpDef {
    get,
    len
}

impl MakeOpDef for ConstArrayOpDef {
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, _extension_ref: &sync::Weak<Extension>) -> SignatureFunc {
        use TypeBound::*;
        match self {
            Self::get => {
                let t_param = TypeParam::from(Copyable);
                let elem_ty = Type::new_var_use(0, Copyable);
                let array_ty = const_array_type(elem_ty.clone());
                PolyFuncType::new([t_param], Signature::new(vec![array_ty, usize_t()], elem_ty)).into()
            }
,
            Self::len => {
                let t_param = TypeParam::from(Copyable);
                let elem_ty = Type::new_var_use(0, Copyable);
                let array_ty = const_array_type(elem_ty.clone());
                PolyFuncType::new([t_param], Signature::new(array_ty, usize_t())).into()
            },
        }
    }

    fn extension_ref(&self) -> sync::Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn description(&self) -> String {
        match self {
            Self::get => "TODO",
            Self::len => "TODO",
        }
        .into()
    }
}
