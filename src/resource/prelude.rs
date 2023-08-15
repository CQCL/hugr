//! Prelude extension - available in all contexts, defining common types,
//! operations and constants.
use lazy_static::lazy_static;
use smol_str::SmolStr;

use crate::{
    resource::TypeDefBound,
    types::{
        type_param::{TypeArg, TypeParam},
        CustomType, Type, TypeBound,
    },
    Resource,
};

lazy_static! {
    /// Prelude resource
    pub static ref PRELUDE: Resource = {
        let mut prelude = Resource::new(SmolStr::new_inline("prelude"));
        prelude
            .add_type(
                SmolStr::new_inline("float64"),
                vec![],
                "float64".into(),
                TypeDefBound::Explicit(crate::types::TypeBound::Copyable),
            )
            .unwrap();

            prelude
            .add_type(
                SmolStr::new_inline("usize"),
                vec![],
                "usize".into(),
                TypeDefBound::Explicit(crate::types::TypeBound::Eq),
            )
            .unwrap();


            prelude
            .add_type(
                SmolStr::new_inline("array"),
                vec![TypeParam::Type(None), TypeParam::USize],
                "array".into(),
                TypeDefBound::FromParams(vec![0]),
            )
            .unwrap();

            prelude
            .add_type(
                SmolStr::new_inline("qubit"),
                vec![],
                "qubit".into(),
                TypeDefBound::NoBound,
            )
            .unwrap();
        prelude
    };
}

pub(crate) const USIZE_CUSTOM_T: CustomType = CustomType::new_simple(
    SmolStr::new_inline("usize"),
    SmolStr::new_inline("prelude"),
    Some(TypeBound::Eq),
);

pub(crate) const QB_CUSTOM_T: CustomType = CustomType::new_simple(
    SmolStr::new_inline("qubit"),
    SmolStr::new_inline("prelude"),
    None,
);

pub(crate) const QB_T: Type = Type::new_extension(QB_CUSTOM_T);
pub(crate) const USIZE_T: Type = Type::new_extension(USIZE_CUSTOM_T);

/// Initialize a new array of type `typ` of length `size`
pub fn new_array(typ: Type, size: u64) -> Type {
    let array_def = PRELUDE.get_type("array").unwrap();
    let custom_t = array_def
        .instantiate_concrete(vec![TypeArg::Type(typ), TypeArg::USize(size)])
        .unwrap();
    Type::new_extension(custom_t)
}

pub(crate) const ERROR_TYPE: Type = Type::new_extension(CustomType::new_simple(
    smol_str::SmolStr::new_inline("error"),
    smol_str::SmolStr::new_inline("prelude"),
    Some(TypeBound::Eq),
));
