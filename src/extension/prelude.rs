//! Prelude extension - available in all contexts, defining common types,
//! operations and constants.
use lazy_static::lazy_static;
use smol_str::SmolStr;

use crate::{
    extension::TypeDefBound,
    types::{
        type_param::{TypeArg, TypeParam},
        CustomCheckFailure, CustomType, Type, TypeBound,
    },
    values::{CustomConst, KnownTypeConst},
    Extension,
};

use super::ExtensionRegistry;

lazy_static! {
    /// Prelude extension
    pub static ref PRELUDE: Extension = {
        let mut prelude = Extension::new(SmolStr::new_inline("prelude"));
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
                vec![TypeParam::Type(TypeBound::Any), TypeParam::max_nat()],
                "array".into(),
                TypeDefBound::FromParams(vec![0]),
            )
            .unwrap();

        prelude
            .add_type(
                SmolStr::new_inline("qubit"),
                vec![],
                "qubit".into(),
                TypeDefBound::Explicit(TypeBound::Any),
            )
            .unwrap();
        prelude
    };
}

/// An extension registry containing only the prelude
pub fn prelude_registry() -> ExtensionRegistry {
    [PRELUDE.to_owned()].into()
}

pub(crate) const USIZE_CUSTOM_T: CustomType = CustomType::new_simple(
    SmolStr::new_inline("usize"),
    SmolStr::new_inline("prelude"),
    TypeBound::Eq,
);

pub(crate) const QB_CUSTOM_T: CustomType = CustomType::new_simple(
    SmolStr::new_inline("qubit"),
    SmolStr::new_inline("prelude"),
    TypeBound::Any,
);

/// Qubit type.
pub const QB_T: Type = Type::new_extension(QB_CUSTOM_T);
/// Unsigned size type.
pub const USIZE_T: Type = Type::new_extension(USIZE_CUSTOM_T);
/// Boolean type - Sum of two units.
pub const BOOL_T: Type = Type::new_simple_predicate(2);

/// Initialize a new array of type `typ` of length `size`
pub fn new_array(typ: Type, size: u64) -> Type {
    let array_def = PRELUDE.get_type("array").unwrap();
    let custom_t = array_def
        .instantiate_concrete(vec![TypeArg::Type(typ), TypeArg::BoundedNat(size)])
        .unwrap();
    Type::new_extension(custom_t)
}

pub(crate) const ERROR_TYPE: Type = Type::new_extension(CustomType::new_simple(
    smol_str::SmolStr::new_inline("error"),
    smol_str::SmolStr::new_inline("prelude"),
    TypeBound::Eq,
));

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Structure for holding constant usize values.
pub struct ConstUsize(u64);

impl ConstUsize {
    /// Creates a new [`ConstUsize`].
    pub fn new(value: u64) -> Self {
        Self(value)
    }
}

#[typetag::serde]
impl CustomConst for ConstUsize {
    fn name(&self) -> SmolStr {
        format!("ConstUsize({:?})", self.0).into()
    }

    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        self.check_known_type(typ)
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::values::downcast_equal_consts(self, other)
    }
}

impl KnownTypeConst for ConstUsize {
    const TYPE: CustomType = USIZE_CUSTOM_T;
}
