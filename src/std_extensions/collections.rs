//! List type and operations.

use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::{
    extension::{ExtensionId, TypeDef, TypeDefBound},
    types::{
        type_param::{TypeArg, TypeParam},
        CustomCheckFailure, CustomType, FunctionType, PolyFuncType, Type, TypeBound,
    },
    values::{CustomConst, Value},
    Extension,
};

/// Reported unique name of the list type.
pub const LIST_TYPENAME: SmolStr = SmolStr::new_inline("List");
/// Pop operation name.
pub const POP_NAME: SmolStr = SmolStr::new_inline("pop");
/// Push operation name.
pub const PUSH_NAME: SmolStr = SmolStr::new_inline("push");
/// Reported unique name of the extension
pub const EXTENSION_NAME: ExtensionId = ExtensionId::new_unchecked("Collections");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Dynamically sized list of values, all of the same type.
pub struct ListValue(Vec<Value>);

impl ListValue {
    /// Create a new [CustomConst] for a list of values.
    /// (The caller will need these to all be of the same type, but that is not checked here.)
    pub fn new(contents: Vec<Value>) -> Self {
        Self(contents)
    }
}

#[typetag::serde]
impl CustomConst for ListValue {
    fn name(&self) -> SmolStr {
        SmolStr::new_inline("list")
    }

    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        let error = || {
            // TODO more bespoke errors
            CustomCheckFailure::Message("List type check fail.".to_string())
        };

        get_type(&LIST_TYPENAME)
            .check_custom(typ)
            .map_err(|_| error())?;

        // constant can only hold classic type.
        let [TypeArg::Type { ty: t }] = typ.args() else {
            return Err(error());
        };

        // check all values are instances of the element type
        for val in &self.0 {
            t.check_type(val).map_err(|_| error())?;
        }
        Ok(())
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::values::downcast_equal_consts(self, other)
    }
}
const TP: TypeParam = TypeParam::Type(TypeBound::Any);

fn extension() -> Extension {
    let mut extension = Extension::new(EXTENSION_NAME);

    extension
        .add_type(
            LIST_TYPENAME,
            vec![TP],
            "Generic dynamically sized list of type T.".into(),
            TypeDefBound::FromParams(vec![0]),
        )
        .unwrap();
    let list_type_def = extension.get_type(&LIST_TYPENAME).unwrap();

    let (l, e) = list_and_elem_type(list_type_def);
    extension
        .add_op_type_scheme_simple(
            POP_NAME,
            "Pop from back of list".into(),
            PolyFuncType::new(
                vec![TP],
                FunctionType::new(vec![l.clone()], vec![l.clone(), e.clone()]),
            ),
        )
        .unwrap();
    extension
        .add_op_type_scheme_simple(
            PUSH_NAME,
            "Push to back of list".into(),
            PolyFuncType::new(vec![TP], FunctionType::new(vec![l.clone(), e], vec![l])),
        )
        .unwrap();
    extension
}

lazy_static! {
    /// Collections extension definition.
    pub static ref EXTENSION: Extension = extension();
}

fn get_type(name: &str) -> &TypeDef {
    EXTENSION.get_type(name).unwrap()
}

fn list_and_elem_type(list_type_def: &TypeDef) -> (Type, Type) {
    let elem_type = Type::new_var_use(0, TypeBound::Any);
    let list_type = Type::new_extension(
        list_type_def
            .instantiate(vec![TypeArg::new_var_use(0, TP)])
            .unwrap(),
    );
    (list_type, elem_type)
}
#[cfg(test)]
mod test {
    use crate::{
        extension::{
            prelude::{ConstUsize, QB_T, USIZE_T},
            ExtensionRegistry, OpDef, PRELUDE,
        },
        std_extensions::arithmetic::float_types::{self, ConstF64, FLOAT64_TYPE},
        types::{type_param::TypeArg, Type, TypeRow},
        Extension,
    };

    use super::*;
    fn get_op(name: &str) -> &OpDef {
        EXTENSION.get_op(name).unwrap()
    }
    #[test]
    fn test_extension() {
        let r: Extension = extension();
        assert_eq!(r.name(), &EXTENSION_NAME);
        let ops = r.operations();
        assert_eq!(ops.count(), 2);
    }

    #[test]
    fn test_list() {
        let r: Extension = extension();
        let list_def = r.get_type(&LIST_TYPENAME).unwrap();

        let list_type = list_def
            .instantiate([TypeArg::Type { ty: USIZE_T }])
            .unwrap();

        assert!(list_def
            .instantiate([TypeArg::BoundedNat { n: 3 }])
            .is_err());

        list_def.check_custom(&list_type).unwrap();
        let list_value = ListValue(vec![ConstUsize::new(3).into()]);

        list_value.check_custom_type(&list_type).unwrap();

        let wrong_list_value = ListValue(vec![ConstF64::new(1.2).into()]);
        assert!(wrong_list_value.check_custom_type(&list_type).is_err());
    }

    #[test]
    fn test_list_ops() {
        let reg = ExtensionRegistry::try_new([
            EXTENSION.to_owned(),
            PRELUDE.to_owned(),
            float_types::extension(),
        ])
        .unwrap();
        let pop_sig = get_op(&POP_NAME)
            .compute_signature(&[TypeArg::Type { ty: QB_T }], &reg)
            .unwrap();

        let list_type = Type::new_extension(CustomType::new(
            LIST_TYPENAME,
            vec![TypeArg::Type { ty: QB_T }],
            EXTENSION_NAME,
            TypeBound::Any,
        ));

        let both_row: TypeRow = vec![list_type.clone(), QB_T].into();
        let just_list_row: TypeRow = vec![list_type].into();
        assert_eq!(pop_sig.input(), &just_list_row);
        assert_eq!(pop_sig.output(), &both_row);

        let push_sig = get_op(&PUSH_NAME)
            .compute_signature(&[TypeArg::Type { ty: FLOAT64_TYPE }], &reg)
            .unwrap();

        let list_type = Type::new_extension(CustomType::new(
            LIST_TYPENAME,
            vec![TypeArg::Type { ty: FLOAT64_TYPE }],
            EXTENSION_NAME,
            TypeBound::Copyable,
        ));
        let both_row: TypeRow = vec![list_type.clone(), FLOAT64_TYPE].into();
        let just_list_row: TypeRow = vec![list_type].into();

        assert_eq!(push_sig.input(), &both_row);
        assert_eq!(push_sig.output(), &just_list_row);
    }
}
