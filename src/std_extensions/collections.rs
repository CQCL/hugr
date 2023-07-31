//! Basic logical operations.

use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::{
    extension::{ExtensionSet, SignatureError, TypeDef, TypeDefBound},
    types::{
        type_param::{TypeArg, TypeParam},
        CustomCheckFailure, CustomType, Type, TypeBound, TypeRow,
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
pub const RESOURCE_NAME: SmolStr = SmolStr::new_inline("Collections");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Dynamically sized list of values, all of the same type.
pub struct ListValue(Vec<Value>);

#[typetag::serde]
impl CustomConst for ListValue {
    fn name(&self) -> SmolStr {
        if self.0.is_empty() {
            "list".into()
        } else {
            format!("list:<{}>", self.0[0].name()).into()
        }
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
        let [TypeArg::Type(t)] =  typ.args() else {
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

fn extension() -> Extension {
    let mut extension = Extension::new(RESOURCE_NAME);

    extension
        .add_type(
            LIST_TYPENAME,
            vec![TypeParam::Type(TypeBound::Any)],
            "Generic dynamically sized list of type T.".into(),
            TypeDefBound::FromParams(vec![0]),
        )
        .unwrap();
    extension
        .add_op_custom_sig(
            POP_NAME,
            "Pop from back of list".into(),
            vec![TypeParam::Type(TypeBound::Any)],
            Default::default(),
            vec![],
            move |args: &[TypeArg]| {
                let (list_type, element_type) = list_types(args)?;

                let inputs = TypeRow::from(vec![list_type.clone()]);
                let outputs = TypeRow::from(vec![list_type, element_type]);
                let resource_set = ExtensionSet::singleton(&RESOURCE_NAME);
                Ok((inputs, outputs, resource_set))
            },
        )
        .unwrap();
    extension
        .add_op_custom_sig(
            PUSH_NAME,
            "Push to back of list".into(),
            vec![TypeParam::Type(TypeBound::Any)],
            Default::default(),
            vec![],
            move |args: &[TypeArg]| {
                let (list_type, element_type) = list_types(args)?;

                let outputs = TypeRow::from(vec![list_type.clone()]);
                let inputs = TypeRow::from(vec![list_type, element_type]);
                let resource_set = ExtensionSet::singleton(&RESOURCE_NAME);
                Ok((inputs, outputs, resource_set))
            },
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

fn list_types(args: &[TypeArg]) -> Result<(Type, Type), SignatureError> {
    let list_custom_type = get_type(&LIST_TYPENAME).instantiate_concrete(args)?;
    let [TypeArg::Type(element_type)] = args else {panic!("should be checked by def.")};

    let list_type: Type = Type::new_extension(list_custom_type);
    Ok((list_type, element_type.clone()))
}

#[cfg(test)]
mod test {
    use crate::{
        extension::{
            prelude::{ConstUsize, QB_T, USIZE_T},
            OpDef,
        },
        std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE},
        types::{type_param::TypeArg, Type},
        Extension,
    };

    use super::*;
    fn get_op(name: &str) -> &OpDef {
        EXTENSION.get_op(name).unwrap()
    }
    #[test]
    fn test_resource() {
        let r: Extension = extension();
        assert_eq!(r.name(), &RESOURCE_NAME);
        let ops = r.operations();
        assert_eq!(ops.count(), 2);
    }

    #[test]
    fn test_list() {
        let r: Extension = extension();
        let list_def = r.get_type(&LIST_TYPENAME).unwrap();

        let list_type = list_def
            .instantiate_concrete([TypeArg::Type(USIZE_T)])
            .unwrap();

        assert!(list_def.instantiate_concrete([TypeArg::USize(3)]).is_err());

        list_def.check_custom(&list_type).unwrap();
        let list_value = ListValue(vec![ConstUsize::new(3).into()]);

        list_value.check_custom_type(&list_type).unwrap();

        let wrong_list_value = ListValue(vec![ConstF64::new(1.2).into()]);
        assert!(wrong_list_value.check_custom_type(&list_type).is_err());
    }

    #[test]
    fn test_list_ops() {
        let pop_sig = get_op(&POP_NAME)
            .compute_signature(&[TypeArg::Type(QB_T)])
            .unwrap();

        let list_type = Type::new_extension(CustomType::new(
            LIST_TYPENAME,
            vec![TypeArg::Type(QB_T)],
            RESOURCE_NAME,
            TypeBound::Any,
        ));

        let both_row: TypeRow = vec![list_type.clone(), QB_T].into();
        let just_list_row: TypeRow = vec![list_type].into();
        assert_eq!(pop_sig.input(), &just_list_row);
        assert_eq!(pop_sig.output(), &both_row);

        let push_sig = get_op(&PUSH_NAME)
            .compute_signature(&[TypeArg::Type(FLOAT64_TYPE)])
            .unwrap();

        let list_type = Type::new_extension(CustomType::new(
            LIST_TYPENAME,
            vec![TypeArg::Type(FLOAT64_TYPE)],
            RESOURCE_NAME,
            TypeBound::Copyable,
        ));
        let both_row: TypeRow = vec![list_type.clone(), FLOAT64_TYPE].into();
        let just_list_row: TypeRow = vec![list_type].into();

        assert_eq!(push_sig.input(), &both_row);
        assert_eq!(push_sig.output(), &just_list_row);
    }
}
