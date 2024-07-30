//! List type and operations.

use itertools::Itertools;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

use crate::ops::constant::ValueName;
use crate::ops::{OpName, Value};
use crate::types::TypeName;
use crate::{
    extension::{
        simple_op::{MakeExtensionOp, OpLoadError},
        ConstFold, ExtensionId, ExtensionRegistry, ExtensionSet, SignatureError, TypeDef,
        TypeDefBound,
    },
    ops::constant::CustomConst,
    ops::{self, custom::ExtensionOp, NamedOp},
    types::{
        type_param::{TypeArg, TypeParam},
        CustomCheckFailure, CustomType, FuncValueType, PolyFuncTypeRV, Type, TypeBound,
    },
    utils::sorted_consts,
    Extension,
};

/// Reported unique name of the list type.
pub const LIST_TYPENAME: TypeName = TypeName::new_inline("List");
/// Pop operation name.
pub const POP_NAME: OpName = OpName::new_inline("pop");
/// Push operation name.
pub const PUSH_NAME: OpName = OpName::new_inline("push");
/// Reported unique name of the extension
pub const EXTENSION_NAME: ExtensionId = ExtensionId::new_unchecked("Collections");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Dynamically sized list of values, all of the same type.
pub struct ListValue(Vec<Value>, Type);

impl ListValue {
    /// Create a new [CustomConst] for a list of values of type `typ`.
    /// That all values ore of type `typ` is not checked here.
    pub fn new(typ: Type, contents: impl IntoIterator<Item = Value>) -> Self {
        Self(contents.into_iter().collect_vec(), typ)
    }

    /// Create a new [CustomConst] for an empty list of values of type `typ`.
    pub fn new_empty(typ: Type) -> Self {
        Self(vec![], typ)
    }

    /// Returns the type of the `[ListValue]` as a `[CustomType]`.`
    pub fn custom_type(&self) -> CustomType {
        list_custom_type(self.1.clone())
    }
}

#[typetag::serde]
impl CustomConst for ListValue {
    fn name(&self) -> ValueName {
        ValueName::new_inline("list")
    }

    fn get_type(&self) -> Type {
        self.custom_type().into()
    }

    fn validate(&self) -> Result<(), CustomCheckFailure> {
        let typ = self.custom_type();
        let error = || {
            // TODO more bespoke errors
            CustomCheckFailure::Message("List type check fail.".to_string())
        };

        EXTENSION
            .get_type(&LIST_TYPENAME)
            .unwrap()
            .check_custom(&typ)
            .map_err(|_| error())?;

        // constant can only hold classic type.
        let [TypeArg::Type { ty }] = typ.args() else {
            return Err(error());
        };

        // check all values are instances of the element type
        for v in &self.0 {
            if v.get_type() != *ty {
                return Err(error());
            }
        }

        Ok(())
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::union_over(self.0.iter().map(Value::extension_reqs))
            .union(EXTENSION_NAME.into())
    }
}

struct PopFold;

impl ConstFold for PopFold {
    fn fold(
        &self,
        _type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, ops::Value)],
    ) -> crate::extension::ConstFoldResult {
        let [list]: [&ops::Value; 1] = sorted_consts(consts).try_into().ok()?;
        let list: &ListValue = list.get_custom_value().expect("Should be list value.");
        let mut list = list.clone();
        let elem = list.0.pop()?; // empty list fails to evaluate "pop"

        Some(vec![(0.into(), list.into()), (1.into(), elem)])
    }
}

struct PushFold;

impl ConstFold for PushFold {
    fn fold(
        &self,
        _type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, ops::Value)],
    ) -> crate::extension::ConstFoldResult {
        let [list, elem]: [&ops::Value; 2] = sorted_consts(consts).try_into().ok()?;
        let list: &ListValue = list.get_custom_value().expect("Should be list value.");
        let mut list = list.clone();
        list.0.push(elem.clone());

        Some(vec![(0.into(), list.into())])
    }
}
const TP: TypeParam = TypeParam::Type { b: TypeBound::Any };

fn extension() -> Extension {
    let mut extension = Extension::new(EXTENSION_NAME, VERSION);

    extension
        .add_type(
            LIST_TYPENAME,
            vec![TP],
            "Generic dynamically sized list of type T.".into(),
            TypeDefBound::from_params(vec![0]),
        )
        .unwrap();
    let list_type_def = extension.get_type(&LIST_TYPENAME).unwrap();

    let (l, e) = list_and_elem_type_vars(list_type_def);
    extension
        .add_op(
            POP_NAME,
            "Pop from back of list".into(),
            PolyFuncTypeRV::new(
                vec![TP],
                FuncValueType::new(vec![l.clone()], vec![l.clone(), e.clone()]),
            ),
        )
        .unwrap()
        .set_constant_folder(PopFold);
    extension
        .add_op(
            PUSH_NAME,
            "Push to back of list".into(),
            PolyFuncTypeRV::new(vec![TP], FuncValueType::new(vec![l.clone(), e], vec![l])),
        )
        .unwrap()
        .set_constant_folder(PushFold);

    extension
}

lazy_static! {
    /// Collections extension definition.
    pub static ref EXTENSION: Extension = extension();
}

/// Get the type of a list of `elem_type` as a `CustomType`.
pub fn list_custom_type(elem_type: Type) -> CustomType {
    EXTENSION
        .get_type(&LIST_TYPENAME)
        .unwrap()
        .instantiate(vec![TypeArg::Type { ty: elem_type }])
        .unwrap()
}

/// Get the `Type` of a list of `elem_type`.
pub fn list_type(elem_type: Type) -> Type {
    list_custom_type(elem_type).into()
}

fn list_and_elem_type_vars(list_type_def: &TypeDef) -> (Type, Type) {
    let elem_type = Type::new_var_use(0, TypeBound::Any);
    let list_type = Type::new_extension(
        list_type_def
            .instantiate(vec![TypeArg::new_var_use(0, TP)])
            .unwrap(),
    );
    (list_type, elem_type)
}

/// A list operation
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ListOp {
    /// Pop from end of list
    Pop,
    /// Push to end of list
    Push,
}

impl ListOp {
    /// Instantiate a list operation with an `element_type`
    pub fn with_type(self, element_type: Type) -> ListOpInst {
        ListOpInst {
            elem_type: element_type,
            op: self,
        }
    }
}

/// A list operation with a concrete element type.
#[derive(Debug, Clone, PartialEq)]
pub struct ListOpInst {
    op: ListOp,
    elem_type: Type,
}

impl NamedOp for ListOpInst {
    fn name(&self) -> OpName {
        match self.op {
            ListOp::Pop => POP_NAME,
            ListOp::Push => PUSH_NAME,
        }
    }
}

impl MakeExtensionOp for ListOpInst {
    fn from_extension_op(
        ext_op: &ExtensionOp,
    ) -> Result<Self, crate::extension::simple_op::OpLoadError> {
        let [TypeArg::Type { ty }] = ext_op.args() else {
            return Err(SignatureError::InvalidTypeArgs.into());
        };
        let name = ext_op.def().name();
        let op = match name {
            // can't use const SmolStr in pattern
            _ if name == &POP_NAME => ListOp::Pop,
            _ if name == &PUSH_NAME => ListOp::Push,
            _ => return Err(OpLoadError::NotMember(name.to_string())),
        };

        Ok(Self {
            elem_type: ty.clone(),
            op,
        })
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![TypeArg::Type {
            ty: self.elem_type.clone(),
        }]
    }
}

impl ListOpInst {
    /// Convert this list operation to an [`ExtensionOp`] by providing a
    /// registry to validate the element type against.
    pub fn to_extension_op(self, elem_type_registry: &ExtensionRegistry) -> Option<ExtensionOp> {
        let registry = ExtensionRegistry::try_new(
            elem_type_registry
                .clone()
                .into_iter()
                // ignore self if already in registry
                .filter_map(|(_, ext)| (ext.name() != EXTENSION.name()).then_some(ext))
                .chain(std::iter::once(EXTENSION.to_owned())),
        )
        .unwrap();
        ExtensionOp::new(
            registry.get(&EXTENSION_NAME)?.get_op(&self.name())?.clone(),
            self.type_args(),
            &registry,
        )
        .ok()
    }
}

#[cfg(test)]
mod test {
    use crate::ops::OpTrait;
    use crate::{
        extension::{
            prelude::{ConstUsize, QB_T, USIZE_T},
            PRELUDE,
        },
        std_extensions::arithmetic::float_types::{self, ConstF64, FLOAT64_TYPE},
        types::TypeRow,
    };

    use super::*;

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
        let list_value = ListValue(vec![ConstUsize::new(3).into()], USIZE_T);

        list_value.validate().unwrap();

        let wrong_list_value = ListValue(vec![ConstF64::new(1.2).into()], USIZE_T);
        assert!(wrong_list_value.validate().is_err());
    }

    #[test]
    fn test_list_ops() {
        let reg =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), float_types::EXTENSION.to_owned()])
                .unwrap();
        let pop_op = ListOp::Pop.with_type(QB_T);
        let pop_ext = pop_op.clone().to_extension_op(&reg).unwrap();
        assert_eq!(ListOpInst::from_extension_op(&pop_ext).unwrap(), pop_op);
        let pop_sig = pop_ext.dataflow_signature().unwrap();

        let list_t = list_type(QB_T);

        let both_row: TypeRow = vec![list_t.clone(), QB_T].into();
        let just_list_row: TypeRow = vec![list_t].into();
        assert_eq!(pop_sig.input(), &just_list_row);
        assert_eq!(pop_sig.output(), &both_row);

        let push_op = ListOp::Push.with_type(FLOAT64_TYPE);
        let push_ext = push_op.clone().to_extension_op(&reg).unwrap();
        assert_eq!(ListOpInst::from_extension_op(&push_ext).unwrap(), push_op);
        let push_sig = push_ext.dataflow_signature().unwrap();

        let list_t = list_type(FLOAT64_TYPE);

        let both_row: TypeRow = vec![list_t.clone(), FLOAT64_TYPE].into();
        let just_list_row: TypeRow = vec![list_t].into();

        assert_eq!(push_sig.input(), &both_row);
        assert_eq!(push_sig.output(), &just_list_row);
    }
}
