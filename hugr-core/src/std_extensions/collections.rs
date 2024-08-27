//! List type and operations.

mod list_fold;

use std::str::FromStr;

use itertools::Itertools;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use crate::extension::simple_op::{MakeOpDef, MakeRegisteredOp};
use crate::extension::{ExtensionBuildError, OpDef, SignatureFunc, PRELUDE};
use crate::ops::constant::ValueName;
use crate::ops::{OpName, Value};
use crate::types::{TypeName, TypeRowRV};
use crate::{
    extension::{
        simple_op::{MakeExtensionOp, OpLoadError},
        ExtensionId, ExtensionRegistry, ExtensionSet, SignatureError, TypeDef, TypeDefBound,
    },
    ops::constant::CustomConst,
    ops::{custom::ExtensionOp, NamedOp},
    types::{
        type_param::{TypeArg, TypeParam},
        CustomCheckFailure, CustomType, FuncValueType, PolyFuncTypeRV, Type, TypeBound,
    },
    Extension,
};

/// Reported unique name of the list type.
pub const LIST_TYPENAME: TypeName = TypeName::new_inline("List");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("collections");
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
            .union(EXTENSION_ID.into())
    }
}

/// A list operation
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(non_camel_case_types)]
#[non_exhaustive]
pub enum ListOp {
    /// Pop from end of list
    pop,
    /// Push to end of list
    push,
}

impl ListOp {
    /// Type parameter used in the list types.
    const TP: TypeParam = TypeParam::Type { b: TypeBound::Any };

    /// Instantiate a list operation with an `element_type`
    pub fn with_type(self, element_type: Type) -> ListOpInst {
        ListOpInst {
            elem_type: element_type,
            op: self,
        }
    }

    /// Compute the signature of the operation, given the list type definition.
    fn compute_signature(self, list_type_def: &TypeDef) -> SignatureFunc {
        use ListOp::*;
        let e = Type::new_var_use(0, TypeBound::Any);
        let l = self.list_type(list_type_def, 0);
        match self {
            pop => self
                .list_polytype(vec![l.clone()], vec![l.clone(), e.clone()])
                .into(),
            push => self.list_polytype(vec![l.clone(), e], vec![l]).into(),
        }
    }

    /// Compute a polymorphic function type for a list operation.
    fn list_polytype(
        self,
        input: impl Into<TypeRowRV>,
        output: impl Into<TypeRowRV>,
    ) -> PolyFuncTypeRV {
        PolyFuncTypeRV::new(vec![Self::TP], FuncValueType::new(input, output))
    }

    /// Returns the type of a generic list, associated with the element type parameter at index `idx`.
    fn list_type(self, list_type_def: &TypeDef, idx: usize) -> Type {
        Type::new_extension(
            list_type_def
                .instantiate(vec![TypeArg::new_var_use(idx, Self::TP)])
                .unwrap(),
        )
    }
}

impl MakeOpDef for ListOp {
    fn from_def(op_def: &OpDef) -> Result<Self, crate::extension::simple_op::OpLoadError> {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    /// Add an operation implemented as an [MakeOpDef], which can provide the data
    /// required to define an [OpDef], to an extension.
    //
    // This method is re-defined here since we need to pass the list type def while computing the signature,
    // to avoid recursive loops initializing the extension.
    fn add_to_extension(&self, extension: &mut Extension) -> Result<(), ExtensionBuildError> {
        let sig = self.compute_signature(extension.get_type(&LIST_TYPENAME).unwrap());
        let def = extension.add_op(self.name(), self.description(), sig)?;

        self.post_opdef(def);

        Ok(())
    }

    fn signature(&self) -> SignatureFunc {
        self.compute_signature(list_type_def())
    }

    fn description(&self) -> String {
        use ListOp::*;

        match self {
            pop => "Pop from back of list",
            push => "Push to back of list",
        }
        .into()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        list_fold::set_fold(self, def)
    }
}

lazy_static! {
    /// Extension for list operations.
    pub static ref EXTENSION: Extension = {
        println!("creating collections extension");
        let mut extension = Extension::new(EXTENSION_ID, VERSION);

        // The list type must be defined before the operations are added.
        extension.add_type(
            LIST_TYPENAME,
            vec![ListOp::TP],
            "Generic dynamically sized list of type T.".into(),
            TypeDefBound::from_params(vec![0]),
        )
        .unwrap();

        ListOp::load_all_ops(&mut extension).unwrap();

        extension
    };

    /// Registry of extensions required to validate list operations.
    pub static ref COLLECTIONS_REGISTRY: ExtensionRegistry  = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        EXTENSION.to_owned(),
    ])
    .unwrap();
}

impl MakeRegisteredOp for ListOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
        &COLLECTIONS_REGISTRY
    }
}

/// Get the type of a list of `elem_type` as a `CustomType`.
pub fn list_type_def() -> &'static TypeDef {
    // This must not be called while the extension is being built.
    EXTENSION.get_type(&LIST_TYPENAME).unwrap()
}

/// Get the type of a list of `elem_type` as a `CustomType`.
pub fn list_custom_type(elem_type: Type) -> CustomType {
    list_type_def()
        .instantiate(vec![TypeArg::Type { ty: elem_type }])
        .unwrap()
}

/// Get the `Type` of a list of `elem_type`.
pub fn list_type(elem_type: Type) -> Type {
    list_custom_type(elem_type).into()
}

/// A list operation with a concrete element type.
///
/// See [ListOp] for the parametric version.
#[derive(Debug, Clone, PartialEq)]
pub struct ListOpInst {
    op: ListOp,
    elem_type: Type,
}

impl NamedOp for ListOpInst {
    fn name(&self) -> OpName {
        let name: &str = self.op.into();
        name.into()
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
        let Ok(op) = ListOp::from_str(name) else {
            return Err(OpLoadError::NotMember(name.to_string()));
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
            registry.get(&EXTENSION_ID)?.get_op(&self.name())?.clone(),
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
        assert_eq!(&ListOp::push.extension_id(), EXTENSION.name());
        assert_eq!(&ListOp::push.extension(), EXTENSION.name());
        assert!(ListOp::pop.registry().contains(EXTENSION.name()));
        for (_, op_def) in EXTENSION.operations() {
            assert_eq!(op_def.extension(), &EXTENSION_ID);
        }
    }

    #[test]
    fn test_list() {
        let list_def = list_type_def();

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
        let pop_op = ListOp::pop.with_type(QB_T);
        let pop_ext = pop_op.clone().to_extension_op(&reg).unwrap();
        assert_eq!(ListOpInst::from_extension_op(&pop_ext).unwrap(), pop_op);
        let pop_sig = pop_ext.dataflow_signature().unwrap();

        let list_t = list_type(QB_T);

        let both_row: TypeRow = vec![list_t.clone(), QB_T].into();
        let just_list_row: TypeRow = vec![list_t].into();
        assert_eq!(pop_sig.input(), &just_list_row);
        assert_eq!(pop_sig.output(), &both_row);

        let push_op = ListOp::push.with_type(FLOAT64_TYPE);
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
