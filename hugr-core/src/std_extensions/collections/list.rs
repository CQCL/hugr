//! List type and operations.

mod list_fold;

use std::hash::{Hash, Hasher};

use std::str::FromStr;
use std::sync::{Arc, LazyLock, Weak};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use strum::{EnumIter, EnumString, IntoStaticStr};

use crate::extension::prelude::{either_type, option_type, usize_t};
use crate::extension::resolution::{
    ExtensionResolutionError, WeakExtensionRegistry, resolve_type_extensions,
    resolve_value_extensions,
};
use crate::extension::simple_op::{MakeOpDef, MakeRegisteredOp};
use crate::extension::{ExtensionBuildError, OpDef, SignatureFunc};
use crate::ops::constant::{TryHash, ValueName, maybe_hash_values};
use crate::ops::{OpName, Value};
use crate::types::{Term, TypeName, TypeRowRV};
use crate::{
    Extension,
    extension::{
        ExtensionId, SignatureError, TypeDef, TypeDefBound,
        simple_op::{MakeExtensionOp, OpLoadError},
    },
    ops::constant::CustomConst,
    ops::custom::ExtensionOp,
    types::{
        CustomCheckFailure, CustomType, FuncValueType, PolyFuncTypeRV, Type, TypeBound,
        type_param::{TypeArg, TypeParam},
    },
};

/// Reported unique name of the list type.
pub const LIST_TYPENAME: TypeName = TypeName::new_inline("List");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("collections.list");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Dynamically sized list of values, all of the same type.
pub struct ListValue(Vec<Value>, Type);

impl ListValue {
    /// Create a new [`CustomConst`] for a list of values of type `typ`.
    /// That all values are of type `typ` is not checked here.
    pub fn new(typ: Type, contents: impl IntoIterator<Item = Value>) -> Self {
        Self(contents.into_iter().collect_vec(), typ)
    }

    /// Create a new [`CustomConst`] for an empty list of values of type `typ`.
    #[must_use]
    pub fn new_empty(typ: Type) -> Self {
        Self(vec![], typ)
    }

    /// Returns the type of the `[ListValue]` as a `[CustomType]`.`
    #[must_use]
    pub fn custom_type(&self) -> CustomType {
        list_custom_type(self.1.clone())
    }

    /// Returns the type of values inside the `[ListValue]`.
    #[must_use]
    pub fn get_element_type(&self) -> &Type {
        &self.1
    }

    /// Returns the values contained inside the `[ListValue]`.
    #[must_use]
    pub fn get_contents(&self) -> &[Value] {
        &self.0
    }
}

impl TryHash for ListValue {
    fn try_hash(&self, mut st: &mut dyn Hasher) -> bool {
        maybe_hash_values(&self.0, &mut st) && {
            self.1.hash(&mut st);
            true
        }
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
        let [TypeArg::Runtime(ty)] = typ.args() else {
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

    fn update_extensions(
        &mut self,
        extensions: &WeakExtensionRegistry,
    ) -> Result<(), ExtensionResolutionError> {
        for val in &mut self.0 {
            resolve_value_extensions(val, extensions)?;
        }
        resolve_type_extensions(&mut self.1, extensions)
    }
}

/// A list operation
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(non_camel_case_types)]
#[non_exhaustive]
pub enum ListOp {
    /// Pop from the end of list. Return an optional value.
    pop,
    /// Push to end of list. Return the new list.
    push,
    /// Lookup an element in a list by index.
    get,
    /// Replace the element at index `i` with value `v`, and return the old value.
    ///
    /// If the index is out of bounds, returns the input value as an error.
    set,
    /// Insert an element at index `i`.
    ///
    /// Elements at higher indices are shifted one position to the right.
    /// Returns an Err with the element if the index is out of bounds.
    insert,
    /// Get the length of a list.
    length,
}

impl ListOp {
    /// Type parameter used in the list types.
    const TP: TypeParam = TypeParam::RuntimeType(TypeBound::Linear);

    /// Instantiate a list operation with an `element_type`.
    #[must_use]
    pub fn with_type(self, element_type: Type) -> ListOpInst {
        ListOpInst {
            elem_type: element_type,
            op: self,
        }
    }

    /// Compute the signature of the operation, given the list type definition.
    fn compute_signature(self, list_type_def: &TypeDef) -> SignatureFunc {
        use ListOp::{get, insert, length, pop, push, set};
        let e = Type::new_var_use(0, TypeBound::Linear);
        let l = self.list_type(list_type_def, 0);
        match self {
            pop => self
                .list_polytype(vec![l.clone()], vec![l, Type::from(option_type(e))])
                .into(),
            push => self.list_polytype(vec![l.clone(), e], vec![l]).into(),
            get => self
                .list_polytype(vec![l, usize_t()], vec![Type::from(option_type(e))])
                .into(),
            set => self
                .list_polytype(
                    vec![l.clone(), usize_t(), e.clone()],
                    vec![l, Type::from(either_type(e.clone(), e))],
                )
                .into(),
            insert => self
                .list_polytype(
                    vec![l.clone(), usize_t(), e.clone()],
                    vec![l, either_type(e, Type::UNIT).into()],
                )
                .into(),
            length => self
                .list_polytype(vec![l.clone()], vec![l, usize_t()])
                .into(),
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
    fn opdef_id(&self) -> OpName {
        <&Self as Into<&'static str>>::into(self).into()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, crate::extension::simple_op::OpLoadError> {
        crate::extension::simple_op::try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    /// Add an operation implemented as an [`MakeOpDef`], which can provide the data
    /// required to define an [`OpDef`], to an extension.
    //
    // This method is re-defined here since we need to pass the list type def while computing the signature,
    // to avoid recursive loops initializing the extension.
    fn add_to_extension(
        &self,
        extension: &mut Extension,
        extension_ref: &Weak<Extension>,
    ) -> Result<(), ExtensionBuildError> {
        let sig = self.compute_signature(extension.get_type(&LIST_TYPENAME).unwrap());
        let def = extension.add_op(self.opdef_id(), self.description(), sig, extension_ref)?;

        self.post_opdef(def);

        Ok(())
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        self.compute_signature(list_type_def())
    }

    fn description(&self) -> String {
        use ListOp::*;

        match self {
            pop => "Pop from the back of list. Returns an optional value.",
            push => "Push to the back of list",
            get => "Lookup an element in a list by index. Panics if the index is out of bounds.",
            set => "Replace the element at index `i` with value `v`.",
            insert => "Insert an element at index `i`. Elements at higher indices are shifted one position to the right. Panics if the index is out of bounds.",
            length => "Get the length of a list",
        }
        .into()
    }

    fn post_opdef(&self, def: &mut OpDef) {
        list_fold::set_fold(self, def);
    }
}

/// Extension for list operations.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(|| {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        extension
            .add_type(
                LIST_TYPENAME,
                vec![ListOp::TP],
                "Generic dynamically sized list of type T.".into(),
                TypeDefBound::from_params(vec![0]),
                extension_ref,
            )
            .unwrap();

        // The list type must be defined before the operations are added.
        ListOp::load_all_ops(extension, extension_ref).unwrap();
    })
});

impl MakeRegisteredOp for ListOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

/// Get the type of a list of `elem_type` as a `CustomType`.
#[must_use]
pub fn list_type_def() -> &'static TypeDef {
    // This must not be called while the extension is being built.
    EXTENSION.get_type(&LIST_TYPENAME).unwrap()
}

/// Get the type of a list of `elem_type` as a `CustomType`.
#[must_use]
pub fn list_custom_type(elem_type: Type) -> CustomType {
    list_type_def().instantiate(vec![elem_type.into()]).unwrap()
}

/// Get the `Type` of a list of `elem_type`.
#[must_use]
pub fn list_type(elem_type: Type) -> Type {
    list_custom_type(elem_type).into()
}

/// A list operation with a concrete element type.
///
/// See [`ListOp`] for the parametric version.
#[derive(Debug, Clone, PartialEq)]
pub struct ListOpInst {
    op: ListOp,
    elem_type: Type,
}

impl MakeExtensionOp for ListOpInst {
    fn op_id(&self) -> OpName {
        self.op.opdef_id()
    }

    fn from_extension_op(
        ext_op: &ExtensionOp,
    ) -> Result<Self, crate::extension::simple_op::OpLoadError> {
        let [Term::Runtime(ty)] = ext_op.args() else {
            return Err(SignatureError::InvalidTypeArgs.into());
        };
        let name = ext_op.unqualified_id();
        let Ok(op) = ListOp::from_str(name) else {
            return Err(OpLoadError::NotMember(name.to_string()));
        };

        Ok(Self {
            elem_type: ty.clone(),
            op,
        })
    }

    fn type_args(&self) -> Vec<Term> {
        vec![self.elem_type.clone().into()]
    }
}

impl ListOpInst {
    /// Convert this list operation to an [`ExtensionOp`] by providing a
    /// registry to validate the element type against.
    #[must_use]
    pub fn to_extension_op(self) -> Option<ExtensionOp> {
        ExtensionOp::new(EXTENSION.get_op(&self.op_id())?.clone(), self.type_args()).ok()
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::PortIndex;
    use crate::extension::prelude::{
        const_fail_tuple, const_none, const_ok_tuple, const_some_tuple,
    };
    use crate::ops::OpTrait;
    use crate::{
        extension::prelude::{ConstUsize, qb_t, usize_t},
        std_extensions::arithmetic::float_types::{ConstF64, float64_type},
        types::TypeRow,
    };

    use super::*;

    #[test]
    fn test_extension() {
        assert_eq!(&ListOp::push.extension_id(), EXTENSION.name());
        assert_eq!(&ListOp::push.extension(), EXTENSION.name());
        for (_, op_def) in EXTENSION.operations() {
            assert_eq!(op_def.extension_id(), &EXTENSION_ID);
        }
    }

    #[test]
    fn test_list() {
        let list_def = list_type_def();

        let list_type = list_def.instantiate([usize_t().into()]).unwrap();

        assert!(list_def.instantiate([3u64.into()]).is_err());

        list_def.check_custom(&list_type).unwrap();
        let list_value = ListValue(vec![ConstUsize::new(3).into()], usize_t());

        list_value.validate().unwrap();

        let wrong_list_value = ListValue(vec![ConstF64::new(1.2).into()], usize_t());
        assert!(wrong_list_value.validate().is_err());
    }

    #[test]
    fn test_list_ops() {
        let pop_op = ListOp::pop.with_type(qb_t());
        let pop_ext = pop_op.clone().to_extension_op().unwrap();
        assert_eq!(ListOpInst::from_extension_op(&pop_ext).unwrap(), pop_op);
        let pop_sig = pop_ext.dataflow_signature().unwrap();

        let list_t = list_type(qb_t());

        let both_row: TypeRow = vec![list_t.clone(), option_type(qb_t()).into()].into();
        let just_list_row: TypeRow = vec![list_t].into();
        assert_eq!(pop_sig.input(), &just_list_row);
        assert_eq!(pop_sig.output(), &both_row);

        let push_op = ListOp::push.with_type(float64_type());
        let push_ext = push_op.clone().to_extension_op().unwrap();
        assert_eq!(ListOpInst::from_extension_op(&push_ext).unwrap(), push_op);
        let push_sig = push_ext.dataflow_signature().unwrap();

        let list_t = list_type(float64_type());

        let both_row: TypeRow = vec![list_t.clone(), float64_type()].into();
        let just_list_row: TypeRow = vec![list_t].into();

        assert_eq!(push_sig.input(), &both_row);
        assert_eq!(push_sig.output(), &just_list_row);
    }

    /// Values used in the `list_fold` test cases.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum TestVal {
        Idx(usize),
        List(Vec<usize>),
        Elem(usize),
        Some(Vec<TestVal>),
        None(TypeRow),
        Ok(Vec<TestVal>, TypeRow),
        Err(TypeRow, Vec<TestVal>),
    }

    impl TestVal {
        fn to_value(&self) -> Value {
            match self {
                TestVal::Idx(i) => Value::extension(ConstUsize::new(*i as u64)),
                TestVal::Elem(e) => Value::extension(ConstUsize::new(*e as u64)),
                TestVal::List(l) => {
                    let elems = l
                        .iter()
                        .map(|&i| Value::extension(ConstUsize::new(i as u64)))
                        .collect();
                    Value::extension(ListValue(elems, usize_t()))
                }
                TestVal::Some(l) => {
                    let elems = l.iter().map(TestVal::to_value);
                    const_some_tuple(elems)
                }
                TestVal::None(tr) => const_none(tr.clone()),
                TestVal::Ok(l, tr) => {
                    let elems = l.iter().map(TestVal::to_value);
                    const_ok_tuple(elems, tr.clone())
                }
                TestVal::Err(tr, l) => {
                    let elems = l.iter().map(TestVal::to_value);
                    const_fail_tuple(elems, tr.clone())
                }
            }
        }
    }

    #[rstest]
    #[case::pop(ListOp::pop, &[TestVal::List(vec![77,88, 42])], &[TestVal::List(vec![77,88]), TestVal::Some(vec![TestVal::Elem(42)])])]
    #[case::pop_empty(ListOp::pop, &[TestVal::List(vec![])], &[TestVal::List(vec![]), TestVal::None(vec![usize_t()].into())])]
    #[case::push(ListOp::push, &[TestVal::List(vec![77,88]), TestVal::Elem(42)], &[TestVal::List(vec![77,88,42])])]
    #[case::set(ListOp::set, &[TestVal::List(vec![77,88,42]), TestVal::Idx(1), TestVal::Elem(99)], &[TestVal::List(vec![77,99,42]), TestVal::Ok(vec![TestVal::Elem(88)], vec![usize_t()].into())])]
    #[case::set_invalid(ListOp::set, &[TestVal::List(vec![77,88,42]), TestVal::Idx(123), TestVal::Elem(99)], &[TestVal::List(vec![77,88,42]), TestVal::Err(vec![usize_t()].into(), vec![TestVal::Elem(99)])])]
    #[case::get(ListOp::get, &[TestVal::List(vec![77,88,42]), TestVal::Idx(1)], &[TestVal::Some(vec![TestVal::Elem(88)])])]
    #[case::get_invalid(ListOp::get, &[TestVal::List(vec![77,88,42]), TestVal::Idx(99)], &[TestVal::None(vec![usize_t()].into())])]
    #[case::insert(ListOp::insert, &[TestVal::List(vec![77,88,42]), TestVal::Idx(1), TestVal::Elem(99)], &[TestVal::List(vec![77,99,88,42]), TestVal::Ok(vec![], vec![usize_t()].into())])]
    #[case::insert_invalid(ListOp::insert, &[TestVal::List(vec![77,88,42]), TestVal::Idx(52), TestVal::Elem(99)], &[TestVal::List(vec![77,88,42]), TestVal::Err(Type::UNIT.into(), vec![TestVal::Elem(99)])])]
    #[case::length(ListOp::length, &[TestVal::List(vec![77,88,42])], &[TestVal::Elem(3)])]
    fn list_fold(#[case] op: ListOp, #[case] inputs: &[TestVal], #[case] outputs: &[TestVal]) {
        let consts: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| (i.into(), x.to_value()))
            .collect();

        let res = op
            .with_type(usize_t())
            .to_extension_op()
            .unwrap()
            .constant_fold(&consts)
            .unwrap();

        for (i, expected) in outputs.iter().enumerate() {
            let expected = expected.to_value();
            let res_val = res
                .iter()
                .find(|(port, _)| port.index() == i)
                .unwrap()
                .1
                .clone();

            assert_eq!(res_val, expected);
        }
    }
}
