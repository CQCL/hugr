//! Prelude extension - available in all contexts, defining common types,
//! operations and constants.
use std::sync::{Arc, Weak};

use itertools::Itertools;
use lazy_static::lazy_static;

use crate::extension::const_fold::fold_out_row;
use crate::extension::simple_op::{
    try_from_name, MakeExtensionOp, MakeOpDef, MakeRegisteredOp, OpLoadError,
};
use crate::extension::{
    ConstFold, ExtensionId, ExtensionSet, OpDef, SignatureError, SignatureFunc, TypeDefBound,
};
use crate::ops::constant::{CustomCheckFailure, CustomConst, ValueName};
use crate::ops::OpName;
use crate::ops::{NamedOp, Value};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{
    CustomType, FuncValueType, PolyFuncType, PolyFuncTypeRV, Signature, SumType, Type, TypeBound,
    TypeName, TypeRV, TypeRow, TypeRowRV,
};
use crate::utils::sorted_consts;
use crate::{type_row, Extension};

use strum_macros::{EnumIter, EnumString, IntoStaticStr};

use super::ExtensionRegistry;

mod unwrap_builder;

pub use unwrap_builder::UnwrapBuilder;

/// Array type and operations.
pub mod array;
pub use array::{array_type, new_array_op, ArrayOp, ArrayOpDef, ARRAY_TYPE_NAME, NEW_ARRAY_OP_ID};

/// Name of prelude extension.
pub const PRELUDE_ID: ExtensionId = ExtensionId::new_unchecked("prelude");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 0);
lazy_static! {
    /// Prelude extension, containing common types and operations.
    pub static ref PRELUDE: Arc<Extension> = {
        Extension::new_arc(PRELUDE_ID, VERSION, |prelude, extension_ref| {

            // Construct the list and error types using the passed extension
            // reference.
            //
            // If we tried to use `string_type()` or `error_type()` directly it
            // would try to access the `PRELUDE` lazy static recursively,
            // causing a deadlock.
            let string_type: Type = string_custom_type(extension_ref).into();
            let error_type: CustomType = error_custom_type(extension_ref);

            prelude
                .add_type(
                    TypeName::new_inline("usize"),
                    vec![],
                    "usize".into(),
                    TypeDefBound::copyable(),
                    extension_ref,
                )
                .unwrap();
            prelude.add_type(
                    STRING_TYPE_NAME,
                    vec![],
                    "string".into(),
                    TypeDefBound::copyable(),
                    extension_ref,
                )
                .unwrap();
            prelude.add_op(
                    PRINT_OP_ID,
                    "Print the string to standard output".to_string(),
                    Signature::new(vec![string_type], type_row![]),
                    extension_ref,
                )
                .unwrap();
            prelude.add_type(
                    TypeName::new_inline(ARRAY_TYPE_NAME),
                    vec![ TypeParam::max_nat(), TypeBound::Any.into()],
                    "array".into(),
                    TypeDefBound::from_params(vec![1] ),
                    extension_ref,
                )
                .unwrap();
            prelude
                .add_type(
                    TypeName::new_inline("qubit"),
                    vec![],
                    "qubit".into(),
                    TypeDefBound::any(),
                    extension_ref,
                )
                .unwrap();
            prelude
                .add_type(
                    ERROR_TYPE_NAME,
                    vec![],
                    "Simple opaque error type.".into(),
                    TypeDefBound::copyable(),
                    extension_ref,
                )
                .unwrap();
            prelude
                .add_op(
                    PANIC_OP_ID,
                    "Panic with input error".to_string(),
                    PolyFuncTypeRV::new(
                        [TypeParam::new_list(TypeBound::Any), TypeParam::new_list(TypeBound::Any)],
                        FuncValueType::new(
                            vec![TypeRV::new_extension(error_type), TypeRV::new_row_var_use(0, TypeBound::Any)],
                            vec![TypeRV::new_row_var_use(1, TypeBound::Any)],
                        ),
                    ),
                    extension_ref,
                )
                .unwrap();

            TupleOpDef::load_all_ops(prelude, extension_ref).unwrap();
            NoopDef.add_to_extension(prelude, extension_ref).unwrap();
            LiftDef.add_to_extension(prelude, extension_ref).unwrap();
            array::ArrayOpDef::load_all_ops(prelude, extension_ref).unwrap();
            array::ArrayScanDef.add_to_extension(prelude, extension_ref).unwrap();
        })
    };

    /// An extension registry containing only the prelude
    pub static ref PRELUDE_REGISTRY: ExtensionRegistry = ExtensionRegistry::new([PRELUDE.clone()]);
}

pub(crate) fn usize_custom_t(extension_ref: &Weak<Extension>) -> CustomType {
    CustomType::new(
        TypeName::new_inline("usize"),
        vec![],
        PRELUDE_ID,
        TypeBound::Copyable,
        extension_ref,
    )
}

pub(crate) fn qb_custom_t(extension_ref: &Weak<Extension>) -> CustomType {
    CustomType::new(
        TypeName::new_inline("qubit"),
        vec![],
        PRELUDE_ID,
        TypeBound::Any,
        extension_ref,
    )
}

/// Qubit type.
pub fn qb_t() -> Type {
    qb_custom_t(&Arc::downgrade(&PRELUDE)).into()
}
/// Unsigned size type.
pub fn usize_t() -> Type {
    usize_custom_t(&Arc::downgrade(&PRELUDE)).into()
}
/// Boolean type - Sum of two units.
pub fn bool_t() -> Type {
    Type::new_unit_sum(2)
}

/// Name of the prelude panic operation.
///
/// This operation can have any input and any output wires; it is instantiated
/// with two [TypeArg::Sequence]s representing these. The first input to the
/// operation is always an error type; the remaining inputs correspond to the
/// first sequence of types in its instantiation; the outputs correspond to the
/// second sequence of types in its instantiation. Note that the inputs and
/// outputs only exist so that structural constraints such as linearity can be
/// satisfied.
pub const PANIC_OP_ID: OpName = OpName::new_inline("panic");

/// Name of the string type.
pub const STRING_TYPE_NAME: TypeName = TypeName::new_inline("string");

/// Custom type for strings.
///
/// Receives a reference to the prelude extensions as a parameter.
/// This avoids deadlocks when we are in the process of creating the prelude.
fn string_custom_type(extension_ref: &Weak<Extension>) -> CustomType {
    CustomType::new(
        STRING_TYPE_NAME,
        vec![],
        PRELUDE_ID,
        TypeBound::Copyable,
        extension_ref,
    )
}

/// String type.
pub fn string_type() -> Type {
    string_custom_type(&Arc::downgrade(&PRELUDE)).into()
}

#[derive(Debug, Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
/// Structure for holding constant string values.
pub struct ConstString(String);

impl ConstString {
    /// Creates a new [`ConstString`].
    pub fn new(value: String) -> Self {
        Self(value)
    }

    /// Returns the value of the constant.
    pub fn value(&self) -> &str {
        &self.0
    }
}

#[typetag::serde]
impl CustomConst for ConstString {
    fn name(&self) -> ValueName {
        format!("ConstString({:?})", self.0).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(PRELUDE_ID)
    }

    fn get_type(&self) -> Type {
        string_type()
    }
}

/// Name of the print operation
pub const PRINT_OP_ID: OpName = OpName::new_inline("print");

/// The custom type for Errors.
///
/// Receives a reference to the prelude extensions as a parameter.
/// This avoids deadlocks when we are in the process of creating the prelude.
fn error_custom_type(extension_ref: &Weak<Extension>) -> CustomType {
    CustomType::new(
        ERROR_TYPE_NAME,
        vec![],
        PRELUDE_ID,
        TypeBound::Copyable,
        extension_ref,
    )
}

/// Unspecified opaque error type.
pub fn error_type() -> Type {
    error_custom_type(&Arc::downgrade(&PRELUDE)).into()
}

/// The string name of the error type.
pub const ERROR_TYPE_NAME: TypeName = TypeName::new_inline("error");

/// Return a Sum type with the second variant as the given type and the first an Error.
pub fn sum_with_error(ty: impl Into<TypeRowRV>) -> SumType {
    either_type(error_type(), ty)
}

/// An optional type, i.e. a Sum type with the second variant as the given type and the first as an empty tuple.
#[inline]
pub fn option_type(ty: impl Into<TypeRowRV>) -> SumType {
    either_type(TypeRow::new(), ty)
}

/// An "either" type, i.e. a Sum type with a "left" and a "right" variant.
///
/// When used as a fallible value, the "right" variant represents a successful computation,
/// and the "left" variant represents a failure.
#[inline]
pub fn either_type(ty_left: impl Into<TypeRowRV>, ty_right: impl Into<TypeRowRV>) -> SumType {
    SumType::new([ty_left.into(), ty_right.into()])
}

/// A constant optional value with a given value.
///
/// See [option_type].
pub fn const_some(value: Value) -> Value {
    const_some_tuple([value])
}

/// A constant optional value with a row of values.
///
/// For single values, use [const_some].
///
/// See [option_type].
pub fn const_some_tuple(values: impl IntoIterator<Item = Value>) -> Value {
    const_right_tuple(TypeRow::new(), values)
}

/// A constant optional value with no value.
///
/// See [option_type].
pub fn const_none(ty: impl Into<TypeRowRV>) -> Value {
    const_left_tuple([], ty)
}

/// A constant Either value with a left variant.
///
/// In fallible computations, this represents a failure.
///
/// See [either_type].
pub fn const_left(value: Value, ty_right: impl Into<TypeRowRV>) -> Value {
    const_left_tuple([value], ty_right)
}

/// A constant Either value with a row of left values.
///
/// In fallible computations, this represents a failure.
///
/// See [either_type].
pub fn const_left_tuple(
    values: impl IntoIterator<Item = Value>,
    ty_right: impl Into<TypeRowRV>,
) -> Value {
    let values = values.into_iter().collect_vec();
    let types: TypeRowRV = values
        .iter()
        .map(|v| TypeRV::from(v.get_type()))
        .collect_vec()
        .into();
    let typ = either_type(types, ty_right);
    Value::sum(0, values, typ).unwrap()
}

/// A constant Either value with a right variant.
///
/// In fallible computations, this represents a successful result.
///
/// See [either_type].
pub fn const_right(ty_left: impl Into<TypeRowRV>, value: Value) -> Value {
    const_right_tuple(ty_left, [value])
}

/// A constant Either value with a row of right values.
///
/// In fallible computations, this represents a successful result.
///
/// See [either_type].
pub fn const_right_tuple(
    ty_left: impl Into<TypeRowRV>,
    values: impl IntoIterator<Item = Value>,
) -> Value {
    let values = values.into_iter().collect_vec();
    let types: TypeRowRV = values
        .iter()
        .map(|v| TypeRV::from(v.get_type()))
        .collect_vec()
        .into();
    let typ = either_type(ty_left, types);
    Value::sum(1, values, typ).unwrap()
}

/// A constant Either value with a success variant.
///
/// Alias for [const_right].
pub fn const_ok(value: Value, ty_fail: impl Into<TypeRowRV>) -> Value {
    const_right(ty_fail, value)
}

/// A constant Either with a row of success values.
///
/// Alias for [const_right_tuple].
pub fn const_ok_tuple(
    values: impl IntoIterator<Item = Value>,
    ty_fail: impl Into<TypeRowRV>,
) -> Value {
    const_right_tuple(ty_fail, values)
}

/// A constant Either value with a failure variant.
///
/// Alias for [const_left].
pub fn const_fail(value: Value, ty_ok: impl Into<TypeRowRV>) -> Value {
    const_left(value, ty_ok)
}

/// A constant Either with a row of failure values.
///
/// Alias for [const_left_tuple].
pub fn const_fail_tuple(
    values: impl IntoIterator<Item = Value>,
    ty_ok: impl Into<TypeRowRV>,
) -> Value {
    const_left_tuple(values, ty_ok)
}

#[derive(Debug, Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
/// Structure for holding constant usize values.
pub struct ConstUsize(u64);

impl ConstUsize {
    /// Creates a new [`ConstUsize`].
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the value of the constant.
    pub fn value(&self) -> u64 {
        self.0
    }
}

#[typetag::serde]
impl CustomConst for ConstUsize {
    fn name(&self) -> ValueName {
        format!("ConstUsize({})", self.0).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(PRELUDE_ID)
    }

    fn get_type(&self) -> Type {
        usize_t()
    }
}

#[derive(Debug, Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
/// Structure for holding constant usize values.
pub struct ConstError {
    /// Integer tag/signal for the error.
    pub signal: u32,
    /// Error message.
    pub message: String,
}

impl ConstError {
    /// Define a new error value.
    pub fn new(signal: u32, message: impl ToString) -> Self {
        Self {
            signal,
            message: message.to_string(),
        }
    }

    /// Returns an "either" value with a failure variant.
    ///
    /// args:
    ///     ty_ok: The type of the success variant.
    pub fn as_either(self, ty_ok: impl Into<TypeRowRV>) -> Value {
        const_fail(self.into(), ty_ok)
    }
}

#[typetag::serde]
impl CustomConst for ConstError {
    fn name(&self) -> ValueName {
        format!("ConstError({}, {:?})", self.signal, self.message).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(PRELUDE_ID)
    }
    fn get_type(&self) -> Type {
        error_type()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// A structure for holding references to external symbols.
pub struct ConstExternalSymbol {
    /// The symbol name that this value refers to. Must be nonempty.
    pub symbol: String,
    /// The type of the value found at this symbol reference.
    pub typ: Type,
    /// Whether the value at the symbol reference is constant or mutable.
    pub constant: bool,
}

impl ConstExternalSymbol {
    /// Construct a new [ConstExternalSymbol].
    pub fn new(symbol: impl Into<String>, typ: impl Into<Type>, constant: bool) -> Self {
        Self {
            symbol: symbol.into(),
            typ: typ.into(),
            constant,
        }
    }
}

impl PartialEq<dyn CustomConst> for ConstExternalSymbol {
    fn eq(&self, other: &dyn CustomConst) -> bool {
        self.equal_consts(other)
    }
}

#[typetag::serde]
impl CustomConst for ConstExternalSymbol {
    fn name(&self) -> ValueName {
        format!("@{}", &self.symbol).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(PRELUDE_ID)
    }
    fn get_type(&self) -> Type {
        self.typ.clone()
    }

    fn validate(&self) -> Result<(), CustomCheckFailure> {
        if self.symbol.is_empty() {
            Err(CustomCheckFailure::Message(
                "External symbol name is empty.".into(),
            ))
        } else {
            Ok(())
        }
    }
}

/// Logic extension operation definitions.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum TupleOpDef {
    MakeTuple,
    UnpackTuple,
}

impl ConstFold for TupleOpDef {
    fn fold(
        &self,
        _type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, Value)],
    ) -> crate::extension::ConstFoldResult {
        match self {
            TupleOpDef::MakeTuple => {
                fold_out_row([Value::tuple(sorted_consts(consts).into_iter().cloned())])
            }
            TupleOpDef::UnpackTuple => {
                let c = &consts.first()?.1;
                let Some(vs) = c.as_tuple() else {
                    panic!("This op always takes a Tuple input.");
                };
                fold_out_row(vs.iter().cloned())
            }
        }
    }
}
impl MakeOpDef for TupleOpDef {
    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        let rv = TypeRV::new_row_var_use(0, TypeBound::Any);
        let tuple_type = TypeRV::new_tuple(vec![rv.clone()]);

        let param = TypeParam::new_list(TypeBound::Any);
        match self {
            TupleOpDef::MakeTuple => {
                PolyFuncTypeRV::new([param], FuncValueType::new(rv, tuple_type))
            }
            TupleOpDef::UnpackTuple => {
                PolyFuncTypeRV::new([param], FuncValueType::new(tuple_type, rv))
            }
        }
        .into()
    }

    fn description(&self) -> String {
        match self {
            TupleOpDef::MakeTuple => "MakeTuple operation",
            TupleOpDef::UnpackTuple => "UnpackTuple operation",
        }
        .to_string()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        PRELUDE_ID.to_owned()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&PRELUDE)
    }

    fn post_opdef(&self, def: &mut OpDef) {
        def.set_constant_folder(*self);
    }
}
/// An operation that packs all its inputs into a tuple.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
#[non_exhaustive]
pub struct MakeTuple(pub TypeRow);

impl MakeTuple {
    /// Create a new MakeTuple operation.
    pub fn new(tys: TypeRow) -> Self {
        Self(tys)
    }
}

impl NamedOp for MakeTuple {
    fn name(&self) -> OpName {
        TupleOpDef::MakeTuple.name()
    }
}

impl MakeExtensionOp for MakeTuple {
    fn from_extension_op(ext_op: &crate::ops::ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = TupleOpDef::from_def(ext_op.def())?;
        if def != TupleOpDef::MakeTuple {
            return Err(OpLoadError::NotMember(ext_op.def().name().to_string()))?;
        }
        let [TypeArg::Sequence { elems }] = ext_op.args() else {
            return Err(SignatureError::InvalidTypeArgs)?;
        };
        let tys: Result<Vec<Type>, _> = elems
            .iter()
            .map(|a| match a {
                TypeArg::Type { ty } => Ok(ty.clone()),
                _ => Err(SignatureError::InvalidTypeArgs),
            })
            .collect();
        Ok(Self(tys?.into()))
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![TypeArg::Sequence {
            elems: self
                .0
                .iter()
                .map(|t| TypeArg::Type { ty: t.clone() })
                .collect(),
        }]
    }
}

impl MakeRegisteredOp for MakeTuple {
    fn extension_id(&self) -> ExtensionId {
        PRELUDE_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &PRELUDE_REGISTRY
    }
}

/// An operation that unpacks a tuple into its components.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
#[non_exhaustive]
pub struct UnpackTuple(pub TypeRow);

impl UnpackTuple {
    /// Create a new UnpackTuple operation.
    pub fn new(tys: TypeRow) -> Self {
        Self(tys)
    }
}

impl NamedOp for UnpackTuple {
    fn name(&self) -> OpName {
        TupleOpDef::UnpackTuple.name()
    }
}

impl MakeExtensionOp for UnpackTuple {
    fn from_extension_op(ext_op: &crate::ops::ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = TupleOpDef::from_def(ext_op.def())?;
        if def != TupleOpDef::UnpackTuple {
            return Err(OpLoadError::NotMember(ext_op.def().name().to_string()))?;
        }
        let [TypeArg::Sequence { elems }] = ext_op.args() else {
            return Err(SignatureError::InvalidTypeArgs)?;
        };
        let tys: Result<Vec<Type>, _> = elems
            .iter()
            .map(|a| match a {
                TypeArg::Type { ty } => Ok(ty.clone()),
                _ => Err(SignatureError::InvalidTypeArgs),
            })
            .collect();
        Ok(Self(tys?.into()))
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![TypeArg::Sequence {
            elems: self
                .0
                .iter()
                .map(|t| TypeArg::Type { ty: t.clone() })
                .collect(),
        }]
    }
}

impl MakeRegisteredOp for UnpackTuple {
    fn extension_id(&self) -> ExtensionId {
        PRELUDE_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &PRELUDE_REGISTRY
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// A no-op operation definition.
pub struct NoopDef;

impl NamedOp for NoopDef {
    fn name(&self) -> OpName {
        "Noop".into()
    }
}

impl std::str::FromStr for NoopDef {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == NoopDef.name() {
            Ok(Self)
        } else {
            Err(())
        }
    }
}
impl MakeOpDef for NoopDef {
    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        let tv = Type::new_var_use(0, TypeBound::Any);
        PolyFuncType::new([TypeBound::Any.into()], Signature::new_endo(tv)).into()
    }

    fn description(&self) -> String {
        "Noop gate".to_string()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        PRELUDE_ID.to_owned()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&PRELUDE)
    }

    fn post_opdef(&self, def: &mut OpDef) {
        def.set_constant_folder(*self);
    }
}

impl ConstFold for NoopDef {
    fn fold(
        &self,
        _type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, Value)],
    ) -> crate::extension::ConstFoldResult {
        fold_out_row([consts.first()?.1.clone()])
    }
}

/// A no-op operation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Noop(pub Type);

impl Noop {
    /// Create a new Noop operation.
    pub fn new(ty: Type) -> Self {
        Self(ty)
    }
}

impl Default for Noop {
    fn default() -> Self {
        Self(Type::UNIT)
    }
}
impl NamedOp for Noop {
    fn name(&self) -> OpName {
        NoopDef.name()
    }
}

impl MakeExtensionOp for Noop {
    fn from_extension_op(ext_op: &crate::ops::ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let _def = NoopDef::from_def(ext_op.def())?;
        let [TypeArg::Type { ty }] = ext_op.args() else {
            return Err(SignatureError::InvalidTypeArgs)?;
        };
        Ok(Self(ty.clone()))
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![TypeArg::Type { ty: self.0.clone() }]
    }
}

impl MakeRegisteredOp for Noop {
    fn extension_id(&self) -> ExtensionId {
        PRELUDE_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &PRELUDE_REGISTRY
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// A lift operation definition.
pub struct LiftDef;

impl NamedOp for LiftDef {
    fn name(&self) -> OpName {
        "Lift".into()
    }
}

impl std::str::FromStr for LiftDef {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == LiftDef.name() {
            Ok(Self)
        } else {
            Err(())
        }
    }
}

impl MakeOpDef for LiftDef {
    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        PolyFuncTypeRV::new(
            vec![TypeParam::Extensions, TypeParam::new_list(TypeBound::Any)],
            FuncValueType::new_endo(TypeRV::new_row_var_use(1, TypeBound::Any))
                .with_extension_delta(ExtensionSet::type_var(0)),
        )
        .into()
    }

    fn description(&self) -> String {
        "Add extension requirements to a row of values".to_string()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        PRELUDE_ID.to_owned()
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&PRELUDE)
    }
}

/// A node which adds a extension req to the types of the wires it is passed
/// It has no effect on the values passed along the edge
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
#[non_exhaustive]
pub struct Lift {
    /// The types of the edges
    pub type_row: TypeRow,
    /// The extensions which we're adding to the inputs
    pub new_extensions: ExtensionSet,
}

impl Lift {
    /// Create a new Lift operation with the extensions to add.
    pub fn new(type_row: TypeRow, set: impl Into<ExtensionSet>) -> Self {
        Self {
            type_row,
            new_extensions: set.into(),
        }
    }
}

impl NamedOp for Lift {
    fn name(&self) -> OpName {
        LiftDef.name()
    }
}

impl MakeExtensionOp for Lift {
    fn from_extension_op(ext_op: &crate::ops::ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let _def = LiftDef::from_def(ext_op.def())?;

        let [TypeArg::Extensions { es }, TypeArg::Sequence { elems }] = ext_op.args() else {
            return Err(SignatureError::InvalidTypeArgs)?;
        };
        let tys: Result<Vec<Type>, _> = elems
            .iter()
            .map(|a| match a {
                TypeArg::Type { ty } => Ok(ty.clone()),
                _ => Err(SignatureError::InvalidTypeArgs),
            })
            .collect();
        Ok(Self {
            type_row: tys?.into(),
            new_extensions: es.clone(),
        })
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![
            TypeArg::Extensions {
                es: self.new_extensions.clone(),
            },
            TypeArg::Sequence {
                elems: self
                    .type_row
                    .iter()
                    .map(|t| TypeArg::Type { ty: t.clone() })
                    .collect(),
            },
        ]
    }
}

impl MakeRegisteredOp for Lift {
    fn extension_id(&self) -> ExtensionId {
        PRELUDE_ID.to_owned()
    }

    fn registry<'s, 'r: 's>(&'s self) -> &'r crate::extension::ExtensionRegistry {
        &PRELUDE_REGISTRY
    }
}

#[cfg(test)]
mod test {
    use crate::builder::inout_sig;
    use crate::std_extensions::arithmetic::float_ops::FLOAT_OPS_REGISTRY;
    use crate::std_extensions::arithmetic::float_types::{float64_type, ConstF64};
    use crate::utils::test_quantum_extension;
    use crate::{
        builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
        utils::test_quantum_extension::cx_gate,
        Hugr, Wire,
    };

    use super::*;
    use crate::{
        ops::{OpTrait, OpType},
        type_row,
    };

    #[test]
    fn test_make_tuple() {
        let op = MakeTuple::new(type_row![Type::UNIT]);
        let optype: OpType = op.clone().into();
        assert_eq!(
            optype.dataflow_signature().unwrap().io(),
            (
                &type_row![Type::UNIT],
                &vec![Type::new_tuple(type_row![Type::UNIT])].into(),
            )
        );

        let new_op = MakeTuple::from_extension_op(optype.as_extension_op().unwrap()).unwrap();
        assert_eq!(new_op, op);
    }

    #[test]
    fn test_unmake_tuple() {
        let op = UnpackTuple::new(type_row![Type::UNIT]);
        let optype: OpType = op.clone().into();
        assert_eq!(
            optype.dataflow_signature().unwrap().io(),
            (
                &vec![Type::new_tuple(type_row![Type::UNIT])].into(),
                &type_row![Type::UNIT],
            )
        );

        let new_op = UnpackTuple::from_extension_op(optype.as_extension_op().unwrap()).unwrap();
        assert_eq!(new_op, op);
    }

    #[test]
    fn test_noop() {
        let op = Noop::new(Type::UNIT);
        let optype: OpType = op.clone().into();
        assert_eq!(
            optype.dataflow_signature().unwrap().io(),
            (&type_row![Type::UNIT], &type_row![Type::UNIT])
        );

        let new_op = Noop::from_extension_op(optype.as_extension_op().unwrap()).unwrap();
        assert_eq!(new_op, op);
    }

    #[test]
    fn test_lift() {
        const XA: ExtensionId = ExtensionId::new_unchecked("xa");
        let op = Lift::new(type_row![Type::UNIT], ExtensionSet::singleton(XA));
        let optype: OpType = op.clone().into();
        assert_eq!(
            optype.dataflow_signature().unwrap(),
            Signature::new_endo(type_row![Type::UNIT])
                .with_extension_delta(XA)
                .with_prelude()
        );

        let new_op = Lift::from_extension_op(optype.as_extension_op().unwrap()).unwrap();
        assert_eq!(new_op, op);
    }

    #[test]
    /// Test building a HUGR involving a new_array operation.
    fn test_new_array() {
        let mut b = DFGBuilder::new(inout_sig(
            vec![qb_t(), qb_t()],
            array_type(TypeArg::BoundedNat { n: 2 }, qb_t()),
        ))
        .unwrap();

        let [q1, q2] = b.input_wires_arr();

        let op = new_array_op(qb_t(), 2);

        let out = b.add_dataflow_op(op, [q1, q2]).unwrap();

        b.finish_prelude_hugr_with_outputs(out.outputs()).unwrap();
    }

    #[test]
    fn test_option() {
        let typ: Type = option_type(bool_t()).into();
        let const_val1 = const_some(Value::true_val());
        let const_val2 = const_none(bool_t());

        let mut b = DFGBuilder::new(inout_sig(type_row![], vec![typ.clone(), typ])).unwrap();

        let some = b.add_load_value(const_val1);
        let none = b.add_load_value(const_val2);

        b.finish_prelude_hugr_with_outputs([some, none]).unwrap();
    }

    #[test]
    fn test_result() {
        let typ: Type = either_type(bool_t(), float64_type()).into();
        let const_bool = const_left(Value::true_val(), float64_type());
        let const_float = const_right(bool_t(), ConstF64::new(0.5).into());

        let mut b = DFGBuilder::new(inout_sig(type_row![], vec![typ.clone(), typ])).unwrap();

        let bool = b.add_load_value(const_bool);
        let float = b.add_load_value(const_float);

        b.finish_hugr_with_outputs([bool, float], &FLOAT_OPS_REGISTRY)
            .unwrap();
    }

    #[test]
    /// test the prelude error type and panic op.
    fn test_error_type() {
        let ext_def = PRELUDE
            .get_type(&ERROR_TYPE_NAME)
            .unwrap()
            .instantiate([])
            .unwrap();

        let ext_type = Type::new_extension(ext_def);
        assert_eq!(ext_type, error_type());

        let error_val = ConstError::new(2, "my message");

        assert_eq!(error_val.name(), "ConstError(2, \"my message\")");

        assert!(error_val.validate().is_ok());

        assert_eq!(
            error_val.extension_reqs(),
            ExtensionSet::singleton(PRELUDE_ID)
        );
        assert!(error_val.equal_consts(&ConstError::new(2, "my message")));
        assert!(!error_val.equal_consts(&ConstError::new(3, "my message")));

        let mut b = DFGBuilder::new(endo_sig(type_row![])).unwrap();

        let err = b.add_load_value(error_val);

        const TYPE_ARG_NONE: TypeArg = TypeArg::Sequence { elems: vec![] };
        let op = PRELUDE
            .instantiate_extension_op(
                &PANIC_OP_ID,
                [TYPE_ARG_NONE, TYPE_ARG_NONE],
                &PRELUDE_REGISTRY,
            )
            .unwrap();

        b.add_dataflow_op(op, [err]).unwrap();

        b.finish_prelude_hugr_with_outputs([]).unwrap();
    }

    #[test]
    /// test the panic operation with input and output wires
    fn test_panic_with_io() {
        let error_val = ConstError::new(42, "PANIC");
        let type_arg_q: TypeArg = TypeArg::Type { ty: qb_t() };
        let type_arg_2q: TypeArg = TypeArg::Sequence {
            elems: vec![type_arg_q.clone(), type_arg_q],
        };
        let panic_op = PRELUDE
            .instantiate_extension_op(
                &PANIC_OP_ID,
                [type_arg_2q.clone(), type_arg_2q.clone()],
                &PRELUDE_REGISTRY,
            )
            .unwrap();

        let mut b = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()])).unwrap();
        let [q0, q1] = b.input_wires_arr();
        let [q0, q1] = b
            .add_dataflow_op(cx_gate(), [q0, q1])
            .unwrap()
            .outputs_arr();
        let err = b.add_load_value(error_val);
        let [q0, q1] = b
            .add_dataflow_op(panic_op, [err, q0, q1])
            .unwrap()
            .outputs_arr();
        b.finish_hugr_with_outputs([q0, q1], &test_quantum_extension::REG)
            .unwrap();
    }

    #[test]
    /// Test string type.
    fn test_string_type() {
        let string_custom_type: CustomType = PRELUDE
            .get_type(&STRING_TYPE_NAME)
            .unwrap()
            .instantiate([])
            .unwrap();
        let string_ty: Type = Type::new_extension(string_custom_type);
        assert_eq!(string_ty, string_type());
        let string_const: ConstString = ConstString::new("Lorem ipsum".into());
        assert_eq!(string_const.name(), "ConstString(\"Lorem ipsum\")");
        assert!(string_const.validate().is_ok());
        assert_eq!(
            string_const.extension_reqs(),
            ExtensionSet::singleton(PRELUDE_ID)
        );
        assert!(string_const.equal_consts(&ConstString::new("Lorem ipsum".into())));
        assert!(!string_const.equal_consts(&ConstString::new("Lorem ispum".into())));
    }

    #[test]
    /// Test print operation
    fn test_print() {
        let mut b: DFGBuilder<Hugr> = DFGBuilder::new(endo_sig(vec![])).unwrap();
        let greeting: ConstString = ConstString::new("Hello, world!".into());
        let greeting_out: Wire = b.add_load_value(greeting);
        let print_op = PRELUDE
            .instantiate_extension_op(&PRINT_OP_ID, [], &PRELUDE_REGISTRY)
            .unwrap();
        b.add_dataflow_op(print_op, [greeting_out]).unwrap();
        b.finish_prelude_hugr_with_outputs([]).unwrap();
    }

    #[test]
    fn test_external_symbol() {
        let subject = ConstExternalSymbol::new("foo", Type::UNIT, false);
        assert_eq!(subject.get_type(), Type::UNIT);
        assert_eq!(subject.name(), "@foo");
        assert!(subject.validate().is_ok());
        assert_eq!(
            subject.extension_reqs(),
            ExtensionSet::singleton(PRELUDE_ID)
        );
        assert!(subject.equal_consts(&ConstExternalSymbol::new("foo", Type::UNIT, false)));
        assert!(!subject.equal_consts(&ConstExternalSymbol::new("bar", Type::UNIT, false)));
        assert!(!subject.equal_consts(&ConstExternalSymbol::new("foo", string_type(), false)));
        assert!(!subject.equal_consts(&ConstExternalSymbol::new("foo", Type::UNIT, true)));

        assert!(ConstExternalSymbol::new("", Type::UNIT, true)
            .validate()
            .is_err())
    }
}
