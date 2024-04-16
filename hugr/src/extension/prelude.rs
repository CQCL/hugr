//! Prelude extension - available in all contexts, defining common types,
//! operations and constants.
use lazy_static::lazy_static;
use smol_str::SmolStr;

use crate::ops::CustomOp;
use crate::types::SumType;
use crate::{
    extension::{ExtensionId, TypeDefBound},
    ops::constant::CustomConst,
    type_row,
    types::{
        type_param::{TypeArg, TypeParam},
        CustomType, FunctionType, PolyFuncType, Type, TypeBound,
    },
    Extension,
};

use super::{ExtensionRegistry, ExtensionSet, SignatureError, SignatureFromArgs};
struct ArrayOpCustom;

const MAX: &[TypeParam; 1] = &[TypeParam::max_nat()];
impl SignatureFromArgs for ArrayOpCustom {
    fn compute_signature(&self, arg_values: &[TypeArg]) -> Result<PolyFuncType, SignatureError> {
        let [TypeArg::BoundedNat { n }] = *arg_values else {
            return Err(SignatureError::InvalidTypeArgs);
        };
        let elem_ty_var = Type::new_var_use(0, TypeBound::Any);

        let var_arg_row = vec![elem_ty_var.clone(); n as usize];
        let other_row = vec![array_type(TypeArg::BoundedNat { n }, elem_ty_var.clone())];

        Ok(PolyFuncType::new(
            vec![TypeBound::Any.into()],
            FunctionType::new(var_arg_row, other_row),
        ))
    }

    fn static_params(&self) -> &[TypeParam] {
        MAX
    }
}

/// Name of prelude extension.
pub const PRELUDE_ID: ExtensionId = ExtensionId::new_unchecked("prelude");
lazy_static! {
    static ref PRELUDE_DEF: Extension = {
        let mut prelude = Extension::new(PRELUDE_ID);
        prelude
            .add_type(
                SmolStr::new_inline("usize"),
                vec![],
                "usize".into(),
                TypeDefBound::Explicit(crate::types::TypeBound::Eq),
            )
            .unwrap();
        prelude.add_type(
                SmolStr::new_inline("string"),
                vec![],
                "string".into(),
                TypeDefBound::Explicit(crate::types::TypeBound::Eq),
            )
            .unwrap();
        prelude.add_op(
            SmolStr::new_inline(PRINT_OP_ID),
            "Print the string to standard output".to_string(),
            FunctionType::new(type_row![Type::new_extension(STRING_CUSTOM_TYPE)], type_row![]),
            )
            .unwrap();
        prelude.add_type(
                SmolStr::new_inline("array"),
                vec![ TypeParam::max_nat(), TypeBound::Any.into()],
                "array".into(),
                TypeDefBound::FromParams(vec![1]),
            )
            .unwrap();
        prelude
            .add_op(
                SmolStr::new_inline(NEW_ARRAY_OP_ID),
                "Create a new array from elements".to_string(),
                ArrayOpCustom,
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
        .add_type(
            ERROR_TYPE_NAME,
            vec![],
            "Simple opaque error type.".into(),
            TypeDefBound::Explicit(TypeBound::Eq),
        )
        .unwrap();
        prelude
        .add_op(
            SmolStr::new_inline(PANIC_OP_ID),
            "Panic with input error".to_string(),
            FunctionType::new(type_row![Type::new_extension(ERROR_CUSTOM_TYPE)], type_row![]),
        )
        .unwrap();
        prelude
    };
    /// An extension registry containing only the prelude
    pub static ref PRELUDE_REGISTRY: ExtensionRegistry =
        ExtensionRegistry::try_new([PRELUDE_DEF.to_owned()]).unwrap();

    /// Prelude extension
    pub static ref PRELUDE: &'static Extension = PRELUDE_REGISTRY.get(&PRELUDE_ID).unwrap();

}

pub(crate) const USIZE_CUSTOM_T: CustomType =
    CustomType::new_simple(SmolStr::new_inline("usize"), PRELUDE_ID, TypeBound::Eq);

pub(crate) const QB_CUSTOM_T: CustomType =
    CustomType::new_simple(SmolStr::new_inline("qubit"), PRELUDE_ID, TypeBound::Any);

/// Qubit type.
pub const QB_T: Type = Type::new_extension(QB_CUSTOM_T);
/// Unsigned size type.
pub const USIZE_T: Type = Type::new_extension(USIZE_CUSTOM_T);
/// Boolean type - Sum of two units.
pub const BOOL_T: Type = Type::new_unit_sum(2);

/// Initialize a new array of element type `element_ty` of length `size`
pub fn array_type(size: TypeArg, element_ty: Type) -> Type {
    let array_def = PRELUDE.get_type("array").unwrap();
    let custom_t = array_def
        .instantiate(vec![size, TypeArg::Type { ty: element_ty }])
        .unwrap();
    Type::new_extension(custom_t)
}

/// Name of the operation in the prelude for creating new arrays.
pub const NEW_ARRAY_OP_ID: &str = "new_array";
/// Name of the prelude panic operation.
pub const PANIC_OP_ID: &str = "panic";

/// Initialize a new array op of element type `element_ty` of length `size`
pub fn new_array_op(element_ty: Type, size: u64) -> CustomOp {
    PRELUDE
        .instantiate_extension_op(
            NEW_ARRAY_OP_ID,
            vec![
                TypeArg::BoundedNat { n: size },
                TypeArg::Type { ty: element_ty },
            ],
            &PRELUDE_REGISTRY,
        )
        .unwrap()
        .into()
}

/// Name of the string type.
pub const STRING_TYPE_NAME: SmolStr = SmolStr::new_inline("string");

/// Custom type for strings.
pub const STRING_CUSTOM_TYPE: CustomType =
    CustomType::new_simple(STRING_TYPE_NAME, PRELUDE_ID, TypeBound::Eq);

/// String type.
pub const STRING_TYPE: Type = Type::new_extension(STRING_CUSTOM_TYPE);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Structure for holding constant string values.
pub struct ConstString(String);

impl ConstString {
    /// Creates a new [`ConstString`].
    pub fn new(value: String) -> Self {
        Self(value)
    }

    /// Returns the value of the constant.
    pub fn value(&self) -> String {
        self.0.clone()
    }
}

#[typetag::serde]
impl CustomConst for ConstString {
    fn name(&self) -> SmolStr {
        format!("ConstString({:?})", self.0).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&PRELUDE_ID)
    }

    fn get_type(&self) -> Type {
        STRING_TYPE
    }
}

/// Name of the print operation
pub const PRINT_OP_ID: &str = "print";

/// The custom type for Errors.
pub const ERROR_CUSTOM_TYPE: CustomType =
    CustomType::new_simple(ERROR_TYPE_NAME, PRELUDE_ID, TypeBound::Eq);
/// Unspecified opaque error type.
pub const ERROR_TYPE: Type = Type::new_extension(ERROR_CUSTOM_TYPE);

/// The string name of the error type.
pub const ERROR_TYPE_NAME: SmolStr = SmolStr::new_inline("error");

/// Return a Sum type with the first variant as the given type and the second an Error.
pub fn sum_with_error(ty: Type) -> SumType {
    SumType::new([vec![ty], vec![ERROR_TYPE]])
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    fn name(&self) -> SmolStr {
        format!("ConstUsize({:?})", self.0).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&PRELUDE_ID)
    }

    fn get_type(&self) -> Type {
        USIZE_T
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
}

#[typetag::serde]
impl CustomConst for ConstError {
    fn name(&self) -> SmolStr {
        format!("ConstError({:?}, {:?})", self.signal, self.message).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&PRELUDE_ID)
    }
    fn get_type(&self) -> Type {
        ERROR_TYPE
    }
}

#[cfg(test)]
mod test {
    use crate::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        types::FunctionType,
        Hugr, Wire,
    };

    use super::*;

    #[test]
    /// Test building a HUGR involving a new_array operation.
    fn test_new_array() {
        let mut b = DFGBuilder::new(FunctionType::new(
            vec![QB_T, QB_T],
            vec![array_type(TypeArg::BoundedNat { n: 2 }, QB_T)],
        ))
        .unwrap();

        let [q1, q2] = b.input_wires_arr();

        let op = new_array_op(QB_T, 2);

        let out = b.add_dataflow_op(op, [q1, q2]).unwrap();

        b.finish_prelude_hugr_with_outputs(out.outputs()).unwrap();
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
        assert_eq!(ext_type, ERROR_TYPE);

        let error_val = ConstError::new(2, "my message");

        assert_eq!(error_val.name(), "ConstError(2, \"my message\")");

        assert!(error_val.validate().is_ok());

        assert_eq!(
            error_val.extension_reqs(),
            ExtensionSet::singleton(&PRELUDE_ID)
        );
        assert!(error_val.equal_consts(&ConstError::new(2, "my message")));
        assert!(!error_val.equal_consts(&ConstError::new(3, "my message")));

        let mut b = DFGBuilder::new(FunctionType::new_endo(type_row![])).unwrap();

        let err = b.add_load_const(error_val);

        let op = PRELUDE
            .instantiate_extension_op(PANIC_OP_ID, [], &PRELUDE_REGISTRY)
            .unwrap();

        b.add_dataflow_op(op, [err]).unwrap();

        b.finish_prelude_hugr_with_outputs([]).unwrap();
    }

    #[test]
    /// Test string type.
    fn test_string_type() {
        let string_custom_type: CustomType = PRELUDE
            .get_type(&STRING_TYPE_NAME)
            .unwrap()
            .instantiate([])
            .unwrap();
        let string_type: Type = Type::new_extension(string_custom_type);
        assert_eq!(string_type, STRING_TYPE);
        let string_const: ConstString = ConstString::new("Lorem ipsum".into());
        assert_eq!(string_const.name(), "ConstString(\"Lorem ipsum\")");
        assert!(string_const.validate().is_ok());
        assert_eq!(
            string_const.extension_reqs(),
            ExtensionSet::singleton(&PRELUDE_ID)
        );
        assert!(string_const.equal_consts(&ConstString::new("Lorem ipsum".into())));
        assert!(!string_const.equal_consts(&ConstString::new("Lorem ispum".into())));
    }

    #[test]
    /// Test print operation
    fn test_print() {
        let mut b: DFGBuilder<Hugr> = DFGBuilder::new(FunctionType::new(vec![], vec![])).unwrap();
        let greeting: ConstString = ConstString::new("Hello, world!".into());
        let greeting_out: Wire = b.add_load_const(greeting);
        let print_op = PRELUDE
            .instantiate_extension_op(PRINT_OP_ID, [], &PRELUDE_REGISTRY)
            .unwrap();
        b.add_dataflow_op(print_op, [greeting_out]).unwrap();
        b.finish_prelude_hugr_with_outputs([]).unwrap();
    }
}
