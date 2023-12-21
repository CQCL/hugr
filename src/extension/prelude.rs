//! Prelude extension - available in all contexts, defining common types,
//! operations and constants.
use lazy_static::lazy_static;
use smol_str::SmolStr;

use crate::{
    extension::{ExtensionId, TypeDefBound},
    ops::LeafOp,
    types::{
        type_param::{TypeArg, TypeParam},
        CustomCheckFailure, CustomType, FunctionType, PolyFuncType, Type, TypeBound,
    },
    values::{CustomConst, KnownTypeConst},
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


        prelude
            .add_type(
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

/// Initialize a new array op of element type `element_ty` of length `size`
pub fn new_array_op(element_ty: Type, size: u64) -> LeafOp {
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

/// The custom type for Errors.
pub const ERROR_CUSTOM_TYPE: CustomType =
    CustomType::new_simple(ERROR_TYPE_NAME, PRELUDE_ID, TypeBound::Eq);
/// Unspecified opaque error type.
pub const ERROR_TYPE: Type = Type::new_extension(ERROR_CUSTOM_TYPE);

/// The string name of the error type.
pub const ERROR_TYPE_NAME: SmolStr = SmolStr::new_inline("error");

/// Return a Sum type with the first variant as the given type and the second an Error.
pub fn sum_with_error(ty: Type) -> Type {
    Type::new_sum(vec![ty, ERROR_TYPE])
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

    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        self.check_known_type(typ)
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::values::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&PRELUDE_ID)
    }
}

impl KnownTypeConst for ConstUsize {
    const TYPE: CustomType = USIZE_CUSTOM_T;
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

    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        self.check_known_type(typ)
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::values::downcast_equal_consts(self, other)
    }

    fn extension_reqs(&self) -> ExtensionSet {
        ExtensionSet::singleton(&PRELUDE_ID)
    }
}

impl KnownTypeConst for ConstError {
    const TYPE: CustomType = ERROR_CUSTOM_TYPE;
}

#[cfg(test)]
mod test {
    use crate::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        types::FunctionType,
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
    /// test the prelude error type.
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

        assert!(error_val.check_custom_type(&ERROR_CUSTOM_TYPE).is_ok());

        assert_eq!(
            error_val.extension_reqs(),
            ExtensionSet::singleton(&PRELUDE_ID)
        );
        assert!(error_val.equal_consts(&ConstError::new(2, "my message")));
        assert!(!error_val.equal_consts(&ConstError::new(3, "my message")));
    }
}
