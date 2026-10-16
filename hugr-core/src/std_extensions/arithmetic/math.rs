//! Double-precision math operations. Angles are in radians.

use std::sync::{Arc, LazyLock, Weak};

use strum::{EnumIter, EnumString, IntoStaticStr};

use super::float_types::float64_type;
use crate::{
    Extension, Wire,
    builder::{BuildError, Dataflow},
    extension::{
        ExtensionId, OpDef, SignatureFunc, Version,
        simple_op::{MakeOpDef, MakeRegisteredOp, OpLoadError, try_from_name},
    },
    ops::OpName,
    types::Signature,
};

/// The math extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.math");
/// The math extension version.
pub const VERSION: Version = Version::new(0, 1, 0);

/// Math functions operating on 64-bit floats.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
#[allow(non_camel_case_types)]
#[non_exhaustive]
pub enum MathOps {
    /// Sine of an angle in radians.
    sin,
    /// Cosine of an angle in radians.
    cos,
    /// Tangent of an angle in radians.
    tan,
    /// Inverse tangent, in radians.
    atan,
    /// Four-quadrant inverse tangent of (y, x), in radians.
    atan2,
    /// Inverse sine, in radians.
    asin,
    /// Inverse cosine, in radians.
    acos,
    /// Natural exponential.
    exp,
    /// Base-two exponential.
    exp2,
    /// Natural logarithm.
    log,
    /// Base-two logarithm.
    log2,
    /// Base-ten logarithm.
    log10,
    /// First input raised to the power of the second input.
    ///
    /// `pow` fits most naturally the math extension, but `fpow` is retained in the
    /// float extension to avoid unexpected breakages.
    pow,
    /// Remainder of the first input divided by the second, truncating toward zero.
    fmod,
}

impl MathOps {
    fn arity(self) -> (usize, usize) {
        match self {
            Self::pow | Self::fmod | Self::atan2 => (2, 1),
            _ => (1, 1),
        }
    }
}

impl MakeOpDef for MathOps {
    fn opdef_id(&self) -> OpName {
        <&Self as Into<&'static str>>::into(self).into()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError> {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
        let (inputs, outputs) = self.arity();
        Signature::new(vec![float64_type(); inputs], vec![float64_type(); outputs]).into()
    }

    fn description(&self) -> String {
        format!("Double-precision {} math function", self.opdef_id())
    }
}

/// HUGR definitions for the math operations.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(|| {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        MathOps::load_all_ops(extension, extension_ref).unwrap();
    })
});

impl MakeRegisteredOp for MathOps {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID
    }

    fn extension_ref(&self) -> Arc<Extension> {
        EXTENSION.clone()
    }
}

/// Extend dataflow builders with double-precision math operations.
pub trait MathOpBuilder: Dataflow {
    /// Add a math operation and return its result wire.
    ///
    /// Inputs must be `float64` wires in the operation's argument order.
    /// In particular, `atan2` takes `(y, x)`, `pow` takes `(base, exponent)`,
    /// and `fmod` takes `(dividend, divisor)`.
    ///
    /// # Errors
    ///
    /// Returns a [`BuildError`] if the inputs cannot be connected to the operation.
    fn add_math_op(
        &mut self,
        op: MathOps,
        inputs: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        Ok(self.add_dataflow_op(op, inputs)?.out_wire(0))
    }
}

impl<D: Dataflow> MathOpBuilder for D {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        HugrView,
        builder::{DFGBuilder, DataflowHugr},
        extension::prelude::bool_t,
        ops::{OpTrait, OpType},
        std_extensions::STD_REG,
    };
    use rstest::rstest;

    #[rstest]
    #[case(MathOps::sin, 1)]
    #[case(MathOps::cos, 1)]
    #[case(MathOps::tan, 1)]
    #[case(MathOps::atan, 1)]
    #[case(MathOps::atan2, 2)]
    #[case(MathOps::asin, 1)]
    #[case(MathOps::acos, 1)]
    #[case(MathOps::exp, 1)]
    #[case(MathOps::exp2, 1)]
    #[case(MathOps::log, 1)]
    #[case(MathOps::log2, 1)]
    #[case(MathOps::log10, 1)]
    #[case(MathOps::pow, 2)]
    #[case(MathOps::fmod, 2)]
    fn build_math_op(#[case] op: MathOps, #[case] input_count: usize) {
        let signature = Signature::new(vec![float64_type(); input_count], [float64_type()]);
        let mut builder = DFGBuilder::new(signature.clone()).unwrap();
        let inputs: Vec<_> = builder.input_wires().collect();
        let output = builder.add_math_op(op, inputs.iter().copied()).unwrap();
        let hugr = builder.finish_hugr_with_outputs([output]).unwrap();
        hugr.validate().unwrap();

        let OpType::ExtensionOp(extension_op) = hugr.get_optype(output.node()) else {
            panic!("Expected math operation");
        };
        assert_eq!(
            extension_op.dataflow_signature().unwrap().as_ref(),
            &signature
        );
        assert_eq!(MathOps::from_op(extension_op).unwrap(), op);
        assert_eq!(
            MathOps::from_def(EXTENSION.get_op(&op.opdef_id()).unwrap()).unwrap(),
            op
        );
        assert!(Arc::ptr_eq(
            STD_REG.get(EXTENSION_ID.as_ref()).unwrap(),
            &EXTENSION
        ));
        for (port, input) in inputs.into_iter().enumerate() {
            assert_eq!(
                hugr.single_linked_output(output.node(), port),
                Some((input.node(), input.source()))
            );
        }
    }

    #[rstest]
    #[case::missing_input(MathOps::sin, vec![])]
    #[case::extra_input(MathOps::sin, vec![float64_type(); 2])]
    #[case::missing_binary_input(MathOps::atan2, vec![float64_type()])]
    #[case::wrong_type(MathOps::sin, vec![bool_t()])]
    fn invalid_inputs(#[case] op: MathOps, #[case] inputs: Vec<crate::types::Type>) {
        let mut builder = DFGBuilder::new(Signature::new(inputs, [float64_type()])).unwrap();
        let inputs: Vec<_> = builder.input_wires().collect();
        let result = builder
            .add_math_op(op, inputs)
            .and_then(|output| builder.finish_hugr_with_outputs([output]));
        assert!(result.is_err());
    }
}
