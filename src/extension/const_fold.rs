use std::fmt::Formatter;

use std::fmt::Debug;

use crate::types::TypeArg;

use crate::OutgoingPort;

use crate::ops;

/// Output of constant folding an operation, None indicates folding was either
/// not possible or unsuccessful. An empty vector indicates folding was
/// successful and no values are output.
pub type ConstFoldResult = Option<Vec<(OutgoingPort, ops::Const)>>;

/// Trait implemented by extension operations that can perform constant folding.
pub trait ConstFold: Send + Sync {
    /// Given type arguments `type_args` and
    /// [`crate::ops::Const`] values for inputs at [`crate::IncomingPort`]s,
    /// try to evaluate the operation.
    fn fold(
        &self,
        type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, crate::ops::Const)],
    ) -> ConstFoldResult;
}

impl Debug for Box<dyn ConstFold> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "<custom constant folding>")
    }
}

impl Default for Box<dyn ConstFold> {
    fn default() -> Self {
        Box::new(|&_: &_| None)
    }
}

/// Blanket implementation for functions that only require the constants to
/// evaluate - type arguments are not relevant.
impl<T> ConstFold for T
where
    T: Fn(&[(crate::IncomingPort, crate::ops::Const)]) -> ConstFoldResult + Send + Sync,
{
    fn fold(
        &self,
        _type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, crate::ops::Const)],
    ) -> ConstFoldResult {
        self(consts)
    }
}
