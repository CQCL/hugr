use std::fmt::Formatter;

use std::fmt::Debug;

use crate::types::TypeArg;

use crate::OutgoingPort;

use crate::IncomingPort;

use crate::ops;
use derive_more::From;

#[derive(From, Clone, PartialEq, Debug)]
pub enum FoldOutput {
    /// Value from port can be replaced with a constant
    Value(Box<ops::Const>),
    /// Value from port corresponds to one of the incoming values.
    Input(IncomingPort),
}

impl From<ops::Const> for FoldOutput {
    fn from(value: ops::Const) -> Self {
        Self::Value(Box::new(value))
    }
}

pub type ConstFoldResult = Option<Vec<(OutgoingPort, FoldOutput)>>;

pub trait ConstFold: Send + Sync {
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
