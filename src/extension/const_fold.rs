use std::fmt::Formatter;

use std::fmt::Debug;

use crate::types::TypeArg;

use crate::OutgoingPort;

use crate::ops;

pub type ConstFoldResult = Option<Vec<(OutgoingPort, ops::Const)>>;

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
