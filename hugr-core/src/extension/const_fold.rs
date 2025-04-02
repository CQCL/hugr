use std::fmt::Formatter;

use std::fmt::Debug;

use crate::ops::Value;
use crate::types::TypeArg;

use crate::IncomingPort;
use crate::OutgoingPort;

use crate::ops;

/// Output of constant folding an operation, None indicates folding was either
/// not possible or unsuccessful. An empty vector indicates folding was
/// successful and no values are output.
pub type ConstFoldResult = Option<Vec<(OutgoingPort, ops::Value)>>;

/// Tag some output constants with [`OutgoingPort`] inferred from the ordering.
pub fn fold_out_row(consts: impl IntoIterator<Item = Value>) -> ConstFoldResult {
    let vec = consts
        .into_iter()
        .enumerate()
        .map(|(i, c)| (i.into(), c))
        .collect();
    Some(vec)
}

/// Trait implemented by extension operations that can perform constant folding.
pub trait ConstFold: Send + Sync {
    /// Given type arguments `type_args` and
    /// [`crate::ops::Const`] values for inputs at [`crate::IncomingPort`]s,
    /// try to evaluate the operation.
    fn fold(
        &self,
        type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, crate::ops::Value)],
    ) -> ConstFoldResult;
}

impl Debug for Box<dyn ConstFold> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "<custom constant folding>")
    }
}

/// Blanket implementation for functions that only require the constants to
/// evaluate - type arguments are not relevant.
impl<T> ConstFold for T
where
    T: Fn(&[(crate::IncomingPort, crate::ops::Value)]) -> ConstFoldResult + Send + Sync,
{
    fn fold(
        &self,
        _type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, crate::ops::Value)],
    ) -> ConstFoldResult {
        self(consts)
    }
}

type FoldFn = dyn Fn(&[TypeArg], &[(IncomingPort, Value)]) -> ConstFoldResult + Send + Sync;

/// Type holding a boxed const-folding function.
pub struct Folder {
    /// Const-folding function.
    pub folder: Box<FoldFn>,
}

impl ConstFold for Folder {
    fn fold(&self, type_args: &[TypeArg], consts: &[(IncomingPort, Value)]) -> ConstFoldResult {
        (self.folder)(type_args, consts)
    }
}
