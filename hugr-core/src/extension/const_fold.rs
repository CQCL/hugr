use std::fmt::Formatter;

use std::fmt::Debug;

use crate::ops::Value;
use crate::ops::constant::{OpaqueValue, Sum};
use crate::types::{SumType, TypeArg};
use crate::{IncomingPort, Node, OutgoingPort, PortIndex};

/// Representation of values used for constant folding.
/// See [ConstFold], which is used as `dyn` so we cannot parametrize by [HugrNode]
// Should we be non-exhaustive??
#[derive(Clone, PartialEq, Default)]
pub enum FoldVal {
    /// Value is unknown, must assume that it could be anything
    #[default]
    Unknown,
    /// A variant of a [SumType]
    Sum {
        /// Which variant of the sum type this value is.
        tag: usize,
        /// Describes the type of the whole value.
        // Can we deprecate this immediately? It is only for converting to Value
        sum_type: SumType,
        /// A value for each element (type) within the variant
        items: Vec<FoldVal>,
    },
    /// A constant value defined by an extension
    Extension(OpaqueValue),
    /// A function pointer loaded from a [FuncDefn](crate::ops::FuncDefn) or `FuncDecl`
    LoadedFunction(Node, Vec<TypeArg>), // Deliberately skipping Function(Box<Hugr>) ATM
}

impl TryFrom<FoldVal> for Value {
    type Error = Option<Node>;

    fn try_from(value: FoldVal) -> Result<Self, Self::Error> {
        match value {
            FoldVal::Unknown => Err(None),
            FoldVal::Sum {
                tag,
                sum_type,
                items,
            } => {
                let values = items
                    .into_iter()
                    .map(Value::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Value::Sum(Sum {
                    tag,
                    values,
                    sum_type,
                }))
            }
            FoldVal::Extension(e) => Ok(Value::Extension { e }),
            FoldVal::LoadedFunction(node, _) => Err(Some(node)),
        }
    }
}

impl From<Value> for FoldVal {
    fn from(value: Value) -> Self {
        match value {
            Value::Extension { e } => FoldVal::Extension(e),
            Value::Function { .. } => FoldVal::Unknown,
            Value::Sum(Sum {
                tag,
                values,
                sum_type,
            }) => {
                let items = values.into_iter().map(FoldVal::from).collect();
                FoldVal::Sum {
                    tag,
                    sum_type,
                    items,
                }
            }
        }
    }
}

/// Output of constant folding an operation, None indicates folding was either
/// not possible or unsuccessful. An empty vector indicates folding was
/// successful and no values are output.
pub type ConstFoldResult = Option<Vec<(OutgoingPort, Value)>>;

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
    /// Given type arguments `type_args` and [`FoldVal`]s for each input,
    /// update the outputs (these will be initialized to [FoldVal::Unknown]).
    ///
    /// Defaults to calling [Self::fold] with those arguments that can be converted ---
    /// [FoldVal::LoadedFunction]s will be lost as these are not representable as [Value]s.
    fn fold2(&self, type_args: &[TypeArg], inputs: &[FoldVal], outputs: &mut [FoldVal]) {
        let consts = inputs
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(p, fv)| Some((p.into(), fv.try_into().ok()?)))
            .collect::<Vec<_>>();
        let outs = self.fold(type_args, &consts);
        for (p, v) in outs.unwrap_or_default() {
            outputs[p.index()] = v.into();
        }
    }

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
