//! Tags for sets of operation kinds.

use std::{cmp, fmt::Display};

/// Tags for sets of operation kinds.
///
/// This can mark either specific operations, or sets of operations allowed in
/// regions.
///
/// Uses a flat representation for all the variants, in contrast to the complex
/// `OpType` structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OpTag {
    /// All operations allowed.
    #[default]
    Any,
    /// No valid operation types.
    None,

    /// Non-root module operations.
    ModuleOp,
    /// Root module operation.
    ModuleRoot,
    /// A function definition.
    Def,
    /// A function definition or declaration.
    Function,
    /// A type alias.
    Alias,
    /// A constant declaration.
    Const,

    /// Any dataflow operation.
    DataflowOp,
    /// A nested data-flow operation.
    Dfg,
    /// A nested control-flow operation.
    Cfg,
    /// A dataflow input.
    Input,
    /// A dataflow output.
    Output,
    /// A function call.
    FnCall,
    /// A constant load operation.
    LoadConst,
    /// A tail-recursive loop.
    TailLoop,
    /// A conditional operation.
    Conditional,
    /// A case op inside a conditional.
    Case,
    /// A leaf operation.
    Leaf,

    /// A control flow basic block.
    BasicBlock,
    /// A control flow exit node.
    BasicBlockExit,
}

impl OpTag {
    /// Returns true if the tag is more general than the given tag.
    #[inline]
    pub fn contains(self, other: OpTag) -> bool {
        self == other || other.parent_tags().iter().any(|&tag| self.contains(tag))
    }

    /// Returns the infimum of the set of tags that strictly contain this tag
    #[inline]
    fn parent_tags<'a>(self) -> &'a [OpTag] {
        match self {
            OpTag::Any => &[],
            OpTag::None => &[OpTag::Any],
            OpTag::ModuleOp => &[OpTag::Any],
            OpTag::DataflowOp => &[OpTag::Any],
            OpTag::Input => &[OpTag::DataflowOp],
            OpTag::Output => &[OpTag::DataflowOp],
            OpTag::Function => &[OpTag::ModuleOp],
            OpTag::Alias => &[OpTag::ModuleOp],
            OpTag::Def => &[OpTag::Function],
            OpTag::BasicBlock => &[OpTag::Any],
            OpTag::BasicBlockExit => &[OpTag::BasicBlock],
            OpTag::Case => &[OpTag::Any],
            OpTag::ModuleRoot => &[OpTag::Any],
            // Technically, this should be ModuleOp, but we will allow it outside modules soon.
            OpTag::Const => &[OpTag::ModuleOp, OpTag::DataflowOp],
            OpTag::Dfg => &[OpTag::DataflowOp],
            OpTag::Cfg => &[OpTag::DataflowOp],
            OpTag::TailLoop => &[OpTag::DataflowOp],
            OpTag::Conditional => &[OpTag::DataflowOp],
            OpTag::FnCall => &[OpTag::DataflowOp],
            OpTag::LoadConst => &[OpTag::DataflowOp],
            OpTag::Leaf => &[OpTag::DataflowOp],
        }
    }

    /// Returns a user-friendly description of the set.
    pub fn description(&self) -> &str {
        match self {
            OpTag::Any => "Any",
            OpTag::None => "None",
            OpTag::ModuleOp => "Module operations",
            OpTag::DataflowOp => "Dataflow operations",
            OpTag::Input => "Input node",
            OpTag::Output => "Output node",
            OpTag::Def => "Function definition",
            OpTag::BasicBlock => "Basic block",
            OpTag::BasicBlockExit => "Exit basic block node",
            OpTag::Case => "Case",
            OpTag::ModuleRoot => "Module root node",
            OpTag::Function => "Function definition or declaration",
            OpTag::Alias => "Type alias",
            OpTag::Const => "Constant declaration",
            OpTag::Dfg => "Nested data-flow operation",
            OpTag::Cfg => "Nested control-flow operation",
            OpTag::TailLoop => "Tail-recursive loop",
            OpTag::Conditional => "Conditional operation",
            OpTag::FnCall => "Function call",
            OpTag::LoadConst => "Constant load operation",
            OpTag::Leaf => "Leaf operation",
        }
    }

    /// Returns whether the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, OpTag::None)
    }
}

impl Display for OpTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

impl PartialOrd for OpTag {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        if self == other {
            Some(cmp::Ordering::Equal)
        } else if self.contains(*other) {
            Some(cmp::Ordering::Greater)
        } else if other.contains(*self) {
            Some(cmp::Ordering::Less)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn tag_contains() {
        assert!(OpTag::Any.contains(OpTag::Any));
        assert!(OpTag::None.contains(OpTag::None));
        assert!(OpTag::ModuleOp.contains(OpTag::ModuleOp));
        assert!(OpTag::DataflowOp.contains(OpTag::DataflowOp));
        assert!(OpTag::BasicBlock.contains(OpTag::BasicBlock));

        assert!(OpTag::Any.contains(OpTag::None));
        assert!(OpTag::Any.contains(OpTag::ModuleOp));
        assert!(OpTag::Any.contains(OpTag::DataflowOp));
        assert!(OpTag::Any.contains(OpTag::BasicBlock));

        assert!(!OpTag::None.contains(OpTag::Any));
        assert!(!OpTag::None.contains(OpTag::ModuleOp));
        assert!(!OpTag::None.contains(OpTag::DataflowOp));
        assert!(!OpTag::None.contains(OpTag::BasicBlock));
    }
}
