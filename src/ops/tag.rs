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
    /// A function definition or declaration.
    Function,
    /// A type alias.
    Alias,
    /// A constant declaration.
    Const,
    /// A function definition.
    FuncDef,

    /// Node in a Dataflow Sibling Graph.
    DataflowChild,
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
    /// Operations taking const inputs.
    ConstInput,
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
            OpTag::DataflowChild => &[OpTag::Any],
            OpTag::Input => &[OpTag::DataflowChild],
            OpTag::Output => &[OpTag::DataflowChild],
            OpTag::Function => &[OpTag::ModuleOp],
            OpTag::Alias => &[OpTag::ModuleOp],
            OpTag::FuncDef => &[OpTag::Function, OpTag::DataflowChild],
            OpTag::BasicBlock => &[OpTag::Any],
            OpTag::BasicBlockExit => &[OpTag::BasicBlock],
            OpTag::Case => &[OpTag::Any],
            OpTag::ModuleRoot => &[OpTag::Any],
            OpTag::Const => &[OpTag::ModuleOp, OpTag::DataflowChild],
            OpTag::Dfg => &[OpTag::DataflowChild],
            OpTag::Cfg => &[OpTag::DataflowChild],
            OpTag::ConstInput => &[OpTag::DataflowChild],
            OpTag::TailLoop => &[OpTag::DataflowChild],
            OpTag::Conditional => &[OpTag::DataflowChild],
            OpTag::FnCall => &[OpTag::ConstInput],
            OpTag::LoadConst => &[OpTag::ConstInput],
            OpTag::Leaf => &[OpTag::DataflowChild],
        }
    }

    /// Returns a user-friendly description of the set.
    pub fn description(&self) -> &str {
        match self {
            OpTag::Any => "Any",
            OpTag::None => "None",
            OpTag::ModuleOp => "Module operations",
            OpTag::DataflowChild => "Node in a Dataflow Sibling Graph",
            OpTag::Input => "Input node",
            OpTag::Output => "Output node",
            OpTag::FuncDef => "Function definition",
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
            OpTag::ConstInput => "Dataflow operations that take a Const input.",
        }
    }

    /// Returns whether the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self == &OpTag::None
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
        assert!(OpTag::DataflowChild.contains(OpTag::DataflowChild));
        assert!(OpTag::BasicBlock.contains(OpTag::BasicBlock));

        assert!(OpTag::Any.contains(OpTag::None));
        assert!(OpTag::Any.contains(OpTag::ModuleOp));
        assert!(OpTag::Any.contains(OpTag::DataflowChild));
        assert!(OpTag::Any.contains(OpTag::BasicBlock));

        assert!(!OpTag::None.contains(OpTag::Any));
        assert!(!OpTag::None.contains(OpTag::ModuleOp));
        assert!(!OpTag::None.contains(OpTag::DataflowChild));
        assert!(!OpTag::None.contains(OpTag::BasicBlock));
    }
}
