//! Tags for sets of operation kinds.

use std::{cmp, fmt::Display};

/// Tags for sets of operation kinds.
///
/// This can mark either specific operations, or sets of operations allowed in
/// regions.
///
/// Uses a flat representation for all the variants, in contrast to the complex
/// `OpType` structures.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Default)]
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
    FuncDefn,

    /// Node in a Control-flow Sibling Graph.
    ControlFlowChild,
    /// Node in a Dataflow Sibling Graph.
    DataflowChild,
    /// Parent node of a Dataflow Sibling Graph.
    DataflowParent,

    /// A nested data-flow operation.
    Dfg,
    /// A nested control-flow operation.
    Cfg,
    /// A dataflow input.
    Input,
    /// A dataflow output.
    Output,
    /// Dataflow node that has a static input
    StaticInput,
    /// Node that has a static output
    StaticOutput,
    /// A function call.
    FnCall,
    /// A constant load operation.
    LoadConst,
    /// A function load operation.
    LoadFunc,
    /// A definition that could be at module level or inside a DSG.
    /// Note that this means only Constants, as all other defn/decls
    /// must be at Module level.
    ScopedDefn,
    /// A tail-recursive loop.
    TailLoop,
    /// A conditional operation.
    Conditional,
    /// A case op inside a conditional.
    Case,
    /// A leaf operation.
    Leaf,

    /// A control flow basic block defining a dataflow graph.
    DataflowBlock,
    /// A control flow exit node.
    BasicBlockExit,
}

impl OpTag {
    /// Returns true if this tag is more general than `other`.
    #[inline]
    #[must_use]
    pub const fn is_superset(self, other: OpTag) -> bool {
        // We cannot call iter().any() or even do for loops in const fn yet.
        // So we have to write this ugly code.
        if self.eq(other) {
            return true;
        }
        let parents = other.immediate_supersets();
        let mut i = 0;
        while i < parents.len() {
            if self.is_superset(parents[i]) {
                return true;
            }
            i += 1;
        }
        false
    }

    /// Returns the infimum of the set of tags that strictly contain this tag
    ///
    /// Tags are sets of operations. The `parent_tags` of T define the immediate
    /// supersets of T. In mathematical terms:
    /// ```text
    /// R ∈ parent_tags(T) if R ⊃ T and ∄ Q st. R ⊃ Q ⊃ T .
    /// ```
    #[inline]
    const fn immediate_supersets<'a>(self) -> &'a [OpTag] {
        match self {
            OpTag::Any => &[],
            OpTag::None => &[OpTag::Any],
            OpTag::ModuleOp => &[OpTag::Any],
            OpTag::ControlFlowChild => &[OpTag::Any],
            OpTag::DataflowChild => &[OpTag::Any],
            OpTag::Input => &[OpTag::DataflowChild],
            OpTag::Output => &[OpTag::DataflowChild],
            OpTag::Function => &[OpTag::ModuleOp, OpTag::StaticOutput],
            OpTag::Alias => &[OpTag::ModuleOp],
            OpTag::FuncDefn => &[OpTag::Function, OpTag::DataflowParent],
            OpTag::DataflowBlock => &[OpTag::ControlFlowChild, OpTag::DataflowParent],
            OpTag::BasicBlockExit => &[OpTag::ControlFlowChild],
            OpTag::Case => &[OpTag::Any, OpTag::DataflowParent],
            OpTag::ModuleRoot => &[OpTag::Any],
            OpTag::Const => &[OpTag::ScopedDefn, OpTag::StaticOutput],
            OpTag::Dfg => &[OpTag::DataflowChild, OpTag::DataflowParent],
            OpTag::Cfg => &[OpTag::DataflowChild],
            OpTag::ScopedDefn => &[
                OpTag::DataflowChild,
                OpTag::ControlFlowChild,
                OpTag::ModuleOp,
            ],
            OpTag::TailLoop => &[OpTag::DataflowChild, OpTag::DataflowParent],
            OpTag::Conditional => &[OpTag::DataflowChild],
            OpTag::StaticInput => &[OpTag::Any],
            OpTag::StaticOutput => &[OpTag::Any],
            OpTag::FnCall => &[OpTag::StaticInput, OpTag::DataflowChild],
            OpTag::LoadConst => &[OpTag::StaticInput, OpTag::DataflowChild],
            OpTag::LoadFunc => &[OpTag::StaticInput, OpTag::DataflowChild],
            OpTag::Leaf => &[OpTag::DataflowChild],
            OpTag::DataflowParent => &[OpTag::Any],
        }
    }

    /// Returns a user-friendly description of the set.
    #[must_use]
    pub const fn description(&self) -> &str {
        match self {
            OpTag::Any => "Any",
            OpTag::None => "None",
            OpTag::ModuleOp => "Module operations",
            OpTag::ControlFlowChild => "Node in a Controlflow Sibling Graph",
            OpTag::DataflowChild => "Node in a Dataflow Sibling Graph",
            OpTag::Input => "Input node",
            OpTag::Output => "Output node",
            OpTag::FuncDefn => "Function definition",
            OpTag::DataflowBlock => "Basic block containing a dataflow graph",
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
            OpTag::StaticInput => "Node with static input (LoadConst, LoadFunc, or FnCall)",
            OpTag::StaticOutput => "Node with static output (FuncDefn, FuncDecl, Const)",
            OpTag::FnCall => "Function call",
            OpTag::LoadConst => "Constant load operation",
            OpTag::LoadFunc => "Function load operation",
            OpTag::Leaf => "Leaf operation",
            OpTag::ScopedDefn => "Definitions that can live at global or local scope",
            OpTag::DataflowParent => "Operation whose children form a Dataflow Sibling Graph",
        }
    }

    /// Returns whether the set is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, &OpTag::None)
    }

    /// Constant equality check.
    #[inline]
    #[must_use]
    pub const fn eq(self, other: OpTag) -> bool {
        self as u32 == other as u32
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
        } else if self.is_superset(*other) {
            Some(cmp::Ordering::Greater)
        } else if other.is_superset(*self) {
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
        assert!(OpTag::Any.is_superset(OpTag::Any));
        assert!(OpTag::None.is_superset(OpTag::None));
        assert!(OpTag::ModuleOp.is_superset(OpTag::ModuleOp));
        assert!(OpTag::DataflowChild.is_superset(OpTag::DataflowChild));
        assert!(OpTag::ControlFlowChild.is_superset(OpTag::ControlFlowChild));

        assert!(OpTag::Any.is_superset(OpTag::None));
        assert!(OpTag::Any.is_superset(OpTag::ModuleOp));
        assert!(OpTag::Any.is_superset(OpTag::DataflowChild));
        assert!(OpTag::Any.is_superset(OpTag::ControlFlowChild));

        assert!(!OpTag::None.is_superset(OpTag::Any));
        assert!(!OpTag::None.is_superset(OpTag::ModuleOp));
        assert!(!OpTag::None.is_superset(OpTag::DataflowChild));
        assert!(!OpTag::None.is_superset(OpTag::ControlFlowChild));

        // Other specific checks
        assert!(!OpTag::DataflowParent.is_superset(OpTag::BasicBlockExit));
        assert!(!OpTag::DataflowParent.is_superset(OpTag::Cfg));
    }
}
