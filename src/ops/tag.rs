//! Tags for sets of operation kinds.

use std::{cmp, fmt::Display};
use strum_macros::FromRepr;

pub type TagRepr = u8;

/// Tags for sets of operation kinds.
///
/// This can mark either specific operations, or sets of operations allowed in
/// regions.
///
/// Uses a flat representation for all the variants, in contrast to the complex
/// `OpType` structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, FromRepr)]
#[repr(u8)]
#[non_exhaustive]
pub enum OpTag {
    /// All operations allowed.
    #[default]
    Any = 0,
    /// No valid operation types.
    None = 1,

    /// Non-root module operations.
    ModuleOp = 2,
    /// Root module operation.
    ModuleRoot = 3,
    /// A function definition or declaration.
    Function = 4,
    /// A type alias.
    Alias = 5,
    /// A constant declaration.
    Const = 6,
    /// A function definition.
    FuncDefn = 7,

    /// Node in a Control-flow Sibling Graph.
    ControlFlowChild = 8,
    /// Node in a Dataflow Sibling Graph.
    DataflowChild = 9,
    /// Parent node of a Dataflow Sibling Graph.
    DataflowParent = 10,

    /// A nested data-flow operation.
    Dfg = 11,
    /// A nested control-flow operation.
    Cfg = 12,
    /// A dataflow input.
    Input = 13,
    /// A dataflow output.
    Output = 14,
    /// A function call.
    FnCall = 15,
    /// A constant load operation.
    LoadConst = 16,
    /// A definition that could be at module level or inside a DSG.
    ScopedDefn = 17,
    /// A tail-recursive loop.
    TailLoop = 18,
    /// A conditional operation.
    Conditional = 19,
    /// A case op inside a conditional.
    Case = 20,
    /// A leaf operation.
    Leaf = 21,

    /// A control flow basic block.
    BasicBlock = 22,
    /// A control flow exit node.
    BasicBlockExit = 23,
}

impl OpTag {
    /// Returns true if the tag is more general than the given tag.
    #[inline]
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

    #[inline]
    pub const fn repr(&self) -> TagRepr {
        *self as u8
    }

    /// Returns the infimum of the set of tags that strictly contain this tag
    ///
    /// Tags are sets of operations. The parent_tags of T define the immediate
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
            OpTag::Function => &[OpTag::ModuleOp],
            OpTag::Alias => &[OpTag::ScopedDefn],
            OpTag::FuncDefn => &[OpTag::Function, OpTag::ScopedDefn, OpTag::DataflowParent],
            OpTag::BasicBlock => &[OpTag::ControlFlowChild, OpTag::DataflowParent],
            OpTag::BasicBlockExit => &[OpTag::BasicBlock],
            OpTag::Case => &[OpTag::Any, OpTag::DataflowParent],
            OpTag::ModuleRoot => &[OpTag::Any],
            OpTag::Const => &[OpTag::ScopedDefn],
            OpTag::Dfg => &[OpTag::DataflowChild, OpTag::DataflowParent],
            OpTag::Cfg => &[OpTag::DataflowChild],
            OpTag::ScopedDefn => &[
                OpTag::DataflowChild,
                OpTag::ControlFlowChild,
                OpTag::ModuleOp,
            ],
            OpTag::TailLoop => &[OpTag::DataflowChild, OpTag::DataflowParent],
            OpTag::Conditional => &[OpTag::DataflowChild],
            OpTag::FnCall => &[OpTag::DataflowChild],
            OpTag::LoadConst => &[OpTag::DataflowChild],
            OpTag::Leaf => &[OpTag::DataflowChild],
            OpTag::DataflowParent => &[OpTag::Any],
        }
    }

    /// Returns a user-friendly description of the set.
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
            OpTag::ScopedDefn => "Definitions that can live at global or local scope",
            OpTag::DataflowParent => "Operation whose children form a Dataflow Sibling Graph",
        }
    }

    /// Returns whether the set is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        matches!(self, &OpTag::None)
    }

    /// Constant equality check.
    #[inline]
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
    use crate::Node;

    use super::*;
    pub struct NewHandle<const C: TagRepr>(Node);

    impl<const C: TagRepr> std::fmt::Debug for NewHandle<C> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_tuple(&format!("NewHandle<{:?}>", OpTag::from_repr(C).unwrap()))
                .field(&self.0)
                .finish()
        }
    }

    #[test]
    fn tag_contains() {
        assert!(OpTag::Any.is_superset(OpTag::Any));
        assert!(OpTag::None.is_superset(OpTag::None));
        assert!(OpTag::ModuleOp.is_superset(OpTag::ModuleOp));
        assert!(OpTag::DataflowChild.is_superset(OpTag::DataflowChild));
        assert!(OpTag::BasicBlock.is_superset(OpTag::BasicBlock));

        assert!(OpTag::Any.is_superset(OpTag::None));
        assert!(OpTag::Any.is_superset(OpTag::ModuleOp));
        assert!(OpTag::Any.is_superset(OpTag::DataflowChild));
        assert!(OpTag::Any.is_superset(OpTag::BasicBlock));

        assert!(!OpTag::None.is_superset(OpTag::Any));
        assert!(!OpTag::None.is_superset(OpTag::ModuleOp));
        assert!(!OpTag::None.is_superset(OpTag::DataflowChild));
        assert!(!OpTag::None.is_superset(OpTag::BasicBlock));

        const c: TagRepr = OpTag::ModuleRoot.repr();
        let node = Node::from(portgraph::NodeIndex::new(1));
        let h: NewHandle<c> = NewHandle(node);

        assert_eq!(
            format!("{:?}", h),
            "NewHandle<ModuleRoot>(Node { index: NodeIndex(1) })"
        )
    }
}
