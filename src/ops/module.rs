//! Module-level operations

use smol_str::SmolStr;

use crate::types::{ClassicType, EdgeKind, Signature, SimpleType};

use super::OpTagged;
use super::{impl_op_name, OpTag, OpTrait};

/// The root of a module, parent of all other `OpType`s.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Module;

impl_op_name!(Module);

impl OpTagged for Module {
    fn static_tag() -> OpTag {
        OpTag::ModuleRoot
    }
}

impl OpTrait for Module {
    fn description(&self) -> &str {
        "The root of a module, parent of all other `OpType`s"
    }

    fn tag(&self) -> super::OpTag {
        <Self as OpTagged>::static_tag()
    }
}

/// A function definition.
///
/// Children nodes are the body of the definition.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FuncDefn {
    /// Name of function
    pub name: String,
    /// Signature of the function
    pub signature: Signature,
}

impl_op_name!(FuncDefn);
impl OpTagged for FuncDefn {
    fn static_tag() -> OpTag {
        OpTag::FuncDefn
    }
}
impl OpTrait for FuncDefn {
    fn description(&self) -> &str {
        "A function definition"
    }

    fn tag(&self) -> OpTag {
        <Self as OpTagged>::static_tag()
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Static(ClassicType::graph_from_sig(
            self.signature.clone(),
        )))
    }
}

/// External function declaration, linked at runtime.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FuncDecl {
    /// Name of function
    pub name: String,
    /// Signature of the function
    pub signature: Signature,
}

impl_op_name!(FuncDecl);
impl OpTagged for FuncDecl {
    fn static_tag() -> OpTag {
        OpTag::Function
    }
}
impl OpTrait for FuncDecl {
    fn description(&self) -> &str {
        "External function declaration, linked at runtime"
    }

    fn tag(&self) -> OpTag {
        <Self as OpTagged>::static_tag()
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Static(ClassicType::graph_from_sig(
            self.signature.clone(),
        )))
    }
}

/// A type alias definition, used only for debug/metadata.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AliasDefn {
    /// Alias name
    pub name: SmolStr,
    /// Aliased type
    pub definition: SimpleType,
}
impl_op_name!(AliasDefn);
impl OpTagged for AliasDefn {
    fn static_tag() -> OpTag {
        OpTag::Alias
    }
}
impl OpTrait for AliasDefn {
    fn description(&self) -> &str {
        "A type alias definition"
    }

    fn tag(&self) -> OpTag {
        <Self as OpTagged>::static_tag()
    }
}

/// A type alias declaration. Resolved at link time.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AliasDecl {
    /// Alias name
    pub name: SmolStr,
    /// Flag to signify type is linear
    pub linear: bool,
}

impl_op_name!(AliasDecl);
impl OpTagged for AliasDecl {
    fn static_tag() -> OpTag {
        OpTag::Alias
    }
}
impl OpTrait for AliasDecl {
    fn description(&self) -> &str {
        "A type alias declaration"
    }

    fn tag(&self) -> OpTag {
        <Self as OpTagged>::static_tag()
    }
}
