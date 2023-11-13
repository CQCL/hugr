//! Module-level operations

use smol_str::SmolStr;

use crate::types::{EdgeKind, PolyFuncType};
use crate::types::{Type, TypeBound};

use super::StaticTag;
use super::{impl_op_name, OpTag, OpTrait};

/// The root of a module, parent of all other `OpType`s.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Module;

impl_op_name!(Module);

impl StaticTag for Module {
    const TAG: OpTag = OpTag::ModuleRoot;
}

impl OpTrait for Module {
    fn description(&self) -> &str {
        "The root of a module, parent of all other `OpType`s"
    }

    fn tag(&self) -> super::OpTag {
        <Self as StaticTag>::TAG
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
    pub signature: PolyFuncType,
}

impl_op_name!(FuncDefn);
impl StaticTag for FuncDefn {
    const TAG: OpTag = OpTag::FuncDefn;
}
impl OpTrait for FuncDefn {
    fn description(&self) -> &str {
        "A function definition"
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Static(Type::new_function(self.signature.clone())))
    }
}

/// External function declaration, linked at runtime.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FuncDecl {
    /// Name of function
    pub name: String,
    /// Signature of the function
    pub signature: PolyFuncType,
}

impl_op_name!(FuncDecl);
impl StaticTag for FuncDecl {
    const TAG: OpTag = OpTag::Function;
}
impl OpTrait for FuncDecl {
    fn description(&self) -> &str {
        "External function declaration, linked at runtime"
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Static(Type::new_function(self.signature.clone())))
    }
}

/// A type alias definition, used only for debug/metadata.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AliasDefn {
    /// Alias name
    pub name: SmolStr,
    /// Aliased type
    pub definition: Type,
}
impl_op_name!(AliasDefn);
impl StaticTag for AliasDefn {
    const TAG: OpTag = OpTag::Alias;
}
impl OpTrait for AliasDefn {
    fn description(&self) -> &str {
        "A type alias definition"
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }
}

/// A type alias declaration. Resolved at link time.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AliasDecl {
    /// Alias name
    pub name: SmolStr,
    /// Flag to signify type is classical
    pub bound: TypeBound,
}

impl AliasDecl {
    /// Construct a new Alias declaration.
    pub fn new(name: impl Into<SmolStr>, bound: TypeBound) -> Self {
        Self {
            name: name.into(),
            bound,
        }
    }

    /// Returns a reference to the name of this [`AliasDecl`].
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }
}

impl_op_name!(AliasDecl);
impl StaticTag for AliasDecl {
    const TAG: OpTag = OpTag::Alias;
}
impl OpTrait for AliasDecl {
    fn description(&self) -> &str {
        "A type alias declaration"
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }
}
