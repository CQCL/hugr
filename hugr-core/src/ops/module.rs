//! Module-level operations

use std::borrow::Cow;

use smol_str::SmolStr;
#[cfg(test)]
use {
    crate::proptest::{any_nonempty_smolstr, any_nonempty_string},
    ::proptest_derive::Arbitrary,
};

use crate::Visibility;
use crate::types::{EdgeKind, PolyFuncType, Signature, Type, TypeBound};

use super::dataflow::DataflowParent;
use super::{OpTag, OpTrait, StaticTag, impl_op_name};

/// The root of a module, parent of all other `OpType`s.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct Module {
    // can't be simple unit struct due to flattened serialization issues
    // see https://github.com/CQCL/hugr/issues/1270
}

impl Module {
    /// Construct a new Module.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }
}

impl_op_name!(Module);

impl StaticTag for Module {
    const TAG: OpTag = OpTag::ModuleRoot;
}

impl OpTrait for Module {
    fn description(&self) -> &'static str {
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
#[cfg_attr(test, derive(Arbitrary))]
pub struct FuncDefn {
    #[cfg_attr(test, proptest(strategy = "any_nonempty_string()"))]
    name: String,
    signature: PolyFuncType,
    #[serde(default = "priv_vis")] // sadly serde does not pick this up from the schema
    visibility: Visibility,
}

fn priv_vis() -> Visibility {
    Visibility::Private
}

impl FuncDefn {
    /// Create a new, [Visibility::Private], instance with the given name and signature.
    /// See also [Self::new_vis].
    pub fn new(name: impl Into<String>, signature: impl Into<PolyFuncType>) -> Self {
        Self::new_vis(name, signature, Visibility::Private)
    }

    /// Create a new instance with the specified name and visibility
    pub fn new_vis(
        name: impl Into<String>,
        signature: impl Into<PolyFuncType>,
        visibility: Visibility,
    ) -> Self {
        Self {
            name: name.into(),
            signature: signature.into(),
            visibility,
        }
    }

    /// The name of the function (not the name of the Op)
    pub fn func_name(&self) -> &String {
        &self.name
    }

    /// Allows mutating the name of the function (as per [Self::func_name])
    pub fn func_name_mut(&mut self) -> &mut String {
        &mut self.name
    }

    /// Gets the signature of the function
    pub fn signature(&self) -> &PolyFuncType {
        &self.signature
    }

    /// Allows mutating the signature of the function
    pub fn signature_mut(&mut self) -> &mut PolyFuncType {
        &mut self.signature
    }

    /// The visibility of the function, e.g. for linking
    pub fn visibility(&self) -> &Visibility {
        &self.visibility
    }

    /// Allows changing [Self::visibility]
    pub fn visibility_mut(&mut self) -> &mut Visibility {
        &mut self.visibility
    }
}

impl_op_name!(FuncDefn);
impl StaticTag for FuncDefn {
    const TAG: OpTag = OpTag::FuncDefn;
}

impl DataflowParent for FuncDefn {
    fn inner_signature(&self) -> Cow<'_, Signature> {
        Cow::Borrowed(self.signature.body())
    }
}

impl OpTrait for FuncDefn {
    fn description(&self) -> &'static str {
        "A function definition"
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    fn static_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Function(self.signature.clone()))
    }

    // Cannot refer to TypeArgs of enclosing Hugr (it binds its own), so no substitute()
}

/// External function declaration, linked at runtime.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct FuncDecl {
    #[cfg_attr(test, proptest(strategy = "any_nonempty_string()"))]
    name: String,
    signature: PolyFuncType,
    // (again) sadly serde does not pick this up from the schema
    #[serde(default = "pub_vis")] // Note opposite of FuncDefn
    visibility: Visibility,
}

fn pub_vis() -> Visibility {
    Visibility::Public
}

impl FuncDecl {
    /// Create a new [Visibility::Public] instance with the given name and signature.
    /// See also [Self::new_vis]
    pub fn new(name: impl Into<String>, signature: impl Into<PolyFuncType>) -> Self {
        Self::new_vis(name, signature, Visibility::Public)
    }

    /// Create a new instance with the given name, signature and visibility
    pub fn new_vis(
        name: impl Into<String>,
        signature: impl Into<PolyFuncType>,
        visibility: Visibility,
    ) -> Self {
        Self {
            name: name.into(),
            signature: signature.into(),
            visibility,
        }
    }

    /// The name of the function (not the name of the Op)
    pub fn func_name(&self) -> &String {
        &self.name
    }

    /// The visibility of the function, e.g. for linking
    pub fn visibility(&self) -> &Visibility {
        &self.visibility
    }

    /// Allows mutating the name of the function (as per [Self::func_name])
    pub fn func_name_mut(&mut self) -> &mut String {
        &mut self.name
    }

    /// Allows mutating the [Self::visibility] of the function
    pub fn visibility_mut(&mut self) -> &mut Visibility {
        &mut self.visibility
    }

    /// Gets the signature of the function
    pub fn signature(&self) -> &PolyFuncType {
        &self.signature
    }

    /// Allows mutating the signature of the function
    pub fn signature_mut(&mut self) -> &mut PolyFuncType {
        &mut self.signature
    }
}

impl_op_name!(FuncDecl);
impl StaticTag for FuncDecl {
    const TAG: OpTag = OpTag::Function;
}

impl OpTrait for FuncDecl {
    fn description(&self) -> &'static str {
        "External function declaration, linked at runtime"
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    fn static_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Function(self.signature.clone()))
    }

    // Cannot refer to TypeArgs of enclosing Hugr (the type binds its own), so no substitute()
}

/// A type alias definition, used only for debug/metadata.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct AliasDefn {
    /// Alias name
    #[cfg_attr(test, proptest(strategy = "any_nonempty_smolstr()"))]
    pub name: SmolStr,
    /// Aliased type
    pub definition: Type,
}
impl_op_name!(AliasDefn);
impl StaticTag for AliasDefn {
    const TAG: OpTag = OpTag::Alias;
}
impl OpTrait for AliasDefn {
    fn description(&self) -> &'static str {
        "A type alias definition"
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    // Cannot refer to TypeArgs of enclosing Hugr (? - we planned to make this
    // polymorphic so it binds its own, and we never combine binders), so no substitute()
}

/// A type alias declaration. Resolved at link time.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct AliasDecl {
    /// Alias name
    #[cfg_attr(test, proptest(strategy = "any_nonempty_smolstr()"))]
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
    #[must_use]
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }
}

impl_op_name!(AliasDecl);
impl StaticTag for AliasDecl {
    const TAG: OpTag = OpTag::Alias;
}
impl OpTrait for AliasDecl {
    fn description(&self) -> &'static str {
        "A type alias declaration"
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    // Cannot refer to TypeArgs of enclosing Hugr (? - we planned to make this
    // polymorphic so it binds its own, and we never combine binders), so no substitute()
}
