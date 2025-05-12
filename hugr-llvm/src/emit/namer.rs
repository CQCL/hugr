use hugr_core::{Node, NodeIndex as _};

/// A type with features for mangling the naming of symbols.
#[derive(Clone)]
pub struct Namer {
    prefix: String,
    postfix_node: bool,
}

impl Namer {
    /// The [Default] impl of `Namer` uses this as a prefix.
    pub const DEFAULT_PREFIX: &'static str = "_hl.";

    /// Create a new `Namer` that for each symbol:
    /// * prefixes `prefix`
    /// * if `post_fix_node` is true, postfixes ".{`node.index()`}"
    ///
    /// # Example
    ///
    /// ```
    /// use hugr_llvm::emit::Namer;
    /// use hugr_core::Node;
    /// let node = Node::from(portgraph::NodeIndex::new(7));
    /// let namer = Namer::default();
    /// assert_eq!(namer.name_func("name", node), "_hl.name.7");
    ///
    /// let namer = Namer::new("prefix.", false);
    /// assert_eq!(namer.name_func("name", node), "prefix.name")
    /// ```
    pub fn new(prefix: impl Into<String>, postfix_node: bool) -> Self {
        Self {
            prefix: prefix.into(),
            postfix_node,
        }
    }

    /// Mangle the the name of a [`hugr_core::ops::FuncDefn`] or [`hugr_core::ops::FuncDecl`].
    pub fn name_func(&self, name: impl AsRef<str>, node: Node) -> String {
        let prefix = &self.prefix;
        let name = name.as_ref();
        let postfix = if self.postfix_node {
            format!(".{}", node.index())
        } else {
            String::new()
        };
        format!("{prefix}{name}{postfix}")
    }
}

impl Default for Namer {
    fn default() -> Self {
        Self::new(Self::DEFAULT_PREFIX, true)
    }
}
