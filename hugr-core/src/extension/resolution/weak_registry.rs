use std::collections::BTreeMap;
use std::sync::{Arc, Weak};

use itertools::Itertools;

use derive_more::Display;

use crate::Extension;
use crate::extension::{ExtensionId, ExtensionRegistry};

/// The equivalent to an [`ExtensionRegistry`] that only contains weak
/// references.
///
/// This is used to resolve extensions pointers while the extensions themselves
/// (and the [`Arc`] that contains them) are being initialized.
#[derive(Debug, Display, Default, Clone)]
#[display("WeakExtensionRegistry[{}]", exts.keys().join(", "))]
pub struct WeakExtensionRegistry {
    /// The extensions in the registry.
    exts: BTreeMap<ExtensionId, Weak<Extension>>,
}

impl WeakExtensionRegistry {
    /// Create a new weak registry from a list of extensions and their ids.
    pub fn new(extensions: impl IntoIterator<Item = (ExtensionId, Weak<Extension>)>) -> Self {
        let mut res = Self::default();
        for (id, ext) in extensions {
            res.register(id, ext);
        }
        res
    }

    /// Gets the Extension with the given name
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Weak<Extension>> {
        self.exts.get(name)
    }

    /// Returns `true` if the registry contains an extension with the given name.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.exts.contains_key(name)
    }

    /// Register a new extension in the registry.
    ///
    /// Returns `true` if the extension was added, `false` if it was already present.
    pub fn register(&mut self, id: ExtensionId, ext: impl Into<Weak<Extension>>) -> bool {
        self.exts.insert(id, ext.into()).is_none()
    }

    /// Returns an iterator over the weak references in the registry and their ids.
    pub fn iter(&self) -> impl Iterator<Item = (&ExtensionId, &Weak<Extension>)> {
        self.exts.iter()
    }

    /// Returns an iterator over the weak references in the registry.
    pub fn extensions(&self) -> impl Iterator<Item = &Weak<Extension>> {
        self.exts.values()
    }

    /// Returns an iterator over the extension ids in the registry.
    pub fn ids(&self) -> impl Iterator<Item = &ExtensionId> {
        self.exts.keys()
    }
}

impl IntoIterator for WeakExtensionRegistry {
    type Item = Weak<Extension>;
    type IntoIter = std::collections::btree_map::IntoValues<ExtensionId, Weak<Extension>>;

    fn into_iter(self) -> Self::IntoIter {
        self.exts.into_values()
    }
}

impl<'a> TryFrom<&'a WeakExtensionRegistry> for ExtensionRegistry {
    type Error = ();

    fn try_from(weak: &'a WeakExtensionRegistry) -> Result<Self, Self::Error> {
        let exts: Vec<Arc<Extension>> = weak
            .extensions()
            .map(|w| w.upgrade().ok_or(()))
            .try_collect()?;
        Ok(ExtensionRegistry::new(exts))
    }
}

impl TryFrom<WeakExtensionRegistry> for ExtensionRegistry {
    type Error = ();

    fn try_from(weak: WeakExtensionRegistry) -> Result<Self, Self::Error> {
        let exts: Vec<Arc<Extension>> = weak
            .into_iter()
            .map(|w| w.upgrade().ok_or(()))
            .try_collect()?;
        Ok(ExtensionRegistry::new(exts))
    }
}

impl<'a> From<&'a ExtensionRegistry> for WeakExtensionRegistry {
    fn from(reg: &'a ExtensionRegistry) -> Self {
        let exts = reg
            .iter()
            .map(|ext| (ext.name().clone(), Arc::downgrade(ext)))
            .collect();
        Self { exts }
    }
}
