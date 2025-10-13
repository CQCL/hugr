use std::collections::HashMap;

use crate::CommitId;

/// A cache for storing computed properties of a `PersistentHugr`.
#[derive(Debug, Default, Clone)]
pub(super) struct PersistentHugrCache {
    children_cache: HashMap<CommitId, Vec<CommitId>>,
}

impl PersistentHugrCache {
    /// Remove the cached list of children of `commit`.
    ///
    /// This should be called whenever the children of `commit` may have
    /// changed.
    pub fn invalidate_children(&mut self, commit: CommitId) {
        self.children_cache.remove(&commit);
    }

    /// Get the cached list of children of `commit`, or compute and cache them
    /// using the callable.
    pub fn children_or_insert(
        &mut self,
        commit: CommitId,
        children: impl FnOnce() -> Vec<CommitId>,
    ) -> &Vec<CommitId> {
        self.children_cache.entry(commit).or_insert_with(children)
    }
}
