//! Store of commits that are refered to in [`PersistentHugr`]s.

use std::{
    cell::{Ref, RefCell},
    mem,
    rc::Rc,
};

use derive_more::From;
use hugr_core::{Hugr, Node};
use itertools::Itertools;
use relrc::Registry;

use crate::{Commit, InvalidCommit, PersistentHugr, PersistentReplacement};

pub mod serial;

/// A copyable handle to a [`Commit`] vertex within a [`CommitStateSpace`].
pub type CommitId = relrc::NodeId;

/// A HUGR node within a commit of the commit state space
#[derive(
    Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct PatchNode(pub CommitId, pub Node);

impl PatchNode {
    /// Get the commit ID of the commit that owns this node.
    pub fn owner(&self) -> CommitId {
        self.0
    }
}

// Print out PatchNodes as `Node(x)@commit_hex`
impl std::fmt::Debug for PatchNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}@{:?}", self.1, self.0)
    }
}

impl std::fmt::Display for PatchNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

mod hidden {
    use super::*;

    /// The data stored in a [`Commit`], either the base [`Hugr`] (on which all
    /// other commits apply), or a [`PersistentReplacement`]
    ///
    /// This is a "unnamable" type: we do not expose this struct publicly in our
    /// API, but we can still use it in public trait bounds (see
    /// [`Resolver`](crate::resolver::Resolver)).
    #[derive(Debug, Clone, From)]
    pub enum CommitData {
        Base(Hugr),
        Replacement(PersistentReplacement),
    }
}
pub(crate) use hidden::CommitData;

/// The set of all current commits, assigning every commit a unique ID.
///
/// A [`CommitStateSpace`] always has a unique base commit (the root of the
/// graph). All other commits are [`PersistentReplacement`]s that apply on top
/// of it. Commits are stored as [`relrc::RelRc`]s: they are reference-counted
/// pointers to the patch data that also maintain strong references to the
/// commit's parents. This means that commits can be cloned cheaply and dropped
/// freely; the memory of a commit will be released whenever no other commit in
/// scope depends on it.
///
/// Note that a [`CommitStateSpace`] only keeps weak references to commits, so
/// it is invalid to keep commit IDs beyond the lifetime of the commit. IDs will
/// be invalidated as soon as the commits are dropped.
///
/// Commits in a [`CommitStateSpace`] DO NOT represent a valid history in the
/// general case: pairs of commits may be mutually exclusive if they modify the
/// same subgraph. Use [`Self::try_create`] to get a [`PersistentHugr`]
/// with a set of compatible commits.
///
/// Cloning a [`CommitStateSpace`] value corresponds to creating a new handle to
/// the same underlying state space.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct CommitStateSpace {
    registry: Rc<RefCell<Registry<CommitData, ()>>>,
}

impl PartialEq for CommitStateSpace {
    fn eq(&self, other: &Self) -> bool {
        self.registry.as_ptr() == other.registry.as_ptr()
    }
}

impl Eq for CommitStateSpace {}

impl From<Registry<CommitData, ()>> for CommitStateSpace {
    fn from(registry: Registry<CommitData, ()>) -> Self {
        Self {
            registry: Rc::new(RefCell::new(registry)),
        }
    }
}

impl From<Rc<RefCell<Registry<CommitData, ()>>>> for CommitStateSpace {
    fn from(registry: Rc<RefCell<Registry<CommitData, ()>>>) -> Self {
        Self { registry }
    }
}

impl<'a> From<&'a Rc<RefCell<Registry<CommitData, ()>>>> for &'a CommitStateSpace {
    fn from(rc: &'a Rc<RefCell<Registry<CommitData, ()>>>) -> Self {
        // SAFETY: Commit is a transparent wrapper around the registry Rc
        unsafe { mem::transmute(rc) }
    }
}

impl Default for CommitStateSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl CommitStateSpace {
    /// Create a new empty commit state space.
    pub fn new() -> Self {
        let registry = Rc::new(RefCell::new(Registry::new()));
        Self { registry }
    }

    /// Set the base commit of the state space.
    ///
    /// This will only succeed if the state space is currently empty
    /// (otherwise a base commit already exists).
    pub fn try_set_base(&self, hugr: Hugr) -> Option<Commit<'_>> {
        if !self.registry.borrow().is_empty() {
            return None;
        }
        Some(Commit::new_base(hugr, self))
    }

    /// Check if `commit` is in the commit state space.
    pub fn contains(&self, commit: &Commit) -> bool {
        self.borrow().contains(commit.as_relrc())
    }

    /// Check if `commit_id` is in the commit state space.
    pub fn contains_id(&self, commit_id: CommitId) -> bool {
        self.borrow().contains_id(commit_id)
    }

    /// Get the ID of `commit` in the commit state space.
    pub fn get_id(&self, commit: &Commit) -> Option<CommitId> {
        self.borrow().get_id(commit.as_relrc())
    }

    /// A reverse lookup to obtain the commit from `commit_id`.
    ///
    /// Will return `None` if the commit_id does not exist in the registry or
    /// if it has already been dropped.
    pub fn try_upgrade<'a>(&'a self, commit_id: CommitId) -> Option<Commit<'a>> {
        self.borrow()
            .get(commit_id)
            // SAFETY: the commit will be alive as long as the state space
            // is alive
            .map(|rc| unsafe { Commit::from_relrc(rc) })
    }

    fn borrow(&self) -> Ref<'_, Registry<CommitData, ()>> {
        self.registry.as_ref().borrow()
    }

    pub fn as_registry(&self) -> &Rc<RefCell<Registry<CommitData, ()>>> {
        &self.registry
    }

    pub fn to_registry(&self) -> Rc<RefCell<Registry<CommitData, ()>>> {
        self.registry.clone()
    }

    /// Get all commits in the state space as a vector.
    pub fn all_commits(&self) -> Vec<(CommitId, Commit<'_>)> {
        self.borrow()
            .iter()
            .map(|(id, rc)| (id, unsafe { Commit::from_relrc(rc) }))
            .collect()
    }

    /// Create a new [`PersistentHugr`] in this state space, consisting of
    /// `commits` and their ancestors.
    ///
    /// All commits in the resulting `PersistentHugr` are guaranteed to be
    /// compatible. If the selected commits would include two commits which
    /// are incompatible, a [`InvalidCommit::IncompatibleHistory`] error is
    /// returned. If `commits` is empty, a [`InvalidCommit::NonUniqueBase`]
    /// error is returned.
    pub fn try_create(
        &self,
        commits: impl IntoIterator<Item = CommitId>,
    ) -> Result<PersistentHugr, InvalidCommit> {
        let commits: Vec<_> = commits
            .into_iter()
            .map(|id| {
                self.try_upgrade(id)
                    .ok_or(InvalidCommit::UnknownCommitId(id))
            })
            .try_collect()?;
        PersistentHugr::try_new(commits)
    }

    /// Get the (unique) base commit of the state space.
    ///
    /// Return `None` if `self` is empty.
    pub fn base_commit<'a>(&'a self) -> Option<Commit<'a>> {
        // get any commit
        let (_, relrc) = self.borrow().iter().next()?;
        // SAFETY: commit will be alive as long as the state space
        // is alive
        let commit: Commit<'a> = unsafe { Commit::from_relrc(relrc) };
        Some(commit.base_commit().clone())
    }
}
