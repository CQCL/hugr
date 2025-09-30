//! [`Commit`] type and data associated with it.

use std::{marker::PhantomData, mem};

use delegate::delegate;
use hugr_core::{
    Direction, Hugr, HugrView, IncomingPort, Node, OutgoingPort, Port,
    hugr::{
        NodeMetadataMap, internal::HugrInternals, patch::simple_replace::InvalidReplacement,
        views::InvalidSignature,
    },
    ops::OpType,
};
use itertools::{Either, Itertools};
use relrc::RelRc;
use thiserror::Error;

use crate::{
    CommitData, CommitId, CommitStateSpace, PatchNode, PersistentReplacement,
    subgraph::InvalidPinnedSubgraph,
};

mod boundary;

/// A single unit of change in a [`PersistentHugr`].
///
/// Invariant: there is always a unique root commit (i.e. a commit with variant
/// `CommitData::Base`) in the ancestors of a commit.
///
/// The data within a commit is a patch, representing a rewrite that can be
/// performed on the Hugr defined by the ancestors of the commit. Currently,
/// patches must be [`SimpleReplacement`]s.
///
/// # Lifetime of commits
///
/// A commit remains valid as long as the [`CommitStateSpace`] containing it is
/// alive. Note that it is also sufficient that a [`PersistentHugr`] containing
/// the commit is alive, given that the [`CommitStateSpace`] is guaranteed to
/// be alive as long as any of its contained [`PersistentHugr`]s. In other
/// words, the lifetime dependency is:
/// ```ignore
/// PersistentHugr -> CommitStateSpace -> Commit
/// ```
/// where `->` can be read as "is outlived by". Note that the dependencies are
/// NOT valid in the other direction: a [`Commit`] only maintains a weak
/// reference to its [`CommitStateSpace`].
///
/// When a [`CommitStateSpace`] goes out of scope, all its commits become
/// invalid. The implementation uses lifetimes to ensure at compile time that
/// the commit is valid throughout its lifetime. All constructors of [`Commit`]
/// thus expect a reference to the state space that the commit should be added
/// to, which fixes the lifetime of the commit.
///
/// Methods that directly modify the lifetime are marked as `unsafe`. It is up
/// to the user to ensure that the commit is valid throughout its updated
/// lifetime.
///
/// [`PersistentHugr`]: crate::PersistentHugr
/// [`SimpleReplacement`]: hugr_core::SimpleReplacement
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct Commit<'a>(RelRc<CommitData, ()>, PhantomData<&'a ()>);

impl<'a> Commit<'a> {
    /// Create a commit from a simple replacement.
    ///
    /// Requires a reference to the commit state space that the commit should
    /// be added to.
    ///
    /// Use [`Self::try_new`] instead if the parents of the commit cannot be
    /// inferred from the invalidation set of `replacement` alone.
    ///
    /// The replacement must act on a non-empty subgraph, otherwise this
    /// function will return an [`InvalidCommit::EmptyReplacement`] error.
    ///
    /// If any of the parents of the replacement are not in the commit state
    /// space, this function will return an [`InvalidCommit::UnknownParent`]
    /// error.
    pub fn try_from_replacement(
        replacement: PersistentReplacement,
        state_space: &'a CommitStateSpace,
    ) -> Result<Self, InvalidCommit> {
        Self::try_new(replacement, [], state_space)
    }

    /// Create a new base commit.
    ///
    /// Note that the base commit must be unique in the state space. This does
    /// not check for uniqueness.
    ///
    /// Prefer using [`CommitStateSpace::try_set_base`] or
    /// [`PersistentHugr::with_base`] to create base commits.
    ///
    /// [`PersistentHugr::with_base`]: crate::PersistentHugr::with_base
    pub(crate) fn new_base(hugr: Hugr, state_space: &'a CommitStateSpace) -> Self {
        let commit = RelRc::new(CommitData::Base(hugr));
        commit
            .try_register_in(state_space.as_registry())
            .expect("new node is not yet registered");

        Commit(commit, PhantomData)
    }

    /// Create a new commit
    ///
    /// Requires a reference to the commit state space that the commit should
    /// be added to.
    ///
    /// The returned commit will correspond to the application of `replacement`
    /// and will be the child of the commits in `parents` as well as of all
    /// the commits in the invalidation set of `replacement`.
    ///
    /// The replacement must act on a non-empty subgraph, otherwise this
    /// function will return an [`InvalidCommit::EmptyReplacement`] error.
    /// If any of the parents of the replacement are not in the commit state
    /// space, this function will return an [`InvalidCommit::UnknownParent`]
    /// error.
    pub fn try_new<'b>(
        replacement: PersistentReplacement,
        parents: impl IntoIterator<Item = Commit<'b>>,
        state_space: &'a CommitStateSpace,
    ) -> Result<Self, InvalidCommit> {
        // TODO: clearly this needs to check that all state_space are the same??
        if replacement.subgraph().nodes().is_empty() {
            return Err(InvalidCommit::EmptyReplacement);
        }
        let repl_parents = get_parent_commits(&replacement, state_space)?
            .into_iter()
            .map_into::<RelRc<_, ()>>();
        let parents = parents
            .into_iter()
            .map_into::<RelRc<_, ()>>()
            .chain(repl_parents)
            .unique_by(|p| p.as_ptr());
        let rc = RelRc::with_parents(replacement.into(), parents.into_iter().map(|p| (p, ())));

        if let Err(err) = get_base_ancestors(&rc).exactly_one() {
            return Err(InvalidCommit::NonUniqueBase(err.count()));
        }

        rc.try_register_in(state_space.as_registry())
            .expect("new node is not yet registered");

        Ok(Self(rc, PhantomData))
    }

    /// Create a commit from a relrc.
    ///
    /// This is unsafe because it cannot be guaranteed that the commit will
    /// live as long as the lifetime 'a.
    pub(crate) unsafe fn from_relrc(rc: RelRc<CommitData, ()>) -> Self {
        Self(rc, PhantomData)
    }

    /// The state space that `self` belongs to.
    pub fn state_space(&self) -> CommitStateSpace {
        self.0
            .registry()
            .expect("invalid commit: not registered")
            .into()
    }

    /// Get the ID of `self`.
    pub fn id(&self) -> CommitId {
        self.state_space()
            .get_id(self)
            .expect("invalid commit: not registered")
    }

    /// Get the parents of `commit_id`
    pub fn parents(&self) -> impl Iterator<Item = &Self> + '_ {
        self.as_relrc()
            .all_parents()
            .map_into()
            // SAFETY: the parents will be alive as long as self
            .map(|cm: &Commit| unsafe { upgrade_lifetime(cm) })
    }

    /// Get all commits that have `self` as parent in `state_space`.
    pub fn children(&self, _state_space: &'a CommitStateSpace) -> impl Iterator<Item = Self> + '_ {
        self.as_relrc()
            .all_children()
            // SAFETY: the children will be alive as long as the state space
            // is alive
            .map(|rc| unsafe { Self::from_relrc(rc) })
    }

    /// Check if `self` is a valid commit.
    ///
    /// This checks if
    ///  1. there is exactly one ancestor of `self` that is a base commit, and
    ///  2. `self` is registered in a state space.
    pub fn is_valid(&self) -> bool {
        get_base_ancestors(&self.0).exactly_one().is_ok() && self.0.registry().is_some()
    }

    pub(crate) fn as_relrc(&self) -> &RelRc<CommitData, ()> {
        &self.0
    }

    /// Get the set of nodes inserted by `self`.
    pub fn inserted_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        match self.0.value() {
            CommitData::Base(base) => Either::Left(base.nodes()),
            CommitData::Replacement(repl) => {
                // Skip the entrypoint and the IO nodes
                Either::Right(repl.replacement().entry_descendants().skip(3))
            }
        }
    }

    /// Get the patch that `self` represents.
    pub fn replacement(&self) -> Option<&PersistentReplacement> {
        match self.0.value() {
            CommitData::Base(_) => None,
            CommitData::Replacement(replacement) => Some(replacement),
        }
    }

    /// Get the set of nodes in parent commits deleted by applying `self`.
    ///
    /// Currently this is the same as [`Self::invalidated_parent_nodes`].
    pub fn deleted_parent_nodes(&self) -> impl Iterator<Item = PatchNode> + '_ {
        self.replacement()
            .into_iter()
            .flat_map(|r| r.invalidation_set())
    }

    /// Get the Hugr inserted by `self`.
    ///
    /// This is either the replacement Hugr of a [`CommitData::Replacement`] or
    /// the base Hugr of a [`CommitData::Base`].
    pub(crate) fn commit_hugr(&self) -> &Hugr {
        match self.value() {
            CommitData::Base(base) => base,
            CommitData::Replacement(repl) => repl.replacement(),
        }
    }

    delegate! {
        to self.0 {
            pub(crate) fn value(&self) -> &CommitData;
            pub(crate) fn as_ptr(&self) -> *const relrc::node::InnerData<CommitData, ()>;
        }
    }

    /// Get the base commit of `self`.
    pub(crate) fn base_commit(&self) -> &Self {
        let rc = get_base_ancestors(&self.0)
            .next()
            .expect("no base commit found");
        let commit: &Commit = rc.into();
        // SAFETY: base commit lives at least as long as `self`
        unsafe { upgrade_lifetime(commit) }
    }

    /// Check if `(node, port)` is a value port in `self`.
    pub(crate) fn is_value_port(&self, node: Node, port: impl Into<Port>) -> bool {
        self.commit_hugr()
            .get_optype(node)
            .port_kind(port)
            .expect("invalid port")
            .is_value()
    }

    /// All value ports of `node` in `dir`.
    pub(crate) fn value_ports(
        &self,
        node: Node,
        dir: Direction,
    ) -> impl Iterator<Item = (Node, Port)> + '_ {
        let ports = self.node_ports(node, dir);
        ports.filter_map(move |p| self.is_value_port(node, p).then_some((node, p)))
    }

    /// All outgoing value ports of `node` in `self`.
    pub(crate) fn output_value_ports(
        &self,
        node: Node,
    ) -> impl Iterator<Item = (Node, OutgoingPort)> + '_ {
        self.value_ports(node, Direction::Outgoing)
            .map(|(n, p)| (n, p.as_outgoing().expect("unexpected port direction")))
    }

    /// All incoming value ports of `node` in `self`.
    pub(crate) fn input_value_ports(
        &self,
        node: Node,
    ) -> impl Iterator<Item = (Node, IncomingPort)> + '_ {
        self.value_ports(node, Direction::Incoming)
            .map(|(n, p)| (n, p.as_incoming().expect("unexpected port direction")))
    }

    /// Change the lifetime of the commit.
    ///
    /// This is unsafe because it cannot be guaranteed that the commit will
    /// live as long as the lifetime 'b. The user must guuarantee that the
    /// rewrite space of `self` is valid as long as the lifetime 'b.
    pub unsafe fn upgrade_lifetime<'b>(self) -> Commit<'b> {
        Commit(self.0, PhantomData)
    }
}

/// Change the lifetime of the commit reference.
///
/// This is unsafe because it cannot be guaranteed that the commit will
/// live as long as the lifetime 'b.
pub(crate) unsafe fn upgrade_lifetime<'a, 'b, 'c>(commit: &'c Commit<'a>) -> &'c Commit<'b> {
    unsafe { mem::transmute(commit) }
}

// The subset of HugrView methods that can be implemented on Commits
// by simplify delegating to the patches' respective HUGRs
impl Commit<'_> {
    /// Get the type of the operation at `node`.
    pub fn get_optype(&self, node: Node) -> &OpType {
        let hugr = self.commit_hugr();
        hugr.get_optype(node)
    }

    /// Get the number of ports of `node` in `dir`.
    pub fn num_ports(&self, node: Node, dir: Direction) -> usize {
        self.commit_hugr().num_ports(node, dir)
    }

    /// Iterator over output ports of node.
    /// Like [`CommitStateSpace::node_ports`](node, Direction::Outgoing)`
    /// but preserves knowledge that the ports are [OutgoingPort]s.
    #[inline]
    pub fn node_outputs(&self, node: Node) -> impl Iterator<Item = OutgoingPort> + Clone + '_ {
        self.node_ports(node, Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }

    /// Iterator over inputs ports of node.
    /// Like [`CommitStateSpace::node_ports`](node, Direction::Incoming)`
    /// but preserves knowledge that the ports are [IncomingPort]s.
    #[inline]
    pub fn node_inputs(&self, node: Node) -> impl Iterator<Item = IncomingPort> + Clone + '_ {
        self.node_ports(node, Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// Get an iterator over the ports of `node` in `dir`.
    pub fn node_ports(
        &self,
        node: Node,
        dir: Direction,
    ) -> impl Iterator<Item = Port> + Clone + '_ {
        self.commit_hugr().node_ports(node, dir)
    }

    /// Get an iterator over all ports of `node`.
    pub fn all_node_ports(&self, node: Node) -> impl Iterator<Item = Port> + Clone + '_ {
        self.commit_hugr().all_node_ports(node)
    }

    /// Get the metadata map of `node`.
    pub fn node_metadata_map(&self, node: Node) -> &NodeMetadataMap {
        self.commit_hugr().node_metadata_map(node)
    }
}

fn get_base_ancestors(arg: &RelRc<CommitData, ()>) -> impl Iterator<Item = &RelRc<CommitData, ()>> {
    arg.all_ancestors()
        .filter(|c| matches!(c.value(), CommitData::Base(_)))
}

impl From<Commit<'_>> for RelRc<CommitData, ()> {
    fn from(Commit(data, _): Commit) -> Self {
        data
    }
}

impl<'a> From<&'a RelRc<CommitData, ()>> for &'a Commit<'a> {
    fn from(rc: &'a RelRc<CommitData, ()>) -> Self {
        // SAFETY: Commit is a transparent wrapper around RelRc
        unsafe { mem::transmute(rc) }
    }
}

/// An error that occurs when trying to add a commit to a commit state space.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum InvalidCommit {
    /// The commit conflicts with existing commits in the state space.
    #[error("Incompatible history: children of commit {0:?} conflict in {1:?}")]
    IncompatibleHistory(CommitId, Node),

    /// The commit has a parent not present in the state space.
    #[error("Missing parent commit: {0:?}")]
    UnknownParent(CommitId),

    /// The commit is not a replacement.
    #[error("Commit is not a replacement")]
    NotReplacement,

    /// The set of commits contains zero or more than one base commit.
    #[error("{0} base commits found (should be 1)")]
    NonUniqueBase(usize),

    /// The commit is an empty replacement.
    #[error("Not allowed: empty replacement")]
    EmptyReplacement,

    #[error("Invalid subgraph: {0}")]
    /// The subgraph of the replacement is not convex.
    InvalidSubgraph(#[from] InvalidPinnedSubgraph),

    /// The replacement of the commit is invalid.
    #[error("Invalid replacement: {0}")]
    InvalidReplacement(#[from] InvalidReplacement),

    /// The signature of the replacement is invalid.
    #[error("Invalid signature: {0}")]
    InvalidSignature(#[from] InvalidSignature),

    /// A wire has an unpinned port.
    #[error("Incomplete wire: {0} is unpinned")]
    IncompleteWire(PatchNode, Port),

    /// The commit ID is not in the state space.
    #[error("Unknown commit ID: {0:?}")]
    UnknownCommitId(CommitId),
}

fn get_parent_commits<'a>(
    replacement: &PersistentReplacement,
    state_space: &'a CommitStateSpace,
) -> Result<Vec<Commit<'a>>, InvalidCommit> {
    let parent_ids = replacement.invalidation_set().map(|n| n.owner()).unique();
    parent_ids
        .map(|id| {
            state_space
                .try_upgrade(id)
                .ok_or(InvalidCommit::UnknownParent(id))
        })
        .collect()
}
