//! Directives and errors relating to linking Hugrs.

use std::{collections::HashMap, fmt::Display};

use itertools::{Either, Itertools};

use crate::{HugrView, Node, Visibility, core::HugrNode, ops::OpType, types::PolyFuncType};

/// An error resulting from an [NodeLinkingDirective] passed to [insert_hugr_link_nodes]
/// or [insert_from_view_link_nodes].
///
/// [insert_hugr_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_hugr_link_nodes
/// [insert_from_view_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_from_view_link_nodes
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum NodeLinkingError<N: Display> {
    /// Inserting the whole Hugr, yet also asked to insert some of its children
    /// (so the inserted Hugr's entrypoint was its module-root).
    #[error(
        "Cannot insert children (e.g. {_0}) when already inserting whole Hugr (with module entrypoint)"
    )]
    ChildOfEntrypoint(N),
    /// A module-child requested contained (or was) the entrypoint
    #[error("Requested to insert module-child {_0} but this contains the entrypoint")]
    ChildContainsEntrypoint(N),
    /// A module-child requested was not a child of the module root
    #[error("{_0} was not a child of the module root")]
    NotChildOfRoot(N),
}

/// Directive for how to treat a particular FuncDefn/FuncDecl in the source Hugr.
/// (TN is a node in the target Hugr.)
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum NodeLinkingDirective<TN = Node> {
    /// Insert the FuncDecl, or the FuncDefn and its subtree, into the target Hugr.
    Add {
        // TODO If non-None, change the name of the inserted function
        //rename: Option<String>,
        /// Existing/old nodes in the target which will be removed (with their subtrees),
        /// and any ([EdgeKind::Function]) edges from them changed to leave the newly-inserted
        /// node instead. (Typically, this `Vec` would contain at most one `FuncDefn`,
        /// or perhaps-multiple, aliased, `FuncDecl`s.)
        ///
        /// [EdgeKind::Function]: crate::types::EdgeKind::Function
        replace: Vec<TN>,
    },
    /// Do not insert the node/subtree from the source, but for any inserted node
    /// with an ([EdgeKind::Function]) edge from it, change that edge to come from
    /// the specified existing node instead.
    ///
    /// [EdgeKind::Function]: crate::types::EdgeKind::Function
    UseExisting(TN),
}

impl<TN> NodeLinkingDirective<TN> {
    /// Just add the node (and any subtree) into the target.
    /// (Could lead to an invalid Hugr if the target Hugr
    /// already has another with the same name and both are [Public])
    ///
    /// [Public]: crate::Visibility::Public
    pub const fn add() -> Self {
        Self::Add { replace: vec![] }
    }

    /// The new node should replace the specified node (already existing)
    /// in the target. (Could lead to an invalid Hugr if they have
    /// different signatures.)
    pub fn replace(nodes: impl IntoIterator<Item = TN>) -> Self {
        Self::Add {
            replace: nodes.into_iter().collect(),
        }
    }
}

/// Describes ways to link a "Source" Hugr being inserted into a target Hugr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameLinkingPolicy {
    /// Do not use linking - just insert all functions from source Hugr into target.
    /// (This can lead to an invalid Hugr if there are name/signature conflicts on public functions).
    AddAll,
    /// Do not use linking - break any edges from functions in the source Hugr to the insert part.
    AddNone,
    /// Identify public functions in source and target Hugr by name.
    /// Multiple FuncDecls, and FuncDecl+FuncDefn pairs, with the same name and signature
    /// will be combined, taking the FuncDefn from either Hugr.
    LinkByName {
        /// If true, all private functions from the source hugr are inserted into the target.
        /// (Since these are private, name conflicts do not make the Hugr invalid.)
        /// If false, instead edges from said private functions to any inserted parts
        /// of the source Hugr will be broken, making the target Hugr invalid.
        copy_private_funcs: bool,
        /// How to handle cases where the same (public) name is present in both
        /// inserted and target Hugr but with different signatures.
        /// `true` means an error is raised and nothing is added to the target Hugr.
        /// `false` means the new function will be added alongside the existing one
        ///   - this will give an invalid Hugr (duplicate names).
        // NOTE there are other possible handling schemes, both where we don't insert the new function, both leading to an invalid Hugr:
        //   * don't insert but break edges --> Unconnected ports (or, replace and break existing edges)
        //   * use (or replace) the existing function --> incompatible ports
        // but given you'll need to patch the Hugr up afterwards, you can get there just
        // by setting this to `false` (and maybe removing one FuncDefn), or via explicit node linking.
        error_on_conflicting_sig: bool,
        /// How to handle cases where both target and inserted Hugr have a FuncDefn
        /// with the same name and signature.
        multi_impls: MultipleImplHandling,
        // TODO Renames to apply to public functions in the inserted Hugr. These take effect
        // before [error_on_conflicting_sig] or [take_existing_and_new_impls].
        // rename_map: HashMap<String, String>
    },
}

/// What to do when [NameLinkingPolicy::LinkByName] finds both target and inserted Hugr
/// have a [Visibility::Public] FuncDefn with the same name and signature.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum MultipleImplHandling {
    /// Do not perform insertion; raise an error instead
    ErrorDontInsert,
    /// Keep the implementation already in the target Hugr. (Edges in the source
    /// Hugr will be redirected to use the function from the target.)
    UseExisting,
    /// Keep the implementation in the source Hugr. (Edges in the target Hugr
    /// will be redirected to use the function from the source; the previously-existing
    /// function in the target Hugr will be removed.)
    UseNew,
    /// Add the new function alongside the existing one in the target Hugr,
    /// preserving (separately) uses of both. (The Hugr will be invalid because
    /// of duplicate names.)
    UseBoth,
}

/// An error in using names to determine how to link functions in source and target Hugrs.
/// (SN = Source Node, TN = Target Node)
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
pub enum NameLinkingError<SN: Display, TN: Display + std::fmt::Debug> {
    /// Both source and target contained a [FuncDefn] (public and with same name
    /// and signature).
    ///
    /// [FuncDefn]: crate::ops::FuncDefn
    #[error("Source ({_1}) and target ({_2}) both contained FuncDefn with same public name {_0}")]
    MultipleImpls(String, SN, TN),
    /// Source and target containing public functions with conflicting signatures
    // TODO ALAN Should we indicate which were decls or defns? via an extra enum?
    #[error(
        "Conflicting signatures for name {name} - Source ({src_node}) has {src_sig}, Target ({tgt_node}) has ({tgt_sig})"
    )]
    #[allow(missing_docs)]
    Signatures {
        name: String,
        src_node: SN,
        src_sig: Box<PolyFuncType>,
        tgt_node: TN,
        tgt_sig: Box<PolyFuncType>,
    },
    /// A [Visibility::Public] function in the source, whose body is being added
    /// to the target, contained the entrypoint (which needs to be added
    /// in a different place).
    #[error("The entrypoint is contained within function {_0} which will be added as {_1:?}")]
    AddFunctionContainingEntrypoint(SN, NodeLinkingDirective<TN>),
}

impl NameLinkingPolicy {
    /// The default policy used by [HugrMut::insert_hugr].
    /// * All private functions are copied
    /// * If public functions have conflicting signatures, we keep both
    /// * If both existing and inserted Hugrs have same-signature public FuncDefns,
    ///   the newly inserted one replaces the original
    ///
    /// [HugrMut::insert_hugr]: crate::hugr::HugrMut::insert_hugr
    pub fn default_for_hugr() -> Self {
        Self::LinkByName {
            copy_private_funcs: true,
            error_on_conflicting_sig: false,
            multi_impls: MultipleImplHandling::UseNew,
        }
    }

    /// The default policy used by [HugrMut::insert_from_view].
    /// * Private functions are not copied, i.e. edges from them into the inserted portion, are disconnected in the target
    /// * If public functions have conflicting signatures, we keep both
    /// * If both existing and inserted Hugrs have same-signature public FuncDefns,
    ///   the original is used in place of the new.
    ///
    /// [HugrMut::insert_from_view]: crate::hugr::HugrMut::insert_from_view
    pub fn default_for_view() -> Self {
        Self::LinkByName {
            copy_private_funcs: false,
            error_on_conflicting_sig: false,
            multi_impls: MultipleImplHandling::UseExisting,
        }
    }

    /// Builds an explicit map of [NodeLinkingDirective]s that implements this policy for a given
    /// source (inserted) and target (inserted-into) Hugr.
    /// The map should be such that no [NodeLinkingError] will occur.
    #[allow(clippy::type_complexity)]
    pub fn to_node_linking<T: HugrView + ?Sized, S: HugrView + ?Sized>(
        &self,
        target: &T,
        source: &S,
    ) -> Result<NodeLinkingPolicy<S::Node, T::Node>, NameLinkingError<S::Node, T::Node>> {
        let existing = gather_existing(target);
        let mut res = NodeLinkingPolicy::new();

        for n in source.children(source.module_root()) {
            if let Some((name, is_defn, vis, sig)) = link_sig(source, n) {
                let mut dirv = NodeLinkingDirective::add();
                match self {
                    NameLinkingPolicy::AddAll => (),
                    NameLinkingPolicy::AddNone => continue,
                    NameLinkingPolicy::LinkByName {
                        copy_private_funcs,
                        error_on_conflicting_sig,
                        multi_impls,
                    } => {
                        if !vis.is_public() {
                            if !copy_private_funcs {
                                continue;
                            };
                        } else if let Some((ex_ns, ex_sig)) = existing.get(name) {
                            if sig == *ex_sig {
                                dirv = directive(name, n, is_defn, ex_ns, multi_impls)?
                            } else if *error_on_conflicting_sig {
                                return Err(NameLinkingError::Signatures {
                                    name: name.clone(),
                                    src_node: n,
                                    src_sig: Box::new(sig.clone()),
                                    tgt_node: *ex_ns.as_ref().left_or_else(|(n, _)| n),
                                    tgt_sig: Box::new((*ex_sig).clone()),
                                });
                            }
                        };
                    }
                };
                res.insert(n, dirv);
            }
        }
        Ok(res)
    }
}

fn directive<SN: Display, TN: HugrNode>(
    name: &str,
    new_n: SN,
    new_defn: bool,
    ex_ns: &Either<TN, (TN, Vec<TN>)>,
    multi_impls: &MultipleImplHandling,
) -> Result<NodeLinkingDirective<TN>, NameLinkingError<SN, TN>> {
    Ok(match (new_defn, ex_ns) {
        (false, Either::Right(_)) => NodeLinkingDirective::add(), // another alias
        (false, Either::Left(defn)) => NodeLinkingDirective::UseExisting(*defn), // resolve decl
        (true, Either::Right((decl, decls))) => {
            NodeLinkingDirective::replace(std::iter::once(decl).chain(decls).cloned())
        }
        (true, &Either::Left(defn)) => match multi_impls {
            MultipleImplHandling::UseExisting => NodeLinkingDirective::UseExisting(defn),
            MultipleImplHandling::UseNew => NodeLinkingDirective::replace([defn]),
            MultipleImplHandling::ErrorDontInsert => {
                return Err(NameLinkingError::MultipleImpls(
                    name.to_owned(),
                    new_n,
                    defn,
                ));
            }
            MultipleImplHandling::UseBoth => NodeLinkingDirective::add(),
        },
    })
}

type PubFuncs<'a, N> = (Either<N, (N, Vec<N>)>, &'a PolyFuncType);

fn link_sig<H: HugrView + ?Sized>(
    h: &H,
    n: H::Node,
) -> Option<(&String, bool, &Visibility, &PolyFuncType)> {
    match h.get_optype(n) {
        OpType::FuncDecl(fd) => Some((fd.func_name(), false, fd.visibility(), fd.signature())),
        OpType::FuncDefn(fd) => Some((fd.func_name(), true, fd.visibility(), fd.signature())),
        _ => None,
    }
}

fn gather_existing<H: HugrView + ?Sized>(h: &H) -> HashMap<&String, PubFuncs<H::Node>> {
    let left_if = |b| if b { Either::Left } else { Either::Right };
    h.children(h.module_root())
        .filter_map(|n| {
            link_sig(h, n).and_then(|(fname, is_defn, vis, sig)| {
                vis.is_public()
                    .then_some((fname, (left_if(is_defn)(n), sig)))
            })
        })
        .into_grouping_map()
        .aggregate(|acc: Option<PubFuncs<H::Node>>, name, (new, sig2)| {
            let Some((mut acc, sig1)) = acc else {
                return Some((new.map_right(|n| (n, vec![])), sig2));
            };
            assert_eq!(sig1, sig2, "Invalid Hugr: different signatures for {name}");
            let (Either::Right((_, decls)), Either::Right(ndecl)) = (&mut acc, &new) else {
                let err = match acc.is_left() && new.is_left() {
                    true => "Multiple FuncDefns",
                    false => "FuncDefn and FuncDecl(s)",
                };
                panic!("Invalid Hugr: {err} for {name}");
            };
            decls.push(*ndecl);
            Some((acc, sig2))
        })
}

/// Details, node-by-node, how module-children of a source Hugr should be inserted into a
/// target Hugr (beneath the module root). For use with [insert_hugr_link_nodes] and
/// [insert_from_view_link_nodes].
///
/// [insert_hugr_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_hugr_link_nodes
/// [insert_from_view_link_nodes]: crate::hugr::hugrmut::HugrMut::insert_from_view_link_nodes
pub type NodeLinkingPolicy<SN, TN> = HashMap<SN, NodeLinkingDirective<TN>>;

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use itertools::Itertools as _;
    use rstest::rstest;

    use crate::builder::{
        DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder, HugrBuilder,
        ModuleBuilder,
    };
    use crate::extension::prelude::{ConstUsize, usize_t};
    use crate::hugr::linking::{
        MultipleImplHandling, NameLinkingError, NameLinkingPolicy, NodeLinkingDirective,
    };
    use crate::hugr::{ValidationError, hugrmut::HugrMut};
    use crate::ops::OpType;
    use crate::ops::handle::NodeHandle;
    use crate::std_extensions::arithmetic::int_ops::IntOpDef;
    use crate::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
    use crate::types::Signature;
    use crate::{Hugr, HugrView, Visibility};

    fn list_decls_defns<H: HugrView>(h: &H) -> (HashMap<H::Node, &str>, HashMap<H::Node, &str>) {
        let mut decls = HashMap::new();
        let mut defns = HashMap::new();
        for n in h.children(h.module_root()) {
            match h.get_optype(n) {
                OpType::FuncDecl(fd) => decls.insert(n, fd.func_name().as_str()),
                OpType::FuncDefn(fd) => defns.insert(n, fd.func_name().as_str()),
                _ => None,
            };
        }
        (decls, defns)
    }

    fn call_targets<H: HugrView>(h: &H) -> HashMap<H::Node, H::Node> {
        h.nodes()
            .filter(|n| h.get_optype(*n).is_call())
            .map(|n| {
                let tgt = h.static_source(n).expect(format!("For node {n}").as_str());
                (n, tgt)
            })
            .collect()
    }

    #[test]
    fn combines_decls_defn() {
        let i64_t = || INT_TYPES[6].to_owned();
        let foo_sig = Signature::new_endo(i64_t());
        let bar_sig = Signature::new(vec![i64_t(); 2], i64_t());
        let orig_target = {
            let mut fb =
                FunctionBuilder::new_vis("foo", foo_sig.clone(), Visibility::Public).unwrap();
            let mut mb = fb.module_root_builder();
            let bar1 = mb.declare("bar", bar_sig.clone().into()).unwrap();
            let bar2 = mb.declare("bar", bar_sig.clone().into()).unwrap(); // alias
            let [i] = fb.input_wires_arr();
            let [c] = fb.call(&bar1, &[], [i, i]).unwrap().outputs_arr();
            let r = fb.call(&bar2, &[], [i, c]).unwrap();
            let h = fb.finish_hugr_with_outputs(r.outputs()).unwrap();
            assert_eq!(
                list_decls_defns(&h),
                (
                    HashMap::from([(bar1.node(), "bar"), (bar2.node(), "bar")]),
                    HashMap::from([(h.entrypoint(), "foo")])
                )
            );
            h
        };

        let inserted = {
            let mut dfb = DFGBuilder::new(Signature::new(vec![], i64_t())).unwrap();
            let mut mb = dfb.module_root_builder();
            let foo1 = mb.declare("foo", foo_sig.clone().into()).unwrap();
            let foo2 = mb.declare("foo", foo_sig.clone().into()).unwrap();
            let mut bar = mb
                .define_function_vis("bar", bar_sig.clone(), Visibility::Public)
                .unwrap();
            let res = bar
                .add_dataflow_op(IntOpDef::iadd.with_log_width(6), bar.input_wires())
                .unwrap();
            let bar = bar.finish_with_outputs(res.outputs()).unwrap();
            let i = dfb.add_load_value(ConstInt::new_u(6, 257).unwrap());
            let c = dfb.call(&foo1, &[], [i]).unwrap();
            let r = dfb.call(&foo2, &[], c.outputs()).unwrap();
            let h = dfb.finish_hugr_with_outputs(r.outputs()).unwrap();
            assert_eq!(
                list_decls_defns(&h),
                (
                    HashMap::from([(foo1.node(), "foo"), (foo2.node(), "foo")]),
                    HashMap::from([
                        (h.get_parent(h.entrypoint()).unwrap(), "main"),
                        (bar.node(), "bar")
                    ])
                )
            );
            h
        };

        // AddNone: inserted DFG has all calls disconnected
        let mut target = orig_target.clone();
        let node_map = target
            .insert_from_view_link_names(
                Some(target.entrypoint()),
                &inserted,
                NameLinkingPolicy::AddNone,
            )
            .unwrap();
        assert_eq!(list_decls_defns(&target), list_decls_defns(&orig_target)); // no new Funcs
        assert!(matches!(
            target.validate(),
            Err(ValidationError::UnconnectedPort { .. })
        ));
        let dfg = node_map[&inserted.entrypoint()];
        assert_eq!(
            target.children(dfg).flat_map(|n| target.children(n)).next(),
            None
        );
        for c in target.nodes().filter(|n| target.get_optype(*n).is_call()) {
            assert_eq!(
                target.static_source(c).is_none(),
                target.get_parent(c) == Some(dfg)
            );
        }
        target.remove_subtree(dfg);
        target.validate().unwrap();
        // Hugrs will not be equal because of internal graph representation details
        assert_eq!(target.num_nodes(), orig_target.num_nodes());
        for n in target.nodes() {
            assert_eq!(target.get_optype(n), orig_target.get_optype(n));
            for inp in target.node_inputs(n) {
                assert_eq!(
                    target.linked_outputs(n, inp).collect_vec(),
                    orig_target.linked_outputs(n, inp).collect_vec()
                );
            }
        }

        // AddAll (w/out entrypoint): conflicting FuncDecls / FuncDefns.
        let mut target = orig_target.clone();
        // Do not add entrypoint subtree - it is contained in an added subtree. (Hence, AddAll + Some useless for any non-module entrypoint....TODO what about AddAll+Some w/module entrypoint?)
        // TODO - allow adding only public, w/out linking ??
        // Or, make AddAll exclude the entrypoint-container ?? (Like current insert_hugr)
        target
            .insert_from_view_link_names(None, &inserted, NameLinkingPolicy::AddAll)
            .unwrap();
        assert!(matches!(
            target.validate(),
            Err(ValidationError::DuplicateExport { .. })
        ));
        let (decls, defns) = list_decls_defns(&target);
        assert_eq!(
            decls.values().copied().sorted().collect_vec(),
            ["bar", "bar", "foo", "foo"]
        );
        assert_eq!(
            defns.values().copied().sorted().collect_vec(),
            ["bar", "foo", "main"]
        );
        let call_tgts = call_targets(&target);
        for decl in decls.keys() {
            assert_eq!(call_tgts.values().filter(|tgt| *tgt == decl).count(), 1); // as before
        }
        for defn in defns.keys() {
            assert_eq!(call_tgts.values().find(|tgt| *tgt == defn), None); // as before
        }

        // Linking by name...neither of the looped-over params should make any difference:
        for error_on_conflicting_sig in [false, true] {
            for multi_impls in [
                MultipleImplHandling::ErrorDontInsert,
                MultipleImplHandling::UseNew,
                MultipleImplHandling::UseExisting,
                MultipleImplHandling::UseBoth,
            ] {
                let pol = |copy_private_funcs| NameLinkingPolicy::LinkByName {
                    copy_private_funcs,
                    error_on_conflicting_sig,
                    multi_impls,
                };
                let mut target = orig_target.clone();
                let res = target.insert_from_view_link_names(
                    Some(target.entrypoint()),
                    &inserted,
                    pol(true),
                );
                assert_eq!(
                    res.err().unwrap(),
                    NameLinkingError::AddFunctionContainingEntrypoint(
                        inserted.get_parent(inserted.entrypoint()).unwrap(),
                        NodeLinkingDirective::add()
                    )
                );
                assert_eq!(target, orig_target);

                target
                    .insert_hugr_link_names(Some(target.entrypoint()), inserted.clone(), pol(false))
                    .unwrap();
                target.validate().unwrap();
                let (decls, defns) = list_decls_defns(&target);
                assert_eq!(decls, HashMap::new());
                assert_eq!(
                    defns.values().copied().sorted().collect_vec(),
                    ["bar", "foo"]
                );
                let call_tgts = call_targets(&target);
                for defn in defns.keys() {
                    // Defns now have two calls each (was one to each alias)
                    assert_eq!(call_tgts.values().filter(|tgt| *tgt == defn).count(), 2);
                }
            }
        }
    }

    // TODO test copy_private_funcs actually copying; presence/absence of parent when inserting (subtree of) public func
    #[rstest]
    fn sig_conflict(
        #[values(false, true)] host_defn: bool,
        #[values(false, true)] inserted_defn: bool,
    ) {
        let mk_def_or_decl = |n, sig: Signature, defn| {
            let mut mb = ModuleBuilder::new();
            let node = if defn {
                let fb = mb.define_function_vis(n, sig, Visibility::Public).unwrap();
                let ins = fb.input_wires();
                fb.finish_with_outputs(ins).unwrap().node()
            } else {
                mb.declare(n, sig.into()).unwrap().node()
            };
            (mb.finish_hugr().unwrap(), node)
        };

        let old_sig = Signature::new_endo(usize_t());
        let (orig_host, orig_fn) = mk_def_or_decl("foo", old_sig.clone(), host_defn);
        let new_sig = Signature::new_endo(INT_TYPES[3].clone());
        let (inserted, inserted_fn) = mk_def_or_decl("foo", new_sig.clone(), inserted_defn);

        let mk_pol = |error_on_conflicting_sig| NameLinkingPolicy::LinkByName {
            copy_private_funcs: true,
            error_on_conflicting_sig,
            multi_impls: MultipleImplHandling::ErrorDontInsert,
        };
        let mut host = orig_host.clone();
        let res = host.insert_hugr_link_names(None, inserted.clone(), mk_pol(true));
        assert_eq!(host, orig_host); // Did nothing
        assert_eq!(
            res,
            Err(NameLinkingError::Signatures {
                name: "foo".to_string(),
                src_node: inserted_fn,
                src_sig: Box::new(new_sig.into()),
                tgt_node: orig_fn,
                tgt_sig: Box::new(old_sig.into())
            })
        );

        let node_map = host
            .insert_hugr_link_names(None, inserted, mk_pol(false))
            .unwrap();
        assert_eq!(
            host.validate(),
            Err(ValidationError::DuplicateExport {
                link_name: "foo".to_string(),
                children: [orig_fn, node_map[&inserted_fn]]
            })
        );
    }

    #[rstest]
    #[case(MultipleImplHandling::UseNew, vec![11])]
    #[case(MultipleImplHandling::UseExisting, vec![5])]
    #[case(MultipleImplHandling::UseBoth, vec![5, 11])]
    #[case(MultipleImplHandling::ErrorDontInsert, vec![])]
    fn impl_conflict(#[case] multi_impls: MultipleImplHandling, #[case] expected: Vec<u64>) {
        fn build_hugr(cst: u64) -> Hugr {
            let mut mb = ModuleBuilder::new();
            let mut fb = mb
                .define_function_vis("foo", Signature::new(vec![], usize_t()), Visibility::Public)
                .unwrap();
            let c = fb.add_load_value(ConstUsize::new(cst));
            fb.finish_with_outputs([c]).unwrap();
            mb.finish_hugr().unwrap()
        }
        let backup = build_hugr(5);
        let mut host = backup.clone();
        let inserted = build_hugr(11);

        let res = host.insert_hugr_link_names(
            None,
            inserted,
            NameLinkingPolicy::LinkByName {
                copy_private_funcs: true,
                error_on_conflicting_sig: false,
                multi_impls,
            },
        );
        if multi_impls == MultipleImplHandling::ErrorDontInsert {
            assert!(matches!(res, Err(NameLinkingError::MultipleImpls(n, _, _)) if n == "foo"));
            assert_eq!(host, backup);
            return;
        }
        res.unwrap();
        let res = host.validate();
        if multi_impls == MultipleImplHandling::UseBoth {
            assert!(
                matches!(res, Err(ValidationError::DuplicateExport { link_name, .. }) if link_name == "foo")
            );
        } else {
            res.unwrap();
        }
        let func_consts = host
            .children(host.module_root())
            .filter(|n| host.get_optype(*n).is_func_defn())
            .map(|n| {
                host.children(n)
                    .filter_map(|ch| host.get_optype(ch).as_const())
                    .exactly_one()
                    .ok()
                    .unwrap()
                    .get_custom_value::<ConstUsize>()
                    .unwrap()
                    .value()
            })
            .collect_vec();
        assert_eq!(func_consts, expected);
    }
}
