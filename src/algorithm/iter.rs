//! Routines for iterating over all nodes in a Hugr, attending to visiting
// in_neighbors of a node before that node.
use crate::hugr::views::PetgraphWrapper;
use crate::{
    hugr::views::{HierarchyView, SiblingGraph},
    ops::{OpTag, OpTrait},
    HugrView, Node,
};
use petgraph::visit::{Dfs, Topo, Walker};

type PetgraphMap<'a> = <PetgraphWrapper<'a, SiblingGraph<'a>> as petgraph::visit::Visitable>::Map;

enum SiblingWalker<'a> {
    Dataflow(Topo<Node, PetgraphMap<'a>>),
    ControlFlow(Dfs<Node, PetgraphMap<'a>>),
}

impl<'a> Walker<SiblingGraph<'a>> for SiblingWalker<'a> {
    type Item = Node;
    fn walk_next(&mut self, context: SiblingGraph<'a>) -> Option<Self::Item> {
        match self {
            Self::Dataflow(ref mut topo) => topo.walk_next(context.as_petgraph()),
            Self::ControlFlow(ref mut dfs) => dfs.walk_next(context.as_petgraph()),
        }
    }
}

/// Some traversals of a graph will visit a node with children twice, once
/// before the children and once after. The visit before the children will come
/// as `PreOrder(node)` while the visit after the children will come as
/// `PostOrder(node)`. Note that nodes without children will be visited only
/// once, in a `PostOrder(node)`.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum VisitOrder<X> {
    #[allow(missing_docs)]
    PreOrder(X),
    #[allow(missing_docs)]
    PostOrder(X),
}

impl<X> VisitOrder<X> {
    fn into_postorder(self) -> Option<X> {
        match self {
            VisitOrder::PostOrder(x) => Some(x),
            _ => None,
        }
    }
}

/// Return an iterator over root and the immediate children of root.
/// `VisitOrder::PreOrder(root)` will be the first result of the iterator and
/// `VisitOrder::PostOrder(root)` will be the last.  No other [VisitOrder::PreOrder] nodes will be
/// present.  Children will be visited before any of their out_neighbours when
/// the graph is acyclic. Otherwise they will be visited in a Dfs order starting
/// from the first child(as returned by hugr.children()).
pub fn children(hugr: &impl HugrView, root: Node) -> impl '_ + Iterator<Item = VisitOrder<Node>> {
    let region = SiblingGraph::try_new(hugr, root).expect("Failed to create SiblingGraph");
    let walker = if OpTag::DataflowParent.is_superset(hugr.get_optype(root).tag()) {
        // These nodes never have cycles, so a Topo will give all nodes in topological order
        SiblingWalker::Dataflow(Topo::new(region.as_petgraph()))
    } else {
        // These nodes may have cycles, but they must have an Input node, whence
        // we can DFS.
        SiblingWalker::ControlFlow(Dfs::new(region.as_petgraph(), root))
    };
    // walker will visit all children, then root. We wrap those in PostOrder,
    // and prefix with PreOrder(root)
    std::iter::once(VisitOrder::PreOrder(root))
        .chain(walker.iter(region).map(VisitOrder::PostOrder))
}

/// Return an iterator over root and the immediate children of `root`.
/// `root` will be the last node returned. Children of `root` will be visited
/// before any of their out_neighbours when the graph is acyclic. Otherwise they
/// will be visited in a Dfs order starting from the first child(as returned by
/// hugr.children()).
pub fn children_postorder(hugr: &impl HugrView, root: Node) -> impl '_ + Iterator<Item = Node> {
    children(hugr, root).filter_map(VisitOrder::into_postorder)
}

enum RecurseIterator<'a> {
    One(std::iter::Once<VisitOrder<Node>>),
    Many(Box<dyn 'a + Iterator<Item = VisitOrder<Node>>>),
}

impl<'a> Iterator for RecurseIterator<'a> {
    type Item = VisitOrder<Node>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::One(ref mut once) => once.next(),
            Self::Many(ref mut many) => many.next(),
        }
    }
}

impl<'a> RecurseIterator<'a> {
    fn new(hugr: &'a impl HugrView, root: Node, node: VisitOrder<Node>) -> Self {
        match node {
            VisitOrder::PostOrder(y) if y == root || hugr.children(y).next().is_none() => {
                Self::One(std::iter::once(VisitOrder::PostOrder(y)))
            }
            VisitOrder::PostOrder(y) => Self::Many(Box::new(recursive_children(hugr, y))),
            x => Self::One(std::iter::once(x)),
        }
    }
}

/// Return an iterator over root and transitive children of root.
/// `[Preorder](node)` items will be returned for nodes with children, before
/// any of those children. `[PostOrder](node)` items will be returned for all
/// nodes, after any children of that node.  Nodes will be visited before any of
/// their `out_neighbours` when their [SiblingGraph] is acyclic. Otherwise
/// they will be visited in a Dfs order starting from the first child of the
/// parent (as returned by hugr.children()).
pub fn recursive_children(
    hugr: &impl HugrView,
    root: Node,
) -> impl '_ + Iterator<Item = VisitOrder<Node>> {
    children(hugr, root).flat_map(move |x| RecurseIterator::new(hugr, root, x))
}

/// Return an iterator over root and transitive children of root. Nodes will be
/// visited before any of their `out_neighbours` when their [SiblingGraph]
/// is acyclic. Otherwise they will be visited in a Dfs order starting from the
/// first child of the parent (as returned by hugr.children()).
pub fn recursive_children_postorder(
    hugr: &impl HugrView,
    root: Node,
) -> impl '_ + Iterator<Item = Node> {
    recursive_children(hugr, root).filter_map(VisitOrder::into_postorder)
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use rstest::rstest;

    use crate::{
        builder::{
            test::simple_dfg_hugr, Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder,
        },
        extension::prelude::USIZE_T,
        hugr::hugrmut::sealed::HugrMutInternals,
        ops,
        ops::OpType,
        type_row,
        types::FunctionType,
    };

    use super::*;

    fn node_string(hugr: &impl HugrView, node: Node) -> &'static str {
        match *hugr.get_optype(node) {
            OpType::FuncDefn(_) => "F",
            OpType::DFG(_) => "D",
            OpType::Input(_) => "I",
            OpType::Output(_) => "O",
            OpType::LeafOp(ops::LeafOp::Noop { .. }) => "N",
            _ => "?",
        }
    }

    fn visit_nodeorder_string(hugr: &impl HugrView, node: VisitOrder<Node>) -> String {
        match node {
            VisitOrder::PostOrder(x) => node_string(hugr, x).to_string(),
            VisitOrder::PreOrder(x) => format!("Pre({})", node_string(hugr, x)),
        }
    }

    fn visitorder_string(
        hugr: &impl HugrView,
        iter: impl IntoIterator<Item = VisitOrder<Node>>,
    ) -> String {
        let mut s = String::new();
        for x in iter {
            s += &visit_nodeorder_string(hugr, x)
        }
        s
    }

    fn visitnode_string(hugr: &impl HugrView, iter: impl IntoIterator<Item = Node>) -> String {
        let mut s = String::new();
        for x in iter {
            s += node_string(hugr, x)
        }
        s
    }

    #[rstest]
    fn simple(simple_dfg_hugr: crate::Hugr) {
        assert_eq!(
            visitorder_string(
                &simple_dfg_hugr,
                super::recursive_children(&simple_dfg_hugr, simple_dfg_hugr.root())
            ),
            "Pre(D)IOD"
        );
        assert_eq!(
            visitnode_string(
                &simple_dfg_hugr,
                super::recursive_children_postorder(&simple_dfg_hugr, simple_dfg_hugr.root())
            ),
            "IOD"
        );
    }

    #[test]
    fn out_of_order() -> Result<(), Box<dyn Error>> {
        use crate::ops::handle::NodeHandle;
        let sig = FunctionType::new_endo(type_row![USIZE_T]);
        let mut fun_builder = FunctionBuilder::new("f3", sig.clone().into())?;
        let [i] = fun_builder.input_wires_arr();
        let noop = fun_builder.add_dataflow_op(ops::LeafOp::Noop { ty: USIZE_T }, [i])?;
        let dfg_builder = fun_builder.dfg_builder(sig.clone(), None, [noop.out_wire(0)])?;
        let [i1] = dfg_builder.input_wires_arr();
        let dfg = dfg_builder.finish_with_outputs([i1])?;
        let mut h = fun_builder.finish_prelude_hugr_with_outputs([dfg.out_wire(0)])?;
        h.hugr_mut()
            .move_before_sibling(dfg.handle().node(), noop.handle().node())?;

        assert_eq!(
            visitorder_string(&h, super::children(&h, h.root())),
            "Pre(F)INDOF"
        );
        assert_eq!(
            visitnode_string(&h, super::children_postorder(&h, h.root())),
            "INDOF"
        );
        assert_eq!(
            visitorder_string(&h, super::recursive_children(&h, h.root())),
            "Pre(F)INPre(D)IODOF"
        );
        assert_eq!(
            visitnode_string(&h, super::recursive_children_postorder(&h, h.root())),
            "INIODOF"
        );
        Ok(())
    }
}
