//! Implementation of the core hugr traits for different wrappers of a `Hugr`.

use std::{borrow::Cow, rc::Rc, sync::Arc};

use super::HugrView;
use crate::hugr::internal::{HugrInternals, HugrMutInternals};
use crate::hugr::{HugrMut, hugrmut::InsertForestResult};

macro_rules! hugr_internal_methods {
    // The extra ident here is because invocations of the macro cannot pass `self` as argument
    ($arg:ident, $e:expr) => {
        delegate::delegate! {
            to ({let $arg=self; $e}) {
                fn region_portgraph(&self, parent: Self::Node) -> (portgraph::view::FlatRegion<'_, Self::RegionPortgraph<'_>>, Self::RegionPortgraphNodes);
                fn node_metadata_map(&self, node: Self::Node) -> &crate::hugr::NodeMetadataMap;
            }
        }
    };
}
pub(crate) use hugr_internal_methods;

macro_rules! hugr_view_methods {
    // The extra ident here is because invocations of the macro cannot pass `self` as argument
    ($arg:ident, $e:expr) => {
        delegate::delegate! {
            to ({let $arg=self; $e}) {
                fn entrypoint(&self) -> Self::Node;
                fn entrypoint_optype(&self) -> &crate::ops::OpType;
                fn module_root(&self) -> Self::Node;
                fn contains_node(&self, node: Self::Node) -> bool;
                fn get_parent(&self, node: Self::Node) -> Option<Self::Node>;
                fn get_metadata(&self, node: Self::Node, key: impl AsRef<str>) -> Option<&crate::hugr::NodeMetadata>;
                fn get_optype(&self, node: Self::Node) -> &crate::ops::OpType;
                fn num_nodes(&self) -> usize;
                fn num_edges(&self) -> usize;
                fn num_ports(&self, node: Self::Node, dir: crate::Direction) -> usize;
                fn num_inputs(&self, node: Self::Node) -> usize;
                fn num_outputs(&self, node: Self::Node) -> usize;
                fn nodes(&self) -> impl Iterator<Item = Self::Node> + Clone;
                fn node_ports(&self, node: Self::Node, dir: crate::Direction) -> impl Iterator<Item = crate::Port> + Clone;
                fn node_outputs(&self, node: Self::Node) -> impl Iterator<Item = crate::OutgoingPort> + Clone;
                fn node_inputs(&self, node: Self::Node) -> impl Iterator<Item = crate::IncomingPort> + Clone;
                fn all_node_ports(&self, node: Self::Node) -> impl Iterator<Item = crate::Port> + Clone;
                fn linked_ports(&self, node: Self::Node, port: impl Into<crate::Port>) -> impl Iterator<Item = (Self::Node, crate::Port)> + Clone;
                fn all_linked_ports(&self, node: Self::Node, dir: crate::Direction) -> itertools::Either<impl Iterator<Item = (Self::Node, crate::OutgoingPort)>, impl Iterator<Item = (Self::Node, crate::IncomingPort)>>;
                fn all_linked_outputs(&self, node: Self::Node) -> impl Iterator<Item = (Self::Node, crate::OutgoingPort)>;
                fn all_linked_inputs(&self, node: Self::Node) -> impl Iterator<Item = (Self::Node, crate::IncomingPort)>;
                fn single_linked_port(&self, node: Self::Node, port: impl Into<crate::Port>) -> Option<(Self::Node, crate::Port)>;
                fn single_linked_output(&self, node: Self::Node, port: impl Into<crate::IncomingPort>) -> Option<(Self::Node, crate::OutgoingPort)>;
                fn single_linked_input(&self, node: Self::Node, port: impl Into<crate::OutgoingPort>) -> Option<(Self::Node, crate::IncomingPort)>;
                fn linked_outputs(&self, node: Self::Node, port: impl Into<crate::IncomingPort>) -> impl Iterator<Item = (Self::Node, crate::OutgoingPort)>;
                fn linked_inputs(&self, node: Self::Node, port: impl Into<crate::OutgoingPort>) -> impl Iterator<Item = (Self::Node, crate::IncomingPort)>;
                fn node_connections(&self, node: Self::Node, other: Self::Node) -> impl Iterator<Item = [crate::Port; 2]> + Clone;
                fn is_linked(&self, node: Self::Node, port: impl Into<crate::Port>) -> bool;
                fn children(&self, node: Self::Node) -> impl DoubleEndedIterator<Item = Self::Node> + Clone;
                fn descendants(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone;
                fn first_child(&self, node: Self::Node) -> Option<Self::Node>;
                fn neighbours(&self, node: Self::Node, dir: crate::Direction) -> impl Iterator<Item = Self::Node> + Clone;
                fn all_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone;
                fn mermaid_string(&self) -> String;
                #[allow(deprecated)]
                fn mermaid_string_with_config(&self, config: crate::hugr::views::render::RenderConfig<Self::Node>) -> String;
                fn mermaid_string_with_formatter(&self, #[into] formatter: crate::hugr::views::render::MermaidFormatter<Self>) -> String;
                fn dot_string(&self) -> String;
                fn static_source(&self, node: Self::Node) -> Option<Self::Node>;
                fn static_targets(&self, node: Self::Node) -> Option<impl Iterator<Item = (Self::Node, crate::IncomingPort)>>;
                fn value_types(&self, node: Self::Node, dir: crate::Direction) -> impl Iterator<Item = (crate::Port, crate::types::Type)>;
                fn extensions(&self) -> &crate::extension::ExtensionRegistry;
                fn validate(&self) -> Result<(), crate::hugr::ValidationError<Self::Node>>;
                fn extract_hugr(&self, parent: Self::Node) -> (crate::Hugr, impl crate::hugr::views::ExtractionResult<Self::Node> + 'static);
            }
        }
    }
}
pub(crate) use hugr_view_methods;

macro_rules! hugr_mut_internal_methods {
    // The extra ident here is because invocations of the macro cannot pass `self` as argument
    ($arg:ident, $e:expr) => {
        delegate::delegate! {
            to ({let $arg=self; $e}) {
                fn set_module_root(&mut self, root: Self::Node);
                fn set_num_ports(&mut self, node: Self::Node, incoming: usize, outgoing: usize);
                fn add_ports(&mut self, node: Self::Node, direction: crate::Direction, amount: isize) -> std::ops::Range<usize>;
                fn insert_ports(&mut self, node: Self::Node, direction: crate::Direction, index: usize, amount: usize) -> std::ops::Range<usize>;
                fn set_parent(&mut self, node: Self::Node, parent: Self::Node);
                fn move_after_sibling(&mut self, node: Self::Node, after: Self::Node);
                fn move_before_sibling(&mut self, node: Self::Node, before: Self::Node);
                fn replace_op(&mut self, node: Self::Node, op: impl Into<crate::ops::OpType>) -> crate::ops::OpType;
                fn optype_mut(&mut self, node: Self::Node) -> &mut crate::ops::OpType;
                fn node_metadata_map_mut(&mut self, node: Self::Node) -> &mut crate::hugr::NodeMetadataMap;
                fn extensions_mut(&mut self) -> &mut crate::extension::ExtensionRegistry;
            }
        }
    };
}
pub(crate) use hugr_mut_internal_methods;

macro_rules! hugr_mut_methods {
    // The extra ident here is because invocations of the macro cannot pass `self` as argument
    ($arg:ident, $e:expr) => {
        delegate::delegate! {
            to ({let $arg=self; $e}) {
                fn set_entrypoint(&mut self, root: Self::Node);
                fn get_metadata_mut(&mut self, node: Self::Node, key: impl AsRef<str>) -> &mut crate::hugr::NodeMetadata;
                fn set_metadata(&mut self, node: Self::Node, key: impl AsRef<str>, metadata: impl Into<crate::hugr::NodeMetadata>);
                fn remove_metadata(&mut self, node: Self::Node, key: impl AsRef<str>);
                fn add_node_with_parent(&mut self, parent: Self::Node, op: impl Into<crate::ops::OpType>) -> Self::Node;
                fn add_node_before(&mut self, sibling: Self::Node, nodetype: impl Into<crate::ops::OpType>) -> Self::Node;
                fn add_node_after(&mut self, sibling: Self::Node, op: impl Into<crate::ops::OpType>) -> Self::Node;
                fn remove_node(&mut self, node: Self::Node) -> crate::ops::OpType;
                fn remove_subtree(&mut self, node: Self::Node);
                fn copy_descendants(&mut self, root: Self::Node, new_parent: Self::Node, subst: Option<crate::types::Substitution>) -> std::collections::BTreeMap<Self::Node, Self::Node>;
                fn connect(&mut self, src: Self::Node, src_port: impl Into<crate::OutgoingPort>, dst: Self::Node, dst_port: impl Into<crate::IncomingPort>);
                fn disconnect(&mut self, node: Self::Node, port: impl Into<crate::Port>);
                fn add_other_edge(&mut self, src: Self::Node, dst: Self::Node) -> (crate::OutgoingPort, crate::IncomingPort);
                fn insert_forest(&mut self, other: crate::Hugr, roots: impl IntoIterator<Item=(crate::Node, Self::Node)>) -> InsertForestResult<crate::Node, Self::Node>;
                fn insert_view_forest<Other: crate::hugr::HugrView>(&mut self, other: &Other, nodes: impl Iterator<Item=Other::Node> + Clone, roots: impl IntoIterator<Item=(Other::Node, Self::Node)>) -> InsertForestResult<Other::Node, Self::Node>;
                fn use_extension(&mut self, extension: impl Into<std::sync::Arc<crate::extension::Extension>>);
                fn use_extensions<Reg>(&mut self, registry: impl IntoIterator<Item = Reg>) where crate::extension::ExtensionRegistry: Extend<Reg>;
            }
        }
    };
}
pub(crate) use hugr_mut_methods;

// -------- Immutable borrow
impl<T: HugrView> HugrInternals for &T {
    type RegionPortgraph<'p>
        = T::RegionPortgraph<'p>
    where
        Self: 'p;

    type Node = T::Node;

    type RegionPortgraphNodes = T::RegionPortgraphNodes;

    hugr_internal_methods! {this, *this}
}
impl<T: HugrView> HugrView for &T {
    hugr_view_methods! {this, *this}
}

// -------- Mutable borrow
impl<T: HugrView> HugrInternals for &mut T {
    type RegionPortgraph<'p>
        = T::RegionPortgraph<'p>
    where
        Self: 'p;

    type Node = T::Node;

    type RegionPortgraphNodes = T::RegionPortgraphNodes;

    hugr_internal_methods! {this, &**this}
}
impl<T: HugrView> HugrView for &mut T {
    hugr_view_methods! {this, &**this}
}
impl<T: HugrMutInternals> HugrMutInternals for &mut T {
    hugr_mut_internal_methods! {this, &mut **this}
}
impl<T: HugrMut> HugrMut for &mut T {
    hugr_mut_methods! {this, &mut **this}
}

// -------- Rc
impl<T: HugrView> HugrInternals for Rc<T> {
    type RegionPortgraph<'p>
        = T::RegionPortgraph<'p>
    where
        Self: 'p;

    type Node = T::Node;

    type RegionPortgraphNodes = T::RegionPortgraphNodes;

    hugr_internal_methods! {this, this.as_ref()}
}
impl<T: HugrView> HugrView for Rc<T> {
    hugr_view_methods! {this, this.as_ref()}
}

// -------- Arc
impl<T: HugrView> HugrInternals for Arc<T> {
    type RegionPortgraph<'p>
        = T::RegionPortgraph<'p>
    where
        Self: 'p;

    type Node = T::Node;

    type RegionPortgraphNodes = T::RegionPortgraphNodes;

    hugr_internal_methods! {this, this.as_ref()}
}
impl<T: HugrView> HugrView for Arc<T> {
    hugr_view_methods! {this, this.as_ref()}
}

// -------- Box
impl<T: HugrView> HugrInternals for Box<T> {
    type RegionPortgraph<'p>
        = T::RegionPortgraph<'p>
    where
        Self: 'p;

    type Node = T::Node;

    type RegionPortgraphNodes = T::RegionPortgraphNodes;

    hugr_internal_methods! {this, this.as_ref()}
}
impl<T: HugrView> HugrView for Box<T> {
    hugr_view_methods! {this, this.as_ref()}
}
impl<T: HugrMutInternals> HugrMutInternals for Box<T> {
    hugr_mut_internal_methods! {this, this.as_mut()}
}
impl<T: HugrMut> HugrMut for Box<T> {
    hugr_mut_methods! {this, this.as_mut()}
}

// -------- Cow
impl<T: HugrView + ToOwned> HugrInternals for Cow<'_, T> {
    type RegionPortgraph<'p>
        = T::RegionPortgraph<'p>
    where
        Self: 'p;

    type Node = T::Node;

    type RegionPortgraphNodes = T::RegionPortgraphNodes;

    hugr_internal_methods! {this, this.as_ref()}
}
impl<T: HugrView + ToOwned> HugrView for Cow<'_, T> {
    hugr_view_methods! {this, this.as_ref()}
}
impl<T> HugrMutInternals for Cow<'_, T>
where
    T: HugrMutInternals + ToOwned,
    <T as ToOwned>::Owned: HugrMutInternals<Node = T::Node>,
{
    hugr_mut_internal_methods! {this, this.to_mut()}
}
impl<T> HugrMut for Cow<'_, T>
where
    T: HugrMut + ToOwned,
    <T as ToOwned>::Owned: HugrMut<Node = T::Node>,
{
    hugr_mut_methods! {this, this.to_mut()}
}

#[cfg(test)]
mod test {
    use std::{rc::Rc, sync::Arc};

    use crate::{Hugr, HugrView};

    struct ViewWrapper<H>(H);
    impl<H: HugrView> ViewWrapper<H> {
        fn nodes(&self) -> impl Iterator<Item = H::Node> + '_ {
            self.0.nodes()
        }
    }

    #[test]
    fn test_refs_to_view() {
        let h = Hugr::default();
        let v = ViewWrapper(&h);
        let c = h.nodes().count();
        assert_eq!(v.nodes().count(), c);

        let v2 = ViewWrapper(h.with_entrypoint(h.entrypoint()));
        assert_eq!(v2.nodes().count(), v.nodes().count());
        assert_eq!(ViewWrapper(&v2.0).nodes().count(), v.nodes().count());

        let vh = ViewWrapper(h);
        assert_eq!(vh.nodes().count(), c);
        let h: Hugr = vh.0;
        assert_eq!(h.nodes().count(), c);

        let vb = ViewWrapper(Box::new(&h));
        assert_eq!(vb.nodes().count(), c);
        let va = ViewWrapper(Arc::new(h));
        assert_eq!(va.nodes().count(), c);
        let h = Arc::try_unwrap(va.0).unwrap();
        let vr = Rc::new(&h);
        assert_eq!(ViewWrapper(&vr).nodes().count(), h.nodes().count());
    }
}
