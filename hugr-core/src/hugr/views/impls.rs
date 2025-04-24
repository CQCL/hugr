//! Implementation of the core hugr traits for different wrappers of a `Hugr`.

use std::{borrow::Cow, rc::Rc, sync::Arc};

use super::HugrView;
use super::RootTagged;
use crate::hugr::internal::{HugrInternals, HugrMutInternals};
use crate::hugr::HugrMut;
use crate::Hugr;
use crate::Node;

macro_rules! hugr_internal_methods {
    // The extra ident here is because invocations of the macro cannot pass `self` as argument
    ($arg:ident, $e:expr) => {
        delegate::delegate! {
            to ({let $arg=self; $e}) {
                fn portgraph(&self) -> Self::Portgraph<'_>;
                fn hierarchy(&self) -> Cow<'_, portgraph::Hierarchy>;
                fn base_hugr(&self) -> &crate::Hugr;
                fn root_node(&self) -> Self::Node;
                fn get_pg_index(&self, node: impl crate::ops::handle::NodeHandle<Self::Node>) -> portgraph::NodeIndex;
                fn get_node(&self, index: portgraph::NodeIndex) -> Self::Node;
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
                fn root(&self) -> Self::Node;
                fn root_type(&self) -> &crate::ops::OpType;
                fn contains_node(&self, node: Self::Node) -> bool;
                fn valid_node(&self, node: Self::Node) -> bool;
                fn valid_non_root(&self, node: Self::Node) -> bool;
                fn get_parent(&self, node: Self::Node) -> Option<Self::Node>;
                fn get_optype(&self, node: Self::Node) -> &crate::ops::OpType;
                fn get_metadata(&self, node: Self::Node, key: impl AsRef<str>) -> Option<&crate::hugr::NodeMetadata>;
                fn get_node_metadata(&self, node: Self::Node) -> Option<&crate::hugr::NodeMetadataMap>;
                fn node_count(&self) -> usize;
                fn edge_count(&self) -> usize;
                fn nodes(&self) -> impl Iterator<Item = Self::Node> + Clone;
                fn node_ports(&self, node: Self::Node, dir: crate::Direction) -> impl Iterator<Item = crate::Port> + Clone;
                fn node_outputs(&self, node: Self::Node) -> impl Iterator<Item = crate::OutgoingPort> + Clone;
                fn node_inputs(&self, node: Self::Node) -> impl Iterator<Item = crate::IncomingPort> + Clone;
                fn all_node_ports(&self, node: Self::Node) -> impl Iterator<Item = crate::Port> + Clone;
                fn linked_ports(
                    &self,
                    node: Self::Node,
                    port: impl Into<crate::Port>,
                ) -> impl Iterator<Item = (Self::Node, crate::Port)> + Clone;
                fn all_linked_ports(
                    &self,
                    node: Self::Node,
                    dir: crate::Direction,
                ) -> itertools::Either<
                    impl Iterator<Item = (Self::Node, crate::OutgoingPort)>,
                    impl Iterator<Item = (Self::Node, crate::IncomingPort)>,
                >;
                fn all_linked_outputs(&self, node: Self::Node) -> impl Iterator<Item = (Self::Node, crate::OutgoingPort)>;
                fn all_linked_inputs(&self, node: Self::Node) -> impl Iterator<Item = (Self::Node, crate::IncomingPort)>;
                fn single_linked_port(&self, node: Self::Node, port: impl Into<crate::Port>) -> Option<(Self::Node, crate::Port)>;
                fn single_linked_output(&self, node: Self::Node, port: impl Into<crate::IncomingPort>) -> Option<(Self::Node, crate::OutgoingPort)>;
                fn single_linked_input(&self, node: Self::Node, port: impl Into<crate::OutgoingPort>) -> Option<(Self::Node, crate::IncomingPort)>;
                fn linked_outputs(&self, node: Self::Node, port: impl Into<crate::IncomingPort>) -> impl Iterator<Item = (Self::Node, crate::OutgoingPort)>;
                fn linked_inputs(&self, node: Self::Node, port: impl Into<crate::OutgoingPort>) -> impl Iterator<Item = (Self::Node, crate::IncomingPort)>;
                fn node_connections(&self, node: Self::Node, other: Self::Node) -> impl Iterator<Item = [crate::Port; 2]> + Clone;
                fn is_linked(&self, node: Self::Node, port: impl Into<crate::Port>) -> bool;
                fn num_ports(&self, node: Self::Node, dir: crate::Direction) -> usize;
                fn num_inputs(&self, node: Self::Node) -> usize;
                fn num_outputs(&self, node: Self::Node) -> usize;
                fn children(&self, node: Self::Node) -> impl DoubleEndedIterator<Item = Self::Node> + Clone;
                fn first_child(&self, node: Self::Node) -> Option<Self::Node>;
                fn neighbours(&self, node: Self::Node, dir: crate::Direction) -> impl Iterator<Item = Self::Node> + Clone;
                fn input_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone;
                fn output_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone;
                fn all_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone;
                fn get_io(&self, node: Self::Node) -> Option<[Self::Node; 2]>;
                fn inner_function_type(&self) -> Option<Cow<'_, crate::types::Signature>>;
                fn poly_func_type(&self) -> Option<crate::types::PolyFuncType>;
                // TODO: cannot use delegate here. `PetgraphWrapper` is a thin
                // wrapper around `Self`, so falling back to the default impl
                // should be harmless.
                // fn as_petgraph(&self) -> PetgraphWrapper<'_, Self>;
                fn mermaid_string(&self) -> String;
                fn mermaid_string_with_config(&self, config: crate::hugr::views::render::RenderConfig) -> String;
                fn dot_string(&self) -> String;
                fn static_source(&self, node: Self::Node) -> Option<Self::Node>;
                fn static_targets(&self, node: Self::Node) -> Option<impl Iterator<Item = (Self::Node, crate::IncomingPort)>>;
                fn signature(&self, node: Self::Node) -> Option<Cow<'_, crate::types::Signature>>;
                fn value_types(&self, node: Self::Node, dir: crate::Direction) -> impl Iterator<Item = (crate::Port, crate::types::Type)>;
                fn in_value_types(&self, node: Self::Node) -> impl Iterator<Item = (crate::IncomingPort, crate::types::Type)>;
                fn out_value_types(&self, node: Self::Node) -> impl Iterator<Item = (crate::OutgoingPort, crate::types::Type)>;
                fn extensions(&self) -> &crate::extension::ExtensionRegistry;
                fn validate(&self) -> Result<(), crate::hugr::ValidationError>;
                fn validate_no_extensions(&self) -> Result<(), crate::hugr::ValidationError>;
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
                fn set_root(&mut self, root: Self::Node);
                fn set_num_ports(&mut self, node: Self::Node, incoming: usize, outgoing: usize);
                fn add_ports(&mut self, node: Self::Node, direction: crate::Direction, amount: isize) -> std::ops::Range<usize>;
                fn insert_ports(&mut self, node: Self::Node, direction: crate::Direction, index: usize, amount: usize) -> std::ops::Range<usize>;
                fn set_parent(&mut self, node: Self::Node, parent: Self::Node);
                fn move_after_sibling(&mut self, node: Self::Node, after: Self::Node);
                fn move_before_sibling(&mut self, node: Self::Node, before: Self::Node);
                fn replace_op(&mut self, node: Self::Node, op: impl Into<crate::ops::OpType>) -> Result<crate::ops::OpType, crate::hugr::HugrError>;
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
                fn add_node_with_parent(&mut self, parent: Self::Node, op: impl Into<crate::ops::OpType>) -> Self::Node;
                fn add_node_before(&mut self, sibling: Self::Node, nodetype: impl Into<crate::ops::OpType>) -> Self::Node;
                fn add_node_after(&mut self, sibling: Self::Node, op: impl Into<crate::ops::OpType>) -> Self::Node;
                fn remove_node(&mut self, node: Self::Node) -> crate::ops::OpType;
                fn remove_subtree(&mut self, node: Self::Node);
                fn copy_descendants(&mut self, root: Self::Node, new_parent: Self::Node, subst: Option<crate::types::Substitution>) -> std::collections::BTreeMap<Self::Node, Self::Node>;
                fn connect(&mut self, src: Self::Node, src_port: impl Into<crate::OutgoingPort>, dst: Self::Node, dst_port: impl Into<crate::IncomingPort>);
                fn disconnect(&mut self, node: Self::Node, port: impl Into<crate::Port>);
                fn add_other_edge(&mut self, src: Self::Node, dst: Self::Node) -> (crate::OutgoingPort, crate::IncomingPort);
                fn insert_hugr(&mut self, root: Self::Node, other: crate::Hugr) -> crate::hugr::hugrmut::InsertionResult<crate::Node, Self::Node>;
                fn insert_from_view<Other: crate::hugr::HugrView>(&mut self, root: Self::Node, other: &Other) -> crate::hugr::hugrmut::InsertionResult<Other::Node, Self::Node>;
                fn insert_subgraph<Other: crate::hugr::HugrView>(&mut self, root: Self::Node, other: &Other, subgraph: &crate::hugr::views::SiblingSubgraph<Other::Node>) -> std::collections::HashMap<Other::Node, Self::Node>;
            }
        }
    };
}
pub(crate) use hugr_mut_methods;

// -------- Base Hugr implementation
impl RootTagged for Hugr {
    type RootHandle = Node;
}

// -------- Immutable borrow
impl<T: HugrView> HugrInternals for &T {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    hugr_internal_methods! {this, *this}
}
impl<T: HugrView> HugrView for &T {
    hugr_view_methods! {this, *this}
}
impl<T: RootTagged> RootTagged for &T {
    type RootHandle = T::RootHandle;
}

// -------- Mutable borrow
impl<T: HugrView> HugrInternals for &mut T {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    hugr_internal_methods! {this, &**this}
}
impl<T: HugrView> HugrView for &mut T {
    hugr_view_methods! {this, &**this}
}
impl<T: RootTagged> RootTagged for &mut T {
    type RootHandle = T::RootHandle;
}
impl<T: HugrMutInternals> HugrMutInternals for &mut T {
    hugr_mut_internal_methods! {this, &mut **this}
}
impl<T: HugrMut> HugrMut for &mut T {
    hugr_mut_methods! {this, &mut **this}
}

// -------- Rc
impl<T: HugrView> HugrInternals for Rc<T> {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    hugr_internal_methods! {this, this.as_ref()}
}
impl<T: HugrView> HugrView for Rc<T> {
    hugr_view_methods! {this, this.as_ref()}
}
impl<T: RootTagged> RootTagged for Rc<T> {
    type RootHandle = T::RootHandle;
}

// -------- Arc
impl<T: HugrView> HugrInternals for Arc<T> {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    hugr_internal_methods! {this, this.as_ref()}
}
impl<T: HugrView> HugrView for Arc<T> {
    hugr_view_methods! {this, this.as_ref()}
}
impl<T: RootTagged> RootTagged for Arc<T> {
    type RootHandle = T::RootHandle;
}

// -------- Box
impl<T: HugrView> HugrInternals for Box<T> {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    hugr_internal_methods! {this, this.as_ref()}
}
impl<T: HugrView> HugrView for Box<T> {
    hugr_view_methods! {this, this.as_ref()}
}
impl<T: RootTagged> RootTagged for Box<T> {
    type RootHandle = T::RootHandle;
}
impl<T: HugrMutInternals> HugrMutInternals for Box<T> {
    hugr_mut_internal_methods! {this, this.as_mut()}
}
impl<T: HugrMut> HugrMut for Box<T> {
    hugr_mut_methods! {this, this.as_mut()}
}

// -------- Cow
impl<T: HugrView + ToOwned> HugrInternals for Cow<'_, T> {
    type Portgraph<'p>
        = T::Portgraph<'p>
    where
        Self: 'p;
    type Node = T::Node;

    hugr_internal_methods! {this, this.as_ref()}
}
impl<T: HugrView + ToOwned> HugrView for Cow<'_, T> {
    hugr_view_methods! {this, this.as_ref()}
}
impl<T: RootTagged + ToOwned> RootTagged for Cow<'_, T> {
    type RootHandle = T::RootHandle;
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

    use crate::hugr::views::{DescendantsGraph, HierarchyView};
    use crate::{Hugr, HugrView, Node};

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
        let v2 = ViewWrapper(DescendantsGraph::<Node>::try_new(&h, h.root()).unwrap());
        // v2 owns the DescendantsGraph, but that only borrows `h`, so we still have both
        assert_eq!(v2.nodes().count(), v.nodes().count());
        // And we can borrow the DescendantsGraph, even just a reference to that counts as a HugrView
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
