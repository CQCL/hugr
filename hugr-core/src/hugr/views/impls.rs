use std::{borrow::Cow, rc::Rc, sync::Arc};

use delegate::delegate;
use itertools::Either;

use super::{render::RenderConfig, HugrView, RootChecked};
use crate::{
    extension::ExtensionRegistry,
    hugr::{NodeMetadata, NodeMetadataMap, ValidationError},
    ops::OpType,
    types::{PolyFuncType, Signature, Type},
    Direction, Hugr, IncomingPort, Node, OutgoingPort, Port,
};

macro_rules! hugr_view_methods {
    // The extra ident here is because invocations of the macro cannot pass `self` as argument
    ($arg:ident, $e:expr) => {
        delegate! {
            to ({let $arg=self; $e}) {
                fn root(&self) -> Node;
                fn root_type(&self) -> &OpType;
                fn contains_node(&self, node: Node) -> bool;
                fn valid_node(&self, node: Node) -> bool;
                fn valid_non_root(&self, node: Node) -> bool;
                fn get_parent(&self, node: Node) -> Option<Node>;
                fn get_optype(&self, node: Node) -> &OpType;
                fn get_metadata(&self, node: Node, key: impl AsRef<str>) -> Option<&NodeMetadata>;
                fn get_node_metadata(&self, node: Node) -> Option<&NodeMetadataMap>;
                fn node_count(&self) -> usize;
                fn edge_count(&self) -> usize;
                fn nodes(&self) -> impl Iterator<Item = Node> + Clone;
                fn node_ports(&self, node: Node, dir: Direction) -> impl Iterator<Item = Port> + Clone;
                fn node_outputs(&self, node: Node) -> impl Iterator<Item = OutgoingPort> + Clone;
                fn node_inputs(&self, node: Node) -> impl Iterator<Item = IncomingPort> + Clone;
                fn all_node_ports(&self, node: Node) -> impl Iterator<Item = Port> + Clone;
                fn linked_ports(
                    &self,
                    node: Node,
                    port: impl Into<Port>,
                ) -> impl Iterator<Item = (Node, Port)> + Clone;
                fn all_linked_ports(
                    &self,
                    node: Node,
                    dir: Direction,
                ) -> Either<
                    impl Iterator<Item = (Node, OutgoingPort)>,
                    impl Iterator<Item = (Node, IncomingPort)>,
                >;
                fn all_linked_outputs(&self, node: Node) -> impl Iterator<Item = (Node, OutgoingPort)>;
                fn all_linked_inputs(&self, node: Node) -> impl Iterator<Item = (Node, IncomingPort)>;
                fn single_linked_port(&self, node: Node, port: impl Into<Port>) -> Option<(Node, Port)>;
                fn single_linked_output(&self, node: Node, port: impl Into<IncomingPort>) -> Option<(Node, OutgoingPort)>;
                fn single_linked_input(&self, node: Node, port: impl Into<OutgoingPort>) -> Option<(Node, IncomingPort)>;
                fn linked_outputs(&self, node: Node, port: impl Into<IncomingPort>) -> impl Iterator<Item = (Node, OutgoingPort)>;
                fn linked_inputs(&self, node: Node, port: impl Into<OutgoingPort>) -> impl Iterator<Item = (Node, IncomingPort)>;
                fn node_connections(&self, node: Node, other: Node) -> impl Iterator<Item = [Port; 2]> + Clone;
                fn is_linked(&self, node: Node, port: impl Into<Port>) -> bool;
                fn num_ports(&self, node: Node, dir: Direction) -> usize;
                fn num_inputs(&self, node: Node) -> usize;
                fn num_outputs(&self, node: Node) -> usize;
                fn children(&self, node: Node) -> impl DoubleEndedIterator<Item = Node> + Clone;
                fn first_child(&self, node: Node) -> Option<Node>;
                fn neighbours(&self, node: Node, dir: Direction) -> impl Iterator<Item = Node> + Clone;
                fn input_neighbours(&self, node: Node) -> impl Iterator<Item = Node> + Clone;
                fn output_neighbours(&self, node: Node) -> impl Iterator<Item = Node> + Clone;
                fn all_neighbours(&self, node: Node) -> impl Iterator<Item = Node> + Clone;
                fn get_io(&self, node: Node) -> Option<[Node; 2]>;
                fn inner_function_type(&self) -> Option<Cow<'_, Signature>>;
                fn poly_func_type(&self) -> Option<PolyFuncType>;
                // TODO: cannot use delegate: return type of `as_petgraph`
                // depends on Self
                // fn as_petgraph(&self) -> PetgraphWrapper<'_, Self>;
                fn mermaid_string(&self) -> String;
                fn mermaid_string_with_config(&self, config: RenderConfig) -> String;
                fn dot_string(&self) -> String;
                fn static_source(&self, node: Node) -> Option<Node>;
                fn static_targets(&self, node: Node) -> Option<impl Iterator<Item = (Node, IncomingPort)>>;
                fn signature(&self, node: Node) -> Option<Cow<'_, Signature>>;
                fn value_types(&self, node: Node, dir: Direction) -> impl Iterator<Item = (Port, Type)>;
                fn in_value_types(&self, node: Node) -> impl Iterator<Item = (IncomingPort, Type)>;
                fn out_value_types(&self, node: Node) -> impl Iterator<Item = (OutgoingPort, Type)>;
                fn extensions(&self) -> &ExtensionRegistry;
                fn validate(&self) -> Result<(), ValidationError>;
                fn validate_no_extensions(&self) -> Result<(), ValidationError>;
            }
        }
    }
}

impl<T: HugrView> HugrView for &T {
    hugr_view_methods! {this, *this}
}

impl<T: HugrView> HugrView for &mut T {
    hugr_view_methods! {this, &**this}
}

impl<T: HugrView> HugrView for Rc<T> {
    hugr_view_methods! {this, this.as_ref()}
}

impl<T: HugrView> HugrView for Arc<T> {
    hugr_view_methods! {this, this.as_ref()}
}

impl<T: HugrView> HugrView for Box<T> {
    hugr_view_methods! {this, this.as_ref()}
}

impl<T: HugrView + ToOwned> HugrView for Cow<'_, T> {
    hugr_view_methods! {this, this.as_ref()}
}

impl<H: AsRef<Hugr>, Root> HugrView for RootChecked<H, Root> {
    hugr_view_methods! {this, this.as_ref()}
}

#[cfg(test)]
mod test {
    use std::{rc::Rc, sync::Arc};

    use crate::hugr::views::{DescendantsGraph, HierarchyView};
    use crate::{Hugr, HugrView, Node};

    struct ViewWrapper<H>(H);
    impl<H: HugrView> ViewWrapper<H> {
        fn nodes(&self) -> impl Iterator<Item = Node> + '_ {
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
