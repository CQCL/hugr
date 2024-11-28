use std::rc::Rc;
use std::sync::Arc;

use delegate::delegate;

use super::{HugrView, RootChecked};
use crate::{Direction, Hugr, Node, Port};

macro_rules! hugr_view_methods {
    // The extra ident here is because invocations of the macro cannot pass `self` as argument
    ($arg:ident, $e:expr) => {
        delegate! {
            to ({let $arg=self; $e}) {
                fn contains_node(&self, node: Node) -> bool;
                fn node_count(&self) -> usize;
                fn edge_count(&self) -> usize;
                fn nodes(&self) -> impl Iterator<Item = Node> + Clone;
                fn node_ports(&self, node: Node, dir: Direction) -> impl Iterator<Item = Port> + Clone;
                fn all_node_ports(&self, node: Node) -> impl Iterator<Item = Port> + Clone;
                fn linked_ports(
                    &self,
                    node: Node,
                    port: impl Into<Port>,
                ) -> impl Iterator<Item = (Node, Port)> + Clone;
                fn node_connections(&self, node: Node, other: Node) -> impl Iterator<Item = [Port; 2]> + Clone;
                fn num_ports(&self, node: Node, dir: Direction) -> usize;
                fn children(&self, node: Node) -> impl DoubleEndedIterator<Item = Node> + Clone;
                fn neighbours(&self, node: Node, dir: Direction) -> impl Iterator<Item = Node> + Clone;
                fn all_neighbours(&self, node: Node) -> impl Iterator<Item = Node> + Clone;
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

impl<H: AsRef<Hugr>, Root> HugrView for RootChecked<H, Root> {
    hugr_view_methods! {this, this.as_ref()}
}

#[cfg(test)]
mod test {
    use crate::hugr::views::{DescendantsGraph, HierarchyView};
    use crate::{Hugr, HugrView, Node};

    struct ViewWrapper<H>(H);
    impl<H: HugrView> ViewWrapper<H> {
        fn nodes<'a>(&'a self) -> impl Iterator<Item = Node> + 'a {
            self.0.nodes()
        }
    }

    #[test]
    fn test_views() {
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
    }
}
