use std::ops::Deref;

use delegate::delegate;

use super::HugrView;
use crate::{Direction, Node, Port};

impl<H:HugrView, T: Deref<Target=H>> HugrView for T {
    delegate! {
        to (**self) {
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
