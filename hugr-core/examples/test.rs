//! bar
use std::str::FromStr;

use hugr::Node;
use hugr_core::hugr::attributes::{Attr, AttrGroup, AttrStore};
use hugr_core::impl_attr_sparse;
use portgraph::{PortGraph, PortMut};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

pub fn main() {
    let mut graph = PortGraph::new();
    let mut attrs = AttrGroup::new();

    let node0 = graph.add_node(0, 0).into();
    let node1 = graph.add_node(0, 0).into();

    {
        let docs = attrs.register::<DocString>();
        docs.insert(node0, "Lorem ipsum".into());
        docs.insert(node1, "dolor sit".into());

        let funcs = attrs.register::<FuncRef>();
        funcs.insert(node1, FuncRef(node0));
    }

    {
        let docs0 = attrs.borrow::<DocString>();
        let docs1 = attrs.borrow::<DocString>();
        let func_ref = attrs.borrow_mut::<FuncRef>();

        println!("{:#?}", docs0[node0]);
    }

    println!("{:#?}", attrs);
    println!("{}", serde_json::to_string_pretty(&attrs).unwrap());
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocString(pub SmolStr);

impl From<&str> for DocString {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

impl_attr_sparse!(DocString, "core/doc");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuncRef(pub Node);

impl_attr_sparse!(FuncRef, "core/func-ref");
