//! bar
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
        let docs = attrs.get_or_insert::<DocString>();
        docs.insert(node0, DocString("Lorem ipsum".into()));
        docs.insert(node1, DocString("dolor sit".into()));

        let funcs = attrs.get_or_insert::<FuncRef>();
        funcs.insert(node1, FuncRef(node0));
    }

    {
        let docs0 = attrs.borrow::<DocString>().unwrap();
        let docs1 = attrs.borrow::<DocString>().unwrap();
        let func_ref = attrs.borrow_mut::<FuncRef>().unwrap();
    }

    println!("{}", serde_json::to_string_pretty(&attrs).unwrap());
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocString(pub SmolStr);

impl_attr_sparse!(DocString, "core/doc");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuncRef(pub Node);

impl_attr_sparse!(FuncRef, "core/func-ref");
