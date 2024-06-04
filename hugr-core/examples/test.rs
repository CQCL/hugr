//! bar
use std::sync::Arc;

use hugr::builder::HugrBuilder;
use hugr::extension::ExtensionRegistry;
use hugr::types::{FunctionType, PolyFuncType, TypeRow};
use hugr_core::builder::ModuleBuilder;
use hugr_core::hugr::attributes::{Attr, AttrGroup, Sparse};
use hugr_core::{impl_attr_sparse, Hugr};
use serde::{Deserialize, Serialize};

pub fn main() {
    let mut group = AttrGroup::new();

    let docs = group.get_or_insert::<DocString>();

    group.get::<DocString>().unwrap();

    println!("{}", serde_json::to_string_pretty(&group).unwrap());
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocString(pub Arc<str>);

impl Default for DocString {
    fn default() -> Self {
        Self("".into())
    }
}

impl_attr_sparse!(DocString, "core/doc");

// foo
// pub fn main() {
//     let extension_registry = ExtensionRegistry::try_new([]).unwrap();

//     let mut module = ModuleBuilder::new();

//     let ft = PolyFuncType::new([], FunctionType::new(TypeRow::new(), TypeRow::new()));
//     let f = module.declare("foo", ft);

//     let hugr = module.finish_hugr(&extension_registry).unwrap();

//     // println!("{:#?}");
//     println!("{}", serde_json::to_string_pretty(&hugr).unwrap());

//     // println!("Hello world");
// }
