//! Conversion of model to and from binary format.
//!
//!
#![allow(missing_docs)]

use std::{collections::HashMap, sync::Arc};

use capnp::message::HeapAllocator;

use crate::v0;
pub mod hugr_capnp {
    include!(concat!(env!("OUT_DIR"), "/hugr_capnp.rs"));
}

/// Convert v0::Module to hugr_capnp::module.
pub fn convert_module(
    module: &v0::Module,
) -> Result<capnp::message::Builder<HeapAllocator>, Box<dyn std::error::Error>> {
    let mut builder = capnp::message::Builder::new_default();
    let mut module_builder = builder.init_root::<hugr_capnp::module::Builder>();
    {
        let mut nodes = module_builder
            .reborrow()
            .init_nodes(module.nodes.len() as u32);
        for (i, node) in module.nodes.iter().enumerate() {
            let mut node_builder = nodes.reborrow().get(i as u32);
            build_node(node, &mut node_builder)?;
        }
    }
    {
        let mut ports = module_builder
            .reborrow()
            .init_ports(module.ports.len() as u32);
        for (i, port) in module.ports.iter().enumerate() {
            let mut port_builder = ports.reborrow().get(i as u32);
            build_port(port, &mut port_builder)?;
        }
    }

    let term_hash: HashMap<&Arc<v0::Term>, u32> = module
        .term_table
        .iter()
        .enumerate()
        .map(|(i, term)| (term, i as u32))
        .collect();
    {
        let mut terms = module_builder
            .reborrow()
            .init_terms(module.terms.len() as u32);
        for (i, term) in module.terms.iter().enumerate() {
            terms.set(i as u32, *term_hash.get(term).expect("Term not in table."));
        }
    }

    {
        let mut term_table = module_builder
            .reborrow()
            .init_term_table(term_hash.len() as u32);
        for (term, id) in term_hash {
            let mut term_builder = term_table.reborrow().get(id);
            build_term(term, &mut term_builder)?;
        }
    }

    // Set the fields of the module_builder using the data from the v0::Module

    Ok(builder)
}

fn build_node(
    node: &v0::Node,
    builder: &mut hugr_capnp::node::Builder,
) -> Result<(), Box<dyn std::error::Error>> {
    // Set the fields of the node_builder using the data from the v0::Node
    Ok(())
}

fn build_port(
    port: &v0::Port,
    builder: &mut hugr_capnp::port::Builder,
) -> Result<(), Box<dyn std::error::Error>> {
    // Set the fields of the port_builder using the data from the v0::Port
    Ok(())
}

fn build_term(
    term: &v0::Term,
    builder: &mut hugr_capnp::term::Builder,
) -> Result<(), Box<dyn std::error::Error>> {
    match term {
        v0::Term::Wildcard => todo!(),
        v0::Term::Type => todo!(),
        v0::Term::Constraint => todo!(),
        v0::Term::Var(_) => todo!(),
        v0::Term::Named(_) => todo!(),
        v0::Term::List(_) => todo!(),
        v0::Term::ListType(_) => todo!(),
        v0::Term::Str(s) => {
            let mut s_build = builder.reborrow().init_str(s.len() as u32);
            s_build.push_str(s);
        }
        v0::Term::StrType => todo!(),
        v0::Term::Nat(_) => todo!(),
        v0::Term::NatType => todo!(),
        v0::Term::ExtSet(_) => todo!(),
        v0::Term::ExtSetType => todo!(),
        v0::Term::Tuple(_) => todo!(),
        v0::Term::ProductType(_) => todo!(),
        v0::Term::Tagged(_) => todo!(),
        v0::Term::SumType(_) => todo!(),
        v0::Term::FuncType(_) => todo!(),
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use smol_str::SmolStr;

    use super::*;

    #[test]
    fn test_hashcons() {
        let mut module = v0::Module::default();
        let s = SmolStr::new_inline("Hello, world!");
        module.add_term(v0::Term::Str(s.clone()));
        module.add_term(v0::Term::Str(s.clone()));
        module.add_term(v0::Term::Str(SmolStr::new_inline("Hello, world")));

        let message = convert_module(&module).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        capnp::serialize_packed::write_message(&mut buf, &message).unwrap();

        let message_reader =
            capnp::serialize_packed::read_message(&buf[..], ::capnp::message::ReaderOptions::new())
                .unwrap();

        let message: hugr_capnp::module::Reader = message_reader.get_root().unwrap();
        let terms = message.get_terms().unwrap();
        assert_eq!(terms.len(), 3);
        assert_eq!(message.get_term_table().unwrap().len(), 2);
    }
}
