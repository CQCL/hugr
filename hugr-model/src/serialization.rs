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
pub fn convert_module(module: &v0::Module) -> capnp::message::Builder<HeapAllocator> {
    let mut builder = capnp::message::Builder::new_default();
    let mut module_builder = builder.init_root::<hugr_capnp::module::Builder>();
    {
        let mut nodes = module_builder
            .reborrow()
            .init_nodes(module.nodes.len() as u32);
        for (i, node) in module.nodes.iter().enumerate() {
            let mut node_builder = nodes.reborrow().get(i as u32);
            node.build(&mut node_builder);
        }
    }
    {
        let mut ports = module_builder
            .reborrow()
            .init_ports(module.ports.len() as u32);
        for (i, port) in module.ports.iter().enumerate() {
            let mut port_builder = ports.reborrow().get(i as u32);
            port.build(&mut port_builder);
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
            term.build(&mut term_table.reborrow().get(id));
        }
    }

    // Set the fields of the module_builder using the data from the v0::Module

    builder
}

trait Build {
    type SerBuilder<'a>;
    fn build(&self, builder: &mut Self::SerBuilder<'_>);
}
impl Build for v0::Node {
    type SerBuilder<'a> = hugr_capnp::node::Builder<'a>;
    fn build(&self, builder: &mut Self::SerBuilder<'_>) {
        let mut operation_build = builder.reborrow().init_operation();
        self.operation.build(&mut operation_build);

        let mut ports = builder.reborrow().init_params(self.params.len() as u32);
        for (i, param_id) in self.params.iter().enumerate() {
            ports.set(i as u32, param_id.0);
        }

        let mut inputs = builder.reborrow().init_inputs(self.inputs.len() as u32);
        for (i, input_id) in self.inputs.iter().enumerate() {
            inputs.set(i as u32, input_id.0);
        }

        let mut outputs = builder.reborrow().init_outputs(self.outputs.len() as u32);
        for (i, output_id) in self.outputs.iter().enumerate() {
            outputs.set(i as u32, output_id.0);
        }

        let mut children = builder.reborrow().init_children(self.children.len() as u32);
        for (i, child_id) in self.children.iter().enumerate() {
            children.set(i as u32, child_id.0);
        }

        build_meta_list(
            self.meta.iter(),
            builder.reborrow().init_meta(self.meta.len() as u32),
        );
    }
}

impl Build for v0::MetaItem {
    type SerBuilder<'a> = hugr_capnp::meta_item::Builder<'a>;
    fn build(&self, builder: &mut Self::SerBuilder<'_>) {
        builder.set_name(&self.name);
        self.value.build(&mut builder.reborrow().init_value());
    }
}

fn build_meta_list<'a>(
    metas: impl Iterator<Item = &'a v0::MetaItem>,
    mut metadata: capnp::struct_list::Builder<hugr_capnp::meta_item::Owned>,
) {
    for (i, meta_item) in metas.enumerate() {
        meta_item.build(&mut metadata.reborrow().get(i as u32));
    }
}

impl Build for v0::Operation {
    type SerBuilder<'a> = hugr_capnp::operation::Builder<'a>;
    fn build(&self, builder: &mut Self::SerBuilder<'_>) {
        match self {
            v0::Operation::Module => builder.set_module(()),
            v0::Operation::Input => builder.set_input(()),
            v0::Operation::Output => builder.set_output(()),
            v0::Operation::Dfg => builder.set_dfg(()),
            v0::Operation::Cfg => builder.set_cfg(()),
            v0::Operation::Block => builder.set_block(()),
            v0::Operation::Exit => builder.set_exit(()),
            v0::Operation::Case => builder.set_case(()),
            v0::Operation::DefineFunc(defn) => {
                defn.build(&mut builder.reborrow().init_define_func());
            }
            v0::Operation::DeclareFunc(decl) => {
                decl.build(&mut builder.reborrow().init_define_func());
            }
            v0::Operation::CallFunc(_) => todo!(),
            v0::Operation::LoadFunc(_) => todo!(),
            v0::Operation::Custom(cust) => {
                cust.build(&mut builder.reborrow().init_custom());
            }
            v0::Operation::DefineAlias(_) => todo!(),
            v0::Operation::DeclareAlias(_) => todo!(),
        }
    }
}

impl Build for v0::operation::DefineFunc {
    type SerBuilder<'a> = hugr_capnp::operation::define_func::Builder<'a>;
    fn build(&self, builder: &mut Self::SerBuilder<'_>) {
        builder.set_name(&self.name.0);

        self.r#type.build(&mut builder.reborrow().init_type());
    }
}

impl Build for v0::operation::DeclareFunc {
    type SerBuilder<'a> = hugr_capnp::operation::define_func::Builder<'a>;
    fn build(&self, builder: &mut Self::SerBuilder<'_>) {
        builder.set_name(&self.name.0);
        self.r#type.build(&mut builder.reborrow().init_type());
    }
}

impl Build for v0::operation::CallFunc {
    type SerBuilder<'a> = hugr_capnp::operation::call_func::Builder<'a>;
    fn build(&self, builder: &mut Self::SerBuilder<'_>) {
        builder.set_name(&self.name.0);
    }
}

impl Build for v0::operation::Custom {
    type SerBuilder<'a> = hugr_capnp::operation::custom::Builder<'a>;
    fn build(&self, builder: &mut Self::SerBuilder<'_>) {
        builder.set_name(&self.name.0);
    }
}

impl Build for v0::Scheme {
    type SerBuilder<'a> = hugr_capnp::scheme::Builder<'a>;

    fn build(&self, _builder: &mut Self::SerBuilder<'_>) {
        todo!()
    }
}

impl Build for v0::Port {
    type SerBuilder<'a> = hugr_capnp::port::Builder<'a>;
    fn build(&self, builder: &mut Self::SerBuilder<'_>) {
        builder.set_type(self.r#type.0);
        let meta_builder = builder.reborrow().init_meta(self.meta.len() as u32);
        build_meta_list(self.meta.iter(), meta_builder);
    }
}

impl Build for v0::Term {
    type SerBuilder<'a> = hugr_capnp::term::Builder<'a>;
    fn build(&self, builder: &mut Self::SerBuilder<'_>) {
        match self {
            v0::Term::Wildcard => builder.set_wildcard(()),
            v0::Term::Type => builder.set_type(()),
            v0::Term::Constraint => todo!(),
            v0::Term::Var(_) => todo!(),
            v0::Term::Named(_) => todo!(),
            v0::Term::List(_) => todo!(),
            v0::Term::ListType(_) => todo!(),
            v0::Term::Str(s) => builder.set_str(s),
            v0::Term::StrType => builder.set_str_type(()),
            v0::Term::Nat(v) => builder.set_nat(*v),
            v0::Term::NatType => builder.set_nat_type(()),
            v0::Term::ExtSet(_) => todo!(),
            v0::Term::ExtSetType => builder.set_ext_set_type(()),
            v0::Term::Tuple(_) => todo!(),
            v0::Term::ProductType(_) => todo!(),
            v0::Term::Tagged(_) => todo!(),
            v0::Term::SumType(_) => todo!(),
            v0::Term::FuncType(_) => todo!(),
        };
    }
}

pub fn to_bytes(module: &v0::Module) -> Vec<u8> {
    let message = convert_module(module);
    let mut buf: Vec<u8> = Vec::new();
    capnp::serialize_packed::write_message(&mut buf, &message).unwrap();
    buf
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

        let buf = to_bytes(&module);
        let message_reader =
            capnp::serialize_packed::read_message(&buf[..], ::capnp::message::ReaderOptions::new())
                .unwrap();

        let message: hugr_capnp::module::Reader = message_reader.get_root().unwrap();
        let terms = message.get_terms().unwrap();
        assert_eq!(terms.len(), 3);
        assert_eq!(message.get_term_table().unwrap().len(), 2);
    }
}
