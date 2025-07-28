use std::io::Write;

use crate::capnp::hugr_v0_capnp as hugr_capnp;
use crate::v0 as model;
use crate::v0::table;

/// An error encounter while serializing a model.
#[derive(Debug, derive_more::From, derive_more::Display, derive_more::Error)]
#[non_exhaustive]
pub enum WriteError {
    /// An error encountered while encoding a `capnproto` buffer.
    EncodingError(capnp::Error),
}

/// Write a list of items into a list builder.
macro_rules! write_list {
    ($builder:expr, $init:ident, $write:expr, $list:expr) => {
        let mut __list_builder = $builder.reborrow().$init($list.len() as _);
        for (index, item) in $list.iter().enumerate() {
            $write(__list_builder.reborrow().get(index as _), item);
        }
    };
}

/// Writes a package to an impl of [Write].
pub fn write_to_writer(package: &table::Package, writer: impl Write) -> Result<(), WriteError> {
    let mut message = capnp::message::Builder::new_default();
    let builder = message.init_root();
    write_package(builder, package);

    Ok(capnp::serialize_packed::write_message(writer, &message)?)
}

/// Writes a package to a byte vector.
#[must_use]
pub fn write_to_vec(package: &table::Package) -> Vec<u8> {
    let mut message = capnp::message::Builder::new_default();
    let builder = message.init_root();
    write_package(builder, package);

    let mut output = Vec::new();
    let _ = capnp::serialize_packed::write_message(&mut output, &message);
    output
}

fn write_package(mut builder: hugr_capnp::package::Builder, package: &table::Package) {
    write_list!(builder, init_modules, write_module, package.modules);
}

fn write_module(mut builder: hugr_capnp::module::Builder, module: &table::Module) {
    builder.set_root(module.root.0);
    write_list!(builder, init_nodes, write_node, module.nodes);
    write_list!(builder, init_regions, write_region, module.regions);
    write_list!(builder, init_terms, write_term, module.terms);
}

fn write_node(mut builder: hugr_capnp::node::Builder, node: &table::Node) {
    write_operation(builder.reborrow().init_operation(), &node.operation);
    let _ = builder.set_inputs(table::LinkIndex::unwrap_slice(node.inputs));
    let _ = builder.set_outputs(table::LinkIndex::unwrap_slice(node.outputs));
    let _ = builder.set_meta(table::TermId::unwrap_slice(node.meta));
    let _ = builder.set_regions(table::RegionId::unwrap_slice(node.regions));
    builder.set_signature(node.signature.map_or(0, |t| t.0 + 1));
}

fn write_operation(mut builder: hugr_capnp::operation::Builder, operation: &table::Operation) {
    match operation {
        table::Operation::Dfg => builder.set_dfg(()),
        table::Operation::Cfg => builder.set_cfg(()),
        table::Operation::Block => builder.set_block(()),
        table::Operation::TailLoop => builder.set_tail_loop(()),
        table::Operation::Conditional => builder.set_conditional(()),
        table::Operation::Custom(operation) => builder.set_custom(operation.0),

        table::Operation::DefineFunc(symbol) => {
            let builder = builder.init_func_defn();
            write_symbol(builder, symbol);
        }
        table::Operation::DeclareFunc(symbol) => {
            let builder = builder.init_func_decl();
            write_symbol(builder, symbol);
        }

        table::Operation::DefineAlias(symbol, value) => {
            let mut builder = builder.init_alias_defn();
            write_symbol(builder.reborrow().init_symbol(), symbol);
            builder.set_value(value.0);
        }
        table::Operation::DeclareAlias(symbol) => {
            let builder = builder.init_alias_decl();
            write_symbol(builder, symbol);
        }

        table::Operation::DeclareConstructor(symbol) => {
            let builder = builder.init_constructor_decl();
            write_symbol(builder, symbol);
        }
        table::Operation::DeclareOperation(symbol) => {
            let builder = builder.init_operation_decl();
            write_symbol(builder, symbol);
        }

        table::Operation::Import { name } => {
            builder.set_import(*name);
        }

        table::Operation::Invalid => builder.set_invalid(()),
    }
}

fn write_symbol(mut builder: hugr_capnp::symbol::Builder, symbol: &table::Symbol) {
    builder.set_name(symbol.name);
    write_list!(builder, init_params, write_param, symbol.params);
    let _ = builder.set_constraints(table::TermId::unwrap_slice(symbol.constraints));
    builder.set_signature(symbol.signature.0);
}

fn write_param(mut builder: hugr_capnp::param::Builder, param: &table::Param) {
    builder.set_name(param.name);
    builder.set_type(param.r#type.0);
}

fn write_region(mut builder: hugr_capnp::region::Builder, region: &table::Region) {
    builder.set_kind(match region.kind {
        model::RegionKind::DataFlow => hugr_capnp::RegionKind::DataFlow,
        model::RegionKind::ControlFlow => hugr_capnp::RegionKind::ControlFlow,
        model::RegionKind::Module => hugr_capnp::RegionKind::Module,
    });

    let _ = builder.set_sources(table::LinkIndex::unwrap_slice(region.sources));
    let _ = builder.set_targets(table::LinkIndex::unwrap_slice(region.targets));
    let _ = builder.set_children(table::NodeId::unwrap_slice(region.children));
    let _ = builder.set_meta(table::TermId::unwrap_slice(region.meta));
    builder.set_signature(region.signature.map_or(0, |t| t.0 + 1));

    if let Some(scope) = &region.scope {
        write_region_scope(builder.init_scope(), scope);
    }
}

fn write_region_scope(mut builder: hugr_capnp::region_scope::Builder, scope: &table::RegionScope) {
    builder.set_links(scope.links);
    builder.set_ports(scope.ports);
}

fn write_term(mut builder: hugr_capnp::term::Builder, term: &table::Term) {
    match term {
        table::Term::Wildcard => builder.set_wildcard(()),
        table::Term::Var(table::VarId(node, index)) => {
            let mut builder = builder.init_variable();
            builder.set_node(node.0);
            builder.set_index(*index);
        }

        table::Term::Literal(value) => match value {
            model::Literal::Str(value) => builder.set_string(value),
            model::Literal::Nat(value) => builder.set_nat(*value),
            model::Literal::Bytes(value) => builder.set_bytes(value),
            model::Literal::Float(value) => builder.set_float(value.into_inner()),
        },

        table::Term::Func(region) => builder.set_func(region.0),
        table::Term::Apply(symbol, args) => {
            let mut builder = builder.init_apply();
            builder.set_symbol(symbol.0);
            let _ = builder.set_args(table::TermId::unwrap_slice(args));
        }

        table::Term::List(parts) => {
            write_list!(builder, init_list, write_seq_part, parts);
        }

        table::Term::Tuple(parts) => {
            write_list!(builder, init_tuple, write_seq_part, parts);
        }
    }
}

fn write_seq_part(mut builder: hugr_capnp::term::seq_part::Builder, part: &table::SeqPart) {
    match part {
        table::SeqPart::Item(term_id) => builder.set_item(term_id.0),
        table::SeqPart::Splice(term_id) => builder.set_splice(term_id.0),
    }
}
