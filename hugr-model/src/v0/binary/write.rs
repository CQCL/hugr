use crate::hugr_v0_capnp as hugr_capnp;
use crate::v0 as model;

/// Write a list of items into a list builder.
macro_rules! write_list {
    ($builder:expr, $init:ident, $write:expr, $list:expr) => {
        let mut __list_builder = $builder.reborrow().$init($list.len() as _);
        for (index, item) in $list.iter().enumerate() {
            $write(__list_builder.reborrow().get(index as _), item);
        }
    };
}

/// Writes a module to a byte vector.
pub fn write_to_vec(module: &model::Module) -> Vec<u8> {
    let mut message = capnp::message::Builder::new_default();
    let builder = message.init_root();
    write_module(builder, module);

    let mut output = Vec::new();
    let _ = capnp::serialize_packed::write_message(&mut output, &message);
    output
}

fn write_module(mut builder: hugr_capnp::module::Builder, module: &model::Module) {
    builder.set_root(module.root.0);
    write_list!(builder, init_nodes, write_node, module.nodes);
    write_list!(builder, init_regions, write_region, module.regions);
    write_list!(builder, init_terms, write_term, module.terms);
}

fn write_node(mut builder: hugr_capnp::node::Builder, node: &model::Node) {
    write_operation(builder.reborrow().init_operation(), &node.operation);
    let _ = builder.set_inputs(model::LinkIndex::unwrap_slice(node.inputs));
    let _ = builder.set_outputs(model::LinkIndex::unwrap_slice(node.outputs));
    let _ = builder.set_meta(model::TermId::unwrap_slice(node.meta));
    let _ = builder.set_params(model::TermId::unwrap_slice(node.params));
    let _ = builder.set_regions(model::RegionId::unwrap_slice(node.regions));
    builder.set_signature(node.signature.map_or(0, |t| t.0 + 1));
}

fn write_operation(mut builder: hugr_capnp::operation::Builder, operation: &model::Operation) {
    match operation {
        model::Operation::Dfg => builder.set_dfg(()),
        model::Operation::Cfg => builder.set_cfg(()),
        model::Operation::Block => builder.set_block(()),
        model::Operation::TailLoop => builder.set_tail_loop(()),
        model::Operation::Conditional => builder.set_conditional(()),
        model::Operation::Custom(operation) => builder.set_custom(operation.0),

        model::Operation::DefineFunc(symbol) => {
            let builder = builder.init_func_defn();
            write_symbol(builder, symbol);
        }
        model::Operation::DeclareFunc(symbol) => {
            let builder = builder.init_func_decl();
            write_symbol(builder, symbol);
        }

        model::Operation::DefineAlias(symbol) => {
            let builder = builder.init_alias_defn();
            write_symbol(builder, symbol);
        }
        model::Operation::DeclareAlias(symbol) => {
            let builder = builder.init_alias_decl();
            write_symbol(builder, symbol);
        }

        model::Operation::DeclareConstructor(symbol) => {
            let builder = builder.init_constructor_decl();
            write_symbol(builder, symbol);
        }
        model::Operation::DeclareOperation(symbol) => {
            let builder = builder.init_operation_decl();
            write_symbol(builder, symbol);
        }

        model::Operation::Import { name } => {
            builder.set_import(*name);
        }

        model::Operation::Invalid => builder.set_invalid(()),
    }
}

fn write_symbol(mut builder: hugr_capnp::symbol::Builder, symbol: &model::Symbol) {
    let _ = builder.set_name(symbol.name);
    write_list!(builder, init_params, write_param, symbol.params);
    let _ = builder.set_constraints(model::TermId::unwrap_slice(symbol.constraints));
    builder.set_signature(symbol.signature.0);
}

fn write_param(mut builder: hugr_capnp::param::Builder, param: &model::Param) {
    builder.set_name(param.name);
    builder.set_type(param.r#type.0);
}

fn write_region(mut builder: hugr_capnp::region::Builder, region: &model::Region) {
    builder.set_kind(match region.kind {
        model::RegionKind::DataFlow => hugr_capnp::RegionKind::DataFlow,
        model::RegionKind::ControlFlow => hugr_capnp::RegionKind::ControlFlow,
        model::RegionKind::Module => hugr_capnp::RegionKind::Module,
    });

    let _ = builder.set_sources(model::LinkIndex::unwrap_slice(region.sources));
    let _ = builder.set_targets(model::LinkIndex::unwrap_slice(region.targets));
    let _ = builder.set_children(model::NodeId::unwrap_slice(region.children));
    let _ = builder.set_meta(model::TermId::unwrap_slice(region.meta));
    builder.set_signature(region.signature.map_or(0, |t| t.0 + 1));

    if let Some(scope) = &region.scope {
        write_region_scope(builder.init_scope(), scope);
    }
}

fn write_region_scope(mut builder: hugr_capnp::region_scope::Builder, scope: &model::RegionScope) {
    builder.set_links(scope.links);
    builder.set_ports(scope.ports);
}

fn write_term(mut builder: hugr_capnp::term::Builder, term: &model::Term) {
    match term {
        model::Term::Wildcard => builder.set_wildcard(()),
        model::Term::Var(model::VarId(node, index)) => {
            let mut builder = builder.init_variable();
            builder.set_node(node.0);
            builder.set_index(*index);
        }
        model::Term::Str(value) => builder.set_string(value),
        model::Term::Nat(value) => builder.set_nat(*value),
        model::Term::ConstFunc(region) => builder.set_const_func(region.0),
        model::Term::Bytes(data) => builder.set_bytes(data),
        model::Term::Float(value) => builder.set_float(value.into_inner()),

        model::Term::Apply(symbol, args) => {
            let mut builder = builder.init_apply();
            builder.set_symbol(symbol.0);
            let _ = builder.set_args(model::TermId::unwrap_slice(args));
        }

        model::Term::List(parts) => {
            write_list!(builder, init_list, write_list_part, parts);
        }

        model::Term::ExtSet(parts) => {
            write_list!(builder, init_ext_set, write_ext_set_part, parts);
        }

        model::Term::Tuple(parts) => {
            write_list!(builder, init_tuple, write_tuple_part, parts);
        }
    }
}

fn write_list_part(mut builder: hugr_capnp::term::list_part::Builder, part: &model::ListPart) {
    match part {
        model::ListPart::Item(term_id) => builder.set_item(term_id.0),
        model::ListPart::Splice(term_id) => builder.set_splice(term_id.0),
    }
}

fn write_tuple_part(mut builder: hugr_capnp::term::tuple_part::Builder, item: &model::TuplePart) {
    match item {
        model::TuplePart::Item(term_id) => builder.set_item(term_id.0),
        model::TuplePart::Splice(term_id) => builder.set_splice(term_id.0),
    }
}

fn write_ext_set_part(
    mut builder: hugr_capnp::term::ext_set_part::Builder,
    part: &model::ExtSetPart,
) {
    match part {
        model::ExtSetPart::Extension(ext) => builder.set_extension(ext),
        model::ExtSetPart::Splice(term_id) => builder.set_splice(term_id.0),
    }
}
