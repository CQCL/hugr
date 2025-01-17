use std::io::Write;

use crate::hugr_v0_capnp as hugr_capnp;
use crate::v0 as model;

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

/// Writes a module to an impl of [Write].
pub fn write_to_writer(module: &model::Module, writer: impl Write) -> Result<(), WriteError> {
    let mut message = capnp::message::Builder::new_default();
    let builder = message.init_root();
    write_module(builder, module);

    Ok(capnp::serialize_packed::write_message(writer, &message)?)
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
        model::Operation::Tag { tag } => builder.set_tag(*tag),
        model::Operation::Custom { operation } => builder.set_custom(operation.0),
        model::Operation::CustomFull { operation } => {
            builder.set_custom_full(operation.0);
        }
        model::Operation::CallFunc { func } => builder.set_call_func(func.0),
        model::Operation::LoadFunc { func } => builder.set_load_func(func.0),

        model::Operation::DefineFunc { decl } => {
            let mut builder = builder.init_func_defn();
            builder.set_name(decl.name);
            write_list!(builder, init_params, write_param, decl.params);
            let _ = builder.set_constraints(model::TermId::unwrap_slice(decl.constraints));
            builder.set_signature(decl.signature.0);
        }
        model::Operation::DeclareFunc { decl } => {
            let mut builder = builder.init_func_decl();
            builder.set_name(decl.name);
            write_list!(builder, init_params, write_param, decl.params);
            let _ = builder.set_constraints(model::TermId::unwrap_slice(decl.constraints));
            builder.set_signature(decl.signature.0);
        }

        model::Operation::DefineAlias { decl, value } => {
            let mut builder = builder.init_alias_defn();
            builder.set_name(decl.name);
            write_list!(builder, init_params, write_param, decl.params);
            builder.set_type(decl.r#type.0);
            builder.set_value(value.0);
        }
        model::Operation::DeclareAlias { decl } => {
            let mut builder = builder.init_alias_decl();
            builder.set_name(decl.name);
            write_list!(builder, init_params, write_param, decl.params);
            builder.set_type(decl.r#type.0);
        }

        model::Operation::DeclareConstructor { decl } => {
            let mut builder = builder.init_constructor_decl();
            builder.set_name(decl.name);
            write_list!(builder, init_params, write_param, decl.params);
            let _ = builder.set_constraints(model::TermId::unwrap_slice(decl.constraints));
            builder.set_type(decl.r#type.0);
        }
        model::Operation::DeclareOperation { decl } => {
            let mut builder = builder.init_operation_decl();
            builder.set_name(decl.name);
            write_list!(builder, init_params, write_param, decl.params);
            let _ = builder.set_constraints(model::TermId::unwrap_slice(decl.constraints));
            builder.set_type(decl.r#type.0);
        }

        model::Operation::Import { name } => {
            builder.set_import(*name);
        }

        model::Operation::Invalid => builder.set_invalid(()),

        model::Operation::Const { value } => builder.set_const(value.0),
    }
}

fn write_param(mut builder: hugr_capnp::param::Builder, param: &model::Param) {
    builder.set_name(param.name);
    builder.set_type(param.r#type.0);
    builder.set_sort(match param.sort {
        model::ParamSort::Implicit => hugr_capnp::ParamSort::Implicit,
        model::ParamSort::Explicit => hugr_capnp::ParamSort::Explicit,
    });
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
        model::Term::Type => builder.set_runtime_type(()),
        model::Term::StaticType => builder.set_static_type(()),
        model::Term::Constraint => builder.set_constraint(()),
        model::Term::Var(model::VarId(node, index)) => {
            let mut builder = builder.init_variable();
            builder.set_variable_node(node.0);
            builder.set_variable_index(*index);
        }
        model::Term::ListType { item_type } => builder.set_list_type(item_type.0),
        model::Term::Str(value) => builder.set_string(value),
        model::Term::StrType => builder.set_string_type(()),
        model::Term::Nat(value) => builder.set_nat(*value),
        model::Term::NatType => builder.set_nat_type(()),
        model::Term::ExtSetType => builder.set_ext_set_type(()),
        model::Term::Adt { variants } => builder.set_adt(variants.0),
        model::Term::Const { r#type, extensions } => {
            let mut builder = builder.init_const();
            builder.set_type(r#type.0);
            builder.set_extensions(extensions.0);
        }
        model::Term::Control { values } => builder.set_control(values.0),
        model::Term::ControlType => builder.set_control_type(()),

        model::Term::Apply { symbol, args } => {
            let mut builder = builder.init_apply();
            builder.set_symbol(symbol.0);
            let _ = builder.set_args(model::TermId::unwrap_slice(args));
        }

        model::Term::ApplyFull { symbol, args } => {
            let mut builder = builder.init_apply_full();
            builder.set_symbol(symbol.0);
            let _ = builder.set_args(model::TermId::unwrap_slice(args));
        }

        model::Term::List { parts } => {
            let mut builder = builder.init_list();
            write_list!(builder, init_items, write_list_item, parts);
        }

        model::Term::ExtSet { parts } => {
            let mut builder = builder.init_ext_set();
            write_list!(builder, init_items, write_ext_set_item, parts);
        }

        model::Term::FuncType {
            inputs,
            outputs,
            extensions,
        } => {
            let mut builder = builder.init_func_type();
            builder.set_inputs(inputs.0);
            builder.set_outputs(outputs.0);
            builder.set_extensions(extensions.0);
        }

        model::Term::NonLinearConstraint { term } => {
            builder.set_non_linear_constraint(term.0);
        }

        model::Term::ConstFunc { region } => {
            builder.set_const_func(region.0);
        }

        model::Term::ConstAdt { tag, values } => {
            let mut builder = builder.init_const_adt();
            builder.set_tag(*tag);
            builder.set_values(values.0);
        }

        model::Term::Bytes { data } => {
            builder.set_bytes(data);
        }

        model::Term::BytesType => {
            builder.set_bytes_type(());
        }

        model::Term::Meta => {
            builder.set_meta(());
        }
    }
}

fn write_list_item(mut builder: hugr_capnp::term::list_part::Builder, item: &model::ListPart) {
    match item {
        model::ListPart::Item(term_id) => builder.set_item(term_id.0),
        model::ListPart::Splice(term_id) => builder.set_splice(term_id.0),
    }
}

fn write_ext_set_item(
    mut builder: hugr_capnp::term::ext_set_part::Builder,
    item: &model::ExtSetPart,
) {
    match item {
        model::ExtSetPart::Extension(ext) => builder.set_extension(ext),
        model::ExtSetPart::Splice(term_id) => builder.set_splice(term_id.0),
    }
}
