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
    write_list!(builder, init_inputs, write_link_ref, node.inputs);
    write_list!(builder, init_outputs, write_link_ref, node.outputs);
    write_list!(builder, init_meta, write_meta_item, node.meta);
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
        model::Operation::Custom { operation } => {
            write_global_ref(builder.init_custom(), operation)
        }
        model::Operation::CustomFull { operation } => {
            write_global_ref(builder.init_custom_full(), operation)
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

        model::Operation::Invalid => builder.set_invalid(()),
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

fn write_global_ref(mut builder: hugr_capnp::global_ref::Builder, global_ref: &model::GlobalRef) {
    match global_ref {
        model::GlobalRef::Direct(node) => builder.set_node(node.0),
        model::GlobalRef::Named(name) => builder.set_named(name),
    }
}

fn write_link_ref(mut builder: hugr_capnp::link_ref::Builder, link_ref: &model::LinkRef) {
    match link_ref {
        model::LinkRef::Id(id) => builder.set_id(id.0),
        model::LinkRef::Named(name) => builder.set_named(name),
    }
}

fn write_local_ref(mut builder: hugr_capnp::local_ref::Builder, local_ref: &model::LocalRef) {
    match local_ref {
        model::LocalRef::Index(node, index) => {
            let mut builder = builder.init_direct();
            builder.set_node(node.0);
            builder.set_index(*index);
        }
        model::LocalRef::Named(name) => builder.set_named(name),
    }
}

fn write_meta_item(mut builder: hugr_capnp::meta_item::Builder, meta_item: &model::MetaItem) {
    builder.set_name(meta_item.name);
    builder.set_value(meta_item.value.0)
}

fn write_region(mut builder: hugr_capnp::region::Builder, region: &model::Region) {
    builder.set_kind(match region.kind {
        model::RegionKind::DataFlow => hugr_capnp::RegionKind::DataFlow,
        model::RegionKind::ControlFlow => hugr_capnp::RegionKind::ControlFlow,
        model::RegionKind::Module => hugr_capnp::RegionKind::Module,
    });

    write_list!(builder, init_sources, write_link_ref, region.sources);
    write_list!(builder, init_targets, write_link_ref, region.targets);
    let _ = builder.set_children(model::NodeId::unwrap_slice(region.children));
    write_list!(builder, init_meta, write_meta_item, region.meta);
    builder.set_signature(region.signature.map_or(0, |t| t.0 + 1));
}

fn write_term(mut builder: hugr_capnp::term::Builder, term: &model::Term) {
    match term {
        model::Term::Wildcard => builder.set_wildcard(()),
        model::Term::Type => builder.set_runtime_type(()),
        model::Term::StaticType => builder.set_static_type(()),
        model::Term::Constraint => builder.set_constraint(()),
        model::Term::Var(local_ref) => write_local_ref(builder.init_variable(), local_ref),
        model::Term::ListType { item_type } => builder.set_list_type(item_type.0),
        model::Term::Str(value) => builder.set_string(value),
        model::Term::StrType => builder.set_string_type(()),
        model::Term::Nat(value) => builder.set_nat(*value),
        model::Term::NatType => builder.set_nat_type(()),
        model::Term::ExtSetType => builder.set_ext_set_type(()),
        model::Term::Adt { variants } => builder.set_adt(variants.0),
        model::Term::Quote { r#type } => builder.set_quote(r#type.0),
        model::Term::Control { values } => builder.set_control(values.0),
        model::Term::ControlType => builder.set_control_type(()),

        model::Term::Apply { global, args } => {
            let mut builder = builder.init_apply();
            write_global_ref(builder.reborrow().init_global(), global);
            let _ = builder.set_args(model::TermId::unwrap_slice(args));
        }

        model::Term::ApplyFull { global, args } => {
            let mut builder = builder.init_apply_full();
            write_global_ref(builder.reborrow().init_global(), global);
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
