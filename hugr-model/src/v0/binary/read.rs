use crate::hugr_v0_capnp as hugr_capnp;
use crate::v0 as model;
use bumpalo::collections::Vec as BumpVec;
use bumpalo::Bump;

type ReadResult<T> = Result<T, capnp::Error>;

/// Read a hugr module from a byte slice.
pub fn read_from_slice<'a>(slice: &[u8], bump: &'a Bump) -> ReadResult<model::Module<'a>> {
    let reader =
        capnp::serialize_packed::read_message(slice, capnp::message::ReaderOptions::new())?;
    let root = reader.get_root::<hugr_capnp::module::Reader>()?;
    read_module(bump, root)
}

/// Read a list of structs from a reader into a slice allocated through the bump allocator.
macro_rules! read_list {
    ($bump:expr, $reader:expr, $get:ident, $read:expr) => {{
        let mut __list_reader = $reader.$get()?;
        let mut __list = BumpVec::with_capacity_in(__list_reader.len() as _, $bump);
        for __item_reader in __list_reader.iter() {
            __list.push($read($bump, __item_reader)?);
        }
        __list.into_bump_slice()
    }};
}

/// Read a list of scalars from a reader into a slice allocated through the bump allocator.
macro_rules! read_scalar_list {
    ($bump:expr, $reader:expr, $get:ident, $wrap:path) => {{
        let mut __list_reader = $reader.$get()?;
        let mut __list = BumpVec::with_capacity_in(__list_reader.len() as _, $bump);
        for __item in __list_reader.iter() {
            __list.push($wrap(__item));
        }
        __list.into_bump_slice()
    }};
}

fn read_module<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::module::Reader,
) -> ReadResult<model::Module<'a>> {
    let root = model::RegionId(reader.get_root());

    let nodes = reader
        .get_nodes()?
        .iter()
        .map(|r| read_node(bump, r))
        .collect::<ReadResult<_>>()?;

    let regions = reader
        .get_regions()?
        .iter()
        .map(|r| read_region(bump, r))
        .collect::<ReadResult<_>>()?;

    let terms = reader
        .get_terms()?
        .iter()
        .map(|r| read_term(bump, r))
        .collect::<ReadResult<_>>()?;

    Ok(model::Module {
        root,
        nodes,
        regions,
        terms,
    })
}

fn read_node<'a>(bump: &'a Bump, reader: hugr_capnp::node::Reader) -> ReadResult<model::Node<'a>> {
    let operation = read_operation(bump, reader.get_operation()?)?;
    let inputs = read_list!(bump, reader, get_inputs, read_link_ref);
    let outputs = read_list!(bump, reader, get_outputs, read_link_ref);
    let params = read_scalar_list!(bump, reader, get_params, model::TermId);
    let regions = read_scalar_list!(bump, reader, get_regions, model::RegionId);
    let meta = read_list!(bump, reader, get_meta, read_meta_item);
    let signature = reader.get_signature().checked_sub(1).map(model::TermId);

    Ok(model::Node {
        operation,
        inputs,
        outputs,
        params,
        regions,
        meta,
        signature,
    })
}

fn read_local_ref<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::local_ref::Reader,
) -> ReadResult<model::LocalRef<'a>> {
    use hugr_capnp::local_ref::Which;
    Ok(match reader.which()? {
        Which::Direct(reader) => {
            let index = reader.get_index();
            let node = model::NodeId(reader.get_node());
            model::LocalRef::Index(node, index)
        }
        Which::Named(name) => model::LocalRef::Named(bump.alloc_str(name?.to_str()?)),
    })
}

fn read_global_ref<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::global_ref::Reader,
) -> ReadResult<model::GlobalRef<'a>> {
    use hugr_capnp::global_ref::Which;
    Ok(match reader.which()? {
        Which::Node(node) => model::GlobalRef::Direct(model::NodeId(node)),
        Which::Named(name) => model::GlobalRef::Named(bump.alloc_str(name?.to_str()?)),
    })
}

fn read_link_ref<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::link_ref::Reader,
) -> ReadResult<model::LinkRef<'a>> {
    use hugr_capnp::link_ref::Which;
    Ok(match reader.which()? {
        Which::Id(id) => model::LinkRef::Id(model::LinkId(id)),
        Which::Named(name) => model::LinkRef::Named(bump.alloc_str(name?.to_str()?)),
    })
}

fn read_operation<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::operation::Reader,
) -> ReadResult<model::Operation<'a>> {
    use hugr_capnp::operation::Which;
    Ok(match reader.which()? {
        Which::Invalid(()) => model::Operation::Invalid,
        Which::Dfg(()) => model::Operation::Dfg,
        Which::Cfg(()) => model::Operation::Cfg,
        Which::Block(()) => model::Operation::Block,
        Which::FuncDefn(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader, get_params, read_param);
            let signature = model::TermId(reader.get_signature());
            let decl = bump.alloc(model::FuncDecl {
                name,
                params,
                signature,
            });
            model::Operation::DefineFunc { decl }
        }
        Which::FuncDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader, get_params, read_param);
            let signature = model::TermId(reader.get_signature());
            let decl = bump.alloc(model::FuncDecl {
                name,
                params,
                signature,
            });
            model::Operation::DeclareFunc { decl }
        }
        Which::AliasDefn(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader, get_params, read_param);
            let r#type = model::TermId(reader.get_type());
            let value = model::TermId(reader.get_value());
            let decl = bump.alloc(model::AliasDecl {
                name,
                params,
                r#type,
            });
            model::Operation::DefineAlias { decl, value }
        }
        Which::AliasDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader, get_params, read_param);
            let r#type = model::TermId(reader.get_type());
            let decl = bump.alloc(model::AliasDecl {
                name,
                params,
                r#type,
            });
            model::Operation::DeclareAlias { decl }
        }
        Which::ConstructorDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader, get_params, read_param);
            let r#type = model::TermId(reader.get_type());
            let decl = bump.alloc(model::ConstructorDecl {
                name,
                params,
                r#type,
            });
            model::Operation::DeclareConstructor { decl }
        }
        Which::OperationDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader, get_params, read_param);
            let r#type = model::TermId(reader.get_type());
            let decl = bump.alloc(model::OperationDecl {
                name,
                params,
                r#type,
            });
            model::Operation::DeclareOperation { decl }
        }
        Which::Custom(name) => model::Operation::Custom {
            operation: read_global_ref(bump, name?)?,
        },
        Which::CustomFull(name) => model::Operation::CustomFull {
            operation: read_global_ref(bump, name?)?,
        },
        Which::Tag(tag) => model::Operation::Tag { tag },
        Which::TailLoop(()) => model::Operation::TailLoop,
        Which::Conditional(()) => model::Operation::Conditional,
        Which::CallFunc(func) => model::Operation::CallFunc {
            func: model::TermId(func),
        },
        Which::LoadFunc(func) => model::Operation::LoadFunc {
            func: model::TermId(func),
        },
    })
}

fn read_region<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::region::Reader,
) -> ReadResult<model::Region<'a>> {
    let kind = match reader.get_kind()? {
        hugr_capnp::RegionKind::DataFlow => model::RegionKind::DataFlow,
        hugr_capnp::RegionKind::ControlFlow => model::RegionKind::ControlFlow,
    };

    let sources = read_list!(bump, reader, get_sources, read_link_ref);
    let targets = read_list!(bump, reader, get_targets, read_link_ref);
    let children = read_scalar_list!(bump, reader, get_children, model::NodeId);
    let meta = read_list!(bump, reader, get_meta, read_meta_item);
    let signature = reader.get_signature().checked_sub(1).map(model::TermId);

    Ok(model::Region {
        kind,
        sources,
        targets,
        children,
        meta,
        signature,
    })
}

fn read_term<'a>(bump: &'a Bump, reader: hugr_capnp::term::Reader) -> ReadResult<model::Term<'a>> {
    use hugr_capnp::term::Which;
    Ok(match reader.which()? {
        Which::Wildcard(()) => model::Term::Wildcard,
        Which::RuntimeType(()) => model::Term::Type,
        Which::StaticType(()) => model::Term::StaticType,
        Which::Constraint(()) => model::Term::Constraint,
        Which::String(value) => model::Term::Str(bump.alloc_str(value?.to_str()?)),
        Which::StringType(()) => model::Term::StrType,
        Which::Nat(value) => model::Term::Nat(value),
        Which::NatType(()) => model::Term::NatType,
        Which::ExtSetType(()) => model::Term::ExtSetType,
        Which::ControlType(()) => model::Term::ControlType,
        Which::Variable(local_ref) => model::Term::Var(read_local_ref(bump, local_ref?)?),

        Which::Apply(reader) => {
            let reader = reader?;
            let global = read_global_ref(bump, reader.get_global()?)?;
            let args = read_scalar_list!(bump, reader, get_args, model::TermId);
            model::Term::Apply { global, args }
        }

        Which::ApplyFull(reader) => {
            let reader = reader?;
            let global = read_global_ref(bump, reader.get_global()?)?;
            let args = read_scalar_list!(bump, reader, get_args, model::TermId);
            model::Term::ApplyFull { global, args }
        }

        Which::Quote(r#type) => model::Term::Quote {
            r#type: model::TermId(r#type),
        },

        Which::List(reader) => {
            let reader = reader?;
            let items = read_scalar_list!(bump, reader, get_items, model::TermId);
            let tail = reader.get_tail().checked_sub(1).map(model::TermId);
            model::Term::List { items, tail }
        }

        Which::ListType(item_type) => model::Term::ListType {
            item_type: model::TermId(item_type),
        },

        Which::ExtSet(reader) => {
            let reader = reader?;

            let extensions = {
                let extensions_reader = reader.get_extensions()?;
                let mut extensions = BumpVec::with_capacity_in(extensions_reader.len() as _, bump);
                for extension_reader in extensions_reader.iter() {
                    extensions.push(bump.alloc_str(extension_reader?.to_str()?) as &str);
                }
                extensions.into_bump_slice()
            };

            let rest = reader.get_rest().checked_sub(1).map(model::TermId);
            model::Term::ExtSet { extensions, rest }
        }

        Which::Adt(variants) => model::Term::Adt {
            variants: model::TermId(variants),
        },

        Which::FuncType(reader) => {
            let reader = reader?;
            let inputs = model::TermId(reader.get_inputs());
            let outputs = model::TermId(reader.get_outputs());
            let extensions = model::TermId(reader.get_extensions());
            model::Term::FuncType {
                inputs,
                outputs,
                extensions,
            }
        }

        Which::Control(values) => model::Term::Control {
            values: model::TermId(values),
        },
    })
}

fn read_meta_item<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::meta_item::Reader,
) -> ReadResult<model::MetaItem<'a>> {
    let name = bump.alloc_str(reader.get_name()?.to_str()?);
    let value = model::TermId(reader.get_value());
    Ok(model::MetaItem { name, value })
}

fn read_param<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::param::Reader,
) -> ReadResult<model::Param<'a>> {
    use hugr_capnp::param::Which;
    Ok(match reader.which()? {
        Which::Implicit(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let r#type = model::TermId(reader.get_type());
            model::Param::Implicit { name, r#type }
        }
        Which::Explicit(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let r#type = model::TermId(reader.get_type());
            model::Param::Explicit { name, r#type }
        }
        Which::Constraint(constraint) => {
            let constraint = model::TermId(constraint);
            model::Param::Constraint { constraint }
        }
    })
}
