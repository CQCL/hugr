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
    ($bump:expr, $reader:expr, $read:expr) => {{
        let mut __list_reader = $reader;
        let mut __list = BumpVec::with_capacity_in(__list_reader.len() as _, $bump);
        for __item_reader in __list_reader.iter() {
            __list.push($read($bump, __item_reader)?);
        }
        __list.into_bump_slice()
    }};
}

/// Read a list of scalars from a reader into a slice allocated through the bump allocator.
macro_rules! read_scalar_list {
    ($bump:expr, $reader:expr, $wrap:path) => {{
        let mut __list_reader = $reader;
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
    let inputs = read_scalar_list!(bump, reader.get_inputs()?, model::LinkIndex);
    let outputs = read_scalar_list!(bump, reader.get_outputs()?, model::LinkIndex);
    let params = read_scalar_list!(bump, reader.get_params()?, model::TermId);
    let meta = read_list!(bump, reader.get_meta()?, read_meta_item);
    let signature = reader.get_signature().checked_sub(1).map(model::TermId);

    Ok(model::Node {
        operation,
        inputs,
        outputs,
        params,
        meta,
        signature,
    })
}

fn read_operation<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::operation::Reader,
) -> ReadResult<model::Operation<'a>> {
    use hugr_capnp::operation::Which;
    Ok(match reader.which()? {
        Which::Invalid(()) => model::Operation::Invalid,
        Which::Dfg(body) => model::Operation::Dfg {
            body: model::RegionId(body),
        },
        Which::Cfg(body) => model::Operation::Cfg {
            body: model::RegionId(body),
        },
        Which::Block(body) => model::Operation::Block {
            body: model::RegionId(body),
        },
        Which::FuncDefn(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader.get_constraints()?, model::TermId);
            let signature = model::TermId(reader.get_signature());
            let body = model::RegionId(reader.get_body());
            let decl = bump.alloc(model::FuncDecl {
                name,
                params,
                constraints,
                signature,
            });
            model::Operation::DefineFunc { decl, body }
        }
        Which::FuncDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader.get_constraints()?, model::TermId);
            let signature = model::TermId(reader.get_signature());
            let decl = bump.alloc(model::FuncDecl {
                name,
                params,
                constraints,
                signature,
            });
            model::Operation::DeclareFunc { decl }
        }
        Which::AliasDefn(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
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
            let params = read_list!(bump, reader.get_params()?, read_param);
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
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader.get_constraints()?, model::TermId);
            let r#type = model::TermId(reader.get_type());
            let decl = bump.alloc(model::ConstructorDecl {
                name,
                params,
                constraints,
                r#type,
            });
            model::Operation::DeclareConstructor { decl }
        }
        Which::OperationDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader.get_constraints()?, model::TermId);
            let r#type = model::TermId(reader.get_type());
            let decl = bump.alloc(model::OperationDecl {
                name,
                params,
                constraints,
                r#type,
            });
            model::Operation::DeclareOperation { decl }
        }
        Which::Custom(operation) => model::Operation::Custom {
            operation: model::NodeId(operation),
        },
        Which::CustomFull(operation) => model::Operation::CustomFull {
            operation: model::NodeId(operation),
        },
        Which::Tag(tag) => model::Operation::Tag { tag },
        Which::TailLoop(body) => model::Operation::TailLoop {
            body: model::RegionId(body),
        },
        Which::Conditional(branches) => {
            let branches = read_scalar_list!(bump, branches?, model::RegionId);
            model::Operation::Conditional { branches }
        }
        Which::CallFunc(func) => model::Operation::CallFunc {
            func: model::TermId(func),
        },
        Which::LoadFunc(func) => model::Operation::LoadFunc {
            func: model::TermId(func),
        },
        Which::Import(name) => model::Operation::Import {
            name: bump.alloc_str(name?.to_str()?),
        },
        Which::Const(value) => model::Operation::Const {
            value: model::TermId(value),
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
        hugr_capnp::RegionKind::Module => model::RegionKind::Module,
    };

    let sources = read_scalar_list!(bump, reader.get_sources()?, model::LinkIndex);
    let targets = read_scalar_list!(bump, reader.get_targets()?, model::LinkIndex);
    let children = read_scalar_list!(bump, reader.get_children()?, model::NodeId);
    let meta = read_list!(bump, reader.get_meta()?, read_meta_item);
    let signature = reader.get_signature().checked_sub(1).map(model::TermId);

    let scope = if reader.has_scope() {
        Some(read_region_scope(reader.get_scope()?)?)
    } else {
        None
    };

    Ok(model::Region {
        kind,
        sources,
        targets,
        children,
        meta,
        signature,
        scope,
    })
}

fn read_region_scope(reader: hugr_capnp::region_scope::Reader) -> ReadResult<model::RegionScope> {
    let links = reader.get_links();
    let ports = reader.get_ports();
    Ok(model::RegionScope { links, ports })
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

        Which::Variable(reader) => {
            let node = model::NodeId(reader.get_variable_node());
            let index = reader.get_variable_index();
            model::Term::Var(model::VarId(node, index))
        }

        Which::Apply(reader) => {
            let reader = reader?;
            let symbol = model::NodeId(reader.get_symbol());
            let args = read_scalar_list!(bump, reader.get_args()?, model::TermId);
            model::Term::Apply { symbol, args }
        }

        Which::ApplyFull(reader) => {
            let reader = reader?;
            let symbol = model::NodeId(reader.get_symbol());
            let args = read_scalar_list!(bump, reader.get_args()?, model::TermId);
            model::Term::ApplyFull { symbol, args }
        }

        Which::Const(r#type) => model::Term::Const {
            r#type: model::TermId(r#type),
        },

        Which::List(reader) => {
            let reader = reader?;
            let parts = read_list!(bump, reader.get_items()?, read_list_part);
            model::Term::List { parts }
        }

        Which::ListType(item_type) => model::Term::ListType {
            item_type: model::TermId(item_type),
        },

        Which::ExtSet(reader) => {
            let reader = reader?;
            let parts = read_list!(bump, reader.get_items()?, read_ext_set_part);
            model::Term::ExtSet { parts }
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

        Which::NonLinearConstraint(term) => model::Term::NonLinearConstraint {
            term: model::TermId(term),
        },

        Which::ConstFunc(region) => model::Term::ConstFunc {
            region: model::RegionId(region),
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

fn read_list_part(
    _: &Bump,
    reader: hugr_capnp::term::list_part::Reader,
) -> ReadResult<model::ListPart> {
    use hugr_capnp::term::list_part::Which;
    Ok(match reader.which()? {
        Which::Item(term) => model::ListPart::Item(model::TermId(term)),
        Which::Splice(list) => model::ListPart::Splice(model::TermId(list)),
    })
}

fn read_ext_set_part<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::term::ext_set_part::Reader,
) -> ReadResult<model::ExtSetPart<'a>> {
    use hugr_capnp::term::ext_set_part::Which;
    Ok(match reader.which()? {
        Which::Extension(ext) => model::ExtSetPart::Extension(bump.alloc_str(ext?.to_str()?)),
        Which::Splice(list) => model::ExtSetPart::Splice(model::TermId(list)),
    })
}

fn read_param<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::param::Reader,
) -> ReadResult<model::Param<'a>> {
    let name = bump.alloc_str(reader.get_name()?.to_str()?);
    let r#type = model::TermId(reader.get_type());

    let sort = match reader.get_sort()? {
        hugr_capnp::ParamSort::Implicit => model::ParamSort::Implicit,
        hugr_capnp::ParamSort::Explicit => model::ParamSort::Explicit,
    };

    Ok(model::Param { name, r#type, sort })
}
