use crate::capnp::hugr_v0_capnp as hugr_capnp;
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
    let inputs = read_scalar_list!(bump, reader, get_inputs, model::LinkIndex);
    let outputs = read_scalar_list!(bump, reader, get_outputs, model::LinkIndex);
    let params = read_scalar_list!(bump, reader, get_params, model::TermId);
    let regions = read_scalar_list!(bump, reader, get_regions, model::RegionId);
    let meta = read_scalar_list!(bump, reader, get_meta, model::TermId);
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
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader, get_constraints, model::TermId);
            let signature = model::TermId(reader.get_signature());
            let symbol = bump.alloc(model::Symbol {
                name,
                params,
                constraints,
                signature,
            });
            model::Operation::DefineFunc(symbol)
        }
        Which::FuncDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader, get_constraints, model::TermId);
            let signature = model::TermId(reader.get_signature());
            let symbol = bump.alloc(model::Symbol {
                name,
                params,
                constraints,
                signature,
            });
            model::Operation::DeclareFunc(symbol)
        }
        Which::AliasDefn(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let signature = model::TermId(reader.get_signature());
            let symbol = bump.alloc(model::Symbol {
                name,
                params,
                constraints: &[],
                signature,
            });
            model::Operation::DefineAlias(symbol)
        }
        Which::AliasDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let signature = model::TermId(reader.get_signature());
            let symbol = bump.alloc(model::Symbol {
                name,
                params,
                constraints: &[],
                signature,
            });
            model::Operation::DeclareAlias(symbol)
        }
        Which::ConstructorDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader, get_constraints, model::TermId);
            let signature = model::TermId(reader.get_signature());
            let symbol = bump.alloc(model::Symbol {
                name,
                params,
                constraints,
                signature,
            });
            model::Operation::DeclareConstructor(symbol)
        }
        Which::OperationDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader, get_constraints, model::TermId);
            let signature = model::TermId(reader.get_signature());
            let symbol = bump.alloc(model::Symbol {
                name,
                params,
                constraints,
                signature,
            });
            model::Operation::DeclareOperation(symbol)
        }
        Which::Custom(operation) => model::Operation::Custom(model::NodeId(operation)),
        Which::TailLoop(()) => model::Operation::TailLoop,
        Which::Conditional(()) => model::Operation::Conditional,
        Which::Import(name) => model::Operation::Import {
            name: bump.alloc_str(name?.to_str()?),
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

    let sources = read_scalar_list!(bump, reader, get_sources, model::LinkIndex);
    let targets = read_scalar_list!(bump, reader, get_targets, model::LinkIndex);
    let children = read_scalar_list!(bump, reader, get_children, model::NodeId);
    let meta = read_scalar_list!(bump, reader, get_meta, model::TermId);
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
        Which::String(value) => model::Term::Str(bump.alloc_str(value?.to_str()?)),
        Which::Nat(value) => model::Term::Nat(value),

        Which::Variable(reader) => {
            let node = model::NodeId(reader.get_node());
            let index = reader.get_index();
            model::Term::Var(model::VarId(node, index))
        }

        Which::Apply(reader) => {
            let symbol = model::NodeId(reader.get_symbol());
            let args = read_scalar_list!(bump, reader, get_args, model::TermId);
            model::Term::Apply(symbol, args)
        }

        Which::List(reader) => {
            let parts = read_list!(bump, reader?, read_list_part);
            model::Term::List(parts)
        }

        Which::ExtSet(reader) => {
            let parts = read_list!(bump, reader?, read_ext_set_part);
            model::Term::ExtSet(parts)
        }

        Which::Tuple(reader) => {
            let parts = read_list!(bump, reader?, read_tuple_part);
            model::Term::Tuple(parts)
        }

        Which::ConstFunc(region) => model::Term::ConstFunc(model::RegionId(region)),

        Which::Bytes(bytes) => model::Term::Bytes(bump.alloc_slice_copy(bytes?)),
        Which::Float(value) => model::Term::Float(value.into()),
    })
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

fn read_tuple_part(
    _: &Bump,
    reader: hugr_capnp::term::tuple_part::Reader,
) -> ReadResult<model::TuplePart> {
    use hugr_capnp::term::tuple_part::Which;
    Ok(match reader.which()? {
        Which::Item(term) => model::TuplePart::Item(model::TermId(term)),
        Which::Splice(list) => model::TuplePart::Splice(model::TermId(list)),
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
    Ok(model::Param { name, r#type })
}
