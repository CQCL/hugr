use crate::capnp::hugr_v0_capnp as hugr_capnp;
use crate::v0 as model;
use crate::v0::table;
use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use std::io::BufRead;

/// An error encounted while deserialising a model.
#[derive(Debug, derive_more::From, derive_more::Display, derive_more::Error)]
#[non_exhaustive]
pub enum ReadError {
    #[from(forward)]
    /// An error encounted while decoding a model from a `capnproto` buffer.
    DecodingError(capnp::Error),
}

type ReadResult<T> = Result<T, ReadError>;

/// Read a hugr package from a byte slice.
pub fn read_from_slice<'a>(slice: &[u8], bump: &'a Bump) -> ReadResult<table::Package<'a>> {
    read_from_reader(slice, bump)
}

/// Read a hugr package from an impl of [`BufRead`].
pub fn read_from_reader(reader: impl BufRead, bump: &Bump) -> ReadResult<table::Package<'_>> {
    let reader =
        capnp::serialize_packed::read_message(reader, capnp::message::ReaderOptions::new())?;
    let root = reader.get_root::<hugr_capnp::package::Reader>()?;
    read_package(bump, root)
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

fn read_package<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::package::Reader,
) -> ReadResult<table::Package<'a>> {
    let modules = reader
        .get_modules()?
        .iter()
        .map(|m| read_module(bump, m))
        .collect::<ReadResult<_>>()?;

    Ok(table::Package { modules })
}

fn read_module<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::module::Reader,
) -> ReadResult<table::Module<'a>> {
    let root = table::RegionId(reader.get_root());

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

    Ok(table::Module {
        root,
        nodes,
        regions,
        terms,
    })
}

fn read_node<'a>(bump: &'a Bump, reader: hugr_capnp::node::Reader) -> ReadResult<table::Node<'a>> {
    let operation = read_operation(bump, reader.get_operation()?)?;
    let inputs = read_scalar_list!(bump, reader, get_inputs, table::LinkIndex);
    let outputs = read_scalar_list!(bump, reader, get_outputs, table::LinkIndex);
    let regions = read_scalar_list!(bump, reader, get_regions, table::RegionId);
    let meta = read_scalar_list!(bump, reader, get_meta, table::TermId);
    let signature = reader.get_signature().checked_sub(1).map(table::TermId);

    Ok(table::Node {
        operation,
        inputs,
        outputs,
        regions,
        meta,
        signature,
    })
}

fn read_operation<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::operation::Reader,
) -> ReadResult<table::Operation<'a>> {
    use hugr_capnp::operation::Which;
    Ok(match reader.which()? {
        Which::Invalid(()) => table::Operation::Invalid,
        Which::Dfg(()) => table::Operation::Dfg,
        Which::Cfg(()) => table::Operation::Cfg,
        Which::Block(()) => table::Operation::Block,
        Which::FuncDefn(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader, get_constraints, table::TermId);
            let signature = table::TermId(reader.get_signature());
            let symbol = bump.alloc(table::Symbol {
                name,
                params,
                constraints,
                signature,
            });
            table::Operation::DefineFunc(symbol)
        }
        Which::FuncDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader, get_constraints, table::TermId);
            let signature = table::TermId(reader.get_signature());
            let symbol = bump.alloc(table::Symbol {
                name,
                params,
                constraints,
                signature,
            });
            table::Operation::DeclareFunc(symbol)
        }
        Which::AliasDefn(reader) => {
            let symbol = reader.get_symbol()?;
            let value = table::TermId(reader.get_value());
            let name = bump.alloc_str(symbol.get_name()?.to_str()?);
            let params = read_list!(bump, symbol.get_params()?, read_param);
            let signature = table::TermId(symbol.get_signature());
            let symbol = bump.alloc(table::Symbol {
                name,
                params,
                constraints: &[],
                signature,
            });
            table::Operation::DefineAlias(symbol, value)
        }
        Which::AliasDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let signature = table::TermId(reader.get_signature());
            let symbol = bump.alloc(table::Symbol {
                name,
                params,
                constraints: &[],
                signature,
            });
            table::Operation::DeclareAlias(symbol)
        }
        Which::ConstructorDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader, get_constraints, table::TermId);
            let signature = table::TermId(reader.get_signature());
            let symbol = bump.alloc(table::Symbol {
                name,
                params,
                constraints,
                signature,
            });
            table::Operation::DeclareConstructor(symbol)
        }
        Which::OperationDecl(reader) => {
            let reader = reader?;
            let name = bump.alloc_str(reader.get_name()?.to_str()?);
            let params = read_list!(bump, reader.get_params()?, read_param);
            let constraints = read_scalar_list!(bump, reader, get_constraints, table::TermId);
            let signature = table::TermId(reader.get_signature());
            let symbol = bump.alloc(table::Symbol {
                name,
                params,
                constraints,
                signature,
            });
            table::Operation::DeclareOperation(symbol)
        }
        Which::Custom(operation) => table::Operation::Custom(table::TermId(operation)),
        Which::TailLoop(()) => table::Operation::TailLoop,
        Which::Conditional(()) => table::Operation::Conditional,
        Which::Import(name) => table::Operation::Import {
            name: bump.alloc_str(name?.to_str()?),
        },
    })
}

fn read_region<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::region::Reader,
) -> ReadResult<table::Region<'a>> {
    let kind = match reader.get_kind()? {
        hugr_capnp::RegionKind::DataFlow => model::RegionKind::DataFlow,
        hugr_capnp::RegionKind::ControlFlow => model::RegionKind::ControlFlow,
        hugr_capnp::RegionKind::Module => model::RegionKind::Module,
    };

    let sources = read_scalar_list!(bump, reader, get_sources, table::LinkIndex);
    let targets = read_scalar_list!(bump, reader, get_targets, table::LinkIndex);
    let children = read_scalar_list!(bump, reader, get_children, table::NodeId);
    let meta = read_scalar_list!(bump, reader, get_meta, table::TermId);
    let signature = reader.get_signature().checked_sub(1).map(table::TermId);

    let scope = if reader.has_scope() {
        Some(read_region_scope(reader.get_scope()?)?)
    } else {
        None
    };

    Ok(table::Region {
        kind,
        sources,
        targets,
        children,
        meta,
        signature,
        scope,
    })
}

fn read_region_scope(reader: hugr_capnp::region_scope::Reader) -> ReadResult<table::RegionScope> {
    let links = reader.get_links();
    let ports = reader.get_ports();
    Ok(table::RegionScope { links, ports })
}

fn read_term<'a>(bump: &'a Bump, reader: hugr_capnp::term::Reader) -> ReadResult<table::Term<'a>> {
    use hugr_capnp::term::Which;
    Ok(match reader.which()? {
        Which::Wildcard(()) => table::Term::Wildcard,
        Which::String(value) => table::Term::Literal(model::Literal::Str(value?.to_str()?.into())),
        Which::Nat(value) => table::Term::Literal(model::Literal::Nat(value)),

        Which::Variable(reader) => {
            let node = table::NodeId(reader.get_node());
            let index = reader.get_index();
            table::Term::Var(table::VarId(node, index))
        }

        Which::Apply(reader) => {
            let symbol = table::NodeId(reader.get_symbol());
            let args = read_scalar_list!(bump, reader, get_args, table::TermId);
            table::Term::Apply(symbol, args)
        }

        Which::List(reader) => {
            let parts = read_list!(bump, reader?, read_seq_part);
            table::Term::List(parts)
        }

        Which::Tuple(reader) => {
            let parts = read_list!(bump, reader?, read_seq_part);
            table::Term::Tuple(parts)
        }

        Which::Func(region) => table::Term::Func(table::RegionId(region)),

        Which::Bytes(bytes) => table::Term::Literal(model::Literal::Bytes(bytes?.into())),
        Which::Float(value) => table::Term::Literal(model::Literal::Float(value.into())),
    })
}

fn read_seq_part(
    _: &Bump,
    reader: hugr_capnp::term::seq_part::Reader,
) -> ReadResult<table::SeqPart> {
    use hugr_capnp::term::seq_part::Which;
    Ok(match reader.which()? {
        Which::Item(term) => table::SeqPart::Item(table::TermId(term)),
        Which::Splice(list) => table::SeqPart::Splice(table::TermId(list)),
    })
}

fn read_param<'a>(
    bump: &'a Bump,
    reader: hugr_capnp::param::Reader,
) -> ReadResult<table::Param<'a>> {
    let name = bump.alloc_str(reader.get_name()?.to_str()?);
    let r#type = table::TermId(reader.get_type());
    Ok(table::Param { name, r#type })
}
