//! Importing HUGR graphs from their `hugr-model` representation.
//!
//! **Warning**: This module is still under development and is expected to change.
//! It is included in the library to allow for early experimentation, and for
//! the core and model to converge incrementally.
use std::sync::Arc;

use crate::{
    Direction, Hugr, HugrView, Node, Port,
    extension::{
        ExtensionId, ExtensionRegistry, SignatureError, resolution::ExtensionResolutionError,
    },
    hugr::{HugrMut, NodeMetadata},
    ops::{
        AliasDecl, AliasDefn, CFG, Call, CallIndirect, Case, Conditional, Const, DFG,
        DataflowBlock, ExitBlock, FuncDecl, FuncDefn, Input, LoadConstant, LoadFunction, OpType,
        OpaqueOp, Output, Tag, TailLoop, Value,
        constant::{CustomConst, CustomSerialized, OpaqueValue},
    },
    package::Package,
    std_extensions::{
        arithmetic::{float_types::ConstF64, int_types::ConstInt},
        collections::array::ArrayValue,
    },
    types::{
        CustomType, FuncTypeBase, MaybeRV, PolyFuncType, PolyFuncTypeBase, RowVariable, Signature,
        Term, Type, TypeArg, TypeBase, TypeBound, TypeEnum, TypeName, TypeRow,
        type_param::{SeqPart, TypeParam},
        type_row::TypeRowBase,
    },
};
use fxhash::FxHashMap;
use hugr_model::v0 as model;
use hugr_model::v0::table;
use itertools::{Either, Itertools};
use smol_str::{SmolStr, ToSmolStr};
use thiserror::Error;

/// An error that can occur during import.
#[derive(Debug, Clone, Error)]
#[error("failed to import hugr")]
pub struct ImportError(#[from] ImportErrorInner);

#[derive(Debug, Clone, Error)]
enum ImportErrorInner {
    /// The model contains a feature that is not supported by the importer yet.
    /// Errors of this kind are expected to be removed as the model format and
    /// the core HUGR representation converge.
    #[error("currently unsupported: {0}")]
    Unsupported(String),

    /// The model contains implicit information that has not yet been inferred.
    /// This includes wildcards and application of functions with implicit parameters.
    #[error("uninferred implicit: {0}")]
    Uninferred(String),

    /// The model is not well-formed.
    #[error("{0}")]
    Invalid(String),

    /// An error with additional context.
    #[error("import failed in context: {1}")]
    Context(#[source] Box<ImportErrorInner>, String),

    /// A signature mismatch was detected during import.
    #[error("signature error")]
    Signature(#[from] SignatureError),

    /// An error relating to the loaded extension registry.
    #[error("extension error")]
    Extension(#[from] ExtensionError),

    /// Incorrect order hints.
    #[error("incorrect order hint")]
    OrderHint(#[from] OrderHintError),

    /// Extension resolution.
    #[error("extension resolution error")]
    ExtensionResolution(#[from] ExtensionResolutionError),
}

#[derive(Debug, Clone, Error)]
enum ExtensionError {
    /// An extension is missing.
    #[error("Importing the hugr requires extension {missing_ext}, which was not found in the registry. The available extensions are: [{}]",
            available.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(", "))]
    Missing {
        /// The missing extension.
        missing_ext: ExtensionId,
        /// The available extensions in the registry.
        available: Vec<ExtensionId>,
    },

    /// An extension type is missing.
    #[error(
        "Importing the hugr requires extension {ext} to have a type named {name}, but it was not found."
    )]
    MissingType {
        /// The extension that is missing the type.
        ext: ExtensionId,
        /// The name of the missing type.
        name: TypeName,
    },
}

impl From<ExtensionError> for ImportError {
    fn from(value: ExtensionError) -> Self {
        Self::from(ImportErrorInner::from(value))
    }
}

/// Import error caused by incorrect order hints.
#[derive(Debug, Clone, Error)]
enum OrderHintError {
    /// Duplicate order hint key in the same region.
    #[error("duplicate order hint key {0}")]
    DuplicateKey(table::NodeId, u64),
    /// Order hint including a key not defined in the region.
    #[error("order hint with unknown key {0}")]
    UnknownKey(u64),
    /// Order hint involving a node with no order port.
    #[error("order hint on node with no order port: {0}")]
    NoOrderPort(table::NodeId),
}

impl From<OrderHintError> for ImportError {
    fn from(value: OrderHintError) -> Self {
        Self::from(ImportErrorInner::from(value))
    }
}

/// Helper macro to create an `ImportError::Unsupported` error with a formatted message.
macro_rules! error_unsupported {
    ($($e:expr),*) => { ImportError(ImportErrorInner::Unsupported(format!($($e),*))) }
}

/// Helper macro to create an `ImportError::Uninferred` error with a formatted message.
macro_rules! error_uninferred {
    ($($e:expr),*) => { ImportError(ImportErrorInner::Uninferred(format!($($e),*))) }
}

/// Helper macro to create an `ImportError::Invalid` error with a formatted message.
macro_rules! error_invalid {
    ($($e:expr),*) => { ImportError(ImportErrorInner::Invalid(format!($($e),*))) }
}

/// Helper macro to create an `ImportError::Context` error with a formatted message.
macro_rules! error_context {
    ($err:expr, $($e:expr),*) => {
        {
            let ImportError(__err) = $err;
            ImportError(ImportErrorInner::Context(Box::new(__err), format!($($e),*)))
        }
    }
}

/// Import a [`Package`] from its model representation.
pub fn import_package(
    package: &table::Package,
    extensions: &ExtensionRegistry,
) -> Result<Package, ImportError> {
    let modules = package
        .modules
        .iter()
        .map(|module| import_hugr(module, extensions))
        .collect::<Result<Vec<_>, _>>()?;

    // This does not panic since the import already requires a module root.
    let package = Package::new(modules);
    Ok(package)
}

/// Import a [`Hugr`] module from its model representation.
pub fn import_hugr(
    module: &table::Module,
    extensions: &ExtensionRegistry,
) -> Result<Hugr, ImportError> {
    // TODO: Module should know about the number of edges, so that we can use a vector here.
    // For now we use a hashmap, which will be slower.
    let mut ctx = Context {
        module,
        hugr: Hugr::new(),
        link_ports: FxHashMap::default(),
        static_edges: Vec::new(),
        extensions,
        nodes: FxHashMap::default(),
        local_vars: FxHashMap::default(),
        custom_name_cache: FxHashMap::default(),
        region_scope: table::RegionId::default(),
    };

    ctx.import_root()?;
    ctx.link_ports()?;
    ctx.link_static_ports()?;

    ctx.hugr
        .resolve_extension_defs(extensions)
        .map_err(ImportErrorInner::ExtensionResolution)?;

    Ok(ctx.hugr)
}

struct Context<'a> {
    /// The module being imported.
    module: &'a table::Module<'a>,

    /// The HUGR graph being constructed.
    hugr: Hugr,

    /// The ports that are part of each link. This is used to connect the ports at the end of the
    /// import process.
    link_ports: FxHashMap<(table::RegionId, table::LinkIndex), Vec<(Node, Port)>>,

    /// Pairs of nodes that should be connected by a static edge.
    /// These are collected during the import process and connected at the end.
    static_edges: Vec<(table::NodeId, table::NodeId)>,

    /// The ambient extension registry to use for importing.
    extensions: &'a ExtensionRegistry,

    /// A map from `NodeId` to the imported `Node`.
    nodes: FxHashMap<table::NodeId, Node>,

    local_vars: FxHashMap<table::VarId, LocalVar>,

    custom_name_cache: FxHashMap<&'a str, (ExtensionId, SmolStr)>,

    region_scope: table::RegionId,
}

impl<'a> Context<'a> {
    /// Get the signature of the node with the given `NodeId`.
    fn get_node_signature(&mut self, node: table::NodeId) -> Result<Signature, ImportError> {
        let node_data = self.get_node(node)?;
        let signature = node_data
            .signature
            .ok_or_else(|| error_uninferred!("node signature"))?;
        self.import_func_type(signature)
    }

    /// Get the node with the given `NodeId`, or return an error if it does not exist.
    #[inline]
    fn get_node(&self, node_id: table::NodeId) -> Result<&'a table::Node<'a>, ImportError> {
        self.module
            .get_node(node_id)
            .ok_or_else(|| error_invalid!("unknown node {}", node_id))
    }

    /// Get the term with the given `TermId`, or return an error if it does not exist.
    #[inline]
    fn get_term(&self, term_id: table::TermId) -> Result<&'a table::Term<'a>, ImportError> {
        self.module
            .get_term(term_id)
            .ok_or_else(|| error_invalid!("unknown term {}", term_id))
    }

    /// Get the region with the given `RegionId`, or return an error if it does not exist.
    #[inline]
    fn get_region(&self, region_id: table::RegionId) -> Result<&'a table::Region<'a>, ImportError> {
        self.module
            .get_region(region_id)
            .ok_or_else(|| error_invalid!("unknown region {}", region_id))
    }

    fn make_node(
        &mut self,
        node_id: table::NodeId,
        op: OpType,
        parent: Node,
    ) -> Result<Node, ImportError> {
        let node = self.hugr.add_node_with_parent(parent, op);
        self.nodes.insert(node_id, node);

        let node_data = self.get_node(node_id)?;
        self.record_links(node, Direction::Incoming, node_data.inputs);
        self.record_links(node, Direction::Outgoing, node_data.outputs);

        for meta_item in node_data.meta {
            self.import_node_metadata(node, *meta_item)
                .map_err(|err| error_context!(err, "node metadata"))?;
        }

        Ok(node)
    }

    fn import_node_metadata(
        &mut self,
        node: Node,
        meta_item: table::TermId,
    ) -> Result<(), ImportError> {
        // Import the JSON metadata
        if let Some([name_arg, json_arg]) = self.match_symbol(meta_item, model::COMPAT_META_JSON)? {
            let table::Term::Literal(model::Literal::Str(name)) = self.get_term(name_arg)? else {
                return Err(error_invalid!(
                    "`{}` expects a string literal as its first argument",
                    model::COMPAT_META_JSON
                ));
            };

            let table::Term::Literal(model::Literal::Str(json_str)) = self.get_term(json_arg)?
            else {
                return Err(error_invalid!(
                    "`{}` expects a string literal as its second argument",
                    model::COMPAT_CONST_JSON
                ));
            };

            let json_value: NodeMetadata = serde_json::from_str(json_str).map_err(|_| {
                error_invalid!(
                    "failed to parse JSON string for `{}` metadata",
                    model::COMPAT_CONST_JSON
                )
            })?;

            self.hugr.set_metadata(node, name, json_value);
        }

        // Set the entrypoint
        if let Some([]) = self.match_symbol(meta_item, model::CORE_ENTRYPOINT)? {
            self.hugr.set_entrypoint(node);
        }

        Ok(())
    }

    /// Associate links with the ports of the given node in the given direction.
    fn record_links(&mut self, node: Node, direction: Direction, links: &'a [table::LinkIndex]) {
        let optype = self.hugr.get_optype(node);
        // NOTE: `OpType::port_count` copies the signature, which significantly slows down the import.
        debug_assert!(links.len() <= optype.port_count(direction));

        for (link, port) in links.iter().zip(self.hugr.node_ports(node, direction)) {
            self.link_ports
                .entry((self.region_scope, *link))
                .or_default()
                .push((node, port));
        }
    }

    /// Link up the ports in the hugr graph, according to the connectivity information that
    /// has been gathered in the `link_ports` map.
    fn link_ports(&mut self) -> Result<(), ImportError> {
        // For each edge, we group the ports by their direction. We reuse the `inputs` and
        // `outputs` vectors to avoid unnecessary allocations.
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for (link_id, link_ports) in std::mem::take(&mut self.link_ports) {
            // Skip the edge if it doesn't have any ports.
            if link_ports.is_empty() {
                continue;
            }

            for (node, port) in link_ports {
                match port.as_directed() {
                    Either::Left(input) => inputs.push((node, input)),
                    Either::Right(output) => outputs.push((node, output)),
                }
            }

            match (inputs.as_slice(), outputs.as_slice()) {
                ([], []) => {
                    unreachable!();
                }
                (_, [output]) => {
                    for (node, port) in &inputs {
                        self.hugr.connect(output.0, output.1, *node, *port);
                    }
                }
                ([input], _) => {
                    for (node, port) in &outputs {
                        self.hugr.connect(*node, *port, input.0, input.1);
                    }
                }
                _ => {
                    return Err(error_unsupported!(
                        "link {:?} would require hyperedge",
                        link_id
                    ));
                }
            }

            inputs.clear();
            outputs.clear();
        }

        Ok(())
    }

    fn link_static_ports(&mut self) -> Result<(), ImportError> {
        for (src_id, dst_id) in std::mem::take(&mut self.static_edges) {
            // None of these lookups should fail given how we constructed `static_edges`.
            let src = self.nodes[&src_id];
            let dst = self.nodes[&dst_id];
            let src_port = self.hugr.get_optype(src).static_output_port().unwrap();
            let dst_port = self.hugr.get_optype(dst).static_input_port().unwrap();
            self.hugr.connect(src, src_port, dst, dst_port);
        }

        Ok(())
    }

    fn get_symbol_name(&self, node_id: table::NodeId) -> Result<&'a str, ImportError> {
        let node_data = self.get_node(node_id)?;
        let name = node_data
            .operation
            .symbol()
            .ok_or_else(|| error_invalid!("node {} is expected to be a symbol", node_id))?;
        Ok(name)
    }

    fn get_func_signature(
        &mut self,
        func_node: table::NodeId,
    ) -> Result<PolyFuncType, ImportError> {
        let symbol = match self.get_node(func_node)?.operation {
            table::Operation::DefineFunc(symbol) => symbol,
            table::Operation::DeclareFunc(symbol) => symbol,
            _ => {
                return Err(error_invalid!(
                    "node {} is expected to be a function declaration or definition",
                    func_node
                ));
            }
        };

        self.import_poly_func_type(func_node, *symbol, |_, signature| Ok(signature))
    }

    /// Import the root region of the module.
    fn import_root(&mut self) -> Result<(), ImportError> {
        self.region_scope = self.module.root;
        let region_data = self.get_region(self.module.root)?;

        for node in region_data.children {
            self.import_node(*node, self.hugr.entrypoint())?;
        }

        for meta_item in region_data.meta {
            self.import_node_metadata(self.hugr.module_root(), *meta_item)?;
        }

        Ok(())
    }

    fn import_node(
        &mut self,
        node_id: table::NodeId,
        parent: Node,
    ) -> Result<Option<Node>, ImportError> {
        let node_data = self.get_node(node_id)?;

        let result = match node_data.operation {
            table::Operation::Invalid => {
                return Err(error_invalid!("tried to import an `invalid` operation"));
            }

            table::Operation::Dfg => Some(
                self.import_node_dfg(node_id, parent, node_data)
                    .map_err(|err| error_context!(err, "`dfg` node with id {}", node_id))?,
            ),

            table::Operation::Cfg => Some(
                self.import_node_cfg(node_id, parent, node_data)
                    .map_err(|err| error_context!(err, "`cfg` node with id {}", node_id))?,
            ),

            table::Operation::Block => Some(
                self.import_node_block(node_id, parent)
                    .map_err(|err| error_context!(err, "`block` node with id {}", node_id))?,
            ),

            table::Operation::DefineFunc(symbol) => Some(
                self.import_node_define_func(node_id, symbol, node_data, parent)
                    .map_err(|err| error_context!(err, "`define-func` node with id {}", node_id))?,
            ),

            table::Operation::DeclareFunc(symbol) => Some(
                self.import_node_declare_func(node_id, symbol, parent)
                    .map_err(|err| {
                        error_context!(err, "`declare-func` node with id {}", node_id)
                    })?,
            ),

            table::Operation::TailLoop => Some(
                self.import_tail_loop(node_id, parent)
                    .map_err(|err| error_context!(err, "`tail-loop` node with id {}", node_id))?,
            ),

            table::Operation::Conditional => Some(
                self.import_conditional(node_id, parent)
                    .map_err(|err| error_context!(err, "`cond` node with id {}", node_id))?,
            ),

            table::Operation::Custom(operation) => Some(
                self.import_node_custom(node_id, operation, node_data, parent)
                    .map_err(|err| error_context!(err, "custom node with id {}", node_id))?,
            ),

            table::Operation::DefineAlias(symbol, value) => Some(
                self.import_node_define_alias(node_id, symbol, value, parent)
                    .map_err(|err| {
                        error_context!(err, "`define-alias` node with id {}", node_id)
                    })?,
            ),

            table::Operation::DeclareAlias(symbol) => Some(
                self.import_node_declare_alias(node_id, symbol, parent)
                    .map_err(|err| {
                        error_context!(err, "`declare-alias` node with id {}", node_id)
                    })?,
            ),

            table::Operation::Import { .. } => None,

            table::Operation::DeclareConstructor { .. } => None,
            table::Operation::DeclareOperation { .. } => None,
        };

        Ok(result)
    }

    fn import_node_dfg(
        &mut self,
        node_id: table::NodeId,
        parent: Node,
        node_data: &'a table::Node<'a>,
    ) -> Result<Node, ImportError> {
        let signature = self
            .get_node_signature(node_id)
            .map_err(|err| error_context!(err, "node signature"))?;

        let optype = OpType::DFG(DFG { signature });
        let node = self.make_node(node_id, optype, parent)?;

        let [region] = node_data.regions else {
            return Err(error_invalid!("dfg region expects a single region"));
        };

        self.import_dfg_region(*region, node)?;
        Ok(node)
    }

    fn import_node_cfg(
        &mut self,
        node_id: table::NodeId,
        parent: Node,
        node_data: &'a table::Node<'a>,
    ) -> Result<Node, ImportError> {
        let signature = self
            .get_node_signature(node_id)
            .map_err(|err| error_context!(err, "node signature"))?;

        let optype = OpType::CFG(CFG { signature });
        let node = self.make_node(node_id, optype, parent)?;

        let [region] = node_data.regions else {
            return Err(error_invalid!("cfg nodes expect a single region"));
        };

        self.import_cfg_region(*region, node)?;
        Ok(node)
    }

    fn import_dfg_region(
        &mut self,
        region: table::RegionId,
        node: Node,
    ) -> Result<(), ImportError> {
        let region_data = self.get_region(region)?;

        let prev_region = self.region_scope;
        if region_data.scope.is_some() {
            self.region_scope = region;
        }

        if region_data.kind != model::RegionKind::DataFlow {
            return Err(error_invalid!("expected dfg region"));
        }

        let signature = self
            .import_func_type(
                region_data
                    .signature
                    .ok_or_else(|| error_uninferred!("region signature"))?,
            )
            .map_err(|err| error_context!(err, "signature of dfg region with id {}", region))?;

        // Create the input and output nodes
        let input = self.hugr.add_node_with_parent(
            node,
            OpType::Input(Input {
                types: signature.input,
            }),
        );
        let output = self.hugr.add_node_with_parent(
            node,
            OpType::Output(Output {
                types: signature.output,
            }),
        );

        // Make sure that the ports of the input/output nodes are connected correctly
        self.record_links(input, Direction::Outgoing, region_data.sources);
        self.record_links(output, Direction::Incoming, region_data.targets);

        for child in region_data.children {
            self.import_node(*child, node)?;
        }

        self.create_order_edges(region)?;

        for meta_item in region_data.meta {
            self.import_node_metadata(node, *meta_item)?;
        }

        self.region_scope = prev_region;

        Ok(())
    }

    /// Create order edges between nodes of a dataflow region based on order hint metadata.
    ///
    /// This method assumes that the nodes for the children of the region have already been imported.
    fn create_order_edges(&mut self, region_id: table::RegionId) -> Result<(), ImportError> {
        let region_data = self.get_region(region_id)?;
        debug_assert_eq!(region_data.kind, model::RegionKind::DataFlow);

        // Collect order hint keys
        // PERFORMANCE: It might be worthwhile to reuse the map to avoid allocations.
        let mut order_keys = FxHashMap::<u64, table::NodeId>::default();

        for child_id in region_data.children {
            let child_data = self.get_node(*child_id)?;

            for meta_id in child_data.meta {
                let Some([key]) = self.match_symbol(*meta_id, model::ORDER_HINT_KEY)? else {
                    continue;
                };

                let table::Term::Literal(model::Literal::Nat(key)) = self.get_term(key)? else {
                    continue;
                };

                if order_keys.insert(*key, *child_id).is_some() {
                    return Err(OrderHintError::DuplicateKey(*child_id, *key).into());
                }
            }
        }

        // Insert order edges
        for meta_id in region_data.meta {
            let Some([a, b]) = self.match_symbol(*meta_id, model::ORDER_HINT_ORDER)? else {
                continue;
            };

            let table::Term::Literal(model::Literal::Nat(a)) = self.get_term(a)? else {
                continue;
            };

            let table::Term::Literal(model::Literal::Nat(b)) = self.get_term(b)? else {
                continue;
            };

            let a = order_keys.get(a).ok_or(OrderHintError::UnknownKey(*a))?;
            let b = order_keys.get(b).ok_or(OrderHintError::UnknownKey(*b))?;

            // NOTE: The lookups here are expected to succeed since we only
            // process the order metadata after we have imported the nodes.
            let a_node = self.nodes[a];
            let b_node = self.nodes[b];

            let a_port = self
                .hugr
                .get_optype(a_node)
                .other_output_port()
                .ok_or(OrderHintError::NoOrderPort(*a))?;

            let b_port = self
                .hugr
                .get_optype(b_node)
                .other_input_port()
                .ok_or(OrderHintError::NoOrderPort(*b))?;

            self.hugr.connect(a_node, a_port, b_node, b_port);
        }

        Ok(())
    }

    fn import_adt_and_rest(
        &mut self,
        list: table::TermId,
    ) -> Result<(Vec<TypeRow>, TypeRow), ImportError> {
        let items = self.import_closed_list(list)?;

        let Some((first, rest)) = items.split_first() else {
            return Err(error_invalid!("expected list to have at least one element"));
        };

        let sum_rows: Vec<_> = {
            let [variants] = self.expect_symbol(*first, model::CORE_ADT)?;
            self.import_type_rows(variants)?
        };

        let rest = rest
            .iter()
            .map(|term| self.import_type(*term))
            .collect::<Result<Vec<_>, _>>()?
            .into();

        Ok((sum_rows, rest))
    }

    fn import_tail_loop(
        &mut self,
        node_id: table::NodeId,
        parent: Node,
    ) -> Result<Node, ImportError> {
        let node_data = self.get_node(node_id)?;
        debug_assert_eq!(node_data.operation, table::Operation::TailLoop);

        let [region] = node_data.regions else {
            return Err(error_invalid!(
                "loop node {} expects a single region",
                node_id
            ));
        };

        let region_data = self.get_region(*region)?;

        let (just_inputs, just_outputs, rest) = (|| {
            let [_, region_outputs] = self.get_func_type(
                region_data
                    .signature
                    .ok_or_else(|| error_uninferred!("region signature"))?,
            )?;
            let (sum_rows, rest) = self.import_adt_and_rest(region_outputs)?;

            if sum_rows.len() != 2 {
                return Err(error_invalid!(
                    "loop nodes expect their first target to be an ADT with two variants"
                ));
            }

            let mut sum_rows = sum_rows.into_iter();
            let just_inputs = sum_rows.next().unwrap();
            let just_outputs = sum_rows.next().unwrap();

            Ok((just_inputs, just_outputs, rest))
        })()
        .map_err(|err| error_context!(err, "region signature"))?;

        let optype = OpType::TailLoop(TailLoop {
            just_inputs,
            just_outputs,
            rest,
        });

        let node = self.make_node(node_id, optype, parent)?;

        self.import_dfg_region(*region, node)?;
        Ok(node)
    }

    fn import_conditional(
        &mut self,
        node_id: table::NodeId,
        parent: Node,
    ) -> Result<Node, ImportError> {
        let node_data = self.get_node(node_id)?;
        debug_assert_eq!(node_data.operation, table::Operation::Conditional);

        let (sum_rows, other_inputs, outputs) = (|| {
            let [inputs, outputs] = self.get_func_type(
                node_data
                    .signature
                    .ok_or_else(|| error_uninferred!("node signature"))?,
            )?;
            let (sum_rows, other_inputs) = self.import_adt_and_rest(inputs)?;
            let outputs = self.import_type_row(outputs)?;

            Ok((sum_rows, other_inputs, outputs))
        })()
        .map_err(|err| error_context!(err, "node signature"))?;

        let optype = OpType::Conditional(Conditional {
            sum_rows,
            other_inputs,
            outputs,
        });

        let node = self.make_node(node_id, optype, parent)?;

        for region in node_data.regions {
            let region_data = self.get_region(*region)?;
            let signature = self.import_func_type(
                region_data
                    .signature
                    .ok_or_else(|| error_uninferred!("region signature"))?,
            )?;

            let case_node = self
                .hugr
                .add_node_with_parent(node, OpType::Case(Case { signature }));

            self.import_dfg_region(*region, case_node)?;
        }

        Ok(node)
    }

    fn import_cfg_region(
        &mut self,
        region: table::RegionId,
        node: Node,
    ) -> Result<(), ImportError> {
        let region_data = self.get_region(region)?;

        if region_data.kind != model::RegionKind::ControlFlow {
            return Err(error_invalid!("expected cfg region"));
        }

        let prev_region = self.region_scope;
        if region_data.scope.is_some() {
            self.region_scope = region;
        }

        let region_target_types = (|| {
            let [_, region_targets] = self.get_ctrl_type(
                region_data
                    .signature
                    .ok_or_else(|| error_uninferred!("region signature"))?,
            )?;

            self.import_closed_list(region_targets)
        })()
        .map_err(|err| error_context!(err, "signature of cfg region with id {}", region))?;

        // Identify the entry node of the control flow region by looking for
        // a block whose input is linked to the sole source port of the CFG region.
        let entry_node = 'find_entry: {
            let [entry_link] = region_data.sources else {
                return Err(error_invalid!("cfg region expects a single source"));
            };

            for child in region_data.children {
                let child_data = self.get_node(*child)?;
                let is_entry = child_data.inputs.iter().any(|link| link == entry_link);

                if is_entry {
                    break 'find_entry *child;
                }
            }

            // TODO: We should allow for the case in which control flows
            // directly from the source to the target of the region. This is
            // currently not allowed in hugr core directly, but may be simulated
            // by constructing an empty entry block.
            return Err(error_invalid!("cfg region without entry node"));
        };

        // The entry node in core control flow regions is identified by being
        // the first child node of the CFG node. We therefore import the entry node first.
        self.import_node(entry_node, node)?;

        // Create the exit node for the control flow region. This always needs
        // to be second in the node list.
        {
            let cfg_outputs = {
                let [target_types] = region_target_types.as_slice() else {
                    return Err(error_invalid!("cfg region expects a single target"));
                };

                self.import_type_row(*target_types)?
            };

            let exit = self
                .hugr
                .add_node_with_parent(node, OpType::ExitBlock(ExitBlock { cfg_outputs }));
            self.record_links(exit, Direction::Incoming, region_data.targets);
        }

        // Finally we import all other nodes.
        for child in region_data.children {
            if *child != entry_node {
                self.import_node(*child, node)?;
            }
        }

        for meta_item in region_data.meta {
            self.import_node_metadata(node, *meta_item)
                .map_err(|err| error_context!(err, "node metadata"))?;
        }

        self.region_scope = prev_region;

        Ok(())
    }

    fn import_node_block(
        &mut self,
        node_id: table::NodeId,
        parent: Node,
    ) -> Result<Node, ImportError> {
        let node_data = self.get_node(node_id)?;
        debug_assert_eq!(node_data.operation, table::Operation::Block);

        let [region] = node_data.regions else {
            return Err(error_invalid!("basic block expects a single region"));
        };
        let region_data = self.get_region(*region)?;
        let [inputs, outputs] = self.get_func_type(
            region_data
                .signature
                .ok_or_else(|| error_uninferred!("region signature"))?,
        )?;
        let inputs = self.import_type_row(inputs)?;
        let (sum_rows, other_outputs) = self.import_adt_and_rest(outputs)?;

        let optype = OpType::DataflowBlock(DataflowBlock {
            inputs,
            other_outputs,
            sum_rows,
        });
        let node = self.make_node(node_id, optype, parent)?;

        self.import_dfg_region(*region, node).map_err(|err| {
            error_context!(err, "block body defined by region with id {}", *region)
        })?;
        Ok(node)
    }

    fn import_node_define_func(
        &mut self,
        node_id: table::NodeId,
        symbol: &'a table::Symbol<'a>,
        node_data: &'a table::Node<'a>,
        parent: Node,
    ) -> Result<Node, ImportError> {
        self.import_poly_func_type(node_id, *symbol, |ctx, signature| {
            let optype = OpType::FuncDefn(FuncDefn::new(symbol.name, signature));

            let node = ctx.make_node(node_id, optype, parent)?;

            let [region] = node_data.regions else {
                return Err(error_invalid!(
                    "function definition nodes expect a single region"
                ));
            };

            ctx.import_dfg_region(*region, node).map_err(|err| {
                error_context!(err, "function body defined by region with id {}", *region)
            })?;

            Ok(node)
        })
    }

    fn import_node_declare_func(
        &mut self,
        node_id: table::NodeId,
        symbol: &'a table::Symbol<'a>,
        parent: Node,
    ) -> Result<Node, ImportError> {
        self.import_poly_func_type(node_id, *symbol, |ctx, signature| {
            let optype = OpType::FuncDecl(FuncDecl::new(symbol.name, signature));
            let node = ctx.make_node(node_id, optype, parent)?;
            Ok(node)
        })
    }

    fn import_node_custom(
        &mut self,
        node_id: table::NodeId,
        operation: table::TermId,
        node_data: &'a table::Node<'a>,
        parent: Node,
    ) -> Result<Node, ImportError> {
        if let Some([inputs, outputs]) = self.match_symbol(operation, model::CORE_CALL_INDIRECT)? {
            let inputs = self.import_type_row(inputs)?;
            let outputs = self.import_type_row(outputs)?;
            let signature = Signature::new(inputs, outputs);
            let optype = OpType::CallIndirect(CallIndirect { signature });
            let node = self.make_node(node_id, optype, parent)?;
            return Ok(node);
        }

        if let Some([_, _, func]) = self.match_symbol(operation, model::CORE_CALL)? {
            let table::Term::Apply(symbol, args) = self.get_term(func)? else {
                return Err(error_invalid!(
                    "expected a symbol application to be passed to `{}`",
                    model::CORE_CALL
                ));
            };

            let func_sig = self.get_func_signature(*symbol)?;

            let type_args = args
                .iter()
                .map(|term| self.import_term(*term))
                .collect::<Result<Vec<TypeArg>, _>>()?;

            self.static_edges.push((*symbol, node_id));
            let optype = OpType::Call(
                Call::try_new(func_sig, type_args).map_err(ImportErrorInner::Signature)?,
            );

            let node = self.make_node(node_id, optype, parent)?;
            return Ok(node);
        }

        if let Some([_, value]) = self.match_symbol(operation, model::CORE_LOAD_CONST)? {
            // If the constant refers directly to a function, import this as the `LoadFunc` operation.
            if let table::Term::Apply(symbol, args) = self.get_term(value)? {
                let func_node_data = self.get_node(*symbol)?;

                if let table::Operation::DefineFunc(_) | table::Operation::DeclareFunc(_) =
                    func_node_data.operation
                {
                    let func_sig = self.get_func_signature(*symbol)?;
                    let type_args = args
                        .iter()
                        .map(|term| self.import_term(*term))
                        .collect::<Result<Vec<TypeArg>, _>>()?;

                    self.static_edges.push((*symbol, node_id));

                    let optype = OpType::LoadFunction(
                        LoadFunction::try_new(func_sig, type_args)
                            .map_err(ImportErrorInner::Signature)?,
                    );

                    let node = self.make_node(node_id, optype, parent)?;
                    return Ok(node);
                }
            }

            // Otherwise use const nodes
            let signature = node_data
                .signature
                .ok_or_else(|| error_uninferred!("node signature"))?;
            let [_, outputs] = self.get_func_type(signature)?;
            let outputs = self.import_closed_list(outputs)?;
            let output = outputs.first().ok_or_else(|| {
                error_invalid!("`{}` expects a single output", model::CORE_LOAD_CONST)
            })?;
            let datatype = self.import_type(*output)?;

            let imported_value = self.import_value(value, *output)?;

            let load_const_node = self.make_node(
                node_id,
                OpType::LoadConstant(LoadConstant {
                    datatype: datatype.clone(),
                }),
                parent,
            )?;

            let const_node = self
                .hugr
                .add_node_with_parent(parent, OpType::Const(Const::new(imported_value)));

            self.hugr.connect(const_node, 0, load_const_node, 0);

            return Ok(load_const_node);
        }

        if let Some([_, _, tag]) = self.match_symbol(operation, model::CORE_MAKE_ADT)? {
            let table::Term::Literal(model::Literal::Nat(tag)) = self.get_term(tag)? else {
                return Err(error_invalid!(
                    "`{}` expects a nat literal tag",
                    model::CORE_MAKE_ADT
                ));
            };

            let signature = node_data
                .signature
                .ok_or_else(|| error_uninferred!("node signature"))?;
            let [_, outputs] = self.get_func_type(signature)?;
            let (variants, _) = self.import_adt_and_rest(outputs)?;
            let node = self.make_node(
                node_id,
                OpType::Tag(Tag {
                    variants,
                    tag: *tag as usize,
                }),
                parent,
            )?;
            return Ok(node);
        }

        let table::Term::Apply(node, params) = self.get_term(operation)? else {
            return Err(error_invalid!(
                "custom operations expect a symbol application referencing an operation"
            ));
        };
        let name = self.get_symbol_name(*node)?;
        let args = params
            .iter()
            .map(|param| self.import_term(*param))
            .collect::<Result<Vec<_>, _>>()?;
        let (extension, name) = self.import_custom_name(name)?;
        let signature = self.get_node_signature(node_id)?;

        // TODO: Currently we do not have the description or any other metadata for
        // the custom op. This will improve with declarative extensions being able
        // to declare operations as a node, in which case the description will be attached
        // to that node as metadata.

        let optype = OpType::OpaqueOp(OpaqueOp::new(extension, name, args, signature));
        self.make_node(node_id, optype, parent)
    }

    fn import_node_define_alias(
        &mut self,
        node_id: table::NodeId,
        symbol: &'a table::Symbol<'a>,
        value: table::TermId,
        parent: Node,
    ) -> Result<Node, ImportError> {
        if !symbol.params.is_empty() {
            return Err(error_unsupported!(
                "parameters or constraints in alias definition"
            ));
        }

        let optype = OpType::AliasDefn(AliasDefn {
            name: symbol.name.to_smolstr(),
            definition: self.import_type(value)?,
        });

        let node = self.make_node(node_id, optype, parent)?;
        Ok(node)
    }

    fn import_node_declare_alias(
        &mut self,
        node_id: table::NodeId,
        symbol: &'a table::Symbol<'a>,
        parent: Node,
    ) -> Result<Node, ImportError> {
        if !symbol.params.is_empty() {
            return Err(error_unsupported!(
                "parameters or constraints in alias declaration"
            ));
        }

        let optype = OpType::AliasDecl(AliasDecl {
            name: symbol.name.to_smolstr(),
            bound: TypeBound::Copyable,
        });

        let node = self.make_node(node_id, optype, parent)?;
        Ok(node)
    }

    fn import_poly_func_type<RV: MaybeRV, T>(
        &mut self,
        node: table::NodeId,
        symbol: table::Symbol<'a>,
        in_scope: impl FnOnce(&mut Self, PolyFuncTypeBase<RV>) -> Result<T, ImportError>,
    ) -> Result<T, ImportError> {
        (|| {
            let mut imported_params = Vec::with_capacity(symbol.params.len());

            for (index, param) in symbol.params.iter().enumerate() {
                self.local_vars
                    .insert(table::VarId(node, index as _), LocalVar::new(param.r#type));
            }

            for constraint in symbol.constraints {
                if let Some([term]) = self.match_symbol(*constraint, model::CORE_NON_LINEAR)? {
                    let table::Term::Var(var) = self.get_term(term)? else {
                        return Err(error_unsupported!(
                            "constraint on term that is not a variable"
                        ));
                    };

                    self.local_vars
                        .get_mut(var)
                        .ok_or_else(|| error_invalid!("unknown variable {}", var))?
                        .bound = TypeBound::Copyable;
                } else {
                    return Err(error_unsupported!("constraint other than copy or discard"));
                }
            }

            for (index, param) in symbol.params.iter().enumerate() {
                // NOTE: `PolyFuncType` only has explicit type parameters at present.
                let bound = self.local_vars[&table::VarId(node, index as _)].bound;
                imported_params.push(
                    self.import_term_with_bound(param.r#type, bound)
                        .map_err(|err| error_context!(err, "type of parameter `{}`", param.name))?,
                );
            }

            let body = self.import_func_type::<RV>(symbol.signature)?;
            in_scope(self, PolyFuncTypeBase::new(imported_params, body))
        })()
        .map_err(|err| error_context!(err, "symbol `{}` defined by node {}", symbol.name, node))
    }

    /// Import a [`Term`] from a term that represents a static type or value.
    fn import_term(&mut self, term_id: table::TermId) -> Result<Term, ImportError> {
        self.import_term_with_bound(term_id, TypeBound::Any)
    }

    fn import_term_with_bound(
        &mut self,
        term_id: table::TermId,
        bound: TypeBound,
    ) -> Result<Term, ImportError> {
        (|| {
            if let Some([]) = self.match_symbol(term_id, model::CORE_STR_TYPE)? {
                return Ok(Term::StringType);
            }

            if let Some([]) = self.match_symbol(term_id, model::CORE_NAT_TYPE)? {
                return Ok(Term::max_nat_type());
            }

            if let Some([]) = self.match_symbol(term_id, model::CORE_BYTES_TYPE)? {
                return Ok(Term::BytesType);
            }

            if let Some([]) = self.match_symbol(term_id, model::CORE_FLOAT_TYPE)? {
                return Ok(Term::FloatType);
            }

            if let Some([]) = self.match_symbol(term_id, model::CORE_TYPE)? {
                return Ok(TypeParam::RuntimeType(bound));
            }

            if let Some([]) = self.match_symbol(term_id, model::CORE_CONSTRAINT)? {
                return Err(error_unsupported!("`{}`", model::CORE_CONSTRAINT));
            }

            if let Some([]) = self.match_symbol(term_id, model::CORE_STATIC)? {
                return Ok(Term::StaticType);
            }

            if let Some([]) = self.match_symbol(term_id, model::CORE_CONST)? {
                return Err(error_unsupported!("`{}`", model::CORE_CONST));
            }

            if let Some([item_type]) = self.match_symbol(term_id, model::CORE_LIST_TYPE)? {
                // At present `hugr-model` has no way to express that the item
                // type of a list must be copyable. Therefore we import it as `Any`.
                let item_type = self
                    .import_term(item_type)
                    .map_err(|err| error_context!(err, "item type of list type"))?;
                return Ok(TypeParam::new_list_type(item_type));
            }

            if let Some([item_types]) = self.match_symbol(term_id, model::CORE_TUPLE_TYPE)? {
                // At present `hugr-model` has no way to express that the item
                // types of a tuple must be copyable. Therefore we import it as `Any`.
                let item_types = self
                    .import_term(item_types)
                    .map_err(|err| error_context!(err, "item types of tuple type"))?;
                return Ok(TypeParam::new_tuple_type(item_types));
            }

            match self.get_term(term_id)? {
                table::Term::Wildcard => Err(error_uninferred!("wildcard")),

                table::Term::Var(var) => {
                    let var_info = self
                        .local_vars
                        .get(var)
                        .ok_or_else(|| error_invalid!("unknown variable {}", var))?;
                    let decl = self.import_term_with_bound(var_info.r#type, var_info.bound)?;
                    Ok(Term::new_var_use(var.1 as _, decl))
                }

                table::Term::List(parts) => {
                    // PERFORMANCE: Can we do this without the additional allocation?
                    let parts: Vec<_> = parts
                        .iter()
                        .map(|part| self.import_seq_part(part))
                        .collect::<Result<_, _>>()
                        .map_err(|err| error_context!(err, "list parts"))?;
                    Ok(TypeArg::new_list_from_parts(parts))
                }

                table::Term::Tuple(parts) => {
                    // PERFORMANCE: Can we do this without the additional allocation?
                    let parts: Vec<_> = parts
                        .iter()
                        .map(|part| self.import_seq_part(part))
                        .try_collect()
                        .map_err(|err| error_context!(err, "tuple parts"))?;
                    Ok(TypeArg::new_tuple_from_parts(parts))
                }

                table::Term::Literal(model::Literal::Str(value)) => {
                    Ok(Term::String(value.to_string()))
                }

                table::Term::Literal(model::Literal::Nat(value)) => Ok(Term::BoundedNat(*value)),

                table::Term::Literal(model::Literal::Bytes(value)) => {
                    Ok(Term::Bytes(value.clone()))
                }
                table::Term::Literal(model::Literal::Float(value)) => Ok(Term::Float(*value)),
                table::Term::Func { .. } => Err(error_unsupported!("function constant")),

                table::Term::Apply { .. } => {
                    let ty: Type = self.import_type(term_id)?;
                    Ok(ty.into())
                }
            }
        })()
        .map_err(|err| error_context!(err, "term {}", term_id))
    }

    fn import_seq_part(
        &mut self,
        seq_part: &'a table::SeqPart,
    ) -> Result<SeqPart<TypeArg>, ImportError> {
        Ok(match seq_part {
            table::SeqPart::Item(term_id) => SeqPart::Item(self.import_term(*term_id)?),
            table::SeqPart::Splice(term_id) => SeqPart::Splice(self.import_term(*term_id)?),
        })
    }

    /// Import a `Type` from a term that represents a runtime type.
    fn import_type<RV: MaybeRV>(
        &mut self,
        term_id: table::TermId,
    ) -> Result<TypeBase<RV>, ImportError> {
        (|| {
            if let Some([_, _]) = self.match_symbol(term_id, model::CORE_FN)? {
                let func_type = self.import_func_type::<RowVariable>(term_id)?;
                return Ok(TypeBase::new_function(func_type));
            }

            if let Some([variants]) = self.match_symbol(term_id, model::CORE_ADT)? {
                let variants = (|| {
                    self.import_closed_list(variants)?
                        .iter()
                        .map(|variant| self.import_type_row::<RowVariable>(*variant))
                        .collect::<Result<Vec<_>, _>>()
                })()
                .map_err(|err| error_context!(err, "adt variants"))?;

                return Ok(TypeBase::new_sum(variants));
            }

            match self.get_term(term_id)? {
                table::Term::Wildcard => Err(error_uninferred!("wildcard")),

                table::Term::Apply(symbol, args) => {
                    let name = self.get_symbol_name(*symbol)?;

                    let args = args
                        .iter()
                        .map(|arg| self.import_term(*arg))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| {
                            error_context!(err, "type argument of custom type `{}`", name)
                        })?;

                    let (extension, id) = self.import_custom_name(name)?;

                    let extension_ref =
                        self.extensions
                            .get(&extension)
                            .ok_or_else(|| ExtensionError::Missing {
                                missing_ext: extension.clone(),
                                available: self.extensions.ids().cloned().collect(),
                            })?;

                    let ext_type =
                        extension_ref
                            .get_type(&id)
                            .ok_or_else(|| ExtensionError::MissingType {
                                ext: extension.clone(),
                                name: id.clone(),
                            })?;

                    let bound = ext_type.bound(&args);

                    Ok(TypeBase::new_extension(CustomType::new(
                        id,
                        args,
                        extension,
                        bound,
                        &Arc::downgrade(extension_ref),
                    )))
                }

                table::Term::Var(var @ table::VarId(_, index)) => {
                    let local_var = self
                        .local_vars
                        .get(var)
                        .ok_or(error_invalid!("unknown var {}", var))?;
                    Ok(TypeBase::new_var_use(*index as _, local_var.bound))
                }

                // The following terms are not runtime types, but the core `Type` only contains runtime types.
                // We therefore report a type error here.
                table::Term::List { .. }
                | table::Term::Tuple { .. }
                | table::Term::Literal(_)
                | table::Term::Func { .. } => Err(error_invalid!("expected a runtime type")),
            }
        })()
        .map_err(|err| error_context!(err, "term {} as `Type`", term_id))
    }

    fn get_func_type(&mut self, term_id: table::TermId) -> Result<[table::TermId; 2], ImportError> {
        self.match_symbol(term_id, model::CORE_FN)?
            .ok_or(error_invalid!("expected a function type"))
    }

    fn get_ctrl_type(&mut self, term_id: table::TermId) -> Result<[table::TermId; 2], ImportError> {
        self.match_symbol(term_id, model::CORE_CTRL)?
            .ok_or(error_invalid!("expected a control type"))
    }

    fn import_func_type<RV: MaybeRV>(
        &mut self,
        term_id: table::TermId,
    ) -> Result<FuncTypeBase<RV>, ImportError> {
        (|| {
            let [inputs, outputs] = self.get_func_type(term_id)?;
            let inputs = self
                .import_type_row(inputs)
                .map_err(|err| error_context!(err, "function inputs"))?;
            let outputs = self
                .import_type_row(outputs)
                .map_err(|err| error_context!(err, "function outputs"))?;
            Ok(FuncTypeBase::new(inputs, outputs))
        })()
        .map_err(|err| error_context!(err, "function type"))
    }

    fn import_closed_list(
        &mut self,
        term_id: table::TermId,
    ) -> Result<Vec<table::TermId>, ImportError> {
        fn import_into(
            ctx: &mut Context,
            term_id: table::TermId,
            types: &mut Vec<table::TermId>,
        ) -> Result<(), ImportError> {
            match ctx.get_term(term_id)? {
                table::Term::List(parts) => {
                    types.reserve(parts.len());

                    for part in *parts {
                        match part {
                            table::SeqPart::Item(term_id) => {
                                types.push(*term_id);
                            }
                            table::SeqPart::Splice(term_id) => {
                                import_into(ctx, *term_id, types)?;
                            }
                        }
                    }
                }
                _ => return Err(error_invalid!("expected a closed list")),
            }

            Ok(())
        }

        let mut types = Vec::new();
        import_into(self, term_id, &mut types)?;
        Ok(types)
    }

    fn import_closed_tuple(
        &mut self,
        term_id: table::TermId,
    ) -> Result<Vec<table::TermId>, ImportError> {
        fn import_into(
            ctx: &mut Context,
            term_id: table::TermId,
            types: &mut Vec<table::TermId>,
        ) -> Result<(), ImportError> {
            match ctx.get_term(term_id)? {
                table::Term::Tuple(parts) => {
                    types.reserve(parts.len());

                    for part in *parts {
                        match part {
                            table::SeqPart::Item(term_id) => {
                                types.push(*term_id);
                            }
                            table::SeqPart::Splice(term_id) => {
                                import_into(ctx, *term_id, types)?;
                            }
                        }
                    }
                }
                _ => return Err(error_invalid!("expected a closed tuple")),
            }

            Ok(())
        }

        let mut types = Vec::new();
        import_into(self, term_id, &mut types)?;
        Ok(types)
    }

    fn import_type_rows<RV: MaybeRV>(
        &mut self,
        term_id: table::TermId,
    ) -> Result<Vec<TypeRowBase<RV>>, ImportError> {
        self.import_closed_list(term_id)?
            .into_iter()
            .map(|term_id| self.import_type_row::<RV>(term_id))
            .collect()
    }

    fn import_type_row<RV: MaybeRV>(
        &mut self,
        term_id: table::TermId,
    ) -> Result<TypeRowBase<RV>, ImportError> {
        fn import_into<RV: MaybeRV>(
            ctx: &mut Context,
            term_id: table::TermId,
            types: &mut Vec<TypeBase<RV>>,
        ) -> Result<(), ImportError> {
            match ctx.get_term(term_id)? {
                table::Term::List(parts) => {
                    types.reserve(parts.len());

                    for item in *parts {
                        match item {
                            table::SeqPart::Item(term_id) => {
                                types.push(ctx.import_type::<RV>(*term_id)?);
                            }
                            table::SeqPart::Splice(term_id) => {
                                import_into(ctx, *term_id, types)?;
                            }
                        }
                    }
                }
                table::Term::Var(table::VarId(_, index)) => {
                    let var = RV::try_from_rv(RowVariable(*index as _, TypeBound::Any))
                        .map_err(|_| error_invalid!("expected a closed list"))?;
                    types.push(TypeBase::new(TypeEnum::RowVar(var)));
                }
                _ => return Err(error_invalid!("expected a list")),
            }

            Ok(())
        }

        let mut types = Vec::new();
        import_into(self, term_id, &mut types)?;
        Ok(types.into())
    }

    fn import_custom_name(
        &mut self,
        symbol: &'a str,
    ) -> Result<(ExtensionId, SmolStr), ImportError> {
        use std::collections::hash_map::Entry;
        match self.custom_name_cache.entry(symbol) {
            Entry::Occupied(occupied_entry) => Ok(occupied_entry.get().clone()),
            Entry::Vacant(vacant_entry) => {
                let qualified_name = ExtensionId::new(symbol)
                    .map_err(|_| error_invalid!("`{}` is not a valid symbol name", symbol))?;

                let (extension, id) = qualified_name
                    .split_last()
                    .ok_or_else(|| error_invalid!("`{}` is not a valid symbol name", symbol))?;

                vacant_entry.insert((extension.clone(), id.clone()));
                Ok((extension, id))
            }
        }
    }

    fn import_value(
        &mut self,
        term_id: table::TermId,
        type_id: table::TermId,
    ) -> Result<Value, ImportError> {
        let term_data = self.get_term(term_id)?;

        // NOTE: We have special cased arrays, integers, and floats for now.
        // TODO: Allow arbitrary extension values to be imported from terms.

        if let Some([runtime_type, json]) = self.match_symbol(term_id, model::COMPAT_CONST_JSON)? {
            let table::Term::Literal(model::Literal::Str(json)) = self.get_term(json)? else {
                return Err(error_invalid!(
                    "`{}` expects a string literal",
                    model::COMPAT_CONST_JSON
                ));
            };

            // We attempt to deserialize as the custom const directly.
            // This might fail due to the custom const struct not being included when
            // this code was compiled; in that case, we fall back to the serialized form.
            let value: Option<Box<dyn CustomConst>> = serde_json::from_str(json).ok();

            if let Some(value) = value {
                let opaque_value = OpaqueValue::from(value);
                return Ok(Value::Extension { e: opaque_value });
            } else {
                let runtime_type = self.import_type(runtime_type)?;
                let value: serde_json::Value = serde_json::from_str(json).map_err(|_| {
                    error_invalid!(
                        "unable to parse JSON string for `{}`",
                        model::COMPAT_CONST_JSON
                    )
                })?;
                let custom_const = CustomSerialized::new(runtime_type, value);
                let opaque_value = OpaqueValue::new(custom_const);
                return Ok(Value::Extension { e: opaque_value });
            }
        }

        if let Some([_, element_type_term, contents]) =
            self.match_symbol(term_id, ArrayValue::CTR_NAME)?
        {
            let element_type = self.import_type(element_type_term)?;
            let contents = self.import_closed_list(contents)?;
            let contents = contents
                .iter()
                .map(|item| self.import_value(*item, element_type_term))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(ArrayValue::new(element_type, contents).into());
        }

        if let Some([bitwidth, value]) = self.match_symbol(term_id, ConstInt::CTR_NAME)? {
            let bitwidth = {
                let table::Term::Literal(model::Literal::Nat(bitwidth)) =
                    self.get_term(bitwidth)?
                else {
                    return Err(error_invalid!(
                        "`{}` expects a nat literal in its `bitwidth` argument",
                        ConstInt::CTR_NAME
                    ));
                };
                if *bitwidth > 6 {
                    return Err(error_invalid!(
                        "`{}` expects the bitwidth to be at most 6, got {}",
                        ConstInt::CTR_NAME,
                        bitwidth
                    ));
                }
                *bitwidth as u8
            };

            let value = {
                let table::Term::Literal(model::Literal::Nat(value)) = self.get_term(value)? else {
                    return Err(error_invalid!(
                        "`{}` expects a nat literal value",
                        ConstInt::CTR_NAME
                    ));
                };
                *value
            };

            return Ok(ConstInt::new_u(bitwidth, value)
                .map_err(|_| error_invalid!("failed to create int constant"))?
                .into());
        }

        if let Some([value]) = self.match_symbol(term_id, ConstF64::CTR_NAME)? {
            let table::Term::Literal(model::Literal::Float(value)) = self.get_term(value)? else {
                return Err(error_invalid!(
                    "`{}` expects a float literal value",
                    ConstF64::CTR_NAME
                ));
            };

            return Ok(ConstF64::new(value.into_inner()).into());
        }

        if let Some([_, _, tag, values]) = self.match_symbol(term_id, model::CORE_CONST_ADT)? {
            let [variants] = self.expect_symbol(type_id, model::CORE_ADT)?;
            let values = self.import_closed_tuple(values)?;
            let variants = self.import_closed_list(variants)?;

            let table::Term::Literal(model::Literal::Nat(tag)) = self.get_term(tag)? else {
                return Err(error_invalid!(
                    "`{}` expects a nat literal tag",
                    model::CORE_ADT
                ));
            };

            let variant = variants.get(*tag as usize).ok_or(error_invalid!(
                "the tag of a `{}` must be a valid index into the list of variants",
                model::CORE_CONST_ADT
            ))?;

            let variant = self.import_closed_list(*variant)?;

            let items = values
                .iter()
                .zip(variant.iter())
                .map(|(value, typ)| self.import_value(*value, *typ))
                .collect::<Result<Vec<_>, _>>()?;

            let typ = {
                // TODO: Import as a `SumType` directly and avoid the copy.
                let typ: Type = self.import_type(type_id)?;
                match typ.as_type_enum() {
                    TypeEnum::Sum(sum) => sum.clone(),
                    _ => unreachable!(),
                }
            };

            return Ok(Value::sum(*tag as _, items, typ).unwrap());
        }

        match term_data {
            table::Term::Wildcard => Err(error_uninferred!("wildcard")),
            table::Term::Var(_) => Err(error_unsupported!("constant value containing a variable")),

            table::Term::Apply(symbol, _) => {
                let symbol_name = self.get_symbol_name(*symbol)?;
                Err(error_unsupported!(
                    "unknown custom constant value `{}`",
                    symbol_name
                ))
                // TODO: This should ultimately include the following cases:
                // - function definitions
                // - custom constructors for values
            }

            table::Term::List { .. } | table::Term::Tuple(_) | table::Term::Literal(_) => {
                Err(error_invalid!("expected constant"))
            }

            table::Term::Func { .. } => Err(error_unsupported!("constant function value")),
        }
    }

    fn match_symbol<const N: usize>(
        &self,
        term_id: table::TermId,
        name: &str,
    ) -> Result<Option<[table::TermId; N]>, ImportError> {
        let term = self.get_term(term_id)?;

        // TODO: Follow alias chains?

        let table::Term::Apply(symbol, args) = term else {
            return Ok(None);
        };

        if name != self.get_symbol_name(*symbol)? {
            return Ok(None);
        }

        // We allow the match even if the symbol is applied to fewer arguments
        // than parameters. In that case, the arguments are padded with wildcards
        // at the beginning.
        if args.len() > N {
            return Ok(None);
        }

        let result = std::array::from_fn(|i| {
            (i + args.len())
                .checked_sub(N)
                .map(|i| args[i])
                .unwrap_or_default()
        });

        Ok(Some(result))
    }

    fn expect_symbol<const N: usize>(
        &self,
        term_id: table::TermId,
        name: &str,
    ) -> Result<[table::TermId; N], ImportError> {
        self.match_symbol(term_id, name)?.ok_or(error_invalid!(
            "expected symbol `{}` with arity {}",
            name,
            N
        ))
    }
}

/// Information about a local variable.
#[derive(Debug, Clone, Copy)]
struct LocalVar {
    /// The type of the variable.
    r#type: table::TermId,
    /// The type bound of the variable.
    bound: TypeBound,
}

impl LocalVar {
    pub fn new(r#type: table::TermId) -> Self {
        Self {
            r#type,
            bound: TypeBound::Any,
        }
    }
}
