//! Importing HUGR graphs from their `hugr-model` representation.
//!
//! **Warning**: This module is still under development and is expected to change.
//! It is included in the library to allow for early experimentation, and for
//! the core and model to converge incrementally.
use std::sync::Arc;

use crate::{
    Direction, Hugr, HugrView, Node, Port,
    extension::{ExtensionId, ExtensionRegistry, SignatureError},
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
        Type, TypeArg, TypeBase, TypeBound, TypeEnum, TypeName, TypeRow, type_param::TypeParam,
        type_row::TypeRowBase,
    },
};
use fxhash::FxHashMap;
use hugr_model::v0 as model;
use hugr_model::v0::table;
use itertools::Either;
use smol_str::{SmolStr, ToSmolStr};
use thiserror::Error;

/// Error during import.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum ImportError {
    /// The model contains a feature that is not supported by the importer yet.
    /// Errors of this kind are expected to be removed as the model format and
    /// the core HUGR representation converge.
    #[error("currently unsupported: {0}")]
    Unsupported(String),
    /// The model contains implicit information that has not yet been inferred.
    /// This includes wildcards and application of functions with implicit parameters.
    #[error("uninferred implicit: {0}")]
    Uninferred(String),
    /// A signature mismatch was detected during import.
    #[error("signature error: {0}")]
    Signature(#[from] SignatureError),
    /// A required extension is missing.
    #[error("Importing the hugr requires extension {missing_ext}, which was not found in the registry. The available extensions are: [{}]",
            available.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(", "))]
    Extension {
        /// The missing extension.
        missing_ext: ExtensionId,
        /// The available extensions in the registry.
        available: Vec<ExtensionId>,
    },
    /// An extension type is missing.
    #[error(
        "Importing the hugr requires extension {ext} to have a type named {name}, but it was not found."
    )]
    ExtensionType {
        /// The extension that is missing the type.
        ext: ExtensionId,
        /// The name of the missing type.
        name: TypeName,
    },
    /// The model is not well-formed.
    #[error("validate error: {0}")]
    Model(#[from] table::ModelError),
    /// Incorrect order hints.
    #[error("incorrect order hint: {0}")]
    OrderHint(#[from] OrderHintError),
}

/// Import error caused by incorrect order hints.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum OrderHintError {
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

/// Helper macro to create an `ImportError::Unsupported` error with a formatted message.
macro_rules! error_unsupported {
    ($($e:expr),*) => { ImportError::Unsupported(format!($($e),*)) }
}

/// Helper macro to create an `ImportError::Uninferred` error with a formatted message.
macro_rules! error_uninferred {
    ($($e:expr),*) => { ImportError::Uninferred(format!($($e),*)) }
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
            .ok_or_else(|| table::ModelError::NodeNotFound(node_id).into())
    }

    /// Get the term with the given `TermId`, or return an error if it does not exist.
    #[inline]
    fn get_term(&self, term_id: table::TermId) -> Result<&'a table::Term<'a>, ImportError> {
        self.module
            .get_term(term_id)
            .ok_or_else(|| table::ModelError::TermNotFound(term_id).into())
    }

    /// Get the region with the given `RegionId`, or return an error if it does not exist.
    #[inline]
    fn get_region(&self, region_id: table::RegionId) -> Result<&'a table::Region<'a>, ImportError> {
        self.module
            .get_region(region_id)
            .ok_or_else(|| table::ModelError::RegionNotFound(region_id).into())
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
            self.import_node_metadata(node, *meta_item)?;
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
                return Err(table::ModelError::TypeError(meta_item).into());
            };

            let table::Term::Literal(model::Literal::Str(json_str)) = self.get_term(json_arg)?
            else {
                return Err(table::ModelError::TypeError(meta_item).into());
            };

            let json_value: NodeMetadata = serde_json::from_str(json_str)
                .map_err(|_| table::ModelError::TypeError(meta_item))?;

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
            .ok_or(table::ModelError::InvalidSymbol(node_id))?;
        Ok(name)
    }

    fn get_func_signature(
        &mut self,
        func_node: table::NodeId,
    ) -> Result<PolyFuncType, ImportError> {
        let symbol = match self.get_node(func_node)?.operation {
            table::Operation::DefineFunc(symbol) => symbol,
            table::Operation::DeclareFunc(symbol) => symbol,
            _ => return Err(table::ModelError::UnexpectedOperation(func_node).into()),
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

        Ok(())
    }

    fn import_node(
        &mut self,
        node_id: table::NodeId,
        parent: Node,
    ) -> Result<Option<Node>, ImportError> {
        let node_data = self.get_node(node_id)?;

        match node_data.operation {
            table::Operation::Invalid => Err(table::ModelError::InvalidOperation(node_id).into()),
            table::Operation::Dfg => {
                let signature = self.get_node_signature(node_id)?;
                let optype = OpType::DFG(DFG { signature });
                let node = self.make_node(node_id, optype, parent)?;

                let [region] = node_data.regions else {
                    return Err(table::ModelError::InvalidRegions(node_id).into());
                };

                self.import_dfg_region(node_id, *region, node)?;
                Ok(Some(node))
            }

            table::Operation::Cfg => {
                let signature = self.get_node_signature(node_id)?;
                let optype = OpType::CFG(CFG { signature });
                let node = self.make_node(node_id, optype, parent)?;

                let [region] = node_data.regions else {
                    return Err(table::ModelError::InvalidRegions(node_id).into());
                };

                self.import_cfg_region(node_id, *region, node)?;
                Ok(Some(node))
            }

            table::Operation::Block => {
                let node = self.import_cfg_block(node_id, parent)?;
                Ok(Some(node))
            }

            table::Operation::DefineFunc(symbol) => {
                self.import_poly_func_type(node_id, *symbol, |ctx, signature| {
                    let optype = OpType::FuncDefn(FuncDefn {
                        name: symbol.name.to_string(),
                        signature,
                    });

                    let node = ctx.make_node(node_id, optype, parent)?;

                    let [region] = node_data.regions else {
                        return Err(table::ModelError::InvalidRegions(node_id).into());
                    };

                    ctx.import_dfg_region(node_id, *region, node)?;

                    Ok(Some(node))
                })
            }

            table::Operation::DeclareFunc(symbol) => {
                self.import_poly_func_type(node_id, *symbol, |ctx, signature| {
                    let optype = OpType::FuncDecl(FuncDecl {
                        name: symbol.name.to_string(),
                        signature,
                    });

                    let node = ctx.make_node(node_id, optype, parent)?;

                    Ok(Some(node))
                })
            }

            table::Operation::TailLoop => {
                let node = self.import_tail_loop(node_id, parent)?;
                Ok(Some(node))
            }
            table::Operation::Conditional => {
                let node = self.import_conditional(node_id, parent)?;
                Ok(Some(node))
            }

            table::Operation::Custom(operation) => {
                if let Some([_, _]) = self.match_symbol(operation, model::CORE_CALL_INDIRECT)? {
                    let signature = self.get_node_signature(node_id)?;
                    let optype = OpType::CallIndirect(CallIndirect { signature });
                    let node = self.make_node(node_id, optype, parent)?;
                    return Ok(Some(node));
                }

                if let Some([_, _, func]) = self.match_symbol(operation, model::CORE_CALL)? {
                    let table::Term::Apply(symbol, args) = self.get_term(func)? else {
                        return Err(table::ModelError::TypeError(func).into());
                    };

                    let func_sig = self.get_func_signature(*symbol)?;

                    let type_args = args
                        .iter()
                        .map(|term| self.import_type_arg(*term))
                        .collect::<Result<Vec<TypeArg>, _>>()?;

                    self.static_edges.push((*symbol, node_id));
                    let optype = OpType::Call(Call::try_new(func_sig, type_args)?);

                    let node = self.make_node(node_id, optype, parent)?;
                    return Ok(Some(node));
                }

                if let Some([_, value]) = self.match_symbol(operation, model::CORE_LOAD_CONST)? {
                    // If the constant refers directly to a function, import this as the `LoadFunc` operation.
                    if let table::Term::Apply(symbol, args) = self.get_term(value)? {
                        let func_node_data = self
                            .module
                            .get_node(*symbol)
                            .ok_or(table::ModelError::NodeNotFound(*symbol))?;

                        if let table::Operation::DefineFunc(_) | table::Operation::DeclareFunc(_) =
                            func_node_data.operation
                        {
                            let func_sig = self.get_func_signature(*symbol)?;
                            let type_args = args
                                .iter()
                                .map(|term| self.import_type_arg(*term))
                                .collect::<Result<Vec<TypeArg>, _>>()?;

                            self.static_edges.push((*symbol, node_id));

                            let optype =
                                OpType::LoadFunction(LoadFunction::try_new(func_sig, type_args)?);

                            let node = self.make_node(node_id, optype, parent)?;
                            return Ok(Some(node));
                        }
                    }

                    // Otherwise use const nodes
                    let signature = node_data
                        .signature
                        .ok_or_else(|| error_uninferred!("node signature"))?;
                    let [_, outputs] = self.get_func_type(signature)?;
                    let outputs = self.import_closed_list(outputs)?;
                    let output = outputs
                        .first()
                        .ok_or(table::ModelError::TypeError(signature))?;
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

                    return Ok(Some(load_const_node));
                }

                if let Some([_, _, tag]) = self.match_symbol(operation, model::CORE_MAKE_ADT)? {
                    let table::Term::Literal(model::Literal::Nat(tag)) = self.get_term(tag)? else {
                        return Err(table::ModelError::TypeError(tag).into());
                    };

                    let signature = node_data
                        .signature
                        .ok_or_else(|| error_uninferred!("node signature"))?;
                    let [_, outputs] = self.get_func_type(signature)?;
                    let (variants, _) = self.import_adt_and_rest(node_id, outputs)?;
                    let node = self.make_node(
                        node_id,
                        OpType::Tag(Tag {
                            variants,
                            tag: *tag as usize,
                        }),
                        parent,
                    )?;
                    return Ok(Some(node));
                }

                let table::Term::Apply(node, params) = self.get_term(operation)? else {
                    return Err(table::ModelError::TypeError(operation).into());
                };
                let name = self.get_symbol_name(*node)?;
                let args = params
                    .iter()
                    .map(|param| self.import_type_arg(*param))
                    .collect::<Result<Vec<_>, _>>()?;
                let (extension, name) = self.import_custom_name(name)?;
                let signature = self.get_node_signature(node_id)?;

                // TODO: Currently we do not have the description or any other metadata for
                // the custom op. This will improve with declarative extensions being able
                // to declare operations as a node, in which case the description will be attached
                // to that node as metadata.

                let optype = OpType::OpaqueOp(OpaqueOp::new(
                    extension,
                    name,
                    String::default(),
                    args,
                    signature,
                ));

                let node = self.make_node(node_id, optype, parent)?;

                Ok(Some(node))
            }

            table::Operation::DefineAlias(symbol, value) => {
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
                Ok(Some(node))
            }

            table::Operation::DeclareAlias(symbol) => {
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
                Ok(Some(node))
            }

            table::Operation::Import { .. } => Ok(None),

            table::Operation::DeclareConstructor { .. } => Ok(None),
            table::Operation::DeclareOperation { .. } => Ok(None),
        }
    }

    fn import_dfg_region(
        &mut self,
        node_id: table::NodeId,
        region: table::RegionId,
        node: Node,
    ) -> Result<(), ImportError> {
        let region_data = self.get_region(region)?;

        let prev_region = self.region_scope;
        if region_data.scope.is_some() {
            self.region_scope = region;
        }

        if region_data.kind != model::RegionKind::DataFlow {
            return Err(table::ModelError::InvalidRegions(node_id).into());
        }

        let signature = self.import_func_type(
            region_data
                .signature
                .ok_or_else(|| error_uninferred!("region signature"))?,
        )?;

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
        node_id: table::NodeId,
        list: table::TermId,
    ) -> Result<(Vec<TypeRow>, TypeRow), ImportError> {
        let items = self.import_closed_list(list)?;

        let Some((first, rest)) = items.split_first() else {
            return Err(table::ModelError::InvalidRegions(node_id).into());
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
            return Err(table::ModelError::InvalidRegions(node_id).into());
        };
        let region_data = self.get_region(*region)?;

        let [_, region_outputs] = self.get_func_type(
            region_data
                .signature
                .ok_or_else(|| error_uninferred!("region signature"))?,
        )?;
        let (sum_rows, rest) = self.import_adt_and_rest(node_id, region_outputs)?;

        let (just_inputs, just_outputs) = {
            let mut sum_rows = sum_rows.into_iter();

            let Some(just_inputs) = sum_rows.next() else {
                return Err(table::ModelError::TypeError(region_outputs).into());
            };

            let Some(just_outputs) = sum_rows.next() else {
                return Err(table::ModelError::TypeError(region_outputs).into());
            };

            (just_inputs, just_outputs)
        };

        let optype = OpType::TailLoop(TailLoop {
            just_inputs,
            just_outputs,
            rest,
        });

        let node = self.make_node(node_id, optype, parent)?;

        self.import_dfg_region(node_id, *region, node)?;
        Ok(node)
    }

    fn import_conditional(
        &mut self,
        node_id: table::NodeId,
        parent: Node,
    ) -> Result<Node, ImportError> {
        let node_data = self.get_node(node_id)?;
        debug_assert_eq!(node_data.operation, table::Operation::Conditional);
        let [inputs, outputs] = self.get_func_type(
            node_data
                .signature
                .ok_or_else(|| error_uninferred!("node signature"))?,
        )?;
        let (sum_rows, other_inputs) = self.import_adt_and_rest(node_id, inputs)?;
        let outputs = self.import_type_row(outputs)?;

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

            self.import_dfg_region(node_id, *region, case_node)?;
        }

        Ok(node)
    }

    fn import_cfg_region(
        &mut self,
        node_id: table::NodeId,
        region: table::RegionId,
        node: Node,
    ) -> Result<(), ImportError> {
        let region_data = self.get_region(region)?;

        if region_data.kind != model::RegionKind::ControlFlow {
            return Err(table::ModelError::InvalidRegions(node_id).into());
        }

        let prev_region = self.region_scope;
        if region_data.scope.is_some() {
            self.region_scope = region;
        }

        let [region_source, region_targets] = self.get_func_type(
            region_data
                .signature
                .ok_or_else(|| error_uninferred!("region signature"))?,
        )?;

        let region_source_types = self.import_closed_list(region_source)?;
        let region_target_types = self.import_closed_list(region_targets)?;

        // Create the entry node for the control flow region.
        // Since the core hugr does not have explicit entry blocks yet, we create a dataflow block
        // that simply forwards its inputs to its outputs.
        {
            let types = {
                let [ctrl_type] = region_source_types.as_slice() else {
                    return Err(table::ModelError::TypeError(region_source).into());
                };

                let [types] = self.expect_symbol(*ctrl_type, model::CORE_CTRL)?;
                self.import_type_row(types)?
            };

            let entry = self.hugr.add_node_with_parent(
                node,
                OpType::DataflowBlock(DataflowBlock {
                    inputs: types.clone(),
                    other_outputs: TypeRow::default(),
                    sum_rows: vec![types.clone()],
                }),
            );

            self.record_links(entry, Direction::Outgoing, region_data.sources);

            let node_input = self.hugr.add_node_with_parent(
                entry,
                OpType::Input(Input {
                    types: types.clone(),
                }),
            );

            let node_output = self.hugr.add_node_with_parent(
                entry,
                OpType::Output(Output {
                    types: vec![Type::new_sum([types.clone()])].into(),
                }),
            );

            let node_tag = self.hugr.add_node_with_parent(
                entry,
                OpType::Tag(Tag {
                    tag: 0,
                    variants: vec![types],
                }),
            );

            // Connect the input node to the tag node
            let input_outputs = self.hugr.node_outputs(node_input);
            let tag_inputs = self.hugr.node_inputs(node_tag);
            let mut connections =
                Vec::with_capacity(input_outputs.size_hint().0 + tag_inputs.size_hint().0);

            for (a, b) in input_outputs.zip(tag_inputs) {
                connections.push((node_input, a, node_tag, b));
            }

            // Connect the tag node to the output node
            let tag_outputs = self.hugr.node_outputs(node_tag);
            let output_inputs = self.hugr.node_inputs(node_output);

            for (a, b) in tag_outputs.zip(output_inputs) {
                connections.push((node_tag, a, node_output, b));
            }

            for (src, src_port, dst, dst_port) in connections {
                self.hugr.connect(src, src_port, dst, dst_port);
            }
        }

        for child in region_data.children {
            self.import_node(*child, node)?;
        }

        // Create the exit node for the control flow region.
        {
            let cfg_outputs = {
                let [ctrl_type] = region_target_types.as_slice() else {
                    return Err(table::ModelError::TypeError(region_targets).into());
                };

                let [types] = self.expect_symbol(*ctrl_type, model::CORE_CTRL)?;
                self.import_type_row(types)?
            };

            let exit = self
                .hugr
                .add_node_with_parent(node, OpType::ExitBlock(ExitBlock { cfg_outputs }));
            self.record_links(exit, Direction::Incoming, region_data.targets);
        }

        for meta_item in region_data.meta {
            self.import_node_metadata(node, *meta_item)?;
        }

        self.region_scope = prev_region;

        Ok(())
    }

    fn import_cfg_block(
        &mut self,
        node_id: table::NodeId,
        parent: Node,
    ) -> Result<Node, ImportError> {
        let node_data = self.get_node(node_id)?;
        debug_assert_eq!(node_data.operation, table::Operation::Block);

        let [region] = node_data.regions else {
            return Err(table::ModelError::InvalidRegions(node_id).into());
        };
        let region_data = self.get_region(*region)?;
        let [inputs, outputs] = self.get_func_type(
            region_data
                .signature
                .ok_or_else(|| error_uninferred!("region signature"))?,
        )?;
        let inputs = self.import_type_row(inputs)?;
        let (sum_rows, other_outputs) = self.import_adt_and_rest(node_id, outputs)?;

        let optype = OpType::DataflowBlock(DataflowBlock {
            inputs,
            other_outputs,
            sum_rows,
        });
        let node = self.make_node(node_id, optype, parent)?;

        self.import_dfg_region(node_id, *region, node)?;
        Ok(node)
    }

    fn import_poly_func_type<RV: MaybeRV, T>(
        &mut self,
        node: table::NodeId,
        symbol: table::Symbol<'a>,
        in_scope: impl FnOnce(&mut Self, PolyFuncTypeBase<RV>) -> Result<T, ImportError>,
    ) -> Result<T, ImportError> {
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
                    .ok_or(table::ModelError::InvalidVar(*var))?
                    .bound = TypeBound::Copyable;
            } else {
                return Err(error_unsupported!("constraint other than copy or discard"));
            }
        }

        for (index, param) in symbol.params.iter().enumerate() {
            // NOTE: `PolyFuncType` only has explicit type parameters at present.
            let bound = self.local_vars[&table::VarId(node, index as _)].bound;
            imported_params.push(self.import_type_param(param.r#type, bound)?);
        }

        let body = self.import_func_type::<RV>(symbol.signature)?;
        in_scope(self, PolyFuncTypeBase::new(imported_params, body))
    }

    /// Import a [`TypeParam`] from a term that represents a static type.
    fn import_type_param(
        &mut self,
        term_id: table::TermId,
        bound: TypeBound,
    ) -> Result<TypeParam, ImportError> {
        if let Some([]) = self.match_symbol(term_id, model::CORE_STR_TYPE)? {
            return Ok(TypeParam::String);
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_NAT_TYPE)? {
            return Ok(TypeParam::max_nat());
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_BYTES_TYPE)? {
            return Err(error_unsupported!(
                "`{}` as `TypeParam`",
                model::CORE_BYTES_TYPE
            ));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_FLOAT_TYPE)? {
            return Err(error_unsupported!(
                "`{}` as `TypeParam`",
                model::CORE_FLOAT_TYPE
            ));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_TYPE)? {
            return Ok(TypeParam::Type { b: bound });
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_STATIC)? {
            return Err(error_unsupported!(
                "`{}` as `TypeParam`",
                model::CORE_STATIC
            ));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_CONSTRAINT)? {
            return Err(error_unsupported!(
                "`{}` as `TypeParam`",
                model::CORE_CONSTRAINT
            ));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_CONST)? {
            return Err(error_unsupported!("`{}` as `TypeParam`", model::CORE_CONST));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_CTRL_TYPE)? {
            return Err(error_unsupported!(
                "`{}` as `TypeParam`",
                model::CORE_CTRL_TYPE
            ));
        }

        if let Some([item_type]) = self.match_symbol(term_id, model::CORE_LIST_TYPE)? {
            // At present `hugr-model` has no way to express that the item
            // type of a list must be copyable. Therefore we import it as `Any`.
            let param = Box::new(self.import_type_param(item_type, TypeBound::Any)?);
            return Ok(TypeParam::List { param });
        }

        if let Some([_]) = self.match_symbol(term_id, model::CORE_TUPLE_TYPE)? {
            // At present `hugr-model` has no way to express that the item
            // types of a tuple must be copyable. Therefore we import it as `Any`.
            todo!("import tuple type");
        }

        match self.get_term(term_id)? {
            table::Term::Wildcard => Err(error_uninferred!("wildcard")),

            table::Term::Var { .. } => Err(error_unsupported!("type variable as `TypeParam`")),
            table::Term::Apply(symbol, _) => {
                let name = self.get_symbol_name(*symbol)?;
                Err(error_unsupported!("custom type `{}` as `TypeParam`", name))
            }

            table::Term::Tuple(_)
            | table::Term::List { .. }
            | table::Term::Func { .. }
            | table::Term::Literal(_) => Err(table::ModelError::TypeError(term_id).into()),
        }
    }

    /// Import a `TypeArg` from a term that represents a static type or value.
    fn import_type_arg(&mut self, term_id: table::TermId) -> Result<TypeArg, ImportError> {
        if let Some([]) = self.match_symbol(term_id, model::CORE_STR_TYPE)? {
            return Err(error_unsupported!(
                "`{}` as `TypeArg`",
                model::CORE_STR_TYPE
            ));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_NAT_TYPE)? {
            return Err(error_unsupported!(
                "`{}` as `TypeArg`",
                model::CORE_NAT_TYPE
            ));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_BYTES_TYPE)? {
            return Err(error_unsupported!(
                "`{}` as `TypeArg`",
                model::CORE_BYTES_TYPE
            ));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_FLOAT_TYPE)? {
            return Err(error_unsupported!(
                "`{}` as `TypeArg`",
                model::CORE_FLOAT_TYPE
            ));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_TYPE)? {
            return Err(error_unsupported!("`{}` as `TypeArg`", model::CORE_TYPE));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_CONSTRAINT)? {
            return Err(error_unsupported!(
                "`{}` as `TypeArg`",
                model::CORE_CONSTRAINT
            ));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_STATIC)? {
            return Err(error_unsupported!("`{}` as `TypeArg`", model::CORE_STATIC));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_CTRL_TYPE)? {
            return Err(error_unsupported!(
                "`{}` as `TypeArg`",
                model::CORE_CTRL_TYPE
            ));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_CONST)? {
            return Err(error_unsupported!("`{}` as `TypeArg`", model::CORE_CONST));
        }

        if let Some([]) = self.match_symbol(term_id, model::CORE_LIST_TYPE)? {
            return Err(error_unsupported!(
                "`{}` as `TypeArg`",
                model::CORE_LIST_TYPE
            ));
        }

        match self.get_term(term_id)? {
            table::Term::Wildcard => Err(error_uninferred!("wildcard")),

            table::Term::Var(var) => {
                let var_info = self
                    .local_vars
                    .get(var)
                    .ok_or(table::ModelError::InvalidVar(*var))?;
                let decl = self.import_type_param(var_info.r#type, var_info.bound)?;
                Ok(TypeArg::new_var_use(var.1 as _, decl))
            }

            table::Term::List { .. } => {
                let elems = self
                    .import_closed_list(term_id)?
                    .iter()
                    .map(|item| self.import_type_arg(*item))
                    .collect::<Result<_, _>>()?;

                Ok(TypeArg::Sequence { elems })
            }

            table::Term::Tuple { .. } => {
                // NOTE: While `TypeArg`s can represent tuples as
                // `TypeArg::Sequence`s, this conflates lists and tuples. To
                // avoid ambiguity we therefore report an error here for now.
                Err(error_unsupported!("tuples as `TypeArg`"))
            }

            table::Term::Literal(model::Literal::Str(value)) => Ok(TypeArg::String {
                arg: value.to_string(),
            }),

            table::Term::Literal(model::Literal::Nat(value)) => {
                Ok(TypeArg::BoundedNat { n: *value })
            }

            table::Term::Literal(model::Literal::Bytes(_)) => {
                Err(error_unsupported!("`(bytes ..)` as `TypeArg`"))
            }
            table::Term::Literal(model::Literal::Float(_)) => {
                Err(error_unsupported!("float literal as `TypeArg`"))
            }
            table::Term::Func { .. } => Err(error_unsupported!("function constant as `TypeArg`")),

            table::Term::Apply { .. } => {
                let ty = self.import_type(term_id)?;
                Ok(TypeArg::Type { ty })
            }
        }
    }

    /// Import a `Type` from a term that represents a runtime type.
    fn import_type<RV: MaybeRV>(
        &mut self,
        term_id: table::TermId,
    ) -> Result<TypeBase<RV>, ImportError> {
        if let Some([_, _]) = self.match_symbol(term_id, model::CORE_FN)? {
            let func_type = self.import_func_type::<RowVariable>(term_id)?;
            return Ok(TypeBase::new_function(func_type));
        }

        if let Some([variants]) = self.match_symbol(term_id, model::CORE_ADT)? {
            let variants = self.import_closed_list(variants)?;
            let variants = variants
                .iter()
                .map(|variant| self.import_type_row::<RowVariable>(*variant))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(TypeBase::new_sum(variants));
        }

        match self.get_term(term_id)? {
            table::Term::Wildcard => Err(error_uninferred!("wildcard")),

            table::Term::Apply(symbol, args) => {
                let args = args
                    .iter()
                    .map(|arg| self.import_type_arg(*arg))
                    .collect::<Result<Vec<_>, _>>()?;

                let name = self.get_symbol_name(*symbol)?;
                let (extension, id) = self.import_custom_name(name)?;

                let extension_ref =
                    self.extensions
                        .get(&extension)
                        .ok_or_else(|| ImportError::Extension {
                            missing_ext: extension.clone(),
                            available: self.extensions.ids().cloned().collect(),
                        })?;

                let ext_type =
                    extension_ref
                        .get_type(&id)
                        .ok_or_else(|| ImportError::ExtensionType {
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

            table::Term::Var(table::VarId(_, index)) => {
                Ok(TypeBase::new_var_use(*index as _, TypeBound::Copyable))
            }

            // The following terms are not runtime types, but the core `Type` only contains runtime types.
            // We therefore report a type error here.
            table::Term::List { .. }
            | table::Term::Tuple { .. }
            | table::Term::Literal(_)
            | table::Term::Func { .. } => Err(table::ModelError::TypeError(term_id).into()),
        }
    }

    fn get_func_type(&mut self, term_id: table::TermId) -> Result<[table::TermId; 2], ImportError> {
        self.match_symbol(term_id, model::CORE_FN)?
            .ok_or(table::ModelError::TypeError(term_id).into())
    }

    fn import_func_type<RV: MaybeRV>(
        &mut self,
        term_id: table::TermId,
    ) -> Result<FuncTypeBase<RV>, ImportError> {
        let [inputs, outputs] = self.get_func_type(term_id)?;
        let inputs = self.import_type_row(inputs)?;
        let outputs = self.import_type_row(outputs)?;
        Ok(FuncTypeBase::new(inputs, outputs))
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
                _ => return Err(table::ModelError::TypeError(term_id).into()),
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
                _ => return Err(table::ModelError::TypeError(term_id).into()),
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
                        .map_err(|_| table::ModelError::TypeError(term_id))?;
                    types.push(TypeBase::new(TypeEnum::RowVar(var)));
                }
                _ => return Err(table::ModelError::TypeError(term_id).into()),
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
                    .map_err(|_| table::ModelError::MalformedName(symbol.to_smolstr()))?;

                let (extension, id) = qualified_name
                    .split_last()
                    .ok_or_else(|| table::ModelError::MalformedName(symbol.to_smolstr()))?;

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
                return Err(table::ModelError::TypeError(term_id).into());
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
                let value: serde_json::Value = serde_json::from_str(json)
                    .map_err(|_| table::ModelError::TypeError(term_id))?;
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
                    return Err(table::ModelError::TypeError(term_id).into());
                };
                if *bitwidth > 6 {
                    return Err(table::ModelError::TypeError(term_id).into());
                }
                *bitwidth as u8
            };

            let value = {
                let table::Term::Literal(model::Literal::Nat(value)) = self.get_term(value)? else {
                    return Err(table::ModelError::TypeError(term_id).into());
                };
                *value
            };

            return Ok(ConstInt::new_u(bitwidth, value)
                .map_err(|_| table::ModelError::TypeError(term_id))?
                .into());
        }

        if let Some([value]) = self.match_symbol(term_id, ConstF64::CTR_NAME)? {
            let table::Term::Literal(model::Literal::Float(value)) = self.get_term(value)? else {
                return Err(table::ModelError::TypeError(term_id).into());
            };

            return Ok(ConstF64::new(value.into_inner()).into());
        }

        if let Some([_, _, tag, values]) = self.match_symbol(term_id, model::CORE_CONST_ADT)? {
            let [variants] = self.expect_symbol(type_id, model::CORE_ADT)?;
            let values = self.import_closed_tuple(values)?;
            let variants = self.import_closed_list(variants)?;

            let table::Term::Literal(model::Literal::Nat(tag)) = self.get_term(tag)? else {
                return Err(table::ModelError::TypeError(term_id).into());
            };

            let variant = variants
                .get(*tag as usize)
                .ok_or(table::ModelError::TypeError(term_id))?;

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
                Err(table::ModelError::TypeError(term_id).into())
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

        Ok((*args).try_into().ok())
    }

    fn expect_symbol<const N: usize>(
        &self,
        term_id: table::TermId,
        name: &str,
    ) -> Result<[table::TermId; N], ImportError> {
        self.match_symbol(term_id, name)?
            .ok_or(table::ModelError::TypeError(term_id).into())
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
