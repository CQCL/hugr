//! Importing HUGR graphs from their `hugr-model` representation.
//!
//! **Warning**: This module is still under development and is expected to change.
//! It is included in the library to allow for early experimentation, and for
//! the core and model to converge incrementally.
use std::sync::Arc;

use crate::{
    export::OP_FUNC_CALL_INDIRECT,
    extension::{ExtensionId, ExtensionRegistry, ExtensionSet, SignatureError},
    hugr::{HugrMut, IdentList},
    ops::{
        constant::{CustomConst, CustomSerialized, OpaqueValue},
        AliasDecl, AliasDefn, Call, CallIndirect, Case, Conditional, Const, DataflowBlock,
        ExitBlock, FuncDecl, FuncDefn, Input, LoadConstant, LoadFunction, Module, OpType, OpaqueOp,
        Output, Tag, TailLoop, Value, CFG, DFG,
    },
    std_extensions::{
        arithmetic::{float_types::ConstF64, int_types::ConstInt},
        collections::array::ArrayValue,
    },
    types::{
        type_param::TypeParam, type_row::TypeRowBase, CustomType, FuncTypeBase, MaybeRV,
        PolyFuncType, PolyFuncTypeBase, RowVariable, Signature, Type, TypeArg, TypeBase, TypeBound,
        TypeEnum, TypeName, TypeRow,
    },
    Direction, Hugr, HugrView, Node, Port,
};
use fxhash::FxHashMap;
use hugr_model::v0::{self as model};
use itertools::Either;
use smol_str::{SmolStr, ToSmolStr};
use thiserror::Error;

/// Error during import.
#[derive(Debug, Clone, Error)]
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
            available.iter().map(|ext| ext.to_string()).collect::<Vec<_>>().join(", "))]
    Extension {
        /// The missing extension.
        missing_ext: ExtensionId,
        /// The available extensions in the registry.
        available: Vec<ExtensionId>,
    },
    /// An extension type is missing.
    #[error("Importing the hugr requires extension {ext} to have a type named {name}, but it was not found.")]
    ExtensionType {
        /// The extension that is missing the type.
        ext: ExtensionId,
        /// The name of the missing type.
        name: TypeName,
    },
    /// The model is not well-formed.
    #[error("validate error: {0}")]
    Model(#[from] model::ModelError),
}

/// Helper macro to create an `ImportError::Unsupported` error with a formatted message.
macro_rules! error_unsupported {
    ($($e:expr),*) => { ImportError::Unsupported(format!($($e),*)) }
}

/// Helper macro to create an `ImportError::Uninferred` error with a formatted message.
macro_rules! error_uninferred {
    ($($e:expr),*) => { ImportError::Uninferred(format!($($e),*)) }
}

/// Import a `hugr` module from its model representation.
pub fn import_hugr(
    module: &model::Module,
    extensions: &ExtensionRegistry,
) -> Result<Hugr, ImportError> {
    // TODO: Module should know about the number of edges, so that we can use a vector here.
    // For now we use a hashmap, which will be slower.
    let mut ctx = Context {
        module,
        hugr: Hugr::new(OpType::Module(Module {})),
        link_ports: FxHashMap::default(),
        static_edges: Vec::new(),
        extensions,
        nodes: FxHashMap::default(),
        local_vars: FxHashMap::default(),
        custom_name_cache: FxHashMap::default(),
        region_scope: model::RegionId::default(),
    };

    ctx.import_root()?;
    ctx.link_ports()?;
    ctx.link_static_ports()?;

    Ok(ctx.hugr)
}

struct Context<'a> {
    /// The module being imported.
    module: &'a model::Module<'a>,

    /// The HUGR graph being constructed.
    hugr: Hugr,

    /// The ports that are part of each link. This is used to connect the ports at the end of the
    /// import process.
    link_ports: FxHashMap<(model::RegionId, model::LinkIndex), Vec<(Node, Port)>>,

    /// Pairs of nodes that should be connected by a static edge.
    /// These are collected during the import process and connected at the end.
    static_edges: Vec<(model::NodeId, model::NodeId)>,

    /// The ambient extension registry to use for importing.
    extensions: &'a ExtensionRegistry,

    /// A map from `NodeId` to the imported `Node`.
    nodes: FxHashMap<model::NodeId, Node>,

    local_vars: FxHashMap<model::VarId, LocalVar>,

    custom_name_cache: FxHashMap<&'a str, (ExtensionId, SmolStr)>,

    region_scope: model::RegionId,
}

impl<'a> Context<'a> {
    /// Get the signature of the node with the given `NodeId`.
    fn get_node_signature(&mut self, node: model::NodeId) -> Result<Signature, ImportError> {
        let node_data = self.get_node(node)?;
        let signature = node_data
            .signature
            .ok_or_else(|| error_uninferred!("node signature"))?;
        self.import_func_type(signature)
    }

    /// Get the node with the given `NodeId`, or return an error if it does not exist.
    #[inline]
    fn get_node(&self, node_id: model::NodeId) -> Result<&'a model::Node<'a>, ImportError> {
        self.module
            .get_node(node_id)
            .ok_or_else(|| model::ModelError::NodeNotFound(node_id).into())
    }

    /// Get the term with the given `TermId`, or return an error if it does not exist.
    #[inline]
    fn get_term(&self, term_id: model::TermId) -> Result<&'a model::Term<'a>, ImportError> {
        self.module
            .get_term(term_id)
            .ok_or_else(|| model::ModelError::TermNotFound(term_id).into())
    }

    /// Get the region with the given `RegionId`, or return an error if it does not exist.
    #[inline]
    fn get_region(&self, region_id: model::RegionId) -> Result<&'a model::Region<'a>, ImportError> {
        self.module
            .get_region(region_id)
            .ok_or_else(|| model::ModelError::RegionNotFound(region_id).into())
    }

    fn make_node(
        &mut self,
        node_id: model::NodeId,
        op: OpType,
        parent: Node,
    ) -> Result<Node, ImportError> {
        let node = self.hugr.add_node_with_parent(parent, op);
        self.nodes.insert(node_id, node);

        let node_data = self.get_node(node_id)?;
        self.record_links(node, Direction::Incoming, node_data.inputs);
        self.record_links(node, Direction::Outgoing, node_data.outputs);

        for meta_item in node_data.meta {
            // TODO: For now we expect all metadata to be JSON since this is how
            // it is handled in `hugr-core`.
            let (name, value) = self.import_json_meta(*meta_item)?;
            self.hugr.set_metadata(node, name, value);
        }

        Ok(node)
    }

    /// Associate links with the ports of the given node in the given direction.
    fn record_links(&mut self, node: Node, direction: Direction, links: &'a [model::LinkIndex]) {
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
                    for (node, port) in inputs.iter() {
                        self.hugr.connect(output.0, output.1, *node, *port);
                    }
                }
                ([input], _) => {
                    for (node, port) in outputs.iter() {
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

    fn get_symbol_name(&self, node_id: model::NodeId) -> Result<&'a str, ImportError> {
        let node_data = self.get_node(node_id)?;
        let name = node_data
            .operation
            .symbol()
            .ok_or(model::ModelError::InvalidSymbol(node_id))?;
        Ok(name)
    }

    fn get_func_signature(
        &mut self,
        func_node: model::NodeId,
    ) -> Result<PolyFuncType, ImportError> {
        let decl = match self.get_node(func_node)?.operation {
            model::Operation::DefineFunc { decl } => decl,
            model::Operation::DeclareFunc { decl } => decl,
            _ => return Err(model::ModelError::UnexpectedOperation(func_node).into()),
        };

        self.import_poly_func_type(func_node, *decl, |_, signature| Ok(signature))
    }

    /// Import the root region of the module.
    fn import_root(&mut self) -> Result<(), ImportError> {
        self.region_scope = self.module.root;
        let region_data = self.get_region(self.module.root)?;

        for node in region_data.children {
            self.import_node(*node, self.hugr.root())?;
        }

        Ok(())
    }

    fn import_node(
        &mut self,
        node_id: model::NodeId,
        parent: Node,
    ) -> Result<Option<Node>, ImportError> {
        let node_data = self.get_node(node_id)?;

        match node_data.operation {
            model::Operation::Invalid => Err(model::ModelError::InvalidOperation(node_id).into()),
            model::Operation::Dfg => {
                let signature = self.get_node_signature(node_id)?;
                let optype = OpType::DFG(DFG { signature });
                let node = self.make_node(node_id, optype, parent)?;

                let [region] = node_data.regions else {
                    return Err(model::ModelError::InvalidRegions(node_id).into());
                };

                self.import_dfg_region(node_id, *region, node)?;
                Ok(Some(node))
            }

            model::Operation::Cfg => {
                let signature = self.get_node_signature(node_id)?;
                let optype = OpType::CFG(CFG { signature });
                let node = self.make_node(node_id, optype, parent)?;

                let [region] = node_data.regions else {
                    return Err(model::ModelError::InvalidRegions(node_id).into());
                };

                self.import_cfg_region(node_id, *region, node)?;
                Ok(Some(node))
            }

            model::Operation::Block => {
                let node = self.import_cfg_block(node_id, parent)?;
                Ok(Some(node))
            }

            model::Operation::DefineFunc { decl } => {
                self.import_poly_func_type(node_id, *decl, |ctx, signature| {
                    let optype = OpType::FuncDefn(FuncDefn {
                        name: decl.name.to_string(),
                        signature,
                    });

                    let node = ctx.make_node(node_id, optype, parent)?;

                    let [region] = node_data.regions else {
                        return Err(model::ModelError::InvalidRegions(node_id).into());
                    };

                    ctx.import_dfg_region(node_id, *region, node)?;

                    Ok(Some(node))
                })
            }

            model::Operation::DeclareFunc { decl } => {
                self.import_poly_func_type(node_id, *decl, |ctx, signature| {
                    let optype = OpType::FuncDecl(FuncDecl {
                        name: decl.name.to_string(),
                        signature,
                    });

                    let node = ctx.make_node(node_id, optype, parent)?;

                    Ok(Some(node))
                })
            }

            model::Operation::CallFunc { func } => {
                let model::Term::ApplyFull { symbol, args } = self.get_term(func)? else {
                    return Err(model::ModelError::TypeError(func).into());
                };

                let func_sig = self.get_func_signature(*symbol)?;

                let type_args = args
                    .iter()
                    .map(|term| self.import_type_arg(*term))
                    .collect::<Result<Vec<TypeArg>, _>>()?;

                self.static_edges.push((*symbol, node_id));
                let optype = OpType::Call(Call::try_new(func_sig, type_args)?);

                let node = self.make_node(node_id, optype, parent)?;
                Ok(Some(node))
            }

            model::Operation::LoadFunc { func } => {
                let model::Term::ApplyFull { symbol, args } = self.get_term(func)? else {
                    return Err(model::ModelError::TypeError(func).into());
                };

                let func_sig = self.get_func_signature(*symbol)?;

                let type_args = args
                    .iter()
                    .map(|term| self.import_type_arg(*term))
                    .collect::<Result<Vec<TypeArg>, _>>()?;

                self.static_edges.push((*symbol, node_id));

                let optype = OpType::LoadFunction(LoadFunction::try_new(func_sig, type_args)?);

                let node = self.make_node(node_id, optype, parent)?;
                Ok(Some(node))
            }

            model::Operation::TailLoop => {
                let node = self.import_tail_loop(node_id, parent)?;
                Ok(Some(node))
            }
            model::Operation::Conditional => {
                let node = self.import_conditional(node_id, parent)?;
                Ok(Some(node))
            }

            model::Operation::CustomFull { operation } => {
                let name = self.get_symbol_name(operation)?;

                if name == OP_FUNC_CALL_INDIRECT {
                    let signature = self.get_node_signature(node_id)?;
                    let optype = OpType::CallIndirect(CallIndirect { signature });
                    let node = self.make_node(node_id, optype, parent)?;
                    return Ok(Some(node));
                }

                let signature = self.get_node_signature(node_id)?;
                let args = node_data
                    .params
                    .iter()
                    .map(|param| self.import_type_arg(*param))
                    .collect::<Result<Vec<_>, _>>()?;

                let (extension, name) = self.import_custom_name(name)?;

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

            model::Operation::Custom { .. } => Err(error_unsupported!(
                "custom operation with implicit parameters"
            )),

            model::Operation::DefineAlias { decl, value } => {
                if !decl.params.is_empty() {
                    return Err(error_unsupported!(
                        "parameters or constraints in alias definition"
                    ));
                }

                let optype = OpType::AliasDefn(AliasDefn {
                    name: decl.name.to_smolstr(),
                    definition: self.import_type(value)?,
                });

                let node = self.make_node(node_id, optype, parent)?;
                Ok(Some(node))
            }

            model::Operation::DeclareAlias { decl } => {
                if !decl.params.is_empty() {
                    return Err(error_unsupported!(
                        "parameters or constraints in alias declaration"
                    ));
                }

                let optype = OpType::AliasDecl(AliasDecl {
                    name: decl.name.to_smolstr(),
                    bound: TypeBound::Copyable,
                });

                let node = self.make_node(node_id, optype, parent)?;
                Ok(Some(node))
            }

            model::Operation::Tag { tag } => {
                let signature = node_data
                    .signature
                    .ok_or_else(|| error_uninferred!("node signature"))?;
                let (_, outputs, _) = self.get_func_type(signature)?;
                let (variants, _) = self.import_adt_and_rest(node_id, outputs)?;
                let node = self.make_node(
                    node_id,
                    OpType::Tag(Tag {
                        variants,
                        tag: tag as _,
                    }),
                    parent,
                )?;
                Ok(Some(node))
            }

            model::Operation::Import { .. } => Ok(None),

            model::Operation::DeclareConstructor { .. } => Ok(None),
            model::Operation::DeclareOperation { .. } => Ok(None),

            model::Operation::Const { value } => {
                let signature = node_data
                    .signature
                    .ok_or_else(|| error_uninferred!("node signature"))?;
                let (_, outputs, _) = self.get_func_type(signature)?;
                let outputs = self.import_closed_list(outputs)?;
                let output = outputs
                    .first()
                    .ok_or(model::ModelError::TypeError(signature))?;
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

                Ok(Some(load_const_node))
            }
        }
    }

    fn import_dfg_region(
        &mut self,
        node_id: model::NodeId,
        region: model::RegionId,
        node: Node,
    ) -> Result<(), ImportError> {
        let region_data = self.get_region(region)?;

        let prev_region = self.region_scope;
        if region_data.scope.is_some() {
            self.region_scope = region;
        }

        if region_data.kind != model::RegionKind::DataFlow {
            return Err(model::ModelError::InvalidRegions(node_id).into());
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

        self.region_scope = prev_region;

        Ok(())
    }

    fn import_adt_and_rest(
        &mut self,
        node_id: model::NodeId,
        list: model::TermId,
    ) -> Result<(Vec<TypeRow>, TypeRow), ImportError> {
        let items = self.import_closed_list(list)?;

        let Some((first, rest)) = items.split_first() else {
            return Err(model::ModelError::InvalidRegions(node_id).into());
        };

        let sum_rows: Vec<_> = {
            let model::Term::Adt { variants } = self.get_term(*first)? else {
                return Err(model::ModelError::TypeError(*first).into());
            };

            self.import_type_rows(*variants)?
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
        node_id: model::NodeId,
        parent: Node,
    ) -> Result<Node, ImportError> {
        let node_data = self.get_node(node_id)?;
        debug_assert_eq!(node_data.operation, model::Operation::TailLoop);

        let [region] = node_data.regions else {
            return Err(model::ModelError::InvalidRegions(node_id).into());
        };
        let region_data = self.get_region(*region)?;

        let (_, region_outputs, _) = self.get_func_type(
            region_data
                .signature
                .ok_or_else(|| error_uninferred!("region signature"))?,
        )?;
        let (sum_rows, rest) = self.import_adt_and_rest(node_id, region_outputs)?;

        let (just_inputs, just_outputs) = {
            let mut sum_rows = sum_rows.into_iter();

            let Some(just_inputs) = sum_rows.next() else {
                return Err(model::ModelError::TypeError(region_outputs).into());
            };

            let Some(just_outputs) = sum_rows.next() else {
                return Err(model::ModelError::TypeError(region_outputs).into());
            };

            (just_inputs, just_outputs)
        };

        let optype = OpType::TailLoop(TailLoop {
            just_inputs,
            just_outputs,
            rest,
            extension_delta: ExtensionSet::new(),
        });

        let node = self.make_node(node_id, optype, parent)?;

        self.import_dfg_region(node_id, *region, node)?;
        Ok(node)
    }

    fn import_conditional(
        &mut self,
        node_id: model::NodeId,
        parent: Node,
    ) -> Result<Node, ImportError> {
        let node_data = self.get_node(node_id)?;
        debug_assert_eq!(node_data.operation, model::Operation::Conditional);
        let (inputs, outputs, _) = self.get_func_type(
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
            extension_delta: ExtensionSet::new(),
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
        node_id: model::NodeId,
        region: model::RegionId,
        node: Node,
    ) -> Result<(), ImportError> {
        let region_data = self.get_region(region)?;

        if region_data.kind != model::RegionKind::ControlFlow {
            return Err(model::ModelError::InvalidRegions(node_id).into());
        }

        let prev_region = self.region_scope;
        if region_data.scope.is_some() {
            self.region_scope = region;
        }

        let (region_source, region_targets, _) = self.get_func_type(
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
                    return Err(model::ModelError::TypeError(region_source).into());
                };

                let model::Term::Control { values: types } = self.get_term(*ctrl_type)? else {
                    return Err(model::ModelError::TypeError(*ctrl_type).into());
                };

                self.import_type_row(*types)?
            };

            let entry = self.hugr.add_node_with_parent(
                node,
                OpType::DataflowBlock(DataflowBlock {
                    inputs: types.clone(),
                    other_outputs: TypeRow::default(),
                    sum_rows: vec![types.clone()],
                    extension_delta: ExtensionSet::default(),
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
                    return Err(model::ModelError::TypeError(region_targets).into());
                };

                let model::Term::Control { values: types } = self.get_term(*ctrl_type)? else {
                    return Err(model::ModelError::TypeError(*ctrl_type).into());
                };

                self.import_type_row(*types)?
            };

            let exit = self
                .hugr
                .add_node_with_parent(node, OpType::ExitBlock(ExitBlock { cfg_outputs }));
            self.record_links(exit, Direction::Incoming, region_data.targets);
        }

        self.region_scope = prev_region;

        Ok(())
    }

    fn import_cfg_block(
        &mut self,
        node_id: model::NodeId,
        parent: Node,
    ) -> Result<Node, ImportError> {
        let node_data = self.get_node(node_id)?;
        debug_assert_eq!(node_data.operation, model::Operation::Block);

        let [region] = node_data.regions else {
            return Err(model::ModelError::InvalidRegions(node_id).into());
        };
        let region_data = self.get_region(*region)?;
        let (inputs, outputs, extensions) = self.get_func_type(
            region_data
                .signature
                .ok_or_else(|| error_uninferred!("region signature"))?,
        )?;
        let inputs = self.import_type_row(inputs)?;
        let (sum_rows, other_outputs) = self.import_adt_and_rest(node_id, outputs)?;
        let extension_delta = self.import_extension_set(extensions)?;

        let optype = OpType::DataflowBlock(DataflowBlock {
            inputs,
            other_outputs,
            sum_rows,
            extension_delta,
        });
        let node = self.make_node(node_id, optype, parent)?;

        self.import_dfg_region(node_id, *region, node)?;
        Ok(node)
    }

    fn import_poly_func_type<RV: MaybeRV, T>(
        &mut self,
        node: model::NodeId,
        decl: model::FuncDecl<'a>,
        in_scope: impl FnOnce(&mut Self, PolyFuncTypeBase<RV>) -> Result<T, ImportError>,
    ) -> Result<T, ImportError> {
        let mut imported_params = Vec::with_capacity(decl.params.len());

        for (index, param) in decl.params.iter().enumerate() {
            self.local_vars
                .insert(model::VarId(node, index as _), LocalVar::new(param.r#type));
        }

        for constraint in decl.constraints {
            match self.get_term(*constraint)? {
                model::Term::NonLinearConstraint { term } => {
                    let model::Term::Var(var) = self.get_term(*term)? else {
                        return Err(error_unsupported!(
                            "constraint on term that is not a variable"
                        ));
                    };

                    self.local_vars
                        .get_mut(var)
                        .ok_or(model::ModelError::InvalidVar(*var))?
                        .bound = TypeBound::Copyable;
                }
                _ => return Err(error_unsupported!("constraint other than copy or discard")),
            }
        }

        for (index, param) in decl.params.iter().enumerate() {
            // NOTE: `PolyFuncType` only has explicit type parameters at present.
            let bound = self.local_vars[&model::VarId(node, index as _)].bound;
            imported_params.push(self.import_type_param(param.r#type, bound)?);
        }

        let body = self.import_func_type::<RV>(decl.signature)?;
        in_scope(self, PolyFuncTypeBase::new(imported_params, body))
    }

    /// Import a [`TypeParam`] from a term that represents a static type.
    fn import_type_param(
        &mut self,
        term_id: model::TermId,
        bound: TypeBound,
    ) -> Result<TypeParam, ImportError> {
        match self.get_term(term_id)? {
            model::Term::Wildcard => Err(error_uninferred!("wildcard")),

            model::Term::Type => Ok(TypeParam::Type { b: bound }),

            model::Term::StaticType => Err(error_unsupported!("`type` as `TypeParam`")),
            model::Term::Constraint => Err(error_unsupported!("`constraint` as `TypeParam`")),
            model::Term::Var { .. } => Err(error_unsupported!("type variable as `TypeParam`")),
            model::Term::Apply { .. } => Err(error_unsupported!("custom type as `TypeParam`")),
            model::Term::ApplyFull { .. } => Err(error_unsupported!("custom type as `TypeParam`")),
            model::Term::BytesType { .. } => Err(error_unsupported!("`bytes` as `TypeParam`")),
            model::Term::FloatType { .. } => Err(error_unsupported!("`float` as `TypeParam`")),
            model::Term::Const { .. } => Err(error_unsupported!("`(const ...)` as `TypeParam`")),
            model::Term::FuncType { .. } => Err(error_unsupported!("`(fn ...)` as `TypeParam`")),

            model::Term::ListType { item_type } => {
                // At present `hugr-model` has no way to express that the item
                // type of a list must be copyable. Therefore we import it as `Any`.
                let param = Box::new(self.import_type_param(*item_type, TypeBound::Any)?);
                Ok(TypeParam::List { param })
            }

            model::Term::StrType => Ok(TypeParam::String),
            model::Term::ExtSetType => Ok(TypeParam::Extensions),

            model::Term::NatType => Ok(TypeParam::max_nat()),

            model::Term::Nat(_)
            | model::Term::Str(_)
            | model::Term::List { .. }
            | model::Term::ExtSet { .. }
            | model::Term::Adt { .. }
            | model::Term::Control { .. }
            | model::Term::NonLinearConstraint { .. }
            | model::Term::ConstFunc { .. }
            | model::Term::Bytes { .. }
            | model::Term::Meta
            | model::Term::Float { .. }
            | model::Term::ConstAdt { .. } => Err(model::ModelError::TypeError(term_id).into()),

            model::Term::ControlType => {
                Err(error_unsupported!("type of control types as `TypeParam`"))
            }
        }
    }

    /// Import a `TypeArg` from a term that represents a static type or value.
    fn import_type_arg(&mut self, term_id: model::TermId) -> Result<TypeArg, ImportError> {
        match self.get_term(term_id)? {
            model::Term::Wildcard => Err(error_uninferred!("wildcard")),
            model::Term::Apply { .. } => {
                Err(error_uninferred!("application with implicit parameters"))
            }

            model::Term::Var(var) => {
                let var_info = self
                    .local_vars
                    .get(var)
                    .ok_or(model::ModelError::InvalidVar(*var))?;
                let decl = self.import_type_param(var_info.r#type, var_info.bound)?;
                Ok(TypeArg::new_var_use(var.1 as _, decl))
            }

            model::Term::List { .. } => {
                let elems = self
                    .import_closed_list(term_id)?
                    .iter()
                    .map(|item| self.import_type_arg(*item))
                    .collect::<Result<_, _>>()?;

                Ok(TypeArg::Sequence { elems })
            }

            model::Term::Str(value) => Ok(TypeArg::String {
                arg: value.to_string(),
            }),

            model::Term::Nat(value) => Ok(TypeArg::BoundedNat { n: *value }),
            model::Term::ExtSet { .. } => Ok(TypeArg::Extensions {
                es: self.import_extension_set(term_id)?,
            }),

            model::Term::StrType => Err(error_unsupported!("`str` as `TypeArg`")),
            model::Term::NatType => Err(error_unsupported!("`nat` as `TypeArg`")),
            model::Term::ListType { .. } => Err(error_unsupported!("`(list ...)` as `TypeArg`")),
            model::Term::ExtSetType => Err(error_unsupported!("`ext-set` as `TypeArg`")),
            model::Term::Type => Err(error_unsupported!("`type` as `TypeArg`")),
            model::Term::Constraint => Err(error_unsupported!("`constraint` as `TypeArg`")),
            model::Term::StaticType => Err(error_unsupported!("`static` as `TypeArg`")),
            model::Term::ControlType => Err(error_unsupported!("`ctrl` as `TypeArg`")),
            model::Term::BytesType => Err(error_unsupported!("`bytes` as `TypeArg`")),
            model::Term::FloatType => Err(error_unsupported!("`float` as `TypeArg`")),
            model::Term::Bytes { .. } => Err(error_unsupported!("`(bytes ..)` as `TypeArg`")),
            model::Term::Const { .. } => Err(error_unsupported!("`const` as `TypeArg`")),
            model::Term::Float { .. } => Err(error_unsupported!("float literal as `TypeArg`")),
            model::Term::ConstAdt { .. } => Err(error_unsupported!("adt constant as `TypeArg`")),
            model::Term::ConstFunc { .. } => {
                Err(error_unsupported!("function constant as `TypeArg`"))
            }

            model::Term::FuncType { .. }
            | model::Term::Adt { .. }
            | model::Term::ApplyFull { .. } => {
                let ty = self.import_type(term_id)?;
                Ok(TypeArg::Type { ty })
            }

            model::Term::Control { .. }
            | model::Term::Meta
            | model::Term::NonLinearConstraint { .. } => {
                Err(model::ModelError::TypeError(term_id).into())
            }
        }
    }

    fn import_extension_set(
        &mut self,
        term_id: model::TermId,
    ) -> Result<ExtensionSet, ImportError> {
        let mut es = ExtensionSet::new();
        let mut stack = vec![term_id];

        while let Some(term_id) = stack.pop() {
            match self.get_term(term_id)? {
                model::Term::Wildcard => return Err(error_uninferred!("wildcard")),

                model::Term::Var(model::VarId(_, index)) => {
                    es.insert_type_var(*index as _);
                }

                model::Term::ExtSet { parts } => {
                    for part in *parts {
                        match part {
                            model::ExtSetPart::Extension(ext) => {
                                let ext_ident = IdentList::new(*ext).map_err(|_| {
                                    model::ModelError::MalformedName(ext.to_smolstr())
                                })?;
                                es.insert(ext_ident);
                            }
                            model::ExtSetPart::Splice(term_id) => {
                                // The order in an extension set does not matter.
                                stack.push(*term_id);
                            }
                        }
                    }
                }
                _ => return Err(model::ModelError::TypeError(term_id).into()),
            }
        }

        Ok(es)
    }

    /// Import a `Type` from a term that represents a runtime type.
    fn import_type<RV: MaybeRV>(
        &mut self,
        term_id: model::TermId,
    ) -> Result<TypeBase<RV>, ImportError> {
        match self.get_term(term_id)? {
            model::Term::Wildcard => Err(error_uninferred!("wildcard")),
            model::Term::Apply { .. } => {
                Err(error_uninferred!("application with implicit parameters"))
            }

            model::Term::ApplyFull { symbol, args } => {
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

            model::Term::Var(model::VarId(_, index)) => {
                Ok(TypeBase::new_var_use(*index as _, TypeBound::Copyable))
            }

            model::Term::FuncType { .. } => {
                let func_type = self.import_func_type::<RowVariable>(term_id)?;
                Ok(TypeBase::new_function(func_type))
            }

            model::Term::Adt { variants } => {
                let variants = self.import_closed_list(*variants)?;
                let variants = variants
                    .iter()
                    .map(|variant| self.import_type_row::<RowVariable>(*variant))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(TypeBase::new_sum(variants))
            }

            // The following terms are not runtime types, but the core `Type` only contains runtime types.
            // We therefore report a type error here.
            model::Term::ListType { .. }
            | model::Term::StrType
            | model::Term::NatType
            | model::Term::ExtSetType
            | model::Term::StaticType
            | model::Term::Type
            | model::Term::Constraint
            | model::Term::Const { .. }
            | model::Term::Str(_)
            | model::Term::ExtSet { .. }
            | model::Term::List { .. }
            | model::Term::Control { .. }
            | model::Term::ControlType
            | model::Term::Nat(_)
            | model::Term::NonLinearConstraint { .. }
            | model::Term::Bytes { .. }
            | model::Term::BytesType
            | model::Term::FloatType
            | model::Term::Float { .. }
            | model::Term::ConstFunc { .. }
            | model::Term::Meta
            | model::Term::ConstAdt { .. } => Err(model::ModelError::TypeError(term_id).into()),
        }
    }

    fn get_func_type(
        &mut self,
        term_id: model::TermId,
    ) -> Result<(model::TermId, model::TermId, model::TermId), ImportError> {
        match self.get_term(term_id)? {
            model::Term::FuncType {
                inputs,
                outputs,
                extensions,
            } => Ok((*inputs, *outputs, *extensions)),
            _ => Err(model::ModelError::TypeError(term_id).into()),
        }
    }

    fn import_func_type<RV: MaybeRV>(
        &mut self,
        term_id: model::TermId,
    ) -> Result<FuncTypeBase<RV>, ImportError> {
        let (inputs, outputs, extensions) = self.get_func_type(term_id)?;
        let inputs = self.import_type_row(inputs)?;
        let outputs = self.import_type_row(outputs)?;
        let extensions = self.import_extension_set(extensions)?;
        Ok(FuncTypeBase::new(inputs, outputs).with_extension_delta(extensions))
    }

    fn import_closed_list(
        &mut self,
        term_id: model::TermId,
    ) -> Result<Vec<model::TermId>, ImportError> {
        fn import_into(
            ctx: &mut Context,
            term_id: model::TermId,
            types: &mut Vec<model::TermId>,
        ) -> Result<(), ImportError> {
            match ctx.get_term(term_id)? {
                model::Term::List { parts } => {
                    types.reserve(parts.len());

                    for part in *parts {
                        match part {
                            model::ListPart::Item(term_id) => {
                                types.push(*term_id);
                            }
                            model::ListPart::Splice(term_id) => {
                                import_into(ctx, *term_id, types)?;
                            }
                        }
                    }
                }
                _ => return Err(model::ModelError::TypeError(term_id).into()),
            }

            Ok(())
        }

        let mut types = Vec::new();
        import_into(self, term_id, &mut types)?;
        Ok(types)
    }

    fn import_type_rows<RV: MaybeRV>(
        &mut self,
        term_id: model::TermId,
    ) -> Result<Vec<TypeRowBase<RV>>, ImportError> {
        self.import_closed_list(term_id)?
            .into_iter()
            .map(|term_id| self.import_type_row::<RV>(term_id))
            .collect()
    }

    fn import_type_row<RV: MaybeRV>(
        &mut self,
        term_id: model::TermId,
    ) -> Result<TypeRowBase<RV>, ImportError> {
        fn import_into<RV: MaybeRV>(
            ctx: &mut Context,
            term_id: model::TermId,
            types: &mut Vec<TypeBase<RV>>,
        ) -> Result<(), ImportError> {
            match ctx.get_term(term_id)? {
                model::Term::List { parts } => {
                    types.reserve(parts.len());

                    for item in *parts {
                        match item {
                            model::ListPart::Item(term_id) => {
                                types.push(ctx.import_type::<RV>(*term_id)?);
                            }
                            model::ListPart::Splice(term_id) => {
                                import_into(ctx, *term_id, types)?;
                            }
                        }
                    }
                }
                model::Term::Var(model::VarId(_, index)) => {
                    let var = RV::try_from_rv(RowVariable(*index as _, TypeBound::Any))
                        .map_err(|_| model::ModelError::TypeError(term_id))?;
                    types.push(TypeBase::new(TypeEnum::RowVar(var)));
                }
                _ => return Err(model::ModelError::TypeError(term_id).into()),
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
                    .map_err(|_| model::ModelError::MalformedName(symbol.to_smolstr()))?;

                let (extension, id) = qualified_name
                    .split_last()
                    .ok_or_else(|| model::ModelError::MalformedName(symbol.to_smolstr()))?;

                vacant_entry.insert((extension.clone(), id.clone()));
                Ok((extension, id))
            }
        }
    }

    fn import_json_meta(
        &mut self,
        term_id: model::TermId,
    ) -> Result<(&'a str, serde_json::Value), ImportError> {
        let (global, args) = match self.get_term(term_id)? {
            model::Term::Apply { symbol, args } | model::Term::ApplyFull { symbol, args } => {
                (symbol, args)
            }
            _ => return Err(model::ModelError::TypeError(term_id).into()),
        };

        let global = self.get_symbol_name(*global)?;
        if global != model::COMPAT_META_JSON {
            return Err(model::ModelError::TypeError(term_id).into());
        }

        let [name_arg, json_arg] = args else {
            return Err(model::ModelError::TypeError(term_id).into());
        };

        let model::Term::Str(name) = self.get_term(*name_arg)? else {
            return Err(model::ModelError::TypeError(term_id).into());
        };

        let model::Term::Str(json_str) = self.get_term(*json_arg)? else {
            return Err(model::ModelError::TypeError(term_id).into());
        };

        let json_value =
            serde_json::from_str(json_str).map_err(|_| model::ModelError::TypeError(term_id))?;

        Ok((name, json_value))
    }

    fn import_value(
        &mut self,
        term_id: model::TermId,
        type_id: model::TermId,
    ) -> Result<Value, ImportError> {
        let term_data = self.get_term(term_id)?;

        match term_data {
            model::Term::Wildcard => Err(error_uninferred!("wildcard")),
            model::Term::Apply { .. } => {
                Err(error_uninferred!("application with implicit parameters"))
            }
            model::Term::Var(_) => Err(error_unsupported!("constant value containing a variable")),

            model::Term::ApplyFull { symbol, args } => {
                let symbol_name = self.get_symbol_name(*symbol)?;

                if symbol_name == model::COMPAT_CONST_JSON {
                    let value = args.get(1).ok_or(model::ModelError::TypeError(term_id))?;

                    let model::Term::Str(json) = self.get_term(*value)? else {
                        return Err(model::ModelError::TypeError(term_id).into());
                    };

                    // We attempt to deserialize as the custom const directly.
                    // This might fail due to the custom const struct not being included when
                    // this code was compiled; in that case, we fall back to the serialized form.
                    let value: Option<Box<dyn CustomConst>> = serde_json::from_str(json).ok();

                    if let Some(value) = value {
                        let opaque_value = OpaqueValue::from(value);
                        return Ok(Value::Extension { e: opaque_value });
                    } else {
                        let runtime_type =
                            args.first().ok_or(model::ModelError::TypeError(term_id))?;
                        let runtime_type = self.import_type(*runtime_type)?;

                        let extensions =
                            args.get(2).ok_or(model::ModelError::TypeError(term_id))?;
                        let extensions = self.import_extension_set(*extensions)?;

                        let value: serde_json::Value = serde_json::from_str(json)
                            .map_err(|_| model::ModelError::TypeError(term_id))?;
                        let custom_const = CustomSerialized::new(runtime_type, value, extensions);
                        let opaque_value = OpaqueValue::new(custom_const);
                        return Ok(Value::Extension { e: opaque_value });
                    }
                }

                // NOTE: We have special cased arrays, integers, and floats for now.
                // TODO: Allow arbitrary extension values to be imported from terms.

                if symbol_name == ArrayValue::CTR_NAME {
                    let element_type_term =
                        args.get(1).ok_or(model::ModelError::TypeError(term_id))?;
                    let element_type = self.import_type(*element_type_term)?;

                    let contents = {
                        let contents = args.get(2).ok_or(model::ModelError::TypeError(term_id))?;
                        let contents = self.import_closed_list(*contents)?;
                        contents
                            .iter()
                            .map(|item| self.import_value(*item, *element_type_term))
                            .collect::<Result<Vec<_>, _>>()?
                    };

                    return Ok(ArrayValue::new(element_type, contents).into());
                }

                if symbol_name == ConstInt::CTR_NAME {
                    let bitwidth = {
                        let bitwidth = args.first().ok_or(model::ModelError::TypeError(term_id))?;
                        let model::Term::Nat(bitwidth) = self.get_term(*bitwidth)? else {
                            return Err(model::ModelError::TypeError(term_id).into());
                        };
                        if *bitwidth > 6 {
                            return Err(model::ModelError::TypeError(term_id).into());
                        }
                        *bitwidth as u8
                    };

                    let value = {
                        let value = args.get(1).ok_or(model::ModelError::TypeError(term_id))?;
                        let model::Term::Nat(value) = self.get_term(*value)? else {
                            return Err(model::ModelError::TypeError(term_id).into());
                        };
                        *value
                    };

                    return Ok(ConstInt::new_u(bitwidth, value)
                        .map_err(|_| model::ModelError::TypeError(term_id))?
                        .into());
                }

                if symbol_name == ConstF64::CTR_NAME {
                    let value = {
                        let value = args.first().ok_or(model::ModelError::TypeError(term_id))?;
                        let model::Term::Float { value } = self.get_term(*value)? else {
                            return Err(model::ModelError::TypeError(term_id).into());
                        };
                        value.into_inner()
                    };

                    return Ok(ConstF64::new(value).into());
                }

                Err(error_unsupported!("unknown custom constant value"))
                // TODO: This should ultimately include the following cases:
                // - function definitions
                // - custom constructors for values
            }

            model::Term::StaticType
            | model::Term::Constraint
            | model::Term::Const { .. }
            | model::Term::List { .. }
            | model::Term::ListType { .. }
            | model::Term::Str(_)
            | model::Term::StrType
            | model::Term::Nat(_)
            | model::Term::NatType
            | model::Term::ExtSet { .. }
            | model::Term::ExtSetType
            | model::Term::Adt { .. }
            | model::Term::FuncType { .. }
            | model::Term::Control { .. }
            | model::Term::ControlType
            | model::Term::Type
            | model::Term::Bytes { .. }
            | model::Term::BytesType
            | model::Term::Meta
            | model::Term::Float { .. }
            | model::Term::FloatType
            | model::Term::NonLinearConstraint { .. } => {
                Err(model::ModelError::TypeError(term_id).into())
            }

            model::Term::ConstFunc { .. } => Err(error_unsupported!("constant function value")),

            model::Term::ConstAdt { tag, values } => {
                let model::Term::Adt { variants } = self.get_term(type_id)? else {
                    return Err(model::ModelError::TypeError(term_id).into());
                };

                let values = self.import_closed_list(*values)?;
                let variants = self.import_closed_list(*variants)?;

                let variant = variants
                    .get(*tag as usize)
                    .ok_or(model::ModelError::TypeError(term_id))?;
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

                Ok(Value::sum(*tag as _, items, typ).unwrap())
            }
        }
    }
}

/// Information about a local variable.
#[derive(Debug, Clone, Copy)]
struct LocalVar {
    /// The type of the variable.
    r#type: model::TermId,
    /// The type bound of the variable.
    bound: TypeBound,
}

impl LocalVar {
    pub fn new(r#type: model::TermId) -> Self {
        Self {
            r#type,
            bound: TypeBound::Any,
        }
    }
}
