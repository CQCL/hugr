//! Importing HUGR graphs from their `hugr-model` representation.
//!
//! **Warning**: This module is still under development and is expected to change.
//! It is included in the library to allow for early experimentation, and for
//! the core and model to converge incrementally.
use crate::{
    export::OP_FUNC_CALL_INDIRECT,
    extension::{ExtensionId, ExtensionRegistry, ExtensionSet, SignatureError},
    hugr::{HugrMut, IdentList},
    ops::{
        AliasDecl, AliasDefn, Call, CallIndirect, Case, Conditional, DataflowBlock, ExitBlock,
        FuncDecl, FuncDefn, Input, LoadFunction, Module, OpType, OpaqueOp, Output, Tag, TailLoop,
        CFG, DFG,
    },
    types::{
        type_param::TypeParam, type_row::TypeRowBase, CustomType, FuncTypeBase, MaybeRV, NoRV,
        PolyFuncType, PolyFuncTypeBase, RowVariable, Signature, Type, TypeArg, TypeBase, TypeBound,
        TypeRow,
    },
    Direction, Hugr, HugrView, Node, Port,
};
use fxhash::FxHashMap;
use hugr_model::v0::{self as model, GlobalRef};
use indexmap::IndexMap;
use itertools::Either;
use smol_str::{SmolStr, ToSmolStr};
use thiserror::Error;

type FxIndexMap<K, V> = IndexMap<K, V, fxhash::FxBuildHasher>;

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
    let names = Names::new(module)?;

    // TODO: Module should know about the number of edges, so that we can use a vector here.
    // For now we use a hashmap, which will be slower.
    let edge_ports = FxHashMap::default();

    let mut ctx = Context {
        module,
        names,
        hugr: Hugr::new(OpType::Module(Module {})),
        link_ports: edge_ports,
        static_edges: Vec::new(),
        extensions,
        nodes: FxHashMap::default(),
        local_variables: IndexMap::default(),
        custom_name_cache: FxHashMap::default(),
    };

    ctx.import_root()?;
    ctx.link_ports()?;
    ctx.link_static_ports()?;

    Ok(ctx.hugr)
}

struct Context<'a> {
    /// The module being imported.
    module: &'a model::Module<'a>,

    names: Names<'a>,

    /// The HUGR graph being constructed.
    hugr: Hugr,

    /// The ports that are part of each link. This is used to connect the ports at the end of the
    /// import process.
    link_ports: FxHashMap<model::LinkRef<'a>, Vec<(Node, Port)>>,

    /// Pairs of nodes that should be connected by a static edge.
    /// These are collected during the import process and connected at the end.
    static_edges: Vec<(model::NodeId, model::NodeId)>,

    // /// The `(Node, Port)` pairs for each `PortId` in the module.
    // imported_ports: Vec<Option<(Node, Port)>>,
    /// The ambient extension registry to use for importing.
    extensions: &'a ExtensionRegistry,

    /// A map from `NodeId` to the imported `Node`.
    nodes: FxHashMap<model::NodeId, Node>,

    /// The types of the local variables that are currently in scope.
    local_variables: FxIndexMap<&'a str, model::TermId>,

    custom_name_cache: FxHashMap<&'a str, (ExtensionId, SmolStr)>,
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

    /// Looks up a [`LocalRef`] within the current scope and returns its index and type.
    fn resolve_local_ref(
        &self,
        local_ref: &model::LocalRef,
    ) -> Result<(usize, model::TermId), ImportError> {
        let term = match local_ref {
            model::LocalRef::Index(_, index) => self
                .local_variables
                .get_index(*index as usize)
                .map(|(_, term)| (*index as usize, *term)),
            model::LocalRef::Named(name) => self
                .local_variables
                .get_full(name)
                .map(|(index, _, term)| (index, *term)),
        };

        term.ok_or_else(|| model::ModelError::InvalidLocal(local_ref.to_string()).into())
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
        Ok(node)
    }

    /// Associate links with the ports of the given node in the given direction.
    fn record_links(&mut self, node: Node, direction: Direction, links: &'a [model::LinkRef<'a>]) {
        let optype = self.hugr.get_optype(node);
        // NOTE: `OpType::port_count` copies the signature, which significantly slows down the import.
        debug_assert!(links.len() <= optype.port_count(direction));

        for (link, port) in links.iter().zip(self.hugr.node_ports(node, direction)) {
            self.link_ports.entry(*link).or_default().push((node, port));
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

            if inputs.is_empty() || outputs.is_empty() {
                return Err(error_unsupported!(
                    "link {} is missing either an input or an output port",
                    link_id
                ));
            }

            // We connect the first output to all the inputs, and the first input to all the outputs
            // (except the first one, which we already connected to the first input). This should
            // result in the hugr having a (hyper)edge that connects all the ports.
            // There should be a better way to do this.
            for (node, port) in inputs.iter() {
                self.hugr.connect(outputs[0].0, outputs[0].1, *node, *port);
            }

            for (node, port) in outputs.iter().skip(1) {
                self.hugr.connect(*node, *port, inputs[0].0, inputs[0].1);
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

    fn with_local_socpe<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, ImportError>,
    ) -> Result<T, ImportError> {
        let previous = std::mem::take(&mut self.local_variables);
        let result = f(self);
        self.local_variables = previous;
        result
    }

    fn resolve_global_ref(
        &self,
        global_ref: &model::GlobalRef,
    ) -> Result<model::NodeId, ImportError> {
        match global_ref {
            model::GlobalRef::Direct(node_id) => Ok(*node_id),
            model::GlobalRef::Named(name) => {
                let item = self
                    .names
                    .items
                    .get(name)
                    .ok_or_else(|| model::ModelError::InvalidGlobal(global_ref.to_string()))?;

                match item {
                    NamedItem::FuncDecl(node) => Ok(*node),
                    NamedItem::FuncDefn(node) => Ok(*node),
                    NamedItem::CtrDecl(node) => Ok(*node),
                    NamedItem::OperationDecl(node) => Ok(*node),
                }
            }
        }
    }

    fn get_global_name(&self, global_ref: model::GlobalRef<'a>) -> Result<&'a str, ImportError> {
        match global_ref {
            model::GlobalRef::Direct(node_id) => {
                let node_data = self.get_node(node_id)?;

                let name = match node_data.operation {
                    model::Operation::DefineFunc { decl } => decl.name,
                    model::Operation::DeclareFunc { decl } => decl.name,
                    model::Operation::DefineAlias { decl, .. } => decl.name,
                    model::Operation::DeclareAlias { decl } => decl.name,
                    model::Operation::DeclareConstructor { decl } => decl.name,
                    model::Operation::DeclareOperation { decl } => decl.name,
                    _ => {
                        return Err(model::ModelError::InvalidGlobal(global_ref.to_string()).into());
                    }
                };

                Ok(name)
            }
            model::GlobalRef::Named(name) => Ok(name),
        }
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

        self.import_poly_func_type(*decl, |_, signature| Ok(signature))
    }

    /// Import the root region of the module.
    fn import_root(&mut self) -> Result<(), ImportError> {
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
                self.import_poly_func_type(*decl, |ctx, signature| {
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
                self.import_poly_func_type(*decl, |ctx, signature| {
                    let optype = OpType::FuncDecl(FuncDecl {
                        name: decl.name.to_string(),
                        signature,
                    });

                    let node = ctx.make_node(node_id, optype, parent)?;

                    Ok(Some(node))
                })
            }

            model::Operation::CallFunc { func } => {
                let model::Term::ApplyFull { global: name, args } = self.get_term(func)? else {
                    return Err(model::ModelError::TypeError(func).into());
                };

                let func_node = self.resolve_global_ref(name)?;
                let func_sig = self.get_func_signature(func_node)?;

                let type_args = args
                    .iter()
                    .map(|term| self.import_type_arg(*term))
                    .collect::<Result<Vec<TypeArg>, _>>()?;

                self.static_edges.push((func_node, node_id));
                let optype = OpType::Call(Call::try_new(func_sig, type_args, self.extensions)?);

                let node = self.make_node(node_id, optype, parent)?;
                Ok(Some(node))
            }

            model::Operation::LoadFunc { func } => {
                let model::Term::ApplyFull { global: name, args } = self.get_term(func)? else {
                    return Err(model::ModelError::TypeError(func).into());
                };

                let func_node = self.resolve_global_ref(name)?;
                let func_sig = self.get_func_signature(func_node)?;

                let type_args = args
                    .iter()
                    .map(|term| self.import_type_arg(*term))
                    .collect::<Result<Vec<TypeArg>, _>>()?;

                self.static_edges.push((func_node, node_id));

                let optype = OpType::LoadFunction(LoadFunction::try_new(
                    func_sig,
                    type_args,
                    self.extensions,
                )?);

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

            model::Operation::CustomFull {
                operation: GlobalRef::Named(name),
            } if name == OP_FUNC_CALL_INDIRECT => {
                let signature = self.get_node_signature(node_id)?;
                let optype = OpType::CallIndirect(CallIndirect { signature });
                let node = self.make_node(node_id, optype, parent)?;
                Ok(Some(node))
            }

            model::Operation::CustomFull { operation } => {
                let signature = self.get_node_signature(node_id)?;
                let args = node_data
                    .params
                    .iter()
                    .map(|param| self.import_type_arg(*param))
                    .collect::<Result<Vec<_>, _>>()?;

                let name = self.get_global_name(operation)?;
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

                match node_data.regions {
                    [] => {}
                    [region] => self.import_dfg_region(node_id, *region, node)?,
                    _ => return Err(error_unsupported!("multiple regions in custom operation")),
                }

                Ok(Some(node))
            }

            model::Operation::Custom { .. } => Err(error_unsupported!(
                "custom operation with implicit parameters"
            )),

            model::Operation::DefineAlias { decl, value } => self.with_local_socpe(|ctx| {
                if !decl.params.is_empty() {
                    return Err(error_unsupported!(
                        "parameters or constraints in alias definition"
                    ));
                }

                let optype = OpType::AliasDefn(AliasDefn {
                    name: decl.name.to_smolstr(),
                    definition: ctx.import_type(value)?,
                });

                let node = ctx.make_node(node_id, optype, parent)?;
                Ok(Some(node))
            }),

            model::Operation::DeclareAlias { decl } => self.with_local_socpe(|ctx| {
                if !decl.params.is_empty() {
                    return Err(error_unsupported!(
                        "parameters or constraints in alias declaration"
                    ));
                }

                let optype = OpType::AliasDecl(AliasDecl {
                    name: decl.name.to_smolstr(),
                    bound: TypeBound::Copyable,
                });

                let node = ctx.make_node(node_id, optype, parent)?;
                Ok(Some(node))
            }),

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

            model::Operation::DeclareConstructor { .. } => Ok(None),
            model::Operation::DeclareOperation { .. } => Ok(None),
        }
    }

    fn import_dfg_region(
        &mut self,
        node_id: model::NodeId,
        region: model::RegionId,
        node: Node,
    ) -> Result<(), ImportError> {
        let region_data = self.get_region(region)?;

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

            for (a, b) in input_outputs.zip(tag_inputs) {
                self.hugr.connect(node_input, a, node_tag, b);
            }

            // Connect the tag node to the output node
            let tag_outputs = self.hugr.node_outputs(node_tag);
            let output_inputs = self.hugr.node_inputs(node_output);

            for (a, b) in tag_outputs.zip(output_inputs) {
                self.hugr.connect(node_tag, a, node_output, b);
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
        decl: model::FuncDecl<'a>,
        in_scope: impl FnOnce(&mut Self, PolyFuncTypeBase<RV>) -> Result<T, ImportError>,
    ) -> Result<T, ImportError> {
        self.with_local_socpe(|ctx| {
            let mut imported_params = Vec::with_capacity(decl.params.len());

            for param in decl.params {
                // TODO: `PolyFuncType` should be able to handle constraints
                // and distinguish between implicit and explicit parameters.
                match param {
                    model::Param::Implicit { name, r#type } => {
                        imported_params.push(ctx.import_type_param(*r#type)?);
                        ctx.local_variables.insert(name, *r#type);
                    }
                    model::Param::Explicit { name, r#type } => {
                        imported_params.push(ctx.import_type_param(*r#type)?);
                        ctx.local_variables.insert(name, *r#type);
                    }
                    model::Param::Constraint { constraint: _ } => {
                        return Err(error_unsupported!("constraints"));
                    }
                }
            }

            let body = ctx.import_func_type::<RV>(decl.signature)?;
            in_scope(ctx, PolyFuncTypeBase::new(imported_params, body))
        })
    }

    /// Import a [`TypeParam`] from a term that represents a static type.
    fn import_type_param(&mut self, term_id: model::TermId) -> Result<TypeParam, ImportError> {
        match self.get_term(term_id)? {
            model::Term::Wildcard => Err(error_uninferred!("wildcard")),

            model::Term::Type => {
                // As part of the migration from `TypeBound`s to constraints, we pretend that all
                // `TypeBound`s are copyable.
                Ok(TypeParam::Type {
                    b: TypeBound::Copyable,
                })
            }

            model::Term::StaticType => Err(error_unsupported!("`type` as `TypeParam`")),
            model::Term::Constraint => Err(error_unsupported!("`constraint` as `TypeParam`")),
            model::Term::Var(_) => Err(error_unsupported!("type variable as `TypeParam`")),
            model::Term::Apply { .. } => Err(error_unsupported!("custom type as `TypeParam`")),
            model::Term::ApplyFull { .. } => Err(error_unsupported!("custom type as `TypeParam`")),

            model::Term::Quote { .. } => Err(error_unsupported!("`(quote ...)` as `TypeParam`")),
            model::Term::FuncType { .. } => Err(error_unsupported!("`(fn ...)` as `TypeParam`")),

            model::Term::ListType { item_type } => {
                let param = Box::new(self.import_type_param(*item_type)?);
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
            | model::Term::Control { .. } => Err(model::ModelError::TypeError(term_id).into()),

            model::Term::ControlType => {
                Err(error_unsupported!("type of control types as `TypeParam`"))
            }
        }
    }

    /// Import a `TypeArg` froma term that represents a static type or value.
    fn import_type_arg(&mut self, term_id: model::TermId) -> Result<TypeArg, ImportError> {
        match self.get_term(term_id)? {
            model::Term::Wildcard => Err(error_uninferred!("wildcard")),
            model::Term::Apply { .. } => {
                Err(error_uninferred!("application with implicit parameters"))
            }

            model::Term::Var(var) => {
                let (index, var_type) = self.resolve_local_ref(var)?;
                let decl = self.import_type_param(var_type)?;
                Ok(TypeArg::new_var_use(index, decl))
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

            model::Term::Quote { .. } => Ok(TypeArg::Type {
                ty: self.import_type(term_id)?,
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
            model::Term::ApplyFull { .. } => Err(error_unsupported!("custom types as `TypeArg`")),
            model::Term::Constraint => Err(error_unsupported!("`constraint` as `TypeArg`")),
            model::Term::StaticType => Err(error_unsupported!("`static` as `TypeArg`")),
            model::Term::ControlType => Err(error_unsupported!("`ctrl` as `TypeArg`")),

            model::Term::FuncType { .. }
            | model::Term::Adt { .. }
            | model::Term::Control { .. } => Err(model::ModelError::TypeError(term_id).into()),
        }
    }

    fn import_extension_set(
        &mut self,
        term_id: model::TermId,
    ) -> Result<ExtensionSet, ImportError> {
        match self.get_term(term_id)? {
            model::Term::Wildcard => Err(error_uninferred!("wildcard")),

            model::Term::Var(var) => {
                let mut es = ExtensionSet::new();
                let (index, _) = self.resolve_local_ref(var)?;
                es.insert_type_var(index);
                Ok(es)
            }

            model::Term::ExtSet { extensions, rest } => {
                let mut es = match rest {
                    Some(rest) => self.import_extension_set(*rest)?,
                    None => ExtensionSet::new(),
                };

                for ext in extensions.iter() {
                    let ext_ident = IdentList::new(*ext)
                        .map_err(|_| model::ModelError::MalformedName(ext.to_smolstr()))?;
                    es.insert(&ext_ident);
                }

                Ok(es)
            }
            _ => Err(model::ModelError::TypeError(term_id).into()),
        }
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

            model::Term::ApplyFull { global: name, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.import_type_arg(*arg))
                    .collect::<Result<Vec<_>, _>>()?;

                let name = self.get_global_name(*name)?;
                let (extension, id) = self.import_custom_name(name)?;

                Ok(TypeBase::new_extension(CustomType::new(
                    id,
                    args,
                    extension,
                    // As part of the migration from `TypeBound`s to constraints, we pretend that all
                    // `TypeBound`s are copyable.
                    TypeBound::Copyable,
                )))
            }

            model::Term::Var(var) => {
                // We pretend that all `TypeBound`s are copyable.
                let (index, _) = self.resolve_local_ref(var)?;
                Ok(TypeBase::new_var_use(index, TypeBound::Copyable))
            }

            model::Term::FuncType { .. } => {
                let func_type = self.import_func_type::<NoRV>(term_id)?;
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
            | model::Term::Quote { .. }
            | model::Term::Str(_)
            | model::Term::ExtSet { .. }
            | model::Term::List { .. }
            | model::Term::Control { .. }
            | model::Term::ControlType
            | model::Term::Nat(_) => Err(model::ModelError::TypeError(term_id).into()),
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
        let inputs = self.import_type_row::<RV>(inputs)?;
        let outputs = self.import_type_row::<RV>(outputs)?;
        let extensions = self.import_extension_set(extensions)?;
        Ok(FuncTypeBase::new(inputs, outputs).with_extension_delta(extensions))
    }

    fn import_closed_list(
        &mut self,
        mut term_id: model::TermId,
    ) -> Result<Vec<model::TermId>, ImportError> {
        // PERFORMANCE: We currently allocate a Vec here to collect list items
        // into, in order to handle the case where the tail of the list is another
        // list. We should avoid this.
        let mut list_items = Vec::new();

        loop {
            match self.get_term(term_id)? {
                model::Term::Var(_) => return Err(error_unsupported!("open lists")),
                model::Term::List { items, tail } => {
                    list_items.extend(items.iter());

                    match tail {
                        Some(tail) => term_id = *tail,
                        None => break,
                    }
                }
                _ => {
                    return Err(model::ModelError::TypeError(term_id).into());
                }
            }
        }

        Ok(list_items)
    }

    fn import_type_rows<RV: MaybeRV>(
        &mut self,
        term_id: model::TermId,
    ) -> Result<Vec<TypeRowBase<RV>>, ImportError> {
        self.import_closed_list(term_id)?
            .iter()
            .map(|row| self.import_type_row::<RV>(*row))
            .collect()
    }

    fn import_type_row<RV: MaybeRV>(
        &mut self,
        term_id: model::TermId,
    ) -> Result<TypeRowBase<RV>, ImportError> {
        let items = self
            .import_closed_list(term_id)?
            .iter()
            .map(|item| self.import_type(*item))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(items.into())
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum NamedItem {
    FuncDecl(model::NodeId),
    FuncDefn(model::NodeId),
    CtrDecl(model::NodeId),
    OperationDecl(model::NodeId),
}

struct Names<'a> {
    items: FxHashMap<&'a str, NamedItem>,
}

impl<'a> Names<'a> {
    pub fn new(module: &model::Module<'a>) -> Result<Self, ImportError> {
        let mut items = FxHashMap::default();

        for (node_id, node_data) in module.nodes.iter().enumerate() {
            let node_id = model::NodeId(node_id as _);

            let item = match node_data.operation {
                model::Operation::DefineFunc { decl } => {
                    Some((decl.name, NamedItem::FuncDecl(node_id)))
                }
                model::Operation::DeclareFunc { decl } => {
                    Some((decl.name, NamedItem::FuncDefn(node_id)))
                }
                model::Operation::DeclareConstructor { decl } => {
                    Some((decl.name, NamedItem::CtrDecl(node_id)))
                }
                model::Operation::DeclareOperation { decl } => {
                    Some((decl.name, NamedItem::OperationDecl(node_id)))
                }
                _ => None,
            };

            if let Some((name, item)) = item {
                // TODO: Deal with duplicates
                items.insert(name, item);
            }
        }

        Ok(Self { items })
    }
}
