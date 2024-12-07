//! Exporting HUGR graphs to their `hugr-model` representation.
use crate::{
    extension::{ExtensionId, ExtensionSet, OpDef, SignatureFunc},
    hugr::{IdentList, NodeMetadataMap},
    ops::{DataflowBlock, OpName, OpTrait, OpType},
    types::{
        type_param::{TypeArgVariable, TypeParam},
        type_row::TypeRowBase,
        CustomType, FuncTypeBase, MaybeRV, PolyFuncTypeBase, RowVariable, SumType, TypeArg,
        TypeBase, TypeBound, TypeEnum,
    },
    Direction, Hugr, HugrView, IncomingPort, Node, Port,
};
use bumpalo::{collections::String as BumpString, collections::Vec as BumpVec, Bump};
use fxhash::FxHashMap;
use hugr_model::v0::{self as model};
use indexmap::IndexSet;
use std::fmt::Write;

type FxIndexSet<T> = IndexSet<T, fxhash::FxBuildHasher>;

pub(crate) const OP_FUNC_CALL_INDIRECT: &str = "func.call-indirect";
const TERM_PARAM_TUPLE: &str = "param.tuple";
const TERM_JSON: &str = "prelude.json";
const META_DESCRIPTION: &str = "docs.description";

/// Export a [`Hugr`] graph to its representation in the model.
pub fn export_hugr<'a>(hugr: &'a Hugr, bump: &'a Bump) -> model::Module<'a> {
    let mut ctx = Context::new(hugr, bump);
    ctx.export_root();
    ctx.module
}

/// State for converting a HUGR graph to its representation in the model.
struct Context<'a> {
    /// The HUGR graph to convert.
    hugr: &'a Hugr,
    /// The module that is being built.
    module: model::Module<'a>,
    /// Mapping from ports to link indices.
    /// This only includes the minimum port among groups of linked ports.
    links: FxIndexSet<(Node, Port)>,
    /// The arena in which the model is allocated.
    bump: &'a Bump,
    /// Stores the terms that we have already seen to avoid duplicates.
    term_map: FxHashMap<model::Term<'a>, model::TermId>,

    /// The current scope for local variables.
    ///
    /// This is set to the id of the smallest enclosing node that defines a polymorphic type.
    /// We use this when exporting local variables in terms.
    local_scope: Option<model::NodeId>,

    /// Constraints to be added to the local scope.
    ///
    /// When exporting a node that defines a polymorphic type, we use this field
    /// to collect the constraints that need to be added to that polymorphic
    /// type. Currently this is used to record `nonlinear` constraints on uses
    /// of `TypeParam::Type` with a `TypeBound::Copyable` bound.
    local_constraints: Vec<model::TermId>,

    /// Mapping from extension operations to their declarations.
    decl_operations: FxHashMap<(ExtensionId, OpName), model::NodeId>,
}

impl<'a> Context<'a> {
    pub fn new(hugr: &'a Hugr, bump: &'a Bump) -> Self {
        let mut module = model::Module::default();
        module.nodes.reserve(hugr.node_count());

        Self {
            hugr,
            module,
            bump,
            links: IndexSet::default(),
            term_map: FxHashMap::default(),
            local_scope: None,
            decl_operations: FxHashMap::default(),
            local_constraints: Vec::new(),
        }
    }

    /// Exports the root module of the HUGR graph.
    pub fn export_root(&mut self) {
        let hugr_children = self.hugr.children(self.hugr.root());
        let mut children = Vec::with_capacity(hugr_children.size_hint().0);

        for child in self.hugr.children(self.hugr.root()) {
            children.push(self.export_node(child));
        }

        children.extend(self.decl_operations.values().copied());

        let root = self.module.insert_region(model::Region {
            kind: model::RegionKind::Module,
            sources: &[],
            targets: &[],
            children: self.bump.alloc_slice_copy(&children),
            meta: &[], // TODO: Export metadata
            signature: None,
        });

        self.module.root = root;
    }

    /// Returns the edge id for a given port, creating a new edge if necessary.
    ///
    /// Any two ports that are linked will be represented by the same link.
    fn get_link_id(&mut self, node: Node, port: impl Into<Port>) -> model::LinkId {
        // To ensure that linked ports are represented by the same edge, we take the minimum port
        // among all the linked ports, including the one we started with.
        let port = port.into();
        let linked_ports = self.hugr.linked_ports(node, port);
        let all_ports = std::iter::once((node, port)).chain(linked_ports);
        let repr = all_ports.min().unwrap();
        let edge = self.links.insert_full(repr).0 as _;
        model::LinkId(edge)
    }

    pub fn make_ports(
        &mut self,
        node: Node,
        direction: Direction,
        num_ports: usize,
    ) -> &'a [model::LinkRef<'a>] {
        let ports = self.hugr.node_ports(node, direction);
        let mut links = BumpVec::with_capacity_in(ports.size_hint().0, self.bump);

        for port in ports.take(num_ports) {
            links.push(model::LinkRef::Id(self.get_link_id(node, port)));
        }

        links.into_bump_slice()
    }

    pub fn make_term(&mut self, term: model::Term<'a>) -> model::TermId {
        // Wildcard terms do not all represent the same term, so we should not deduplicate them.
        if term == model::Term::Wildcard {
            return self.module.insert_term(term);
        }

        *self
            .term_map
            .entry(term.clone())
            .or_insert_with(|| self.module.insert_term(term))
    }

    pub fn make_qualified_name(
        &mut self,
        extension: &ExtensionId,
        name: impl AsRef<str>,
    ) -> &'a str {
        let capacity = extension.len() + name.as_ref().len() + 1;
        let mut output = BumpString::with_capacity_in(capacity, self.bump);
        let _ = write!(&mut output, "{}.{}", extension, name.as_ref());
        output.into_bump_str()
    }

    pub fn make_named_global_ref(
        &mut self,
        extension: &IdentList,
        name: impl AsRef<str>,
    ) -> model::GlobalRef<'a> {
        model::GlobalRef::Named(self.make_qualified_name(extension, name))
    }

    /// Get the node that declares or defines the function associated with the given
    /// node via the static input. Returns `None` if the node is not connected to a function.
    fn connected_function(&self, node: Node) -> Option<Node> {
        let func_node = self.hugr.static_source(node)?;

        match self.hugr.get_optype(func_node) {
            OpType::FuncDecl(_) => Some(func_node),
            OpType::FuncDefn(_) => Some(func_node),
            _ => None,
        }
    }

    /// Get the name of a function definition or declaration node. Returns `None` if not
    /// one of those operations.
    fn get_func_name(&self, func_node: Node) -> Option<&'a str> {
        match self.hugr.get_optype(func_node) {
            OpType::FuncDecl(func_decl) => Some(&func_decl.name),
            OpType::FuncDefn(func_defn) => Some(&func_defn.name),
            _ => None,
        }
    }

    fn with_local_scope<T>(&mut self, node: model::NodeId, f: impl FnOnce(&mut Self) -> T) -> T {
        let prev_local_scope = self.local_scope.replace(node);
        let prev_local_constraints = std::mem::take(&mut self.local_constraints);
        let result = f(self);
        self.local_scope = prev_local_scope;
        self.local_constraints = prev_local_constraints;
        result
    }

    pub fn export_node(&mut self, node: Node) -> model::NodeId {
        // We insert a dummy node with the invalid operation at this point to reserve
        // the node id. This is necessary to establish the correct node id for the
        // local scope introduced by some operations. We will overwrite this node later.
        let node_id = self.module.insert_node(model::Node::default());

        let mut params: &[_] = &[];
        let mut regions: &[_] = &[];

        let optype = self.hugr.get_optype(node);

        let operation = match optype {
            OpType::Module(_) => todo!("this should be an error"),

            OpType::Input(_) => {
                panic!("input nodes should have been handled by the region export")
            }

            OpType::Output(_) => {
                panic!("output nodes should have been handled by the region export")
            }

            OpType::DFG(dfg) => {
                let extensions = self.export_ext_set(&dfg.signature.extension_reqs);
                regions = self
                    .bump
                    .alloc_slice_copy(&[self.export_dfg(node, extensions)]);
                model::Operation::Dfg
            }

            OpType::CFG(_) => {
                regions = self.bump.alloc_slice_copy(&[self.export_cfg(node)]);
                model::Operation::Cfg
            }

            OpType::ExitBlock(_) => {
                panic!("exit blocks should have been handled by the region export")
            }

            OpType::Case(_) => {
                todo!("case nodes should have been handled by the region export")
            }

            OpType::DataflowBlock(block) => {
                let extensions = self.export_ext_set(&block.extension_delta);
                regions = self
                    .bump
                    .alloc_slice_copy(&[self.export_dfg(node, extensions)]);
                model::Operation::Block
            }

            OpType::FuncDefn(func) => self.with_local_scope(node_id, |this| {
                let name = this.get_func_name(node).unwrap();
                let (params, constraints, signature) = this.export_poly_func_type(&func.signature);
                let decl = this.bump.alloc(model::FuncDecl {
                    name,
                    params,
                    constraints,
                    signature,
                });
                let extensions = this.export_ext_set(&func.signature.body().extension_reqs);
                regions = this
                    .bump
                    .alloc_slice_copy(&[this.export_dfg(node, extensions)]);
                model::Operation::DefineFunc { decl }
            }),

            OpType::FuncDecl(func) => self.with_local_scope(node_id, |this| {
                let name = this.get_func_name(node).unwrap();
                let (params, constraints, func) = this.export_poly_func_type(&func.signature);
                let decl = this.bump.alloc(model::FuncDecl {
                    name,
                    params,
                    constraints,
                    signature: func,
                });
                model::Operation::DeclareFunc { decl }
            }),

            OpType::AliasDecl(alias) => self.with_local_scope(node_id, |this| {
                // TODO: We should support aliases with different types and with parameters
                let r#type = this.make_term(model::Term::Type);
                let decl = this.bump.alloc(model::AliasDecl {
                    name: &alias.name,
                    params: &[],
                    r#type,
                });
                model::Operation::DeclareAlias { decl }
            }),

            OpType::AliasDefn(alias) => self.with_local_scope(node_id, |this| {
                let value = this.export_type(&alias.definition);
                // TODO: We should support aliases with different types and with parameters
                let r#type = this.make_term(model::Term::Type);
                let decl = this.bump.alloc(model::AliasDecl {
                    name: &alias.name,
                    params: &[],
                    r#type,
                });
                model::Operation::DefineAlias { decl, value }
            }),

            OpType::Call(call) => {
                // TODO: If the node is not connected to a function, we should do better than panic.
                let node = self.connected_function(node).unwrap();
                let name = model::GlobalRef::Named(self.get_func_name(node).unwrap());

                let mut args = BumpVec::new_in(self.bump);
                args.extend(call.type_args.iter().map(|arg| self.export_type_arg(arg)));
                let args = args.into_bump_slice();

                let func = self.make_term(model::Term::ApplyFull { global: name, args });
                model::Operation::CallFunc { func }
            }

            OpType::LoadFunction(load) => {
                // TODO: If the node is not connected to a function, we should do better than panic.
                let node = self.connected_function(node).unwrap();
                let name = model::GlobalRef::Named(self.get_func_name(node).unwrap());

                let mut args = BumpVec::new_in(self.bump);
                args.extend(load.type_args.iter().map(|arg| self.export_type_arg(arg)));
                let args = args.into_bump_slice();

                let func = self.make_term(model::Term::ApplyFull { global: name, args });
                model::Operation::LoadFunc { func }
            }

            OpType::Const(_) => todo!("Export const nodes?"),
            OpType::LoadConstant(_) => todo!("Export load constant?"),

            OpType::CallIndirect(_) => model::Operation::CustomFull {
                operation: model::GlobalRef::Named(OP_FUNC_CALL_INDIRECT),
            },

            OpType::Tag(tag) => model::Operation::Tag { tag: tag.tag as _ },

            OpType::TailLoop(tail_loop) => {
                let extensions = self.export_ext_set(&tail_loop.extension_delta);
                regions = self
                    .bump
                    .alloc_slice_copy(&[self.export_dfg(node, extensions)]);
                model::Operation::TailLoop
            }

            OpType::Conditional(_) => {
                regions = self.export_conditional_regions(node);
                model::Operation::Conditional
            }

            // Opaque/extension operations should in the future support having multiple different
            // regions of potentially different kinds. At the moment, we check if the node has any
            // children, in which case we create a dataflow region with those children.
            OpType::ExtensionOp(op) => {
                let operation = self.export_opdef(op.def());

                params = self
                    .bump
                    .alloc_slice_fill_iter(op.args().iter().map(|arg| self.export_type_arg(arg)));

                // PERFORMANCE: Currently the API does not appear to allow to get the extension
                // set without copying it.
                // NOTE: We assume here that the extension set of the dfg region must be the same
                // as that of the node. This might change in the future.
                let extensions = self.export_ext_set(&op.extension_delta());

                if let Some(region) = self.export_dfg_if_present(node, extensions) {
                    regions = self.bump.alloc_slice_copy(&[region]);
                }

                model::Operation::CustomFull { operation }
            }

            OpType::OpaqueOp(op) => {
                let operation = self.make_named_global_ref(op.extension(), op.op_name());

                params = self
                    .bump
                    .alloc_slice_fill_iter(op.args().iter().map(|arg| self.export_type_arg(arg)));

                // PERFORMANCE: Currently the API does not appear to allow to get the extension
                // set without copying it.
                // NOTE: We assume here that the extension set of the dfg region must be the same
                // as that of the node. This might change in the future.
                let extensions = self.export_ext_set(&op.extension_delta());

                if let Some(region) = self.export_dfg_if_present(node, extensions) {
                    regions = self.bump.alloc_slice_copy(&[region]);
                }

                model::Operation::CustomFull { operation }
            }
        };

        let (signature, num_inputs, num_outputs) = match optype {
            OpType::DataflowBlock(block) => {
                let signature = self.export_block_signature(block);
                (Some(signature), 1, block.sum_rows.len())
            }

            // PERFORMANCE: As it stands, `OpType::dataflow_signature` copies and/or allocates.
            // That might not seem like a big deal, but it's a significant portion of the time spent
            // when exporting. However it is not trivial to change this at the moment.
            _ => match &optype.dataflow_signature() {
                Some(signature) => {
                    let num_inputs = signature.input_types().len();
                    let num_outputs = signature.output_types().len();
                    let signature = self.export_func_type(signature);
                    (Some(signature), num_inputs, num_outputs)
                }
                None => (None, 0, 0),
            },
        };

        let inputs = self.make_ports(node, Direction::Incoming, num_inputs);
        let outputs = self.make_ports(node, Direction::Outgoing, num_outputs);

        let meta = match self.hugr.get_node_metadata(node) {
            Some(metadata_map) => self.export_node_metadata(metadata_map),
            None => &[],
        };

        // Replace the placeholder node with the actual node.
        *self.module.get_node_mut(node_id).unwrap() = model::Node {
            operation,
            inputs,
            outputs,
            params,
            regions,
            meta,
            signature,
        };

        node_id
    }

    /// Export an `OpDef` as an operation declaration.
    ///
    /// Operations that allow a declarative form are exported as a reference to
    /// an operation declaration node, and this node is reused for all instances
    /// of the operation. The node is added to the `decl_operations` map so that
    /// at the end of the export, the operation declaration nodes can be added
    /// to the module as children of the module region.
    pub fn export_opdef(&mut self, opdef: &OpDef) -> model::GlobalRef<'a> {
        use std::collections::hash_map::Entry;

        let poly_func_type = match opdef.signature_func() {
            SignatureFunc::PolyFuncType(poly_func_type) => poly_func_type,
            _ => return self.make_named_global_ref(opdef.extension_id(), opdef.name()),
        };

        let key = (opdef.extension_id().clone(), opdef.name().clone());
        let entry = self.decl_operations.entry(key);

        let node = match entry {
            Entry::Occupied(occupied_entry) => {
                return model::GlobalRef::Direct(*occupied_entry.get())
            }
            Entry::Vacant(vacant_entry) => {
                *vacant_entry.insert(self.module.insert_node(model::Node {
                    operation: model::Operation::Invalid,
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    regions: &[],
                    meta: &[],
                    signature: None,
                }))
            }
        };

        let decl = self.with_local_scope(node, |this| {
            let name = this.make_qualified_name(opdef.extension_id(), opdef.name());
            let (params, constraints, r#type) = this.export_poly_func_type(poly_func_type);
            let decl = this.bump.alloc(model::OperationDecl {
                name,
                params,
                constraints,
                r#type,
            });
            decl
        });

        let meta = {
            let description = Some(opdef.description()).filter(|d| !d.is_empty());
            let meta_len = opdef.iter_misc().len() + description.is_some() as usize;
            let mut meta = BumpVec::with_capacity_in(meta_len, self.bump);

            if let Some(description) = description {
                let name = META_DESCRIPTION;
                let value = self.make_term(model::Term::Str(self.bump.alloc_str(description)));
                meta.push(model::MetaItem { name, value })
            }

            for (name, value) in opdef.iter_misc() {
                let name = self.bump.alloc_str(name);
                let value = self.export_json(value);
                meta.push(model::MetaItem { name, value });
            }

            self.bump.alloc_slice_copy(&meta)
        };

        let node_data = self.module.get_node_mut(node).unwrap();
        node_data.operation = model::Operation::DeclareOperation { decl };
        node_data.meta = meta;

        model::GlobalRef::Direct(node)
    }

    /// Export the signature of a `DataflowBlock`. Here we can't use `OpType::dataflow_signature`
    /// like for the other nodes since the ports are control flow ports.
    pub fn export_block_signature(&mut self, block: &DataflowBlock) -> model::TermId {
        let inputs = {
            let inputs = self.export_type_row(&block.inputs);
            let inputs = self.make_term(model::Term::Control { values: inputs });
            self.make_term(model::Term::List {
                parts: self.bump.alloc_slice_copy(&[model::ListPart::Item(inputs)]),
            })
        };

        let tail = self.export_type_row(&block.other_outputs);

        let outputs = {
            let mut outputs = BumpVec::with_capacity_in(block.sum_rows.len(), self.bump);
            for sum_row in block.sum_rows.iter() {
                let variant = self.export_type_row_with_tail(sum_row, Some(tail));
                let control = self.make_term(model::Term::Control { values: variant });
                outputs.push(model::ListPart::Item(control));
            }
            self.make_term(model::Term::List {
                parts: outputs.into_bump_slice(),
            })
        };

        let extensions = self.export_ext_set(&block.extension_delta);
        self.make_term(model::Term::FuncType {
            inputs,
            outputs,
            extensions,
        })
    }

    /// Create a region from the given node's children, if it has any.
    ///
    /// See [`Self::export_dfg`].
    pub fn export_dfg_if_present(
        &mut self,
        node: Node,
        extensions: model::TermId,
    ) -> Option<model::RegionId> {
        if self.hugr.children(node).next().is_none() {
            None
        } else {
            Some(self.export_dfg(node, extensions))
        }
    }

    /// Creates a data flow region from the given node's children.
    ///
    /// `Input` and `Output` nodes are used to determine the source and target ports of the region.
    pub fn export_dfg(&mut self, node: Node, extensions: model::TermId) -> model::RegionId {
        let mut children = self.hugr.children(node);

        // The first child is an `Input` node, which we use to determine the region's sources.
        let input_node = children.next().unwrap();
        let OpType::Input(input_op) = self.hugr.get_optype(input_node) else {
            panic!("expected an `Input` node as the first child node");
        };
        let sources = self.make_ports(input_node, Direction::Outgoing, input_op.types.len());

        // The second child is an `Output` node, which we use to determine the region's targets.
        let output_node = children.next().unwrap();
        let OpType::Output(output_op) = self.hugr.get_optype(output_node) else {
            panic!("expected an `Output` node as the second child node");
        };
        let targets = self.make_ports(output_node, Direction::Incoming, output_op.types.len());

        // Export the remaining children of the node.
        let mut region_children = BumpVec::with_capacity_in(children.size_hint().0, self.bump);

        for child in children {
            region_children.push(self.export_node(child));
        }

        let signature = {
            let inputs = self.export_type_row(&input_op.types);
            let outputs = self.export_type_row(&output_op.types);

            Some(self.make_term(model::Term::FuncType {
                inputs,
                outputs,
                extensions,
            }))
        };

        self.module.insert_region(model::Region {
            kind: model::RegionKind::DataFlow,
            sources,
            targets,
            children: region_children.into_bump_slice(),
            meta: &[], // TODO: Export metadata
            signature,
        })
    }

    /// Creates a control flow region from the given node's children.
    pub fn export_cfg(&mut self, node: Node) -> model::RegionId {
        let mut children = self.hugr.children(node);
        let mut region_children = BumpVec::with_capacity_in(children.size_hint().0 + 1, self.bump);

        // The first child is the entry block.
        // We create a source port on the control flow region and connect it to the
        // first input port of the exported entry block.
        let entry_block = children.next().unwrap();

        let OpType::DataflowBlock(_) = self.hugr.get_optype(entry_block) else {
            panic!("expected a `DataflowBlock` node as the first child node");
        };

        let source = model::LinkRef::Id(self.get_link_id(entry_block, IncomingPort::from(0)));
        region_children.push(self.export_node(entry_block));

        // The last child is the exit block.
        // Contrary to the entry block, the exit block does not have a dataflow subgraph.
        // We therefore do not export the block itself, but simply use its output ports
        // as the target ports of the control flow region.
        let exit_block = children.next_back().unwrap();

        // Export the remaining children of the node, except for the last one.
        for child in children {
            region_children.push(self.export_node(child));
        }

        let OpType::ExitBlock(_) = self.hugr.get_optype(exit_block) else {
            panic!("expected an `ExitBlock` node as the last child node");
        };

        let targets = self.make_ports(exit_block, Direction::Incoming, 1);

        // Get the signature of the control flow region.
        // This is the same as the signature of the parent node.
        let signature = Some(self.export_func_type(&self.hugr.signature(node).unwrap()));

        self.module.insert_region(model::Region {
            kind: model::RegionKind::ControlFlow,
            sources: self.bump.alloc_slice_copy(&[source]),
            targets,
            children: region_children.into_bump_slice(),
            meta: &[], // TODO: Export metadata
            signature,
        })
    }

    /// Export the `Case` node children of a `Conditional` node as data flow regions.
    pub fn export_conditional_regions(&mut self, node: Node) -> &'a [model::RegionId] {
        let children = self.hugr.children(node);
        let mut regions = BumpVec::with_capacity_in(children.size_hint().0, self.bump);

        for child in children {
            let OpType::Case(case_op) = self.hugr.get_optype(child) else {
                panic!("expected a `Case` node as a child of a `Conditional` node");
            };

            let extensions = self.export_ext_set(&case_op.signature.extension_reqs);
            regions.push(self.export_dfg(child, extensions));
        }

        regions.into_bump_slice()
    }

    /// Exports a polymorphic function type.
    ///
    /// The returned triple consists of:
    ///  - The static parameters of the polymorphic function type.
    ///  - The constraints of the polymorphic function type.
    ///  - The function type itself.
    pub fn export_poly_func_type<RV: MaybeRV>(
        &mut self,
        t: &PolyFuncTypeBase<RV>,
    ) -> (&'a [model::Param<'a>], &'a [model::TermId], model::TermId) {
        let mut params = BumpVec::with_capacity_in(t.params().len(), self.bump);
        let scope = self
            .local_scope
            .expect("exporting poly func type outside of local scope");

        for (i, param) in t.params().iter().enumerate() {
            let name = self.bump.alloc_str(&i.to_string());
            let r#type = self.export_type_param(param, Some(model::LocalRef::Index(scope, i as _)));
            let param = model::Param {
                name,
                r#type,
                sort: model::ParamSort::Implicit,
            };
            params.push(param)
        }

        let constraints = self.bump.alloc_slice_copy(&self.local_constraints);
        let body = self.export_func_type(t.body());

        (params.into_bump_slice(), constraints, body)
    }

    pub fn export_type<RV: MaybeRV>(&mut self, t: &TypeBase<RV>) -> model::TermId {
        self.export_type_enum(t.as_type_enum())
    }

    pub fn export_type_enum<RV: MaybeRV>(&mut self, t: &TypeEnum<RV>) -> model::TermId {
        match t {
            TypeEnum::Extension(ext) => self.export_custom_type(ext),
            TypeEnum::Alias(alias) => {
                let name = model::GlobalRef::Named(self.bump.alloc_str(alias.name()));
                let args = &[];
                self.make_term(model::Term::ApplyFull { global: name, args })
            }
            TypeEnum::Function(func) => self.export_func_type(func),
            TypeEnum::Variable(index, _) => {
                let node = self.local_scope.expect("local variable out of scope");
                self.make_term(model::Term::Var(model::LocalRef::Index(node, *index as _)))
            }
            TypeEnum::RowVar(rv) => self.export_row_var(rv.as_rv()),
            TypeEnum::Sum(sum) => self.export_sum_type(sum),
        }
    }

    pub fn export_func_type<RV: MaybeRV>(&mut self, t: &FuncTypeBase<RV>) -> model::TermId {
        let inputs = self.export_type_row(t.input());
        let outputs = self.export_type_row(t.output());
        let extensions = self.export_ext_set(&t.extension_reqs);
        self.make_term(model::Term::FuncType {
            inputs,
            outputs,
            extensions,
        })
    }

    pub fn export_custom_type(&mut self, t: &CustomType) -> model::TermId {
        let global = self.make_named_global_ref(t.extension(), t.name());

        let args = self
            .bump
            .alloc_slice_fill_iter(t.args().iter().map(|p| self.export_type_arg(p)));
        let term = model::Term::ApplyFull { global, args };
        self.make_term(term)
    }

    pub fn export_type_arg(&mut self, t: &TypeArg) -> model::TermId {
        match t {
            TypeArg::Type { ty } => self.export_type(ty),
            TypeArg::BoundedNat { n } => self.make_term(model::Term::Nat(*n)),
            TypeArg::String { arg } => self.make_term(model::Term::Str(self.bump.alloc_str(arg))),
            TypeArg::Sequence { elems } => {
                // For now we assume that the sequence is meant to be a list.
                let parts = self.bump.alloc_slice_fill_iter(
                    elems
                        .iter()
                        .map(|elem| model::ListPart::Item(self.export_type_arg(elem))),
                );
                self.make_term(model::Term::List { parts })
            }
            TypeArg::Extensions { es } => self.export_ext_set(es),
            TypeArg::Variable { v } => self.export_type_arg_var(v),
        }
    }

    pub fn export_type_arg_var(&mut self, var: &TypeArgVariable) -> model::TermId {
        let node = self.local_scope.expect("local variable out of scope");
        self.make_term(model::Term::Var(model::LocalRef::Index(
            node,
            var.index() as _,
        )))
    }

    pub fn export_row_var(&mut self, t: &RowVariable) -> model::TermId {
        let node = self.local_scope.expect("local variable out of scope");
        self.make_term(model::Term::Var(model::LocalRef::Index(node, t.0 as _)))
    }

    pub fn export_sum_type(&mut self, t: &SumType) -> model::TermId {
        match t {
            SumType::Unit { size } => {
                let parts = self.bump.alloc_slice_fill_iter((0..*size).map(|_| {
                    model::ListPart::Item(self.make_term(model::Term::List { parts: &[] }))
                }));
                let variants = self.make_term(model::Term::List { parts });
                self.make_term(model::Term::Adt { variants })
            }
            SumType::General { rows } => {
                let parts = self.bump.alloc_slice_fill_iter(
                    rows.iter()
                        .map(|row| model::ListPart::Item(self.export_type_row(row))),
                );
                let list = model::Term::List { parts };
                let variants = { self.make_term(list) };
                self.make_term(model::Term::Adt { variants })
            }
        }
    }

    #[inline]
    pub fn export_type_row<RV: MaybeRV>(&mut self, row: &TypeRowBase<RV>) -> model::TermId {
        self.export_type_row_with_tail(row, None)
    }

    pub fn export_type_row_with_tail<RV: MaybeRV>(
        &mut self,
        row: &TypeRowBase<RV>,
        tail: Option<model::TermId>,
    ) -> model::TermId {
        let mut parts = BumpVec::with_capacity_in(row.len() + tail.is_some() as usize, self.bump);

        for t in row.iter() {
            match t.as_type_enum() {
                TypeEnum::RowVar(var) => {
                    parts.push(model::ListPart::Splice(self.export_row_var(var.as_rv())));
                }
                _ => {
                    parts.push(model::ListPart::Item(self.export_type(t)));
                }
            }
        }

        if let Some(tail) = tail {
            parts.push(model::ListPart::Splice(tail));
        }

        let parts = parts.into_bump_slice();
        self.make_term(model::Term::List { parts })
    }

    /// Exports a `TypeParam` to a term.
    ///
    /// The `var` argument is set when the type parameter being exported is the
    /// type of a parameter to a polymorphic definition. In that case we can
    /// generate a `nonlinear` constraint for the type of runtime types marked as
    /// `TypeBound::Copyable`.
    pub fn export_type_param(
        &mut self,
        t: &TypeParam,
        var: Option<model::LocalRef<'static>>,
    ) -> model::TermId {
        match t {
            TypeParam::Type { b } => {
                if let (Some(var), TypeBound::Copyable) = (var, b) {
                    let term = self.make_term(model::Term::Var(var));
                    let non_linear = self.make_term(model::Term::NonLinearConstraint { term });
                    self.local_constraints.push(non_linear);
                }

                self.make_term(model::Term::Type)
            }
            // This ignores the bound on the natural for now.
            TypeParam::BoundedNat { .. } => self.make_term(model::Term::NatType),
            TypeParam::String => self.make_term(model::Term::StrType),
            TypeParam::List { param } => {
                let item_type = self.export_type_param(param, None);
                self.make_term(model::Term::ListType { item_type })
            }
            TypeParam::Tuple { params } => {
                let parts = self.bump.alloc_slice_fill_iter(
                    params
                        .iter()
                        .map(|param| model::ListPart::Item(self.export_type_param(param, None))),
                );
                let types = self.make_term(model::Term::List { parts });
                self.make_term(model::Term::ApplyFull {
                    global: model::GlobalRef::Named(TERM_PARAM_TUPLE),
                    args: self.bump.alloc_slice_copy(&[types]),
                })
            }
            TypeParam::Extensions => {
                let term = model::Term::ExtSetType;
                self.make_term(term)
            }
        }
    }

    pub fn export_ext_set(&mut self, ext_set: &ExtensionSet) -> model::TermId {
        let capacity = ext_set.iter().size_hint().0;
        let mut parts = BumpVec::with_capacity_in(capacity, self.bump);

        for ext in ext_set.iter() {
            // `ExtensionSet`s represent variables by extension names that parse to integers.
            match ext.parse::<u16>() {
                Ok(var) => {
                    let node = self.local_scope.expect("local variable out of scope");
                    let local_ref = model::LocalRef::Index(node, var);
                    let term = self.make_term(model::Term::Var(local_ref));
                    parts.push(model::ExtSetPart::Splice(term));
                }
                Err(_) => parts.push(model::ExtSetPart::Extension(self.bump.alloc_str(ext))),
            }
        }

        self.make_term(model::Term::ExtSet {
            parts: parts.into_bump_slice(),
        })
    }

    pub fn export_node_metadata(
        &mut self,
        metadata_map: &NodeMetadataMap,
    ) -> &'a [model::MetaItem<'a>] {
        let mut meta = BumpVec::with_capacity_in(metadata_map.len(), self.bump);

        for (name, value) in metadata_map {
            let name = self.bump.alloc_str(name);
            let value = self.export_json(value);
            meta.push(model::MetaItem { name, value });
        }

        meta.into_bump_slice()
    }

    pub fn export_json(&mut self, value: &serde_json::Value) -> model::TermId {
        let value = serde_json::to_string(value).expect("json values are always serializable");
        let value = self.make_term(model::Term::Str(self.bump.alloc_str(&value)));
        let value = self.bump.alloc_slice_copy(&[value]);
        self.make_term(model::Term::ApplyFull {
            global: model::GlobalRef::Named(TERM_JSON),
            args: value,
        })
    }
}

#[cfg(test)]
mod test {
    use rstest::{fixture, rstest};

    use crate::{
        builder::{Dataflow, DataflowSubContainer},
        extension::prelude::qb_t,
        std_extensions::arithmetic::float_types,
        types::Signature,
        utils::test_quantum_extension::{self, cx_gate, h_gate},
        Hugr,
    };

    #[fixture]
    fn test_simple_circuit() -> Hugr {
        crate::builder::test::build_main(
            Signature::new_endo(vec![qb_t(), qb_t()])
                .with_extension_delta(test_quantum_extension::EXTENSION_ID)
                .with_extension_delta(float_types::EXTENSION_ID)
                .into(),
            |mut f_build| {
                let wires: Vec<_> = f_build.input_wires().collect();
                let mut linear = f_build.as_circuit(wires);

                assert_eq!(linear.n_wires(), 2);

                linear
                    .append(h_gate(), [0])?
                    .append(cx_gate(), [0, 1])?
                    .append(cx_gate(), [1, 0])?;

                let outs = linear.finish();
                f_build.finish_with_outputs(outs)
            },
        )
        .unwrap()
    }

    #[rstest]
    #[case(test_simple_circuit())]
    fn test_export(#[case] hugr: Hugr) {
        use bumpalo::Bump;
        let bump = Bump::new();
        let _model = super::export_hugr(&hugr, &bump);
    }
}
