//! Exporting HUGR graphs to their `hugr-model` representation.
use crate::{
    extension::{ExtensionId, OpDef, SignatureFunc},
    hugr::{IdentList, NodeMetadataMap},
    ops::{
        constant::CustomSerialized, DataflowBlock, DataflowOpTrait, OpName, OpTrait, OpType, Value,
    },
    package::Package,
    std_extensions::{
        arithmetic::{float_types::ConstF64, int_types::ConstInt},
        collections::array::ArrayValue,
    },
    types::{
        type_param::{TypeArgVariable, TypeParam},
        type_row::TypeRowBase,
        CustomType, FuncTypeBase, MaybeRV, PolyFuncTypeBase, RowVariable, SumType, TypeArg,
        TypeBase, TypeBound, TypeEnum, TypeRow,
    },
    Direction, Hugr, HugrView, IncomingPort, Node, Port,
};

use fxhash::{FxBuildHasher, FxHashMap};
use hugr_model::v0::{
    self as model,
    bumpalo::{collections::Vec as BumpVec, Bump},
    table,
};
use petgraph::unionfind::UnionFind;
use smol_str::{format_smolstr, SmolStr, ToSmolStr};

/// Export a [`Package`] to its representation in the model.
pub fn export_package<'a>(package: &'a Package, bump: &'a Bump) -> table::Package<'a> {
    let modules = package
        .modules
        .iter()
        .map(|module| export_hugr(module, bump))
        .collect();
    table::Package { modules }
}

/// Export a [`Hugr`] graph to its representation in the model.
pub fn export_hugr<'a>(hugr: &'a Hugr, bump: &'a Bump) -> table::Module<'a> {
    let mut ctx = Context::new(hugr, bump);
    ctx.export_root();
    ctx.module
}

/// State for converting a HUGR graph to its representation in the model.
struct Context<'a> {
    /// The HUGR graph to convert.
    hugr: &'a Hugr,
    /// The module that is being built.
    module: table::Module<'a>,
    /// The arena in which the model is allocated.
    bump: &'a Bump,
    /// Stores the terms that we have already seen to avoid duplicates.
    term_map: FxHashMap<table::Term<'a>, table::TermId>,

    /// The current scope for local variables.
    ///
    /// This is set to the id of the smallest enclosing node that defines a polymorphic type.
    /// We use this when exporting local variables in terms.
    local_scope: Option<table::NodeId>,

    /// Constraints to be added to the local scope.
    ///
    /// When exporting a node that defines a polymorphic type, we use this field
    /// to collect the constraints that need to be added to that polymorphic
    /// type. Currently this is used to record `nonlinear` constraints on uses
    /// of `TypeParam::Type` with a `TypeBound::Copyable` bound.
    local_constraints: Vec<table::TermId>,

    /// Mapping from extension operations to their declarations.
    decl_operations: FxHashMap<(ExtensionId, OpName), table::NodeId>,

    /// Auxiliary structure for tracking the links between ports.
    links: Links,

    /// The symbol table tracking symbols that are currently in scope.
    symbols: model::scope::SymbolTable<CoreSymbol>,

    /// Mapping from implicit imports to their node ids.
    implicit_imports: FxHashMap<CoreSymbol, table::NodeId>,

    /// Map from node ids in the [`Hugr`] to the corresponding node ids in the model.
    node_to_id: FxHashMap<Node, table::NodeId>,

    /// Mapping from node ids in the [`Hugr`] to the corresponding model nodes.
    id_to_node: FxHashMap<table::NodeId, Node>,

    static_symbols: FxHashMap<&'static str, CoreSymbol>,
}

impl<'a> Context<'a> {
    pub fn new(hugr: &'a Hugr, bump: &'a Bump) -> Self {
        let mut module = table::Module::default();
        module.nodes.reserve(hugr.node_count());
        let links = Links::new(hugr);

        Self {
            hugr,
            module,
            bump,
            links,
            term_map: FxHashMap::default(),
            local_scope: None,
            decl_operations: FxHashMap::default(),
            local_constraints: Vec::new(),
            symbols: model::scope::SymbolTable::default(),
            implicit_imports: FxHashMap::default(),
            node_to_id: FxHashMap::default(),
            id_to_node: FxHashMap::default(),
            static_symbols: FxHashMap::default(),
        }
    }

    /// Exports the root module of the HUGR graph.
    pub fn export_root(&mut self) {
        self.module.root = self.module.insert_region(table::Region::default());
        self.symbols.enter(self.module.root);
        self.links.enter(self.module.root);

        let hugr_children = self.hugr.children(self.hugr.root());
        let mut children = Vec::with_capacity(hugr_children.size_hint().0);

        for child in hugr_children.clone() {
            if let Some(child_id) = self.export_node_shallow(child) {
                children.push(child_id);
            }
        }

        for child in &children {
            self.export_node_deep(*child);
        }

        let mut all_children = BumpVec::with_capacity_in(
            children.len() + self.decl_operations.len() + self.implicit_imports.len(),
            self.bump,
        );

        all_children.extend(self.implicit_imports.drain().map(|(_, id)| id));
        all_children.extend(self.decl_operations.values().copied());
        all_children.extend(children);

        let (links, ports) = self.links.exit();
        self.symbols.exit();

        self.module.regions[self.module.root.index()] = table::Region {
            kind: model::RegionKind::Module,
            sources: &[],
            targets: &[],
            children: all_children.into_bump_slice(),
            meta: &[], // TODO: Export metadata
            signature: None,
            scope: Some(table::RegionScope { links, ports }),
        };
    }

    pub fn make_ports(
        &mut self,
        node: Node,
        direction: Direction,
        num_ports: usize,
    ) -> &'a [table::LinkIndex] {
        let ports = self.hugr.node_ports(node, direction);
        let mut links = BumpVec::with_capacity_in(ports.size_hint().0, self.bump);

        for port in ports.take(num_ports) {
            links.push(self.links.use_link(node, port));
        }

        links.into_bump_slice()
    }

    pub fn make_term(&mut self, term: table::Term<'a>) -> table::TermId {
        // Wildcard terms do not all represent the same term, so we should not deduplicate them.
        if term == table::Term::Wildcard {
            return self.module.insert_term(term);
        }

        *self
            .term_map
            .entry(term.clone())
            .or_insert_with(|| self.module.insert_term(term))
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

    fn with_local_scope<T>(&mut self, node: table::NodeId, f: impl FnOnce(&mut Self) -> T) -> T {
        let prev_local_scope = self.local_scope.replace(node);
        let prev_local_constraints = std::mem::take(&mut self.local_constraints);
        let result = f(self);
        self.local_scope = prev_local_scope;
        self.local_constraints = prev_local_constraints;
        result
    }

    fn export_node_shallow(&mut self, node: Node) -> Option<table::NodeId> {
        let optype = self.hugr.get_optype(node);

        // We skip nodes that are not exported as nodes in the model.
        if let OpType::Const(_)
        | OpType::Input(_)
        | OpType::Output(_)
        | OpType::ExitBlock(_)
        | OpType::Case(_) = optype
        {
            return None;
        }

        let node_id = self.module.insert_node(table::Node::default());
        self.node_to_id.insert(node, node_id);
        self.id_to_node.insert(node_id, node);

        // We record the name of the symbol defined by the node, if any.
        let symbol = match optype {
            OpType::FuncDefn(func_defn) => Some(func_defn.name.to_smolstr()),
            OpType::FuncDecl(func_decl) => Some(func_decl.name.to_smolstr()),
            OpType::AliasDecl(alias_decl) => Some(alias_decl.name.clone()),
            OpType::AliasDefn(alias_defn) => Some(alias_defn.name.clone()),
            _ => None,
        };

        if let Some(symbol) = symbol {
            self.symbols
                .insert(CoreSymbol::Local(symbol), node_id)
                .expect("duplicate symbol");
        }

        Some(node_id)
    }

    fn export_node_deep(&mut self, node_id: table::NodeId) {
        // We insert a dummy node with the invalid operation at this point to reserve
        // the node id. This is necessary to establish the correct node id for the
        // local scope introduced by some operations. We will overwrite this node later.
        let mut regions: &[_] = &[];

        let node = self.id_to_node[&node_id];
        let optype = self.hugr.get_optype(node);

        let operation = match optype {
            OpType::Module(_) => todo!("this should be an error"),

            OpType::Input(_) => {
                panic!("input nodes should have been handled by the region export")
            }

            OpType::Output(_) => {
                panic!("output nodes should have been handled by the region export")
            }

            OpType::DFG(_) => {
                regions = self
                    .bump
                    .alloc_slice_copy(&[self.export_dfg(node, model::ScopeClosure::Open)]);
                table::Operation::Dfg
            }

            OpType::CFG(_) => {
                regions = self
                    .bump
                    .alloc_slice_copy(&[self.export_cfg(node, model::ScopeClosure::Open)]);
                table::Operation::Cfg
            }

            OpType::ExitBlock(_) => {
                panic!("exit blocks should have been handled by the region export")
            }

            OpType::Case(_) => {
                todo!("case nodes should have been handled by the region export")
            }

            OpType::DataflowBlock(_) => {
                regions = self
                    .bump
                    .alloc_slice_copy(&[self.export_dfg(node, model::ScopeClosure::Open)]);
                table::Operation::Block
            }

            OpType::FuncDefn(func) => self.with_local_scope(node_id, |this| {
                let name = this.symbols.symbol_name(node_id).unwrap().clone();
                let symbol = this.export_poly_func_type(name, &func.signature);
                regions = this
                    .bump
                    .alloc_slice_copy(&[this.export_dfg(node, model::ScopeClosure::Closed)]);
                table::Operation::DefineFunc(symbol)
            }),

            OpType::FuncDecl(func) => self.with_local_scope(node_id, |this| {
                let name = this.symbols.symbol_name(node_id).unwrap().clone();
                let symbol = this.export_poly_func_type(name, &func.signature);
                table::Operation::DeclareFunc(symbol)
            }),

            OpType::AliasDecl(alias) => self.with_local_scope(node_id, |this| {
                // TODO: We should support aliases with different types and with parameters
                let signature = this.make_term_apply(model::CORE_TYPE, &[]);
                let symbol = this.bump.alloc(table::Symbol {
                    name: model::SymbolName::new(alias.name.clone()),
                    params: &[],
                    constraints: &[],
                    signature,
                });
                table::Operation::DeclareAlias(symbol)
            }),

            OpType::AliasDefn(alias) => self.with_local_scope(node_id, |this| {
                let value = this.export_type(&alias.definition);
                // TODO: We should support aliases with different types and with parameters
                let signature = this.make_term_apply(model::CORE_TYPE, &[]);
                let symbol = this.bump.alloc(table::Symbol {
                    name: model::SymbolName::new(alias.name.clone()),
                    params: &[],
                    constraints: &[],
                    signature,
                });
                table::Operation::DefineAlias(symbol, value)
            }),

            OpType::Call(call) => {
                // TODO: If the node is not connected to a function, we should do better than panic.
                let node = self.connected_function(node).unwrap();
                let symbol = self.node_to_id[&node];
                let mut args = BumpVec::new_in(self.bump);
                args.extend(call.type_args.iter().map(|arg| self.export_type_arg(arg)));
                let args = args.into_bump_slice();
                let func = self.make_term(table::Term::Apply(symbol, args));

                // TODO PERFORMANCE: Avoid exporting the signature here again.
                let signature = call.signature();
                let inputs = self.export_type_row(&signature.input);
                let outputs = self.export_type_row(&signature.output);
                let operation = self.make_term_apply(model::CORE_CALL, &[inputs, outputs, func]);
                table::Operation::Custom(operation)
            }

            OpType::LoadFunction(load) => {
                let node = self.connected_function(node).unwrap();
                let symbol = self.node_to_id[&node];
                let mut args = BumpVec::new_in(self.bump);
                args.extend(load.type_args.iter().map(|arg| self.export_type_arg(arg)));
                let args = args.into_bump_slice();
                let func = self.make_term(table::Term::Apply(symbol, args));
                let runtime_type = self.make_term(table::Term::Wildcard);
                let operation = self.make_term_apply(model::CORE_LOAD_CONST, &[runtime_type, func]);
                table::Operation::Custom(operation)
            }

            OpType::Const(_) => {
                unreachable!("const nodes are filtered out by `export_node_shallow`")
            }

            OpType::LoadConstant(_) => {
                // TODO: If the node is not connected to a constant, we should do better than panic.
                let const_node = self.hugr.static_source(node).unwrap();
                let const_node_op = self.hugr.get_optype(const_node);

                let OpType::Const(const_node_data) = const_node_op else {
                    panic!("expected `LoadConstant` node to be connected to a `Const` node");
                };

                // TODO: Share the constant value between all nodes that load it.

                let runtime_type = self.make_term(table::Term::Wildcard);
                let value = self.export_value(&const_node_data.value);
                let operation =
                    self.make_term_apply(model::CORE_LOAD_CONST, &[runtime_type, value]);
                table::Operation::Custom(operation)
            }

            OpType::CallIndirect(call) => {
                let inputs = self.export_type_row(&call.signature.input);
                let outputs = self.export_type_row(&call.signature.output);
                let operation = self.make_term_apply(model::CORE_CALL_INDIRECT, &[inputs, outputs]);
                table::Operation::Custom(operation)
            }

            OpType::Tag(tag) => {
                let variants = self.make_term(table::Term::Wildcard);
                let types = self.make_term(table::Term::Wildcard);
                let tag = self.make_term(model::Literal::Nat(tag.tag as u64).into());
                let operation = self.make_term_apply(model::CORE_MAKE_ADT, &[variants, types, tag]);
                table::Operation::Custom(operation)
            }

            OpType::TailLoop(_) => {
                regions = self
                    .bump
                    .alloc_slice_copy(&[self.export_dfg(node, model::ScopeClosure::Open)]);
                table::Operation::TailLoop
            }

            OpType::Conditional(_) => {
                regions = self.export_conditional_regions(node);
                table::Operation::Conditional
            }

            OpType::ExtensionOp(op) => {
                let node = self.export_opdef(op.def());
                let params = self
                    .bump
                    .alloc_slice_fill_iter(op.args().iter().map(|arg| self.export_type_arg(arg)));
                let operation = self.make_term(table::Term::Apply(node, params));
                table::Operation::Custom(operation)
            }

            OpType::OpaqueOp(op) => {
                let name = CoreSymbol::Qualified(op.extension().clone(), op.op_name().clone());
                let node = self.resolve_symbol(&name);
                let params = self
                    .bump
                    .alloc_slice_fill_iter(op.args().iter().map(|arg| self.export_type_arg(arg)));
                let operation = self.make_term(table::Term::Apply(node, params));
                table::Operation::Custom(operation)
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

        self.module.nodes[node_id.index()] = table::Node {
            operation,
            inputs,
            outputs,
            regions,
            meta,
            signature,
        };
    }

    /// Export an `OpDef` as an operation declaration.
    ///
    /// Operations that allow a declarative form are exported as a reference to
    /// an operation declaration node, and this node is reused for all instances
    /// of the operation. The node is added to the `decl_operations` map so that
    /// at the end of the export, the operation declaration nodes can be added
    /// to the module as children of the module region.
    pub fn export_opdef(&mut self, opdef: &OpDef) -> table::NodeId {
        use std::collections::hash_map::Entry;

        let poly_func_type = match opdef.signature_func() {
            SignatureFunc::PolyFuncType(poly_func_type) => poly_func_type,
            _ => {
                let name =
                    CoreSymbol::Qualified(opdef.extension_id().clone(), opdef.name().clone());
                return self.resolve_symbol(&name);
            }
        };

        let key = (opdef.extension_id().clone(), opdef.name().clone());
        let entry = self.decl_operations.entry(key);

        let node = match entry {
            Entry::Occupied(occupied_entry) => return *occupied_entry.get(),
            Entry::Vacant(vacant_entry) => {
                *vacant_entry.insert(self.module.insert_node(table::Node::default()))
            }
        };

        let symbol = self.with_local_scope(node, |this| {
            let name = CoreSymbol::Qualified(opdef.extension_id().clone(), opdef.name().clone());
            this.export_poly_func_type(name, poly_func_type)
        });

        let meta = {
            let description = Some(opdef.description()).filter(|d| !d.is_empty());
            let meta_len = opdef.iter_misc().len() + description.is_some() as usize;
            let mut meta = BumpVec::with_capacity_in(meta_len, self.bump);

            if let Some(description) = description {
                let value = self.make_term(model::Literal::Str(description.into()).into());
                meta.push(self.make_term_apply(model::CORE_META_DESCRIPTION, &[value]));
            }

            for (name, value) in opdef.iter_misc() {
                meta.push(self.export_json_meta(name, value));
            }

            self.bump.alloc_slice_copy(&meta)
        };

        let node_data = self.module.get_node_mut(node).unwrap();
        node_data.operation = table::Operation::DeclareOperation(symbol);
        node_data.meta = meta;

        node
    }

    /// Export the signature of a `DataflowBlock`. Here we can't use `OpType::dataflow_signature`
    /// like for the other nodes since the ports are control flow ports.
    pub fn export_block_signature(&mut self, block: &DataflowBlock) -> table::TermId {
        let inputs = {
            let inputs = self.export_type_row(&block.inputs);
            let inputs = self.make_term_apply(model::CORE_CTRL, &[inputs]);
            self.make_term(table::Term::List(
                self.bump.alloc_slice_copy(&[table::SeqPart::Item(inputs)]),
            ))
        };

        let tail = self.export_type_row(&block.other_outputs);

        let outputs = {
            let mut outputs = BumpVec::with_capacity_in(block.sum_rows.len(), self.bump);
            for sum_row in block.sum_rows.iter() {
                let variant = self.export_type_row_with_tail(sum_row, Some(tail));
                let control = self.make_term_apply(model::CORE_CTRL, &[variant]);
                outputs.push(table::SeqPart::Item(control));
            }
            self.make_term(table::Term::List(outputs.into_bump_slice()))
        };

        self.make_term_apply(model::CORE_FN, &[inputs, outputs])
    }

    /// Creates a data flow region from the given node's children.
    ///
    /// `Input` and `Output` nodes are used to determine the source and target ports of the region.
    pub fn export_dfg(&mut self, node: Node, closure: model::ScopeClosure) -> table::RegionId {
        let region = self.module.insert_region(table::Region::default());

        self.symbols.enter(region);
        if closure == model::ScopeClosure::Closed {
            self.links.enter(region);
        }

        let mut sources: &[_] = &[];
        let mut targets: &[_] = &[];
        let mut input_types = None;
        let mut output_types = None;

        let children = self.hugr.children(node);
        let mut region_children = BumpVec::with_capacity_in(children.size_hint().0 - 2, self.bump);

        for child in children {
            match self.hugr.get_optype(child) {
                OpType::Input(input) => {
                    sources = self.make_ports(child, Direction::Outgoing, input.types.len());
                    input_types = Some(&input.types);
                }
                OpType::Output(output) => {
                    targets = self.make_ports(child, Direction::Incoming, output.types.len());
                    output_types = Some(&output.types);
                }
                _ => {
                    if let Some(child_id) = self.export_node_shallow(child) {
                        region_children.push(child_id);
                    }
                }
            }
        }

        for child_id in &region_children {
            self.export_node_deep(*child_id);
        }

        let signature = {
            let inputs = self.export_type_row(input_types.unwrap());
            let outputs = self.export_type_row(output_types.unwrap());
            Some(self.make_term_apply(model::CORE_FN, &[inputs, outputs]))
        };

        let scope = match closure {
            model::ScopeClosure::Closed => {
                let (links, ports) = self.links.exit();
                Some(table::RegionScope { links, ports })
            }
            model::ScopeClosure::Open => None,
        };
        self.symbols.exit();

        self.module.regions[region.index()] = table::Region {
            kind: model::RegionKind::DataFlow,
            sources,
            targets,
            children: region_children.into_bump_slice(),
            meta: &[], // TODO: Export metadata
            signature,
            scope,
        };

        region
    }

    /// Creates a control flow region from the given node's children.
    pub fn export_cfg(&mut self, node: Node, closure: model::ScopeClosure) -> table::RegionId {
        let region = self.module.insert_region(table::Region::default());
        self.symbols.enter(region);

        if closure == model::ScopeClosure::Closed {
            self.links.enter(region);
        }

        let mut source = None;
        let mut targets: &[_] = &[];

        let children = self.hugr.children(node);
        let mut region_children = BumpVec::with_capacity_in(children.size_hint().0 - 1, self.bump);

        for child in children {
            match self.hugr.get_optype(child) {
                OpType::ExitBlock(_) => {
                    targets = self.make_ports(child, Direction::Incoming, 1);
                }
                _ => {
                    if let Some(child_id) = self.export_node_shallow(child) {
                        region_children.push(child_id);
                    }

                    if source.is_none() {
                        source = Some(self.links.use_link(child, IncomingPort::from(0)));
                    }
                }
            }
        }

        for child_id in &region_children {
            self.export_node_deep(*child_id);
        }

        // Get the signature of the control flow region.
        let signature = {
            let node_signature = self.hugr.signature(node).unwrap();

            let mut wrap_ctrl = |types: &TypeRow| {
                let types = self.export_type_row(types);
                let types_ctrl = self.make_term_apply(model::CORE_CTRL, &[types]);
                self.make_term(table::Term::List(
                    self.bump
                        .alloc_slice_copy(&[table::SeqPart::Item(types_ctrl)]),
                ))
            };

            let inputs = wrap_ctrl(node_signature.input());
            let outputs = wrap_ctrl(node_signature.output());
            Some(self.make_term_apply(model::CORE_FN, &[inputs, outputs]))
        };

        let scope = match closure {
            model::ScopeClosure::Closed => {
                let (links, ports) = self.links.exit();
                Some(table::RegionScope { links, ports })
            }
            model::ScopeClosure::Open => None,
        };
        self.symbols.exit();

        self.module.regions[region.index()] = table::Region {
            kind: model::RegionKind::ControlFlow,
            sources: self.bump.alloc_slice_copy(&[source.unwrap()]),
            targets,
            children: region_children.into_bump_slice(),
            meta: &[], // TODO: Export metadata
            signature,
            scope,
        };

        region
    }

    /// Export the `Case` node children of a `Conditional` node as data flow regions.
    pub fn export_conditional_regions(&mut self, node: Node) -> &'a [table::RegionId] {
        let children = self.hugr.children(node);
        let mut regions = BumpVec::with_capacity_in(children.size_hint().0, self.bump);

        for child in children {
            let OpType::Case(_) = self.hugr.get_optype(child) else {
                panic!("expected a `Case` node as a child of a `Conditional` node");
            };

            regions.push(self.export_dfg(child, model::ScopeClosure::Open));
        }

        regions.into_bump_slice()
    }

    /// Exports a polymorphic function type.
    pub fn export_poly_func_type<RV: MaybeRV>(
        &mut self,
        name: CoreSymbol,
        t: &PolyFuncTypeBase<RV>,
    ) -> &'a table::Symbol<'a> {
        let mut params = BumpVec::with_capacity_in(t.params().len(), self.bump);
        let scope = self
            .local_scope
            .expect("exporting poly func type outside of local scope");

        for (i, param) in t.params().iter().enumerate() {
            let name = model::VarName::new(i.to_smolstr());
            let r#type = self.export_type_param(param, Some((scope, i as _)));
            let param = table::Param { name, r#type };
            params.push(param)
        }

        let constraints = self.bump.alloc_slice_copy(&self.local_constraints);
        let body = self.export_func_type(t.body());

        self.bump.alloc(table::Symbol {
            name: name.into(),
            params: params.into_bump_slice(),
            constraints,
            signature: body,
        })
    }

    pub fn export_type<RV: MaybeRV>(&mut self, t: &TypeBase<RV>) -> table::TermId {
        self.export_type_enum(t.as_type_enum())
    }

    pub fn export_type_enum<RV: MaybeRV>(&mut self, t: &TypeEnum<RV>) -> table::TermId {
        match t {
            TypeEnum::Extension(ext) => self.export_custom_type(ext),
            TypeEnum::Alias(alias) => {
                let name = CoreSymbol::Local(alias.name().to_smolstr());
                let symbol = self.resolve_symbol(&name);
                self.make_term(table::Term::Apply(symbol, &[]))
            }
            TypeEnum::Function(func) => self.export_func_type(func),
            TypeEnum::Variable(index, _) => {
                let node = self.local_scope.expect("local variable out of scope");
                self.make_term(table::Term::Var(table::VarId(node, *index as _)))
            }
            TypeEnum::RowVar(rv) => self.export_row_var(rv.as_rv()),
            TypeEnum::Sum(sum) => self.export_sum_type(sum),
        }
    }

    pub fn export_func_type<RV: MaybeRV>(&mut self, t: &FuncTypeBase<RV>) -> table::TermId {
        let inputs = self.export_type_row(t.input());
        let outputs = self.export_type_row(t.output());
        self.make_term_apply(model::CORE_FN, &[inputs, outputs])
    }

    pub fn export_custom_type(&mut self, t: &CustomType) -> table::TermId {
        let name = CoreSymbol::Qualified(t.extension().clone(), t.name().clone());
        let symbol = self.resolve_symbol(&name);

        let args = self
            .bump
            .alloc_slice_fill_iter(t.args().iter().map(|p| self.export_type_arg(p)));
        let term = table::Term::Apply(symbol, args);
        self.make_term(term)
    }

    pub fn export_type_arg(&mut self, t: &TypeArg) -> table::TermId {
        match t {
            TypeArg::Type { ty } => self.export_type(ty),
            TypeArg::BoundedNat { n } => self.make_term(model::Literal::Nat(*n).into()),
            TypeArg::String { arg } => self.make_term(model::Literal::Str(arg.into()).into()),
            TypeArg::Sequence { elems } => {
                // For now we assume that the sequence is meant to be a list.
                let parts = self.bump.alloc_slice_fill_iter(
                    elems
                        .iter()
                        .map(|elem| table::SeqPart::Item(self.export_type_arg(elem))),
                );
                self.make_term(table::Term::List(parts))
            }
            TypeArg::Extensions { .. } => self.make_term_apply("compat.ext_set", &[]),
            TypeArg::Variable { v } => self.export_type_arg_var(v),
        }
    }

    pub fn export_type_arg_var(&mut self, var: &TypeArgVariable) -> table::TermId {
        let node = self.local_scope.expect("local variable out of scope");
        self.make_term(table::Term::Var(table::VarId(node, var.index() as _)))
    }

    pub fn export_row_var(&mut self, t: &RowVariable) -> table::TermId {
        let node = self.local_scope.expect("local variable out of scope");
        self.make_term(table::Term::Var(table::VarId(node, t.0 as _)))
    }

    pub fn export_sum_variants(&mut self, t: &SumType) -> table::TermId {
        match t {
            SumType::Unit { size } => {
                let parts = self.bump.alloc_slice_fill_iter(
                    (0..*size)
                        .map(|_| table::SeqPart::Item(self.make_term(table::Term::List(&[])))),
                );
                self.make_term(table::Term::List(parts))
            }
            SumType::General { rows } => {
                let parts = self.bump.alloc_slice_fill_iter(
                    rows.iter()
                        .map(|row| table::SeqPart::Item(self.export_type_row(row))),
                );
                self.make_term(table::Term::List(parts))
            }
        }
    }

    pub fn export_sum_type(&mut self, t: &SumType) -> table::TermId {
        let variants = self.export_sum_variants(t);
        self.make_term_apply(model::CORE_ADT, &[variants])
    }

    #[inline]
    pub fn export_type_row<RV: MaybeRV>(&mut self, row: &TypeRowBase<RV>) -> table::TermId {
        self.export_type_row_with_tail(row, None)
    }

    pub fn export_type_row_with_tail<RV: MaybeRV>(
        &mut self,
        row: &TypeRowBase<RV>,
        tail: Option<table::TermId>,
    ) -> table::TermId {
        let mut parts = BumpVec::with_capacity_in(row.len() + tail.is_some() as usize, self.bump);

        for t in row.iter() {
            match t.as_type_enum() {
                TypeEnum::RowVar(var) => {
                    parts.push(table::SeqPart::Splice(self.export_row_var(var.as_rv())));
                }
                _ => {
                    parts.push(table::SeqPart::Item(self.export_type(t)));
                }
            }
        }

        if let Some(tail) = tail {
            parts.push(table::SeqPart::Splice(tail));
        }

        let parts = parts.into_bump_slice();
        self.make_term(table::Term::List(parts))
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
        var: Option<(table::NodeId, table::VarIndex)>,
    ) -> table::TermId {
        match t {
            TypeParam::Type { b } => {
                if let (Some((node, index)), TypeBound::Copyable) = (var, b) {
                    let term = self.make_term(table::Term::Var(table::VarId(node, index)));
                    let non_linear = self.make_term_apply(model::CORE_NON_LINEAR, &[term]);
                    self.local_constraints.push(non_linear);
                }

                self.make_term_apply(model::CORE_TYPE, &[])
            }
            // This ignores the bound on the natural for now.
            TypeParam::BoundedNat { .. } => self.make_term_apply(model::CORE_NAT_TYPE, &[]),
            TypeParam::String => self.make_term_apply(model::CORE_STR_TYPE, &[]),
            TypeParam::List { param } => {
                let item_type = self.export_type_param(param, None);
                self.make_term_apply(model::CORE_LIST_TYPE, &[item_type])
            }
            TypeParam::Tuple { params } => {
                let parts = self.bump.alloc_slice_fill_iter(
                    params
                        .iter()
                        .map(|param| table::SeqPart::Item(self.export_type_param(param, None))),
                );
                let types = self.make_term(table::Term::List(parts));
                self.make_term_apply(model::CORE_TUPLE_TYPE, &[types])
            }
            TypeParam::Extensions => self.make_term_apply("compat.ext_set_type", &[]),
        }
    }

    fn export_value(&mut self, value: &'a Value) -> table::TermId {
        match value {
            Value::Extension { e } => {
                // NOTE: We have special cased arrays, integers, and floats for now.
                // TODO: Allow arbitrary extension values to be exported as terms.

                if let Some(array) = e.value().downcast_ref::<ArrayValue>() {
                    let len = self
                        .make_term(model::Literal::Nat(array.get_contents().len() as u64).into());
                    let element_type = self.export_type(array.get_element_type());
                    let mut contents =
                        BumpVec::with_capacity_in(array.get_contents().len(), self.bump);

                    for element in array.get_contents() {
                        contents.push(table::SeqPart::Item(self.export_value(element)));
                    }

                    let contents = self.make_term(table::Term::List(contents.into_bump_slice()));
                    return self
                        .make_term_apply(ArrayValue::CTR_NAME, &[len, element_type, contents]);
                }

                if let Some(v) = e.value().downcast_ref::<ConstInt>() {
                    let bitwidth = self.make_term(model::Literal::Nat(v.log_width() as u64).into());
                    let literal = self.make_term(model::Literal::Nat(v.value_u()).into());
                    return self.make_term_apply(ConstInt::CTR_NAME, &[bitwidth, literal]);
                }

                if let Some(v) = e.value().downcast_ref::<ConstF64>() {
                    let literal = self.make_term(model::Literal::Float(v.value().into()).into());
                    return self.make_term_apply(ConstF64::CTR_NAME, &[literal]);
                }

                let json = match e.value().downcast_ref::<CustomSerialized>() {
                    Some(custom) => serde_json::to_string(custom.value()).unwrap(),
                    None => serde_json::to_string(e.value())
                        .expect("custom extension values should be serializable"),
                };

                let json = self.make_term(model::Literal::Str(json.into()).into());
                let runtime_type = self.export_type(&e.get_type());
                return self.make_term_apply(model::COMPAT_CONST_JSON, &[runtime_type, json]);
            }

            Value::Function { hugr } => {
                let outer_hugr = std::mem::replace(&mut self.hugr, hugr);
                let outer_node_to_id = std::mem::take(&mut self.node_to_id);

                let region = match hugr.root_type() {
                    OpType::DFG(_) => self.export_dfg(hugr.root(), model::ScopeClosure::Closed),
                    _ => panic!("Value::Function root must be a DFG"),
                };

                self.node_to_id = outer_node_to_id;
                self.hugr = outer_hugr;

                self.make_term(table::Term::Func(region))
            }

            Value::Sum(sum) => {
                let variants = self.export_sum_variants(&sum.sum_type);
                let types = self.make_term(table::Term::Wildcard);
                let tag = self.make_term(model::Literal::Nat(sum.tag as u64).into());

                let values = {
                    let mut values = BumpVec::with_capacity_in(sum.values.len(), self.bump);

                    for value in &sum.values {
                        values.push(table::SeqPart::Item(self.export_value(value)));
                    }

                    self.make_term(table::Term::Tuple(values.into_bump_slice()))
                };

                self.make_term_apply(model::CORE_CONST_ADT, &[variants, types, tag, values])
            }
        }
    }

    pub fn export_node_metadata(&mut self, metadata_map: &NodeMetadataMap) -> &'a [table::TermId] {
        let mut meta = BumpVec::with_capacity_in(metadata_map.len(), self.bump);

        for (name, value) in metadata_map {
            meta.push(self.export_json_meta(name, value));
        }

        meta.into_bump_slice()
    }

    pub fn export_json_meta(&mut self, name: &str, value: &serde_json::Value) -> table::TermId {
        let value = serde_json::to_string(value).expect("json values are always serializable");
        let value = self.make_term(model::Literal::Str(value.into()).into());
        let name = self.make_term(model::Literal::Str(name.into()).into());
        self.make_term_apply(model::COMPAT_META_JSON, &[name, value])
    }

    fn resolve_symbol(&mut self, name: &CoreSymbol) -> table::NodeId {
        let result = self.symbols.resolve(name);

        if let Ok(node) = result {
            return node;
        }

        // NOTE: We do not use the entry API here in order to avoid allocating
        // a new `SymbolName` when the import already exists.
        if let Some(node) = self.implicit_imports.get(name) {
            return *node;
        }

        let node = self.module.insert_node(table::Node {
            operation: table::Operation::Import(name.clone().into()),
            ..table::Node::default()
        });
        self.implicit_imports.insert(name.clone(), node);
        node
    }

    fn make_term_apply(&mut self, name: &'static str, args: &[table::TermId]) -> table::TermId {
        let name = self
            .static_symbols
            .entry(name)
            .or_insert_with(|| {
                let (ext, name) = name.rsplit_once(".").unwrap();
                let ext = IdentList::new_unchecked(ext);
                let name = name.to_smolstr();
                CoreSymbol::Qualified(ext, name)
            })
            .clone();

        let symbol = self.resolve_symbol(&name);
        let args = self.bump.alloc_slice_copy(args);
        self.make_term(table::Term::Apply(symbol, args))
    }
}

type FxIndexSet<T> = indexmap::IndexSet<T, FxBuildHasher>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CoreSymbol {
    Qualified(ExtensionId, SmolStr),
    Local(SmolStr),
}

impl From<CoreSymbol> for model::SymbolName {
    fn from(value: CoreSymbol) -> Self {
        Self::new(match value {
            CoreSymbol::Qualified(ext, name) => format_smolstr!("{}.{}", ext, name),
            CoreSymbol::Local(name) => name,
        })
    }
}

/// Data structure for translating the edges between ports in the `Hugr` graph
/// into the hypergraph representation used by `hugr_model`.
struct Links {
    /// Scoping helper that keeps track of the current nesting of regions
    /// and translates the group of connected ports into a link index.
    scope: model::scope::LinkTable<u32>,

    /// A mapping from each port to the group of connected ports it belongs to.
    groups: FxHashMap<(Node, Port), u32>,
}

impl Links {
    /// Create the `Links` data structure from a `Hugr` graph by recording the
    /// connectivity of the ports.
    pub fn new(hugr: &Hugr) -> Self {
        let scope = model::scope::LinkTable::new();

        // We collect all ports that are in the hugr into an index set so that
        // we have an association between the port and a numeric index.
        let node_ports: FxIndexSet<(Node, Port)> = hugr
            .nodes()
            .flat_map(|node| hugr.all_node_ports(node).map(move |port| (node, port)))
            .collect();

        // We then use a union-find data structure to group together all ports that are connected.
        let mut uf = UnionFind::<u32>::new(node_ports.len());

        for (i, (node, port)) in node_ports.iter().enumerate() {
            if let Ok(port) = port.as_incoming() {
                for (other_node, other_port) in hugr.linked_outputs(*node, port) {
                    let other_port = Port::from(other_port);
                    let j = node_ports.get_index_of(&(other_node, other_port)).unwrap();
                    uf.union(i as u32, j as u32);
                }
            }
        }

        // We then collect the association between the port and the group of connected ports it belongs to.
        let groups = node_ports
            .into_iter()
            .enumerate()
            .map(|(i, node_port)| (node_port, uf.find(i as u32)))
            .collect();

        Self { scope, groups }
    }

    /// Enter an isolated region.
    pub fn enter(&mut self, region: table::RegionId) {
        self.scope.enter(region);
    }

    /// Leave an isolated region, returning the number of links and ports in the region.
    ///
    /// # Panics
    ///
    /// Panics if there is no remaining open scope to exit.
    pub fn exit(&mut self) -> (u32, u32) {
        self.scope.exit()
    }

    /// Obtain the link index for a node and port.
    ///
    /// # Panics
    ///
    /// Panics if the port does not exist in the [`Hugr`] that was passed to `[Self::new]`.
    pub fn use_link(&mut self, node: Node, port: impl Into<Port>) -> table::LinkIndex {
        let port = port.into();
        let group = self.groups[&(node, port)];
        self.scope.use_link(group)
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
        use hugr_model::v0::bumpalo::Bump;
        let bump = Bump::new();
        let _model = super::export_hugr(&hugr, &bump);
    }
}
