//! Exporting HUGR graphs to their `hugr-model` representation.
use crate::{
    extension::ExtensionSet,
    ops::OpType,
    types::{
        type_param::{TypeArgVariable, TypeParam},
        type_row::TypeRowBase,
        CustomType, EdgeKind, FuncTypeBase, MaybeRV, PolyFuncTypeBase, RowVariable, SumType,
        TypeArg, TypeBase, TypeEnum,
    },
    Direction, Hugr, HugrView, IncomingPort, Node, Port, PortIndex,
};
use bumpalo::{collections::Vec as BumpVec, Bump};
use hugr_model::v0::{self as model};
use indexmap::IndexSet;
use smol_str::ToSmolStr;

pub(crate) const OP_FUNC_CALL_INDIRECT: &str = "func.call-indirect";
const TERM_PARAM_TUPLE: &str = "param.tuple";

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
    links: IndexSet<(Node, Port)>,
    bump: &'a Bump,
}

impl<'a> Context<'a> {
    pub fn new(hugr: &'a Hugr, bump: &'a Bump) -> Self {
        let mut module = model::Module::default();
        module.nodes.reserve(hugr.node_count());

        Self {
            hugr,
            module,
            bump,
            links: IndexSet::new(),
        }
    }

    pub fn export_root(&mut self) {
        let r#type = self.module.insert_term(model::Term::Wildcard);

        let hugr_children = self.hugr.children(self.hugr.root());
        let mut children = BumpVec::with_capacity_in(hugr_children.len(), self.bump);

        for child in self.hugr.children(self.hugr.root()) {
            children.push(self.export_node(child));
        }

        let root = self.module.insert_region(model::Region {
            kind: model::RegionKind::DataFlow,
            sources: &[],
            targets: &[],
            children: children.into_bump_slice(),
            meta: &[],
            signature: r#type,
        });

        self.module.root = root;
    }

    /// Returns the edge id for a given port, creating a new edge if necessary.
    ///
    /// Any two ports that are linked will be represented by the same link.
    fn get_link_id(&mut self, node: Node, port: Port) -> model::LinkId {
        // To ensure that linked ports are represented by the same edge, we take the minimum port
        // among all the linked ports, including the one we started with.
        let linked_ports = self.hugr.linked_ports(node, port);
        let all_ports = std::iter::once((node, port)).chain(linked_ports);
        let repr = all_ports.min().unwrap();
        let edge = self.links.insert_full(repr).0 as _;
        model::LinkId(edge)
    }

    pub fn make_ports(&mut self, node: Node, direction: Direction) -> &'a [model::Port<'a>] {
        let ports = self.hugr.node_ports(node, direction);
        let mut model_ports = BumpVec::with_capacity_in(ports.len(), self.bump);

        for port in ports {
            if let Some(model_port) = self.make_port(node, port) {
                model_ports.push(model_port);
            }
        }

        model_ports.into_bump_slice()
    }

    pub fn make_port(&mut self, node: Node, port: impl Into<Port>) -> Option<model::Port<'a>> {
        let port: Port = port.into();
        let op_type = self.hugr.get_optype(node);

        let r#type = match op_type.port_kind(port)? {
            EdgeKind::ControlFlow => {
                // TODO: This should ideally be reported by the op itself
                let types: Vec<_> = match (op_type, port.direction()) {
                    (OpType::DataflowBlock(block), Direction::Incoming) => {
                        block.inputs.iter().map(|t| self.export_type(t)).collect()
                    }
                    (OpType::DataflowBlock(block), Direction::Outgoing) => {
                        let mut types = Vec::new();
                        types.extend(
                            block.sum_rows[port.index()]
                                .iter()
                                .map(|t| self.export_type(t)),
                        );
                        types.extend(block.other_outputs.iter().map(|t| self.export_type(t)));
                        types
                    }
                    (OpType::ExitBlock(block), Direction::Incoming) => block
                        .cfg_outputs
                        .iter()
                        .map(|t| self.export_type(t))
                        .collect(),
                    (OpType::ExitBlock(_), Direction::Outgoing) => vec![],
                    _ => unreachable!("unexpected control flow port on non-control-flow op"),
                };

                let types = self.bump.alloc_slice_copy(&types);
                let values = self.module.insert_term(model::Term::List {
                    items: types,
                    tail: None,
                });
                self.module.insert_term(model::Term::Control { values })
            }
            EdgeKind::Value(r#type) => self.export_type(&r#type),
            EdgeKind::Const(_) => return None,
            EdgeKind::Function(_) => return None,
            EdgeKind::StateOrder => return None,
        };

        let link = model::LinkRef::Id(self.get_link_id(node, port));

        Some(model::Port {
            r#type: Some(r#type),
            link,
        })
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

    pub fn export_node(&mut self, node: Node) -> model::NodeId {
        let inputs = self.make_ports(node, Direction::Incoming);
        let outputs = self.make_ports(node, Direction::Outgoing);
        let mut params: &[_] = &[];
        let mut regions: &[_] = &[];

        fn make_custom(name: &'static str) -> model::Operation {
            model::Operation::Custom {
                name: model::GlobalRef::Named(name),
            }
        }

        let operation = match self.hugr.get_optype(node) {
            OpType::Module(_) => todo!("this should be an error"),

            OpType::Input(_) => {
                panic!("input nodes should have been handled by the region export")
            }

            OpType::Output(_) => {
                panic!("output nodes should have been handled by the region export")
            }

            OpType::DFG(_) => {
                regions = self.bump.alloc_slice_copy(&[self.export_dfg(node)]);
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

            OpType::DataflowBlock(_) => {
                regions = self.bump.alloc_slice_copy(&[self.export_dfg(node)]);
                model::Operation::Block
            }

            OpType::FuncDefn(func) => {
                let name = self.get_func_name(node).unwrap();
                let (params, func) = self.export_poly_func_type(&func.signature);
                let decl = self.bump.alloc(model::FuncDecl {
                    name,
                    params,
                    signature: func,
                });
                regions = self.bump.alloc_slice_copy(&[self.export_dfg(node)]);
                model::Operation::DefineFunc { decl }
            }

            OpType::FuncDecl(func) => {
                let name = self.get_func_name(node).unwrap();
                let (params, func) = self.export_poly_func_type(&func.signature);
                let decl = self.bump.alloc(model::FuncDecl {
                    name,
                    params,
                    signature: func,
                });
                model::Operation::DeclareFunc { decl }
            }

            OpType::AliasDecl(alias) => {
                // TODO: We should support aliases with different types and with parameters
                let r#type = self.module.insert_term(model::Term::Type);
                let decl = self.bump.alloc(model::AliasDecl {
                    name: &alias.name,
                    params: &[],
                    r#type,
                });
                model::Operation::DeclareAlias { decl }
            }

            OpType::AliasDefn(alias) => {
                let value = self.export_type(&alias.definition);
                // TODO: We should support aliases with different types and with parameters
                let r#type = self.module.insert_term(model::Term::Type);
                let decl = self.bump.alloc(model::AliasDecl {
                    name: &alias.name,
                    params: &[],
                    r#type,
                });
                model::Operation::DefineAlias { decl, value }
            }

            OpType::Call(call) => {
                // TODO: If the node is not connected to a function, we should do better than panic.
                let node = self.connected_function(node).unwrap();
                let name = model::GlobalRef::Named(self.get_func_name(node).unwrap());

                let mut args = BumpVec::new_in(self.bump);
                args.extend(call.type_args.iter().map(|arg| self.export_type_arg(arg)));
                let args = args.into_bump_slice();

                let func = self
                    .module
                    .insert_term(model::Term::ApplyFull { global: name, args });
                model::Operation::CallFunc { func }
            }

            OpType::LoadFunction(load) => {
                // TODO: If the node is not connected to a function, we should do better than panic.
                let node = self.connected_function(node).unwrap();
                let name = model::GlobalRef::Named(self.get_func_name(node).unwrap());

                let mut args = BumpVec::new_in(self.bump);
                args.extend(load.type_args.iter().map(|arg| self.export_type_arg(arg)));
                let args = args.into_bump_slice();

                let func = self
                    .module
                    .insert_term(model::Term::ApplyFull { global: name, args });
                model::Operation::LoadFunc { func }
            }

            OpType::Const(_) => todo!("Export const nodes?"),
            OpType::LoadConstant(_) => todo!("Export load constant?"),

            OpType::CallIndirect(_) => make_custom(OP_FUNC_CALL_INDIRECT),

            OpType::Tag(tag) => model::Operation::Tag { tag: tag.tag as _ },

            OpType::TailLoop(_) => {
                regions = self.bump.alloc_slice_copy(&[self.export_dfg(node)]);
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
                let name =
                    self.bump
                        .alloc_str(&format!("{}.{}", op.def().extension(), op.def().name()));
                let name = model::GlobalRef::Named(name);

                params = self
                    .bump
                    .alloc_slice_fill_iter(op.args().iter().map(|arg| self.export_type_arg(arg)));

                if let Some(region) = self.export_dfg_if_present(node) {
                    regions = self.bump.alloc_slice_copy(&[region]);
                }

                model::Operation::Custom { name }
            }

            OpType::OpaqueOp(op) => {
                let name = self
                    .bump
                    .alloc_str(&format!("{}.{}", op.extension(), op.op_name()));
                let name = model::GlobalRef::Named(name);

                params = self
                    .bump
                    .alloc_slice_fill_iter(op.args().iter().map(|arg| self.export_type_arg(arg)));

                if let Some(region) = self.export_dfg_if_present(node) {
                    regions = self.bump.alloc_slice_copy(&[region]);
                }

                model::Operation::Custom { name }
            }
        };

        let r#type = self.module.insert_term(model::Term::Wildcard);

        self.module.insert_node(model::Node {
            operation,
            inputs,
            outputs,
            params,
            regions,
            meta: &[],
            signature: r#type,
        })
    }

    /// Create a region from the given node's children, if it has any.
    ///
    /// See [`Self::export_dfg`].
    pub fn export_dfg_if_present(&mut self, node: Node) -> Option<model::RegionId> {
        if self.hugr.children(node).next().is_none() {
            None
        } else {
            Some(self.export_dfg(node))
        }
    }

    /// Creates a data flow region from the given node's children.
    ///
    /// `Input` and `Output` nodes are used to determine the source and target ports of the region.
    pub fn export_dfg(&mut self, node: Node) -> model::RegionId {
        let mut children = self.hugr.children(node);

        // The first child is an `Input` node, which we use to determine the region's sources.
        let input_node = children.next().unwrap();
        assert!(matches!(self.hugr.get_optype(input_node), OpType::Input(_)));
        let sources = self.make_ports(input_node, Direction::Outgoing);

        // The second child is an `Output` node, which we use to determine the region's targets.
        let output_node = children.next().unwrap();
        assert!(matches!(
            self.hugr.get_optype(output_node),
            OpType::Output(_)
        ));
        let targets = self.make_ports(output_node, Direction::Incoming);

        // Export the remaining children of the node.
        let mut region_children = BumpVec::with_capacity_in(children.len(), self.bump);

        for child in children {
            region_children.push(self.export_node(child));
        }

        // TODO: We can determine the type of the region
        let r#type = self.module.insert_term(model::Term::Wildcard);

        self.module.insert_region(model::Region {
            kind: model::RegionKind::DataFlow,
            sources,
            targets,
            children: region_children.into_bump_slice(),
            meta: &[],
            signature: r#type,
        })
    }

    /// Creates a control flow region from the given node's children.
    pub fn export_cfg(&mut self, node: Node) -> model::RegionId {
        let mut children = self.hugr.children(node);
        let mut region_children = BumpVec::with_capacity_in(children.len() + 1, self.bump);

        // The first child is the entry block.
        // We create a source port on the control flow region and connect it to the
        // first input port of the exported entry block.
        let entry_block = children.next().unwrap();

        assert!(matches!(
            self.hugr.get_optype(entry_block),
            OpType::DataflowBlock(_)
        ));

        let source = self.make_port(entry_block, IncomingPort::from(0)).unwrap();
        region_children.push(self.export_node(entry_block));

        // Export the remaining children of the node, except for the last one.
        for _ in 0..children.len() - 1 {
            region_children.push(self.export_node(children.next().unwrap()));
        }

        // The last child is the exit block.
        // Contrary to the entry block, the exit block does not have a dataflow subgraph.
        // We therefore do not export the block itself, but simply use its output ports
        // as the target ports of the control flow region.
        let exit_block = children.next().unwrap();

        assert!(matches!(
            self.hugr.get_optype(exit_block),
            OpType::ExitBlock(_)
        ));

        let targets = self.make_ports(exit_block, Direction::Incoming);

        // TODO: We can determine the type of the region
        let r#type = self.module.insert_term(model::Term::Wildcard);

        self.module.insert_region(model::Region {
            kind: model::RegionKind::ControlFlow,
            sources: self.bump.alloc_slice_copy(&[source]),
            targets,
            children: region_children.into_bump_slice(),
            meta: &[],
            signature: r#type,
        })
    }

    /// Export the `Case` node children of a `Conditional` node as data flow regions.
    pub fn export_conditional_regions(&mut self, node: Node) -> &'a [model::RegionId] {
        let children = self.hugr.children(node);
        let mut regions = BumpVec::with_capacity_in(children.len(), self.bump);

        for child in children {
            assert!(matches!(self.hugr.get_optype(child), OpType::Case(_)));
            regions.push(self.export_dfg(child));
        }

        regions.into_bump_slice()
    }

    pub fn export_poly_func_type<RV: MaybeRV>(
        &mut self,
        t: &PolyFuncTypeBase<RV>,
    ) -> (&'a [model::Param<'a>], model::TermId) {
        let mut params = BumpVec::with_capacity_in(t.params().len(), self.bump);

        for (i, param) in t.params().iter().enumerate() {
            let name = self.bump.alloc_str(&i.to_string());
            let r#type = self.export_type_param(param);
            let param = model::Param::Implicit { name, r#type };
            params.push(param)
        }

        let body = self.export_func_type(t.body());

        (params.into_bump_slice(), body)
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
                self.module
                    .insert_term(model::Term::ApplyFull { global: name, args })
            }
            TypeEnum::Function(func) => self.export_func_type(func),
            TypeEnum::Variable(index, _) => {
                // This ignores the type bound for now
                self.module
                    .insert_term(model::Term::Var(model::LocalRef::Index(*index as _)))
            }
            TypeEnum::RowVar(rv) => self.export_row_var(rv.as_rv()),
            TypeEnum::Sum(sum) => self.export_sum_type(sum),
        }
    }

    pub fn export_func_type<RV: MaybeRV>(&mut self, t: &FuncTypeBase<RV>) -> model::TermId {
        let inputs = self.export_type_row(t.input());
        let outputs = self.export_type_row(t.output());
        let extensions = self.export_ext_set(&t.extension_reqs);
        self.module.insert_term(model::Term::FuncType {
            inputs,
            outputs,
            extensions,
        })
    }

    pub fn export_custom_type(&mut self, t: &CustomType) -> model::TermId {
        let name = format!("{}.{}", t.extension(), t.name());
        let name = model::GlobalRef::Named(self.bump.alloc_str(&name));

        let args = self
            .bump
            .alloc_slice_fill_iter(t.args().iter().map(|p| self.export_type_arg(p)));
        let term = model::Term::ApplyFull { global: name, args };
        self.module.insert_term(term)
    }

    pub fn export_type_arg(&mut self, t: &TypeArg) -> model::TermId {
        match t {
            TypeArg::Type { ty } => self.export_type(ty),
            TypeArg::BoundedNat { n } => self.module.insert_term(model::Term::Nat(*n)),
            TypeArg::String { arg } => self
                .module
                .insert_term(model::Term::Str(self.bump.alloc_str(arg))),
            TypeArg::Sequence { elems } => {
                // For now we assume that the sequence is meant to be a list.
                let items = self
                    .bump
                    .alloc_slice_fill_iter(elems.iter().map(|elem| self.export_type_arg(elem)));
                self.module
                    .insert_term(model::Term::List { items, tail: None })
            }
            TypeArg::Extensions { es } => self.export_ext_set(es),
            TypeArg::Variable { v } => self.export_type_arg_var(v),
        }
    }

    pub fn export_type_arg_var(&mut self, var: &TypeArgVariable) -> model::TermId {
        self.module
            .insert_term(model::Term::Var(model::LocalRef::Index(var.index() as _)))
    }

    pub fn export_row_var(&mut self, t: &RowVariable) -> model::TermId {
        self.module
            .insert_term(model::Term::Var(model::LocalRef::Index(t.0 as _)))
    }

    pub fn export_sum_type(&mut self, t: &SumType) -> model::TermId {
        match t {
            SumType::Unit { size } => {
                let items = self.bump.alloc_slice_fill_iter((0..*size).map(|_| {
                    self.module.insert_term(model::Term::List {
                        items: &[],
                        tail: None,
                    })
                }));
                let list = model::Term::List { items, tail: None };
                let variants = self.module.insert_term(list);
                self.module.insert_term(model::Term::Adt { variants })
            }
            SumType::General { rows } => {
                let items = self
                    .bump
                    .alloc_slice_fill_iter(rows.iter().map(|row| self.export_type_row(row)));
                let list = model::Term::List { items, tail: None };
                let variants = { self.module.insert_term(list) };
                self.module.insert_term(model::Term::Adt { variants })
            }
        }
    }

    pub fn export_type_row<RV: MaybeRV>(&mut self, t: &TypeRowBase<RV>) -> model::TermId {
        let mut items = BumpVec::with_capacity_in(t.len(), self.bump);
        items.extend(t.iter().map(|row| self.export_type(row)));
        let items = items.into_bump_slice();
        self.module
            .insert_term(model::Term::List { items, tail: None })
    }

    pub fn export_type_param(&mut self, t: &TypeParam) -> model::TermId {
        match t {
            // This ignores the type bound for now.
            TypeParam::Type { .. } => self.module.insert_term(model::Term::Type),
            // This ignores the type bound for now.
            TypeParam::BoundedNat { .. } => self.module.insert_term(model::Term::NatType),
            TypeParam::String => self.module.insert_term(model::Term::StrType),
            TypeParam::List { param } => {
                let item_type = self.export_type_param(param);
                self.module.insert_term(model::Term::ListType { item_type })
            }
            TypeParam::Tuple { params } => {
                let items = self.bump.alloc_slice_fill_iter(
                    params.iter().map(|param| self.export_type_param(param)),
                );
                let types = self
                    .module
                    .insert_term(model::Term::List { items, tail: None });
                self.module.insert_term(model::Term::ApplyFull {
                    global: model::GlobalRef::Named(TERM_PARAM_TUPLE),
                    args: self.bump.alloc_slice_copy(&[types]),
                })
            }
            TypeParam::Extensions => {
                let term = model::Term::ExtSetType;
                self.module.insert_term(term)
            }
        }
    }

    pub fn export_ext_set(&mut self, t: &ExtensionSet) -> model::TermId {
        // Extension sets with variables are encoded using a hack: a variable in the
        // extension set is represented by converting its index into a string.
        // Until we have a better representation for extension sets, we therefore
        // need to try and parse each extension as a number to determine if it is
        // a variable or an extension.
        let mut extensions = Vec::new();
        let mut variables = Vec::new();

        for ext in t.iter() {
            if let Ok(index) = ext.parse::<usize>() {
                variables.push({
                    self.module
                        .insert_term(model::Term::Var(model::LocalRef::Index(index as _)))
                });
            } else {
                extensions.push(ext.to_smolstr());
            }
        }

        // Extension sets in the model support at most one variable. This is a
        // deliberate limitation so that extension sets behave like polymorphic rows.
        // The type theory of such rows and how to apply them to model (co)effects
        // is well understood.
        //
        // Extension sets in `hugr-core` at this point have no such restriction.
        // However, it appears that so far we never actually use extension sets with
        // multiple variables, except for extension sets that are generated through
        // property testing.
        let rest = match variables.as_slice() {
            [] => None,
            [var] => Some(*var),
            _ => {
                // TODO: We won't need this anymore once we have a core representation
                // that ensures that extension sets have at most one variable.
                panic!("Extension set with multiple variables")
            }
        };

        let mut extensions = BumpVec::with_capacity_in(extensions.len(), self.bump);
        extensions.extend(t.iter().map(|ext| self.bump.alloc_str(ext) as &str));
        let extensions = extensions.into_bump_slice();

        self.module
            .insert_term(model::Term::ExtSet { extensions, rest })
    }
}

#[cfg(test)]
mod test {
    use rstest::{fixture, rstest};

    use crate::{
        builder::{Dataflow, DataflowSubContainer},
        extension::prelude::QB_T,
        std_extensions::arithmetic::float_types,
        type_row,
        types::Signature,
        utils::test_quantum_extension::{self, cx_gate, h_gate},
        Hugr,
    };

    #[fixture]
    fn test_simple_circuit() -> Hugr {
        crate::builder::test::build_main(
            Signature::new_endo(type_row![QB_T, QB_T])
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
