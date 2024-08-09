//! Text syntax.
use crate::{
    extension::ExtensionSet,
    ops::{NamedOp as _, OpType},
    types::{
        type_param::{TypeArgVariable, TypeParam},
        type_row::TypeRowBase,
        CustomType, FuncTypeBase, MaybeRV, PolyFuncTypeBase, RowVariable, SumType, TypeArg,
        TypeBase, TypeEnum,
    },
    Direction, Hugr, HugrView, Node, Port,
};
use hugr_model::v0 as model;
use indexmap::IndexMap;
use smol_str::ToSmolStr;
use tinyvec::TinyVec;

/// Export a [`Hugr`] graph to its representation in the model.
pub fn export(hugr: &Hugr) -> model::Module {
    let mut context = Context::new(hugr);
    hugr.root().export(&mut context);
    context.module
}

/// State for converting a HUGR graph to its representation in the model.
struct Context<'a> {
    /// The HUGR graph to convert.
    hugr: &'a Hugr,
    module: model::Module,
    /// Mapping from ports to edge indices.
    /// This only includes the minimum port among groups of linked ports.
    edges: IndexMap<(Node, Port), usize>,
}

impl<'a> Context<'a> {
    pub fn new(hugr: &'a Hugr) -> Self {
        Self {
            hugr,
            module: model::Module::default(),
            edges: IndexMap::new(),
        }
    }

    pub fn make_term(&mut self, term: model::Term) -> model::TermId {
        let index = self.module.terms.len();
        self.module.terms.push(term);
        model::TermId(index as _)
    }

    pub fn make_node(&mut self, node: model::Node) -> model::NodeId {
        let index = self.module.terms.len();
        self.module.nodes.push(node);
        model::NodeId(index as _)
    }

    pub fn make_port(&mut self, node: Node, port: impl Into<Port>) -> model::PortId {
        let port = port.into();
        let index = self.module.ports.len();
        let port_id = model::PortId(index as _);
        let r#type = self.make_term(model::Term::Wildcard); // TODO
        self.module.ports.push(model::Port {
            r#type,
            meta: Vec::new(),
        });

        // To ensure that linked ports are represented by the same edge, we take the minimum port
        // among all the linked ports, including the one we started with.
        let linked_ports = self.hugr.linked_ports(node, port);
        let all_ports = std::iter::once((node, port)).chain(linked_ports);
        let repr = all_ports.min().unwrap();

        let edge_id = *self.edges.entry(repr).or_insert_with(|| {
            let edge_id = self.module.edges.len();
            let edge = model::Edge::default();
            self.module.edges.push(edge);
            edge_id
        });

        match port.direction() {
            Direction::Incoming => self.module.edges[edge_id].inputs.push(port_id),
            Direction::Outgoing => self.module.edges[edge_id].outputs.push(port_id),
        }

        port_id
    }

    /// Get the name of the function associated with a node that takes a static input
    /// connected to a function definition or declaration. Returns `None` otherwise.
    fn get_func_name(&self, node: Node) -> Option<model::Symbol> {
        let port = self.hugr.node_inputs(node).last()?;
        let (defn_node, _) = self.hugr.linked_outputs(node, port).next()?;
        match self.hugr.get_optype(defn_node) {
            OpType::FuncDecl(func_decl) => Some(model::Symbol(func_decl.name())),
            OpType::FuncDefn(func_defn) => Some(model::Symbol(func_defn.name())),
            _ => None,
        }
    }
}

/// Trait for core types that can be exported into the model format.
trait Export {
    /// The target type to export to.
    type Target;

    /// Export the value into the target type.
    fn export(&self, ctx: &mut Context) -> Self::Target;
}

impl Export for Node {
    type Target = model::NodeId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        let mut params = TinyVec::new();

        let inputs = ctx
            .hugr
            .node_inputs(*self)
            .map(|port| ctx.make_port(*self, port))
            .collect();

        let outputs = ctx
            .hugr
            .node_outputs(*self)
            .map(|port| ctx.make_port(*self, port))
            .collect();

        let children = ctx
            .hugr
            .children(*self)
            .map(|child| child.export(ctx))
            .collect();

        fn make_custom(name: &'static str) -> model::Operation {
            model::Operation::Custom(model::operation::Custom {
                name: model::Symbol(name.to_smolstr()),
            })
        }

        let operation = match ctx.hugr.get_optype(*self) {
            OpType::Module(_) => model::Operation::Module,
            OpType::Input(_) => model::Operation::Input,
            OpType::Output(_) => model::Operation::Output,
            OpType::DFG(_) => model::Operation::Dfg,
            OpType::CFG(_) => model::Operation::Cfg,
            OpType::ExitBlock(_) => model::Operation::Exit,
            OpType::Case(_) => model::Operation::Case,
            OpType::DataflowBlock(_) => model::Operation::Block,

            OpType::FuncDefn(func) => {
                // TODO: If the node is not connected to a function, we should do better than panic.
                let name = ctx.get_func_name(*self).unwrap();
                let r#type = Box::new(func.signature.export(ctx));
                model::Operation::DefineFunc(model::operation::DefineFunc { name, r#type })
            }

            OpType::FuncDecl(func) => {
                // TODO: If the node is not connected to a function, we should do better than panic.
                let name = ctx.get_func_name(*self).unwrap();
                let r#type = Box::new(func.signature.export(ctx));
                model::Operation::DeclareFunc(model::operation::DeclareFunc { name, r#type })
            }

            OpType::AliasDecl(alias) => {
                let name = model::Symbol(alias.name().to_smolstr());
                // TODO: We should support aliases with different types
                let r#type = ctx.make_term(model::Term::Type);
                model::Operation::DeclareAlias(model::operation::DeclareAlias { name, r#type })
            }

            OpType::AliasDefn(alias) => {
                let name = model::Symbol(alias.name().to_smolstr());
                let value = alias.definition.export(ctx);
                model::Operation::DefineAlias(model::operation::DefineAlias { name, value })
            }

            OpType::Call(call) => {
                // TODO: If the node is not connected to a function, we should do better than panic.
                let name = ctx.get_func_name(*self).unwrap();
                params.extend(call.type_args.iter().map(|arg| arg.export(ctx)));
                model::Operation::CallFunc(model::operation::CallFunc { name })
            }

            OpType::LoadFunction(func) => {
                // TODO: If the node is not connected to a function, we should do better than panic.
                let name = ctx.get_func_name(*self).unwrap();
                params.extend(func.type_args.iter().map(|arg| arg.export(ctx)));
                model::Operation::LoadFunc(model::operation::LoadFunc { name })
            }

            OpType::Const(_) => todo!("Export const nodes?"),
            OpType::LoadConstant(_) => todo!("Export load constant?"),

            OpType::CallIndirect(_) => make_custom("core.call-indirect"),
            OpType::Noop(_) => make_custom("core.id"),
            OpType::MakeTuple(_) => make_custom("core.make-tuple"),
            OpType::UnpackTuple(_) => make_custom("core.unpack-tuple"),
            OpType::Tag(_) => make_custom("core.make-tagged"),
            OpType::Lift(_) => make_custom("core.lift"),
            OpType::TailLoop(_) => make_custom("core.tail-loop"),
            OpType::Conditional(_) => make_custom("core.cond"),
            OpType::ExtensionOp(op) => {
                let name = model::Symbol(op.name());
                params.extend(op.args().iter().map(|arg| arg.export(ctx)));
                model::Operation::Custom(model::operation::Custom { name })
            }
            OpType::OpaqueOp(op) => {
                let name = model::Symbol(op.name());
                params.extend(op.args().iter().map(|arg| arg.export(ctx)));
                model::Operation::Custom(model::operation::Custom { name })
            }
        };

        ctx.make_node(model::Node {
            operation,
            params,
            inputs,
            outputs,
            children,
            meta: Vec::new(),
        })
    }
}

impl<RV: MaybeRV> Export for PolyFuncTypeBase<RV> {
    type Target = model::Scheme;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        let params = self
            .params()
            .iter()
            .enumerate()
            .map(|(i, param)| model::SchemeParam {
                name: model::TermVar(i.to_smolstr()),
                r#type: param.export(ctx),
            })
            .collect();
        let constraints = TinyVec::new();
        let body = self.body().export(ctx);
        model::Scheme {
            params,
            constraints,
            body,
        }
    }
}

impl<RV: MaybeRV> Export for TypeBase<RV> {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        self.as_type_enum().export(ctx)
    }
}

impl<RV: MaybeRV> Export for TypeEnum<RV> {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        match self {
            TypeEnum::Extension(ext) => ext.export(ctx),
            TypeEnum::Alias(alias) => {
                let name = model::Symbol(alias.name().to_smolstr());
                let args = TinyVec::new();
                ctx.make_term(model::Term::Named(model::term::Named { name, args }))
            }
            TypeEnum::Function(func) => func.export(ctx),
            TypeEnum::Variable(index, _) => {
                // This ignores the type bound for now
                ctx.make_term(model::Term::Var(model::TermVar(index.to_smolstr())))
            }
            TypeEnum::RowVar(rv) => rv.as_rv().export(ctx),
            TypeEnum::Sum(sum) => sum.export(ctx),
        }
    }
}

impl<RV: MaybeRV> Export for FuncTypeBase<RV> {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        let inputs = self.input().export(ctx);
        let outputs = self.output().export(ctx);
        let extensions = self.extension_reqs.export(ctx);
        ctx.make_term(model::Term::FuncType(model::term::FuncType {
            inputs,
            outputs,
            extensions,
        }))
    }
}

impl Export for CustomType {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        let name = model::Symbol(format!("{}.{}", self.extension(), self.name()).into());
        let args = self.args().iter().map(|arg| arg.export(ctx)).collect();
        ctx.make_term(model::Term::Named(model::term::Named { name, args }))
    }
}

impl Export for TypeArg {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        match self {
            TypeArg::Type { ty } => ty.export(ctx),
            TypeArg::BoundedNat { n } => ctx.make_term(model::Term::Nat(*n)),
            TypeArg::String { arg } => ctx.make_term(model::Term::Str(arg.into())),
            TypeArg::Sequence { elems } => {
                // For now we assume that the sequence is meant to be a list.
                let items = elems.iter().map(|elem| elem.export(ctx)).collect();
                ctx.make_term(model::Term::List(model::term::List { items, tail: None }))
            }
            TypeArg::Extensions { es } => es.export(ctx),
            TypeArg::Variable { v } => v.export(ctx),
        }
    }
}

impl Export for TypeArgVariable {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        ctx.make_term(model::Term::Var(model::TermVar(self.index().to_smolstr())))
    }
}

impl Export for RowVariable {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        ctx.make_term(model::Term::Var(model::TermVar(self.0.to_smolstr())))
    }
}

impl Export for SumType {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        match self {
            SumType::Unit { size } => {
                let items = (0..*size)
                    .map(|_| {
                        let types = ctx.make_term(model::Term::List(model::term::List {
                            items: TinyVec::new(),
                            tail: None,
                        }));
                        ctx.make_term(model::Term::ProductType(model::term::ProductType { types }))
                    })
                    .collect();
                let types =
                    ctx.make_term(model::Term::List(model::term::List { items, tail: None }));
                ctx.make_term(model::Term::SumType(model::term::SumType { types }))
            }
            SumType::General { rows } => {
                let items = rows.iter().map(|row| row.export(ctx)).collect();
                let types =
                    ctx.make_term(model::Term::List(model::term::List { items, tail: None }));
                ctx.make_term(model::Term::SumType(model::term::SumType { types }))
            }
        }
    }
}

impl<RV: MaybeRV> Export for TypeRowBase<RV> {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        let items = self
            .as_slice()
            .iter()
            .map(|item| item.export(ctx))
            .collect();
        ctx.make_term(model::Term::List(model::term::List { items, tail: None }))
    }
}

impl Export for TypeParam {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        match self {
            // This ignores the type bound for now.
            TypeParam::Type { .. } => ctx.make_term(model::Term::Type),
            // This ignores the type bound for now.
            TypeParam::BoundedNat { .. } => ctx.make_term(model::Term::NatType),
            TypeParam::String => ctx.make_term(model::Term::StrType),
            TypeParam::List { param } => {
                let item_type = param.export(ctx);
                ctx.make_term(model::Term::ListType(model::term::ListType { item_type }))
            }
            TypeParam::Tuple { params } => {
                let items = params.iter().map(|param| param.export(ctx)).collect();
                let types =
                    ctx.make_term(model::Term::List(model::term::List { items, tail: None }));
                ctx.make_term(model::Term::ProductType(model::term::ProductType { types }))
            }
            TypeParam::Extensions => ctx.make_term(model::Term::ExtSetType),
        }
    }
}

impl Export for ExtensionSet {
    type Target = model::TermId;

    fn export(&self, ctx: &mut Context) -> Self::Target {
        // This ignores that type variables in the extension set are represented
        // by converting their index into a string. We should probably have a deeper
        // look at how extension sets are represented.
        let extensions = self.iter().map(|ext| ext.to_smolstr()).collect();
        ctx.make_term(model::Term::ExtSet(model::term::ExtSet {
            extensions,
            rest: None,
        }))
    }
}
