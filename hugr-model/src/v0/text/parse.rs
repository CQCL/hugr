use bumpalo::{collections::String as BumpString, collections::Vec as BumpVec, Bump};
use fxhash::FxHashMap;
use pest::{
    iterators::{Pair, Pairs},
    Parser, RuleType,
};
use thiserror::Error;

use crate::v0::{
    scope::{LinkTable, SymbolTable, UnknownSymbolError, VarTable},
    AliasDecl, ConstructorDecl, ExtSetPart, FuncDecl, LinkIndex, ListPart, MetaItem, Module, Node,
    NodeId, Operation, OperationDecl, Param, ParamSort, Region, RegionId, RegionKind, RegionScope,
    Term, TermId,
};

mod pest_parser {
    use pest_derive::Parser;

    // NOTE: The pest derive macro generates a `Rule` enum. We do not want this to be
    // part of the public API, and so we hide it within this private module.

    #[derive(Parser)]
    #[grammar = "v0/text/hugr.pest"]
    pub struct HugrParser;
}

use pest_parser::{HugrParser, Rule};

/// A parsed HUGR module.
///
/// This consists of the module itself, together with additional information that was
/// extracted from the text format.
#[derive(Debug, Clone)]
pub struct ParsedModule<'a> {
    /// The parsed module.
    pub module: Module<'a>,
    // TODO: Spans
}

/// Parses a HUGR module from its text representation.
pub fn parse<'a>(input: &'a str, bump: &'a Bump) -> Result<ParsedModule<'a>, ParseError> {
    let mut context = ParseContext::new(bump);
    let mut pairs =
        HugrParser::parse(Rule::module, input).map_err(|err| ParseError(Box::new(err)))?;
    context.parse_module(pairs.next().unwrap())?;

    Ok(ParsedModule {
        module: context.module,
    })
}

struct ParseContext<'a> {
    module: Module<'a>,
    bump: &'a Bump,
    vars: VarTable<'a>,
    links: LinkTable<&'a str>,
    symbols: SymbolTable<'a>,
    implicit_imports: FxHashMap<&'a str, NodeId>,
}

impl<'a> ParseContext<'a> {
    fn new(bump: &'a Bump) -> Self {
        Self {
            module: Module::default(),
            symbols: SymbolTable::default(),
            links: LinkTable::default(),
            vars: VarTable::default(),
            implicit_imports: FxHashMap::default(),
            bump,
        }
    }

    fn parse_module(&mut self, pair: Pair<'a, Rule>) -> ParseResult<()> {
        debug_assert_eq!(pair.as_rule(), Rule::module);
        let mut inner = pair.into_inner();

        self.module.root = self.module.insert_region(Region::default());
        self.symbols.enter(self.module.root);
        self.links.enter(self.module.root);

        // TODO: What scope does the metadata live in?
        let meta = self.parse_meta(&mut inner)?;
        let explicit_children = self.parse_nodes(&mut inner)?;

        let mut children = BumpVec::with_capacity_in(
            explicit_children.len() + self.implicit_imports.len(),
            self.bump,
        );
        children.extend(explicit_children);
        children.extend(self.implicit_imports.drain().map(|(_, node)| node));
        let children = children.into_bump_slice();

        let (link_count, port_count) = self.links.exit();
        self.symbols.exit();

        self.module.regions[self.module.root.index()] = Region {
            kind: RegionKind::Module,
            sources: &[],
            targets: &[],
            children,
            meta,
            signature: None,
            scope: Some(RegionScope {
                links: link_count,
                ports: port_count,
            }),
        };

        Ok(())
    }

    fn parse_term(&mut self, pair: Pair<'a, Rule>) -> ParseResult<TermId> {
        debug_assert_eq!(pair.as_rule(), Rule::term);
        let pair = pair.into_inner().next().unwrap();
        let rule = pair.as_rule();
        let mut inner = pair.into_inner();

        let term =
            match rule {
                Rule::term_wildcard => Term::Wildcard,
                Rule::term_type => Term::Type,
                Rule::term_static => Term::StaticType,
                Rule::term_constraint => Term::Constraint,
                Rule::term_str_type => Term::StrType,
                Rule::term_nat_type => Term::NatType,
                Rule::term_ctrl_type => Term::ControlType,
                Rule::term_ext_set_type => Term::ExtSetType,

                Rule::term_var => {
                    let name_token = inner.next().unwrap();
                    let name = name_token.as_str();

                    let var = self.vars.resolve(name).map_err(|err| {
                        ParseError::custom(&err.to_string(), name_token.as_span())
                    })?;

                    Term::Var(var)
                }

                Rule::term_apply => {
                    let symbol = self.parse_symbol_use(&mut inner)?;
                    let mut args = Vec::new();

                    for token in inner {
                        args.push(self.parse_term(token)?);
                    }

                    Term::Apply {
                        symbol,
                        args: self.bump.alloc_slice_copy(&args),
                    }
                }

                Rule::term_apply_full => {
                    let symbol = self.parse_symbol_use(&mut inner)?;
                    let mut args = Vec::new();

                    for token in inner {
                        args.push(self.parse_term(token)?);
                    }

                    Term::ApplyFull {
                        symbol,
                        args: self.bump.alloc_slice_copy(&args),
                    }
                }

                Rule::term_const => {
                    let r#type = self.parse_term(inner.next().unwrap())?;
                    Term::Const { r#type }
                }

                Rule::term_list => {
                    let mut parts = BumpVec::with_capacity_in(inner.len(), self.bump);

                    for token in inner {
                        match token.as_rule() {
                            Rule::term => parts.push(ListPart::Item(self.parse_term(token)?)),
                            Rule::spliced_term => {
                                let term_token = token.into_inner().next().unwrap();
                                parts.push(ListPart::Splice(self.parse_term(term_token)?))
                            }
                            _ => unreachable!(),
                        }
                    }

                    Term::List {
                        parts: parts.into_bump_slice(),
                    }
                }

                Rule::term_list_type => {
                    let item_type = self.parse_term(inner.next().unwrap())?;
                    Term::ListType { item_type }
                }

                Rule::term_str => {
                    let value = self.parse_string(inner.next().unwrap())?;
                    Term::Str(value)
                }

                Rule::term_nat => {
                    let value = inner.next().unwrap().as_str().parse().unwrap();
                    Term::Nat(value)
                }

                Rule::term_ext_set => {
                    let mut parts = BumpVec::with_capacity_in(inner.len(), self.bump);

                    for token in inner {
                        match token.as_rule() {
                            Rule::ext_name => parts
                                .push(ExtSetPart::Extension(self.bump.alloc_str(token.as_str()))),
                            Rule::spliced_term => {
                                let term_token = token.into_inner().next().unwrap();
                                parts.push(ExtSetPart::Splice(self.parse_term(term_token)?))
                            }
                            _ => unreachable!(),
                        }
                    }

                    Term::ExtSet {
                        parts: parts.into_bump_slice(),
                    }
                }

                Rule::term_adt => {
                    let variants = self.parse_term(inner.next().unwrap())?;
                    Term::Adt { variants }
                }

                Rule::term_func_type => {
                    let inputs = self.parse_term(inner.next().unwrap())?;
                    let outputs = self.parse_term(inner.next().unwrap())?;
                    let extensions = self.parse_term(inner.next().unwrap())?;
                    Term::FuncType {
                        inputs,
                        outputs,
                        extensions,
                    }
                }

                Rule::term_ctrl => {
                    let values = self.parse_term(inner.next().unwrap())?;
                    Term::Control { values }
                }

                Rule::term_non_linear => {
                    let term = self.parse_term(inner.next().unwrap())?;
                    Term::NonLinearConstraint { term }
                }

                r => unreachable!("term: {:?}", r),
            };

        Ok(self.module.insert_term(term))
    }

    fn parse_node_shallow(&mut self, pair: Pair<'a, Rule>) -> ParseResult<NodeId> {
        debug_assert_eq!(pair.as_rule(), Rule::node);
        let pair = pair.into_inner().next().unwrap();
        let span = pair.as_span();
        let rule = pair.as_rule();
        let mut inner = pair.into_inner();

        let symbol = match rule {
            Rule::node_define_func => {
                let mut func_header = inner.next().unwrap().into_inner();
                Some(self.parse_symbol(&mut func_header)?)
            }
            Rule::node_declare_func => {
                let mut func_header = inner.next().unwrap().into_inner();
                Some(self.parse_symbol(&mut func_header)?)
            }
            Rule::node_define_alias => {
                let mut alias_header = inner.next().unwrap().into_inner();
                Some(self.parse_symbol(&mut alias_header)?)
            }
            Rule::node_declare_alias => {
                let mut alias_header = inner.next().unwrap().into_inner();
                Some(self.parse_symbol(&mut alias_header)?)
            }
            Rule::node_declare_ctr => {
                let mut ctr_header = inner.next().unwrap().into_inner();
                Some(self.parse_symbol(&mut ctr_header)?)
            }
            Rule::node_declare_operation => {
                let mut op_header = inner.next().unwrap().into_inner();
                Some(self.parse_symbol(&mut op_header)?)
            }
            Rule::node_import => Some(self.parse_symbol(&mut inner)?),
            _ => None,
        };

        let node = self.module.insert_node(Node::default());

        if let Some(symbol) = symbol {
            self.symbols
                .insert(symbol, node)
                .map_err(|err| ParseError::custom(&err.to_string(), span))?;
        }

        Ok(node)
    }

    fn parse_node_deep(&mut self, pair: Pair<'a, Rule>, node: NodeId) -> ParseResult<Node<'a>> {
        debug_assert_eq!(pair.as_rule(), Rule::node);
        let pair = pair.into_inner().next().unwrap();
        let rule = pair.as_rule();

        let mut inner = pair.into_inner();

        let node = match rule {
            Rule::node_dfg => {
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let body = self.parse_region(&mut inner, false)?;
                Node {
                    operation: Operation::Dfg { body },
                    inputs,
                    outputs,
                    params: &[],
                    meta,
                    signature,
                }
            }

            Rule::node_cfg => {
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let body = self.parse_region(&mut inner, false)?;
                Node {
                    operation: Operation::Cfg { body },
                    inputs,
                    outputs,
                    params: &[],
                    meta,
                    signature,
                }
            }

            Rule::node_block => {
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let body = self.parse_region(&mut inner, false)?;
                Node {
                    operation: Operation::Block { body },
                    inputs,
                    outputs,
                    params: &[],
                    meta,
                    signature,
                }
            }

            Rule::node_define_func => {
                self.vars.enter(node);
                let decl = self.parse_func_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                let body = self.parse_region(&mut inner, true)?;
                self.vars.exit();
                Node {
                    operation: Operation::DefineFunc { decl, body },
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_declare_func => {
                self.vars.enter(node);
                let decl = self.parse_func_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                self.vars.exit();
                Node {
                    operation: Operation::DeclareFunc { decl },
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_call_func => {
                let func = self.parse_term(inner.next().unwrap())?;
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                Node {
                    operation: Operation::CallFunc { func },
                    inputs,
                    outputs,
                    params: &[],
                    meta,
                    signature,
                }
            }

            Rule::node_load_func => {
                let func = self.parse_term(inner.next().unwrap())?;
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                Node {
                    operation: Operation::LoadFunc { func },
                    inputs,
                    outputs,
                    params: &[],
                    meta,
                    signature,
                }
            }

            Rule::node_define_alias => {
                self.vars.enter(node);
                let decl = self.parse_alias_header(inner.next().unwrap())?;
                let value = self.parse_term(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                self.vars.exit();
                Node {
                    operation: Operation::DefineAlias { decl, value },
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_declare_alias => {
                self.vars.enter(node);
                let decl = self.parse_alias_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                self.vars.exit();
                Node {
                    operation: Operation::DeclareAlias { decl },
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_custom => {
                let op = inner.next().unwrap();
                debug_assert!(matches!(
                    op.as_rule(),
                    Rule::term_apply | Rule::term_apply_full
                ));
                let op_rule = op.as_rule();
                let mut op_inner = op.into_inner();

                let operation = self.parse_symbol_use(&mut op_inner)?;

                let mut params = Vec::new();

                for token in filter_rule(&mut inner, Rule::term) {
                    params.push(self.parse_term(token)?);
                }

                let operation = match op_rule {
                    Rule::term_apply_full => Operation::CustomFull { operation },
                    Rule::term_apply => Operation::Custom { operation },
                    _ => unreachable!(),
                };

                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                Node {
                    operation,
                    inputs,
                    outputs,
                    params: self.bump.alloc_slice_copy(&params),
                    meta,
                    signature,
                }
            }

            Rule::node_tail_loop => {
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let body = self.parse_region(&mut inner, true)?;
                Node {
                    operation: Operation::TailLoop { body },
                    inputs,
                    outputs,
                    params: &[],
                    meta,
                    signature,
                }
            }

            Rule::node_cond => {
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let branches = self.parse_regions(&mut inner, false)?;
                Node {
                    operation: Operation::Conditional { branches },
                    inputs,
                    outputs,
                    params: &[],
                    meta,
                    signature,
                }
            }

            Rule::node_tag => {
                let tag = inner.next().unwrap().as_str().parse::<u16>().unwrap();
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                Node {
                    operation: Operation::Tag { tag },
                    inputs,
                    outputs,
                    params: &[],
                    meta,
                    signature,
                }
            }

            Rule::node_declare_ctr => {
                self.vars.enter(node);
                let decl = self.parse_ctr_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                self.vars.exit();
                Node {
                    operation: Operation::DeclareConstructor { decl },
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_declare_operation => {
                self.vars.enter(node);
                let decl = self.parse_op_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                self.vars.exit();
                Node {
                    operation: Operation::DeclareOperation { decl },
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    meta,
                    signature: None,
                }
            }

            _ => unreachable!(),
        };

        Ok(node)
    }

    fn parse_regions(
        &mut self,
        pairs: &mut Pairs<'a, Rule>,
        closed: bool,
    ) -> ParseResult<&'a [RegionId]> {
        let mut regions = Vec::new();

        while let Some(pair) = pairs.peek() {
            if pair.as_rule() != Rule::region {
                break;
            }

            regions.push(self.parse_region(pairs, closed)?);
        }

        Ok(self.bump.alloc_slice_copy(&regions))
    }

    fn parse_region(&mut self, pairs: &mut Pairs<'a, Rule>, closed: bool) -> ParseResult<RegionId> {
        let pair = pairs.next().unwrap();
        debug_assert_eq!(pair.as_rule(), Rule::region);
        let pair = pair.into_inner().next().unwrap();
        let rule = pair.as_rule();
        let mut inner = pair.into_inner();

        let region = self.module.insert_region(Region::default());
        self.symbols.enter(region);

        if closed {
            self.links.enter(region);
        }

        let kind = match rule {
            Rule::region_cfg => RegionKind::ControlFlow,
            Rule::region_dfg => RegionKind::DataFlow,
            _ => unreachable!(),
        };

        let sources = self.parse_port_list(&mut inner)?;
        let targets = self.parse_port_list(&mut inner)?;
        let signature = self.parse_signature(&mut inner)?;
        let meta = self.parse_meta(&mut inner)?;
        let children = self.parse_nodes(&mut inner)?;

        let scope = if closed {
            let (links, ports) = self.links.exit();
            Some(RegionScope { links, ports })
        } else {
            None
        };

        self.symbols.exit();

        self.module.regions[region.index()] = Region {
            kind,
            sources,
            targets,
            children,
            meta,
            signature,
            scope,
        };

        Ok(region)
    }

    fn parse_nodes(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [NodeId]> {
        let nodes = {
            let mut pairs = pairs.clone();
            let mut nodes = BumpVec::with_capacity_in(pairs.len(), self.bump);

            for pair in filter_rule(&mut pairs, Rule::node) {
                nodes.push(self.parse_node_shallow(pair)?);
            }

            nodes.into_bump_slice()
        };

        for (i, pair) in filter_rule(pairs, Rule::node).enumerate() {
            let node = nodes[i];
            let node_data = self.parse_node_deep(pair, node)?;
            self.module.nodes[node.index()] = node_data;
        }

        Ok(nodes)
    }

    fn parse_func_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a FuncDecl<'a>> {
        debug_assert_eq!(pair.as_rule(), Rule::func_header);

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;
        let constraints = self.parse_constraints(&mut inner)?;

        let inputs = self.parse_term(inner.next().unwrap())?;
        let outputs = self.parse_term(inner.next().unwrap())?;
        let extensions = self.parse_term(inner.next().unwrap())?;

        // Assemble the inputs, outputs and extensions into a function type.
        let func = self.module.insert_term(Term::FuncType {
            inputs,
            outputs,
            extensions,
        });

        Ok(self.bump.alloc(FuncDecl {
            name,
            params,
            constraints,
            signature: func,
        }))
    }

    fn parse_alias_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a AliasDecl<'a>> {
        debug_assert_eq!(pair.as_rule(), Rule::alias_header);

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;
        let r#type = self.parse_term(inner.next().unwrap())?;

        Ok(self.bump.alloc(AliasDecl {
            name,
            params,
            r#type,
        }))
    }

    fn parse_ctr_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a ConstructorDecl<'a>> {
        debug_assert_eq!(pair.as_rule(), Rule::ctr_header);

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;
        let constraints = self.parse_constraints(&mut inner)?;
        let r#type = self.parse_term(inner.next().unwrap())?;

        Ok(self.bump.alloc(ConstructorDecl {
            name,
            params,
            constraints,
            r#type,
        }))
    }

    fn parse_op_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a OperationDecl<'a>> {
        debug_assert_eq!(pair.as_rule(), Rule::operation_header);

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;
        let constraints = self.parse_constraints(&mut inner)?;
        let r#type = self.parse_term(inner.next().unwrap())?;

        Ok(self.bump.alloc(OperationDecl {
            name,
            params,
            constraints,
            r#type,
        }))
    }

    fn parse_params(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [Param<'a>]> {
        let mut params = Vec::new();

        for pair in filter_rule(pairs, Rule::param) {
            let param = pair.into_inner().next().unwrap();
            let param_span = param.as_span();

            let param = match param.as_rule() {
                Rule::param_implicit => {
                    let mut inner = param.into_inner();
                    let name = &inner.next().unwrap().as_str()[1..];
                    let r#type = self.parse_term(inner.next().unwrap())?;
                    Param {
                        name,
                        r#type,
                        sort: ParamSort::Implicit,
                    }
                }
                Rule::param_explicit => {
                    let mut inner = param.into_inner();
                    let name = &inner.next().unwrap().as_str()[1..];
                    let r#type = self.parse_term(inner.next().unwrap())?;
                    Param {
                        name,
                        r#type,
                        sort: ParamSort::Explicit,
                    }
                }
                _ => unreachable!(),
            };

            self.vars
                .insert(param.name)
                .map_err(|err| ParseError::custom(&err.to_string(), param_span))?;

            params.push(param);
        }

        Ok(self.bump.alloc_slice_copy(&params))
    }

    fn parse_constraints(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [TermId]> {
        let mut constraints = Vec::new();

        for pair in filter_rule(pairs, Rule::where_clause) {
            let constraint = self.parse_term(pair.into_inner().next().unwrap())?;
            constraints.push(constraint);
        }

        Ok(self.bump.alloc_slice_copy(&constraints))
    }

    fn parse_signature(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<Option<TermId>> {
        let Some(Rule::signature) = pairs.peek().map(|p| p.as_rule()) else {
            return Ok(None);
        };

        let pair = pairs.next().unwrap();
        let signature = self.parse_term(pair.into_inner().next().unwrap())?;
        Ok(Some(signature))
    }

    fn parse_port_list(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [LinkIndex]> {
        let Some(Rule::port_list) = pairs.peek().map(|p| p.as_rule()) else {
            return Ok(&[]);
        };

        let pair = pairs.next().unwrap();
        let inner = pair.into_inner();
        let mut links = BumpVec::with_capacity_in(inner.len(), self.bump);

        for token in inner {
            links.push(self.parse_port(token)?);
        }

        Ok(links.into_bump_slice())
    }

    fn parse_port(&mut self, pair: Pair<'a, Rule>) -> ParseResult<LinkIndex> {
        debug_assert_eq!(pair.as_rule(), Rule::port);
        let mut inner = pair.into_inner();
        let name = &inner.next().unwrap().as_str()[1..];
        Ok(self.links.use_link(name))
    }

    fn parse_meta(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [MetaItem<'a>]> {
        let mut items = Vec::new();

        for meta in filter_rule(pairs, Rule::meta) {
            let mut inner = meta.into_inner();
            let name = self.parse_symbol(&mut inner)?;
            let value = self.parse_term(inner.next().unwrap())?;
            items.push(MetaItem { name, value })
        }

        Ok(self.bump.alloc_slice_copy(&items))
    }

    fn parse_symbol_use(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<NodeId> {
        let name = self.parse_symbol(pairs)?;
        let resolved = self.symbols.resolve(name);

        Ok(match resolved {
            Ok(node) => node,
            Err(UnknownSymbolError(_)) => *self.implicit_imports.entry(name).or_insert_with(|| {
                self.module.insert_node(Node {
                    operation: Operation::Import { name },
                    ..Node::default()
                })
            }),
        })
    }

    fn parse_symbol(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a str> {
        let pair = pairs.next().unwrap();
        if let Rule::symbol = pair.as_rule() {
            Ok(pair.as_str())
        } else {
            unreachable!("expected a symbol");
        }
    }

    fn parse_string(&self, token: Pair<'a, Rule>) -> ParseResult<&'a str> {
        debug_assert_eq!(token.as_rule(), Rule::string);

        // Any escape sequence is longer than the character it represents.
        // Therefore the length of this token (minus 2 for the quotes on either
        // side) is an upper bound for the length of the string.
        let capacity = token.as_str().len() - 2;
        let mut string = BumpString::with_capacity_in(capacity, self.bump);
        let tokens = token.into_inner();

        for token in tokens {
            match token.as_rule() {
                Rule::string_raw => string.push_str(token.as_str()),
                Rule::string_escape => match token.as_str().chars().nth(1).unwrap() {
                    '"' => string.push('"'),
                    '\\' => string.push('\\'),
                    'n' => string.push('\n'),
                    'r' => string.push('\r'),
                    't' => string.push('\t'),
                    _ => unreachable!(),
                },
                Rule::string_unicode => {
                    let token_str = token.as_str();
                    debug_assert_eq!(&token_str[0..3], r"\u{");
                    debug_assert_eq!(&token_str[token_str.len() - 1..], "}");
                    let code_str = &token_str[3..token_str.len() - 1];
                    let code = u32::from_str_radix(code_str, 16).map_err(|_| {
                        ParseError::custom("invalid unicode escape sequence", token.as_span())
                    })?;
                    let char = std::char::from_u32(code).ok_or_else(|| {
                        ParseError::custom("invalid unicode code point", token.as_span())
                    })?;
                    string.push(char);
                }
                _ => unreachable!(),
            }
        }

        Ok(string.into_bump_str())
    }
}

/// Draw from a pest pair iterator only the pairs that match a given rule.
///
/// This is similar to a `take_while`, except that it does not take the iterator
/// by value and so lets us continue using it after the filter.
#[inline]
fn filter_rule<'a, 'i, R: RuleType>(
    pairs: &'a mut Pairs<'i, R>,
    rule: R,
) -> impl Iterator<Item = Pair<'i, R>> + 'a {
    std::iter::from_fn(move || {
        let peek = pairs.peek()?;
        if peek.as_rule() == rule {
            Some(pairs.next().unwrap())
        } else {
            None
        }
    })
}

/// An error that occurred during parsing.
#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub struct ParseError(Box<pest::error::Error<Rule>>);

impl ParseError {
    /// Line of the error in the input string.
    pub fn line(&self) -> usize {
        use pest::error::LineColLocation;
        match self.0.line_col {
            LineColLocation::Pos((line, _)) => line,
            LineColLocation::Span((line, _), _) => line,
        }
    }

    /// Column of the error in the input string.
    pub fn column(&self) -> usize {
        use pest::error::LineColLocation;
        match self.0.line_col {
            LineColLocation::Pos((_, col)) => col,
            LineColLocation::Span((_, col), _) => col,
        }
    }

    /// Location of the error in the input string in bytes.
    pub fn location(&self) -> usize {
        use pest::error::InputLocation;
        match self.0.location {
            InputLocation::Pos(offset) => offset,
            InputLocation::Span((offset, _)) => offset,
        }
    }

    fn custom(message: &str, span: pest::Span) -> Self {
        let error = pest::error::Error::new_from_span(
            pest::error::ErrorVariant::CustomError {
                message: message.to_string(),
            },
            span,
        );
        ParseError(Box::new(error))
    }
}

// NOTE: `ParseError` does not implement `From<pest::error::Error<Rule>>` so that
// pest does not become part of the public API.

type ParseResult<T> = Result<T, ParseError>;
