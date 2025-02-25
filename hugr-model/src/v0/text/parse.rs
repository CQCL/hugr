use base64::{prelude::BASE64_STANDARD, Engine};
use bumpalo::{
    collections::{String as BumpString, Vec as BumpVec},
    Bump,
};
use fxhash::FxHashMap;
use pest::{
    iterators::{Pair, Pairs},
    Parser, RuleType,
};
use thiserror::Error;

use crate::v0::{
    scope::{LinkTable, SymbolTable, UnknownSymbolError, VarTable},
    ExtSetPart, LinkIndex, ListPart, Module, Node, NodeId, Operation, Param, Region, RegionId,
    RegionKind, RegionScope, ScopeClosure, Symbol, Term, TermId, TuplePart,
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
        let str_slice = pair.as_str();
        let mut inner = pair.into_inner();

        let term =
            match rule {
                Rule::term_wildcard => Term::Wildcard,

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

                    Term::Apply(symbol, self.bump.alloc_slice_copy(&args))
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

                    Term::List(parts.into_bump_slice())
                }

                Rule::term_tuple => {
                    let mut parts = BumpVec::with_capacity_in(inner.len(), self.bump);

                    for token in inner {
                        match token.as_rule() {
                            Rule::term => parts.push(TuplePart::Item(self.parse_term(token)?)),
                            Rule::spliced_term => {
                                let term_token = token.into_inner().next().unwrap();
                                parts.push(TuplePart::Splice(self.parse_term(term_token)?))
                            }
                            _ => unreachable!(),
                        }
                    }

                    Term::Tuple(parts.into_bump_slice())
                }

                Rule::term_str => {
                    let value = self.parse_string(inner.next().unwrap())?;
                    Term::Str(value)
                }

                Rule::term_nat => {
                    let value = str_slice.trim().parse().unwrap();
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

                    Term::ExtSet(parts.into_bump_slice())
                }

                Rule::term_const_func => {
                    let region = self.parse_region(inner.next().unwrap(), ScopeClosure::Closed)?;
                    Term::ConstFunc(region)
                }

                Rule::term_bytes => {
                    let token = inner.next().unwrap();
                    let slice = token.as_str();
                    // Remove the quotes
                    let slice = &slice[1..slice.len() - 1];
                    let data = BASE64_STANDARD.decode(slice).map_err(|_| {
                        ParseError::custom("invalid base64 encoding", token.as_span())
                    })?;
                    let data = self.bump.alloc_slice_copy(&data);
                    Term::Bytes(data)
                }

                Rule::term_float => {
                    let value: f64 = str_slice.trim().parse().unwrap();
                    Term::Float(value.into())
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
                let regions = self.parse_regions(&mut inner, ScopeClosure::Open)?;
                Node {
                    operation: Operation::Dfg,
                    inputs,
                    outputs,
                    params: &[],
                    regions,
                    meta,
                    signature,
                }
            }

            Rule::node_cfg => {
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let regions = self.parse_regions(&mut inner, ScopeClosure::Open)?;
                Node {
                    operation: Operation::Cfg,
                    inputs,
                    outputs,
                    params: &[],
                    regions,
                    meta,
                    signature,
                }
            }

            Rule::node_block => {
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let regions = self.parse_regions(&mut inner, ScopeClosure::Open)?;
                Node {
                    operation: Operation::Block,
                    inputs,
                    outputs,
                    params: &[],
                    regions,
                    meta,
                    signature,
                }
            }

            Rule::node_define_func => {
                self.vars.enter(node);
                let symbol = self.parse_func_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                let regions = self.parse_regions(&mut inner, ScopeClosure::Closed)?;
                self.vars.exit();
                Node {
                    operation: Operation::DefineFunc(symbol),
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    regions,
                    meta,
                    signature: None,
                }
            }

            Rule::node_declare_func => {
                self.vars.enter(node);
                let symbol = self.parse_func_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                self.vars.exit();
                Node {
                    operation: Operation::DeclareFunc(symbol),
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    regions: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_define_alias => {
                self.vars.enter(node);
                let symbol = self.parse_alias_header(inner.next().unwrap())?;
                let value = self.parse_term(inner.next().unwrap())?;
                let params = self.bump.alloc_slice_copy(&[value]);
                let meta = self.parse_meta(&mut inner)?;
                self.vars.exit();
                Node {
                    operation: Operation::DefineAlias(symbol),
                    inputs: &[],
                    outputs: &[],
                    params,
                    regions: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_declare_alias => {
                self.vars.enter(node);
                let symbol = self.parse_alias_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                self.vars.exit();
                Node {
                    operation: Operation::DeclareAlias(symbol),
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    regions: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_custom => {
                let op = inner.next().unwrap();
                debug_assert!(matches!(op.as_rule(), Rule::term_apply));
                let mut op_inner = op.into_inner();

                let operation = self.parse_symbol_use(&mut op_inner)?;

                let mut params = Vec::new();

                for token in filter_rule(&mut op_inner, Rule::term) {
                    params.push(self.parse_term(token)?);
                }

                let operation = Operation::Custom(operation);
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let regions = self.parse_regions(&mut inner, ScopeClosure::Closed)?;
                Node {
                    operation,
                    inputs,
                    outputs,
                    params: self.bump.alloc_slice_copy(&params),
                    regions,
                    meta,
                    signature,
                }
            }

            Rule::node_tail_loop => {
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let regions = self.parse_regions(&mut inner, ScopeClosure::Open)?;
                Node {
                    operation: Operation::TailLoop,
                    inputs,
                    outputs,
                    params: &[],
                    regions,
                    meta,
                    signature,
                }
            }

            Rule::node_cond => {
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let regions = self.parse_regions(&mut inner, ScopeClosure::Open)?;
                Node {
                    operation: Operation::Conditional,
                    inputs,
                    outputs,
                    params: &[],
                    regions,
                    meta,
                    signature,
                }
            }

            Rule::node_declare_ctr => {
                self.vars.enter(node);
                let symbol = self.parse_ctr_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                self.vars.exit();
                Node {
                    operation: Operation::DeclareConstructor(symbol),
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    regions: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_declare_operation => {
                self.vars.enter(node);
                let symbol = self.parse_op_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                self.vars.exit();
                Node {
                    operation: Operation::DeclareOperation(symbol),
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    regions: &[],
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
        closure: ScopeClosure,
    ) -> ParseResult<&'a [RegionId]> {
        let mut regions = Vec::new();
        for pair in filter_rule(pairs, Rule::region) {
            regions.push(self.parse_region(pair, closure)?);
        }
        Ok(self.bump.alloc_slice_copy(&regions))
    }

    fn parse_region(
        &mut self,
        pair: Pair<'a, Rule>,
        closure: ScopeClosure,
    ) -> ParseResult<RegionId> {
        debug_assert_eq!(pair.as_rule(), Rule::region);
        let pair = pair.into_inner().next().unwrap();
        let rule = pair.as_rule();
        let mut inner = pair.into_inner();

        let region = self.module.insert_region(Region::default());
        self.symbols.enter(region);

        if closure == ScopeClosure::Closed {
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

        let scope = match closure {
            ScopeClosure::Closed => {
                let (links, ports) = self.links.exit();
                Some(RegionScope { links, ports })
            }
            ScopeClosure::Open => None,
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

    fn parse_func_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a Symbol<'a>> {
        debug_assert_eq!(pair.as_rule(), Rule::func_header);

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;
        let constraints = self.parse_constraints(&mut inner)?;
        let signature = self.parse_term(inner.next().unwrap())?;

        Ok(self.bump.alloc(Symbol {
            name,
            params,
            constraints,
            signature,
        }))
    }

    fn parse_alias_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a Symbol<'a>> {
        debug_assert_eq!(pair.as_rule(), Rule::alias_header);

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;
        let signature = self.parse_term(inner.next().unwrap())?;

        Ok(self.bump.alloc(Symbol {
            name,
            params,
            constraints: &[],
            signature,
        }))
    }

    fn parse_ctr_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a Symbol<'a>> {
        debug_assert_eq!(pair.as_rule(), Rule::ctr_header);

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;
        let constraints = self.parse_constraints(&mut inner)?;
        let signature = self.parse_term(inner.next().unwrap())?;

        Ok(self.bump.alloc(Symbol {
            name,
            params,
            constraints,
            signature,
        }))
    }

    fn parse_op_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a Symbol<'a>> {
        debug_assert_eq!(pair.as_rule(), Rule::operation_header);

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;
        let constraints = self.parse_constraints(&mut inner)?;
        let signature = self.parse_term(inner.next().unwrap())?;

        Ok(self.bump.alloc(Symbol {
            name,
            params,
            constraints,
            signature,
        }))
    }

    fn parse_params(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [Param<'a>]> {
        let mut params = Vec::new();

        for pair in filter_rule(pairs, Rule::param) {
            let param_span = pair.as_span();
            let mut inner = pair.into_inner();
            let name = &inner.next().unwrap().as_str()[1..];
            let r#type = self.parse_term(inner.next().unwrap())?;
            let param = Param { name, r#type };

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

    fn parse_meta(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [TermId]> {
        let mut items = Vec::new();

        for meta in filter_rule(pairs, Rule::meta) {
            let mut inner = meta.into_inner();
            let value = self.parse_term(inner.next().unwrap())?;
            items.push(value)
        }

        Ok(self.bump.alloc_slice_copy(&items))
    }

    fn parse_symbol_use(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<NodeId> {
        let name = self.parse_symbol(pairs)?;
        self.use_symbol(name)
    }

    fn use_symbol(&mut self, name: &'a str) -> ParseResult<NodeId> {
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
