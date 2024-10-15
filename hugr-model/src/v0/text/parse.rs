use bumpalo::Bump;
use pest::{
    iterators::{Pair, Pairs},
    Parser, RuleType,
};
use thiserror::Error;

use crate::v0::{
    AliasDecl, ConstructorDecl, FuncDecl, GlobalRef, LinkRef, LocalRef, MetaItem, Module, Node,
    NodeId, Operation, OperationDecl, Param, Region, RegionId, RegionKind, Term, TermId,
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
}

impl<'a> ParseContext<'a> {
    fn new(bump: &'a Bump) -> Self {
        Self {
            module: Module::default(),
            bump,
        }
    }

    fn parse_module(&mut self, pair: Pair<'a, Rule>) -> ParseResult<()> {
        debug_assert!(matches!(pair.as_rule(), Rule::module));
        let mut inner = pair.into_inner();
        let meta = self.parse_meta(&mut inner)?;

        let children = self.parse_nodes(&mut inner)?;

        let root_region = self.module.insert_region(Region {
            kind: RegionKind::DataFlow,
            sources: &[],
            targets: &[],
            children,
            meta,
            signature: None,
        });

        self.module.root = root_region;

        Ok(())
    }

    fn parse_term(&mut self, pair: Pair<'a, Rule>) -> ParseResult<TermId> {
        debug_assert!(matches!(pair.as_rule(), Rule::term));
        let pair = pair.into_inner().next().unwrap();
        let rule = pair.as_rule();
        let mut inner = pair.into_inner();

        let term = match rule {
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
                Term::Var(LocalRef::Named(name))
            }

            Rule::term_apply => {
                let name = GlobalRef::Named(self.parse_symbol(&mut inner)?);
                let mut args = Vec::new();

                for token in inner {
                    args.push(self.parse_term(token)?);
                }

                Term::Apply {
                    global: name,
                    args: self.bump.alloc_slice_copy(&args),
                }
            }

            Rule::term_apply_full => {
                let name = GlobalRef::Named(self.parse_symbol(&mut inner)?);
                let mut args = Vec::new();

                for token in inner {
                    args.push(self.parse_term(token)?);
                }

                Term::ApplyFull {
                    global: name,
                    args: self.bump.alloc_slice_copy(&args),
                }
            }

            Rule::term_quote => {
                let r#type = self.parse_term(inner.next().unwrap())?;
                Term::Quote { r#type }
            }

            Rule::term_list => {
                let mut items = Vec::new();
                let mut tail = None;

                for token in filter_rule(&mut inner, Rule::term) {
                    items.push(self.parse_term(token)?);
                }

                if inner.next().is_some() {
                    let token = inner.next().unwrap();
                    tail = Some(self.parse_term(token)?);
                }

                Term::List {
                    items: self.bump.alloc_slice_copy(&items),
                    tail,
                }
            }

            Rule::term_list_type => {
                let item_type = self.parse_term(inner.next().unwrap())?;
                Term::ListType { item_type }
            }

            Rule::term_str => {
                // TODO: Escaping?
                let value = inner.next().unwrap().as_str();
                let value = &value[1..value.len() - 1];
                Term::Str(value)
            }

            Rule::term_nat => {
                let value = inner.next().unwrap().as_str().parse().unwrap();
                Term::Nat(value)
            }

            Rule::term_ext_set => {
                let mut extensions = Vec::new();
                let mut rest = None;

                for token in filter_rule(&mut inner, Rule::ext_name) {
                    extensions.push(token.as_str());
                }

                if inner.next().is_some() {
                    let token = inner.next().unwrap();
                    rest = Some(self.parse_term(token)?);
                }

                Term::ExtSet {
                    extensions: self.bump.alloc_slice_copy(&extensions),
                    rest,
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

            r => unreachable!("term: {:?}", r),
        };

        Ok(self.module.insert_term(term))
    }

    fn parse_node(&mut self, pair: Pair<'a, Rule>) -> ParseResult<NodeId> {
        debug_assert!(matches!(pair.as_rule(), Rule::node));
        let pair = pair.into_inner().next().unwrap();
        let rule = pair.as_rule();

        let mut inner = pair.into_inner();

        let node = match rule {
            Rule::node_dfg => {
                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let regions = self.parse_regions(&mut inner)?;
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
                let regions = self.parse_regions(&mut inner)?;
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
                let regions = self.parse_regions(&mut inner)?;
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
                let decl = self.parse_func_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                let regions = self.parse_regions(&mut inner)?;
                Node {
                    operation: Operation::DefineFunc { decl },
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    regions,
                    meta,
                    signature: None,
                }
            }

            Rule::node_declare_func => {
                let decl = self.parse_func_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                Node {
                    operation: Operation::DeclareFunc { decl },
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    regions: &[],
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
                    regions: &[],
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
                    regions: &[],
                    meta,
                    signature,
                }
            }

            Rule::node_define_alias => {
                let decl = self.parse_alias_header(inner.next().unwrap())?;
                let value = self.parse_term(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                Node {
                    operation: Operation::DefineAlias { decl, value },
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    regions: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_declare_alias => {
                let decl = self.parse_alias_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                Node {
                    operation: Operation::DeclareAlias { decl },
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
                debug_assert!(matches!(
                    op.as_rule(),
                    Rule::term_apply | Rule::term_apply_full
                ));
                let op_rule = op.as_rule();
                let mut op_inner = op.into_inner();

                let name = GlobalRef::Named(self.parse_symbol(&mut op_inner)?);

                let mut params = Vec::new();

                for token in filter_rule(&mut inner, Rule::term) {
                    params.push(self.parse_term(token)?);
                }

                let operation = match op_rule {
                    Rule::term_apply_full => Operation::CustomFull { operation: name },
                    Rule::term_apply => Operation::Custom { operation: name },
                    _ => unreachable!(),
                };

                let inputs = self.parse_port_list(&mut inner)?;
                let outputs = self.parse_port_list(&mut inner)?;
                let signature = self.parse_signature(&mut inner)?;
                let meta = self.parse_meta(&mut inner)?;
                let regions = self.parse_regions(&mut inner)?;
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
                let regions = self.parse_regions(&mut inner)?;
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
                let regions = self.parse_regions(&mut inner)?;
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
                    regions: &[],
                    meta,
                    signature,
                }
            }

            Rule::node_declare_ctr => {
                let decl = self.parse_ctr_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                Node {
                    operation: Operation::DeclareConstructor { decl },
                    inputs: &[],
                    outputs: &[],
                    params: &[],
                    regions: &[],
                    meta,
                    signature: None,
                }
            }

            Rule::node_declare_operation => {
                let decl = self.parse_op_header(inner.next().unwrap())?;
                let meta = self.parse_meta(&mut inner)?;
                Node {
                    operation: Operation::DeclareOperation { decl },
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

        let node_id = self.module.insert_node(node);

        Ok(node_id)
    }

    fn parse_regions(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [RegionId]> {
        let mut regions = Vec::new();
        for pair in filter_rule(pairs, Rule::region) {
            regions.push(self.parse_region(pair)?);
        }
        Ok(self.bump.alloc_slice_copy(&regions))
    }

    fn parse_region(&mut self, pair: Pair<'a, Rule>) -> ParseResult<RegionId> {
        debug_assert!(matches!(pair.as_rule(), Rule::region));
        let pair = pair.into_inner().next().unwrap();
        let rule = pair.as_rule();
        let mut inner = pair.into_inner();

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

        Ok(self.module.insert_region(Region {
            kind,
            sources,
            targets,
            children,
            meta,
            signature,
        }))
    }

    fn parse_nodes(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [NodeId]> {
        let mut nodes = Vec::new();

        for pair in filter_rule(pairs, Rule::node) {
            nodes.push(self.parse_node(pair)?);
        }

        Ok(self.bump.alloc_slice_copy(&nodes))
    }

    fn parse_func_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a FuncDecl<'a>> {
        debug_assert!(matches!(pair.as_rule(), Rule::func_header));

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;

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
            signature: func,
        }))
    }

    fn parse_alias_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a AliasDecl<'a>> {
        debug_assert!(matches!(pair.as_rule(), Rule::alias_header));

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
        debug_assert!(matches!(pair.as_rule(), Rule::ctr_header));

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;
        let r#type = self.parse_term(inner.next().unwrap())?;

        Ok(self.bump.alloc(ConstructorDecl {
            name,
            params,
            r#type,
        }))
    }

    fn parse_op_header(&mut self, pair: Pair<'a, Rule>) -> ParseResult<&'a OperationDecl<'a>> {
        debug_assert!(matches!(pair.as_rule(), Rule::operation_header));

        let mut inner = pair.into_inner();
        let name = self.parse_symbol(&mut inner)?;
        let params = self.parse_params(&mut inner)?;
        let r#type = self.parse_term(inner.next().unwrap())?;

        Ok(self.bump.alloc(OperationDecl {
            name,
            params,
            r#type,
        }))
    }

    fn parse_params(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [Param<'a>]> {
        let mut params = Vec::new();

        for pair in filter_rule(pairs, Rule::param) {
            let param = pair.into_inner().next().unwrap();

            let param = match param.as_rule() {
                Rule::param_implicit => {
                    let mut inner = param.into_inner();
                    let name = &inner.next().unwrap().as_str()[1..];
                    let r#type = self.parse_term(inner.next().unwrap())?;
                    Param::Implicit { name, r#type }
                }
                Rule::param_explicit => {
                    let mut inner = param.into_inner();
                    let name = &inner.next().unwrap().as_str()[1..];
                    let r#type = self.parse_term(inner.next().unwrap())?;
                    Param::Explicit { name, r#type }
                }
                Rule::param_constraint => {
                    let mut inner = param.into_inner();
                    let constraint = self.parse_term(inner.next().unwrap())?;
                    Param::Constraint { constraint }
                }
                _ => unreachable!(),
            };

            params.push(param);
        }

        Ok(self.bump.alloc_slice_copy(&params))
    }

    fn parse_signature(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<Option<TermId>> {
        let Some(Rule::signature) = pairs.peek().map(|p| p.as_rule()) else {
            return Ok(None);
        };

        let pair = pairs.next().unwrap();
        let signature = self.parse_term(pair.into_inner().next().unwrap())?;
        Ok(Some(signature))
    }

    fn parse_port_list(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a [LinkRef<'a>]> {
        let Some(Rule::port_list) = pairs.peek().map(|p| p.as_rule()) else {
            return Ok(&[]);
        };

        let pair = pairs.next().unwrap();
        let inner = pair.into_inner();
        let mut links = Vec::new();

        for token in inner {
            links.push(self.parse_port(token)?);
        }

        Ok(self.bump.alloc_slice_copy(&links))
    }

    fn parse_port(&mut self, pair: Pair<'a, Rule>) -> ParseResult<LinkRef<'a>> {
        debug_assert!(matches!(pair.as_rule(), Rule::port));
        let mut inner = pair.into_inner();
        let link = LinkRef::Named(&inner.next().unwrap().as_str()[1..]);
        Ok(link)
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

    fn parse_symbol(&mut self, pairs: &mut Pairs<'a, Rule>) -> ParseResult<&'a str> {
        let pair = pairs.next().unwrap();
        if let Rule::symbol = pair.as_rule() {
            Ok(pair.as_str())
        } else {
            unreachable!("expected a symbol");
        }
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
}

// NOTE: `ParseError` does not implement `From<pest::error::Error<Rule>>` so that
// pest does not become part of the public API.

type ParseResult<T> = Result<T, ParseError>;
