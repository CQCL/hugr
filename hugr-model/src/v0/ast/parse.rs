// NOTE: We use the `pest` library for parsing. This library is convenient, but
// performance is mediocre. In the case that we find that parsing is too slow,
// we can replace the parser.

// NOTE: The `pest` library returns a parsed AST which we then transform into
// our AST data structures. The `pest` AST is guaranteed to conform to the
// grammar, but this is not automatically visible from the types. Therefore this
// module contains many `unwrap`s and `unreachable!`s, which will not fail on
// any input unless there is a bug in the parser. This is perhaps aesthetically
// unsatisfying but it is aligned with the intended usage pattern of `pest`.

// NOTE: The `parse_` functions are implementation details since they refer to
// `pest` data structures. We expose parsing via implementations of the
// `FromStr` trait.

use std::str::FromStr;
use std::sync::Arc;

use base64::Engine as _;
use base64::prelude::BASE64_STANDARD;
use ordered_float::OrderedFloat;
use pest::Parser as _;
use pest::iterators::{Pair, Pairs};
use pest_parser::{HugrParser, Rule};
use smol_str::SmolStr;
use thiserror::Error;

use crate::v0::ast::{LinkName, Module, Operation, SeqPart};
use crate::v0::{Literal, RegionKind};

use super::{Node, Package, Param, Region, Symbol, VarName};
use super::{SymbolName, Term};

mod pest_parser {
    use pest_derive::Parser;

    // NOTE: The pest derive macro generates a `Rule` enum. We do not want this to be
    // part of the public API, and so we hide it within this private module.

    #[derive(Parser)]
    #[grammar = "v0/ast/hugr.pest"]
    pub struct HugrParser;
}

fn parse_symbol_name(pair: Pair<Rule>) -> ParseResult<SymbolName> {
    debug_assert_eq!(Rule::symbol_name, pair.as_rule());
    Ok(SymbolName(pair.as_str().into()))
}

fn parse_var_name(pair: Pair<Rule>) -> ParseResult<VarName> {
    debug_assert_eq!(Rule::term_var, pair.as_rule());
    Ok(VarName(pair.as_str()[1..].into()))
}

fn parse_link_name(pair: Pair<Rule>) -> ParseResult<LinkName> {
    debug_assert_eq!(Rule::link_name, pair.as_rule());
    Ok(LinkName(pair.as_str()[1..].into()))
}

fn parse_term(pair: Pair<Rule>) -> ParseResult<Term> {
    debug_assert_eq!(Rule::term, pair.as_rule());
    let pair = pair.into_inner().next().unwrap();

    Ok(match pair.as_rule() {
        Rule::term_wildcard => Term::Wildcard,
        Rule::term_var => Term::Var(parse_var_name(pair)?),
        Rule::term_apply => {
            let mut pairs = pair.into_inner();
            let symbol = parse_symbol_name(pairs.next().unwrap())?;
            let terms = pairs.map(parse_term).collect::<ParseResult<_>>()?;
            Term::Apply(symbol, terms)
        }
        Rule::term_list => {
            let pairs = pair.into_inner();
            let parts = pairs.map(parse_seq_part).collect::<ParseResult<_>>()?;
            Term::List(parts)
        }
        Rule::term_tuple => {
            let pairs = pair.into_inner();
            let parts = pairs.map(parse_seq_part).collect::<ParseResult<_>>()?;
            Term::Tuple(parts)
        }
        Rule::literal => {
            let literal = parse_literal(pair)?;
            Term::Literal(literal)
        }
        Rule::term_const_func => {
            let mut pairs = pair.into_inner();
            let region = parse_region(pairs.next().unwrap())?;
            Term::Func(Arc::new(region))
        }
        _ => unreachable!(),
    })
}

fn parse_literal(pair: Pair<Rule>) -> ParseResult<Literal> {
    debug_assert_eq!(pair.as_rule(), Rule::literal);
    let pair = pair.into_inner().next().unwrap();

    Ok(match pair.as_rule() {
        Rule::literal_string => Literal::Str(parse_string(pair)?),
        Rule::literal_nat => Literal::Nat(parse_nat(pair)?),
        Rule::literal_bytes => Literal::Bytes(parse_bytes(pair)?),
        Rule::literal_float => Literal::Float(parse_float(pair)?),
        _ => unreachable!("expected literal"),
    })
}

fn parse_seq_part(pair: Pair<Rule>) -> ParseResult<SeqPart> {
    debug_assert_eq!(pair.as_rule(), Rule::part);
    let pair = pair.into_inner().next().unwrap();

    Ok(match pair.as_rule() {
        Rule::term => SeqPart::Item(parse_term(pair)?),
        Rule::spliced_term => {
            let mut pairs = pair.into_inner();
            let term = parse_term(pairs.next().unwrap())?;
            SeqPart::Splice(term)
        }
        _ => unreachable!("expected term or spliced term"),
    })
}

fn parse_package(pair: Pair<Rule>) -> ParseResult<Package> {
    debug_assert_eq!(pair.as_rule(), Rule::package);
    let mut pairs = pair.into_inner();

    let modules = take_rule(&mut pairs, Rule::module)
        .map(parse_module)
        .collect::<ParseResult<_>>()?;

    Ok(Package { modules })
}

fn parse_module(pair: Pair<Rule>) -> ParseResult<Module> {
    debug_assert_eq!(pair.as_rule(), Rule::module);
    let mut pairs = pair.into_inner();
    let meta = parse_meta_items(&mut pairs)?;
    let children = parse_nodes(&mut pairs)?;

    Ok(Module {
        root: Region {
            kind: RegionKind::Module,
            children,
            meta,
            ..Default::default()
        },
    })
}

fn parse_region(pair: Pair<Rule>) -> ParseResult<Region> {
    debug_assert_eq!(pair.as_rule(), Rule::region);
    let mut pairs = pair.into_inner();

    let kind = parse_region_kind(pairs.next().unwrap())?;
    let sources = parse_port_list(&mut pairs)?;
    let targets = parse_port_list(&mut pairs)?;
    let signature = parse_optional_signature(&mut pairs)?;
    let meta = parse_meta_items(&mut pairs)?;
    let children = parse_nodes(&mut pairs)?;

    Ok(Region {
        kind,
        sources,
        targets,
        children,
        meta,
        signature,
    })
}

fn parse_region_kind(pair: Pair<Rule>) -> ParseResult<RegionKind> {
    debug_assert_eq!(pair.as_rule(), Rule::region_kind);

    Ok(match pair.as_str() {
        "dfg" => RegionKind::DataFlow,
        "cfg" => RegionKind::ControlFlow,
        "mod" => RegionKind::Module,
        _ => unreachable!(),
    })
}

fn parse_nodes(pairs: &mut Pairs<Rule>) -> ParseResult<Box<[Node]>> {
    take_rule(pairs, Rule::node).map(parse_node).collect()
}

fn parse_node(pair: Pair<Rule>) -> ParseResult<Node> {
    debug_assert_eq!(pair.as_rule(), Rule::node);
    let mut pairs = pair.into_inner();
    let pair = pairs.next().unwrap();
    let rule = pair.as_rule();
    let mut pairs = pair.into_inner();

    let operation = match rule {
        Rule::node_dfg => Operation::Dfg,
        Rule::node_cfg => Operation::Cfg,
        Rule::node_block => Operation::Block,
        Rule::node_tail_loop => Operation::TailLoop,
        Rule::node_cond => Operation::Conditional,

        Rule::node_import => {
            let name = parse_symbol_name(pairs.next().unwrap())?;
            Operation::Import(name)
        }

        Rule::node_custom => {
            let term = parse_term(pairs.next().unwrap())?;
            Operation::Custom(term)
        }

        Rule::node_define_func => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            Operation::DefineFunc(Box::new(symbol))
        }
        Rule::node_declare_func => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            Operation::DeclareFunc(Box::new(symbol))
        }
        Rule::node_define_alias => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            let value = parse_term(pairs.next().unwrap())?;
            Operation::DefineAlias(Box::new(symbol), value)
        }
        Rule::node_declare_alias => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            Operation::DeclareAlias(Box::new(symbol))
        }
        Rule::node_declare_ctr => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            Operation::DeclareConstructor(Box::new(symbol))
        }
        Rule::node_declare_operation => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            Operation::DeclareOperation(Box::new(symbol))
        }

        _ => unreachable!(),
    };

    let inputs = parse_port_list(&mut pairs)?;
    let outputs = parse_port_list(&mut pairs)?;
    let signature = parse_optional_signature(&mut pairs)?;
    let meta = parse_meta_items(&mut pairs)?;
    let regions = pairs
        .map(|pair| parse_region(pair))
        .collect::<ParseResult<_>>()?;

    Ok(Node {
        operation,
        inputs,
        outputs,
        regions,
        meta,
        signature,
    })
}

fn parse_meta_items(pairs: &mut Pairs<Rule>) -> ParseResult<Box<[Term]>> {
    take_rule(pairs, Rule::meta).map(parse_meta_item).collect()
}

fn parse_meta_item(pair: Pair<Rule>) -> ParseResult<Term> {
    debug_assert_eq!(pair.as_rule(), Rule::meta);
    let mut pairs = pair.into_inner();
    parse_term(pairs.next().unwrap())
}

fn parse_optional_signature(pairs: &mut Pairs<Rule>) -> ParseResult<Option<Term>> {
    match take_rule(pairs, Rule::signature).next() {
        Some(pair) => Ok(Some(parse_signature(pair)?)),
        _ => Ok(None),
    }
}

fn parse_signature(pair: Pair<Rule>) -> ParseResult<Term> {
    debug_assert_eq!(Rule::signature, pair.as_rule());
    let mut pairs = pair.into_inner();
    parse_term(pairs.next().unwrap())
}

fn parse_params(pairs: &mut Pairs<Rule>) -> ParseResult<Box<[Param]>> {
    take_rule(pairs, Rule::param).map(parse_param).collect()
}

fn parse_param(pair: Pair<Rule>) -> ParseResult<Param> {
    debug_assert_eq!(Rule::param, pair.as_rule());
    let mut pairs = pair.into_inner();
    let name = parse_var_name(pairs.next().unwrap())?;
    let r#type = parse_term(pairs.next().unwrap())?;
    Ok(Param { name, r#type })
}

fn parse_symbol(pair: Pair<Rule>) -> ParseResult<Symbol> {
    debug_assert_eq!(Rule::symbol, pair.as_rule());
    let mut pairs = pair.into_inner();
    let name = parse_symbol_name(pairs.next().unwrap())?;
    let params = parse_params(&mut pairs)?;
    let constraints = parse_constraints(&mut pairs)?;
    let signature = parse_term(pairs.next().unwrap())?;

    Ok(Symbol {
        name,
        params,
        constraints,
        signature,
    })
}

fn parse_constraints(pairs: &mut Pairs<Rule>) -> ParseResult<Box<[Term]>> {
    take_rule(pairs, Rule::where_clause)
        .map(parse_constraint)
        .collect()
}

fn parse_constraint(pair: Pair<Rule>) -> ParseResult<Term> {
    debug_assert_eq!(Rule::where_clause, pair.as_rule());
    let mut pairs = pair.into_inner();
    parse_term(pairs.next().unwrap())
}

fn parse_port_list(pairs: &mut Pairs<Rule>) -> ParseResult<Box<[LinkName]>> {
    let Some(pair) = take_rule(pairs, Rule::port_list).next() else {
        return Ok(Default::default());
    };

    let pairs = pair.into_inner();
    pairs.map(parse_link_name).collect()
}

fn parse_string(pair: Pair<Rule>) -> ParseResult<SmolStr> {
    debug_assert_eq!(pair.as_rule(), Rule::literal_string);

    // Any escape sequence is longer than the character it represents.
    // Therefore the length of this token (minus 2 for the quotes on either
    // side) is an upper bound for the length of the string.
    let capacity = pair.as_str().len() - 2;
    let mut string = String::with_capacity(capacity);
    let pairs = pair.into_inner();

    for pair in pairs {
        match pair.as_rule() {
            Rule::literal_string_raw => string.push_str(pair.as_str()),
            Rule::literal_string_escape => match pair.as_str().chars().nth(1).unwrap() {
                '"' => string.push('"'),
                '\\' => string.push('\\'),
                'n' => string.push('\n'),
                'r' => string.push('\r'),
                't' => string.push('\t'),
                _ => unreachable!(),
            },
            Rule::literal_string_unicode => {
                let token_str = pair.as_str();
                debug_assert_eq!(&token_str[0..3], r"\u{");
                debug_assert_eq!(&token_str[token_str.len() - 1..], "}");
                let code_str = &token_str[3..token_str.len() - 1];
                let code = u32::from_str_radix(code_str, 16).map_err(|_| {
                    ParseError::custom("invalid unicode escape sequence", pair.as_span())
                })?;
                let char = std::char::from_u32(code).ok_or_else(|| {
                    ParseError::custom("invalid unicode code point", pair.as_span())
                })?;
                string.push(char);
            }
            _ => unreachable!(),
        }
    }

    Ok(string.into())
}

fn parse_bytes(pair: Pair<Rule>) -> ParseResult<Arc<[u8]>> {
    debug_assert_eq!(pair.as_rule(), Rule::literal_bytes);
    let pair = pair.into_inner().next().unwrap();
    debug_assert_eq!(pair.as_rule(), Rule::base64_string);

    let slice = pair.as_str().as_bytes();

    // Remove the quotes
    let slice = &slice[1..slice.len() - 1];

    let data = BASE64_STANDARD
        .decode(slice)
        .map_err(|_| ParseError::custom("invalid base64 encoding", pair.as_span()))?;

    Ok(data.into())
}

fn parse_nat(pair: Pair<Rule>) -> ParseResult<u64> {
    debug_assert_eq!(pair.as_rule(), Rule::literal_nat);
    let value = pair.as_str().trim().parse().unwrap();
    Ok(value)
}

fn parse_float(pair: Pair<Rule>) -> ParseResult<OrderedFloat<f64>> {
    debug_assert_eq!(pair.as_rule(), Rule::literal_float);
    let value = pair.as_str().trim().parse().unwrap();
    Ok(OrderedFloat(value))
}

fn take_rule<'a, 'i>(
    pairs: &'i mut Pairs<'a, Rule>,
    rule: Rule,
) -> impl Iterator<Item = Pair<'a, Rule>> + 'i {
    std::iter::from_fn(move || {
        if pairs.peek()?.as_rule() == rule {
            pairs.next()
        } else {
            None
        }
    })
}

type ParseResult<T> = Result<T, ParseError>;

/// An error that occurred during parsing.
#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub struct ParseError(Box<pest::error::Error<Rule>>);

impl ParseError {
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

macro_rules! impl_from_str {
    ($ident:ident, $rule:ident, $parse:expr) => {
        impl FromStr for $ident {
            type Err = ParseError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let mut pairs =
                    HugrParser::parse(Rule::$rule, s).map_err(|err| ParseError(Box::new(err)))?;
                $parse(pairs.next().unwrap())
            }
        }
    };
}

impl_from_str!(SymbolName, symbol_name, parse_symbol_name);
impl_from_str!(VarName, term_var, parse_var_name);
impl_from_str!(LinkName, link_name, parse_link_name);
impl_from_str!(Term, term, parse_term);
impl_from_str!(Node, node, parse_node);
impl_from_str!(Region, region, parse_region);
impl_from_str!(Param, param, parse_param);
impl_from_str!(Package, package, parse_package);
impl_from_str!(Module, module, parse_module);
impl_from_str!(SeqPart, part, parse_seq_part);
impl_from_str!(Literal, literal, parse_literal);
impl_from_str!(Symbol, symbol, parse_symbol);
