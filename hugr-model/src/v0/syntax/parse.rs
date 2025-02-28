mod pest_parser {
    use pest_derive::Parser;

    // NOTE: The pest derive macro generates a `Rule` enum. We do not want this to be
    // part of the public API, and so we hide it within this private module.

    #[derive(Parser)]
    #[grammar = "v0/syntax/hugr.pest"]
    pub struct HugrParser;
}

use std::str::FromStr;
use std::sync::Arc;

use base64::prelude::BASE64_STANDARD;
use base64::Engine as _;
use pest::iterators::{Pair, Pairs};
use pest::Parser as _;
use pest_parser::{HugrParser, Rule};
use smol_str::SmolStr;
use thiserror::Error;

use crate::v0::syntax::{LinkName, Module, Operation, SeqPart};
use crate::v0::RegionKind;

use super::{Node, Param, Region, Symbol, VarName};
use super::{SymbolName, Term};

fn parse_symbol_name<'a>(pair: Pair<'a, Rule>) -> ParseResult<SymbolName> {
    debug_assert_eq!(Rule::symbol_name, pair.as_rule());
    Ok(SymbolName(pair.as_str().into()))
}

fn parse_var_name<'a>(pair: Pair<'a, Rule>) -> ParseResult<VarName> {
    debug_assert_eq!(Rule::term_var, pair.as_rule());
    Ok(VarName(pair.as_str()[1..].into()))
}

fn parse_link_name<'a>(pair: Pair<'a, Rule>) -> ParseResult<LinkName> {
    debug_assert_eq!(Rule::link_name, pair.as_rule());
    Ok(LinkName(pair.as_str()[1..].into()))
}

fn parse_term<'a>(pair: Pair<'a, Rule>) -> ParseResult<Term> {
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
        Rule::term_str => {
            let mut pairs = pair.into_inner();
            let string = parse_string(pairs.next().unwrap())?;
            Term::Str(string)
        }
        Rule::term_nat => {
            let value = pair.as_str().trim().parse().unwrap();
            Term::Nat(value)
        }
        Rule::term_ext_set => Term::ExtSet,
        Rule::term_bytes => {
            let mut pairs = pair.into_inner();
            let bytes = parse_bytes(pairs.next().unwrap())?;
            Term::Bytes(bytes)
        }
        Rule::term_float => {
            let value: f64 = pair.as_str().trim().parse().unwrap();
            Term::Float(value.into())
        }
        Rule::term_const_func => {
            let mut pairs = pair.into_inner();
            let region = parse_region(pairs.next().unwrap())?;
            Term::Func(Arc::new(region))
        }
        _ => unreachable!(),
    })
}

fn parse_seq_part<'a>(pair: Pair<'a, Rule>) -> ParseResult<SeqPart> {
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

fn parse_module<'a>(pair: Pair<'a, Rule>) -> ParseResult<Module> {
    debug_assert_eq!(pair.as_rule(), Rule::module);
    let mut pairs = pair.into_inner();
    let meta = parse_meta_items(&mut pairs)?;
    let children = pairs.map(parse_node).collect::<ParseResult<_>>()?;

    Ok(Module {
        root: Region {
            kind: RegionKind::Module,
            children,
            meta,
            ..Default::default()
        },
    })
}

fn parse_region<'a>(pair: Pair<'a, Rule>) -> ParseResult<Region> {
    debug_assert_eq!(pair.as_rule(), Rule::region);
    let mut pairs = pair.into_inner();

    let kind = parse_region_kind(pairs.next().unwrap())?;
    let sources = parse_port_list(&mut pairs)?;
    let targets = parse_port_list(&mut pairs)?;
    let signature = parse_optional_signature(&mut pairs)?;
    let meta = parse_meta_items(&mut pairs)?;
    let children = pairs.map(parse_node).collect::<ParseResult<_>>()?;

    Ok(Region {
        kind,
        sources,
        targets,
        signature,
        meta,
        children,
        scope: Default::default(), // TODO
    })
}

fn parse_region_kind<'a>(pair: Pair<'a, Rule>) -> ParseResult<RegionKind> {
    debug_assert_eq!(pair.as_rule(), Rule::region_kind);

    Ok(match pair.as_str() {
        "dfg" => RegionKind::DataFlow,
        "cfg" => RegionKind::ControlFlow,
        "mod" => RegionKind::Module,
        _ => unreachable!(),
    })
}

fn parse_node<'a>(pair: Pair<'a, Rule>) -> ParseResult<Node> {
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
            Operation::DefineFunc(Arc::new(symbol))
        }
        Rule::node_declare_func => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            Operation::DeclareFunc(Arc::new(symbol))
        }
        Rule::node_define_alias => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            let value = parse_term(pairs.next().unwrap())?;
            Operation::DefineAlias(Arc::new(symbol), value)
        }
        Rule::node_declare_alias => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            Operation::DeclareAlias(Arc::new(symbol))
        }
        Rule::node_declare_ctr => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            Operation::DeclareConstructor(Arc::new(symbol))
        }
        Rule::node_declare_operation => {
            let symbol = parse_symbol(pairs.next().unwrap())?;
            Operation::DeclareOperation(Arc::new(symbol))
        }

        _ => unreachable!(),
    };

    let inputs = parse_port_list(&mut pairs)?;
    let outputs = parse_port_list(&mut pairs)?;
    let signature = parse_optional_signature(&mut pairs)?;
    let meta = parse_meta_items(&mut pairs)?;
    let regions = pairs.map(parse_region).collect::<ParseResult<_>>()?;

    Ok(Node {
        operation,
        inputs,
        outputs,
        regions,
        meta,
        signature,
    })
}

fn parse_meta_items<'a>(pairs: &mut Pairs<'a, Rule>) -> ParseResult<Box<[Term]>> {
    take_rule(pairs, Rule::meta).map(parse_meta_item).collect()
}

fn parse_meta_item<'a>(pair: Pair<'a, Rule>) -> ParseResult<Term> {
    debug_assert_eq!(pair.as_rule(), Rule::meta);
    let mut pairs = pair.into_inner();
    parse_term(pairs.next().unwrap())
}

fn parse_optional_signature<'a>(pairs: &mut Pairs<'a, Rule>) -> ParseResult<Option<Term>> {
    if let Some(pair) = take_rule(pairs, Rule::signature).next() {
        Ok(Some(parse_signature(pair)?))
    } else {
        Ok(None)
    }
}

fn parse_signature<'a>(pair: Pair<'a, Rule>) -> ParseResult<Term> {
    debug_assert_eq!(Rule::signature, pair.as_rule());
    let mut pairs = pair.into_inner();
    parse_term(pairs.next().unwrap())
}

fn parse_params<'a>(pairs: &mut Pairs<'a, Rule>) -> ParseResult<Arc<[Param]>> {
    take_rule(pairs, Rule::param).map(parse_param).collect()
}

fn parse_param<'a>(pair: Pair<'a, Rule>) -> ParseResult<Param> {
    debug_assert_eq!(Rule::param, pair.as_rule());
    let mut pairs = pair.into_inner();
    let name = parse_var_name(pairs.next().unwrap())?;
    let r#type = parse_term(pairs.next().unwrap())?;
    Ok(Param { name, r#type })
}

fn parse_symbol<'a>(pair: Pair<'a, Rule>) -> ParseResult<Symbol> {
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

fn parse_constraints<'a>(pairs: &mut Pairs<'a, Rule>) -> ParseResult<Arc<[Term]>> {
    take_rule(pairs, Rule::where_clause)
        .map(parse_constraint)
        .collect()
}

fn parse_constraint<'a>(pair: Pair<'a, Rule>) -> ParseResult<Term> {
    debug_assert_eq!(Rule::where_clause, pair.as_rule());
    let mut pairs = pair.into_inner();
    parse_term(pairs.next().unwrap())
}

fn parse_port_list<'a>(pairs: &mut Pairs<'a, Rule>) -> ParseResult<Box<[LinkName]>> {
    let Some(pair) = take_rule(pairs, Rule::port_list).next() else {
        return Ok(Default::default());
    };

    let pairs = pair.into_inner();
    pairs.map(parse_link_name).collect()
}

fn parse_string<'a>(pair: Pair<'a, Rule>) -> ParseResult<SmolStr> {
    debug_assert_eq!(pair.as_rule(), Rule::string);

    // Any escape sequence is longer than the character it represents.
    // Therefore the length of this token (minus 2 for the quotes on either
    // side) is an upper bound for the length of the string.
    let capacity = pair.as_str().len() - 2;
    let mut string = String::with_capacity(capacity);
    let pairs = pair.into_inner();

    for pair in pairs {
        match pair.as_rule() {
            Rule::string_raw => string.push_str(pair.as_str()),
            Rule::string_escape => match pair.as_str().chars().nth(1).unwrap() {
                '"' => string.push('"'),
                '\\' => string.push('\\'),
                'n' => string.push('\n'),
                'r' => string.push('\r'),
                't' => string.push('\t'),
                _ => unreachable!(),
            },
            Rule::string_unicode => {
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

fn parse_bytes<'a>(pair: Pair<'a, Rule>) -> ParseResult<Arc<[u8]>> {
    let slice = pair.as_str().as_bytes();

    // Remove the quotes
    let slice = &slice[1..slice.len() - 1];

    let data = BASE64_STANDARD
        .decode(slice)
        .map_err(|_| ParseError::custom("invalid base64 encoding", pair.as_span()))?;

    Ok(data.into())
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
impl_from_str!(Module, module, parse_module);
impl_from_str!(SeqPart, part, parse_seq_part);
