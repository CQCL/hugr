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

use crate::v0::syntax::{LinkName, ListPart, Module, Operation, TuplePart};
use crate::v0::{RegionKind, ScopeClosure};

use super::{Constraint, MetaItem, Node, Param, Region, Signature, Symbol, VarName};

use super::{SymbolName, Term};

trait Parse: Sized {
    const RULE: Rule;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self>;

    fn parse_pairs<'a>(pairs: &mut Pairs<'a, Rule>) -> ParseResult<Self> {
        Self::parse_pair(pairs.next().unwrap())
    }

    fn parse_many<'a, C>(pairs: &mut Pairs<'a, Rule>) -> ParseResult<C>
    where
        C: FromIterator<Self>,
    {
        take_rule(pairs, Self::RULE).map(Self::parse_pair).collect()
    }

    fn parse_opt<'a>(pairs: &mut Pairs<'a, Rule>) -> ParseResult<Option<Self>> {
        if let Some(pair) = take_rule(pairs, Self::RULE).next() {
            Ok(Some(Self::parse_pair(pair)?))
        } else {
            Ok(None)
        }
    }
}

impl Parse for SymbolName {
    const RULE: Rule = Rule::symbol_name;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());
        Ok(SymbolName(pair.as_str().into()))
    }
}

impl Parse for VarName {
    const RULE: Rule = Rule::term_var;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());
        Ok(VarName(pair.as_str()[1..].into()))
    }
}

impl Parse for LinkName {
    const RULE: Rule = Rule::link_name;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());
        Ok(LinkName(pair.as_str()[1..].into()))
    }
}

impl Parse for Term {
    const RULE: Rule = Rule::term;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());
        let pair = pair.into_inner().next().unwrap();

        Ok(match pair.as_rule() {
            Rule::term_wildcard => Term::Wildcard,
            Rule::term_var => Term::Var(VarName::parse_pair(pair)?),
            Rule::term_apply => {
                let mut pairs = pair.into_inner();
                let symbol = SymbolName::parse_pairs(&mut pairs)?;
                let terms = Term::parse_many(&mut pairs)?;
                Term::Apply(symbol, terms)
            }
            Rule::term_list => {
                let mut pairs = pair.into_inner();
                let parts = ListPart::parse_many(&mut pairs)?;
                Term::List(parts)
            }
            Rule::term_tuple => {
                let mut pairs = pair.into_inner();
                let parts = TuplePart::parse_many(&mut pairs)?;
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
                let region = Region::parse_pairs(&mut pairs)?;
                Term::Func(Arc::new(region))
            }
            _ => unreachable!(),
        })
    }
}

impl Parse for ListPart {
    const RULE: Rule = Rule::part;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(pair.as_rule(), Self::RULE);
        let pair = pair.into_inner().next().unwrap();

        Ok(match pair.as_rule() {
            Rule::term => Self::Item(Term::parse_pair(pair)?),
            Rule::spliced_term => {
                let mut pairs = pair.into_inner();
                let term = Term::parse_pairs(&mut pairs)?;
                Self::Splice(term)
            }
            _ => unreachable!("expected term or spliced term"),
        })
    }
}

impl Parse for TuplePart {
    const RULE: Rule = Rule::part;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(pair.as_rule(), Self::RULE);
        let pair = pair.into_inner().next().unwrap();

        Ok(match pair.as_rule() {
            Rule::term => Self::Item(Term::parse_pair(pair)?),
            Rule::spliced_term => {
                let mut pairs = pair.into_inner();
                let term = Term::parse_pairs(&mut pairs)?;
                Self::Splice(term)
            }
            _ => unreachable!("expected term or spliced term"),
        })
    }
}

impl Parse for Module {
    const RULE: Rule = Rule::module;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(pair.as_rule(), Self::RULE);
        let mut pairs = pair.into_inner();
        let meta = MetaItem::parse_many(&mut pairs)?;
        let children = Node::parse_many(&mut pairs)?;
        Ok(Module {
            root: Region {
                kind: RegionKind::Module,
                children,
                meta,
                ..Default::default()
            },
        })
    }
}

impl Parse for Region {
    const RULE: Rule = Rule::region;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(pair.as_rule(), Self::RULE);
        let mut pairs = pair.into_inner();

        let kind = RegionKind::parse_pairs(&mut pairs)?;
        let sources = parse_port_list(&mut pairs)?;
        let targets = parse_port_list(&mut pairs)?;
        let signature = Signature::parse_opt(&mut pairs)?;
        let meta = MetaItem::parse_many(&mut pairs)?;
        let children = Node::parse_many(&mut pairs)?;

        Ok(Self {
            kind,
            sources,
            targets,
            signature,
            meta,
            children,
            scope: Default::default(), // TODO
        })
    }
}

impl Parse for RegionKind {
    const RULE: Rule = Rule::region_kind;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());

        Ok(match pair.as_str() {
            "dfg" => Self::DataFlow,
            "cfg" => Self::ControlFlow,
            "mod" => Self::Module,
            _ => unreachable!(),
        })
    }
}

impl Parse for Node {
    const RULE: Rule = Rule::node;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());
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
                let name = SymbolName::parse_pairs(&mut pairs)?;
                Operation::Import(name)
            }

            Rule::node_custom => {
                let term = Term::parse_pairs(&mut pairs)?;
                Operation::Custom(term)
            }

            Rule::node_define_func => {
                let symbol = Symbol::parse_pairs(&mut pairs)?;
                Operation::DefineFunc(Arc::new(symbol))
            }
            Rule::node_declare_func => {
                let symbol = Symbol::parse_pairs(&mut pairs)?;
                Operation::DeclareFunc(Arc::new(symbol))
            }
            Rule::node_define_alias => {
                let symbol = Symbol::parse_pairs(&mut pairs)?;
                let value = Term::parse_pairs(&mut pairs)?;
                Operation::DefineAlias(Arc::new(symbol), value)
            }
            Rule::node_declare_alias => {
                let symbol = Symbol::parse_pairs(&mut pairs)?;
                Operation::DeclareAlias(Arc::new(symbol))
            }
            Rule::node_declare_ctr => {
                let symbol = Symbol::parse_pairs(&mut pairs)?;
                Operation::DeclareConstructor(Arc::new(symbol))
            }
            Rule::node_declare_operation => {
                let symbol = Symbol::parse_pairs(&mut pairs)?;
                Operation::DeclareOperation(Arc::new(symbol))
            }

            _ => unreachable!(),
        };

        let inputs = parse_port_list(&mut pairs)?;
        let outputs = parse_port_list(&mut pairs)?;
        let signature = Signature::parse_opt(&mut pairs)?;
        let meta = MetaItem::parse_many(&mut pairs)?;
        let regions = Region::parse_many(&mut pairs)?;

        Ok(Node {
            operation,
            inputs,
            outputs,
            regions,
            meta,
            signature,
        })
    }
}

impl Parse for MetaItem {
    const RULE: Rule = Rule::meta;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());
        let mut pairs = pair.into_inner();
        let term = Term::parse_pairs(&mut pairs)?;
        Ok(Self(term))
    }
}

impl Parse for Signature {
    const RULE: Rule = Rule::signature;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());
        let mut pairs = pair.into_inner();
        let term = Term::parse_pairs(&mut pairs)?;
        Ok(Self(term))
    }
}

impl Parse for Param {
    const RULE: Rule = Rule::param;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());
        let mut pairs = pair.into_inner();
        let name = VarName::parse_pairs(&mut pairs)?;
        let r#type = Term::parse_pairs(&mut pairs)?;
        Ok(Self { name, r#type })
    }
}

impl Parse for Symbol {
    const RULE: Rule = Rule::symbol;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());
        let mut pairs = pair.into_inner();
        let name = SymbolName::parse_pairs(&mut pairs)?;
        let params = Param::parse_many(&mut pairs)?;
        let constraints = Constraint::parse_many(&mut pairs)?;
        let signature = Term::parse_pairs(&mut pairs)?;

        Ok(Self {
            name,
            params,
            constraints,
            signature,
        })
    }
}

impl Parse for Constraint {
    const RULE: Rule = Rule::where_clause;

    fn parse_pair<'a>(pair: Pair<'a, Rule>) -> ParseResult<Self> {
        debug_assert_eq!(Self::RULE, pair.as_rule());
        let mut pairs = pair.into_inner();
        let term = Term::parse_pairs(&mut pairs)?;
        Ok(Self(term))
    }
}

fn parse_port_list<'a>(pairs: &mut Pairs<'a, Rule>) -> ParseResult<Box<[LinkName]>> {
    let Some(pair) = take_rule(pairs, Rule::port_list).next() else {
        return Ok(Default::default());
    };

    let mut pairs = pair.into_inner();
    let links = LinkName::parse_many(&mut pairs)?;
    Ok(links)
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
    ($ident:ident, $rule:ident) => {
        impl FromStr for $ident {
            type Err = ParseError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let mut pairs =
                    HugrParser::parse(Rule::$rule, s).map_err(|err| ParseError(Box::new(err)))?;
                Self::parse_pairs(&mut pairs)
            }
        }
    };
}

impl_from_str!(SymbolName, symbol_name);
impl_from_str!(VarName, term_var);
impl_from_str!(LinkName, link_name);
impl_from_str!(Term, term);
impl_from_str!(Node, node);
impl_from_str!(Region, region);
impl_from_str!(MetaItem, meta);
impl_from_str!(Signature, signature);
impl_from_str!(Param, param);
impl_from_str!(Module, module);
