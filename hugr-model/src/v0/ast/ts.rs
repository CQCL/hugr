use std::{
    iter::Peekable,
    ops::Range,
    str::{CharIndices, Chars},
    sync::{Arc, LazyLock},
};

use itertools::Itertools;
use regex::Regex;
use smol_str::{SmolStr, SmolStrBuilder, ToSmolStr};
use thiserror::Error;
use tree_sitter::{Node as TSNode, TreeCursor};
use tree_sitter_hugr;

use crate::v0::{
    CORE_FN, CORE_META_DESCRIPTION, LinkName, Literal, SymbolName, VarName, Visibility,
};

use super::{Module, Node, Operation, Param, Region, SeqPart, Symbol, Term};

#[derive(Debug, Error)]
pub enum ParseError {}

type ParseResult<T> = Result<T, ParseError>;

#[test]
fn test() {
    let src = include_str!("../../../core.thugr");
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_hugr::LANGUAGE.into())
        .unwrap();

    let ast = parser.parse(src, None).unwrap();
    let root = ast.root_node();

    if root.has_error() {
        eprintln!("found a parse error");
    }

    // let mut walk = Walk::new(ast.walk());

    // walk.debug();
    panic!("{}", root.to_sexp());
}

fn parse_node(token: Token) -> ParseResult<Node> {
    match token.name() {
        "symbol_function" => parse_symbol_function(token),
        "symbol_ctr" => parse_symbol_ctr(token),
        "symbol_op" => parse_symbol_op(token),
        "operation" => parse_operation(token),
        _ => panic!("expected node, got `{}`", token.name()),
    }
}

fn parse_symbol(tokens: &mut Tokens) -> ParseResult<Symbol> {
    let visibility = Some(parse_visibility(tokens)?);
    let name = tokens.parse_one("symbol", parse_symbol_name)?;
    let params = tokens.parse_many("param", parse_param)?;
    let constraints = tokens
        .parse_opt("constraints", parse_constraints)?
        .unwrap_or_default();
    let signature = tokens.parse_opt("term", parse_term)?.unwrap_or_default();
    Ok(Symbol {
        visibility,
        name,
        params,
        constraints,
        signature,
    })
}

fn parse_symbol_function(token: Token) -> ParseResult<Node> {
    assert_eq!(token.name(), "symbol_function");
    let mut inner = token.children();

    let meta = parse_meta_seq(&mut inner)?;
    let symbol = parse_symbol(&mut inner)?;
    let body = inner.parse_opt("region", parse_region)?;

    Ok(Node {
        operation: match body {
            Some(_) => Operation::DefineFunc(Box::new(symbol)),
            None => Operation::DeclareFunc(Box::new(symbol)),
        },
        inputs: Default::default(),
        outputs: Default::default(),
        regions: body.into_iter().collect(),
        meta,
        signature: None,
    })
}

fn parse_symbol_ctr(token: Token) -> ParseResult<Node> {
    assert_eq!(token.name(), "symbol_ctr");
    let mut inner = token.children();

    let meta = parse_meta_seq(&mut inner)?;
    let symbol = parse_symbol(&mut inner)?;

    Ok(Node {
        operation: Operation::DeclareConstructor(Box::new(symbol)),
        inputs: Default::default(),
        outputs: Default::default(),
        regions: Default::default(),
        meta,
        signature: None,
    })
}

fn parse_symbol_op(token: Token) -> ParseResult<Node> {
    assert_eq!(token.name(), "symbol_op");
    let mut inner = token.children();

    let meta = parse_meta_seq(&mut inner)?;
    let symbol = parse_symbol(&mut inner)?;

    Ok(Node {
        operation: Operation::DeclareConstructor(Box::new(symbol)),
        inputs: Default::default(),
        outputs: Default::default(),
        regions: Default::default(),
        meta,
        signature: None,
    })
}

fn parse_operation(token: Token) -> ParseResult<Node> {
    assert_eq!(token.name(), "operation");
    let mut inner = token.children();
    let meta = inner.parse_many("meta", parse_meta)?;
    let outputs = inner.parse_many("link", parse_link_name)?;
    let operation = inner.parse_one("term", parse_term)?;
    let inputs = inner.parse_many("link", parse_link_name)?;
    let regions = inner.parse_many("region", parse_region)?;
    let signature = inner.parse_opt("term", parse_term)?;
    Ok(Node {
        operation: Operation::Custom(operation),
        inputs,
        outputs,
        regions,
        meta,
        signature,
    })
}

fn parse_region(token: Token) -> ParseResult<Region> {
    assert_eq!(token.name(), "region");
    todo!()
}

fn parse_visibility(tokens: &mut Tokens) -> ParseResult<Visibility> {
    Ok(tokens
        .take_rule("pub")
        .next()
        .map(|_| Visibility::Public)
        .unwrap_or(Visibility::Private))
}

fn parse_constraints(token: Token) -> ParseResult<Box<[Term]>> {
    assert_eq!(token.name(), "constraints");
    token.children().map(parse_term).try_collect()
}

fn parse_param(token: Token) -> ParseResult<Param> {
    assert_eq!(token.name(), "param");
    let mut inner = token.children();
    let name = inner.parse_one("var", parse_var_name)?;
    let r#type = inner.parse_one("term", parse_term)?;
    Ok(Param { name, r#type })
}

fn parse_meta_seq(tokens: &mut Tokens) -> ParseResult<Box<[Term]>> {
    let doc_lines: Vec<_> = tokens.parse_many("doc_comment", parse_doc_comment)?;
    let mut meta: Vec<_> = tokens.parse_many("meta", parse_meta)?;

    if !doc_lines.is_empty() {
        let doc = Term::Literal(Literal::Str(doc_lines.join("").to_smolstr()));
        meta.insert(
            0,
            Term::Apply(SymbolName::new(CORE_META_DESCRIPTION), [doc].into()),
        );
    }

    Ok(meta.into())
}

fn parse_doc_comment<'a>(token: Token<'a>) -> ParseResult<&'a str> {
    Ok(token.slice().strip_prefix("/// ").unwrap())
}

fn parse_meta(token: Token) -> ParseResult<Term> {
    assert_eq!(token.name(), "meta");
    token.children().parse_one("term", parse_term)
}

fn parse_term(token: Token) -> ParseResult<Term> {
    assert_eq!(token.name(), "term");

    let mut inner = token.children();
    let node = inner.next().unwrap();
    let mut inner = node.children();

    match node.name() {
        "term_apply" => {
            let symbol = inner.parse_one("symbol", parse_symbol_name)?;
            let args = inner.map(parse_term).try_collect()?;
            Ok(Term::Apply(symbol, args))
        }
        "term_list" => {
            let parts = inner.map(parse_seq_part).try_collect()?;
            Ok(Term::List(parts))
        }
        "term_tuple" => {
            let parts = inner.map(parse_seq_part).try_collect()?;
            Ok(Term::Tuple(parts))
        }
        "func_type" => {
            let inputs = inner.parse_one("term", parse_term)?;
            let outputs = inner.parse_one("term", parse_term)?;
            Ok(Term::Apply(
                SymbolName::new(CORE_FN),
                [inputs, outputs].into(),
            ))
        }
        "var" => {
            let var = parse_var_name(node)?;
            Ok(Term::Var(var))
        }
        "string" => {
            let value = parse_string(node)?;
            Ok(Term::Literal(Literal::Str(value)))
        }
        "nat" => {
            let value = parse_nat(node)?;
            Ok(Term::Literal(Literal::Nat(value)))
        }
        "wildcard" => Ok(Term::Wildcard),
        "term_parens" => inner.parse_one("term", parse_term),
        _ => panic!("expected term but got `{}`", node.name()),
    }
}

fn parse_seq_part(token: Token) -> ParseResult<SeqPart> {
    match token.name() {
        "splice" => {
            let mut inner = token.children();
            let term = inner.parse_one("term", parse_term)?;
            Ok(SeqPart::Splice(term))
        }
        "term" => Ok(SeqPart::Item(parse_term(token)?)),
        _ => panic!("expected sequence part but got `{}`", token.name()),
    }
}

fn parse_symbol_name(token: Token) -> ParseResult<SymbolName> {
    todo!()
}

fn parse_link_name(token: Token) -> ParseResult<LinkName> {
    todo!()
}

fn parse_var_name(token: Token) -> ParseResult<VarName> {
    todo!()
}

fn parse_string(node: Token) -> ParseResult<SmolStr> {
    todo!()
    // parse_string_literal(node.slice())
}

fn parse_nat(node: Token) -> ParseResult<u64> {
    node.slice().parse().map_err(|_| todo!())
}

fn parse_sigil_id(sigil: char, str: &str) -> Result<SmolStr, IdParseError> {
    let Some(str) = str.strip_prefix(sigil) else {
        return Err(IdParseError::UnexpectedSigil { expected: sigil });
    };

    if str.starts_with('"') {
        Ok(parse_string_literal(str)?)
    } else {
        Ok(str.to_smolstr())
    }
}

#[derive(Debug, Error)]
pub enum IdParseError {
    #[error("expected sigil `{expected}`")]
    UnexpectedSigil { expected: char },
    #[error("error parsing string escaped id")]
    String(#[from] StringParseError),
}

fn parse_string_literal(str: &str) -> Result<SmolStr, StringParseError> {
    let mut builder = SmolStrBuilder::new();
    let mut chars = str.char_indices();

    while let Some((_, char)) = chars.next() {
        let unescaped = match char {
            '"' => return Err(StringParseError::UnescapedQuote),
            '\\' => {
                let (start, char) = chars.next().ok_or(StringParseError::MissingEscape)?;
                match char {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    'u' => {
                        let rest = str[start..]
                            .strip_prefix("u{")
                            .ok_or(StringParseError::BadUnicode)?;
                        let (code_str, rest) =
                            rest.split_once("}").ok_or(StringParseError::BadUnicode)?;
                        let code = u32::from_str_radix(code_str, 16)
                            .map_err(|_| StringParseError::BadUnicode)?;
                        let char = char::from_u32(code)
                            .ok_or_else(|| StringParseError::UnknownUnicode(code))?;
                        chars = rest.char_indices();
                        char
                    }
                    _ => return Err(StringParseError::UnknownEscape(char)),
                }
            }
            char => char,
        };

        builder.push(unescaped);
    }

    Ok(builder.finish())
}

#[derive(Debug, Error)]
pub enum StringParseError {
    #[error("unknown escape char `{0}`")]
    UnknownEscape(char),
    #[error("missing escaped char after backslash")]
    MissingEscape,
    #[error("unescaped quote")]
    UnescapedQuote,
    #[error("badly formatted unicode escape sequence")]
    BadUnicode,
    #[error("unknown unicode code point {0}")]
    UnknownUnicode(u32),
}

#[derive(Debug, Clone, Copy)]
struct Token<'a> {
    source: &'a str,
    node: TSNode<'a>,
}

impl<'a> Token<'a> {
    pub fn name(&self) -> &str {
        self.node.grammar_name()
    }

    pub fn range(&self) -> Range<usize> {
        self.node.byte_range()
    }

    pub fn slice(&self) -> &'a str {
        &self.source[self.range()]
    }

    pub fn children(&self) -> Tokens<'a> {
        let node = self.node.named_child(0);
        let count = self.node.named_child_count();
        let token = node.map(|node| Token {
            node,
            source: self.source,
        });
        Tokens { token, count }
    }
}

struct Tokens<'a> {
    token: Option<Token<'a>>,
    count: usize,
}

impl<'a> Tokens<'a> {
    /// Peeks at the next node in the parser.
    pub fn peek(&self) -> Option<Token<'a>> {
        self.token
    }

    pub fn take_rule<'p>(&'p mut self, rule: &'static str) -> RuleParser<'p, 'a> {
        RuleParser { parser: self, rule }
    }

    pub fn parse_one<T, F>(&mut self, rule: &'static str, parse: F) -> ParseResult<T>
    where
        F: FnOnce(Token<'a>) -> ParseResult<T>,
    {
        let Some(token) = self.next() else {
            panic!("expected `{}` but got nothing", rule);
        };

        if token.name() != rule {
            panic!("expected `{}` but got `{}`", rule, token.name());
        }

        parse(token)
    }

    pub fn parse_many<T, C, F>(&mut self, rule: &'static str, parse: F) -> ParseResult<C>
    where
        C: FromIterator<T>,
        F: FnMut(Token<'a>) -> ParseResult<T>,
    {
        self.take_rule(rule).map(parse).try_collect()
    }

    pub fn parse_opt<T, F>(&mut self, rule: &'static str, parse: F) -> ParseResult<Option<T>>
    where
        F: FnOnce(Token<'a>) -> ParseResult<T>,
    {
        self.take_rule(rule).next().map(parse).transpose()
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let token = self.token.take()?;
        self.count -= 1;
        assert!(token.node.is_error());
        let node = token.node.next_named_sibling();
        self.token = node.map(|node| Token {
            node,
            source: token.source,
        });
        Some(token)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

struct RuleParser<'p, 'a> {
    parser: &'p mut Tokens<'a>,
    rule: &'static str,
}

impl<'p, 'a> Iterator for RuleParser<'p, 'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.parser.peek()?;

        if node.name() == self.rule {
            self.parser.next()
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.parser.size_hint().1)
    }
}

struct Nodes<'a> {
    cursor: Option<TreeCursor<'a>>,
    size: usize,
}

impl<'a> Nodes<'a> {
    pub fn new(cursor: TreeCursor<'a>) -> Self {
        Self {
            size: cursor.node().descendant_count(),
            cursor: Some(cursor),
        }
    }
}

impl<'a> Iterator for Nodes<'a> {
    type Item = TSNode<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let cursor = self.cursor.as_mut()?;
        let node = cursor.node();
        self.size -= 1;

        if cursor.goto_first_child() {
            return Some(node);
        }

        while !cursor.goto_next_sibling() {
            if !cursor.goto_parent() {
                self.cursor = None;
                return Some(node);
            }
        }

        Some(node)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}
