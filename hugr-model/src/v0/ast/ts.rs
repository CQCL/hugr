use itertools::Itertools;
use smol_str::ToSmolStr;
use std::{io, iter::FusedIterator, ops::Range};
use thiserror::Error;
use tree_sitter::{Node as TSNode, TreeCursor};
use tree_sitter_hugr;

use crate::v0::{
    CORE_FN, CORE_META_DESCRIPTION, LinkName, Literal, RegionKind, SymbolName, VarName, Visibility,
};

use super::{
    Module, Node, Operation, Package, Param, Region, SeqPart, Symbol, Term,
    literals::LiteralParseError, names::NameParseError,
};

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("parse error")]
    Error { location: Span },

    #[error("failed to parse literal")]
    Literal {
        #[source]
        error: LiteralParseError,
        location: Span,
    },
    #[error("failed to parse name")]
    Name {
        #[source]
        error: NameParseError,
        location: Span,
    },
}

impl ParseError {
    pub fn location(&self) -> Span {
        match self {
            ParseError::Error { location } => location,
            ParseError::Literal { location, .. } => location,
            ParseError::Name { location, .. } => location,
        }
        .clone()
    }

    pub fn eprint(&self, source: &str) -> io::Result<()> {
        self.to_report().eprint(ariadne::Source::from(source))
    }

    fn to_report(&self) -> ariadne::Report {
        ariadne::Report::build(ariadne::ReportKind::Error, self.location())
            .with_message("parse error")
            .with_label(
                ariadne::Label::new(self.location())
                    .with_color(ariadne::Color::Red)
                    .with_message("parse error"),
            )
            .finish()
    }
}

pub type Span = Range<usize>;

type ParseResult<T> = Result<T, ParseError>;

#[test]
fn test() {
    let src = include_str!("../../../core.thugr");

    let package = match parse(src) {
        Ok(package) => package,
        Err(err) => {
            err.eprint(src).unwrap();
            panic!();
        }
    };

    panic!("{}", package);
}

fn parse(source: &str) -> ParseResult<Package> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_hugr::LANGUAGE.into())
        .expect("failed to set `hugr` language in tree-sitter parser");

    // NOTE: The `parse` method should always succeed. Parse errors are communicated
    // via error ndoes in the produced AST.
    let ast = parser.parse(source, None).unwrap();
    let root = ast.root_node();

    if root.has_error() {
        // Find the first error node
        let error = AllNodes::new(root.walk())
            .find(|node| node.is_error())
            .unwrap();

        return Err(ParseError::Error {
            location: error.byte_range(),
        });
    }

    parse_package(Token { node: root, source })
}

fn parse_package(token: Token) -> ParseResult<Package> {
    let mut inner = token.children();
    let modules = inner.parse_many("module", parse_module)?;
    Ok(Package { modules })
}

fn parse_module(token: Token) -> ParseResult<Module> {
    debug_assert_eq!(token.name(), "module");
    let mut inner = token.children();
    let meta = parse_meta_seq(&mut inner)?;
    let children = inner.map(parse_node).try_collect()?;

    Ok(Module {
        root: Region {
            kind: RegionKind::Module,
            sources: Default::default(),
            targets: Default::default(),
            children,
            meta,
            signature: None,
        },
    })
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
    let name = tokens.parse_one("symbol_name", parse_symbol_name)?;
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
    let outputs = inner.parse_many("link_name", parse_link_name)?;
    let operation = inner.parse_one("term", parse_term)?;
    let inputs = inner.parse_many("link_name", parse_link_name)?;
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
    let name = inner.parse_one("var_name", parse_var_name)?;
    let r#type = inner.parse_one("term", parse_term)?;
    Ok(Param { name, r#type })
}

fn parse_meta_seq(tokens: &mut Tokens) -> ParseResult<Box<[Term]>> {
    let doc_lines: Vec<_> = tokens.parse_many("doc_comment", parse_doc_comment)?;
    let mut meta: Vec<_> = tokens.parse_many("meta", parse_meta)?;

    if !doc_lines.is_empty() {
        let doc = Term::Literal(Literal::Str(doc_lines.join("\n").to_smolstr()));
        meta.insert(
            0,
            Term::Apply(SymbolName::new(CORE_META_DESCRIPTION), [doc].into()),
        );
    }

    Ok(meta.into())
}

fn parse_doc_comment<'a>(token: Token<'a>) -> ParseResult<&'a str> {
    let comment = token.slice().strip_prefix("///").unwrap().trim();
    println!("Comment: `{}`", comment);
    Ok(comment)
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
            let symbol = inner.parse_one("symbol_name", parse_symbol_name)?;
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
        "var_name" => {
            let var = parse_var_name(node)?;
            Ok(Term::Var(var))
        }
        "literal" => {
            let value = parse_literal(node)?;
            Ok(Term::Literal(value))
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
    assert_eq!(token.name(), "symbol_name");
    token.slice().parse().map_err(|error| ParseError::Name {
        error,
        location: token.range(),
    })
}

fn parse_link_name(token: Token) -> ParseResult<LinkName> {
    assert_eq!(token.name(), "link_name");
    token.slice().parse().map_err(|error| ParseError::Name {
        error,
        location: token.range(),
    })
}

fn parse_var_name(token: Token) -> ParseResult<VarName> {
    assert_eq!(token.name(), "var_name");
    token.slice().parse().map_err(|error| ParseError::Name {
        error,
        location: token.range(),
    })
}

fn parse_literal(token: Token) -> ParseResult<Literal> {
    assert_eq!(token.name(), "literal");
    token.slice().parse().map_err(|error| ParseError::Literal {
        error,
        location: token.range(),
    })
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

/// Sequence of [`Token`]s.
struct Tokens<'a> {
    /// The current token or `None` if the sequence is empty.
    token: Option<Token<'a>>,

    /// The number of tokens remaining.
    count: usize,
}

impl<'a> Tokens<'a> {
    /// Create an iterator over all subsequent tokens for a specified rule.
    pub fn take_rule<'p>(&'p mut self, rule: &'static str) -> RuleTokens<'p, 'a> {
        RuleTokens { parser: self, rule }
    }

    /// Parse one token for a specified rule.
    ///
    /// Use this method when the grammar guarantees that such a token will be present.
    ///
    /// # Panics
    ///
    /// Panics when there are no tokens left or the next token is for a different rule.
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

    /// Parse many tokens for a specified rule.
    pub fn parse_many<T, C, F>(&mut self, rule: &'static str, parse: F) -> ParseResult<C>
    where
        C: FromIterator<T>,
        F: FnMut(Token<'a>) -> ParseResult<T>,
    {
        self.take_rule(rule).map(parse).try_collect()
    }

    /// Parse up to one token for a specified rule.
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
        debug_assert!(!token.node.is_error());
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

impl<'a> ExactSizeIterator for Tokens<'a> {}
impl<'a> FusedIterator for Tokens<'a> {}

/// Sequence of [`Token`]s for a specific rule.
struct RuleTokens<'p, 'a> {
    parser: &'p mut Tokens<'a>,
    rule: &'static str,
}

impl<'p, 'a> Iterator for RuleTokens<'p, 'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.parser.token?;

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

impl<'p, 'a> FusedIterator for RuleTokens<'p, 'a> {}

struct AllNodes<'a> {
    cursor: Option<TreeCursor<'a>>,
    size: usize,
}

impl<'a> AllNodes<'a> {
    pub fn new(cursor: TreeCursor<'a>) -> Self {
        Self {
            size: cursor.node().descendant_count(),
            cursor: Some(cursor),
        }
    }
}

impl<'a> Iterator for AllNodes<'a> {
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
