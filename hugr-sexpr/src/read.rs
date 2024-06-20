//! Reading s-expressions from strings.
use logos::Logos;
use smol_str::SmolStr;
use std::collections::VecDeque;
use std::ops::Range;
use thiserror::Error;

use crate::input::{Input, InputStream, ParseError, TokenTree};
use crate::Symbol;

#[derive(Debug, Clone, PartialEq, Eq, Logos)]
#[logos(skip r"[ \t\n\f]+")]
enum Token {
    #[token("(", |_| 0)]
    OpenList(usize),

    #[token(")")]
    CloseList,

    #[regex(
        r#""([^"\\]|\\["\\bnfrt]|u[a-fA-F0-9]{4})*""#,
        |lex| unescape_string(&lex.slice()[1..lex.slice().len() - 1])
    )]
    String(SmolStr),

    #[regex(
        "[a-zA-Z!$%&*\\./<>=@\\^_~][a-zA-Z0-9!$%&*+\\-\\./:<>=@\\^_~]*",
        |lex| Symbol::new(lex.slice())
    )]
    Symbol(Symbol),

    #[regex(";[^\n]*\n")]
    Comment,

    #[token("#t", |_| Some(true))]
    #[token("#f", |_| Some(false))]
    Bool(bool),

    // TODO: Better number parsing
    #[regex("[+-]?[0-9]+", |lex| lex.slice().parse().map_err(|_| ()))]
    Int(i64),
}

/// Span within a string.
pub type Span = Range<usize>;

/// Error while reading a value from an s-expression string.
#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum ReadError {
    #[error("unrecognized syntax")]
    Syntax { span: Span },
    #[error("unexpected end of file")]
    EndOfFile,
    #[error("unexpected closing delimiter")]
    UnexpectedClose { span: Span },
    #[error(transparent)]
    Parse(#[from] ParseError<Span>),
}

/// Read a value of type `T` from an s-expression string.
pub fn from_str<T>(str: &str) -> Result<T, ReadError>
where
    T: for<'a> Input<ReaderStream<'a>>,
{
    let mut tokens: Vec<_> = Token::lexer(str)
        .spanned()
        .filter(|(token, _)| !matches!(token, Ok(Token::Comment)))
        .map(|(token, span)| match token {
            Ok(token) => Ok((token, span)),
            Err(()) => Err(ReadError::Syntax { span: span.clone() }),
        })
        .collect::<Result<_, _>>()?;

    balance_lists(&mut tokens)?;

    let result = T::parse(&mut ReaderStream {
        tokens: &tokens,
        cur_span: 0..0,
        parent_span: 0..str.len(),
    })?;

    Ok(result)
}

/// Replaces escape sequences with their corresponding characters.
fn unescape_string(str: &str) -> Option<SmolStr> {
    let mut input = str.chars().collect::<VecDeque<_>>();
    let mut output = String::new();
    let mut unicode = String::new();

    while let Some(c) = input.pop_front() {
        if c != '\\' {
            output.push(c);
            continue;
        }

        match input.pop_front() {
            Some('b') => output.push('\u{0008}'),
            Some('n') => output.push('\n'),
            Some('f') => output.push('\u{000C}'),
            Some('r') => output.push('\r'),
            Some('t') => output.push('\t'),
            Some('"') => output.push('"'),
            Some('\\') => output.push('\\'),
            Some('u') => {
                unicode.extend(input.drain(..4));
                let codepoint = u32::from_str_radix(&unicode, 16).ok()?;
                unicode.clear();
                output.push(char::from_u32(codepoint)?);
            }
            _ => return None,
        }
    }

    Some(output.into())
}

/// Check that the parentheses are well-balanced and make the OpenList
/// tokens reflect the distance to their associated CloseList tokens.
fn balance_lists(tokens: &mut [(Token, Span)]) -> Result<(), ReadError> {
    // Stack that holds the indices of all currently unclosed `(`s.
    let mut stack = Vec::new();

    for i in 0..tokens.len() {
        let (token, span) = &tokens[i];

        match token {
            Token::OpenList(_) => stack.push(i),
            Token::CloseList => {
                let Some(j) = stack.pop() else {
                    return Err(ReadError::UnexpectedClose { span: span.clone() });
                };

                tokens[j].0 = Token::OpenList(i - j);
            }
            _ => {}
        }
    }

    if !stack.is_empty() {
        return Err(ReadError::EndOfFile);
    }

    Ok(())
}

/// Input stream used by [`from_str`].
#[derive(Clone)]
pub struct ReaderStream<'a> {
    tokens: &'a [(Token, Span)],
    cur_span: Span,
    parent_span: Span,
}

impl<'a> InputStream for ReaderStream<'a> {
    type Span = Span;

    fn next(&mut self) -> Option<TokenTree<Self>> {
        match self.peek()? {
            TokenTree::List(inner) => {
                self.cur_span = inner.parent_span.clone();
                self.tokens = &self.tokens[inner.tokens.len() + 1..];
                Some(TokenTree::List(inner))
            }
            token_tree => {
                self.cur_span = self.tokens[0].1.clone();
                self.tokens = &self.tokens[1..];
                Some(token_tree)
            }
        }
    }

    fn peek(&self) -> Option<TokenTree<Self>> {
        let (token, span) = self.tokens.first()?;

        match token {
            Token::OpenList(skip) => Some(TokenTree::List(ReaderStream {
                tokens: &self.tokens[1..=*skip],
                cur_span: span.end..span.end,
                parent_span: span.end..self.tokens[*skip].1.end,
            })),
            Token::CloseList => None,
            Token::String(string) => Some(TokenTree::String(string.clone())),
            Token::Symbol(symbol) => Some(TokenTree::Symbol(symbol.clone())),
            Token::Comment => unreachable!("comments have been stripped before"),
            Token::Bool(bool) => Some(TokenTree::Bool(*bool)),
            Token::Int(int) => Some(TokenTree::Int(*int)),
        }
    }

    fn span(&self) -> Self::Span {
        self.cur_span.clone()
    }

    fn parent_span(&self) -> Self::Span {
        self.parent_span.clone()
    }
}
