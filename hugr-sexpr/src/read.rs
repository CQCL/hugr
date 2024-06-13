//! Read [`Value`]s from a string.
use logos::Logos;
use std::ops::Range;
use thiserror::Error;

use crate::{Symbol, Value};

// TODO: Unescape strings!

#[derive(Debug, Clone, PartialEq, Eq, Logos)]
#[logos(skip r"[ \t\n\f]+")]
enum Token {
    #[token("(")]
    OpenList,

    #[token(")")]
    CloseList,

    #[regex(
        r#""([^"\\]|\\["\\bnfrt]|u[a-fA-F0-9]{4})*""#,
        |lex| String::from(&lex.slice()[1..lex.slice().len() - 1])
    )]
    String(String),

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

/// Error while reading a [`Value`] from a string.
#[derive(Debug, Clone, Error)]
#[allow(missing_docs)]
pub enum ReadError {
    #[error("unrecognized syntax")]
    Syntax { span: Span },
    #[error("unexpected end of file")]
    EndOfFile,
    #[error("unexpected closing delimiter")]
    UnexpectedClose { span: Span },
}

/// Reads a sequence of [`Value`]s from a string.
pub fn read_values(input: &str) -> Result<Vec<Value>, ReadError> {
    // TODO: Avoid putting in spans in the first place
    Ok(read_values_with_span(input)?
        .into_iter()
        .map(|value| value.map_meta(|_| ()))
        .collect())
}

/// Reads a sequence of [`Value`]s from a string, including spans.
pub fn read_values_with_span(input: &str) -> Result<Vec<Value<Span>>, ReadError> {
    let lexer = Token::lexer(input);
    let mut tokens = lexer.spanned().map(|(token, span)| {
        let token = token.map_err(|()| ReadError::Syntax { span: span.clone() });
        (token, span)
    });
    read_values_from_lexer(&mut tokens)
}

/// Reads a sequence of top level values.
fn read_values_from_lexer(
    lexer: &mut impl Iterator<Item = (Result<Token, ReadError>, Span)>,
) -> Result<Vec<Value<Span>>, ReadError> {
    let mut values = Vec::new();

    while let Some((token, span)) = lexer.next() {
        let token = token?;

        let value = match token {
            Token::OpenList => read_list_from_lexer(lexer, span.start)?,
            Token::String(string) => Value::String(string, span),
            Token::Symbol(symbol) => Value::Symbol(symbol, span),
            Token::Comment => continue,
            Token::Bool(bool) => Value::Bool(bool, span),
            Token::Int(int) => Value::Int(int, span),
            Token::CloseList => return Err(ReadError::UnexpectedClose { span }),
        };

        values.push(value);
    }

    Ok(values)
}

/// Reads a list value, assuming that the opening `(` has already been read.
fn read_list_from_lexer(
    lexer: &mut impl Iterator<Item = (Result<Token, ReadError>, Span)>,
    open_loc: usize,
) -> Result<Value<Span>, ReadError> {
    let mut values = Vec::new();

    while let Some((token, span)) = lexer.next() {
        let token = token?;

        let value = match token {
            Token::OpenList => read_list_from_lexer(lexer, span.start)?,
            Token::String(string) => Value::String(string, span),
            Token::Symbol(symbol) => Value::Symbol(symbol, span),
            Token::Comment => continue,
            Token::Bool(bool) => Value::Bool(bool, span),
            Token::Int(int) => Value::Int(int, span),
            Token::CloseList => return Ok(Value::List(values, open_loc..span.end)),
        };

        values.push(value);
    }

    Err(ReadError::EndOfFile)
}
