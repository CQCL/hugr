//! Types that can be constructed from s-expressions.
use smol_str::SmolStr;
use std::{error::Error, fmt::Display};
use thiserror::Error;

use crate::{Symbol, Value};

/// Input streams that emits s-expression tokens.
pub trait InputStream: Sized {
    /// Span that identifies the location of a token.
    type Span;

    /// Advance to the next token and return it.
    fn next(&mut self) -> Option<TokenTree<Self>>;

    /// Return the next token without advancing.
    fn peek(&self) -> Option<TokenTree<Self>>;

    /// The span of the last token returned by [`InputStream::next`].
    fn span(&self) -> Self::Span;

    /// The span of the parent stream, if any.
    ///
    /// For the root stream of some input, this returns the span of the entire input.
    fn parent_span(&self) -> Self::Span;
}

impl InputStream for &[Value] {
    type Span = ();

    fn next(&mut self) -> Option<TokenTree<Self>> {
        let (value, rest) = self.split_first()?;
        *self = rest;
        Some(value_to_token(value))
    }

    fn peek(&self) -> Option<TokenTree<Self>> {
        let value = self.first()?;
        Some(value_to_token(value))
    }

    fn span(&self) -> Self::Span {}

    fn parent_span(&self) -> Self::Span {}
}

fn value_to_token(value: &Value) -> TokenTree<&[Value]> {
    match value {
        Value::List(list) => TokenTree::List(list),
        Value::String(string) => TokenTree::String(string.clone()),
        Value::Symbol(symbol) => TokenTree::Symbol(symbol.clone()),
        Value::Bool(bool) => TokenTree::Bool(*bool),
        Value::Int(int) => TokenTree::Int(*int),
    }
}

/// Types that can be constructed from s-expressions.
pub trait Input<I>: Sized
where
    I: InputStream,
{
    /// Parse a value by consuming tokens from an input stream.
    fn parse(stream: &mut I) -> Result<Self, ParseError<I::Span>>;
}

impl<I: InputStream> Input<I> for SmolStr {
    fn parse(stream: &mut I) -> Result<Self, ParseError<I::Span>> {
        let Some(TokenTree::String(string)) = stream.next() else {
            return Err(ParseError::new("expected string", stream.span()));
        };

        Ok(string)
    }
}

impl<I: InputStream> Input<I> for String {
    fn parse(stream: &mut I) -> Result<Self, ParseError<I::Span>> {
        let Some(TokenTree::String(string)) = stream.next() else {
            return Err(ParseError::new("expected string", stream.span()));
        };

        Ok(string.into())
    }
}

impl<I: InputStream> Input<I> for Symbol {
    fn parse(stream: &mut I) -> Result<Self, ParseError<I::Span>> {
        let Some(TokenTree::Symbol(symbol)) = stream.next() else {
            return Err(ParseError::new("expected symbol", stream.span()));
        };

        Ok(symbol)
    }
}

impl<I: InputStream> Input<I> for Value {
    fn parse(stream: &mut I) -> Result<Self, ParseError<I::Span>> {
        let Some(token_tree) = stream.next() else {
            return Err(ParseError::new("expected value", stream.span()));
        };

        let value = match token_tree {
            TokenTree::List(mut list) => Value::List(Input::parse(&mut list)?),
            TokenTree::String(string) => Value::String(string),
            TokenTree::Symbol(symbol) => Value::Symbol(symbol),
            TokenTree::Bool(bool) => Value::Bool(bool),
            TokenTree::Int(int) => Value::Int(int),
        };

        Ok(value)
    }
}

impl<I: InputStream> Input<I> for Vec<Value> {
    fn parse(stream: &mut I) -> Result<Self, ParseError<I::Span>> {
        let mut values = Vec::new();

        while let Some(token_tree) = stream.next() {
            values.push(match token_tree {
                TokenTree::List(mut list) => Value::List(Input::parse(&mut list)?),
                TokenTree::String(string) => Value::String(string),
                TokenTree::Symbol(symbol) => Value::Symbol(symbol),
                TokenTree::Bool(bool) => Value::Bool(bool),
                TokenTree::Int(int) => Value::Int(int),
            });
        }

        Ok(values)
    }
}

/// Error while parsing a value.
#[derive(Debug, Error)]
pub enum ParseError<S> {
    /// Parse error together with a span.
    #[error("{message}")]
    Error {
        /// Error message.
        message: String,
        /// Span that indicates where the error occured.
        span: S,
    },
    /// Custom errors
    #[error(transparent)]
    Other(#[from] Box<dyn Error + 'static>),
}

impl<S> ParseError<S> {
    /// Construct a new [`ParseError`] given a message and span.
    pub fn new(message: impl Display, span: S) -> Self {
        Self::Error {
            message: format!("{}", message),
            span,
        }
    }
}

/// Individual token returned by an [`InputStream`].
#[derive(Debug, Clone)]
pub enum TokenTree<L> {
    /// A list with a nested [`InputStream`].
    List(L),
    /// A string.
    String(SmolStr),
    /// A symbol.
    Symbol(Symbol),
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
}

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use hugr_sexpr_derive::Input;
