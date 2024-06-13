//! Import a [`Value`] into a custom type.

use crate::{Symbol, Value};
use std::fmt::Display;
use thiserror::Error;

/// Trait for types that can be imported from a sequence of [`Value`]s.
pub trait Import<'a, A>
where
    A: Clone,
{
    /// Import this type from the given sequence of [`Value`]s.
    /// On success, return the imported object together with the remaining values.
    fn import(values: &'a [Value<A>]) -> ImportResult<'a, Self, A>
    where
        Self: Sized;
}

/// Result type for [`Import::import`].
pub type ImportResult<'a, O, A> = Result<(&'a [Value<A>], O), ImportError<A>>;

impl<'a, A> Import<'a, A> for String
where
    A: Clone,
{
    fn import(values: &'a [Value<A>]) -> ImportResult<'a, Self, A>
    where
        Self: Sized,
    {
        let (value, values) = values
            .split_first()
            .ok_or_else(|| ImportError::new("expected string"))?;

        let value = value
            .as_string()
            .ok_or_else(|| ImportError::new_with_meta("expected string", value.meta().clone()))?;

        Ok((values, value.to_string()))
    }
}

impl<'a, A> Import<'a, A> for Symbol
where
    A: Clone,
{
    fn import(values: &'a [Value<A>]) -> ImportResult<'a, Self, A>
    where
        Self: Sized,
    {
        let (value, values) = values
            .split_first()
            .ok_or_else(|| ImportError::new("expected symbol"))?;

        let value = value
            .as_symbol()
            .ok_or_else(|| ImportError::new_with_meta("expected symbol", value.meta().clone()))?;

        Ok((values, value.clone()))
    }
}

impl<'a, A> Import<'a, A> for Vec<Value<A>>
where
    A: Clone,
{
    fn import(values: &'a [Value<A>]) -> ImportResult<'a, Self, A>
    where
        Self: Sized,
    {
        Ok((&[], values.to_vec()))
    }
}

impl<'a, A> Import<'a, A> for Value<A>
where
    A: Clone,
{
    fn import(values: &'a [Value<A>]) -> ImportResult<'a, Self, A>
    where
        Self: Sized,
    {
        let (value, values) = values
            .split_first()
            .ok_or_else(|| ImportError::new("expected value"))?;

        Ok((values, value.clone()))
    }
}

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use hugr_sexpr_derive::Import;

/// Error while importing from a [`Value`].
#[derive(Debug, Clone, Error)]
#[error("{message}")]
pub struct ImportError<A> {
    message: String,
    meta: Option<A>,
}

impl<A> ImportError<A> {
    /// Create a new [`ImportError`] with the given message.
    pub fn new(message: impl Display) -> Self {
        Self {
            message: format!("{}", message),
            meta: None,
        }
    }

    /// Create a new [`ImportError`] with the given message and metadata.
    pub fn new_with_meta(message: impl Display, meta: A) -> Self {
        Self {
            message: format!("{}", message),
            meta: Some(meta),
        }
    }

    /// The metadata attached to the [`ImportError`].
    pub fn meta(&self) -> Option<&A> {
        self.meta.as_ref()
    }
}
