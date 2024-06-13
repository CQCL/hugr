//! Export a [`Value`] from a custom type.

use crate::{Symbol, Value};

/// Trait for types that can be converted into a sequence of [`Value`]s.
pub trait Export<A> {
    /// Extend the given vector with the exported values.
    fn export_into(&self, into: &mut Vec<Value<A>>);
}

impl<A> Export<A> for String
where
    A: Default,
{
    fn export_into(&self, into: &mut Vec<Value<A>>) {
        into.push(Value::String(self.into(), A::default()));
    }
}

impl<A> Export<A> for Symbol
where
    A: Default,
{
    fn export_into(&self, into: &mut Vec<Value<A>>) {
        into.push(Value::Symbol(self.clone(), A::default()));
    }
}

impl<A> Export<A> for Vec<Value<A>>
where
    A: Clone,
{
    fn export_into(&self, into: &mut Vec<Value<A>>) {
        into.extend(self.iter().cloned())
    }
}

impl<A> Export<A> for Value<A>
where
    A: Clone,
{
    fn export_into(&self, into: &mut Vec<Value<A>>) {
        into.push(self.clone());
    }
}

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use hugr_sexpr_derive::Export;

/// Export a type into a [`Vec`] of values.
pub fn export_values<A, T>(object: &T) -> Vec<Value<A>>
where
    T: Export<A>,
{
    let mut values = Vec::new();
    object.export_into(&mut values);
    values
}
