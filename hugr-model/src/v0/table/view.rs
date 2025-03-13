use super::Module;
use std::sync::Arc;

/// Trait for views into a [`Module`].
pub trait View<'a, S>: Sized {
    /// Attempt to interpret a subpart of a module as this type.
    fn view(module: &'a Module<'a>, src: S) -> Option<Self>;
}

impl<'a, S, T> View<'a, &S> for T
where
    T: View<'a, S>,
    S: Copy,
{
    fn view(module: &'a Module<'a>, id: &S) -> Option<Self> {
        T::view(module, *id)
    }
}

impl<'a, S, T> View<'a, Option<S>> for Option<T>
where
    T: View<'a, S>,
{
    fn view(module: &'a Module<'a>, src: Option<S>) -> Option<Self> {
        Some(match src {
            Some(src) => Some(T::view(module, src)?),
            None => None,
        })
    }
}

impl<'a, S, T> View<'a, &'a [S]> for Vec<T>
where
    T: View<'a, &'a S>,
{
    fn view(module: &'a Module<'a>, sources: &'a [S]) -> Option<Self> {
        sources
            .iter()
            .map(|source| T::view(module, source))
            .collect()
    }
}

impl<'a, S, T> View<'a, &'a [S]> for Box<[T]>
where
    T: View<'a, &'a S>,
{
    fn view(module: &'a Module<'a>, sources: &'a [S]) -> Option<Self> {
        sources
            .iter()
            .map(|source| T::view(module, source))
            .collect()
    }
}

impl<'a, S, T> View<'a, &'a [S]> for Arc<[T]>
where
    T: View<'a, &'a S>,
{
    fn view(module: &'a Module<'a>, sources: &'a [S]) -> Option<Self> {
        sources
            .iter()
            .map(|source| T::view(module, source))
            .collect()
    }
}
