use std::fmt::{self, Debug, Display};

use itertools::Itertools;

/// Write a comma separated list of of some types.
/// Like debug_list, but using the Display instance rather than Debug,
/// and not adding surrounding square brackets.
pub fn display_list<T>(ts: &[T], f: &mut fmt::Formatter) -> fmt::Result
where
    T: Display,
{
    let mut first = true;
    for t in ts.iter() {
        if !first {
            f.write_str(", ")?;
        }
        t.fmt(f)?;
        if first {
            first = false;
        }
    }
    Ok(())
}

/// Collect a vector into an array.
pub fn collect_array<const N: usize, T: Debug>(arr: &[T]) -> [&T; N] {
    arr.iter().collect_vec().try_into().unwrap()
}

#[allow(dead_code)]
// Test only utils
#[cfg(test)]
pub(crate) mod test {
    /// Open a browser page to render a dot string graph.
    #[cfg(not(ci_run))]
    pub(crate) fn viz_dotstr(dotstr: &str) {
        let mut base: String = "https://dreampuf.github.io/GraphvizOnline/#".into();
        base.push_str(&urlencoding::encode(dotstr));
        webbrowser::open(&base).unwrap();
    }
}

#[derive(Clone, Debug, Eq)]
pub(crate) enum MaybeStatic<T: 'static> {
    Value(T),
    Static(&'static T),
}

impl<T> MaybeStatic<T> {
    pub(super) const fn new_static(v_ref: &'static T) -> Self {
        Self::Static(v_ref)
    }

    pub(super) const fn new_value(v: T) -> Self {
        Self::Value(v)
    }
}

impl<T: Clone> MaybeStatic<T> {
    pub(crate) fn into_inner(self) -> T {
        match self {
            MaybeStatic::Value(v) => v,
            MaybeStatic::Static(v_ref) => v_ref.clone(),
        }
    }
}

impl<T> AsRef<T> for MaybeStatic<T> {
    fn as_ref(&self) -> &T {
        match self {
            MaybeStatic::Value(v) => v,
            MaybeStatic::Static(v_ref) => v_ref,
        }
    }
}

// can use pointer equality to compare static instances
impl<T: PartialEq> PartialEq for MaybeStatic<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // pointer equality can give false-negative
            (Self::Static(l0), Self::Static(r0)) => std::ptr::eq(*l0, *r0) || l0 == r0,
            (Self::Value(l0), Self::Value(r0)) => l0 == r0,
            (Self::Value(v), Self::Static(v_ref)) | (Self::Static(v_ref), Self::Value(v)) => {
                v == *v_ref
            }
        }
    }
}
