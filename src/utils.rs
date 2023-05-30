use std::fmt::{self, Display};

pub fn display_list<T>(ts: &[T], f: &mut fmt::Formatter) -> fmt::Result
where
    T: Display,
{
    if ts.is_empty() {
        return Ok(());
    }
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

#[allow(dead_code)]
// Test only utils
#[cfg(test)]
pub(crate) mod test {
    /// Open a browser page to render a dot string graph.
    #[cfg(not(ci_run))]
    pub(crate) fn viz_dotstr(dotstr: &str) {
        let mut base: String = "https://dreampuf.github.io/GraphvizOnline/#".into();
        url_escape::encode_query_to_string(dotstr, &mut base);
        webbrowser::open(&base).unwrap();
    }
}
