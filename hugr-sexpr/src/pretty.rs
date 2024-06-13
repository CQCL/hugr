use crate::Value;
use pretty::{Arena, DocAllocator, RefDoc};

/// Constructs a pretty printing document from a [`Value`].
pub fn value_to_doc<'a, 'v: 'a, A>(arena: &'a Arena<'a>, value: &'v Value<A>) -> RefDoc<'a> {
    match value {
        Value::List(values, _) => {
            let values_doc = values.iter().map(|value| value_to_doc(arena, value));
            let doc = arena.intersperse(values_doc, arena.line());

            arena
                .text("(")
                .append(doc.nest(2).group())
                .append(arena.text(")"))
                .into_doc()
        }

        Value::String(string, _) => {
            // TODO: Escape and deal with newlines
            // The pretty printer assumes that there are no newlines in the text
            arena
                .text("\"")
                .append(arena.text(string.as_str()))
                .append(arena.text("\""))
                .into_doc()
        }

        Value::Symbol(symbol, _) => {
            // TODO: Escape and deal with newlines
            // The pretty printer assumes that there are no newlines in the text
            arena.text(symbol.as_ref()).into_doc()
        }

        Value::Bool(bool, _) => match bool {
            true => arena.text("#t").into_doc(),
            false => arena.text("#f").into_doc(),
        },

        Value::Int(int, _) => arena.text(int.to_string()).into_doc(),
    }
}

/// Pretty prints a [`Value`] into a string.
pub fn value_to_string<A>(value: &Value<A>, width: usize) -> String {
    let mut string = String::new();
    // We can ignore the `Err` case here since writing to a string does not fail.
    let _ = write_value(value, width, &mut string);
    string
}

/// Pretty pints a [`Value`] into an [`std::fmt::Write`].
pub fn write_value<A, W>(value: &Value<A>, width: usize, f: &mut W) -> std::fmt::Result
where
    W: std::fmt::Write,
{
    let arena = Arena::new();
    let doc = value_to_doc(&arena, value);
    doc.render_fmt(width, f)
}
