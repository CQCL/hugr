use std::{borrow::Cow, fmt::Display};

use base64::{prelude::BASE64_STANDARD, Engine as _};
use pretty::{Arena, DocAllocator as _, RefDoc};

use super::{Link, ListPart, Symbol, Term, TuplePart, Var};

struct Printer<'a> {
    /// The arena in which to allocate the pretty-printed documents.
    arena: &'a Arena<'a>,
    /// Parts of the document to be concatenated.
    docs: Vec<RefDoc<'a>>,
    /// Stack of indices into `docs` denoting the current nesting.
    docs_stack: Vec<usize>,
}

fn print_to_fmt<P: Print>(f: &mut std::fmt::Formatter<'_>, what: &P) -> std::fmt::Result {
    let arena = Arena::new();
    let mut printer = Printer::new(&arena);
    printer.print(what);
    let doc = printer.finish();
    doc.render_fmt(80, f)
}

impl<'a> Printer<'a> {
    fn new(arena: &'a Arena<'a>) -> Self {
        Self {
            arena,
            docs: Vec::new(),
            docs_stack: Vec::new(),
        }
    }

    fn finish(self) -> RefDoc<'a> {
        let sep = self
            .arena
            .concat([self.arena.hardline(), self.arena.hardline()]);
        self.arena.intersperse(self.docs, sep).into_doc()
    }

    fn parens_enter(&mut self) {
        self.delim_open();
    }

    fn parens_exit(&mut self) {
        self.delim_close("(", ")", 2);
    }

    fn brackets_enter(&mut self) {
        self.delim_open();
    }

    fn brackets_exit(&mut self) {
        self.delim_close("[", "]", 1);
    }

    fn delim_open(&mut self) {
        self.docs_stack.push(self.docs.len());
    }

    fn delim_close(&mut self, open: &'static str, close: &'static str, nesting: isize) {
        let docs = self.docs.drain(self.docs_stack.pop().unwrap()..);
        let doc = self.arena.concat([
            self.arena.text(open),
            self.arena
                .intersperse(docs, self.arena.line())
                .nest(nesting)
                .group(),
            self.arena.text(close),
        ]);
        self.docs.push(doc.into_doc());
    }

    fn text(&mut self, text: impl Into<Cow<'a, str>>) {
        self.docs.push(self.arena.text(text).into_doc());
    }

    fn int(&mut self, value: u64) {
        self.text(format!("{}", value));
    }

    fn string(&mut self, string: &str) {
        let mut output = String::with_capacity(string.len() + 2);
        output.push('"');

        for c in string.chars() {
            match c {
                '\\' => output.push_str("\\\\"),
                '"' => output.push_str("\\\""),
                '\n' => output.push_str("\\n"),
                '\r' => output.push_str("\\r"),
                '\t' => output.push_str("\\t"),
                _ => output.push(c),
            }
        }

        output.push('"');
        self.text(output);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        // every 3 bytes are encoded into 4 characters
        let mut output = String::with_capacity(2 + bytes.len().div_ceil(3) * 4);
        output.push('"');
        BASE64_STANDARD.encode_string(bytes, &mut output);
        output.push('"');
        self.text(output);
    }

    fn print<P: Print>(&mut self, what: &'a P) {
        what.print(self)
    }
}

trait Print {
    fn print<'a>(&'a self, printer: &mut Printer<'a>);
}

impl Print for Term {
    fn print<'a>(&'a self, printer: &mut Printer<'a>) {
        match self {
            Term::Wildcard => printer.text("_"),
            Term::Var(var) => printer.print(var),
            Term::Apply(symbol, terms) => {
                if terms.is_empty() {
                    printer.print(symbol);
                } else {
                    printer.parens_enter();
                    printer.print(symbol);
                    printer.print(terms);
                    printer.parens_exit();
                }
            }
            Term::List(list_parts) => {
                printer.brackets_enter();
                printer.print(list_parts);
                printer.brackets_exit();
            }
            Term::Str(str) => {
                printer.string(str);
            }
            Term::Nat(nat) => {
                printer.int(*nat);
            }
            Term::Bytes(bytes) => {
                printer.bytes(bytes);
            }
            Term::Float(float) => {
                // The debug representation of a float always includes a decimal point.
                printer.text(format!("{:?}", float.into_inner()));
            }
            Term::Tuple(tuple_parts) => {
                printer.parens_enter();
                printer.text("tuple");
                printer.print(tuple_parts);
                printer.parens_exit();
            }
            Term::ExtSet => {
                printer.parens_enter();
                printer.text("ext");
                printer.parens_exit();
            }
        }
    }
}

impl Print for ListPart {
    fn print<'a>(&'a self, printer: &mut Printer<'a>) {
        match self {
            ListPart::Item(term) => {
                printer.print(term);
            }
            ListPart::Splice(term) => {
                printer.print(term);
                printer.text("...");
            }
        }
    }
}

impl Print for TuplePart {
    fn print<'a>(&'a self, printer: &mut Printer<'a>) {
        match self {
            TuplePart::Item(term) => {
                printer.print(term);
            }
            TuplePart::Splice(term) => {
                printer.print(term);
                printer.text("...");
            }
        }
    }
}

impl Print for Symbol {
    fn print<'a>(&'a self, printer: &mut Printer<'a>) {
        printer.text(self.0.as_str())
    }
}

impl Print for Var {
    fn print<'a>(&'a self, printer: &mut Printer<'a>) {
        printer.text(format!("{}", self))
    }
}

impl<P: Print> Print for Vec<P> {
    fn print<'a>(&'a self, printer: &mut Printer<'a>) {
        for item in self {
            printer.print(item);
        }
    }
}

macro_rules! impl_display {
    ($t:ident) => {
        impl Display for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                print_to_fmt(f, self)
            }
        }
    };
}

impl_display!(Term);
impl_display!(ListPart);
impl_display!(TuplePart);

impl Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "?{}", self.0)
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for Link {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}
