use std::{borrow::Cow, fmt::Display};

use base64::{prelude::BASE64_STANDARD, Engine as _};
use pretty::{Arena, DocAllocator as _, RefDoc};

use crate::v0::RegionKind;

use super::{
    LinkName, ListPart, Module, Node, Operation, Param, Region, Symbol, SymbolName, Term,
    TuplePart, VarName,
};

struct Printer<'a> {
    /// The arena in which to allocate the pretty-printed documents.
    arena: &'a Arena<'a>,
    /// Parts of the document to be concatenated.
    docs: Vec<RefDoc<'a>>,
    /// Stack of indices into `docs` denoting the current nesting.
    docs_stack: Vec<usize>,
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

    fn group_enter(&mut self) {
        self.delim_open();
    }

    fn group_exit(&mut self) {
        self.delim_close("", "", 0);
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
}

fn print_term<'a>(printer: &mut Printer<'a>, term: &'a Term) {
    match term {
        Term::Wildcard => printer.text("_"),
        Term::Var(var) => print_var_name(printer, var),
        Term::Apply(symbol, terms) => {
            if terms.is_empty() {
                print_symbol_name(printer, symbol);
            } else {
                printer.parens_enter();
                print_symbol_name(printer, symbol);

                for term in terms.iter() {
                    print_term(printer, term);
                }

                printer.parens_exit();
            }
        }
        Term::List(list_parts) => {
            printer.brackets_enter();
            for part in list_parts.iter() {
                print_list_part(printer, part);
            }
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
            for part in tuple_parts.iter() {
                print_tuple_part(printer, part);
            }
            printer.parens_exit();
        }
        Term::ExtSet => {
            printer.parens_enter();
            printer.text("ext");
            printer.parens_exit();
        }
        Term::Func(region) => {
            printer.parens_enter();
            printer.text("fn");
            print_region(printer, region);
            printer.parens_exit();
        }
    }
}

fn print_list_part<'a>(printer: &mut Printer<'a>, part: &'a ListPart) {
    match part {
        ListPart::Item(term) => {
            print_term(printer, term);
        }
        ListPart::Splice(term) => {
            printer.group_enter();
            print_term(printer, term);
            printer.text("...");
            printer.group_exit();
        }
    }
}

fn print_tuple_part<'a>(printer: &mut Printer<'a>, part: &'a TuplePart) {
    match part {
        TuplePart::Item(term) => {
            print_term(printer, term);
        }
        TuplePart::Splice(term) => {
            printer.group_enter();
            print_term(printer, term);
            printer.text("...");
            printer.group_exit();
        }
    }
}

fn print_symbol_name<'a>(printer: &mut Printer<'a>, name: &'a SymbolName) {
    printer.text(name.0.as_str())
}

fn print_var_name<'a>(printer: &mut Printer<'a>, name: &'a VarName) {
    printer.text(format!("{}", name))
}

fn print_link_name<'a>(printer: &mut Printer<'a>, name: &'a LinkName) {
    printer.text(format!("{}", name))
}

fn print_port_lists<'a>(
    printer: &mut Printer<'a>,
    inputs: &'a [LinkName],
    outputs: &'a [LinkName],
) {
    if inputs.is_empty() && outputs.is_empty() {
        return;
    }

    printer.brackets_enter();
    for input in inputs {
        print_link_name(printer, input);
    }
    printer.brackets_exit();

    printer.brackets_enter();
    for output in outputs {
        print_link_name(printer, output);
    }
    printer.brackets_exit();
}

fn print_module<'a>(printer: &mut Printer<'a>, module: &'a Module) {
    printer.parens_enter();
    printer.text("hugr");
    printer.text("0");
    printer.parens_exit();

    for meta in &module.root.meta {
        print_meta_item(printer, meta);
    }

    for child in &module.root.children {
        print_node(printer, child);
    }
}

fn print_node<'a>(printer: &mut Printer<'a>, node: &'a Node) {
    printer.parens_enter();

    printer.group_enter();
    match &node.operation {
        Operation::Invalid => printer.text("invalid"),
        Operation::Dfg => printer.text("dfg"),
        Operation::Cfg => printer.text("cfg"),
        Operation::Block => printer.text("block"),
        Operation::DefineFunc(symbol_signature) => {
            printer.text("define-func");
            print_symbol(printer, symbol_signature);
        }
        Operation::DeclareFunc(symbol_signature) => {
            printer.text("declare-func");
            print_symbol(printer, symbol_signature);
        }
        Operation::Custom(term) => {
            print_term(printer, term);
        }
        Operation::DefineAlias(symbol_signature, value) => {
            printer.text("define-alias");
            print_symbol(printer, symbol_signature);
            print_term(printer, value);
        }
        Operation::DeclareAlias(symbol_signature) => {
            printer.text("declare-alias");
            print_symbol(printer, symbol_signature);
        }
        Operation::TailLoop => printer.text("tail-loop"),
        Operation::Conditional => printer.text("cond"),
        Operation::DeclareConstructor(symbol_signature) => {
            printer.text("declare-ctr");
            print_symbol(printer, symbol_signature);
        }
        Operation::DeclareOperation(symbol_signature) => {
            printer.text("declare-operation");
            print_symbol(printer, symbol_signature);
        }
        Operation::Import(symbol) => {
            printer.text("import");
            print_symbol_name(printer, symbol);
        }
    }

    print_port_lists(printer, &node.inputs, &node.outputs);
    printer.group_exit();

    if let Some(signature) = &node.signature {
        print_signature(printer, signature);
    }

    for meta in &node.meta {
        print_meta_item(printer, meta);
    }

    for region in &node.regions {
        print_region(printer, region);
    }

    printer.parens_exit();
}

fn print_region<'a>(printer: &mut Printer<'a>, region: &'a Region) {
    printer.parens_enter();
    printer.group_enter();

    printer.text(match region.kind {
        RegionKind::DataFlow => "dfg",
        RegionKind::ControlFlow => "cfg",
        RegionKind::Module => "mod",
    });

    print_port_lists(printer, &region.sources, &region.targets);
    printer.group_exit();

    if let Some(signature) = &region.signature {
        print_signature(printer, signature);
    }

    for meta in &region.meta {
        print_meta_item(printer, meta);
    }

    for child in &region.children {
        print_node(printer, child);
    }

    printer.parens_exit();
}

fn print_symbol<'a>(printer: &mut Printer<'a>, symbol: &'a Symbol) {
    print_symbol_name(printer, &symbol.name);

    for param in symbol.params.iter() {
        print_param(printer, param);
    }

    for constraint in symbol.constraints.iter() {
        print_constraint(printer, constraint);
    }

    print_term(printer, &symbol.signature);
}

fn print_param<'a>(printer: &mut Printer<'a>, param: &'a Param) {
    printer.parens_enter();
    printer.text("param");
    print_var_name(printer, &param.name);
    print_term(printer, &param.r#type);
    printer.parens_exit();
}

fn print_constraint<'a>(printer: &mut Printer<'a>, constraint: &'a Term) {
    printer.parens_enter();
    printer.text("where");
    print_term(printer, &constraint);
    printer.parens_exit();
}

fn print_meta_item<'a>(printer: &mut Printer<'a>, meta: &'a Term) {
    printer.parens_enter();
    printer.text("meta");
    print_term(printer, &meta);
    printer.parens_exit();
}

fn print_signature<'a>(printer: &mut Printer<'a>, signature: &'a Term) {
    printer.parens_enter();
    printer.text("signature");
    print_term(printer, &signature);
    printer.parens_exit();
}

macro_rules! impl_display {
    ($t:ident, $print:expr) => {
        impl Display for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let arena = Arena::new();
                let mut printer = Printer::new(&arena);
                $print(&mut printer, self);
                let doc = printer.finish();
                doc.render_fmt(80, f)
            }
        }
    };
}

impl_display!(Module, print_module);
impl_display!(Node, print_node);
impl_display!(Region, print_region);
impl_display!(Param, print_param);
impl_display!(Term, print_term);
impl_display!(ListPart, print_list_part);
impl_display!(TuplePart, print_tuple_part);

impl Display for VarName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "?{}", self.0)
    }
}

impl Display for SymbolName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for LinkName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}
