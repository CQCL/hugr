use super::{SeqPart, Term};
use crate::v0::{CORE_FN, Literal, SymbolName};
use itertools::Itertools;
use std::sync::Arc;

#[rust_sitter::grammar("hugr")]
mod grammar {
    use smol_str::{SmolStr, ToSmolStr as _};

    #[rust_sitter::language]
    #[derive(Debug, Clone)]
    pub struct Package {
        #[rust_sitter::repeat]
        modules: Vec<Module>,
    }

    #[derive(Debug, Clone)]
    pub struct Module {
        #[rust_sitter::repeat]
        pub doc: Vec<DocComment>,
        #[rust_sitter::repeat]
        pub meta: Vec<Meta>,
        #[rust_sitter::leaf(text = "mod")]
        _mod: (),
        #[rust_sitter::leaf(text = "{")]
        _open: (),
        #[rust_sitter::repeat]
        items: Vec<Node>,
        #[rust_sitter::leaf(text = "}")]
        _close: (),
    }

    #[derive(Debug, Clone)]
    pub struct FunctionSymbol {
        #[rust_sitter::repeat]
        pub doc: Vec<DocComment>,
        #[rust_sitter::repeat]
        pub meta: Vec<Meta>,
        pub visibility: Option<Public>,
        #[rust_sitter::leaf(text = "fn")]
        _fn: (),
        pub name: Symbol,
        pub body: Option<Region>,
        pub signature: Option<Signature>,
        #[rust_sitter::leaf(text = ";")]
        _end: (),
    }

    #[derive(Debug, Clone)]
    pub struct CtrSymbol {
        #[rust_sitter::repeat]
        pub doc: Vec<DocComment>,
        #[rust_sitter::repeat]
        pub meta: Vec<Meta>,
        pub visibility: Option<Public>,
        #[rust_sitter::leaf(text = "ctr")]
        _ctr: (),
        pub name: Symbol,
        pub signature: Option<Signature>,
        #[rust_sitter::leaf(text = ";")]
        _end: (),
    }

    #[derive(Debug, Clone)]
    pub enum Region {
        Data {
            #[rust_sitter::leaf(text = "{")]
            _open: (),
            #[rust_sitter::repeat]
            meta: Vec<Meta>,
            #[rust_sitter::repeat]
            body: Vec<Operation>,
            #[rust_sitter::leaf(text = "}")]
            _close: (),
        },
    }

    #[derive(Debug, Clone)]
    pub enum Operation {
        Term(Term),
        Function {
            #[rust_sitter::leaf(text = "fn")]
            _fn: (),
            symbol: Symbol,
        },
    }

    #[derive(Debug, Clone)]
    pub struct Symbol {
        pub name: SymbolName,
        pub params: Option<Parameters>,
        pub constraints: Option<Constraints>,
    }

    #[derive(Debug, Clone)]
    pub struct Parameters {
        #[rust_sitter::leaf(text = "(")]
        _open: (),
        #[rust_sitter::repeat]
        #[rust_sitter::delimiter(rust_sitter::leaf(text = ","))]
        parameters: Vec<Parameter>,
        #[rust_sitter::leaf(text = ")")]
        _close: (),
    }

    #[derive(Debug, Clone)]
    pub struct Parameter {
        pub name: VarName,
        #[rust_sitter::leaf(text = ":")]
        _colon: (),
        pub type_: Term,
    }

    #[derive(Debug, Clone)]
    pub struct Constraints {
        #[rust_sitter::leaf(text = "where")]
        _where: (),
        #[rust_sitter::repeat]
        #[rust_sitter::delimiter(rust_sitter::leaf(text = ","))]
        constraints: Vec<Term>,
    }

    #[derive(Debug, Clone)]
    pub struct Node {
        #[rust_sitter::repeat]
        pub meta: Vec<Meta>,
        pub outputs: Option<OperationOutputs>,
        pub operation: Operation,
        #[rust_sitter::repeat]
        #[rust_sitter::delimited(rust_sitter::leaf(","))]
        pub inputs: Vec<Link>,
        pub signature: Option<Signature>,
        #[rust_sitter::leaf(text = ";")]
        _end: (),
    }

    #[derive(Debug, Clone)]
    pub struct Signature {
        #[rust_sitter::leaf(text = ":")]
        _sep: (),
        pub term: Term,
    }

    #[derive(Debug, Clone)]
    pub struct OperationOutputs {
        #[rust_sitter::repeat]
        #[rust_sitter::delimited(rust_sitter::leaf(","))]
        outputs: Vec<Link>,
        #[rust_sitter::leaf(text = "=")]
        _equals: (),
    }

    #[derive(Debug, Clone)]
    pub struct Meta {
        #[rust_sitter::leaf(text = "#[")]
        _open: (),
        pub term: Term,
        #[rust_sitter::leaf(text = "]")]
        _close: (),
    }

    #[derive(Debug, Clone)]
    pub enum Term {
        #[rust_sitter::leaf(text = "_")]
        Wildcard,
        Apply {
            symbol: Symbol,
            args: Option<Arguments>,
        },
        List {
            #[rust_sitter::leaf(text = "[")]
            _open: (),
            #[rust_sitter::repeat]
            #[rust_sitter::delimited(rust_sitter::leaf(text = ","))]
            parts: Vec<SeqPart>,
            #[rust_sitter::leaf(text = "]")]
            _close: (),
        },
        Nat(Nat),
        #[rust_sitter::prec_right(1)]
        FuncType {
            inputs: Box<Term>,
            #[rust_sitter::leaf(text = "->")]
            _arrow: (),
            outputs: Box<Term>,
        },
    }

    #[derive(Debug, Clone)]
    pub enum SeqPart {
        Item(Term),
        Splice {
            #[rust_sitter::leaf(text = "...")]
            _ellipsis: (),
            term: Term,
        },
    }

    #[derive(Debug, Clone)]
    pub struct Arguments {
        #[rust_sitter::leaf(text = "(")]
        _open: (),
        #[rust_sitter::repeat]
        #[rust_sitter::delimited(rust_sitter::leaf(text = ","))]
        pub args: Vec<Term>,
        #[rust_sitter::leaf(text = ")")]
        _close: (),
    }

    #[derive(Debug, Clone)]
    pub struct SymbolName(
        #[rust_sitter::leaf(pattern = "@(([0-9]+)|([a-zA-Z_][\\.a-zA-Z0-9_]*))", transform = |s| s.to_smolstr())]
        pub SmolStr,
    );

    #[derive(Debug, Clone)]
    pub struct Link(
        #[rust_sitter::leaf(pattern = "%(([0-9]+)|([a-zA-Z_][\\.a-zA-Z0-9_]*))", transform = |s| s.to_smolstr())]
        pub SmolStr,
    );

    #[derive(Debug, Clone)]
    pub struct VarName(
        #[rust_sitter::leaf(pattern = "\\?(([0-9]+)|([a-zA-Z_][\\.a-zA-Z0-9_]*))", transform = |s| s.to_smolstr())]
        pub SmolStr,
    );

    #[derive(Debug, Clone)]
    pub struct Nat(
        #[rust_sitter::leaf(pattern = r"[1-9][0-9]*", transform = |s| s.parse().unwrap())] pub u64,
    );

    #[rust_sitter::leaf(text = "pub")]
    #[derive(Debug, Clone)]
    struct Public;

    #[rust_sitter::extra]
    #[rust_sitter::leaf(pattern = r"\s")]
    struct Whitespace;

    #[rust_sitter::extra]
    #[rust_sitter::leaf(pattern = r"//[^/].*")]
    struct Comment;

    #[rust_sitter::extra]
    #[rust_sitter::leaf(pattern = r"\r?\n")]
    struct Newline;

    #[derive(Debug, Clone)]
    pub struct DocComment(
        #[rust_sitter::leaf(pattern = r"///[^\r\n]*\r?\n", transform = |s| s.parse().unwrap())]
        pub SmolStr,
    );
}

impl From<grammar::Symbol> for SymbolName {
    fn from(value: grammar::Symbol) -> Self {
        SymbolName::new(value.0)
    }
}

impl From<grammar::Term> for Term {
    fn from(value: grammar::Term) -> Self {
        match value {
            grammar::Term::Wildcard => Self::Wildcard,
            grammar::Term::Apply { symbol, args } => {
                let symbol = symbol.into();
                let args = args.map(|args| args.into()).unwrap_or_default();
                Self::Apply(symbol, args)
            }
            grammar::Term::List { parts, .. } => {
                Self::List(parts.into_iter().map(SeqPart::from).collect())
            }
            grammar::Term::Nat(nat) => Self::Literal(nat.into()),
            grammar::Term::FuncType {
                inputs, outputs, ..
            } => Self::Apply(
                SymbolName::new(CORE_FN),
                vec![(*inputs).into(), (*outputs).into()].into(),
            ),
        }
    }
}

impl From<grammar::Arguments> for Arc<[Term]> {
    fn from(value: grammar::Arguments) -> Self {
        value.args.into_iter().map_into().collect()
    }
}

impl From<grammar::SeqPart> for SeqPart {
    fn from(value: grammar::SeqPart) -> Self {
        match value {
            grammar::SeqPart::Item(term) => SeqPart::Item(term.into()),
            grammar::SeqPart::Splice { term, .. } => SeqPart::Splice(term.into()),
        }
    }
}

impl From<grammar::Nat> for Literal {
    fn from(value: grammar::Nat) -> Self {
        Self::Nat(value.0)
    }
}

#[test]
fn test() {}
