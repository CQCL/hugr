//! Hugr graphs.

use super::ident::{Symbol, VarName};
use super::keywords as kw;
use super::terms::Term;
use parens::parser::{self, Parse, Parser, Peek, Span};
use parens::printer::{Print, Printer};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

/// A hugr graph describing a computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hugr {
    /// The name of this hugr within the module.
    pub name: SmolStr,
    /// The root nodes of this hugr graph.
    pub roots: Vec<Node>,
    /// Source location.
    #[serde(skip)]
    pub span: Span,
}

impl Parse for Hugr {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        parser.parse::<kw::hugr>()?;
        let name = parser.parse()?;
        let mut roots = Vec::new();

        while !parser.is_empty() {
            if parser.peek::<Node>() {
                roots.push(parser.list(Node::parse)?);
            } else {
                return Err(parser.error("expected root node"));
            }
        }

        let span = parser.parent_span();
        Ok(Hugr { name, roots, span })
    }
}

impl Print for Hugr {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.group(|printer| {
            printer.print(kw::hugr)?;
            printer.print(&self.name)
        })?;

        for root in &self.roots {
            printer.list(|printer| printer.print(root))?;
        }

        Ok(())
    }
}
/// A node in a HUGR graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// The operation that the node runs.
    pub operation: Operation,
    /// The inputs to the operation.
    #[serde(default, skip_serializing_if = "VarList::is_empty")]
    pub inputs: VarList,
    /// The outputs of the operation.
    #[serde(default, skip_serializing_if = "VarList::is_empty")]
    pub outputs: VarList,
    /// The child nodes that are nested within this node.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Node>,
    /// Source location.
    #[serde(skip)]
    pub span: Span,
}

impl Parse for Node {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        let operation = parser.parse()?;
        let inputs = parser.parse()?;
        let outputs = parser.parse()?;
        let mut children = Vec::new();

        while !parser.is_empty() {
            if parser.peek::<Node>() {
                children.push(parser.list(Node::parse)?);
            } else {
                return Err(parser.error("expected child nodes"));
            }
        }

        let span = parser.parent_span();
        Ok(Node {
            operation,
            inputs,
            outputs,
            children,
            span,
        })
    }
}

impl Print for Node {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.group(|printer| {
            printer.print(&self.operation)?;
            printer.print(&self.inputs)?;
            printer.print(&self.outputs)
        })?;

        for child in &self.children {
            printer.list(|printer| printer.print(child))?;
        }

        Ok(())
    }
}

impl Peek for Node {
    fn peek(cursor: parser::Cursor<'_>) -> bool {
        cursor.peek_list(|cursor| cursor.peek::<Operation>())
    }
}

/// An operation together with an optional list of custom arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    /// The name of the operation.
    pub name: Symbol,
    /// The list of custom arguments.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<Term>,
}

impl Parse for Operation {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        if parser.peek::<Symbol>() {
            let name = parser.parse()?;
            let args = vec![];
            Ok(Operation { name, args })
        } else {
            parser.list(|parser| {
                let name = parser.parse()?;
                let args = parser.parse()?;
                Ok(Operation { name, args })
            })
        }
    }
}

impl Peek for Operation {
    fn peek(cursor: parser::Cursor<'_>) -> bool {
        cursor.peek::<Symbol>() || cursor.peek_list(|cursor| cursor.peek::<Symbol>())
    }
}

impl Print for Operation {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        if self.args.is_empty() {
            printer.print(&self.name)
        } else {
            printer.list(|printer| {
                printer.print(&self.name)?;
                printer.print(&self.args)
            })
        }
    }
}

/// A list of variables used for node inputs and outputs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VarList(pub Vec<VarName>);

impl VarList {
    fn is_empty(&self) -> bool {
        // Used for serde's `skip_serializing_if`
        self.0.is_empty()
    }
}

impl Parse for VarList {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        Ok(Self(parser.seq(|parser| parser.parse())?))
    }
}

impl Print for VarList {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.seq(|printer| printer.print(&self.0))
    }
}
