use pretty::{Arena, DocAllocator, RefDoc};
use std::borrow::Cow;

use crate::v0::{
    GlobalRef, LinkRef, LocalRef, MetaItem, ModelError, Module, NodeId, Operation, Param, RegionId,
    RegionKind, Term, TermId,
};

type PrintError = ModelError;
type PrintResult<T> = Result<T, PrintError>;

/// Pretty-print a module to a string.
pub fn print_to_string(module: &Module, width: usize) -> PrintResult<String> {
    let arena = Arena::new();
    let doc = PrintContext::create_doc(&arena, module)?;
    let mut out = String::new();
    let _ = doc.render_fmt(width, &mut out);
    Ok(out)
}

struct PrintContext<'p, 'a: 'p> {
    /// The arena in which to allocate the pretty-printed documents.
    arena: &'p Arena<'p>,
    /// The module to be printed.
    module: &'a Module<'a>,
    /// Parts of the document to be concatenated.
    docs: Vec<RefDoc<'p>>,
    /// Stack of indices into `docs` denoting the current nesting.
    docs_stack: Vec<usize>,
    /// The names of local variables that are in scope.
    locals: Vec<&'a str>,
}

impl<'p, 'a: 'p> PrintContext<'p, 'a> {
    fn create_doc(arena: &'p Arena<'p>, module: &'a Module) -> PrintResult<RefDoc<'p>> {
        let mut this = Self {
            arena,
            module,
            docs: Vec::new(),
            docs_stack: Vec::new(),
            locals: Vec::new(),
        };

        this.print_parens(|this| {
            this.print_text("hugr");
            this.print_text("0");
        });

        this.print_root()?;
        Ok(this.finish())
    }

    fn finish(self) -> RefDoc<'p> {
        let sep = self
            .arena
            .concat([self.arena.hardline(), self.arena.hardline()]);
        self.arena.intersperse(self.docs, sep).into_doc()
    }

    fn print_text(&mut self, text: impl Into<Cow<'p, str>>) {
        self.docs.push(self.arena.text(text).into_doc());
    }

    /// Print a delimited sequence of elements.
    ///
    /// See [`print_group`], [`print_parens`], and [`print_brackets`].
    fn print_delimited<T>(
        &mut self,
        start: &'static str,
        end: &'static str,
        nesting: isize,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.docs_stack.push(self.docs.len());
        let result = f(self);
        let docs = self.docs.drain(self.docs_stack.pop().unwrap()..);
        let doc = self.arena.concat([
            self.arena.text(start),
            self.arena
                .intersperse(docs, self.arena.line())
                .nest(nesting)
                .group(),
            self.arena.text(end),
        ]);
        self.docs.push(doc.into_doc());
        result
    }

    /// Print a sequence of elements that are preferrably laid out together in the same line.
    #[inline]
    fn print_group<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.print_delimited("", "", 0, f)
    }

    /// Print a sequence of elements in a parenthesized list.
    #[inline]
    fn print_parens<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.print_delimited("(", ")", 2, f)
    }

    /// Print a sequence of elements in a bracketed list.
    #[inline]
    fn print_brackets<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.print_delimited("[", "]", 1, f)
    }

    fn print_root(&mut self) -> PrintResult<()> {
        let root_id = self.module.root;
        let root_data = self
            .module
            .get_region(root_id)
            .ok_or_else(|| PrintError::RegionNotFound(root_id))?;

        self.print_meta(root_data.meta)?;
        self.print_nodes(root_id)?;
        Ok(())
    }

    fn with_local_scope<T>(
        &mut self,
        params: &'a [Param<'a>],
        f: impl FnOnce(&mut Self) -> PrintResult<T>,
    ) -> PrintResult<T> {
        let locals = std::mem::take(&mut self.locals);

        for param in params {
            match param {
                Param::Implicit { name, .. } => self.locals.push(name),
                Param::Explicit { name, .. } => self.locals.push(name),
                Param::Constraint { .. } => {}
            }
        }

        let result = f(self);
        self.locals = locals;
        result
    }

    fn print_node(&mut self, node_id: NodeId) -> PrintResult<()> {
        let node_data = self
            .module
            .get_node(node_id)
            .ok_or_else(|| PrintError::NodeNotFound(node_id))?;

        self.print_parens(|this| match &node_data.operation {
            Operation::Invalid => Err(ModelError::InvalidOperation(node_id)),
            Operation::Dfg => {
                this.print_group(|this| {
                    this.print_text("dfg");
                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_regions(node_data.regions)
            }
            Operation::Cfg => {
                this.print_group(|this| {
                    this.print_text("cfg");
                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_regions(node_data.regions)
            }
            Operation::Block => {
                this.print_group(|this| {
                    this.print_text("block");
                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_regions(node_data.regions)
            }

            Operation::DefineFunc { decl } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("define-func");
                    this.print_text(decl.name);
                });

                for param in decl.params {
                    this.print_param(*param)?;
                }

                match self.module.get_term(decl.signature) {
                    Some(Term::FuncType {
                        inputs,
                        outputs,
                        extensions,
                    }) => {
                        this.print_group(|this| {
                            this.print_term(*inputs)?;
                            this.print_term(*outputs)?;
                            this.print_term(*extensions)
                        })?;
                    }
                    Some(_) => return Err(PrintError::TypeError(decl.signature)),
                    None => return Err(PrintError::TermNotFound(decl.signature)),
                }

                this.print_meta(node_data.meta)?;
                this.print_regions(node_data.regions)
            }),

            Operation::DeclareFunc { decl } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("declare-func");
                    this.print_text(decl.name);
                });

                for param in decl.params {
                    this.print_param(*param)?;
                }

                match self.module.get_term(decl.signature) {
                    Some(Term::FuncType {
                        inputs,
                        outputs,
                        extensions,
                    }) => {
                        this.print_group(|this| {
                            this.print_term(*inputs)?;
                            this.print_term(*outputs)?;
                            this.print_term(*extensions)
                        })?;
                    }
                    Some(_) => return Err(PrintError::TypeError(decl.signature)),
                    None => return Err(PrintError::TermNotFound(decl.signature)),
                }

                this.print_meta(node_data.meta)?;
                Ok(())
            }),

            Operation::CallFunc { func } => {
                this.print_group(|this| {
                    this.print_text("call");
                    this.print_term(*func)?;
                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                Ok(())
            }

            Operation::LoadFunc { func } => {
                this.print_group(|this| {
                    this.print_text("load-func");
                    this.print_term(*func)?;
                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                Ok(())
            }

            Operation::Custom { operation } => {
                this.print_group(|this| {
                    if node_data.params.is_empty() {
                        this.print_global_ref(*operation)?;
                    } else {
                        this.print_parens(|this| {
                            this.print_global_ref(*operation)?;

                            for param in node_data.params {
                                this.print_term(*param)?;
                            }

                            Ok(())
                        })?;
                    }

                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_regions(node_data.regions)
            }

            Operation::CustomFull { operation } => {
                this.print_group(|this| {
                    this.print_parens(|this| {
                        this.print_text("@");
                        this.print_global_ref(*operation)?;

                        for param in node_data.params {
                            this.print_term(*param)?;
                        }

                        Ok(())
                    })?;

                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_regions(node_data.regions)
            }

            Operation::DefineAlias { decl, value } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("define-alias");
                    this.print_text(decl.name);
                });

                for param in decl.params {
                    this.print_param(*param)?;
                }

                this.print_term(decl.r#type)?;
                this.print_term(*value)?;
                this.print_meta(node_data.meta)?;
                Ok(())
            }),
            Operation::DeclareAlias { decl } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("declare-alias");
                    this.print_text(decl.name);
                });

                for param in decl.params {
                    this.print_param(*param)?;
                }

                this.print_term(decl.r#type)?;
                this.print_meta(node_data.meta)?;
                Ok(())
            }),

            Operation::DeclareConstructor { decl } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("declare-ctr");
                    this.print_text(decl.name);
                });

                for param in decl.params {
                    this.print_param(*param)?;
                }

                this.print_term(decl.r#type)?;
                this.print_meta(node_data.meta)?;
                Ok(())
            }),

            Operation::DeclareOperation { decl } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("declare-operation");
                    this.print_text(decl.name);
                });

                for param in decl.params {
                    this.print_param(*param)?;
                }

                this.print_term(decl.r#type)?;
                this.print_meta(node_data.meta)?;
                Ok(())
            }),

            Operation::TailLoop => {
                this.print_text("tail-loop");
                this.print_port_lists(node_data.inputs, node_data.outputs)?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_regions(node_data.regions)
            }

            Operation::Conditional => {
                this.print_text("cond");
                this.print_port_lists(node_data.inputs, node_data.outputs)?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_regions(node_data.regions)
            }

            Operation::Tag { tag } => {
                this.print_text("tag");
                this.print_text(format!("{}", tag));
                this.print_port_lists(node_data.inputs, node_data.outputs)?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)
            }
        })
    }

    fn print_regions(&mut self, regions: &'a [RegionId]) -> PrintResult<()> {
        for region in regions {
            self.print_region(*region)?;
        }
        Ok(())
    }

    fn print_region(&mut self, region: RegionId) -> PrintResult<()> {
        let region_data = self
            .module
            .get_region(region)
            .ok_or(PrintError::RegionNotFound(region))?;

        self.print_parens(|this| {
            match region_data.kind {
                RegionKind::DataFlow => {
                    this.print_text("dfg");
                }
                RegionKind::ControlFlow => {
                    this.print_text("cfg");
                }
                RegionKind::Module => {
                    this.print_text("module");
                }
            };

            this.print_port_lists(region_data.sources, region_data.targets)?;
            this.print_signature(region_data.signature)?;
            this.print_meta(region_data.meta)?;
            this.print_nodes(region)
        })
    }

    fn print_nodes(&mut self, region: RegionId) -> PrintResult<()> {
        let region_data = self
            .module
            .get_region(region)
            .ok_or(PrintError::RegionNotFound(region))?;

        for node_id in region_data.children {
            self.print_node(*node_id)?;
        }

        Ok(())
    }

    fn print_port_lists(
        &mut self,
        first: &'a [LinkRef<'a>],
        second: &'a [LinkRef<'a>],
    ) -> PrintResult<()> {
        if !first.is_empty() && !second.is_empty() {
            self.print_group(|this| {
                this.print_port_list(first)?;
                this.print_port_list(second)
            })
        } else {
            Ok(())
        }
    }

    fn print_port_list(&mut self, links: &'a [LinkRef<'a>]) -> PrintResult<()> {
        self.print_brackets(|this| {
            for link in links {
                this.print_link_ref(*link);
            }
            Ok(())
        })
    }

    fn print_link_ref(&mut self, link_ref: LinkRef<'a>) {
        match link_ref {
            LinkRef::Id(link_id) => self.print_text(format!("%{}", link_id.0)),
            LinkRef::Named(name) => self.print_text(format!("%{}", name)),
        }
    }

    fn print_param(&mut self, param: Param<'a>) -> PrintResult<()> {
        self.print_parens(|this| match param {
            Param::Implicit { name, r#type } => {
                this.print_text("forall");
                this.print_text(format!("?{}", name));
                this.print_term(r#type)
            }
            Param::Explicit { name, r#type } => {
                this.print_text("param");
                this.print_text(format!("?{}", name));
                this.print_term(r#type)
            }
            Param::Constraint { constraint } => {
                this.print_text("where");
                this.print_term(constraint)
            }
        })
    }

    fn print_term(&mut self, term_id: TermId) -> PrintResult<()> {
        let term_data = self
            .module
            .get_term(term_id)
            .ok_or_else(|| PrintError::TermNotFound(term_id))?;

        match term_data {
            Term::Wildcard => {
                self.print_text("_");
                Ok(())
            }
            Term::Type => {
                self.print_text("type");
                Ok(())
            }
            Term::StaticType => {
                self.print_text("static");
                Ok(())
            }
            Term::Constraint => {
                self.print_text("constraint");
                Ok(())
            }
            Term::Var(local_ref) => self.print_local_ref(*local_ref),
            Term::Apply { global: name, args } => {
                if args.is_empty() {
                    self.print_global_ref(*name)?;
                } else {
                    self.print_parens(|this| {
                        this.print_global_ref(*name)?;
                        for arg in args.iter() {
                            this.print_term(*arg)?;
                        }
                        Ok(())
                    })?;
                }

                Ok(())
            }
            Term::ApplyFull { global: name, args } => self.print_parens(|this| {
                this.print_text("@");
                this.print_global_ref(*name)?;
                for arg in args.iter() {
                    this.print_term(*arg)?;
                }

                Ok(())
            }),
            Term::Quote { r#type } => self.print_parens(|this| {
                this.print_text("quote");
                this.print_term(*r#type)
            }),
            Term::List { items, tail } => self.print_brackets(|this| {
                for item in items.iter() {
                    this.print_term(*item)?;
                }
                if let Some(tail) = tail {
                    this.print_text(".");
                    this.print_term(*tail)?;
                }
                Ok(())
            }),
            Term::ListType { item_type } => self.print_parens(|this| {
                this.print_text("list");
                this.print_term(*item_type)
            }),
            Term::Str(str) => {
                self.print_string(str);
                Ok(())
            }
            Term::StrType => {
                self.print_text("str");
                Ok(())
            }
            Term::Nat(n) => {
                self.print_text(n.to_string());
                Ok(())
            }
            Term::NatType => {
                self.print_text("nat");
                Ok(())
            }
            Term::ExtSet { extensions, rest } => self.print_parens(|this| {
                this.print_text("ext");
                for extension in *extensions {
                    this.print_text(*extension);
                }
                if let Some(rest) = rest {
                    this.print_text(".");
                    this.print_term(*rest)?;
                }
                Ok(())
            }),
            Term::ExtSetType => {
                self.print_text("ext-set");
                Ok(())
            }
            Term::Adt { variants } => self.print_parens(|this| {
                this.print_text("adt");
                this.print_term(*variants)
            }),
            Term::FuncType {
                inputs,
                outputs,
                extensions,
            } => self.print_parens(|this| {
                this.print_text("fn");
                this.print_term(*inputs)?;
                this.print_term(*outputs)?;
                this.print_term(*extensions)
            }),
            Term::Control { values } => self.print_parens(|this| {
                this.print_text("ctrl");
                this.print_term(*values)
            }),
            Term::ControlType => {
                self.print_text("ctrl");
                Ok(())
            }
        }
    }

    fn print_local_ref(&mut self, local_ref: LocalRef<'a>) -> PrintResult<()> {
        let name = match local_ref {
            LocalRef::Index(_, i) => {
                let Some(name) = self.locals.get(i as usize) else {
                    return Err(PrintError::InvalidLocal(local_ref.to_string()));
                };

                name
            }
            LocalRef::Named(name) => name,
        };

        self.print_text(format!("?{}", name));
        Ok(())
    }

    fn print_global_ref(&mut self, global_ref: GlobalRef<'a>) -> PrintResult<()> {
        match global_ref {
            GlobalRef::Direct(node_id) => {
                let node_data = self
                    .module
                    .get_node(node_id)
                    .ok_or_else(|| PrintError::NodeNotFound(node_id))?;

                let name = match &node_data.operation {
                    Operation::DefineFunc { decl } => decl.name,
                    Operation::DeclareFunc { decl } => decl.name,
                    Operation::DefineAlias { decl, .. } => decl.name,
                    Operation::DeclareAlias { decl } => decl.name,
                    _ => return Err(PrintError::UnexpectedOperation(node_id)),
                };

                self.print_text(name)
            }

            GlobalRef::Named(symbol) => self.print_text(symbol),
        }

        Ok(())
    }

    fn print_meta(&mut self, meta: &'a [MetaItem<'a>]) -> PrintResult<()> {
        for item in meta {
            self.print_parens(|this| {
                this.print_group(|this| {
                    this.print_text("meta");
                    this.print_text(item.name);
                });
                this.print_term(item.value)
            })?;
        }

        Ok(())
    }

    fn print_signature(&mut self, term: Option<TermId>) -> PrintResult<()> {
        if let Some(term) = term {
            self.print_parens(|this| {
                this.print_text("signature");
                this.print_term(term)
            })?;
        }

        Ok(())
    }

    /// Print a string literal.
    fn print_string(&mut self, string: &str) {
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
        self.print_text(output);
    }
}
