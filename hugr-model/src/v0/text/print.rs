use pretty::{Arena, DocAllocator, RefDoc};
use std::borrow::Cow;

use crate::v0::{
    ExtSetPart, LinkIndex, ListPart, MetaItem, ModelError, Module, NodeId, Operation, Param,
    ParamSort, RegionId, RegionKind, Term, TermId, VarId,
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
            .ok_or(PrintError::RegionNotFound(root_id))?;

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
        self.locals.extend(params.iter().map(|param| param.name));
        let result = f(self);
        self.locals = locals;
        result
    }

    fn print_node(&mut self, node_id: NodeId) -> PrintResult<()> {
        let node_data = self
            .module
            .get_node(node_id)
            .ok_or(PrintError::NodeNotFound(node_id))?;

        self.print_parens(|this| match &node_data.operation {
            Operation::Invalid => Err(ModelError::InvalidOperation(node_id)),
            Operation::Dfg { body } => {
                this.print_group(|this| {
                    this.print_text("dfg");
                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_region(*body)
            }
            Operation::Cfg { body } => {
                this.print_group(|this| {
                    this.print_text("cfg");
                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_region(*body)
            }
            Operation::Block { body } => {
                this.print_group(|this| {
                    this.print_text("block");
                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_region(*body)
            }

            Operation::DefineFunc { decl, body } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("define-func");
                    this.print_text(decl.name);
                });

                this.print_params(decl.params)?;
                this.print_constraints(decl.constraints)?;

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
                this.print_region(*body)
            }),

            Operation::DeclareFunc { decl } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("declare-func");
                    this.print_text(decl.name);
                });

                this.print_params(decl.params)?;
                this.print_constraints(decl.constraints)?;

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
                        this.print_symbol(*operation)?;
                    } else {
                        this.print_parens(|this| {
                            this.print_symbol(*operation)?;

                            for param in node_data.params {
                                this.print_term(*param)?;
                            }

                            Ok(())
                        })?;
                    }

                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)
            }

            Operation::CustomFull { operation } => {
                this.print_group(|this| {
                    this.print_parens(|this| {
                        this.print_text("@");
                        this.print_symbol(*operation)?;

                        for param in node_data.params {
                            this.print_term(*param)?;
                        }

                        Ok(())
                    })?;

                    this.print_port_lists(node_data.inputs, node_data.outputs)
                })?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)
            }

            Operation::DefineAlias { decl, value } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("define-alias");
                    this.print_text(decl.name);
                });

                this.print_params(decl.params)?;

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

                this.print_params(decl.params)?;

                this.print_term(decl.r#type)?;
                this.print_meta(node_data.meta)?;
                Ok(())
            }),

            Operation::DeclareConstructor { decl } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("declare-ctr");
                    this.print_text(decl.name);
                });

                this.print_params(decl.params)?;
                this.print_constraints(decl.constraints)?;

                this.print_term(decl.r#type)?;
                this.print_meta(node_data.meta)?;
                Ok(())
            }),

            Operation::DeclareOperation { decl } => this.with_local_scope(decl.params, |this| {
                this.print_group(|this| {
                    this.print_text("declare-operation");
                    this.print_text(decl.name);
                });

                this.print_params(decl.params)?;
                this.print_constraints(decl.constraints)?;

                this.print_term(decl.r#type)?;
                this.print_meta(node_data.meta)?;
                Ok(())
            }),

            Operation::TailLoop { body } => {
                this.print_text("tail-loop");
                this.print_port_lists(node_data.inputs, node_data.outputs)?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_region(*body)
            }

            Operation::Conditional { branches } => {
                this.print_text("cond");
                this.print_port_lists(node_data.inputs, node_data.outputs)?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)?;
                this.print_regions(branches)
            }

            Operation::Tag { tag } => {
                this.print_text("tag");
                this.print_text(format!("{}", tag));
                this.print_port_lists(node_data.inputs, node_data.outputs)?;
                this.print_signature(node_data.signature)?;
                this.print_meta(node_data.meta)
            }

            Operation::Import { name } => {
                this.print_text("import");
                this.print_text(*name);
                this.print_meta(node_data.meta)
            }

            Operation::Const { value } => {
                this.print_text("const");
                this.print_term(*value)?;
                this.print_meta(node_data.meta)
            }
        })
    }

    fn print_regions(&mut self, regions: &'a [RegionId]) -> PrintResult<()> {
        regions
            .iter()
            .try_for_each(|region| self.print_region(*region))
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

        region_data
            .children
            .iter()
            .try_for_each(|node_id| self.print_node(*node_id))
    }

    fn print_port_lists(
        &mut self,
        first: &'a [LinkIndex],
        second: &'a [LinkIndex],
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

    fn print_port_list(&mut self, links: &'a [LinkIndex]) -> PrintResult<()> {
        self.print_brackets(|this| {
            for link in links {
                this.print_link_index(*link);
            }
            Ok(())
        })
    }

    fn print_link_index(&mut self, link_index: LinkIndex) {
        self.print_text(format!("%{}", link_index.0));
    }

    fn print_params(&mut self, params: &'a [Param<'a>]) -> PrintResult<()> {
        params.iter().try_for_each(|param| self.print_param(*param))
    }

    fn print_param(&mut self, param: Param<'a>) -> PrintResult<()> {
        self.print_parens(|this| {
            match param.sort {
                ParamSort::Implicit => this.print_text("forall"),
                ParamSort::Explicit => this.print_text("param"),
            };

            this.print_text(format!("?{}", param.name));
            this.print_term(param.r#type)
        })
    }

    fn print_constraints(&mut self, terms: &'a [TermId]) -> PrintResult<()> {
        for term in terms {
            self.print_parens(|this| {
                this.print_text("where");
                this.print_term(*term)
            })?;
        }

        Ok(())
    }

    fn print_term(&mut self, term_id: TermId) -> PrintResult<()> {
        let term_data = self
            .module
            .get_term(term_id)
            .ok_or(PrintError::TermNotFound(term_id))?;

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
            Term::Var(var) => self.print_var(*var),
            Term::Apply { symbol, args } => {
                if args.is_empty() {
                    self.print_symbol(*symbol)?;
                } else {
                    self.print_parens(|this| {
                        this.print_symbol(*symbol)?;
                        for arg in args.iter() {
                            this.print_term(*arg)?;
                        }
                        Ok(())
                    })?;
                }

                Ok(())
            }
            Term::ApplyFull { symbol, args } => self.print_parens(|this| {
                this.print_text("@");
                this.print_symbol(*symbol)?;
                for arg in args.iter() {
                    this.print_term(*arg)?;
                }

                Ok(())
            }),
            Term::Const { r#type } => self.print_parens(|this| {
                this.print_text("const");
                this.print_term(*r#type)
            }),
            Term::List { .. } => self.print_brackets(|this| this.print_list_parts(term_id)),
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
            Term::ExtSet { .. } => self.print_parens(|this| {
                this.print_text("ext");
                this.print_ext_set_parts(term_id)?;
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
            Term::NonLinearConstraint { term } => self.print_parens(|this| {
                this.print_text("nonlinear");
                this.print_term(*term)
            }),
            Term::ConstFunc { region } => self.print_parens(|this| {
                this.print_text("fn");
                this.print_region(*region)
            }),
        }
    }

    /// Prints the contents of a list.
    ///
    /// This is used so that spliced lists are merged into the parent list.
    fn print_list_parts(&mut self, term_id: TermId) -> PrintResult<()> {
        let term_data = self
            .module
            .get_term(term_id)
            .ok_or(PrintError::TermNotFound(term_id))?;

        if let Term::List { parts } = term_data {
            for part in *parts {
                match part {
                    ListPart::Item(term) => self.print_term(*term)?,
                    ListPart::Splice(list) => self.print_list_parts(*list)?,
                }
            }
        } else {
            self.print_term(term_id)?;
            self.print_text("...");
        }

        Ok(())
    }

    /// Prints the contents of an extension set.
    ///
    /// This is used so that spliced extension sets are merged into the parent extension set.
    fn print_ext_set_parts(&mut self, term_id: TermId) -> PrintResult<()> {
        let term_data = self
            .module
            .get_term(term_id)
            .ok_or(PrintError::TermNotFound(term_id))?;

        if let Term::ExtSet { parts } = term_data {
            for part in *parts {
                match part {
                    ExtSetPart::Extension(ext) => self.print_text(*ext),
                    ExtSetPart::Splice(list) => self.print_ext_set_parts(*list)?,
                }
            }
        } else {
            self.print_term(term_id)?;
            self.print_text("...");
        }

        Ok(())
    }

    fn print_var(&mut self, var: VarId) -> PrintResult<()> {
        let Some(name) = self.locals.get(var.1 as usize) else {
            return Err(PrintError::InvalidVar(var));
        };

        self.print_text(format!("?{}", name));
        Ok(())
    }

    fn print_symbol(&mut self, node_id: NodeId) -> PrintResult<()> {
        let node_data = self
            .module
            .get_node(node_id)
            .ok_or(PrintError::NodeNotFound(node_id))?;

        let name = node_data
            .operation
            .symbol()
            .ok_or(PrintError::UnexpectedOperation(node_id))?;

        self.print_text(name);
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
