#![allow(missing_docs)]
use quote::{format_ident, quote};

use bumpalo::Bump;
use hugr_model::v0 as model;

mod core;

pub fn main() {
    let bump = Bump::new();
    let input = include_str!("../../extensions/core.edn");
    let extension = "core";
    let module = model::text::parse(input, &bump).unwrap().module;

    let root = module.get_region(module.root).unwrap();
    let mut out = Vec::new();
    // let mut out = String::new();

    for node_id in root.children {
        let node = module.get_node(*node_id).unwrap();

        let symbol = match node.operation {
            model::Operation::DeclareConstructor(symbol) => symbol,
            model::Operation::DeclareOperation(symbol) => symbol,
            _ => continue,
        };

        let symbol_string = symbol.name;
        let (symbol_ext, symbol_name) = symbol.name.rsplit_once(".").unwrap();
        let symbol_ident = format_ident!("r#{}", symbol_name);

        if symbol_ext != extension {
            continue;
        }

        let docs = match find_doc_meta(&module, *node_id) {
            Some(docs) => format!("`{}`: {}", symbol.name, docs),
            None => format!("`{}`.", symbol.name),
        };

        let mut field_decls = Vec::new();
        let mut field_names = Vec::new();

        for param in symbol.params {
            let param_ident = format_ident!("r#{}", param.name);
            field_decls.push(quote! {
                #[allow(missing_docs)]
                pub #param_ident: model::TermId
            });
            field_names.push(param_ident);
        }

        let view_impl = match node.operation {
            model::Operation::DeclareConstructor(_) => quote! {
                impl<'a> View<'a> for #symbol_ident {
                    type Id = model::TermId;
                    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
                        let [#(#field_names),*] = view_term_apply(module, id, #symbol_string)?;
                        Some(Self { #(#field_names),* })
                    }
                }
            },
            model::Operation::DeclareOperation(_) => quote! {
                impl<'a> View<'a> for #symbol_ident {
                    type Id = model::NodeId;
                    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
                        let [#(#field_names),*] = view_node_custom(module, id, #symbol_string)?;
                        Some(Self { #(#field_names),* })
                    }
                }
            },
            _ => unreachable!(),
        };

        out.push(quote! {
            #[doc = #docs]
            #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
            #[allow(non_camel_case_types)]
            pub struct #symbol_ident {
                #(#field_decls),*
            }

            #view_impl
        });
    }

    let out = quote! {
        use hugr_model::v0 as model;
        use super::{view_term_apply, View};
        #(#out)*
    };

    let ast = syn::parse2(out).unwrap();
    let pretty = prettyplease::unparse(&ast);
    println!("{}", pretty);
}

fn find_doc_meta<'a>(module: &'a model::Module<'a>, node_id: model::NodeId) -> Option<&'a str> {
    let node = module.get_node(node_id)?;

    for term_id in node.meta {
        let Some([doc]) = view_term_apply(module, *term_id, "core.meta.description") else {
            continue;
        };

        match module.get_term(doc)? {
            model::Term::Str(doc) => return Some(doc),
            _ => {}
        }
    }

    None
}

fn view_term_apply<const N: usize>(
    module: &model::Module,
    term_id: model::TermId,
    name: &str,
) -> Option<[model::TermId; N]> {
    let term = module.get_term(term_id)?;

    // TODO: Follow alias chains?

    let model::Term::Apply(symbol, args) = term else {
        return None;
    };

    let symbol_name = module.get_node(*symbol)?.operation.symbol()?;

    if name != symbol_name {
        return None;
    }

    (*args).try_into().ok()
}

fn view_node_custom<const N: usize>(
    module: &model::Module,
    node_id: model::NodeId,
    name: &str,
) -> Option<[model::TermId; N]> {
    let node = module.get_node(node_id)?;

    // TODO: Follow alias chains?

    let model::Operation::Custom(symbol) = &node.operation else {
        return None;
    };

    let symbol_name = module.get_node(*symbol)?.operation.symbol()?;

    if name != symbol_name {
        return None;
    }

    (*node.params).try_into().ok()
}

trait View<'a>: Sized {
    type Id;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self>;
}

struct r#fn {
    pub inputs: model::TermId,
    pub outputs: model::TermId,
}

impl<'a> View<'a> for r#fn {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [inputs, outputs] = view_term_apply(module, id, "core.fn")?;
        Some(Self { inputs, outputs })
    }
}
