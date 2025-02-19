use crate::util::find_node_docs;
use hugr_model::v0::{Module, Operation};
use quote::{format_ident, quote};
use std::collections::HashMap;

pub fn generate(module: &Module) -> String {
    let root = module.get_region(module.root).unwrap();

    // We group the symbols by their extension in order to generate a Rust module per extension.
    let mut modules = HashMap::<&str, Vec<_>>::new();

    for node_id in root.children {
        let node = module.get_node(*node_id).unwrap();

        let symbol = match node.operation {
            Operation::DeclareConstructor(symbol) => symbol,
            Operation::DeclareOperation(symbol) => symbol,
            _ => continue,
        };

        let symbol_string = symbol.name;
        let (symbol_ext, symbol_name) = symbol.name.rsplit_once(".").unwrap();
        let symbol_ident = format_ident!("r#{}", symbol_name);

        // We use metadata in order to find human-readable documentation for the symbol.
        let docs = match find_node_docs(&module, *node_id) {
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
            Operation::DeclareConstructor(_) => quote! {
                impl<'a> ::hugr_model::v0::view::View<'a> for #symbol_ident {
                    type Id = ::hugr_model::v0::TermId;
                    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
                        let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;

                        if apply.symbol != #symbol_string {
                            return None;
                        }

                        let [#(#field_names),*] = apply.args.try_into().ok()?;
                        Some(Self { #(#field_names),* })
                    }
                }
            },
            Operation::DeclareOperation(_) => quote! {
                impl<'a> ::hugr_model::v0::view::View<'a> for #symbol_ident {
                    type Id = ::hugr_model::v0::NodeId;
                    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
                        let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;

                        if operation.name != #symbol_string {
                            return None;
                        }

                        let [#(#field_names),*] = operation.params.try_into().ok()?;
                        Some(Self { #(#field_names),* })
                    }
                }
            },
            _ => unreachable!(),
        };

        modules.entry(symbol_ext).or_default().push(quote! {
            #[doc = #docs]
            #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
            #[allow(non_camel_case_types)]
            pub struct #symbol_ident {
                #(#field_decls),*
            }

            #view_impl
        });
    }

    let mut out = Vec::new();

    for (symbol_ext, content) in modules {
        let module_name = format_ident!("r#{}", symbol_ext.replace(".", "_"));

        out.push(quote! {
            pub mod #module_name {
                #(#content)*
            }
        });
    }

    let out = quote! { #(#out)* };

    // The generated Rust code is pretty-printed.
    let ast = syn::parse2(out).unwrap();
    prettyplease::unparse(&ast)
}
