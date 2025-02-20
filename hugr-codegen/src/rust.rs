use crate::util::find_node_docs;
use hugr_model::v0::syntax;
use hugr_model::v0::{Module, Operation};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Generator {
    // We group the symbols by their extension in order to generate a Rust
    // module per extension.
    code_by_extension: HashMap<String, Vec<TokenStream>>,
}

impl Generator {
    pub fn new() -> Self {
        Self {
            code_by_extension: HashMap::new(),
        }
    }

    pub fn add_module(&mut self, module: &Module) {
        let root = module.get_region(module.root).unwrap();

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

            let mut field_decls = Vec::new();
            let mut field_names = Vec::new();
            let mut field_docs = Vec::new();

            for param in symbol.params {
                let param_ident = format_ident!("r#{}", param.name);
                field_decls.push(quote! {
                    #[allow(missing_docs)]
                    pub #param_ident: hugr_model::v0::TermId
                });
                field_names.push(param_ident);

                let param_type = module.view::<syntax::Term>(param.r#type).unwrap();
                field_docs.push(format!("`{} : {}`", param.name, param_type));
            }

            let sig = module.view::<syntax::Term>(symbol.signature).unwrap();

            // We use metadata in order to find human-readable documentation for the symbol.
            let doc_head = match find_node_docs(&module, *node_id) {
                Some(docs) => format!("{}", docs),
                None => format!("`{}`.", symbol.name),
            };

            let doc = format!("{}\n\n__Type__:\n```text\n{}\n```\n", doc_head, sig);

            let view_impl = match node.operation {
                Operation::DeclareConstructor(_) => quote! {
                    impl<'a> ::hugr_model::v0::view::View<'a> for #symbol_ident {
                        type Id = ::hugr_model::v0::TermId;
                        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
                            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;

                            if apply.name != #symbol_string {
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
                        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
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

            self.code_by_extension
                .entry(symbol_ext.into())
                .or_default()
                .push(quote! {
                    #[doc = #doc]
                    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
                    #[allow(non_camel_case_types)]
                    pub struct #symbol_ident {
                        #(
                            #[doc = #field_docs]
                            #field_decls
                        ),*
                    }

                    #view_impl
                });
        }
    }

    pub fn as_tokens(&self) -> TokenStream {
        let mut out = Vec::new();

        for (symbol_ext, content) in &self.code_by_extension {
            let module_name = format_ident!("r#{}", symbol_ext.replace(".", "_"));

            out.push(quote! {
                pub mod #module_name {
                    #(#content)*
                }
            });
        }

        quote! { #(#out)* }
    }

    pub fn as_str(&self) -> String {
        let tokens = self.as_tokens();
        let ast = syn::parse2(tokens).unwrap();
        prettyplease::unparse(&ast)
    }
}

impl Default for Generator {
    fn default() -> Self {
        Self::new()
    }
}
