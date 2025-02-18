use hugr_model::v0::{
    view::{NamedConstructor, View},
    Module, NodeId, Operation, TermId,
};
use quote::{format_ident, quote};

pub fn generate(module: &Module, extension: &str) {
    let root = module.get_region(module.root).unwrap();

    let mut out = Vec::new();

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

    let out = quote! { #(#out)* };
    let ast = syn::parse2(out).unwrap();
    let pretty = prettyplease::unparse(&ast);
    println!("{}", pretty);
}

struct MetaDoc<'a>(pub &'a str);

impl<'a> View<'a> for MetaDoc<'a> {
    type Id = TermId;

    fn view(module: &'a Module<'a>, id: Self::Id) -> Option<Self> {
        let apply: NamedConstructor = module.view(id)?;

        if apply.name != "core.meta.description" {
            return None;
        }

        let [doc] = apply.args.try_into().ok()?;
        Some(MetaDoc(module.view(doc)?))
    }
}

fn find_doc_meta<'a>(module: &'a Module<'a>, node_id: NodeId) -> Option<&'a str> {
    module
        .get_node(node_id)?
        .meta
        .iter()
        .find_map(|meta| module.view::<MetaDoc>(*meta))
        .map(|MetaDoc(doc)| doc)
}
