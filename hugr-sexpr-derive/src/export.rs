use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::{spanned::Spanned, DataStruct, DeriveInput};

use crate::common::{get_first_type_arg, parse_sexpr_attributes, FieldKind};

pub fn derive_export_impl(derive_input: DeriveInput) -> syn::Result<TokenStream> {
    match &derive_input.data {
        syn::Data::Struct(data_struct) => derive_export_struct(&derive_input, data_struct),
        syn::Data::Enum(_) => Err(syn::Error::new(
            derive_input.span(),
            "Can not derive Export for enums.",
        )),
        syn::Data::Union(_) => Err(syn::Error::new(
            derive_input.span(),
            "Can not derive Export for unions.",
        )),
    }
}

fn derive_export_struct(
    derive_input: &DeriveInput,
    data_struct: &DataStruct,
) -> syn::Result<TokenStream> {
    let struct_ident = &derive_input.ident;
    let struct_generics = &derive_input.generics.params;

    let mut code_fields = Vec::new();
    let mut code_where = Vec::new();

    for field in &data_struct.fields {
        let Some(field_ident) = &field.ident else {
            // TODO: Derive Export for tuple structs
            return Err(syn::Error::new_spanned(
                field,
                "Fields must be named to derive Export.",
            ));
        };

        let field_type = &field.ty;
        let field_data = parse_sexpr_attributes(&field.attrs)?;
        let field_name = field_data
            .rename
            .unwrap_or_else(|| format!("{}", field_ident.to_token_stream()));

        let field_symbol = quote! {
            ::hugr_sexpr::Value::Symbol(
                ::hugr_sexpr::Symbol::new(#field_name),
                ::std::default::Default::default()
            )
        };

        match field_data.kind {
            FieldKind::Positional => {
                code_fields.push(quote! {
                    <_ as ::hugr_sexpr::export::Export<A>>::export_into(&self.#field_ident, into);
                });

                code_where.push(quote! {
                    #field_type: ::hugr_sexpr::export::Export<A>,
                });
            }
            FieldKind::NamedRequired => {
                code_fields.push(quote! {
                    let field_value = &self.#field_ident;
                    let mut inner = vec![#field_symbol];
                    <_ as ::hugr_sexpr::export::Export<A>>::export_into(field_value, &mut inner);
                    into.push(::hugr_sexpr::Value::List(inner, ::std::default::Default::default()));
                });

                code_where.push(quote! {
                    #field_type: ::hugr_sexpr::export::Export<A>,
                });
            }
            FieldKind::NamedOptional => {
                code_fields.push(quote! {
                    if let Some(field_value) = &self.#field_ident {
                        let mut inner = vec![#field_symbol];
                        <_ as ::hugr_sexpr::export::Export<A>>::export_into(field_value, &mut inner);
                        into.push(::hugr_sexpr::Value::List(inner, ::std::default::Default::default()));
                    }
                });

                let inner_type = get_first_type_arg(field_type).ok_or(syn::Error::new_spanned(
                    field,
                    "Optional field must have type `Option<T>` for some `T`.",
                ))?;

                code_where.push(quote! {
                    #inner_type: ::hugr_sexpr::export::Export<A>,
                });
            }
            FieldKind::NamedRepeated => {
                code_fields.push(quote! {
                    into.extend(self.#field_ident.iter().map(|field_value| {
                        let mut inner = vec![#field_symbol];
                        <_ as ::hugr_sexpr::export::Export<A>>::export_into(field_value, &mut inner);
                        ::hugr_sexpr::Value::List(inner, ::std::default::Default::default())
                    }));
                });

                let inner_type = get_first_type_arg(field_type).ok_or(syn::Error::new_spanned(
                    field,
                    "Repeated field must have type `Vec<T>` for some `T`.",
                ))?;

                code_where.push(quote! {
                    #inner_type: ::hugr_sexpr::export::Export<A>,
                });
            }
        }
    }

    let code_fields: TokenStream = code_fields.into_iter().collect();
    let code_where: TokenStream = code_where.into_iter().collect();

    Ok(quote! {
        impl<A, #struct_generics> ::hugr_sexpr::export::Export<A> for #struct_ident<#struct_generics>
        where A: ::std::default::Default, #code_where {
            fn export_into(&self, into: &mut ::std::vec::Vec<::hugr_sexpr::Value<A>>) {
                #code_fields
            }
        }
    })
}
