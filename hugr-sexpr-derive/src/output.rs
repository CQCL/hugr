use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::{spanned::Spanned, DataStruct, DeriveInput};

use crate::common::{get_first_type_arg, parse_sexpr_attributes, FieldKind};

pub fn derive_output_impl(derive_input: DeriveInput) -> syn::Result<TokenStream> {
    match &derive_input.data {
        syn::Data::Struct(data_struct) => derive_output_struct(&derive_input, data_struct),
        syn::Data::Enum(_) => Err(syn::Error::new(
            derive_input.span(),
            "Can not derive Output for enums.",
        )),
        syn::Data::Union(_) => Err(syn::Error::new(
            derive_input.span(),
            "Can not derive Output for unions.",
        )),
    }
}

fn derive_output_struct(
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
                "Fields must be named to derive Output.",
            ));
        };

        let field_type = &field.ty;
        let field_data = parse_sexpr_attributes(&field.attrs)?;
        let field_name = field_data
            .rename
            .unwrap_or_else(|| format!("{}", field_ident.to_token_stream()));

        match field_data.kind {
            FieldKind::Positional => {
                code_fields.push(quote! {
                    <_ as ::hugr_sexpr::output::Output<O>>::print(&self.#field_ident, output)?;
                });

                code_where.push(quote! {
                    #field_type: ::hugr_sexpr::output::Output<O>,
                });
            }
            FieldKind::NamedRequired => {
                code_fields.push(quote! {
                    output.list(|output| {
                        output.symbol(#field_name)?;
                        <_ as ::hugr_sexpr::output::Output<O>>::print(&self.#field_ident, output)
                    })?;
                });

                code_where.push(quote! {
                    #field_type: ::hugr_sexpr::output::Output<O>,
                });
            }
            FieldKind::NamedOptional => {
                code_fields.push(quote! {
                    if let Some(field_value) = &self.#field_ident {
                        output.list(|output| {
                            output.symbol(#field_name)?;
                            <_ as ::hugr_sexpr::output::Output<O>>::print(field_value, output)
                        })?;
                    }
                });

                let inner_type = get_first_type_arg(field_type).ok_or(syn::Error::new_spanned(
                    field,
                    "Optional field must have type `Option<T>` for some `T`.",
                ))?;

                code_where.push(quote! {
                    #inner_type: ::hugr_sexpr::output::Output<O>,
                });
            }
            FieldKind::NamedRepeated => {
                code_fields.push(quote! {
                    for field_value in self.#field_ident.iter() {
                        output.list(|output| {
                            output.symbol(#field_name)?;
                            <_ as ::hugr_sexpr::output::Output<O>>::print(field_value, output)
                        })?;
                    }
                });

                let inner_type = get_first_type_arg(field_type).ok_or(syn::Error::new_spanned(
                    field,
                    "Repeated field must have type `Vec<T>` for some `T`.",
                ))?;

                code_where.push(quote! {
                    #inner_type: ::hugr_sexpr::output::Output<O>,
                });
            }
        }
    }

    let code_fields: TokenStream = code_fields.into_iter().collect();
    let code_where: TokenStream = code_where.into_iter().collect();

    Ok(quote! {
        impl<O, #struct_generics> ::hugr_sexpr::output::Output<O> for #struct_ident<#struct_generics>
        where O: ::hugr_sexpr::output::OutputStream, #code_where {
            fn print(&self, output: &mut O) -> std::result::Result<(), O::Error> {
                #code_fields
                Ok(())
            }
        }
    })
}
