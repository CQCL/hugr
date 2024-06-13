use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::{spanned::Spanned, DataStruct, DeriveInput};

use crate::common::{get_first_type_arg, parse_sexpr_attributes, FieldKind};

pub fn derive_import_impl(derive_input: DeriveInput) -> syn::Result<TokenStream> {
    match &derive_input.data {
        syn::Data::Struct(data_struct) => derive_import_struct(&derive_input, data_struct),
        syn::Data::Enum(_) => Err(syn::Error::new(
            derive_input.span(),
            "Can not derive Import for enums.",
        )),
        syn::Data::Union(_) => Err(syn::Error::new(
            derive_input.span(),
            "Can not derive Import for unions.",
        )),
    }
}

fn derive_import_struct(
    derive_input: &DeriveInput,
    data_struct: &DataStruct,
) -> syn::Result<TokenStream> {
    let struct_ident = &derive_input.ident;
    let struct_generics = &derive_input.generics.params;

    // The code used to parse positional fields
    let mut code_positional = Vec::new();

    // The setup code for named fields
    let mut code_field_setup = Vec::new();

    // The code that checks if required fields have been set
    let mut code_field_required = Vec::new();

    // The match branch for a named field
    let mut code_named_match = Vec::new();

    // The type bounds
    let mut code_where = Vec::new();

    let mut constr_fields = Vec::new();

    // Whether we have seen a named field so far.
    // This is used to guarantee that positional fields must come before named ones.
    let mut seen_named = false;

    for field in &data_struct.fields {
        let Some(field_ident) = &field.ident else {
            return Err(syn::Error::new_spanned(
                field,
                "Fields must be named to derive Import.",
            ));
        };

        let field_data = parse_sexpr_attributes(&field.attrs)?;
        let field_type = &field.ty;

        let field_name = field_data
            .rename
            .unwrap_or_else(|| format!("{}", field_ident.to_token_stream()));

        let field_ident_var = syn::Ident::new(
            &format!("var_{}", field_ident.to_token_stream()),
            field_ident.span(),
        );

        match field_data.kind {
            FieldKind::Positional => {
                if seen_named {
                    return Err(syn::Error::new_spanned(
                        field,
                        "Positional fields must come before named fields.",
                    ));
                }

                code_positional.push(quote! {
                    let (values, #field_ident_var) = <_ as ::hugr_sexpr::import::Import<'a, A>>::import(values)?;
                });

                code_where.push(quote! {
                    #field_type: ::hugr_sexpr::import::Import<'a, A>,
                });
            }
            FieldKind::NamedRequired => {
                seen_named = true;

                code_field_setup.push(quote! {
                    let mut #field_ident_var = None;
                });

                let missing_field_message = format!("Missing required field `{}`.", field_name);

                code_field_required.push(quote! {
                    let Some(#field_ident_var) = #field_ident_var else {
                        return Err(::hugr_sexpr::import::ImportError::new(
                            #missing_field_message
                        ));
                    };
                });

                let duplicate_field_message = format!("Duplicate field `{}`.", field_name);

                code_named_match.push(quote! {
                    #field_name => {
                        if #field_ident_var.is_some() {
                            return Err(::hugr_sexpr::import::ImportError::new_with_meta(
                                #duplicate_field_message,
                                field_value.meta().clone()
                            ));
                        }

                        let (values, value) = <_ as ::hugr_sexpr::import::Import<'a, A>>::import(values)?;
                        #field_ident_var = Some(value);
                    },
                });

                code_where.push(quote! {
                    #field_type: ::hugr_sexpr::import::Import<'a, A>,
                });
            }
            FieldKind::NamedOptional => {
                seen_named = true;

                code_field_setup.push(quote! {
                    let mut #field_ident_var = None;
                });

                let duplicate_field_message = format!("Duplicate field `{}`.", field_name);

                code_named_match.push(quote! {
                    #field_name => {
                        if #field_ident_var.is_some() {
                            return Err(::hugr_sexpr::import::ImportError::new_with_meta(
                                #duplicate_field_message,
                                field_value.meta().clone()
                            ));
                        }

                        let (values, value) = <_ as ::hugr_sexpr::import::Import<'a, A>>::import(values)?;
                        #field_ident_var = Some(value);
                    }
                });

                // As with the positional and required fields, we need to ensure that
                // the type of the field is parseable by adding a constraint bound for `Import`.
                // But since the type of an optional field is wrapped in an `Option`, we
                // first need to extract it.
                let inner_type = get_first_type_arg(field_type).ok_or(syn::Error::new_spanned(
                    field,
                    "Optional field must have type `Option<T>` for some `T`.",
                ))?;

                code_where.push(quote! {
                    #inner_type: ::hugr_sexpr::import::Import<'a, A>,
                });
            }
            FieldKind::NamedRepeated => {
                seen_named = true;

                code_field_setup.push(quote! {
                    let mut #field_ident_var = Vec::new();
                });

                code_named_match.push(quote! {
                    #field_name => {
                        let (values, value) = <_ as ::hugr_sexpr::import::Import<'a, A>>::import(values)?;
                        #field_ident_var.push(value);
                    }
                });

                // Analogous to the optional fields, the type for a repeated field is wrapped
                // within a `Vec` and therefore needs to be extracted first before we can add
                // the constraint bound.
                let inner_type = get_first_type_arg(field_type).ok_or(syn::Error::new_spanned(
                    field,
                    "Repeated field must have type `Option<T>` for some `T`.",
                ))?;

                code_where.push(quote! {
                    #inner_type: ::hugr_sexpr::import::Import<'a, A>,
                });
            }
        };

        constr_fields.push(quote! {
            #field_ident: #field_ident_var,
        });
    }

    let code_named_match: TokenStream = code_named_match.into_iter().collect();
    let code_named = quote! {
        for field_value in values {
            let (head, values) = field_value.as_list_with_head().ok_or_else(||
                ::hugr_sexpr::import::ImportError::new_with_meta(
                    "Expected field.",
                    field_value.meta().clone()
                )
            )?;
            match head.as_ref() {
                #code_named_match
                _ => {
                    return Err(::hugr_sexpr::import::ImportError::new_with_meta(
                        "Unknown field.",
                        field_value.meta().clone()
                    ));
                }
            };
        }
    };

    let code_positional: TokenStream = code_positional.into_iter().collect();
    let code_field_setup: TokenStream = code_field_setup.into_iter().collect();
    let code_field_required: TokenStream = code_field_required.into_iter().collect();
    let code_where: TokenStream = code_where.into_iter().collect();
    let constr_fields: TokenStream = constr_fields.into_iter().collect();

    Ok(quote! {
        #[automatically_derived]
        impl<'a, A, #struct_generics> ::hugr_sexpr::import::Import<'a, A> for #struct_ident<#struct_generics>
        where A: Clone, #code_where {
            fn import(values: &'a [::hugr_sexpr::Value<A>]) -> ::hugr_sexpr::import::ImportResult<'a, Self, A>
            where
                Self: Sized {
                #code_positional
                #code_field_setup
                #code_named
                #code_field_required
                Ok((&[], Self {
                    #constr_fields
                }))
            }
        }
    })
}
