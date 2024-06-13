//! Procedural macros for converting from s-expressions to Rust types and back.
//! See the `hugr_sexpr` crate for more details on s-expressions and on how
//! to use the derive macros.
use syn::{parse_macro_input, DeriveInput};

pub(crate) mod common;
mod export;
mod import;

/// Derive the [`Import`] trait.
#[proc_macro_derive(Import, attributes(sexpr))]
pub fn derive_import(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let derive_input = parse_macro_input!(input as DeriveInput);
    import::derive_import_impl(derive_input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derive the [`Export`] trait.
#[proc_macro_derive(Export, attributes(sexpr))]
pub fn derive_export(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let derive_input = parse_macro_input!(input as DeriveInput);
    export::derive_export_impl(derive_input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
