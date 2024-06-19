//! Procedural macros for converting from s-expressions to Rust types and back.
//! See the `hugr_sexpr` crate for more details on s-expressions and on how
//! to use the derive macros.
use syn::{parse_macro_input, DeriveInput};

pub(crate) mod common;
mod input;
mod output;

/// Derive the [`Input`] trait.
#[proc_macro_derive(Input, attributes(sexpr))]
pub fn derive_input(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let derive_input = parse_macro_input!(input as DeriveInput);
    input::derive_input_impl(derive_input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derive the [`Output`] trait.
#[proc_macro_derive(Output, attributes(sexpr))]
pub fn derive_export(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let derive_input = parse_macro_input!(input as DeriveInput);
    output::derive_output_impl(derive_input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
