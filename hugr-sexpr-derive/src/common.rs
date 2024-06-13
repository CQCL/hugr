use syn::{Attribute, LitStr};

#[derive(Debug, Clone, Copy)]
pub enum FieldKind {
    Positional,
    NamedRequired,
    NamedOptional,
    NamedRepeated,
}

pub struct FieldData {
    pub kind: FieldKind,
    pub rename: Option<String>,
}

/// Parse the `sexpr` attributes on a field.
pub fn parse_sexpr_attributes(attrs: &[Attribute]) -> syn::Result<FieldData> {
    let mut field_data = FieldData {
        kind: FieldKind::Positional,
        rename: None,
    };

    for attr in attrs {
        if !attr.path().is_ident("sexpr") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            let path = &meta.path;
            if path.is_ident("required") {
                field_data.kind = FieldKind::NamedRequired;
                Ok(())
            } else if path.is_ident("optional") {
                field_data.kind = FieldKind::NamedOptional;
                Ok(())
            } else if path.is_ident("repeated") {
                field_data.kind = FieldKind::NamedRepeated;
                Ok(())
            } else if path.is_ident("rename") {
                let value = meta.value()?;
                let name: LitStr = value.parse()?;
                field_data.rename = Some(name.value());
                Ok(())
            } else {
                Err(meta.error("unrecognized sexpr attribute"))
            }
        })?;
    }

    Ok(field_data)
}

/// Extract the first type argument from a type, if any.
pub fn get_first_type_arg(ty: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::Path(path) = ty else {
        return None;
    };

    let syn::PathArguments::AngleBracketed(args) = &path.path.segments.last()?.arguments else {
        return None;
    };

    let syn::GenericArgument::Type(ty) = args.args.first()? else {
        return None;
    };

    Some(ty)
}
