use std::{str::FromStr, sync::Arc};

use base64::Engine as _;
use base64::prelude::BASE64_STANDARD;
use smol_str::{SmolStr, SmolStrBuilder};
use thiserror::Error;

use crate::v0::Literal;

pub(crate) fn parse_string_literal(str: &str) -> Result<SmolStr, StringParseError> {
    let Some(str) = str.strip_prefix('"').and_then(|str| str.strip_suffix('"')) else {
        return Err(StringParseError::Delimiters);
    };

    let mut builder = SmolStrBuilder::new();
    let mut chars = str.char_indices();

    while let Some((_, char)) = chars.next() {
        let unescaped = match char {
            '"' => return Err(StringParseError::UnescapedQuote),
            '\\' => {
                let (start, char) = chars.next().ok_or(StringParseError::MissingEscape)?;
                match char {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    'u' => {
                        let rest = str[start..]
                            .strip_prefix("u{")
                            .ok_or(StringParseError::BadUnicode)?;
                        let (code_str, rest) =
                            rest.split_once("}").ok_or(StringParseError::BadUnicode)?;
                        let code = u32::from_str_radix(code_str, 16)
                            .map_err(|_| StringParseError::BadUnicode)?;
                        let char = char::from_u32(code)
                            .ok_or_else(|| StringParseError::UnknownUnicode(code))?;
                        chars = rest.char_indices();
                        char
                    }
                    _ => return Err(StringParseError::UnknownEscape(char)),
                }
            }
            char => char,
        };

        builder.push(unescaped);
    }

    Ok(builder.finish())
}

#[derive(Debug, Error)]
pub enum StringParseError {
    #[error("unknown escape char `{0}`")]
    UnknownEscape(char),
    #[error("missing escaped char after backslash")]
    MissingEscape,
    #[error("unescaped quote")]
    UnescapedQuote,
    #[error("badly formatted unicode escape sequence")]
    BadUnicode,
    #[error("unknown unicode code point {0}")]
    UnknownUnicode(u32),
    #[error(r#"string literal must start and end with a double quote `"`"#)]
    Delimiters,
}

fn parse_bytes_literal(str: &str) -> Result<Arc<[u8]>, BytesParseError> {
    let Some(str) = str
        .strip_prefix("b\"")
        .and_then(|str| str.strip_suffix('"'))
    else {
        return Err(BytesParseError::Delimiters);
    };

    let data = BASE64_STANDARD
        .decode(str)
        .map_err(|_| BytesParseError::Base64)?;

    Ok(data.into())
}

#[derive(Debug, Error)]
pub enum BytesParseError {
    #[error(r#"byte string literals must start with `b"` and end with `"`"#)]
    Delimiters,
    #[error("failed to decode base64 string")]
    Base64,
}

impl FromStr for Literal {
    type Err = LiteralParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.starts_with('"') {
            Ok(Self::Str(parse_string_literal(s)?))
        } else if s.starts_with('b') {
            Ok(Self::Bytes(parse_bytes_literal(s)?))
        } else {
            todo!()
        }
    }
}

#[derive(Debug, Error)]
pub enum LiteralParseError {
    #[error("unexpected literal")]
    Unexpected,
    #[error("failed to parse string")]
    String(#[from] StringParseError),
    #[error("failed to byte string")]
    Bytes(#[from] BytesParseError),
}
