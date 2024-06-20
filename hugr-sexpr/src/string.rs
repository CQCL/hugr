/// Replaces escape sequences with their corresponding characters.
pub fn unescape_string(str: &str) -> Option<String> {
    let mut input = str.chars();
    let mut output = String::new();
    let mut unicode = String::new();

    while let Some(c) = input.next() {
        if c != '\\' {
            output.push(c);
            continue;
        }

        match input.next() {
            Some('b') => output.push('\u{0008}'),
            Some('n') => output.push('\n'),
            Some('f') => output.push('\u{000C}'),
            Some('r') => output.push('\r'),
            Some('t') => output.push('\t'),
            Some('"') => output.push('"'),
            Some('\\') => output.push('\\'),
            Some('u') => {
                for _ in 0..4 {
                    unicode.push(input.next()?);
                }
                let codepoint = u32::from_str_radix(&unicode, 16).ok()?;
                unicode.clear();
                output.push(char::from_u32(codepoint)?);
            }
            _ => return None,
        }
    }

    Some(output)
}

/// Escapes a string so that it can be used within double quotes.
pub fn escape_string(str: &str) -> String {
    let mut output = String::new();

    for c in str.chars() {
        match c {
            '\u{0008}' => output.push_str(r#"\b"#),
            '\n' => output.push_str(r#"\n"#),
            '\u{000C}' => output.push_str(r#"\f"#),
            '\r' => output.push_str(r#"\r"#),
            '\t' => output.push_str(r#"\t"#),
            '"' => output.push_str(r#"\""#),
            '\\' => output.push_str(r#"\\"#),
            c if c.is_control() => output.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => output.push(c),
        }
    }

    output
}
