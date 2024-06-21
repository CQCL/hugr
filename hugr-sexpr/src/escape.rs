use logos::Logos;

/// Lexer token for an escaped string or symbol.
#[derive(Debug, Clone, Logos)]
enum EscapedToken {
    #[token(r#"\n"#, |_| '\n')]
    #[token(r#"\r"#, |_| '\r')]
    #[token(r#"\t"#, |_| '\t')]
    #[token(r#"\""#, |_| '"')]
    #[token(r#"\|"#, |_| '|')]
    #[token(r#"\\"#, |_| '\\')]
    Escaped(char),

    #[regex(r#"\\u\{[a-fA-F0-9]+\}"#, |lex| parse_unicode(lex.slice()))]
    Unicode(char),

    #[regex(r#"[^\\]"#)]
    Literal,
}

/// Parses a unicode escape sequence of the form `\u{HEX}` where `HEX` is a
/// hexadecimal number representing a unicode codepoint.
fn parse_unicode(str: &str) -> Option<char> {
    // Skip the '\u{' prefix and '}' suffix
    let hex = str.get(3..str.len() - 1)?;
    let code = u32::from_str_radix(hex, 16).ok()?;
    char::from_u32(code)
}

/// Replaces escape sequences with their corresponding characters.
pub fn unescape(str: &str) -> Option<String> {
    let mut lexer = EscapedToken::lexer(str);
    let mut output = String::with_capacity(str.len());

    while let Some(token) = lexer.next() {
        let token = token.ok()?;

        match token {
            EscapedToken::Escaped(c) => output.push(c),
            EscapedToken::Unicode(c) => output.push(c),
            EscapedToken::Literal => output.push_str(lexer.slice()),
        }
    }

    Some(output)
}

pub fn escape_string(str: &str) -> String {
    let mut output = String::with_capacity(str.len());

    for c in str.chars() {
        match c {
            '\n' => output.push_str(r#"\n"#),
            '\r' => output.push_str(r#"\r"#),
            '\t' => output.push_str(r#"\t"#),
            '"' => output.push_str(r#"\""#),
            '\\' => output.push_str(r#"\\"#),
            c => output.push(c),
        }
    }

    output
}

/// Lexer token that matches a symbol which can be printed without escaping.
///
/// Since we use the logos crate anyway for parsing, we might as well use it to
/// check if a symbol needs to be escaped. This avoids some brittle manual code
/// or pulling in the `regex` crate needlessly.
#[derive(Debug, Clone, PartialEq, Logos)]
enum BareSymbol {
    #[regex(r#"[a-zA-Z!$%&*/:<=>?\^_~\.@][a-zA-Z!$%&*/:<=>?\^_~0-9+\-\.@]*"#)]
    #[regex(r#"[+-]([a-zA-Z!$%&*/:<=>?\^_~\.@][a-zA-Z!$%&*/:<=>?\^_~0-9+\-\.@]*)?"#)]
    BareSymbol,
}

/// Escape a symbol. If the symbol can occur on its own, it is returned as is.
/// Otherwise it is escaped and surrounded by `|` characters.
pub fn escape_symbol(str: &str) -> String {
    // If the symbol is fine without escaping, we can return it directly.
    {
        let mut lexer = BareSymbol::lexer(str);
        let first_token = lexer.next();
        let second_token = lexer.next();

        if matches!(first_token, Some(Ok(_))) && second_token.is_none() {
            return str.to_string();
        }
    }

    let mut output = String::with_capacity(str.len() + 2);
    output.push('|');

    for c in str.chars() {
        match c {
            '\n' => output.push_str(r#"\n"#),
            '\r' => output.push_str(r#"\r"#),
            '\t' => output.push_str(r#"\t"#),
            '|' => output.push_str(r#"\|"#),
            '\\' => output.push_str(r#"\\"#),
            c => output.push(c),
        }
    }

    output.push('|');
    output
}

#[cfg(test)]
mod test {
    use super::{escape_string, escape_symbol, unescape};
    use rstest::rstest;

    #[rstest]
    #[case("symbol", "symbol")]
    #[case("\n", r#"|\n|"#)]
    #[case("3", "|3|")]
    #[case("+", "+")]
    #[case("-", "-")]
    #[case("-3", "|-3|")]
    #[case(".3", ".3")]
    #[case("|", r#"|\||"#)]
    #[case("", "||")]
    #[case(r#"\"#, r#"|\\|"#)]
    #[case(r#"""#, r#"|"|"#)]
    #[case("+any", "+any")]
    #[case("-any", "-any")]
    #[case("#symbol", "|#symbol|")]
    fn test_escape_symbol(#[case] symbol: &str, #[case] expected: &str) {
        assert_eq!(expected, escape_symbol(symbol));
    }

    #[rstest]
    #[case("string", "string")]
    #[case("\n", r"\n")]
    #[case(r"\", r"\\")]
    #[case(r#"""#, r#"\""#)]
    #[case("|", "|")]
    #[case("", "")]
    fn test_escape_string(#[case] string: &str, #[case] expected: &str) {
        assert_eq!(expected, escape_string(string));
    }

    #[rstest]
    #[case(r#"\""#, r#"""#)]
    #[case(r"\|", "|")]
    #[case(r"\u{1F60A}", "\u{1F60A}")]
    fn test_unescape(#[case] escaped: &str, #[case] expected: &str) {
        assert_eq!(expected, unescape(escaped).unwrap());
    }
}
