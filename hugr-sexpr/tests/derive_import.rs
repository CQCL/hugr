use hugr_sexpr::{import::Import, read_values, Symbol};

#[test]
pub fn positional() {
    #[derive(Import)]
    struct Test {
        first: Symbol,
        second: String,
    }

    let text = r#"symbol "string""#;
    let values = read_values(text).unwrap();
    let (_, test) = Test::import(&values).unwrap();

    assert_eq!(test.first, Symbol::new("symbol"));
    assert_eq!(test.second, "string");
}

#[test]
pub fn optional_given() {
    #[derive(Import)]
    struct Test {
        #[sexpr(optional)]
        field: Option<String>,
    }

    let text = r#"(field "string")"#;
    let values = read_values(text).unwrap();
    let (_, test) = Test::import(&values).unwrap();

    assert_eq!(test.field.unwrap(), "string");
}

pub fn optional_absent() {
    #[derive(Import)]
    struct Test {
        #[sexpr(optional)]
        field: Option<String>,
    }

    let text = r#""#;
    let values = read_values(text).unwrap();
    let (_, test) = Test::import(&values).unwrap();

    assert_eq!(test.field, None);
}

#[test]
pub fn optional_duplicate() {
    #[derive(Import)]
    struct Test {
        #[allow(dead_code)]
        #[sexpr(optional)]
        field: Option<String>,
    }

    let text = r#"(field "string") (field "another")"#;
    let values = read_values(text).unwrap();
    assert!(Test::import(&values).is_err());
}

#[test]
pub fn required_given() {
    #[derive(Import)]
    struct Test {
        #[sexpr(required)]
        field: String,
    }

    let text = r#"(field "string")"#;
    let values = read_values(text).unwrap();
    let (_, test) = Test::import(&values).unwrap();

    assert_eq!(test.field, "string");
}

pub fn required_absent() {
    #[derive(Import)]
    struct Test {
        #[allow(dead_code)]
        #[sexpr(optional)]
        field: Option<String>,
    }

    let text = r#""#;
    let values = read_values(text).unwrap();

    assert!(Test::import(&values).is_err());
}

#[test]
pub fn required_duplicate() {
    #[derive(Import)]
    struct Test {
        #[allow(dead_code)]
        #[sexpr(required)]
        field: String,
    }

    let text = r#"(field "string") (field "another")"#;
    let values = read_values(text).unwrap();

    assert!(Test::import(&values).is_err());
}

#[test]
pub fn repeated() {
    #[derive(Import)]
    struct Test {
        #[sexpr(repeated)]
        values: Vec<String>,
    }

    let mut text = String::new();
    let mut expected = Vec::new();

    for i in 0..3 {
        let values = read_values(&text).unwrap();
        let (_, test) = Test::import(&values).unwrap();
        assert_eq!(test.values, expected);

        text.push_str(&format!(r#" (values "{}")"#, i));
        expected.push(format!("{}", i));
    }
}

#[test]
pub fn resursive_field() {
    #[derive(Import, PartialEq, Eq, Debug)]
    struct Outer {
        #[sexpr(repeated)]
        inner: Vec<Inner>,
    }

    #[derive(Import, PartialEq, Eq, Debug)]
    struct Inner {
        positional: Symbol,
        #[sexpr(required)]
        field: String,
    }

    let text = r#"
        (inner first (field "first"))
        (inner second (field "second"))
    "#;

    let expected = Outer {
        inner: vec![
            Inner {
                positional: "first".into(),
                field: "first".into(),
            },
            Inner {
                positional: "second".into(),
                field: "second".into(),
            },
        ],
    };

    let values = read_values(&text).unwrap();
    let (_, test) = Outer::import(&values).unwrap();
    assert_eq!(test, expected);
}
