#![allow(missing_docs)]

use anyhow::Result;
use std::str::FromStr;

use hugr::{envelope::read_envelope, std_extensions::std_reg};
use hugr_core::{export::export_package, import::import_package};
use hugr_model::v0 as model;

fn roundtrip(source: &str) -> Result<String> {
    let bump = model::bumpalo::Bump::new();
    let package_ast = model::ast::Package::from_str(source)?;
    let package_table = package_ast.resolve(&bump)?;
    let core = import_package(&package_table, Default::default(), &std_reg())?;
    let exported_table = export_package(&core.modules, &core.extensions, &bump);
    let exported_ast = exported_table.as_ast().unwrap();

    Ok(exported_ast.to_string())
}

macro_rules! test_roundtrip {
    ($name: ident, $file: expr) => {
        #[test]
        #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
        pub fn $name() {
            let ast = roundtrip(include_str!($file)).unwrap_or_else(|err| panic!("{:?}", err));
            insta::assert_snapshot!(ast)
        }
    };
}

test_roundtrip!(
    test_roundtrip_add,
    "../../hugr-model/tests/fixtures/model-add.edn"
);

test_roundtrip!(
    test_roundtrip_call,
    "../../hugr-model/tests/fixtures/model-call.edn"
);

test_roundtrip!(
    test_roundtrip_alias,
    "../../hugr-model/tests/fixtures/model-alias.edn"
);

test_roundtrip!(
    test_roundtrip_cfg,
    "../../hugr-model/tests/fixtures/model-cfg.edn"
);

test_roundtrip!(
    test_roundtrip_cond,
    "../../hugr-model/tests/fixtures/model-cond.edn"
);

test_roundtrip!(
    test_roundtrip_loop,
    "../../hugr-model/tests/fixtures/model-loop.edn"
);

test_roundtrip!(
    test_roundtrip_params,
    "../../hugr-model/tests/fixtures/model-params.edn"
);

test_roundtrip!(
    test_roundtrip_constraints,
    "../../hugr-model/tests/fixtures/model-constraints.edn"
);

test_roundtrip!(
    test_roundtrip_const,
    "../../hugr-model/tests/fixtures/model-const.edn"
);

test_roundtrip!(
    test_roundtrip_order,
    "../../hugr-model/tests/fixtures/model-order.edn"
);

test_roundtrip!(
    test_roundtrip_entrypoint,
    "../../hugr-model/tests/fixtures/model-entrypoint.edn"
);

#[test]
fn import_package_with_extensions() {
    let simple_with_exts = r#"HUGRiHJv)@[{"version":"0.1.0","name":"miniquantum","types":{},"operations":{"H":{"extension":"miniquantum","name":"H","description":"Hadamard gate","signature":{"params":[],"body":{"input":[{"t":"Q"}],"output":[{"t":"Q"}]}},"binary":false}}}](hugr 0)
        (mod)

        (import core.entrypoint)

        (import core.title)

        (import prelude.qubit)

        (import core.fn)

        (import core.meta.description)

        (declare-operation miniquantum.H (core.fn [prelude.qubit] [prelude.qubit])
        (meta (core.meta.description "Hadamard gate")))

        (define-func private _1 (core.fn [prelude.qubit] [prelude.qubit])
        (meta (core.title "main"))
        (dfg [%0] [%1]
            (signature (core.fn [prelude.qubit] [prelude.qubit]))
            (dfg [%0] [%1]
            (signature (core.fn [prelude.qubit] [prelude.qubit]))
            (meta core.entrypoint)
            (dfg [%2] [%3]
                (signature (core.fn [prelude.qubit] [prelude.qubit]))
                (miniquantum.H [%2] [%3]
                (signature (core.fn [prelude.qubit] [prelude.qubit])))))))
    "#;

    let bytes = simple_with_exts.as_bytes();
    let buff = std::io::BufReader::new(bytes);
    let (_, pkg) = read_envelope(buff, &std_reg()).unwrap();

    assert_eq!(pkg.modules.len(), 1);
    assert_eq!(pkg.extensions.len(), 1);

    assert_eq!(
        pkg.extensions.iter().next().unwrap().name(),
        &hugr::extension::ExtensionId::new_unchecked("miniquantum")
    );
}
