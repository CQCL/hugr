#![allow(missing_docs)]

use anyhow::Result;
use rstest::{fixture, rstest};
use std::str::FromStr;

use hugr::{
    Extension, Hugr,
    builder::{Dataflow as _, DataflowHugr as _},
    envelope::{EnvelopeConfig, EnvelopeFormat, read_envelope, write_envelope},
    extension::prelude::bool_t,
    package::Package,
    std_extensions::std_reg,
    types::Signature,
};
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

#[fixture]
fn simple_dfg_hugr() -> Hugr {
    let dfg_builder =
        hugr::builder::DFGBuilder::new(Signature::new(vec![bool_t()], vec![bool_t()])).unwrap();
    let [i1] = dfg_builder.input_wires_arr();
    dfg_builder.finish_hugr_with_outputs([i1]).unwrap()
}

#[rstest]
#[case(EnvelopeFormat::ModelTextWithExtensions)]
#[case(EnvelopeFormat::ModelWithExtensions)]
fn import_package_with_extensions(#[case] format: EnvelopeFormat, simple_dfg_hugr: Hugr) {
    let ext = Extension::new_arc(
        "miniquantum".try_into().unwrap(),
        hugr::extension::Version::new(0, 1, 0),
        |_, _| {},
    );
    let mut package = Package::new([simple_dfg_hugr]);
    package.extensions.register_updated(ext);

    let mut bytes: Vec<u8> = Vec::new();
    write_envelope(&mut bytes, &package, EnvelopeConfig::new(format)).unwrap();

    let buff = std::io::BufReader::new(bytes.as_slice());
    let (_, loaded_pkg) = read_envelope(buff, &std_reg()).unwrap();

    assert_eq!(loaded_pkg.extensions.len(), 1);
    let read_ext = loaded_pkg.extensions.iter().next().unwrap();
    assert_eq!(read_ext.name(), &"miniquantum".try_into().unwrap());

    assert_eq!(package, loaded_pkg);
}
