//! Supporting Rust library for the hugr Python bindings.

use hugr_core::{
    envelope::{EnvelopeConfig, EnvelopeFormat, read_envelope, write_envelope},
    std_extensions::STD_REG,
};
use hugr_model::v0::ast;
use pyo3::{exceptions::PyValueError, prelude::*};

macro_rules! syntax_to_and_from_string {
    ($name:ident, $ty:ty) => {
        pastey::paste! {
            #[pyfunction]
            fn [<$name _to_string>](ob: ast::$ty) -> PyResult<String> {
                Ok(format!("{}", ob))
            }

            #[pyfunction]
            fn [<string_to_ $name>](string: String) -> PyResult<ast::$ty> {
                string
                    .parse::<ast::$ty>()
                    .map_err(|err| PyValueError::new_err(err.to_string()))
            }
        }
    };
}

syntax_to_and_from_string!(term, Term);
syntax_to_and_from_string!(node, Node);
syntax_to_and_from_string!(region, Region);
syntax_to_and_from_string!(module, Module);
syntax_to_and_from_string!(param, Param);
syntax_to_and_from_string!(symbol, Symbol);
syntax_to_and_from_string!(package, Package);

#[pyfunction]
fn package_to_bytes(package: ast::Package) -> PyResult<Vec<u8>> {
    let bump = bumpalo::Bump::new();
    let resolved = package
        .resolve(&bump)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let bytes = hugr_model::v0::binary::write_to_vec(&resolved);
    Ok(bytes)
}

#[pyfunction]
fn bytes_to_package(bytes: &[u8]) -> PyResult<ast::Package> {
    let bump = bumpalo::Bump::new();
    let table = hugr_model::v0::binary::read_from_slice(bytes, &bump)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let package = table
        .as_ast()
        .ok_or_else(|| PyValueError::new_err("Malformed package"))?;
    Ok(package)
}

/// Convert an envelope to a new envelope in JSON format.
#[pyfunction]
fn to_json_envelope(bytes: &[u8]) -> PyResult<Vec<u8>> {
    let (_, pkg) =
        read_envelope(bytes, &STD_REG).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let config_json = EnvelopeConfig::new(EnvelopeFormat::PackageJson);
    let mut json_data: Vec<u8> = Vec::new();
    write_envelope(&mut json_data, &pkg, config_json).unwrap();
    Ok(json_data)
}

/// Returns the current version of the HUGR model format as a tuple of (major, minor, patch).
#[pyfunction]
fn current_model_version() -> (u64, u64, u64) {
    (
        hugr_model::CURRENT_VERSION.major,
        hugr_model::CURRENT_VERSION.minor,
        hugr_model::CURRENT_VERSION.patch,
    )
}

#[pymodule]
fn _hugr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(term_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_term, m)?)?;
    m.add_function(wrap_pyfunction!(node_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_node, m)?)?;
    m.add_function(wrap_pyfunction!(region_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_region, m)?)?;
    m.add_function(wrap_pyfunction!(module_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_module, m)?)?;
    m.add_function(wrap_pyfunction!(package_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(package_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_package, m)?)?;
    m.add_function(wrap_pyfunction!(bytes_to_package, m)?)?;
    m.add_function(wrap_pyfunction!(param_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_param, m)?)?;
    m.add_function(wrap_pyfunction!(symbol_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(current_model_version, m)?)?;
    m.add_function(wrap_pyfunction!(to_json_envelope, m)?)?;
    Ok(())
}
