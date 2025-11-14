//! Supporting Rust library for the hugr Python bindings.

use hugr_cli::{CliArgs, RunWithIoError};
use hugr_model::v0::ast;
use pyo3::{create_exception, exceptions::PyException, exceptions::PyValueError, prelude::*};

// Define custom exceptions
create_exception!(
    _hugr,
    HugrCliError,
    PyException,
    "Base exception for HUGR CLI errors."
);
create_exception!(
    _hugr,
    HugrCliDescribeError,
    HugrCliError,
    "Exception for HUGR CLI describe command errors with partial output."
);

/// Helper to convert RunWithIoError to Python exception
fn cli_error_to_py(err: RunWithIoError) -> PyErr {
    match err {
        RunWithIoError::Describe { source, output } => {
            // Convert output bytes to string, falling back to empty string if invalid UTF-8
            let output_str = String::from_utf8(output).unwrap_or_else(|e| {
                format!("<Invalid UTF-8 output: {} bytes>", e.as_bytes().len())
            });

            HugrCliDescribeError::new_err((format!("{:?}", source), output_str))
        }
        RunWithIoError::Other(e) => HugrCliError::new_err(format!("{:?}", e)),
        _ => {
            // Catch-all for any future error variants (non_exhaustive enum)
            HugrCliError::new_err(format!("{:?}", err))
        }
    }
}

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
fn bytes_to_package(bytes: &[u8]) -> PyResult<(ast::Package, Vec<u8>)> {
    let bump = bumpalo::Bump::new();
    let (table, suffix) = hugr_model::v0::binary::read_from_slice_with_suffix(bytes, &bump)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let package = table
        .as_ast()
        .ok_or_else(|| PyValueError::new_err("Malformed package"))?;
    Ok((package, suffix))
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

#[pyfunction]
/// Directly invoke the HUGR CLI entrypoint.
/// Arguments are extracted from std::env::args().
fn run_cli() {
    // python is the first arg so skip it
    CliArgs::new_from_args(std::env::args().skip(1)).run_cli();
}

/// Run a CLI command with bytes input and return bytes output.
///
/// This function provides a programmatic interface to the HUGR CLI,
/// allowing Python code to pass input data as bytes and receive output
/// as bytes, without needing to use stdin/stdout or temporary files.
///
/// # Arguments
///
/// * `args` - Command line arguments as a list of strings, not including the executable name.
/// * `input_bytes` - Optional input data as bytes (e.g., a HUGR package)
///
/// # Returns
///
/// Returns the command output as bytes, maybe empty.
/// Raises an exception on error.
///
/// Errors or tracing may still be printed to stderr as normal.
#[pyfunction]
#[pyo3(signature = (args, input_bytes=None))]
fn cli_with_io(mut args: Vec<String>, input_bytes: Option<&[u8]>) -> PyResult<Vec<u8>> {
    // placeholder for executable
    args.insert(0, String::new());
    let cli_args = CliArgs::new_from_args(args);
    let input = input_bytes.unwrap_or(&[]);
    cli_args.run_with_io(input).map_err(cli_error_to_py)
}

#[pymodule]
fn _hugr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register custom exceptions
    m.add("HugrCliError", m.py().get_type::<HugrCliError>())?;
    m.add(
        "HugrCliDescribeError",
        m.py().get_type::<HugrCliDescribeError>(),
    )?;

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
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    m.add_function(wrap_pyfunction!(cli_with_io, m)?)?;
    Ok(())
}
