use hugr_model::v0::ast;
use pyo3::{exceptions::PyValueError, prelude::*};

macro_rules! syntax_to_and_from_string {
    ($name:ident, $ty:ty) => {
        paste::paste! {
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

#[pyfunction]
fn module_to_bytes(module: ast::Module) -> PyResult<Vec<u8>> {
    let bump = bumpalo::Bump::new();
    let resolved = module
        .resolve(&bump)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let bytes = hugr_model::v0::binary::write_to_vec(&resolved);
    Ok(bytes)
}

#[pyfunction]
fn bytes_to_module(bytes: &[u8]) -> PyResult<ast::Module> {
    let bump = bumpalo::Bump::new();
    let table = hugr_model::v0::binary::read_from_slice(bytes, &bump)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let module = table
        .as_ast()
        .ok_or_else(|| PyValueError::new_err("Malformed module"))?;
    Ok(module)
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
    m.add_function(wrap_pyfunction!(module_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_module, m)?)?;
    m.add_function(wrap_pyfunction!(bytes_to_module, m)?)?;
    m.add_function(wrap_pyfunction!(param_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_param, m)?)?;
    m.add_function(wrap_pyfunction!(symbol_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_symbol, m)?)?;
    Ok(())
}
