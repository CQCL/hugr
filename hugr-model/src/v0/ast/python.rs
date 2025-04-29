use std::sync::Arc;

use super::{Module, Node, Operation, Package, Param, Region, SeqPart, Symbol, Term};
use pyo3::{
    Bound, PyAny, PyResult,
    exceptions::PyTypeError,
    types::{PyAnyMethods, PyStringMethods as _, PyTypeMethods as _},
};

impl<'py> pyo3::FromPyObject<'py> for Term {
    fn extract_bound(term: &Bound<'py, PyAny>) -> PyResult<Self> {
        let name = term.get_type().name()?;

        Ok(match name.to_str()? {
            "Wildcard" => Self::Wildcard,
            "Var" => {
                let name = term.getattr("name")?.extract()?;
                Self::Var(name)
            }
            "Apply" => {
                let symbol = term.getattr("symbol")?.extract()?;
                let args: Vec<_> = term.getattr("args")?.extract()?;
                Self::Apply(symbol, args.into())
            }
            "List" => {
                let parts: Vec<_> = term.getattr("parts")?.extract()?;
                Self::List(parts.into())
            }
            "Tuple" => {
                let parts: Vec<_> = term.getattr("parts")?.extract()?;
                Self::Tuple(parts.into())
            }
            "Literal" => {
                let literal = term.getattr("value")?.extract()?;
                Self::Literal(literal)
            }
            "Func" => {
                let region = term.getattr("region")?.extract()?;
                Self::Func(Arc::new(region))
            }
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Unknown Term type: {}.",
                    name.to_str()?
                )));
            }
        })
    }
}

impl<'py> pyo3::IntoPyObject<'py> for &Term {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        match self {
            Term::Wildcard => {
                let py_class = py_module.getattr("Wildcard")?;
                py_class.call0()
            }
            Term::Var(var_name) => {
                let py_class = py_module.getattr("Var")?;
                py_class.call1((var_name.as_ref(),))
            }
            Term::Apply(symbol_name, terms) => {
                let py_class = py_module.getattr("Apply")?;
                py_class.call1((symbol_name.as_ref(), terms.as_ref()))
            }
            Term::List(parts) => {
                let py_class = py_module.getattr("List")?;
                py_class.call1((parts.as_ref(),))
            }
            Term::Literal(literal) => {
                let py_class = py_module.getattr("Literal")?;
                py_class.call1((literal,))
            }
            Term::Tuple(parts) => {
                let py_class = py_module.getattr("Tuple")?;
                py_class.call1((parts.as_ref(),))
            }
            Term::Func(region) => {
                let py_class = py_module.getattr("Func")?;
                py_class.call1((region.as_ref(),))
            }
        }
    }
}

impl<'py> pyo3::FromPyObject<'py> for SeqPart {
    fn extract_bound(part: &Bound<'py, PyAny>) -> PyResult<Self> {
        let name = part.get_type().name()?;

        if name.to_str()? == "Splice" {
            let term = part.getattr("seq")?.extract()?;
            Ok(Self::Splice(term))
        } else {
            let term = part.extract()?;
            Ok(Self::Item(term))
        }
    }
}

impl<'py> pyo3::IntoPyObject<'py> for &SeqPart {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        match self {
            SeqPart::Item(term) => term.into_pyobject(py),
            SeqPart::Splice(term) => {
                let py_class = py_module.getattr("Splice")?;
                py_class.call1((term,))
            }
        }
    }
}

impl<'py> pyo3::FromPyObject<'py> for Param {
    fn extract_bound(symbol: &Bound<'py, PyAny>) -> PyResult<Self> {
        let name = symbol.getattr("name")?.extract()?;
        let r#type = symbol.getattr("type")?.extract()?;
        Ok(Self { name, r#type })
    }
}

impl<'py> pyo3::IntoPyObject<'py> for &Param {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        let py_class = py_module.getattr("Param")?;
        py_class.call1((self.name.as_ref(), &self.r#type))
    }
}

impl<'py> pyo3::FromPyObject<'py> for Symbol {
    fn extract_bound(symbol: &Bound<'py, PyAny>) -> PyResult<Self> {
        let name = symbol.getattr("name")?.extract()?;
        let params: Vec<_> = symbol.getattr("params")?.extract()?;
        let constraints: Vec<_> = symbol.getattr("constraints")?.extract()?;
        let signature = symbol.getattr("signature")?.extract()?;
        Ok(Self {
            name,
            signature,
            params: params.into(),
            constraints: constraints.into(),
        })
    }
}

impl<'py> pyo3::IntoPyObject<'py> for &Symbol {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        let py_class = py_module.getattr("Symbol")?;
        py_class.call1((
            self.name.as_ref(),
            self.params.as_ref(),
            self.constraints.as_ref(),
            &self.signature,
        ))
    }
}

impl<'py> pyo3::FromPyObject<'py> for Node {
    fn extract_bound(node: &Bound<'py, PyAny>) -> PyResult<Self> {
        let operation = node.getattr("operation")?.extract()?;
        let inputs: Vec<_> = node.getattr("inputs")?.extract()?;
        let outputs: Vec<_> = node.getattr("outputs")?.extract()?;
        let regions: Vec<_> = node.getattr("regions")?.extract()?;
        let meta: Vec<_> = node.getattr("meta")?.extract()?;
        let signature = node.getattr("signature")?.extract()?;

        Ok(Self {
            operation,
            inputs: inputs.into(),
            outputs: outputs.into(),
            regions: regions.into(),
            meta: meta.into(),
            signature,
        })
    }
}

impl<'py> pyo3::IntoPyObject<'py> for &Node {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        let py_class = py_module.getattr("Node")?;
        py_class.call1((
            &self.operation,
            self.inputs.as_ref(),
            self.outputs.as_ref(),
            self.regions.as_ref(),
            self.meta.as_ref(),
            &self.signature,
        ))
    }
}

impl<'py> pyo3::FromPyObject<'py> for Operation {
    fn extract_bound(op: &Bound<'py, PyAny>) -> PyResult<Self> {
        let name = op.get_type().name()?;

        Ok(match name.to_str()? {
            "InvalidOp" => Self::Invalid,
            "Dfg" => Self::Dfg,
            "Cfg" => Self::Cfg,
            "Block" => Self::Block,
            "DefineFunc" => {
                let symbol = op.getattr("symbol")?.extract()?;
                Self::DefineFunc(Box::new(symbol))
            }
            "DeclareFunc" => {
                let symbol = op.getattr("symbol")?.extract()?;
                Self::DeclareFunc(Box::new(symbol))
            }
            "DeclareConstructor" => {
                let symbol = op.getattr("symbol")?.extract()?;
                Self::DeclareConstructor(Box::new(symbol))
            }
            "DeclareOperation" => {
                let symbol = op.getattr("symbol")?.extract()?;
                Self::DeclareOperation(Box::new(symbol))
            }
            "DeclareAlias" => {
                let symbol = op.getattr("symbol")?.extract()?;
                Self::DeclareAlias(Box::new(symbol))
            }
            "DefineAlias" => {
                let symbol = op.getattr("symbol")?.extract()?;
                let value = op.getattr("value")?.extract()?;
                Self::DefineAlias(Box::new(symbol), value)
            }
            "TailLoop" => Self::TailLoop,
            "Conditional" => Self::Conditional,
            "Import" => {
                let name = op.getattr("name")?.extract()?;
                Self::Import(name)
            }
            "CustomOp" => {
                let operation = op.getattr("operation")?.extract()?;
                Self::Custom(operation)
            }
            _ => return Err(PyTypeError::new_err("Unknown Operation type.")),
        })
    }
}

impl<'py> pyo3::IntoPyObject<'py> for &Operation {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        match self {
            Operation::Invalid => {
                let py_class = py_module.getattr("InvalidOp")?;
                py_class.call0()
            }
            Operation::Dfg => {
                let py_class = py_module.getattr("Dfg")?;
                py_class.call0()
            }
            Operation::Cfg => {
                let py_class = py_module.getattr("Cfg")?;
                py_class.call0()
            }
            Operation::Block => {
                let py_class = py_module.getattr("Block")?;
                py_class.call0()
            }
            Operation::DefineFunc(symbol) => {
                let py_class = py_module.getattr("DefineFunc")?;
                py_class.call1((symbol.as_ref(),))
            }
            Operation::DeclareFunc(symbol) => {
                let py_class = py_module.getattr("DeclareFunc")?;
                py_class.call1((symbol.as_ref(),))
            }
            Operation::DeclareConstructor(symbol) => {
                let py_class = py_module.getattr("DeclareConstructor")?;
                py_class.call1((symbol.as_ref(),))
            }
            Operation::DeclareOperation(symbol) => {
                let py_class = py_module.getattr("DeclareOperation")?;
                py_class.call1((symbol.as_ref(),))
            }
            Operation::DeclareAlias(symbol) => {
                let py_class = py_module.getattr("DeclareAlias")?;
                py_class.call1((symbol.as_ref(),))
            }
            Operation::DefineAlias(symbol, value) => {
                let py_class = py_module.getattr("DefineAlias")?;
                py_class.call1((symbol.as_ref(), value))
            }
            Operation::TailLoop => {
                let py_class = py_module.getattr("TailLoop")?;
                py_class.call0()
            }
            Operation::Conditional => {
                let py_class = py_module.getattr("Conditional")?;
                py_class.call0()
            }
            Operation::Import(name) => {
                let py_class = py_module.getattr("Import")?;
                py_class.call1((name.as_ref(),))
            }
            Operation::Custom(term) => {
                let py_class = py_module.getattr("CustomOp")?;
                py_class.call1((term,))
            }
        }
    }
}

impl<'py> pyo3::FromPyObject<'py> for Region {
    fn extract_bound(region: &Bound<'py, PyAny>) -> PyResult<Self> {
        let kind = region.getattr("kind")?.extract()?;
        let sources: Vec<_> = region.getattr("sources")?.extract()?;
        let targets: Vec<_> = region.getattr("targets")?.extract()?;
        let children: Vec<_> = region.getattr("children")?.extract()?;
        let meta: Vec<_> = region.getattr("meta")?.extract()?;
        let signature = region.getattr("signature")?.extract()?;

        Ok(Self {
            kind,
            sources: sources.into(),
            targets: targets.into(),
            children: children.into(),
            meta: meta.into(),
            signature,
        })
    }
}

impl<'py> pyo3::IntoPyObject<'py> for &Region {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        let py_class = py_module.getattr("Region")?;
        py_class.call1((
            self.kind,
            self.sources.as_ref(),
            self.targets.as_ref(),
            self.children.as_ref(),
            self.meta.as_ref(),
            &self.signature,
        ))
    }
}

impl<'py> pyo3::FromPyObject<'py> for Module {
    fn extract_bound(module: &Bound<'py, PyAny>) -> PyResult<Self> {
        let root = module.getattr("root")?.extract()?;
        Ok(Self { root })
    }
}

impl<'py> pyo3::IntoPyObject<'py> for &Module {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        let py_class = py_module.getattr("Module")?;
        py_class.call1((&self.root,))
    }
}

impl<'py> pyo3::FromPyObject<'py> for Package {
    fn extract_bound(package: &Bound<'py, PyAny>) -> PyResult<Self> {
        let modules = package.getattr("modules")?.extract()?;
        Ok(Self { modules })
    }
}

impl<'py> pyo3::IntoPyObject<'py> for &Package {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        let py_class = py_module.getattr("Package")?;
        py_class.call1((&self.modules,))
    }
}

macro_rules! impl_into_pyobject_owned {
    ($ident:ty) => {
        impl<'py> pyo3::IntoPyObject<'py> for $ident {
            type Target = pyo3::PyAny;
            type Output = pyo3::Bound<'py, Self::Target>;
            type Error = pyo3::PyErr;

            fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
                (&self).into_pyobject(py)
            }
        }
    };
}

impl_into_pyobject_owned!(Term);
impl_into_pyobject_owned!(SeqPart);
impl_into_pyobject_owned!(Param);
impl_into_pyobject_owned!(Symbol);
impl_into_pyobject_owned!(Module);
impl_into_pyobject_owned!(Package);
impl_into_pyobject_owned!(Node);
impl_into_pyobject_owned!(Region);
impl_into_pyobject_owned!(Operation);
