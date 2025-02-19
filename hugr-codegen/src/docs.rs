use std::fmt::Display;
use std::fmt::Write;

use hugr_model::v0::{
    view::NamedConstructor, Module, Operation, CORE_CONSTRAINT, CORE_META, CORE_STATIC, CORE_TYPE,
};

use crate::util::find_node_docs;

pub fn generate(module: &Module) {
    // let mut modules = HashMap::<&str, Vec<_>>::new();

    let root = module.get_region(module.root).unwrap();

    for node_id in root.children {
        let node = module.get_node(*node_id).unwrap();

        let (sort, symbol) = match node.operation {
            Operation::DeclareConstructor(symbol) => {
                let mut sort = Sort::Constructor;

                if let Some(sig) = module.view::<NamedConstructor>(symbol.signature) {
                    sort = match sig.name {
                        CORE_META => Sort::Metadata,
                        CORE_STATIC => Sort::StaticType,
                        CORE_TYPE => Sort::RuntimeType,
                        CORE_CONSTRAINT => Sort::Constraint,
                        _ => Sort::Constructor,
                    };
                }

                (sort, symbol)
            }
            Operation::DeclareOperation(symbol) => (Sort::Operation, symbol),
            _ => continue,
        };

        let (symbol_ext, symbol_name) = symbol.name.rsplit_once(".").unwrap();

        let mut out = String::new();
        let _ = writeln!(&mut out, "## {} - `{}`", sort, symbol.name);

        if let Some(docs) = find_node_docs(module, *node_id) {
            let _ = writeln!(&mut out, "\n{}", docs);
        }

        let _ = writeln!(&mut out, "");
        for param in symbol.params {
            // TODO: Write the type of the parameter
            let _ = writeln!(&mut out, " - *Parameter* `{}`", param.name);
        }

        // TODO: Constraint

        // for constraint in symbol.constraints {
        //     let _ = writeln!(&mut out, " - *Constraint* `{}`", constraint.name);
        // }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sort {
    Operation,
    RuntimeType,
    StaticType,
    Metadata,
    Constraint,
    Constructor,
}

impl Display for Sort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sort::Operation => write!(f, "Operation"),
            Sort::RuntimeType => write!(f, "RuntimeType"),
            Sort::StaticType => write!(f, "StaticType"),
            Sort::Metadata => write!(f, "Metadata"),
            Sort::Constraint => write!(f, "Constraint"),
            Sort::Constructor => write!(f, "Constructor"),
        }
    }
}
