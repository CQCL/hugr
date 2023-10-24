use super::{
    build_traits::HugrBuilder,
    dataflow::{DFGBuilder, FunctionBuilder},
    BuildError, Container,
};

use crate::{
    extension::ExtensionRegistry,
    hugr::{hugrmut::sealed::HugrMutInternals, views::HugrView, ValidationError},
    ops,
    types::{Type, TypeBound},
};

use crate::ops::handle::{AliasID, FuncID, NodeHandle};
use crate::ops::OpType;

use crate::types::Signature;

use crate::Node;
use smol_str::SmolStr;

use crate::{hugr::NodeType, Hugr};

/// Builder for a HUGR module.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleBuilder<T>(pub(super) T);

impl<T: AsMut<Hugr> + AsRef<Hugr>> Container for ModuleBuilder<T> {
    #[inline]
    fn container_node(&self) -> Node {
        self.0.as_ref().root()
    }

    #[inline]
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.0.as_mut()
    }

    fn hugr(&self) -> &Hugr {
        self.0.as_ref()
    }
}

impl ModuleBuilder<Hugr> {
    /// Begin building a new module.
    #[must_use]
    pub fn new() -> Self {
        Self(Default::default())
    }
}

impl Default for ModuleBuilder<Hugr> {
    fn default() -> Self {
        Self::new()
    }
}

impl HugrBuilder for ModuleBuilder<Hugr> {
    fn finish_hugr(
        mut self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Hugr, ValidationError> {
        self.0.update_validate(extension_registry)?;
        Ok(self.0)
    }
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> ModuleBuilder<T> {
    /// Replace a [`ops::FuncDecl`] with [`ops::FuncDefn`] and return a builder for
    /// the defining graph.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`OpType::FuncDefn`] node.
    pub fn define_declaration(
        &mut self,
        f_id: &FuncID<false>,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        let f_node = f_id.node();
        let (signature, name) = if let OpType::FuncDecl(ops::FuncDecl { signature, name }) =
            self.hugr().get_optype(f_node)
        {
            (signature.clone(), name.clone())
        } else {
            return Err(BuildError::UnexpectedType {
                node: f_node,
                op_desc: "OpType::FuncDecl",
            });
        };
        self.hugr_mut().replace_op(
            f_node,
            NodeType::pure(ops::FuncDefn {
                name,
                signature: signature.clone(),
            }),
        )?;

        let db = DFGBuilder::create_with_io(self.hugr_mut(), f_node, signature, None)?;
        Ok(FunctionBuilder::from_dfg_builder(db))
    }

    /// Declare a function with `signature` and return a handle to the declaration.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`OpType::FuncDecl`] node.
    pub fn declare(
        &mut self,
        name: impl Into<String>,
        signature: Signature,
    ) -> Result<FuncID<false>, BuildError> {
        // TODO add param names to metadata
        let rs = signature.input_extensions.clone();
        let declare_n = self.add_child_node(NodeType::new(
            ops::FuncDecl {
                signature: signature.into(),
                name: name.into(),
            },
            rs,
        ))?;

        Ok(declare_n.into())
    }

    /// Add a [`OpType::AliasDefn`] node and return a handle to the Alias.
    ///
    /// # Errors
    ///
    /// Error in adding [`OpType::AliasDefn`] child node.
    pub fn add_alias_def(
        &mut self,
        name: impl Into<SmolStr>,
        typ: Type,
    ) -> Result<AliasID<true>, BuildError> {
        // TODO: add AliasDefn in other containers
        // This is currently tricky as they are not connected to anything so do
        // not appear in topological traversals.
        // Could be fixed by removing single-entry requirement and sorting from
        // every 0-input node.
        let name: SmolStr = name.into();
        let bound = typ.least_upper_bound();
        let node = self.add_child_op(ops::AliasDefn {
            name: name.clone(),
            definition: typ,
        })?;

        Ok(AliasID::new(node, name, bound))
    }

    /// Add a [`OpType::AliasDecl`] node and return a handle to the Alias.
    /// # Errors
    ///
    /// Error in adding [`OpType::AliasDecl`] child node.
    pub fn add_alias_declare(
        &mut self,
        name: impl Into<SmolStr>,
        bound: TypeBound,
    ) -> Result<AliasID<false>, BuildError> {
        let name: SmolStr = name.into();
        let node = self.add_child_op(ops::AliasDecl {
            name: name.clone(),
            bound,
        })?;

        Ok(AliasID::new(node, name, bound))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cool_asserts::assert_matches;
    #[test]
    fn basic_recurse() -> Result<(), BuildError> {
        assert_matches!(hugr_utils::examples::basic_recurse(), Ok(_));
        Ok(())
    }

    #[test]
    fn simple_alias() -> Result<(), BuildError> {
        assert_matches!(hugr_utils::examples::simple_alias(), Ok(_));
        Ok(())
    }

    #[test]
    fn local_def() -> Result<(), BuildError> {
        assert_matches!(hugr_utils::examples::local_def(), Ok(_));
        Ok(())
    }
}
