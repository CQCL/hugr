use super::{
    BuildError, Container,
    build_traits::HugrBuilder,
    dataflow::{DFGBuilder, FunctionBuilder},
};

use crate::hugr::ValidationError;
use crate::hugr::internal::HugrMutInternals;
use crate::hugr::views::HugrView;
use crate::ops;
use crate::ops::handle::{AliasID, FuncID, NodeHandle};
use crate::types::{PolyFuncType, Type, TypeBound};
use crate::{Hugr, Node};

use smol_str::SmolStr;

/// Builder for a HUGR module.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ModuleBuilder<T>(pub(super) T);

impl<T: AsMut<Hugr> + AsRef<Hugr>> Container for ModuleBuilder<T> {
    #[inline]
    fn container_node(&self) -> Node {
        self.0.as_ref().module_root()
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
        Self::default()
    }
}

impl HugrBuilder for ModuleBuilder<Hugr> {
    fn finish_hugr(self) -> Result<Hugr, ValidationError<Node>> {
        self.0.validate()?;
        Ok(self.0)
    }
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> ModuleBuilder<T> {
    /// Replace a [`ops::FuncDecl`] with public [`ops::FuncDefn`] and return a builder for
    /// the defining graph.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`crate::ops::OpType::FuncDefn`] node.
    pub fn define_declaration(
        &mut self,
        f_id: &FuncID<false>,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        let f_node = f_id.node();
        let opty = self.hugr_mut().optype_mut(f_node);
        let ops::OpType::FuncDecl(decl) = opty else {
            return Err(BuildError::UnexpectedType {
                node: f_node,
                op_desc: "crate::ops::OpType::FuncDecl",
            });
        };

        let body = decl.signature().body().clone();
        // TODO look for `name_hint` metadata on the FuncDecl and copy to FuncDefn
        *opty = ops::FuncDefn::new_link_name(
            format!("From Decl {}", decl.link_name()),
            decl.signature().clone(),
            decl.link_name().clone(),
        )
        .into();

        let db = DFGBuilder::create_with_io(self.hugr_mut(), f_node, body)?;
        Ok(FunctionBuilder::from_dfg_builder(db))
    }

    /// Add a [`ops::FuncDefn`] node, with both `name` and `link_name` explicitly specified.
    /// Returns a builder to define the function body graph.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`ops::FuncDefn`] node.
    pub fn define_function_link_name(
        &mut self,
        name: impl Into<String>,
        signature: impl Into<PolyFuncType>,
        link_name: impl Into<Option<String>>,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        let signature: PolyFuncType = signature.into();
        let body = signature.body().clone();
        let f_node = self.add_child_node(ops::FuncDefn::new_link_name(name, signature, link_name));

        // Add the extensions used by the function types.
        self.use_extensions(
            body.used_extensions().unwrap_or_else(|e| {
                panic!("Build-time signatures should have valid extensions. {e}")
            }),
        );

        let db = DFGBuilder::create_with_io(self.hugr_mut(), f_node, body)?;
        Ok(FunctionBuilder::from_dfg_builder(db))
    }

    /// Declare a function with `signature` and return a handle to the declaration.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`crate::ops::OpType::FuncDecl`] node.
    pub fn declare(
        &mut self,
        name: impl Into<String>,
        signature: PolyFuncType,
    ) -> Result<FuncID<false>, BuildError> {
        let body = signature.body().clone();
        // TODO add param names to metadata
        let declare_n = self.add_child_node(ops::FuncDecl::new(name, signature));

        // Add the extensions used by the function types.
        self.use_extensions(
            body.used_extensions().unwrap_or_else(|e| {
                panic!("Build-time signatures should have valid extensions. {e}")
            }),
        );

        Ok(declare_n.into())
    }

    /// Adds a private [`ops::FuncDefn`] node and returns a builder to define the function
    /// body graph.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`ops::FuncDefn`] node.
    // ALAN TODO? deprecate and rename to define_private_func(tion)? (Rather long...)
    pub fn define_function(
        &mut self,
        name: impl Into<String>,
        signature: impl Into<PolyFuncType>,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        self.define_function_link_name(name, signature, None)
    }

    /// Adds a public [`ops::FuncDefn`] node, with `link_name` the same as `name`,
    /// and returns a builder to define the function body graph.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`ops::FuncDefn`] node.
    pub fn define_function_pub(
        &mut self,
        name: impl Into<String>,
        signature: impl Into<PolyFuncType>,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        let name = name.into();
        self.define_function_link_name(name.clone(), signature, name)
    }

    /// Add a [`crate::ops::OpType::AliasDefn`] node and return a handle to the Alias.
    ///
    /// # Errors
    ///
    /// Error in adding [`crate::ops::OpType::AliasDefn`] child node.
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
        let node = self.add_child_node(ops::AliasDefn {
            name: name.clone(),
            definition: typ,
        });

        Ok(AliasID::new(node, name, bound))
    }

    /// Add a [`crate::ops::OpType::AliasDecl`] node and return a handle to the Alias.
    /// # Errors
    ///
    /// Error in adding [`crate::ops::OpType::AliasDecl`] child node.
    pub fn add_alias_declare(
        &mut self,
        name: impl Into<SmolStr>,
        bound: TypeBound,
    ) -> Result<AliasID<false>, BuildError> {
        let name: SmolStr = name.into();
        let node = self.add_child_node(ops::AliasDecl {
            name: name.clone(),
            bound,
        });

        Ok(AliasID::new(node, name, bound))
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::extension::prelude::usize_t;
    use crate::{
        builder::{Dataflow, DataflowSubContainer, test::n_identity},
        types::Signature,
    };

    use super::*;
    #[test]
    fn basic_recurse() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let f_id = module_builder.declare(
                "main",
                Signature::new(vec![usize_t()], vec![usize_t()]).into(),
            )?;

            let mut f_build = module_builder.define_declaration(&f_id)?;
            let call = f_build.call(&f_id, &[], f_build.input_wires())?;

            f_build.finish_with_outputs(call.outputs())?;
            module_builder.finish_hugr()
        };
        assert_matches!(build_result, Ok(_));
        Ok(())
    }

    #[test]
    fn simple_alias() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let qubit_state_type =
                module_builder.add_alias_declare("qubit_state", TypeBound::Any)?;

            let f_build = module_builder.define_function(
                "main",
                Signature::new_endo(qubit_state_type.get_alias_type()),
            )?;
            n_identity(f_build)?;
            module_builder.finish_hugr()
        };
        assert_matches!(build_result, Ok(_));
        Ok(())
    }
}
