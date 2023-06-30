use super::{
    build_traits::HugrBuilder,
    dataflow::{DFGBuilder, FunctionBuilder},
    BuildError, Container,
};

use crate::{
    hugr::{view::HugrView, ValidationError},
    ops,
    types::SimpleType,
};

use crate::ops::handle::{AliasID, FuncID, NodeHandle};
use crate::ops::OpType;

use crate::types::Signature;

use crate::Node;
use smol_str::SmolStr;

use crate::{hugr::HugrMut, Hugr};

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
    fn finish_hugr(self) -> Result<Hugr, ValidationError> {
        self.0.validate()?;
        Ok(self.0)
    }
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> ModuleBuilder<T> {
    /// Replace a [`ops::FuncDeclare`] with [`ops::FuncDef`] and return a builder for
    /// the defining graph.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`OpType::FuncDef`] node.
    pub fn define_declaration(
        &mut self,
        f_id: &FuncID<false>,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        let f_node = f_id.node();
        let (signature, name) = if let OpType::FuncDeclare(ops::FuncDeclare { signature, name }) =
            self.hugr().get_optype(f_node)
        {
            (signature.clone(), name.clone())
        } else {
            return Err(BuildError::UnexpectedType {
                node: f_node,
                op_desc: "OpType::FuncDeclare",
            });
        };
        self.hugr_mut().replace_op(
            f_node,
            ops::FuncDef {
                name,
                signature: signature.clone(),
            },
        );

        let db = DFGBuilder::create_with_io(self.hugr_mut(), f_node, signature)?;
        Ok(FunctionBuilder::from_dfg_builder(db))
    }

    /// Declare a function with `signature` and return a handle to the declaration.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`OpType::FuncDeclare`] node.
    pub fn declare(
        &mut self,
        name: impl Into<String>,
        signature: Signature,
    ) -> Result<FuncID<false>, BuildError> {
        // TODO add param names to metadata
        let declare_n = self.add_child_op(ops::FuncDeclare {
            signature,
            name: name.into(),
        })?;

        Ok(declare_n.into())
    }

    /// Add a [`OpType::AliasDef`] node and return a handle to the Alias.
    ///
    /// # Errors
    ///
    /// Error in adding [`OpType::AliasDef`] child node.
    pub fn add_alias_def(
        &mut self,
        name: impl Into<SmolStr>,
        typ: SimpleType,
    ) -> Result<AliasID<true>, BuildError> {
        // TODO: add AliasDef in other containers
        // This is currently tricky as they are not connected to anything so do
        // not appear in topological traversals.
        // Could be fixed by removing single-entry requirement and sorting from
        // every 0-input node.
        let name: SmolStr = name.into();
        let linear = typ.is_linear();
        let node = self.add_child_op(ops::AliasDef {
            name: name.clone(),
            definition: typ,
        })?;

        Ok(AliasID::new(node, name, linear))
    }

    /// Add a [`OpType::AliasDeclare`] node and return a handle to the Alias.
    /// # Errors
    ///
    /// Error in adding [`OpType::AliasDeclare`] child node.
    pub fn add_alias_declare(
        &mut self,
        name: impl Into<SmolStr>,
        linear: bool,
    ) -> Result<AliasID<false>, BuildError> {
        let name: SmolStr = name.into();
        let node = self.add_child_op(ops::AliasDeclare {
            name: name.clone(),
            linear,
        })?;

        Ok(AliasID::new(node, name, linear))
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::{
        builder::{
            test::{n_identity, NAT},
            Dataflow, DataflowSubContainer,
        },
        type_row,
    };

    use super::*;
    #[test]
    fn basic_recurse() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let f_id = module_builder
                .declare("main", Signature::new_df(type_row![NAT], type_row![NAT]))?;

            let mut f_build = module_builder.define_declaration(&f_id)?;
            let call = f_build.call(&f_id, f_build.input_wires())?;

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

            let qubit_state_type = module_builder.add_alias_declare("qubit_state", true)?;

            let f_build = module_builder.define_function(
                "main",
                Signature::new_df(
                    vec![qubit_state_type.get_alias_type()],
                    vec![qubit_state_type.get_alias_type()],
                ),
            )?;
            n_identity(f_build)?;
            module_builder.finish_hugr()
        };
        assert_matches!(build_result, Ok(_));
        Ok(())
    }

    #[test]
    fn local_def() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let mut f_build = module_builder
                .define_function("main", Signature::new_df(type_row![NAT], type_row![NAT]))?;
            let local_build = f_build
                .define_function("local", Signature::new_df(type_row![NAT], type_row![NAT]))?;
            let [wire] = local_build.input_wires_arr();
            let f_id = local_build.finish_with_outputs([wire])?;

            let call = f_build.call(f_id.handle(), f_build.input_wires())?;

            f_build.finish_with_outputs(call.outputs())?;
            module_builder.finish_hugr()
        };
        assert_matches!(build_result, Ok(_));
        Ok(())
    }
}
