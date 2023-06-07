use super::{
    build_traits::HugrBuilder,
    dataflow::{DFGBuilder, FunctionBuilder},
    BuildError, Container, HugrMutRef,
};

use crate::{
    hugr::{view::HugrView, ValidationError},
    types::SimpleType,
};

use crate::ops::handle::{AliasID, FuncID, NodeHandle};
use crate::ops::{ModuleOp, OpType};

use crate::types::Signature;

use crate::Node;
use smol_str::SmolStr;

use crate::{hugr::HugrMut, Hugr};

/// Builder for a HUGR module.
pub struct ModuleBuilder<T>(pub(super) T);

impl<T: HugrMutRef> Container for ModuleBuilder<T> {
    #[inline]
    fn container_node(&self) -> Node {
        self.0.as_ref().root()
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        self.0.as_mut()
    }

    fn hugr(&self) -> &Hugr {
        self.0.as_ref().hugr()
    }
}

impl ModuleBuilder<HugrMut> {
    /// Begin building a new module.
    #[must_use]
    pub fn new() -> Self {
        Self(HugrMut::new_module())
    }
}

impl Default for ModuleBuilder<HugrMut> {
    fn default() -> Self {
        Self::new()
    }
}

impl HugrBuilder for ModuleBuilder<HugrMut> {
    fn finish_hugr(self) -> Result<Hugr, ValidationError> {
        self.0.finish()
    }
}

impl<T: HugrMutRef> ModuleBuilder<T> {
    /// Generate a builder for defining a function body graph.
    ///
    /// Replaces a [`ModuleOp::Declare`] node as specified by `f_id`
    /// with a [`ModuleOp::Def`] node.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the node.
    pub fn define_function(
        &mut self,
        f_id: &FuncID<false>,
    ) -> Result<FunctionBuilder<&mut HugrMut>, BuildError> {
        let f_node = f_id.node();
        let signature = match self.hugr().get_optype(f_node) {
            OpType::Module(ModuleOp::Declare { signature }) => signature.clone(),
            _ => {
                return Err(BuildError::UnexpectedType {
                    node: f_node,
                    op_desc: "ModuleOp::Declare",
                })
            }
        };
        self.base().replace_op(
            f_node,
            OpType::Module(ModuleOp::Def {
                signature: signature.clone(),
            }),
        );

        let db = DFGBuilder::create_with_io(self.base(), f_node, signature)?;
        Ok(FunctionBuilder::from_dfg_builder(db))
    }

    /// Add a [`ModuleOp::Def`] node and returns a builder to define the function
    /// body graph.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`ModuleOp::Def`] node.
    pub fn declare_and_def(
        &mut self,
        name: impl Into<String>,
        signature: Signature,
    ) -> Result<FunctionBuilder<&mut HugrMut>, BuildError> {
        let fid = self.declare(name, signature)?;
        self.define_function(&fid)
    }

    /// Declare a function with `signature` and return a handle to the declaration.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`ModuleOp::Declare`] node.
    pub fn declare(
        &mut self,
        _name: impl Into<String>,
        signature: Signature,
    ) -> Result<FuncID<false>, BuildError> {
        // TODO add name and param names to metadata
        let declare_n = self.add_child_op(ModuleOp::Declare { signature })?;

        Ok(declare_n.into())
    }

    /// Add a [`ModuleOp::AliasDef`] node and return a handle to the Alias.
    ///
    /// # Errors
    ///
    /// Error in adding [`ModuleOp::AliasDef`] child node.
    pub fn add_alias_def(
        &mut self,
        name: impl Into<SmolStr>,
        typ: SimpleType,
    ) -> Result<AliasID<true>, BuildError> {
        let name: SmolStr = name.into();
        let linear = typ.is_linear();
        let node = self.add_child_op(ModuleOp::AliasDef {
            name: name.clone(),
            definition: typ,
        })?;

        Ok(AliasID::new(node, name, linear))
    }

    /// Add a [`ModuleOp::AliasDeclare`] node and return a handle to the Alias.
    /// # Errors
    ///
    /// Error in adding [`ModuleOp::AliasDeclare`] child node.
    pub fn add_alias_declare(
        &mut self,
        name: impl Into<SmolStr>,
        linear: bool,
    ) -> Result<AliasID<false>, BuildError> {
        let name: SmolStr = name.into();
        let node = self.add_child_op(ModuleOp::AliasDeclare {
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

            let mut f_build = module_builder.define_function(&f_id)?;
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

            let f_build = module_builder.declare_and_def(
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
}
