use super::{
    dataflow::{DFGBuilder, FunctionBuilder},
    BuildError, Container,
};

use crate::types::SimpleType;

use crate::ops::handle::{ConstID, FuncID, NewTypeID, NodeHandle};
use crate::ops::{ConstValue, ModuleOp, OpType};

use crate::types::Signature;

use portgraph::NodeIndex;
use smol_str::SmolStr;

use crate::{hugr::HugrMut, Hugr};

#[derive(Default)]
/// Builder for a HUGR module.
/// Top level builder which can generate sub-builders.
/// Validates and returns the HUGR on `finish`.
pub struct ModuleBuilder(HugrMut);

impl ModuleBuilder {
    /// New builder for a new HUGR.
    pub fn new() -> Self {
        Self(HugrMut::new())
    }
}

impl Container for ModuleBuilder {
    type ContainerHandle = Result<Hugr, BuildError>;

    #[inline]
    fn container_node(&self) -> NodeIndex {
        self.0.root()
    }

    #[inline]
    fn base(&mut self) -> &mut HugrMut {
        &mut self.0
    }

    #[inline]
    fn finish(self) -> Self::ContainerHandle {
        Ok(self.0.finish()?)
    }

    fn hugr(&self) -> &Hugr {
        self.0.hugr()
    }
}

impl ModuleBuilder {
    /// Generate a builder for defining a function body graph.
    ///
    /// Replaces a [`ModuleOp::Declare`] node as specified by `f_id`
    /// with a [`ModuleOp::Def`] node.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the node.
    pub fn define_function<'a: 'b, 'b>(
        &'a mut self,
        f_id: &FuncID,
    ) -> Result<FunctionBuilder<'b>, BuildError> {
        let f_node = f_id.node();
        let (inputs, outputs) = if let OpType::Module(ModuleOp::Declare { signature }) =
            self.hugr().get_optype(f_node)
        {
            (signature.input.clone(), signature.output.clone())
        } else {
            return Err(BuildError::UnexpectedType {
                node: f_node,
                op_desc: "ModuleOp::Declare",
            });
        };
        self.base().replace_op(
            f_node,
            OpType::Module(ModuleOp::Def {
                signature: Signature::new(inputs.clone(), outputs.clone(), None),
            }),
        );

        let db = DFGBuilder::create_with_io(self.base(), f_node, inputs, outputs)?;
        Ok(FunctionBuilder::new(db))
    }

    /// Add a [`ModuleOp::Def`] node and returns a builder to define the function
    /// body graph.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`ModuleOp::Def`] node.
    pub fn declare_and_def<'a: 'b, 'b>(
        &'a mut self,
        _name: impl Into<String>,
        signature: Signature,
    ) -> Result<FunctionBuilder<'b>, BuildError> {
        let fid = self.declare(_name, signature)?;
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
    ) -> Result<FuncID, BuildError> {
        // TODO add name and param names to metadata
        let declare_n = self.add_child_op(ModuleOp::Declare { signature })?;

        Ok(declare_n.into())
    }

    /// Add a constant value to the module and return a handle to it.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`ModuleOp::Const`] node.
    pub fn constant(&mut self, val: ConstValue) -> Result<ConstID, BuildError> {
        let typ = val.const_type();
        let const_n = self.add_child_op(ModuleOp::Const(val))?;

        Ok((const_n, typ).into())
    }

    /// Add a [`ModuleOp::NewType`] node and return a handle to the NewType.
    pub fn add_new_type(
        &mut self,
        name: impl Into<SmolStr>,
        typ: SimpleType,
    ) -> Result<NewTypeID, BuildError> {
        let name: SmolStr = name.into();

        let node = self.add_child_op(ModuleOp::NewType {
            name: name.clone(),
            definition: typ.clone(),
        })?;

        Ok((node, name, typ).into())
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use crate::{
        builder::{
            test::{BIT, NAT, QB},
            Dataflow,
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
            module_builder.finish()
        };
        assert_matches!(build_result, Ok(_));
        Ok(())
    }

    #[test]
    fn simple_newtype() -> Result<(), BuildError> {
        let inputs = type_row![QB, BIT];
        let build_result = {
            let mut module_builder = ModuleBuilder::new();

            let qubit_state_type = module_builder
                .add_new_type("qubit_state", SimpleType::new_tuple(inputs.clone()))?;

            let mut f_build = module_builder.declare_and_def(
                "main",
                Signature::new_df(inputs, vec![qubit_state_type.get_new_type()]),
            )?;
            {
                let tuple = f_build.make_tuple(f_build.input_wires())?;
                let q_s_val = f_build.make_new_type(&qubit_state_type, tuple)?;
                f_build.finish_with_outputs([q_s_val])?;
            }

            module_builder.finish()
        };
        assert_matches!(build_result, Ok(_));
        Ok(())
    }
}
