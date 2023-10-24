use super::{
    build_traits::SubContainer,
    dataflow::{DFGBuilder, DFGWrapper},
    handle::BuildHandle,
    BasicBlockID, BuildError, CfgID, Container, Dataflow, HugrBuilder, Wire,
};

use crate::ops::{self, BasicBlock, OpType};
use crate::{
    extension::{ExtensionRegistry, ExtensionSet},
    types::FunctionType,
};
use crate::{hugr::views::HugrView, types::TypeRow};
use crate::{ops::handle::NodeHandle, types::Type};

use crate::Node;
use crate::{
    hugr::{HugrMut, NodeType},
    type_row, Hugr,
};

/// Builder for a [`crate::ops::CFG`] child control
/// flow graph
#[derive(Debug, PartialEq)]
pub struct CFGBuilder<T> {
    pub(super) base: T,
    pub(super) cfg_node: Node,
    pub(super) inputs: Option<TypeRow>,
    pub(super) exit_node: Node,
    pub(super) n_out_wires: usize,
}

impl<B: AsMut<Hugr> + AsRef<Hugr>> Container for CFGBuilder<B> {
    #[inline]
    fn container_node(&self) -> Node {
        self.cfg_node
    }

    #[inline]
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.base.as_mut()
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.base.as_ref()
    }
}

impl<H: AsMut<Hugr> + AsRef<Hugr>> SubContainer for CFGBuilder<H> {
    type ContainerHandle = BuildHandle<CfgID>;
    #[inline]
    fn finish_sub_container(self) -> Result<Self::ContainerHandle, BuildError> {
        Ok((self.cfg_node, self.n_out_wires).into())
    }
}

impl CFGBuilder<Hugr> {
    /// New CFG rooted HUGR builder
    pub fn new(signature: FunctionType) -> Result<Self, BuildError> {
        let cfg_op = ops::CFG {
            signature: signature.clone(),
        };

        let base = Hugr::new(NodeType::open_extensions(cfg_op));
        let cfg_node = base.root();
        CFGBuilder::create(base, cfg_node, signature.input, signature.output)
    }
}

impl HugrBuilder for CFGBuilder<Hugr> {
    fn finish_hugr(
        mut self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Hugr, crate::hugr::ValidationError> {
        self.base.update_validate(extension_registry)?;
        Ok(self.base)
    }
}

impl<B: AsMut<Hugr> + AsRef<Hugr>> CFGBuilder<B> {
    pub(super) fn create(
        mut base: B,
        cfg_node: Node,
        input: TypeRow,
        output: TypeRow,
    ) -> Result<Self, BuildError> {
        let n_out_wires = output.len();
        let exit_block_type = OpType::BasicBlock(BasicBlock::Exit {
            cfg_outputs: output,
        });
        let exit_node = base
            .as_mut()
            // Make the extensions a parameter
            .add_op_with_parent(cfg_node, exit_block_type)?;
        Ok(Self {
            base,
            cfg_node,
            n_out_wires,
            exit_node,
            inputs: Some(input),
        })
    }

    /// Return a builder for a non-entry [`BasicBlock::DFB`] child graph with `inputs`
    /// and `outputs` and the variants of the branching TupleSum value
    /// specified by `tuple_sum_rows`.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the node.
    pub fn block_builder(
        &mut self,
        inputs: TypeRow,
        tuple_sum_rows: impl IntoIterator<Item = TypeRow>,
        extension_delta: ExtensionSet,
        other_outputs: TypeRow,
    ) -> Result<BlockBuilder<&mut Hugr>, BuildError> {
        self.any_block_builder(
            inputs,
            tuple_sum_rows,
            other_outputs,
            extension_delta,
            false,
        )
    }

    fn any_block_builder(
        &mut self,
        inputs: TypeRow,
        tuple_sum_rows: impl IntoIterator<Item = TypeRow>,
        other_outputs: TypeRow,
        extension_delta: ExtensionSet,
        entry: bool,
    ) -> Result<BlockBuilder<&mut Hugr>, BuildError> {
        let tuple_sum_rows: Vec<_> = tuple_sum_rows.into_iter().collect();
        let op = OpType::BasicBlock(BasicBlock::DFB {
            inputs: inputs.clone(),
            other_outputs: other_outputs.clone(),
            tuple_sum_rows: tuple_sum_rows.clone(),
            extension_delta,
        });
        let parent = self.container_node();
        let block_n = if entry {
            let exit = self.exit_node;
            // TODO: Make extensions a parameter
            self.hugr_mut().add_op_before(exit, op)
        } else {
            // TODO: Make extensions a parameter
            self.hugr_mut().add_op_with_parent(parent, op)
        }?;

        BlockBuilder::create(
            self.hugr_mut(),
            block_n,
            tuple_sum_rows,
            other_outputs,
            inputs,
        )
    }

    /// Return a builder for a non-entry [`BasicBlock::DFB`] child graph with `inputs`
    /// and `outputs` and a UnitSum type: a Sum of `n_cases` unit types.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the node.
    pub fn simple_block_builder(
        &mut self,
        signature: FunctionType,
        n_cases: usize,
    ) -> Result<BlockBuilder<&mut Hugr>, BuildError> {
        self.block_builder(
            signature.input,
            vec![type_row![]; n_cases],
            signature.extension_reqs,
            signature.output,
        )
    }

    /// Return a builder for the entry [`BasicBlock::DFB`] child graph with `inputs`
    /// and `outputs` and the variants of the branching TupleSum value
    /// specified by `tuple_sum_rows`.
    ///
    /// # Errors
    ///
    /// This function will return an error if an entry block has already been built.
    pub fn entry_builder(
        &mut self,
        tuple_sum_rows: impl IntoIterator<Item = TypeRow>,
        other_outputs: TypeRow,
        extension_delta: ExtensionSet,
    ) -> Result<BlockBuilder<&mut Hugr>, BuildError> {
        let inputs = self
            .inputs
            .take()
            .ok_or(BuildError::EntryBuiltError(self.cfg_node))?;
        self.any_block_builder(inputs, tuple_sum_rows, other_outputs, extension_delta, true)
    }

    /// Return a builder for the entry [`BasicBlock::DFB`] child graph with `inputs`
    /// and `outputs` and a UnitSum type: a Sum of `n_cases` unit types.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the node.
    pub fn simple_entry_builder(
        &mut self,
        outputs: TypeRow,
        n_cases: usize,
        extension_delta: ExtensionSet,
    ) -> Result<BlockBuilder<&mut Hugr>, BuildError> {
        self.entry_builder(vec![type_row![]; n_cases], outputs, extension_delta)
    }

    /// Returns the exit block of this [`CFGBuilder`].
    pub fn exit_block(&self) -> BasicBlockID {
        self.exit_node.into()
    }

    /// Set the `branch` index `successor` block of `predecessor`.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error connecting the blocks.
    pub fn branch(
        &mut self,
        predecessor: &BasicBlockID,
        branch: usize,
        successor: &BasicBlockID,
    ) -> Result<(), BuildError> {
        let from = predecessor.node();
        let to = successor.node();
        Ok(self.hugr_mut().connect(from, branch, to, 0)?)
    }
}

/// Builder for a [`BasicBlock::DFB`] child graph.
pub type BlockBuilder<B> = DFGWrapper<B, BasicBlockID>;

impl<B: AsMut<Hugr> + AsRef<Hugr>> BlockBuilder<B> {
    /// Set the outputs of the block, with `branch_wire` carrying  the value of the
    /// branch controlling TupleSum value.  `outputs` are the remaining outputs.
    pub fn set_outputs(
        &mut self,
        branch_wire: Wire,
        outputs: impl IntoIterator<Item = Wire>,
    ) -> Result<(), BuildError> {
        Dataflow::set_outputs(self, [branch_wire].into_iter().chain(outputs))
    }
    fn create(
        base: B,
        block_n: Node,
        tuple_sum_rows: impl IntoIterator<Item = TypeRow>,
        other_outputs: TypeRow,
        inputs: TypeRow,
    ) -> Result<Self, BuildError> {
        // The node outputs a TupleSum before the data outputs of the block node
        let tuple_sum_type = Type::new_tuple_sum(tuple_sum_rows);
        let mut node_outputs = vec![tuple_sum_type];
        node_outputs.extend_from_slice(&other_outputs);
        let signature = FunctionType::new(inputs, TypeRow::from(node_outputs));
        let inp_ex = base
            .as_ref()
            .get_nodetype(block_n)
            .input_extensions()
            .cloned();
        let db = DFGBuilder::create_with_io(base, block_n, signature, inp_ex)?;
        Ok(BlockBuilder::from_dfg_builder(db))
    }

    /// [Set outputs](BlockBuilder::set_outputs) and [finish](`BlockBuilder::finish_sub_container`).
    pub fn finish_with_outputs(
        mut self,
        branch_wire: Wire,
        outputs: impl IntoIterator<Item = Wire>,
    ) -> Result<<Self as SubContainer>::ContainerHandle, BuildError>
    where
        Self: Sized,
    {
        self.set_outputs(branch_wire, outputs)?;
        self.finish_sub_container()
    }
}

impl BlockBuilder<Hugr> {
    /// Initialize a [`BasicBlock::DFB`] rooted HUGR builder
    pub fn new(
        inputs: impl Into<TypeRow>,
        input_extensions: impl Into<Option<ExtensionSet>>,
        tuple_sum_rows: impl IntoIterator<Item = TypeRow>,
        other_outputs: impl Into<TypeRow>,
        extension_delta: ExtensionSet,
    ) -> Result<Self, BuildError> {
        let inputs = inputs.into();
        let tuple_sum_rows: Vec<_> = tuple_sum_rows.into_iter().collect();
        let other_outputs = other_outputs.into();
        let op = BasicBlock::DFB {
            inputs: inputs.clone(),
            other_outputs: other_outputs.clone(),
            tuple_sum_rows: tuple_sum_rows.clone(),
            extension_delta,
        };

        let base = Hugr::new(NodeType::new(op, input_extensions));
        let root = base.root();
        Self::create(base, root, tuple_sum_rows, other_outputs, inputs)
    }

    /// [Set outputs](BlockBuilder::set_outputs) and [finish_hugr](`BlockBuilder::finish_hugr`).
    pub fn finish_hugr_with_outputs(
        mut self,
        branch_wire: Wire,
        outputs: impl IntoIterator<Item = Wire>,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Hugr, BuildError> {
        self.set_outputs(branch_wire, outputs)?;
        self.finish_hugr(extension_registry)
            .map_err(BuildError::InvalidHUGR)
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    #[test]
    fn basic_module_cfg() {
        assert_matches!(hugr_utils::examples::basic_module_cfg(), Ok(_));
    }
    #[test]
    fn basic_cfg_hugr() {
        assert_matches!(hugr_utils::examples::basic_cfg(), Ok(_));
    }
}
