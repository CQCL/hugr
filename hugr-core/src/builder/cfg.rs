use super::{
    build_traits::SubContainer,
    dataflow::{DFGBuilder, DFGWrapper},
    handle::BuildHandle,
    BasicBlockID, BuildError, CfgID, Container, Dataflow, HugrBuilder, Wire,
};

use crate::extension::{ExtensionRegistry, ExtensionSet};
use crate::{hugr::views::HugrView, types::TypeRow};
use crate::{
    ops::{self, handle::NodeHandle, DataflowBlock, DataflowParent, ExitBlock, OpType},
    types::FunctionType,
};

use crate::Node;
use crate::{
    hugr::{HugrMut, NodeType},
    type_row, Hugr,
};

/// Builder for a [`crate::ops::CFG`] child control
/// flow graph.
///
/// These builder methods should ensure that the first two children of a CFG
/// node are the entry node and the exit node.
///
/// # Example
/// ```
/// /*  Build a control flow graph with the following structure:
///            +-----------+
///            |   Entry   |
///            +-/-----\---+
///             /       \
///            /         \
///           /           \
///          /             \
///   +-----/----+       +--\-------+
///   | Branch A |       | Branch B |
///   +-----\----+       +----/-----+
///          \               /
///           \             /
///            \           /
///             \         /
///            +-\-------/--+
///            |    Exit    |
///            +------------+
/// */
/// use hugr::{
///     builder::{BuildError, CFGBuilder, Container, Dataflow, HugrBuilder},
///     extension::{prelude, ExtensionSet},
///     ops, type_row,
///     types::{FunctionType, SumType, Type},
///     Hugr,
/// };
///
/// const NAT: Type = prelude::USIZE_T;
///
/// fn make_cfg() -> Result<Hugr, BuildError> {
///     let mut cfg_builder = CFGBuilder::new(FunctionType::new(type_row![NAT], type_row![NAT]))?;
///
///     // Outputs from basic blocks must be packed in a sum which corresponds to
///     // which successor to pick. We'll either choose the first branch and pass
///     // it a NAT, or the second branch and pass it nothing.
///     let sum_variants = vec![type_row![NAT], type_row![]];
///
///     // The second argument says what types will be passed through to every
///     // successor, in addition to the appropriate `sum_variants` type.
///     let mut entry_b =
///         cfg_builder.entry_builder(sum_variants.clone(), type_row![NAT], ExtensionSet::new())?;
///
///     let [inw] = entry_b.input_wires_arr();
///     let entry = {
///         // Pack the const "42" into the appropriate sum type.
///         let left_42 = ops::Value::sum(
///             0,
///             [prelude::ConstUsize::new(42).into()],
///             SumType::new(sum_variants.clone()),
///         )?;
///         let sum = entry_b.add_load_value(left_42);
///
///         entry_b.finish_with_outputs(sum, [inw])?
///     };
///
///     // This block will be the first successor of the entry node. It takes two
///     // `NAT` arguments: one from the `sum_variants` type, and another from the
///     // entry node's `other_outputs`.
///     let mut successor_builder = cfg_builder.simple_block_builder(
///         FunctionType::new(type_row![NAT, NAT], type_row![NAT]),
///         1, // only one successor to this block
///     )?;
///     let successor_a = {
///         // This block has one successor. The choice is denoted by a unary sum.
///         let sum_unary = successor_builder.add_load_const(ops::Value::unary_unit_sum());
///
///         // The input wires of a node start with the data embedded in the variant
///         // which selected this block.
///         let [_forty_two, in_wire] = successor_builder.input_wires_arr();
///         successor_builder.finish_with_outputs(sum_unary, [in_wire])?
///     };
///
///     // The only argument to this block is the entry node's `other_outputs`.
///     let mut successor_builder = cfg_builder
///         .simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?;
///     let successor_b = {
///         let sum_unary = successor_builder.add_load_value(ops::Value::unary_unit_sum());
///         let [in_wire] = successor_builder.input_wires_arr();
///         successor_builder.finish_with_outputs(sum_unary, [in_wire])?
///     };
///     let exit = cfg_builder.exit_block();
///     cfg_builder.branch(&entry, 0, &successor_a)?; // branch 0 goes to successor_a
///     cfg_builder.branch(&entry, 1, &successor_b)?; // branch 1 goes to successor_b
///     cfg_builder.branch(&successor_a, 0, &exit)?;
///     cfg_builder.branch(&successor_b, 0, &exit)?;
///     let hugr = cfg_builder.finish_prelude_hugr()?;
///     Ok(hugr)
/// };
/// #[cfg(not(feature = "extension_inference"))]
/// assert!(make_cfg().is_ok());
/// ```
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

        let base = Hugr::new(NodeType::new_open(cfg_op));
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
        let exit_block_type = OpType::ExitBlock(ExitBlock {
            cfg_outputs: output,
        });
        let exit_node = base
            .as_mut()
            // Make the extensions a parameter
            .add_node_with_parent(cfg_node, exit_block_type);
        Ok(Self {
            base,
            cfg_node,
            n_out_wires,
            exit_node,
            inputs: Some(input),
        })
    }

    /// Return a builder for a non-entry [`DataflowBlock`] child graph with `inputs`
    /// and `outputs` and the variants of the branching Sum value
    /// specified by `sum_rows`.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error adding the node.
    pub fn block_builder(
        &mut self,
        inputs: TypeRow,
        sum_rows: impl IntoIterator<Item = TypeRow>,
        extension_delta: ExtensionSet,
        other_outputs: TypeRow,
    ) -> Result<BlockBuilder<&mut Hugr>, BuildError> {
        self.any_block_builder(inputs, sum_rows, other_outputs, extension_delta, false)
    }

    fn any_block_builder(
        &mut self,
        inputs: TypeRow,
        sum_rows: impl IntoIterator<Item = TypeRow>,
        other_outputs: TypeRow,
        extension_delta: ExtensionSet,
        entry: bool,
    ) -> Result<BlockBuilder<&mut Hugr>, BuildError> {
        let sum_rows: Vec<_> = sum_rows.into_iter().collect();
        let op = OpType::DataflowBlock(DataflowBlock {
            inputs: inputs.clone(),
            other_outputs: other_outputs.clone(),
            sum_rows,
            extension_delta,
        });
        let parent = self.container_node();
        let block_n = if entry {
            let exit = self.exit_node;
            // TODO: Make extensions a parameter
            self.hugr_mut().add_node_before(exit, op)
        } else {
            // TODO: Make extensions a parameter
            self.hugr_mut().add_node_with_parent(parent, op)
        };

        BlockBuilder::create(self.hugr_mut(), block_n)
    }

    /// Return a builder for a non-entry [`DataflowBlock`] child graph with `inputs`
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

    /// Return a builder for the entry [`DataflowBlock`] child graph with `inputs`
    /// and `outputs` and the variants of the branching Sum value
    /// specified by `sum_rows`.
    ///
    /// # Errors
    ///
    /// This function will return an error if an entry block has already been built.
    pub fn entry_builder(
        &mut self,
        sum_rows: impl IntoIterator<Item = TypeRow>,
        other_outputs: TypeRow,
        extension_delta: ExtensionSet,
    ) -> Result<BlockBuilder<&mut Hugr>, BuildError> {
        let inputs = self
            .inputs
            .take()
            .ok_or(BuildError::EntryBuiltError(self.cfg_node))?;
        self.any_block_builder(inputs, sum_rows, other_outputs, extension_delta, true)
    }

    /// Return a builder for the entry [`DataflowBlock`] child graph with `inputs`
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
        self.hugr_mut().connect(from, branch, to, 0);
        Ok(())
    }
}

/// Builder for a [`DataflowBlock`] child graph.
pub type BlockBuilder<B> = DFGWrapper<B, BasicBlockID>;

impl<B: AsMut<Hugr> + AsRef<Hugr>> BlockBuilder<B> {
    /// Set the outputs of the block, with `branch_wire` carrying  the value of the
    /// branch controlling Sum value.  `outputs` are the remaining outputs.
    pub fn set_outputs(
        &mut self,
        branch_wire: Wire,
        outputs: impl IntoIterator<Item = Wire>,
    ) -> Result<(), BuildError> {
        Dataflow::set_outputs(self, [branch_wire].into_iter().chain(outputs))
    }
    fn create(base: B, block_n: Node) -> Result<Self, BuildError> {
        let block_op = base.get_optype(block_n).as_dataflow_block().unwrap();
        let signature = block_op.inner_signature();
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
    /// Initialize a [`DataflowBlock`] rooted HUGR builder
    pub fn new(
        inputs: impl Into<TypeRow>,
        input_extensions: impl Into<Option<ExtensionSet>>,
        sum_rows: impl IntoIterator<Item = TypeRow>,
        other_outputs: impl Into<TypeRow>,
        extension_delta: ExtensionSet,
    ) -> Result<Self, BuildError> {
        let inputs = inputs.into();
        let sum_rows: Vec<_> = sum_rows.into_iter().collect();
        let other_outputs = other_outputs.into();
        let op = DataflowBlock {
            inputs: inputs.clone(),
            other_outputs: other_outputs.clone(),
            sum_rows,
            extension_delta,
        };

        let base = Hugr::new(NodeType::new(op, input_extensions));
        let root = base.root();
        Self::create(base, root)
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
pub(crate) mod test {
    use crate::builder::{DataflowSubContainer, ModuleBuilder};

    use crate::hugr::validate::InterGraphEdgeError;
    use crate::hugr::ValidationError;
    use crate::types::FunctionType;
    use crate::{builder::test::NAT, type_row};
    use cool_asserts::assert_matches;

    use super::*;
    #[test]
    fn basic_module_cfg() -> Result<(), BuildError> {
        let build_result = {
            let mut module_builder = ModuleBuilder::new();
            let mut func_builder = module_builder
                .define_function("main", FunctionType::new(vec![NAT], type_row![NAT]).into())?;
            let _f_id = {
                let [int] = func_builder.input_wires_arr();

                let cfg_id = {
                    let mut cfg_builder = func_builder.cfg_builder(
                        vec![(NAT, int)],
                        None,
                        type_row![NAT],
                        ExtensionSet::new(),
                    )?;
                    build_basic_cfg(&mut cfg_builder)?;

                    cfg_builder.finish_sub_container()?
                };

                func_builder.finish_with_outputs(cfg_id.outputs())?
            };
            module_builder.finish_prelude_hugr()
        };

        assert_eq!(build_result.err(), None);

        Ok(())
    }
    #[test]
    fn basic_cfg_hugr() -> Result<(), BuildError> {
        let mut cfg_builder = CFGBuilder::new(FunctionType::new(type_row![NAT], type_row![NAT]))?;
        build_basic_cfg(&mut cfg_builder)?;
        assert_matches!(cfg_builder.finish_prelude_hugr(), Ok(_));

        Ok(())
    }

    pub(crate) fn build_basic_cfg<T: AsMut<Hugr> + AsRef<Hugr>>(
        cfg_builder: &mut CFGBuilder<T>,
    ) -> Result<(), BuildError> {
        let sum2_variants = vec![type_row![NAT], type_row![NAT]];
        let mut entry_b =
            cfg_builder.entry_builder(sum2_variants.clone(), type_row![], ExtensionSet::new())?;
        let entry = {
            let [inw] = entry_b.input_wires_arr();

            let sum = entry_b.make_sum(1, sum2_variants, [inw])?;
            entry_b.finish_with_outputs(sum, [])?
        };
        let mut middle_b = cfg_builder
            .simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?;
        let middle = {
            let c = middle_b.add_load_const(ops::Value::unary_unit_sum());
            let [inw] = middle_b.input_wires_arr();
            middle_b.finish_with_outputs(c, [inw])?
        };
        let exit = cfg_builder.exit_block();
        cfg_builder.branch(&entry, 0, &middle)?;
        cfg_builder.branch(&middle, 0, &exit)?;
        cfg_builder.branch(&entry, 1, &exit)?;
        Ok(())
    }
    #[test]
    fn test_dom_edge() -> Result<(), BuildError> {
        let mut cfg_builder = CFGBuilder::new(FunctionType::new(type_row![NAT], type_row![NAT]))?;
        let sum_tuple_const = cfg_builder.add_constant(ops::Value::unary_unit_sum());
        let sum_variants = vec![type_row![]];

        let mut entry_b =
            cfg_builder.entry_builder(sum_variants.clone(), type_row![], ExtensionSet::new())?;
        let [inw] = entry_b.input_wires_arr();
        let entry = {
            let sum = entry_b.load_const(&sum_tuple_const);

            entry_b.finish_with_outputs(sum, [])?
        };
        let mut middle_b =
            cfg_builder.simple_block_builder(FunctionType::new(type_row![], type_row![NAT]), 1)?;
        let middle = {
            let c = middle_b.load_const(&sum_tuple_const);
            middle_b.finish_with_outputs(c, [inw])?
        };
        let exit = cfg_builder.exit_block();
        cfg_builder.branch(&entry, 0, &middle)?;
        cfg_builder.branch(&middle, 0, &exit)?;
        assert_matches!(cfg_builder.finish_prelude_hugr(), Ok(_));

        Ok(())
    }

    #[test]
    fn test_non_dom_edge() -> Result<(), BuildError> {
        let mut cfg_builder = CFGBuilder::new(FunctionType::new(type_row![NAT], type_row![NAT]))?;
        let sum_tuple_const = cfg_builder.add_constant(ops::Value::unary_unit_sum());
        let sum_variants = vec![type_row![]];
        let mut middle_b = cfg_builder
            .simple_block_builder(FunctionType::new(type_row![NAT], type_row![NAT]), 1)?;
        let [inw] = middle_b.input_wires_arr();
        let middle = {
            let c = middle_b.load_const(&sum_tuple_const);
            middle_b.finish_with_outputs(c, [inw])?
        };

        let mut entry_b =
            cfg_builder.entry_builder(sum_variants.clone(), type_row![NAT], ExtensionSet::new())?;
        let entry = {
            let sum = entry_b.load_const(&sum_tuple_const);
            // entry block uses wire from middle block even though middle block
            // does not dominate entry
            entry_b.finish_with_outputs(sum, [inw])?
        };
        let exit = cfg_builder.exit_block();
        cfg_builder.branch(&entry, 0, &middle)?;
        cfg_builder.branch(&middle, 0, &exit)?;
        assert_matches!(
            cfg_builder.finish_prelude_hugr(),
            Err(ValidationError::InterGraphEdgeError(
                InterGraphEdgeError::NonDominatedAncestor { .. }
            ))
        );

        Ok(())
    }
}
