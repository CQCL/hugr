use super::build_traits::{HugrBuilder, SubContainer};
use super::handle::BuildHandle;
use super::{BuildError, Container, Dataflow, DfgID, FuncID};

use std::marker::PhantomData;

use crate::hugr::{HugrView, NodeType, ValidationError};
use crate::ops;

use crate::types::{FunctionType, Signature};

use crate::extension::{ExtensionRegistry, ExtensionSet};
use crate::Node;
use crate::{hugr::HugrMut, Hugr};

/// Builder for a [`ops::DFG`] node.
#[derive(Debug, Clone, PartialEq)]
pub struct DFGBuilder<T> {
    pub(crate) base: T,
    pub(crate) dfg_node: Node,
    pub(crate) num_in_wires: usize,
    pub(crate) num_out_wires: usize,
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> DFGBuilder<T> {
    pub(super) fn create_with_io(
        mut base: T,
        parent: Node,
        signature: FunctionType,
        input_extensions: Option<ExtensionSet>,
    ) -> Result<Self, BuildError> {
        let num_in_wires = signature.input().len();
        let num_out_wires = signature.output().len();
        /* For a given dataflow graph with extension requirements IR -> IR + dR,
         - The output node's extension requirements are IR + dR -> IR + dR
           (but we expect no output wires)

         - The input node's extension requirements are IR -> IR, though we
           expect no input wires. We must avoid the case where the difference
           in extensions is an open variable, as it would be if the requirements
           were 0 -> IR.
           N.B. This means that for input nodes, we can't infer the extensions
           from the input wires as we normally expect, but have to infer the
           output wires and make use of the equality between the two.
        */
        let input = ops::Input {
            types: signature.input().clone(),
        };
        let output = ops::Output {
            types: signature.output().clone(),
        };
        base.as_mut()
            .add_node_with_parent(parent, NodeType::new(input, input_extensions.clone()))?;
        base.as_mut().add_node_with_parent(
            parent,
            NodeType::new(
                output,
                input_extensions.map(|inp| inp.union(&signature.extension_reqs)),
            ),
        )?;

        Ok(Self {
            base,
            dfg_node: parent,
            num_in_wires,
            num_out_wires,
        })
    }
}

impl DFGBuilder<Hugr> {
    /// Begin building a new DFG rooted HUGR.
    /// Input extensions default to being an open variable
    ///
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn new(signature: FunctionType) -> Result<DFGBuilder<Hugr>, BuildError> {
        let dfg_op = ops::DFG {
            signature: signature.clone(),
        };
        let base = Hugr::new(NodeType::open_extensions(dfg_op));
        let root = base.root();
        DFGBuilder::create_with_io(base, root, signature, None)
    }
}

impl HugrBuilder for DFGBuilder<Hugr> {
    fn finish_hugr(
        mut self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Hugr, ValidationError> {
        self.base.update_validate(extension_registry)?;
        Ok(self.base)
    }
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> Container for DFGBuilder<T> {
    #[inline]
    fn container_node(&self) -> Node {
        self.dfg_node
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

impl<T: AsMut<Hugr> + AsRef<Hugr>> SubContainer for DFGBuilder<T> {
    type ContainerHandle = BuildHandle<DfgID>;
    #[inline]
    fn finish_sub_container(self) -> Result<Self::ContainerHandle, BuildError> {
        Ok((self.dfg_node, self.num_out_wires).into())
    }
}

impl<T: AsMut<Hugr> + AsRef<Hugr>> Dataflow for DFGBuilder<T> {
    #[inline]
    fn num_inputs(&self) -> usize {
        self.num_in_wires
    }
}

/// Wrapper around [`DFGBuilder`] used to build other dataflow regions.
// Stores option of DFGBuilder so it can be taken out without moving.
#[derive(Debug, Clone, PartialEq)]
pub struct DFGWrapper<B, T>(DFGBuilder<B>, PhantomData<T>);

impl<B, T> DFGWrapper<B, T> {
    pub(super) fn from_dfg_builder(db: DFGBuilder<B>) -> Self {
        Self(db, PhantomData)
    }
}

/// Builder for a [`ops::FuncDefn`] node
pub type FunctionBuilder<B> = DFGWrapper<B, BuildHandle<FuncID<true>>>;

impl FunctionBuilder<Hugr> {
    /// Initialize a builder for a FuncDefn rooted HUGR
    /// # Errors
    ///
    /// Error in adding DFG child nodes.
    pub fn new(name: impl Into<String>, signature: Signature) -> Result<Self, BuildError> {
        let op = ops::FuncDefn {
            signature: signature.clone().into(),
            name: name.into(),
        };

        let base = Hugr::new(NodeType::new(op, signature.input_extensions.clone()));
        let root = base.root();

        let db = DFGBuilder::create_with_io(
            base,
            root,
            signature.signature,
            Some(signature.input_extensions),
        )?;
        Ok(Self::from_dfg_builder(db))
    }
}

impl<B: AsMut<Hugr> + AsRef<Hugr>, T> Container for DFGWrapper<B, T> {
    #[inline]
    fn container_node(&self) -> Node {
        self.0.container_node()
    }

    #[inline]
    fn hugr_mut(&mut self) -> &mut Hugr {
        self.0.hugr_mut()
    }

    #[inline]
    fn hugr(&self) -> &Hugr {
        self.0.hugr()
    }
}

impl<B: AsMut<Hugr> + AsRef<Hugr>, T> Dataflow for DFGWrapper<B, T> {
    #[inline]
    fn num_inputs(&self) -> usize {
        self.0.num_inputs()
    }
}

impl<B: AsMut<Hugr> + AsRef<Hugr>, T: From<BuildHandle<DfgID>>> SubContainer for DFGWrapper<B, T> {
    type ContainerHandle = T;

    #[inline]
    fn finish_sub_container(self) -> Result<Self::ContainerHandle, BuildError> {
        self.0.finish_sub_container().map(Into::into)
    }
}

impl<T> HugrBuilder for DFGWrapper<Hugr, T> {
    fn finish_hugr(self, extension_registry: &ExtensionRegistry) -> Result<Hugr, ValidationError> {
        self.0.finish_hugr(extension_registry)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use cool_asserts::assert_matches;
    use rstest::rstest;
    use serde_json::json;

    use crate::builder::build_traits::DataflowHugr;
    use crate::builder::{DataflowSubContainer, ModuleBuilder};
    use crate::extension::prelude::BOOL_T;
    use crate::extension::{ExtensionId, EMPTY_REG};
    use crate::hugr::validate::InterGraphEdgeError;
    use crate::ops::{handle::NodeHandle, LeafOp, OpTag};

    use crate::std_extensions::logic::test::and_op;
    use crate::std_extensions::quantum::test::h_gate;
    use crate::{
        builder::{
            test::{n_identity, BIT, NAT, QB},
            BuildError,
        },
        extension::ExtensionSet,
        type_row, Wire,
    };

    use super::super::test::simple_dfg_hugr;
    use super::*;
    #[test]
    fn nested_identity() {
        assert_matches!(hugr_utils::examples::nested_identity(), Ok(_));
    }

    #[test]
    fn copy_insertion() {
        assert_matches!(hugr_utils::examples::copy_input_and_output(), Ok(_));
        assert_matches!(hugr_utils::examples::copy_input_and_output(), Ok(_));
        assert_matches!(hugr_utils::examples::copy_multiple_times(), Ok(_));
    }

    #[test]
    fn copy_insertion_qubit() {
        let builder = || {
            let mut module_builder = ModuleBuilder::new();

            let f_build = module_builder.define_function(
                "main",
                FunctionType::new(type_row![QB], type_row![QB, QB]).pure(),
            )?;

            let [q1] = f_build.input_wires_arr();
            f_build.finish_with_outputs([q1, q1])?;

            Ok(module_builder.finish_prelude_hugr()?)
        };

        assert_eq!(builder(), Err(BuildError::NoCopyLinear(QB)));
    }

    #[test]
    fn simple_inter_graph_edge() {
        assert_matches!(hugr_utils::examples::simple_inter_graph_edge(), Ok(_));
    }

    #[test]
    fn error_on_linear_inter_graph_edge() -> Result<(), BuildError> {
        let mut f_build = FunctionBuilder::new(
            "main",
            FunctionType::new(type_row![QB], type_row![QB]).pure(),
        )?;

        let [i1] = f_build.input_wires_arr();
        let noop = f_build.add_dataflow_op(LeafOp::Noop { ty: QB }, [i1])?;
        let i1 = noop.out_wire(0);

        let mut nested =
            f_build.dfg_builder(FunctionType::new(type_row![], type_row![QB]), None, [])?;

        let id_res = nested.add_dataflow_op(LeafOp::Noop { ty: QB }, [i1]);

        // The error would anyway be caught in validation when we finish the Hugr,
        // but the builder catches it earlier
        assert_matches!(
            id_res.map(|bh| bh.handle().node()), // Transform into something that impl's Debug
            Err(BuildError::InvalidHUGR(
                ValidationError::InterGraphEdgeError(InterGraphEdgeError::NonCopyableData { .. })
            ))
        );

        Ok(())
    }

    #[rstest]
    fn dfg_hugr(simple_dfg_hugr: Hugr) {
        assert_eq!(simple_dfg_hugr.node_count(), 3);
        assert_matches!(simple_dfg_hugr.root_type().tag(), OpTag::Dfg);
    }

    #[test]
    fn insert_hugr() -> Result<(), BuildError> {
        // Create a simple DFG
        let mut dfg_builder = DFGBuilder::new(FunctionType::new(type_row![BIT], type_row![BIT]))?;
        let [i1] = dfg_builder.input_wires_arr();
        dfg_builder.set_metadata(json!(42));
        let dfg_hugr = dfg_builder.finish_hugr_with_outputs([i1], &EMPTY_REG)?;

        // Create a module, and insert the DFG into it
        let mut module_builder = ModuleBuilder::new();

        {
            let mut f_build = module_builder.define_function(
                "main",
                FunctionType::new(type_row![BIT], type_row![BIT]).pure(),
            )?;

            let [i1] = f_build.input_wires_arr();
            let id = f_build.add_hugr_with_wires(dfg_hugr, [i1])?;
            f_build.finish_with_outputs([id.out_wire(0)])?;
        }

        assert_eq!(module_builder.finish_hugr(&EMPTY_REG)?.node_count(), 7);

        Ok(())
    }

    #[test]
    fn lift_node() -> Result<(), BuildError> {
        let xa: ExtensionId = "A".try_into().unwrap();
        let xb: ExtensionId = "B".try_into().unwrap();
        let xc = "C".try_into().unwrap();
        let ab_extensions = ExtensionSet::from_iter([xa.clone(), xb.clone()]);
        let c_extensions = ExtensionSet::singleton(&xc);
        let abc_extensions = ab_extensions.clone().union(&c_extensions);

        let parent_sig =
            FunctionType::new(type_row![BIT], type_row![BIT]).with_extension_delta(&abc_extensions);
        let mut parent = DFGBuilder::new(parent_sig)?;

        let add_c_sig = FunctionType::new(type_row![BIT], type_row![BIT])
            .with_extension_delta(&c_extensions)
            .with_input_extensions(ab_extensions.clone());

        let [w] = parent.input_wires_arr();

        let add_ab_sig =
            FunctionType::new(type_row![BIT], type_row![BIT]).with_extension_delta(&ab_extensions);

        // A box which adds extensions A and B, via child Lift nodes
        let mut add_ab = parent.dfg_builder(add_ab_sig, Some(ExtensionSet::new()), [w])?;
        let [w] = add_ab.input_wires_arr();

        let lift_a = add_ab.add_dataflow_op(
            LeafOp::Lift {
                type_row: type_row![BIT],
                new_extension: xa.clone(),
            },
            [w],
        )?;
        let [w] = lift_a.outputs_arr();

        let lift_b = add_ab.add_dataflow_node(
            NodeType::new(
                LeafOp::Lift {
                    type_row: type_row![BIT],
                    new_extension: xb,
                },
                ExtensionSet::from_iter([xa]),
            ),
            [w],
        )?;
        let [w] = lift_b.outputs_arr();

        let add_ab = add_ab.finish_with_outputs([w])?;
        let [w] = add_ab.outputs_arr();

        // Add another node (a sibling to add_ab) which adds extension C
        // via a child lift node
        let mut add_c =
            parent.dfg_builder(add_c_sig.signature, Some(add_c_sig.input_extensions), [w])?;
        let [w] = add_c.input_wires_arr();
        let lift_c = add_c.add_dataflow_node(
            NodeType::new(
                LeafOp::Lift {
                    type_row: type_row![BIT],
                    new_extension: xc,
                },
                ab_extensions,
            ),
            [w],
        )?;
        let wires: Vec<Wire> = lift_c.outputs().collect();

        let add_c = add_c.finish_with_outputs(wires)?;
        let [w] = add_c.outputs_arr();
        parent.finish_hugr_with_outputs([w], &EMPTY_REG)?;

        Ok(())
    }
}
