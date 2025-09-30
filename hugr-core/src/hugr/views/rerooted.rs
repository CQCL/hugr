//! A HUGR wrapper with a modified entrypoint node, returned by
//! [`HugrView::with_entrypoint`] and [`HugrMut::with_entrypoint_mut`].

use crate::hugr::internal::{HugrInternals, HugrMutInternals};
use crate::hugr::{HugrMut, hugrmut::InsertForestResult};

use super::{HugrView, panic_invalid_node};

/// A HUGR wrapper with a modified entrypoint node.
///
/// All nodes from the original are still present, but the main entrypoint used
/// for traversals and optimizations is altered.
#[derive(Clone)]
pub struct Rerooted<H: HugrView> {
    hugr: H,
    entrypoint: H::Node,
}

impl<H: HugrView> Rerooted<H> {
    /// Create a hierarchical view of a whole HUGR
    ///
    /// # Panics
    ///
    /// If the new entrypoint is not in the HUGR.
    ///
    /// [`OpTag`]: crate::ops::OpTag
    pub fn new(hugr: H, entrypoint: H::Node) -> Self {
        panic_invalid_node(&hugr, entrypoint);
        Self { hugr, entrypoint }
    }

    /// Returns the HUGR wrapped in this view.
    pub fn into_unwrapped(self) -> H {
        self.hugr
    }
}

impl<H: HugrView> HugrInternals for Rerooted<H> {
    type RegionPortgraph<'p>
        = H::RegionPortgraph<'p>
    where
        Self: 'p;

    type Node = H::Node;

    type RegionPortgraphNodes = H::RegionPortgraphNodes;

    super::impls::hugr_internal_methods! {this, &this.hugr}
}

impl<H: HugrView> HugrView for Rerooted<H> {
    #[inline]
    fn entrypoint(&self) -> Self::Node {
        self.entrypoint
    }

    #[inline]
    fn entrypoint_optype(&self) -> &crate::ops::OpType {
        self.hugr.get_optype(self.entrypoint)
    }

    fn mermaid_string_with_formatter(
        &self,
        formatter: crate::hugr::views::render::MermaidFormatter<Self>,
    ) -> String {
        self.hugr
            .mermaid_string_with_formatter(formatter.with_hugr(&self.hugr))
    }

    delegate::delegate! {
        to (&self.hugr) {
                fn module_root(&self) -> Self::Node;
                fn contains_node(&self, node: Self::Node) -> bool;
                fn get_parent(&self, node: Self::Node) -> Option<Self::Node>;
                fn get_metadata(&self, node: Self::Node, key: impl AsRef<str>) -> Option<&crate::hugr::NodeMetadata>;
                fn get_optype(&self, node: Self::Node) -> &crate::ops::OpType;
                fn num_nodes(&self) -> usize;
                fn num_edges(&self) -> usize;
                fn num_ports(&self, node: Self::Node, dir: crate::Direction) -> usize;
                fn num_inputs(&self, node: Self::Node) -> usize;
                fn num_outputs(&self, node: Self::Node) -> usize;
                fn nodes(&self) -> impl Iterator<Item = Self::Node> + Clone;
                fn node_ports(&self, node: Self::Node, dir: crate::Direction) -> impl Iterator<Item = crate::Port> + Clone;
                fn node_outputs(&self, node: Self::Node) -> impl Iterator<Item = crate::OutgoingPort> + Clone;
                fn node_inputs(&self, node: Self::Node) -> impl Iterator<Item = crate::IncomingPort> + Clone;
                fn all_node_ports(&self, node: Self::Node) -> impl Iterator<Item = crate::Port> + Clone;
                fn linked_ports(&self, node: Self::Node, port: impl Into<crate::Port>) -> impl Iterator<Item = (Self::Node, crate::Port)> + Clone;
                fn all_linked_ports(&self, node: Self::Node, dir: crate::Direction) -> itertools::Either<impl Iterator<Item = (Self::Node, crate::OutgoingPort)>, impl Iterator<Item = (Self::Node, crate::IncomingPort)>>;
                fn all_linked_outputs(&self, node: Self::Node) -> impl Iterator<Item = (Self::Node, crate::OutgoingPort)>;
                fn all_linked_inputs(&self, node: Self::Node) -> impl Iterator<Item = (Self::Node, crate::IncomingPort)>;
                fn single_linked_port(&self, node: Self::Node, port: impl Into<crate::Port>) -> Option<(Self::Node, crate::Port)>;
                fn single_linked_output(&self, node: Self::Node, port: impl Into<crate::IncomingPort>) -> Option<(Self::Node, crate::OutgoingPort)>;
                fn single_linked_input(&self, node: Self::Node, port: impl Into<crate::OutgoingPort>) -> Option<(Self::Node, crate::IncomingPort)>;
                fn linked_outputs(&self, node: Self::Node, port: impl Into<crate::IncomingPort>) -> impl Iterator<Item = (Self::Node, crate::OutgoingPort)>;
                fn linked_inputs(&self, node: Self::Node, port: impl Into<crate::OutgoingPort>) -> impl Iterator<Item = (Self::Node, crate::IncomingPort)>;
                fn node_connections(&self, node: Self::Node, other: Self::Node) -> impl Iterator<Item = [crate::Port; 2]> + Clone;
                fn is_linked(&self, node: Self::Node, port: impl Into<crate::Port>) -> bool;
                fn children(&self, node: Self::Node) -> impl DoubleEndedIterator<Item = Self::Node> + Clone;
                fn descendants(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone;
                fn first_child(&self, node: Self::Node) -> Option<Self::Node>;
                fn neighbours(&self, node: Self::Node, dir: crate::Direction) -> impl Iterator<Item = Self::Node> + Clone;
                fn all_neighbours(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + Clone;
                #[expect(deprecated)]
                fn mermaid_string_with_config(&self, config: crate::hugr::views::render::RenderConfig<Self::Node>) -> String;
                fn dot_string(&self) -> String;
                fn static_source(&self, node: Self::Node) -> Option<Self::Node>;
                fn static_targets(&self, node: Self::Node) -> Option<impl Iterator<Item = (Self::Node, crate::IncomingPort)>>;
                fn value_types(&self, node: Self::Node, dir: crate::Direction) -> impl Iterator<Item = (crate::Port, crate::types::Type)>;
                fn extensions(&self) -> &crate::extension::ExtensionRegistry;
                fn validate(&self) -> Result<(), crate::hugr::ValidationError<Self::Node>>;
                fn extract_hugr(&self, parent: Self::Node) -> (crate::Hugr, impl crate::hugr::views::ExtractionResult<Self::Node> + 'static);
        }
    }
}

impl<H: HugrMutInternals> HugrMutInternals for Rerooted<H> {
    super::impls::hugr_mut_internal_methods! {this, &mut this.hugr}
}

impl<H: HugrMut> HugrMut for Rerooted<H> {
    fn set_entrypoint(&mut self, root: Self::Node) {
        self.entrypoint = root;
        self.hugr.set_entrypoint(root);
    }

    delegate::delegate! {
        to (&mut self.hugr) {
                fn get_metadata_mut(&mut self, node: Self::Node, key: impl AsRef<str>) -> &mut crate::hugr::NodeMetadata;
                fn set_metadata(&mut self, node: Self::Node, key: impl AsRef<str>, metadata: impl Into<crate::hugr::NodeMetadata>);
                fn remove_metadata(&mut self, node: Self::Node, key: impl AsRef<str>);
                fn add_node_with_parent(&mut self, parent: Self::Node, op: impl Into<crate::ops::OpType>) -> Self::Node;
                fn add_node_before(&mut self, sibling: Self::Node, nodetype: impl Into<crate::ops::OpType>) -> Self::Node;
                fn add_node_after(&mut self, sibling: Self::Node, op: impl Into<crate::ops::OpType>) -> Self::Node;
                fn remove_node(&mut self, node: Self::Node) -> crate::ops::OpType;
                fn remove_subtree(&mut self, node: Self::Node);
                fn copy_descendants(&mut self, root: Self::Node, new_parent: Self::Node, subst: Option<crate::types::Substitution>) -> std::collections::BTreeMap<Self::Node, Self::Node>;
                fn connect(&mut self, src: Self::Node, src_port: impl Into<crate::OutgoingPort>, dst: Self::Node, dst_port: impl Into<crate::IncomingPort>);
                fn disconnect(&mut self, node: Self::Node, port: impl Into<crate::Port>);
                fn add_other_edge(&mut self, src: Self::Node, dst: Self::Node) -> (crate::OutgoingPort, crate::IncomingPort);
                fn insert_forest(&mut self, other: crate::Hugr, roots: impl IntoIterator<Item=(crate::Node, Self::Node)>) -> InsertForestResult<crate::Node, Self::Node>;
                fn insert_view_forest<Other: crate::hugr::HugrView>(&mut self, other: &Other, nodes: impl Iterator<Item=Other::Node> + Clone, roots: impl IntoIterator<Item=(Other::Node, Self::Node)>) -> InsertForestResult<Other::Node, Self::Node>;
                fn use_extension(&mut self, extension: impl Into<std::sync::Arc<crate::extension::Extension>>);
                fn use_extensions<Reg>(&mut self, registry: impl IntoIterator<Item = Reg>) where crate::extension::ExtensionRegistry: Extend<Reg>;
        }
    }
}

#[cfg(test)]
mod test {
    use crate::builder::test::simple_cfg_hugr;
    use crate::builder::{Dataflow, FunctionBuilder, HugrBuilder, SubContainer};
    use crate::hugr::HugrMut;
    use crate::hugr::internal::HugrMutInternals;
    use crate::hugr::views::ExtractionResult;
    use crate::ops::handle::NodeHandle;
    use crate::ops::{DataflowBlock, OpType};
    use crate::{HugrView, type_row, types::Signature};

    #[test]
    fn rerooted() {
        let mut builder = FunctionBuilder::new("main", Signature::new(vec![], vec![])).unwrap();
        let dfg = builder
            .dfg_builder_endo([])
            .unwrap()
            .finish_sub_container()
            .unwrap()
            .node();
        let mut h = builder.finish_hugr().unwrap();
        let _func = h.entrypoint();

        // Immutable wrappers
        let dfg_v = h.with_entrypoint(dfg);
        assert_eq!(dfg_v.module_root(), h.module_root());
        assert_eq!(dfg_v.entrypoint(), dfg);
        assert!(dfg_v.entrypoint_optype().is_dfg());
        assert!(dfg_v.get_optype(dfg_v.module_root().node()).is_module());

        // Mutable wrappers
        let mut dfg_v = h.with_entrypoint_mut(dfg);
        {
            // That is a HugrMutInternal, so we can try:
            let root = dfg_v.entrypoint();
            let bb: OpType = DataflowBlock {
                inputs: type_row![],
                other_outputs: type_row![],
                sum_rows: vec![type_row![]],
            }
            .into();
            dfg_v.replace_op(root, bb.clone());

            assert!(dfg_v.entrypoint_optype().is_dataflow_block());
            assert!(dfg_v.get_optype(dfg_v.module_root().node()).is_module());
        }
        // That modified the original HUGR
        assert!(h.get_optype(dfg).is_dataflow_block());
        assert!(h.entrypoint_optype().is_func_defn());
        assert!(h.get_optype(h.module_root().node()).is_module());
    }

    #[test]
    fn extract_rerooted() {
        let mut hugr = simple_cfg_hugr();
        let cfg = hugr.entrypoint();
        let basic_block = hugr.first_child(cfg).unwrap();
        hugr.set_entrypoint(basic_block);
        assert!(hugr.get_optype(hugr.entrypoint()).is_dataflow_block());

        let rerooted = hugr.with_entrypoint(cfg);
        assert!(rerooted.get_optype(rerooted.entrypoint()).is_cfg());

        // Extract the basic block
        let (extracted_hugr, map) = rerooted.extract_hugr(basic_block);
        let extracted_cfg = map.extracted_node(cfg);
        let extracted_bb = map.extracted_node(basic_block);
        assert_eq!(extracted_hugr.entrypoint(), extracted_bb);
        assert!(extracted_hugr.get_optype(extracted_cfg).is_cfg());
        assert_eq!(
            extracted_hugr.first_child(extracted_cfg),
            Some(extracted_bb)
        );
        assert!(extracted_hugr.get_optype(extracted_bb).is_dataflow_block());

        // Extract the cfg (and current entrypoint)
        let (extracted_hugr, map) = rerooted.extract_hugr(cfg);
        let extracted_cfg = map.extracted_node(cfg);
        let extracted_bb = map.extracted_node(basic_block);
        assert_eq!(extracted_hugr.entrypoint(), extracted_cfg);
        assert!(extracted_hugr.get_optype(extracted_cfg).is_cfg());
        assert_eq!(
            extracted_hugr.first_child(extracted_cfg),
            Some(extracted_bb)
        );
        assert!(extracted_hugr.get_optype(extracted_bb).is_dataflow_block());
    }

    #[test]
    fn mermaid_format() {
        let h = simple_cfg_hugr();
        let rerooted = h.with_entrypoint(h.entrypoint());
        let mermaid_str = rerooted.mermaid_format().finish();
        assert_eq!(mermaid_str, h.mermaid_format().finish());
    }
}
