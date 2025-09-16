use super::{
    BuildError, Container,
    build_traits::HugrBuilder,
    dataflow::{DFGBuilder, FunctionBuilder},
};

use crate::hugr::linking::{HugrLinking, NodeLinkingDirectives, NodeLinkingError};
use crate::hugr::{
    ValidationError, hugrmut::InsertedForest, internal::HugrMutInternals, views::HugrView,
};
use crate::ops;
use crate::ops::handle::{AliasID, FuncID, NodeHandle};
use crate::types::{PolyFuncType, Type, TypeBound};
use crate::{Hugr, Node, Visibility, ops::FuncDefn};

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
    /// Continue building a module from an existing hugr.
    #[must_use]
    pub fn with_hugr(hugr: T) -> Self {
        ModuleBuilder(hugr)
    }

    /// Replace a [`ops::FuncDecl`] with [`ops::FuncDefn`] and return a builder for
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
        *opty = ops::FuncDefn::new_vis(
            decl.func_name(),
            decl.signature().clone(),
            decl.visibility().clone(),
        )
        .into();

        let db = DFGBuilder::create_with_io(self.hugr_mut(), f_node, body)?;
        Ok(FunctionBuilder::from_dfg_builder(db))
    }

    /// Add a [`ops::FuncDefn`] node of the specified visibility.
    /// Returns a builder to define the function body graph.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`ops::FuncDefn`] node.
    pub fn define_function_vis(
        &mut self,
        name: impl Into<String>,
        signature: impl Into<PolyFuncType>,
        visibility: Visibility,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        self.define_function_op(FuncDefn::new_vis(name, signature, visibility))
    }

    fn define_function_op(
        &mut self,
        op: FuncDefn,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        let body = op.signature().body().clone();
        let f_node = self.add_child_node(op);

        // Add the extensions used by the function types.
        self.use_extensions(
            body.used_extensions().unwrap_or_else(|e| {
                panic!("Build-time signatures should have valid extensions. {e}")
            }),
        );

        let db = DFGBuilder::create_with_io(self.hugr_mut(), f_node, body)?;
        Ok(FunctionBuilder::from_dfg_builder(db))
    }

    /// Declare a [Visibility::Public] function with `signature` and return a handle to the declaration.
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
        self.declare_vis(name, signature, Visibility::Public)
    }

    /// Declare a function with the specified `signature` and [Visibility],
    /// and return a handle to the declaration.
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`crate::ops::OpType::FuncDecl`] node.
    pub fn declare_vis(
        &mut self,
        name: impl Into<String>,
        signature: PolyFuncType,
        visibility: Visibility,
    ) -> Result<FuncID<false>, BuildError> {
        let body = signature.body().clone();
        // TODO add param names to metadata
        let declare_n = self.add_child_node(ops::FuncDecl::new_vis(name, signature, visibility));

        // Add the extensions used by the function types.
        self.use_extensions(
            body.used_extensions().unwrap_or_else(|e| {
                panic!("Build-time signatures should have valid extensions. {e}")
            }),
        );

        Ok(declare_n.into())
    }

    /// Adds a [`ops::FuncDefn`] node and returns a builder to define the function
    /// body graph. The function will be private. (See [Self::define_function_vis].)
    ///
    /// # Errors
    ///
    /// This function will return an error if there is an error in adding the
    /// [`ops::FuncDefn`] node.
    pub fn define_function(
        &mut self,
        name: impl Into<String>,
        signature: impl Into<PolyFuncType>,
    ) -> Result<FunctionBuilder<&mut Hugr>, BuildError> {
        self.define_function_op(FuncDefn::new(name, signature))
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

    /// Add some module-children of another Hugr to this module, with
    /// linking directives specified explicitly by [Node].
    ///
    /// `children` contains a map from the children of `other` to insert,
    /// to how they should be combined with the nodes in `self`. Note if
    /// this map is empty, nothing is added.
    pub fn link_hugr_by_node(
        &mut self,
        other: Hugr,
        children: NodeLinkingDirectives<Node, Node>,
    ) -> Result<InsertedForest, NodeLinkingError> {
        self.hugr_mut()
            .insert_link_hugr_by_node(None, other, children)
    }

    /// Copy module-children from a HugrView into this module, with
    /// linking directives specified explicitly by [Node].
    ///
    /// `children` contains a map from the children of `other` to copy,
    /// to how they should be combined with the nodes in `self`. Note if
    /// this map is empty, nothing is added.
    pub fn link_view_by_node<H: HugrView>(
        &mut self,
        other: &H,
        children: NodeLinkingDirectives<H::Node, Node>,
    ) -> Result<InsertedForest<H::Node>, NodeLinkingError<H::Node>> {
        self.hugr_mut()
            .insert_link_view_by_node(None, other, children)
    }
}

#[cfg(test)]
mod test {
    use std::collections::{HashMap, HashSet};

    use cool_asserts::assert_matches;

    use crate::builder::test::dfg_calling_defn_decl;
    use crate::builder::{Dataflow, DataflowSubContainer, test::n_identity};
    use crate::extension::prelude::usize_t;
    use crate::{hugr::linking::NodeLinkingDirective, ops::OpType, types::Signature};

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
                module_builder.add_alias_declare("qubit_state", TypeBound::Linear)?;

            let f_build = module_builder.define_function(
                "main",
                Signature::new(
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
    fn builder_from_existing() -> Result<(), BuildError> {
        let hugr = Hugr::new();

        let fn_builder = FunctionBuilder::with_hugr(hugr, "main", Signature::new_endo(vec![]))?;
        let mut hugr = fn_builder.finish_hugr()?;

        let mut module_builder = ModuleBuilder::with_hugr(&mut hugr);
        module_builder.declare("other", Signature::new_endo(vec![]).into())?;

        hugr.validate()?;

        Ok(())
    }

    #[test]
    fn link_by_node() {
        let mut mb = ModuleBuilder::new();
        let (dfg, defn, decl) = dfg_calling_defn_decl();
        let added = mb
            .link_view_by_node(
                &dfg,
                HashMap::from([
                    (defn.node(), NodeLinkingDirective::add()),
                    (decl.node(), NodeLinkingDirective::add()),
                ]),
            )
            .unwrap();
        let n_defn = added.node_map[&defn.node()];
        let n_decl = added.node_map[&decl.node()];
        let h = mb.hugr();
        assert_eq!(h.children(h.module_root()).count(), 2);
        h.validate().unwrap();
        let old_name = match mb.hugr_mut().optype_mut(n_defn) {
            OpType::FuncDefn(fd) => std::mem::replace(fd.func_name_mut(), "new".to_string()),
            _ => panic!(),
        };
        let main = dfg.get_parent(dfg.entrypoint()).unwrap();
        assert_eq!(
            dfg.get_optype(main).as_func_defn().unwrap().func_name(),
            "main"
        );
        mb.link_hugr_by_node(
            dfg,
            HashMap::from([
                (main, NodeLinkingDirective::add()),
                (decl.node(), NodeLinkingDirective::UseExisting(n_defn)),
                (defn.node(), NodeLinkingDirective::replace([n_decl])),
            ]),
        )
        .unwrap();
        let h = mb.finish_hugr().unwrap();
        assert_eq!(
            h.children(h.module_root())
                .map(|n| h.get_optype(n).as_func_defn().unwrap().func_name().as_str())
                .collect::<HashSet<_>>(),
            HashSet::from(["main", "new", old_name.as_str()])
        );
    }
}
