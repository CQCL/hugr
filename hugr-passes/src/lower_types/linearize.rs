use std::{collections::HashMap, sync::Arc};

use hugr_core::{
    extension::TypeDef,
    hugr::hugrmut::HugrMut,
    types::{Type, TypeArg, TypeEnum},
    IncomingPort, Node, OutgoingPort,
};

use super::{OpReplacement, ParametricType};

#[derive(Clone, Default)]
pub struct Linearizer {
    // Keyed by lowered type, as only needed when there is an op outputting such
    copy_discard: HashMap<Type, (OpReplacement, OpReplacement)>,
    // Copy/discard of parametric types handled by a function that receives the new/lowered type.
    // We do not allow linearization to "parametrized" non-extension types, at least not
    // in one step. We could do that using a trait, but it seems enough of a corner case.
    // Instead that can be achieved by *firstly* lowering to a custom linear type, with copy/dup
    // inserted; *secondly* by lowering that to the desired non-extension linear type,
    // including lowering of the copy/dup operations to...whatever.
    copy_discard_parametric: HashMap<
        ParametricType,
        (
            Arc<dyn Fn(&[TypeArg], &Linearizer) -> Result<OpReplacement, LinearizeError>>,
            Arc<dyn Fn(&[TypeArg], &Linearizer) -> Result<OpReplacement, LinearizeError>>,
        ),
    >,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum LinearizeError {
    #[error("Need copy op for {_0}")]
    NeedCopy(Type),
    #[error("Need discard op for {_0}")]
    NeedDiscard(Type),
    #[error("Cannot add nonlocal edge for linear type from {src} (with parent {src_parent}) to {tgt} (with parent {tgt_parent})")]
    NoLinearNonLocalEdges {
        src: Node,
        src_parent: Node,
        tgt: Node,
        tgt_parent: Node,
    },
}

impl Linearizer {
    pub fn register(&mut self, typ: Type, copy: OpReplacement, discard: OpReplacement) {
        self.copy_discard.insert(typ, (copy, discard));
    }

    pub fn register_parametric(
        &mut self,
        src: &TypeDef,
        copy_fn: Box<dyn Fn(&[TypeArg], &Linearizer) -> Result<OpReplacement, LinearizeError>>,
        discard_fn: Box<dyn Fn(&[TypeArg], &Linearizer) -> Result<OpReplacement, LinearizeError>>,
    ) {
        self.copy_discard_parametric
            .insert(src.into(), (Arc::from(copy_fn), Arc::from(discard_fn)));
    }

    /// Insert copy or discard operations (as appropriate) enough to wire `src_port` of `src_node`
    /// up to all `targets`.
    ///
    /// # Errors
    ///
    /// If needed copy or discard ops cannot be found;
    pub fn insert_copy_discard(
        &self,
        hugr: &mut impl HugrMut,
        mut src_node: Node,
        mut src_port: OutgoingPort,
        typ: &Type, // Or better to get the signature ourselves??
        targets: &[(Node, IncomingPort)],
    ) -> Result<(), LinearizeError> {
        let (last_node, last_inport) = match targets.last() {
            None => {
                let parent = hugr.get_parent(src_node).unwrap();
                (self.discard_op(typ)?.add(hugr, parent), 0.into())
            }
            Some(last) => *last,
        };

        if targets.len() > 1 {
            // Fail fast if the edges are nonlocal. (TODO transform to local edges!)
            let src_parent = hugr
                .get_parent(src_node)
                .expect("Root node cannot have out edges");
            if let Some((tgt, tgt_parent)) = targets.iter().find_map(|(tgt, _)| {
                let tgt_parent = hugr
                    .get_parent(*tgt)
                    .expect("Root node cannot have incoming edges");
                (tgt_parent != src_parent).then_some((*tgt, tgt_parent))
            }) {
                return Err(LinearizeError::NoLinearNonLocalEdges {
                    src: src_node,
                    src_parent,
                    tgt,
                    tgt_parent,
                });
            }

            let copy_op = self.copy_op(typ)?;

            for (tgt_node, tgt_port) in &targets[..targets.len() - 1] {
                let n = copy_op.add(hugr, hugr.get_parent(src_node).unwrap());
                hugr.connect(src_node, src_port, n, 0);
                hugr.connect(n, 0, *tgt_node, *tgt_port);
                (src_node, src_port) = (n, 1.into());
            }
        }
        hugr.connect(src_node, src_port, last_node, last_inport);
        Ok(())
    }

    fn copy_op(&self, typ: &Type) -> Result<OpReplacement, LinearizeError> {
        if let Some((copy, _)) = self.copy_discard.get(typ) {
            return Ok(copy.clone());
        }
        let TypeEnum::Extension(exty) = typ.as_type_enum() else {
            todo!() // handle sums, etc....
        };
        let (copy_fn, _) = self
            .copy_discard_parametric
            .get(&exty.into())
            .ok_or_else(|| LinearizeError::NeedCopy(typ.clone()))?;
        copy_fn(exty.args(), self)
    }

    fn discard_op(&self, typ: &Type) -> Result<OpReplacement, LinearizeError> {
        if let Some((_, discard)) = self.copy_discard.get(typ) {
            return Ok(discard.clone());
        }
        let TypeEnum::Extension(exty) = typ.as_type_enum() else {
            todo!() // handle sums, etc...
        };
        let (_, discard_fn) = self
            .copy_discard_parametric
            .get(&exty.into())
            .ok_or_else(|| LinearizeError::NeedDiscard(typ.clone()))?;
        discard_fn(exty.args(), self)
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use hugr_core::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use hugr_core::extension::{TypeDefBound, Version};
    use hugr_core::hugr::IdentList;
    use hugr_core::ops::{ExtensionOp, NamedOp, OpName};
    use hugr_core::std_extensions::collections::array::{array_type, ArrayOpDef};
    use hugr_core::types::{Type, TypeEnum};
    use hugr_core::{extension::prelude::usize_t, types::Signature};
    use hugr_core::{Extension, HugrView};

    use crate::lower_types::OpReplacement;
    use crate::LowerTypes;

    #[test]
    fn single_values() {
        // Extension with a linear type, a copy and discard op
        let e = Extension::new_arc(
            IdentList::new_unchecked("TestExt"),
            Version::new(0, 0, 0),
            |e, w| {
                let lin = Type::new_extension(
                    e.add_type("Lin".into(), vec![], String::new(), TypeDefBound::any(), w)
                        .unwrap()
                        .instantiate([])
                        .unwrap(),
                );
                e.add_op(
                    "copy".into(),
                    String::new(),
                    Signature::new(lin.clone(), vec![lin.clone(); 2]),
                    w,
                )
                .unwrap();
                e.add_op(
                    "discard".into(),
                    String::new(),
                    Signature::new(lin, vec![]),
                    w,
                )
                .unwrap();
            },
        );
        let lin_t = Type::new_extension(e.get_type("Lin").unwrap().instantiate([]).unwrap());

        // Configure to lower usize_t to the linear type above
        let copy_op = ExtensionOp::new(e.get_op("copy").unwrap().clone(), []).unwrap();
        let discard_op = ExtensionOp::new(e.get_op("discard").unwrap().clone(), []).unwrap();
        let mut lowerer = LowerTypes::default();
        let TypeEnum::Extension(usize_custom_t) = usize_t().as_type_enum().clone() else {
            panic!()
        };
        lowerer.lower_type(usize_custom_t, lin_t.clone());
        lowerer.linearize(
            lin_t,
            OpReplacement::SingleOp(copy_op.into()),
            OpReplacement::SingleOp(discard_op.into()),
        );

        // Build Hugr - uses first input three times, discards second input (both usize)
        let mut outer = DFGBuilder::new(Signature::new(
            vec![usize_t(); 2],
            vec![usize_t(), array_type(2, usize_t())],
        ))
        .unwrap();
        let [inp, _] = outer.input_wires_arr();
        let new_array = outer
            .add_dataflow_op(ArrayOpDef::new_array.to_concrete(usize_t(), 2), [inp, inp])
            .unwrap();
        let [arr] = new_array.outputs_arr();
        let mut h = outer.finish_hugr_with_outputs([inp, arr]).unwrap();

        assert!(lowerer.run(&mut h).unwrap());

        let ext_ops = h.nodes().filter_map(|n| h.get_optype(n).as_extension_op());
        let mut counts = HashMap::<OpName, u32>::new();
        for e in ext_ops {
            *counts.entry(e.name()).or_default() += 1;
        }
        assert_eq!(
            counts,
            HashMap::from([
                ("TestExt.copy".into(), 2),
                ("TestExt.discard".into(), 1),
                ("collections.array.new_array".into(), 1)
            ])
        );
    }
}
