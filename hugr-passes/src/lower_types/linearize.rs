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
        // TODO should pass &LowerTypes, or at least some way to call copy_op / discard_op, to these
        (
            Arc<dyn Fn(&[TypeArg]) -> OpReplacement>,
            Arc<dyn Fn(&[TypeArg]) -> OpReplacement>,
        ),
    >,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum LinearizeError {
    #[error("Need copy op for {_0}")]
    NeedCopy(Type),
    #[error("Need discard op for {_0}")]
    NeedDiscard(Type),
}

impl Linearizer {
    pub fn register(&mut self, typ: Type, copy: OpReplacement, discard: OpReplacement) {
        self.copy_discard.insert(typ, (copy, discard));
    }

    pub fn register_parametric(
        &mut self,
        src: TypeDef,
        copy_fn: Box<dyn Fn(&[TypeArg]) -> OpReplacement>,
        discard_fn: Box<dyn Fn(&[TypeArg]) -> OpReplacement>,
    ) {
        self.copy_discard_parametric
            .insert((&src).into(), (Arc::from(copy_fn), Arc::from(discard_fn)));
    }

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
        Ok(copy_fn(exty.args()))
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
        Ok(discard_fn(exty.args()))
    }
}
