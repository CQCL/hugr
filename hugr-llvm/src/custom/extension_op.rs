use std::{collections::HashMap, rc::Rc};

use hugr_core::{
    extension::{
        simple_op::{MakeExtensionOp, MakeOpDef},
        ExtensionId,
    },
    ops::{ExtensionOp, OpName},
    HugrView,
};

use anyhow::{bail, Result};

use strum::IntoEnumIterator;

use crate::emit::{EmitFuncContext, EmitOpArgs};

/// A helper trait for describing the callback used for emitting [ExtensionOp]s,
/// and for hanging documentation. We have the appropriate `Fn` as a supertrait,
/// and there is a blanket impl for that `Fn`. We do not intend users to impl
/// this trait.
///
/// `ExtensionOpFn` callbacks are registered against a fully qualified [OpName],
/// i.e. including it's [ExtensionId]. Callbacks can assume that the provided
/// [EmitOpArgs::node] holds an op matching that fully qualified name, and that
/// the signature of that op determinies the length and types of
/// [EmitOpArgs::inputs], and [EmitOpArgs::outputs] via
/// [EmitFuncContext::llvm_type].
///
/// Callbacks should use the supplied [EmitFuncContext] to emit LLVM to match
/// the desired semantics of the op. If a callback returns success then the callback must:
///  - ensure that [crate::emit::func::RowPromise::finish] has been called on the outputs.
///  - ensure that the contexts [inkwell::builder::Builder] is positioned at the end of a basic
///    block, logically after the execution of the just-emitted op.
///
/// Callbacks may hold references with lifetimes older than `'a`.
pub trait ExtensionOpFn<'a, H>:
    for<'c> Fn(&mut EmitFuncContext<'c, 'a, H>, EmitOpArgs<'c, '_, ExtensionOp, H>) -> Result<()> + 'a
{
}

impl<
        'a,
        H,
        F: for<'c> Fn(
                &mut EmitFuncContext<'c, 'a, H>,
                EmitOpArgs<'c, '_, ExtensionOp, H>,
            ) -> Result<()>
            + ?Sized
            + 'a,
    > ExtensionOpFn<'a, H> for F
{
}

/// A collection of [ExtensionOpFn] callbacks keyed the fully qualified [OpName].
///
/// Those callbacks may hold references with lifetimes older than `'a`.
#[derive(Default)]
pub struct ExtensionOpMap<'a, H>(HashMap<(ExtensionId, OpName), Box<dyn ExtensionOpFn<'a, H>>>);

impl<'a, H: HugrView> ExtensionOpMap<'a, H> {
    /// Register a callback to emit a [ExtensionOp], keyed by fully
    /// qualified [OpName].
    pub fn extension_op(
        &mut self,
        extension: ExtensionId,
        op: OpName,
        handler: impl ExtensionOpFn<'a, H>,
    ) {
        self.0.insert((extension, op), Box::new(handler));
    }

    /// Register callbacks to emit [ExtensionOp]s that match the
    /// definitions generated by `Op`s impl of [strum::IntoEnumIterator]>
    pub fn simple_extension_op<Op: MakeOpDef + IntoEnumIterator>(
        &mut self,
        handler: impl 'a
            + for<'c> Fn(
                &mut EmitFuncContext<'c, 'a, H>,
                EmitOpArgs<'c, '_, ExtensionOp, H>,
                Op,
            ) -> Result<()>,
    ) {
        let handler = Rc::new(handler);
        for op in Op::iter() {
            let handler = handler.clone();
            self.extension_op(op.extension(), op.name().clone(), move |context, args| {
                let op = Op::from_extension_op(&args.node())?;
                handler(context, args, op)
            });
        }
    }

    /// Emit an [ExtensionOp]  by delegating to the collected callbacks.
    ///
    /// If no handler is registered for the op an error will be returned.
    pub fn emit_extension_op<'c>(
        &self,
        context: &mut EmitFuncContext<'c, 'a, H>,
        args: EmitOpArgs<'c, '_, ExtensionOp, H>,
    ) -> Result<()> {
        let node = args.node();
        let key = (node.def().extension().clone(), node.def().name().clone());
        let Some(handler) = self.0.get(&key) else {
            bail!("No extension could emit extension op: {key:?}")
        };
        (handler.as_ref())(context, args)
    }
}
