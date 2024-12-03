//! Resolve weak links inside `CustomType`s in an optype's signature.

use std::sync::Arc;

use super::{ExtensionRegistry, ExtensionResolutionError};
use crate::ops::OpType;
use crate::types::type_row::TypeRowBase;
use crate::types::{CustomType, MaybeRV, Signature, SumType, TypeBase, TypeEnum};
use crate::Node;

/// Replace the dangling extension pointer in the [`CustomType`]s inside a
/// signature with a valid pointer to the extension in the `extensions`
/// registry.
///
/// When a pointer is replaced, the extension is added to the
/// `used_extensions` registry and the new type definition is returned.
///
/// This is a helper function used right after deserializing a Hugr.
pub fn update_op_types_extensions(
    node: Node,
    op: &mut OpType,
    extensions: &ExtensionRegistry,
    used_extensions: &mut ExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    match op {
        OpType::ExtensionOp(ext) => {
            update_signature_exts(node, ext.signature_mut(), extensions, used_extensions)?
        }
        OpType::FuncDefn(f) => {
            update_signature_exts(node, f.signature.body_mut(), extensions, used_extensions)?
        }
        OpType::FuncDecl(f) => {
            update_signature_exts(node, f.signature.body_mut(), extensions, used_extensions)?
        }
        OpType::Const(_c) => {
            // TODO: Is it OK to assume that `Value::get_type` returns a well-resolved value?
        }
        OpType::Input(inp) => {
            update_type_row_exts(node, &mut inp.types, extensions, used_extensions)?
        }
        OpType::Output(out) => {
            update_type_row_exts(node, &mut out.types, extensions, used_extensions)?
        }
        OpType::Call(c) => {
            update_signature_exts(node, c.func_sig.body_mut(), extensions, used_extensions)?;
            update_signature_exts(node, &mut c.instantiation, extensions, used_extensions)?;
        }
        OpType::CallIndirect(c) => {
            update_signature_exts(node, &mut c.signature, extensions, used_extensions)?
        }
        OpType::LoadConstant(lc) => {
            update_type_exts(node, &mut lc.datatype, extensions, used_extensions)?
        }
        OpType::LoadFunction(lf) => {
            update_signature_exts(node, lf.func_sig.body_mut(), extensions, used_extensions)?;
            update_signature_exts(node, &mut lf.signature, extensions, used_extensions)?;
        }
        OpType::DFG(dfg) => {
            update_signature_exts(node, &mut dfg.signature, extensions, used_extensions)?
        }
        OpType::OpaqueOp(op) => {
            update_signature_exts(node, op.signature_mut(), extensions, used_extensions)?
        }
        OpType::Tag(t) => {
            for variant in t.variants.iter_mut() {
                update_type_row_exts(node, variant, extensions, used_extensions)?
            }
        }
        OpType::DataflowBlock(db) => {
            update_type_row_exts(node, &mut db.inputs, extensions, used_extensions)?;
            update_type_row_exts(node, &mut db.other_outputs, extensions, used_extensions)?;
            for row in db.sum_rows.iter_mut() {
                update_type_row_exts(node, row, extensions, used_extensions)?;
            }
        }
        OpType::ExitBlock(e) => {
            update_type_row_exts(node, &mut e.cfg_outputs, extensions, used_extensions)?;
        }
        OpType::TailLoop(tl) => {
            update_type_row_exts(node, &mut tl.just_inputs, extensions, used_extensions)?;
            update_type_row_exts(node, &mut tl.just_outputs, extensions, used_extensions)?;
            update_type_row_exts(node, &mut tl.rest, extensions, used_extensions)?;
        }
        OpType::CFG(cfg) => {
            update_signature_exts(node, &mut cfg.signature, extensions, used_extensions)?;
        }
        OpType::Conditional(cond) => {
            for row in cond.sum_rows.iter_mut() {
                update_type_row_exts(node, row, extensions, used_extensions)?;
            }
            update_type_row_exts(node, &mut cond.other_inputs, extensions, used_extensions)?;
            update_type_row_exts(node, &mut cond.outputs, extensions, used_extensions)?;
        }
        OpType::Case(case) => {
            update_signature_exts(node, &mut case.signature, extensions, used_extensions)?;
        }
        // Ignore optypes that do not store a signature.
        _ => {}
    }
    Ok(())
}

/// Update all weak Extension pointers in the [`CustomType`]s inside a signature.
///
/// Adds the extensions used in the signature to the `used_extensions` registry.
fn update_signature_exts(
    node: Node,
    signature: &mut Signature,
    extensions: &ExtensionRegistry,
    used_extensions: &mut ExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    update_type_row_exts(node, &mut signature.input, extensions, used_extensions)?;
    update_type_row_exts(node, &mut signature.output, extensions, used_extensions)?;
    Ok(())
}

/// Update all weak Extension pointers in the [`CustomType`]s inside a type row.
///
/// Adds the extensions used in the row to the `used_extensions` registry.
fn update_type_row_exts<RV: MaybeRV>(
    node: Node,
    row: &mut TypeRowBase<RV>,
    extensions: &ExtensionRegistry,
    used_extensions: &mut ExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    for ty in row.iter_mut() {
        update_type_exts(node, ty, extensions, used_extensions)?;
    }
    Ok(())
}

/// Update all weak Extension pointers in the [`CustomType`]s inside a type.
///
/// Adds the extensions used in the type to the `used_extensions` registry.
fn update_type_exts<RV: MaybeRV>(
    node: Node,
    typ: &mut TypeBase<RV>,
    extensions: &ExtensionRegistry,
    used_extensions: &mut ExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    match typ.as_type_enum_mut() {
        TypeEnum::Extension(custom) => {
            let ext_id = custom.extension();
            let ext = extensions.get(ext_id).ok_or_else(|| {
                ExtensionResolutionError::missing_type_extension(
                    node,
                    custom.name(),
                    ext_id,
                    extensions,
                )
            })?;

            // Add the extension to the used extensions registry,
            // and update the CustomType with the valid pointer.
            used_extensions.register_updated_ref(ext);
            *custom = CustomType::new(
                custom.name().clone(),
                custom.args(),
                custom.extension().clone(),
                custom.bound(),
                &Arc::downgrade(ext),
            );
        }
        TypeEnum::Function(f) => {
            update_type_row_exts(node, &mut f.input, extensions, used_extensions)?;
            update_type_row_exts(node, &mut f.output, extensions, used_extensions)?;
        }
        TypeEnum::Sum(SumType::General { rows }) => {
            for row in rows.iter_mut() {
                update_type_row_exts(node, row, extensions, used_extensions)?;
            }
        }
        _ => {}
    }
    Ok(())
}
