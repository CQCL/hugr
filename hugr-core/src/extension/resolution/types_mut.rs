//! Resolve weak links inside `CustomType`s in an optype's signature, while
//! collecting all used extensions.
//!
//! For a non-mutating option see [`super::collect_op_types_extensions`].

use std::collections::HashSet;
use std::sync::Arc;

use super::types::collect_type_exts;
use super::{ExtensionRegistry, ExtensionResolutionError};
use crate::ops::OpType;
use crate::types::type_row::TypeRowBase;
use crate::types::{MaybeRV, Signature, SumType, TypeArg, TypeBase, TypeEnum};
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
        OpType::Const(c) => {
            let typ = c.get_type();
            let mut missing = HashSet::new();
            collect_type_exts(&typ, used_extensions, &mut missing);
            // We expect that the `CustomConst::get_type` binary calls always return valid extensions.
            // As we cannot update the `CustomConst` type, we ignore the result.
            //
            // Some exotic consts may need https://github.com/CQCL/hugr/issues/1742 to be implemented
            // to pass this test.
            //assert!(missing.is_empty());
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
    // Note that we do not include the signature's `extension_reqs` here, as those refer
    // to _runtime_ requirements that may not be currently present.
    // See https://github.com/CQCL/hugr/issues/1734
    // TODO: Update comment once that issue gets implemented.
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
            for arg in custom.args_mut() {
                update_typearg_exts(node, arg, extensions, used_extensions)?;
            }

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
            custom.update_extension(Arc::downgrade(ext));
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

/// Update all weak Extension pointers in the [`CustomType`]s inside a type arg.
///
/// Adds the extensions used in the type to the `used_extensions` registry.
fn update_typearg_exts(
    node: Node,
    arg: &mut TypeArg,
    extensions: &ExtensionRegistry,
    used_extensions: &mut ExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    match arg {
        TypeArg::Type { ty } => update_type_exts(node, ty, extensions, used_extensions)?,
        TypeArg::Sequence { elems } => {
            for elem in elems.iter_mut() {
                update_typearg_exts(node, elem, extensions, used_extensions)?;
            }
        }
        _ => {}
    }
    Ok(())
}
