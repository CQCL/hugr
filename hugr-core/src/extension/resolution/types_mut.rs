//! Resolve weak links inside `CustomType`s in an optype's signature, while
//! collecting all used extensions.
//!
//! For a non-mutating option see [`super::collect_op_types_extensions`].

use std::sync::Weak;

use super::types::collect_type_exts;
use super::{ExtensionResolutionError, WeakExtensionRegistry};
use crate::extension::ExtensionSet;
use crate::ops::{OpType, Value};
use crate::types::type_row::TypeRowBase;
use crate::types::{CustomType, FuncTypeBase, MaybeRV, SumType, TypeArg, TypeBase, TypeEnum};
use crate::{Extension, Node};

/// Replace the dangling extension pointer in the [`CustomType`]s inside an
/// optype with a valid pointer to the extension in the `extensions`
/// registry.
///
/// Returns an iterator over the used extensions.
///
/// This is a helper function used right after deserializing a Hugr.
pub fn resolve_op_types_extensions(
    node: Option<Node>,
    op: &mut OpType,
    extensions: &WeakExtensionRegistry,
) -> Result<impl Iterator<Item = Weak<Extension>> + use<>, ExtensionResolutionError> {
    let mut used = WeakExtensionRegistry::default();
    let used_extensions = &mut used;
    match op {
        OpType::ExtensionOp(ext) => {
            for arg in ext.args_mut() {
                resolve_typearg_exts(node, arg, extensions, used_extensions)?;
            }
            resolve_signature_exts(node, ext.signature_mut(), extensions, used_extensions)?
        }
        OpType::FuncDefn(f) => {
            resolve_signature_exts(node, f.signature.body_mut(), extensions, used_extensions)?
        }
        OpType::FuncDecl(f) => {
            resolve_signature_exts(node, f.signature.body_mut(), extensions, used_extensions)?
        }
        OpType::Const(c) => resolve_value_exts(node, &mut c.value, extensions, used_extensions)?,
        OpType::Input(inp) => {
            resolve_type_row_exts(node, &mut inp.types, extensions, used_extensions)?
        }
        OpType::Output(out) => {
            resolve_type_row_exts(node, &mut out.types, extensions, used_extensions)?
        }
        OpType::Call(c) => {
            resolve_signature_exts(node, c.func_sig.body_mut(), extensions, used_extensions)?;
            resolve_signature_exts(node, &mut c.instantiation, extensions, used_extensions)?;
            for arg in &mut c.type_args {
                resolve_typearg_exts(node, arg, extensions, used_extensions)?;
            }
        }
        OpType::CallIndirect(c) => {
            resolve_signature_exts(node, &mut c.signature, extensions, used_extensions)?
        }
        OpType::LoadConstant(lc) => {
            resolve_type_exts(node, &mut lc.datatype, extensions, used_extensions)?
        }
        OpType::LoadFunction(lf) => {
            resolve_signature_exts(node, lf.func_sig.body_mut(), extensions, used_extensions)?;
            resolve_signature_exts(node, &mut lf.instantiation, extensions, used_extensions)?;
            for arg in &mut lf.type_args {
                resolve_typearg_exts(node, arg, extensions, used_extensions)?;
            }
        }
        OpType::DFG(dfg) => {
            resolve_signature_exts(node, &mut dfg.signature, extensions, used_extensions)?
        }
        OpType::OpaqueOp(op) => {
            for arg in op.args_mut() {
                resolve_typearg_exts(node, arg, extensions, used_extensions)?;
            }
            resolve_signature_exts(node, op.signature_mut(), extensions, used_extensions)?
        }
        OpType::Tag(t) => {
            for variant in t.variants.iter_mut() {
                resolve_type_row_exts(node, variant, extensions, used_extensions)?
            }
        }
        OpType::DataflowBlock(db) => {
            resolve_type_row_exts(node, &mut db.inputs, extensions, used_extensions)?;
            resolve_type_row_exts(node, &mut db.other_outputs, extensions, used_extensions)?;
            for row in db.sum_rows.iter_mut() {
                resolve_type_row_exts(node, row, extensions, used_extensions)?;
            }
        }
        OpType::ExitBlock(e) => {
            resolve_type_row_exts(node, &mut e.cfg_outputs, extensions, used_extensions)?;
        }
        OpType::TailLoop(tl) => {
            resolve_type_row_exts(node, &mut tl.just_inputs, extensions, used_extensions)?;
            resolve_type_row_exts(node, &mut tl.just_outputs, extensions, used_extensions)?;
            resolve_type_row_exts(node, &mut tl.rest, extensions, used_extensions)?;
        }
        OpType::CFG(cfg) => {
            resolve_signature_exts(node, &mut cfg.signature, extensions, used_extensions)?;
        }
        OpType::Conditional(cond) => {
            for row in cond.sum_rows.iter_mut() {
                resolve_type_row_exts(node, row, extensions, used_extensions)?;
            }
            resolve_type_row_exts(node, &mut cond.other_inputs, extensions, used_extensions)?;
            resolve_type_row_exts(node, &mut cond.outputs, extensions, used_extensions)?;
        }
        OpType::Case(case) => {
            resolve_signature_exts(node, &mut case.signature, extensions, used_extensions)?;
        }
        // Ignore optypes that do not store a signature.
        OpType::Module(_) | OpType::AliasDecl(_) | OpType::AliasDefn(_) => {}
    }
    Ok(used.into_iter())
}

/// Update all weak Extension pointers in the [`CustomType`]s inside a signature.
///
/// Adds the extensions used in the signature to the `used_extensions` registry.
pub(super) fn resolve_signature_exts<RV: MaybeRV>(
    node: Option<Node>,
    signature: &mut FuncTypeBase<RV>,
    extensions: &WeakExtensionRegistry,
    used_extensions: &mut WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    resolve_type_row_exts(node, &mut signature.input, extensions, used_extensions)?;
    resolve_type_row_exts(node, &mut signature.output, extensions, used_extensions)?;
    Ok(())
}

/// Update all weak Extension pointers in the [`CustomType`]s inside a type row.
///
/// Adds the extensions used in the row to the `used_extensions` registry.
pub(super) fn resolve_type_row_exts<RV: MaybeRV>(
    node: Option<Node>,
    row: &mut TypeRowBase<RV>,
    extensions: &WeakExtensionRegistry,
    used_extensions: &mut WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    for ty in row.iter_mut() {
        resolve_type_exts(node, ty, extensions, used_extensions)?;
    }
    Ok(())
}

/// Update all weak Extension pointers in the [`CustomType`]s inside a type.
///
/// Adds the extensions used in the type to the `used_extensions` registry.
pub(super) fn resolve_type_exts<RV: MaybeRV>(
    node: Option<Node>,
    typ: &mut TypeBase<RV>,
    extensions: &WeakExtensionRegistry,
    used_extensions: &mut WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    match typ.as_type_enum_mut() {
        TypeEnum::Extension(custom) => {
            resolve_custom_type_exts(node, custom, extensions, used_extensions)?;
        }
        TypeEnum::Function(f) => {
            resolve_type_row_exts(node, &mut f.input, extensions, used_extensions)?;
            resolve_type_row_exts(node, &mut f.output, extensions, used_extensions)?;
        }
        TypeEnum::Sum(SumType::General { rows }) => {
            for row in rows.iter_mut() {
                resolve_type_row_exts(node, row, extensions, used_extensions)?;
            }
        }
        // Other types do not store extensions.
        TypeEnum::Alias(_)
        | TypeEnum::RowVar(_)
        | TypeEnum::Variable(_, _)
        | TypeEnum::Sum(SumType::Unit { .. }) => {}
    }
    Ok(())
}

/// Update all weak Extension pointers in a [`CustomType`].
///
/// Adds the extensions used in the type to the `used_extensions` registry.
pub(super) fn resolve_custom_type_exts(
    node: Option<Node>,
    custom: &mut CustomType,
    extensions: &WeakExtensionRegistry,
    used_extensions: &mut WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    for arg in custom.args_mut() {
        resolve_typearg_exts(node, arg, extensions, used_extensions)?;
    }

    let ext_id = custom.extension();
    let ext = extensions.get(ext_id).ok_or_else(|| {
        ExtensionResolutionError::missing_type_extension(node, custom.name(), ext_id, extensions)
    })?;

    // Add the extension to the used extensions registry,
    // and update the CustomType with the valid pointer.
    used_extensions.register(ext_id.clone(), ext.clone());
    custom.update_extension(ext.clone());

    Ok(())
}

/// Update all weak Extension pointers in the [`CustomType`]s inside a type arg.
///
/// Adds the extensions used in the type to the `used_extensions` registry.
pub(super) fn resolve_typearg_exts(
    node: Option<Node>,
    arg: &mut TypeArg,
    extensions: &WeakExtensionRegistry,
    used_extensions: &mut WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    match arg {
        TypeArg::Type { ty } => resolve_type_exts(node, ty, extensions, used_extensions)?,
        TypeArg::Sequence { elems } => {
            for elem in elems.iter_mut() {
                resolve_typearg_exts(node, elem, extensions, used_extensions)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Update all weak Extension pointers in the [`CustomType`]s inside a [`Value`].
///
/// Adds the extensions used in the row to the `used_extensions` registry.
pub(super) fn resolve_value_exts(
    node: Option<Node>,
    value: &mut Value,
    extensions: &WeakExtensionRegistry,
    used_extensions: &mut WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    match value {
        Value::Extension { e } => {
            e.value_mut().update_extensions(extensions)?;

            // We expect that the `CustomConst::get_type` binary calls
            // return types with valid extensions after we call `update_extensions`.
            let typ = e.get_type();
            let mut missing = ExtensionSet::new();
            collect_type_exts(&typ, used_extensions, &mut missing);
            if !missing.is_empty() {
                return Err(ExtensionResolutionError::InvalidConstTypes {
                    value: e.name(),
                    missing_extensions: missing,
                });
            }
        }
        Value::Function { hugr } => {
            // We don't need to add the nested hugr's extensions to the main one here,
            // but we run resolution on it independently.
            if let Ok(exts) = extensions.try_into() {
                hugr.resolve_extension_defs(&exts)?;
            }
        }
        Value::Sum(s) => {
            if let SumType::General { rows } = &mut s.sum_type {
                for row in rows.iter_mut() {
                    resolve_type_row_exts(node, row, extensions, used_extensions)?;
                }
            }
            s.values
                .iter_mut()
                .try_for_each(|v| resolve_value_exts(node, v, extensions, used_extensions))?;
        }
    }
    Ok(())
}
