//! Collect the extensions referenced inside `CustomType`s in an optype or
//! signature.
//!
//! Fails if any of the weak extension pointers have been invalidated.
//!
//! See [`super::update_op_types_extensions`] for a mutating version that
//! updates the weak links to point to the correct extensions.

use super::ExtensionCollectionError;
use crate::extension::{ExtensionRegistry, ExtensionSet};
use crate::ops::{DataflowOpTrait, OpType};
use crate::types::type_row::TypeRowBase;
use crate::types::{FuncTypeBase, MaybeRV, SumType, TypeArg, TypeBase, TypeEnum};
use crate::Node;

/// Collects every extension used te define the types in an operation.
///
/// Custom types store a [`Weak`] reference to their extension, which can be
/// invalidated if the original `Arc<Extension>` is dropped. This normally
/// happens when deserializing a HUGR. On such cases, we return an error with
/// the missing extension names.
///
/// Use [`collect_op_types_extensions`] instead to update the weak references and
/// ensure they point to valid extensions.
///
/// # Attributes
///
/// - `node`: The node where the operation is located, if available.
///   This is used to provide context in the error message.
/// - `op`: The operation to collect the extensions from.
pub fn collect_op_types_extensions(
    node: Option<Node>,
    op: &OpType,
) -> Result<ExtensionRegistry, ExtensionCollectionError> {
    let mut used = ExtensionRegistry::default();
    let mut missing = ExtensionSet::new();

    match op {
        OpType::ExtensionOp(ext) => {
            for arg in ext.args() {
                collect_typearg_exts(arg, &mut used, &mut missing);
            }
            collect_signature_exts(&ext.signature(), &mut used, &mut missing)
        }
        OpType::FuncDefn(f) => collect_signature_exts(f.signature.body(), &mut used, &mut missing),
        OpType::FuncDecl(f) => collect_signature_exts(f.signature.body(), &mut used, &mut missing),
        OpType::Const(c) => {
            let typ = c.get_type();
            collect_type_exts(&typ, &mut used, &mut missing);
        }
        OpType::Input(inp) => collect_type_row_exts(&inp.types, &mut used, &mut missing),
        OpType::Output(out) => collect_type_row_exts(&out.types, &mut used, &mut missing),
        OpType::Call(c) => {
            collect_signature_exts(c.func_sig.body(), &mut used, &mut missing);
            collect_signature_exts(&c.instantiation, &mut used, &mut missing);
        }
        OpType::CallIndirect(c) => collect_signature_exts(&c.signature, &mut used, &mut missing),
        OpType::LoadConstant(lc) => collect_type_exts(&lc.datatype, &mut used, &mut missing),
        OpType::LoadFunction(lf) => {
            collect_signature_exts(lf.func_sig.body(), &mut used, &mut missing);
            collect_signature_exts(&lf.signature, &mut used, &mut missing);
        }
        OpType::DFG(dfg) => collect_signature_exts(&dfg.signature, &mut used, &mut missing),
        OpType::OpaqueOp(op) => {
            for arg in op.args() {
                collect_typearg_exts(arg, &mut used, &mut missing);
            }
            collect_signature_exts(&op.signature(), &mut used, &mut missing)
        }
        OpType::Tag(t) => {
            for variant in t.variants.iter() {
                collect_type_row_exts(variant, &mut used, &mut missing)
            }
        }
        OpType::DataflowBlock(db) => {
            collect_type_row_exts(&db.inputs, &mut used, &mut missing);
            collect_type_row_exts(&db.other_outputs, &mut used, &mut missing);
            for row in db.sum_rows.iter() {
                collect_type_row_exts(row, &mut used, &mut missing);
            }
        }
        OpType::ExitBlock(e) => {
            collect_type_row_exts(&e.cfg_outputs, &mut used, &mut missing);
        }
        OpType::TailLoop(tl) => {
            collect_type_row_exts(&tl.just_inputs, &mut used, &mut missing);
            collect_type_row_exts(&tl.just_outputs, &mut used, &mut missing);
            collect_type_row_exts(&tl.rest, &mut used, &mut missing);
        }
        OpType::CFG(cfg) => {
            collect_signature_exts(&cfg.signature, &mut used, &mut missing);
        }
        OpType::Conditional(cond) => {
            for row in cond.sum_rows.iter() {
                collect_type_row_exts(row, &mut used, &mut missing);
            }
            collect_type_row_exts(&cond.other_inputs, &mut used, &mut missing);
            collect_type_row_exts(&cond.outputs, &mut used, &mut missing);
        }
        OpType::Case(case) => {
            collect_signature_exts(&case.signature, &mut used, &mut missing);
        }
        // Ignore optypes that do not store a signature.
        _ => {}
    };

    missing
        .is_empty()
        .then_some(used)
        .ok_or(ExtensionCollectionError::dropped_op_extension(
            node, op, missing,
        ))
}

/// Collect the Extension pointers in the [`CustomType`]s inside a signature.
///
/// # Attributes
///
/// - `signature`: The signature to collect the extensions from.
/// - `used_extensions`: A The registry where to store the used extensions.
/// - `missing_extensions`: A set of `ExtensionId`s of which the
///   `Weak<Extension>` pointer has been invalidated.
pub(crate) fn collect_signature_exts<RV: MaybeRV>(
    signature: &FuncTypeBase<RV>,
    used_extensions: &mut ExtensionRegistry,
    missing_extensions: &mut ExtensionSet,
) {
    // Note that we do not include the signature's `extension_reqs` here, as those refer
    // to _runtime_ requirements that may not be currently present.
    // See https://github.com/CQCL/hugr/issues/1734
    // TODO: Update comment once that issue gets implemented.
    collect_type_row_exts(&signature.input, used_extensions, missing_extensions);
    collect_type_row_exts(&signature.output, used_extensions, missing_extensions);
}

/// Collect the Extension pointers in the [`CustomType`]s inside a type row.
///
/// # Attributes
///
/// - `row`: The type row to collect the extensions from.
/// - `used_extensions`: A The registry where to store the used extensions.
/// - `missing_extensions`: A set of `ExtensionId`s of which the
///   `Weak<Extension>` pointer has been invalidated.
fn collect_type_row_exts<RV: MaybeRV>(
    row: &TypeRowBase<RV>,
    used_extensions: &mut ExtensionRegistry,
    missing_extensions: &mut ExtensionSet,
) {
    for ty in row.iter() {
        collect_type_exts(ty, used_extensions, missing_extensions);
    }
}

/// Collect the Extension pointers in the [`CustomType`]s inside a type.
///
/// # Attributes
///
/// - `typ`: The type to collect the extensions from.
/// - `used_extensions`: A The registry where to store the used extensions.
/// - `missing_extensions`: A set of `ExtensionId`s of which the
///   `Weak<Extension>` pointer has been invalidated.
pub(super) fn collect_type_exts<RV: MaybeRV>(
    typ: &TypeBase<RV>,
    used_extensions: &mut ExtensionRegistry,
    missing_extensions: &mut ExtensionSet,
) {
    match typ.as_type_enum() {
        TypeEnum::Extension(custom) => {
            for arg in custom.args() {
                collect_typearg_exts(arg, used_extensions, missing_extensions);
            }
            match custom.extension_ref().upgrade() {
                Some(ext) => {
                    // The extension pointer is still valid.
                    used_extensions.register_updated(ext);
                }
                None => {
                    // The extension has been dropped.
                    // Register it in the missing set.
                    missing_extensions.insert(custom.extension().clone());
                }
            }
        }
        TypeEnum::Function(f) => {
            collect_type_row_exts(&f.input, used_extensions, missing_extensions);
            collect_type_row_exts(&f.output, used_extensions, missing_extensions);
        }
        TypeEnum::Sum(SumType::General { rows }) => {
            for row in rows.iter() {
                collect_type_row_exts(row, used_extensions, missing_extensions);
            }
        }
        _ => {}
    }
}

/// Collect the Extension pointers in the [`CustomType`]s inside a type argument.
///
/// # Attributes
///
/// - `arg`: The type argument to collect the extensions from.
/// - `used_extensions`: A The registry where to store the used extensions.
/// - `missing_extensions`: A set of `ExtensionId`s of which the
///   `Weak<Extension>` pointer has been invalidated.
fn collect_typearg_exts(
    arg: &TypeArg,
    used_extensions: &mut ExtensionRegistry,
    missing_extensions: &mut ExtensionSet,
) {
    match arg {
        TypeArg::Type { ty } => collect_type_exts(ty, used_extensions, missing_extensions),
        TypeArg::Sequence { elems } => {
            for elem in elems.iter() {
                collect_typearg_exts(elem, used_extensions, missing_extensions);
            }
        }
        _ => {}
    }
}
