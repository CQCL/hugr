//! Collect the extensions referenced inside `CustomType`s in an optype or
//! signature.
//!
//! Fails if any of the weak extension pointers have been invalidated.
//!
//! See [`super::resolve_op_types_extensions`] for a mutating version that
//! updates the weak links to point to the correct extensions.

use super::{ExtensionCollectionError, WeakExtensionRegistry};
use crate::Node;
use crate::extension::{ExtensionRegistry, ExtensionSet};
use crate::ops::{DataflowOpTrait, OpType, Value};
use crate::types::type_row::TypeRowBase;
use crate::types::{FuncTypeBase, MaybeRV, SumType, Term, TypeBase, TypeEnum};

/// Collects every extension used to define the types in an operation.
///
/// Custom types store a [`Weak`] reference to their extension, which can be
/// invalidated if the original `Arc<Extension>` is dropped. This normally
/// happens when deserializing a HUGR. On such cases, we return an error with
/// the missing extension names.
///
/// Use [`super::resolve_op_types_extensions`] instead to update the weak references and
/// ensure they point to valid extensions.
///
/// # Attributes
///
/// - `node`: The node where the operation is located, if available.
///   This is used to provide context in the error message.
/// - `op`: The operation to collect the extensions from.
pub(crate) fn collect_op_types_extensions(
    node: Option<Node>,
    op: &OpType,
) -> Result<ExtensionRegistry, ExtensionCollectionError> {
    let mut used = WeakExtensionRegistry::default();
    let mut missing = ExtensionSet::new();

    match op {
        OpType::ExtensionOp(ext) => {
            for arg in ext.args() {
                collect_term_exts(arg, &mut used, &mut missing);
            }
            collect_signature_exts(&ext.signature(), &mut used, &mut missing);
        }
        OpType::FuncDefn(f) => {
            collect_signature_exts(f.signature().body(), &mut used, &mut missing)
        }
        OpType::FuncDecl(f) => {
            collect_signature_exts(f.signature().body(), &mut used, &mut missing)
        }
        OpType::Const(c) => collect_value_exts(&c.value, &mut used, &mut missing),
        OpType::Input(inp) => collect_type_row_exts(&inp.types, &mut used, &mut missing),
        OpType::Output(out) => collect_type_row_exts(&out.types, &mut used, &mut missing),
        OpType::Call(c) => {
            collect_signature_exts(c.func_sig.body(), &mut used, &mut missing);
            collect_signature_exts(&c.instantiation, &mut used, &mut missing);
            for arg in &c.type_args {
                collect_term_exts(arg, &mut used, &mut missing);
            }
        }
        OpType::CallIndirect(c) => collect_signature_exts(&c.signature, &mut used, &mut missing),
        OpType::LoadConstant(lc) => collect_type_exts(&lc.datatype, &mut used, &mut missing),
        OpType::LoadFunction(lf) => {
            collect_signature_exts(lf.func_sig.body(), &mut used, &mut missing);
            collect_signature_exts(&lf.instantiation, &mut used, &mut missing);
            for arg in &lf.type_args {
                collect_term_exts(arg, &mut used, &mut missing);
            }
        }
        OpType::DFG(dfg) => collect_signature_exts(&dfg.signature, &mut used, &mut missing),
        OpType::OpaqueOp(op) => {
            for arg in op.args() {
                collect_term_exts(arg, &mut used, &mut missing);
            }
            collect_signature_exts(&op.signature(), &mut used, &mut missing);
        }
        OpType::Tag(t) => {
            for variant in &t.variants {
                collect_type_row_exts(variant, &mut used, &mut missing);
            }
        }
        OpType::DataflowBlock(db) => {
            collect_type_row_exts(&db.inputs, &mut used, &mut missing);
            collect_type_row_exts(&db.other_outputs, &mut used, &mut missing);
            for row in &db.sum_rows {
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
            for row in &cond.sum_rows {
                collect_type_row_exts(row, &mut used, &mut missing);
            }
            collect_type_row_exts(&cond.other_inputs, &mut used, &mut missing);
            collect_type_row_exts(&cond.outputs, &mut used, &mut missing);
        }
        OpType::Case(case) => {
            collect_signature_exts(&case.signature, &mut used, &mut missing);
        }
        // Ignore optypes that do not store a signature.
        OpType::Module(_) | OpType::AliasDecl(_) | OpType::AliasDefn(_) => {}
    }

    if missing.is_empty() {
        // We know there are no missing extensions, so this should not fail.
        Ok(used.try_into().expect("All extensions are valid"))
    } else {
        Err(ExtensionCollectionError::dropped_op_extension(
            node, op, missing,
        ))
    }
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
    used_extensions: &mut WeakExtensionRegistry,
    missing_extensions: &mut ExtensionSet,
) {
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
    used_extensions: &mut WeakExtensionRegistry,
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
pub(crate) fn collect_type_exts<RV: MaybeRV>(
    typ: &TypeBase<RV>,
    used_extensions: &mut WeakExtensionRegistry,
    missing_extensions: &mut ExtensionSet,
) {
    match typ.as_type_enum() {
        TypeEnum::Extension(custom) => {
            for arg in custom.args() {
                collect_term_exts(arg, used_extensions, missing_extensions);
            }
            let ext_ref = custom.extension_ref();
            // Check if the extension reference is still valid.
            match ext_ref.upgrade() {
                Some(ext) => {
                    used_extensions.register(ext.name().clone(), ext_ref);
                }
                None => {
                    missing_extensions.insert(custom.extension().clone());
                }
            }
        }
        TypeEnum::Function(f) => {
            collect_type_row_exts(&f.input, used_extensions, missing_extensions);
            collect_type_row_exts(&f.output, used_extensions, missing_extensions);
        }
        TypeEnum::Sum(SumType::General { rows }) => {
            for row in rows {
                collect_type_row_exts(row, used_extensions, missing_extensions);
            }
        }
        // Other types do not store extensions.
        TypeEnum::Alias(_)
        | TypeEnum::RowVar(_)
        | TypeEnum::Variable(_, _)
        | TypeEnum::Sum(SumType::Unit { .. }) => {}
    }
}

/// Collect the Extension pointers in the [`CustomType`]s inside a [`Term`].
///
/// # Attributes
///
/// - `term`: The term argument to collect the extensions from.
/// - `used_extensions`: A The registry where to store the used extensions.
/// - `missing_extensions`: A set of `ExtensionId`s of which the
///   `Weak<Extension>` pointer has been invalidated.
pub(super) fn collect_term_exts(
    term: &Term,
    used_extensions: &mut WeakExtensionRegistry,
    missing_extensions: &mut ExtensionSet,
) {
    match term {
        Term::Runtime(ty) => collect_type_exts(ty, used_extensions, missing_extensions),
        Term::ConstType(ty) => collect_type_exts(ty, used_extensions, missing_extensions),
        Term::List(elems) => {
            for elem in elems.iter() {
                collect_term_exts(elem, used_extensions, missing_extensions);
            }
        }
        Term::Tuple(elems) => {
            for elem in elems.iter() {
                collect_term_exts(elem, used_extensions, missing_extensions);
            }
        }
        Term::ListType(item_type) => {
            collect_term_exts(item_type, used_extensions, missing_extensions)
        }
        Term::TupleType(item_types) => {
            collect_term_exts(item_types, used_extensions, missing_extensions)
        }
        Term::ListConcat(lists) => {
            for list in lists {
                collect_term_exts(list, used_extensions, missing_extensions);
            }
        }
        Term::TupleConcat(tuples) => {
            for tuple in tuples {
                collect_term_exts(tuple, used_extensions, missing_extensions);
            }
        }
        Term::Variable(_)
        | Term::RuntimeType(_)
        | Term::StaticType
        | Term::BoundedNatType(_)
        | Term::StringType
        | Term::BytesType
        | Term::FloatType
        | Term::BoundedNat(_)
        | Term::String(_)
        | Term::Bytes(_)
        | Term::Float(_) => {}
    }
}

/// Collect the Extension pointers in the [`CustomType`]s inside a value.
///
/// # Attributes
///
/// - `value`: The value to collect the extensions from.
/// - `used_extensions`: A The registry where to store the used extensions.
/// - `missing_extensions`: A set of `ExtensionId`s of which the
///   `Weak<Extension>` pointer has been invalidated.
fn collect_value_exts(
    value: &Value,
    used_extensions: &mut WeakExtensionRegistry,
    missing_extensions: &mut ExtensionSet,
) {
    match value {
        Value::Extension { e } => {
            let typ = e.get_type();
            collect_type_exts(&typ, used_extensions, missing_extensions);
        }
        Value::Function { hugr: _ } => {
            // The extensions used by nested hugrs do not need to be counted for the root hugr.
        }
        Value::Sum(s) => {
            if let SumType::General { rows } = &s.sum_type {
                for row in rows {
                    collect_type_row_exts(row, used_extensions, missing_extensions);
                }
            }
            s.values
                .iter()
                .for_each(|v| collect_value_exts(v, used_extensions, missing_extensions));
        }
    }
}
