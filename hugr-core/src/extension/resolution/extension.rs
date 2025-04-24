//! Resolve weak links inside `CustomType`s in an extension definition.
//!
//! This module is used when loading serialized extensions, to ensure that all
//! weak links are resolved.
#![allow(dead_code, unused_variables)]

use std::mem;
use std::sync::Arc;

use crate::extension::{Extension, ExtensionId, ExtensionRegistry, OpDef, SignatureFunc, TypeDef};

use super::types_mut::resolve_signature_exts;
use super::{ExtensionResolutionError, WeakExtensionRegistry};

impl ExtensionRegistry {
    /// Given a list of extensions that has been deserialized, create a new
    /// registry while updating any internal `Weak<Extension>` reference to
    /// point to the newly created [`Arc`]s in the registry.
    ///
    /// # Errors
    ///
    /// - If an opaque operation cannot be resolved to an extension operation.
    /// - If an extension operation references an extension that is missing from
    ///   the registry.
    /// - If a custom type references an extension that is missing from the
    ///   registry.
    pub fn new_with_extension_resolution(
        extensions: impl IntoIterator<Item = Extension>,
        other_extensions: &WeakExtensionRegistry,
    ) -> Result<ExtensionRegistry, ExtensionResolutionError> {
        Self::new_cyclic(extensions, |mut exts, weak_registry| {
            let mut weak_registry = weak_registry.clone();
            for (other_id, other) in other_extensions.iter() {
                weak_registry.register(other_id.clone(), other.clone());
            }
            for ext in &mut exts {
                ext.resolve_references(&weak_registry)?;
            }
            Ok(exts)
        })
    }
}

impl Extension {
    /// Resolve all extension references inside the extension.
    ///
    /// This is internally called when after deserializing an extension, to
    /// update all `Weak` references that were dropped from the types.
    ///
    /// This method will clone all opdef `Arc`s in the extension, so it should
    /// not be called on locally defined extensions to prevent unnecessary
    /// cloning.
    fn resolve_references(
        &mut self,
        extensions: &WeakExtensionRegistry,
    ) -> Result<(), ExtensionResolutionError> {
        let mut used_extensions = WeakExtensionRegistry::default();

        for type_def in self.types.values_mut() {
            resolve_typedef_exts(&self.name, type_def, extensions, &mut used_extensions)?;
        }

        let ops = mem::take(&mut self.operations);
        for (op_id, mut op_def) in ops {
            // TODO: We should be able to clone the definition if needed by using `make_mut`,
            // but `OpDef` does not implement clone -.-
            let op_ref = Arc::<OpDef>::get_mut(&mut op_def).expect("OpDef is not unique");
            resolve_opdef_exts(&self.name, op_ref, extensions, &mut used_extensions)?;
            self.operations.insert(op_id, op_def);
        }

        Ok(())
    }
}

/// Update all weak Extension pointers in the [`CustomType`]s inside a type
/// definition.
///
/// Adds the extensions used in the type to the `used_extensions` registry.
pub(super) fn resolve_typedef_exts(
    extension: &ExtensionId,
    def: &mut TypeDef,
    extensions: &WeakExtensionRegistry,
    used_extensions: &mut WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    match extensions.get(def.extension_id()) {
        Some(ext) => {
            *def.extension_mut() = ext.clone();
        }
        None => {
            return Err(ExtensionResolutionError::WrongTypeDefExtension {
                extension: extension.clone(),
                def: def.name().clone(),
                wrong_extension: def.extension_id().clone(),
            });
        }
    }

    Ok(())
}

/// Update all weak Extension pointers in the [`CustomType`]s inside an
/// operation definition.
///
/// Adds the extensions used in the type to the `used_extensions` registry.
pub(super) fn resolve_opdef_exts(
    extension: &ExtensionId,
    def: &mut OpDef,
    extensions: &WeakExtensionRegistry,
    used_extensions: &mut WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    match extensions.get(def.extension_id()) {
        Some(ext) => {
            *def.extension_mut() = ext.clone();
        }
        None => {
            return Err(ExtensionResolutionError::WrongOpDefExtension {
                extension: extension.clone(),
                def: def.name().clone(),
                wrong_extension: def.extension_id().clone(),
            });
        }
    }

    resolve_signature_func_exts(
        extension,
        def.signature_func_mut(),
        extensions,
        used_extensions,
    )?;

    // We ignore the lowering functions in the operation definition.
    // They may contain an unresolved hugr, but it's the responsibility of the
    // lowering call to resolve it, is it may use a different set of extensions.

    Ok(())
}

/// Update all weak Extension pointers in the [`CustomType`]s inside a
/// signature computation function.
///
/// Adds the extensions used in the type to the `used_extensions` registry.
pub(super) fn resolve_signature_func_exts(
    extension: &ExtensionId,
    signature: &mut SignatureFunc,
    extensions: &WeakExtensionRegistry,
    used_extensions: &mut WeakExtensionRegistry,
) -> Result<(), ExtensionResolutionError> {
    let signature_body = match signature {
        SignatureFunc::PolyFuncType(p) => p.body_mut(),
        SignatureFunc::CustomValidator(v) => v.poly_func_mut().body_mut(),
        SignatureFunc::MissingValidateFunc(p) => p.body_mut(),
        // Binary computation functions should always return valid types.
        SignatureFunc::CustomFunc(_) | SignatureFunc::MissingComputeFunc => {
            return Ok(());
        }
    };
    resolve_signature_exts(None, signature_body, extensions, used_extensions)
}
