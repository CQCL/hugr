//! A trait that enum for op definitions that gathers up some shared functionality.
use std::str::FromStr;

use strum::IntoEnumIterator;

use crate::{
    ops::{custom::ExtensionOp, LeafOp, OpType},
    types::TypeArg,
    Extension,
};

use super::{op_def::SignatureFunc, ExtensionBuildError, ExtensionId, ExtensionRegistry, OpDef};
use thiserror::Error;

/// Error when definition extension does not match that of the [OpEnum]
#[derive(Debug, Error, PartialEq)]
#[error("Expected extension ID {expected} but found {provided}.")]
pub struct WrongExtension {
    expected: ExtensionId,
    provided: ExtensionId,
}

/// Error loading [OpEnum]
#[derive(Debug, Error, PartialEq)]
#[error("{0}")]
#[allow(missing_docs)]
pub enum OpLoadError<T> {
    WrongExtension(#[from] WrongExtension),
    LoadError(T),
}

/// A trait that operation sets defined by simple (C-style) enums can implement
/// to simplify interactions with the extension.
/// Relies on `strum_macros::{EnumIter, EnumString, IntoStaticStr}`
pub trait OpEnum: Into<&'static str> + FromStr + Copy + IntoEnumIterator {
    /// The name of the extension these ops belong to.
    const EXTENSION_ID: ExtensionId;

    // TODO can be removed after rust 1.75
    /// Error thrown when loading from string fails.
    type LoadError: std::error::Error;
    /// Description type.
    type Description: ToString;

    /// Return the signature (polymorphic function type) of the operation.
    fn signature(&self) -> SignatureFunc;

    /// Description of the operation.
    fn description(&self) -> Self::Description;

    /// Edit the opdef before finalising.
    fn post_opdef(&self, _def: &mut OpDef) {}

    /// Name of the operation - derived from strum serialization.
    fn name(&self) -> &str {
        (*self).into()
    }

    /// Load an operation from the name of the operation.
    fn from_extension_name(op_name: &str) -> Result<Self, Self::LoadError>;

    /// Try to load one of the operations of this set from an [OpDef].
    fn try_from_op_def(op_def: &OpDef) -> Result<Self, OpLoadError<Self::LoadError>> {
        if op_def.extension() != &Self::EXTENSION_ID {
            return Err(WrongExtension {
                expected: Self::EXTENSION_ID.clone(),
                provided: op_def.extension().clone(),
            }
            .into());
        }
        Self::from_extension_name(op_def.name()).map_err(OpLoadError::LoadError)
    }

    /// Add an operation to an extension.
    fn add_to_extension<'e>(
        &self,
        ext: &'e mut Extension,
    ) -> Result<&'e OpDef, ExtensionBuildError> {
        let def = ext.add_op(
            self.name().into(),
            self.description().to_string(),
            self.signature(),
        )?;

        self.post_opdef(def);

        Ok(def)
    }

    /// Iterator over all operations in the set.
    fn all_variants() -> <Self as IntoEnumIterator>::Iterator {
        <Self as IntoEnumIterator>::iter()
    }

    /// load all variants of a `SimpleOpEnum` in to an extension as op defs.
    fn load_all_ops(extension: &mut Extension) -> Result<(), ExtensionBuildError> {
        for op in Self::all_variants() {
            op.add_to_extension(extension)?;
        }
        Ok(())
    }

    /// Generate an [OpType].
    fn to_optype(
        &self,
        extension: &Extension,
        args: &[TypeArg],
        exts: &ExtensionRegistry,
    ) -> Option<OpType> {
        let leaf: LeafOp = ExtensionOp::new(extension.get_op(self.name())?.clone(), args, exts)
            .ok()?
            .into();

        Some(leaf.into())
    }

    /// Try to instantiate a variant from an [OpType]. Default behaviour assumes
    /// an [ExtensionOp] and loads from the name.
    fn from_optype(op: &OpType) -> Option<Self> {
        let ext: &ExtensionOp = op.as_leaf_op()?.as_extension_op()?;
        Self::try_from_op_def(ext.def()).ok()
    }
}

#[cfg(test)]
mod test {
    use crate::{extension::EMPTY_REG, type_row, types::FunctionType};

    use super::*;
    use strum_macros::{EnumIter, EnumString, IntoStaticStr};
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
    enum DummyEnum {
        Dumb,
    }
    #[derive(Debug, thiserror::Error, PartialEq)]
    #[error("Dummy")]
    struct DummyError;
    impl OpEnum for DummyEnum {
        const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("dummy");

        type LoadError = DummyError;

        type Description = &'static str;

        fn signature(&self) -> SignatureFunc {
            FunctionType::new_endo(type_row![]).into()
        }

        fn description(&self) -> Self::Description {
            "dummy"
        }

        fn from_extension_name(_op_name: &str) -> Result<Self, Self::LoadError> {
            Ok(Self::Dumb)
        }
    }

    #[test]
    fn test_dummy_enum() {
        let o = DummyEnum::Dumb;

        let good_name = ExtensionId::new("dummy").unwrap();
        let mut e = Extension::new(good_name.clone());

        o.add_to_extension(&mut e).unwrap();

        assert_eq!(
            DummyEnum::try_from_op_def(e.get_op(o.name()).unwrap()).unwrap(),
            o
        );

        assert_eq!(
            DummyEnum::from_optype(&o.to_optype(&e, &[], &EMPTY_REG).unwrap()).unwrap(),
            o
        );
        let bad_name = ExtensionId::new("not_dummy").unwrap();
        let mut e = Extension::new(bad_name.clone());

        o.add_to_extension(&mut e).unwrap();

        assert_eq!(
            DummyEnum::try_from_op_def(e.get_op(o.name()).unwrap()),
            Err(OpLoadError::WrongExtension(WrongExtension {
                expected: good_name,
                provided: bad_name
            }))
        );
    }
}
