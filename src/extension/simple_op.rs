//! A trait that enum for op definitions that gathers up some shared functionality.

use strum::IntoEnumIterator;

use crate::{
    ops::{custom::ExtensionOp, LeafOp, OpType},
    types::{FunctionType, TypeArg},
    Extension,
};

use super::{
    op_def::SignatureFunc, ExtensionBuildError, ExtensionId, ExtensionRegistry, OpDef,
    SignatureError,
};
use delegate::delegate;
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
pub enum OpLoadError {
    WrongExtension(#[from] WrongExtension),
    #[error("Op with name {0} is not a member of this enum.")]
    NotEnumMember(String),
    #[error("Type args invalid: {0}.")]
    InvalidArgs(#[from] SignatureError),
}

pub trait OpEnumName {
    fn name(&self) -> &str;
}

impl<T> OpEnumName for T
where
    for<'a> &'a T: Into<&'static str>,
{
    fn name(&self) -> &str {
        self.into()
    }
}
/// A trait that operation sets defined by simple (C-style) enums can implement
/// to simplify interactions with the extension.
/// Relies on `strum_macros::{EnumIter, EnumString, IntoStaticStr}`
pub trait OpEnum: OpEnumName {
    /// Description type.
    type Description: ToString;

    /// Return the signature (polymorphic function type) of the operation.
    fn def_signature(&self) -> SignatureFunc;

    /// Description of the operation.
    fn description(&self) -> Self::Description;

    /// Any type args which define this operation. Default is no type arguments.
    fn type_args(&self) -> Vec<TypeArg> {
        vec![]
    }

    /// Edit the opdef before finalising. By default does nothing.
    fn post_opdef(&self, _def: &mut OpDef) {}

    /// Try to load one of the operations of this set from an [OpDef].
    fn from_op_def(op_def: &OpDef, args: &[TypeArg]) -> Result<Self, OpLoadError>
    where
        Self: Sized;

    /// Try to instantiate a variant from an [OpType]. Default behaviour assumes
    /// an [ExtensionOp] and loads from the name.
    fn from_optype(op: &OpType) -> Option<Self>
    where
        Self: Sized,
    {
        let ext: &ExtensionOp = op.as_leaf_op()?.as_extension_op()?;
        Self::from_op_def(ext.def(), ext.args()).ok()
    }

    fn to_registered<'r>(
        self,
        extension_id: ExtensionId,
        registry: &'r ExtensionRegistry,
    ) -> RegisteredEnum<'r, Self>
    where
        Self: Sized,
    {
        RegisteredEnum {
            extension_id,
            registry,
            op_enum: self,
        }
    }

    /// Iterator over all operations in the set. Non-trivial variants will have
    /// default values used for the members.
    fn all_variants() -> <Self as IntoEnumIterator>::Iterator
    where
        Self: IntoEnumIterator,
    {
        <Self as IntoEnumIterator>::iter()
    }

    /// load all variants of a [OpEnum] in to an extension as op defs.
    fn load_all_ops(extension: &mut Extension) -> Result<(), ExtensionBuildError>
    where
        Self: IntoEnumIterator,
    {
        for op in Self::all_variants() {
            extension.add_op_enum(&op)?;
        }
        Ok(())
    }
}

/// Load an [OpEnum] from its name. Works best for C-style enums where each
/// variant corresponds to an [OpDef] and an [OpType], i,e, there are no type parameters.
/// See [strum_macros::EnumString].
pub fn try_from_name<T>(name: &str) -> Result<T, OpLoadError>
where
    T: std::str::FromStr + OpEnum,
{
    T::from_str(name).map_err(|_| OpLoadError::NotEnumMember(name.to_string()))
}

/// Wrap an [OpEnum] with an extension registry to allow type computation.
/// Generate from [OpEnum::to_registered]
pub struct RegisteredEnum<'r, T> {
    /// The name of the extension these ops belong to.
    extension_id: ExtensionId,
    /// A registry of all extensions, used for type computation.
    registry: &'r ExtensionRegistry,
    /// The inner [OpEnum]
    op_enum: T,
}

impl<'a, T: OpEnum> RegisteredEnum<'a, T> {
    /// Generate an [OpType].
    pub fn to_optype(&self) -> Option<OpType> {
        let leaf: LeafOp = ExtensionOp::new(
            self.registry
                .get(&self.extension_id)?
                .get_op(self.name())?
                .clone(),
            self.type_args(),
            self.registry,
        )
        .ok()?
        .into();

        Some(leaf.into())
    }

    /// Compute the [FunctionType] for this operation, instantiating with type arguments.
    pub fn function_type(&self) -> Result<FunctionType, SignatureError> {
        self.op_enum.def_signature().compute_signature(
            self.registry
                .get(&self.extension_id)
                .expect("should return 'Extension not in registry' error here.")
                .get_op(self.name())
                .expect("should return 'Op not in extension' error here."),
            &self.type_args(),
            self.registry,
        )
    }
    delegate! {
        to self.op_enum {
            /// Name of the operation - derived from strum serialization.
            pub fn name(&self) -> &str;
            /// Any type args which define this operation. Default is no type arguments.
            pub fn type_args(&self) -> Vec<TypeArg>;
            /// Description of the operation.
            pub fn description(&self) -> T::Description;
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{type_row, types::FunctionType};

    use super::*;
    use strum_macros::{EnumIter, EnumString, IntoStaticStr};
    #[derive(Clone, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
    enum DummyEnum {
        Dumb,
    }
    #[derive(Debug, thiserror::Error, PartialEq)]
    #[error("Dummy")]
    struct DummyError;
    impl OpEnum for DummyEnum {
        type Description = &'static str;

        fn def_signature(&self) -> SignatureFunc {
            FunctionType::new_endo(type_row![]).into()
        }

        fn description(&self) -> Self::Description {
            "dummy"
        }

        fn from_op_def(_op_def: &OpDef, _args: &[TypeArg]) -> Result<Self, OpLoadError> {
            Ok(Self::Dumb)
        }
    }

    #[test]
    fn test_dummy_enum() {
        let o = DummyEnum::Dumb;

        let ext_name = ExtensionId::new("dummy").unwrap();
        let mut e = Extension::new(ext_name.clone());

        e.add_op_enum(&o).unwrap();

        assert_eq!(
            DummyEnum::from_op_def(e.get_op(o.name()).unwrap(), &[]).unwrap(),
            o
        );

        assert_eq!(
            DummyEnum::from_optype(
                &o.clone()
                    .to_registered(
                        ext_name,
                        &ExtensionRegistry::try_new([e.to_owned()]).unwrap()
                    )
                    .to_optype()
                    .unwrap()
            )
            .unwrap(),
            o
        );
    }
}
