//! A trait that enum for op definitions that gathers up some shared functionality.

use strum::IntoEnumIterator;

use crate::ops::{OpName, OpNameRef};
use crate::{
    ops::{custom::ExtensionOp, NamedOp, OpType},
    types::TypeArg,
    Extension,
};

use super::{
    op_def::SignatureFunc, ExtensionBuildError, ExtensionId, ExtensionRegistry, OpDef,
    SignatureError,
};
use delegate::delegate;
use thiserror::Error;

/// Error loading operation.
#[derive(Debug, Error, PartialEq)]
#[error("{0}")]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum OpLoadError {
    #[error("Op with name {0} is not a member of this set.")]
    NotMember(String),
    #[error("Type args invalid: {0}.")]
    InvalidArgs(#[from] SignatureError),
    #[error("OpDef belongs to extension {0}, expected {1}.")]
    WrongExtension(ExtensionId, ExtensionId),
}

impl<T> NamedOp for T
where
    for<'a> &'a T: Into<&'static str>,
{
    fn name(&self) -> OpName {
        let s = self.into();
        s.into()
    }
}

/// Traits implemented by types which can add themselves to [`Extension`]s as
/// [`OpDef`]s or load themselves from an [`OpDef`].
/// Particularly useful with C-style enums that implement [strum::IntoEnumIterator],
/// as then all definitions can be added to an extension at once.
pub trait MakeOpDef: NamedOp {
    /// Try to load one of the operations of this set from an [OpDef].
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized;

    /// Return the signature (polymorphic function type) of the operation.
    fn signature(&self) -> SignatureFunc;

    /// The ID of the extension this operation is defined in.
    fn extension_id() -> ExtensionId;

    /// Description of the operation. By default, the same as `self.name()`.
    fn description(&self) -> String {
        self.name().to_string()
    }

    /// Edit the opdef before finalising. By default does nothing.
    fn post_opdef(&self, _def: &mut OpDef) {}

    /// Add an operation implemented as an [MakeOpDef], which can provide the data
    /// required to define an [OpDef], to an extension.
    fn add_to_extension(&self, extension: &mut Extension) -> Result<(), ExtensionBuildError> {
        let def = extension.add_op(self.name(), self.description(), self.signature())?;

        self.post_opdef(def);

        Ok(())
    }

    /// Load all variants of an enum of op definitions in to an extension as op defs.
    /// See [strum::IntoEnumIterator].
    fn load_all_ops(extension: &mut Extension) -> Result<(), ExtensionBuildError>
    where
        Self: IntoEnumIterator,
    {
        for op in Self::iter() {
            op.add_to_extension(extension)?;
        }
        Ok(())
    }
}

/// Traits implemented by types which can be loaded from [`ExtensionOp`]s,
/// i.e. concrete instances of [`OpDef`]s, with defined type arguments.
pub trait MakeExtensionOp: NamedOp {
    /// Try to load one of the operations of this set from an [OpDef].
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized;
    /// Try to instantiate a variant from an [OpType]. Default behaviour assumes
    /// an [ExtensionOp] and loads from the name.
    fn from_optype(op: &OpType) -> Option<Self>
    where
        Self: Sized,
    {
        let ext: &ExtensionOp = op.as_custom_op()?.as_extension_op()?;
        Self::from_extension_op(ext).ok()
    }

    /// Any type args which define this operation.
    fn type_args(&self) -> Vec<TypeArg>;

    /// Given the ID of the extension this operation is defined in, and a
    /// registry containing that extension, return a [RegisteredOp].
    fn to_registered(
        self,
        extension_id: ExtensionId,
        registry: &ExtensionRegistry,
    ) -> RegisteredOp<'_, Self>
    where
        Self: Sized,
    {
        RegisteredOp {
            extension_id,
            registry,
            op: self,
        }
    }
}

/// Blanket implementation for non-polymorphic operations - [OpDef]s with no type parameters.
impl<T: MakeOpDef> MakeExtensionOp for T {
    #[inline]
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        Self::from_def(ext_op.def())
    }

    #[inline]
    fn type_args(&self) -> Vec<TypeArg> {
        vec![]
    }
}

/// Load an [MakeOpDef] from its name.
/// See [strum_macros::EnumString].
pub fn try_from_name<T>(def: &OpDef) -> Result<T, OpLoadError>
where
    T: std::str::FromStr + MakeOpDef,
{
    let expected_extension = T::extension_id();
    if def.extension() != &expected_extension {
        return Err(OpLoadError::WrongExtension(
            def.extension().clone(),
            expected_extension,
        ));
    }
    let name: &OpNameRef = def.name();

    T::from_str(name).map_err(|_| OpLoadError::NotMember(name.to_string()))
}

/// Wrap an [MakeExtensionOp] with an extension registry to allow type computation.
/// Generate from [MakeExtensionOp::to_registered]
#[derive(Clone, Debug)]
pub struct RegisteredOp<'r, T> {
    /// The name of the extension these ops belong to.
    extension_id: ExtensionId,
    /// A registry of all extensions, used for type computation.
    registry: &'r ExtensionRegistry,
    /// The inner [MakeExtensionOp]
    op: T,
}

impl<T> RegisteredOp<'_, T> {
    /// Extract the inner wrapped value
    pub fn to_inner(self) -> T {
        self.op
    }
}

impl<T: MakeExtensionOp> RegisteredOp<'_, T> {
    /// Generate an [OpType].
    pub fn to_extension_op(&self) -> Option<ExtensionOp> {
        ExtensionOp::new(
            self.registry
                .get(&self.extension_id)?
                .get_op(&self.name())?
                .clone(),
            self.type_args(),
            self.registry,
        )
        .ok()
    }

    delegate! {
        to self.op {
            /// Name of the operation - derived from strum serialization.
            pub fn name(&self) -> OpName;
            /// Any type args which define this operation. Default is no type arguments.
            pub fn type_args(&self) -> Vec<TypeArg>;
        }
    }
}

/// Trait for operations that can self report the extension ID they belong to
/// and the registry required to compute their types.
/// Allows conversion to [`ExtensionOp`]
pub trait MakeRegisteredOp: MakeExtensionOp {
    /// The ID of the extension this op belongs to.
    fn extension_id(&self) -> ExtensionId;
    /// A reference to an [ExtensionRegistry] which is sufficient to generate
    /// the signature of this op.
    fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry;

    /// Convert this operation in to an [ExtensionOp]. Returns None if the type
    /// cannot be computed.
    fn to_extension_op(self) -> Option<ExtensionOp>
    where
        Self: Sized,
    {
        let registered: RegisteredOp<_> = self.into();
        registered.to_extension_op()
    }
}

impl<T: MakeRegisteredOp> From<T> for RegisteredOp<'_, T> {
    fn from(ext_op: T) -> Self {
        let extension_id = ext_op.extension_id();
        let registry = ext_op.registry();
        ext_op.to_registered(extension_id, registry)
    }
}

impl<T: MakeRegisteredOp> From<T> for OpType {
    /// Convert
    fn from(ext_op: T) -> Self {
        ext_op.to_extension_op().unwrap().into()
    }
}

#[cfg(test)]
mod test {
    use crate::{const_extension_ids, type_row, types::FunctionType};

    use super::*;
    use lazy_static::lazy_static;
    use strum_macros::{EnumIter, EnumString, IntoStaticStr};

    #[derive(Clone, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
    enum DummyEnum {
        Dumb,
    }

    impl MakeOpDef for DummyEnum {
        fn signature(&self) -> SignatureFunc {
            FunctionType::new_endo(type_row![]).into()
        }

        fn from_def(_op_def: &OpDef) -> Result<Self, OpLoadError> {
            Ok(Self::Dumb)
        }

        fn extension_id() -> ExtensionId {
            EXT_ID.to_owned()
        }
    }
    const_extension_ids! {
        const EXT_ID: ExtensionId = "DummyExt";
    }

    lazy_static! {
        static ref EXT: Extension = {
            let mut e = Extension::new(EXT_ID.clone());
            DummyEnum::Dumb.add_to_extension(&mut e).unwrap();
            e
        };
        static ref DUMMY_REG: ExtensionRegistry =
            ExtensionRegistry::try_new([EXT.to_owned()]).unwrap();
    }
    impl MakeRegisteredOp for DummyEnum {
        fn extension_id(&self) -> ExtensionId {
            EXT_ID.to_owned()
        }

        fn registry<'s, 'r: 's>(&'s self) -> &'r ExtensionRegistry {
            &DUMMY_REG
        }
    }

    #[test]
    fn test_dummy_enum() {
        let o = DummyEnum::Dumb;

        assert_eq!(
            DummyEnum::from_def(EXT.get_op(&o.name()).unwrap()).unwrap(),
            o
        );

        assert_eq!(
            DummyEnum::from_optype(&o.clone().to_extension_op().unwrap().into()).unwrap(),
            o
        );
        let registered: RegisteredOp<_> = o.clone().into();
        assert_eq!(registered.to_inner(), o);
    }
}
