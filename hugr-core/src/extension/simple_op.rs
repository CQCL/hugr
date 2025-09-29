//! A trait that enum for op definitions that gathers up some shared functionality.

use std::sync::Weak;

use strum::IntoEnumIterator;

use crate::ops::{ExtensionOp, OpName, OpNameRef};
use crate::{Extension, ops::OpType, types::TypeArg};

use super::{ExtensionBuildError, ExtensionId, OpDef, SignatureError, op_def::SignatureFunc};
use delegate::delegate;
use thiserror::Error;

/// Error loading operation.
#[derive(Debug, Error, PartialEq, Clone)]
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

/// Traits implemented by types which can add themselves to [`Extension`]s as
/// [`OpDef`]s or load themselves from an [`OpDef`].
///
/// Particularly useful with C-style enums that implement [`strum::IntoEnumIterator`],
/// as then all definitions can be added to an extension at once.
///
/// [`MakeExtensionOp`] has a blanket impl for types that impl [`MakeOpDef`].
pub trait MakeOpDef {
    /// The [`OpDef::name`] which will be used when `Self`  is added to an [Extension]
    /// or when `Self` is loaded from an [`OpDef`].
    ///
    /// This identifer must be unique within the extension with which the
    /// [`OpDef`] is registered. An [`ExtensionOp`] instantiating this [`OpDef`] will
    /// report `self.opdef_id()` as its [`ExtensionOp::unqualified_id`].
    ///
    /// [`MakeExtensionOp::op_id`] must match this function.
    fn opdef_id(&self) -> OpName;

    /// Try to load one of the operations of this set from an [`OpDef`].
    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized;

    /// The ID of the extension this operation is defined in.
    fn extension(&self) -> ExtensionId;

    /// Returns a weak reference to the extension this operation is defined in.
    fn extension_ref(&self) -> Weak<Extension>;

    /// Compute the signature of the operation while the extension definition is being built.
    ///
    /// Requires a [`Weak`] reference to the extension defining the operation.
    /// This method is intended to be used inside the closure passed to [`Extension::new_arc`],
    /// and it is normally internally called by [`MakeOpDef::add_to_extension`].
    fn init_signature(&self, extension_ref: &Weak<Extension>) -> SignatureFunc;

    /// Return the signature (polymorphic function type) of the operation.
    fn signature(&self) -> SignatureFunc {
        self.init_signature(&self.extension_ref())
    }

    /// Description of the operation. By default, the same as `self.opdef_id()`.
    fn description(&self) -> String {
        self.opdef_id().to_string()
    }

    /// Edit the opdef before finalising. By default does nothing.
    fn post_opdef(&self, _def: &mut OpDef) {}

    /// Add an operation implemented as an [`MakeOpDef`], which can provide the data
    /// required to define an [`OpDef`], to an extension.
    ///
    /// Requires a [`Weak`] reference to the extension defining the operation.
    /// This method is intended to be used inside the closure passed to [`Extension::new_arc`].
    fn add_to_extension(
        &self,
        extension: &mut Extension,
        extension_ref: &Weak<Extension>,
    ) -> Result<(), ExtensionBuildError> {
        let def = extension.add_op(
            self.opdef_id(),
            self.description(),
            self.init_signature(extension_ref),
            extension_ref,
        )?;

        self.post_opdef(def);

        Ok(())
    }

    /// Load all variants of an enum of op definitions in to an extension as op defs.
    /// See [`strum::IntoEnumIterator`].
    ///
    /// Requires a [`Weak`] reference to the extension defining the operation.
    /// This method is intended to be used inside the closure passed to [`Extension::new_arc`].
    fn load_all_ops(
        extension: &mut Extension,
        extension_ref: &Weak<Extension>,
    ) -> Result<(), ExtensionBuildError>
    where
        Self: IntoEnumIterator,
    {
        for op in Self::iter() {
            op.add_to_extension(extension, extension_ref)?;
        }
        Ok(())
    }

    /// If the definition can be loaded from a string, load from an [`ExtensionOp`].
    fn from_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized + std::str::FromStr,
    {
        Self::from_extension_op(ext_op)
    }
}

/// [`MakeOpDef`] with an associate concrete Op type which can be instantiated with type arguments.
pub trait HasConcrete: MakeOpDef {
    /// Associated concrete type.
    type Concrete: MakeExtensionOp;

    /// Instantiate the operation with type arguments.
    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError>;
}

/// [`MakeExtensionOp`] with an associated [`HasConcrete`].
pub trait HasDef: MakeExtensionOp {
    /// Associated [`HasConcrete`] type.
    type Def: HasConcrete<Concrete = Self> + std::str::FromStr;

    /// Load the operation from a [`ExtensionOp`].
    fn from_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        Self::from_extension_op(ext_op)
    }
}

/// Traits implemented by types which can be loaded from [`ExtensionOp`]s,
/// i.e. concrete instances of [`OpDef`]s, with defined type arguments.
pub trait MakeExtensionOp {
    /// The [`OpDef::name`] of [`ExtensionOp`]s from which `Self` can be loaded.
    ///
    /// This identifer must be unique within the extension with which the
    /// [`OpDef`] is registered. An [`ExtensionOp`] instantiating this [`OpDef`] will
    /// report `self.opdef_id()` as its [`ExtensionOp::unqualified_id`].
    fn op_id(&self) -> OpName;

    /// Try to load one of the operations of this set from an [`OpDef`].
    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized;
    /// Try to instantiate a variant from an [`OpType`]. Default behaviour assumes
    /// an [`ExtensionOp`] and loads from the name.
    #[must_use]
    fn from_optype(op: &OpType) -> Option<Self>
    where
        Self: Sized,
    {
        let ext: &ExtensionOp = op.as_extension_op()?;
        Self::from_extension_op(ext).ok()
    }

    /// Any type args which define this operation.
    fn type_args(&self) -> Vec<TypeArg>;

    /// Given the ID of the extension this operation is defined in, and a
    /// registry containing that extension, return a [`RegisteredOp`].
    fn to_registered(
        self,
        extension_id: ExtensionId,
        extension: Weak<Extension>,
    ) -> RegisteredOp<Self>
    where
        Self: Sized,
    {
        RegisteredOp {
            extension_id,
            extension,
            op: self,
        }
    }
}

/// Blanket implementation for non-polymorphic operations - [`OpDef`]s with no type parameters.
impl<T: MakeOpDef> MakeExtensionOp for T {
    fn op_id(&self) -> OpName {
        self.opdef_id()
    }

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

/// Load an [`MakeOpDef`] from its name.
/// See [`strum::EnumString`].
pub fn try_from_name<T>(name: &OpNameRef, def_extension: &ExtensionId) -> Result<T, OpLoadError>
where
    T: std::str::FromStr + MakeOpDef,
{
    let op = T::from_str(name).map_err(|_| OpLoadError::NotMember(name.to_string()))?;
    let expected_extension = op.extension();
    if def_extension != &expected_extension {
        return Err(OpLoadError::WrongExtension(
            def_extension.clone(),
            expected_extension,
        ));
    }

    Ok(op)
}

/// Wrap an [`MakeExtensionOp`] with an extension registry to allow type computation.
/// Generate from [`MakeExtensionOp::to_registered`]
#[derive(Clone, Debug)]
pub struct RegisteredOp<T> {
    /// The name of the extension these ops belong to.
    pub extension_id: ExtensionId,
    /// A registry of all extensions, used for type computation.
    extension: Weak<Extension>,
    /// The inner [`MakeExtensionOp`]
    op: T,
}

impl<T> RegisteredOp<T> {
    /// Extract the inner wrapped value
    pub fn to_inner(self) -> T {
        self.op
    }
}

impl<T: MakeExtensionOp> RegisteredOp<T> {
    /// Generate an [`OpType`].
    pub fn to_extension_op(&self) -> Option<ExtensionOp> {
        ExtensionOp::new(
            self.extension.upgrade()?.get_op(&self.op_id())?.clone(),
            self.type_args(),
        )
        .ok()
    }

    delegate! {
        to self.op {
            /// Name of the operation - derived from strum serialization.
            pub fn op_id(&self) -> OpName;
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
    /// A reference to the [Extension] which defines this operation.
    fn extension_ref(&self) -> Weak<Extension>;

    /// Convert this operation in to an [`ExtensionOp`]. Returns None if the type
    /// cannot be computed.
    fn to_extension_op(self) -> Option<ExtensionOp>
    where
        Self: Sized,
    {
        let registered: RegisteredOp<_> = self.into();
        registered.to_extension_op()
    }
}

impl<T: MakeRegisteredOp> From<T> for RegisteredOp<T> {
    fn from(ext_op: T) -> Self {
        let extension_id = ext_op.extension_id();
        let extension = ext_op.extension_ref();
        ext_op.to_registered(extension_id, extension)
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
    use std::sync::{Arc, LazyLock};

    use crate::{
        const_extension_ids, type_row,
        types::{Signature, Term},
    };

    use super::*;
    use strum::{EnumIter, EnumString, IntoStaticStr};

    #[derive(Clone, Debug, Hash, PartialEq, Eq, EnumIter, IntoStaticStr, EnumString)]
    enum DummyEnum {
        Dumb,
    }

    impl MakeOpDef for DummyEnum {
        fn opdef_id(&self) -> OpName {
            <&'static str>::from(self).into()
        }

        fn init_signature(&self, _extension_ref: &Weak<Extension>) -> SignatureFunc {
            Signature::new_endo(type_row![]).into()
        }

        fn extension_ref(&self) -> Weak<Extension> {
            Arc::downgrade(&EXT)
        }

        fn from_def(_op_def: &OpDef) -> Result<Self, OpLoadError> {
            Ok(Self::Dumb)
        }

        fn extension(&self) -> ExtensionId {
            EXT_ID.clone()
        }
    }

    impl HasConcrete for DummyEnum {
        type Concrete = Self;

        fn instantiate(&self, _type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
            if _type_args.is_empty() {
                Ok(self.clone())
            } else {
                Err(OpLoadError::InvalidArgs(SignatureError::InvalidTypeArgs))
            }
        }
    }
    const_extension_ids! {
        const EXT_ID: ExtensionId = "DummyExt";
    }

    static EXT: LazyLock<Arc<Extension>> = LazyLock::new(|| {
        Extension::new_test_arc(EXT_ID.clone(), |ext, extension_ref| {
            DummyEnum::Dumb
                .add_to_extension(ext, extension_ref)
                .unwrap();
        })
    });

    impl MakeRegisteredOp for DummyEnum {
        fn extension_id(&self) -> ExtensionId {
            EXT_ID.clone()
        }

        fn extension_ref(&self) -> Weak<Extension> {
            Arc::downgrade(&EXT)
        }
    }

    #[test]
    fn test_dummy_enum() {
        let o = DummyEnum::Dumb;

        assert_eq!(
            DummyEnum::from_def(EXT.get_op(&o.opdef_id()).unwrap()).unwrap(),
            o
        );

        assert_eq!(
            DummyEnum::from_optype(&o.clone().to_extension_op().unwrap().into()).unwrap(),
            o
        );
        let registered: RegisteredOp<_> = o.clone().into();
        assert_eq!(registered.to_inner(), o);

        assert_eq!(o.instantiate(&[]), Ok(o.clone()));
        assert_eq!(
            o.instantiate(&[Term::from(1u64)]),
            Err(OpLoadError::InvalidArgs(SignatureError::InvalidTypeArgs))
        );
    }
}
