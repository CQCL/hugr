use std::{borrow::Cow, rc::Rc};

use anyhow::{Result, bail};
use delegate::delegate;
use inkwell::{
    builder::Builder,
    types::{BasicType, BasicTypeEnum},
    values::{BasicValue, BasicValueEnum, PointerValue},
};
use itertools::{Itertools as _, zip_eq};

#[derive(Eq, PartialEq, Clone)]
pub struct ValueMailBox<'c> {
    typ: BasicTypeEnum<'c>,
    ptr: PointerValue<'c>,
    name: Cow<'static, str>,
}

fn join_names<'a>(names: impl IntoIterator<Item = &'a str>) -> String {
    names
        .into_iter()
        .filter(|x| !x.is_empty())
        .join("_")
        .to_string()
}

impl<'c> ValueMailBox<'c> {
    pub(super) fn new(
        typ: impl BasicType<'c>,
        ptr: PointerValue<'c>,
        name: Option<String>,
    ) -> Self {
        Self {
            typ: typ.as_basic_type_enum(),
            ptr,
            name: name.map_or(Cow::Borrowed(""), Cow::Owned),
        }
    }
    pub fn get_type(&self) -> BasicTypeEnum<'c> {
        self.typ
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn promise(&self) -> ValuePromise<'c> {
        ValuePromise(self.clone())
    }

    pub fn read<'a>(
        &'a self,
        builder: &Builder<'c>,
        labels: impl IntoIterator<Item = &'a str>,
    ) -> Result<BasicValueEnum<'c>> {
        let r = builder.build_load(
            self.ptr,
            &join_names(
                labels
                    .into_iter()
                    .chain(std::iter::once(self.name.as_ref())),
            ),
        )?;
        debug_assert_eq!(r.get_type(), self.get_type());
        Ok(r)
    }

    fn write(&self, builder: &Builder<'c>, v: impl BasicValue<'c>) -> Result<()> {
        builder.build_store(self.ptr, v)?;
        Ok(())
    }
}

#[must_use]
pub struct ValuePromise<'c>(ValueMailBox<'c>);

impl<'c> ValuePromise<'c> {
    pub fn finish(self, builder: &Builder<'c>, v: impl BasicValue<'c>) -> Result<()> {
        self.0.write(builder, v)
    }

    delegate! {
        to self.0 {
            pub fn get_type(&self) -> BasicTypeEnum<'c>;
        }
    }
}

/// Holds a vector of [`PointerValue`]s pointing to `alloca`s in the first block
/// of a function.
#[derive(Eq, PartialEq, Clone)]
#[allow(clippy::len_without_is_empty)]
pub struct RowMailBox<'c>(Rc<Vec<ValueMailBox<'c>>>, Cow<'static, str>);

impl<'c> RowMailBox<'c> {
    #[must_use]
    pub fn new_empty() -> Self {
        Self::new(std::iter::empty(), None)
    }

    pub(super) fn new(
        mbs: impl IntoIterator<Item = ValueMailBox<'c>>,
        name: Option<String>,
    ) -> Self {
        Self(
            Rc::new(mbs.into_iter().collect_vec()),
            name.map_or(Cow::Borrowed(""), Cow::Owned),
        )
    }

    /// Returns a [`RowPromise`] that when [`RowPromise::finish`]ed will write to this `RowMailBox`.
    pub fn promise(&self) -> RowPromise<'c> {
        RowPromise(self.clone())
    }

    /// Get the LLVM types of this `RowMailBox`.
    pub fn get_types(&'_ self) -> impl Iterator<Item = BasicTypeEnum<'c>> + '_ {
        self.0.iter().map(ValueMailBox::get_type)
    }

    /// Returns the number of values in this `RowMailBox`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Read from the inner pointers.
    pub fn read_vec<'a>(
        &'a self,
        builder: &Builder<'c>,
        labels: impl IntoIterator<Item = &'a str>,
    ) -> Result<Vec<BasicValueEnum<'c>>> {
        self.read(builder, labels)
    }

    /// Read from the inner pointers.
    pub fn read<'a, R: FromIterator<BasicValueEnum<'c>>>(
        &'a self,
        builder: &Builder<'c>,
        labels: impl IntoIterator<Item = &'a str>,
    ) -> Result<R> {
        let labels = labels.into_iter().collect_vec();
        self.mailboxes()
            .map(|mb| mb.read(builder, labels.clone()))
            .collect::<Result<_>>()
    }

    pub(crate) fn write(
        &self,
        builder: &Builder<'c>,
        vs: impl IntoIterator<Item = BasicValueEnum<'c>>,
    ) -> Result<()> {
        let vs = vs.into_iter().collect_vec();
        #[cfg(debug_assertions)]
        {
            let actual_types = vs.clone().into_iter().map(|x| x.get_type()).collect_vec();
            let expected_types = self.get_types().collect_vec();
            if actual_types != expected_types {
                bail!(
                    "RowMailbox::write: Expected types {:?}, got {:?}",
                    expected_types,
                    actual_types
                );
            }
        }
        zip_eq(self.0.iter(), vs).try_for_each(|(mb, v)| mb.write(builder, v))
    }

    fn mailboxes(&'_ self) -> impl Iterator<Item = ValueMailBox<'c>> + '_ {
        self.0.iter().cloned()
    }
}

impl<'c> FromIterator<ValueMailBox<'c>> for RowMailBox<'c> {
    fn from_iter<T: IntoIterator<Item = ValueMailBox<'c>>>(iter: T) -> Self {
        Self::new(iter, None)
    }
}

/// A promise to write values into a `RowMailBox`
#[must_use]
#[allow(clippy::len_without_is_empty)]
pub struct RowPromise<'c>(RowMailBox<'c>);

impl<'c> RowPromise<'c> {
    /// Consumes the `RowPromise`, writing the values to the promised [`RowMailBox`].
    pub fn finish(
        self,
        builder: &Builder<'c>,
        vs: impl IntoIterator<Item = BasicValueEnum<'c>>,
    ) -> Result<()> {
        self.0.write(builder, vs)
    }

    delegate! {
        to self.0 {
            /// Get the LLVM types of this `RowMailBox`.
            pub fn get_types(&'_ self) -> impl Iterator<Item=BasicTypeEnum<'c>> + '_;
            /// Returns the number of values promised to be written.
            #[must_use] pub fn len(&self) -> usize;
        }
    }
}
