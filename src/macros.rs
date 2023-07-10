//! Helper macros.

/// Helper macro for declaring downcast traits that can be cloned when inside a `Box<dyn $trait>`.
/// Implements `Clone` for `Box<dyn $trait>`.
macro_rules! impl_box_clone {
    ($trait:ident, $clone_trait:ident) => {
        /// Auto-implemented trait for cloning `Box<dyn $trait>`.
        pub trait $clone_trait {
            /// Clone the trait object into a Sized box.
            fn clone_box(&self) -> Box<dyn $trait>;
        }

        impl<T> $clone_trait for T
        where
            T: $trait + Clone,
        {
            fn clone_box(&self) -> Box<dyn $trait> {
                Box::new(self.clone())
            }
        }

        impl Clone for Box<dyn $trait> {
            fn clone(&self) -> Box<dyn $trait> {
                self.clone_box()
            }
        }
    };
}
pub(crate) use impl_box_clone;

/// Creates a [`TypeRow`] backed by statically defined data, avoiding
/// allocations.
///
/// The parameters must be constants of type [`SimpleType`].
///
/// For type rows that cannot be statically defined, use a vector or slice with
/// [`TypeRow::from`] instead.
///
/// [`SimpleType`]: crate::types::SimpleType
/// [`TypeRow`]: crate::types::TypeRow
/// [`TypeRow::from`]: crate::types::TypeRow::from
///
/// Example:
/// ```
/// # use hugr::macros::type_row;
/// # use hugr::types::{ClassicType, SimpleType, Signature, TypeRow};
/// const B: SimpleType = SimpleType::Classic(ClassicType::bit());
/// let static_row: TypeRow<SimpleType> = type_row![B, B];
/// let dynamic_row: TypeRow<SimpleType> = vec![B, B, B].into();
/// let sig: Signature = Signature::new_df(static_row.clone(), dynamic_row);
///
/// let repeated_row: TypeRow<SimpleType> = type_row![B; 2];
/// assert_eq!(repeated_row, static_row);
/// ```
#[allow(unused_macros)]
#[macro_export]
macro_rules! type_row {
    () => {
        {
            $crate::types::TypeRow::new()
        }
    };
    ($($t:expr),+ $(,)?) => {
        {
            use $crate::types;
            static ROW: &[types::SimpleType] = &[$($t),*];
            let row: types::TypeRow<_> = ROW.into();
            row
        }
    };
    ($t:expr; $n:expr) => {
        {
            use $crate::types;
            static ROW: &[types::SimpleType] = &[$t; $n];
            let row: types::TypeRow<_> = ROW.into();
            row
        }
    };
}

/// Like [type_row!] but for rows of [ClassicType]s
///
/// [ClassicType]: types::ClassicType
#[allow(unused_macros)]
#[macro_export]
macro_rules! classic_row {
    () => {
        {
            $crate::types::TypeRow::new()
        }
    };
    ($($t:expr),+ $(,)?) => {
        {
            use $crate::types;
            static ROW: &[types::ClassicType] = &[$($t),*];
            let row: types::TypeRow<_> = ROW.into();
            row
        }
    };
    ($t:expr; $n:expr) => {
        {
            use $crate::types;
            static ROW: &[types::ClassicType] = &[$t; $n];
            let row: types::TypeRow<_> = ROW.into();
            row
        }
    };
}

#[allow(unused_imports)]
pub use type_row;

#[allow(unused_imports)]
pub use classic_row;
