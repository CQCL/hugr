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
/// The parameters must be constants of type [`Type`].
///
/// For type rows that cannot be statically defined, use a vector or slice with
/// [`TypeRow::from`] instead.
///
/// [`Type`]: crate::types::Type
/// [`TypeRow`]: crate::types::TypeRow
/// [`TypeRow::from`]: crate::types::TypeRow::from
///
/// Example:
/// ```
/// # use hugr::macros::type_row;
/// # use hugr::types::{AbstractSignature, Type, TypeRow};
/// const U: Type = Type::UNIT;
/// let static_row: TypeRow = type_row![U, U];
/// let dynamic_row: TypeRow = vec![U, U, U].into();
/// let sig = AbstractSignature::new(static_row, dynamic_row).pure();
///
/// let repeated_row: TypeRow = type_row![U; 3];
/// assert_eq!(repeated_row, *sig.output());
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
            static ROW: &[types::Type] = &[$($t),*];
            let row: types::TypeRow = ROW.into();
            row
        }
    };
    ($t:expr; $n:expr) => {
        {
            use $crate::types;
            static ROW: &[types::Type] = &[$t; $n];
            let row: types::TypeRow = ROW.into();
            row
        }
    };
}

#[allow(unused_imports)]
pub use type_row;
