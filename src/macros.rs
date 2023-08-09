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

/// Creates a [`SimpleRow`] backed by statically defined data, avoiding
/// allocations.
///
/// The parameters must be constants of type [`SimpleType`].
///
/// For type rows that cannot be statically defined, use a vector or slice with
/// [`SimpleRow::from`] instead.
///
/// [`SimpleType`]: crate::types::SimpleType
/// [`SimpleRow`]: crate::types::SimpleRow
/// [`SimpleRow::from`]: crate::types::SimpleRow::from
///
/// Example:
/// ```
/// # use hugr::macros::type_row;
/// # use hugr::types::{AbstractSignature, ClassicType, SimpleType, SimpleRow};
/// const B: SimpleType = SimpleType::Classic(ClassicType::usize());
/// const QB: SimpleType = SimpleType::Qubit;
/// let static_row: SimpleRow = type_row![B, QB];
/// let dynamic_row: SimpleRow = vec![B, B, B].into();
/// let sig = AbstractSignature::new_df(static_row, dynamic_row).pure();
///
/// let repeated_row: SimpleRow = type_row![B; 3];
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
            static ROW: &[types::SimpleType] = &[$($t),*];
            let row: types::SimpleRow = ROW.into();
            row
        }
    };
    ($t:expr; $n:expr) => {
        {
            use $crate::types;
            static ROW: &[types::SimpleType] = &[$t; $n];
            let row: types::SimpleRow = ROW.into();
            row
        }
    };
}

/// Like [type_row!] but creates a [`ClassicRow`], from parameters
/// that must all be constants of type [`ClassicType`].
///
/// For type rows that cannot be statically defined, use a vector or slice with
/// [`ClassicRow::from`] instead.
///
/// [`ClassicType`]: crate::types::ClassicType
/// [`ClassicRow`]: crate::types::ClassicRow
/// [`ClassicRow::from`]: crate::types::ClassicRow::from
///
/// Example:
/// ```
/// # use hugr::macros::classic_row;
/// # use hugr::types::{ClassicType, Signature, ClassicRow};
/// const B: ClassicType = ClassicType::usize();
/// const I: ClassicType = ClassicType::usize();
/// let static_row: ClassicRow = classic_row![B, B];
/// let dynamic_row: ClassicRow = vec![B, B, I].into();
///
/// let repeated_row: ClassicRow = classic_row![B; 2];
/// assert_eq!(repeated_row, static_row);
/// ```
#[allow(unused_macros)]
#[macro_export]
macro_rules! classic_row {
    () => {
        {
            $crate::types::ClassicRow::new()
        }
    };
    ($($t:expr),+ $(,)?) => {
        {
            use $crate::types;
            static ROW: &[types::ClassicType] = &[$($t),*];
            let row: types::ClassicRow = ROW.into();
            row
        }
    };
    ($t:expr; $n:expr) => {
        {
            use $crate::types;
            static ROW: &[types::ClassicType] = &[$t; $n];
            let row: types::ClassicRow = ROW.into();
            row
        }
    };
}

#[allow(unused_imports)]
pub use type_row;

#[allow(unused_imports)]
pub use classic_row;
