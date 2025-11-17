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
/// # use hugr::type_row;
/// # use hugr::types::{Signature, Type, TypeRow};
/// const U: Type = Type::UNIT;
/// let static_row: TypeRow = type_row![U, U];
/// let dynamic_row: TypeRow = vec![U, U, U].into();
/// let sig = Signature::new(static_row, dynamic_row);
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

/// Declare 'const' variables holding new `ExtensionIds`, validating
/// that they are well-formed as separate tests - hence, usable at the top level
/// of a test module only. Example:
/// ```rust
/// # mod test {
/// # use hugr::const_extension_ids;
/// const_extension_ids! {
///   pub const EXT_A: ExtensionId = "A";
///   /// A doc comment
///   #[cfg(foobar)] pub (super) const EXT_A_B: ExtensionId = "A.B";
///   const EXT_BAD: ExtensionId = "..55"; // this will generate a failing #[test] fn ....
/// }
/// # }
/// ```
#[macro_export]
macro_rules! const_extension_ids {
    ($($(#[$attr:meta])* $v:vis const $field_name:ident : ExtensionId = $ext_name:expr;)+) => {
        $($(#[$attr])* $v const $field_name: $crate::extension::ExtensionId =
            $crate::extension::ExtensionId::new_unchecked($ext_name);

        pastey::paste! {
            #[test]
            fn [<check_ $field_name:lower _wellformed>]() {
                $crate::extension::ExtensionId::new($ext_name).unwrap();
            }
        })*
    };
}

pub use const_extension_ids;

#[cfg(test)]
/// Get the full path to a test file given its path relative to the
/// `resources/test` directory in this crate.
#[macro_export]
macro_rules! test_file {
    ($fname:expr) => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/resources/test/", $fname)
    };
}
