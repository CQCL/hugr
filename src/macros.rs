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
