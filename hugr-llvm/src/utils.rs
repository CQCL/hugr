//! Module for utilities that do not depend on LLVM. These are candidates for
//! upstreaming.
#[deprecated(note = "This module is deprecated and will be removed in a future release.")]
pub mod array_op_builder;
pub mod fat;
pub mod inline_constant_functions;
pub mod int_op_builder;
pub mod logic_op_builder;
pub mod type_map;

#[deprecated(note = "Import from hugr_core::std_extensions::collections::array.")]
pub use hugr_core::std_extensions::collections::array::ArrayOpBuilder;
pub use inline_constant_functions::inline_constant_functions;
pub use int_op_builder::IntOpBuilder;
pub use logic_op_builder::LogicOpBuilder;
