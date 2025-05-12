//! # A crate for lowering `HUGR`s into LLVM.
//!
//! ## Five minute Introduction to LLVM and [inkwell]
//!
//! References:
//! * The full specification for LLVM IR as on the main branch: <https://llvm.org/docs/LangRef.html>.
//! * The full specification for LLVM IR version for version 14.0: <https://releases.llvm.org/14.0.0/docs/LangRef.html>
//! * The documentation for the [inkwell] crate <https://thedan64.github.io/inkwell/inkwell/index.html>
//!
//! LLVM offers a stable C interface to most of it's functionality. These
//! bindings are exposed to rust through the `llvm-sys` crate; there is quite a
//! lot of feature-ing and build-time logic done there to support various
//! linking configurations and llvm versions.
//!
//! The [inkwell] crate offers safe wrappers around these bindings and is what
//! we use throughout.
//!
//! ### Definition of LLVM terms:
//!
//! * [Context](inkwell::context::Context): A context owns all of the many LLVM
//!   objects. Most all `inkwell` types take a lifetime parameter which ensures
//!   they do not outlive their owning `Context`. A `Context` is not thread safe.
//!   A Context is used to construct modules, types, builders, and basic blocks.
//!
//! * [Module](inkwell::module::Module): A module is owned by a `Context`. It is
//!   a container for globals and functions. A `foo.ll` file containing LLVM IR
//!   would be loaded into a `Module`.
//!
//! * [Function](inkwell::values::FunctionValue): A function has a name(symbol),
//!   parameters, a return type, linkage(symbol visibility) and various other
//!   attributes. It may contain basic blocks, in which case compiling the owning
//!   module will produce object code for the function. If it does not contain
//!   basic blocks then it represents an external symbol that can be called, and
//!   object code for that symbol must be linked with the object code from this
//!   module.
//!
//! * [Instruction](inkwell::values::InstructionOpcode): A basic block is an
//!   ordered list of instructions. Examples: `load`, `store`, `iadd`, `ret`. They
//!   are all defined in the `LangRef`.
//!
//! * [Intrinsic](inkwell::intrinsics::Intrinsic): A "special function" provided
//!   by LLVM. These are called like functions but are treated like an
//!   instruction(i.e. special cased) by various passes. The differences between
//!   `Instruction` and `Intrinsic` are not well motivated and are largely an
//!   accident of history. They are all defined in the `LangRef`.
//!
//! * [Values](inkwell::values): The things `Instruction`s take and return. They
//!   can be the parameters of functions, the results of instructions,
//!   constants, symbol references to globals, and other more esoteric things.
//!   Mostly we use [`BasicValueEnum`] or [`BasicValue`].
//!
//! * [Types](inkwell::types): Every `Value` has a type. For example `i32`,
//!   `f64`, `ptr`. In particular types are used to construct constant values.
//!
//! * [Builder](inkwell::builder::Builder): This is the mechanism by which one
//!   inserts instructions into a basic block into a basic block. A builder has
//!   a "current position" where the next instruction will be inserted. It has
//!   many functions such as [`inkwell::builder::Builder::build_call`] which are
//!   used to create and insert instructions.
//!
//! [`BasicValueEnum`]: [inkwell::values::BasicValueEnum]
//! [`BasicValue`]: [inkwell::values::BasicValue]
//!
#![expect(missing_docs)] // TODO: Fix...

pub mod custom;
pub mod emit;
pub mod extension;
pub mod sum;
pub mod types;
pub mod utils;

#[allow(unreachable_code)]
#[must_use]
pub fn llvm_version() -> &'static str {
    #[cfg(feature = "llvm14-0")]
    return "llvm14";
    panic!("No recognised llvm feature.")
}

#[cfg(any(test, feature = "test-utils"))]
pub mod test;

pub use custom::{CodegenExtension, CodegenExtsBuilder};

pub use inkwell;
pub use inkwell::llvm_sys;
