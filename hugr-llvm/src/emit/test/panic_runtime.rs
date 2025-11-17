//! This module provides a *very* basic panic runtime for LLVM execution
//! tests involving panics.
//!
//! Note that this is strictly for *execution testing* and is not used in
//! production! See [`crate::emit::test::PanicTestPreludeCodegen`] and
//! [`crate::emit::test::Emission::exec_panicking`] for the test harness.
//!
//! The progam entry point should be invoked via the [`trampoline`] function
//! which launches the program. Calls to [`panic_exit`] will exit the program
//! and make [`trampoline`] return prematurely. Note that the runtime will
//! perform a direct jump to the exit and no unrolling is performed. This
//! is fine for our use case since stack frames from hugr-llvm generated
//! bytecode don't require cleanup. However, using [`panic_exit`] to jump
//! over Rust frames is undefined behaviour.
//!
//! Note that the implementation relies on setjmp-longjmp (SJLJ) jumping to
//! exit the program. As SJLJ is not available in Rust, this had to be
//! implemented in C in linked into the Rust binary.

use std::ffi::{c_char, c_void};

#[link(name = "test_panic_runtime")]
unsafe extern "C" {
    /// Runs the `entry` function inside the panic runtime.
    ///
    /// # Safety
    ///
    /// `jmp_buf` must have size of at least [`jmp_buf_size`].
    pub(crate) unsafe fn trampoline(jmp_buf: *mut c_void, entry: Option<unsafe extern "C" fn()>);

    /// Exits a program launched by [`trampoline`].
    ///
    /// This function will never return. Also copies the first `msg_limit`
    /// characters of the string `msg` into `msg_buf`.
    ///
    /// # Safety
    ///
    /// Calls to this function are only allowed within stack frames that were
    /// launched by [`trampoline`]. Furthermore, [`panic_exit`] may only be used
    /// to jump over C or LLVM execution engine frames. Using [`panic_exit`] to
    /// jump over Rust frames is undefined behaviour.
    ///
    /// `jmp_buf` must be the same buffer given to [`trampoline`] and `msg_buf`
    /// must have size of at least `msg_limit`.
    pub(crate) unsafe fn panic_exit(
        jmp_buf: *mut c_void,
        msg_buf: *mut c_char,
        msg: *const c_char,
        msg_limit: usize,
    ) -> !;

    /// Minimum size that needs to be alloced for jump buffers passed to
    /// [`trampoline`] and [`panic_exit`].
    pub(crate) unsafe fn jmp_buf_size() -> usize;

}
