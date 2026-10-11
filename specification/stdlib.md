(standard-library)=
# Standard Library

A collection of extensions form the "standard library" for HUGR, and are defined
in this repository.

## Prelude

The prelude extension is assumed to be valid and available in all contexts, and
so must be supported by all third-party tooling.

### Types

`usize`: a positive integer size type.

`string`: a string type.

`array<N, T>`: a known-size (N) array of type T.

`qubit`: a linear (non-copyable) qubit type.

`error`: an error type which operations use as a variant of sum to indicate when an error may occur. See [Arithmetic Extensions](#arithmetic-extensions) for some examples.

### Operations

| Name              | Inputs       | Outputs       | Meaning                                                                                                                                                                                                            |
|-------------------|--------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `print`           | `string`     | -             | Append the string to the program's output stream[^1] (atomically).                                                                                                                                                 |
| `new_array<N, T>` | `T` x N      | `array<N, T>` | Create an array from all the inputs.                                                                                                                                                                               |
| `panic`           | `error`, ... | ...           | Immediately end execution and pass contents of error to context. Inputs following the `error`, and all outputs, are arbitrary; these only exist so that structural constraints such as linearity can be satisfied. |

[^1]: The existence of an output stream, and the processing of it either during
or after program execution, is runtime-dependent. If no output stream exists
then the `print` function is a no-op.

## Logic Extension

The Logic Extension provides a boolean type and basic logical operations.

The boolean type `bool` is defined to be `Sum(#(),#())`, with the convention that the
first option in the sum represents "false" and the second represents "true".

The following operations are defined:

| Name     | Inputs     | Outputs | Meaning                       |
| -------- | ---------- | ------- | ----------------------------- |
| `not`    | `bool`     | `bool`  | logical "not"                 |
| `and<N>` | `bool` x N | `bool`  | N-ary logical "and" (N \>= 0) |
| `or<N>`  | `bool` x N | `bool`  | N-ary logical "or"  (N \>= 0) |

Note that an `and<0>` operation produces the constant value "true" and an
`or<0>` operation produces the constant value "false".

(arithmetic-extensions)=
## Arithmetic Extensions

Types and operations for integer and
floating-point operations are provided by a collection of extensions under the
namespace `arithmetic`.

We largely adopt (a subset of) the definitions of
[WebAssembly 2.0](https://webassembly.github.io/spec/core/index.html),
including the names of the operations. Where WebAssembly specifies a
"partial" operation (i.e. when the result is not defined on certain
inputs), we use a Sum type to hold the result.

A few additional operations not included in WebAssembly are also
specified, and there are some other small differences (highlighted
below).

### `arithmetic.int.types`

The `int<N>` type is parameterized by its width `N`, which is a positive
integer.

The possible values of `N` are 2^i for i in the range [0,6].

The `int<N>` type represents an arbitrary bit string of length `N`.
Semantics are defined by the operations. There are three possible
interpretations of a value:

- as a bit string $(a_{N-1}, a_{N-2}, \ldots, a_0)$ where
  $a_i \in \\{0,1\\}$;
- as an unsigned integer $\sum_{i \lt N} 2^i a_i$;
- as a signed integer $\sum_{i \lt N-1} 2^i a_i - 2^{N-1} a_{N-1}$.

An asterix ( \* ) in the tables below indicates that the definition
either differs from or is not part of the
[WebAssembly](https://webassembly.github.io/spec/core/exec/numerics.html)
specification.

### `arithmetic.int`

This extension defines operations on the integer types.

Casts:

| Name                   | Inputs   | Outputs                    | Meaning                                                                                      |
| ---------------------- | -------- |----------------------------| -------------------------------------------------------------------------------------------- |
| `iwiden_u<M,N>`( \* )  | `int<M>` | `int<N>`                   | widen an unsigned integer to a wider one with the same value (where M \<= N)                 |
| `iwiden_s<M,N>`( \* )  | `int<M>` | `int<N>`                   | widen a signed integer to a wider one with the same value (where M \<= N)                    |
| `inarrow_u<M,N>`( \* ) | `int<M>` | `Sum(#(int<N>), #(error))` | narrow an unsigned integer to a narrower one with the same value if possible, and an error otherwise (where M \>= N) |
| `inarrow_s<M,N>`( \* ) | `int<M>` | `Sum(#(int<N>), #(error))` | narrow a signed integer to a narrower one with the same value if possible, and an error otherwise (where M \>= N)    |
| `itobool` ( \* )       | `int<1>` | `bool`                     | convert to `bool` (1 is true, 0 is false)                                                    |
| `ifrombool` ( \* )     | `bool`   | `int<1>`                   | convert from `bool` (1 is true, 0 is false)                                                  |

Comparisons:

| Name       | Inputs             | Outputs | Meaning                                      |
| ---------- | ------------------ | ------- | -------------------------------------------- |
| `ieq<N>`   | `int<N>`, `int<N>` | `bool`  | equality test                                |
| `ine<N>`   | `int<N>`, `int<N>` | `bool`  | inequality test                              |
| `ilt_u<N>` | `int<N>`, `int<N>` | `bool`  | "less than" as unsigned integers             |
| `ilt_s<N>` | `int<N>`, `int<N>` | `bool`  | "less than" as signed integers               |
| `igt_u<N>` | `int<N>`, `int<N>` | `bool`  | "greater than" as unsigned integers          |
| `igt_s<N>` | `int<N>`, `int<N>` | `bool`  | "greater than" as signed integers            |
| `ile_u<N>` | `int<N>`, `int<N>` | `bool`  | "less than or equal" as unsigned integers    |
| `ile_s<N>` | `int<N>`, `int<N>` | `bool`  | "less than or equal" as signed integers      |
| `ige_u<N>` | `int<N>`, `int<N>` | `bool`  | "greater than or equal" as unsigned integers |
| `ige_s<N>` | `int<N>`, `int<N>` | `bool`  | "greater than or equal" as signed integers   |

Other operations:

| Name                         | Inputs             | Outputs                            | Meaning                                                                                                                                                      |
|------------------------------|--------------------|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `imax_u<N>`                  | `int<N>`, `int<N>` | `int<N>`                           | maximum of unsigned integers                                                                                                                                 |
| `imax_s<N>`                  | `int<N>`, `int<N>` | `int<N>`                           | maximum of signed integers                                                                                                                                   |
| `imin_u<N>`                  | `int<N>`, `int<N>` | `int<N>`                           | minimum of unsigned integers                                                                                                                                 |
| `imin_s<N>`                  | `int<N>`, `int<N>` | `int<N>`                           | minimum of signed integers                                                                                                                                   |
| `iadd<N>`                    | `int<N>`, `int<N>` | `int<N>`                           | addition modulo 2^N (signed and unsigned versions are the same op)                                                                                           |
| `isub<N>`                    | `int<N>`, `int<N>` | `int<N>`                           | subtraction modulo 2^N (signed and unsigned versions are the same op)                                                                                        |
| `ineg<N>`                    | `int<N>`           | `int<N>`                           | negation modulo 2^N (signed and unsigned versions are the same op)                                                                                           |
| `imul<N>`                    | `int<N>`, `int<N>` | `int<N>`                           | multiplication modulo 2^N (signed and unsigned versions are the same op)                                                                                     |
| `idivmod_checked_u<N>`( \* ) | `int<N>`, `int<N>` | `Sum(#(int<N>, int<N>), #(error))` | given unsigned integers 0 \<= n \< 2^N, 0 \<= m \< 2^N, generates unsigned q, r where q\*m+r=n, 0\<=r\<m (m=0 is an error)                                   |
| `idivmod_u<N>`               | `int<N>`, `int<N>` | `(int<N>, int<N>)`                 | given unsigned integers 0 \<= n \< 2^N, 0 \<= m \< 2^N, generates unsigned q, r where q\*m+r=n, 0\<=r\<m (m=0 will call panic)                               |
| `idivmod_checked_s<N>`( \* ) | `int<N>`, `int<N>` | `Sum(#(int<N>, int<N>), #(error))` | given signed integer -2^{N-1} \<= n \< 2^{N-1} and unsigned 0 \<= m \< 2^N, generates signed q and unsigned r where q\*m+r=n, 0\<=r\<m (m=0 is an error)     |
| `idivmod_s<N>`( \* )         | `int<N>`, `int<N>` | `(int<N>, int<N>)`                 | given signed integer -2^{N-1} \<= n \< 2^{N-1} and unsigned 0 \<= m \< 2^N, generates signed q and unsigned r where q\*m+r=n, 0\<=r\<m (m=0 will call panic) |
| `idiv_checked_u<N>` ( \* )   | `int<N>`, `int<N>` | `Sum(#(int<N>), #(error))`         | as `idivmod_checked_u` but discarding the second output                                                                                                      |
| `idiv_u<N>`                  | `int<N>`, `int<N>` | `int<N>`                           | as `idivmod_u` but discarding the second output                                                                                                              |
| `imod_checked_u<N>` ( \* )   | `int<N>`, `int<N>` | `Sum(#(int<N>), #(error))`         | as `idivmod_checked_u` but discarding the first output                                                                                                       |
| `imod_u<N>`                  | `int<N>`, `int<N>` | `int<N>`                           | as `idivmod_u` but discarding the first output                                                                                                               |
| `idiv_checked_s<N>`( \* )    | `int<N>`, `int<N>` | `Sum(#(int<N>), #(error))`         | as `idivmod_checked_s` but discarding the second output                                                                                                      |
| `idiv_s<N>`                  | `int<N>`, `int<N>` | `int<N>`                           | as `idivmod_s` but discarding the second output                                                                                                              |
| `imod_checked_s<N>`( \* )    | `int<N>`, `int<N>` | `Sum(#(int<N>), #(error))`         | as `idivmod_checked_s` but discarding the first output                                                                                                       |
| `imod_s<N>`( \* )            | `int<N>`, `int<M>` | `int<M>`                           | as `idivmod_s` but discarding the first output                                                                                                               |
| `iabs<N>`                    | `int<N>`           | `int<N>`                           | convert signed to unsigned by taking absolute value                                                                                                          |
| `iand<N>`                    | `int<N>`, `int<N>` | `int<N>`                           | bitwise AND                                                                                                                                                  |
| `ior<N>`                     | `int<N>`, `int<N>` | `int<N>`                           | bitwise OR                                                                                                                                                   |
| `ixor<N>`                    | `int<N>`, `int<N>` | `int<N>`                           | bitwise XOR                                                                                                                                                  |
| `inot<N>`                    | `int<N>`           | `int<N>`                           | bitwise NOT                                                                                                                                                  |
| `ishl<N>`( \* )              | `int<N>`, `int<N>` | `int<N>`                           | shift first input left by k bits where k is unsigned interpretation of second input (leftmost bits dropped, rightmost bits set to zero)                      |
| `ishr<N>`( \* )              | `int<N>`, `int<N>` | `int<N>`                           | shift first input right by k bits where k is unsigned interpretation of second input (rightmost bits dropped, leftmost bits set to zero)                     |
| `irotl<N>`( \* )             | `int<N>`, `int<N>` | `int<N>`                           | rotate first input left by k bits where k is unsigned interpretation of second input (leftmost bits replace rightmost bits)                                  |
| `irotr<N>`( \* )             | `int<N>`, `int<N>` | `int<N>`                           | rotate first input right by k bits where k is unsigned interpretation of second input (rightmost bits replace leftmost bits)                                 |
| `itostring_u<N>`             | `int<N>`           | `string`                           | decimal string representation of unsigned integer                                                                                                            |
| `itostring_s<N>`             | `int<N>`           | `string`                           | decimal string representation of signed integer                                                                                                              |


### `arithmetic.float.types`

The `float64` type represents IEEE 754-2019 floating-point data of 64
bits.

Non-finite `float64` values (i.e. NaN and ±infinity) are not allowed in `Const`
nodes.

### `arithmetic.float`

Floating-point operations are defined as follows. All operations below
follow
[WebAssembly](https://webassembly.github.io/spec/core/exec/numerics.html#floating-point-operations)
except where stated.

| Name              | Inputs               | Outputs   | Meaning                                                                  |
| ----------------- | -------------------- | --------- | ------------------------------------------------------------------------ |
| `feq`( \* )       | `float64`, `float64` | `bool`    | equality test (as WASM but with 0 and 1 interpreted as `bool`)           |
| `fne`( \* )       | `float64`, `float64` | `bool`    | inequality test (as WASM but with 0 and 1 interpreted as `bool`)         |
| `flt`( \* )       | `float64`, `float64` | `bool`    | "less than" (as WASM but with 0 and 1 interpreted as `bool`)             |
| `fgt`( \* )       | `float64`, `float64` | `bool`    | "greater than" (as WASM but with 0 and 1 interpreted as `bool`)          |
| `fle`( \* )       | `float64`, `float64` | `bool`    | "less than or equal" (as WASM but with 0 and 1 interpreted as `bool`)    |
| `fge`( \* )       | `float64`, `float64` | `bool`    | "greater than or equal" (as WASM but with 0 and 1 interpreted as `bool`) |
| `fmax`            | `float64`, `float64` | `float64` | maximum                                                                  |
| `fmin`            | `float64`, `float64` | `float64` | minimum                                                                  |
| `fadd`            | `float64`, `float64` | `float64` | addition                                                                 |
| `fsub`            | `float64`, `float64` | `float64` | subtraction                                                              |
| `fneg`            | `float64`            | `float64` | negation                                                                 |
| `fabs`            | `float64`            | `float64` | absolute value                                                           |
| `fmul`            | `float64`, `float64` | `float64` | multiplication                                                           |
| `fdiv`            | `float64`, `float64` | `float64` | division                                                                 |
| `ffloor`          | `float64`            | `float64` | floor                                                                    |
| `fceil`           | `float64`            | `float64` | ceiling                                                                  |
| `ftostring`       | `float64`            | `string`  | string representation[^2]                                                  |

[^2]: The exact specification of the float-to-string conversion is
implementation-dependent.

### `arithmetic.math`

This extension provides math operations on `arithmetic.float.types.float64`. All 
angles, including the results of inverse trigonometric operations, are in radians.

| Name    | Inputs               | Outputs   | Meaning                                                                                                                                   |
| ------- | -------------------- | --------- |-------------------------------------------------------------------------------------------------------------------------------------------|
| `sin`   | `float64`            | `float64` | sine                                                                                                                                      |
| `cos`   | `float64`            | `float64` | cosine                                                                                                                                    |
| `tan`   | `float64`            | `float64` | tangent                                                                                                                                   |
| `asin`  | `float64`            | `float64` | inverse sine                                                                                                                              |
| `acos`  | `float64`            | `float64` | inverse cosine                                                                                                                            |
| `atan`  | `float64`            | `float64` | inverse tangent                                                                                                                           |
| `atan2` | `float64`, `float64` | `float64` | four-quadrant inverse tangent, with inputs in the order `(y, x)`                                                                          |
| `exp`   | `float64`            | `float64` | natural exponential, $e^x$                                                                                                                |
| `exp2`  | `float64`            | `float64` | base-2 exponential, $2^x$                                                                                                                 |
| `log`   | `float64`            | `float64` | natural logarithm                                                                                                                         |
| `log2`  | `float64`            | `float64` | base-2 logarithm                                                                                                                          |
| `log10` | `float64`            | `float64` | base-10 logarithm                                                                                                                         |
| `pow`   | `float64`, `float64` | `float64` | first input raised to the power of the second, with inputs `(base, exponent)`                                                             |
| `fmod`  | `float64`, `float64` | `float64` | remainder of the first input divided by the second, using a quotient truncated toward zero; the remainder has the sign of the first input |

### `arithmetic.conversions`

Conversions between integers and floats:

| Name           | Inputs    | Outputs                    | Meaning               |
| -------------- | --------- |----------------------------| --------------------- |
| `trunc_u<N>`   | `float64` | `Sum(#(int<N>), #(error))` | float to unsigned int, rounding towards zero. Returns an error when the float is non-finite. |
| `trunc_s<N>`   | `float64` | `Sum(#(int<N>), #(error))` | float to signed int, rounding towards zero. Returns an error when the float is non-finite. |
| `convert_u<N>` | `int<N>`  | `float64`                  | unsigned int to float |
| `convert_s<N>` | `int<N>`  | `float64`                  | signed int to float   |
| `bytecast_int64_to_float64` | `int<6>`  | `float64`     | reinterpret an int64 as a float64 based on its bytes, with the same endianness. |
| `bytecast_float64_to_int64` | `float64` | `int64`       | reinterpret an float64 as an int based on its bytes, with the same endianness. |

## Collections Extensions

There are multiple extensions defining types, values and operations to work with collections of data:

- `collections.array`: The standard linear and fixed-length array type, parameterized by length and element type.
- `collections.borrow_arr`: A linear and fixed-length array type that provides additional unsafe operations for borrowing elements from the array, parameterized by length and element type.
- `collections.static_array`: An array type for modelling globally available constant arrays of copyable values, parameterized only by element type.
- `collections.list`: A variable-length list type, parameterized by element type.


### `collections.array`

This extension provides the `array` type and value with the following operations:


| Operation       | Inputs          | Outputs         | Meaning         |
|-----------------|-----------------|-----------------|-----------------|
| `new_array`     | `elem_ty^SIZE` | `array<SIZE, elemty>` | Make a new array, given distinct inputs equal to its length. `SIZE` must be statically known (not a variable). |
| `get`           | `array<size, elemty>`, `usize` | `option<elemty>`, `array` | Copy an element out of the array (**copyable** elements only). Return none if the index is out of bounds. |
| `set`           | `array<size, elemty>`, `usize`, `elemty` | `either<elemty, array>` | Exchange an element of the array with an external value. Tagged for failure/success if index is out of bounds respectively. |
| `swap`          | `array<size, elemty>`, `usize`, `usize` | `either<array, array>` | Exchange the elements at two indices within the array. Tagged for failure/success if index is out of bounds respectively. |
| `pop_left`      | `array<SIZE, elemty>` | `option<elemty, array<SIZE-1, elemty>>` | Pop an element from the left of the array. `SIZE` must be statically known (not a variable). Return none if the input array is size 0. |
| `pop_right`     | `array<SIZE, elemty>` | `option<elemty, array<SIZE-1, elemty>>` | Pop an element from the right of the array. `SIZE` must be statically known (not a variable). Return none if the input array is size 0. |
| `discard_empty` | `array<0, elemty>` | `()`  | Discard an empty array. |
| `discard`       | `array<SIZE, elemty>` | `()`  | Discard an array with **copyable** elements. |
| `clone`         | `array<SIZE, elemty>` | `array<SIZE, elemty>`, `array<SIZE, elemty>` | Clone an array with **copyable** elements. |
| `unpack`        | `array<SIZE, elemty>` | `elemty^SIZE` | Unpack an array into its individual elements. `SIZE` must be statically known (not a variable). |
| `repeat`        | `(() -> elemty)` | `array<SIZE, elemty>` | Create a new array whose elements are initialized by calling the given function `SIZE` times. |
| `scan`          | `array<SIZE, elemty_src>`,  `(elemty_src, list<acc_ty> -> elemty_dest, list<acc_ty>)`, `list<acc_ty>` | `array<SIZE, elemty_dest>`, `list<acc_ty>`  | A combination of map and foldl. Apply a function to each element of the array with an accumulator that is passed through from start to finish. Return the resulting array and the final state of the accumulator. |


### `collections.borrow_arr`

This extension contains the `borrow_array` type and value. It has all the operations that the array extension does (see previous section), with additional unsafe operations to deal with borrowing elements. Borrowing means taking elements out of an array while relying on the underlying implementation to keep track of which elements have already been taken out.

| Operation             | Inputs                | Outputs               | Meaning               |
|-----------------------|-----------------------|-----------------------|-----------------------|
| `borrow`              | `borrow_array<size, elemty>`, `usize`| `borrow_array<size, elemty>`, `elemty` | Borrow an element from the array at the given index. The element already being borrowed should result in a panic. |
| `return`              | `borrow_array<size, elemty>`, `usize`, `elemty` | `borrow_array<size, elemty>`| Return an element to the array at the given index. There already being an element at this index should result in a panic. |
| `discard_all_borrowed`| `borrow_array<size, elemty>`| `()`| Discard an array where all elements have been borrowed. Should panic if there are still elements in the array. |
| `new_all_borrowed`   | `()` | `borrow_array<size, elemty>`| Create a new borrow array where all elements are borrowed. |
| `is_borrowed`         | `borrow_array<size, elemty>`, `usize` | `bool`, `borrow_array<size, elemty>` | Check if the element at the given index is borrowed. |

There are also conversion operations to convert borrow arrays to and from standard arrays:

| Operation    | Inputs       | Outputs      | Meaning      |
|--------------|--------------|--------------|--------------|
| `from_array` | `array<size, elemty>` | `borrow_array<size, elemty>` | Turn an array into a borrow array. |
| `to_array`   | `borrow_array<size, elemty>` | `array<size, elemty>` | Turn a borrow array into an array. |

### `collections.static_array`

This extension contains the `static_array` type and value for modelling constant statically sized arrays. The only way of obtaining a value of type `static_array` is by creating a `StaticArrayValue`. There are only two operations provided:

| Operation    | Inputs       | Outputs      | Meaning      |
|--------------|--------------|--------------|--------------|
| `get`        | `static_array<elemty>`, `usize` | `option<elemty>`| Get the element at the given index. Return none if the index is out of bounds. |
| `len`        | `static_array<elemty>` | `usize` | Gets the length of the array. |

### `collections.list`

This extension contains the `list` type and value. Lists are dynamically sized, with all elements having the same type.

| Operation | Inputs | Outputs | Meaning |
|-----------|--------|---------|---------|
| `pop`     | `list<elemty>` | `list<elemty>`, `option<elemty>` | Pop from the end of a list. Return the new list and the popped value (or none if the list was empty). |
| `push`    | `list<elemty>`, `elemty` | `list<elemty>` | Push to the end of a list. Return the new list. |
| `get`     | `list<elemty>`, `usize` | `option<elemty>` | Look up an element in a list by index. |
| `set`     | `list<elemty>`, `usize`, `elemty` | `list<elemty>`, `either<elemty, elemty>`,  | Replace the element at the given index, and return the old value. If the index is out of bounds, return the input value as an error. |
| `insert`  | `list<elemty>`, `usize`, `elemty` | `list<elemty>`, `either<elem_ty, ()>` | Insert an element at the given index. Elements at higher indices are shifted one position to the right. Return an error with the element if the index is out of bounds. |
| `length`  | `list<elemty>` | `list<elemty>`, `usize` | Get the length of a list. |

## Pointer Extension

The `ptr` extension provides shared mutable cells. A `ptr<T>` is a linear handle
that must be used exactly once, even when `T` is copyable. `Dup` creates another
handle to the same cell; it does not copy the stored value. Each handle must
ultimately be released with `Free`, which returns `Some(value)` when releasing
the last handle and `None` otherwise. The cell may contain a linear value.

Pointer handles are threaded through operations so that the operations can be
ordered. Passing a pointer returned by one operation into the next establishes a
dataflow dependency: the next operation waits for the first to finish. After
`Dup`, operations on the separate handles have unspecified order unless another
dependency orders them. Sharing the same cell does not itself establish an
execution order.

| Operation | Inputs | Outputs | Meaning |
|-----------|--------|---------|---------|
| `New<T>` | `T` | `ptr<T>` | Create a cell containing the input value. |
| `Read<T>` | `ptr<T>` | `ptr<T>`, `T` | Copy the stored value. Requires copyable `T`. |
| `Write<T>` | `ptr<T>`, `T` | `ptr<T>` | Replace and discard the old value. Requires copyable `T`. |
| `Swap<T>` | `ptr<T>`, `T` | `ptr<T>`, `T` | Replace the stored value and return the old value. |
| `Dup<T>` | `ptr<T>` | `ptr<T>`, `ptr<T>` | Create two handles to the same cell. |
| `Eq<T>` | `ptr<T>`, `ptr<T>` | `ptr<T>`, `ptr<T>`, `bool` | Compare cell identity, returning both handles in input order. Does not read or compare the stored values. |
| `Free<T>` | `ptr<T>` | `option<T>` | Release a handle and return the value if this was the last handle. |
| `Map<T, A, B>` | `ptr<T>`, `(T, A -> T, B)`, `A` | `ptr<T>`, `B` | Apply the function to the stored value and extra inputs, store its first result, and return its remaining results. |

`A` and `B` are rows of zero or more types, including linear types. `Map` preserves
the stored value's type and must provide exclusive access to it while the callback
runs. Operations on shared handles must not duplicate or discard a linear stored
value. Implementations must synchronize concurrent accesses to the same cell.
