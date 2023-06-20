//! Type Arguments
//!
//! Values for TypeParams declared in [`OpDef`]s.
//!
//! [`OpDef`]: crate::resource::OpDef

use crate::types::{simple::HInt, ClassicType, SimpleType};

/// An argument value for a type parameter
pub enum TypeArgValue {
    /// Where the OpDef declares that an argument is a `Type`
    Type(SimpleType),
    /// Where the OpDef declares that an argument is a `ClassicType`,
    /// it'll get one of these (rather than embedding inside a Type)
    ClassicType(ClassicType),
    /// Where the OpDef declares that an argument is an `F64`
    F64(f64),
    /// Where the OpDef declares an argument that's an `Int`
    /// - using the same representation as [`ClassicType`].
    Int(HInt),
    /// Serialized representation of an opaque value...?
    Opaque(Vec<u8>),
    /// Where an argument has type `List<T>` - all elements will implicitly
    /// be of the same variety of TypeArgValue, representing a T.
    List(Vec<TypeArgValue>),
}
