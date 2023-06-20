//! Type Parameters
//!
//! Parameters for [`OpDef`]s provided by extensions
//!
//! [`OpDef`]: crate::resource::OpDef

use super::{simple::HInt, ClassicType, SimpleType, TypeRow};

/// A Type Parameter declared by an OpDef. Specifies
/// the values that must be provided by each operation node.
pub enum TypeParam {
    /// Node must provide a [TypeArgValue::Type] - classic or linear
    Type,
    /// Node must provide a [TypeArgValue::ClassicType]
    ClassicType,
    /// Node must provide a [TypeArgValue::F64] floating-point value
    F64,
    /// Node must provide a [TypeArgValue::Int] integer
    Int,
    /// Node must provide some Opaque data in an [TypeArgValue::Opaque].
    /// TODO is the typerow here correct?
    Opaque(String, Box<TypeRow>),
    /// Node must provide a [TypeArgValue::List] (of whatever length)
    List(Box<TypeParam>)
}

/// An argument value for a type parameter
pub enum TypeArgValue {
    /// Where the OpDef declares that an argument is a [TypeParam::Type]
    Type(SimpleType),
    /// Where the OpDef declares that an argument is a [TypeParam::ClassicType],
    /// it'll get one of these (rather than embedding inside a Type)
    ClassicType(ClassicType),
    /// Where the OpDef declares that an argument is a [TypeParam::F64]
    F64(f64),
    /// Where the OpDef declares an argument that's a [TypeParam::Int]
    /// - using the same representation as [`ClassicType`].
    Int(HInt),
    /// Where the OpDef declares a [TypeParam::Opaque], this must be the
    /// serialized representation of such a value....??
    Opaque(Vec<u8>),
    /// Where an argument has type [TypeParam::List]`<T>` - all elements will implicitly
    /// be of the same variety of TypeArgValue, representing a `T`.
    List(Vec<TypeArgValue>),
}
