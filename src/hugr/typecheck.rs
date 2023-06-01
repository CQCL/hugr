//! Simple type checking - takes a hugr and some extra info and checks whether
//! the types at the sources of each wire match those of the targets

use crate::hugr::*;
use crate::types::SimpleType;

// For static typechecking
use crate::ops::ConstValue;
use crate::types::{ClassicType, Container};

use std::fmt::{self, Display};

/// Errors that arise from typechecking constants
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum TypeError {
    /// This case hasn't been implemented (possibly because we don't have value
    /// variants to check against it
    Unimplemented,
    /// The given type and term are incompatible
    Failed,
    /// The value exceeds the max value of its `I<n>` type
    /// E.g. checking 300 against I8
    IntTooLarge,
    /// Width (n) of an `I<n>` type doesn't fit into a u32
    IntTypeTooLarge,
    /// Found a Var type constructor when we're checking a const val
    ConstCantBeVar,
    /// The length of the tuple value doesn't match the length of the tuple type
    TupleWrongLength,
    /// Const values aren't allowed to be linear
    LinearTypeDisallowed,
    /// Tag for a sum value exceeded the number of variants
    InvalidSumTag,
    /// For a value which embeds its type (e.g. sum or opaque) - a mismatch
    /// between the embedded type and the type we're checking against
    TypeMismatch,
}

impl Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str = match self {
            TypeError::Unimplemented => "Unimplemented",
            TypeError::Failed => "Typechecking failed",
            TypeError::IntTooLarge => "Int too large for type",
            TypeError::IntTypeTooLarge => "Int type too large",
            TypeError::ConstCantBeVar => "Type of a const value can't be Var",
            TypeError::TupleWrongLength => "Tuple of wrong length",
            TypeError::LinearTypeDisallowed => "Linear types not allowed in const nodes",
            TypeError::InvalidSumTag => "Tag of Sum value is invalid",
            TypeError::TypeMismatch => "Type mismatch",
        };
        f.write_str(str)
    }
}

/// Typecheck a constant value - for ensuring that it's valid before creating
/// a const node
pub fn typecheck_const(typ: &ClassicType, val: &ConstValue) -> Result<(), TypeError> {
    match (typ, val) {
        // If the width is larger than u8, our const type (which takes an i64 arg)
        // wont be able to accomodate the value
        // N.B. This diverges from the spec, which allows arbitrary ints as constants
        (ClassicType::Int(width), ConstValue::Int(n)) => match u8::try_from(*width) {
            Ok(width) => {
                if isize::abs(*n as isize) < isize::pow(2, width as u32) {
                    Ok(())
                } else {
                    Err(TypeError::IntTooLarge)
                }
            }
            Err(_) => Err(TypeError::IntTypeTooLarge),
        },
        (ClassicType::F64, _) => Err(TypeError::Unimplemented),
        (ClassicType::Container(c), tm) => match (c, tm) {
            (Container::Tuple(row), ConstValue::Tuple(xs)) => {
                if row.len() != xs.len() {
                    return Err(TypeError::TupleWrongLength);
                }
                for (ty, tm) in row.iter().zip(xs.iter()) {
                    match ty {
                        SimpleType::Classic(ty) => typecheck_const(ty, tm)?,
                        _ => return Err(TypeError::LinearTypeDisallowed),
                    }
                }
                Ok(())
            }
            (Container::Sum(row), ConstValue::Sum { tag, variants, val }) => {
                if tag > &row.len() {
                    return Err(TypeError::InvalidSumTag);
                }
                if **row != *variants {
                    return Err(TypeError::TypeMismatch);
                }
                let ty = variants.get(*tag).unwrap();
                match ty {
                    SimpleType::Classic(ty) => typecheck_const(ty, val.as_ref()),
                    _ => Err(TypeError::LinearTypeDisallowed),
                }
            }
            _ => Err(TypeError::Unimplemented),
        },
        (ClassicType::Graph(_), _) => Err(TypeError::Unimplemented),
        (ClassicType::String, _) => Err(TypeError::Unimplemented),
        (ClassicType::Variable(_), _) => Err(TypeError::ConstCantBeVar),
        (ClassicType::Opaque(ty), ConstValue::Opaque(_tm, ty2)) => {
            if ty.clone().classic_type() != ty2.const_type() {
                return Err(TypeError::TypeMismatch);
            }
            Ok(())
        }
        _ => Err(TypeError::Failed),
    }
}
