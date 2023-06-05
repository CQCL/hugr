//! Simple type checking - takes a hugr and some extra info and checks whether
//! the types at the sources of each wire match those of the targets

use crate::hugr::*;
use crate::types::{SimpleType, TypeRow};

// For static typechecking
use crate::ops::ConstValue;
use crate::types::{ClassicType, Container};

use std::fmt::{self, Display};

/// Errors that arise from typechecking constants
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum TypeError {
    /// This case hasn't been implemented. Possibly because we don't have value
    /// constructors to check against it
    Unimplemented(ClassicType),
    /// The given type and term are incompatible
    Failed(ClassicType),
    /// The value exceeds the max value of its `I<n>` type
    /// E.g. checking 300 against I8
    IntTooLarge(u32, isize),
    /// Width (n) of an `I<n>` type doesn't fit into a u32
    IntTypeTooLarge(usize),
    /// Expected width (packed with const int) doesn't match type
    IntWidthMismatch(usize, usize),
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
    TypeMismatch(ClassicType, ClassicType),
    /// A mismatch between the embedded type and the type we're checking
    /// against, as above, but for rows instead of simple types
    TypeRowMismatch(TypeRow, TypeRow),
}

impl Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str = match self {
            TypeError::Unimplemented(ty) => {
                format!("Const type checking unimplemented for {}", ty)
            }
            TypeError::Failed(typ) => format!("Invalid const value for type {}", typ),
            TypeError::IntTooLarge(width, val) => {
                format!("Const int {} too large for type I{}", val, width)
            }
            TypeError::IntTypeTooLarge(w) => format!("Int type too large: I{}", w),
            TypeError::IntWidthMismatch(exp, act) => format!("Type mismatch for int: expected I{}, but found I{}", exp, act),
            TypeError::ConstCantBeVar => "Type of a const value can't be Var".to_string(),
            TypeError::TupleWrongLength => "Tuple of wrong length".to_string(),
            TypeError::LinearTypeDisallowed => {
                "Linear types not allowed in const nodes".to_string()
            }
            TypeError::InvalidSumTag => "Tag of Sum value is invalid".to_string(),
            TypeError::TypeMismatch(exp, act) => {
                format!("Type mismatch for const - expected {}, found {}", exp, act)
            }
            TypeError::TypeRowMismatch(exp, act) => {
                format!("Type mismatch for const - expected {}, found {}", exp, act)
            }
        };
        f.write_str(&str)
    }
}

/// Typecheck a constant value - for ensuring that it's valid before creating
/// a const node
pub fn typecheck_const(typ: &ClassicType, val: &ConstValue) -> Result<(), TypeError> {
    match (typ, val) {
        // Const int widths are here being limited to the range of u32, but if
        // the width is larger than u6, our const type (which takes an i64 arg)
        // wont be able to accomodate the value anyway.
        // N.B. This diverges from the spec, which allows arbitrary ints as constants
        (ClassicType::Int(exp_width), ConstValue::Int { value, width }) => if exp_width == width {
            match u32::try_from(*width) {
                Ok(width) => if isize::abs(*value as isize) < isize::pow(2, width as u32) {
                    Ok(())
                } else {
                    Err(TypeError::IntTooLarge(width, *value as isize))
                },
                _ => Err(TypeError::IntTypeTooLarge(*width))
            }
        } else {
            Err(TypeError::IntWidthMismatch(*exp_width, *width))
        },
        (ty @ ClassicType::F64, _) => Err(TypeError::Unimplemented(ty.clone())),
        (ty @ ClassicType::Container(c), tm) => match (c, tm) {
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
                    return Err(TypeError::TypeRowMismatch(*row.clone(), variants.clone()));
                }
                let ty = variants.get(*tag).unwrap();
                match ty {
                    SimpleType::Classic(ty) => typecheck_const(ty, val.as_ref()),
                    _ => Err(TypeError::LinearTypeDisallowed),
                }
            }
            _ => Err(TypeError::Unimplemented(ty.clone())),
        },
        (ty @ ClassicType::Graph(_), _) => Err(TypeError::Unimplemented(ty.clone())),
        (ty @ ClassicType::String, _) => Err(TypeError::Unimplemented(ty.clone())),
        (ClassicType::Variable(_), _) => Err(TypeError::ConstCantBeVar),
        (ClassicType::Opaque(ty), ConstValue::Opaque(_tm, ty2)) => {
            // The type we're checking against
            let ty_exp = ty.clone().classic_type();
            let ty_act = ty2.const_type();
            if ty_exp != ty_act {
                return Err(TypeError::TypeMismatch(ty_exp, ty_act));
            }
            Ok(())
        }
        (ty, _) => Err(TypeError::Failed(ty.clone())),
    }
}
