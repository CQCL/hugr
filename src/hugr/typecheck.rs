//! Simple type checking - takes a hugr and some extra info and checks whether
//! the types at the sources of each wire match those of the targets

use crate::hugr::*;
use crate::types::{SimpleType, TypeRow};

// For static typechecking
use crate::ops::ConstValue;
use crate::types::{ClassicType, Container};

use crate::ops::module::{HugrIntValueStore, HugrIntWidthStore, HUGR_MAX_INT_WIDTH};

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
    IntTooLarge(HugrIntWidthStore, HugrIntValueStore),
    /// Width (n) of an `I<n>` type doesn't fit into a HugrIntWidthStore
    IntWidthTooLarge(HugrIntWidthStore),
    /// The width of an integer type wasn't a power of 2
    IntWidthInvalid(HugrIntWidthStore),
    /// Expected width (packed with const int) doesn't match type
    IntWidthMismatch(HugrIntWidthStore, HugrIntWidthStore),
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
            TypeError::IntWidthTooLarge(w) => format!("Int type too large: I{}", w),
            TypeError::IntWidthInvalid(w) => {
                format!(
                    "The int type I{} is invalid, because {} is not a power of 2",
                    w, w
                )
            }
            TypeError::IntWidthMismatch(exp, act) => format!(
                "Type mismatch for int: expected I{}, but found I{}",
                exp, act
            ),
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

/// Per the spec, valid widths for integers are 2^n for all n in [0,7]
fn check_valid_width(width: HugrIntWidthStore) -> Result<(), TypeError> {
    if width > HUGR_MAX_INT_WIDTH {
        return Err(TypeError::IntWidthTooLarge(width));
    }

    let valid_widths: Vec<HugrIntWidthStore> =
        (0..8).map(|a| HugrIntWidthStore::pow(2, a)).collect();
    if valid_widths.contains(&width) {
        Ok(())
    } else {
        Err(TypeError::IntWidthInvalid(width))
    }
}

/// Typecheck a constant value
pub fn typecheck_const(typ: &ClassicType, val: &ConstValue) -> Result<(), TypeError> {
    match (typ, val) {
        (ClassicType::Int(exp_width), ConstValue::Int { value, width }) => {
            // Check that the types make sense
            check_valid_width(*exp_width)?;
            check_valid_width(*width)?;
            // Check that the terms make sense against the types
            if exp_width == width {
                let max_value = HugrIntValueStore::pow(2, *width as u32);
                if value < &max_value {
                    Ok(())
                } else {
                    Err(TypeError::IntTooLarge(*width, *value))
                }
            } else {
                Err(TypeError::IntWidthMismatch(*exp_width, *width))
            }
        }
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
