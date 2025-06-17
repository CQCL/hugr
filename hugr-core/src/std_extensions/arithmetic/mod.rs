//! Basic arithmetic operations.

pub mod conversions;
pub mod float_ops;
pub mod float_types;
pub mod int_ops;
pub mod int_types;

#[cfg(test)]
mod test {
    use crate::{
        std_extensions::arithmetic::int_types::{INT_TYPES, int_type},
        types::type_param::TypeArg,
    };

    use super::int_types::LOG_WIDTH_BOUND;

    #[test]
    fn test_int_types() {
        for i in 0..LOG_WIDTH_BOUND {
            assert_eq!(
                INT_TYPES[i as usize],
                int_type(TypeArg::BoundedNat(u64::from(i)))
            );
        }
    }
}
