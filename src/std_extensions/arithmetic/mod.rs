//! Basic arithmetic operations.

pub mod conversions;
pub mod float_ops;
pub mod float_types;
pub mod int_ops;
pub mod int_types;

#[cfg(test)]
mod test {
    use crate::{
        std_extensions::arithmetic::int_types::{int_type, INT_TYPES},
        types::type_param::TypeArg,
    };

    use super::int_types::MAX_LOG_WIDTH;

    #[test]
    fn test_int_types() {
        for i in 0..MAX_LOG_WIDTH + 1 {
            assert_eq!(
                INT_TYPES[i as usize],
                int_type(TypeArg::BoundedUSize(i as u64))
            )
        }
    }
}
