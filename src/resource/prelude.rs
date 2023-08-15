//! Prelude extension - available in all contexts, defining common types,
//! operations and constants.
use lazy_static::lazy_static;
use smol_str::SmolStr;

use crate::{resource::TypeDefBound, types::type_param::TypeParam, Resource};

lazy_static! {
    /// Prelude resource
    pub static ref PRELUDE: Resource = {
        let mut prelude = Resource::new(SmolStr::new_inline("prelude"));
        prelude
            .add_type(
                SmolStr::new_inline("float64"),
                vec![],
                "float64".into(),
                TypeDefBound::Explicit(crate::types::TypeBound::Copyable),
            )
            .unwrap();

            prelude
            .add_type(
                SmolStr::new_inline("usize"),
                vec![],
                "usize".into(),
                TypeDefBound::Explicit(crate::types::TypeBound::Eq),
            )
            .unwrap();


            prelude
            .add_type(
                SmolStr::new_inline("array"),
                vec![TypeParam::Type(None), TypeParam::USize],
                "array".into(),
                TypeDefBound::FromParams(vec![0]),
            )
            .unwrap();

            prelude
            .add_type(
                SmolStr::new_inline("qubit"),
                vec![],
                "qubit".into(),
                TypeDefBound::NoBound,
            )
            .unwrap();
        prelude
    };
}
