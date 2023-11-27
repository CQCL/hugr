//! Basic integer operations.

use super::int_types::{get_log_width, int_type_var, LOG_WIDTH_TYPE_PARAM};
use crate::extension::prelude::{sum_with_error, BOOL_T};
use crate::extension::{CustomValidator, ValidateJustArgs};
use crate::type_row;
use crate::types::{FunctionType, PolyFuncType};
use crate::utils::collect_array;
use crate::{
    extension::{ExtensionId, ExtensionSet, SignatureError},
    types::{type_param::TypeArg, Type, TypeRow},
    Extension,
};

use lazy_static::lazy_static;

/// The extension identifier.
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("arithmetic.int");

struct IOValidator {
    // whether the first type argument should be greater than the second
    f_gt_s: bool,
}

impl ValidateJustArgs for IOValidator {
    fn validate(&self, arg_values: &[TypeArg]) -> Result<(), SignatureError> {
        let [arg0, arg1] = collect_array(arg_values);
        let i: u8 = get_log_width(arg0)?;
        let o: u8 = get_log_width(arg1)?;
        let cmp = if self.f_gt_s { i > o } else { i < o };
        if !cmp {
            return Err(SignatureError::InvalidTypeArgs);
        }
        Ok(())
    }
}

fn int_polytype(
    n_vars: usize,
    input: impl Into<TypeRow>,
    output: impl Into<TypeRow>,
) -> PolyFuncType {
    PolyFuncType::new(
        vec![LOG_WIDTH_TYPE_PARAM; n_vars],
        FunctionType::new(input, output),
    )
}

fn ibinop_sig() -> PolyFuncType {
    let int_type_var = int_type_var(0);

    int_polytype(1, vec![int_type_var.clone(); 2], vec![int_type_var])
}

fn iunop_sig() -> PolyFuncType {
    let int_type_var = int_type_var(0);
    int_polytype(1, vec![int_type_var.clone()], vec![int_type_var])
}

fn idivmod_checked_sig() -> PolyFuncType {
    let intpair: TypeRow = vec![int_type_var(0), int_type_var(1)].into();
    int_polytype(
        2,
        intpair.clone(),
        vec![sum_with_error(Type::new_tuple(intpair))],
    )
}

fn idivmod_sig() -> PolyFuncType {
    let intpair: TypeRow = vec![int_type_var(0), int_type_var(1)].into();
    int_polytype(2, intpair.clone(), vec![Type::new_tuple(intpair)])
}

/// Extension for basic integer operations.
fn extension() -> Extension {
    let itob_sig = int_polytype(1, vec![int_type_var(0)], type_row![BOOL_T]);

    let btoi_sig = int_polytype(1, type_row![BOOL_T], vec![int_type_var(0)]);

    let icmp_sig = int_polytype(1, vec![int_type_var(0); 2], type_row![BOOL_T]);

    let idiv_checked_sig = int_polytype(
        2,
        vec![int_type_var(0), int_type_var(1)],
        vec![sum_with_error(int_type_var(0))],
    );

    let idiv_sig = int_polytype(
        2,
        vec![int_type_var(1), int_type_var(1)],
        vec![int_type_var(0)],
    );

    let imod_checked_sig = int_polytype(
        2,
        vec![int_type_var(0), int_type_var(1).clone()],
        vec![sum_with_error(int_type_var(1))],
    );

    let imod_sig = int_polytype(
        2,
        vec![int_type_var(0), int_type_var(1).clone()],
        vec![int_type_var(1)],
    );

    let ish_sig = int_polytype(2, vec![int_type_var(1)], vec![int_type_var(0)]);

    let widen_poly = int_polytype(2, vec![int_type_var(0)], vec![int_type_var(1)]);
    let narrow_poly = int_polytype(
        2,
        vec![int_type_var(0)],
        vec![sum_with_error(int_type_var(1))],
    );
    let mut extension = Extension::new_with_reqs(
        EXTENSION_ID,
        ExtensionSet::singleton(&super::int_types::EXTENSION_ID),
    );

    extension
        .add_op(
            "iwiden_u".into(),
            "widen an unsigned integer to a wider one with the same value".to_owned(),
            CustomValidator::new_with_validator(widen_poly.clone(), IOValidator { f_gt_s: false }),
        )
        .unwrap();

    extension
        .add_op(
            "iwiden_s".into(),
            "widen a signed integer to a wider one with the same value".to_owned(),
            CustomValidator::new_with_validator(widen_poly, IOValidator { f_gt_s: false }),
        )
        .unwrap();
    extension
        .add_op(
            "inarrow_u".into(),
            "narrow an unsigned integer to a narrower one with the same value if possible"
                .to_owned(),
            CustomValidator::new_with_validator(narrow_poly.clone(), IOValidator { f_gt_s: true }),
        )
        .unwrap();
    extension
        .add_op(
            "inarrow_s".into(),
            "narrow a signed integer to a narrower one with the same value if possible".to_owned(),
            CustomValidator::new_with_validator(narrow_poly, IOValidator { f_gt_s: true }),
        )
        .unwrap();
    extension
        .add_op(
            "itobool".into(),
            "convert to bool (1 is true, 0 is false)".to_owned(),
            itob_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "ifrombool".into(),
            "convert from bool (1 is true, 0 is false)".to_owned(),
            btoi_sig.clone(),
        )
        .unwrap();
    extension
        .add_op("ieq".into(), "equality test".to_owned(), icmp_sig.clone())
        .unwrap();
    extension
        .add_op("ine".into(), "inequality test".to_owned(), icmp_sig.clone())
        .unwrap();
    extension
        .add_op(
            "ilt_u".into(),
            "\"less than\" as unsigned integers".to_owned(),
            icmp_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "ilt_s".into(),
            "\"less than\" as signed integers".to_owned(),
            icmp_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "igt_u".into(),
            "\"greater than\" as unsigned integers".to_owned(),
            icmp_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "igt_s".into(),
            "\"greater than\" as signed integers".to_owned(),
            icmp_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "ile_u".into(),
            "\"less than or equal\" as unsigned integers".to_owned(),
            icmp_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "ile_s".into(),
            "\"less than or equal\" as signed integers".to_owned(),
            icmp_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "ige_u".into(),
            "\"greater than or equal\" as unsigned integers".to_owned(),
            icmp_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "ige_s".into(),
            "\"greater than or equal\" as signed integers".to_owned(),
            icmp_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "imax_u".into(),
            "maximum of unsigned integers".to_owned(),
            ibinop_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "imax_s".into(),
            "maximum of signed integers".to_owned(),
            ibinop_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "imin_u".into(),
            "minimum of unsigned integers".to_owned(),
            ibinop_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "imin_s".into(),
            "minimum of signed integers".to_owned(),
            ibinop_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "iadd".into(),
            "addition modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            ibinop_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "isub".into(),
            "subtraction modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            ibinop_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "ineg".into(),
            "negation modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            iunop_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "imul".into(),
            "multiplication modulo 2^N (signed and unsigned versions are the same op)".to_owned(),
            ibinop_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "idivmod_checked_u".into(),
            "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^M, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 is an error)"
                .to_owned(),
            idivmod_checked_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "idivmod_u".into(),
            "given unsigned integers 0 <= n < 2^N, 0 <= m < 2^M, generates unsigned q, r where \
            q*m+r=n, 0<=r<m (m=0 will call panic)"
                .to_owned(),
            idivmod_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "idivmod_checked_s".into(),
            "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^M, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 is an error)"
                .to_owned(),
            idivmod_checked_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "idivmod_s".into(),
            "given signed integer -2^{N-1} <= n < 2^{N-1} and unsigned 0 <= m < 2^M, generates \
            signed q and unsigned r where q*m+r=n, 0<=r<m (m=0 will call panic)"
                .to_owned(),
            idivmod_sig(),
        )
        .unwrap();
    extension
        .add_op(
            "idiv_checked_u".into(),
            "as idivmod_checked_u but discarding the second output".to_owned(),
            idiv_checked_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "idiv_u".into(),
            "as idivmod_u but discarding the second output".to_owned(),
            idiv_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "imod_checked_u".into(),
            "as idivmod_checked_u but discarding the first output".to_owned(),
            imod_checked_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "imod_u".into(),
            "as idivmod_u but discarding the first output".to_owned(),
            imod_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "idiv_checked_s".into(),
            "as idivmod_checked_s but discarding the second output".to_owned(),
            idiv_checked_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "idiv_s".into(),
            "as idivmod_s but discarding the second output".to_owned(),
            idiv_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "imod_checked_s".into(),
            "as idivmod_checked_s but discarding the first output".to_owned(),
            imod_checked_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "imod_s".into(),
            "as idivmod_s but discarding the first output".to_owned(),
            imod_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "iabs".into(),
            "convert signed to unsigned by taking absolute value".to_owned(),
            iunop_sig(),
        )
        .unwrap();
    extension
        .add_op("iand".into(), "bitwise AND".to_owned(), ibinop_sig())
        .unwrap();
    extension
        .add_op("ior".into(), "bitwise OR".to_owned(), ibinop_sig())
        .unwrap();
    extension
        .add_op("ixor".into(), "bitwise XOR".to_owned(), ibinop_sig())
        .unwrap();
    extension
        .add_op("inot".into(), "bitwise NOT".to_owned(), iunop_sig())
        .unwrap();
    extension
        .add_op(
            "ishl".into(),
            "shift first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits dropped, rightmost bits set to zero"
                .to_owned(),
            ish_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "ishr".into(),
            "shift first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits dropped, leftmost bits set to zero)"
                .to_owned(),
            ish_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "irotl".into(),
            "rotate first input left by k bits where k is unsigned interpretation of second input \
            (leftmost bits replace rightmost bits)"
                .to_owned(),
            ish_sig.clone(),
        )
        .unwrap();
    extension
        .add_op(
            "irotr".into(),
            "rotate first input right by k bits where k is unsigned interpretation of second input \
            (rightmost bits replace leftmost bits)"
                .to_owned(),
            ish_sig.clone(),
        )
        .unwrap();

    extension
}

lazy_static! {
    /// Extension for basic integer operations.
    pub static ref EXTENSION: Extension = extension();
}

#[cfg(test)]
mod test {
    use crate::{
        extension::{ExtensionRegistry, PRELUDE},
        std_extensions::arithmetic::int_types::int_type,
    };

    use super::*;

    #[test]
    fn test_int_ops_extension() {
        assert_eq!(EXTENSION.name() as &str, "arithmetic.int");
        assert_eq!(EXTENSION.types().count(), 0);
        for (name, _) in EXTENSION.operations() {
            assert!(name.starts_with('i'));
        }
    }

    const fn ta(n: u64) -> TypeArg {
        TypeArg::BoundedNat { n }
    }
    #[test]
    fn test_binary_signatures() {
        let iwiden_s = EXTENSION.get_op("iwiden_s").unwrap();
        let reg = ExtensionRegistry::try_new([
            EXTENSION.to_owned(),
            super::super::int_types::EXTENSION.to_owned(),
            PRELUDE.to_owned(),
        ])
        .unwrap();
        assert_eq!(
            iwiden_s.compute_signature(&[ta(3), ta(4)], &reg).unwrap(),
            FunctionType::new(vec![int_type(ta(3))], vec![int_type(ta(4))],)
        );

        let iwiden_u = EXTENSION.get_op("iwiden_u").unwrap();
        iwiden_u
            .compute_signature(&[ta(4), ta(3)], &reg)
            .unwrap_err();

        let inarrow_s = EXTENSION.get_op("inarrow_s").unwrap();

        assert_eq!(
            inarrow_s.compute_signature(&[ta(2), ta(1)], &reg).unwrap(),
            FunctionType::new(vec![int_type(ta(2))], vec![sum_with_error(int_type(ta(1)))],)
        );

        let inarrow_u = EXTENSION.get_op("inarrow_u").unwrap();
        inarrow_u
            .compute_signature(&[ta(1), ta(2)], &reg)
            .unwrap_err();
    }
}
