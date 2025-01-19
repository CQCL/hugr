//! The algorithm for computing the fields of the struct representing a
//! [HugrSumType].
//!
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::RangeFrom;

use inkwell::types::{BasicType, BasicTypeEnum};
use itertools::Itertools as _;

use super::elidable_type;

/// Compute the layout of the non-tag fields of the struct representing a
/// HugrSumType
///
/// The first return value are the non-tag fields of the struct.
///
/// The second return value is a per-variant-per-field `Option<usize>`. If that
/// value is `None`, then the field should be elided from the struct.  Values
/// for this field should be materialized via `undef`. If that value is
/// `Some(i)` then values of this field should be read and written to the `i`th
/// field of the struct.
pub(super) fn layout_variants<'c>(
    variants: impl AsRef<[Vec<BasicTypeEnum<'c>>]>,
) -> (Vec<BasicTypeEnum<'c>>, Vec<Vec<Option<usize>>>) {
    let variants = variants.as_ref();
    let (sorted_fields, layout) = layout_variants_impl::<BasicTypeOrd>(
        variants
            .iter()
            .cloned()
            .map(|x| x.into_iter().map_into().collect_vec())
            .collect_vec(),
        |x| elidable_type(x.0),
    );
    let sorted_fields = sorted_fields.into_iter().map_into().collect_vec();
    (sorted_fields, layout)
}

fn size_of_type<'c>(t: impl BasicType<'c>) -> Option<u64> {
    match t.as_basic_type_enum() {
        BasicTypeEnum::ArrayType(t) => t.size_of().and_then(|x| x.get_zero_extended_constant()),
        BasicTypeEnum::FloatType(t) => t.size_of().get_zero_extended_constant(),
        BasicTypeEnum::IntType(t) => t.size_of().get_zero_extended_constant(),
        BasicTypeEnum::PointerType(t) => t.size_of().get_zero_extended_constant(),
        BasicTypeEnum::StructType(t) => t.size_of().and_then(|x| x.get_zero_extended_constant()),
        BasicTypeEnum::VectorType(t) => t.size_of().and_then(|x| x.get_zero_extended_constant()),
    }
}

#[derive(derive_more::Debug, Clone, PartialEq, Eq)]
/// `BasicTypeEnum` does not `impl` `Ord`. We use this type to put an ordering `BasicTypeEnum`,
/// first by reverse-size and then by, string representation.
struct BasicTypeOrd<'c>(
    BasicTypeEnum<'c>,
    #[debug(skip)] u64,
    #[debug(skip)] Cow<'c, String>,
);

impl<'c> From<BasicTypeEnum<'c>> for BasicTypeOrd<'c> {
    fn from(t: BasicTypeEnum<'c>) -> Self {
        let size = size_of_type(t).unwrap_or(u64::MAX);
        Self(t, size, Cow::Owned(format!("{t}")))
    }
}

impl<'c> From<BasicTypeOrd<'c>> for BasicTypeEnum<'c> {
    fn from(t: BasicTypeOrd<'c>) -> Self {
        t.0
    }
}

impl Ord for BasicTypeOrd<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let key = |x: &Self| Reverse((x.1, x.2.clone()));
        key(self).cmp(&key(other))
    }
}

impl PartialOrd for BasicTypeOrd<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The implemenation of the layout algorithm.
/// We write this generically so that we can test it with simple types.
///
/// Panics if `variants` is empty.
fn layout_variants_impl<T: Ord + Clone + fmt::Debug>(
    variants: impl AsRef<[Vec<T>]>,
    elide: impl Fn(&T) -> bool,
) -> (Vec<T>, Vec<Vec<Option<usize>>>) {
    let variants = variants.as_ref();
    assert!(!variants.is_empty());
    // * sorted_fields is a Vec<T> with enough copies of each T to represent any
    //   one variant. It will be the first return value.
    // * t_to_index_map maps types to the index of the first
    //   occurence of that type.
    let (sorted_fields, t_to_index_map) = {
        // t_counts tracks, per-type, the maximum number of fields a variant
        // has of that type.
        let t_counts = {
            let t_counts_per_variant = variants.iter().map(|variant| {
                let mut t_counts = BTreeMap::<T, usize>::new();
                for t in variant.iter().flat_map(|t| (!elide(t)).then_some(t)) {
                    t_counts
                        .entry(t.clone())
                        .and_modify(|count| *count += 1)
                        .or_insert(1);
                }
                t_counts
            });
            let mut t_counts = BTreeMap::<T, usize>::new();
            for (t, count) in t_counts_per_variant.flatten() {
                t_counts
                    .entry(t)
                    .and_modify(|x| *x = count.max(*x))
                    .or_insert(count);
            }
            t_counts
        };
        let mut t_to_index_map = BTreeMap::<T, usize>::default();
        let mut last_t = None;
        let sorted_fields = t_counts
            .into_iter()
            .flat_map(|(t, count)| itertools::repeat_n(t, count))
            .enumerate()
            .map(|(i, t)| {
                if last_t.as_ref() != Some(&t) {
                    last_t = Some(t.clone());
                    let _overwritten = t_to_index_map.insert(t.clone(), i).is_some();
                    debug_assert!(!_overwritten);
                }
                t
            })
            .collect_vec();
        (sorted_fields, t_to_index_map)
    };

    // the second return value. Here we record, per-variant, which field of
    // `sorted_fields` represents each field.
    let layout = variants
        .iter()
        .map(|variant| {
            let mut t_to_range_map = BTreeMap::<T, RangeFrom<usize>>::new();
            variant
                .iter()
                .map(|t| {
                    (!elide(t)).then(|| {
                        let field_index_iter = t_to_range_map
                            .entry(t.clone())
                            .or_insert(t_to_index_map[t]..);
                        field_index_iter
                            .next()
                            .expect("We have ensured that there are enough fields of type t")
                    })
                })
                .collect_vec()
        })
        .collect_vec();

    #[cfg(debug_assertions)]
    {
        for (variant, variant_layout) in itertools::zip_eq(variants, layout.iter()) {
            for (t, &mb_field_index) in itertools::zip_eq(variant, variant_layout) {
                if elide(t) {
                    assert!(mb_field_index.is_none())
                } else {
                    assert_eq!(
                        Some(t),
                        mb_field_index.map(|field_index| &sorted_fields[field_index])
                    );
                }
            }
        }
    }

    (sorted_fields, layout)
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[should_panic]
    #[case::none([], [], [])]
    #[case::one_empty([vec![]], [vec![]], [])]
    #[case::multi_empty([vec![],vec![]], [vec![],vec![]], [])]
    #[case::one_nonempty([vec![5]], [vec![Some(0)]], [5])]
    #[case::one_nonempty_dups([vec![5,5]], [vec![Some(0),Some(1)]], [5,5])]
    #[case::one_nonempty_all_elidable([vec![0,0]], [vec![None,None]], [])]
    #[case::one_nonempty_some_elidable([vec![0, 1]], [vec![None,Some(0)]], [1])]
    #[case::one_nonempty_one_empty([vec![5],vec![]], [vec![Some(0)],vec![]], [5])]
    #[case::two_nonempty_no_dups_in_order([vec![5],vec![6]], [vec![Some(0)],vec![Some(1)]], [5,6])]
    #[case::two_nonempty_no_dups_rev_order([vec![8],vec![7]], [vec![Some(1)],vec![Some(0)]], [7,8])]
    #[case::two_nonempty_all_dups([vec![9,10],vec![10,9]], [vec![Some(0),Some(1)],vec![Some(1),Some(0)]], [9,10])]
    #[case::three_nonempty_some_dups([vec![9,10],vec![9],vec![11,10,-1]], [vec![Some(1),Some(2)],vec![Some(1)],vec![Some(3),Some(2),Some(0)]], [-1,9,10,11])]
    // #[case::two_nonempty_all_elidable([vec![0],vec![0],vec![11,10,-1]], [vec![Some(1),Some(2)],vec![Some(1)],vec![Some(3),Some(2),Some(0)]], [-1,9,10,11])]
    fn layout_variants(
        #[case] variants: impl AsRef<[Vec<i32>]>,
        #[case] expected_layout: impl AsRef<[Vec<Option<usize>>]>,
        #[case] expected_fields: impl AsRef<[i32]>,
    ) {
        fn elidable(&x: &i32) -> bool {
            x == 0
        }
        let (sorted_fields, layout) = layout_variants_impl(variants, elidable);
        assert_eq!(sorted_fields.as_slice(), expected_fields.as_ref());
        assert_eq!(layout.as_slice(), expected_layout.as_ref());
    }
}