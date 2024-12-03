//! Tests for extension resolution.

use rstest::rstest;

use crate::extension::resolution::{update_op_extensions, update_op_types_extensions};
use crate::extension::ExtensionRegistry;
use crate::ops::{Input, OpType};
use crate::std_extensions::arithmetic::int_ops;
use crate::std_extensions::arithmetic::int_types;
use crate::type_row;

#[rstest]
#[case::empty(Input { types: type_row![]}, ExtensionRegistry::default())]
// A type with extra extensions in its instantiated type arguments.
#[case::parametric_op(int_ops::IntOpDef::ieq.with_log_width(4),
    ExtensionRegistry::new([int_ops::EXTENSION.to_owned(), int_types::EXTENSION.to_owned()]
))]
fn collect_type_extensions(#[case] op: impl Into<OpType>, #[case] extensions: ExtensionRegistry) {
    let op = op.into();
    let resolved = op.used_extensions().unwrap();
    assert_eq!(resolved, extensions);
}

#[rstest]
#[case::empty(Input { types: type_row![]}, ExtensionRegistry::default())]
// A type with extra extensions in its instantiated type arguments.
#[case::parametric_op(int_ops::IntOpDef::ieq.with_log_width(4),
    ExtensionRegistry::new([int_ops::EXTENSION.to_owned(), int_types::EXTENSION.to_owned()]
))]
fn resolve_type_extensions(#[case] op: impl Into<OpType>, #[case] extensions: ExtensionRegistry) {
    let op = op.into();

    // Ensure that all the `Weak` pointers get invalidated by round-tripping via serialization.
    let ser = serde_json::to_string(&op).unwrap();
    let mut deser_op: OpType = serde_json::from_str(&ser).unwrap();

    let dummy_node = portgraph::NodeIndex::new(0).into();

    let mut used_exts = ExtensionRegistry::default();
    update_op_extensions(dummy_node, &mut deser_op, &extensions).unwrap();
    update_op_types_extensions(dummy_node, &mut deser_op, &extensions, &mut used_exts).unwrap();

    let deser_extensions = deser_op.used_extensions().unwrap();

    assert_eq!(
        deser_extensions, extensions,
        "{deser_extensions} != {extensions}"
    );
}
