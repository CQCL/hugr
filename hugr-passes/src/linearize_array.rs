//! Provides [LinearizeArrayPass] which turns 'value_array`s into regular linear `array`s.

use hugr_core::{
    Node,
    extension::{
        prelude::Noop,
        simple_op::{HasConcrete, MakeOpDef as _, MakeRegisteredOp},
    },
    hugr::hugrmut::HugrMut,
    std_extensions::collections::{
        array::{
            ARRAY_REPEAT_OP_ID, ARRAY_SCAN_OP_ID, Array, ArrayKind, ArrayOpDef, ArrayRepeatDef,
            ArrayScanDef, ArrayValue, array_type_parametric,
        },
        value_array::{self, VArrayFromArrayDef, VArrayToArrayDef, VArrayValue, ValueArray},
    },
    types::Transformable,
};
use itertools::Itertools;
use strum::IntoEnumIterator;

use crate::{
    ComposablePass, ReplaceTypes,
    replace_types::{DelegatingLinearizer, NodeTemplate, ReplaceTypesError},
};

/// A HUGR -> HUGR pass that turns 'value_array`s into regular linear `array`s.
///
/// # Panics
///
/// - If the Hugr has inter-graph edges whose type contains `value_array`s
/// - If the Hugr contains [`ArrayOpDef::get`] operations on `value_array`s that
///   contain nested `value_array`s.
#[derive(Clone)]
pub struct LinearizeArrayPass(ReplaceTypes);

impl Default for LinearizeArrayPass {
    fn default() -> Self {
        let mut pass = ReplaceTypes::default();
        pass.replace_parametrized_type(ValueArray::type_def(), |args| {
            Some(Array::ty_parametric(args[0].clone(), args[1].clone()).unwrap())
        });
        pass.replace_consts_parametrized(ValueArray::type_def(), |v, replacer| {
            let v: &VArrayValue = v.value().downcast_ref().unwrap();
            let mut ty = v.get_element_type().clone();
            let mut contents = v.get_contents().iter().cloned().collect_vec();
            ty.transform(replacer).unwrap();
            for v in &mut contents {
                replacer.change_value(v).unwrap();
            }
            Ok(Some(ArrayValue::new(ty, contents).into()))
        });
        for op_def in ArrayOpDef::iter() {
            pass.replace_parametrized_op(
                value_array::EXTENSION.get_op(&op_def.opdef_id()).unwrap(),
                move |args| {
                    // `get` is only allowed for copyable elements. Assuming the Hugr was
                    // valid when we started, the only way for the element to become linear
                    // is if it used to contain nested `value_array`s. In that case, we
                    // have to get rid of the `get`.
                    // TODO: But what should we replace it with? Can't be a `set` since we
                    // don't have anything to put in. Maybe we need a new `get_copy` op
                    // that takes a function ptr to copy the element? For now, let's just
                    // error out and make sure we're not emitting `get`s for nested value
                    // arrays.
                    assert!(
                        op_def != ArrayOpDef::get || args[1].as_runtime().unwrap().copyable(),
                        "Cannot linearise arrays in this Hugr: \
                            Contains a `get` operation on nested value arrays"
                    );
                    Some(NodeTemplate::SingleOp(
                        op_def.instantiate(args).unwrap().into(),
                    ))
                },
            );
        }
        pass.replace_parametrized_op(
            value_array::EXTENSION.get_op(&ARRAY_REPEAT_OP_ID).unwrap(),
            |args| {
                Some(NodeTemplate::SingleOp(
                    ArrayRepeatDef::new().instantiate(args).unwrap().into(),
                ))
            },
        );
        pass.replace_parametrized_op(
            value_array::EXTENSION.get_op(&ARRAY_SCAN_OP_ID).unwrap(),
            |args| {
                Some(NodeTemplate::SingleOp(
                    ArrayScanDef::new().instantiate(args).unwrap().into(),
                ))
            },
        );
        pass.replace_parametrized_op(
            value_array::EXTENSION
                .get_op(&VArrayFromArrayDef::new().opdef_id())
                .unwrap(),
            |args| {
                let array_ty = array_type_parametric(args[0].clone(), args[1].clone()).unwrap();
                Some(NodeTemplate::SingleOp(
                    Noop::new(array_ty).to_extension_op().unwrap().into(),
                ))
            },
        );
        pass.replace_parametrized_op(
            value_array::EXTENSION
                .get_op(&VArrayToArrayDef::new().opdef_id())
                .unwrap(),
            |args| {
                let array_ty = array_type_parametric(args[0].clone(), args[1].clone()).unwrap();
                Some(NodeTemplate::SingleOp(
                    Noop::new(array_ty).to_extension_op().unwrap().into(),
                ))
            },
        );
        Self(pass)
    }
}

impl<H: HugrMut<Node = Node>> ComposablePass<H> for LinearizeArrayPass {
    type Error = ReplaceTypesError;
    type Result = bool;

    fn run(&self, hugr: &mut H) -> Result<Self::Result, Self::Error> {
        self.0.run(hugr)
    }
}

impl LinearizeArrayPass {
    /// Returns a new [`LinearizeArrayPass`] that handles all standard extensions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allows to configure how to clone and discard arrays that are nested
    /// inside opaque extension values.
    pub fn linearizer(&mut self) -> &mut DelegatingLinearizer {
        self.0.linearizer()
    }
}

#[cfg(test)]
mod test {
    use hugr_core::builder::ModuleBuilder;
    use hugr_core::extension::prelude::{ConstUsize, Noop};
    use hugr_core::ops::handle::NodeHandle;
    use hugr_core::ops::{Const, OpType};
    use hugr_core::std_extensions::collections::array::{
        self, ArrayValue, Direction, FROM, INTO, array_type,
    };
    use hugr_core::std_extensions::collections::value_array::{
        VArrayFromArray, VArrayRepeat, VArrayScan, VArrayToArray, VArrayValue,
    };
    use hugr_core::types::Transformable;
    use hugr_core::{
        HugrView,
        builder::{Container, DFGBuilder, Dataflow, HugrBuilder},
        extension::prelude::{qb_t, usize_t},
        std_extensions::collections::{
            array::{
                ArrayRepeat, ArrayScan,
                op_builder::{build_all_array_ops, build_all_value_array_ops},
            },
            value_array::{self, value_array_type},
        },
        types::{Signature, Type},
    };
    use itertools::Itertools;
    use rstest::rstest;

    use crate::{ComposablePass, composable::ValidatingPass};

    use super::LinearizeArrayPass;

    #[test]
    fn all_value_array_ops() {
        let sig = Signature::new_endo(Type::EMPTY_TYPEROW);
        let mut hugr = build_all_value_array_ops(DFGBuilder::new(sig.clone()).unwrap())
            .finish_hugr()
            .unwrap();
        ValidatingPass::new(LinearizeArrayPass::default())
            .run(&mut hugr)
            .unwrap();

        let target_hugr = build_all_array_ops(DFGBuilder::new(sig).unwrap())
            .finish_hugr()
            .unwrap();
        for (n1, n2) in hugr.nodes().zip_eq(target_hugr.nodes()) {
            assert_eq!(hugr.get_optype(n1), target_hugr.get_optype(n2));
        }
    }

    #[rstest]
    #[case(usize_t(), 2)]
    #[case(qb_t(), 2)]
    #[case(value_array_type(4, usize_t()), 2)]
    fn repeat(#[case] elem_ty: Type, #[case] size: u64) {
        let mut builder = ModuleBuilder::new();
        let repeat_decl = builder
            .declare(
                "foo",
                Signature::new(Type::EMPTY_TYPEROW, elem_ty.clone()).into(),
            )
            .unwrap();
        let mut f = builder
            .define_function(
                "bar",
                Signature::new(Type::EMPTY_TYPEROW, value_array_type(size, elem_ty.clone())),
            )
            .unwrap();
        let repeat_f = f.load_func(&repeat_decl, &[]).unwrap();
        let repeat = f
            .add_dataflow_op(VArrayRepeat::new(elem_ty.clone(), size), [repeat_f])
            .unwrap();
        let [arr] = repeat.outputs_arr();
        f.set_outputs([arr]).unwrap();
        let mut hugr = builder.finish_hugr().unwrap();

        let pass = LinearizeArrayPass::default();
        ValidatingPass::new(pass.clone()).run(&mut hugr).unwrap();
        let new_repeat: ArrayRepeat = hugr.get_optype(repeat.node()).cast().unwrap();
        let mut new_elem_ty = elem_ty.clone();
        new_elem_ty.transform(&pass.0).unwrap();
        assert_eq!(new_repeat, ArrayRepeat::new(new_elem_ty, size));
    }

    #[rstest]
    #[case(usize_t(), qb_t(), 2)]
    #[case(usize_t(), value_array_type(4, usize_t()), 2)]
    #[case(value_array_type(4, usize_t()), value_array_type(8, usize_t()), 2)]
    fn scan(#[case] src_ty: Type, #[case] tgt_ty: Type, #[case] size: u64) {
        let mut builder = ModuleBuilder::new();
        let scan_decl = builder
            .declare("foo", Signature::new(src_ty.clone(), tgt_ty.clone()).into())
            .unwrap();
        let mut f = builder
            .define_function(
                "bar",
                Signature::new(
                    value_array_type(size, src_ty.clone()),
                    value_array_type(size, tgt_ty.clone()),
                ),
            )
            .unwrap();
        let [arr] = f.input_wires_arr();
        let scan_f = f.load_func(&scan_decl, &[]).unwrap();
        let scan = f
            .add_dataflow_op(
                VArrayScan::new(src_ty.clone(), tgt_ty.clone(), vec![], size),
                [arr, scan_f],
            )
            .unwrap();
        let [arr] = scan.outputs_arr();
        f.set_outputs([arr]).unwrap();
        let mut hugr = builder.finish_hugr().unwrap();

        let pass = LinearizeArrayPass::default();
        ValidatingPass::new(pass.clone()).run(&mut hugr).unwrap();
        let new_scan: ArrayScan = hugr.get_optype(scan.node()).cast().unwrap();
        let mut new_src_ty = src_ty.clone();
        let mut new_tgt_ty = tgt_ty.clone();
        new_src_ty.transform(&pass.0).unwrap();
        new_tgt_ty.transform(&pass.0).unwrap();

        assert_eq!(
            new_scan,
            ArrayScan::new(new_src_ty, new_tgt_ty, vec![], size)
        );
    }

    #[rstest]
    #[case(INTO, usize_t(), 2)]
    #[case(FROM, usize_t(), 2)]
    #[case(INTO, array_type(4, usize_t()), 2)]
    #[case(FROM, array_type(4, usize_t()), 2)]
    #[case(INTO, value_array_type(4, usize_t()), 2)]
    #[case(FROM, value_array_type(4, usize_t()), 2)]
    fn convert(#[case] dir: Direction, #[case] elem_ty: Type, #[case] size: u64) {
        let (src, tgt) = match dir {
            INTO => (
                value_array_type(size, elem_ty.clone()),
                array_type(size, elem_ty.clone()),
            ),
            FROM => (
                array_type(size, elem_ty.clone()),
                value_array_type(size, elem_ty.clone()),
            ),
        };
        let sig = Signature::new(src, tgt);
        let mut builder = DFGBuilder::new(sig).unwrap();
        let [arr] = builder.input_wires_arr();
        let op: OpType = match dir {
            INTO => VArrayToArray::new(elem_ty.clone(), size).into(),
            FROM => VArrayFromArray::new(elem_ty.clone(), size).into(),
        };
        let convert = builder.add_dataflow_op(op, [arr]).unwrap();
        let [arr] = convert.outputs_arr();
        builder.set_outputs(vec![arr]).unwrap();
        let mut hugr = builder.finish_hugr().unwrap();

        let pass = LinearizeArrayPass::default();
        ValidatingPass::new(pass.clone()).run(&mut hugr).unwrap();
        let new_convert: Noop = hugr.get_optype(convert.node()).cast().unwrap();
        let mut new_elem_ty = elem_ty.clone();
        new_elem_ty.transform(&pass.0).unwrap();

        assert_eq!(new_convert, Noop::new(array_type(size, new_elem_ty)));
    }

    #[rstest]
    #[case(value_array_type(2, usize_t()))]
    #[case(value_array_type(2, value_array_type(4, usize_t())))]
    #[case(value_array_type(2, Type::new_tuple(vec![usize_t(), value_array_type(4, usize_t())])))]
    fn implicit_clone(#[case] array_ty: Type) {
        let sig = Signature::new(array_ty.clone(), vec![array_ty; 2]);
        let mut builder = DFGBuilder::new(sig).unwrap();
        let [arr] = builder.input_wires_arr();
        builder.set_outputs(vec![arr, arr]).unwrap();

        let mut hugr = builder.finish_hugr().unwrap();
        ValidatingPass::new(LinearizeArrayPass::default())
            .run(&mut hugr)
            .unwrap();
    }

    #[rstest]
    #[case(value_array_type(2, usize_t()))]
    #[case(value_array_type(2, value_array_type(4, usize_t())))]
    #[case(value_array_type(2, Type::new_tuple(vec![usize_t(), value_array_type(4, usize_t())])))]
    fn implicit_discard(#[case] array_ty: Type) {
        let sig = Signature::new(array_ty, Type::EMPTY_TYPEROW);
        let mut builder = DFGBuilder::new(sig).unwrap();
        builder.set_outputs(vec![]).unwrap();

        let mut hugr = builder.finish_hugr().unwrap();
        ValidatingPass::new(LinearizeArrayPass::default())
            .run(&mut hugr)
            .unwrap();
    }

    #[test]
    fn array_value() {
        let mut builder = ModuleBuilder::new();
        let array_v = VArrayValue::new(usize_t(), vec![ConstUsize::new(1).into()]);
        let c = builder.add_constant(Const::new(array_v.clone().into()));

        let mut hugr = builder.finish_hugr().unwrap();
        ValidatingPass::new(LinearizeArrayPass::default())
            .run(&mut hugr)
            .unwrap();

        let new_array_v: &ArrayValue = hugr
            .get_optype(c.node())
            .as_const()
            .unwrap()
            .get_custom_value()
            .unwrap();

        assert_eq!(new_array_v.get_element_type(), array_v.get_element_type());
        assert_eq!(new_array_v.get_contents(), array_v.get_contents());
    }

    #[test]
    fn array_value_nested() {
        let mut builder = ModuleBuilder::new();
        let array_v_inner = VArrayValue::new(usize_t(), vec![ConstUsize::new(1).into()]);
        let array_v: array::GenericArrayValue<value_array::ValueArray> = VArrayValue::new(
            value_array_type(1, usize_t()),
            vec![array_v_inner.clone().into()],
        );
        let c = builder.add_constant(Const::new(array_v.clone().into()));

        let mut hugr = builder.finish_hugr().unwrap();
        ValidatingPass::new(LinearizeArrayPass::default())
            .run(&mut hugr)
            .unwrap();

        let new_array_v: &ArrayValue = hugr
            .get_optype(c.node())
            .as_const()
            .unwrap()
            .get_custom_value()
            .unwrap();

        assert_eq!(new_array_v.get_element_type(), &array_type(1, usize_t()));
        assert_eq!(
            new_array_v.get_contents()[0],
            ArrayValue::new(usize_t(), vec![ConstUsize::new(1).into()]).into()
        );
    }
}
