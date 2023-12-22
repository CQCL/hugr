use crate::{
    extension::{ConstFoldResult, OpDef},
    ops,
    std_extensions::arithmetic::int_types::{ConstIntU, INT_TYPES},
    IncomingPort,
};

use super::IntOpDef;

pub(super) fn set_fold(op: &IntOpDef, def: &mut OpDef) {
    match op {
        IntOpDef::iadd => def.set_constant_folder(iadd_fold),
        _ => (),
    }
}

// TODO get width from const
fn iadd_fold(consts: &[(IncomingPort, ops::Const)]) -> ConstFoldResult {
    let width = 5;
    match consts {
        [(_, c1), (_, c2)] => {
            let [c1, c2]: [&ConstIntU; 2] = [c1, c2].map(|c| c.get_custom_value().unwrap());

            Some(vec![(
                0.into(),
                ops::Const::new(
                    ConstIntU::new(width, c1.value() + c2.value())
                        .unwrap()
                        .into(),
                    INT_TYPES[5].to_owned(),
                )
                .unwrap(),
            )])
        }

        _ => None,
    }
}
