use super::{Apply, List, Term};
use hugr_model::v0::SymbolName;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreFn {
    pub inputs: List,
    pub outputs: List,
}

impl CoreFn {
    pub const SYMBOL: SymbolName = SymbolName::new_static("core.fn");
}

impl TryFrom<Term> for CoreFn {
    type Error = ();

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        let Term::Apply(apply) = value else {
            return Err(());
        };

        if apply.name() != &Self::SYMBOL {
            return Err(());
        };

        if apply.args().len() < 2 {
            return Err(());
        };

        let inputs = apply.args()[0].clone().try_into()?;
        let outputs = apply.args()[1].clone().try_into()?;

        Ok(CoreFn { inputs, outputs })
    }
}

impl From<CoreFn> for Term {
    fn from(value: CoreFn) -> Self {
        Apply::new(CoreFn::SYMBOL, [value.inputs.into(), value.outputs.into()]).into()
    }
}
