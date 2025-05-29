use super::{Term, TermKind, View, ViewError, core};

/// The signature of a tail loop region.
#[derive(Debug, Clone)]
pub struct LoopRegionSignature {
    /// Prefix of input types.
    pub just_inputs: Term,
    /// Prefix of output types.
    pub just_outputs: Term,
    /// Suffix of types shared between inputs and outputs.
    pub rest: Term,
}

impl View for LoopRegionSignature {
    fn view(term: &Term) -> Result<Self, ViewError> {
        let core::Fn { outputs, .. } = term.view()?;
        let ListPrefix([adt], rest) = outputs.view()?;
        let core::Adt { variants } = adt.view()?;
        let ExactList([just_inputs, just_outputs], _) = variants.view()?;
        Ok(Self {
            just_inputs,
            just_outputs,
            rest,
        })
    }
}

impl From<LoopRegionSignature> for Term {
    fn from(value: LoopRegionSignature) -> Self {
        let variants = Term::from(ExactList(
            [value.just_inputs.clone(), value.just_outputs.clone()],
            Term::default(),
        ));
        let adt = Term::from(core::Adt { variants });
        let inputs = Term::new(TermKind::ListConcat(&value.just_inputs, &value.rest));
        let outputs = Term::from(ListPrefix([adt], value.rest.clone()));
        core::Fn { inputs, outputs }.into()
    }
}

#[derive(Debug, Clone)]
struct ListPrefix<const N: usize>(pub [Term; N], pub Term);

impl<const N: usize> From<ListPrefix<N>> for Term {
    fn from(value: ListPrefix<N>) -> Self {
        let mut list = value.1.clone();

        for item in value.0.iter().rev() {
            list = Term::new(TermKind::ListCons(item, &list));
        }

        list
    }
}

impl<const N: usize> View for ListPrefix<N> {
    fn view(term: &Term) -> Result<Self, ViewError> {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct ExactList<const N: usize>(pub [Term; N], pub Term);

impl<const N: usize> From<ExactList<N>> for Term {
    fn from(value: ExactList<N>) -> Self {
        let mut list = Term::new(TermKind::ListEmpty(&value.1));

        for item in value.0.iter().rev() {
            list = Term::new(TermKind::ListCons(item, &list));
        }

        list
    }
}

impl<const N: usize> View for ExactList<N> {
    fn view(term: &Term) -> Result<Self, ViewError> {
        todo!()
    }
}
