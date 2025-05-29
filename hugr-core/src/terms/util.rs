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
        let core::Fn { outputs, .. } = term.expect()?;
        let ([adt], rest) = outputs.view_list_prefix()?;
        let core::Adt { variants } = adt.expect()?;
        let [just_inputs, just_outputs] = variants.view_list_exact()?;
        Ok(Self {
            just_inputs,
            just_outputs,
            rest,
        })
    }

    fn expect(term: &Term) -> Result<Self, ViewError> {
        todo!()
    }
}

impl From<LoopRegionSignature> for Term {
    fn from(value: LoopRegionSignature) -> Self {
        let variants = Term::from(ExactList {
            items: [value.just_inputs.clone(), value.just_outputs.clone()],
            item_type: Term::default(),
        });
        let adt = Term::from(core::Adt { variants });
        let inputs = Term::new(TermKind::ListConcat(&value.just_inputs, &value.rest));
        let outputs = Term::from(ListPrefix {
            prefix: [adt],
            suffix: value.rest.clone(),
        });
        core::Fn { inputs, outputs }.into()
    }
}

#[derive(Debug, Clone)]
struct ListPrefix<const N: usize> {
    pub prefix: [Term; N],
    pub suffix: Term,
}

impl<const N: usize> From<ListPrefix<N>> for Term {
    fn from(value: ListPrefix<N>) -> Self {
        let mut list = value.suffix.clone();

        for item in value.prefix.iter().rev() {
            list = Term::new(TermKind::ListCons(item, &list));
        }

        list
    }
}

#[derive(Debug, Clone)]
struct ExactList<const N: usize> {
    pub items: [Term; N],
    pub item_type: Term,
}

impl<const N: usize> From<ExactList<N>> for Term {
    fn from(value: ExactList<N>) -> Self {
        let mut list = Term::new(TermKind::ListEmpty(&value.item_type));

        for item in value.items.iter().rev() {
            list = Term::new(TermKind::ListCons(item, &list));
        }

        list
    }
}
