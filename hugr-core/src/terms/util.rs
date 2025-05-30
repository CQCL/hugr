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
        let outputs = ListPrefix::view(&outputs)?;
        let [adt] = outputs.prefix;
        let core::Adt { variants } = adt.view()?;
        let variants = ListExact::view(&variants)?;
        let [just_inputs, just_outputs] = variants.items;
        Ok(Self {
            just_inputs,
            just_outputs,
            rest: outputs.rest,
        })
    }
}

impl From<LoopRegionSignature> for Term {
    fn from(value: LoopRegionSignature) -> Self {
        let variants = Term::from(ListExact {
            item_type: Term::default(),
            items: [value.just_inputs.clone(), value.just_outputs.clone()],
        });
        let adt = Term::from(core::Adt { variants });
        let inputs = Term::new(TermKind::ListConcat(
            &Term::default(),
            &[value.just_inputs.clone(), value.rest.clone()],
        ));
        let outputs = Term::from(ListPrefix {
            item_type: Term::default(),
            prefix: [adt],
            rest: value.rest.clone(),
        });
        core::Fn { inputs, outputs }.into()
    }
}

/// Utility view for lists with a known-length prefix.
#[derive(Debug, Clone)]
pub struct ListPrefix<const N: usize> {
    /// The type of the items in the list.
    pub item_type: Term,
    /// The prefix of the list.
    pub prefix: [Term; N],
    /// The remainder of the list.
    pub rest: Term,
}

impl<const N: usize> From<ListPrefix<N>> for Term {
    fn from(value: ListPrefix<N>) -> Self {
        let prefix = Term::from(ListExact {
            item_type: value.item_type.clone(),
            items: value.prefix,
        });
        Term::new(TermKind::ListConcat(
            &value.item_type,
            &[prefix, value.rest],
        ))
    }
}

impl<const N: usize> View for ListPrefix<N> {
    fn view(term: &Term) -> Result<Self, ViewError> {
        todo!()
    }
}

/// Utility view for closed lists with a known length.
#[derive(Debug, Clone)]
pub struct ListExact<const N: usize> {
    /// The type of the items in the list.
    pub item_type: Term,
    /// The items in the list.
    pub items: [Term; N],
}

impl<const N: usize> From<ListExact<N>> for Term {
    fn from(value: ListExact<N>) -> Self {
        Term::new(TermKind::List(&value.item_type, &value.items))
    }
}

impl<const N: usize> View for ListExact<N> {
    fn view(term: &Term) -> Result<Self, ViewError> {
        todo!()
    }
}
