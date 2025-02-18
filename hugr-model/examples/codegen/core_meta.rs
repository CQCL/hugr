use hugr_model::v0 as model;
use super::{view_term_apply, view_node_custom, View};
///`core.meta.description`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#description {
    #[allow(missing_docs)]
    pub r#description: model::TermId,
}
impl<'a> View<'a> for r#description {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#description] = view_term_apply(module, id, "core.meta.description")?;
        Some(Self { r#description })
    }
}

