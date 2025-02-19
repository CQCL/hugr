use hugr_model::v0::{
    view::{NamedConstructor, View},
    Module, NodeId, TermId,
};

struct MetaDoc<'a>(pub &'a str);

impl<'a> View<'a> for MetaDoc<'a> {
    type Id = TermId;

    fn view(module: &'a Module<'a>, id: Self::Id) -> Option<Self> {
        let apply: NamedConstructor = module.view(id)?;

        if apply.name != "core.meta.description" {
            return None;
        }

        let [doc] = apply.args.try_into().ok()?;
        Some(MetaDoc(module.view(doc)?))
    }
}

pub fn find_node_docs<'a>(module: &'a Module<'a>, node_id: NodeId) -> Option<&'a str> {
    module
        .get_node(node_id)?
        .meta
        .iter()
        .find_map(|meta| module.view::<MetaDoc>(*meta))
        .map(|MetaDoc(doc)| doc)
}
