use hugr_model::v0 as model;
use super::{view_term_apply, view_node_custom};
///`collections.array.array`: Fixed-length array.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#array {
    #[allow(missing_docs)]
    pub r#len: model::TermId,
    #[allow(missing_docs)]
    pub r#type: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#array {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#len, r#type] = view_term_apply(module, id, "collections.array.array")?;
        Some(Self { r#len, r#type })
    }
}
///`collections.array.const`: Constant array value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#const {
    #[allow(missing_docs)]
    pub r#len: model::TermId,
    #[allow(missing_docs)]
    pub r#type: model::TermId,
    #[allow(missing_docs)]
    pub r#ext: model::TermId,
    #[allow(missing_docs)]
    pub r#values: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#const {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#len, r#type, r#ext, r#values] = view_term_apply(
            module,
            id,
            "collections.array.const",
        )?;
        Some(Self {
            r#len,
            r#type,
            r#ext,
            r#values,
        })
    }
}
///`collections.array.new_array`: Create a new array from elements.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#new_array {
    #[allow(missing_docs)]
    pub r#len: model::TermId,
    #[allow(missing_docs)]
    pub r#type: model::TermId,
    #[allow(missing_docs)]
    pub r#inputs: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#new_array {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#len, r#type, r#inputs] = view_node_custom(
            module,
            id,
            "collections.array.new_array",
        )?;
        Some(Self { r#len, r#type, r#inputs })
    }
}
///`collections.array.get`: Get an element from an array.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#get {
    #[allow(missing_docs)]
    pub r#len: model::TermId,
    #[allow(missing_docs)]
    pub r#type: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#get {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#len, r#type] = view_node_custom(module, id, "collections.array.get")?;
        Some(Self { r#len, r#type })
    }
}
///`collections.array.set`: Set an element in an array.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#set {
    #[allow(missing_docs)]
    pub r#len: model::TermId,
    #[allow(missing_docs)]
    pub r#type: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#set {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#len, r#type] = view_node_custom(module, id, "collections.array.set")?;
        Some(Self { r#len, r#type })
    }
}
///`collections.array.swap`: Swap two elements in an array.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#swap {
    #[allow(missing_docs)]
    pub r#len: model::TermId,
    #[allow(missing_docs)]
    pub r#type: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#swap {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#len, r#type] = view_node_custom(module, id, "collections.array.swap")?;
        Some(Self { r#len, r#type })
    }
}
///`collections.array.discard_empty`: Discard an empty array.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#discard_empty {
    #[allow(missing_docs)]
    pub r#type: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#discard_empty {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#type] = view_node_custom(module, id, "collections.array.discard_empty")?;
        Some(Self { r#type })
    }
}
///`collections.array.pop_left`: Pop an element from the left of an array.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#pop_left {
    #[allow(missing_docs)]
    pub r#len: model::TermId,
    #[allow(missing_docs)]
    pub r#type: model::TermId,
    #[allow(missing_docs)]
    pub r#reduced_len: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#pop_left {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#len, r#type, r#reduced_len] = view_node_custom(
            module,
            id,
            "collections.array.pop_left",
        )?;
        Some(Self {
            r#len,
            r#type,
            r#reduced_len,
        })
    }
}
///`collections.array.pop_right`: Pop an element from the right of an array.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#pop_right {
    #[allow(missing_docs)]
    pub r#len: model::TermId,
    #[allow(missing_docs)]
    pub r#type: model::TermId,
    #[allow(missing_docs)]
    pub r#reduced_len: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#pop_right {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#len, r#type, r#reduced_len] = view_node_custom(
            module,
            id,
            "collections.array.pop_right",
        )?;
        Some(Self {
            r#len,
            r#type,
            r#reduced_len,
        })
    }
}
///`collections.array.repeat`: Creates a new array whose elements are initialised by calling the given function n times.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#repeat {
    #[allow(missing_docs)]
    pub r#len: model::TermId,
    #[allow(missing_docs)]
    pub r#type: model::TermId,
    #[allow(missing_docs)]
    pub r#ext: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#repeat {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#len, r#type, r#ext] = view_node_custom(
            module,
            id,
            "collections.array.repeat",
        )?;
        Some(Self { r#len, r#type, r#ext })
    }
}
///`collections.array.scan`: A combination of map and foldl. Applies a function to each element of the array with an accumulator that is passed through from start to finish. Returns the resulting array and the final state of the accumulator.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#scan {
    #[allow(missing_docs)]
    pub r#len: model::TermId,
    #[allow(missing_docs)]
    pub r#t1: model::TermId,
    #[allow(missing_docs)]
    pub r#t2: model::TermId,
    #[allow(missing_docs)]
    pub r#s: model::TermId,
    #[allow(missing_docs)]
    pub r#ext: model::TermId,
}
impl<'a> ::hugr_model::v0::View<'a> for r#scan {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#len, r#t1, r#t2, r#s, r#ext] = view_node_custom(
            module,
            id,
            "collections.array.scan",
        )?;
        Some(Self {
            r#len,
            r#t1,
            r#t2,
            r#s,
            r#ext,
        })
    }
}

