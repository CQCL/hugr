use hugr_model::v0 as model;
use super::{view_term_apply, View};
///`core.fn`: Runtime function type.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#fn {
    #[allow(missing_docs)]
    pub r#inputs: model::TermId,
    #[allow(missing_docs)]
    pub r#outputs: model::TermId,
}
impl<'a> View<'a> for r#fn {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#inputs, r#outputs] = view_term_apply(module, id, "core.fn")?;
        Some(Self { r#inputs, r#outputs })
    }
}
///`core.type`: Type of types.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#type {}
impl<'a> View<'a> for r#type {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [] = view_term_apply(module, id, "core.type")?;
        Some(Self {})
    }
}
///`core.static`: Type of static values.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#static {}
impl<'a> View<'a> for r#static {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [] = view_term_apply(module, id, "core.static")?;
        Some(Self {})
    }
}
///`core.constraint`: Type of constraints.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#constraint {}
impl<'a> View<'a> for r#constraint {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [] = view_term_apply(module, id, "core.constraint")?;
        Some(Self {})
    }
}
///`core.nonlinear`: Nonlinear constraint.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#nonlinear {
    #[allow(missing_docs)]
    pub r#type: model::TermId,
}
impl<'a> View<'a> for r#nonlinear {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#type] = view_term_apply(module, id, "core.nonlinear")?;
        Some(Self { r#type })
    }
}
///`core.meta`: Type of metadata.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#meta {}
impl<'a> View<'a> for r#meta {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [] = view_term_apply(module, id, "core.meta")?;
        Some(Self {})
    }
}
///`core.adt`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#adt {
    #[allow(missing_docs)]
    pub r#variants: model::TermId,
}
impl<'a> View<'a> for r#adt {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#variants] = view_term_apply(module, id, "core.adt")?;
        Some(Self { r#variants })
    }
}
///`core.str`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#str {}
impl<'a> View<'a> for r#str {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [] = view_term_apply(module, id, "core.str")?;
        Some(Self {})
    }
}
///`core.nat`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#nat {}
impl<'a> View<'a> for r#nat {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [] = view_term_apply(module, id, "core.nat")?;
        Some(Self {})
    }
}
///`core.bytes`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#bytes {}
impl<'a> View<'a> for r#bytes {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [] = view_term_apply(module, id, "core.bytes")?;
        Some(Self {})
    }
}
///`core.float`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#float {}
impl<'a> View<'a> for r#float {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [] = view_term_apply(module, id, "core.float")?;
        Some(Self {})
    }
}
///`core.ctrl`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#ctrl {
    #[allow(missing_docs)]
    pub r#types: model::TermId,
}
impl<'a> View<'a> for r#ctrl {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#types] = view_term_apply(module, id, "core.ctrl")?;
        Some(Self { r#types })
    }
}
///`core.ctrl_type`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#ctrl_type {}
impl<'a> View<'a> for r#ctrl_type {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [] = view_term_apply(module, id, "core.ctrl_type")?;
        Some(Self {})
    }
}
///`core.ext_set`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#ext_set {}
impl<'a> View<'a> for r#ext_set {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [] = view_term_apply(module, id, "core.ext_set")?;
        Some(Self {})
    }
}
///`core.const`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#const {
    #[allow(missing_docs)]
    pub r#type: model::TermId,
    #[allow(missing_docs)]
    pub r#ext: model::TermId,
}
impl<'a> View<'a> for r#const {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#type, r#ext] = view_term_apply(module, id, "core.const")?;
        Some(Self { r#type, r#ext })
    }
}
///`core.list`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#list {
    #[allow(missing_docs)]
    pub r#type: model::TermId,
}
impl<'a> View<'a> for r#list {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#type] = view_term_apply(module, id, "core.list")?;
        Some(Self { r#type })
    }
}
///`core.tuple`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#tuple {
    #[allow(missing_docs)]
    pub r#type: model::TermId,
}
impl<'a> View<'a> for r#tuple {
    type Id = model::TermId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#type] = view_term_apply(module, id, "core.tuple")?;
        Some(Self { r#type })
    }
}
///`core.call`: Call a statically known function.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#call {
    #[allow(missing_docs)]
    pub r#inputs: model::TermId,
    #[allow(missing_docs)]
    pub r#outputs: model::TermId,
    #[allow(missing_docs)]
    pub r#ext: model::TermId,
    #[allow(missing_docs)]
    pub r#fn: model::TermId,
}
impl<'a> View<'a> for r#call {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#inputs, r#outputs, r#ext, r#fn] = view_node_custom(
            module,
            id,
            "core.call",
        )?;
        Some(Self {
            r#inputs,
            r#outputs,
            r#ext,
            r#fn,
        })
    }
}
///`core.call_indirect`: Call a function provided at runtime.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#call_indirect {
    #[allow(missing_docs)]
    pub r#inputs: model::TermId,
    #[allow(missing_docs)]
    pub r#outputs: model::TermId,
    #[allow(missing_docs)]
    pub r#ext: model::TermId,
}
impl<'a> View<'a> for r#call_indirect {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#inputs, r#outputs, r#ext] = view_node_custom(
            module,
            id,
            "core.call_indirect",
        )?;
        Some(Self { r#inputs, r#outputs, r#ext })
    }
}
///`core.load_const`: Load a constant value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#load_const {
    #[allow(missing_docs)]
    pub r#type: model::TermId,
    #[allow(missing_docs)]
    pub r#ext: model::TermId,
    #[allow(missing_docs)]
    pub r#value: model::TermId,
}
impl<'a> View<'a> for r#load_const {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#type, r#ext, r#value] = view_node_custom(module, id, "core.load_const")?;
        Some(Self { r#type, r#ext, r#value })
    }
}
///`core.make_adt`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
pub struct r#make_adt {
    #[allow(missing_docs)]
    pub r#variants: model::TermId,
    #[allow(missing_docs)]
    pub r#types: model::TermId,
    #[allow(missing_docs)]
    pub r#tag: model::TermId,
}
impl<'a> View<'a> for r#make_adt {
    type Id = model::NodeId;
    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let [r#variants, r#types, r#tag] = view_node_custom(
            module,
            id,
            "core.make_adt",
        )?;
        Some(Self { r#variants, r#types, r#tag })
    }
}

