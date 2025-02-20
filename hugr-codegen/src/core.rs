pub mod r#core_const {
    /**`core.const.adt`.

__Type__: `(core.const (core.adt ?variants) ?ext)`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#adt {
        ///`variants : (core.list (core.list core.type))`
        #[allow(missing_docs)]
        pub r#variants: hugr_model::v0::TermId,
        ///`ext : core.ext_set`
        #[allow(missing_docs)]
        pub r#ext: hugr_model::v0::TermId,
        ///`types : (core.list core.static)`
        #[allow(missing_docs)]
        pub r#types: hugr_model::v0::TermId,
        ///`tag : core.nat`
        #[allow(missing_docs)]
        pub r#tag: hugr_model::v0::TermId,
        ///`values : (core.tuple ?types)`
        #[allow(missing_docs)]
        pub r#values: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#adt {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.const.adt" {
                return None;
            }
            let [r#variants, r#ext, r#types, r#tag, r#values] = apply
                .args
                .try_into()
                .ok()?;
            Some(Self {
                r#variants,
                r#ext,
                r#types,
                r#tag,
                r#values,
            })
        }
    }
    /**`core.const.adt_types`.

__Type__: `core.constraint`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#adt_types {
        ///`variants : (core.list (core.list core.type))`
        #[allow(missing_docs)]
        pub r#variants: hugr_model::v0::TermId,
        ///`ext : core.ext_set`
        #[allow(missing_docs)]
        pub r#ext: hugr_model::v0::TermId,
        ///`types : (core.list core.static)`
        #[allow(missing_docs)]
        pub r#types: hugr_model::v0::TermId,
        ///`tag : core.nat`
        #[allow(missing_docs)]
        pub r#tag: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#adt_types {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.const.adt_types" {
                return None;
            }
            let [r#variants, r#ext, r#types, r#tag] = apply.args.try_into().ok()?;
            Some(Self {
                r#variants,
                r#ext,
                r#types,
                r#tag,
            })
        }
    }
}
pub mod r#core {
    /**Runtime function type.

__Type__: `core.type`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#fn {
        ///`inputs : (core.list core.type)`
        #[allow(missing_docs)]
        pub r#inputs: hugr_model::v0::TermId,
        ///`outputs : (core.list core.type)`
        #[allow(missing_docs)]
        pub r#outputs: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#fn {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.fn" {
                return None;
            }
            let [r#inputs, r#outputs] = apply.args.try_into().ok()?;
            Some(Self { r#inputs, r#outputs })
        }
    }
    /**Type of types.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#type {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#type {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.type" {
                return None;
            }
            let [] = apply.args.try_into().ok()?;
            Some(Self {})
        }
    }
    /**Type of static values.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#static {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#static {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.static" {
                return None;
            }
            let [] = apply.args.try_into().ok()?;
            Some(Self {})
        }
    }
    /**Type of constraints.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#constraint {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#constraint {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.constraint" {
                return None;
            }
            let [] = apply.args.try_into().ok()?;
            Some(Self {})
        }
    }
    /**Nonlinear constraint.

__Type__: `core.constraint`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#nonlinear {
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#nonlinear {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.nonlinear" {
                return None;
            }
            let [r#type] = apply.args.try_into().ok()?;
            Some(Self { r#type })
        }
    }
    /**Type of metadata.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#meta {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#meta {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.meta" {
                return None;
            }
            let [] = apply.args.try_into().ok()?;
            Some(Self {})
        }
    }
    /**`core.adt`.

__Type__: `core.type`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#adt {
        ///`variants : (core.list (core.list core.type))`
        #[allow(missing_docs)]
        pub r#variants: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#adt {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.adt" {
                return None;
            }
            let [r#variants] = apply.args.try_into().ok()?;
            Some(Self { r#variants })
        }
    }
    /**`core.str`.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#str {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#str {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.str" {
                return None;
            }
            let [] = apply.args.try_into().ok()?;
            Some(Self {})
        }
    }
    /**`core.nat`.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#nat {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#nat {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.nat" {
                return None;
            }
            let [] = apply.args.try_into().ok()?;
            Some(Self {})
        }
    }
    /**`core.bytes`.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#bytes {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#bytes {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.bytes" {
                return None;
            }
            let [] = apply.args.try_into().ok()?;
            Some(Self {})
        }
    }
    /**`core.float`.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#float {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#float {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.float" {
                return None;
            }
            let [] = apply.args.try_into().ok()?;
            Some(Self {})
        }
    }
    /**`core.ctrl`.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ctrl {
        ///`types : (core.list core.type)`
        #[allow(missing_docs)]
        pub r#types: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ctrl {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.ctrl" {
                return None;
            }
            let [r#types] = apply.args.try_into().ok()?;
            Some(Self { r#types })
        }
    }
    /**`core.ctrl_type`.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ctrl_type {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ctrl_type {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.ctrl_type" {
                return None;
            }
            let [] = apply.args.try_into().ok()?;
            Some(Self {})
        }
    }
    /**`core.ext_set`.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ext_set {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ext_set {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.ext_set" {
                return None;
            }
            let [] = apply.args.try_into().ok()?;
            Some(Self {})
        }
    }
    /**`core.const`.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#const {
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
        ///`ext : core.ext_set`
        #[allow(missing_docs)]
        pub r#ext: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#const {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.const" {
                return None;
            }
            let [r#type, r#ext] = apply.args.try_into().ok()?;
            Some(Self { r#type, r#ext })
        }
    }
    /**`core.list`.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#list {
        ///`type : core.static`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#list {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.list" {
                return None;
            }
            let [r#type] = apply.args.try_into().ok()?;
            Some(Self { r#type })
        }
    }
    /**`core.tuple`.

__Type__: `core.static`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#tuple {
        ///`type : (core.list core.static)`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#tuple {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.tuple" {
                return None;
            }
            let [r#type] = apply.args.try_into().ok()?;
            Some(Self { r#type })
        }
    }
    /**Call a statically known function.

__Type__: `(core.fn ?inputs ?outputs ?ext)`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#call {
        ///`inputs : (core.list core.type)`
        #[allow(missing_docs)]
        pub r#inputs: hugr_model::v0::TermId,
        ///`outputs : (core.list core.type)`
        #[allow(missing_docs)]
        pub r#outputs: hugr_model::v0::TermId,
        ///`ext : core.ext_set`
        #[allow(missing_docs)]
        pub r#ext: hugr_model::v0::TermId,
        ///`fn : (core.const (core.fn ?inputs ?outputs ?ext) ?ext)`
        #[allow(missing_docs)]
        pub r#fn: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#call {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "core.call" {
                return None;
            }
            let [r#inputs, r#outputs, r#ext, r#fn] = operation.params.try_into().ok()?;
            Some(Self {
                r#inputs,
                r#outputs,
                r#ext,
                r#fn,
            })
        }
    }
    /**Call a function provided at runtime.

__Type__: `(core.fn [(core.fn ?inputs ?outputs ?ext) ?inputs ...] ?outputs ?ext)`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#call_indirect {
        ///`inputs : (core.list core.type)`
        #[allow(missing_docs)]
        pub r#inputs: hugr_model::v0::TermId,
        ///`outputs : (core.list core.type)`
        #[allow(missing_docs)]
        pub r#outputs: hugr_model::v0::TermId,
        ///`ext : core.ext_set`
        #[allow(missing_docs)]
        pub r#ext: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#call_indirect {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "core.call_indirect" {
                return None;
            }
            let [r#inputs, r#outputs, r#ext] = operation.params.try_into().ok()?;
            Some(Self { r#inputs, r#outputs, r#ext })
        }
    }
    /**Load a constant value.

__Type__: `(core.fn [] [?type] ?ext)`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#load_const {
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
        ///`ext : core.ext_set`
        #[allow(missing_docs)]
        pub r#ext: hugr_model::v0::TermId,
        ///`value : (core.const ?type ?ext)`
        #[allow(missing_docs)]
        pub r#value: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#load_const {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "core.load_const" {
                return None;
            }
            let [r#type, r#ext, r#value] = operation.params.try_into().ok()?;
            Some(Self { r#type, r#ext, r#value })
        }
    }
    /**`core.make_adt`.

__Type__: `(core.fn ?types [(core.adt ?variants)] (ext))`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#make_adt {
        ///`variants : (core.list (core.list core.type))`
        #[allow(missing_docs)]
        pub r#variants: hugr_model::v0::TermId,
        ///`types : (core.list core.type)`
        #[allow(missing_docs)]
        pub r#types: hugr_model::v0::TermId,
        ///`tag : core.nat`
        #[allow(missing_docs)]
        pub r#tag: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#make_adt {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "core.make_adt" {
                return None;
            }
            let [r#variants, r#types, r#tag] = operation.params.try_into().ok()?;
            Some(Self { r#variants, r#types, r#tag })
        }
    }
}
pub mod r#core_meta {
    /**`core.meta.description`.

__Type__: `core.meta`*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#description {
        ///`description : core.str`
        #[allow(missing_docs)]
        pub r#description: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#description {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "core.meta.description" {
                return None;
            }
            let [r#description] = apply.args.try_into().ok()?;
            Some(Self { r#description })
        }
    }
}

