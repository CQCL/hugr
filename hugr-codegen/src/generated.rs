pub mod r#arithmetic_int {
    /**Constant integer value.

## Type
<pre>
(core.const (arithmetic.int.types.int ?bitwidth) (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#const {
        ///`bitwidth : core.nat`
        #[allow(missing_docs)]
        pub r#bitwidth: hugr_model::v0::TermId,
        ///`value : core.nat`
        #[allow(missing_docs)]
        pub r#value: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#const {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "arithmetic.int.const" {
                return None;
            }
            let [r#bitwidth, r#value] = apply.args.try_into().ok()?;
            Some(Self { r#bitwidth, r#value })
        }
    }
    /**Widen an unsigned integer preserving value

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?from_width)]
  [(arithmetic.int.int ?to_width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#iwiden_u {
        ///`from_width : core.nat`
        #[allow(missing_docs)]
        pub r#from_width: hugr_model::v0::TermId,
        ///`to_width : core.nat`
        #[allow(missing_docs)]
        pub r#to_width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#iwiden_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.iwiden_u" {
                return None;
            }
            let [r#from_width, r#to_width] = operation.params.try_into().ok()?;
            Some(Self { r#from_width, r#to_width })
        }
    }
    /**Widen a signed integer preserving value

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?from_width)]
  [(arithmetic.int.int ?to_width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#iwiden_s {
        ///`from_width : core.nat`
        #[allow(missing_docs)]
        pub r#from_width: hugr_model::v0::TermId,
        ///`to_width : core.nat`
        #[allow(missing_docs)]
        pub r#to_width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#iwiden_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.iwiden_s" {
                return None;
            }
            let [r#from_width, r#to_width] = operation.params.try_into().ok()?;
            Some(Self { r#from_width, r#to_width })
        }
    }
    /**Narrow an unsigned integer, returning error if value doesn't fit

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?from_width)]
  [(core.adt [[] [(arithmetic.int.int ?to_width)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#inarrow_u {
        ///`from_width : core.nat`
        #[allow(missing_docs)]
        pub r#from_width: hugr_model::v0::TermId,
        ///`to_width : core.nat`
        #[allow(missing_docs)]
        pub r#to_width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#inarrow_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.inarrow_u" {
                return None;
            }
            let [r#from_width, r#to_width] = operation.params.try_into().ok()?;
            Some(Self { r#from_width, r#to_width })
        }
    }
    /**Narrow a signed integer, returning error if value doesn't fit

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?from_width)]
  [(core.adt [[] [(arithmetic.int.int ?to_width)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#inarrow_s {
        ///`from_width : core.nat`
        #[allow(missing_docs)]
        pub r#from_width: hugr_model::v0::TermId,
        ///`to_width : core.nat`
        #[allow(missing_docs)]
        pub r#to_width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#inarrow_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.inarrow_s" {
                return None;
            }
            let [r#from_width, r#to_width] = operation.params.try_into().ok()?;
            Some(Self { r#from_width, r#to_width })
        }
    }
    /**Integer equality test

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [prelude.bool]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ieq {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ieq {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ieq" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Unsigned integer less than

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [prelude.bool]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ilt_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ilt_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ilt_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Signed integer less than

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [prelude.bool]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ilt_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ilt_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ilt_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Unsigned integer greater than

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [prelude.bool]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#igt_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#igt_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.igt_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Signed integer greater than

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [prelude.bool]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#igt_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#igt_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.igt_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Unsigned integer less than or equal

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [prelude.bool]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ile_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ile_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ile_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Signed integer less than or equal

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [prelude.bool]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ile_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ile_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ile_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Unsigned integer greater than or equal

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [prelude.bool]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ige_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ige_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ige_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Signed integer greater than or equal

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [prelude.bool]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ige_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ige_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ige_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Maximum of unsigned integers

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#imax_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#imax_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.imax_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Maximum of signed integers

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#imax_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#imax_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.imax_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Minimum of unsigned integers

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#imin_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#imin_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.imin_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Minimum of signed integers

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#imin_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#imin_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.imin_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Integer addition modulo 2^N

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#iadd {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#iadd {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.iadd" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Integer subtraction modulo 2^N

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#isub {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#isub {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.isub" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Integer multiplication modulo 2^N

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#imul {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#imul {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.imul" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Raise first input to the power of second input, the exponent is treated as an unsigned integer

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ipow {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ipow {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ipow" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Convert signed to unsigned by taking absolute value

## Type
<pre>
(core.fn [(arithmetic.int.int ?width)] [(arithmetic.int.int ?width)] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#iabs {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#iabs {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.iabs" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Checked unsigned integer division and modulus with divide-by-zero checking

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(core.adt [[] [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#idivmod_checked_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#idivmod_checked_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.idivmod_checked_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Unchecked unsigned integer division and modulus

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#idivmod_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#idivmod_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.idivmod_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Checked signed integer division and modulus with divide-by-zero checking

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(core.adt [[] [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#idivmod_checked_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#idivmod_checked_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.idivmod_checked_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Unchecked signed integer division and modulus

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#idivmod_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#idivmod_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.idivmod_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Checked unsigned integer division with divide-by-zero checking

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(core.adt [[] [(arithmetic.int.int ?width)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#idiv_checked_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#idiv_checked_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.idiv_checked_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Unchecked unsigned integer division

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#idiv_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#idiv_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.idiv_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Checked unsigned integer modulus with divide-by-zero checking

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(core.adt [[] [(arithmetic.int.int ?width)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#imod_checked_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#imod_checked_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.imod_checked_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Unchecked unsigned integer modulus

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#imod_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#imod_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.imod_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Checked signed integer division with divide-by-zero checking

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(core.adt [[] [(arithmetic.int.int ?width)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#idiv_checked_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#idiv_checked_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.idiv_checked_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Unchecked signed integer division

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#idiv_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#idiv_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.idiv_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Checked signed integer modulus with divide-by-zero checking

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(core.adt [[] [(arithmetic.int.int ?width)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#imod_checked_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#imod_checked_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.imod_checked_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Unchecked signed integer modulus

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#imod_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#imod_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.imod_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Bitwise AND

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#iand {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#iand {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.iand" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Bitwise OR

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ior {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ior {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ior" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Bitwise XOR

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ixor {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ixor {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ixor" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Bitwise NOT

## Type
<pre>
(core.fn [(arithmetic.int.int ?width)] [(arithmetic.int.int ?width)] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#inot {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#inot {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.inot" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Shift first input left by k bits where k is unsigned interpretation of second input (leftmost bits dropped, rightmost bits set to zero)

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ishl {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ishl {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ishl" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Shift first input right by k bits where k is unsigned interpretation of second input (rightmost bits dropped, leftmost bits set to zero)

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#ishr {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#ishr {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.ishr" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Rotate first input left by k bits where k is unsigned interpretation of second input (leftmost bits replace rightmost bits)

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#irotl {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#irotl {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.irotl" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Rotate first input right by k bits where k is unsigned interpretation of second input (rightmost bits replace leftmost bits)

## Type
<pre>
(core.fn
  [(arithmetic.int.int ?width) (arithmetic.int.int ?width)]
  [(arithmetic.int.int ?width)]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#irotr {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#irotr {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.irotr" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Convert signed to unsigned

## Type
<pre>
(core.fn [(arithmetic.int.int ?width)] [(arithmetic.int.int ?width)] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#is_to_u {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#is_to_u {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.is_to_u" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
    /**Convert unsigned to signed

## Type
<pre>
(core.fn [(arithmetic.int.int ?width)] [(arithmetic.int.int ?width)] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#iu_to_s {
        ///`width : core.nat`
        #[allow(missing_docs)]
        pub r#width: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#iu_to_s {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "arithmetic.int.iu_to_s" {
                return None;
            }
            let [r#width] = operation.params.try_into().ok()?;
            Some(Self { r#width })
        }
    }
}
pub mod r#collections_array {
    /**Fixed-length array.

## Type
<pre>
core.type
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#array {
        ///`len : core.nat`
        #[allow(missing_docs)]
        pub r#len: hugr_model::v0::TermId,
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#array {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "collections.array.array" {
                return None;
            }
            let [r#len, r#type] = apply.args.try_into().ok()?;
            Some(Self { r#len, r#type })
        }
    }
    /**Constant array value.

## Type
<pre>
(core.const (collections.array.array ?len ?type) ?ext)
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#const {
        ///`len : core.nat`
        #[allow(missing_docs)]
        pub r#len: hugr_model::v0::TermId,
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
        ///`ext : core.ext_set`
        #[allow(missing_docs)]
        pub r#ext: hugr_model::v0::TermId,
        ///`values : (core.list (core.const ?type ?ext))`
        #[allow(missing_docs)]
        pub r#values: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#const {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "collections.array.const" {
                return None;
            }
            let [r#len, r#type, r#ext, r#values] = apply.args.try_into().ok()?;
            Some(Self {
                r#len,
                r#type,
                r#ext,
                r#values,
            })
        }
    }
    /**Create a new array from elements.

## Type
<pre>
(core.fn ?inputs [(collections.array.array ?len ?type)] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#new_array {
        ///`len : core.nat`
        #[allow(missing_docs)]
        pub r#len: hugr_model::v0::TermId,
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
        ///`inputs : (core.list core.type)`
        #[allow(missing_docs)]
        pub r#inputs: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#new_array {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "collections.array.new_array" {
                return None;
            }
            let [r#len, r#type, r#inputs] = operation.params.try_into().ok()?;
            Some(Self { r#len, r#type, r#inputs })
        }
    }
    /**Get an element from an array.

## Type
<pre>
(core.fn
  [(collections.array.array ?len ?type) prelude.usize]
  [(core.adt [[] [?type]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#get {
        ///`len : core.nat`
        #[allow(missing_docs)]
        pub r#len: hugr_model::v0::TermId,
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#get {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "collections.array.get" {
                return None;
            }
            let [r#len, r#type] = operation.params.try_into().ok()?;
            Some(Self { r#len, r#type })
        }
    }
    /**Set an element in an array.

## Type
<pre>
(core.fn
  [(collections.array.array ?len ?type) prelude.usize ?type]
  [(core.adt
     [[?type (collections.array.array ?len ?type)]
      [?type (collections.array.array ?len ?type)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#set {
        ///`len : core.nat`
        #[allow(missing_docs)]
        pub r#len: hugr_model::v0::TermId,
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#set {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "collections.array.set" {
                return None;
            }
            let [r#len, r#type] = operation.params.try_into().ok()?;
            Some(Self { r#len, r#type })
        }
    }
    /**Swap two elements in an array.

## Type
<pre>
(core.fn
  [(collections.array.array ?len ?type) prelude.usize prelude.usize]
  [(core.adt
     [[(collections.array.array ?len ?type)]
      [(collections.array.array ?len ?type)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#swap {
        ///`len : core.nat`
        #[allow(missing_docs)]
        pub r#len: hugr_model::v0::TermId,
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#swap {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "collections.array.swap" {
                return None;
            }
            let [r#len, r#type] = operation.params.try_into().ok()?;
            Some(Self { r#len, r#type })
        }
    }
    /**Discard an empty array.

## Type
<pre>
(core.fn [(collections.array.array 0 ?type)] [] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#discard_empty {
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#discard_empty {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "collections.array.discard_empty" {
                return None;
            }
            let [r#type] = operation.params.try_into().ok()?;
            Some(Self { r#type })
        }
    }
    /**Pop an element from the left of an array.

## Type
<pre>
(core.fn
  [(collections.array.array ?len ?type)]
  [(core.adt [[] [?type (collections.array.array ?reduced_len ?type)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#pop_left {
        ///`len : core.nat`
        #[allow(missing_docs)]
        pub r#len: hugr_model::v0::TermId,
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
        ///`reduced_len : core.nat`
        #[allow(missing_docs)]
        pub r#reduced_len: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#pop_left {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "collections.array.pop_left" {
                return None;
            }
            let [r#len, r#type, r#reduced_len] = operation.params.try_into().ok()?;
            Some(Self {
                r#len,
                r#type,
                r#reduced_len,
            })
        }
    }
    /**Pop an element from the right of an array.

## Type
<pre>
(core.fn
  [(collections.array.array ?len ?type)]
  [(core.adt [[] [?type (collections.array.array ?reduced_len ?type)]])]
  (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#pop_right {
        ///`len : core.nat`
        #[allow(missing_docs)]
        pub r#len: hugr_model::v0::TermId,
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
        ///`reduced_len : core.nat`
        #[allow(missing_docs)]
        pub r#reduced_len: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#pop_right {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "collections.array.pop_right" {
                return None;
            }
            let [r#len, r#type, r#reduced_len] = operation.params.try_into().ok()?;
            Some(Self {
                r#len,
                r#type,
                r#reduced_len,
            })
        }
    }
    /**Creates a new array whose elements are initialised by calling the given function n times.

## Type
<pre>
(core.fn
  [(core.fn [] [?type] ?ext)]
  [(collections.array.array ?len ?type)]
  ?ext)
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#repeat {
        ///`len : core.nat`
        #[allow(missing_docs)]
        pub r#len: hugr_model::v0::TermId,
        ///`type : core.type`
        #[allow(missing_docs)]
        pub r#type: hugr_model::v0::TermId,
        ///`ext : core.ext_set`
        #[allow(missing_docs)]
        pub r#ext: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#repeat {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "collections.array.repeat" {
                return None;
            }
            let [r#len, r#type, r#ext] = operation.params.try_into().ok()?;
            Some(Self { r#len, r#type, r#ext })
        }
    }
    /**A combination of map and foldl.

Applies a function to each element of the array with an accumulator that is passed through from start to finish. Returns the resulting array and the final state of the accumulator.

## Type
<pre>
(core.fn
  [(collections.array.array ?len ?a)
   (core.fn [?a ?state ...] [?b ?state ...] ?ext)
   ?state ...]
  [(collections.array.array ?len ?b) ?state ...]
  ?ext)
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#scan {
        ///`len : core.nat`
        #[allow(missing_docs)]
        pub r#len: hugr_model::v0::TermId,
        ///`a : core.type`
        #[allow(missing_docs)]
        pub r#a: hugr_model::v0::TermId,
        ///`b : core.type`
        #[allow(missing_docs)]
        pub r#b: hugr_model::v0::TermId,
        ///`state : (core.list core.type)`
        #[allow(missing_docs)]
        pub r#state: hugr_model::v0::TermId,
        ///`ext : core.ext_set`
        #[allow(missing_docs)]
        pub r#ext: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#scan {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "collections.array.scan" {
                return None;
            }
            let [r#len, r#a, r#b, r#state, r#ext] = operation.params.try_into().ok()?;
            Some(Self {
                r#len,
                r#a,
                r#b,
                r#state,
                r#ext,
            })
        }
    }
}
pub mod r#core_const {
    /**`core.const.adt`.

## Type
<pre>
(core.const (core.adt ?variants) ?ext)
</pre>
*/
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

## Type
<pre>
core.constraint
</pre>
*/
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

## Type
<pre>
core.type
</pre>
*/
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

## Type
<pre>
core.static
</pre>
*/
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

## Type
<pre>
core.static
</pre>
*/
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

## Type
<pre>
core.static
</pre>
*/
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

## Type
<pre>
core.constraint
</pre>
*/
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

## Type
<pre>
core.static
</pre>
*/
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
    /**Algebraic data types (ADTs).

## Type
<pre>
core.type
</pre>
*/
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
    /**String literals.

## Type
<pre>
core.static
</pre>
*/
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
    /**Natural number literals.

## Type
<pre>
core.static
</pre>
*/
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
    /**Byte literals.

## Type
<pre>
core.static
</pre>
*/
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
    /**Float literals.

## Type
<pre>
core.static
</pre>
*/
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

## Type
<pre>
core.static
</pre>
*/
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

## Type
<pre>
core.static
</pre>
*/
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

## Type
<pre>
core.static
</pre>
*/
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
    /**Constant values.

## Type
<pre>
core.static
</pre>
*/
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
    /**Static lists.

## Type
<pre>
core.static
</pre>
*/
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
    /**Static tuples.

## Type
<pre>
core.static
</pre>
*/
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

## Type
<pre>
(core.fn ?inputs ?outputs ?ext)
</pre>
*/
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

## Type
<pre>
(core.fn [(core.fn ?inputs ?outputs ?ext) ?inputs ...] ?outputs ?ext)
</pre>
*/
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

## Type
<pre>
(core.fn [] [?type] ?ext)
</pre>
*/
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
    /**Create a value for an ADT.

## Type
<pre>
(core.fn ?types [(core.adt ?variants)] (ext))
</pre>
*/
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
    /**Metadata for documentation.

## Type
<pre>
core.meta
</pre>
*/
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
pub mod r#arithmetic_int_types {
    /**Integer type.

## Type
<pre>
core.type
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#int {
        ///`bitwidth : core.nat`
        #[allow(missing_docs)]
        pub r#bitwidth: hugr_model::v0::TermId,
    }
    impl<'a> ::hugr_model::v0::view::View<'a> for r#int {
        type Id = ::hugr_model::v0::TermId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let apply: ::hugr_model::v0::view::NamedConstructor = module.view(id)?;
            if apply.name != "arithmetic.int.types.int" {
                return None;
            }
            let [r#bitwidth] = apply.args.try_into().ok()?;
            Some(Self { r#bitwidth })
        }
    }
}
pub mod r#logic {
    /**Logical `and`.

## Type
<pre>
(core.fn [(core.adt [[] []]) (core.adt [[] []])] [(core.adt [[] []])] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#and {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#and {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "logic.and" {
                return None;
            }
            let [] = operation.params.try_into().ok()?;
            Some(Self {})
        }
    }
    /**Logical `or`.

## Type
<pre>
(core.fn [(core.adt [[] []]) (core.adt [[] []])] [(core.adt [[] []])] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#or {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#or {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "logic.or" {
                return None;
            }
            let [] = operation.params.try_into().ok()?;
            Some(Self {})
        }
    }
    /**Logical `xor`.

## Type
<pre>
(core.fn [(core.adt [[] []]) (core.adt [[] []])] [(core.adt [[] []])] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#xor {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#xor {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "logic.xor" {
                return None;
            }
            let [] = operation.params.try_into().ok()?;
            Some(Self {})
        }
    }
    /**Boolean equality.

## Type
<pre>
(core.fn [(core.adt [[] []]) (core.adt [[] []])] [(core.adt [[] []])] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#eq {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#eq {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "logic.eq" {
                return None;
            }
            let [] = operation.params.try_into().ok()?;
            Some(Self {})
        }
    }
    /**Logical `not`.

## Type
<pre>
(core.fn [(core.adt [[] []])] [(core.adt [[] []])] (ext))
</pre>
*/
    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[allow(non_camel_case_types)]
    pub struct r#not {}
    impl<'a> ::hugr_model::v0::view::View<'a> for r#not {
        type Id = ::hugr_model::v0::NodeId;
        fn view(module: &'a ::hugr_model::v0::Module<'a>, id: Self::Id) -> Option<Self> {
            let operation: ::hugr_model::v0::view::NamedOperation = module.view(id)?;
            if operation.name != "logic.not" {
                return None;
            }
            let [] = operation.params.try_into().ok()?;
            Some(Self {})
        }
    }
}

