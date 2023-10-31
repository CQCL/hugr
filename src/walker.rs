#![allow(missing_docs)]

use std::collections::HashSet;

use derive_more::{Deref, DerefMut};
use itertools::Itertools;

use crate::{ops::OpType, HugrView, Node};

#[derive(Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Debug)]
pub enum WalkResult {
    Advance,
    Interrupt,
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Debug)]
pub enum WalkOrder {
    Preorder,
    Postorder,
}

#[derive(Deref, DerefMut)]
struct WalkerCallback<'a, T, E>(Box<dyn 'a + FnMut(Node, OpType, T) -> Result<T, E>>);

impl<'a, T, E, F: 'a + FnMut(Node, OpType, T) -> Result<T, E>> From<F>
    for WalkerCallback<'a, T, E>
{
    fn from(f: F) -> Self {
        Self(Box::new(f))
    }
}

pub struct Walker<'a, H: HugrView, T, E> {
    pre_callbacks: Vec<WalkerCallback<'a, T, E>>,
    post_callbacks: Vec<WalkerCallback<'a, T, E>>,
    hugr: &'a H,
}

enum WorkItem {
    Visit(Node),
    Callback(WalkOrder, Node),
}

impl<'a, H: HugrView, T, E> Walker<'a, H, T, E> {
    pub fn new(hugr: &'a H) -> Self {
        Self {
            pre_callbacks: Vec::new(),
            post_callbacks: Vec::new(),
            hugr,
        }
    }

    pub fn previsit<O, F: 'a + FnMut(Node, O, T) -> Result<T, E>>(&mut self, f: F) -> &mut Self
    where
        OpType: TryInto<O>,
    {
        self.visit(WalkOrder::Preorder, f)
    }

    pub fn postvisit<O, F: 'a + FnMut(Node, O, T) -> Result<T, E>>(&mut self, f: F) -> &mut Self
    where
        OpType: TryInto<O>,
    {
        self.visit(WalkOrder::Postorder, f)
    }

    fn mut_callbacks(&mut self, order: WalkOrder) -> &mut Vec<WalkerCallback<'a, T, E>> {
        match order {
            WalkOrder::Preorder => &mut self.pre_callbacks,
            WalkOrder::Postorder => &mut self.post_callbacks,
        }
    }

    pub fn visit<O, F: 'a + FnMut(Node, O, T) -> Result<T, E>>(
        &mut self,
        walk_order: WalkOrder,
        mut f: F,
    ) -> &mut Self
    where
        OpType: TryInto<O>,
    {
        let cb = move |n, o: OpType, t| match o.try_into() {
            Ok(x) => f(n, x, t),
            _ => Ok(t),
        };
        self.mut_callbacks(walk_order).push(cb.into());
        self
    }

    pub fn walk(&mut self, mut t: T) -> Result<T, E> {
        // We intentionally avoid recursion so that we can robustly accept very deep hugrs
        let mut worklist = vec![WorkItem::Visit(self.hugr.root())];

        while let Some(wi) = worklist.pop() {
            match wi {
                WorkItem::Visit(n) => {
                    worklist.push(WorkItem::Callback(WalkOrder::Postorder, n));
                    let mut pushed_children = HashSet::new();
                    // We intend to only visit direct children.
                    //
                    // If the nodes children form a dataflow sibling graph we
                    // visit them in post dfs order. This traversal is not
                    // guaranteed to visit all nodes, only those reachable from
                    // the input node. (For example, LoadConstant nodes may not
                    // be reachable from the input node). So we do a second
                    // unordered traversal afterwards.
                    if let Some([input, _]) = self.hugr.get_io(n) {
                        use crate::hugr::views::PetgraphWrapper;
                        let wrapper = self.hugr.as_petgraph();
                        let mut dfs = ::petgraph::visit::DfsPostOrder::new(&wrapper, input);
                        while let Some(x) = dfs.next(&wrapper) {
                            worklist.push(WorkItem::Visit(x));
                            pushed_children.insert(x);
                        }
                    }

                    let rest_children = self
                        .hugr
                        .children(n)
                        .filter(|x| !pushed_children.contains(x))
                        .collect_vec();
                    // extend in reverse so that the first child is the top of the stack
                    worklist.extend(rest_children.into_iter().rev().map(WorkItem::Visit));
                    worklist.push(WorkItem::Callback(WalkOrder::Preorder, n));
                }
                WorkItem::Callback(order, n) => {
                    let optype = self.hugr.get_optype(n);
                    for cb in self.mut_callbacks(order).iter_mut() {
                        // this clone is unfortunate, to avoid this we would
                        // need a TryInto variant like:
                        // try_into(&O) -> Option<&T>
                        t = cb(n, optype.clone(), t)?;
                    }
                }
            }
        }
        Ok(t)
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use crate::builder::{Dataflow, DataflowHugr};
    use crate::extension::prelude::USIZE_T;
    use crate::hugr::hugrmut::sealed::HugrMutInternals;
    use crate::ops;
    use crate::types::Signature;
    use crate::{
        builder::{Container, FunctionBuilder, HugrBuilder, ModuleBuilder, SubContainer},
        extension::{ExtensionRegistry, ExtensionSet},
        ops::{FuncDefn, Module},
        type_row,
        types::FunctionType,
    };

    use super::*;

    #[test]
    fn test1() -> Result<(), Box<dyn Error>> {
        let mut module_builder = ModuleBuilder::new();
        let sig = Signature {
            signature: FunctionType::new(type_row![], type_row![]),
            input_extensions: ExtensionSet::new(),
        };
        module_builder
            .define_function("f1", sig.clone())?
            .finish_sub_container()?;
        module_builder
            .define_function("f2", sig.clone())?
            .finish_sub_container()?;

        let hugr = module_builder.finish_hugr(&ExtensionRegistry::new())?;

        let s = Walker::<_, _, Box<dyn Error>>::new(&hugr)
            .visit(WalkOrder::Preorder, |_, Module, mut r| {
                r += ";prem";
                Ok(r)
            })
            .visit(WalkOrder::Postorder, |_, Module, mut r| {
                r += ";postm";
                Ok(r)
            })
            .visit(
                WalkOrder::Preorder,
                |_, FuncDefn { ref name, .. }, mut r| {
                    r += ";pre";
                    r += name.as_ref();
                    Ok(r)
                },
            )
            .visit(
                WalkOrder::Postorder,
                |_, FuncDefn { ref name, .. }, mut r| {
                    r += ";post";
                    r += name.as_ref();
                    Ok(r)
                },
            )
            .walk(String::new())?;

        assert_eq!(s, ";prem;pref1;postf1;pref2;postf2;postm");
        Ok(())
    }

    struct Noop;

    impl TryFrom<ops::OpType> for Noop {
        type Error = ops::OpType;
        fn try_from(ot: ops::OpType) -> Result<Self, Self::Error> {
            match ot {
                ops::OpType::LeafOp(ops::LeafOp::Noop { .. }) => Ok(Noop),
                x => Err(x),
            }
        }
    }
    #[test]
    fn test2() -> Result<(), Box<dyn Error>> {
        use ops::handle::NodeHandle;
        let sig = Signature {
            signature: FunctionType::new(type_row![USIZE_T], type_row![USIZE_T]),
            input_extensions: ExtensionSet::new(),
        };
        let mut fun_builder = FunctionBuilder::new("f3", sig)?;
        let [i] = fun_builder.input_wires_arr();
        let noop1 = fun_builder.add_dataflow_op(ops::LeafOp::Noop { ty: USIZE_T }, [i])?;
        let noop2 =
            fun_builder.add_dataflow_op(ops::LeafOp::Noop { ty: USIZE_T }, [noop1.out_wire(0)])?;
        let mut h = fun_builder.finish_prelude_hugr_with_outputs([noop2.out_wire(0)])?;
        h.hugr_mut()
            .move_before_sibling(noop2.handle().node(), noop1.handle().node())?;

        let v = Walker::<_, Vec<Node>, Box<dyn Error>>::new(&h)
            .previsit(|n, Noop, mut v| {
                v.push(n);
                Ok(v)
            })
            .walk(Vec::new())?;
        assert_eq!(
            &[noop1.handle().node(), noop2.handle().node()],
            v.as_slice()
        );
        Ok(())
    }
}
