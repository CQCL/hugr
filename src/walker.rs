#![allow(missing_docs)]

use std::ops::{Deref, DerefMut};

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

struct WalkerCallback<'a, T, E>(Box<dyn 'a + FnMut(Node, OpType, T) -> Result<T, E>>);

impl<'a, T, E> Deref for WalkerCallback<'a, T, E> {
    type Target = dyn 'a + FnMut(Node, OpType, T) -> Result<T, E>;
    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<'a, T, E> DerefMut for WalkerCallback<'a, T, E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}

impl<'a, T, E, F: 'a + FnMut(Node, OpType, T) -> Result<T, E>> From<F>
    for WalkerCallback<'a, T, E>
{
    fn from(f: F) -> Self {
        Self(Box::new(f))
    }
}

pub struct Walker<'a, T, E> {
    pre_callbacks: Vec<WalkerCallback<'a, T, E>>,
    post_callbacks: Vec<WalkerCallback<'a, T, E>>,
}

enum WorkItem {
    Visit(Node),
    Callback(WalkOrder, Node),
}

impl<'a, T, E> Walker<'a, T, E> {
    pub fn new() -> Self {
        Self {
            pre_callbacks: Vec::new(),
            post_callbacks: Vec::new(),
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

    pub fn walk(&mut self, hugr: impl HugrView, mut t: T) -> Result<T, E> {
        // We intentionally avoid recursion so that we can robustly accept very deep hugrs
        let mut worklist = vec![WorkItem::Visit(hugr.root())];

        while let Some(wi) = worklist.pop() {
            match wi {
                WorkItem::Visit(n) => {
                    worklist.push(WorkItem::Callback(WalkOrder::Postorder, n));
                    // TODO we should add children in topological order
                    let children = hugr.children(n).collect_vec();
                    // extend in reverse so that the first child is the top of the stack
                    worklist.extend(children.into_iter().rev().map(WorkItem::Visit));
                    worklist.push(WorkItem::Callback(WalkOrder::Preorder, n));
                }
                WorkItem::Callback(order, n) => {
                    let optype = hugr.get_optype(n);
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

    use crate::types::Signature;
    use crate::{
        builder::{Container, HugrBuilder, ModuleBuilder, SubContainer},
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

        let s = Walker::<_, Box<dyn Error>>::new()
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
            .walk(&hugr, String::new())?;

        assert_eq!(s, ";prem;pref1;postf1;pref2;postf2;postm");
        Ok(())
    }
}
