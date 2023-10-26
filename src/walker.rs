#![allow(missing_docs)]
use std::rc::Rc;

use itertools::Itertools;
use lazy_static::__Deref;

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

struct WalkerCallback<'a, T, E>(Box<dyn 'a + Fn(Node, OpType, T) -> Result<T, E>>);

impl<'a, T, E, F: 'a + Fn(Node, OpType, T) -> Result<T, E>> From<F> for WalkerCallback<'a, T, E> {
    fn from(f: F) -> Self {
        Self(Box::new(f))
    }
}

pub struct Walker<'a, T, E> {
    pre_callbacks: Vec<WalkerCallback<'a, T, E>>,
    post_callbacks: Vec<WalkerCallback<'a, T, E>>,
}

fn call_back<O, T, E>(
    n: Node,
    o: OpType,
    t: T,
    f: &impl Fn(Node, O, T) -> Result<T, E>,
) -> Result<T, E>
where
    OpType: TryInto<O>,
{
    match o.try_into() {
        Ok(x) => f(n, x, t),
        _ => Ok(t),
    }
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

    pub fn visit<O, F: 'a + Fn(Node, O, T) -> Result<T, E>>(
        &mut self,
        walk_order: WalkOrder,
        f: F,
    ) -> &mut Self
    where
        OpType: TryInto<O>,
    {
        let g = Rc::new(f);
        let callbacks = match walk_order {
            WalkOrder::Preorder => &mut self.pre_callbacks,
            WalkOrder::Postorder => &mut self.post_callbacks,
        };
        callbacks.push((move |n, o, t| call_back(n, o, t, g.as_ref())).into());
        self
    }

    pub fn walk(&self, hugr: impl HugrView, mut t: T) -> Result<T, E> {
        // We intentionally avoid recursion so that we can robustly accept very deep hugrs
        let mut worklist = vec![WorkItem::Visit(hugr.root())];

        while let Some(wi) = worklist.pop() {
            match wi {
                WorkItem::Visit(n) => {
                    worklist.push(WorkItem::Callback(WalkOrder::Postorder, n));
                    // TODO we should add children in topological order
                    let children = hugr.children(n).collect_vec();
                    worklist.extend(children.into_iter().rev().map(WorkItem::Visit));
                    worklist.push(WorkItem::Callback(WalkOrder::Preorder, n));
                }
                WorkItem::Callback(order, n) => {
                    let callbacks = match order {
                        WalkOrder::Preorder => &self.pre_callbacks,
                        WalkOrder::Postorder => &self.post_callbacks,
                    };
                    let optype = hugr.get_optype(n);
                    for cb in callbacks.iter() {
                        // this clone is unfortunate, to avoid this we would need a TryInto variant:
                        // try_into(&O) -> Option<&T>
                        t = cb.0.as_ref()(n, optype.clone(), t)?;
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
            .visit(WalkOrder::Preorder, |_, crate::ops::Module, mut r| {
                r += ";prem";
                Ok(r)
            })
            .visit(WalkOrder::Postorder, |_, crate::ops::Module, mut r| {
                r += ";postm";
                Ok(r)
            })
            .visit(
                WalkOrder::Preorder,
                |_, crate::ops::FuncDefn { ref name, .. }, mut r| {
                    r += ";pre";
                    r += name.as_ref();
                    Ok(r)
                },
            )
            .visit(
                WalkOrder::Postorder,
                |_, crate::ops::FuncDefn { ref name, .. }, mut r| {
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
