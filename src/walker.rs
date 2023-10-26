use std::rc::Rc;

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

struct WalkerCallback<'a, T>(Box<dyn 'a + Fn(Node, OpType, &mut T) -> WalkResult>);

impl<'a, T, F: 'a + Fn(Node, OpType, &mut T) -> WalkResult> From<F> for WalkerCallback<'a, T> {
    fn from(f: F) -> Self {
        Self(Box::new(f))
    }
}

pub struct Walker<'a, T> {
    pre_callbacks: Vec<WalkerCallback<'a, T>>,
    post_callbacks: Vec<WalkerCallback<'a, T>>,
}

fn call_back<O, T>(
    n: Node,
    o: OpType,
    t: &mut T,
    f: impl Fn(Node, O, &mut T) -> WalkResult,
) -> WalkResult
where
    OpType: TryInto<O>,
{
    match o.try_into() {
        Ok(x) => f(n, x, t),
        _ => WalkResult::Advance,
    }
}

enum WorkItem {
    Visit(Node),
    Callback(WalkOrder, Node),
}

impl<'a, T> Walker<'a, T> {
    pub fn new() -> Self {
        Self {
            pre_callbacks: Vec::new(),
            post_callbacks: Vec::new(),
        }
    }

    pub fn visit<O, F: 'a + Fn(Node, O, &mut T) -> WalkResult>(
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
        callbacks.push((move |n, o, t: &'_ mut _| call_back(n, o, t, g.as_ref())).into());
        self
    }

    pub fn walk(&self, hugr: impl HugrView, t: &mut T) {
        // We intentionally avoid recursion so that we can robustly accept very deep hugrs
        let mut worklist = vec![WorkItem::Visit(hugr.root())];

        while let Some(wi) = worklist.pop() {
            match wi {
                WorkItem::Visit(n) => {
                    worklist.push(WorkItem::Callback(WalkOrder::Postorder, n));
                    // TODO we should add children in topological order
                    worklist.extend(hugr.children(n).map(WorkItem::Visit));
                    worklist.push(WorkItem::Callback(WalkOrder::Preorder, n));
                }
                WorkItem::Callback(order, n) => {
                    let callbacks = match order {
                        WalkOrder::Preorder => &self.pre_callbacks,
                        WalkOrder::Postorder => &self.post_callbacks,
                    };
                    for cb in callbacks.iter() {
                        // this clone is unfortunate, to avoid this we would need a TryInto variant:
                        // try_into(&O) -> Option<&T>
                        if cb.0.as_ref()(n, hugr.get_optype(n).clone(), t) == WalkResult::Interrupt
                        {
                            return;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::{error::Error, iter::empty};

    use crate::types::Signature;
    use crate::{
        builder::{Container, HugrBuilder, ModuleBuilder},
        extension::{ExtensionRegistry, ExtensionSet},
        type_row,
        types::FunctionType,
    };

    use super::*;

    fn test1() -> Result<(), Box<dyn Error>> {
        let mut module_builder = ModuleBuilder::new();
        let sig = Signature {
            signature: FunctionType::new(type_row![], type_row![]),
            input_extensions: ExtensionSet::new(),
        };
        module_builder.define_function("f1", sig.clone());
        module_builder.define_function("f2", sig.clone());

        let hugr = module_builder.finish_hugr(&ExtensionRegistry::new())?;

        let mut s = String::new();
        Walker::<String>::new()
            .visit(WalkOrder::Preorder, |_, crate::ops::Module, r| {
                r.extend("pre".chars());
                r.extend(['m']);
                WalkResult::Advance
            })
            .visit(WalkOrder::Postorder, |_, crate::ops::Module, r| {
                r.extend("post".chars());
                r.extend(['n']);
                WalkResult::Advance
            })
            .visit(
                WalkOrder::Preorder,
                |_, crate::ops::FuncDecl { ref name, .. }, r| {
                    r.extend("pre".chars());
                    r.extend(name.chars());
                    WalkResult::Advance
                },
            )
            .visit(
                WalkOrder::Postorder,
                |_, crate::ops::FuncDecl { ref name, .. }, r| {
                    r.extend("post".chars());
                    r.extend(name.chars());
                    WalkResult::Advance
                },
            )
            .walk(&hugr, &mut s);

        assert_eq!(s, "prempref1pref2postf2postf1postn");
        Ok(())
    }
}
