//! Çontains [HierarchyTester], a tool for examining the hierarchy
//! about ancestry and descendancy.

use std::collections::{HashMap, hash_map::Entry};

use itertools::Itertools;

use crate::HugrView;

/// Caches enough information on the hierarchy of an immutably-held Hugr
/// to allow efficient querying of [Self::is_ancestor_of] and [Self::which_child_contains]
#[derive(Clone, Debug)]
pub struct HierarchyTester<'a, H: HugrView> {
    /// This both allows us to access the Hugr, but also guarantees the Hugr isn't
    /// changing beneath our back to invalidate the results.
    hugr: &'a H,
    /// The entry and exit indices (inclusive) of every node in the Hugr
    /// beneath the entrypoint. (Note: every entry number is different.)
    entry_exit: HashMap<H::Node, (usize, usize)>,
}

impl<'a, H: HugrView> HierarchyTester<'a, H> {
    /// Creates a new instance that knows about all descendants of the
    /// specified Hugr's entrypoint
    pub fn new(hugr: &'a H) -> Self {
        let mut entry_exit = HashMap::new();
        fn traverse<H: HugrView>(hugr: &H, n: H::Node, ee: &mut HashMap<H::Node, (usize, usize)>) {
            let old = ee.insert(n, (ee.len(), usize::MAX)); // second is placeholder for now
            debug_assert!(old.is_none());
            for ch in hugr.children(n) {
                traverse(hugr, ch, ee)
            }
            let end_idx = ee.len() - 1;
            let Entry::Occupied(oe) = ee.entry(n) else {
                panic!()
            };
            let (_, end) = oe.into_mut();
            *end = end_idx;
            debug_assert!(
                // Could do this on every which_child_contains?!
                hugr.children(n)
                    .tuple_windows()
                    .all(|(a, b)| ee.get(&a).unwrap().1 == ee.get(&b).unwrap().0 - 1)
            );
        }
        traverse(hugr, hugr.entrypoint(), &mut entry_exit);
        Self { hugr, entry_exit }
    }

    /// Returns true if `anc` is an ancestor of `desc`, including `anc == desc`.
    /// (See also [Self::is_strict_ancestor_of].)
    /// Constant time regardless of size/depth of Hugr.
    pub fn is_ancestor_of(&self, anc: H::Node, desc: H::Node) -> bool {
        let anc = self.entry_exit.get(&anc).unwrap();
        let desc = self.entry_exit.get(&desc).unwrap();
        anc.0 <= desc.0 && desc.0 <= anc.1
    }

    /// Returns true if `anc` is an ancestor of `desc`, excluding `anc == desc`.
    /// (See also [Self::is_ancestor_of].)
    /// Constant time regardless of size/depth of Hugr.
    pub fn is_strict_ancestor_of(&self, anc: H::Node, desc: H::Node) -> bool {
        let anc = self.entry_exit.get(&anc).unwrap();
        let desc = self.entry_exit.get(&desc).unwrap();
        anc.0 < desc.0 && desc.1 <= anc.1
    }

    /// Returns the child of `parent` which is an ancestor of `desc` - unique if there is one.
    /// Time O(n) in number of children of `parent` regardless of size/depth of Hugr.
    pub fn which_child_contains(&self, parent: H::Node, desc: H::Node) -> Option<H::Node> {
        // If we cached child *Vecs* then we could do a binary search, but since we don't,
        // even just getting the children into a Vec is linear, so we might as well...
        for c in self.hugr.children(parent) {
            if self.is_ancestor_of(c, desc) {
                return Some(c);
            }
        }
        None
    }
}

#[cfg(test)]
mod test {
    use std::iter;

    use proptest::prelude::{Just, Strategy};
    use proptest::proptest;

    use crate::builder::{DFGBuilder, Dataflow, HugrBuilder, SubContainer};
    use crate::extension::prelude::usize_t;
    use crate::types::Signature;
    use crate::{Hugr, HugrView};

    use super::HierarchyTester;

    #[derive(Clone, Debug)]
    struct Layout(Vec<Layout>);

    fn make<H: AsMut<Hugr> + AsRef<Hugr>>(dfg: &mut DFGBuilder<H>, l: Layout) {
        let [mut val] = dfg.input_wires_arr();
        for c in l.0 {
            let mut c_b = dfg
                .dfg_builder(Signature::new_endo(usize_t()), [val])
                .unwrap();
            make(&mut c_b, c);
            let [res] = c_b.finish_sub_container().unwrap().outputs_arr();
            val = res;
        }
        dfg.set_outputs([val]).unwrap()
    }

    fn any_layout() -> impl Strategy<Value = Layout> {
        Just(Layout(vec![])).prop_recursive(5, 10, 3, |elem| {
            proptest::collection::vec(elem, 1..5).prop_map(Layout)
        })
    }

    fn strict_ancestors<H: HugrView>(h: &H, n: H::Node) -> impl Iterator<Item = H::Node> {
        iter::successors(h.get_parent(n), |n| h.get_parent(*n))
    }

    proptest! {
        #[test]
        fn hierarchy_test(ly in any_layout()) {
            let mut h = DFGBuilder::new(Signature::new_endo(usize_t())).unwrap();
            make(&mut h, ly);
            let h = h.finish_hugr().unwrap();
            let ht = HierarchyTester::new(&h);
            for n1 in h.entry_descendants() {
                for n2 in h.entry_descendants() {
                    let is_strict_ancestor = strict_ancestors(&h, n1).any(|item| item==n2);
                    assert_eq!(ht.is_strict_ancestor_of(n2, n1), is_strict_ancestor);
                    assert_eq!(ht.is_ancestor_of(n2, n1), is_strict_ancestor || n1 == n2);
                }
            }
        }
    }
}
