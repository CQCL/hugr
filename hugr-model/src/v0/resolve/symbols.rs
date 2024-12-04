use std::hash::BuildHasherDefault;

use fxhash::FxHasher;
use indexmap::IndexMap;

use crate::v0::{NodeId, RegionId, SymbolIntroError, SymbolRef, SymbolRefError};

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

pub struct Symbols<'a> {
    symbols: FxIndexMap<&'a str, NodeId>,
    nodes: FxIndexMap<NodeId, NodeData>,
    regions: FxIndexMap<RegionId, RegionData>,
}

impl<'a> Symbols<'a> {
    pub fn new() -> Self {
        Self {
            symbols: FxIndexMap::default(),
            nodes: FxIndexMap::default(),
            regions: FxIndexMap::default(),
        }
    }

    pub fn enter(&mut self, region: RegionId) {
        let region_data = RegionData { binding_count: 0 };
        self.regions.insert(region, region_data);
    }

    pub fn exit(&mut self) {
        let (_, region_data) = self.regions.pop().unwrap();

        for _ in 0..region_data.binding_count {
            let (node, node_data) = self.nodes.pop().unwrap();

            if let Some(shadows) = node_data.shadows {
                self.symbols[node_data.position] = shadows;
            } else {
                debug_assert_eq!(self.symbols.last().unwrap().1, &node);
                self.symbols.pop();
            }
        }
    }

    pub fn insert(&mut self, name: &'a str, node: NodeId) -> Result<(), SymbolIntroError> {
        let mut region_entry = self.regions.last_entry().unwrap();
        let (position, shadowed) = self.symbols.insert_full(name, node);

        if let Some(shadowed) = shadowed {
            if self.nodes[&shadowed].scope_depth == region_entry.index() as _ {
                return Err(SymbolIntroError::DuplicateSymbol(
                    node,
                    shadowed,
                    name.to_string(),
                ));
            }
        }

        region_entry.get_mut().binding_count += 1;

        self.nodes.insert(
            node,
            NodeData {
                scope_depth: region_entry.index() as _,
                shadows: shadowed,
                position,
            },
        );

        Ok(())
    }

    pub fn try_resolve(
        &self,
        symbol_ref: SymbolRef<'a>,
    ) -> Result<(SymbolRef<'a>, ScopeDepth), SymbolRefError> {
        match symbol_ref {
            SymbolRef::Direct(node) => {
                let node_data = self
                    .nodes
                    .get(&node)
                    .ok_or(SymbolRefError::NotVisible(node))?;

                // Check if the symbol has been shadowed at this point.
                if self.symbols[node_data.position] != node {
                    return Err(SymbolRefError::NotVisible(node).into());
                }

                Ok((symbol_ref, node_data.scope_depth))
            }
            SymbolRef::Named(name) => {
                if let Some(node) = self.symbols.get(name) {
                    let node_data = self.nodes.get(node).unwrap();
                    Ok((SymbolRef::Direct(*node), node_data.scope_depth))
                } else {
                    Ok((symbol_ref, 0))
                }
            }
        }
    }

    pub fn region_to_depth(&self, region: RegionId) -> Option<ScopeDepth> {
        Some(self.regions.get_index_of(&region)? as _)
    }

    pub fn depth_to_region(&self, depth: ScopeDepth) -> Option<RegionId> {
        let (region, _) = self.regions.get_index(depth as _)?;
        Some(*region)
    }
}

#[derive(Debug, Clone, Copy)]
struct NodeData {
    scope_depth: ScopeDepth,
    shadows: Option<NodeId>,
    position: usize,
}

#[derive(Debug, Clone, Copy)]
struct RegionData {
    binding_count: u32,
}

pub type ScopeDepth = u16;
