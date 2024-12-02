use fxhash::FxHashMap;
use std::collections::hash_map::Entry as HashMapEntry;
use std::hash::Hash;

pub struct Bindings<K, S, V> {
    key_to_binding: FxHashMap<K, BindingIndex>,
    bindings: Vec<Binding<K, V>>,
    value_to_binding: FxHashMap<V, BindingIndex>,
    scopes: Vec<Scope<S>>,
}

type BindingIndex = usize;
type ScopeIndex = usize;

#[derive(Debug, Clone, Copy)]
struct Binding<K, V> {
    name: K,
    value: V,
    scope: ScopeIndex,
    shadows: BindingIndex,
    shadowed: bool,
}

#[derive(Debug, Clone, Copy)]
struct Scope<S> {
    binding_depth: usize,
    isolation_depth: usize,
    scope: S,
}

impl<K, S, V> Bindings<K, S, V>
where
    K: Eq + Hash + Copy,
    V: Eq + Hash + Copy,
    S: Copy,
{
    pub fn new() -> Self {
        Self {
            key_to_binding: FxHashMap::default(),
            scopes: Vec::new(),
            bindings: Vec::new(),
            value_to_binding: FxHashMap::default(),
        }
    }

    pub fn get_value_by_key(&self, key: K) -> Option<V> {
        let binding_index = *self.key_to_binding.get(&key)?;
        let binding = self.bindings[binding_index];

        if binding.shadowed || binding.scope < self.isolation_depth() {
            None
        } else {
            Some(binding.value)
        }
    }

    pub fn get_key_by_value(&self, value: V) -> Option<K> {
        let binding_index = *self.value_to_binding.get(&value)?;
        let binding = self.bindings[binding_index];

        if binding.shadowed || binding.scope < self.isolation_depth() {
            None
        } else {
            Some(binding.name)
        }
    }

    pub fn try_insert(&mut self, key: K, value: V) -> Result<(), V> {
        let scope = self
            .scopes
            .len()
            .checked_sub(1)
            .expect("no scope to insert into");

        let shadows = match self.key_to_binding.entry(key) {
            HashMapEntry::Occupied(mut entry) => {
                let shadows = *entry.get();
                let shadows_binding = &mut self.bindings[shadows];

                if shadows_binding.scope == scope {
                    return Err(shadows_binding.value);
                }

                shadows_binding.shadowed = true;
                self.value_to_binding.remove(&shadows_binding.value);
                entry.insert(self.bindings.len());
                shadows
            }
            HashMapEntry::Vacant(entry) => {
                entry.insert(self.bindings.len());
                BindingIndex::MAX
            }
        };

        self.value_to_binding.insert(value, self.bindings.len());

        self.bindings.push(Binding {
            name: key,
            value,
            scope,
            shadows,
            shadowed: false,
        });

        Ok(())
    }

    fn isolation_depth(&self) -> usize {
        self.scopes.last().map_or(0, |scope| scope.isolation_depth)
    }

    pub fn enter(&mut self, scope: S) {
        self.scopes.push(Scope {
            scope,
            binding_depth: self.bindings.len(),
            isolation_depth: self.isolation_depth(),
        });
    }

    pub fn enter_isolated(&mut self, scope: S) {
        self.scopes.push(Scope {
            scope,
            binding_depth: self.bindings.len(),
            isolation_depth: self.scopes.len(),
        });
    }

    pub fn exit(&mut self) -> S {
        let scope = self.scopes.pop().expect("no scope to exit");

        while self.bindings.len() > scope.binding_depth {
            let binding = self.bindings.pop().unwrap();
            self.value_to_binding.remove(&binding.value);

            if let Some(shadows) = self.bindings.get_mut(binding.shadows) {
                debug_assert!(shadows.shadowed);
                debug_assert!(shadows.name == binding.name);
                shadows.shadowed = false;
                self.key_to_binding.insert(shadows.name, binding.shadows);
                self.value_to_binding.insert(shadows.value, binding.shadows);
            } else {
                self.key_to_binding.remove(&binding.name);
            }
        }

        scope.scope
    }

    pub fn scope(&self) -> Option<&S> {
        Some(&self.scopes.last()?.scope)
    }
}

#[cfg(test)]
mod test {
    use super::Bindings;

    #[test]
    fn shadow() {
        let mut bindings = Bindings::<&'static str, (), usize>::new();
        bindings.enter(());
        assert_eq!(bindings.try_insert("foo", 1), Ok(()));
        assert_eq!(bindings.get_value_by_key("foo"), Some(1));
        assert_eq!(bindings.get_key_by_value(1), Some("foo"));
        bindings.enter(());
        // The binding for `foo` is still visible in the new scope:
        assert_eq!(bindings.get_value_by_key("foo"), Some(1));
        assert_eq!(bindings.get_key_by_value(1), Some("foo"));
        assert_eq!(bindings.try_insert("foo", 2), Ok(()));
        // The new binding shadows the old one:
        assert_eq!(bindings.get_value_by_key("foo"), Some(2));
        assert_eq!(bindings.get_key_by_value(2), Some("foo"));
        assert_eq!(bindings.get_key_by_value(1), None);
        bindings.exit();
        // The old binding is restored:
        assert_eq!(bindings.get_value_by_key("foo"), Some(1));
        assert_eq!(bindings.get_key_by_value(1), Some("foo"));
        assert_eq!(bindings.get_key_by_value(2), None);
        bindings.exit();
        // The binding is removed:
        assert_eq!(bindings.get_value_by_key("foo"), None);
        assert_eq!(bindings.get_key_by_value(1), None);
    }

    #[test]
    fn shadow_isolated() {
        let mut bindings = Bindings::<&'static str, (), usize>::new();
        bindings.enter(());
        assert_eq!(bindings.try_insert("foo", 1), Ok(()));
        assert_eq!(bindings.get_value_by_key("foo"), Some(1));
        bindings.enter_isolated(());
        // The binding for `foo` is not visible in the new scope:
        assert_eq!(bindings.get_value_by_key("foo"), None);
        assert_eq!(bindings.get_key_by_value(1), None);
        assert_eq!(bindings.try_insert("foo", 2), Ok(()));
        assert_eq!(bindings.get_value_by_key("foo"), Some(2));
        bindings.exit();
        // The old binding is restored:
        assert_eq!(bindings.get_value_by_key("foo"), Some(1));
        assert_eq!(bindings.get_key_by_value(1), Some("foo"));
        bindings.exit();
        assert_eq!(bindings.get_value_by_key("foo"), None);
    }

    #[test]
    fn duplicate() {
        let mut bindings = Bindings::<&'static str, (), usize>::new();
        bindings.enter(());
        assert_eq!(bindings.try_insert("foo", 1), Ok(()));
        assert_eq!(bindings.try_insert("bar", 2), Ok(()));
        assert_eq!(bindings.get_value_by_key("foo"), Some(1));
        assert_eq!(bindings.try_insert("foo", 2), Err(1));
        // The old binding is still there and unchanged:
        assert_eq!(bindings.get_value_by_key("foo"), Some(1));
        assert_eq!(bindings.get_key_by_value(1), Some("foo"));
        assert_eq!(bindings.get_key_by_value(2), Some("bar"));
    }
}
