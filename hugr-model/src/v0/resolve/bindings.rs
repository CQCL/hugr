use fxhash::FxHashMap;
use std::collections::hash_map::Entry as HashMapEntry;
use std::hash::Hash;

#[derive(Debug, Clone)]
pub struct Bindings<K, S, V> {
    /// The bindings that are currently visible.
    bindings: FxHashMap<K, (V, ScopeId)>,

    /// An undo log that keeps track of changes to the bindings.
    undo_log: Vec<(K, Option<(V, ScopeId)>)>,

    /// Stack of scopes together with the depth of the undo log when the scope was entered.
    scopes: Vec<(S, UndoDepth)>,
}

impl<K, S, V> Bindings<K, S, V>
where
    K: Eq + Hash + Copy,
    V: Copy,
{
    /// Create a new `Bindings` instance.
    pub fn new() -> Self {
        Self {
            bindings: FxHashMap::default(),
            undo_log: Vec::new(),
            scopes: vec![],
        }
    }

    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        let scope = self
            .scopes
            .len()
            .checked_sub(1)
            .expect("no scope to insert into");

        let undo_log = &mut self.undo_log;

        match self.bindings.entry(key) {
            HashMapEntry::Occupied(entry) if entry.get().1 == scope => {
                Entry::Occupied(OccupiedEntry { entry })
            }
            HashMapEntry::Occupied(entry) => Entry::Visible(VisibleEntry {
                entry,
                scope,
                undo_log,
            }),
            HashMapEntry::Vacant(entry) => Entry::Vacant(VacantEntry {
                scope,
                entry,
                undo_log,
            }),
        }
    }

    /// Try to insert a new binding into the current scope.
    ///
    /// If the key is already bound in the current scope, the old value is returned
    /// and the bindings remain unchanged.
    ///
    /// # Panics
    ///
    /// Panics if there is no current scope.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<(), V> {
        self.entry(key).try_insert(value)
    }

    /// Retrieve a binding with the given key.
    pub fn get(&self, key: K) -> Option<V> {
        Some(self.bindings.get(&key)?.0)
    }

    /// Enter a new scope.
    pub fn enter(&mut self, scope: S) {
        let depth = self.undo_log.len();
        self.scopes.push((scope, depth));
    }

    /// Enter a new scope that is isolated from the parent scope.
    ///
    /// No bindings from the parent scope or any of its ancestors will be visible in the new scope.
    /// When the new scope is exited, the previous bindings will be restored.
    pub fn enter_isolated(&mut self, scope: S) {
        self.enter(scope);

        self.undo_log.extend(
            self.bindings
                .drain()
                .map(|(name, binding)| (name, Some(binding))),
        );
    }

    /// Exit the current scope.
    ///
    /// This will remove all bindings that were added in the current scope, and
    /// restore the bindings that were shadowed by the current scope.
    ///
    /// # Panics
    ///
    /// Panics if there is no current scope.
    pub fn exit(&mut self) -> S {
        let (scope, depth) = self.scopes.pop().expect("unbalanced scopes");

        while self.undo_log.len() > depth {
            let (key, change) = self.undo_log.pop().unwrap();
            match change {
                Some(binding) => self.bindings.insert(key, binding),
                None => self.bindings.remove(&key),
            };
        }

        scope
    }

    /// Get the current scope.
    pub fn scope(&self) -> Option<&S> {
        Some(&self.scopes.last()?.0)
    }
}

impl<K, S, V> Default for Bindings<K, S, V>
where
    K: Eq + Hash + Copy,
    V: Copy,
{
    fn default() -> Self {
        Self::new()
    }
}

type ScopeId = usize;
type UndoDepth = usize;

/// An entry in a [`Bindings`] map.
pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
    Visible(VisibleEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Eq + Hash + Copy,
    V: Copy,
{
    pub fn key(&self) -> K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
            Entry::Visible(entry) => entry.key(),
        }
    }

    pub fn try_insert(self, value: V) -> Result<(), V> {
        match self {
            Entry::Occupied(entry) => Err(entry.get()),
            Entry::Vacant(entry) => {
                entry.insert(value);
                Ok(())
            }
            Entry::Visible(entry) => {
                entry.insert(value);
                Ok(())
            }
        }
    }
}

/// An entry in a [`Bindings`] map that is occupied by a binding in a parent scope,
/// but vacant in the current scope.
pub struct VisibleEntry<'a, K, V> {
    entry: std::collections::hash_map::OccupiedEntry<'a, K, (V, ScopeId)>,
    undo_log: &'a mut Vec<(K, Option<(V, ScopeId)>)>,
    scope: ScopeId,
}

impl<'a, K, V> VisibleEntry<'a, K, V>
where
    K: Eq + Hash + Copy,
    V: Copy,
{
    pub fn key(&self) -> K {
        *self.entry.key()
    }

    pub fn get(&self) -> V {
        self.entry.get().0
    }

    pub fn insert(mut self, value: V) {
        let key = self.key();
        let shadowed = self.entry.insert((value, self.scope));
        self.undo_log.push((key, Some(shadowed)));
    }
}

/// An entry in a [`Bindings`] map that is occupied in the current scope.
pub struct OccupiedEntry<'a, K, V> {
    entry: std::collections::hash_map::OccupiedEntry<'a, K, (V, ScopeId)>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V>
where
    K: Eq + Hash + Copy,
    V: Copy,
{
    pub fn key(&self) -> K {
        *self.entry.key()
    }

    pub fn get(&self) -> V {
        self.entry.get().0
    }
}

/// An entry in a [`Bindings`] map that is vacant in the current scope and not visible
/// from any parent scope.
pub struct VacantEntry<'a, K, V> {
    entry: std::collections::hash_map::VacantEntry<'a, K, (V, ScopeId)>,
    undo_log: &'a mut Vec<(K, Option<(V, ScopeId)>)>,
    scope: ScopeId,
}

impl<'a, K, V> VacantEntry<'a, K, V>
where
    K: Eq + Hash + Copy,
    V: Copy,
{
    pub fn key(&self) -> K {
        *self.entry.key()
    }

    pub fn insert(self, value: V) {
        let key = self.key();
        self.entry.insert((value, self.scope));
        self.undo_log.push((key, None));
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
        assert_eq!(bindings.get("foo"), Some(1));
        bindings.enter(());
        // The binding for `foo` is still visible in the new scope:
        assert_eq!(bindings.get("foo"), Some(1));
        assert_eq!(bindings.try_insert("foo", 2), Ok(()));
        // The new binding shadows the old one:
        assert_eq!(bindings.get("foo"), Some(2));
        bindings.exit();
        // The old binding is restored:
        assert_eq!(bindings.get("foo"), Some(1));
        bindings.exit();
        // The binding is removed:
        assert_eq!(bindings.get("foo"), None);
    }

    #[test]
    fn shadow_isolated() {
        let mut bindings = Bindings::<&'static str, (), usize>::new();
        bindings.enter(());
        assert_eq!(bindings.try_insert("foo", 1), Ok(()));
        assert_eq!(bindings.get("foo"), Some(1));
        bindings.enter_isolated(());
        // The binding for `foo` is not visible in the new scope:
        assert_eq!(bindings.get("foo"), None);
        assert_eq!(bindings.try_insert("foo", 2), Ok(()));
        assert_eq!(bindings.get("foo"), Some(2));
        bindings.exit();
        // The old binding is restored:
        assert_eq!(bindings.get("foo"), Some(1));
        bindings.exit();
        assert_eq!(bindings.get("foo"), None);
    }

    #[test]
    fn duplicate() {
        let mut bindings = Bindings::<&'static str, (), usize>::new();
        bindings.enter(());
        assert_eq!(bindings.try_insert("foo", 1), Ok(()));
        assert_eq!(bindings.get("foo"), Some(1));
        assert_eq!(bindings.try_insert("foo", 2), Err(1));
        // The old binding is still there and unchanged:
        assert_eq!(bindings.get("foo"), Some(1));
    }
}
