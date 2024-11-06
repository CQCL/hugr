use fxhash::FxHashMap;
use std::hash::Hash;
use thiserror::Error;

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

    /// Insert a new binding into the current scope.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), InsertBindingError<V>> {
        let scope = self
            .scopes
            .len()
            .checked_sub(1)
            .ok_or(InsertBindingError::NoScope)?;

        match self.bindings.insert(key, (value, scope)) {
            Some(shadowed) if shadowed.1 == scope => {
                self.bindings.insert(key, shadowed);
                Err(InsertBindingError::Duplicate(shadowed.0))
            }
            shadowed => {
                self.undo_log.push((key, shadowed));
                Ok(())
            }
        }
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

/// Error that can occur when inserting a binding.
#[derive(Debug, Error)]
pub enum InsertBindingError<V> {
    /// There is no current scope.
    #[error("no current scope")]
    NoScope,
    /// The binding has already been defined in this scope.
    #[error("binding has already been defined in this scope")]
    Duplicate(V),
}

#[cfg(test)]
mod test {
    use super::Bindings;
    use super::InsertBindingError;

    #[test]
    fn no_scope() {
        let mut bindings = Bindings::<&'static str, (), usize>::new();
        assert!(matches!(
            bindings.insert("foo", 1),
            Err(InsertBindingError::NoScope)
        ));
    }

    #[test]
    fn shadow() {
        let mut bindings = Bindings::<&'static str, (), usize>::new();
        bindings.enter(());
        bindings.insert("foo", 1).unwrap();
        assert_eq!(bindings.get("foo"), Some(1));
        bindings.enter(());
        // The binding for `foo` is still visible in the new scope:
        assert_eq!(bindings.get("foo"), Some(1));
        bindings.insert("foo", 2).unwrap();
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
        bindings.insert("foo", 1).unwrap();
        assert_eq!(bindings.get("foo"), Some(1));
        bindings.enter_isolated(());
        // The binding for `foo` is not visible in the new scope:
        assert_eq!(bindings.get("foo"), None);
        bindings.insert("foo", 2).unwrap();
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
        bindings.insert("foo", 1).unwrap();
        assert_eq!(bindings.get("foo"), Some(1));
        assert!(matches!(
            bindings.insert("foo", 2),
            Err(InsertBindingError::Duplicate(1))
        ));
        // The old binding is still there and unchanged:
        assert_eq!(bindings.get("foo"), Some(1));
    }
}
