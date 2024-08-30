#![allow(missing_docs)]

use hugr_core::ops::Value;
use hugr_core::types::{Type, TypeEnum, TypeRow};
use itertools::{zip_eq, Itertools};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Aka, deconstructible into Sum (TryIntoSum ?)
pub trait AbstractValue: Clone + std::fmt::Debug + PartialEq + Eq + Hash {
    /// We write this way to optimize query/inspection (is-it-a-sum),
    /// at the cost of requiring more cloning during actual conversion
    /// (inside the lazy Iterator, or for the error case, as Self remains)
    fn as_sum(&self) -> Option<(usize, impl Iterator<Item = Self> + '_)>;
}

// TODO ALAN inline into PartialValue? Has to be public as it's in a pub enum
#[derive(PartialEq, Clone, Eq)]
pub struct PartialSum<V>(pub HashMap<usize, Vec<PartialValue<V>>>);

impl<V> PartialSum<V> {
    pub fn unit() -> Self {
        Self::variant(0, [])
    }
    pub fn variant(tag: usize, values: impl IntoIterator<Item = PartialValue<V>>) -> Self {
        Self(HashMap::from([(tag, Vec::from_iter(values))]))
    }

    pub fn num_variants(&self) -> usize {
        self.0.len()
    }
}

impl<V: AbstractValue> PartialSum<V> {
    fn assert_variants(&self) {
        assert_ne!(self.num_variants(), 0);
        for pv in self.0.values().flat_map(|x| x.iter()) {
            pv.assert_invariants();
        }
    }

    // Err with key if any common rows have different lengths (self may have been mutated)
    fn try_join_mut(&mut self, other: Self) -> Result<bool, usize> {
        let mut changed = false;

        for (k, v) in other.0 {
            if let Some(row) = self.0.get_mut(&k) {
                if v.len() != row.len() {
                    // Better to check first and avoid mutation, but fine here
                    return Err(k);
                }
                for (lhs, rhs) in zip_eq(row.iter_mut(), v.into_iter()) {
                    changed |= lhs.join_mut(rhs);
                }
            } else {
                self.0.insert(k, v);
                changed = true;
            }
        }
        Ok(changed)
    }

    // Error with key if any common rows have different lengths ( => Bottom)
    fn try_meet_mut(&mut self, other: Self) -> Result<bool, usize> {
        let mut changed = false;
        let mut keys_to_remove = vec![];
        for (k, v) in self.0.iter() {
            match other.0.get(k) {
                None => keys_to_remove.push(*k),
                Some(o_v) => {
                    if v.len() != o_v.len() {
                        return Err(*k);
                    }
                }
            }
        }
        for (k, v) in other.0 {
            if let Some(row) = self.0.get_mut(&k) {
                for (lhs, rhs) in zip_eq(row.iter_mut(), v.into_iter()) {
                    changed |= lhs.meet_mut(rhs);
                }
            } else {
                keys_to_remove.push(k);
            }
        }
        for k in keys_to_remove {
            self.0.remove(&k);
            changed = true;
        }
        Ok(changed)
    }

    pub fn supports_tag(&self, tag: usize) -> bool {
        self.0.contains_key(&tag)
    }
}

impl<V: Clone + Into<Value>> PartialSum<V> {
    pub fn try_into_value(self, typ: &Type) -> Result<Value, Self> {
        let Ok((k, v)) = self.0.iter().exactly_one() else {
            Err(self)?
        };

        let TypeEnum::Sum(st) = typ.as_type_enum() else {
            Err(self)?
        };
        let Some(r) = st.get_variant(*k) else {
            Err(self)?
        };
        let Ok(r): Result<TypeRow, _> = r.clone().try_into() else {
            Err(self)?
        };
        if v.len() != r.len() {
            return Err(self);
        }
        match zip_eq(v.into_iter(), r.into_iter())
            .map(|(v, t)| v.clone().try_into_value(t))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(vs) => Value::sum(*k, vs, st.clone()).map_err(|_| self),
            Err(_) => Err(self),
        }
    }
}

impl<V: Clone> PartialSum<V> {
    pub fn variant_values(&self, variant: usize, len: usize) -> Option<Vec<PartialValue<V>>> {
        let row = self.0.get(&variant)?;
        assert!(row.len() == len);
        Some(row.clone())
    }
}

impl<V: PartialEq> PartialOrd for PartialSum<V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let max_key = self.0.keys().chain(other.0.keys()).copied().max().unwrap();
        let (mut keys1, mut keys2) = (vec![0; max_key + 1], vec![0; max_key + 1]);
        for k in self.0.keys() {
            keys1[*k] = 1;
        }

        for k in other.0.keys() {
            keys2[*k] = 1;
        }

        if let Some(ord) = keys1.partial_cmp(&keys2) {
            if ord != Ordering::Equal {
                return Some(ord);
            }
        } else {
            return None;
        }
        for (k, lhs) in &self.0 {
            let Some(rhs) = other.0.get(&k) else {
                unreachable!()
            };
            match lhs.partial_cmp(rhs) {
                Some(Ordering::Equal) => continue,
                x => {
                    return x;
                }
            }
        }
        Some(Ordering::Equal)
    }
}

impl<V: std::fmt::Debug> std::fmt::Debug for PartialSum<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<V: Hash> Hash for PartialSum<V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (k, v) in &self.0 {
            k.hash(state);
            v.hash(state);
        }
    }
}

#[derive(PartialEq, Clone, Eq, Hash, Debug)]
pub enum PartialValue<V> {
    Bottom,
    Value(V),
    PartialSum(PartialSum<V>),
    Top,
}

impl<V: AbstractValue> From<V> for PartialValue<V> {
    fn from(v: V) -> Self {
        v.as_sum()
            .map(|(tag, values)| Self::variant(tag, values.map(Self::Value)))
            .unwrap_or(Self::Value(v))
    }
}

impl<V> From<PartialSum<V>> for PartialValue<V> {
    fn from(v: PartialSum<V>) -> Self {
        Self::PartialSum(v)
    }
}

impl<V: AbstractValue> PartialValue<V> {
    // const BOTTOM: Self = Self::Bottom;
    // const BOTTOM_REF: &'static Self = &Self::BOTTOM;

    // fn initialised(&self) -> bool {
    //     !self.is_top()
    // }

    // fn is_top(&self) -> bool {
    //     self == &PartialValue::Top
    // }

    fn assert_invariants(&self) {
        match self {
            Self::PartialSum(ps) => {
                ps.assert_variants();
            }
            Self::Value(v) => {
                assert!(v.as_sum().is_none())
            }
            _ => {}
        }
    }

    fn join_mut_value_handle(&mut self, vh: V) -> bool {
        self.assert_invariants();
        match &*self {
            Self::Top => return false,
            Self::Value(v) if v == &vh => return false,
            Self::Value(v) => {
                *self = Self::Top;
            }
            Self::PartialSum(_) => match vh.into() {
                Self::Value(_) => {
                    *self = Self::Top;
                }
                other => return self.join_mut(other),
            },
            Self::Bottom => {
                *self = vh.into();
            }
        };
        true
    }

    fn meet_mut_value_handle(&mut self, vh: V) -> bool {
        self.assert_invariants();
        match &*self {
            Self::Bottom => false,
            Self::Value(v) => {
                if v == &vh {
                    false
                } else {
                    *self = Self::Bottom;
                    true
                }
            }
            Self::PartialSum(_) => match vh.into() {
                Self::Value(_) => {
                    *self = Self::Bottom;
                    true
                }
                other => self.meet_mut(other),
            },
            Self::Top => {
                *self = vh.into();
                true
            }
        }
    }

    pub fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    pub fn join_mut(&mut self, other: Self) -> bool {
        // println!("join {self:?}\n{:?}", &other);
        let changed = match (&*self, other) {
            (Self::Top, _) => false,
            (_, other @ Self::Top) => {
                *self = other;
                true
            }
            (_, Self::Bottom) => false,
            (Self::Bottom, other) => {
                *self = other;
                true
            }
            (Self::Value(h1), Self::Value(h2)) => {
                if h1 == &h2 {
                    false
                } else {
                    *self = Self::Top;
                    true
                }
            }
            (Self::PartialSum(_), Self::PartialSum(ps2)) => {
                let Self::PartialSum(ps1) = self else {
                    unreachable!()
                };
                match ps1.try_join_mut(ps2) {
                    Ok(ch) => ch,
                    Err(_) => {
                        *self = Self::Top;
                        true
                    }
                }
            }
            (Self::Value(_), mut other) => {
                std::mem::swap(self, &mut other);
                let Self::Value(old_self) = other else {
                    unreachable!()
                };
                self.join_mut_value_handle(old_self)
            }
            (_, Self::Value(h)) => self.join_mut_value_handle(h),
            // (new_self, _) => {
            //     **new_self = Self::Top;
            //     false
            // }
        };
        // if changed {
        // println!("join new self: {:?}", s);
        // }
        changed
    }

    pub fn meet(mut self, other: Self) -> Self {
        self.meet_mut(other);
        self
    }

    pub fn meet_mut(&mut self, other: Self) -> bool {
        let changed = match (&*self, other) {
            (Self::Bottom, _) => false,
            (_, other @ Self::Bottom) => {
                *self = other;
                true
            }
            (_, Self::Top) => false,
            (Self::Top, other) => {
                *self = other;
                true
            }
            (Self::Value(h1), Self::Value(h2)) => {
                if h1 == &h2 {
                    false
                } else {
                    *self = Self::Bottom;
                    true
                }
            }
            (Self::PartialSum(_), Self::PartialSum(ps2)) => {
                let Self::PartialSum(ps1) = self else {
                    unreachable!()
                };
                match ps1.try_meet_mut(ps2) {
                    Ok(ch) => ch,
                    Err(_) => {
                        *self = Self::Bottom;
                        true
                    }
                }
            }
            (Self::Value(_), mut other @ Self::PartialSum(_)) => {
                std::mem::swap(self, &mut other);
                let Self::Value(old_self) = other else {
                    unreachable!()
                };
                self.meet_mut_value_handle(old_self)
            }
            (Self::PartialSum(_), Self::Value(h)) => self.meet_mut_value_handle(h),
            // (new_self, _) => {
            //     **new_self = Self::Bottom;
            //     false
            // }
        };
        // if changed {
        // println!("join new self: {:?}", s);
        // }
        changed
    }

    pub fn top() -> Self {
        Self::Top
    }

    pub fn bottom() -> Self {
        Self::Bottom
    }

    pub fn variant(tag: usize, values: impl IntoIterator<Item = Self>) -> Self {
        PartialSum::variant(tag, values).into()
    }

    pub fn unit() -> Self {
        Self::variant(0, [])
    }

    pub fn variant_values(&self, tag: usize, len: usize) -> Option<Vec<PartialValue<V>>> {
        let vals = match self {
            PartialValue::Bottom => return None,
            PartialValue::Value(v) => v
                .as_sum()
                .filter(|(variant, _)| tag == *variant)?
                .1
                .map(PartialValue::Value)
                .collect(),
            PartialValue::PartialSum(ps) => ps.variant_values(tag, len)?,
            PartialValue::Top => vec![PartialValue::Top; len],
        };
        assert_eq!(vals.len(), len);
        Some(vals)
    }

    pub fn supports_tag(&self, tag: usize) -> bool {
        match self {
            PartialValue::Bottom => false,
            PartialValue::Value(v) => v.as_sum().is_some_and(|(v, _)| v == tag),
            PartialValue::PartialSum(ps) => ps.supports_tag(tag),
            PartialValue::Top => true,
        }
    }
}

impl<V: Clone + Into<Value>> PartialValue<V> {
    pub fn try_into_value(self, typ: &Type) -> Result<Value, Self> {
        let r = match self {
            Self::Value(v) => Ok(v.into().clone()),
            Self::PartialSum(ps) => ps.try_into_value(typ).map_err(Self::PartialSum),
            x => Err(x),
        }?;
        assert_eq!(typ, &r.get_type());
        Ok(r)
    }
}

impl<V: PartialEq> PartialOrd for PartialValue<V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (self, other) {
            (Self::Bottom, Self::Bottom) => Some(Ordering::Equal),
            (Self::Top, Self::Top) => Some(Ordering::Equal),
            (Self::Bottom, _) => Some(Ordering::Less),
            (_, Self::Bottom) => Some(Ordering::Greater),
            (Self::Top, _) => Some(Ordering::Greater),
            (_, Self::Top) => Some(Ordering::Less),
            (Self::Value(v1), Self::Value(v2)) => (v1 == v2).then_some(Ordering::Equal),
            (Self::PartialSum(ps1), Self::PartialSum(ps2)) => ps1.partial_cmp(ps2),
            _ => None,
        }
    }
}

#[cfg(test)]
mod test;
