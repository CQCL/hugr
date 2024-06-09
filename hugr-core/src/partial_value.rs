#![allow(missing_docs)]
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use itertools::{zip_eq, Itertools as _};

use crate::ops::Value;
use crate::types::{Type, TypeEnum};

mod value_handle;

pub use value_handle::{ValueKey, ValueHandle};


/// TODO shouldn't be pub
#[derive(PartialEq, Clone, Eq)]
pub struct PartialSum(HashMap<usize, Vec<PartialValue>>);

impl PartialSum {
    pub fn unit() -> Self { Self::variant(0,[]) }
    pub fn variant(tag: usize, values: impl IntoIterator<Item = PartialValue>) -> Self {
        Self([(tag, values.into_iter().collect())].into_iter().collect())
    }

    pub fn num_variants(&self) -> usize {
        self.0.len()
    }

    fn assert_variants(&self) {
        assert_ne!(self.num_variants(), 0);
        for pv in self.0.values().flat_map(|x| x.iter()) {
            pv.assert_invariants();
        }
    }

    pub fn variant_field_value(&self, variant: usize, idx: usize) -> PartialValue {
        if let Some(row) = self.0.get(&variant) {
            assert!(row.len() > idx);
            row[idx].clone()
        } else {
            PartialValue::bottom()
        }
    }

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
        if v.len() != r.len() {
            return Err(self)
        }
        match zip_eq(v.into_iter(), r.into_iter())
            .map(|(v, t)| v.clone().try_into_value(t))
            .collect::<Result<Vec<_>,_>>() {
            Ok(vs) => {
                Value::sum(*k, vs, st.clone()).map_err(|_| self)
            }
            Err(_) => Err(self)
        }
    }

    // unsafe because we panic if any common rows have different lengths
    fn join_mut_unsafe(&mut self, other: Self) -> bool {
        let mut changed = false;

        for (k, v) in other.0 {
            if let Some(row) = self.0.get_mut(&k) {
                for (lhs, rhs) in zip_eq(row.iter_mut(), v.into_iter()) {
                    changed |= lhs.join_mut(rhs);
                }
            } else {
                self.0.insert(k, v);
                changed = true;
            }
        }
        changed
    }

    // unsafe because we panic if any common rows have different lengths
    fn meet_mut_unsafe(&mut self, other: Self) -> bool {
        let mut changed = false;
        let mut keys_to_remove = vec![];
        for k in self.0.keys() {
            if !other.0.contains_key(k) {
                keys_to_remove.push(*k);
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
        changed
    }

    pub fn supports_tag(&self, tag: usize) -> bool {
        self.0.contains_key(&tag)
    }
}

impl PartialOrd for PartialSum {
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

impl std::fmt::Debug for PartialSum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Hash for PartialSum {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (k, v) in &self.0 {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl TryFrom<ValueHandle> for PartialSum {
    type Error = ValueHandle;

    fn try_from(value: ValueHandle) -> Result<Self, Self::Error> {
        match value.value() {
            Value::Tuple { vs } => {
                let vec = (0..vs.len())
                    .map(|i| PartialValue::from(value.index(i)).into())
                    .collect();
                return Ok(Self([(0, vec)].into_iter().collect()));
            }
            Value::Sum { tag, values, .. } => {
                let vec = (0..values.len())
                    .map(|i| PartialValue::from(value.index(i)).into())
                    .collect();
                return Ok(Self([(*tag, vec)].into_iter().collect()));
            }
            _ => ()
        };
        Err(value)
    }
}

#[derive(PartialEq, Clone, Eq, Hash, Debug)]
pub enum PartialValue {
    Bottom,
    Value(ValueHandle),
    PartialSum(PartialSum),
    Top,
}

impl From<ValueHandle> for PartialValue {
    fn from(v: ValueHandle) -> Self {
        TryInto::<PartialSum>::try_into(v).map_or_else(Self::Value, Self::PartialSum)
    }
}

impl From<PartialSum> for PartialValue {
    fn from(v: PartialSum) -> Self {
        Self::PartialSum(v)
    }
}


impl PartialValue {
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
                assert!(matches!(v.clone().into(), Self::Value(_)))
            }
            _ => {}
        }
    }


    pub fn try_into_value(self, typ: &Type) -> Result<Value, Self> {
        let r = match self {
            Self::Value(v) => Ok(v.value().clone()),
            Self::PartialSum(ps) => ps.try_into_value(typ).map_err(Self::PartialSum),
            x => Err(x),
        }?;
        assert_eq!(typ, &r.get_type());
        Ok(r)
    }

    fn join_mut_value_handle(&mut self, vh: ValueHandle) -> bool {
        self.assert_invariants();
        let mut new_self = self;
        match &mut new_self {
            Self::Top => false,
            new_self @ Self::Value(_) => {
                let Self::Value(v) = *new_self else {
                    unreachable!()
                };
                if v == &vh {
                    false
                } else {
                    **new_self = Self::Top;
                    true
                }
            }
            s @ Self::PartialSum(_) => match vh.into() {
                Self::Value(_) => {
                    **s = Self::Top;
                    true
                }
                other => s.join_mut(other),
            },
            new_self @ Self::Bottom => {
                **new_self = vh.into();
                true
            }
        }
    }

    fn meet_mut_value_handle(&mut self, vh: ValueHandle) -> bool {
        self.assert_invariants();
        let mut new_self = self;
        match &mut new_self {
            Self::Bottom => false,
            new_self @ Self::Value(_) => {
                let Self::Value(v) = *new_self else {
                    unreachable!()
                };
                if v == &vh {
                    false
                } else {
                    **new_self = Self::Bottom;
                    true
                }
            }
            new_self @ Self::PartialSum(_) => match vh.into() {
                Self::Value(_) => {
                    **new_self = Self::Bottom;
                    true
                }
                other => new_self.join_mut(other),
            },
            new_self @ Self::Top => {
                **new_self = vh.into();
                true
            }
        }
    }

    fn value_handles_equal(&self, rhs: &ValueHandle) -> bool {
        let Self::Value(lhs) = self else { unreachable!() };
        lhs == rhs
            // The following is a good idea if ValueHandle gains an Eq
            // instance and so does not do this check:
            // || lhs.value() == rhs.value()
    }

    pub fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    pub fn join_mut(&mut self, other: Self) -> bool {
        // println!("join {self:?}\n{:?}", &other);
        let mut new_self = self;
        let changed = match (&mut new_self, other) {
            (Self::Top, _) => false,
            (new_self, other @ Self::Top) => {
                **new_self = other;
                true
            }
            (_, Self::Bottom) => false,
            (new_self @ Self::Bottom, other) => {
                **new_self = other;
                true
            }
            (new_self @ Self::Value(_), Self::Value(h2)) => {
                if new_self.value_handles_equal(&h2) {
                    false
                } else {
                    **new_self = Self::Top;
                    true
                }
            }
            (new_self @ Self::PartialSum(_), Self::PartialSum(ps2)) => {
                let Self::PartialSum(ps1) = *new_self else {
                    unreachable!()
                };

                ps1.join_mut_unsafe(ps2)
            }
            (new_self @ Self::Value(_), other) => {
                let mut old_self = other;
                std::mem::swap(*new_self, &mut old_self);
                let Self::Value(h) = old_self else {
                    unreachable!()
                };
                new_self.join_mut_value_handle(h)
            }
            (new_self, Self::Value(h)) => new_self.join_mut_value_handle(h),
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
        let mut new_self = self;
        let changed = match (&mut new_self, other) {
            (Self::Bottom, _) => false,
            (new_self, other @ Self::Bottom) => {
                **new_self = other;
                true
            }
            (_, Self::Top) => false,
            (new_self @ Self::Top, other) => {
                **new_self = other;
                true
            }
            (new_self @ Self::Value(_), Self::Value(h2)) => {
                if new_self.value_handles_equal(&h2) {
                    false
                } else {
                    **new_self = Self::Bottom;
                    true
                }
            }
            (new_self @ Self::PartialSum(_), Self::PartialSum(ps2)) => {
                let Self::PartialSum(ps1) = *new_self else {
                    unreachable!()
                };
                ps1.meet_mut_unsafe(ps2)
            }
            (new_self @ Self::Value(_), other @ Self::PartialSum(_)) => {
                let mut old_self = other;
                std::mem::swap(*new_self, &mut old_self);
                let Self::Value(h) = old_self else {
                    unreachable!()
                };
                new_self.meet_mut_value_handle(h)
            }
            (s @ Self::PartialSum(_), Self::Value(h)) => s.meet_mut_value_handle(h),
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

    pub fn supports_tag(&self, tag: usize) -> bool {
        match self {
            PartialValue::Bottom => false,
            PartialValue::Value(v) => v.tag() == tag, // can never be a sum or tuple
            PartialValue::PartialSum(ps) => ps.supports_tag(tag),
            PartialValue::Top => true,
        }
    }

    /// TODO docs
    /// just delegate to variant_field_value
    pub fn tuple_field_value(&self, idx: usize) -> Self {
        self.variant_field_value(0, idx)
    }

    /// TODO docs
    pub fn variant_field_value(&self, variant: usize, idx: usize) -> Self {
        match self {
            Self::Bottom => Self::Bottom,
            Self::PartialSum(ps) => {
                ps.variant_field_value(variant, idx)
            }
            Self::Value(v) => {
                if v.tag() == variant {
                    Self::Value(v.index(idx))
                } else {
                    Self::Bottom
                }
            },
            Self::Top => Self::Top,
        }
    }
}

impl PartialOrd for PartialValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (self, other) {
            (Self::Bottom, Self::Bottom) => Some(Ordering::Equal),
            (Self::Top, Self::Top) => Some(Ordering::Equal),
            (Self::Bottom, _) => Some(Ordering::Less),
            (_, Self::Bottom) => Some(Ordering::Greater),
            (Self::Top, _) => Some(Ordering::Greater),
            (_, Self::Top) => Some(Ordering::Less),
            (Self::Value(_), Self::Value(v2)) => self.value_handles_equal(v2).then_some(Ordering::Equal),
            (Self::PartialSum(ps1), Self::PartialSum(ps2)) => ps1.partial_cmp(ps2),
            _ => None,
        }
    }
}

#[cfg(test)]
mod test;
