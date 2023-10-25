use std::borrow::Cow;

use crate::extension::ExtensionRegistry;

use super::type_param::{TypeArg, TypeParam};
use super::{Type, TypeBound, TypeRow};

#[derive(Clone, Debug)]
pub(crate) struct Substitution<'a> {
    /// The number of variables bound locally-to the tree within which we are substituting
    /// (i.e. beneath the point where we started applying the substitution),
    /// so these should be untouched by the substitution
    leave_lowest: usize,
    /// What to do to variables that are affected
    mapping: Mapping<'a>,
    /// Since applying the substitution could narrow [TypeBound]s,
    /// we need this to recompute bounds of any [CustomType]s
    /// 
    /// [CustomType]: super::CustomType
    reg: &'a ExtensionRegistry,
}

#[derive(Clone, Debug)]
enum Mapping<'a> {
    /// An explicit value to substitute for each bound variable
    Values(Cow<'a, [TypeArg]>),
    /// An amount to add to the index of any free variable - that is,
    /// any free var, of index `i`, becomes the variable `(i +` this amount`)`
    AddToIndex(usize),
}

impl<'a> Substitution<'a> {
    pub(crate) fn new(argvals: impl Into<Cow<'a, [TypeArg]>>, reg: &'a ExtensionRegistry) -> Self {
        Self {
            leave_lowest: 0,
            mapping: Mapping::Values(argvals.into()),
            reg,
        }
    }

    pub(crate) fn extension_registry(&self) -> &ExtensionRegistry {
        self.reg
    }

    pub(crate) fn apply_to_var(&self, idx: usize, decl: &TypeParam) -> TypeArg {
        if idx < self.leave_lowest {
            return TypeArg::new_var_use(idx, decl.clone());
        }
        match &self.mapping {
            Mapping::Values(args) => args
                .get(idx - self.leave_lowest)
                .expect("Unexpected free type var")
                .clone(),
            Mapping::AddToIndex(diff) => TypeArg::new_var_use(idx + diff, decl.clone()),
        }
    }

    pub(super) fn apply_to_type_var(&self, idx: usize, bound: TypeBound) -> Type {
        let TypeArg::Type {ty} = self.apply_to_var(idx, &TypeParam::Type(bound))
           else {panic!("Var of kind 'type' did not produce a Type")};
        ty
    }

    pub(super) fn enter_scope(&self, new_vars: usize) -> Self {
        Self {
            leave_lowest: self.leave_lowest + new_vars,
            mapping: match &self.mapping {
                Mapping::Values(vals) => Mapping::Values({
                    // We need to renumber the RHS `vals` to avoid the newly-bound variables
                    let renum = Substitution {
                        leave_lowest: 0,
                        mapping: Mapping::AddToIndex(new_vars),
                        reg: self.reg,
                    };
                    vals.iter().map(|v| v.substitute(&renum)).collect()
                }),
                Mapping::AddToIndex(i) => Mapping::AddToIndex(*i),
            },
            reg: self.reg,
        }
    }

    pub(super) fn apply_row(&self, row: &TypeRow) -> TypeRow {
        let res = row
            .iter()
            .map(|t| t.substitute(self))
            .collect::<Vec<_>>()
            .into();
        res
    }
}
