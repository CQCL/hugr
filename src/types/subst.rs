use crate::extension::ExtensionRegistry;

use super::type_param::{TypeArg, TypeParam};
use super::{Type, TypeBound, TypeRow};

#[derive(Clone, Debug)]
pub(crate) struct Substitution {
    /// The number of variables bound more closely than those being substituted,
    /// so these should be untouched by the substitution
    leave_lowest: usize,
    /// What to do to variables that are affected
    mapping: Mapping,
}

#[derive(Clone, Debug)]
enum Mapping {
    /// An explicit value to substitute for each bound variable
    Values(Vec<TypeArg>),
    /// An amount to add to the index of any free variable
    /// (That is: any free var, of index `i`, becomes the variable `i +` this amount)
    AddToIndex(usize),
}

impl Substitution {
    pub(crate) fn new(args: impl Into<Vec<TypeArg>>) -> Self {
        Self {
            leave_lowest: 0,
            mapping: Mapping::Values(args.into()),
        }
    }

    pub(crate) fn get(&self, idx: usize, decl: &TypeParam) -> TypeArg {
        if idx < self.leave_lowest {
            return TypeArg::use_var(idx, decl.clone());
        }
        match &self.mapping {
            Mapping::Values(args) => args
                .get(idx - self.leave_lowest)
                .expect("Unexpected free type var")
                .clone(),
            Mapping::AddToIndex(diff) => TypeArg::use_var(idx + diff, decl.clone()),
        }
    }

    pub(super) fn get_type(&self, idx: usize, bound: TypeBound) -> Type {
        let TypeArg::Type {ty} = self.get(idx, &TypeParam::Type(bound))
           else {panic!("Var of kind 'type' did not produce a Type")};
        ty
    }

    // A bit unfortunate to need a new extension registry here...move into Substitution?
    pub(super) fn enter_scope(&self, new_vars: usize, exts: &ExtensionRegistry) -> Self {
        Self {
            leave_lowest: self.leave_lowest + new_vars,
            mapping: match &self.mapping {
                Mapping::Values(vals) => Mapping::Values({
                    // We need to renumber the RHS `vals` to avoid the newly-bound variables
                    let renum = Substitution {
                        leave_lowest: 0,
                        mapping: Mapping::AddToIndex(new_vars),
                    };
                    vals.iter().map(|v| v.substitute(exts, &renum)).collect()
                }),
                Mapping::AddToIndex(i) => Mapping::AddToIndex(*i),
            },
        }
    }

    pub(super) fn apply_row(&self, row: &TypeRow, exts: &ExtensionRegistry) -> TypeRow {
        let res = row
            .iter()
            .map(|t| t.substitute(exts, self))
            .collect::<Vec<_>>()
            .into();
        res
    }
}
