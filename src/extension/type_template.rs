use std::collections::HashMap;

use smol_str::SmolStr;

use super::{ExtensionId, ExtensionSet, TypeDefBound, TypeParametrised};
use crate::types::TypeBound;
use crate::{ops::AliasDecl, Extension};

use crate::types::{
    type_param::{CustomTypeArg, TypeArg, TypeParam},
    CustomType, FunctionType, Type, TypeRow,
};

/// deBruijn-indexed Variables
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
struct VariableRef(usize);

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
enum ExtensionSetTemplate {
    Concrete(ExtensionSet),
    TypeVar(VariableRef),
}

/// TypeEnum with Prim inlined (as we need our own version of FunctionType)
/// and including variables
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
enum TypeTemplate {
    Extension(CustomTypeTemplate),
    Alias(AliasDecl),
    Function(Vec<TypeTemplate>, Vec<TypeTemplate>, ExtensionSetTemplate),
    Tuple(Vec<TypeTemplate>),
    Sum(Vec<TypeTemplate>),
    TypeVar(VariableRef),
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
struct CustomTypeTemplate {
    extension: ExtensionId,
    id: SmolStr,
    args: Vec<TypeArgTemplate>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
enum TypeArgTemplate {
    Type(TypeTemplate),
    USize(u64),
    Opaque(CustomTypeArg),
    Sequence(Vec<TypeArgTemplate>),
    Extensions(ExtensionSet),
    TypeVar(VariableRef),
}

impl TypeTemplate {
    fn substitute(&self, exts: &HashMap<SmolStr, Extension>, vars: &Vec<TypeArg>) -> Type {
        match self {
            TypeTemplate::Extension(ctt) => Type::new_extension(ctt.substitute(exts, vars)),
            TypeTemplate::Alias(decl) => Type::new_alias(decl.clone()),
            TypeTemplate::Function(ins, outs, es) => Type::new_function(FunctionType {
                input: TypeRow::from(
                    ins.iter()
                        .map(|tt| tt.substitute(exts, vars))
                        .collect::<Vec<_>>(),
                ),
                output: TypeRow::from(
                    outs.iter()
                        .map(|tt| tt.substitute(exts, vars))
                        .collect::<Vec<_>>(),
                ),
                extension_reqs: match es {
                    ExtensionSetTemplate::Concrete(c) => c.clone(),
                    ExtensionSetTemplate::TypeVar(VariableRef(i)) => {
                        let TypeArg::Extensions(e) = vars.get(*i).unwrap()
                            else {panic!("Variable was not Extension");};
                        e.clone()
                    }
                },
            }),
            TypeTemplate::Tuple(elems) => Type::new_tuple(
                elems
                    .iter()
                    .map(|tt| tt.substitute(exts, vars))
                    .collect::<Vec<_>>(),
            ),
            TypeTemplate::Sum(elems) => Type::new_sum(
                elems
                    .iter()
                    .map(|tt| tt.substitute(exts, vars))
                    .collect::<Vec<_>>(),
            ),
            TypeTemplate::TypeVar(VariableRef(i)) => {
                let TypeArg::Type(t) = vars.get(*i).unwrap()
                else {panic!("Variable was not Type");};
                t.clone()
            }
        }
    }

    fn validate(
        &self,
        exts: &HashMap<SmolStr, Extension>,
        binders: &Vec<TypeParam>,
        bound: TypeBound,
    ) -> Result<(), ()> {
        match self {
            TypeTemplate::Extension(ctt) => ctt.validate(exts, binders, bound)?,
            TypeTemplate::Alias(decl) => {
                if !bound.contains(decl.bound) {
                    return Err(());
                };
            }
            TypeTemplate::Function(ins, outs, es) => {
                if !bound.contains(TypeBound::Copyable) {
                    return Err(());
                };
                ins.iter()
                    .try_for_each(|tt| tt.validate(exts, binders, bound))?;
                outs.iter()
                    .try_for_each(|tt| tt.validate(exts, binders, bound))?;
                if let ExtensionSetTemplate::TypeVar(VariableRef(i)) = es {
                    if binders.get(*i) != Some(&TypeParam::Extensions) {
                        return Err(());
                    }
                };
            }
            TypeTemplate::Tuple(elems) => elems
                .iter()
                .try_for_each(|tt| tt.validate(exts, binders, bound))?,
            TypeTemplate::Sum(elems) => elems
                .iter()
                .try_for_each(|tt| tt.validate(exts, binders, bound))?,
            TypeTemplate::TypeVar(VariableRef(i)) => {
                match binders.get(*i) {
                    Some(TypeParam::Type(decl_bound)) if bound.contains(*decl_bound) => (),
                    _ => return Err(())
                }
                return Err(());
            }
        };
        Ok(())
    }
}

impl CustomTypeTemplate {
    fn substitute(&self, exts: &HashMap<SmolStr, Extension>, vars: &Vec<TypeArg>) -> CustomType {
        let typdef = exts
            .get(&self.extension)
            .unwrap()
            .get_type(self.id.as_str())
            .unwrap();

        typdef
            .instantiate_concrete(
                self.args
                    .iter()
                    .map(|tat| tat.substitute(exts, vars))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    }

    fn validate(
        &self,
        exts: &HashMap<SmolStr, Extension>,
        binders: &Vec<TypeParam>,
        bound: TypeBound,
    ) -> Result<(), ()> {
        let typdef = exts
            .get(&self.extension)
            .ok_or(())?
            .get_type(self.id.as_str())
            .ok_or(())?;
        // Check args fit the params.
        let params = typdef.params();
        if params.len() != self.args.len() {
            return Err(());
        };
        let mut params: Vec<_> = params.into();
        match &typdef.bound {
            TypeDefBound::Explicit(b) => {
                if !bound.contains(*b) {
                    return Err(());
                };
                // Just check args are valid for the params
            }
            TypeDefBound::FromParams(indices) => {
                for i in indices {
                    // Bound of the CustomType depends upon this index
                    let TypeParam::Type(b) = params[*i] else {
                        return Err(()) // Index says to compute CustomType bound from non-type parameter!
                    };
                    // so require the corresponding arg to meet the bound (intersect with existing bound on arg)
                    if b.contains(bound) {
                        params[*i] = TypeParam::Type(bound);
                    }
                }
            }
        };
        self.args
            .iter()
            .zip(params)
            .try_for_each(|(arg, param)| arg.validate(exts, binders, &param))
    }
}

fn check_type_param_fits(tps: (&TypeParam, &TypeParam)) -> Result<(), ()> {
    match tps {
        (TypeParam::Type(hbound), TypeParam::Type(pbound)) => {
            if hbound.contains(*pbound) {
                Ok(())
            } else {
                Err(())
            }
        }
        (TypeParam::USize, TypeParam::USize) => Ok(()),
        (TypeParam::Opaque(t1), TypeParam::Opaque(t2)) => {
            if t1 == t2 {
                Ok(())
            } else {
                Err(())
            }
        }
        (TypeParam::List(hs), TypeParam::List(ps)) => check_type_param_fits((hs, ps)),
        (TypeParam::Tuple(hs), TypeParam::Tuple(ps)) if hs.len() == ps.len() => {
            hs.iter().zip(ps).try_for_each(check_type_param_fits)
        }
        (TypeParam::Extensions, TypeParam::Extensions) => Ok(()),
        _ => Err(()),
    }
}

impl TypeArgTemplate {
    fn substitute(&self, exts: &HashMap<SmolStr, Extension>, vars: &Vec<TypeArg>) -> TypeArg {
        match self {
            TypeArgTemplate::Type(tt) => TypeArg::Type(tt.substitute(exts, vars)),
            TypeArgTemplate::USize(i) => TypeArg::USize(*i),
            TypeArgTemplate::Opaque(cust) => TypeArg::Opaque(cust.clone()),
            TypeArgTemplate::Sequence(elems) => {
                TypeArg::Sequence(elems.iter().map(|tat| tat.substitute(exts, vars)).collect())
            }
            TypeArgTemplate::Extensions(es) => TypeArg::Extensions(es.clone()),
            TypeArgTemplate::TypeVar(VariableRef(i)) => vars.get(*i).unwrap().clone(),
        }
    }

    fn validate(
        &self,
        exts: &HashMap<SmolStr, Extension>,
        binders: &Vec<TypeParam>,
        bound: &TypeParam,
    ) -> Result<(), ()> {
        if let TypeArgTemplate::TypeVar(VariableRef(i)) = self {
            return match binders.get(*i) {
                Some(vdecl) => check_type_param_fits((bound, vdecl)),
                None => Err(()), // Error: undeclared variable
            };
        };
        // self is not a TypeVar!
        match (bound, self) {
            (TypeParam::Type(bound), TypeArgTemplate::Type(tt)) => {
                tt.validate(exts, binders, *bound)
            }
            (TypeParam::USize, TypeArgTemplate::USize(_)) => Ok(()),
            (TypeParam::Opaque(custy), TypeArgTemplate::Opaque(cusarg)) if &cusarg.typ == custy => {
                Ok(())
            }
            (TypeParam::List(ety), TypeArgTemplate::Sequence(elems)) => elems
                .iter()
                .try_for_each(|e| e.validate(exts, binders, ety)),
            (TypeParam::Tuple(etys), TypeArgTemplate::Sequence(elems))
                if etys.len() == elems.len() =>
            {
                elems
                    .iter()
                    .zip(etys)
                    .try_for_each(|(val, ty)| val.validate(exts, binders, ty))
            }
            (TypeParam::Extensions, TypeArgTemplate::Extensions(_)) => Ok(()),
            _ => Err(()),
        }
    }
}
