use std::collections::HashMap;

use smol_str::SmolStr;

use super::{ExtensionId, ExtensionSet, TypeParametrised};
use crate::types::{least_upper_bound, TypeBound};
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
    // Should this be min_bound? It might be narrower after substitution
    //bound: TypeBound,
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

    fn bound(&self, exts: &HashMap<SmolStr, Extension>, binders: &Vec<TypeParam>) -> TypeBound {
        match self {
            TypeTemplate::Extension(ctt) => ctt.bound(exts, binders),
            TypeTemplate::Alias(decl) => decl.bound,
            TypeTemplate::Function(_, _, _) => TypeBound::Copyable,
            TypeTemplate::Tuple(elems) => {
                least_upper_bound(elems.iter().map(|e| e.bound(exts, binders)))
            }
            TypeTemplate::Sum(elems) => {
                least_upper_bound(elems.iter().map(|e| e.bound(exts, binders)))
            }
            TypeTemplate::TypeVar(VariableRef(i)) => match binders.get(*i) {
                Some(TypeParam::Type(bound)) => bound.clone(),
                _ => panic!("Variable is not a Type, should not occur inside a Type"),
            },
        }
    }

    fn validate(
        &self,
        exts: &HashMap<SmolStr, Extension>,
        binders: &Vec<TypeParam>,
    ) -> Result<(), ()> {
        match self {
            TypeTemplate::Extension(ctt) => ctt.validate(exts, binders),
            TypeTemplate::Alias(_) => Ok(()),
            TypeTemplate::Function(ins, outs, es) => {
                ins.iter().try_for_each(|tt| tt.validate(exts, binders))?;
                outs.iter().try_for_each(|tt| tt.validate(exts, binders))?;
                if let ExtensionSetTemplate::TypeVar(VariableRef(i)) = es {
                    if binders.get(*i) == Some(&TypeParam::Extensions) {
                        return Ok(());
                    }
                }
                Err(())
            }
            TypeTemplate::Tuple(elems) => {
                elems.iter().try_for_each(|tt| tt.validate(exts, binders))
            }
            TypeTemplate::Sum(elems) => elems.iter().try_for_each(|tt| tt.validate(exts, binders)),
            TypeTemplate::TypeVar(VariableRef(i)) => {
                if let Some(TypeParam::Type(_)) = binders.get(*i) {
                    Ok(())
                } else {
                    Err(())
                }
            }
        }
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

    fn bound(&self, exts: &HashMap<SmolStr, Extension>, binders: &Vec<TypeParam>) -> TypeBound {
        if self.validate(exts, binders).is_err() {
            // Raise err here, don't return TypeBound
        };
        let typdef = exts
            .get(&self.extension)
            .unwrap()
            .get_type(self.id.as_str())
            .unwrap();

        match &typdef.bound {
            super::TypeDefBound::Explicit(b) => *b,
            super::TypeDefBound::FromParams(indices) => least_upper_bound(indices.iter().map(|idx| match self.args.get(*idx) {
                Some(TypeArgTemplate::Type(tt)) => tt.bound(exts, binders),
                Some(TypeArgTemplate::TypeVar(VariableRef(i))) => match binders.get(*i) {
                    Some(TypeParam::Type(bound)) => *bound,
                    _ => panic!("Trying to instantiate CustomType with a variable that does not hold a Type")
                }
                _ => panic!("TypeDef's bound definition refers to non-type") // we've validated, so the TypeArgTemplate must fit the param

            }))
        }
    }

    fn validate(
        &self,
        exts: &HashMap<SmolStr, Extension>,
        binders: &Vec<TypeParam>,
    ) -> Result<(), ()> {
        let params = exts
            .get(&self.extension)
            .ok_or(())?
            .get_type(self.id.as_str())
            .ok_or(())?
            .params();
        if params.len() == self.args.len()
            && self
                .args
                .iter()
                .zip(params)
                .all(|(arg, param)| arg.will_fit(param, exts, binders))
        {
            Ok(())
        } else {
            Err(())
        }
    }
}

fn check_type_param_fits(tps: (&TypeParam, &TypeParam)) -> bool {
    match tps {
        (TypeParam::Type(hbound), TypeParam::Type(pbound)) => hbound.contains(*pbound),
        (TypeParam::USize, TypeParam::USize) => true,
        (TypeParam::Opaque(t1), TypeParam::Opaque(t2)) => t1 == t2,
        (TypeParam::List(hs), TypeParam::List(ps)) => check_type_param_fits((hs, ps)),
        (TypeParam::Tuple(hs), TypeParam::Tuple(ps)) => {
            hs.len() == ps.len() && hs.iter().zip(ps).all(check_type_param_fits)
        }
        (TypeParam::Extensions, TypeParam::Extensions) => true,
        _ => false,
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

    fn will_fit(
        &self,
        into: &TypeParam,
        exts: &HashMap<SmolStr, Extension>,
        binders: &Vec<TypeParam>,
    ) -> bool {
        if let TypeArgTemplate::TypeVar(VariableRef(i)) = self {
            return match binders.get(*i) {
                Some(vdecl) => check_type_param_fits((into, vdecl)),
                None => false, // Error: undeclared variable
            };
        };
        // self is not a TypeVar!
        match (into, self) {
            (TypeParam::Type(pbound), TypeArgTemplate::Type(tt)) => {
                pbound.contains(tt.bound(exts, binders))
            }
            (TypeParam::USize, TypeArgTemplate::USize(_)) => true,
            (TypeParam::Opaque(custy), TypeArgTemplate::Opaque(cusarg)) => &cusarg.typ == custy,
            (TypeParam::List(ety), TypeArgTemplate::Sequence(elems)) => {
                elems.iter().all(|e| e.will_fit(ety, exts, binders))
            }
            (TypeParam::Tuple(etys), TypeArgTemplate::Sequence(elems)) => {
                etys.len() == elems.len()
                    && elems
                        .iter()
                        .zip(etys)
                        .all(|(val, ty)| val.will_fit(ty, exts, binders))
            }
            (TypeParam::Extensions, TypeArgTemplate::Extensions(_)) => true,
            _ => false,
        }
    }

    // TODO ensure we make all these checks in will_fit
    /*fn validate(
        &self,
        exts: &HashMap<SmolStr, Extension>,
        binders: &Vec<TypeParam>,
    ) -> Result<(), ()> {
        match self {
            TypeArgTemplate::Type(tt) => tt.validate(exts, binders),
            TypeArgTemplate::USize(_) => Ok(()),
            TypeArgTemplate::Opaque(_) => Ok(()),
            TypeArgTemplate::Sequence(elems) => {
                elems.iter().try_for_each(|tat| tat.validate(exts, binders))
            }
            TypeArgTemplate::Extensions(_) => Ok(()),
            TypeArgTemplate::TypeVar(VariableRef(i)) => {
                if binders.get(*i).is_some() {
                    Ok(())
                } else {
                    Err(())
                }
            }
        }
    }*/
}
