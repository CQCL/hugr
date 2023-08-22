use std::collections::HashMap;

use smol_str::SmolStr;

use super::{ExtensionId, ExtensionSet, TypeParametrised};
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
        if params.len() == binders.len() && binders.iter().zip(params).all(check_type_param_fits) {
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

    fn validate(
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
    }
}
