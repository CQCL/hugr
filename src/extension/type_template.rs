//! Type schemes that can be turned into [FunctionType]s by applying
//! them to [TypeArg]s

use std::collections::HashMap;

use smol_str::SmolStr;

use super::{ExtensionId, ExtensionSet, SignatureError, TypeDefBound, TypeParametrised};
use crate::types::type_param::{check_type_arg, TypeArgError};
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
    // TODO possibly allow union of ExtensionSet with any number of vars?
    Concrete(ExtensionSet),
    TypeVar(VariableRef),
}

/// TypeEnum with Prim inlined (as we need our own version of FunctionType)
/// and including variables
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
enum TypeTemplate {
    Extension(CustomTypeTemplate),
    Alias(AliasDecl),
    Function(FunctionTypeTemplate),
    Tuple(Vec<TypeTemplate>),
    Sum(Vec<TypeTemplate>),
    TypeVar(VariableRef),
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
struct FunctionTypeTemplate {
    input: Vec<TypeTemplate>,
    output: Vec<TypeTemplate>,
    extension_reqs: ExtensionSetTemplate,
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

/// Representation of the type scheme for an Op with declared TypeParams.
/// This should be the result of YAML parsing.
pub struct SignatureTemplate(pub Vec<TypeParam>, FunctionTypeTemplate);

impl TypeTemplate {
    fn substitute(&self, exts: &HashMap<SmolStr, Extension>, vars: &Vec<TypeArg>) -> Type {
        match self {
            TypeTemplate::Extension(ctt) => Type::new_extension(ctt.substitute(exts, vars)),
            TypeTemplate::Alias(decl) => Type::new_alias(decl.clone()),
            TypeTemplate::Function(ftt) => Type::new_function(ftt.substitute(exts, vars)),
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
            TypeTemplate::Function(ftt) => {
                if !bound.contains(TypeBound::Copyable) {
                    return Err(());
                };
                ftt.validate(exts, binders)?;
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
                    _ => return Err(()),
                }
                return Err(());
            }
        };
        Ok(())
    }
}

impl FunctionTypeTemplate {
    fn substitute(&self, exts: &HashMap<SmolStr, Extension>, vars: &Vec<TypeArg>) -> FunctionType {
        FunctionType {
            input: TypeRow::from(
                self.input
                    .iter()
                    .map(|tt| tt.substitute(exts, vars))
                    .collect::<Vec<_>>(),
            ),
            output: TypeRow::from(
                self.output
                    .iter()
                    .map(|tt| tt.substitute(exts, vars))
                    .collect::<Vec<_>>(),
            ),
            extension_reqs: match &self.extension_reqs {
                ExtensionSetTemplate::Concrete(c) => c.clone(),
                ExtensionSetTemplate::TypeVar(VariableRef(i)) => {
                    let TypeArg::Extensions(e) = vars.get(*i).unwrap()
                        else {panic!("Variable was not Extension");};
                    e.clone()
                }
            },
        }
    }

    fn validate(
        &self,
        exts: &HashMap<SmolStr, Extension>,
        binders: &Vec<TypeParam>,
    ) -> Result<(), ()> {
        self.input
            .iter()
            .chain(&self.output)
            .try_for_each(|tt| tt.validate(exts, binders, TypeBound::Any))?;
        if let ExtensionSetTemplate::TypeVar(VariableRef(i)) = self.extension_reqs {
            if binders.get(i) != Some(&TypeParam::Extensions) {
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

impl SignatureTemplate {
    /// Validates the definition, i.e. that every set of arguments passing [SignatureTemplate::check_args]
    /// will produce a valid type in [SignatureTemplate::instantiate_concrete]
    pub fn validate(&self, exts: &HashMap<SmolStr, Extension>) -> Result<(), ()> {
        self.1.validate(exts, &self.0)
    }

    /// Check some [TypeArg]s are legal arguments.
    // Copied from [TypeParametrised::check_args_impl] - either perhaps we can implement [TypeParametrised], or
    /// If this is merely hidden inside an [OpDef], the [OpDef]'s `check_args` will do this.
    ///
    /// [OpDef]: super::OpDef
    pub fn check_args(&self, args: &[TypeArg]) -> Result<(), SignatureError> {
        let binders = &self.0;
        if args.len() != binders.len() {
            return Err(SignatureError::TypeArgMismatch(
                TypeArgError::WrongNumberArgs(args.len(), binders.len()),
            ));
        }
        for (a, p) in args.iter().zip(binders.iter()) {
            check_type_arg(a, p).map_err(SignatureError::TypeArgMismatch)?;
        }
        Ok(())
    }

    /// Create a concrete signature from some type args.
    /// Call [SignatureTemplate::validate] first (once only)
    /// and [SignatureTemplate::check_args] with the arguments
    /// - if those return Ok, this should be guaranteed to succeed.
    pub fn instantiate_concrete(
        &self,
        exts: &HashMap<SmolStr, Extension>,
        vars: &Vec<TypeArg>,
    ) -> FunctionType {
        self.1.substitute(exts, vars)
    }
}
