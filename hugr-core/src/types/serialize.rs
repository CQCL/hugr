use std::sync::Arc;

use ordered_float::OrderedFloat;

use super::{FuncValueType, MaybeRV, RowVariable, SumType, TypeBase, TypeBound, TypeEnum};

use super::custom::CustomType;

use crate::extension::SignatureError;
use crate::extension::prelude::{qb_t, usize_t};
use crate::ops::AliasDecl;
use crate::types::type_param::{TermVar, UpperBound};
use crate::types::{Term, Type};

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(crate) enum SerSimpleType {
    Q,
    I,
    G(Box<FuncValueType>),
    Sum(SumType),
    Opaque(CustomType),
    Alias(AliasDecl),
    V { i: usize, b: TypeBound },
    R { i: usize, b: TypeBound },
}

impl<RV: MaybeRV> From<TypeBase<RV>> for SerSimpleType {
    fn from(value: TypeBase<RV>) -> Self {
        if value == qb_t() {
            return SerSimpleType::Q;
        }
        if value == usize_t() {
            return SerSimpleType::I;
        }
        match value.0 {
            TypeEnum::Extension(o) => SerSimpleType::Opaque(o),
            TypeEnum::Alias(a) => SerSimpleType::Alias(a),
            TypeEnum::Function(sig) => SerSimpleType::G(sig),
            TypeEnum::Variable(i, b) => SerSimpleType::V { i, b },
            TypeEnum::RowVar(rv) => {
                let RowVariable(idx, bound) = rv.as_rv();
                SerSimpleType::R { i: *idx, b: *bound }
            }
            TypeEnum::Sum(st) => SerSimpleType::Sum(st),
        }
    }
}

impl<RV: MaybeRV> TryFrom<SerSimpleType> for TypeBase<RV> {
    type Error = SignatureError;
    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        Ok(match value {
            SerSimpleType::Q => qb_t().into_(),
            SerSimpleType::I => usize_t().into_(),
            SerSimpleType::G(sig) => TypeBase::new_function(*sig),
            SerSimpleType::Sum(st) => st.into(),
            SerSimpleType::Opaque(o) => TypeBase::new_extension(o),
            SerSimpleType::Alias(a) => TypeBase::new_alias(a),
            SerSimpleType::V { i, b } => TypeBase::new_var_use(i, b),
            // We can't use new_row_var because that returns TypeRV not TypeBase<RV>.
            SerSimpleType::R { i, b } => TypeBase::new(TypeEnum::RowVar(
                RV::try_from_rv(RowVariable(i, b))
                    .map_err(|var| SignatureError::RowVarWhereTypeExpected { var })?,
            )),
        })
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
#[serde(tag = "tp")]
pub(super) enum TypeParamSer {
    Type { b: TypeBound },
    BoundedNat { bound: UpperBound },
    String,
    Bytes,
    Float,
    StaticType,
    List { param: Box<Term> },
    Tuple { params: ArrayOrTermSer },
    ConstType { ty: Type },
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
#[serde(tag = "tya")]
pub(super) enum TypeArgSer {
    Type {
        ty: Type,
    },
    BoundedNat {
        n: u64,
    },
    String {
        arg: String,
    },
    Bytes {
        #[serde(with = "base64")]
        value: Arc<[u8]>,
    },
    Float {
        value: OrderedFloat<f64>,
    },
    List {
        elems: Vec<Term>,
    },
    ListConcat {
        lists: Vec<Term>,
    },
    Tuple {
        elems: Vec<Term>,
    },
    TupleConcat {
        tuples: Vec<Term>,
    },
    Variable {
        #[serde(flatten)]
        v: TermVar,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub(super) enum TermSer {
    TypeArg(TypeArgSer),
    TypeParam(TypeParamSer),
}

impl From<Term> for TermSer {
    fn from(value: Term) -> Self {
        match value {
            Term::RuntimeType(b) => TermSer::TypeParam(TypeParamSer::Type { b }),
            Term::StaticType => TermSer::TypeParam(TypeParamSer::StaticType),
            Term::BoundedNatType(bound) => TermSer::TypeParam(TypeParamSer::BoundedNat { bound }),
            Term::StringType => TermSer::TypeParam(TypeParamSer::String),
            Term::BytesType => TermSer::TypeParam(TypeParamSer::Bytes),
            Term::FloatType => TermSer::TypeParam(TypeParamSer::Float),
            Term::ListType(param) => TermSer::TypeParam(TypeParamSer::List { param }),
            Term::ConstType(ty) => TermSer::TypeParam(TypeParamSer::ConstType { ty: *ty }),
            Term::Runtime(ty) => TermSer::TypeArg(TypeArgSer::Type { ty }),
            Term::TupleType(params) => TermSer::TypeParam(TypeParamSer::Tuple {
                params: (*params).into(),
            }),
            Term::BoundedNat(n) => TermSer::TypeArg(TypeArgSer::BoundedNat { n }),
            Term::String(arg) => TermSer::TypeArg(TypeArgSer::String { arg }),
            Term::Bytes(value) => TermSer::TypeArg(TypeArgSer::Bytes { value }),
            Term::Float(value) => TermSer::TypeArg(TypeArgSer::Float { value }),
            Term::List(elems) => TermSer::TypeArg(TypeArgSer::List { elems }),
            Term::Tuple(elems) => TermSer::TypeArg(TypeArgSer::Tuple { elems }),
            Term::Variable(v) => TermSer::TypeArg(TypeArgSer::Variable { v }),
            Term::ListConcat(lists) => TermSer::TypeArg(TypeArgSer::ListConcat { lists }),
            Term::TupleConcat(tuples) => TermSer::TypeArg(TypeArgSer::TupleConcat { tuples }),
        }
    }
}

impl From<TermSer> for Term {
    fn from(value: TermSer) -> Self {
        match value {
            TermSer::TypeParam(param) => match param {
                TypeParamSer::Type { b } => Term::RuntimeType(b),
                TypeParamSer::StaticType => Term::StaticType,
                TypeParamSer::BoundedNat { bound } => Term::BoundedNatType(bound),
                TypeParamSer::String => Term::StringType,
                TypeParamSer::Bytes => Term::BytesType,
                TypeParamSer::Float => Term::FloatType,
                TypeParamSer::List { param } => Term::ListType(param),
                TypeParamSer::Tuple { params } => Term::TupleType(Box::new(params.into())),
                TypeParamSer::ConstType { ty } => Term::ConstType(Box::new(ty)),
            },
            TermSer::TypeArg(arg) => match arg {
                TypeArgSer::Type { ty } => Term::Runtime(ty),
                TypeArgSer::BoundedNat { n } => Term::BoundedNat(n),
                TypeArgSer::String { arg } => Term::String(arg),
                TypeArgSer::Bytes { value } => Term::Bytes(value),
                TypeArgSer::Float { value } => Term::Float(value),
                TypeArgSer::List { elems } => Term::List(elems),
                TypeArgSer::Tuple { elems } => Term::Tuple(elems),
                TypeArgSer::Variable { v } => Term::Variable(v),
                TypeArgSer::ListConcat { lists } => Term::ListConcat(lists),
                TypeArgSer::TupleConcat { tuples } => Term::TupleConcat(tuples),
            },
        }
    }
}

/// Helper type that serialises lists as JSON arrays for compatibility.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub(super) enum ArrayOrTermSer {
    Array(Vec<Term>),
    Term(Box<Term>), // TODO JSON Schema does not really support this yet
}

impl From<ArrayOrTermSer> for Term {
    fn from(value: ArrayOrTermSer) -> Self {
        match value {
            ArrayOrTermSer::Array(terms) => Term::new_list(terms),
            ArrayOrTermSer::Term(term) => *term,
        }
    }
}

impl From<Term> for ArrayOrTermSer {
    fn from(term: Term) -> Self {
        match term {
            Term::List(terms) => ArrayOrTermSer::Array(terms),
            term => ArrayOrTermSer::Term(Box::new(term)),
        }
    }
}

/// Helper for to serialize and deserialize the byte string in [`TypeArg::Bytes`] via base64.
mod base64 {
    use std::sync::Arc;

    use base64::Engine as _;
    use base64::prelude::BASE64_STANDARD;
    use serde::{Deserialize, Serialize};
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Arc<[u8]>, s: S) -> Result<S::Ok, S::Error> {
        let base64 = BASE64_STANDARD.encode(v);
        base64.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Arc<[u8]>, D::Error> {
        let base64 = String::deserialize(d)?;
        BASE64_STANDARD
            .decode(base64.as_bytes())
            .map(|v| v.into())
            .map_err(serde::de::Error::custom)
    }
}
