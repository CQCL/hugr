use hugr_core::ops::constant::{CustomConst, Sum};
use hugr_core::types::{Signature, SumType, TypeRow};
use thiserror::Error;

use std::collections::HashMap;
use std::sync::Arc;

use hugr_core::{
    extension::SignatureError,
    hugr::hugrmut::HugrMut,
    ops::{
        AliasDefn, Call, CallIndirect, Case, Conditional, Const, DataflowBlock, ExitBlock,
        ExtensionOp, FuncDecl, FuncDefn, Input, LoadConstant, LoadFunction, OpType, Output, Tag,
        TailLoop, Value, CFG, DFG,
    },
    types::{CustomType, FuncValueType, Type, TypeArg, TypeBound, TypeEnum, TypeRV, TypeRowRV},
    Node,
};

#[derive(Clone)]
pub struct LowerTypes {
    type_fn: Arc<dyn Fn(&CustomType) -> Option<Type>>,
    type_map: HashMap<CustomType, Type>,
    copy_dup: HashMap<CustomType, (OpType, OpType)>, // TODO what about e.g. arrays that have gone from copyable to linear because their elements have?!
    //op_map: HashMap<OpType, OpType>
    //        1. is input op always a single OpType, or a schema/predicate?
    //        2. output might not be an op - might be a node with children
    //        3. do we need checking BEFORE reparametrization as well as after? (after only if not reparametrized?)
    #[allow(unused)]
    const_fn: Arc<dyn Fn(&dyn CustomConst) -> Option<Value>>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ChangeTypeError {
    #[error(transparent)]
    SignatureError(#[from] SignatureError),
}

impl LowerTypes {
    pub fn lower_type(&mut self, src: CustomType, dest: Type) {
        if src.bound() == TypeBound::Copyable && !dest.copyable() {
            // Of course we could try, and fail only if we encounter outports that are not singly-used!
            panic!("Cannot lower copyable type to linear without copy/dup - use lower_type_linearize instead");
        }
        self.type_map.insert(src, dest);
    }

    pub fn lower_type_linearize(&mut self, src: CustomType, dest: Type, copy: OpType, dup: OpType) {
        self.type_map.insert(src.clone(), dest);
        self.copy_dup.insert(src, (copy, dup));
    }

    pub fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<bool, ChangeTypeError> {
        let mut changed = false;
        for n in hugr.nodes().collect::<Vec<_>>() {
            changed |= self.change_node(hugr, n)?;
        }
        Ok(changed)
    }

    fn change_node(&self, hugr: &mut impl HugrMut, n: Node) -> Result<bool, ChangeTypeError> {
        match hugr.optype_mut(n) {
            OpType::FuncDefn(FuncDefn { signature, .. })
            | OpType::FuncDecl(FuncDecl { signature, .. }) => self.change_sig(signature.body_mut()),
            OpType::LoadConstant(LoadConstant { datatype: ty })
            | OpType::AliasDefn(AliasDefn { definition: ty, .. }) => self.change_type(ty),

            OpType::ExitBlock(ExitBlock { cfg_outputs: types })
            | OpType::Input(Input { types })
            | OpType::Output(Output { types }) => self.change_type_row(types),
            OpType::LoadFunction(LoadFunction {
                func_sig,
                type_args,
                instantiation,
            })
            | OpType::Call(Call {
                func_sig,
                type_args,
                instantiation,
            }) => {
                let change =
                    self.change_sig(func_sig.body_mut())? | self.change_type_args(type_args)?;
                let new_inst = func_sig
                    .instantiate(&type_args)
                    .map_err(ChangeTypeError::SignatureError)?;
                *instantiation = new_inst;
                Ok(change)
            }
            OpType::Case(Case { signature })
            | OpType::CFG(CFG { signature })
            | OpType::DFG(DFG { signature })
            | OpType::CallIndirect(CallIndirect { signature }) => self.change_sig(signature),
            OpType::Tag(Tag { variants, .. }) => {
                let mut ch = false;
                for v in variants.iter_mut() {
                    ch |= self.change_type_row(v)?;
                }
                Ok(ch)
            }
            OpType::Conditional(Conditional {
                other_inputs: row1,
                outputs: row2,
                sum_rows,
                ..
            })
            | OpType::DataflowBlock(DataflowBlock {
                inputs: row1,
                other_outputs: row2,
                sum_rows,
                ..
            }) => {
                let mut ch = self.change_type_row(row1)? | self.change_type_row(row2)?;
                for r in sum_rows.iter_mut() {
                    ch |= self.change_type_row(r)?;
                }
                Ok(ch)
            }
            OpType::TailLoop(TailLoop {
                just_inputs,
                just_outputs,
                rest,
                ..
            }) => Ok(self.change_type_row(just_inputs)?
                | self.change_type_row(just_outputs)?
                | self.change_type_row(rest)?),

            OpType::Const(Const { value, .. }) => self.change_value(value),
            OpType::ExtensionOp(ext_op) => {
                let def = ext_op.def_arc();
                let mut args = ext_op.args().to_vec();
                let change = self.change_type_args(args.as_mut_slice())?;
                if change {
                    *ext_op = ExtensionOp::new(def.clone(), args)?;
                }
                // let params = ext_op_params[node].to_owned();
                todo!("Also check whether we should lower op")
            }
            OpType::OpaqueOp(_) => panic!("OpaqueOp should not be in a Hugr"),

            OpType::AliasDecl(_) | OpType::Module(_) => Ok(false),
            _ => todo!(),
        }
    }

    fn change_value(&self, value: &mut Value) -> Result<bool, ChangeTypeError> {
        match value {
            Value::Sum(Sum {
                values, sum_type, ..
            }) => {
                let mut any_change = false;
                for value in values {
                    any_change |= self.change_value(value)?;
                }
                any_change |= self.change_sumtype(sum_type)?;
                Ok(any_change)
            }
            Value::Extension { e } => {
                if let Some(new_const) = self.subst_custom_const(e.value())? {
                    *value = new_const;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            Value::Function { hugr } => self.run_no_validate(&mut **hugr),
        }
    }

    fn subst_custom_const(&self, _cst: &dyn CustomConst) -> Result<Option<Value>, ChangeTypeError> {
        todo!()
    }

    fn change_type_arg(&self, arg: &mut TypeArg) -> Result<bool, ChangeTypeError> {
        match arg {
            TypeArg::Type { ty } => self.change_type(ty),
            TypeArg::BoundedNat { .. }
            | TypeArg::String { .. }
            | TypeArg::Extensions { .. }
            | TypeArg::Variable { .. } => Ok(false),
            TypeArg::Sequence { elems } => self.change_type_args(elems),
            _ => todo!(),
        }
    }

    fn change_type(&self, ty: &mut Type) -> Result<bool, ChangeTypeError> {
        // There is no as_type_enum_mut because mutation could invalidate the cache of TypeBound
        let new_ty = match ty.as_type_enum() {
            TypeEnum::Alias(_) | TypeEnum::RowVar(_) | TypeEnum::Variable(..) => return Ok(false),
            TypeEnum::Extension(ct) => {
                if let Some(t) = self.subst_custom_type(ct)? {
                    t
                } else {
                    return Ok(false);
                }
            }
            TypeEnum::Function(fty) => {
                if let Some(fty) = self.subst_fty(&**fty)? {
                    Type::new_function(fty)
                } else {
                    return Ok(false);
                }
            }
            TypeEnum::Sum(s) => {
                let mut st = s.clone();
                if !self.change_sumtype(&mut st)? {
                    return Ok(false);
                };
                st.into()
            }
        };
        *ty = new_ty;
        Ok(true)
    }

    fn change_tyrv(&self, ty: &mut TypeRV) -> Result<bool, ChangeTypeError> {
        // There is no as_type_enum_mut because mutation could invalidate the cache of TypeBound
        let new_ty = match ty.as_type_enum() {
            TypeEnum::Alias(_) | TypeEnum::RowVar(_) | TypeEnum::Variable(..) => return Ok(false),
            TypeEnum::Extension(ct) => self.subst_custom_type(ct)?.map(TypeRV::from),
            TypeEnum::Function(fty) => self.subst_fty(&**fty)?.map(TypeRV::new_function),
            TypeEnum::Sum(s) => {
                let mut st = s.clone();
                self.change_sumtype(&mut st)?.then(|| TypeRV::from(st))
            }
        };
        if let Some(new_ty) = new_ty {
            *ty = new_ty;
            return Ok(true);
        };
        return Ok(false);
    }

    fn subst_custom_type(&self, ct: &CustomType) -> Result<Option<Type>, ChangeTypeError> {
        let mut nargs = ct.args().to_vec();
        let ch = self.change_type_args(&mut nargs)?;
        let ext = ct.extension_ref().upgrade().unwrap();
        let ct = ext.get_type(ct.name()).unwrap().instantiate(nargs)?;
        Ok(if let Some(r) = (self.type_fn)(&ct) {
            Some(r)
        } else if let Some(r) = self.type_map.get(&ct) {
            Some(r.clone())
        } else if ch {
            Some(ct.into())
        } else {
            None
        })
    }

    fn change_sig(&self, ft: &mut Signature) -> Result<bool, ChangeTypeError> {
        // TODO runtime_reqs?
        Ok(self.change_type_row(&mut ft.input)? | self.change_type_row(&mut ft.output)?)
    }

    fn subst_fty(&self, fty: &FuncValueType) -> Result<Option<FuncValueType>, ChangeTypeError> {
        let mut fty = fty.clone();
        if !self.change_type_row_rv(&mut fty.input)? & !self.change_type_row_rv(&mut fty.output)? {
            return Ok(None);
        }
        // TODO what about runtime_req if we are changing ops??
        Ok(Some(fty))
    }

    fn change_sumtype(&self, st: &mut SumType) -> Result<bool, ChangeTypeError> {
        Ok(match st {
            SumType::Unit { .. } => false,
            SumType::General { rows } => {
                let mut ch = false;
                for row in rows.iter_mut() {
                    ch |= self.change_type_row_rv(row)?
                }
                ch
            }
            _ => todo!("Unexpected SumType {st:?}"),
        })
    }

    fn change_type_args(&self, tas: &mut [TypeArg]) -> Result<bool, ChangeTypeError> {
        let mut ch = false;
        for ta in tas.iter_mut() {
            ch |= self.change_type_arg(ta)?;
        }
        Ok(ch)
    }

    fn change_type_row(&self, row: &mut TypeRow) -> Result<bool, ChangeTypeError> {
        let mut ch = false;
        for t in row.iter_mut() {
            ch |= self.change_type(t)?;
        }
        Ok(ch)
    }

    fn change_type_row_rv(&self, row: &mut TypeRowRV) -> Result<bool, ChangeTypeError> {
        let mut ch = false;
        for t in row.iter_mut() {
            ch |= self.change_tyrv(t)?;
        }
        Ok(ch)
    }
}
