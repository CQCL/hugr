use std::collections::HashMap;

use hugr_core::{hugr::hugrmut::HugrMut, ops::{ExtensionOp, OpType}, types::{CustomType, FuncValueType, Type, TypeArg, TypeBound, TypeEnum, TypeRV, TypeRowRV}};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LowerTypes {
    // TODO allow Fn() to cope with parametrized CustomTypes? What about Aliases?
    type_fn: Arc<dyn Fn(CustomType) -> Option<Type>>,
    type_map: HashMap<CustomType, Type>,
    copy_dup: HashMap<CustomType, (OpType, OpType)>, // TODO what about e.g. arrays that have gone from copyable to linear because their elements have?!
    //op_map: HashMap<OpType, OpType>
    //        1. is input op always a single OpType, or a schema/predicate?
    //        2. output might not be an op - might be a node with children
    //        3. do we need checking BEFORE reparametrization as well as after? (after only if not reparametrized?)
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

    pub fn run_no_validate(&self, h: &mut impl HugrMut) {
        for n in h.nodes().collect::<Vec<_>>() {
            let n_op = match h.get_optype(n) {
                OpType::ExtensionOp(eop) => {
                    // YEUCH eop.def() is &OpDef but we need the Arc
                    let ext = eop.def().extension().upgrade().unwrap(); 
                    let def = ext.get_op(eop.def().name()).unwrap();                                
                    ExtensionOp::new(def.clone(), self.subst_tas(eop.args())).unwrap() // TODO return error
                },
                OpType::Module(_) | OpType::AliasDecl(_) => todo!(),
                OpType::FuncDefn(func_defn) => todo!(),
                OpType::FuncDecl(func_decl) => todo!(),
                OpType::AliasDecl(alias_decl) => todo!(),
                OpType::AliasDefn(alias_defn) => todo!(),
                OpType::Const(_) => todo!(),
                OpType::Input(input) => todo!(),
                OpType::Output(output) => todo!(),
                OpType::Call(call) => todo!(),
                OpType::CallIndirect(call_indirect) => todo!(),
                OpType::LoadConstant(load_constant) => todo!(),
                OpType::LoadFunction(load_function) => todo!(),
                OpType::DFG(dfg) => todo!(),
                OpType::OpaqueOp(opaque_op) => todo!(),
                OpType::Tag(tag) => todo!(),
                OpType::DataflowBlock(dataflow_block) => todo!(),
                OpType::ExitBlock(exit_block) => todo!(),
                OpType::TailLoop(tail_loop) => todo!(),
                OpType::CFG(cfg) => todo!(),
                OpType::Conditional(conditional) => todo!(),
                OpType::Case(case) => todo!(),
                _ => todo!(),
            };
            h.replace_op(n, n_op).unwrap();
            // TODO now sort out outputs - insert copy/dup
        }
        pub fn change_type_hugr(mut hugr: impl HugrMut, change: &mut impl Changer, reg: &ExtensionRegistry) -> Result<()> {
            // let ext_op_params: HashMap<Node,Vec<TypeParam>> = hugr.nodes().filter(|&x| hugr.get_optype(x).is_extension_op()).map(|x| op_params(hugr, x)).collect();
        
            Ok(())
        }
    }


    fn subst_tas(&self, args: &[TypeArg]) -> Vec<TypeArg> {
        args.iter().map(|ta| self.subst_ta(ta)).collect()
    }

    fn subst_ta(&self, arg: &TypeArg) -> TypeArg {
        match arg {
            TypeArg::Type { ty } => TypeArg::Type {ty: self.subst_ty(ty)},
            TypeArg::BoundedNat { .. } |
            TypeArg::String { .. } |
            TypeArg::Extensions { .. } |
            TypeArg::Variable { .. } => arg.clone(), // Or panic on Variable?
            TypeArg::Sequence { elems } => TypeArg::Sequence { elems: self.subst_tas(elems) },
            _ => todo!(),
        }
    }

    fn subst_ty(&self, ty: &Type) -> Type {
        match ty.as_type_enum() {
            TypeEnum::Alias(_) | TypeEnum::RowVar(_) | TypeEnum::Variable(..) => ty.clone(),
            TypeEnum::Extension(ct) => {
                if let Some(r) = self.type_map.get(ct) {
                    return r.clone()
                }
                let ext = ct.extension_ref().upgrade().unwrap();
                let def = ext.get_type(ct.name()).unwrap();
                def.instantiate( self.subst_tas(ct.args())).unwrap().into() // TODO return error
            }
            TypeEnum::Function(fty) => Type::new_function(FuncValueType::new(
                self.subst_tys(&fty.input), self.subst_tys(&fty.output))),
            TypeEnum::Sum(s) => Type::new_sum(s.variants().map(|v| self.subst_tys(v)))
        }
    }
    
    fn subst_tys(&self, r: &TypeRowRV) -> TypeRowRV {
        r.iter().map(|t| {
            match t.clone().try_into_type() {
                Ok(t) => self.subst_ty(&t).into(),
                Err(rv) => {
                    // YEUCH Type::new(TypeEnum) is crate-private so:
                    let mut t=t.clone();
                    *(t.as_type_enum_mut()) = TypeEnum::RowVar(rv);
                    t
                }
            }
        }).collect::<Vec<TypeRV>>().into()
    }
}
