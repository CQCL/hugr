use super::Module;

pub trait View<'a>: Sized {
    type Id;

    fn view(module: &'a Module<'a>, id: &'a Self::Id) -> Option<Self>;
}
