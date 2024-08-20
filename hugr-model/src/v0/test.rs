use super::*;
use smol_str::SmolStr;

#[test]
fn test() {
    let mut module = Module::default();
    let s = SmolStr::new_inline("Hello, world!");
    let t1 = module.add_term(Term::Str(s.clone()));
    let t2 = module.add_term(Term::Str(s.clone()));
    let t3 = module.add_term(Term::Str(SmolStr::new_inline("Hello, world")));

    assert_eq!(t1.0, 0);
    assert_eq!(t2.0, 1);
    assert_eq!(t3.0, 2);
    assert_eq!(module.terms.len(), 3);
    assert_eq!(module.term_table.len(), 2);
}
