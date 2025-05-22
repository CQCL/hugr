use std::fmt::Display;

use hugr_model::v0::SymbolName;
use hugr_model::v0::ast;
use itertools::Itertools as _;
use servo_arc::Arc;
use servo_arc::ThinArc;

use super::Typed;
use super::{Term, ViewError};

/// [`Term`]s obtained by applying a symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Apply(ThinArc<ApplyHeader, Term>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ApplyHeader {
    name: SymbolName,
    type_: Term,
}

impl Apply {
    /// Creates a new symbol application.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hugr_core::terms::{SymbolName, Term, Apply, Literal, Typed};
    /// let symbol = SymbolName::new("some.symbol");
    /// let arg0 = Term::Literal(Literal::Nat(42));
    /// let arg1 = Term::Literal(Literal::Nat(1337));
    /// let type_ = Term::Wildcard;
    ///
    /// let apply = Apply::new(
    ///     symbol.clone(),
    ///     [arg0.clone(), arg1.clone()],
    ///     type_.clone()
    /// );
    ///
    /// assert_eq!(apply.name(), &symbol);
    /// assert_eq!(apply.args(), &[arg0, arg1]);
    /// assert_eq!(apply.type_(), &type_);
    /// ```
    pub fn new<A>(name: SymbolName, args: A, type_: Term) -> Self
    where
        A: IntoIterator<Item = Term>,
        A::IntoIter: ExactSizeIterator,
    {
        Self(ThinArc::from_header_and_iter(
            ApplyHeader { name, type_ },
            args.into_iter(),
        ))
    }

    /// Tries to create a new symbol application.
    ///
    /// # Errors
    ///
    /// When creating an argument fails.
    pub fn try_new<A, E>(name: SymbolName, args: A, type_: Term) -> Result<Self, E>
    where
        A: IntoIterator<Item = Result<Term, E>>,
        A::IntoIter: ExactSizeIterator,
    {
        let args = args.into_iter();

        // Use wildcard terms as placeholders.
        let mut arc = ThinArc::from_header_and_iter(
            ApplyHeader { name, type_ },
            (0..args.len()).map(|_| Term::Wildcard),
        );

        // Then fill in the arguments one by one.
        // The `unwrap` always succeeds since we have the only copy of the `Arc`.
        let arg_slots = Arc::get_mut(&mut arc).unwrap().slice_mut().iter_mut();

        for (arg_slot, arg) in arg_slots.zip_eq(args) {
            *arg_slot = arg?;
        }

        Ok(Self(arc))
    }

    /// The name of the applied symbol.
    pub fn name(&self) -> &SymbolName {
        &self.0.header.name
    }

    /// The arguments to the symbol.
    pub fn args(&self) -> &[Term] {
        self.0.slice()
    }

    /// Attempt to view this term as an application of a particular symbol with a given arity.
    ///
    /// In the case that there are fewer than `N` arguments we still return a match. The returned
    /// argument sequence is padded from the front with [`Term::Wildcard`], indicating that the
    /// omitted arguments are intended to be implicit.
    ///
    /// # Errors
    ///
    /// - [`ViewError::Mismatch`] when the symbol name does not match.
    /// - [`ViewError::Invalid`] when the symbol is applied to too many arguments.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hugr_core::terms::{Term, Apply, SymbolName, ViewError, Literal};
    /// let this = SymbolName::new("this.name");
    /// let that = SymbolName::new("that.name");
    /// let arg1 = Term::Literal(Literal::Nat(1));
    /// let arg2 = Term::Literal(Literal::Nat(2));
    ///
    /// let apply = Apply::new(this.clone(), [arg1.clone(), arg2.clone()], Term::Wildcard);
    ///
    /// assert_eq!(apply.view(&this), Ok([arg1.clone(), arg2.clone()]));
    /// assert_eq!(apply.view(&this), Ok([Term::Wildcard, arg1.clone(), arg2.clone()]));
    /// assert!(matches!(apply.view::<1>(&this), Err(ViewError::Invalid(_))));
    /// assert_eq!(apply.view::<1>(&that), Err(ViewError::Mismatch));
    /// ```
    pub fn view<const N: usize>(&self, symbol: &SymbolName) -> Result<[Term; N], ViewError> {
        if self.name() != symbol {
            return Err(ViewError::Mismatch);
        }

        if self.args().len() > N {
            return Err(ViewError::Invalid(format!(
                "`{}` expects at most {} arguments",
                symbol, N
            )));
        }

        let result = std::array::from_fn(|i| {
            (i + self.args().len())
                .checked_sub(N)
                .map(|i| self.args()[i].clone())
                .unwrap_or_default()
        });

        Ok(result)
    }
}

impl Typed for Apply {
    #[allow(refining_impl_trait)]
    fn type_(&self) -> &Term {
        &self.0.header.type_
    }
}

impl From<&Apply> for ast::Term {
    fn from(value: &Apply) -> Self {
        let name = value.name().clone();
        let args = value.args().iter().map(ast::Term::from).collect();
        ast::Term::Apply(name, args)
    }
}

impl Display for Apply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::Term::from(self))
    }
}

impl TryFrom<Term> for Apply {
    type Error = ViewError;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        match value {
            Term::Apply(apply) => Ok(apply),
            Term::Wildcard => Err(ViewError::Uninferred),
            Term::Literal(_) => Err(ViewError::Mismatch),
            Term::List(_) => Err(ViewError::Mismatch),
            Term::Tuple(_) => Err(ViewError::Mismatch),
            Term::Var(_) => Err(ViewError::Variable),
            Term::StaticType => Err(ViewError::Mismatch),
        }
    }
}
