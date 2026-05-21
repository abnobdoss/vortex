// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexError;
use vortex_proto::expr::binary_opts::BinaryOp;

/// Equalities, inequalities, and boolean operations over possibly null values.
///
/// For most operations, if either side is null, the result is null.
///
/// The Boolean operators (And, Or) obey [Kleene (three-valued) logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics).
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Operator {
    /// Expressions are equal.
    Eq,
    /// Expressions are not equal.
    NotEq,
    /// Expression is greater than another
    Gt,
    /// Expression is greater or equal to another
    Gte,
    /// Expression is less than another
    Lt,
    /// Expression is less or equal to another
    Lte,
    /// Boolean AND (∧).
    // TODO(joe): rename to KleeneAnd
    And,
    /// Boolean OR (∨).
    // TODO(joe): rename to KleeneOr
    Or,
    /// The sum of the arguments.
    ///
    /// Errs at runtime if the sum would overflow or underflow.
    Add,
    /// The difference between the arguments.
    ///
    /// Errs at runtime if the sum would overflow or underflow.
    ///
    /// The result is null at any index that either input is null.
    Sub,
    /// Multiple two numbers
    Mul,
    /// Divide the left side by the right side
    Div,
}

impl From<Operator> for i32 {
    fn from(value: Operator) -> Self {
        let op: BinaryOp = value.into();
        op.into()
    }
}

impl From<Operator> for BinaryOp {
    fn from(value: Operator) -> Self {
        match value {
            Operator::Eq => BinaryOp::Eq,
            Operator::NotEq => BinaryOp::NotEq,
            Operator::Gt => BinaryOp::Gt,
            Operator::Gte => BinaryOp::Gte,
            Operator::Lt => BinaryOp::Lt,
            Operator::Lte => BinaryOp::Lte,
            Operator::And => BinaryOp::And,
            Operator::Or => BinaryOp::Or,
            Operator::Add => BinaryOp::Add,
            Operator::Sub => BinaryOp::Sub,
            Operator::Mul => BinaryOp::Mul,
            Operator::Div => BinaryOp::Div,
        }
    }
}

impl TryFrom<i32> for Operator {
    type Error = VortexError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Ok(BinaryOp::try_from(value)?.into())
    }
}

impl From<BinaryOp> for Operator {
    fn from(value: BinaryOp) -> Self {
        match value {
            BinaryOp::Eq => Operator::Eq,
            BinaryOp::NotEq => Operator::NotEq,
            BinaryOp::Gt => Operator::Gt,
            BinaryOp::Gte => Operator::Gte,
            BinaryOp::Lt => Operator::Lt,
            BinaryOp::Lte => Operator::Lte,
            BinaryOp::And => Operator::And,
            BinaryOp::Or => Operator::Or,
            BinaryOp::Add => Operator::Add,
            BinaryOp::Sub => Operator::Sub,
            BinaryOp::Mul => Operator::Mul,
            BinaryOp::Div => Operator::Div,
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match &self {
            Operator::Eq => "=",
            Operator::NotEq => "!=",
            Operator::Gt => ">",
            Operator::Gte => ">=",
            Operator::Lt => "<",
            Operator::Lte => "<=",
            Operator::And => "and",
            Operator::Or => "or",
            Operator::Add => "+",
            Operator::Sub => "-",
            Operator::Mul => "*",
            Operator::Div => "/",
        };
        Display::fmt(display, f)
    }
}

impl Operator {
    pub fn inverse(self) -> Option<Self> {
        match self {
            Operator::Eq => Some(Operator::NotEq),
            Operator::NotEq => Some(Operator::Eq),
            Operator::Gt => Some(Operator::Lte),
            Operator::Gte => Some(Operator::Lt),
            Operator::Lt => Some(Operator::Gte),
            Operator::Lte => Some(Operator::Gt),
            Operator::And
            | Operator::Or
            | Operator::Add
            | Operator::Sub
            | Operator::Mul
            | Operator::Div => None,
        }
    }

    pub fn logical_inverse(self) -> Option<Self> {
        match self {
            Operator::And => Some(Operator::Or),
            Operator::Or => Some(Operator::And),
            _ => None,
        }
    }

    /// Change the sides of the operator, so that changing lhs and rhs won't change the result of the operation
    pub fn swap(self) -> Option<Self> {
        match self {
            Operator::Eq => Some(Operator::Eq),
            Operator::NotEq => Some(Operator::NotEq),
            Operator::Gt => Some(Operator::Lt),
            Operator::Gte => Some(Operator::Lte),
            Operator::Lt => Some(Operator::Gt),
            Operator::Lte => Some(Operator::Gte),
            Operator::And => Some(Operator::And),
            Operator::Or => Some(Operator::Or),
            Operator::Add => Some(Operator::Add),
            Operator::Mul => Some(Operator::Mul),
            Operator::Sub | Operator::Div => None,
        }
    }

    pub fn is_arithmetic(&self) -> bool {
        matches!(self, Self::Add | Self::Sub | Self::Mul | Self::Div)
    }

    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Eq | Self::NotEq | Self::Gt | Self::Gte | Self::Lt | Self::Lte
        )
    }
}

/// The six comparison operators, providing compile-time guarantees that only
/// comparison variants are used where comparisons are expected.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompareOperator {
    /// Expressions are equal.
    Eq,
    /// Expressions are not equal.
    NotEq,
    /// Expression is greater than another.
    Gt,
    /// Expression is greater or equal to another.
    Gte,
    /// Expression is less than another.
    Lt,
    /// Expression is less or equal to another.
    Lte,
}

impl CompareOperator {
    /// Return the logical inverse of this comparison operator.
    pub fn inverse(self) -> Self {
        match self {
            CompareOperator::Eq => CompareOperator::NotEq,
            CompareOperator::NotEq => CompareOperator::Eq,
            CompareOperator::Gt => CompareOperator::Lte,
            CompareOperator::Gte => CompareOperator::Lt,
            CompareOperator::Lt => CompareOperator::Gte,
            CompareOperator::Lte => CompareOperator::Gt,
        }
    }

    /// Swap the sides of the operator so that swapping lhs and rhs preserves the result.
    pub fn swap(self) -> Self {
        match self {
            CompareOperator::Eq => CompareOperator::Eq,
            CompareOperator::NotEq => CompareOperator::NotEq,
            CompareOperator::Gt => CompareOperator::Lt,
            CompareOperator::Gte => CompareOperator::Lte,
            CompareOperator::Lt => CompareOperator::Gt,
            CompareOperator::Lte => CompareOperator::Gte,
        }
    }
}

impl Display for CompareOperator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let display = match self {
            CompareOperator::Eq => "=",
            CompareOperator::NotEq => "!=",
            CompareOperator::Gt => ">",
            CompareOperator::Gte => ">=",
            CompareOperator::Lt => "<",
            CompareOperator::Lte => "<=",
        };
        Display::fmt(display, f)
    }
}

impl From<CompareOperator> for Operator {
    fn from(value: CompareOperator) -> Self {
        match value {
            CompareOperator::Eq => Operator::Eq,
            CompareOperator::NotEq => Operator::NotEq,
            CompareOperator::Gt => Operator::Gt,
            CompareOperator::Gte => Operator::Gte,
            CompareOperator::Lt => Operator::Lt,
            CompareOperator::Lte => Operator::Lte,
        }
    }
}

impl TryFrom<Operator> for CompareOperator {
    type Error = VortexError;

    fn try_from(value: Operator) -> Result<Self, Self::Error> {
        match value {
            Operator::Eq => Ok(CompareOperator::Eq),
            Operator::NotEq => Ok(CompareOperator::NotEq),
            Operator::Gt => Ok(CompareOperator::Gt),
            Operator::Gte => Ok(CompareOperator::Gte),
            Operator::Lt => Ok(CompareOperator::Lt),
            Operator::Lte => Ok(CompareOperator::Lte),
            other => Err(vortex_error::vortex_err!(
                InvalidArgument: "{other} is not a comparison operator"
            )),
        }
    }
}

#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for CompareOperator {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(match u.int_in_range(0..=5)? {
            0 => CompareOperator::Eq,
            1 => CompareOperator::NotEq,
            2 => CompareOperator::Gt,
            3 => CompareOperator::Gte,
            4 => CompareOperator::Lt,
            5 => CompareOperator::Lte,
            _ => unreachable!(),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::scalar_fn::ScalarFnId;
    use crate::scalar_fn::ScalarFnVTable as _;
    use crate::scalar_fn::fns::list_contains::ListContains;
    use crate::scalar_fn::fns::operators::Operator;

    /// Verify that the test harness itself is functional: `ListContains` is the
    /// scalar fn that the DataFusion converter uses to emulate SQL `IN`, and its
    /// ID must be stable for the gap probes below to be meaningful.
    #[test]
    fn control_list_contains_id_is_stable() {
        assert_eq!(
            ListContains.id(),
            ScalarFnId::new("vortex.list.contains"),
            "ListContains ScalarFnId changed; update the ABA-31 repro tests"
        );
    }

    // ------------------------------------------------------------------
    // ABA-31 (a) — No general string scalar functions
    //
    // The only string-shaped `scalar_fn::fns` module is `like`.  There are no
    // public factories in `expr::exprs` for `length`, `substring`, `concat`,
    // `upper`, `lower`, `replace`, `trim`, etc.  This test documents the
    // absence: when un-ignored it panics with a detailed diagnostic so that
    // CI surfaces the gap clearly.
    // ------------------------------------------------------------------
    #[test]
    #[ignore = "demonstrates ABA-31; see https://linear.app/abanoubdoss/issue/ABA-31"]
    fn issue_aba31_a_general_string_scalar_function_exists() {
        // Inventory of `scalar_fn::fns` modules as of the commit this test was
        // written (see `src/scalar_fn/fns/mod.rs`):
        //   between, binary, case_when, cast, dynamic, fill_null, get_item,
        //   is_not_null, is_null, like, list_contains, literal, mask, merge,
        //   not, operators, pack, root, select, stat, variant_get, zip
        //
        // None of the following SQL-standard string functions appear as a
        // `scalar_fn::fns` module or as a factory in `expr::exprs.rs`:
        let missing: &[&str] = &[
            "length",
            "char_length",
            "octet_length",
            "substring",
            "substr",
            "concat",
            "upper",
            "lower",
            "replace",
            "trim",
            "ltrim",
            "rtrim",
            "strpos",
            "starts_with",
            "ends_with",
        ];
        // The only string-shaped operation that exists:
        let only_string_fn = "like";

        panic!(
            "ABA-31(a) STILL_MISSING: no general string scalar functions. \
             The only string op in scalar_fn::fns is `{only_string_fn}`. \
             None of {missing:?} have a scalar_fn module or expr factory. \
             Engines pushing down SQL string ops must fall back to canonical \
             execution. Fix: add scalar_fn::fns modules for at least length, \
             substring, concat, upper, lower, replace, trim."
        );
    }

    // ------------------------------------------------------------------
    // ABA-31 (b) — No unary numeric operator
    //
    // `Operator` (this file) enumerates only binary operations.  There is no
    // `Abs`, `Neg`, `Floor`, `Ceil`, `Round`, `Sign`, or `Sqrt` variant.
    // The exhaustive `match` below is the structural repro: if upstream ever
    // adds a unary variant, the match will stop compiling (non-exhaustive),
    // forcing the author to revisit this test before merging.
    // ------------------------------------------------------------------
    #[test]
    #[ignore = "demonstrates ABA-31; see https://linear.app/abanoubdoss/issue/ABA-31"]
    fn issue_aba31_b_unary_numeric_operator_exists() {
        // Exhaustive match over all current Operator variants (12 binary ops).
        // Adding any new variant — including a unary one — breaks compilation
        // here, which is the intended signal.
        let sentinel = Operator::Eq;
        let _label: &str = match sentinel {
            Operator::Eq => "Eq",
            Operator::NotEq => "NotEq",
            Operator::Gt => "Gt",
            Operator::Gte => "Gte",
            Operator::Lt => "Lt",
            Operator::Lte => "Lte",
            Operator::And => "And",
            Operator::Or => "Or",
            Operator::Add => "Add",
            Operator::Sub => "Sub",
            Operator::Mul => "Mul",
            Operator::Div => "Div",
        };

        panic!(
            "ABA-31(b) STILL_MISSING: Operator enum has 12 variants, all binary. \
             No unary numeric op (Abs, Neg, Floor, Ceil, Round, Sign, Sqrt) is \
             defined in Operator or as a separate scalar_fn module. The only \
             existing unary scalar fn is `not` (boolean only). Fix: add a \
             UnaryOperator enum (or unary variants in Operator) and a \
             corresponding scalar_fn::fns::unary module."
        );
    }

    // ------------------------------------------------------------------
    // ABA-31 (c) — No native InList operator
    //
    // SQL `x IN (a, b, c)` is lowered by `vortex-datafusion/src/convert/
    // exprs.rs` as `list_contains(lit(List[a,b,c]), x)`.  That conflates
    // relational set-membership with list-element-containment: a `List`
    // literal is a single typed value; an SQL IN predicate is a set of
    // disjuncts amenable to stat-based falsification and separate
    // optimization.  There is no `InList` scalar_fn module and no
    // `in_list` factory in `expr::exprs.rs`.
    // ------------------------------------------------------------------
    #[test]
    #[ignore = "demonstrates ABA-31; see https://linear.app/abanoubdoss/issue/ABA-31"]
    fn issue_aba31_c_native_in_list_operator_exists() {
        // The only set-membership-adjacent facility that exists today:
        assert_eq!(
            ListContains.id(),
            ScalarFnId::new("vortex.list.contains"),
            "ListContains ScalarFnId changed; ABA-31(c) probe needs updating"
        );

        // There is no `vortex.in_list` ScalarFnId anywhere in `scalar_fn::fns/`
        // (inventoried: same list as ABA-31(a)). The DataFusion converter at
        // `vortex-datafusion/src/convert/exprs.rs` lines ~251-272 lowers
        // InListExpr by building a List scalar literal and delegating to
        // list_contains, which is semantically distinct from set membership.
        panic!(
            "ABA-31(c) STILL_MISSING: no native InList scalar fn or in_list \
             expr factory. SQL `IN (...)` is emulated via \
             list_contains(lit(List[...]), x) in the DataFusion converter, \
             losing the relational set-membership semantics needed for stat \
             falsification. Fix: introduce a dedicated InList scalar_fn and \
             an in_list factory in expr::exprs.rs."
        );
    }
}
