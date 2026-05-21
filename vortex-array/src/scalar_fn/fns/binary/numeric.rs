// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrow::Datum;
use crate::arrow::from_arrow_array_with_len;
use crate::executor::ExecutionCtx;
use crate::scalar::NumericOperator;

/// Execute a numeric operation between two arrays.
///
/// This is the entry point for numeric operations from the binary expression.
/// Handles constant-constant directly, otherwise falls back to Arrow.
pub(crate) fn execute_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(result) = constant_numeric(lhs, rhs, op)? {
        return Ok(result);
    }
    arrow_numeric(lhs, rhs, op, ctx)
}

/// Implementation of numeric operations using the Arrow crate.
pub(crate) fn arrow_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    operator: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
    let len = lhs.len();

    let left = Datum::try_new(lhs, ctx)?;
    let right = Datum::try_new_with_target_datatype(rhs, left.data_type(), ctx)?;

    let array = match operator {
        NumericOperator::Add => arrow_arith::numeric::add(&left, &right)?,
        NumericOperator::Sub => arrow_arith::numeric::sub(&left, &right)?,
        NumericOperator::Mul => arrow_arith::numeric::mul(&left, &right)?,
        NumericOperator::Div => arrow_arith::numeric::div(&left, &right)?,
    };

    from_arrow_array_with_len(array.as_ref(), len, nullable)
}

fn constant_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
) -> VortexResult<Option<ArrayRef>> {
    let (Some(lhs), Some(rhs)) = (lhs.as_opt::<Constant>(), rhs.as_opt::<Constant>()) else {
        return Ok(None);
    };

    let Some(result) = lhs
        .scalar()
        .as_primitive()
        .checked_binary_numeric(&rhs.scalar().as_primitive(), op)
    else {
        // Overflow detected — fall through to arrow_numeric which uses wrapping arithmetic.
        return Ok(None);
    };

    Ok(Some(ConstantArray::new(result, lhs.len()).into_array()))
}

/// Repro tests for ABA-23: integer arithmetic panics on overflow / divide-by-zero.
///
/// The bug: `to_primitive()` calls `to_canonical().vortex_expect("to_canonical failed")`,
/// which panics on any `Err` (canonical.rs:489-492). Arrow arithmetic kernels return `Err`
/// on integer overflow and divide-by-zero, but because evaluation is lazy the error only
/// surfaces inside `to_canonical`, where `vortex_expect` converts it to an unrecoverable
/// panic. Callers cannot catch this with normal `VortexResult` handling.
///
/// Each test is marked `#[should_panic]` (current wrong behaviour = panics) AND
/// `#[ignore]` (so CI stays green until a fix lands).
/// When the fix ships, flip `#[should_panic]` to an assertion that `to_canonical` returns
/// `Err`, and remove `#[ignore]`.
///
/// See: <https://linear.app/abanoubdoss/issue/ABA-23>
#[cfg(test)]
#[allow(deprecated)] // intentionally exercising the deprecated to_primitive() panic site
mod tests {
    use std::panic;
    use std::panic::AssertUnwindSafe;

    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar_fn::fns::operators::Operator;

    /// Materializes a lazy binary array via `to_canonical()` (the `VortexResult` path),
    /// NOT via `to_primitive()` (which panics). Returns `Ok(())` on success or graceful
    /// `Err`; returns `Err(String)` if `to_canonical` itself returns `Err`.
    ///
    /// This helper is intentionally separate from `to_primitive()` so the tests can
    /// document *where* the panic is rather than triggering it through the happy path.
    fn eval_to_canonical_result(
        lhs: crate::ArrayRef,
        rhs: crate::ArrayRef,
        op: Operator,
    ) -> Result<(), String> {
        let arr = lhs
            .binary(rhs, op)
            .map_err(|e| format!("binary() Err: {e}"))?;
        arr.to_canonical()
            .map(|_| ())
            .map_err(|e| format!("to_canonical Err: {e}"))
    }

    #[test]
    #[ignore = "demonstrates ABA-23; see https://linear.app/abanoubdoss/issue/ABA-23"]
    #[should_panic(expected = "to_canonical failed")]
    fn issue_aba23_i32_add_max_plus_one_must_not_panic() {
        // i32::MAX + 1 overflows. Arrow returns Err; `to_primitive()` panics via
        // `vortex_expect`. This test locks the wrong behaviour: flip when fixed.
        let lhs = buffer![i32::MAX].into_array();
        let rhs = buffer![1i32].into_array();
        let arr = lhs
            .binary(rhs, Operator::Add)
            .expect("binary() should not fail");
        // to_primitive() is the panic site (canonical.rs:489-492).
        drop(arr.to_primitive());
    }

    #[test]
    #[ignore = "demonstrates ABA-23; see https://linear.app/abanoubdoss/issue/ABA-23"]
    #[should_panic(expected = "to_canonical failed")]
    fn issue_aba23_i64_div_by_zero_must_not_panic() {
        // Integer divide-by-zero: Arrow returns Err; `to_primitive()` panics.
        let lhs = buffer![10i64].into_array();
        let rhs = buffer![0i64].into_array();
        let arr = lhs
            .binary(rhs, Operator::Div)
            .expect("binary() should not fail");
        drop(arr.to_primitive());
    }

    #[test]
    #[ignore = "demonstrates ABA-23; see https://linear.app/abanoubdoss/issue/ABA-23"]
    #[should_panic(expected = "to_canonical failed")]
    fn issue_aba23_u32_sub_underflow_must_not_panic() {
        // 0u32 - 1 underflows (unsigned). Arrow returns Err; `to_primitive()` panics.
        let lhs = buffer![0u32].into_array();
        let rhs = buffer![1u32].into_array();
        let arr = lhs
            .binary(rhs, Operator::Sub)
            .expect("binary() should not fail");
        drop(arr.to_primitive());
    }

    #[test]
    #[ignore = "demonstrates ABA-23; see https://linear.app/abanoubdoss/issue/ABA-23"]
    #[should_panic(expected = "to_canonical failed")]
    fn issue_aba23_i32_mul_overflow_must_not_panic() {
        // i32::MAX * 2 overflows. Arrow returns Err; `to_primitive()` panics.
        let lhs = buffer![i32::MAX].into_array();
        let rhs = buffer![2i32].into_array();
        let arr = lhs
            .binary(rhs, Operator::Mul)
            .expect("binary() should not fail");
        drop(arr.to_primitive());
    }

    /// Verify the `to_canonical()` path (the `VortexResult` path) does surface the error
    /// rather than panicking. This test documents the *desired* graceful behaviour and
    /// should stay green both before and after the fix.
    #[test]
    fn issue_aba23_to_canonical_returns_err_not_panic_on_overflow() {
        // Use catch_unwind to distinguish graceful Err from panic.
        let prev_hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            eval_to_canonical_result(
                buffer![i32::MAX].into_array(),
                buffer![1i32].into_array(),
                Operator::Add,
            )
        }));
        panic::set_hook(prev_hook);

        match result {
            Ok(Err(_graceful_error)) => { /* correct: graceful VortexResult::Err */ }
            Ok(Ok(())) => {
                panic!("ABA-23: i32::MAX + 1 returned Ok — silent integer wrap is also wrong.")
            }
            Err(_panic) => panic!(
                "ABA-23: i32::MAX + 1 panicked via to_canonical() — the VortexResult path \
                 should not panic; only to_primitive() (which calls vortex_expect) should."
            ),
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::RecursiveCanonical;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::binary::numeric::ConstantArray;
    use crate::scalar_fn::fns::operators::Operator;

    fn sub_scalar(array: &ArrayRef, scalar: impl Into<Scalar>) -> VortexResult<ArrayRef> {
        array
            .binary(
                ConstantArray::new(scalar, array.len()).into_array(),
                Operator::Sub,
            )
            .and_then(|a| {
                a.execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            })
            .map(|a| a.0.into_array())
    }

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = buffer![1u16, 2, 3].into_array();
        let result = sub_scalar(&values, 1u16).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]));
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let values = buffer![1i64, 2, 3].into_array();
        let result = sub_scalar(&values, -1i64).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]));
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let result = sub_scalar(&values.into_array(), Some(1u16)).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)])
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let result = sub_scalar(&values, -1f64).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]));
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let _results = sub_scalar(&values, 1.0f32).unwrap();
        let _results = sub_scalar(&values, f32::MAX).unwrap();
    }
}
