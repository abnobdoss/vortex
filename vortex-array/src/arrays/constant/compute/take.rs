// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::AllOr;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::MaskedArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar::Scalar;
use crate::validity::Validity;

impl TakeReduce for Constant {
    fn take(array: ArrayView<'_, Constant>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = match indices
            .validity()?
            .execute_mask(indices.len(), &mut ctx)?
            .bit_buffer()
        {
            AllOr::All => {
                let scalar = Scalar::try_new(
                    array
                        .scalar()
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                    array.scalar().value().cloned(),
                )?;
                ConstantArray::new(scalar, indices.len()).into_array()
            }
            AllOr::None => ConstantArray::new(
                Scalar::null(
                    array
                        .dtype()
                        .union_nullability(indices.dtype().nullability()),
                ),
                indices.len(),
            )
            .into_array(),
            AllOr::Some(v) => {
                let arr = ConstantArray::new(array.scalar().clone(), indices.len()).into_array();

                if array.scalar().is_null() {
                    return Ok(Some(arr));
                }

                MaskedArray::try_new(arr, Validity::from(v.clone()))?.into_array()
            }
        };
        Ok(Some(result))
    }
}

impl Constant {
    pub const TAKE_RULES: ParentRuleSet<Self> =
        ParentRuleSet::new(&[ParentRuleSet::lift(&TakeReduceAdaptor::<Self>(Self))]);
}

#[cfg(test)]
mod tests {
    use std::f64;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_mask::AllOr;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn take_nullable_indices() {
        let array = ConstantArray::new(42, 10).into_array();
        let taken = array
            .take(
                PrimitiveArray::new(
                    buffer![0, 5, 7],
                    Validity::from_iter(vec![false, true, false]),
                )
                .into_array(),
            )
            .unwrap();
        let valid_indices: &[usize] = &[1usize];
        assert_eq!(
            &array.dtype().with_nullability(Nullability::Nullable),
            taken.dtype()
        );
        assert_arrays_eq!(
            #[expect(deprecated)]
            taken.to_primitive(),
            PrimitiveArray::new(
                buffer![42i32, 42, 42],
                Validity::from_iter([false, true, false])
            )
        );
        assert_eq!(
            taken
                .validity()
                .unwrap()
                .execute_mask(taken.len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .indices(),
            AllOr::Some(valid_indices)
        );
    }

    #[test]
    fn take_all_valid_indices() {
        let array = ConstantArray::new(42, 10).into_array();
        let taken = array
            .take(PrimitiveArray::new(buffer![0, 5, 7], Validity::AllValid).into_array())
            .unwrap();
        assert_eq!(
            &array.dtype().with_nullability(Nullability::Nullable),
            taken.dtype()
        );
        assert_arrays_eq!(
            #[expect(deprecated)]
            taken.to_primitive(),
            PrimitiveArray::new(buffer![42i32, 42, 42], Validity::AllValid)
        );
        assert_eq!(
            taken
                .validity()
                .unwrap()
                .execute_mask(taken.len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .indices(),
            AllOr::All
        );
    }

    #[rstest]
    #[case(ConstantArray::new(42i32, 5))]
    #[case(ConstantArray::new(f64::consts::PI, 10))]
    #[case(ConstantArray::new(Scalar::from("hello"), 3))]
    #[case(ConstantArray::new(Scalar::null_native::<i64>(), 5))]
    #[case(ConstantArray::new(true, 1))]
    fn test_take_constant_conformance(#[case] array: ConstantArray) {
        test_take_conformance(&array.into_array());
    }

    /// Regression test for ABA-17.
    ///
    /// `Constant::take` previously branched only on the indices' validity mask
    /// and never inspected the raw index values. A VALID index that fell past
    /// the constant array's length silently produced a row carrying the
    /// constant fill value, rather than an out-of-bounds error — diverging
    /// from every other take kernel (e.g. `Primitive::take`, which rejects
    /// such indices).
    #[test]
    #[ignore = "ABA-17 — fixed in a follow-up commit; un-ignore once the fix lands"]
    fn issue_aba17_take_must_reject_oob_indices() -> vortex_error::VortexResult<()> {
        let array = ConstantArray::new(42i32, 10).into_array();

        // Single VALID index whose value is far past `array.len() = 10`.
        let indices = PrimitiveArray::new(buffer![u32::MAX], Validity::NonNullable).into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Either the lazy `take` or the eager `execute::<Canonical>` must
        // error. We accept either: the contract is that no row is produced
        // for an out-of-bounds index.
        let outcome = array
            .take(indices)
            .and_then(|taken| taken.execute::<crate::Canonical>(&mut ctx));
        assert!(
            outcome.is_err(),
            "ConstantArray::take must reject out-of-bounds indices, but it succeeded"
        );

        // Mixed in-range + OOB indices, all valid: the OOB tail must still
        // be rejected, even though some indices are in range.
        let indices =
            PrimitiveArray::new(buffer![0u32, 5, 1010], Validity::NonNullable).into_array();
        let outcome = array
            .take(indices)
            .and_then(|taken| taken.execute::<crate::Canonical>(&mut ctx));
        assert!(
            outcome.is_err(),
            "ConstantArray::take must reject a mix of in-range and OOB indices"
        );

        // Null index whose underlying value is OOB must be accepted: the
        // bounds check applies only to VALID positions. This guards against
        // a regression where a naive fix would inspect every raw value.
        let indices =
            PrimitiveArray::new(buffer![0u32, u32::MAX], Validity::from_iter([true, false]))
                .into_array();
        let taken = array.take(indices)?;
        assert_eq!(taken.len(), 2);

        Ok(())
    }
}
