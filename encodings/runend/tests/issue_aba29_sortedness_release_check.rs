//! Regression test for ABA-29: `RunEnd::try_new` sortedness validation must
//! be enforced in release builds, not gated behind `#[cfg(debug_assertions)]`.
//!
//! Linear: https://linear.app/abanoubdoss/issue/ABA-29
//!
//! Background
//! ----------
//! `RunEndData::validate_parts` historically wrapped the strict-sorted check
//! on the run-ends array in `#[cfg(debug_assertions)] { ... debug_assert!(...) }`,
//! so the check was compiled out of release builds. The only validation that
//! survived in release was the first/last-coverage check, which an unsorted
//! middle run end (e.g. `[5, 3, 10]`) passes (last == 10 == length).
//! Downstream binary-search-based consumers then returned wrong values.
//!
//! Fallible constructors must return `Err` on invalid input regardless of
//! build profile.

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_buffer::buffer;
    use vortex_runend::RunEnd;

    #[test]
    fn try_new_rejects_unsorted_ends() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Unsorted: 5 then 3 then 10. Last == 10 so the first/last coverage
        // check (the only validation surviving in release before the fix) is
        // satisfied. The strict-sorted check must still reject this input.
        let ends = buffer![5u32, 3, 10].into_array();
        let values = buffer![100i32, 200, 300].into_array();
        let result = RunEnd::try_new(ends, values, &mut ctx);
        assert!(
            result.is_err(),
            "RunEnd::try_new must reject unsorted ends [5, 3, 10] in all \
             build profiles; got Ok"
        );
    }

    #[test]
    fn try_new_rejects_strictly_unsorted_middle() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Strict-sorted violated in the middle ([10, 1, 10]) but last == 10
        // so the coverage check passes.
        let ends = buffer![10u32, 1, 10].into_array();
        let values = buffer![1i32, 2, 3].into_array();
        let result = RunEnd::try_new(ends, values, &mut ctx);
        assert!(
            result.is_err(),
            "RunEnd::try_new must reject [10, 1, 10] (strict-sorted violated \
             in the middle); got Ok"
        );
    }

    #[test]
    fn try_new_accepts_sorted_ends() {
        // Control: sorted ends must still build successfully.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let ends = buffer![3u32, 5, 10].into_array();
        let values = buffer![100i32, 200, 300].into_array();
        let arr =
            RunEnd::try_new(ends, values, &mut ctx).expect("sorted RunEnd::try_new should succeed");
        assert_eq!(arr.len(), 10);
    }
}
