// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::Zero;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::dict::TakeReduce;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;

/// Metadata-only take for [`ListViewArray`]. Always returns `Some`.
///
/// This implementation is deliberately simple and read-optimized. We just take the `offsets` and
/// `sizes` at the requested indices and reuse the original `elements` buffer as-is. This works
/// because `ListView` (unlike `List`) allows non-contiguous and out-of-order lists.
///
/// We don't slice the `elements` array because it would require computing min/max offsets and
/// adjusting all offsets accordingly, which is not really worth the small potential memory we
/// would be able to get back.
///
/// The trade-off is that we may keep unreferenced elements in memory, but this is acceptable
/// since we're optimizing for read performance and the data isn't being copied. Callers that
/// need a compacted result should go through [`TakeExecute`] instead.
impl TakeReduce for ListView {
    fn take(array: ArrayView<'_, ListView>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(take_metadata(array, indices)?.into_array()))
    }
}

/// Execution-path take for [`ListViewArray`].
///
/// This does the same metadata-only take as [`TakeReduce`], then rebuilds the result via
/// [`ListViewRebuildMode::MakeZeroCopyToList`] so the output does not carry unreferenced elements
/// from the source. This path is used during `Dict` canonicalization, where we want to
/// materialize a compacted result.
impl TakeExecute for ListView {
    fn take(
        array: ArrayView<'_, ListView>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(connor)[ListView]: Ideally, we would only rebuild after all `take`s and `filter`
        // compute functions have run, at the "top" of the operator tree. However, we cannot do
        // this right now, so we will just rebuild every time (similar to `ListArray`).
        let taken = take_metadata(array, indices)?;
        Ok(Some(
            taken
                .rebuild(ListViewRebuildMode::MakeZeroCopyToList)?
                .into_array(),
        ))
    }
}

/// Shared metadata-only take: take `offsets`, `sizes` and `validity` at `indices` while reusing
/// the original `elements` buffer as-is.
fn take_metadata(
    array: ArrayView<'_, ListView>,
    indices: &ArrayRef,
) -> VortexResult<ListViewArray> {
    let elements = array.elements();
    let offsets = array.offsets();
    let sizes = array.sizes();

    // Combine the array's validity with the indices' validity.
    let new_validity = array.validity()?.take(indices)?;

    // Take can reorder offsets, create gaps, and may introduce overlaps if `indices` contain
    // duplicates.
    let nullable_new_offsets = offsets.take(indices.clone())?;
    let nullable_new_sizes = sizes.take(indices.clone())?;

    // `take` returns nullable arrays; cast back to non-nullable (filling with zeros to represent
    // the null lists — the validity mask tracks nullness separately).
    let new_offsets = match_each_integer_ptype!(nullable_new_offsets.dtype().as_ptype(), |O| {
        nullable_new_offsets.fill_null(Scalar::primitive(O::zero(), Nullability::NonNullable))?
    });
    let new_sizes = match_each_integer_ptype!(nullable_new_sizes.dtype().as_ptype(), |S| {
        nullable_new_sizes.fill_null(Scalar::primitive(S::zero(), Nullability::NonNullable))?
    });

    // SAFETY: Take operation maintains all `ListViewArray` invariants:
    // - `new_offsets` and `new_sizes` are derived from existing valid child arrays.
    // - `new_offsets` and `new_sizes` are non-nullable.
    // - `new_offsets` and `new_sizes` have the same length (both taken with the same `indices`).
    // - Validity correctly reflects the combination of array and indices validity.
    Ok(unsafe {
        ListViewArray::new_unchecked(elements.clone(), new_offsets, new_sizes, new_validity)
    })
}
