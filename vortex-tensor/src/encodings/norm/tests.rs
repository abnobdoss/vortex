// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Extension;
use vortex::array::arrays::PrimitiveArray;
use vortex::error::VortexResult;
use vortex::scalar::Scalar;

use crate::encodings::norm::NormVectorArray;
use crate::utils::test_helpers::assert_close;
use crate::utils::test_helpers::extract_vector_rows;
use crate::utils::test_helpers::nullable_vector_array;
use crate::utils::test_helpers::vector_array;

#[test]
fn encode_unit_vectors() -> VortexResult<()> {
    // Already unit-length vectors: norms should be 1.0 and vectors unchanged.
    let arr = vector_array(
        3,
        &[
            1.0, 0.0, 0.0, // norm = 1.0
            0.0, 1.0, 0.0, // norm = 1.0
        ],
    )?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let norm = NormVectorArray::compress(arr, &mut ctx)?;
    let norms: PrimitiveArray = norm.norms().clone().execute(&mut ctx)?;
    assert_close(norms.as_slice::<f64>(), &[1.0, 1.0]);

    let rows = extract_vector_rows(norm.vector_array(), &mut ctx)?;
    assert_close(&rows[0], &[1.0, 0.0, 0.0]);
    assert_close(&rows[1], &[0.0, 1.0, 0.0]);

    Ok(())
}

#[test]
fn encode_non_unit_vectors() -> VortexResult<()> {
    let arr = vector_array(
        2,
        &[
            3.0, 4.0, // norm = 5.0
            0.0, 0.0, // norm = 0.0 (zero vector)
        ],
    )?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let norm = NormVectorArray::compress(arr, &mut ctx)?;
    let norms: PrimitiveArray = norm.norms().clone().execute(&mut ctx)?;
    assert_close(norms.as_slice::<f64>(), &[5.0, 0.0]);

    let rows = extract_vector_rows(norm.vector_array(), &mut ctx)?;
    assert_close(&rows[0], &[3.0 / 5.0, 4.0 / 5.0]);
    assert_close(&rows[1], &[0.0, 0.0]);

    Ok(())
}

#[test]
fn execute_round_trip() -> VortexResult<()> {
    let arr = vector_array(
        2,
        &[
            3.0, 4.0, // norm = 5.0
            6.0, 8.0, // norm = 10.0
        ],
    )?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let norm = NormVectorArray::compress(arr, &mut ctx)?;

    // Execute to reconstruct the original vectors.
    let reconstructed = norm.decompress(&mut ctx)?;

    // The reconstructed array should be a Vector extension array.
    assert!(reconstructed.as_opt::<Extension>().is_some());

    let rows = extract_vector_rows(&reconstructed, &mut ctx)?;
    assert_close(&rows[0], &[3.0, 4.0]);
    assert_close(&rows[1], &[6.0, 8.0]);

    Ok(())
}

#[test]
fn execute_round_trip_zero_vector() -> VortexResult<()> {
    let arr = vector_array(2, &[0.0, 0.0])?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let norm = NormVectorArray::compress(arr, &mut ctx)?;

    let reconstructed = norm.decompress(&mut ctx)?;

    let rows = extract_vector_rows(&reconstructed, &mut ctx)?;
    // Zero vector should remain zero after round-trip.
    assert_close(&rows[0], &[0.0, 0.0]);

    Ok(())
}

#[test]
fn scalar_at_returns_original_vector() -> VortexResult<()> {
    let arr = vector_array(
        2,
        &[
            3.0, 4.0, // norm = 5.0
            6.0, 8.0, // norm = 10.0
        ],
    )?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = NormVectorArray::compress(arr, &mut ctx)?;

    // `scalar_at` on the NormVectorArray should match `scalar_at` on the decompressed result.
    let decompressed = encoded.decompress(&mut ctx)?;

    let norm_array = encoded.into_array();
    for i in 0..2 {
        assert_eq!(norm_array.scalar_at(i)?, decompressed.scalar_at(i)?);
    }

    Ok(())
}

#[test]
fn compress_preserves_validity() -> VortexResult<()> {
    let arr = nullable_vector_array(
        2,
        &[
            3.0, 4.0, // row 0: valid, norm = 5.0
            0.0, 0.0, // row 1: null (data is unspecified)
            6.0, 8.0, // row 2: valid, norm = 10.0
        ],
        [true, false, true],
    )?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let norm = NormVectorArray::compress(arr, &mut ctx)?;

    // Valid rows should have correct norms.
    assert!(!norm.norms().is_invalid(0)?);
    assert!(norm.norms().is_invalid(1)?);
    assert!(!norm.norms().is_invalid(2)?);

    // The vector array should also preserve validity.
    assert!(!norm.vector_array().is_invalid(0)?);
    assert!(norm.vector_array().is_invalid(1)?);
    assert!(!norm.vector_array().is_invalid(2)?);

    Ok(())
}

#[test]
fn round_trip_with_nulls() -> VortexResult<()> {
    let arr = nullable_vector_array(
        2,
        &[
            3.0, 4.0, // row 0: valid, norm = 5.0
            0.0, 0.0, // row 1: null
            6.0, 8.0, // row 2: valid, norm = 10.0
        ],
        [true, false, true],
    )?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let norm = NormVectorArray::compress(arr, &mut ctx)?;
    let reconstructed = norm.decompress(&mut ctx)?;

    // Valid rows should round-trip correctly.
    let rows = extract_vector_rows(&reconstructed, &mut ctx)?;
    assert_close(&rows[0], &[3.0, 4.0]);
    assert_close(&rows[2], &[6.0, 8.0]);

    // Null rows should remain null.
    assert!(reconstructed.is_invalid(1)?);

    Ok(())
}

#[test]
fn scalar_at_null_row_returns_null() -> VortexResult<()> {
    let arr = nullable_vector_array(
        2,
        &[
            3.0, 4.0, // row 0: valid
            0.0, 0.0, // row 1: null
        ],
        [true, false],
    )?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = NormVectorArray::compress(arr, &mut ctx)?;
    let norm_array = encoded.into_array();

    // Valid row should return a non-null scalar.
    assert!(!norm_array.scalar_at(0)?.is_null());

    // Null row should return a null scalar.
    assert_eq!(
        norm_array.scalar_at(1)?,
        Scalar::null(norm_array.dtype().clone())
    );

    Ok(())
}

#[test]
fn scalar_at_with_nulls_matches_decompressed() -> VortexResult<()> {
    let arr = nullable_vector_array(
        2,
        &[
            3.0, 4.0, // row 0: valid, norm = 5.0
            0.0, 0.0, // row 1: null
            6.0, 8.0, // row 2: valid, norm = 10.0
        ],
        [true, false, true],
    )?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let encoded = NormVectorArray::compress(arr, &mut ctx)?;
    let decompressed = encoded.decompress(&mut ctx)?;

    let norm_array = encoded.into_array();
    for i in 0..3 {
        assert_eq!(norm_array.scalar_at(i)?, decompressed.scalar_at(i)?);
    }

    Ok(())
}
