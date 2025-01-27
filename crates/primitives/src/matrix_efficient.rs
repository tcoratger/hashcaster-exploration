use crate::{
    binary_field::BinaryField128b,
    frobenius::FROBENIUS,
    utils::{cpu_v_movemask_epi8, drop_top_bit, v_slli_epi64},
};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::{
    array,
    ops::Deref,
    sync::atomic::{AtomicU64, Ordering},
};

/// Number of rows in the binary matrix.
const NUM_ROWS: usize = 128;

/// Number of columns in the binary matrix.
const NUM_COLS: usize = 128;

/// Number of 16-bit chunks in a 128-bit value.
const CHUNK_COUNT: usize = 128 / 16;

/// Precomputed `(idx_u64, shift)` values for each chunk index.
///
/// Each tuple specifies:
/// - `idx_u64`: The index of the atomic 64-bit value.
/// - `shift`: The bit-shift required for the chunk's contribution.
const IDX_U64_SHIFT: [(usize, usize); CHUNK_COUNT] = {
    let mut table = [(0, 0); CHUNK_COUNT];
    let mut i = 0;
    while i < CHUNK_COUNT {
        table[i] = (i / 4, 16 * (i % 4));
        i += 1;
    }
    table
};

/// Represents a 128x128 binary matrix optimized using the Method of Four Russians.
///
/// Internally, the matrix is stored as precomputed sums for efficient subset operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EfficientMatrix(pub [BinaryField128b; 256 * 16]);

impl Default for EfficientMatrix {
    fn default() -> Self {
        Self([BinaryField128b::ZERO; 256 * 16])
    }
}

impl EfficientMatrix {
    /// Constructs an `EfficientMatrix` using Frobenius inverse linear combinations.
    ///
    /// This function computes a new matrix from the given `gammas` and a
    /// precomputed Frobenius table.
    ///
    /// # Parameters
    /// - `gammas`: An array of 128 elements (`BinaryField128b`) used for linear combinations.
    ///
    /// # Returns
    /// An `EfficientMatrix` constructed from the computed columns.
    pub fn from_frobenius_inv_lc(gammas: &[BinaryField128b; NUM_COLS]) -> Self {
        // Precompute `minus_i` values for all rows
        let minus_indices: [_; NUM_COLS] = array::from_fn(|i| (NUM_COLS - i) % NUM_COLS);

        // Compute the result columns using Frobenius inverse coefficients
        let ret: [_; NUM_COLS] = array::from_fn(|j| {
            minus_indices.iter().enumerate().fold(BinaryField128b::ZERO, |acc, (i, &minus_i)| {
                acc + gammas[i] * BinaryField128b::from(FROBENIUS[minus_i][j])
            })
        });

        // Construct the EfficientMatrix from the computed columns
        Self::from_cols(&ret)
    }

    /// Constructs an `EfficientMatrix` from its rows.
    ///
    /// This function processes the binary matrix row-wise, computes column contributions
    /// in parallel, and precomputes sums for subset operations.
    ///
    /// # Parameters
    /// - `rows`: A reference to an array of 128 rows, each represented as a `BinaryField128b`.
    ///
    /// # Returns
    /// An instance of `EfficientMatrix` with precomputed values.
    pub fn from_rows(rows: &[BinaryField128b; NUM_ROWS]) -> Self {
        // Initialize atomic columns for concurrent updates
        //
        // Atomic operations ensure thread-safe updates in parallel computations.
        let cols: [_; 256] = array::from_fn(|_| AtomicU64::default());

        // Transmute rows into a byte representation for SIMD processing.
        // SAFETY: Assumes `BinaryField128b` has a 16-byte memory layout.
        let rows: [[u8; 16]; NUM_ROWS] = unsafe { std::mem::transmute(*rows) };

        // Process rows in parallel, in chunks of 16 rows
        //
        // Chunks allow SIMD-friendly parallel computation over 16 rows at a time.
        rows.par_chunks_exact(16).enumerate().for_each(|(chunk_idx, chunk)| {
            // Retrieve precomputed index and shift values for the current chunk.
            let (idx_u64, shift) = IDX_U64_SHIFT[chunk_idx];

            // Process each byte in the chunk
            for i in 0..16 {
                // Extract the `i`th byte from all 16 rows to form an array `t`.
                //
                // This creates a SIMD-friendly array for bit manipulation.
                let mut t = array::from_fn(|j| chunk[j][i]);

                // Process each bit in the byte
                //
                // Iterate over all 8 bits of the byte, since each byte contains 8 bits.
                for j in 0..8 {
                    #[allow(clippy::cast_sign_loss)]
                    // Compute a bitmask representing the high bits of the array `t`.
                    let bits = (cpu_v_movemask_epi8(t) as u64) << shift;

                    // Update the atomic column value with the computed bitmask.
                    cols[2 * (8 * i + 7 - j) + idx_u64].fetch_xor(bits, Ordering::SeqCst);

                    // Shift all bits in the array `t` to the left by 1 for the next iteration.
                    t = v_slli_epi64::<1>(t);
                }
            }
        });

        // Finalize atomic column values into a regular array.
        //
        // Use relaxed memory ordering to retrieve the values safely from atomic storage.
        //
        // Transmute the atomic array back into the binary field representation.
        let cols: [_; NUM_COLS] =
            unsafe { std::mem::transmute(cols.map(|x| x.load(Ordering::Relaxed))) };

        // Construct and return the matrix from the finalized columns.
        Self::from_cols(&cols)
    }

    /// Constructs an `EfficientMatrix` from its columns.
    ///
    /// This function precomputes all possible subset sums for the binary matrix
    /// to enable efficient matrix-vector multiplication.
    ///
    /// # Parameters
    /// - `cols`: A reference to an array of 128 columns, each represented as a `BinaryField128b`.
    ///
    /// # Returns
    /// An instance of `EfficientMatrix` with precomputed subset sums.
    pub fn from_cols(cols: &[BinaryField128b; NUM_COLS]) -> Self {
        // Initialize storage for precomputed subset sums
        let mut precomp = [BinaryField128b::ZERO; 256 * 16];

        // Compute sums in parallel for each set of 8 columns
        cols.par_chunks(8).zip(precomp.par_chunks_mut(256)).for_each(|(cols, sums)| {
            // Initialize the first sum as zero
            sums[0] = BinaryField128b::ZERO;

            // Compute subset sums
            for i in 1..256 {
                let (sum_idx, row_idx) = drop_top_bit(i);
                sums[i] = sums[sum_idx] + cols[row_idx];
            }
        });

        // Return the matrix with precomputed values
        Self(precomp)
    }

    /// Applies the precomputed matrix transformation to the given input.
    ///
    /// The function uses precomputed values stored in the matrix to quickly compute
    /// the result for the given `rhs` (right-hand side) input.
    ///
    /// # Parameters
    /// - `rhs`: A `BinaryField128b` input vector to be transformed by the matrix.
    ///
    /// # Returns
    /// The result of the transformation as a `BinaryField128b`.
    pub fn apply(&self, rhs: BinaryField128b) -> BinaryField128b {
        // Transmute `rhs` into a 16-byte array for indexing.
        // SAFETY: Assumes `BinaryField128b` is 128 bits (16 bytes) with a compatible layout.
        let rhs: [u8; 16] = unsafe { std::mem::transmute(rhs) };

        // Accumulate contributions using an iterator, starting from the first element.
        rhs.iter()
            .enumerate()
            .fold(BinaryField128b::ZERO, |acc, (i, &byte)| acc + self[byte as usize + 256 * i])
    }
}

impl Deref for EfficientMatrix {
    type Target = [BinaryField128b; 256 * 16];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use rand::Rng;

    #[test]
    fn test_from_cols_all_zeros() {
        // Initialize input with all zeros.
        let cols = [BinaryField128b::ZERO; 128];

        // Call the `from_cols` function.
        let matrix = EfficientMatrix::from_cols(&cols);

        // Manually compute the expected result.
        // All precomputed sums will be zero because the input columns are all zero.
        let expected = EfficientMatrix([BinaryField128b::ZERO; 256 * 16]);

        // Assert equality.
        assert_eq!(matrix, expected, "Precomputed sums for all-zero input should be zero.");
    }

    #[test]
    fn test_from_cols_single_non_zero_column() {
        // Initialize input with one non-zero column and the rest zeros.
        let mut cols = [BinaryField128b::ZERO; 128];
        // Set only the first column to 1.
        cols[0] = BinaryField128b::from(1);

        // Call the `from_cols` function.
        let matrix = EfficientMatrix::from_cols(&cols);

        // Manually compute the expected result for a few indices.
        let mut expected_precomp = [BinaryField128b::ZERO; 256 * 16];

        // Explanation of manual computation:
        // Each index `i` corresponds to a binary number. The top bit of `i`
        // determines the subset of columns to include in the sum.
        //
        // For example:
        // - `i = 0b00000001` (1 in decimal): Includes column 0.
        // - `i = 0b00000010` (2 in decimal): Includes column 1 (which is zero in this case).
        // - `i = 0b00000011` (3 in decimal): Includes columns 0 and 1.
        //
        // Since only column 0 is non-zero, all sums involving it will equal column 0.
        for (i, precomp) in expected_precomp.iter_mut().enumerate().take(256) {
            // - If the index is even, the sum should be zero (not including column 0).
            // - If the index is odd, the sum should be column 0 (the only non-zero column).
            if i % 2 == 0 {
                *precomp = BinaryField128b::ZERO;
            } else {
                *precomp = cols[0];
            }
        }

        // Construct the expected EfficientMatrix.
        let expected = EfficientMatrix(expected_precomp);

        // Assert that the function output matches the manually computed result.
        assert_eq!(
            matrix, expected,
            "Precomputed sums for a single non-zero column are incorrect."
        );
    }

    #[test]
    #[allow(clippy::unnecessary_cast, clippy::unreadable_literal)]
    fn test_from_cols_with_queries() {
        // Initialize some non-zero columns.
        let c0 = BinaryField128b::from(568);
        let c1 = BinaryField128b::from(123);
        let c2 = BinaryField128b::from(456);
        let c3 = BinaryField128b::from(789);
        let c4 = BinaryField128b::from(1011);

        // Initialize input with some non-zero columns.
        let mut cols = [BinaryField128b::ZERO; 128];
        cols[0] = c0;
        cols[1] = c1;
        cols[2] = c2;
        cols[3] = c3;
        cols[4] = c4;

        // Call the `from_cols` function.
        let matrix = EfficientMatrix::from_cols(&cols);

        assert_eq!(matrix[0b00000001 as usize], c0);
        assert_eq!(matrix[0b00000010 as usize], c1);
        assert_eq!(matrix[0b00000011 as usize], c0 + c1);
        assert_eq!(matrix[0b00000100 as usize], c2);
        assert_eq!(matrix[0b00000101 as usize], c0 + c2);
        assert_eq!(matrix[0b00000110 as usize], c1 + c2);
        assert_eq!(matrix[0b00000111 as usize], c0 + c1 + c2);
        assert_eq!(matrix[0b00001000 as usize], c3);
        assert_eq!(matrix[0b00001001 as usize], c0 + c3);
        assert_eq!(matrix[0b00001010 as usize], c1 + c3);
        assert_eq!(matrix[0b00001011 as usize], c0 + c1 + c3);
        assert_eq!(matrix[0b00001100 as usize], c2 + c3);
        assert_eq!(matrix[0b00001101 as usize], c0 + c2 + c3);
        assert_eq!(matrix[0b00001110 as usize], c1 + c2 + c3);
        assert_eq!(matrix[0b00001111 as usize], c0 + c1 + c2 + c3);
        assert_eq!(matrix[0b00010000 as usize], c4);
        assert_eq!(matrix[0b00010001 as usize], c0 + c4);
        assert_eq!(matrix[0b00010010 as usize], c1 + c4);
        assert_eq!(matrix[0b00010011 as usize], c0 + c1 + c4);
        assert_eq!(matrix[0b00010100 as usize], c2 + c4);
        assert_eq!(matrix[0b00010101 as usize], c0 + c2 + c4);
        assert_eq!(matrix[0b00010110 as usize], c1 + c2 + c4);
        assert_eq!(matrix[0b00010111 as usize], c0 + c1 + c2 + c4);
        assert_eq!(matrix[0b00011000 as usize], c3 + c4);
    }

    #[test]
    fn test_from_rows_all_zeros() {
        // Initialize input rows with all zeros.
        let rows = [BinaryField128b::ZERO; 128];

        // Call the `from_rows` function.
        let matrix = EfficientMatrix::from_rows(&rows);

        // Manually compute the expected result.
        // All columns should also be zero since the input rows are all zero.
        let expected = EfficientMatrix([BinaryField128b::ZERO; 256 * 16]);

        // Assert equality.
        assert_eq!(matrix, expected, "Resulting matrix should be all zeros.");
    }

    #[test]
    fn test_from_rows_all_ones() {
        // Initialize input rows with one non-zero row.
        // In little-endian binary representation the matrix will look like:
        // [1, 0, 0, ..., 0, , 0, 0, ..., 0]
        // [1, 0, 0, ..., 0, , 0, 0, ..., 0]
        // [1, 0, 0, ..., 0, , 0, 0, ..., 0]
        // ...
        let rows = [BinaryField128b::from(1); 128];

        // Call the `from_rows` function.
        let matrix = EfficientMatrix::from_rows(&rows);

        // Initialize input columns with all zeros.
        let mut cols = [BinaryField128b::ZERO; 128];
        // Set only the first column to u128::MAX
        // The objective is to match the following matrix:
        // [1, 0, 0, ..., 0, , 0, 0, ..., 0]
        // [1, 0, 0, ..., 0, , 0, 0, ..., 0]
        // [1, 0, 0, ..., 0, , 0, 0, ..., 0]
        // ...
        cols[0] = BinaryField128b::from(u128::MAX);

        // Manually compute the expected result.
        let expected = EfficientMatrix::from_cols(&cols);

        // Assert equality.
        assert_eq!(matrix, expected, "Resulting matrix does not match the expected result.");
    }

    #[test]
    fn test_from_rows_single_element_in_first_row() {
        // Initialize rows with all zeros.
        let mut rows = [BinaryField128b::ZERO; 128];
        // Set only the first row's first element to 1.
        // In little-endian binary representation the matrix will look like:
        // [1, 0, 0, ..., 0, , 0, 0, ..., 0]
        // [0, 0, 0, ..., 0, , 0, 0, ..., 0]
        // [0, 0, 0, ..., 0, , 0, 0, ..., 0]
        // ...
        rows[0] = BinaryField128b::from(1);

        // Call the `from_rows` function.
        let matrix = EfficientMatrix::from_rows(&rows);

        // Manually compute the expected result:
        // Only the first column will have the first bit set (1).
        let mut cols = [BinaryField128b::ZERO; 128];
        // In little-endian binary representation the matrix will look like:
        // [1, 0, 0, ..., 0, , 0, 0, ..., 0]
        // [0, 0, 0, ..., 0, , 0, 0, ..., 0]
        // [0, 0, 0, ..., 0, , 0, 0, ..., 0]
        // ...
        cols[0] = BinaryField128b::from(1);

        // Construct the expected matrix.
        let expected = EfficientMatrix::from_cols(&cols);

        // Assert equality.
        assert_eq!(
            matrix, expected,
            "Resulting matrix does not match expected with single non-zero element in the first row."
        );
    }

    #[test]
    fn test_from_rows_alternate_rows_full_128_bits() {
        // Create 128-bit alternating patterns:
        // - Even-indexed rows: 128 bits of alternating 10101010...
        // - Odd-indexed rows: 128 bits of alternating 01010101...
        let even_pattern = 0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA;
        let odd_pattern = 0x5555_5555_5555_5555_5555_5555_5555_5555;

        // Repeat each pattern to fill 128 bits (already fits for u128).
        // In little-endian binary representation the matrix will look like:
        // [1, 0, 1, 0, ..., 1, 0, 1, 0]
        // [0, 1, 0, 1, ..., 0, 1, 0, 1]
        // [1, 0, 1, 0, ..., 1, 0, 1, 0]
        // ...
        let rows: Vec<_> = (0..128)
            .map(|i| {
                if i % 2 == 0 {
                    // Even rows: 10101010...
                    even_pattern.into()
                } else {
                    // Odd rows: 01010101...
                    odd_pattern.into()
                }
            })
            .collect();

        // Call the `from_rows` function.
        let matrix = EfficientMatrix::from_rows(&rows.try_into().unwrap());

        // Compute expected column values:
        // In little-endian binary representation the matrix will look like:
        // [1, 0, 1, 0, ..., 1, 0, 1, 0]
        // [0, 1, 0, 1, ..., 0, 1, 0, 1]
        // [1, 0, 1, 0, ..., 1, 0, 1, 0]
        // ...
        let cols: Vec<_> = (0..128)
            .map(|i| {
                if i % 2 == 0 {
                    // Even columns: 10101010...
                    even_pattern.into()
                } else {
                    // Odd columns: 01010101...
                    odd_pattern.into()
                }
            })
            .collect();

        // Construct the expected matrix.
        let expected = EfficientMatrix::from_cols(&cols.try_into().unwrap());

        // Assert equality.
        assert_eq!(
            matrix, expected,
            "Resulting matrix does not match expected with full 128-bit alternating row patterns."
        );
    }

    #[test]
    fn test_from_rows_all_rows_max() {
        // Initialize input rows with all ones.
        // In little-endian binary representation the matrix will look like:
        // [1, 1, 1, ..., 1, , 1, 1, ..., 1]
        // [1, 1, 1, ..., 1, , 1, 1, ..., 1]
        // [1, 1, 1, ..., 1, , 1, 1, ..., 1]
        // ...
        let rows = vec![BinaryField128b::from(u128::MAX); 128];

        // Call the `from_rows` function.
        let matrix = EfficientMatrix::from_rows(&rows.try_into().unwrap());

        // Initialize input columns with all ones.
        // In little-endian binary representation the matrix will look like:
        // [1, 1, 1, ..., 1, , 1, 1, ..., 1]
        // [1, 1, 1, ..., 1, , 1, 1, ..., 1]
        // [1, 1, 1, ..., 1, , 1, 1, ..., 1]
        // ...
        let cols = [BinaryField128b::from(u128::MAX); 128];

        // Manually compute the expected result.
        let expected = EfficientMatrix::from_cols(&cols);

        // Assert equality.
        assert_eq!(matrix, expected, "Resulting matrix does not match the expected result.");
    }

    #[test]
    fn test_apply_all_zeros() {
        // Create an EfficientMatrix filled with zeros.
        let matrix = EfficientMatrix::default();

        // Input vector filled with zeros.
        let input = BinaryField128b::ZERO;

        // Applying the matrix should result in zero output.
        let result = matrix.apply(input);

        // Expected output is zero.
        assert_eq!(
            result,
            BinaryField128b::ZERO,
            "Applying all-zero matrix to zero vector should result in zero."
        );
    }

    #[test]
    fn test_apply_single_nonzero_byte() {
        // Create an EfficientMatrix with a single non-zero value.
        let mut matrix_data = [BinaryField128b::ZERO; 256 * 16];
        matrix_data[1] = BinaryField128b::from(42); // Set a specific value.
        let matrix = EfficientMatrix(matrix_data);

        // Input vector with the first byte set to 1.
        let input = BinaryField128b::from(1);

        // Applying the matrix should return the value at index 1 in the matrix.
        let result = matrix.apply(input);

        // Expected output is the value set earlier.
        assert_eq!(
            result,
            BinaryField128b::from(42),
            "Applying matrix with single non-zero entry should return that entry."
        );
    }

    #[test]
    fn test_apply_against_traditional_matrix() {
        // Setup random columns of u128
        let mut rng = rand::thread_rng();
        let cols: [_; 128] = array::from_fn(|_| rng.gen::<u128>());

        // Create an EfficientMatrix from the columns
        let matrix = EfficientMatrix::from_cols(&cols.map(BinaryField128b::from));

        // Create a traditional matrix from the columns (assuming Matrix::new exists)
        let traditional_matrix = Matrix::new(cols);

        // Generate a random u128 input vector
        let rhs = rng.gen::<u128>();

        // Apply the EfficientMatrix to the input
        let result_efficient = matrix.apply(BinaryField128b::from(rhs));

        // Apply the traditional matrix to the input
        let result_traditional = traditional_matrix.apply(rhs);

        // Assert that the results are equal
        assert_eq!(
            result_efficient,
            BinaryField128b::from(result_traditional),
            "EfficientMatrix and traditional matrix gave different results"
        );
    }

    #[test]
    fn test_frobenius_inv_lc_all_zeros() {
        // Initialize `gammas` with all zeros
        let gammas = [BinaryField128b::ZERO; NUM_COLS];

        // Create the matrix
        let matrix = EfficientMatrix::from_frobenius_inv_lc(&gammas);

        // Expected result: all columns should also be zero
        let expected = EfficientMatrix::default();

        // Assert equality
        assert_eq!(matrix, expected, "Matrix should be all zeros when gammas are zero.");
    }

    #[test]
    fn test_frobenius_inv_lc_single_gamma() {
        // Initialize `gammas` with a single non-zero element
        let mut gammas = [BinaryField128b::ZERO; NUM_COLS];
        // Set the first gamma to be non-zero
        gammas[0] = BinaryField128b::from(5467);

        // Create the matrix
        let matrix = EfficientMatrix::from_frobenius_inv_lc(&gammas);

        // Expected result: each column should be the corresponding Frobenius value
        let expected_cols: [_; NUM_COLS] = array::from_fn(|j| {
            BinaryField128b::from(5467) * BinaryField128b::from(FROBENIUS[0][j])
        });

        let expected = EfficientMatrix::from_cols(&expected_cols);

        // Assert equality
        assert_eq!(
            matrix, expected,
            "Matrix should match the expected result for a single non-zero gamma."
        );
    }

    #[test]
    #[allow(clippy::large_stack_frames)]
    fn test_frobenius_inv_lc_multiple_nonzero_gammas() {
        // Initialize `gammas` with multiple non-zero elements
        let mut gammas = [BinaryField128b::ZERO; NUM_COLS];
        // First gamma
        gammas[0] = BinaryField128b::from(383);
        // Second gamma
        gammas[1] = BinaryField128b::from(463);

        // Create the matrix
        let matrix = EfficientMatrix::from_frobenius_inv_lc(&gammas);

        // Expected result: compute each column based on multiple gammas
        let expected_cols: [_; NUM_COLS] = array::from_fn(|j| {
            BinaryField128b::from(383) * BinaryField128b::from(FROBENIUS[0][j]) +
                BinaryField128b::from(463) * BinaryField128b::from(FROBENIUS[127][j])
        });

        // Construct the expected matrix
        let expected = EfficientMatrix::from_cols(&expected_cols);

        // Assert equality
        assert_eq!(
            matrix, expected,
            "Matrix should match the expected result for multiple non-zero gammas."
        );
    }

    #[test]
    fn test_frobenius_inv_lc_wraparound_indices() {
        // Initialize `gammas` with the last element non-zero
        let mut gammas = [BinaryField128b::ZERO; NUM_COLS];
        // Last gamma
        gammas[127] = BinaryField128b::from(5363);

        // Create the matrix
        let matrix = EfficientMatrix::from_frobenius_inv_lc(&gammas);

        // Expected result: compute each column using wraparound Frobenius indices
        // Wraparound for minus_i = 1
        let expected_cols: [_; NUM_COLS] = array::from_fn(|j| {
            BinaryField128b::from(5363) * BinaryField128b::from(FROBENIUS[1][j])
        });

        // Construct the expected matrix
        let expected = EfficientMatrix::from_cols(&expected_cols);

        // Assert equality
        assert_eq!(
            matrix, expected,
            "Matrix should match the expected result for wraparound indices."
        );
    }
}
