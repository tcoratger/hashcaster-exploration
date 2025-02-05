#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
pub mod backend;
pub mod binary_field;
pub mod bit_iterator;
pub mod frobenius;
pub mod frobenius_cobasis;
pub mod linear_trait;
pub mod matrix;
pub mod matrix_efficient;
pub mod matrix_lin;
pub mod poly;
pub mod sumcheck;
pub mod utils;

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::frobenius_cobasis::{COBASIS, COBASIS_FROBENIUS};
    use binary_field::BinaryField128b;
    use frobenius::FROBENIUS;
    use frobenius_cobasis::COBASIS_FROBENIUS_TRANSPOSE;
    use matrix::Matrix;

    /// Converts a slice of booleans to a `u128` integer.
    pub fn u128_from_bits(bits: &[bool]) -> u128 {
        // Ensure the bit array length does not exceed the capacity of a `u128`.
        assert!(bits.len() <= 128, "Bit array length exceeds u128 capacity");
        // Fold the bits into a `u128` integer, setting each bit from the slice.
        bits.iter().enumerate().fold(0, |acc, (i, &bit)| acc | ((u128::from(bit)) << i))
    }

    #[test]
    fn test_frobenius_precompute_table() {
        // Fetch the precomputed Frobenius table
        let table = FROBENIUS;

        // Initialize the basis as a diagonal matrix of powers of 2
        let mut basis = Matrix::diag();

        // Verify each Frobenius transformation and its precomputed counterpart
        for (i, precomputed_row) in table.iter().enumerate() {
            // Check that the current state of the basis matches the precomputed row
            assert_eq!(precomputed_row, &basis.cols, "Mismatch at row {i}");

            // Apply the Frobenius transformation (square each field element)
            basis.cols.iter_mut().for_each(|col| {
                let x = BinaryField128b::new(*col);
                *col = (x * x).into_inner();
            });
        }

        // Ensure the basis returns to its initial diagonal state
        assert_eq!(basis, Matrix::diag(), "Basis did not return to its original state");
    }

    #[test]
    fn test_frobenius_cobasis_precompute_table() {
        // Fetch the precomputed Frobenius cobasis table
        let table = COBASIS_FROBENIUS;

        // Initialize the basis as a diagonal matrix of powers of 2
        let mut cobasis = Matrix::new(COBASIS);

        // Verify each Frobenius transformation and its precomputed counterpart
        for (i, precomputed_row) in table.iter().enumerate() {
            // Check that the current state of the cobasis matches the precomputed row
            assert_eq!(precomputed_row, &cobasis.cols, "Mismatch at row {i}");

            // Apply the Frobenius transformation (square each field element)
            cobasis.cols.iter_mut().for_each(|col| {
                let x = BinaryField128b::new(*col);
                *col = (x * x).into_inner();
            });
        }

        // Ensure the cobasis returns to its initial state
        assert_eq!(cobasis, Matrix::new(COBASIS), "Basis did not return to its original state");
    }

    #[test]
    #[allow(clippy::large_stack_frames)]
    fn test_frobenius_cobasis_precompute_table_transpose() {
        // Fetch the precomputed Frobenius cobasis table
        let table = COBASIS_FROBENIUS;

        // Fetch the precomputed Frobenius cobasis transpose table
        let table_transpose = COBASIS_FROBENIUS_TRANSPOSE;

        // Compute the expected transpose of the Frobenius cobasis table
        let expected_table_transpose: [[u128; 128]; 128] = {
            let mut table_transpose = [[0; 128]; 128];
            for (i, t) in table.iter().enumerate() {
                for (j, tt) in table_transpose.iter_mut().enumerate() {
                    tt[i] = t[j];
                }
            }
            table_transpose
        };

        // Ensure the precomputed Frobenius cobasis transpose table matches the expected transpose
        assert_eq!(table_transpose, expected_table_transpose);
    }

    #[test]
    fn test_frobenius_cobasis() {
        // Initialize a 128x128 boolean matrix with all values set to `false`.
        // Each row will correspond to a basis vector, and `true` will represent a set bit.
        let mut matrix = [[false; 128]; 128];

        // Iterate over all rows of the matrix to compute the cobasis matrix.
        // `i` represents the index of the current row and corresponds to a basis element \( b_i \).
        matrix.iter_mut().enumerate().for_each(|(i, row)| {
            // Compute the \( i \)-th basis element \( b_i \) in \( \mathbb{F}_{2^{128}} \).
            let b_i = BinaryField128b::basis(i);

            // Iterate over each column of the current row to compute \( \pi_i(b_j) \).
            // `j` represents the index of the current column and corresponds to a basis element \(
            // b_j \).
            row.iter_mut().enumerate().for_each(|(j, cell)| {
                // Compute the product \( x = b_j \cdot b_i \).
                // This initializes the Frobenius transformation for \( b_j \) with respect to \(
                // b_i \).
                let mut x = BinaryField128b::basis(j) * b_i;

                // Initialize the accumulator \( s \) to zero.
                // \( s \) will accumulate the sum of Frobenius transformations \( x^{2^k} \) over
                // 128 iterations.
                let mut s = BinaryField128b::ZERO;

                // Apply the Frobenius map iteratively: \( x \mapsto x^2 \), 128 times.
                // This corresponds to summing \( x^{2^k} \) for \( k = 0 \) to \( 127 \).
                for _ in 0..128 {
                    // Add the current \( x \) to the accumulator \( s \).
                    s += x;
                    // Update \( x \) to \( x^2 \) (Frobenius map).
                    x *= x;
                }

                // If the accumulated value \( s \) equals 1, set the matrix cell to `true`.
                // This ensures \( \pi_i(b_j) = 1 \) for the correct pairing.
                *cell = s == BinaryField128b::ONE;
            });
        });

        // Convert each row of the boolean matrix into a `u128` representation.
        // Each row is treated as a bit vector, where `true` corresponds to a `1` bit.
        let matrix_columns: [u128; 128] = matrix
            .iter()
            .map(|row| u128_from_bits(row)) // Convert each row to a `u128`.
            .collect::<Vec<_>>()
            .try_into() // Ensure the result is a fixed-size array of 128 elements.
            .expect("Failed to convert to array");

        // Create a `Matrix` instance from the 128 `u128` columns.
        // This prepares the cobasis matrix for inversion.
        let matrix = Matrix::new(matrix_columns);

        // Compute the inverse of the cobasis matrix.
        // The inversion ensures that the relationship \( b_i = \sum_j r_j \pi_j \) holds.
        let ret = matrix.inverse().unwrap().cols;

        // Verify that the computed cobasis matrix matches the precomputed `COBASIS`.
        // This confirms that the cobasis was correctly constructed and satisfies theoretical
        // expectations.
        assert_eq!(ret, COBASIS);
    }
}
