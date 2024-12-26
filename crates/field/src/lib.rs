pub mod backend;
pub mod binary_field;
pub mod bit_iterator;
pub mod frobenius;
pub mod frobenius_cobasis;
pub mod matrix;
pub mod small_uint;

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::frobenius_cobasis::{COBASIS, COBASIS_FROBENIUS};
    use binary_field::BinaryField128b;
    use frobenius::FROBENIUS;
    use frobenius_cobasis::COBASIS_FROBENIUS_TRANSPOSE;
    use matrix::Matrix;

    pub fn u128_from_bits(bits: &[bool]) -> u128 {
        assert!(bits.len() <= 128, "Bit array length exceeds u128 capacity");
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

    // #[test]
    // fn test_frobenius_cobasis() {
    //     let mut matrix = vec![vec![false; 128]; 128];
    //     for i in 0..128 {
    //         let b_i = BinaryField128b::basis(i);

    //         // compute pi_i linear function
    //         for j in 0..128 {
    //             let b_j = BinaryField128b::basis(j);
    //             let mut x = b_j * b_i;

    //             let mut s = BinaryField128b::zero();
    //             for k in 0..128 {
    //                 s += x;
    //                 x *= x;
    //             }

    //             if s == BinaryField128b::one() {
    //                 matrix[i][j] = true;
    //             }
    //         }
    //     }

    //     let mut matrix_columns = [0; 128];
    //     for i in 0..128 {
    //         matrix_columns[i] = u128_from_bits(&matrix[i]);
    //     }

    //     let matrix = Matrix::new(matrix_columns);
    //     let ret = matrix.inverse().unwrap().cols;
    // }
}
