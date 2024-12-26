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
    use matrix::Matrix;

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
}
