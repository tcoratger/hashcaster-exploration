macro_rules! frobenius_table {
    () => {{
        use crate::{binary_field::BinaryField128b, matrix::Matrix};

        // Initialize the diagonal matrix and result table.
        let mut basis = Matrix::diag();
        let mut result = [[0; 128]; 128];

        // Precompute Frobenius transformations.
        for i in 0..128 {
            result[i] = basis.cols;

            // Apply Frobenius map to all columns.
            basis.cols.iter_mut().for_each(|col| {
                let x = BinaryField128b::new(*col);
                *col = (x * x).into_inner();
            });
        }

        result
    }};
}

#[cfg(test)]
pub mod tests {
    use crate::{binary_field::BinaryField128b, matrix::Matrix};

    #[test]
    fn test_frobenius_precompute_table() {
        // Fetch the precomputed Frobenius table
        let table = frobenius_table!();

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
}
