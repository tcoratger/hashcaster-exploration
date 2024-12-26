/// Represents a mathematical matrix with columns represented as 128-bit unsigned integers.
///
/// The `Matrix` struct is specifically designed for operations on binary matrices.
/// Each column is a `u128`, where each bit represents a single binary value in the matrix.
///
/// # Fields
/// - `cols`: An array of 128 columns, each represented by a 128-bit unsigned integer (`u128`).
///   - Each bit in a column represents a binary value in the matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix {
    /// The columns of the matrix, where each column is a `u128`.
    ///
    /// - Each `u128` contains 128 binary values, one per bit.
    /// - The total size of the matrix is 128x128 (128 columns, 128 bits per column).
    pub cols: [u128; 128],
}

impl Matrix {
    /// Creates a new `Matrix` from an array of 128 columns.
    ///
    /// # Parameters
    /// - `cols`: An array of 128 `u128` values, each representing a column of the matrix.
    ///
    /// # Returns
    /// A `Matrix` with the specified columns.
    ///
    /// # Panics
    /// None. Assumes that `cols` always has a length of 128.
    pub const fn new(cols: [u128; 128]) -> Self {
        Self { cols }
    }

    /// Generates a diagonal matrix where the diagonal entries are powers of 2.
    ///
    /// The diagonal matrix has the following properties:
    /// - Each column contains a single `1` (in binary) at a unique row.
    /// - The `1`s are aligned along the diagonal of the matrix.
    ///
    /// For example:
    /// - Column 0: `0b000...0001`
    /// - Column 1: `0b000...0010`
    /// - Column 2: `0b000...0100`
    /// - ...
    /// - Column 127: `0b100...0000`
    ///
    /// # Returns
    /// A diagonal `Matrix` with `1`s on the diagonal.
    pub const fn diag() -> Self {
        // Initialize an array to store the diagonal matrix columns.
        let mut cols = [0; 128];

        // Start with the first diagonal value (2^0 = 1).
        let mut value = 1;

        // Initialize the column index.
        let mut i = 0;

        // Loop to populate the columns with diagonal entries.
        while i < 128 {
            // Set the `i`-th column to the current diagonal value.
            cols[i] = value;

            // Shift the value to the left (multiply by 2) for the next diagonal entry.
            value <<= 1;

            // Increment the column index.
            i += 1;
        }

        // Create and return the matrix with the diagonal columns.
        Self::new(cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diag() {
        // Create the diagonal matrix using the diag function
        let diag_matrix = Matrix::diag();

        // Ensure the diagonal matrix has the correct number of columns
        assert_eq!(diag_matrix.cols.len(), 128);

        // Check that each element is a power of 2
        for (i, col) in diag_matrix.cols.iter().enumerate() {
            assert_eq!(*col, 1 << i, "Column {i} did not match expected value.");
        }
    }
}
