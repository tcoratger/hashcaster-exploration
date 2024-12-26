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

    /// Swaps two columns in the matrix.
    ///
    /// This function exchanges the contents of the `i`-th and `j`-th columns of the matrix.
    /// The operation is performed in place and is efficient even for large matrices.
    ///
    /// # Parameters
    /// - `i`: The index of the first column to be swapped.
    /// - `j`: The index of the second column to be swapped.
    ///
    /// # Panics
    /// - This function does not explicitly check for out-of-bounds indices. Passing indices outside
    ///   the range `[0, 127]` for the 128x128 matrix will result in undefined behavior.
    ///
    /// # Safety
    /// - The function uses `unsafe` code to bypass Rust's borrowing rules, ensuring efficient
    ///   swapping. However, it guarantees correctness as long as the indices `i` and `j` are valid.
    ///
    /// # Complexity
    /// - \(O(1)\): The swap is performed in constant time.
    pub fn swap_cols(&mut self, i: usize, j: usize) {
        // Check if the indices are different to avoid unnecessary operations.
        if i != j {
            // Unsafe block is required for `std::ptr::swap` to allow mutable access
            // to two separate elements of the same array without violating borrowing rules.
            unsafe {
                // Perform the swap between the `i`-th and `j`-th columns.
                std::ptr::swap(&mut self.cols[i], &mut self.cols[j]);
            }
        }
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

    #[test]
    fn test_swap_cols_basic() {
        let mut matrix = Matrix::diag();

        // Swap two columns
        matrix.swap_cols(0, 1);

        // Check that the columns have been swapped
        assert_eq!(matrix.cols[0], 1 << 1, "Column 0 should contain the value of Column 1.");
        assert_eq!(matrix.cols[1], 1 << 0, "Column 1 should contain the value of Column 0.");
    }

    #[test]
    fn test_swap_cols_noop() {
        let mut matrix = Matrix::diag();

        // Swap a column with itself
        matrix.swap_cols(3, 3);

        // Ensure the matrix remains unchanged
        for i in 0..128 {
            assert_eq!(matrix.cols[i], 1 << i, "Column {i} should remain unchanged.");
        }
    }

    #[test]
    fn test_swap_cols_random_indices() {
        let mut matrix = Matrix::diag();

        // Swap two arbitrary columns
        matrix.swap_cols(5, 10);

        // Check that the swap was successful
        assert_eq!(matrix.cols[5], 1 << 10, "Column 5 should contain the value of Column 10.");
        assert_eq!(matrix.cols[10], 1 << 5, "Column 10 should contain the value of Column 5.");

        // Ensure other columns remain unchanged
        for i in 0..128 {
            if i != 5 && i != 10 {
                assert_eq!(matrix.cols[i], 1 << i, "Column {i} should remain unchanged.");
            }
        }
    }

    #[test]
    fn test_swap_cols_edge_indices() {
        let mut matrix = Matrix::diag();

        // Swap the first and last columns
        matrix.swap_cols(0, 127);

        // Check that the swap was successful
        assert_eq!(matrix.cols[0], 1 << 127, "Column 0 should contain the value of Column 127.");
        assert_eq!(matrix.cols[127], 1 << 0, "Column 127 should contain the value of Column 0.");

        // Ensure other columns remain unchanged
        for i in 1..127 {
            assert_eq!(matrix.cols[i], 1 << i, "Column {i} should remain unchanged.");
        }
    }

    #[test]
    fn test_swap_cols_multiple_swaps() {
        let mut matrix = Matrix::diag();

        // Perform multiple swaps
        matrix.swap_cols(0, 5);
        matrix.swap_cols(5, 10);
        matrix.swap_cols(10, 0);

        // Check the final state of the swapped columns
        assert_eq!(matrix.cols[0], 1);
        assert_eq!(matrix.cols[5], 1 << 10);
        assert_eq!(matrix.cols[10], 1 << 5);

        // Ensure other columns remain unchanged
        for i in 0..128 {
            if i != 0 && i != 5 && i != 10 {
                assert_eq!(matrix.cols[i], 1 << i, "Column {i} should remain unchanged.");
            }
        }
    }
}
