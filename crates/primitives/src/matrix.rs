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

    /// Applies a 128x128 matrix in column form to a 128-bit vector.
    ///
    /// This function computes the product of a binary matrix (128x128) and a binary vector (128
    /// bits) using operations in GF(2) (binary field arithmetic). Each bit in the resulting
    /// vector is determined by XOR-ing the appropriate entries of the matrix columns based on
    /// the bits in the input vector.
    ///
    /// ### Parameters
    /// - `self`: A reference to the `Matrix`, where each column of the 128x128 binary matrix is
    ///   represented as a `u128`. Each bit in a column represents a row.
    /// - `vec`: A 128-bit unsigned integer (`u128`), where each bit represents an element of the
    ///   input vector.
    ///
    /// ### Returns
    /// A 128-bit unsigned integer (`u128`), representing the resulting vector after applying the
    /// matrix to the input vector.
    ///
    /// ### Example
    /// Let's take a simple 4x4 matrix and a 4-bit vector for illustration purposes (even though the
    /// actual implementation works with 128x128 matrices and 128-bit vectors):
    ///
    /// Matrix (in column form):
    /// ```text
    /// Column 0: 1011 (binary)
    /// Column 1: 1100 (binary)
    /// Column 2: 0111 (binary)
    /// Column 3: 1001 (binary)
    /// ```
    ///
    /// Vector:
    /// ```text
    /// vec = 1010 (binary)
    /// ```
    ///
    /// Steps to compute the result:
    /// - Start with a result vector `ret = 0000` (binary).
    /// - For each bit in `vec`:
    ///   - If `vec[0] = 1`, XOR `ret` with Column 0: `ret = 0000 XOR 1011 = 1011`.
    ///   - If `vec[1] = 0`, skip Column 1.
    ///   - If `vec[2] = 1`, XOR `ret` with Column 2: `ret = 1011 XOR 0111 = 1100`.
    ///   - If `vec[3] = 0`, skip Column 3.
    ///
    /// Final result:
    /// ```text
    /// ret = 1100 (binary)
    /// ```
    pub fn apply(&self, vec: u128) -> u128 {
        // Initialize the result vector as 0.
        let mut ret = 0;

        // Iterate over each column of the matrix and check the corresponding bit in the input
        // vector.
        for (i, &col) in self.cols.iter().enumerate() {
            // Extract the i-th bit of the vector and create a mask.
            // - Creates `!0` if the bit is 1,
            // - Creates `0` otherwise.
            let mask = (vec >> i & 1).wrapping_neg();

            // XOR the column into the result if the bit is set.
            ret ^= col & mask;
        }

        // Return the accumulated result vector.
        ret
    }

    /// Performs a triangular XOR operation on two columns of the matrix.
    ///
    /// This function modifies the `j`-th column of the matrix by XOR-ing it with the `i`-th column:
    /// ```text
    /// self.cols[j] = self.cols[j] XOR self.cols[i]
    /// ```
    ///
    /// ### Parameters
    /// - `i`: The index of the source column used for the XOR operation.
    /// - `j`: The index of the target column that will be updated.
    ///
    /// ### Purpose
    /// This operation is commonly used in Gaussian elimination to eliminate elements below or above
    /// the pivot in a specific column, as part of transforming the matrix into row echelon form.
    ///
    /// ### Examples
    /// - **Eliminating a column**: Given two columns: ```text Column i: 1010 Column j: 1100 ```
    ///   After calling `triang(0, 1)`, column `j` becomes: ```text Column j: 1100 XOR 1010 =
    ///   0110```
    pub fn triang(&mut self, i: usize, j: usize) {
        self.cols[j] ^= self.cols[i];
    }

    /// Computes the inverse of a 128x128 binary matrix, if it exists.
    ///
    /// This function uses Gaussian elimination over the binary field (GF(2)) to compute the inverse
    /// of the given matrix. The process consists of two main phases:
    /// 1. Forward elimination: Converts the matrix into an upper triangular form.
    /// 2. Backward elimination: Converts the matrix into a diagonal (identity) form.
    ///
    /// If the matrix is invertible, the inverse is returned. Otherwise, `None` is returned.
    ///
    /// ### Returns
    /// - `Some(Self)`: The inverse of the matrix, represented as a `Matrix` struct.
    /// - `None`: If the matrix is not invertible.
    pub fn inverse(&self) -> Option<Self> {
        // Clone the original matrix `self` into `a`, which will be manipulated during the process.
        let mut a = self.clone();

        // Create an identity matrix `b`, which will be transformed into the inverse of `a`.
        let mut b = Self::diag();

        // Phase 1: Forward elimination to create an upper triangular matrix
        for i in 0..128 {
            // Check if the pivot (diagonal element) in column `i` is 0.
            if (a.cols[i] >> i) & 1 == 0 {
                // Search for a row below with a 1 in the current column.
                let pivot = (i + 1..128).find(|&j| (a.cols[j] >> i) & 1 != 0)?;

                // Swap the current row with the pivot row in both `a` and `b`.
                a.cols.swap(i, pivot);
                b.cols.swap(i, pivot);
            }

            // The `i`-th row is now the pivot row. Use it to eliminate entries below the pivot.
            let row_mask = a.cols[i]; // The pivot row in `a`.
            let inv_mask = b.cols[i]; // The corresponding row in `b`.
            for j in i + 1..128 {
                // If the `j`-th row in column `i` has a 1:
                // - Eliminate it by XORing with the pivot row.
                if (a.cols[j] >> i) & 1 != 0 {
                    a.cols[j] ^= row_mask; // Eliminate the element in `a`.
                    b.cols[j] ^= inv_mask; // Apply the same transformation to `b`.
                }
            }
        }

        // Phase 2: Backward elimination to reduce to the identity matrix
        for i in (1..128).rev() {
            // The `i`-th row is now the pivot row. Use it to eliminate entries above the pivot.
            let row_mask = a.cols[i]; // The pivot row in `a`.
            let inv_mask = b.cols[i]; // The corresponding row in `b`.
            for j in 0..i {
                // If the `j`-th row in column `i` has a 1:
                // - Eliminate it by XORing with the pivot row.
                if (a.cols[j] >> i) & 1 != 0 {
                    a.cols[j] ^= row_mask; // Eliminate the element in `a`.
                    b.cols[j] ^= inv_mask; // Apply the same transformation to `b`.
                }
            }
        }

        // At this point:
        // - `a` has been transformed into the identity matrix,
        // - `b` is now the inverse of the original matrix.
        Some(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::OsRng, Rng, RngCore};

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

    #[test]
    fn test_invert_matrix() {
        // Create a random number generator using OsRng.
        let rng = &mut OsRng;

        // Initialize the matrix as a diagonal matrix (identity matrix).
        // This ensures the starting matrix is invertible.
        let mut matrix = Matrix::diag();

        // Apply 100,000 random triangular transforms and swaps to generate
        // a random invertible matrix.
        for _ in 0..100_000 {
            // Generate a random u64 value and cast it to usize for indexing.
            let r = rng.next_u64() as usize;

            // Extract components from the random value:
            // - r1 determines the type of operation (swap or XOR).
            // - r2 and r3 are indices for selecting columns to operate on.
            let r1 = r % (1 << 4);
            let r2 = (r >> 4) & 127;
            let r3 = (r >> 32) & 127;

            if r1 == 0 {
                // If r1 is 0, perform a column swap between r2 and r3.
                matrix.swap_cols(r2, r3);
            } else {
                // Otherwise, perform a triangular XOR operation if r2 != r3.
                if r2 != r3 {
                    matrix.triang(r2, r3);
                }
            }
        }

        // Generate a random 128-bit test vector.
        let test_vector = rng.gen::<u128>();

        // Compute the inverse of the matrix.
        // If the matrix is not invertible, this will panic due to unwrap.
        let inv = matrix.inverse().unwrap();

        // Verify that applying the inverse matrix to the test vector,
        // then applying the original matrix, results in the original vector.
        assert_eq!(matrix.apply(inv.apply(test_vector)), test_vector);

        // Optionally, verify that the composition of the matrix and its inverse
        // produces the identity matrix.
        // This line is commented out but can be uncommented for additional checks.
        // assert_eq!(inv.compose(&matrix), Matrix::diag());
    }

    #[test]
    fn test_apply_single_column_active() {
        // Matrix with only the first column set (all bits active in the first column).
        let matrix = {
            let mut cols = [0; 128];
            // First column has bits [1, 1, 1, 1]
            cols[0] = 0b1111;
            Matrix::new(cols)
        };

        // Input vector with only the first bit active.
        // Binary: [1, 0, 0, 0, ...]
        let vec = 0b1;

        // Expected result: The first column of the matrix.
        // Only the first column contributes.
        let expected = 0b1111;

        // Apply the matrix to the vector and check the result.
        let result = matrix.apply(vec);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_apply_multiple_columns_active() {
        // Matrix with specific columns set.
        let matrix = {
            let mut cols = [0; 128];
            // First column: [1, 0, 1, 0]
            cols[0] = 0b1010;
            // Second column: [0, 1, 0, 1]
            cols[1] = 0b0101;
            Matrix::new(cols)
        };

        // Input vector with the first two bits active.
        // Binary: [1, 1, 0, 0, ...]
        let vec = 0b11;

        // Expected result: XOR of the first and second columns.
        // 0b1010 XOR 0b0101 = 0b1111.
        let expected = 0b1111;

        // Apply the matrix to the vector and check the result.
        let result = matrix.apply(vec);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_apply_no_columns_active() {
        // Matrix with arbitrary values (doesn't matter since vec = 0).
        let matrix = {
            let mut cols = [0; 128];
            // Arbitrary column data.
            cols[0] = 0b1010;
            cols[1] = 0b0101;
            Matrix::new(cols)
        };

        // Input vector with no bits active.
        // Binary: [0, 0, 0, 0, ...]
        let vec = 0b0;

        // Expected result: All bits in the output are 0.
        let expected = 0b0;

        // Apply the matrix to the vector and check the result.
        let result = matrix.apply(vec);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_apply_edge_case() {
        // Matrix with the first and last columns set.
        let matrix = {
            let mut cols = [0; 128];
            // First column: [1, 0, 1, 0]
            cols[0] = 0b1010;
            // Last column: [1, 1, 1, 1]
            cols[127] = 0b1111;
            Matrix::new(cols)
        };

        // Input vector with the first and last bits active.
        // Binary: [1, 0, ..., 0, 1]
        let vec = 0b1 | (1 << 127);

        // Expected result: XOR of the first and last columns.
        // 0b1010 XOR 0b1111 = 0b0101.
        let expected = 0b101;

        // Apply the matrix to the vector and check the result.
        let result = matrix.apply(vec);
        assert_eq!(result, expected);
    }
}
