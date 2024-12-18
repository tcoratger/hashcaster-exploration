use crate::{bool_trait::CompressedFoldedOps, package::BooleanPackage, BoolCheck};
use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::{
    multinear_lagrangian::MultilinearLagrangianPolynomials, univariate::UnivariatePolynomial,
};
use num_traits::{identities::Zero, One};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[derive(Clone, Debug, Default)]
pub struct BoolCheckBuilder<const M: usize> {
    c: usize,
    points: Vec<BinaryField128b>,
    boolean_package: BooleanPackage,
    gammas: Vec<BinaryField128b>,
}

impl<const M: usize> BoolCheckBuilder<M> {
    pub fn new(c: usize, points: Vec<BinaryField128b>, boolean_package: BooleanPackage) -> Self {
        assert!(c < points.len());

        Self { c, points, boolean_package, gammas: vec![] }
    }

    /// This function calculates two mappings for a ternary (base-3) representation of integers:
    /// - **Trit Mapping**: Maps every integer `i` to either a power of 3 or a scaled binary
    ///   equivalent, depending on the digits in its base-3 representation.
    /// - **Bit Mapping**: Reversely maps binary numbers to their ternary equivalents with only the
    ///   digits `0` and `1`.
    ///
    /// ### Mathematical Principle
    /// For an integer `i`:
    /// - **Trit Mapping Logic**:
    ///   - If the ternary representation of `i` contains a `2`, the mapping is determined by the
    ///     position (most significant digit) of the first `2`: `trit_mapping[i] = 3^k`, where `k`
    ///     is the index of the most significant `2`.
    ///   - If no `2` is present, the mapping is twice the binary equivalent of the ternary number:
    ///     `trit_mapping[i] = 2 * b`, where `b` is the binary equivalent of `i`.
    ///
    /// - **Bit Mapping Logic**:
    ///   - Maps binary numbers to their equivalent ternary number containing only `0` and `1`.
    ///     `bit_mapping[b] = j`, where `j` is the ternary equivalent of `b`.
    ///
    /// ### Parameters
    /// - `c`: The highest index for ternary digits. Determines the range of ternary numbers to
    ///   process.
    ///
    /// ### Returns
    /// - `(Vec<u16>, Vec<u16>)`: A tuple containing:
    ///   - `bit_mapping`: Maps binary numbers to ternary equivalents.
    ///   - `trit_mapping`: Maps ternary numbers based on their digit properties.
    pub fn trit_mapping(&self) -> (Vec<u16>, Vec<u16>) {
        let c = self.c;
        // Calculate 3^(c+1), the total number of ternary numbers to process
        let pow3 = 3usize.pow((c + 1) as u32);

        // Preallocate memory for efficiency
        let mut bit_mapping = Vec::with_capacity(1 << (c + 1));
        let mut trit_mapping = Vec::with_capacity(pow3);

        // Iterate over all possible ternary numbers from 0 to 3^(c+1)-1
        for i in 0..pow3 {
            // Initialize variables for current ternary number, binary equivalent, and most
            // significant digit (msd)
            let (mut current, mut bin_value, mut msd) = (i, 0u16, None);

            // Decompose the number `i` into base-3 and calculate mappings
            for idx in 0..=c {
                // Extract the least significant ternary digit
                let digit = (current % 3) as u16;
                current /= 3;

                // Record the position of the most significant `2` if found
                if digit == 2 {
                    msd = Some(idx);
                } else {
                    // Update the binary equivalent by masking the digit with `1`
                    bin_value |= (digit & 1) << idx;
                }
            }

            // Determine the mapping based on the presence of a `2`
            if let Some(msd) = msd {
                // If a `2` exists, map to the corresponding power of 3
                trit_mapping.push(3u16.pow(msd as u32));
            } else {
                // Otherwise, map to twice the binary equivalent
                trit_mapping.push(bin_value << 1);
                bit_mapping.push(i as u16);
            }
        }

        // Return both mappings
        (bit_mapping, trit_mapping)
    }

    /// Extends multiple tables (`N` tables) using ternary-based recursion and applies two
    /// operations:
    /// - **Linear operation** (`f_lin`): Combines values linearly.
    /// - **Quadratic operation** (`f_quad`): Combines values quadratically.
    ///
    /// ### Overview of the Principle
    /// The function extends input tables hierarchically using a **ternary mapping**:
    /// - The ternary mapping determines how table entries are combined at each level.
    /// - At every index `j` in the extended output:
    ///     - If `j` is in the **primary range** (small indices), values are directly fetched and
    ///       combined.
    ///     - If `j` is beyond the primary range, values are **recursively combined** from previous
    ///       results.
    ///
    /// The recursion exploits precomputed offsets (`trit_mapping`) to efficiently locate
    /// dependencies and avoids redundant computations.
    ///
    /// ### Parameters:
    /// - `tables`: An array of `N` tables (`Vec<BinaryField128b>`) to be extended.
    /// - `trit_mapping`: A precomputed ternary mapping array that determines offsets.
    /// - `f_lin`: A **linear function** that takes a slice of values and computes a linear
    ///   combination.
    /// - `f_quad`: A **quadratic function** that takes a slice of values and computes a quadratic
    ///   combination.
    ///
    /// ### Constants:
    /// - `N`: The number of tables to process.
    ///
    /// ### Output:
    /// - Returns an extended table as a `Vec<BinaryField128b>` containing the computed values.
    pub fn extend_n_tables<const N: usize, L, Q>(
        &self,
        tables: &[Vec<BinaryField128b>; N],
        trit_mapping: &[u16],
        f_lin: L,
        f_quad: Q,
    ) -> Vec<BinaryField128b>
    where
        L: Fn(&[BinaryField128b]) -> BinaryField128b + Send + Sync,
        Q: Fn(&[BinaryField128b]) -> BinaryField128b + Send + Sync,
    {
        // Recursion depth parameter.
        let c = self.c;
        // Log2 of table size, determines the dimensions.
        let dims = tables[0].len().ilog2() as usize;

        // pow3: Total number of ternary indices to process (3^(c+1)).
        let pow3 = 3usize.pow((c + 1) as u32);

        // pow3_adj: Adjusted size for indices that require recursive processing.
        let pow3_adj = 2 * pow3 / 3;

        // pow2: Determines the number of chunks to process based on table size and recursion depth.
        let pow2 = 2usize.pow((dims - c - 1) as u32);

        // base_stride: Offset step for table chunks.
        let base_stride = 1 << (c + 1);

        // Preallocate the result vector to store the extended table.
        let mut result = vec![BinaryField128b::zero(); pow3 * pow2];

        // Parallelize over chunks of the result to maximize performance.
        let chunk_id_iter = result.par_chunks_mut(pow3);

        chunk_id_iter.enumerate().for_each(|(chunk_id, result_chunk)| {
            // `tables_ext` stores intermediate results for each ternary index.
            let mut tables_ext = vec![[BinaryField128b::zero(); N]; pow3_adj];

            // Base offset to determine which part of the input tables we are processing.
            let base_tab_offset = chunk_id * base_stride;

            for j in 0..pow3 {
                // Determine the offset for the current index `j` from the ternary mapping.
                let offset = trit_mapping[j] as usize;

                // Case 1: Small indices (primary range).
                if j < pow3_adj {
                    if offset % 2 == 0 {
                        // Even offset: Fetch values directly from the input tables.
                        let idx = base_tab_offset + (offset >> 1);
                        let tab_ext = &mut tables_ext[j];

                        for z in 0..N {
                            // Copy table values at the current offset.
                            tab_ext[z] = tables[z][idx];
                        }

                        // Apply the linear and quadratic functions.
                        result_chunk[j] = f_quad(&tab_ext[0..N]) + f_lin(&tab_ext[0..N]);
                    } else {
                        // Odd offset: Combine results from previous indices.
                        let tab_ext1 = tables_ext[j - offset];
                        let tab_ext2 = tables_ext[j - 2 * offset];

                        let tab_ext = &mut tables_ext[j];
                        for z in 0..N {
                            // Combine values recursively.
                            tab_ext[z] = tab_ext1[z] + tab_ext2[z];
                        }

                        // Apply the quadratic function.
                        result_chunk[j] = f_quad(&tab_ext[0..N]);
                    }
                } else {
                    // Case 2: Large indices (recursive range).
                    let mut args = [BinaryField128b::zero(); N];

                    let tab_ext1 = &tables_ext[j - offset];
                    let tab_ext2 = &tables_ext[j - 2 * offset];

                    for z in 0..N {
                        // Combine values recursively.
                        args[z] = tab_ext1[z] + tab_ext2[z];
                    }

                    // Apply the quadratic function.
                    result_chunk[j] = f_quad(&args);
                }
            }
        });

        // Return the fully extended table.
        result
    }

    pub fn build<const N: usize>(
        &mut self,
        polynomials: &[Vec<BinaryField128b>; N],
        claim_polynomial: &UnivariatePolynomial,
        gamma: BinaryField128b,
    ) -> BoolCheck {
        // Ensure all input polynomials have the expected length
        let expected_poly_len = 1 << self.points.len();
        for poly in polynomials {
            assert_eq!(poly.len(), expected_poly_len, "Polynomial length mismatch");
        }

        // Generate bit and trit mappings
        let (bit_mapping, trit_mapping) = self.trit_mapping();

        let mut tmp = BinaryField128b::one();
        for _ in 0..claim_polynomial.coeffs.len() {
            self.gammas.push(tmp);
            tmp *= gamma;
        }

        // Return the constructed BoolCheck
        BoolCheck {
            c: self.c,
            bit_mapping,
            eq_sequence: MultilinearLagrangianPolynomials::new_eq_poly_sequence(&self.points[1..]),
            claim: claim_polynomial.evaluate_at(&gamma),
            ext: Some(self.extend_n_tables(
                polynomials,
                &trit_mapping,
                |args| self.compress_linear(args),
                |args| self.compress_quadratic(args),
            )),
            pt: self.points.clone(),
            ..Default::default()
        }
    }

    pub fn exec_lin_compressed(&self, arg: &[BinaryField128b]) -> [BinaryField128b; M] {
        match self.boolean_package {
            BooleanPackage::And => {
                assert_eq!(M, 1, "Invalid output size for AND package");
                assert_eq!(arg.len(), 2, "Invalid input size for AND package");
                [BinaryField128b::zero(); M]
            }
        }
    }

    pub fn exec_quad_compressed(&self, arg: &[BinaryField128b]) -> [BinaryField128b; M] {
        match self.boolean_package {
            BooleanPackage::And => {
                assert_eq!(M, 1, "Invalid output size for AND package");
                assert_eq!(arg.len(), 2, "Invalid input size for AND package");
                [arg[0] & arg[1]; M]
            }
        }
    }

    pub fn exec_alg(
        &self,
        data: &[BinaryField128b],
        mut idx_a: usize,
        offset: usize,
    ) -> [[BinaryField128b; M]; 3] {
        match self.boolean_package {
            BooleanPackage::And => {
                assert_eq!(M, 1, "Invalid output size for AND package");

                // Double the starting index to account for the structure of the data.
                idx_a *= 2;

                // Compute the starting index for the second operand in the AND operation.
                let mut idx_b = idx_a + offset * 128;

                // Initialize the return array with zeros, ensuring the correct size.
                let mut ret = [[BinaryField128b::zero(); M]; 3];

                // Populate the first element of each sub-array based on the AND operation logic.
                ret[0][0] = BinaryField128b::basis(0) * data[idx_a] * data[idx_b];
                ret[1][0] = BinaryField128b::basis(0) * data[idx_a + 1] * data[idx_b + 1];
                ret[2][0] = BinaryField128b::basis(0) *
                    (data[idx_a] + data[idx_a + 1]) *
                    (data[idx_b] + data[idx_b + 1]);

                // Iterate over the remaining 127 basis elements to aggregate evaluations.
                for i in 1..128 {
                    // Move to the next indices for both operands in the AND operation.
                    idx_a += offset;
                    idx_b += offset;

                    // Add the contributions of the current basis element to the evaluations.
                    ret[0][0] += BinaryField128b::basis(i) * data[idx_a] * data[idx_b];
                    ret[1][0] += BinaryField128b::basis(i) * data[idx_a + 1] * data[idx_b + 1];
                    ret[2][0] += BinaryField128b::basis(i) *
                        (data[idx_a] + data[idx_a + 1]) *
                        (data[idx_b] + data[idx_b + 1]);
                }

                // Return the aggregated evaluations as a 3-element array.
                ret
            }
        }
    }
}

impl<const M: usize> CompressedFoldedOps for BoolCheckBuilder<M> {
    fn compress_linear(&self, arg: &[BinaryField128b]) -> BinaryField128b {
        // Compute the intermediate result by delegating to the wrapped `FnPackage`'s linear
        // computation.
        let tmp = self.exec_lin_compressed(arg);

        // Initialize the accumulator to zero.
        let mut acc = BinaryField128b::zero();

        // Iterate over the output size `M` and compute the folded sum using `gammas`.
        for (i, &t) in tmp.iter().enumerate() {
            // Multiply the result by the corresponding gamma and accumulate.
            acc += t * self.gammas[i];
        }

        // Return the final compressed result.
        acc
    }

    fn compress_quadratic(&self, arg: &[BinaryField128b]) -> BinaryField128b {
        // Compute the intermediate result by delegating to the wrapped `FnPackage`'s quadratic
        // computation.
        let tmp = self.exec_quad_compressed(arg);

        // Initialize the accumulator to zero.
        let mut acc = BinaryField128b::zero();

        // Iterate over the output size `M` and compute the folded sum using `gammas`.
        for (i, &t) in tmp.iter().enumerate() {
            // Multiply the result by the corresponding gamma and accumulate.
            acc += t * self.gammas[i];
        }

        // Return the final compressed result.
        acc
    }

    fn compress_algebraic(
        &self,
        data: &[BinaryField128b],
        start: usize,
        offset: usize,
    ) -> [BinaryField128b; 3] {
        // Compute the intermediate algebraic results by delegating to the wrapped `FnPackage`.
        let tmp = self.exec_alg(data, start, offset);

        // Initialize the accumulators for each of the 3 output values to zero.
        let mut acc = [BinaryField128b::zero(); 3];

        // Iterate over the output size `M` and compute the folded sums for each output value.
        for i in 0..M {
            // Compress the first output using gammas.
            acc[0] += tmp[0][i] * self.gammas[i];
            // Compress the second output using gammas.
            acc[1] += tmp[1][i] * self.gammas[i];
            // Compress the third output using gammas.
            acc[2] += tmp[2][i] * self.gammas[i];
        }

        // Return the array of compressed results.
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_mapping_small_c() {
        // Create an instance of BoolCheckBuilder with c = 1 (two ternary digits: 0, 1, 2).
        let bool_check: BoolCheckBuilder<0> = BoolCheckBuilder { c: 1, ..Default::default() };

        // Call the trit_mapping method to compute the mappings.
        let (bit_mapping, trit_mapping) = bool_check.trit_mapping();

        // Check that the binary-to-ternary mapping matches the expected result.
        assert_eq!(bit_mapping, vec![0, 1, 3, 4]);

        // Check that the ternary-to-binary mapping matches the expected result.
        assert_eq!(trit_mapping, vec![0, 2, 1, 4, 6, 1, 3, 3, 3]);
    }

    #[test]
    fn test_trit_mapping_medium_c() {
        // Create an instance of BoolCheckBuilder with c = 2 (three ternary digits: 0, 1, 2).
        let bool_check: BoolCheckBuilder<0> = BoolCheckBuilder { c: 2, ..Default::default() };

        // Call the trit_mapping method to compute the mappings.
        let (bit_mapping, trit_mapping) = bool_check.trit_mapping();

        // Check that the binary-to-ternary mapping matches the expected result.
        assert_eq!(bit_mapping, vec![0, 1, 3, 4, 9, 10, 12, 13]);

        // Check that the ternary-to-binary mapping matches the expected result.
        assert_eq!(
            trit_mapping,
            vec![
                0, 2, 1, 4, 6, 1, 3, 3, 3, 8, 10, 1, 12, 14, 1, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9
            ]
        );
    }

    #[test]
    fn test_trit_mapping_large_c() {
        let bool_check: BoolCheckBuilder<0> = BoolCheckBuilder { c: 4, ..Default::default() };

        let (bit_mapping, trit_mapping) = bool_check.trit_mapping();

        assert_eq!(
            bit_mapping,
            vec![
                0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31, 36, 37, 39, 40, 81, 82, 84, 85, 90, 91,
                93, 94, 108, 109, 111, 112, 117, 118, 120, 121
            ]
        );

        assert_eq!(
            trit_mapping,
            vec![
                0, 2, 1, 4, 6, 1, 3, 3, 3, 8, 10, 1, 12, 14, 1, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                16, 18, 1, 20, 22, 1, 3, 3, 3, 24, 26, 1, 28, 30, 1, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
                27, 27, 27, 27, 27, 27, 27, 27, 32, 34, 1, 36, 38, 1, 3, 3, 3, 40, 42, 1, 44, 46,
                1, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 48, 50, 1, 52, 54, 1, 3, 3, 3, 56, 58, 1,
                60, 62, 1, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27, 27,
                27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 81, 81, 81,
                81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
                81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
                81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
                81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81
            ]
        );
    }

    #[test]
    fn test_trit_mapping_no_c() {
        // Create an instance of BoolCheckBuilder with c = 0 (single ternary digit).
        let bool_check: BoolCheckBuilder<0> = BoolCheckBuilder { c: 0, ..Default::default() };

        // Call the trit_mapping method to compute the mappings.
        let (bit_mapping, trit_mapping) = bool_check.trit_mapping();

        // Check that the binary-to-ternary mapping matches the expected result.
        assert_eq!(bit_mapping, vec![0, 1]);

        // Check that the ternary-to-binary mapping matches the expected result.
        assert_eq!(trit_mapping, vec![0, 2, 1]);
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_extend_n_tables() {
        // Define the size of the input tables
        // Table size is determined as 2^dims. Here, dims = 3, so table size = 8 (2^3).
        let dims = 3;

        // Create a BoolCheckBuilder instance
        // The `c` parameter sets the recursion depth for ternary mappings.
        // Here, `c = 2`, meaning we work with ternary numbers up to 3^(2+1) = 27.
        let bool_check: BoolCheckBuilder<0> = BoolCheckBuilder { c: 2, ..Default::default() };

        // Generate test data for the input tables
        // Table1: Contains consecutive integers starting at 0 up to 7.
        let table1: Vec<BinaryField128b> = (0u128..(1 << dims)).map(Into::into).collect();
        // Table2: Contains integers starting at 10 up to 17.
        let table2: Vec<BinaryField128b> = (10u128..(10 + (1 << dims))).map(Into::into).collect();

        // Table3: Contains integers starting at 20 up to 27.
        let table3: Vec<BinaryField128b> = (20u128..(20 + (1 << dims))).map(Into::into).collect();

        // Combine the three tables into an array.
        let tabs = [table1, table2, table3];

        // Compute the ternary mapping for the current value of `c`
        // `trit_mapping` is a precomputed array that determines how table values are combined
        // recursively.
        let (_, trit_mapping) = bool_check.trit_mapping();

        // Define the linear function (f_lin)
        // `f_lin` computes the sum of all values in the input slice.
        let f_lin = |args: &[BinaryField128b]| {
            let mut res = BinaryField128b::zero();
            for &x in args {
                res += x;
            }
            res
        };

        // Define the quadratic function (f_quad)
        // `f_quad` computes the sum of the squares of all values in the input slice.
        let f_quad = |args: &[BinaryField128b]| {
            let mut res = BinaryField128b::zero();
            for &x in args {
                res += x * x;
            }
            res
        };

        // Call the `extend_n_tables` function
        // This function extends the input tables using the ternary mapping and applies the linear
        // and quadratic functions to combine the table values hierarchically.
        let result = bool_check.extend_n_tables(&tabs, &trit_mapping, f_lin, f_quad);

        // Validate the result
        // The result is compared to a hardcoded expected output. The expected output is the result
        // of applying the recursive extension logic to the input tables.
        assert_eq!(
            result,
            vec![
                BinaryField128b::from(286199402842439698884600957666611691959u128),
                BinaryField128b::from(92152884645276120751159248850998264247u128),
                BinaryField128b::from(194088056572031856754469952786247188481u128),
                BinaryField128b::from(299076299051606071403356588563077530026u128),
                BinaryField128b::from(152881988702699464694451933917556507050u128),
                BinaryField128b::from(194088056572031856754469952786247188481u128),
                BinaryField128b::from(72193695521068243347088020961476214811u128),
                BinaryField128b::from(72193695521068243347088020961476214811u128),
                BinaryField128b::from(0u128),
                BinaryField128b::from(18692268690725379462709786785192346071u128),
                BinaryField128b::from(207463413279617572725564511330318156247u128),
                BinaryField128b::from(194088056572031856754469952786247188481u128),
                BinaryField128b::from(288774782084272973388352083845904859232u128),
                BinaryField128b::from(100045175870249058746525603271412809824u128),
                BinaryField128b::from(194088056572031856754469952786247188481u128),
                BinaryField128b::from(286199402842439698884600957666611691945u128),
                BinaryField128b::from(286199402842439698884600957666611691945u128),
                BinaryField128b::from(0u128),
                BinaryField128b::from(288774782084272973388352083845904859244u128),
                BinaryField128b::from(288774782084272973388352083845904859244u128),
                BinaryField128b::from(0u128),
                BinaryField128b::from(74769074762901517850839147140769382878u128),
                BinaryField128b::from(74769074762901517850839147140769382878u128),
                BinaryField128b::from(0u128),
                BinaryField128b::from(299076299051606071403356588563077530034u128),
                BinaryField128b::from(299076299051606071403356588563077530034u128),
                BinaryField128b::from(0u128)
            ]
        );
    }
}
