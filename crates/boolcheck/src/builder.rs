use crate::{
    algebraic::AlgebraicOps, bool_trait::CompressedFoldedOps, ternary_mapping::TernaryMapping,
    BoolCheck,
};
use hashcaster_poly::{
    multinear_lagrangian::MultilinearLagrangianPolynomial,
    point::{Point, Points},
    univariate::UnivariatePolynomial,
};
use hashcaster_primitives::binary_field::BinaryField128b;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::array;

/// A builder for creating instances of `BoolCheck`.
///
/// # Overview
/// The `BoolCheckBuilder` struct is designed to provide a streamlined and configurable way
/// to construct `BoolCheck` objects, which are used for verifying Boolean relations over
/// multilinear polynomials.
///
/// It encapsulates the configuration parameters and input data required to construct
/// `BoolCheck`, allowing users to build instances with minimal effort while ensuring
/// correctness and consistency.
///
/// # Fields
/// - `N`: The number of multilinear polynomials (`polys`) involved in the `BoolCheck`.
/// - `M`: The number of outputs produced by the Boolean operations (e.g., AND, OR).
///
/// # Purpose
/// The primary purpose of `BoolCheckBuilder` is to:
/// - Validate and process input data.
/// - Precompute intermediate parameters such as folding challenges and mappings.
/// - Construct the extended tables and claims required for the `BoolCheck` protocol.
#[derive(Clone, Debug)]
pub struct BoolCheckBuilder<const N: usize, const M: usize, A: AlgebraicOps<N, M>> {
    /// The phase switch parameter (`c`).
    ///
    /// This parameter controls the number of initial rounds in the protocol.
    pub c: usize,

    /// The evaluation points for the multilinear polynomials.
    ///
    /// A collection of field elements where the Boolean operations are evaluated.
    /// These points represent the input space for the `BoolCheck` protocol.
    pub points: Points,

    /// The folding challenge `gamma`.
    ///
    /// A random field element used to compute folding challenges.
    /// Folding challenges are applied to compress the polynomials and evaluate
    /// the Boolean relations.
    pub gamma: Point,

    /// Precomputed folding challenges (`gammas`).
    ///
    /// An array of field elements derived from `gamma`. These values are used
    /// for compressing the Boolean operations during the `BoolCheck` protocol.
    pub gammas: [BinaryField128b; M],

    /// The initial claims for the Boolean operations.
    ///
    /// An array of field elements representing the initial claims for each
    /// Boolean operation at the given evaluation points.
    pub claims: [BinaryField128b; M],

    /// The multilinear polynomials to be used in the `BoolCheck` protocol.
    ///
    /// An array of polynomials representing the operands for the Boolean operations.
    /// Each polynomial must have a length equal to `2^dim`, where `dim` is the
    /// dimensionality of the input space.
    pub polys: [MultilinearLagrangianPolynomial; N],

    /// Abstract algebraic operations.
    pub algebraic_operations: A,
}

impl<const N: usize, const M: usize, A: AlgebraicOps<N, M> + Default> Default
    for BoolCheckBuilder<N, M, A>
{
    fn default() -> Self {
        Self {
            c: 0,
            points: Default::default(),
            gamma: Default::default(),
            gammas: array::from_fn(|_| Default::default()),
            claims: array::from_fn(|_| Default::default()),
            polys: array::from_fn(|_| Default::default()),
            algebraic_operations: Default::default(),
        }
    }
}

impl<const N: usize, const M: usize, A> BoolCheckBuilder<N, M, A>
where
    A: AlgebraicOps<N, M> + Default + Clone + Send + Sync,
{
    /// Creates a new instance of `BoolCheckBuilder`.
    ///
    /// # Parameters
    /// - `c`: The phase switch parameter, representing the number of initial rounds in the
    ///   protocol.
    /// - `points`: A collection of field elements (`Points`) where the Boolean operations are
    ///   evaluated.
    /// - `gamma`: A reference to a random field element (`Point`) used to compute folding
    ///   challenges.
    /// - `claims`: An array of initial claims (`[BinaryField128b; M]`) for the Boolean operations.
    /// - `polys`: An array of multilinear polynomials (`[MultilinearLagrangianPolynomial; N]`) used
    ///   as inputs to the Boolean operations.
    ///
    /// # Returns
    /// - A new instance of `BoolCheckBuilder` initialized with the provided parameters.
    ///
    /// # Panics
    /// - The function panics if `c >= points.len()`, as this would result in an invalid phase
    ///   switch configuration.
    ///
    /// # Notes
    /// - The `gammas` field is computed from the provided `gamma` using a folding strategy.
    pub fn new(
        algebraic_operations: A,
        c: usize,
        points: Points,
        gamma: &Point,
        claims: [BinaryField128b; M],
        polys: [MultilinearLagrangianPolynomial; N],
    ) -> Self {
        // Ensure the phase switch parameter `c` is valid.
        // `c` must be less than the number of evaluation points (`points.len()`).
        //
        // This ensures the phase switch does not exceed the dimensionality of the input space.
        assert!(c < points.len());
        Self {
            c,
            points,
            gamma: gamma.clone(),
            gammas: BinaryField128b::compute_gammas_folding::<M>(**gamma),
            claims,
            polys,
            algebraic_operations,
        }
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
    pub fn extend_n_tables<L, Q>(
        &self,
        trit_mapping: &[u16],
        f_lin: L,
        f_quad: Q,
    ) -> Vec<BinaryField128b>
    where
        L: Fn(&[BinaryField128b; N]) -> BinaryField128b + Send + Sync,
        Q: Fn(&[BinaryField128b; N]) -> BinaryField128b + Send + Sync,
    {
        // Recursion depth parameter.
        let c = self.c;
        // Log2 of table size, determines the dimensions.
        let dims = self.polys[0].len().ilog2() as usize;

        // pow3: Total number of ternary indices to process (3^(c+1)).
        let pow3 = 3usize.pow((c + 1) as u32);

        // pow3_adj: Adjusted size for indices that require recursive processing.
        let pow3_adj = 2 * pow3 / 3;

        // pow2: Determines the number of chunks to process based on table size and recursion depth.
        let pow2 = 1 << (dims - c - 1);

        // base_stride: Offset step for table chunks.
        let base_stride = 1 << (c + 1);

        // Preallocate the result vector to store the extended table.
        let mut result = vec![BinaryField128b::ZERO; pow3 * pow2];

        // Parallelize over chunks of the result to maximize performance.
        let chunk_id_iter = result.par_chunks_mut(pow3);

        chunk_id_iter.enumerate().for_each(|(chunk_id, result_chunk)| {
            // `tables_ext` stores intermediate results for each ternary index.
            let mut tables_ext = vec![[BinaryField128b::ZERO; N]; pow3_adj];

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
                        for (z, tab) in tab_ext.iter_mut().enumerate() {
                            // Copy table values at the current offset.
                            *tab = self.polys[z][idx];
                        }

                        // Sum the linear and quadratic parts.
                        result_chunk[j] = f_quad(tab_ext) + f_lin(tab_ext);
                    } else {
                        // Odd offset: Combine results from previous indices.
                        let tab_ext1 = tables_ext[j - offset];
                        let tab_ext2 = tables_ext[j - 2 * offset];
                        let tab_ext = &mut tables_ext[j];
                        for (z, tab) in tab_ext.iter_mut().enumerate() {
                            // Combine values recursively.
                            *tab = tab_ext1[z] + tab_ext2[z];
                        }

                        // Compute the quadratic part.
                        result_chunk[j] = f_quad(tab_ext);
                    }
                } else {
                    // Case 2: Large indices (recursive range).
                    let mut args = [BinaryField128b::ZERO; N];
                    let tab_ext1 = &tables_ext[j - offset];
                    let tab_ext2 = &tables_ext[j - 2 * offset];
                    for (z, arg) in args.iter_mut().enumerate() {
                        // Combine values recursively.
                        *arg = tab_ext1[z] + tab_ext2[z];
                    }

                    // Apply the quadratic function.
                    result_chunk[j] = f_quad(&args);
                }
            }
        });

        // Return the fully extended table.
        result
    }

    // This is an experimental version of the `extend_n_tables` with no improvements right now.
    pub fn extend_n_tables_experiment<L, Q>(
        &self,
        tree: &TernaryMapping,
        f_lin: L,
        f_quad: Q,
    ) -> Vec<BinaryField128b>
    where
        L: Fn(&[BinaryField128b; N]) -> BinaryField128b + Send + Sync,
        Q: Fn(&[BinaryField128b; N]) -> BinaryField128b + Send + Sync,
    {
        use rayon::prelude::*;

        // Recursion depth parameter.
        let c = self.c;
        let dims = self.polys[0].len().ilog2() as usize;

        let pow3 = 3usize.pow((c + 1) as u32);
        let pow3_adj = 2 * pow3 / 3;
        let pow2 = 1 << (dims - c - 1);

        let chunk_size = 3usize.pow((c - 2) as u32);
        let base_stride = 1 << (c + 1);

        assert_eq!(pow3 % chunk_size, 0, "Chunk size must evenly divide total size");

        let mut result = vec![BinaryField128b::ZERO; pow3 * pow2];

        // Préparation hors des boucles
        let tree_descendants: Vec<_> =
            tree.par_iter().map(|node| node.descendants.clone()).collect();

        result.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_id, chunk)| {
            let base_chunk = (chunk_id * chunk_size) / pow3;
            let base_offset = base_chunk * base_stride;
            let base = (chunk_size * chunk_id) % pow3;

            chunk.iter_mut().enumerate().for_each(|(local_j, elem)| {
                let global_j = base + local_j;
                let offset = &tree_descendants[global_j];

                let mut args = [BinaryField128b::ZERO; N];

                for (z, arg) in args.iter_mut().enumerate().take(N) {
                    let mut acc = BinaryField128b::ZERO;
                    for &x in offset {
                        unsafe {
                            acc +=
                                *self.polys.get_unchecked(z).get_unchecked(base_offset + (x >> 1));
                        }
                    }
                    *arg = acc;
                }

                *elem = if global_j < pow3_adj && offset.len() == 1 {
                    f_quad(&args) + f_lin(&args)
                } else {
                    f_quad(&args)
                };
            });
        });

        result
    }

    pub fn build(&self) -> BoolCheck<N, M, A> {
        // Ensure all input polynomials have the expected length
        let expected_poly_len = 1 << self.points.len();
        for poly in &self.polys {
            assert_eq!(poly.len(), expected_poly_len, "Polynomial length mismatch");
        }

        // Generate bit and trit mappings
        let (bit_mapping, trit_mapping) = self.trit_mapping();

        // Prepare the points to be used in the eq sequence
        let pt_eq_sequence: Points = self.points[1..].into();

        // Return the constructed BoolCheck
        BoolCheck {
            c: self.c,
            bit_mapping,
            eq_sequence: pt_eq_sequence.to_eq_poly_sequence(),
            claim: UnivariatePolynomial::new(self.claims.into()).evaluate_at(&self.gamma),
            extended_table: self.extend_n_tables(
                &trit_mapping,
                |args| self.linear_compressed(args),
                |args| self.quadratic_compressed(args),
            ),
            polys: self.polys.clone(),
            points: self.points.clone(),
            gammas: self.gammas,
            algebraic_operations: self.algebraic_operations.clone(),
            ..Default::default()
        }
    }
}

impl<const N: usize, const M: usize, A: AlgebraicOps<N, M> + Send + Sync> CompressedFoldedOps<N>
    for BoolCheckBuilder<N, M, A>
{
    fn linear_compressed(&self, arg: &[BinaryField128b; N]) -> BinaryField128b {
        // Compute the linear part of the boolean formula.
        let lin = self.algebraic_operations.linear(arg);

        // Initialize the accumulator to zero.
        let mut acc = BinaryField128b::ZERO;

        // Iterate over the output size `M` and compute the folded sum using `gammas`.
        for (i, &t) in lin.iter().enumerate() {
            // Multiply the result by the corresponding gamma and accumulate.
            acc += t * self.gammas[i];
        }

        // Return the final compressed result.
        acc
    }

    fn quadratic_compressed(&self, arg: &[BinaryField128b; N]) -> BinaryField128b {
        // Compute the quadratic part of the boolean formula.
        let quad = self.algebraic_operations.quadratic(arg);

        // Initialize the accumulator to zero.
        let mut acc = BinaryField128b::ZERO;

        // Iterate over the output size `M` and compute the folded sum using `gammas`.
        for (i, &t) in quad.iter().enumerate() {
            // Multiply the result by the corresponding gamma and accumulate.
            acc += t * self.gammas[i];
        }

        // Return the final compressed result.
        acc
    }

    fn algebraic_compressed(
        &self,
        data: &[BinaryField128b],
        idx_a: usize,
        offset: usize,
    ) -> [BinaryField128b; 3] {
        // Compute the algebraic result.
        let alg = self.algebraic_operations.algebraic(data, idx_a, offset);

        // Initialize the accumulators for each of the 3 output values to zero.
        let mut acc = [BinaryField128b::ZERO; 3];

        // Iterate over the output size `M` and compute the folded sums for each output value.
        for i in 0..M {
            // Compress the first output using gammas.
            acc[0] += alg[0][i] * self.gammas[i];
            // Compress the second output using gammas.
            acc[1] += alg[1][i] * self.gammas[i];
            // Compress the third output using gammas.
            acc[2] += alg[2][i] * self.gammas[i];
        }

        // Return the array of results.
        acc
    }
}

#[cfg(test)]
mod tests {
    use crate::and::AndPackage;

    use super::*;

    #[test]
    fn test_trit_mapping_small_c() {
        // Create an instance of BoolCheckBuilder with c = 1 (two ternary digits: 0, 1, 2).
        let bool_check: BoolCheckBuilder<0, 0, AndPackage<0, 0>> =
            BoolCheckBuilder { c: 1, ..Default::default() };

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
        let bool_check: BoolCheckBuilder<0, 0, AndPackage<0, 0>> =
            BoolCheckBuilder { c: 2, ..Default::default() };

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
        let bool_check: BoolCheckBuilder<0, 0, AndPackage<0, 0>> =
            BoolCheckBuilder { c: 4, ..Default::default() };

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
        let bool_check: BoolCheckBuilder<0, 0, AndPackage<0, 0>> =
            BoolCheckBuilder { c: 0, ..Default::default() };

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

        // Generate test data for the input tables
        // Table1: Contains consecutive integers starting at 0 up to 7.
        let table1: MultilinearLagrangianPolynomial =
            (0u128..(1 << dims)).map(Into::into).collect();
        // Table2: Contains integers starting at 10 up to 17.
        let table2: MultilinearLagrangianPolynomial =
            (10u128..(10 + (1 << dims))).map(Into::into).collect();
        // Table3: Contains integers starting at 20 up to 27.
        let table3: MultilinearLagrangianPolynomial =
            (20u128..(20 + (1 << dims))).map(Into::into).collect();

        // Combine the three tables into an array.
        let tabs = [table1, table2, table3];

        // Create a BoolCheckBuilder instance
        // The `c` parameter sets the recursion depth for ternary mappings.
        // Here, `c = 2`, meaning we work with ternary numbers up to 3^(2+1) = 27.
        let bool_check: BoolCheckBuilder<3, 1, AndPackage<3, 1>> =
            BoolCheckBuilder { c: 2, polys: tabs, ..Default::default() };

        // Compute the ternary mapping for the current value of `c`
        // `trit_mapping` is a precomputed array that determines how table values are combined
        // recursively.
        let (_, trit_mapping) = bool_check.trit_mapping();

        // Define the linear function (f_lin)
        // `f_lin` computes the sum of all values in the input slice.
        let f_lin = |args: &[BinaryField128b; 3]| {
            let mut res = BinaryField128b::ZERO;
            for &x in args {
                res += x;
            }
            res
        };

        // Define the quadratic function (f_quad)
        // `f_quad` computes the sum of the squares of all values in the input slice.
        let f_quad = |args: &[BinaryField128b; 3]| {
            let mut res = BinaryField128b::ZERO;
            for &x in args {
                res += x * x;
            }
            res
        };

        // Call the `extend_n_tables` function
        // This function extends the input tables using the ternary mapping and applies the linear
        // and quadratic functions to combine the table values hierarchically.
        let result = bool_check.extend_n_tables(&trit_mapping, f_lin, f_quad);

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
