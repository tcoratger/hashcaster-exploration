use crate::{algebraic::AlgebraicOps, bool_trait::CompressedFoldedOps, BoolCheck};
use bytemuck::zeroed_vec;
use hashcaster_primitives::{
    binary_field::BinaryField128b,
    poly::{
        multinear_lagrangian::MultilinearLagrangianPolynomial,
        point::{Point, Points},
        univariate::UnivariatePolynomial,
    },
    sumcheck::SumcheckBuilder,
};
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
/// - `C`: The phase switch parameter (`c`).
///
/// # Purpose
/// The primary purpose of `BoolCheckBuilder` is to:
/// - Validate and process input data.
/// - Precompute intermediate parameters such as folding challenges and mappings.
/// - Construct the extended tables and claims required for the `BoolCheck` protocol.
#[derive(Clone, Debug)]
pub struct BoolCheckBuilder<
    'a,
    const N: usize,
    const M: usize,
    const C: usize,
    A: AlgebraicOps<N, M>,
> {
    /// The evaluation points for the multilinear polynomials.
    ///
    /// A collection of field elements where the Boolean operations are evaluated.
    /// These points represent the input space for the `BoolCheck` protocol.
    pub points: &'a Points,

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
    pub polys: &'a [MultilinearLagrangianPolynomial; N],

    /// Abstract algebraic operations.
    pub algebraic_operations: &'a A,
}

impl<'a, const N: usize, const M: usize, const C: usize, A> BoolCheckBuilder<'a, N, M, C, A>
where
    A: AlgebraicOps<N, M> + Default + Clone + Send + Sync,
{
    /// Creates a new instance of `BoolCheckBuilder`.
    ///
    /// # Parameters
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
        algebraic_operations: &'a A,
        points: &'a Points,
        claims: [BinaryField128b; M],
        polys: &'a [MultilinearLagrangianPolynomial; N],
    ) -> Self {
        // Ensure the phase switch parameter `c` is valid.
        // `c` must be less than the number of evaluation points (`points.len()`).
        //
        // This ensures the phase switch does not exceed the dimensionality of the input space.
        assert!(C < points.len());
        Self {
            points,
            claims,
            polys,
            algebraic_operations,
            gammas: array::from_fn(|_| Default::default()),
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
        // Calculate 3^(c+1), the total number of ternary numbers to process
        let pow3 = 3usize.pow((C + 1) as u32);

        // Preallocate memory for efficiency
        let mut bit_mapping = Vec::with_capacity(1 << (C + 1));
        let mut trit_mapping = Vec::with_capacity(pow3);

        // Iterate over all possible ternary numbers from 0 to 3^(c+1)-1
        for i in 0..pow3 {
            // Initialize variables for current ternary number, binary equivalent, and most
            // significant digit (msd)
            let (mut current, mut bin_value, mut msd) = (i, 0u16, None);

            // Decompose the number `i` into base-3 and calculate mappings
            for idx in 0..=C {
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
    pub fn extend_n_tables(&self, trit_mapping: &[u16]) -> Vec<BinaryField128b> {
        // Log2 of table size, determines the dimensions.
        let dims = self.polys[0].len().ilog2() as usize;

        // pow3: Total number of ternary indices to process (3^(c+1)).
        let pow3 = 3usize.pow((C + 1) as u32);

        // pow3_adj: Adjusted size for indices that require recursive processing.
        let pow3_adj = 2 * pow3 / 3;

        // pow2: Determines the number of chunks to process based on table size and recursion depth.
        let pow2 = 1 << (dims - C - 1);

        // base_stride: Offset step for table chunks.
        let base_stride = 1 << (C + 1);

        // Preallocate the result vector to store the extended table.
        let mut result = zeroed_vec(pow3 * pow2);

        // Parallelize over chunks of the result to maximize performance.
        result.par_chunks_mut(pow3).enumerate().for_each(|(chunk_id, result_chunk)| {
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
                        result_chunk[j] =
                            self.quadratic_compressed(tab_ext) + self.linear_compressed(tab_ext);
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
                        result_chunk[j] = self.quadratic_compressed(tab_ext);
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
                    result_chunk[j] = self.quadratic_compressed(&args);
                }
            }
        });

        // Return the fully extended table.
        result
    }
}

impl<'a, const N: usize, const M: usize, const C: usize, A> SumcheckBuilder<{ 128 * N }>
    for BoolCheckBuilder<'a, N, M, C, A>
where
    A: AlgebraicOps<N, M> + Default + Clone + Send + Sync,
    [(); 128 * N]:,
{
    type Sumcheck = BoolCheck<'a, N, M, C, A>;

    fn build(mut self, gamma: &Point) -> Self::Sumcheck {
        // Compute the folding challenges using the provided gamma.
        self.gammas = BinaryField128b::compute_gammas_folding(gamma);

        // Ensure all input polynomials have the expected length
        let expected_poly_len = 1 << self.points.len();
        assert!(
            self.polys.iter().all(|poly| { poly.len() == expected_poly_len }),
            "Polynomial length mismatch"
        );

        // Generate bit and trit mappings
        let (bit_mapping, trit_mapping) = self.trit_mapping();

        // Prepare the points to be used in the eq sequence
        let pt_eq_sequence: Points = self.points[1..].into();

        // Return the constructed BoolCheck
        BoolCheck {
            bit_mapping,
            eq_sequence: pt_eq_sequence.to_eq_poly_sequence(),
            claim: UnivariatePolynomial::new(self.claims.into()).evaluate_at(gamma),
            extended_table: self.extend_n_tables(&trit_mapping),
            polys: self.polys,
            points: self.points,
            gammas: self.gammas,
            algebraic_operations: self.algebraic_operations,
            poly_coords: Default::default(),
            challenges: Default::default(),
            round_polys: Default::default(),
        }
    }
}

impl<const N: usize, const M: usize, const C: usize, A: AlgebraicOps<N, M> + Send + Sync>
    CompressedFoldedOps<N> for BoolCheckBuilder<'_, N, M, C, A>
{
    fn linear_compressed(&self, arg: &[BinaryField128b; N]) -> BinaryField128b {
        // Compute and fold the linear part of the boolean formula using gammas.
        self.algebraic_operations
            .linear(arg)
            .iter()
            .zip(self.gammas.iter())
            .fold(BinaryField128b::ZERO, |acc, (&t, &gamma)| acc + t * gamma)
    }

    fn quadratic_compressed(&self, arg: &[BinaryField128b; N]) -> BinaryField128b {
        // Compute and fold the quadratic part of the boolean formula using gammas.
        self.algebraic_operations
            .quadratic(arg)
            .iter()
            .zip(self.gammas.iter())
            .fold(BinaryField128b::ZERO, |acc, (&t, &gamma)| acc + t * gamma)
    }

    fn algebraic_compressed(
        &self,
        data: &[BinaryField128b],
        idx_a: usize,
        offset: usize,
    ) -> [BinaryField128b; 3] {
        // Compute the algebraic result.
        let alg = self.algebraic_operations.algebraic(data, idx_a, offset);

        // Fold each output value using the gammas.
        array::from_fn(|j| {
            (0..M).fold(BinaryField128b::ZERO, |acc, i| acc + alg[j][i] * self.gammas[i])
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct DummyPackage<const I: usize, const O: usize>;

impl<const I: usize, const O: usize> AlgebraicOps<I, O> for DummyPackage<I, O> {
    fn algebraic(
        &self,
        _data: &[BinaryField128b],
        _idx_a: usize,
        _offset: usize,
    ) -> [[BinaryField128b; O]; 3] {
        [[BinaryField128b::ZERO; O]; 3]
    }

    fn linear(&self, data: &[BinaryField128b; I]) -> [BinaryField128b; O] {
        let mut result = BinaryField128b::ZERO;
        for &x in data {
            result += x;
        }
        [result; O]
    }

    fn quadratic(&self, data: &[BinaryField128b; I]) -> [BinaryField128b; O] {
        let mut result = BinaryField128b::ZERO;
        for &x in data {
            result += x * x;
        }
        [result; O]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::and::AndPackage;

    #[test]
    fn test_trit_mapping_small_c() {
        // Create an instance of BoolCheckBuilder with c = 1 (two ternary digits: 0, 1, 2).
        let points = Points::default();
        let bool_check: BoolCheckBuilder<'_, 0, 0, 1, AndPackage<0, 0>> = BoolCheckBuilder {
            points: &points,
            claims: Default::default(),
            polys: &Default::default(),
            gammas: Default::default(),
            algebraic_operations: &Default::default(),
        };

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
        let points = Points::default();
        let bool_check: BoolCheckBuilder<'_, 0, 0, 2, AndPackage<0, 0>> = BoolCheckBuilder {
            points: &points,
            claims: Default::default(),
            polys: &Default::default(),
            gammas: Default::default(),
            algebraic_operations: &Default::default(),
        };

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
        let points = Points::default();
        let bool_check: BoolCheckBuilder<'_, 0, 0, 4, AndPackage<0, 0>> = BoolCheckBuilder {
            points: &points,
            claims: Default::default(),
            polys: &Default::default(),
            gammas: Default::default(),
            algebraic_operations: &Default::default(),
        };

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
        let points = Points::default();
        let bool_check: BoolCheckBuilder<'_, 0, 0, 0, AndPackage<0, 0>> = BoolCheckBuilder {
            points: &points,
            claims: Default::default(),
            polys: &Default::default(),
            gammas: Default::default(),
            algebraic_operations: &Default::default(),
        };

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
        let points = Points::default();
        let bool_check: BoolCheckBuilder<'_, 3, 1, 2, DummyPackage<3, 1>> = BoolCheckBuilder {
            polys: &tabs,
            gammas: [BinaryField128b::ONE; 1],
            points: &points,
            claims: Default::default(),
            algebraic_operations: &Default::default(),
        };

        // Compute the ternary mapping for the current value of `c`
        // `trit_mapping` is a precomputed array that determines how table values are combined
        // recursively.
        let (_, trit_mapping) = bool_check.trit_mapping();

        // Call the `extend_n_tables` function
        // This function extends the input tables using the ternary mapping and applies the linear
        // and quadratic functions to combine the table values hierarchically.
        let result = bool_check.extend_n_tables(&trit_mapping);

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
