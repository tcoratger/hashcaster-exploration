use crate::{
    point::Points,
    utils::{cpu_v_movemask_epi8, drop_top_bit, v_slli_epi64},
};
use bytemuck::cast_slice;
use hashcaster_field::binary_field::BinaryField128b;
use num_traits::{One, Zero};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use std::ops::{BitAnd, Deref, DerefMut};

/// A structure representing a multilinear Lagrangian polynomial.
/// This structure holds the coefficients of the polynomial in a vector.
///
/// Each coefficient corresponds to the evaluation of the polynomial at a specific point
/// in the Boolean hypercube `{0,1}^n`. For example, for `n = 2`, the coefficients
/// represent the evaluations `p(0,0)`, `p(0,1)`, `p(1,0)`, `p(1,1)`, where `p(x)` is the
/// polynomial being constructed.
///
/// The multilinear polynomial is defined as:
///
/// ```text
/// p(X_1, X_2, ..., X_n) = sum_{x in {0,1}^n} p(x) * prod_{i=1}^n phi_{x_i}(X_i)
/// ```
///
/// where `phi_{x_i}(X_i)` is the basis polynomial that ensures the term evaluates
/// to `p(x)` only at `x` and vanishes otherwise.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MultilinearLagrangianPolynomial {
    /// Coefficients of the polynomial, stored as a vector of [`BinaryField128b`] elements.
    ///
    /// Each entry corresponds to the evaluation of the polynomial at a point in the Boolean
    /// hypercube `{0,1}^n`, in lexicographical order of the points. For example, for `n = 2`:
    /// - `coeffs[0]` corresponds to `p(0,0)`,
    /// - `coeffs[1]` corresponds to `p(0,1)`,
    /// - `coeffs[2]` corresponds to `p(1,0)`,
    /// - `coeffs[3]` corresponds to `p(1,1)`.
    pub coeffs: Vec<BinaryField128b>,
}

impl MultilinearLagrangianPolynomial {
    pub fn inner(&self) -> Vec<BinaryField128b> {
        self.coeffs.clone()
    }

    /// Creates a new [`MultilinearLagrangianPolynomial`] with the given coefficients.
    ///
    /// # Arguments
    /// * `coeffs` - A vector of [`BinaryField128b`] elements representing the polynomial's
    ///   coefficients.
    ///
    /// # Returns
    /// A new instance of [`MultilinearLagrangianPolynomial`].
    pub const fn new(coeffs: Vec<BinaryField128b>) -> Self {
        Self { coeffs }
    }

    /// Constructs the equality polynomial based on the provided points.
    ///
    /// # Arguments
    /// * `points` - A slice of `BinaryField128b` elements representing the input points.
    ///
    /// # Returns
    /// An instance of `MultilinearLagrangianPolynomial` containing the coefficients of the equality
    /// polynomial.
    ///
    /// # Explanation
    /// The equality polynomial is a multivariate polynomial constructed to encode the relationship
    /// between multiple input points. It ensures that the polynomial evaluates to `1` when the
    /// input `z` aligns with the combination of input points `points` and appropriately
    /// interpolates over the domain for multivariate inputs.
    ///
    /// ## Definition
    /// Given a set of points `points = [b_1, b_2, ..., b_m]` in the finite field, the equality
    /// polynomial `eq(z, points)` is defined iteratively as:
    ///
    /// ```text
    /// eq(z, points) = prod_{i=1}^m (z_i * b_i + (1 - z_i) * (1 - b_i)).
    /// ```
    /// Here, `z = (z_1, z_2, ..., z_m)` is the evaluation point, and `points` represent the
    /// multivariate configuration for which the polynomial encodes the behavior.
    ///
    /// ## Utility in Multivariate Context
    /// The equality polynomial extends to encode the behavior across multiple input points. It
    /// forms the basis for evaluating multilinear extensions in higher-dimensional spaces. For
    /// a multilinear extension `f_MLE` of a function `f`, we use:
    ///
    /// ```text
    /// f_MLE(z) = sum_{b in {0,1}^m} eq(z, b) * f(b),
    /// ```
    /// where `f(b)` is the value of the function at point `b`, and `eq(z, b)` ensures interpolation
    /// for the multivariate domain.
    ///
    /// ## Multivariate Example
    /// For `points = [pt1, pt2]`:
    /// ```text
    /// eq_poly(z) =
    ///   coeffs[0] * (1 - pt1) * (1 - pt2) +
    ///   coeffs[1] * pt1 * (1 - pt2) +
    ///   coeffs[2] * (1 - pt1) * pt2 +
    ///   coeffs[3] * pt1 * pt2.
    /// ```
    /// The coefficients `coeffs` are updated iteratively to encode this multivariate behavior. This
    /// ensures that the polynomial evaluates correctly for combinations of inputs.
    ///
    /// ## How This Implementation Works
    /// - The polynomial is constructed iteratively starting from a single coefficient (initialized
    ///   to `1`).
    /// - Coefficients are updated using the recurrence relation to encode the multivariate
    ///   relationships.
    /// - The result captures the equality polynomial over the domain, supporting evaluation in
    ///   multilinear extensions and related computations.
    pub fn new_eq_poly(points: &Points) -> Self {
        // Initialize the coefficients with a single 1 (neutral element for multiplication).
        let mut coeffs = vec![BinaryField128b::one()];

        // Preallocate memory for all coefficients, filling with zeros beyond the initial size.
        coeffs.resize(1 << points.len(), BinaryField128b::zero());

        // Iterate over the points to construct the equality polynomial.
        for (i, point) in points.iter().enumerate() {
            // Split the coefficient vector into two parts: `left` and `right`.
            // `left` contains existing coefficients, `right` will store the new coefficients.
            let (left, right) = coeffs.split_at_mut(1 << i);

            // Update coefficients in parallel using iterators over `left` and `right`.
            left.par_iter_mut().zip(right.par_iter_mut()).for_each(|(left_val, right_val)| {
                // Compute the new coefficient in `right` as the product of `left_val` and the
                // current point.
                *right_val = *left_val * **point;
                // Update the existing coefficient in `left` by adding the computed `right_val`.
                *left_val += *right_val;
            });
        }

        // Return the constructed equality polynomial.
        Self { coeffs }
    }

    /// Evaluates the multilinear Lagrangian polynomial at a given point in the Boolean hypercube.
    ///
    /// # Arguments
    /// - `points`: A slice of [`BinaryField128b`] elements representing the evaluation point in the
    ///   hypercube. The number of elements in `points` corresponds to the dimensions of the
    ///   hypercube.
    ///
    /// # Returns
    /// - A [`BinaryField128b`] element representing the value of the polynomial at the given point.
    ///
    /// # Explanation
    /// This function evaluates the polynomial by computing a weighted sum of its coefficients.
    /// The weights are derived from the equality polynomial for the given `points`.
    /// The equality polynomial encodes the interpolation conditions to ensure correct evaluation
    /// in the hypercube.
    ///
    /// ## Method
    /// The evaluation is computed as:
    /// ```text
    /// result = sum_{x in {0,1}^n} coeffs[x] * eq_poly(x, points)
    /// ```
    /// where `coeffs[x]` is the coefficient of the polynomial corresponding to the point `x`,
    /// and `eq_poly(x, points)` is the equality polynomial for `x` and `points`.
    pub fn evaluate_at(&self, points: &Points) -> BinaryField128b {
        // Ensure the number of coefficients matches the number of points.
        // The number of coefficients should be 2^n, where n is the length of `points`.
        assert!(self.coeffs.len() == 1 << points.len());

        // Compute the weighted sum of coefficients.
        // Each coefficient is multiplied by the corresponding weight from the equality polynomial.
        self.coeffs
            .par_iter()
            .zip(Self::new_eq_poly(points).coeffs)
            .map(|(x, y)| *x * y)
            .reduce(BinaryField128b::zero, |a, b| a + b)
    }

    /// Computes the equality sums for the polynomial coefficients.
    ///
    /// # Description
    /// This function computes all possible sums of subsets of polynomial coefficients,
    /// grouped into blocks of size 8. The sums for each block are computed efficiently
    /// using a decomposition strategy (`drop_top_bit`) to incrementally calculate the sums.
    ///
    /// Each subset of coefficients corresponds to a binary representation of the subset index,
    /// where each bit in the index determines whether the corresponding coefficient is included
    /// in the subset. The function ensures efficient calculation by reusing previously computed
    /// sums to avoid redundant operations.
    ///
    /// # Parameters
    /// - `self`: A reference to the polynomial, where the coefficients are stored as
    ///   `BinaryField128b` elements. The polynomial is divided into blocks of 8 coefficients each.
    ///
    /// # Returns
    /// A `Vec<BinaryField128b>` containing the computed sums for all subsets of coefficients in
    /// all blocks of the polynomial. The result includes 256 sums for each block of 8 coefficients.
    ///
    /// # Panics
    /// - If the total number of coefficients in the polynomial (`self.len()`) is not divisible by
    ///   16.
    pub fn eq_sums(&self) -> Vec<BinaryField128b> {
        // Ensure the number of coefficients in the equality polynomial is divisible by 16.
        let eq_len = self.len();
        assert_eq!(
            eq_len % 16,
            0,
            "The number of coefficients in the equality polynomial must be divisible by 16."
        );

        // Preallocate the result vector to store all computed sums.
        let mut results = Vec::with_capacity(256 * (eq_len / 8));

        // Process the polynomial in blocks of 8 coefficients.
        for block_start in (0..eq_len).step_by(8) {
            // Extract the current block of 8 coefficients.
            let block = &self[block_start..block_start + 8];

            // Initialize an array to store the 256 sums for this block.
            let mut block_sums = [BinaryField128b::zero(); 256];

            // Iterate over all subsets (from 1 to 255) to compute their sums.
            for subset in 1..256 {
                // Use drop_top_bit to decompose the subset index.
                let (sum_idx, bit_idx) = drop_top_bit(subset);

                // Compute the sum for the current subset by adding:
                // - The coefficient corresponding to the dropped bit index.
                // - The previously computed sum for the reduced subset.
                block_sums[subset] = block[bit_idx] + block_sums[sum_idx];
            }

            // Append all sums from the current block to the result vector.
            results.extend_from_slice(&block_sums);
        }

        results
    }
}

impl From<Vec<BinaryField128b>> for MultilinearLagrangianPolynomial {
    fn from(coeffs: Vec<BinaryField128b>) -> Self {
        Self::new(coeffs)
    }
}

impl BitAnd for MultilinearLagrangianPolynomial {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        // Ensure the number of coefficients matches between the two polynomials.
        assert_eq!(self.coeffs.len(), rhs.coeffs.len());

        // Perform element-wise AND operation on the coefficients.
        let coeffs = self.coeffs.into_iter().zip(rhs.coeffs).map(|(a, b)| a & b).collect();

        // Return the result as a new polynomial.
        Self::new(coeffs)
    }
}

impl Deref for MultilinearLagrangianPolynomial {
    type Target = Vec<BinaryField128b>;

    fn deref(&self) -> &Self::Target {
        &self.coeffs
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MultilinearLagrangianPolynomials(Vec<MultilinearLagrangianPolynomial>);

impl MultilinearLagrangianPolynomials {
    /// Returns the polynomial at the specified index in the sequence.
    pub fn poly_at(&self, index: usize) -> &MultilinearLagrangianPolynomial {
        &self.0[index]
    }

    /// Constructs the sequence of equality polynomials for a given set of points.
    ///
    /// # Arguments
    /// * `points` - A slice of `BinaryField128b` elements representing the input points.
    ///
    /// # Returns
    /// A [`MultilinearLagrangianPolynomials`] instance containing the sequence of equality
    /// polynomials.
    ///
    /// # Explanation
    /// This method generates a sequence of equality polynomials, where each polynomial encodes the
    /// relationships of subsets of the input points. The sequence starts with a base case
    /// polynomial `eq_0(x) = 1` and iteratively constructs each subsequent polynomial using
    /// a recurrence relation.
    ///
    /// ## Sequence Construction
    /// The equality polynomials in the sequence are defined recursively:
    /// - `eq_0(x) = 1` (base case),
    /// - `eq_k(x) = eq_{k-1}(x) * (1 + m_k * x)`, where `m_k` is the multiplier for the \(k\)-th
    ///   point.
    ///
    /// ## Utility
    /// This sequence is useful in efficiently constructing multilinear extensions, where each
    /// polynomial in the sequence serves as a building block for interpolating functions over
    /// the Boolean hypercube \( \{0, 1\}^n \).
    pub fn new_eq_poly_sequence(points: &Points) -> Self {
        // Start with the base case: eq_0(x) = 1.
        let mut polynomials =
            vec![MultilinearLagrangianPolynomial::new(vec![BinaryField128b::one()])];

        // Iterate over the points in reverse order.
        for (i, multiplier) in points.iter().rev().enumerate() {
            // Reference the previously computed polynomial in the sequence.
            let previous = &polynomials[i];

            // Allocate space for the new polynomial coefficients.
            // The new polynomial will have twice the size of the previous one.
            let mut new_coeffs = vec![BinaryField128b::zero(); 1 << (i + 1)];

            // Compute the new polynomial coefficients using the recurrence relation.
            new_coeffs.par_chunks_exact_mut(2).zip(previous.par_iter()).for_each(
                |(chunk, &prev_coeff)| {
                    // Calculate the updated coefficients.
                    let multiplied = **multiplier * prev_coeff;
                    chunk[0] = prev_coeff + multiplied; // Update the first coefficient.
                    chunk[1] = multiplied; // Update the second coefficient.
                },
            );

            // Append the new polynomial to the list.
            polynomials.push(MultilinearLagrangianPolynomial::new(new_coeffs));
        }

        // Return the constructed sequence of equality polynomials.
        Self(polynomials)
    }

    /// Applies the restriction operation to a set of multilinear Lagrangian polynomials.
    ///
    /// # Overview
    /// The `restrict` function evaluates a set of multilinear Lagrangian polynomials
    /// over a restricted subset of their domain, as defined by `challenges`.
    ///
    /// # Parameters
    /// - `self`: A reference to a `MultilinearLagrangianPolynomials` structure, which contains one
    ///   or more multilinear Lagrangian polynomials.
    /// - `challenges`: A slice of `BinaryField128b` values specifying the challenge points that
    ///   define the restriction.
    /// - `dims`: The dimensionality of the domain of the input polynomials (i.e., the number of
    ///   variables in the Boolean hypercube).
    ///
    /// # Returns
    /// A `Vec<BinaryField128b>` containing the computed restricted evaluations for all input
    /// polynomials.
    ///
    /// # Key Concepts
    /// 1. **Equality Polynomial**:
    ///    - For each `challenges` point, the equality polynomial encodes interpolation conditions
    ///      that ensure correct evaluation at the given challenges.
    ///
    /// 2. **Subset Sums**:
    ///    - Subsets of polynomial coefficients are grouped into chunks, and sums over these subsets
    ///      are computed efficiently.
    ///
    /// 3. **Parallelism**:
    ///    - This implementation leverages parallel iteration (`par_iter`) to achieve efficient
    ///      computation, especially for high-dimensional inputs.
    ///
    /// # Implementation Details
    /// - The function validates that the input polynomials and challenges have appropriate
    ///   dimensions.
    /// - It precomputes sums of the equality polynomial coefficients for efficient evaluation.
    /// - Results are accumulated in a thread-safe manner using intermediate buffers to avoid race
    ///   conditions.
    ///
    /// # Complexity
    /// - The complexity depends on:
    ///     - The number of polynomials (`n`),
    ///     - The number of chunks (`2^(dims-num_challenges)`)
    ///     - The chunk size (`2^num_challenges`).
    ///
    /// Parallelism mitigates the high computational cost for large input sizes.
    ///
    /// # Safety
    /// - The function assumes that input polynomials are properly formatted and aligned in memory.
    /// - Panics occur if dimensions do not match the required constraints.
    ///
    /// # Panics
    /// - If the number of coefficients in a polynomial does not match `2^dims`.
    /// - If the number of challenges exceeds `dims`.
    pub fn restrict(&self, challenges: &Points, dims: usize) -> Vec<BinaryField128b> {
        // Ensure that all input polynomials have the correct length of 2^n.
        // Panics if this condition is violated.
        self.iter().for_each(|poly| assert_eq!(poly.len(), 1 << dims));

        // Ensure the number of challenges does not exceed the dimensionality of the polynomials.
        let num_challenges = challenges.len();
        assert!(num_challenges <= dims);

        // Compute the chunk size and the number of chunks.
        // - Chunk size corresponds to 2^num_challenges.
        // - Number of chunks corresponds to 2^(dims - num_challenges).
        let chunk_size = 1 << num_challenges;
        let num_chunks = 1 << (dims - num_challenges);

        // Compute the equality polynomial for the given challenges.
        let eq = MultilinearLagrangianPolynomial::new_eq_poly(challenges);

        // Verify that the number of coefficients in the equality polynomial is divisible by 16.
        let eq_len = eq.len();
        assert_eq!(
            eq_len % 16,
            0,
            "The number of coefficients in the equality polynomial must be divisible by 16."
        );

        // Compute pre-summed coefficients for the equality polynomial.
        // This is used to efficiently evaluate sums over subsets of coefficients.
        let eq_sums = eq.eq_sums();

        // Initialize a result vector with zeros.
        // The size of the result vector is `num_chunks * 128 * n`, where:
        // - `num_chunks` is the number of chunks,
        // - `128` corresponds to the number of slots for each chunk,
        // - `n` is the number of input polynomials.
        let mut ret = vec![BinaryField128b::zero(); num_chunks * 128 * self.len()];

        // Iterate over each polynomial in the input set.
        self.iter().enumerate().for_each(|(q, poly)| {
            // Iterate over chunks in parallel using `par_iter` for efficiency.
            (0..num_chunks)
                .into_par_iter()
                .map(|i| {
                    // Allocate a thread-local buffer to store intermediate results for this chunk.
                    let mut local_ret = vec![BinaryField128b::zero(); 128];

                    // Process the equality polynomial sums in blocks of size 16.
                    for j in 0..eq_len / 16 {
                        // Retrieve two segments of precomputed sums (`v0` and `v1`):
                        // - `v0` corresponds to lower 8 bits.
                        // - `v1` corresponds to upper 8 bits.
                        let b1 = j * 512;
                        let b2 = b1 + 256;
                        let b3 = b2 + 256;
                        let v0 = &eq_sums[b1..b2];
                        let v1 = &eq_sums[b2..b3];

                        // Extract a block of coefficients from the current polynomial.
                        // This block corresponds to 16 bytes.
                        let bytearr = cast_slice::<BinaryField128b, [u8; 16]>(
                            &poly[i * chunk_size + j * 16..i * chunk_size + (j + 1) * 16],
                        );

                        // Process each byte in the extracted block.
                        for s in 0..16 {
                            // Extract the `s`-th byte from each element in the block.
                            let mut t = [0u8; 16];
                            for (idx, val) in bytearr.iter().enumerate() {
                                t[idx] = val[s];
                            }

                            // Process the 8 bits of the extracted byte, applying bitwise masks.
                            for u in 0..8 {
                                // Compute the bit mask for the current state of the array.
                                #[allow(clippy::cast_sign_loss)]
                                let bits = cpu_v_movemask_epi8(t) as usize;

                                // Update the local result buffer using the precomputed sums.
                                local_ret[s * 8 + 7 - u] += v0[bits & 255] + v1[(bits >> 8) & 255];

                                // Shift the array `t` left by one bit for the next iteration.
                                t = v_slli_epi64::<1>(t);
                            }
                        }
                    }

                    // Return the computed chunk result.
                    local_ret
                })
                .collect::<Vec<_>>()
                .iter()
                .for_each(|local_ret| {
                    for (idx, &value) in local_ret.iter().enumerate() {
                        // Update the global result vector with the chunk results.
                        ret[q * 128 * num_chunks + idx] += value;
                    }
                });
        });

        // Return the final result vector.
        ret
    }
}

impl From<Vec<MultilinearLagrangianPolynomial>> for MultilinearLagrangianPolynomials {
    fn from(polys: Vec<MultilinearLagrangianPolynomial>) -> Self {
        Self(polys)
    }
}

impl From<Vec<Vec<BinaryField128b>>> for MultilinearLagrangianPolynomials {
    fn from(coeffs: Vec<Vec<BinaryField128b>>) -> Self {
        Self(coeffs.into_iter().map(MultilinearLagrangianPolynomial::new).collect())
    }
}

impl Deref for MultilinearLagrangianPolynomials {
    type Target = Vec<MultilinearLagrangianPolynomial>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MultilinearLagrangianPolynomials {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_eq_poly() {
        // Define a simple set of points.
        let points =
            vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];

        // Compute the equality polynomial for the given points.
        let result = MultilinearLagrangianPolynomial::new_eq_poly(&points.into());

        // Assert that the computed equality polynomial matches the expected result.
        assert_eq!(
            result,
            MultilinearLagrangianPolynomial {
                coeffs: vec![
                    BinaryField128b::from(23667462636862719611022351736646926339),
                    BinaryField128b::from(79536576834697464894010492239055159303),
                    BinaryField128b::from(222989354442297682615051044822747971586),
                    BinaryField128b::from(106141905937829921662600755429978931204),
                    BinaryField128b::from(71478132110251412414531161936105570306),
                    BinaryField128b::from(294913050526722114925455479975104741381),
                    BinaryField128b::from(87449637247104542199890968644786585603),
                    BinaryField128b::from(225730887183604071602915146884576182279)
                ]
            }
        );
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_eq_poly_evaluate() {
        // Define a simple set of points.
        let pt1 = BinaryField128b::from(3434);
        let pt2 = BinaryField128b::from(6765);

        // Define the points in a vector.
        let points = vec![pt1, pt2];

        // Compute the equality polynomial for the given points.
        let result = MultilinearLagrangianPolynomial::new_eq_poly(&points.into());

        // Evaluate the equality polynomial at the given points.
        let expected_eq_poly =
            result[0] * (BinaryField128b::one() - pt1) * (BinaryField128b::one() - pt2) +
                result[2] * (BinaryField128b::one() - pt1) * pt2 +
                result[1] * (BinaryField128b::one() - pt2) * pt1 +
                result[3] * pt1 * pt2;

        // Verify that the equality polynomial evaluates to 1.
        // This ensures that the computed polynomial satisfies the expected equality conditions.
        assert_eq!(expected_eq_poly, BinaryField128b::one());
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_eq_poly_sequence() {
        // Define a vector of input points in the field BinaryField128b.
        let points = vec![
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
            BinaryField128b::from(4),
        ];

        // Compute the equality polynomial sequence using the eq_poly_sequence function.
        let eq_sequence = MultilinearLagrangianPolynomials::new_eq_poly_sequence(&points.into());

        // Assert that the computed equality polynomial sequence matches the expected result.
        // This ensures the function is working as intended.
        assert_eq!(
            eq_sequence,
            MultilinearLagrangianPolynomials(vec![
                MultilinearLagrangianPolynomial::new(vec![BinaryField128b::from(
                    257870231182273679343338569694386847745
                )]),
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(257870231182273679343338569694386847749),
                    BinaryField128b::from(4)
                ]),
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(276728653372472173290161332362114236431),
                    BinaryField128b::from(24175334173338157438437990908848766986),
                    BinaryField128b::from(24175334173338157438437990908848766989),
                    BinaryField128b::from(24175334173338157438437990908848766985)
                ]),
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(262194116391218557050997340512544882712),
                    BinaryField128b::from(28499219382283035146096761727006801943),
                    BinaryField128b::from(36391510607255973141463116147421347857),
                    BinaryField128b::from(12382329933390930187138101121107623963),
                    BinaryField128b::from(318146307338789859576041968956220637212),
                    BinaryField128b::from(336838576029515239038751755741412982801),
                    BinaryField128b::from(323629372821402637551770173079877058582),
                    BinaryField128b::from(299454038648064480113332182171028291615)
                ]),
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(156988152385753943545944842966784802826),
                    BinaryField128b::from(238399472904936139640581627536708993042),
                    BinaryField128b::from(103112231144489255218858846742755934222),
                    BinaryField128b::from(118147824772591482324176031488095027225),
                    BinaryField128b::from(214143658130290692373901413934757314572),
                    BinaryField128b::from(247871520449118298941928385465288753181),
                    BinaryField128b::from(114993504431031574539843754968093818891),
                    BinaryField128b::from(127368045919134702485539060344707612688),
                    BinaryField128b::from(138015824183221341818021192535835148297),
                    BinaryField128b::from(181502608447665156913871231917135757333),
                    BinaryField128b::from(311457361334396773361093192980979777548),
                    BinaryField128b::from(30864341024960054608398038725843484701),
                    BinaryField128b::from(197871324294196201012813621595002109966),
                    BinaryField128b::from(137763997785582402678037463463867973656),
                    BinaryField128b::from(30950013923125879264268791915275616264),
                    BinaryField128b::from(326990117386703710211842172749841694743)
                ])
            ])
        );
    }

    #[test]
    fn test_eq_poly_sequence_random_values() {
        use rand::Rng;

        // Define the number of iterations for the test.
        let iterations = 10;

        // Seed a random number generator.
        let mut rng = rand::thread_rng();

        for _ in 0..iterations {
            // Step 1: Generate random points for the test.
            let num_points = rng.gen_range(2..6); // Random number of points between 2 and 5.
            let points: Vec<_> = (0..num_points).map(|_| BinaryField128b::random()).collect();

            // Step 2: Compute the equality polynomial using `new_eq_poly`.
            let result = MultilinearLagrangianPolynomial::new_eq_poly(&points.clone().into());

            // Step 3: Reconstruct the equality polynomial manually to verify correctness.
            let mut expected_eq_poly = BinaryField128b::zero();
            for i in 0..(1 << num_points) {
                // Convert `i` to binary representation to match the current point combination.
                let binary_combination: Vec<bool> =
                    (0..num_points).map(|j| (i & (1 << j)) != 0).collect();

                // Calculate the term for the current binary combination.
                let mut term = result[i];
                for (bit, point) in binary_combination.iter().zip(&points) {
                    term *= if *bit {
                        *point // Include the point if the bit is 1.
                    } else {
                        BinaryField128b::one() - *point // Complement if the bit is 0.
                    };
                }
                expected_eq_poly += term;
            }

            // Step 4: Assert that the computed equality polynomial evaluates to 1.
            assert_eq!(expected_eq_poly, BinaryField128b::one());
        }
    }

    #[test]
    fn test_eq_poly_sequence_cross_check() {
        // Generate a random vector of 20 points in the finite field.
        let points = Points::from((0..20).map(|_| BinaryField128b::random()).collect::<Vec<_>>());

        // Compute the equality polynomial sequence for the points.
        let eq_sequence = MultilinearLagrangianPolynomials::new_eq_poly_sequence(&points);

        // Verify the sequence length matches the expected size (points.len() + 1).
        assert_eq!(eq_sequence.len(), points.len() + 1);

        // Verify the initial polynomial is [1].
        assert_eq!(
            eq_sequence[0],
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::one()])
        );

        // Cross-check each polynomial in the sequence with its direct computation.
        eq_sequence.iter().enumerate().skip(1).for_each(|(i, poly)| {
            assert_eq!(
                poly,
                &MultilinearLagrangianPolynomial::new_eq_poly(&points[points.len() - i..].into())
            );
        });
    }

    #[test]
    fn test_evaluate() {
        // Define a simple multilinear polynomial.
        // Coefficients represent the polynomial's evaluation at the points (0, 0), (0, 1), (1, 0),
        // (1, 1).
        let coeff0 = BinaryField128b::random(); // p(0, 0)
        let coeff1 = BinaryField128b::random(); // p(0, 1)
        let coeff2 = BinaryField128b::random(); // p(1, 0)
        let coeff3 = BinaryField128b::random(); // p(1, 1)

        // Create a multilinear polynomial with the coefficients.
        let polynomial = MultilinearLagrangianPolynomial::new(vec![coeff0, coeff1, coeff2, coeff3]);

        // Define the evaluation points.
        let points = Points::from(vec![BinaryField128b::random(), BinaryField128b::random()]);

        // Compute the evaluation.
        let result = polynomial.evaluate_at(&points);

        // Manually compute the expected result.
        // Using the formula for multilinear polynomial evaluation:
        // result = p(0, 0) * (1 - pt0) * (1 - pt1)
        //        + p(0, 1) * (1 - pt1) * pt0
        //        + p(1, 0) * pt1 * (1 - pt0)
        //        + p(1, 1) * pt1 * pt0
        let expected_result = coeff0 *
            (BinaryField128b::one() - *points[0]) *
            (BinaryField128b::one() - *points[1]) +
            coeff2 * (BinaryField128b::one() - *points[0]) * *points[1] +
            coeff1 * *points[0] * (BinaryField128b::one() - *points[1]) +
            coeff3 * *points[0] * *points[1];

        // Assert the result matches the expectation.
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_multilinear_lagrangian_bitand() {
        // Define two simple multilinear polynomials.
        // Coefficients represent evaluations at points (0, 0), (0, 1), (1, 0), (1, 1).
        let poly1_coeffs = vec![
            BinaryField128b::from(1), // p1(0, 0)
            BinaryField128b::from(0), // p1(0, 1)
            BinaryField128b::from(1), // p1(1, 0)
            BinaryField128b::from(1), // p1(1, 1)
        ];
        let poly2_coeffs = vec![
            BinaryField128b::from(1), // p2(0, 0)
            BinaryField128b::from(1), // p2(0, 1)
            BinaryField128b::from(0), // p2(1, 0)
            BinaryField128b::from(1), // p2(1, 1)
        ];

        // Create two multilinear polynomials with the coefficients.
        let poly1 = MultilinearLagrangianPolynomial::new(poly1_coeffs);
        let poly2 = MultilinearLagrangianPolynomial::new(poly2_coeffs);

        // Step 2: Compute the bitwise AND of the two polynomials.
        let result = poly1 & poly2;

        // Step 3: Manually compute the expected result.
        // Element-wise AND operation on the coefficients:
        let expected_coeffs = vec![
            BinaryField128b::from(1), // 1 & 1 = 1
            BinaryField128b::from(0), // 0 & 1 = 0
            BinaryField128b::from(0), // 1 & 0 = 0
            BinaryField128b::from(1), // 1 & 1 = 1
        ];

        // Create the expected polynomial.
        let expected_result = MultilinearLagrangianPolynomial::new(expected_coeffs);

        // Step 4: Assert the result matches the expected polynomial.
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_eq_sums() {
        // Define a set of points to construct the equality polynomial.
        let points = Points::from(vec![
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
            BinaryField128b::from(4),
        ]);

        // Create the equality polynomial for the points.
        let polynomial = MultilinearLagrangianPolynomial::new_eq_poly(&points);

        // Compute the equality sums.
        let eq_sums = polynomial.eq_sums();

        // Initialize a vector to store all subset sums.
        let mut expected_sums = Vec::new();

        // The polynomial is divided into blocks of 8 coefficients,
        // as each equality block in `eq_sums` corresponds to 8 coefficients.
        for block_start in (0..polynomial.len()).step_by(8) {
            // Extract the current block of 8 coefficients from the polynomial.
            let block = &polynomial[block_start..block_start + 8];

            // Initialize an array to store sums for all 256 subsets of the block.
            let mut block_sums = [BinaryField128b::zero(); 256];

            // Compute all possible sums of the coefficients in the block.
            for (subset, block_sums_subset) in block_sums.iter_mut().enumerate() {
                // Initialize the sum for the current subset.
                let mut sum = BinaryField128b::zero();

                // Iterate through the 8 bits of the subset index.
                for (bit, &block_bit) in block.iter().enumerate().take(8) {
                    // Check if the `bit`-th coefficient is part of the current subset.
                    if (subset & (1 << bit)) != 0 {
                        // Add the `bit`-th coefficient to the subset sum.
                        sum += block_bit;
                    }
                }

                // Store the computed sum in the array.
                *block_sums_subset = sum;
            }

            // Append the subset sums from the block to the overall result vector.
            expected_sums.extend(block_sums);
        }

        // Assert the computed equality sums match the expected result.
        assert_eq!(eq_sums, expected_sums);
    }
}
