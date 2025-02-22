use crate::{
    binary_field::BinaryField128b,
    poly::{evaluation::Evaluations, point::Points},
    utils::{cpu_v_movemask_epi8, drop_top_bit, v_slli_epi64},
};
use bytemuck::{cast_slice, zeroed_vec};
use num_traits::Zero;
use rand::Rng;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::{
    ops::{BitAnd, Deref, DerefMut},
    slice::{Iter, IterMut},
    sync::atomic::{AtomicPtr, Ordering},
};

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

    /// Generates a random multilinear Lagrangian polynomial with the given number of coefficients.
    pub fn random<RNG: Rng>(n: usize, rng: &mut RNG) -> Self {
        (0..n).map(|_| BinaryField128b::random(rng)).collect()
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
        assert_eq!(self.coeffs.len(), 1 << points.len());

        // Compute the weighted sum of coefficients.
        // Each coefficient is multiplied by the corresponding weight from the equality polynomial.
        self.coeffs
            .par_iter()
            .zip(points.to_eq_poly().coeffs)
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
            let mut block_sums = [BinaryField128b::ZERO; 256];

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

impl<'a> IntoIterator for &'a MultilinearLagrangianPolynomial {
    type Item = &'a BinaryField128b;
    type IntoIter = Iter<'a, BinaryField128b>;

    fn into_iter(self) -> Self::IntoIter {
        self.coeffs.iter()
    }
}

impl FromIterator<BinaryField128b> for MultilinearLagrangianPolynomial {
    fn from_iter<T: IntoIterator<Item = BinaryField128b>>(iter: T) -> Self {
        Self { coeffs: iter.into_iter().collect() }
    }
}

impl From<Vec<BinaryField128b>> for MultilinearLagrangianPolynomial {
    fn from(coeffs: Vec<BinaryField128b>) -> Self {
        Self::new(coeffs)
    }
}

impl<'a> IntoIterator for &'a mut MultilinearLagrangianPolynomial {
    type Item = &'a mut BinaryField128b;
    type IntoIter = IterMut<'a, BinaryField128b>;

    fn into_iter(self) -> Self::IntoIter {
        self.coeffs.iter_mut()
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

impl DerefMut for MultilinearLagrangianPolynomial {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.coeffs
    }
}

/// Applies the restriction operation to a set of multilinear Lagrangian polynomials.
///
/// # Overview
/// The `restrict` function evaluates a set of multilinear Lagrangian polynomials
/// over a restricted subset of their domain, as defined by `challenges`.
///
/// # Parameters
/// - `polys`: A reference to an array of `MultilinearLagrangianPolynomial` structures, which
///   contain one or more multilinear Lagrangian polynomials.
/// - `challenges`: A slice of `BinaryField128b` values specifying the challenge points that define
///   the restriction.
/// - `dims`: The dimensionality of the domain of the input polynomials (i.e., the number of
///   variables in the Boolean hypercube).
///
/// # Returns
/// A `Vec<BinaryField128b>` containing the computed restricted evaluations for all input
/// polynomials.
///
/// # Key Concepts
/// 1. **Equality Polynomial**:
///    - For each `challenges` point, the equality polynomial encodes interpolation conditions that
///      ensure correct evaluation at the given challenges.
///
/// 2. **Subset Sums**:
///    - Subsets of polynomial coefficients are grouped into chunks, and sums over these subsets are
///      computed efficiently.
///
/// 3. **Parallelism**:
///    - This implementation leverages parallel iteration (`par_iter`) to achieve efficient
///      computation, especially for high-dimensional inputs.
///
/// # Implementation Details
/// - The function validates that the input polynomials and challenges have appropriate dimensions.
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
pub fn restrict<const N: usize>(
    polys: &[MultilinearLagrangianPolynomial; N],
    challenges: &Points,
    dims: usize,
) -> Evaluations {
    // Ensure that all input polynomials have the correct length of 2^n.
    // Panics if this condition is violated.
    polys.iter().for_each(|poly| assert_eq!(poly.len(), 1 << dims));

    // Ensure the number of challenges does not exceed the dimensionality of the polynomials.
    let num_challenges = challenges.len();
    assert!(num_challenges <= dims);

    // Compute the chunk size and the number of chunks.
    // - Chunk size corresponds to 2^num_challenges.
    // - Number of chunks corresponds to 2^(dims - num_challenges).
    let chunk_size = 1 << num_challenges;
    let num_chunks = 1 << (dims - num_challenges);

    // Compute the equality polynomial for the given challenges.
    let eq = challenges.to_eq_poly();

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
    let mut ret = zeroed_vec::<BinaryField128b>(num_chunks * 128 * N);

    // Create an atomic pointer to the `ret` vector for shared mutable access.
    let ret_ptr = AtomicPtr::new(ret.as_mut_ptr());

    // Iterate over each polynomial in the input set.
    polys.par_iter().enumerate().for_each(|(q, poly)| {
        // Compute the starting index for the current polynomial in the result vector.
        let ret_chunk_start = q * 128 * num_chunks;

        // Iterate over chunks in parallel using `par_iter` for efficiency.
        (0..num_chunks).into_par_iter().for_each(|i| {
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

                        // Update the result vector using the computed sums.
                        unsafe {
                            *ret_ptr
                                .load(Ordering::SeqCst)
                                .add(ret_chunk_start + (s * 8 + 7 - u) * num_chunks + i) +=
                                v0[bits & 255] + v1[(bits >> 8) & 255];
                        }

                        // Shift the array `t` left by one bit for the next iteration.
                        t = v_slli_epi64::<1>(t);
                    }
                }
            }
        });
    });

    // Return the final evaluations.
    Evaluations::new(ret)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::point::to_eq_poly_sequence;
    use rand::rngs::OsRng;

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_eq_poly() {
        // Define a simple set of points.
        let points = Points::from(vec![
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
        ]);

        // Compute the equality polynomial for the given points.
        let result = points.to_eq_poly();

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
        let points = Points::from(vec![pt1, pt2]);

        // Compute the equality polynomial for the given points.
        let result = points.to_eq_poly();

        // Evaluate the equality polynomial at the given points.
        let expected_eq_poly =
            result[0] * (BinaryField128b::ONE - pt1) * (BinaryField128b::ONE - pt2) +
                result[2] * (BinaryField128b::ONE - pt1) * pt2 +
                result[1] * (BinaryField128b::ONE - pt2) * pt1 +
                result[3] * pt1 * pt2;

        // Verify that the equality polynomial evaluates to 1.
        // This ensures that the computed polynomial satisfies the expected equality conditions.
        assert_eq!(expected_eq_poly, BinaryField128b::ONE);
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_eq_poly_sequence() {
        // Define a vector of input points in the field BinaryField128b.
        let points = Points::from(vec![
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
            BinaryField128b::from(4),
        ]);

        // Compute the equality polynomial sequence using the eq_poly_sequence function.
        let eq_sequence = to_eq_poly_sequence(&points);

        // Assert that the computed equality polynomial sequence matches the expected result.
        // This ensures the function is working as intended.
        assert_eq!(
            eq_sequence,
            vec![
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
            ]
        );
    }

    #[test]
    fn test_eq_poly_sequence_random_values() {
        use rand::Rng;

        let rng_pts = &mut OsRng;

        // Define the number of iterations for the test.
        let iterations = 10;

        // Seed a random number generator.
        let mut rng = rand::thread_rng();

        for _ in 0..iterations {
            // Step 1: Generate random points for the test.
            let num_points = rng.gen_range(2..6); // Random number of points between 2 and 5.
            let points = Points::random(num_points, rng_pts);

            // Step 2: Compute the equality polynomial using `new_eq_poly`.
            let result = points.to_eq_poly();

            // Step 3: Reconstruct the equality polynomial manually to verify correctness.
            let mut expected_eq_poly = BinaryField128b::ZERO;
            for i in 0..(1 << num_points) {
                // Convert `i` to binary representation to match the current point combination.
                let binary_combination: Vec<bool> =
                    (0..num_points).map(|j| (i & (1 << j)) != 0).collect();

                // Calculate the term for the current binary combination.
                let mut term = result[i];
                for (bit, point) in binary_combination.iter().zip(&points.0) {
                    term *= if *bit {
                        *point // Include the point if the bit is 1.
                    } else {
                        BinaryField128b::ONE - *point // Complement if the bit is 0.
                    };
                }
                expected_eq_poly += term;
            }

            // Step 4: Assert that the computed equality polynomial evaluates to 1.
            assert_eq!(expected_eq_poly, BinaryField128b::ONE);
        }
    }

    #[test]
    fn test_evaluate() {
        let rng = &mut OsRng;

        // Define a simple multilinear polynomial.
        // Coefficients represent the polynomial's evaluation at the points (0, 0), (0, 1), (1, 0),
        // (1, 1).
        let coeff0 = BinaryField128b::random(rng); // p(0, 0)
        let coeff1 = BinaryField128b::random(rng); // p(0, 1)
        let coeff2 = BinaryField128b::random(rng); // p(1, 0)
        let coeff3 = BinaryField128b::random(rng); // p(1, 1)

        // Create a multilinear polynomial with the coefficients.
        let polynomial = MultilinearLagrangianPolynomial::new(vec![coeff0, coeff1, coeff2, coeff3]);

        // Define the evaluation points.
        let points = Points::random(2, rng);

        // Compute the evaluation.
        let result = polynomial.evaluate_at(&points);

        // Manually compute the expected result.
        // Using the formula for multilinear polynomial evaluation:
        // result = p(0, 0) * (1 - pt0) * (1 - pt1)
        //        + p(0, 1) * (1 - pt1) * pt0
        //        + p(1, 0) * pt1 * (1 - pt0)
        //        + p(1, 1) * pt1 * pt0
        let expected_result =
            coeff0 * (BinaryField128b::ONE - points[0]) * (BinaryField128b::ONE - points[1]) +
                coeff2 * (BinaryField128b::ONE - points[0]) * points[1] +
                coeff1 * points[0] * (BinaryField128b::ONE - points[1]) +
                coeff3 * points[0] * points[1];

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
        let polynomial = points.to_eq_poly();

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
            let mut block_sums = [BinaryField128b::ZERO; 256];

            // Compute all possible sums of the coefficients in the block.
            for (subset, block_sums_subset) in block_sums.iter_mut().enumerate() {
                // Initialize the sum for the current subset.
                let mut sum = BinaryField128b::ZERO;

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

    #[test]
    #[allow(clippy::unreadable_literal, clippy::too_many_lines)]
    fn test_restrict() {
        // Define the number of variables
        let num_vars: usize = 4;
        // Define the number of variables to restrict
        let num_vars_to_restrict = 4;

        // Define the restriction points.
        let points: Points = (0..num_vars_to_restrict).map(BinaryField128b::new).collect();

        // Define three multilinear polynomials.
        let poly0: MultilinearLagrangianPolynomial =
            (0..(1 << num_vars)).map(BinaryField128b::new).collect();
        let poly1: MultilinearLagrangianPolynomial =
            (0..(1 << num_vars)).map(|i| BinaryField128b::new(i + 1)).collect();
        let poly2: MultilinearLagrangianPolynomial =
            (0..(1 << num_vars)).map(|i| BinaryField128b::new(i + 2)).collect();

        // Restrict the polynomials to the given points.
        let restricted_polynomials = restrict(&[poly0, poly1, poly2], &points, num_vars);

        // Verify the restricted evaluations match the expected results.
        assert_eq!(
            restricted_polynomials,
            Evaluations::new(vec![
                BinaryField128b::ZERO,
                BinaryField128b::new(1),
                BinaryField128b::new(2),
                BinaryField128b::new(3),
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::new(257870231182273679343338569694386847745),
                BinaryField128b::new(1),
                BinaryField128b::new(2),
                BinaryField128b::new(3),
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::new(257870231182273679343338569694386847744),
                BinaryField128b::new(3),
                BinaryField128b::new(305763977405398929388903867835113013248),
                BinaryField128b::new(225730887183604071602915146884576182279),
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO
            ])
        );
    }
}
