use algebraic::AlgebraicOps;
use hashcaster_poly::{
    compressed::CompressedPoly,
    evaluation::Evaluations,
    multinear_lagrangian::{MultilinearLagrangianPolynomial, MultilinearLagrangianPolynomials},
    point::{Point, Points},
    univariate::UnivariatePolynomial,
};
use hashcaster_primitives::binary_field::BinaryField128b;
use num_traits::{One, Zero};
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::array;

pub mod algebraic;
pub mod and;
pub mod bool_trait;
pub mod builder;
pub mod ternary_mapping;

/// A struct representing Boolean check protocol.
///
/// The `BoolCheck` struct encapsulates the components and state needed to perform
/// Boolean check protocol. It uses multilinear polynomials and equality sequences to validate
/// claims iteratively over multiple rounds.
///
/// # Type Parameters
/// - `N`: The number of multilinear polynomials used in the protocol.
/// - `M`: The number of folding challenges (gammas) used for compression.
#[derive(Clone, Debug)]
pub struct BoolCheck<const N: usize, const M: usize, A: AlgebraicOps<N, M>> {
    /// The evaluation points for the multilinear polynomials.
    pub points: Points,

    /// An array of multilinear polynomials used in the protocol.
    pub polys: [MultilinearLagrangianPolynomial; N],

    /// A vector storing intermediate evaluations of the polynomials.
    pub extended_table: Vec<BinaryField128b>,

    /// An optional field storing evaluations on a restricted subset of the hypercube.
    poly_coords: Option<Evaluations>,

    /// The phase switch parameter, indicating the number of rounds in phase one.
    pub c: usize,

    /// A sequence of random challenges provided by the verifier.
    pub challenges: Points,

    /// A vector of bit mappings for optimized indexing of polynomial coefficients.
    pub bit_mapping: Vec<u16>,

    /// A sequence of equality polynomials used to verify claims.
    pub eq_sequence: MultilinearLagrangianPolynomials,

    /// A vector of compressed polynomials computed at each round.
    pub round_polys: Vec<CompressedPoly>,

    /// The current claim being verified in the protocol.
    pub claim: BinaryField128b,

    /// An array of folding challenges used for polynomial compression.
    pub gammas: [BinaryField128b; M],

    /// Abstract algebraic operations.
    pub algebraic_operations: A,
}

impl<const N: usize, const M: usize, A: AlgebraicOps<N, M> + Default> Default
    for BoolCheck<N, M, A>
{
    fn default() -> Self {
        Self {
            points: Points::default(),
            polys: array::from_fn(|_| Default::default()),
            extended_table: Vec::new(),
            poly_coords: None,
            c: 0,
            challenges: Points::default(),
            bit_mapping: Vec::new(),
            eq_sequence: MultilinearLagrangianPolynomials::default(),
            round_polys: Vec::new(),
            claim: BinaryField128b::zero(),
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: Default::default(),
        }
    }
}

impl<const N: usize, const M: usize, A: AlgebraicOps<N, M> + Send + Sync> BoolCheck<N, M, A> {
    /// Returns the current round of the protocol.
    ///
    /// The current round is represented by the number of challenges that have been submitted to the
    /// prover.
    pub fn current_round(&self) -> usize {
        self.challenges.len()
    }

    /// Returns the number of evaluation points for the polynomials in the protocol.
    ///
    /// For examples, the multilinear polynomial `p` can be written as:
    /// `p(x) = p_0 + p_1*x_1 + p_2*x_2 + ... + p_n*x_n`
    ///
    /// The number of variables in the polynomial is `n`.
    pub fn number_variables(&self) -> usize {
        self.points.len()
    }

    pub fn compute_round_polynomial(&mut self) -> CompressedPoly {
        // Compute the current round of the protocol.
        let round = self.current_round();

        // Compute the number of variables in the polynomial.
        let number_variables = self.number_variables();

        // In the sumcheck protocol, the number of rounds is limited by the number of variables.
        assert!(round < number_variables, "Protocol has reached the maximum number of rounds.");

        // If the round polynomial has already been computed, return it.
        //
        // We have all the computed polynomials up to the current round in the cache.
        if self.round_polys.len() > round {
            // Return the polynomial from the cache.
            if let Some(poly) = self.round_polys.last() {
                return poly.clone();
            }
            // This situation should never happen.
            return CompressedPoly::default();
        }

        // The protocol is defined in two phases:
        // - The first phase last `c` rounds. It operates over the quadratic polynomial `P Ʌ Q` on
        //   the subset `(0,1,∞)^{c+1} x (0,1)^{n-c-1}`
        // - After `c` phases, the prover no longer has the required data (as we have computed the
        //   extension table only on `3^{c+1} \times 2^{n-c-1}` subset) and cannot proceed further.
        //   This is when we change strategy and compute `P_i(r_0, \ldots, r_c, \mathbf{x}_{>c})`
        //   and `Q_i(r_0, \ldots, r_c, \mathbf{x}_{>c})` (using RESTRICT functionality).
        let poly_deg2 = if round <= self.c {
            // Fetch the equality polynomial for the current round.
            //
            // The equality polynomials for all the points are stored in cache during the setup
            // phase.
            let eq_poly_round = self.eq_sequence.poly_at(number_variables - round - 1);

            // The first phase of the protocol lasts `c` rounds.
            //
            // This is the number of remaining rounds in the first phase.
            let phase1_dimensions = self.c - round;

            // Compute `3^{remaining rounds in the first phase}`.
            let pow3 = 3usize.pow(phase1_dimensions as u32);

            // On the `i-th` round (`i ≤ c`), the computation maintains the restriction:
            // `(P ∧ Q)(r_0,..., r_{i-1}, x_i, ..., x_{n-1})`.
            //
            // When responding to a query, the univariate polynomial `U(t)` is computed as:
            //
            // ```
            // U(t) = Σ_{x_{>i}} (P ∧ Q)(r_{<i}, t, x_{>i}) * eq(r_{<i}, t, x_{>i}; q),
            // ```
            // where `eq` exploits the splitting:
            // ```
            //  eq(r_{<i}, t, x_{>i}; q) = eq(r_{<i}, q_{<i}) * eq(t; q_i) * eq(x_{>i}; q_{>i}).
            // ```
            //
            // This factorization allows the computation to avoid directly manipulating
            // polynomials of degree `3` until the final stage. Specifically:
            // - Terms `eq(r_{<i}, q_{<i}) * eq(t; q_i)` are independent of `x_{>i}` and can be
            //   factored out.
            //
            // By evicting the independent terms, the computation simplifies to the sum:
            //
            // ```
            //  W(t) = Σ_{x_{>i}} (P ∧ Q)(r_{<i}, t, x_{>i}) * eq(x_{>i}; q_{>i}).
            // ```
            (0..(1 << (number_variables - self.c - 1)))
                .into_par_iter()
                .map(|i| {
                    // Base index is: `ì * 2^{number of remaining rounds in the first phase}`.
                    let base_index = i << phase1_dimensions;

                    // Base offset is: `i * 3^{remaining rounds in the first phase + 1}`.
                    let base_offset = 3 * (i * pow3);

                    // Intermediate polynomial for the current round.
                    //
                    // This polynomial is a degree-2 polynomial.
                    let mut pd2_part = [BinaryField128b::zero(); 3];

                    (0..(1 << phase1_dimensions)).for_each(|j| {
                        // The offset is a way to locate the correct position in the extended table
                        // where `(P ∧ Q)(r_{<i}, t, x_{>i})` is stored.
                        let offset = base_offset + 3 * self.bit_mapping[j] as usize;

                        // The multiplier is the equality polynomial at the appropriate location
                        // such that it corresponds to:
                        // `eq(x_{>i};q_{>i})`.
                        let multiplier = eq_poly_round[base_index + j];

                        // Compute:
                        // ```
                        //  W(t) = Σ_{x_{>i}} (P ∧ Q)(r_{<i}, t, x_{>i}) * eq(x_{>i}; q_{>i}).
                        // ```
                        // for `t = 0, 1, ∞`.
                        pd2_part
                            .iter_mut()
                            .zip(&self.extended_table[offset..offset + 3])
                            .for_each(|(acc, value)| *acc += *value * multiplier);
                    });

                    pd2_part
                })
                .reduce(
                    // Finally, we compute
                    // ```
                    //  W(t) = Σ_{x_{>i}} (P ∧ Q)(r_{<i}, t, x_{>i}) * eq(x_{>i}; q_{>i}).
                    // ```
                    // as a sum of the intermediate polynomials for all `t = 0, 1, ∞`.
                    || [BinaryField128b::zero(); 3],
                    |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                )
        } else {
            // During this second phase, after `c` rounds, the prover no longer has the required
            // data because the extension table was computed only on the `3^{c+1} x 2^{n-c-1}`
            // subset.

            // Fetch the equality polynomial for the current round.
            let eq_poly_round = self.eq_sequence.poly_at(self.points.len() - round - 1);

            // Fetch the number of coefficients in the equality polynomial (Lagrangian polynomial).
            let half = eq_poly_round.len();

            // At the current round, the equality polynomial should have `2^{n-round-1}`
            // coefficients.
            assert_eq!(
                half,
                1 << (number_variables - round - 1),
                "Invalid equality polynomial size"
            );

            // Poly coordinates are required for the computation of `P ∧ Q` in the restricted phase.
            let poly_coords = self.poly_coords.as_ref().unwrap();

            // Here we want to compute (in the restricted phase):
            //
            // ```
            //  W(t) = Σ_{x_{>i}} (P ∧ Q)(r_{<i}, t, x_{>i}) * eq(x_{>i}; q_{>i}).
            // ```
            (0..half)
                .into_par_iter()
                .map(|i| {
                    self.compute_algebraic(poly_coords, i, 1 << (number_variables - self.c - 1))
                        .map(|x| x * eq_poly_round[i])
                })
                .reduce(
                    || [BinaryField128b::zero(); 3],
                    |[a, b, c], [d, e, f]| [a + d, b + e, c + f],
                )
        };

        // We want to compute:
        // ```
        // U(t) = eq(r_{<i}; q) eq(t; q) W(t)
        // ```
        //
        // The first step is to compute the equality polynomial `eq(r_{<i}; q)`.
        let eq_y_multiplier = self.challenges.eq_eval_slice(&self.points[..round]);

        // 1. From the intermediate evaluations of the degree-2 polynomial:
        // - `P(0), P(1), P(∞)`,
        // we can construct a degree-2 univariate polynomial in coefficient form.
        //
        // 2. We multiply the polynomial by the equality polynomial to obtain:
        // - `V(t) = eq(r_{<i}; q) W(t)`.
        let univariate_poly_deg2 =
            UnivariatePolynomial::from_evaluations_deg2(poly_deg2) * eq_y_multiplier;

        // We then need to compute the equality polynomial `eq(t; q)`.
        //
        // This is the equality polynomial for the current round (degree-1 univariate
        // polynomial).
        let eq_t = UnivariatePolynomial::new(vec![
            *self.points[round] + BinaryField128b::one(),
            BinaryField128b::one(),
        ]);

        // In the final step, we multiply the univariate polynomial by the equality polynomial:
        // ```
        // U(t) = eq(t; q) V(t)
        // ```
        let u = univariate_poly_deg2.multiply_degree2_by_degree1(&eq_t);

        // Compress the polynomial to obtain:
        // - The compressed round polynomial.
        // - The computed claim.
        let (ret, computed_claim) = CompressedPoly::compress(&u);

        // Check that the computed claim matches the expected value.
        assert_eq!(computed_claim, self.claim, "Claim does not match expected value.");

        // Check that the number of cached round polynomials is correct.
        // It should be equal to the current round.
        assert_eq!(
            self.round_polys.len(),
            round,
            "The number of cached round polynomials is incorrect."
        );

        // Cache the computed round polynomial.
        self.round_polys.push(ret.clone());

        // Return the computed polynomial.
        ret
    }

    pub fn compute_algebraic(
        &self,
        data: &[BinaryField128b],
        idx_a: usize,
        offset: usize,
    ) -> [BinaryField128b; 3] {
        // Compute the algebraic operations for the given data slice.
        let tmp = self.algebraic_operations.algebraic(data, idx_a, offset);

        // Initialize the accumulators for each of the 3 output values to zero.
        let mut acc = [BinaryField128b::zero(); 3];

        // Iterate over the output size `M` and compute the folded sums for each output value.
        for i in 0..M {
            acc[0] += tmp[0][i] * self.gammas[i];
            acc[1] += tmp[1][i] * self.gammas[i];
            acc[2] += tmp[2][i] * self.gammas[i];
        }

        // Return the array of compressed results.
        acc
    }

    pub fn bind(&mut self, r: &Point) {
        // Compute the current round of the protocol.
        let round = self.current_round();

        // Compute the number of variables in the polynomial.
        let number_variables = self.number_variables();

        // In the sumcheck protocol, the number of rounds is limited by the number of variables.
        assert!(round < number_variables, "Protocol has reached the maximum number of rounds.");

        // Compute the round polynomial for the current round:
        // - The round polynomial is a univariate polynomial in `r`.
        // - We first compute the compressed form of the polynomial.
        // - We then decompress the polynomial to obtain the coefficients.
        let round_poly = self.compute_round_polynomial().coeffs(self.claim);
        // Update the current claim:
        // - We evaluate the round polynomial at the random value `r` provided by the verifier.
        self.claim = round_poly.evaluate_at(r);

        // Add the random value sent by the verifier to the list of gammas.
        self.challenges.push(r.clone());

        // Compute `r^2` for future usage in the protocol.
        let r2 = **r * **r;

        if round <= self.c {
            // We compute, by chunk of 3, the new values for the extended table:
            // ```
            // P(0, r) = P(0, r) + (P(0, r) + P(1, r) + P(∞, r)) * r + P(∞, r) * r^2
            // ```
            self.extended_table = self
                .extended_table
                .par_chunks(3)
                .map(|chunk| chunk[0] + (chunk[0] + chunk[1] + chunk[2]) * **r + chunk[2] * r2)
                .collect();
        } else {
            // At each round we halve the number of variables in the hypercube.
            //
            // `half` represents `2^{n-round-1} = 2^{n-round} / 2`.
            let half = 1 << (number_variables - round - 1);

            // Compute the new values for the poly coordinates:
            self.poly_coords
                .as_mut()
                .unwrap()
                .par_chunks_mut(1 << (number_variables - self.c - 1))
                .for_each(|chunk| {
                    for j in 0..half {
                        chunk[j] = chunk[2 * j] + (chunk[2 * j + 1] + chunk[2 * j]) * **r;
                    }
                });
        }

        // Transition to phase 2 if the current round equals `c + 1`.
        //
        // Here we fetch the current round again as it may have changed during the computation.
        // If the current round is equal to `c + 1`:
        // 1. A new challenge has been submitted during binding.
        // 2. `self.current_round()` now returns the updated round.
        // 3. We can transition to phase 2 by applying the `restrict` functionality.
        if self.current_round() == self.c + 1 {
            // At the end of the first phase, we do not need the extended table anymore.
            self.extended_table = Vec::new();

            // Restrict the polynomial coordinates based on the accumulated challenges.
            self.poly_coords = Some(
                MultilinearLagrangianPolynomials::from(self.polys.to_vec())
                    .restrict(&self.challenges, number_variables),
            );
        }
    }

    pub fn finish(&self) -> BoolCheckOutput {
        // Compute the number of variables in the polynomial.
        let number_variables = self.number_variables();
        // The number of rounds should match the number of variables for the protocol to end.
        assert_eq!(number_variables, self.current_round(), "Protocol has not reached the end.");

        // Decompose the `BoolCheck` instance to extract the required fields.
        let Self { poly_coords, round_polys, c, .. } = self;

        // Unwrap the polynomial coordinates, ensuring they are initialized.
        let poly_coords = poly_coords.clone().unwrap();

        // Compute the evaluations on the Frobenius subdomain.
        let base_index = 1 << (number_variables - c - 1);
        let mut frob_evals: Evaluations = (0..128 * self.polys.len())
            .map(|idx| {
                let poly_idx = idx / 128;
                let frob_idx = idx % 128;
                poly_coords[(poly_idx * 128 + frob_idx) * base_index]
            })
            .collect::<Vec<_>>()
            .into();

        // Apply the twist operation to each chunk of 128 evaluations.
        frob_evals.chunks_mut(128).for_each(|chunk| Evaluations::from(chunk.to_vec()).twist());

        // Return the `BoolCheckOutput`
        BoolCheckOutput { frob_evals, round_polys: round_polys.clone() }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BoolCheckOutput {
    /// Evaluations of the polynomials on a Frobenius subdomain.
    pub frob_evals: Evaluations,

    /// A vector of compressed polynomials computed during the protocol's rounds.
    pub round_polys: Vec<CompressedPoly>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use and::AndPackage;
    use builder::BoolCheckBuilder;
    use hashcaster_poly::{multinear_lagrangian::MultilinearLagrangianPolynomial, point::Point};

    #[test]
    fn test_current_rounds() {
        // Define a sample BoolCheck instance.
        // No challenges have been submitted yet.
        let mut bool_check = BoolCheck::<0, 0, AndPackage<0, 0>>::default();

        // Assert the initial round (no challenges yet).
        assert_eq!(bool_check.current_round(), 0);

        // Add a challenge and test the round count.
        bool_check.challenges.push(Point::from(BinaryField128b::from(10)));
        assert_eq!(bool_check.current_round(), 1);

        // Add more challenges and test the updated round count.
        bool_check.challenges.extend(vec![
            Point::from(BinaryField128b::from(20)),
            Point::from(BinaryField128b::from(30)),
            Point::from(BinaryField128b::from(40)),
        ]);
        assert_eq!(bool_check.current_round(), 4);

        // Add another challenge and verify the round count again.
        bool_check.challenges.push(BinaryField128b::from(50).into());
        assert_eq!(bool_check.current_round(), 5);
    }

    #[test]
    fn test_number_variables() {
        // Create a BoolCheck instance with a defined number of points (variables).
        let points =
            vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];
        let bool_check = BoolCheck::<0, 0, AndPackage<0, 0>> {
            points: points.clone().into(),
            ..Default::default()
        };

        // Assert that the number of variables matches the length of the points vector.
        assert_eq!(bool_check.number_variables(), points.len());

        // Test with no points (empty vector).
        let empty_bool_check =
            BoolCheck::<0, 0, AndPackage<0, 0>> { points: Points::default(), ..Default::default() };
        assert_eq!(empty_bool_check.number_variables(), 0);

        // Test with a large number of points.
        let large_points: Vec<_> = (0..1000).map(BinaryField128b::from).collect();
        let large_bool_check = BoolCheck::<0, 0, AndPackage<0, 0>> {
            points: large_points.clone().into(),
            ..Default::default()
        };
        assert_eq!(large_bool_check.number_variables(), large_points.len());
    }

    #[test]
    fn test_new_andcheck() {
        // Set the number of variables for the test.
        let num_vars = 20;

        // Generate a vector `points` of `num_vars` random field elements in `BinaryField128b`.
        // This represents a set of random variables that will be used in the test.
        let points: Vec<_> = (0..num_vars).map(|_| BinaryField128b::random()).collect();

        // Generate a multilinear polynomial `p` with 2^num_vars random elements in
        // `BinaryField128b`. This represents one operand (a polynomial) in the AND
        // operation.
        let p: MultilinearLagrangianPolynomial =
            (0..1 << num_vars).map(|_| BinaryField128b::random()).collect::<Vec<_>>().into();

        // Generate another multilinear polynomial `q` with 2^num_vars random elements in
        // `BinaryField128b`. This represents the second operand (a polynomial) in the AND
        // operation.
        let q: MultilinearLagrangianPolynomial =
            (0..1 << num_vars).map(|_| BinaryField128b::random()).collect::<Vec<_>>().into();

        // Start a timer to measure the execution time of the test.
        let start = std::time::Instant::now();

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q = p.clone() & q.clone();

        // The prover compute the initial claim for the AND operation at the points in `points`.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Set a phase switch parameter, which controls the folding phases.
        let phase_switch = 5;

        // Generate a random folding challenge `gamma` in `BinaryField128b`.
        let gamma = BinaryField128b::random();

        // Create a new `BoolCheckBuilder` instance with:
        // - the phase switch parameter (c),
        // - the points at which the AND operation is evaluated,
        // - the Boolean package (AND operation for this test).
        let boolcheck_builder = BoolCheckBuilder::new(
            AndPackage,
            phase_switch,
            points.clone().into(),
            &Point(gamma),
            [initial_claim],
            [p, q],
        );

        // Build the Boolean check with the following parameters:
        // - the multilinear polynomials `p` and `q` used in the AND operation,
        // - the initial claim for the AND operation,
        // - the folding challenge `gamma`.
        let mut boolcheck = boolcheck_builder.build();

        // Initialize the current claim as the initial claim.
        // The current claim will be updated during each round of the protocol.
        let mut current_claim = initial_claim;

        // Empty vector to store random values sent by the verifier at each round.
        let mut random_values = Vec::new();

        // The loop iterates over the number of variables to perform the rounds of the protocol.
        for _ in 0..num_vars {
            // Compute the round polynomial for the current round.
            let compressed_round_polynomial = boolcheck.compute_round_polynomial();

            // Generate a random value in `BinaryField128b` and store it in the dedicated vector.
            let r = BinaryField128b::random();
            random_values.push(r);

            // Decompress the round polynomial to obtain the coefficients of the univariate round
            // polynomial.
            let round_polynomial = compressed_round_polynomial.coeffs(current_claim);

            // Update the current claim using the round polynomial and the random value.
            // The round polynomial is a univariate polynomial in `r`:
            // ```
            // P(x) = c_0 + c_1 * x + c_2 * x ^ 2 + c_3 * x ^ 3
            // ```
            //
            // Here we compute the updated claim as:
            // ```
            // current_claim = c_0 + r * c_1 + r ^ 2 * c_2 + r ^ 3 * c_3
            // ```
            current_claim = round_polynomial[0] +
                r * round_polynomial[1] +
                r * r * round_polynomial[2] +
                r * r * r * round_polynomial[3];

            // Bind the random value `r` to the Boolean check for the next round.
            boolcheck.bind(&Point(r));
        }

        // Finish the protocol and obtain the output.
        let BoolCheckOutput { mut frob_evals, .. } = boolcheck.finish();

        // Chunk the Frobenius evaluations and untwist each chunk.
        frob_evals
            .as_mut_slice()
            .chunks_mut(128)
            .for_each(|chunk| Evaluations::from(chunk.to_vec()).untwist());

        // Add a zero element to the end of the evaluations for padding.
        // TODO: hack to be removed in the future
        frob_evals.push(BinaryField128b::zero());

        // Compute algebraic AND
        let and_algebraic = AndPackage::<2, 1>.algebraic(&frob_evals, 0, 1);

        // Transform vector of Field elements to Points
        let points: Points = points.iter().map(|p| Point::from(*p)).collect::<Vec<_>>().into();

        // Transform random values to Points
        let random_values: Points =
            random_values.iter().map(|p| Point::from(*p)).collect::<Vec<_>>().into();

        // Get the expected claim
        let expected_claim = and_algebraic[0][0] * points.eq_eval(&random_values).0;

        // Validate the expected claim against the current claim
        assert_eq!(current_claim, expected_claim);

        // Print the execution time of the test.
        println!("Execution time: {:?} ms", start.elapsed().as_millis());
    }

    #[test]
    #[allow(clippy::unreadable_literal, clippy::too_many_lines)]
    fn test_compute_imaginary_round() {
        // Set the number of variables for the test.
        let num_vars = 3;

        // Generate a vector `points` of `num_vars` field elements in `BinaryField128b`.
        let points: Vec<_> = (0..num_vars).map(BinaryField128b::new).collect();

        // Generate a vector `p` with 2^num_vars elements.
        let p: MultilinearLagrangianPolynomial =
            (0..(1 << num_vars)).map(BinaryField128b::new).collect::<Vec<_>>().into();

        // Generate another vector `q` with 2^num_vars elements.
        let q: MultilinearLagrangianPolynomial =
            (0..(1 << num_vars)).map(BinaryField128b::new).collect::<Vec<_>>().into();

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q = p.clone() & q.clone();

        // The prover compute the initial claim for the AND operation at the points in `points`.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Set a phase switch parameter, which controls the folding phases.
        let phase_switch = 2;

        // Generate a folding challenge `gamma`
        let gamma = BinaryField128b::new(1234);

        // Create a new `BoolCheckBuilder` instance with:
        // - the phase switch parameter (c),
        // - the points at which the AND operation is evaluated,
        // - the Boolean package (AND operation for this test).
        let boolcheck_builder = BoolCheckBuilder::new(
            AndPackage,
            phase_switch,
            points.into(),
            &Point(gamma),
            [initial_claim],
            [p, q],
        );

        // Build the Boolean check with the following parameters:
        // - the multilinear polynomials `p` and `q` used in the AND operation,
        // - the initial claim for the AND operation,
        // - the folding challenge `gamma`.
        let mut boolcheck = boolcheck_builder.build();

        // Validate the initial extended table.
        assert_eq!(
            boolcheck.extended_table,
            vec![
                BinaryField128b::new(0),
                BinaryField128b::new(1),
                BinaryField128b::new(1),
                BinaryField128b::new(2),
                BinaryField128b::new(3),
                BinaryField128b::new(1),
                BinaryField128b::new(2),
                BinaryField128b::new(2),
                BinaryField128b::new(0),
                BinaryField128b::new(4),
                BinaryField128b::new(5),
                BinaryField128b::new(1),
                BinaryField128b::new(6),
                BinaryField128b::new(7),
                BinaryField128b::new(1),
                BinaryField128b::new(2),
                BinaryField128b::new(2),
                BinaryField128b::new(0),
                BinaryField128b::new(4),
                BinaryField128b::new(4),
                BinaryField128b::new(0),
                BinaryField128b::new(4),
                BinaryField128b::new(4),
                BinaryField128b::new(0),
                BinaryField128b::new(0),
                BinaryField128b::new(0),
                BinaryField128b::new(0),
            ]
        );

        // Compute the round polynomial for an imaginary round.
        let compressed_round_polynomial = boolcheck.compute_round_polynomial();

        // Verify the compressed round polynomial.
        assert_eq!(
            compressed_round_polynomial,
            CompressedPoly::new(vec![
                BinaryField128b::new(332514690820570361331092984923254947853),
                BinaryField128b::new(1),
                BinaryField128b::new(1)
            ])
        );

        // Decompress the imaginary round polynomial to obtain the coefficients.
        let round_polynomial = compressed_round_polynomial.coeffs(initial_claim);

        // Verify the decompressed round polynomial.
        assert_eq!(
            round_polynomial,
            UnivariatePolynomial::new(vec![
                BinaryField128b::new(332514690820570361331092984923254947853),
                BinaryField128b::new(332514690820570361331092984923254947853),
                BinaryField128b::new(1),
                BinaryField128b::new(1)
            ])
        );

        // Generate an imaginary random challenge (fixed for testing purposes).
        let r = BinaryField128b::new(5678);

        // Update the current claim using the round polynomial and the random value.
        let current_claim = round_polynomial[0] +
            r * round_polynomial[1] +
            r * r * round_polynomial[2] +
            r * r * r * round_polynomial[3];

        // Verify the correctness of the updated claim.
        assert_eq!(current_claim, BinaryField128b::new(284181495769767592368287233794578256034));

        // Bind the random value `r` to the Boolean check for the next round.
        boolcheck.bind(&Point(r));

        // Validate the updated extended table after the first imaginary round.
        assert_eq!(
            boolcheck.extended_table,
            vec![
                BinaryField128b::new(144961788882111902000582228079393390932),
                BinaryField128b::new(144961788882111902000582228079393390934),
                BinaryField128b::new(2),
                BinaryField128b::new(144961788882111902000582228079393390928),
                BinaryField128b::new(144961788882111902000582228079393390930),
                BinaryField128b::new(2),
                BinaryField128b::new(4),
                BinaryField128b::new(4),
                BinaryField128b::new(0),
            ]
        );

        // Validate the update of the boolcheck claim.
        assert_eq!(boolcheck.claim, BinaryField128b::new(284181495769767592368287233794578256034));

        // Verify that the challenge has been integrated inside the boolcheck.
        assert_eq!(
            boolcheck.challenges,
            Points::from(vec![Point::from(BinaryField128b::new(5678))])
        );

        // Verify the correctness of the round polynomial cache.
        assert_eq!(boolcheck.round_polys, vec![compressed_round_polynomial]);

        // Test an imaginary algorithm execution.
        let alg_res = boolcheck.compute_algebraic(
            &(0..4 * 128).map(BinaryField128b::new).collect::<Vec<_>>(),
            0,
            1,
        );

        // Verify the result of the imaginary algorithm execution.
        assert_eq!(
            alg_res,
            [
                BinaryField128b::new(307232209640963015187300467897455873283),
                BinaryField128b::new(194822172689813899653668252817418647041),
                BinaryField128b::new(67733890487442795093175766325246739159)
            ]
        );

        assert_eq!(
            boolcheck.polys.to_vec(),
            vec![
                MultilinearLagrangianPolynomial::from(vec![
                    BinaryField128b::new(0),
                    BinaryField128b::new(1),
                    BinaryField128b::new(2),
                    BinaryField128b::new(3),
                    BinaryField128b::new(4),
                    BinaryField128b::new(5),
                    BinaryField128b::new(6),
                    BinaryField128b::new(7)
                ]);
                2
            ]
        );
    }

    #[test]
    fn test_compute_round_polynomial_cached_result() {
        // Generate a vector `points` of 2 field elements in `BinaryField128b`.
        let points: Vec<_> = vec![BinaryField128b::new(1), BinaryField128b::new(2)];

        // Generate a vector `p` with 2^2 = 4 elements.
        let p: MultilinearLagrangianPolynomial = vec![
            BinaryField128b::new(11),
            BinaryField128b::new(22),
            BinaryField128b::new(33),
            BinaryField128b::new(44),
        ]
        .into();

        // Generate another vector `q` with 2^2 = 4 elements.
        let q: MultilinearLagrangianPolynomial = vec![
            BinaryField128b::new(111),
            BinaryField128b::new(222),
            BinaryField128b::new(333),
            BinaryField128b::new(444),
        ]
        .into();

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q = p.clone() & q.clone();

        // The prover compute the initial claim for the AND operation at the points in `points`.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Set a phase switch parameter, which controls the folding phases.
        let phase_switch = 1;

        // Generate a folding challenge `gamma`
        let gamma = BinaryField128b::new(1234);

        // Create a new `BoolCheckBuilder` instance with:
        // - the phase switch parameter (c),
        // - the points at which the AND operation is evaluated,
        // - the Boolean package (AND operation for this test).
        let boolcheck_builder = BoolCheckBuilder::new(
            AndPackage,
            phase_switch,
            points.into(),
            &Point(gamma),
            [initial_claim],
            [p, q],
        );

        // Build the Boolean check with the following parameters:
        // - the multilinear polynomials `p` and `q` used in the AND operation,
        // - the initial claim for the AND operation,
        // - the folding challenge `gamma`.
        let mut bool_check = boolcheck_builder.build();

        // Add a precomputed round polynomial to the cache.
        let cached_poly = CompressedPoly::new(vec![BinaryField128b::from(1)]);
        bool_check.round_polys.push(cached_poly.clone());

        // Compute the round polynomial for the cached round.
        let round_poly = bool_check.compute_round_polynomial();

        // Ensure the cached polynomial is returned.
        assert_eq!(round_poly, cached_poly);

        // Ensure no new polynomial is added to the cache.
        assert_eq!(bool_check.round_polys.len(), 1);
    }

    #[test]
    fn test_compute_round_polynomial_initial_round() {
        // Generate a vector `points` of 2 field elements in `BinaryField128b`.
        let points: Vec<_> = vec![BinaryField128b::new(1), BinaryField128b::new(2)];

        // Generate a vector `p` with 2^2 = 4 elements.
        let p: MultilinearLagrangianPolynomial = vec![
            BinaryField128b::new(11),
            BinaryField128b::new(22),
            BinaryField128b::new(33),
            BinaryField128b::new(44),
        ]
        .into();

        // Generate another vector `q` with 2^2 = 4 elements.
        let q: MultilinearLagrangianPolynomial = vec![
            BinaryField128b::new(111),
            BinaryField128b::new(222),
            BinaryField128b::new(333),
            BinaryField128b::new(444),
        ]
        .into();

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q = p.clone() & q.clone();

        // The prover compute the initial claim for the AND operation at the points in `points`.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Set a phase switch parameter, which controls the folding phases.
        let phase_switch = 1;

        // Generate a folding challenge `gamma`.
        let gamma = BinaryField128b::new(1234);

        // Create a new `BoolCheckBuilder` instance.
        let boolcheck_builder = BoolCheckBuilder::new(
            AndPackage,
            phase_switch,
            points.into(),
            &Point(gamma),
            [initial_claim],
            [p, q],
        );

        // Build the Boolean check.
        let mut bool_check = boolcheck_builder.build();

        // Compute the round polynomial for the initial round.
        let round_poly = bool_check.compute_round_polynomial();

        // Ensure the computed polynomial is not empty.
        assert!(!round_poly.coeffs(initial_claim).is_empty());

        // Verify that the claim remains unchanged for the initial round.
        assert_eq!(bool_check.claim, initial_claim);

        // Verify that the round polynomial cache now has one polynomial.
        assert_eq!(bool_check.round_polys.len(), 1);
    }

    #[test]
    #[should_panic(expected = "Protocol has reached the maximum number of rounds.")]
    fn test_compute_round_polynomial_exceeds_round_limit() {
        // Generate a vector `points` of 3 field elements in `BinaryField128b`.
        let points: Vec<_> =
            vec![BinaryField128b::new(1), BinaryField128b::new(2), BinaryField128b::new(3)];

        // Generate a vector `p` with 2^3 = 8 elements.
        let p: MultilinearLagrangianPolynomial =
            (0..8).map(BinaryField128b::new).collect::<Vec<_>>().into();

        // Generate another vector `q` with 2^3 = 8 elements.
        let q: MultilinearLagrangianPolynomial =
            (10..18).map(BinaryField128b::new).collect::<Vec<_>>().into();

        // Compute the element-wise AND operation between `p` and `q`.
        let p_and_q = p.clone() & q.clone();

        // Compute the initial claim for the AND operation.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Set a phase switch parameter.
        let phase_switch = 2;

        // Generate a folding challenge `gamma`.
        let gamma = BinaryField128b::new(1234);

        // Create a new `BoolCheckBuilder` instance.
        let boolcheck_builder = BoolCheckBuilder::new(
            AndPackage,
            phase_switch,
            points.into(),
            &Point(gamma),
            [initial_claim],
            [p, q],
        );

        // Build the Boolean check.
        let mut bool_check = boolcheck_builder.build();

        // Simulate adding challenges equal to the number of variables.
        for i in 0..bool_check.number_variables() {
            bool_check.challenges.push(Point(BinaryField128b::new(i as u128)));
        }

        // Attempt to compute a round polynomial beyond the round limit.
        // This should panic.
        bool_check.compute_round_polynomial();
    }

    #[test]
    #[should_panic(expected = "Claim does not match expected value.")]
    fn test_compute_round_polynomial_invalid_claim() {
        // Generate a vector `points` of 2 field elements in `BinaryField128b`.
        let points: Vec<_> = vec![BinaryField128b::new(1), BinaryField128b::new(2)];

        // Generate a vector `p` with 2^2 = 4 elements.
        let p: MultilinearLagrangianPolynomial = vec![
            BinaryField128b::new(11),
            BinaryField128b::new(22),
            BinaryField128b::new(33),
            BinaryField128b::new(44),
        ]
        .into();

        // Generate another vector `q` with 2^2 = 4 elements.
        let q: MultilinearLagrangianPolynomial = vec![
            BinaryField128b::new(111),
            BinaryField128b::new(222),
            BinaryField128b::new(333),
            BinaryField128b::new(444),
        ]
        .into();

        // Compute an incorrect initial claim for testing purposes.
        let incorrect_claim = BinaryField128b::new(999);

        // Set a phase switch parameter.
        let phase_switch = 1;

        // Generate a folding challenge `gamma`.
        let gamma = BinaryField128b::new(1234);

        // Create a new `BoolCheckBuilder` instance.
        let boolcheck_builder = BoolCheckBuilder::new(
            AndPackage,
            phase_switch,
            points.into(),
            &Point(gamma),
            [incorrect_claim],
            [p, q],
        );

        // Build the Boolean check.
        let mut bool_check = boolcheck_builder.build();

        // Attempt to compute a round polynomial with the incorrect claim.
        // This should panic as the claim does not match the computed value.
        bool_check.compute_round_polynomial();
    }

    #[test]
    fn test_compute_round_polynomial_correct_claim_update() {
        // Generate a vector `points` of 2 field elements in `BinaryField128b`.
        let points: Vec<_> = vec![BinaryField128b::new(1), BinaryField128b::new(2)];

        // Generate a vector `p` with 2^2 = 4 elements.
        let p: MultilinearLagrangianPolynomial = vec![
            BinaryField128b::new(11),
            BinaryField128b::new(22),
            BinaryField128b::new(33),
            BinaryField128b::new(44),
        ]
        .into();

        // Generate another vector `q` with 2^2 = 4 elements.
        let q: MultilinearLagrangianPolynomial = vec![
            BinaryField128b::new(111),
            BinaryField128b::new(222),
            BinaryField128b::new(333),
            BinaryField128b::new(444),
        ]
        .into();

        // Compute the element-wise AND operation between `p` and `q`.
        let p_and_q = p.clone() & q.clone();

        // Compute the initial claim for the AND operation.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Set a phase switch parameter.
        let phase_switch = 1;

        // Generate a folding challenge `gamma`.
        let gamma = BinaryField128b::new(1234);

        // Create a new `BoolCheckBuilder` instance.
        let boolcheck_builder = BoolCheckBuilder::new(
            AndPackage,
            phase_switch,
            points.into(),
            &Point(gamma),
            [initial_claim],
            [p, q],
        );

        // Build the Boolean check.
        let mut bool_check = boolcheck_builder.build();

        // Compute the round polynomial.
        let round_poly = bool_check.compute_round_polynomial();

        // Decompress the round polynomial and retrieve its coefficients.
        let coeffs = round_poly.coeffs(initial_claim);

        // Generate a random challenge value.
        let challenge = Point(BinaryField128b::new(3));

        // Compute the round polynomial.
        bool_check.bind(&challenge);

        // Ensure the polynomial has 3 coefficients (degree 2 polynomial).
        assert_eq!(coeffs.len(), 4);

        // Compute the updated claim using the polynomial coefficients.
        let updated_claim = coeffs[0] +
            coeffs[1] * *challenge +
            coeffs[2] * *challenge * *challenge +
            coeffs[3] * *challenge * *challenge * *challenge;

        // Verify that the updated claim matches the computed result.
        assert_eq!(updated_claim, bool_check.claim);
    }

    #[test]
    fn test_current_round_initial_state() {
        // Create a default BoolCheck instance with no challenges added.
        let bool_check: BoolCheck<0, 0, AndPackage<0, 0>> = BoolCheck::default();

        // Assert that the initial round is 0 because no challenges have been added yet.
        assert_eq!(bool_check.current_round(), 0);
    }

    #[test]
    fn test_current_round_after_single_challenge() {
        // Create a default BoolCheck instance.
        let mut bool_check: BoolCheck<0, 0, AndPackage<0, 0>> = BoolCheck::default();

        // Add one challenge to the challenges vector.
        bool_check.challenges.push(Point::from(BinaryField128b::from(10)));

        // Assert that the current round is now 1.
        assert_eq!(bool_check.current_round(), 1);
    }

    #[test]
    fn test_current_round_after_multiple_challenges() {
        // Create a default BoolCheck instance.
        let mut bool_check: BoolCheck<0, 0, AndPackage<0, 0>> = BoolCheck::default();

        // Add multiple challenges to the challenges vector.
        bool_check.challenges.extend(vec![
            Point::from(BinaryField128b::from(10)),
            Point::from(BinaryField128b::from(20)),
            Point::from(BinaryField128b::from(30)),
        ]);

        // Assert that the current round reflects the number of challenges added (3).
        assert_eq!(bool_check.current_round(), 3);
    }

    #[test]
    fn test_current_round_after_removing_challenges() {
        // Create a default BoolCheck instance.
        let mut bool_check: BoolCheck<0, 0, AndPackage<0, 0>> = BoolCheck::default();

        // Add multiple challenges to the challenges vector.
        bool_check.challenges.extend(vec![
            Point::from(BinaryField128b::from(10)),
            Point::from(BinaryField128b::from(20)),
            Point::from(BinaryField128b::from(30)),
        ]);

        // Remove one challenge (simulating a removal operation, if valid in the context).
        bool_check.challenges.pop();

        // Assert that the current round reflects the updated number of challenges (2).
        assert_eq!(bool_check.current_round(), 2);
    }

    #[test]
    fn test_current_round_with_no_challenges_after_clear() {
        // Create a default BoolCheck instance.
        let mut bool_check: BoolCheck<0, 0, AndPackage<0, 0>> = BoolCheck::default();

        // Add multiple challenges to the challenges vector.
        bool_check.challenges.extend(vec![
            Point::from(BinaryField128b::from(10)),
            Point::from(BinaryField128b::from(20)),
        ]);

        // Clear all challenges.
        bool_check.challenges.clear();

        // Assert that the current round is back to 0 after clearing challenges.
        assert_eq!(bool_check.current_round(), 0);
    }

    #[test]
    fn test_number_variables_initial_state() {
        // Create a BoolCheck instance with no points defined (default state).
        let bool_check: BoolCheck<0, 0, AndPackage<0, 0>> = BoolCheck::default();

        // Assert that the number of variables is 0 as no points are defined.
        assert_eq!(bool_check.number_variables(), 0);
    }

    #[test]
    fn test_number_variables_with_single_point() {
        // Create a BoolCheck instance with a single point.
        let bool_check = BoolCheck::<0, 0, AndPackage<0, 0>> {
            points: Points::from(vec![BinaryField128b::from(1)]),
            ..Default::default()
        };

        // Assert that the number of variables is 1, corresponding to the single point.
        assert_eq!(bool_check.number_variables(), 1);
    }

    #[test]
    fn test_number_variables_with_multiple_points() {
        // Create a BoolCheck instance with multiple points.
        let points =
            vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];
        let bool_check = BoolCheck::<0, 0, AndPackage<0, 0>> {
            points: Points::from(points.clone()),
            ..Default::default()
        };

        // Assert that the number of variables matches the number of points (3).
        assert_eq!(bool_check.number_variables(), points.len());
    }

    #[test]
    fn test_number_variables_with_empty_points() {
        // Create a BoolCheck instance with an empty points vector.
        let bool_check =
            BoolCheck::<0, 0, AndPackage<0, 0>> { points: Points::default(), ..Default::default() };

        // Assert that the number of variables is 0 when no points are defined.
        assert_eq!(bool_check.number_variables(), 0);
    }

    #[test]
    fn test_number_variables_with_large_number_of_points() {
        // Generate a large number of points (e.g., 1000 points).
        let points: Vec<BinaryField128b> = (0..1000).map(BinaryField128b::from).collect();

        // Create a BoolCheck instance with these points.
        let bool_check = BoolCheck::<0, 0, AndPackage<0, 0>> {
            points: Points::from(points.clone()),
            ..Default::default()
        };

        // Assert that the number of variables matches the number of points (1000).
        assert_eq!(bool_check.number_variables(), points.len());
    }

    #[test]
    fn test_bind_single() {
        // Create a realistic setup with points and polynomials.
        let points = vec![BinaryField128b::from(1), BinaryField128b::from(2)];
        let p: MultilinearLagrangianPolynomial = vec![
            BinaryField128b::from(11),
            BinaryField128b::from(22),
            BinaryField128b::from(33),
            BinaryField128b::from(44),
        ]
        .into();
        let q: MultilinearLagrangianPolynomial = vec![
            BinaryField128b::from(111),
            BinaryField128b::from(222),
            BinaryField128b::from(333),
            BinaryField128b::from(444),
        ]
        .into();

        // Compute the AND operation between the polynomials and the initial claim.
        let p_and_q = p.clone() & q.clone();
        let initial_claim = p_and_q.evaluate_at(&Points::from(points.clone()));

        // Set a phase switch parameter.
        let phase_switch = 1;

        // Generate a folding challenge `gamma`.
        let gamma = BinaryField128b::new(1234);

        // Create a new `BoolCheckBuilder` instance.
        let boolcheck_builder = BoolCheckBuilder::new(
            AndPackage,
            phase_switch,
            points.into(),
            &Point(gamma),
            [initial_claim],
            [p, q],
        );

        // Build the Boolean check.
        let mut bool_check = boolcheck_builder.build();

        // Bind a random challenge `r` to the BoolCheck instance.
        let r = Point::from(BinaryField128b::from(42));
        bool_check.bind(&r);

        // Verify that the challenge has been added.
        assert_eq!(bool_check.challenges.len(), 1);
        assert_eq!(bool_check.challenges[0], r);

        // Verify that the extended table is updated after binding.
        assert!(!bool_check.extended_table.is_empty());
    }
}
