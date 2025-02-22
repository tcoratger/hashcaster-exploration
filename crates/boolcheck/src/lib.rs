#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]

use std::array;

use algebraic::AlgebraicOps;
use hashcaster_primitives::{
    binary_field::BinaryField128b,
    poly::{
        compressed::CompressedPoly,
        evaluation::{Evaluations, FixedEvaluations},
        multinear_lagrangian::{restrict, MultilinearLagrangianPolynomial},
        point::Points,
        univariate::FixedUnivariatePolynomial,
    },
    sumcheck::{EvaluationProvider, Sumcheck},
};
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

pub mod algebraic;
pub mod and;
pub mod bool_trait;
pub mod builder;

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
pub struct BoolCheck<'a, const N: usize, const M: usize, const C: usize, A: AlgebraicOps<N, M>> {
    /// The evaluation points for the multilinear polynomials.
    pub points: &'a Points,

    /// An array of multilinear polynomials used in the protocol.
    pub polys: &'a [MultilinearLagrangianPolynomial; N],

    /// A vector storing intermediate evaluations of the polynomials.
    pub extended_table: Vec<BinaryField128b>,

    /// An optional field storing evaluations on a restricted subset of the hypercube.
    poly_coords: Evaluations,

    /// A sequence of random challenges provided by the verifier.
    pub challenges: Points,

    /// A vector of bit mappings for optimized indexing of polynomial coefficients.
    pub bit_mapping: Vec<u16>,

    /// A sequence of equality polynomials used to verify claims.
    pub eq_sequence: Vec<MultilinearLagrangianPolynomial>,

    /// A vector of compressed polynomials computed at each round.
    pub round_polys: Vec<CompressedPoly<3>>,

    /// The current claim being verified in the protocol.
    pub claim: BinaryField128b,

    /// An array of folding challenges used for polynomial compression.
    pub gammas: [BinaryField128b; M],

    /// Abstract algebraic operations.
    pub algebraic_operations: &'a A,
}

impl<const N: usize, const M: usize, const C: usize, A> Sumcheck<{ 128 * N }, 3>
    for BoolCheck<'_, N, M, C, A>
where
    A: AlgebraicOps<N, M> + Send + Sync,
    [(); 128 * N]:,
{
    type Output = BoolCheckOutput<{ 128 * N }, 3>;

    fn round_polynomial(&mut self) -> CompressedPoly<3> {
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
            return CompressedPoly(array::from_fn(|_| BinaryField128b::ZERO));
        }

        // The protocol is defined in two phases:
        // - The first phase last `c` rounds. It operates over the quadratic polynomial `P Ʌ Q` on
        //   the subset `(0,1,∞)^{c+1} x (0,1)^{n-c-1}`
        // - After `c` phases, the prover no longer has the required data (as we have computed the
        //   extension table only on `3^{c+1} \times 2^{n-c-1}` subset) and cannot proceed further.
        //   This is when we change strategy and compute `P_i(r_0, \ldots, r_c, \mathbf{x}_{>c})`
        //   and `Q_i(r_0, \ldots, r_c, \mathbf{x}_{>c})` (using RESTRICT functionality).
        let poly_deg2 = if round <= C {
            // Fetch the equality polynomial for the current round.
            //
            // The equality polynomials for all the points are stored in cache during the setup
            // phase.
            let eq_poly_round = &self.eq_sequence[number_variables - round - 1];

            // The first phase of the protocol lasts `c` rounds.
            //
            // This is the number of remaining rounds in the first phase.
            let phase1_dimensions = C - round;

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
            (0..(1 << (number_variables - C - 1)))
                .into_par_iter()
                .map(|i| {
                    // Base index is: `ì * 2^{number of remaining rounds in the first phase}`.
                    let base_index = i << phase1_dimensions;

                    // Base offset is: `i * 3^{remaining rounds in the first phase + 1}`.
                    let base_offset = 3 * (i * pow3);

                    // Intermediate polynomial for the current round.
                    //
                    // This polynomial is a degree-2 polynomial.
                    let mut pd2_part = [BinaryField128b::ZERO; 3];

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
                    || [BinaryField128b::ZERO; 3],
                    |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                )
        } else {
            // During this second phase, after `c` rounds, the prover no longer has the required
            // data because the extension table was computed only on the `3^{c+1} x 2^{n-c-1}`
            // subset.

            // Fetch the equality polynomial for the current round.
            let eq_poly_round = &self.eq_sequence[self.points.len() - round - 1];

            // Fetch the number of coefficients in the equality polynomial (Lagrangian polynomial).
            let half = eq_poly_round.len();

            // At the current round, the equality polynomial should have `2^{n-round-1}`
            // coefficients.
            assert_eq!(
                half,
                1 << (number_variables - round - 1),
                "Invalid equality polynomial size"
            );

            // Here we want to compute (in the restricted phase):
            //
            // ```
            //  W(t) = Σ_{x_{>i}} (P ∧ Q)(r_{<i}, t, x_{>i}) * eq(x_{>i}; q_{>i}).
            // ```
            (0..half)
                .into_par_iter()
                .map(|i| {
                    let mut acc = [BinaryField128b::ZERO; 3];
                    acc.iter_mut()
                        .zip(self.compute_algebraic(i, 1 << (number_variables - C - 1)))
                        .for_each(|(accum, value)| *accum += value * eq_poly_round[i]);
                    acc
                })
                .reduce(
                    || [BinaryField128b::ZERO; 3],
                    |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
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
            FixedUnivariatePolynomial::from_evaluations_deg2(poly_deg2) * eq_y_multiplier;

        // We then need to compute the equality polynomial `eq(t; q)`.
        //
        // This is the equality polynomial for the current round (degree-1 univariate
        // polynomial).
        let eq_t = FixedUnivariatePolynomial::new([
            self.points[round] + BinaryField128b::ONE,
            BinaryField128b::ONE,
        ]);

        // In the final step, we multiply the univariate polynomial by the equality polynomial:
        // ```
        // U(t) = eq(t; q) V(t)
        // ```
        let u = univariate_poly_deg2.multiply_degree2_by_degree1(&eq_t);

        // Compress the polynomial to obtain:
        // - The compressed round polynomial.
        // - The computed claim.
        let (ret, computed_claim) = CompressedPoly::compress(&u.coeffs);

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

    fn bind(&mut self, r: &BinaryField128b) {
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
        let round_poly = self.round_polynomial().coeffs(self.claim);
        // Update the current claim:
        // - We evaluate the round polynomial at the random value `r` provided by the verifier.
        self.claim = round_poly.evaluate_at(r);

        // Add the random value sent by the verifier to the list of gammas.
        self.challenges.push(*r);

        if round <= C {
            // Compute `r^2` for future usage in the protocol.
            let r2 = *r * r;

            // We compute, by chunk of 3, the new values for the extended table:
            // ```
            // P(0, r) = P(0, r) + (P(0, r) + P(1, r) + P(∞, r)) * r + P(∞, r) * r^2
            // ```
            self.extended_table.par_chunks_mut(3).for_each(|chunk| {
                chunk[0] = chunk[0] + (chunk[0] + chunk[1] + chunk[2]) * r + chunk[2] * r2;
            });

            // We want to avoid a new allocation:
            // - We use a linear passing of the data
            // - We truncate the vector to the new size
            let new_len = self.extended_table.len() / 3;
            for i in 0..new_len {
                self.extended_table[i] = self.extended_table[i * 3];
            }

            self.extended_table.truncate(new_len);
        } else {
            // At each round we halve the number of variables in the hypercube.
            //
            // `half` represents `2^{n-round-1} = 2^{n-round} / 2`.
            let half = 1 << (number_variables - round - 1);

            // Compute the new values for the poly coordinates:
            self.poly_coords.par_chunks_mut(1 << (number_variables - C - 1)).for_each(|chunk| {
                for j in 0..half {
                    chunk[j] = chunk[2 * j] + (chunk[2 * j + 1] + chunk[2 * j]) * r;
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
        if self.current_round() == C + 1 {
            // At the end of the first phase, we do not need the extended table anymore.
            self.extended_table.clear();

            // Restrict the polynomial coordinates based on the accumulated challenges.
            self.poly_coords = restrict(self.polys, &self.challenges, number_variables);
        }
    }

    fn finish(self) -> Self::Output {
        // Compute the number of variables in the polynomial.
        let number_variables = self.number_variables();
        // The number of rounds should match the number of variables for the protocol to end.
        assert_eq!(number_variables, self.current_round(), "Protocol has not reached the end.");

        // Decompose the `BoolCheck` instance to extract the required fields.
        let Self { poly_coords, round_polys, .. } = self;

        // Compute the evaluations on the Frobenius subdomain.
        let base_index = 1 << (number_variables - C - 1);
        let frob_evals_array = std::array::from_fn(|idx| {
            // Extract poly index and Frobenius index efficiently
            let (poly_idx, frob_idx) = (idx >> 7, idx & 127);
            // Compute the correct evaluation index
            poly_coords[(poly_idx * 128 + frob_idx) * base_index]
        });

        // Wrap it in `FixedEvaluations`
        let mut frob_evals = FixedEvaluations::new(frob_evals_array);

        // Apply the twist operation to each chunk of 128 evaluations.
        frob_evals.twist();

        // Return the `BoolCheckOutput`
        Self::Output { frob_evals, round_polys }
    }
}

impl<const N: usize, const M: usize, const C: usize, A> BoolCheck<'_, N, M, C, A>
where
    A: AlgebraicOps<N, M> + Send + Sync,
{
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

    /// Computes the algebraic operations for the given data slice using the provided
    /// algebraic operations trait and applies folding challenges.
    ///
    /// # Parameters
    /// - `data`: A slice of `BinaryField128b` elements representing the data to be used in the
    ///   computation.
    /// - `idx_a`: The index of the algebraic operation.
    /// - `offset`: The offset for the folding operation.
    ///
    /// # Returns
    /// A 3-element array of `BinaryField128b` containing the computed algebraic values after
    /// applying folding challenges to the input data.
    ///
    /// # Description
    /// This function applies algebraic operations on the provided data slice. The resulting values
    /// are then folded by applying a set of predefined folding challenges (`gammas`) to produce
    /// a compressed result, which is returned as a 3-element array.
    pub fn compute_algebraic(&self, idx_a: usize, offset: usize) -> [BinaryField128b; 3] {
        // Perform algebraic operations and fold results using gammas.
        self.algebraic_operations.algebraic(&self.poly_coords, idx_a, offset).map(|result| {
            result
                .iter()
                .zip(&self.gammas)
                .fold(BinaryField128b::ZERO, |acc, (&val, &gamma)| acc + val * gamma)
        })
    }
}

#[derive(Clone, Debug)]
pub struct BoolCheckOutput<const N: usize, const M: usize> {
    /// Evaluations of the polynomials on a Frobenius subdomain.
    pub frob_evals: FixedEvaluations<N>,

    /// A vector of compressed polynomials computed during the protocol's rounds.
    pub round_polys: Vec<CompressedPoly<M>>,
}

impl<const N: usize, const M: usize> EvaluationProvider<N> for BoolCheckOutput<N, M> {
    fn evals(self) -> FixedEvaluations<N> {
        self.frob_evals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use and::AndPackage;
    use builder::BoolCheckBuilder;
    use hashcaster_multiclaim::builder::MulticlaimBuilder;
    use hashcaster_primitives::{
        poly::{
            multinear_lagrangian::MultilinearLagrangianPolynomial,
            univariate::FixedUnivariatePolynomial,
        },
        sumcheck::SumcheckBuilder,
    };
    use num_traits::Pow;
    use rand::rngs::OsRng;
    use std::array;

    #[test]
    fn test_current_rounds() {
        // Define a sample BoolCheck instance.
        // No challenges have been submitted yet.
        let points = Points::default();
        let mut bool_check = BoolCheck::<0, 0, 0, AndPackage<0, 0>> {
            points: &points,
            polys: &Default::default(),
            extended_table: Default::default(),
            poly_coords: Default::default(),
            challenges: Default::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: Default::default(),
            algebraic_operations: &Default::default(),
        };

        // Assert the initial round (no challenges yet).
        assert_eq!(bool_check.current_round(), 0);

        // Add a challenge and test the round count.
        bool_check.challenges.push(BinaryField128b::from(10));
        assert_eq!(bool_check.current_round(), 1);

        // Add more challenges and test the updated round count.
        bool_check.challenges.extend(vec![
            BinaryField128b::from(20),
            BinaryField128b::from(30),
            BinaryField128b::from(40),
        ]);
        assert_eq!(bool_check.current_round(), 4);

        // Add another challenge and verify the round count again.
        bool_check.challenges.push(BinaryField128b::from(50));
        assert_eq!(bool_check.current_round(), 5);
    }

    #[test]
    fn test_number_variables() {
        // Create a BoolCheck instance with a defined number of points (variables).
        let points =
            vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];
        let bool_check = BoolCheck::<0, 0, 0, AndPackage<0, 0>> {
            points: &points.clone().into(),
            polys: &Default::default(),
            extended_table: Default::default(),
            poly_coords: Default::default(),
            challenges: Default::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: Default::default(),
            algebraic_operations: &Default::default(),
        };

        // Assert that the number of variables matches the length of the points vector.
        assert_eq!(bool_check.number_variables(), points.len());

        // Test with no points (empty vector).
        let points = Points::default();
        let empty_bool_check = BoolCheck::<0, 0, 0, AndPackage<0, 0>> {
            points: &points,
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };
        assert_eq!(empty_bool_check.number_variables(), 0);

        // Test with a large number of points.
        let large_points: Vec<_> = (0..1000).map(BinaryField128b::from).collect();
        let large_bool_check = BoolCheck::<0, 0, 0, AndPackage<0, 0>> {
            points: &large_points.clone().into(),
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };
        assert_eq!(large_bool_check.number_variables(), large_points.len());
    }

    #[test]
    fn test_new_andcheck1() {
        // Set a phase switch parameter, which controls the folding phases.
        const PHASE_SWITCH: usize = 5;

        // Set the number of variables for the test.
        let num_vars = 20;

        let rng = &mut OsRng;

        // Generate a vector `points` of `num_vars` random field elements in `BinaryField128b`.
        // This represents a set of random variables that will be used in the test.
        let points = Points::random(num_vars, rng);

        // Generate a multilinear polynomial `p` with 2^num_vars random elements in
        // `BinaryField128b`. This represents one operand (a polynomial) in the AND
        // operation.
        let p = MultilinearLagrangianPolynomial::random(1 << num_vars, rng);

        // Generate another multilinear polynomial `q` with 2^num_vars random elements in
        // `BinaryField128b`. This represents the second operand (a polynomial) in the AND
        // operation.
        let q = MultilinearLagrangianPolynomial::random(1 << num_vars, rng);

        // Start a timer to measure the execution time of the test.
        let start = std::time::Instant::now();

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q = p.clone() & q.clone();

        // The prover compute the initial claim for the AND operation at the points in `points`.
        let initial_claim = p_and_q.evaluate_at(&points);

        // Generate a random folding challenge `gamma` in `BinaryField128b`.
        let gamma = BinaryField128b::random(rng);

        // Create a new `BoolCheckBuilder` instance with:
        // - the phase switch parameter (c),
        // - the points at which the AND operation is evaluated,
        // - the Boolean package (AND operation for this test).
        let polys = [p, q];
        let boolcheck_builder = BoolCheckBuilder::<_, _, PHASE_SWITCH, _>::new(
            &AndPackage,
            &points,
            [initial_claim],
            &polys,
        );

        // Build the Boolean check with the following parameters:
        // - the multilinear polynomials `p` and `q` used in the AND operation,
        // - the initial claim for the AND operation,
        // - the folding challenge `gamma`.
        let mut boolcheck = boolcheck_builder.build(&gamma);

        // Initialize the current claim as the initial claim.
        // The current claim will be updated during each round of the protocol.
        let mut current_claim = initial_claim;

        // Empty vector to store random values sent by the verifier at each round.
        let mut challenges = Points::default();

        let mut time_in_bind = 0;
        let mut time_in_round_polynomial = 0;

        // The loop iterates over the number of variables to perform the rounds of the protocol.
        for _ in 0..num_vars {
            let start = std::time::Instant::now();

            // Compute the round polynomial for the current round.
            let compressed_round_polynomial = boolcheck.round_polynomial();

            time_in_round_polynomial += start.elapsed().as_millis();

            // Generate a random value in `BinaryField128b` and store it in the dedicated vector.
            let r = BinaryField128b::random(rng);

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
            current_claim = round_polynomial.evaluate_at(&r);

            let start = std::time::Instant::now();

            // Bind the random value `r` to the Boolean check for the next round.
            boolcheck.bind(&r);

            let end = std::time::Instant::now();
            time_in_bind += end.duration_since(start).as_millis();

            // Store the random value in the challenges vector.
            challenges.push(r);
        }

        // Finish the protocol and obtain the output.
        let BoolCheckOutput { mut frob_evals, .. } = boolcheck.finish();

        // Untwist the Frobenius evaluations to obtain the expected claim.
        frob_evals.untwist();

        // Compute algebraic AND
        let and_algebraic = AndPackage::<2, 1>.algebraic(frob_evals.as_ref(), 0, 1);

        // Get the expected claim
        let expected_claim = and_algebraic[0][0] * points.eq_eval(&challenges);

        // Validate the expected claim against the current claim
        assert_eq!(current_claim, expected_claim);

        // Print the execution time of the test.
        println!("Execution time: {:?} ms", start.elapsed().as_millis());

        println!("Time in bind: {time_in_bind:?} ms");
        println!("Time in round polynomial: {time_in_round_polynomial:?} ms");
    }

    #[test]
    #[allow(clippy::cast_sign_loss)]
    fn test_new_andcheck_with_multiclaim() {
        // Set a phase switch parameter, which controls the folding phases.
        const PHASE_SWITCH: usize = 5;

        // Set the number of variables for the test.
        let num_vars = 20;

        let rng = &mut OsRng;

        // Generate a vector `points` of `num_vars` random field elements in `BinaryField128b`.
        // This represents a set of random variables that will be used in the test.
        let points = Points::random(num_vars, rng);

        // Generate a multilinear polynomial `p` with 2^num_vars random elements in
        // `BinaryField128b`. This represents one operand (a polynomial) in the AND
        // operation.
        let p = MultilinearLagrangianPolynomial::random(1 << num_vars, rng);

        // Generate another multilinear polynomial `q` with 2^num_vars random elements in
        // `BinaryField128b`. This represents the second operand (a polynomial) in the AND
        // operation.
        let q = MultilinearLagrangianPolynomial::random(1 << num_vars, rng);

        // Start a timer to measure the execution time of the test.
        let start = std::time::Instant::now();

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q = p.clone() & q.clone();

        // The prover compute the initial claim for the AND operation at the points in `points`.
        let initial_claim = p_and_q.evaluate_at(&points);

        // Generate a random folding challenge `gamma` in `BinaryField128b`.
        let gamma = BinaryField128b::random(rng);

        // Create a new `BoolCheckBuilder` instance with:
        // - the phase switch parameter (c),
        // - the points at which the AND operation is evaluated,
        // - the Boolean package (AND operation for this test).
        let polys = [p.clone(), q.clone()];
        let boolcheck_builder = BoolCheckBuilder::<_, _, PHASE_SWITCH, _>::new(
            &AndPackage,
            &points,
            [initial_claim],
            &polys,
        );

        // Build the Boolean check with the following parameters:
        // - the multilinear polynomials `p` and `q` used in the AND operation,
        // - the initial claim for the AND operation,
        // - the folding challenge `gamma`.
        let mut boolcheck = boolcheck_builder.build(&gamma);

        // Initialize the current claim as the initial claim.
        // The current claim will be updated during each round of the protocol.
        let mut current_claim = initial_claim;

        // Setup an empty vector to store the challanges in the main loop
        let mut challenges = Points::from(Vec::<BinaryField128b>::with_capacity(num_vars));

        // The loop iterates over the number of variables to perform the rounds of the protocol.
        for _ in 0..num_vars {
            // Compute the round polynomial for the current round.
            let compressed_round_polynomial = boolcheck.round_polynomial();

            // Generate a random value in `BinaryField128b` and store it in the dedicated vector.
            let r = BinaryField128b::random(rng);

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
            current_claim = round_polynomial.evaluate_at(&r);

            // Bind the random value `r` to the Boolean check for the next round.
            boolcheck.bind(&r);

            // Store the random value in the challenges vector.
            challenges.push(r);
        }

        // Finish the protocol and obtain the output.
        let BoolCheckOutput { frob_evals, .. } = boolcheck.finish();

        // Clone the Frobenius evaluations to untwist them (we will need the original evaluations
        // untouched for the multiclaim part of the test).
        let mut untwisted_evals = frob_evals.clone();

        assert_eq!(frob_evals.len(), 256, "Invalid number of Frobenius evaluations.");

        // Untwist the Frobenius evaluations to obtain the expected claim.
        untwisted_evals.untwist();

        // Compute algebraic AND
        let and_algebraic = AndPackage::<2, 1>.algebraic(untwisted_evals.as_ref(), 0, 1);

        // Get the expected claim
        let expected_claim = and_algebraic[0][0] * points.eq_eval(&challenges);

        // Validate the expected claim against the current claim
        assert_eq!(current_claim, expected_claim);

        // Print the execution time of the boolcheck part of the test.
        println!("Execution time of Boolcheck: {:?} ms", start.elapsed().as_millis());

        // New start for the multiclaim part of the test.
        let start = std::time::Instant::now();

        // Setup the new points to be the challenges of the boolcheck.
        let points = challenges;

        // Map the points to the inverse Frobenius orbit using a fixed-size array
        #[allow(clippy::cast_possible_wrap)]
        let points_inv_orbit: [Points; 128] =
            std::array::from_fn(|i| points.iter().map(|x| x.frobenius(-(i as i32))).collect());

        // Generate a random gamma for folding
        let gamma = BinaryField128b::random(rng);

        // Generate `gamma^128` for final evaluation
        let gamma128 = gamma.pow(128);

        // Setup a multiclaim builder
        let polys_multiclaim = [p.clone(), q.clone()];
        let multiclaim_builder = MulticlaimBuilder::new(&polys_multiclaim, &points, &frob_evals);

        // Builder the multiclaim prover via folding
        let mut multiclaim_prover = multiclaim_builder.build(&gamma);

        // Compute the claim
        let mut claim = FixedUnivariatePolynomial::new(frob_evals.0).evaluate_at(&gamma);

        // Setup an empty vector to store the challanges in the main loop
        let mut challenges = Points::from(Vec::<BinaryField128b>::with_capacity(num_vars));

        // Empty vector to store the challenges
        for _ in 0..num_vars {
            // Compute the round polynomial
            let round_polynomial = multiclaim_prover.round_polynomial().coeffs(claim);

            // Check that the round polynomial is of degree 2
            assert_eq!(round_polynomial.len(), 3, "Round polynomial should have degree 2.");

            // Random challenge
            let challenge = BinaryField128b::random(rng);

            // Update the claim with the round polynomial and the challenge
            claim = round_polynomial.evaluate_at(&challenge);

            // Push the challenge to the vector
            challenges.push(challenge);

            // Bind the prover to the challenge
            multiclaim_prover.bind(&challenge);
        }

        // Finish the protocol
        let multiclaim_output = multiclaim_prover.finish();

        // Compute the equality evaluations at the challenges into a fixed-size array
        let eq_evaluations: FixedUnivariatePolynomial<128> = FixedUnivariatePolynomial {
            coeffs: std::array::from_fn(|i| points_inv_orbit[i].eq_eval(&challenges)),
        };

        // Compute the equality evaluation at gamma
        let eq_evaluation = eq_evaluations.evaluate_at(&gamma);

        // Validate the claim
        assert_eq!(
            FixedUnivariatePolynomial::new(multiclaim_output.0).evaluate_at(&gamma128) *
                eq_evaluation,
            claim
        );

        // More validations about P and Q evaluations at the challenges
        assert_eq!(p.evaluate_at(&challenges), multiclaim_output[0]);
        assert_eq!(q.evaluate_at(&challenges), multiclaim_output[1]);

        // Print the execution time of the multiclaim part of the test.
        println!("Execution time of Multiclaim: {:?} ms", start.elapsed().as_millis());
    }

    #[test]
    #[allow(clippy::unreadable_literal, clippy::too_many_lines)]
    fn test_compute_imaginary_round() {
        // Set a phase switch parameter, which controls the folding phases.
        const PHASE_SWITCH: usize = 2;

        // Set the number of variables for the test.
        let num_vars = 3;

        // Generate a vector `points` of `num_vars` field elements in `BinaryField128b`.
        let points: Vec<_> = (0..num_vars).map(BinaryField128b::new).collect();

        // Generate a vector `p` with 2^num_vars elements.
        let p: MultilinearLagrangianPolynomial =
            (0..(1 << num_vars)).map(BinaryField128b::new).collect();

        // Generate another vector `q` with 2^num_vars elements.
        let q: MultilinearLagrangianPolynomial =
            (0..(1 << num_vars)).map(BinaryField128b::new).collect();

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q = p.clone() & q.clone();

        // The prover compute the initial claim for the AND operation at the points in `points`.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Generate a folding challenge `gamma`
        let gamma = BinaryField128b::from(1234);

        // Create a new `BoolCheckBuilder` instance with:
        // - the phase switch parameter (c),
        // - the points at which the AND operation is evaluated,
        // - the Boolean package (AND operation for this test).
        let pts = points.into();
        let polys = [p, q];
        let boolcheck_builder = BoolCheckBuilder::<_, _, PHASE_SWITCH, _>::new(
            &AndPackage,
            &pts,
            [initial_claim],
            &polys,
        );

        // Build the Boolean check with the following parameters:
        // - the multilinear polynomials `p` and `q` used in the AND operation,
        // - the initial claim for the AND operation,
        // - the folding challenge `gamma`.
        let mut boolcheck = boolcheck_builder.build(&gamma);

        // Validate the initial extended table.
        assert_eq!(
            boolcheck.extended_table,
            vec![
                BinaryField128b::ZERO,
                BinaryField128b::new(1),
                BinaryField128b::new(1),
                BinaryField128b::new(2),
                BinaryField128b::new(3),
                BinaryField128b::new(1),
                BinaryField128b::new(2),
                BinaryField128b::new(2),
                BinaryField128b::ZERO,
                BinaryField128b::new(4),
                BinaryField128b::new(5),
                BinaryField128b::new(1),
                BinaryField128b::new(6),
                BinaryField128b::new(7),
                BinaryField128b::new(1),
                BinaryField128b::new(2),
                BinaryField128b::new(2),
                BinaryField128b::ZERO,
                BinaryField128b::new(4),
                BinaryField128b::new(4),
                BinaryField128b::ZERO,
                BinaryField128b::new(4),
                BinaryField128b::new(4),
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
                BinaryField128b::ZERO,
            ]
        );

        // Compute the round polynomial for an imaginary round.
        let compressed_round_polynomial = boolcheck.round_polynomial();

        // Verify the compressed round polynomial.
        assert_eq!(
            compressed_round_polynomial,
            CompressedPoly([
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
            FixedUnivariatePolynomial::new([
                BinaryField128b::new(332514690820570361331092984923254947853),
                BinaryField128b::new(332514690820570361331092984923254947853),
                BinaryField128b::new(1),
                BinaryField128b::new(1)
            ])
        );

        // Generate an imaginary random challenge (fixed for testing purposes).
        let r = BinaryField128b::from(5678);

        // Update the current claim using the round polynomial and the random value.
        let current_claim = round_polynomial.evaluate_at(&r);

        // Verify the correctness of the updated claim.
        assert_eq!(current_claim, BinaryField128b::new(284181495769767592368287233794578256034));

        // Bind the random value `r` to the Boolean check for the next round.
        boolcheck.bind(&r);

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
                BinaryField128b::ZERO,
            ]
        );

        // Validate the update of the boolcheck claim.
        assert_eq!(boolcheck.claim, BinaryField128b::new(284181495769767592368287233794578256034));

        // Verify that the challenge has been integrated inside the boolcheck.
        assert_eq!(boolcheck.challenges, Points(vec![BinaryField128b::new(5678)]));

        // Verify the correctness of the round polynomial cache.
        assert_eq!(boolcheck.round_polys, vec![compressed_round_polynomial]);

        boolcheck.poly_coords =
            Evaluations((0..4 * 128).map(BinaryField128b::new).collect::<Vec<_>>());

        // Test an imaginary algorithm execution.
        let alg_res = boolcheck.compute_algebraic(0, 1);

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
                    BinaryField128b::ZERO,
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
        // Set a phase switch parameter, which controls the folding phases.
        const PHASE_SWITCH: usize = 1;

        // Generate a vector `points` of 2 field elements in `BinaryField128b`.
        let points: Vec<_> = vec![BinaryField128b::new(1), BinaryField128b::new(2)];

        // Generate a vector `p` with 2^2 = 4 elements.
        let p = MultilinearLagrangianPolynomial::new(vec![
            BinaryField128b::new(11),
            BinaryField128b::new(22),
            BinaryField128b::new(33),
            BinaryField128b::new(44),
        ]);

        // Generate another vector `q` with 2^2 = 4 elements.
        let q = MultilinearLagrangianPolynomial::new(vec![
            BinaryField128b::new(111),
            BinaryField128b::new(222),
            BinaryField128b::new(333),
            BinaryField128b::new(444),
        ]);

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q = p.clone() & q.clone();

        // The prover compute the initial claim for the AND operation at the points in `points`.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Generate a folding challenge `gamma`
        let gamma = BinaryField128b::from(1234);

        // Create a new `BoolCheckBuilder` instance with:
        // - the phase switch parameter (c),
        // - the points at which the AND operation is evaluated,
        // - the Boolean package (AND operation for this test).
        let pts = points.into();
        let polys = [p, q];
        let boolcheck_builder = BoolCheckBuilder::<_, _, PHASE_SWITCH, _>::new(
            &AndPackage,
            &pts,
            [initial_claim],
            &polys,
        );

        // Build the Boolean check with the following parameters:
        // - the multilinear polynomials `p` and `q` used in the AND operation,
        // - the initial claim for the AND operation,
        // - the folding challenge `gamma`.
        let mut bool_check = boolcheck_builder.build(&gamma);

        // Add a precomputed round polynomial to the cache.
        let cached_poly = CompressedPoly([
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
        ]);
        bool_check.round_polys.push(cached_poly.clone());

        // Compute the round polynomial for the cached round.
        let round_poly = bool_check.round_polynomial();

        // Ensure the cached polynomial is returned.
        assert_eq!(round_poly, cached_poly);

        // Ensure no new polynomial is added to the cache.
        assert_eq!(bool_check.round_polys.len(), 1);
    }

    #[test]
    fn test_compute_round_polynomial_initial_round() {
        // Set a phase switch parameter, which controls the folding phases.
        const PHASE_SWITCH: usize = 1;

        // Generate a vector `points` of 2 field elements in `BinaryField128b`.
        let points: Vec<_> = vec![BinaryField128b::new(1), BinaryField128b::new(2)];

        // Generate a vector `p` with 2^2 = 4 elements.
        let p = MultilinearLagrangianPolynomial::new(vec![
            BinaryField128b::new(11),
            BinaryField128b::new(22),
            BinaryField128b::new(33),
            BinaryField128b::new(44),
        ]);

        // Generate another vector `q` with 2^2 = 4 elements.
        let q = MultilinearLagrangianPolynomial::new(vec![
            BinaryField128b::new(111),
            BinaryField128b::new(222),
            BinaryField128b::new(333),
            BinaryField128b::new(444),
        ]);

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q = p.clone() & q.clone();

        // The prover compute the initial claim for the AND operation at the points in `points`.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Generate a folding challenge `gamma`.
        let gamma = BinaryField128b::from(1234);

        // Create a new `BoolCheckBuilder` instance.
        let pts = points.into();
        let polys = [p, q];
        let boolcheck_builder = BoolCheckBuilder::<_, _, PHASE_SWITCH, _>::new(
            &AndPackage,
            &pts,
            [initial_claim],
            &polys,
        );

        // Build the Boolean check.
        let mut bool_check = boolcheck_builder.build(&gamma);

        // Compute the round polynomial for the initial round.
        let round_poly = bool_check.round_polynomial();

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
        // Set a phase switch parameter.
        const PHASE_SWITCH: usize = 2;

        // Generate a vector `points` of 3 field elements in `BinaryField128b`.
        let points: Vec<_> =
            vec![BinaryField128b::new(1), BinaryField128b::new(2), BinaryField128b::new(3)];

        // Generate a vector `p` with 2^3 = 8 elements.
        let p: MultilinearLagrangianPolynomial = (0..8).map(BinaryField128b::new).collect();

        // Generate another vector `q` with 2^3 = 8 elements.
        let q: MultilinearLagrangianPolynomial = (10..18).map(BinaryField128b::new).collect();

        // Compute the element-wise AND operation between `p` and `q`.
        let p_and_q = p.clone() & q.clone();

        // Compute the initial claim for the AND operation.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Generate a folding challenge `gamma`.
        let gamma = BinaryField128b::from(1234);

        // Create a new `BoolCheckBuilder` instance.
        let pts = points.into();
        let polys = [p, q];
        let boolcheck_builder = BoolCheckBuilder::<_, _, PHASE_SWITCH, _>::new(
            &AndPackage,
            &pts,
            [initial_claim],
            &polys,
        );

        // Build the Boolean check.
        let mut bool_check = boolcheck_builder.build(&gamma);

        // Simulate adding challenges equal to the number of variables.
        for i in 0..bool_check.number_variables() {
            bool_check.challenges.push(BinaryField128b::from(i as u128));
        }

        // Attempt to compute a round polynomial beyond the round limit.
        // This should panic.
        bool_check.round_polynomial();
    }

    #[test]
    #[should_panic(expected = "Claim does not match expected value.")]
    fn test_compute_round_polynomial_invalid_claim() {
        // Set a phase switch parameter.
        const PHASE_SWITCH: usize = 1;

        // Generate a vector `points` of 2 field elements in `BinaryField128b`.
        let points = vec![BinaryField128b::new(1), BinaryField128b::new(2)];

        // Generate a vector `p` with 2^2 = 4 elements.
        let p = MultilinearLagrangianPolynomial::new(vec![
            BinaryField128b::new(11),
            BinaryField128b::new(22),
            BinaryField128b::new(33),
            BinaryField128b::new(44),
        ]);

        // Generate another vector `q` with 2^2 = 4 elements.
        let q = MultilinearLagrangianPolynomial::new(vec![
            BinaryField128b::new(111),
            BinaryField128b::new(222),
            BinaryField128b::new(333),
            BinaryField128b::new(444),
        ]);

        // Compute an incorrect initial claim for testing purposes.
        let incorrect_claim = BinaryField128b::new(999);

        // Generate a folding challenge `gamma`.
        let gamma = BinaryField128b::from(1234);

        // Create a new `BoolCheckBuilder` instance.
        let pts = points.into();
        let polys = [p, q];
        let boolcheck_builder = BoolCheckBuilder::<_, _, PHASE_SWITCH, _>::new(
            &AndPackage,
            &pts,
            [incorrect_claim],
            &polys,
        );

        // Build the Boolean check.
        let mut bool_check = boolcheck_builder.build(&gamma);

        // Attempt to compute a round polynomial with the incorrect claim.
        // This should panic as the claim does not match the computed value.
        bool_check.round_polynomial();
    }

    #[test]
    fn test_compute_round_polynomial_correct_claim_update() {
        // Set a phase switch parameter.
        const PHASE_SWITCH: usize = 1;

        // Generate a vector `points` of 2 field elements in `BinaryField128b`.
        let points: Vec<_> = vec![BinaryField128b::new(1), BinaryField128b::new(2)];

        // Generate a vector `p` with 2^2 = 4 elements.
        let p = MultilinearLagrangianPolynomial::new(vec![
            BinaryField128b::new(11),
            BinaryField128b::new(22),
            BinaryField128b::new(33),
            BinaryField128b::new(44),
        ]);

        // Generate another vector `q` with 2^2 = 4 elements.
        let q = MultilinearLagrangianPolynomial::new(vec![
            BinaryField128b::new(111),
            BinaryField128b::new(222),
            BinaryField128b::new(333),
            BinaryField128b::new(444),
        ]);

        // Compute the element-wise AND operation between `p` and `q`.
        let p_and_q = p.clone() & q.clone();

        // Compute the initial claim for the AND operation.
        let initial_claim = p_and_q.evaluate_at(&points.clone().into());

        // Generate a folding challenge `gamma`.
        let gamma = BinaryField128b::from(1234);

        // Create a new `BoolCheckBuilder` instance.
        let pts = points.into();
        let polys = [p, q];
        let boolcheck_builder = BoolCheckBuilder::<_, _, PHASE_SWITCH, _>::new(
            &AndPackage,
            &pts,
            [initial_claim],
            &polys,
        );

        // Build the Boolean check.
        let mut bool_check = boolcheck_builder.build(&gamma);

        // Compute the round polynomial.
        let round_poly = bool_check.round_polynomial();

        // Decompress the round polynomial and retrieve its coefficients.
        let coeffs = round_poly.coeffs(initial_claim);

        // Generate a random challenge value.
        let challenge = BinaryField128b::from(3);

        // Compute the round polynomial.
        bool_check.bind(&challenge);

        // Ensure the polynomial has 3 coefficients (degree 2 polynomial).
        assert_eq!(coeffs.len(), 4);

        // Compute the updated claim using the polynomial coefficients.
        let updated_claim = coeffs[0] +
            coeffs[1] * challenge +
            coeffs[2] * challenge * challenge +
            coeffs[3] * challenge * challenge * challenge;

        // Verify that the updated claim matches the computed result.
        assert_eq!(updated_claim, bool_check.claim);
    }

    #[test]
    fn test_current_round_initial_state() {
        // Create a default BoolCheck instance with no challenges added.
        let points = Points::default();
        let bool_check: BoolCheck<'_, 0, 0, 0, AndPackage<0, 0>> = BoolCheck {
            points: &points,
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };

        // Assert that the initial round is 0 because no challenges have been added yet.
        assert_eq!(bool_check.current_round(), 0);
    }

    #[test]
    fn test_current_round_after_single_challenge() {
        // Create a default BoolCheck instance.
        let points = Points::default();
        let mut bool_check: BoolCheck<'_, 0, 0, 0, AndPackage<0, 0>> = BoolCheck {
            points: &points,
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };

        // Add one challenge to the challenges vector.
        bool_check.challenges.push(BinaryField128b::from(10));

        // Assert that the current round is now 1.
        assert_eq!(bool_check.current_round(), 1);
    }

    #[test]
    fn test_current_round_after_multiple_challenges() {
        // Create a default BoolCheck instance.
        let points = Points::default();
        let mut bool_check: BoolCheck<'_, 0, 0, 0, AndPackage<0, 0>> = BoolCheck {
            points: &points,
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };

        // Add multiple challenges to the challenges vector.
        bool_check.challenges.extend(vec![
            BinaryField128b::from(10),
            BinaryField128b::from(20),
            BinaryField128b::from(30),
        ]);

        // Assert that the current round reflects the number of challenges added (3).
        assert_eq!(bool_check.current_round(), 3);
    }

    #[test]
    fn test_current_round_after_removing_challenges() {
        // Create a default BoolCheck instance.
        let points = Points::default();
        let mut bool_check: BoolCheck<'_, 0, 0, 0, AndPackage<0, 0>> = BoolCheck {
            points: &points,
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };

        // Add multiple challenges to the challenges vector.
        bool_check.challenges.extend(vec![
            BinaryField128b::from(10),
            BinaryField128b::from(20),
            BinaryField128b::from(30),
        ]);

        // Remove one challenge (simulating a removal operation, if valid in the context).
        bool_check.challenges.pop();

        // Assert that the current round reflects the updated number of challenges (2).
        assert_eq!(bool_check.current_round(), 2);
    }

    #[test]
    fn test_current_round_with_no_challenges_after_clear() {
        // Create a default BoolCheck instance.
        let points = Points::default();
        let mut bool_check: BoolCheck<'_, 0, 0, 0, AndPackage<0, 0>> = BoolCheck {
            points: &points,
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };

        // Add multiple challenges to the challenges vector.
        bool_check.challenges.extend(vec![BinaryField128b::from(10), BinaryField128b::from(20)]);

        // Clear all challenges.
        bool_check.challenges.clear();

        // Assert that the current round is back to 0 after clearing challenges.
        assert_eq!(bool_check.current_round(), 0);
    }

    #[test]
    fn test_number_variables_initial_state() {
        // Create a BoolCheck instance with no points defined (default state).
        let points = Points::default();
        let bool_check: BoolCheck<'_, 0, 0, 0, AndPackage<0, 0>> = BoolCheck {
            points: &points,
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };

        // Assert that the number of variables is 0 as no points are defined.
        assert_eq!(bool_check.number_variables(), 0);
    }

    #[test]
    fn test_number_variables_with_single_point() {
        // Create a BoolCheck instance with a single point.
        let bool_check = BoolCheck::<0, 0, 0, AndPackage<0, 0>> {
            points: &Points::from(vec![BinaryField128b::from(1)]),
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };

        // Assert that the number of variables is 1, corresponding to the single point.
        assert_eq!(bool_check.number_variables(), 1);
    }

    #[test]
    fn test_number_variables_with_multiple_points() {
        // Create a BoolCheck instance with multiple points.
        let points =
            vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];
        let bool_check = BoolCheck::<0, 0, 0, AndPackage<0, 0>> {
            points: &Points::from(points.clone()),
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };

        // Assert that the number of variables matches the number of points (3).
        assert_eq!(bool_check.number_variables(), points.len());
    }

    #[test]
    fn test_number_variables_with_empty_points() {
        // Create a BoolCheck instance with an empty points vector.
        let points = Points::default();
        let bool_check = BoolCheck::<0, 0, 0, AndPackage<0, 0>> {
            points: &points,
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };

        // Assert that the number of variables is 0 when no points are defined.
        assert_eq!(bool_check.number_variables(), 0);
    }

    #[test]
    fn test_number_variables_with_large_number_of_points() {
        // Generate a large number of points (e.g., 1000 points).
        let points: Vec<_> = (0..1000).map(BinaryField128b::from).collect();

        // Create a BoolCheck instance with these points.
        let bool_check = BoolCheck::<0, 0, 0, AndPackage<0, 0>> {
            points: &Points::from(points.clone()),
            polys: &array::from_fn(|_| Default::default()),
            extended_table: Default::default(),
            poly_coords: Evaluations::default(),
            challenges: Points::default(),
            bit_mapping: Default::default(),
            eq_sequence: Default::default(),
            round_polys: Default::default(),
            claim: BinaryField128b::ZERO,
            gammas: array::from_fn(|_| Default::default()),
            algebraic_operations: &Default::default(),
        };

        // Assert that the number of variables matches the number of points (1000).
        assert_eq!(bool_check.number_variables(), points.len());
    }

    #[test]
    fn test_bind_single() {
        // Set a phase switch parameter.
        const PHASE_SWITCH: usize = 1;

        // Create a realistic setup with points and polynomials.
        let points = vec![BinaryField128b::from(1), BinaryField128b::from(2)];
        let p = MultilinearLagrangianPolynomial::new(vec![
            BinaryField128b::from(11),
            BinaryField128b::from(22),
            BinaryField128b::from(33),
            BinaryField128b::from(44),
        ]);

        let q = MultilinearLagrangianPolynomial::new(vec![
            BinaryField128b::from(111),
            BinaryField128b::from(222),
            BinaryField128b::from(333),
            BinaryField128b::from(444),
        ]);

        // Compute the AND operation between the polynomials and the initial claim.
        let p_and_q = p.clone() & q.clone();
        let initial_claim = p_and_q.evaluate_at(&Points::from(points.clone()));

        // Generate a folding challenge `gamma`.
        let gamma = BinaryField128b::from(1234);

        // Create a new `BoolCheckBuilder` instance.
        let pts = points.into();
        let polys = [p, q];
        let boolcheck_builder = BoolCheckBuilder::<_, _, PHASE_SWITCH, _>::new(
            &AndPackage,
            &pts,
            [initial_claim],
            &polys,
        );

        // Build the Boolean check.
        let mut bool_check = boolcheck_builder.build(&gamma);

        // Bind a random challenge `r` to the BoolCheck instance.
        let r = BinaryField128b::from(42);
        bool_check.bind(&r);

        // Verify that the challenge has been added.
        assert_eq!(bool_check.challenges.len(), 1);
        assert_eq!(bool_check.challenges[0], r);

        // Verify that the extended table is updated after binding.
        assert!(!bool_check.extended_table.is_empty());
    }
}
