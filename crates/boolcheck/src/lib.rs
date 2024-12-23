use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::{
    compressed::CompressedPoly, multinear_lagrangian::MultilinearLagrangianPolynomials,
    point::Points, univariate::UnivariatePolynomial,
};
use num_traits::{One, Zero};
use package::BooleanPackage;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub mod bool_trait;
pub mod builder;
pub mod package;

#[derive(Clone, Debug, Default)]
pub struct BoolCheck {
    points: Points,
    poly: Vec<BinaryField128b>,
    polys: Vec<Vec<BinaryField128b>>,
    extended_table: Vec<BinaryField128b>,
    poly_coords: Option<Vec<BinaryField128b>>,
    c: usize,
    challenges: Points,
    bit_mapping: Vec<u16>,
    eq_sequence: MultilinearLagrangianPolynomials,
    round_polys: Vec<CompressedPoly>,
    claim: BinaryField128b,
    boolean_package: BooleanPackage,
    gammas: Vec<BinaryField128b>,
}

impl BoolCheck {
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
    /// p(x) = p_0 + p_1*x_1 + p_2*x_2 + ... + p_n*x_n
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
            assert_eq!(half, 1 << number_variables - round - 1, "Invalid equality polynomial size");

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
                    self.exec_alg(poly_coords, i, 1 << (number_variables - self.c - 1))
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
        let eq_y_multiplier = self.challenges.eq_eval(&self.points[..round].to_vec().into());

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

    pub fn exec_alg(
        &self,
        data: &[BinaryField128b],
        mut idx_a: usize,
        offset: usize,
    ) -> [BinaryField128b; 3] {
        match self.boolean_package {
            BooleanPackage::And => {
                // `idx_a` is the starting index for the first operand in the AND operation (`P`).
                // Double the starting index to account for the structure of the data.
                idx_a *= 2;

                // Compute the starting index for the second operand in the AND operation (`Q`).
                let idx_b = idx_a + offset * 128;

                // Initialize the result accumulators for the three evaluations. This will hold:
                // - `W(0) = Σ_{x} P(0,x) * Q(0,x) * eq(x; q)`,
                // - `W(1) = Σ_{x} P(1,x) * Q(1,x) * eq(x; q)`,
                // - `W(∞) = Σ_{x} P(∞,x) * Q(∞,x) * eq(x; q)`.
                let mut acc = [BinaryField128b::zero(); 3];

                // Iterate over the 128 basis elements.
                // This loop implements the summation over `x_{>c}`:
                // ```
                // W(t) = Σ_{x_{>c}} (P ∧ Q)(t, x_{>c}) * eq(x_{>c}; q_{>c}).
                // ```
                for i in 0..128 {
                    // Precompute the offsets for this iteration.
                    // This is used to fetch the polynomials:
                    // - `P(t, x_{>c})` and `Q(t, x_{>c})` for the `i-th` basis vector of `x_{>c}`.
                    let offset_a = idx_a + i * offset;
                    let offset_b = idx_b + i * offset;

                    // Fetch the data elements for this iteration.
                    // `a_0 = P(t=0, x)`
                    let a0 = data[offset_a];
                    // `a_1 = P(t=1, x)`
                    let a1 = data[offset_a + 1];
                    // `b_0 = Q(t=0, x)`
                    let b0 = data[offset_b];
                    // `b_1 = Q(t=1, x)`
                    let b1 = data[offset_b + 1];

                    // Compute the contributions for this basis element.
                    let basis = BinaryField128b::basis(i);
                    // `W(0)  = Σ_{x} P(0,x) * Q(0,x) * eq(x; q)`
                    acc[0] += basis * a0 * b0;
                    // `W(1)  = Σ_{x} P(1,x) * Q(1,x) * eq(x; q)`
                    acc[1] += basis * a1 * b1;
                    // `W(∞)  = Σ_{x} (P(0,x) + P(1,x)) * (Q(0,x) + Q(1,x)) * eq(x; q)`
                    acc[2] += basis * (a0 + a1) * (b0 + b1);
                }

                // Compress results using gammas.
                //
                // Gamma is a folding factor that is used to compress the polynomial.
                [acc[0] * self.gammas[0], acc[1] * self.gammas[0], acc[2] * self.gammas[0]]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use builder::BoolCheckBuilder;
    use hashcaster_poly::{multinear_lagrangian::MultilinearLagrangianPolynomial, point::Point};
    use package::BooleanPackage;
    use std::iter::repeat_with;

    #[test]
    fn test_current_rounds() {
        // Define a sample BoolCheck instance.
        // No challenges have been submitted yet.
        let mut bool_check = BoolCheck::default();

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
        let bool_check = BoolCheck { points: points.clone().into(), ..Default::default() };

        // Assert that the number of variables matches the length of the points vector.
        assert_eq!(bool_check.number_variables(), points.len());

        // Test with no points (empty vector).
        let empty_bool_check = BoolCheck { points: Points::default(), ..Default::default() };
        assert_eq!(empty_bool_check.number_variables(), 0);

        // Test with a large number of points.
        let large_points: Vec<_> = (0..1000).map(BinaryField128b::from).collect();
        let large_bool_check =
            BoolCheck { points: large_points.clone().into(), ..Default::default() };
        assert_eq!(large_bool_check.number_variables(), large_points.len());
    }

    #[test]
    fn test_new_andcheck() {
        // Set the number of variables for the test.
        let num_vars = 20;

        // Generate a vector `points` of `num_vars` random field elements in `BinaryField128b`.
        // This represents a set of random variables that will be used in the test.
        let points: Vec<_> = repeat_with(BinaryField128b::random).take(num_vars).collect();

        // Generate a multilinear polynomial `p` with 2^num_vars random elements in
        // `BinaryField128b`. This represents one operand (a polynomial) in the AND
        // operation.
        let p: MultilinearLagrangianPolynomial =
            repeat_with(BinaryField128b::random).take(1 << num_vars).collect::<Vec<_>>().into();

        // Generate another multilinear polynomial `q` with 2^num_vars random elements in
        // `BinaryField128b`. This represents the second operand (a polynomial) in the AND
        // operation.
        let q: MultilinearLagrangianPolynomial =
            repeat_with(BinaryField128b::random).take(1 << num_vars).collect::<Vec<_>>().into();

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q = p.clone() & q.clone();

        // The prover compute the initial claim for the AND operation at the points in `points`.
        let initial_claim = p_and_q.evaluate_at(&points);

        // Set a phase switch parameter, which controls the folding phases.
        let phase_switch = 5;

        // Generate a random folding challenge `gamma` in `BinaryField128b`.
        let gamma = BinaryField128b::random();

        // Create a new `BoolCheckBuilder` instance with:
        // - the phase switch parameter (c),
        // - the points at which the AND operation is evaluated,
        // - the Boolean package (AND operation for this test).
        let boolcheck_builder = BoolCheckBuilder::new(
            phase_switch,
            points,
            BooleanPackage::And,
            [initial_claim],
            gamma,
        );

        // Build the Boolean check with the following parameters:
        // - the multilinear polynomials `p` and `q` used in the AND operation,
        // - the initial claim for the AND operation,
        // - the folding challenge `gamma`.
        let _boolcheck = boolcheck_builder.build(&[p, q]);

        // Initialize the current claim as the initial claim.
        // The current claim will be updated during each round of the protocol.
        let mut current_claim = initial_claim;
    }
}
