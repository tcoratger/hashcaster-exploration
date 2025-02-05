use hashcaster_primitives::{
    binary_field::BinaryField128b,
    poly::{
        compressed::CompressedPoly, evaluation::FixedEvaluations,
        multinear_lagrangian::MultilinearLagrangianPolynomial, point::Points,
    },
    sumcheck::{EvaluationProvider, Sumcheck},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::array::from_fn;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProdCheck<const N: usize> {
    /// Polynomials of the first input set (`P`).
    pub p_polys: [MultilinearLagrangianPolynomial; N],
    /// Polynomials of the second input set (`Q`).
    pub q_polys: [MultilinearLagrangianPolynomial; N],
    /// The initial claim that represents the sum of the products of `P` and `Q`.
    pub claim: BinaryField128b,
    /// A list of challenges (random values) generated during the protocol.
    pub challenges: Points,
    /// The number of variables in the polynomials (`log2(size of each polynomial)`).
    pub num_vars: usize,
    /// Cached result of the round message polynomial, used for optimization.
    pub cached_round_msg: Option<CompressedPoly<2>>,
}

impl<const N: usize> Default for ProdCheck<N> {
    fn default() -> Self {
        Self {
            p_polys: core::array::from_fn(|_| Default::default()),
            q_polys: core::array::from_fn(|_| Default::default()),
            claim: BinaryField128b::ZERO,
            challenges: Default::default(),
            num_vars: 0,
            cached_round_msg: None,
        }
    }
}

impl<const N: usize> Sumcheck<N, 2> for ProdCheck<N> {
    type Output = ProdCheckOutput<N>;

    fn round_polynomial(&mut self) -> CompressedPoly<2> {
        // Fetch the length of the first polynomial to determine the range of the current variable.
        let p0_len = self.p_polys[0].len();

        // Ensure the protocol is not complete (length must be greater than 1).
        assert!(p0_len > 1, "The protocol is already complete");

        // Compute the midpoint to split the range into two halves.
        let half = p0_len / 2;

        // Return the cached round polynomial if it exists.
        if let Some(cache_round_msg) = &self.cached_round_msg {
            return cache_round_msg.clone();
        }

        // Compute `pq_zero`, `pq_one`, and `pq_inf` for each index in parallel.
        let mut poly: [BinaryField128b; 3] = (0..half)
            .into_par_iter()
            .map(|i| {
                // Contribution for `X_i = 0`
                let mut pq_zero = BinaryField128b::ZERO;
                // Contribution for `X_i = 1`
                let mut pq_one = BinaryField128b::ZERO;
                // Contribution of sums across both halves
                let mut pq_inf = BinaryField128b::ZERO;

                // Iterate through all polynomials to compute contributions.
                for j in 0..N {
                    let p = &self.p_polys[j];
                    let q = &self.q_polys[j];

                    // Precompute polynomial values for efficiency.
                    // Lower half of `P`
                    let p_low = p[2 * i];
                    // Upper half of `P`
                    let p_high = p[2 * i + 1];
                    // Lower half of `Q`
                    let q_low = q[2 * i];
                    // Upper half of `Q`
                    let q_high = q[2 * i + 1];

                    // Compute the product for the lower halves.
                    pq_zero += p_low * q_low;

                    // Compute the product for the upper halves.
                    pq_one += p_high * q_high;

                    // Compute the product of sums across both halves.
                    pq_inf += (p_low + p_high) * (q_low + q_high);
                }

                // Return the contributions for this index.
                [pq_zero, pq_one, pq_inf]
            })
            // Combine results from all indices into three accumulated terms.
            .reduce(|| [BinaryField128b::ZERO; 3], |[a, b, c], [d, e, f]| [a + d, b + e, c + f]);

        // Adjust the coefficients for the round polynomial.
        // The second coefficient includes contributions from all terms.
        poly[1] += poly[0] + poly[2];

        // Compress the polynomial into a univariate form.
        let (compressed_poly, computed_claim) = CompressedPoly::compress(&poly);

        // Ensure the computed claim matches the expected value.
        assert_eq!(computed_claim, self.claim, "Claim does not match expected value.");

        // Cache the result for future use.
        self.cached_round_msg = Some(compressed_poly.clone());

        // Return the compressed polynomial.
        compressed_poly
    }

    fn bind(&mut self, r: &BinaryField128b) {
        // Validate that the protocol is not complete.
        // Get the length of the first polynomial.
        let p0_len = self.p_polys[0].len();
        assert!(p0_len > 1, "The protocol is already complete");

        // Compute the midpoint of the polynomial range.
        let half = p0_len / 2;

        // Compute the new claim using the decompressed round polynomial and the challenge.
        //
        // Fetch the coefficients of the round polynomial and evaluate it at the challenge `r`.
        let round_poly = self.round_polynomial().coeffs(self.claim);
        self.claim = round_poly.evaluate_at(r);

        // Add the new challenge to the list of challenges.
        self.challenges.push(*r);

        // Prepare new (halved) polynomials for `P` and `Q`.
        let mut p_new: [MultilinearLagrangianPolynomial; N] =
            from_fn(|_| vec![BinaryField128b::ZERO; half].into());
        let mut q_new: [MultilinearLagrangianPolynomial; N] =
            from_fn(|_| vec![BinaryField128b::ZERO; half].into());

        // Halve the polynomials using the challenge.
        for i in 0..N {
            // Current polynomial from `P`.
            let p = &self.p_polys[i];
            // Current polynomial from `Q`.
            let q = &self.q_polys[i];

            // Compute the new values for the halved polynomials.
            let (p_values, q_values): (Vec<_>, Vec<_>) = (0..half)
                .into_par_iter()
                .map(|j| {
                    (
                        p[2 * j] + (p[2 * j + 1] + p[2 * j]) * r,
                        q[2 * j] + (q[2 * j + 1] + q[2 * j]) * r,
                    )
                })
                .unzip();

            // Assign the computed values to the new polynomials.
            p_new[i] = p_values.into();
            q_new[i] = q_values.into();
        }

        // Replace the old polynomials with the halved ones.
        self.p_polys = p_new;
        self.q_polys = q_new;

        // Clear the cached round message as it is no longer valid.
        self.cached_round_msg = None;
    }

    fn finish(self) -> Self::Output {
        // Extract the final evaluations from `p_polys`
        let p_evaluations = FixedEvaluations::new(from_fn(|i| {
            // Ensure the polynomial is fully reduced (length == 1).
            assert_eq!(self.p_polys[i].len(), 1, "The protocol is not complete");
            // Extract the single value from the polynomial.
            self.p_polys[i][0]
        }));

        // Extract the final evaluations from `q_polys`
        let q_evaluations = FixedEvaluations::new(from_fn(|i| {
            // Ensure the polynomial is fully reduced (length == 1).
            assert_eq!(self.q_polys[i].len(), 1, "The protocol is not complete");
            // Extract the single value from the polynomial.
            self.q_polys[i][0]
        }));

        ProdCheckOutput { p_evaluations, q_evaluations }
    }
}

impl<const N: usize> ProdCheck<N> {
    pub fn new(
        p_polys: [MultilinearLagrangianPolynomial; N],
        q_polys: [MultilinearLagrangianPolynomial; N],
        claim: BinaryField128b,
        check_init_claim: bool,
    ) -> Self {
        // Compute the number of variables from the length of the first polynomial.
        let num_vars = p_polys[0].len().ilog2() as usize;

        // Validate that each polynomial in `P` and `Q` has the expected size.
        assert!(
            p_polys
                .iter()
                .zip(q_polys.iter())
                .all(|(p, q)| p.len() == 1 << num_vars && q.len() == 1 << num_vars),
            "All polynomials must have size 2^num_vars"
        );

        // Optionally verify the correctness of the initial claim.
        if check_init_claim {
            // Compute the expected claim by multiplying each pair of polynomials.
            let expected_claim =
                p_polys.iter().zip(&q_polys).fold(BinaryField128b::ZERO, |acc, (p, q)| {
                    acc + p
                        .iter()
                        .zip(q)
                        .map(|(&p_val, &q_val)| p_val * q_val)
                        .sum::<BinaryField128b>()
                });

            // Ensure the provided claim matches the computed claim.
            assert_eq!(claim, expected_claim, "Initial claim does not match computed value");
        }

        // Return the new `Prodcheck` instance.
        Self { p_polys, q_polys, claim, num_vars, ..Default::default() }
    }
}

/// Represents the final output of the sumcheck protocol.
///
/// # Purpose
/// The `ProdCheckOutput` struct encapsulates the final evaluations of the `p_polys`
/// and `q_polys` after the sumcheck protocol has been completed. These evaluations
/// are derived from fully reduced polynomials.
///
/// # Fields
/// - `p_evaluations`: The final evaluations of the `p_polys`, represented as an `Evaluations`
///   object containing the single values extracted from the reduced `p_polys`.
/// - `q_evaluations`: The final evaluations of the `q_polys`, represented as an `Evaluations`
///   object containing the single values extracted from the reduced `q_polys`.
#[derive(Clone, Debug)]
pub struct ProdCheckOutput<const N: usize> {
    /// Final evaluations of the `p_polys`.
    ///
    /// Each evaluation corresponds to the single value extracted from a fully reduced
    /// polynomial in the `p_polys` set.
    pub p_evaluations: FixedEvaluations<N>,

    /// Final evaluations of the `q_polys`.
    ///
    /// Each evaluation corresponds to the single value extracted from a fully reduced
    /// polynomial in the `q_polys` set.
    pub q_evaluations: FixedEvaluations<N>,
}

impl<const N: usize> EvaluationProvider<N> for ProdCheckOutput<N> {
    fn evals(self) -> FixedEvaluations<N> {
        self.p_evaluations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;
    use std::array;

    #[test]
    fn test_prodcheck_new_valid_claim() {
        // Number of polynomials in P and Q.
        const N: usize = 3;

        // Create valid polynomials for P and Q.
        let p1 = BinaryField128b::from(1);
        let p2 = BinaryField128b::from(2);
        let p3 = BinaryField128b::from(3);
        let p4 = BinaryField128b::from(4);
        let p_polys: [_; N] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![p1, p2, p3, p4]));

        let q1 = BinaryField128b::from(5);
        let q2 = BinaryField128b::from(6);
        let q3 = BinaryField128b::from(7);
        let q4 = BinaryField128b::from(8);
        let q_polys: [_; N] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![q1, q2, q3, q4]));

        // Compute the claim manually.
        let claim = (p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) +
            (p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) +
            (p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4);

        // Create ProdCheck with check_init_claim set to true.
        let prodcheck = ProdCheck::new(p_polys.clone(), q_polys.clone(), claim, true);

        // Assert ProdCheck is created correctly.
        assert_eq!(
            prodcheck,
            ProdCheck {
                p_polys,
                q_polys,
                claim,
                challenges: Points::default(),
                num_vars: 2,
                cached_round_msg: None,
            }
        );
    }

    #[test]
    #[should_panic(expected = "Initial claim does not match computed value")]
    fn test_prodcheck_new_invalid_claim() {
        const N: usize = 2;

        // Create valid polynomials for P and Q.
        let p_polys: [_; N] = array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(1),
                BinaryField128b::from(2),
                BinaryField128b::from(3),
                BinaryField128b::from(4),
            ])
        });
        let q_polys: [_; N] = array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(4),
                BinaryField128b::from(3),
                BinaryField128b::from(2),
                BinaryField128b::from(1),
            ])
        });

        // Use an incorrect claim.
        let incorrect_claim = BinaryField128b::from(999);

        // This should panic.
        ProdCheck::new(p_polys, q_polys, incorrect_claim, true);
    }

    #[test]
    fn test_prodcheck_new_without_checking_claim() {
        const N: usize = 2;

        // Create valid polynomials for P and Q.
        let p_polys: [_; N] = array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(1),
                BinaryField128b::from(2),
                BinaryField128b::from(3),
                BinaryField128b::from(4),
            ])
        });
        let q_polys: [_; N] = array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(4),
                BinaryField128b::from(3),
                BinaryField128b::from(2),
                BinaryField128b::from(1),
            ])
        });

        // Use an arbitrary claim without checking.
        let arbitrary_claim = BinaryField128b::from(123);

        // Create ProdCheck with check_init_claim set to false.
        let prodcheck = ProdCheck::new(p_polys.clone(), q_polys.clone(), arbitrary_claim, false);

        // Assert ProdCheck is created correctly.
        assert_eq!(
            prodcheck,
            ProdCheck {
                p_polys,
                q_polys,
                claim: arbitrary_claim,
                challenges: Points::default(),
                num_vars: 2,
                cached_round_msg: None,
            }
        );
    }

    #[test]
    #[should_panic(expected = "All polynomials must have size 2^num_vars")]
    fn test_prodcheck_new_invalid_polynomial_size_p() {
        const N: usize = 2;

        // Create polynomials of invalid sizes for P and Q.
        // Second polynomial in P has an invalid size.
        let p_polys: [_; N] = [
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(1),
                BinaryField128b::from(2),
            ]),
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::from(1)]),
        ];
        let q_polys: [_; N] = [
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(1),
                BinaryField128b::from(2),
            ]),
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(1),
                BinaryField128b::from(2),
            ]),
        ];

        // This should panic.
        ProdCheck::new(p_polys, q_polys, BinaryField128b::ZERO, true);
    }

    #[test]
    #[should_panic(expected = "All polynomials must have size 2^num_vars")]
    fn test_prodcheck_new_invalid_polynomial_size_q() {
        const N: usize = 2;

        // Create polynomials of invalid sizes for P and Q.
        // First polynomial in Q has an invalid
        let p_polys: [_; N] = [
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(1),
                BinaryField128b::from(2),
            ]),
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(1),
                BinaryField128b::from(2),
            ]),
        ];
        let q_polys: [_; N] = [
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::from(2)]),
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(1),
                BinaryField128b::from(2),
            ]),
        ];

        // This should panic.
        ProdCheck::new(p_polys, q_polys, BinaryField128b::ZERO, true);
    }

    #[test]
    fn test_prodcheck_compute_round_polynomial_valid() {
        const N: usize = 2;

        // Create simple polynomials for P and Q.
        let p1 = BinaryField128b::from(1);
        let p2 = BinaryField128b::from(2);
        let p3 = BinaryField128b::from(3);
        let p4 = BinaryField128b::from(4);
        let p_polys: [_; N] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![p1, p2, p3, p4]));

        let q1 = BinaryField128b::from(5);
        let q2 = BinaryField128b::from(6);
        let q3 = BinaryField128b::from(7);
        let q4 = BinaryField128b::from(8);
        let q_polys: [_; N] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![q1, q2, q3, q4]));

        // Compute the claim manually.
        let claim =
            (p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) + (p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4);

        // Create ProdCheck with check_init_claim set to true.
        let mut prodcheck = ProdCheck::new(p_polys, q_polys, claim, true);

        // Assert the cached round message is initially unset.
        assert!(prodcheck.cached_round_msg.is_none());

        // Compute the round polynomial.
        let compressed_poly = prodcheck.round_polynomial();

        // Manually compute the expected compressed polynomial.
        // - `pq_zero` is the sum of products of lower halves for all polynomials.
        // - `pq_one` is omitted in the compressed polynomial.
        // - `pq_inf` is the sum of products of sums of halves for all polynomials.
        let pq_zero = (p1 * q1 + p3 * q3) + (p1 * q1 + p3 * q3);
        let pq_inf = ((p1 + p2) * (q1 + q2) + (p3 + p4) * (q3 + q4)) +
            ((p1 + p2) * (q1 + q2) + (p3 + p4) * (q3 + q4));

        // Assert the compressed polynomial is correct.
        assert_eq!(compressed_poly.len(), 2);
        assert_eq!(pq_zero, compressed_poly[0]);
        assert_eq!(pq_inf, compressed_poly[1]);

        // Assert the cached round message is set.
        assert_eq!(prodcheck.cached_round_msg, Some(compressed_poly));
    }

    #[test]
    #[should_panic(expected = "Claim does not match expected value.")]
    fn test_prodcheck_compute_round_polynomial_invalid_claim() {
        const N: usize = 2;

        // Create simple polynomials for P and Q.
        let p_polys: [_; N] = array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(1),
                BinaryField128b::from(2),
                BinaryField128b::from(3),
                BinaryField128b::from(4),
            ])
        });
        let q_polys: [_; N] = array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(5),
                BinaryField128b::from(6),
                BinaryField128b::from(7),
                BinaryField128b::from(8),
            ])
        });

        // Use an incorrect claim.
        let incorrect_claim = BinaryField128b::from(999);

        let mut prodcheck = ProdCheck::new(p_polys, q_polys, incorrect_claim, false);

        // This should panic due to mismatched claim.
        prodcheck.round_polynomial();
    }

    #[test]
    #[should_panic(expected = "The protocol is already complete")]
    fn test_prodcheck_compute_round_polynomial_protocol_complete() {
        const N: usize = 2;

        // Create polynomials with size 1 (indicating protocol completion).
        let p1 = BinaryField128b::from(1);
        let p_polys: [_; N] = array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![p1]));
        let q1 = BinaryField128b::from(2);
        let q_polys: [_; N] = array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![q1]));

        // Compute the claim manually.
        let claim = (p1 * q1) + (p1 * q1);

        // Create ProdCheck with check_init_claim set to true.
        let mut prodcheck = ProdCheck::new(p_polys, q_polys, claim, true);

        // This should panic because the protocol is complete.
        prodcheck.round_polynomial();
    }

    #[test]
    fn test_prodcheck_bind_valid_challenge() {
        const N: usize = 3;

        // Create valid polynomials for P and Q.
        let p1 = BinaryField128b::from(1);
        let p2 = BinaryField128b::from(2);
        let p3 = BinaryField128b::from(3);
        let p4 = BinaryField128b::from(4);
        let p_polys: [_; N] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![p1, p2, p3, p4]));

        let q1 = BinaryField128b::from(5);
        let q2 = BinaryField128b::from(6);
        let q3 = BinaryField128b::from(7);
        let q4 = BinaryField128b::from(8);
        let q_polys: [_; N] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![q1, q2, q3, q4]));

        // Compute the initial claim.
        let initial_claim = (p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) +
            (p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4) +
            (p1 * q1 + p2 * q2 + p3 * q3 + p4 * q4);

        // Create ProdCheck with the initial state.
        let mut prodcheck = ProdCheck::new(p_polys, q_polys, initial_claim, true);

        // Perform binding with a valid challenge.
        let challenge = BinaryField128b::from(3);
        prodcheck.bind(&challenge);

        // Manually compute the expected compressed polynomial.
        // - `pq_zero` is the sum of products of lower halves for all polynomials.
        // - `pq_one` is omitted in the compressed polynomial.
        // - `pq_inf` is the sum of products of sums of halves for all polynomials.
        let pq_zero = (p1 * q1 + p3 * q3) + (p1 * q1 + p3 * q3) + (p1 * q1 + p3 * q3);
        let pq_inf = ((p1 + p2) * (q1 + q2) + (p3 + p4) * (q3 + q4)) +
            ((p1 + p2) * (q1 + q2) + (p3 + p4) * (q3 + q4)) +
            ((p1 + p2) * (q1 + q2) + (p3 + p4) * (q3 + q4));

        // Compute the coefficients of the univariate round polynomial.
        let a = pq_zero;
        let b = pq_inf + pq_zero + pq_zero + initial_claim;
        let c = pq_inf;

        // Compute the new claim using the challenge.
        let new_claim = a + b * challenge + c * challenge * challenge;

        // Assert the new claim is updated correctly.
        assert_eq!(new_claim, prodcheck.claim);

        // Compute manually the expected new polynomials for P and Q.
        let expected_p: [_; N] = array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![
                p1 + (p2 + p1) * challenge,
                p3 + (p4 + p3) * challenge,
            ])
        });

        let expected_q: [_; N] = array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![
                q1 + (q2 + q1) * challenge,
                q3 + (q4 + q3) * challenge,
            ])
        });

        // Assert the polynomials are reduced in size.
        assert_eq!(prodcheck.p_polys, expected_p);
        assert_eq!(prodcheck.q_polys, expected_q);

        // Assert the polynomials are reduced in size.
        // Original polynomials have size 4, so the new size should be 2.
        assert_eq!(prodcheck.p_polys[0].len(), 2);
        assert_eq!(prodcheck.q_polys[0].len(), 2);

        // Assert that the challenge is added to the list of challenges.
        assert_eq!(prodcheck.challenges, Points::from(vec![challenge]));

        // Assert the cached round message is cleared.
        assert!(prodcheck.cached_round_msg.is_none());
    }

    #[test]
    #[should_panic(expected = "The protocol is already complete")]
    fn test_prodcheck_bind_protocol_complete() {
        const N: usize = 2;

        // Create polynomials with size 1 (indicating protocol completion).
        let p1 = BinaryField128b::from(1);
        let p_polys: [_; N] = array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![p1]));
        let q1 = BinaryField128b::from(2);
        let q_polys: [_; N] = array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![q1]));

        // Compute the claim manually.
        let claim = (p1 * q1) + (p1 * q1);

        // Create ProdCheck with the final state.
        let mut prodcheck = ProdCheck::new(p_polys, q_polys, claim, true);

        // Attempt to bind with a new challenge (should panic).
        let challenge = BinaryField128b::from(3);
        prodcheck.bind(&challenge);
    }

    #[test]
    fn test_prodcheck_finish_valid_state() {
        const N: usize = 2;

        // Create polynomials with size 1 (indicating protocol completion).
        let p1 = BinaryField128b::from(1);
        let p_polys: [_; N] = array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![p1]));
        let q1 = BinaryField128b::from(2);
        let q_polys: [_; N] = array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![q1]));

        // Compute the claim manually.
        let claim = (p1 * q1) + (p1 * q1);

        // Create ProdCheck in the completed state.
        let prodcheck = ProdCheck::new(p_polys, q_polys, claim, true);

        // Call `finish` and validate the output.
        let output = prodcheck.finish();
        assert_eq!(output.p_evaluations, FixedEvaluations::new([p1; N]));
        assert_eq!(output.q_evaluations, FixedEvaluations::new([q1; N]));
    }

    #[test]
    #[should_panic(expected = "The protocol is not complete")]
    fn test_prodcheck_finish_incomplete_state() {
        const N: usize = 2;

        // Create polynomials with size > 1 (indicating protocol is incomplete).
        let p1 = BinaryField128b::from(1);
        let p2 = BinaryField128b::from(2);
        let p_polys: [_; N] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![p1, p2]));
        let q1 = BinaryField128b::from(3);
        let q2 = BinaryField128b::from(4);
        let q_polys: [_; N] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![q1, q2]));

        // Compute the claim manually.
        let claim = (p1 * q1 + p2 * q2) + (p1 * q1 + p2 * q2);

        // Create ProdCheck in an incomplete state.
        let prodcheck = ProdCheck::new(p_polys, q_polys, claim, true);

        // This should panic as the protocol is incomplete.
        prodcheck.finish();
    }

    #[test]
    fn test_prodcheck_finish_multiple_variables() {
        const N: usize = 3;

        // Create polynomials with size 1 for completion.
        let p_vals = [BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];
        let q_vals = [BinaryField128b::from(4), BinaryField128b::from(5), BinaryField128b::from(6)];
        let p_polys: [_; N] =
            array::from_fn(|i| MultilinearLagrangianPolynomial::new(vec![p_vals[i]]));
        let q_polys: [_; N] =
            array::from_fn(|i| MultilinearLagrangianPolynomial::new(vec![q_vals[i]]));

        // Compute the claim manually.
        let claim = p_vals.iter().zip(&q_vals).map(|(p, q)| *p * *q).sum();

        // Create ProdCheck in the completed state.
        let prodcheck = ProdCheck::new(p_polys, q_polys, claim, true);

        // Call `finish` and validate the output.
        let output = prodcheck.finish();
        assert_eq!(output.p_evaluations, FixedEvaluations::new(p_vals));
        assert_eq!(output.q_evaluations, FixedEvaluations::new(q_vals));
    }

    #[test]
    fn test_prodcheck_full() {
        // Number of polynomials in the test.
        const N: usize = 3;
        // Number of variables for the polynomials.
        const NUM_VARS: usize = 15;

        let rng = &mut OsRng;

        let start = std::time::Instant::now();

        // Generate random polynomials for `p_polys` and `q_polys`.
        // Each polynomial has `2^NUM_VARS` evaluations.
        let p_polys: [_; N] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::random(1 << NUM_VARS, rng));

        let q_polys: [_; N] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::random(1 << NUM_VARS, rng));

        // Compute the initial claim: the sum of the element-wise products of `P` and `Q`.
        let mut current_claim =
            p_polys.iter().zip(&q_polys).fold(BinaryField128b::ZERO, |acc, (p, q)| {
                acc + p.iter().zip(q).map(|(&p_val, &q_val)| p_val * q_val).sum::<BinaryField128b>()
            });

        // Initialize the `ProdCheck` object with the generated polynomials and computed claim.
        let mut prodcheck = ProdCheck::new(p_polys.clone(), q_polys.clone(), current_claim, true);

        // Simulate the rounds of the sumcheck protocol.
        for _ in 0..NUM_VARS {
            // Compute the round polynomial for the current state of the protocol.
            let compressed_round_polynomial = prodcheck.round_polynomial();

            // Generate a random challenge `r` for the current round.
            let r = BinaryField128b::random(rng);

            // Decompress the round polynomial to obtain its coefficients.
            // The round polynomial is represented as a univariate polynomial in `r`.
            let round_polynomial = compressed_round_polynomial.coeffs(current_claim);

            // Update the current claim by evaluating the round polynomial at the challenge `r`.
            // The evaluation is computed as:
            // ```
            // current_claim = c_0 + r * c_1 + r ^ 2 * c_2
            // ```
            current_claim =
                round_polynomial[0] + r * round_polynomial[1] + r * r * round_polynomial[2];

            // Bind the challenge `r` to the `ProdCheck` instance, updating its state.
            prodcheck.bind(&r);
        }

        // Ensure the protocol is complete, meaning all polynomials are reduced to a single value.
        assert!(prodcheck.p_polys.iter().all(|p| p.len() == 1));
        assert!(prodcheck.q_polys.iter().all(|q| q.len() == 1));

        // Extract the final evaluations using the `finish` method.
        let output = prodcheck.clone().finish();

        // Compute evaluations using `from_fn`, avoiding intermediate Vec allocations.
        let p_evaluations = std::array::from_fn(|i| p_polys[i].evaluate_at(&prodcheck.challenges));
        let q_evaluations = std::array::from_fn(|i| q_polys[i].evaluate_at(&prodcheck.challenges));

        // Verify that the final evaluations match the expected values.
        assert_eq!(output.p_evaluations, FixedEvaluations::new(p_evaluations));
        assert_eq!(output.q_evaluations, FixedEvaluations::new(q_evaluations));

        // Compute the final claim by multiplying the final evaluations of `p_polys` and `q_polys`.
        let final_claim = p_evaluations
            .iter()
            .zip(&q_evaluations)
            .map(|(&p_eval, &q_eval)| p_eval * q_eval)
            .fold(BinaryField128b::ZERO, |acc, val| acc + val);

        // Ensure the final computed claim matches the current claim in the protocol.
        assert_eq!(final_claim, current_claim);

        println!("total time {:?}", start.elapsed().as_millis());
    }
}
