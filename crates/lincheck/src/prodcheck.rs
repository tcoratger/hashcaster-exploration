use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::{
    compressed::CompressedPoly, multinear_lagrangian::MultilinearLagrangianPolynomial,
    point::Points,
};
use num_traits::identities::Zero;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
    pub cached_round_msg: Option<CompressedPoly>,
}

impl<const N: usize> Default for ProdCheck<N> {
    fn default() -> Self {
        Self {
            p_polys: core::array::from_fn(|_| Default::default()),
            q_polys: core::array::from_fn(|_| Default::default()),
            claim: BinaryField128b::zero(),
            challenges: Default::default(),
            num_vars: 0,
            cached_round_msg: None,
        }
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
                p_polys.iter().zip(&q_polys).fold(BinaryField128b::zero(), |acc, (p, q)| {
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

    /// Computes the round polynomial for the current state of the prover in the sumcheck protocol.
    ///
    /// ### Purpose
    /// The round polynomial represents the contribution of the current variable in the sumcheck
    /// protocol. It compresses multilinear polynomial evaluations into a smaller univariate
    /// polynomial, which is required for verification in the sumcheck protocol.
    ///
    /// ### Mathematical Context
    /// Given two sets of polynomials, `P` and `Q`, the goal is to compute the contribution of the
    /// current variable `X_i` to the overall product sum. For each index `i`, the contribution
    /// consists of three terms:
    ///
    /// 1. **`pq_zero`**: Product of evaluations when `X_i = 0` for all polynomials.
    /// 2. **`pq_one`**: Product of evaluations when `X_i = 1` for all polynomials.
    /// 3. **`pq_inf`**: Product of sums of evaluations across both halves.
    ///
    /// These terms are combined into a univariate polynomial:
    ///
    /// ```text
    /// g_i(X_i) = pq_zero + (pq_inf - pq_zero - pq_one) * X_i
    /// ```
    ///
    /// This method iteratively reduces the polynomial size and caches intermediate results to
    /// optimize future computations.
    ///
    /// ### Steps
    /// 1. Validate the protocol state to ensure it is incomplete.
    /// 2. Partition the evaluations into two halves based on the current variable.
    /// 3. Compute the contributions (`pq_zero`, `pq_one`, `pq_inf`) for each index in parallel.
    /// 4. Compress the computed polynomial into a univariate form.
    /// 5. Cache the result to optimize future rounds.
    ///
    /// ### Returns
    /// - A compressed polynomial representing the contribution of the current variable.
    pub fn compute_round_polynomial(&mut self) -> CompressedPoly {
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

        // Initialize a parallel iterator to compute contributions for each index in the first half.
        let iter = (0..half).into_par_iter();

        // Compute `pq_zero`, `pq_one`, and `pq_inf` for each index in parallel.
        let mut poly = iter
            .map(|i| {
                // Contribution for `X_i = 0`
                let mut pq_zero = BinaryField128b::zero();
                // Contribution for `X_i = 1`
                let mut pq_one = BinaryField128b::zero();
                // Contribution of sums across both halves
                let mut pq_inf = BinaryField128b::zero();

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
            .reduce(|| [BinaryField128b::zero(); 3], |[a, b, c], [d, e, f]| [a + d, b + e, c + f]);

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
}

#[cfg(test)]
mod tests {
    use super::*;
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
        ProdCheck::new(p_polys, q_polys, BinaryField128b::zero(), true);
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
        ProdCheck::new(p_polys, q_polys, BinaryField128b::zero(), true);
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
        let compressed_poly = prodcheck.compute_round_polynomial();

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
        prodcheck.compute_round_polynomial();
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
        prodcheck.compute_round_polynomial();
    }
}
