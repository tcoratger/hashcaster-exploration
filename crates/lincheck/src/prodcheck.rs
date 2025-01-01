use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::{
    compressed::CompressedPoly, multinear_lagrangian::MultilinearLagrangianPolynomial,
    point::Points,
};
use num_traits::identities::Zero;

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
}
