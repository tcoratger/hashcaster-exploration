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

        // Ensure both `P` and `Q` have the same number of polynomials.
        assert_eq!(p_polys.len(), q_polys.len());

        // Validate that each polynomial in `P` and `Q` has the expected size.
        assert!(p_polys
            .iter()
            .zip(q_polys.iter())
            .all(|(p, q)| p.len() == 1 << num_vars && q.len() == 1 << num_vars));

        // Total number of polynomials in `P` and `Q`.
        let poly_len = p_polys.len();

        // Optionally verify the correctness of the initial claim.
        if check_init_claim {
            // Start with zero.
            let mut expected_claim = BinaryField128b::zero();

            // Compute the sum of products of `P` and `Q` polynomials.
            for i in 0..poly_len {
                for j in 0..1 << num_vars {
                    expected_claim += p_polys[i][j] * q_polys[i][j];
                }
            }

            // Ensure the provided claim matches the computed claim.
            assert_eq!(claim, expected_claim);
        }

        // Return the new `Prodcheck` instance.
        Self { p_polys, q_polys, claim, num_vars, ..Default::default() }
    }
}
