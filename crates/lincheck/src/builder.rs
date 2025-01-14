use crate::prodcheck::ProdCheck;
use hashcaster_primitives::{
    binary_field::BinaryField128b,
    linear_trait::LinearOperations,
    poly::{
        multinear_lagrangian::MultilinearLagrangianPolynomial,
        point::{Point, Points},
        univariate::UnivariatePolynomial,
    },
};
use std::array;

/// A builder for constructing and managing the state of the LinCheck protocol.
///
/// # Generics
/// - `N`: Number of input polynomials.
/// - `M`: Number of output claims.
#[derive(Clone, Debug)]
pub struct LinCheckBuilder<const N: usize, const M: usize, L: LinearOperations> {
    /// Linear transformation matrix applied to the polynomial coefficients.
    matrix: L,
    /// Input polynomials to be evaluated.
    polys: [MultilinearLagrangianPolynomial; N],
    /// Points for polynomial evaluation.
    points: Points,
    /// Total number of variables in the polynomials.
    num_vars: usize,
    /// Number of "active" variables affecting chunk sizes.
    num_active_vars: usize,
    /// Initial claims (evaluations) for the polynomials at specified points.
    initial_claims: [BinaryField128b; M],
}

impl<const N: usize, const M: usize, L: LinearOperations + Default> Default
    for LinCheckBuilder<N, M, L>
{
    fn default() -> Self {
        Self {
            matrix: Default::default(),
            polys: core::array::from_fn(|_| Default::default()),
            points: Default::default(),
            num_vars: 0,
            num_active_vars: 0,
            initial_claims: core::array::from_fn(|_| Default::default()),
        }
    }
}

impl<const N: usize, const M: usize, L: LinearOperations> LinCheckBuilder<N, M, L> {
    /// Constructs a new `LinCheckBuilder` instance.
    ///
    /// # Parameters
    /// - `polys`: Array of `N` polynomials.
    /// - `points`: Evaluation points.
    /// - `matrix`: Linear transformation matrix.
    /// - `num_active_vars`: Number of active variables.
    /// - `initial_claims`: Initial evaluations of the polynomials.
    ///
    /// # Panics
    /// - If matrix dimensions do not match the expected sizes.
    /// - If the number of variables does not match the polynomial sizes.
    pub fn new(
        polys: [MultilinearLagrangianPolynomial; N],
        points: Points,
        matrix: L,
        num_active_vars: usize,
        initial_claims: [BinaryField128b; M],
    ) -> Self {
        // Assert matrix dimensions match expectations.
        assert_eq!(matrix.n_in(), N * (1 << num_active_vars), "Invalid matrix dimensions");
        assert_eq!(matrix.n_out(), M * (1 << num_active_vars), "Invalid matrix dimensions");

        // Determine total number of variables and validate.
        let num_vars = points.len();
        assert!(num_vars >= num_active_vars, "Number of variables must be >= active variables");

        // Validate polynomial sizes match the number of variables.
        for poly in &polys {
            assert!(poly.len() == 1 << num_vars, "Polynomial size mismatch");
        }

        // Return the initialized builder.
        Self { matrix, polys, points, num_vars, num_active_vars, initial_claims }
    }

    /// Constructs the folding challenge protocol for LinCheck.
    ///
    /// ## Description
    /// This function sets up the LinCheck protocol by:
    /// - Restricting the input polynomials based on "dormant" variables.
    /// - Combining the equality polynomial for "active" variables with gamma powers.
    /// - Applying a matrix transposition to compute folded results.
    ///
    /// ## Parameters
    /// - `gamma`: Folding challenge scalar.
    ///
    /// ## Returns
    /// A `ProdCheck` object containing the folded polynomials and evaluations.
    pub fn build(self, gamma: &Point) -> ProdCheck<N> {
        // Compute chunk size based on active variables.
        // Each polynomial is divided into chunks of size `2^num_active_vars`.
        let chunk_size = 1 << self.num_active_vars;

        // Split evaluation points into active and dormant sets.
        // - Active variables (`x_0, ..., x_(a-1)`) directly affect chunk-wise computation.
        // - Dormant variables (`x_a, ..., x_(n-1)`) are used to restrict the polynomial.
        let (pt_active, pt_dormant) = self.points.split_at(self.num_active_vars);
        let pt_active: Points = pt_active.into();
        let pt_dormant: Points = pt_dormant.into();

        // Generate equality polynomial for dormant variables.
        let eq_dormant = pt_dormant.to_eq_poly();

        // Initialize restricted polynomials with zero coefficients.
        // Initializes `N` restricted polynomials, each with `2^num_active_vars` coefficients.
        let mut p_polys: [_; N] = array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::ZERO; chunk_size])
        });

        // Restrict input polynomials using dormant equality polynomial.
        //
        // For each polynomial `P(x)`:
        // `P'(x) = Σ(eq_dormant[j] * P_chunk[j])` for chunks `j`.
        self.polys.into_iter().enumerate().for_each(|(i, poly)| {
            poly.chunks(chunk_size).enumerate().for_each(|(j, chunk)| {
                p_polys[i].iter_mut().zip(chunk).for_each(|(p, &c)| *p += eq_dormant[j] * c);
            });
        });

        // Compute powers of gamma for folding.
        //
        // Computes successive powers `[1, γ, γ², ..., γ^(M-1)]`.
        let gammas = BinaryField128b::compute_gammas_folding::<M>(**gamma);

        // Generate equality polynomial for active variables.
        let eq_active = pt_active.to_eq_poly();

        // Combine gamma powers and active equality polynomial.
        //
        // `gamma_eqs[i] = γ^j * eq_active[i]`, where `j` is derived from `i`.
        let gamma_eqs: Vec<_> =
            gammas.iter().flat_map(|gpow| (0..chunk_size).map(|i| *gpow * eq_active[i])).collect();

        // Apply the transposed matrix operation to compute `q`.
        // `q = Mᵀ * gamma_eqs`.
        let mut q =
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::ZERO; N * chunk_size]);
        self.matrix.apply_transposed(&gamma_eqs, &mut q);

        // Split `q` into separate polynomials.
        //
        // Splits `q` into `N` polynomials `Q_1, ..., Q_N`, each of size `2^num_active_vars`.
        let q_polys: [_; N] = core::array::from_fn(|_| {
            let tmp = q.split_off(chunk_size).into();
            std::mem::replace(&mut q, tmp)
        });

        // Verify all chunks were processed.
        assert_eq!(q.len(), 0, "Unprocessed data remains in `q`");

        // Evaluate the initial claims using a univariate polynomial.
        //
        // `claim(γ) = Σ(initial_claims[i] * γ^i)` for `i = 0, ..., M-1`.
        let claim = UnivariatePolynomial::new(self.initial_claims.to_vec()).evaluate_at(gamma);

        // Return a `ProdCheck` object with the computed data.
        ProdCheck::new(p_polys, q_polys, claim, false)
    }
}

#[cfg(test)]
mod tests {
    use hashcaster_primitives::matrix_lin::MatrixLinear;

    use super::*;

    #[test]
    fn test_default_lincheck() {
        // Create a default LinCheckBuilder instance
        let lincheck: LinCheckBuilder<2, 2, MatrixLinear> = LinCheckBuilder::default();

        // Verify default values
        assert_eq!(lincheck.num_vars, 0);
        assert_eq!(lincheck.num_active_vars, 0);
        assert_eq!(lincheck.matrix.n_in(), 0);
        assert_eq!(lincheck.matrix.n_out(), 0);
        assert!(lincheck.initial_claims.iter().all(|&claim| claim == BinaryField128b::default()));
    }

    #[test]
    fn test_new_lincheck() {
        // Use `Points` with enough length to satisfy `num_vars >= num_active_vars`
        // `num_vars` = 3
        let points = Points::from(vec![BinaryField128b::default(); 3]);

        // Polynomials must have a length of `1 << num_vars`
        let polys: [MultilinearLagrangianPolynomial; 1] = core::array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::default(); 1 << 3])
        });

        // Matrix must match the input and output size based on `num_active_vars`
        let matrix = MatrixLinear::new(4, 4, vec![BinaryField128b::default(); 16]);

        // Initial claims size must match M
        let initial_claims: [BinaryField128b; 1] =
            core::array::from_fn(|_| BinaryField128b::default());

        // Create a new LinCheckBuilder instance
        let lincheck =
            LinCheckBuilder::new(polys, points.clone(), matrix.clone(), 2, initial_claims);

        // Verify the parameters
        // Ensure `num_vars` matches points length
        assert_eq!(lincheck.num_vars, points.len());
        // Ensure `num_active_vars` is correctly set
        assert_eq!(lincheck.num_active_vars, 2);
        // Ensure matrix input size matches expectations
        assert_eq!(lincheck.matrix.n_in(), 4);
        // Ensure matrix output size matches expectations
        assert_eq!(lincheck.matrix.n_out(), 4);
        // Ensure points are correctly stored
        assert_eq!(lincheck.points, points);
        // Ensure matrix is correctly stored
        assert_eq!(lincheck.matrix, matrix);
    }

    #[test]
    #[should_panic(expected = "Invalid matrix dimensions")]
    fn test_invalid_matrix_dimensions() {
        // Create an invalid matrix with incorrect dimensions
        let matrix = MatrixLinear::new(3, 4, vec![BinaryField128b::default(); 16]);
        let polys: [MultilinearLagrangianPolynomial; 1] =
            core::array::from_fn(|_| MultilinearLagrangianPolynomial::default());
        let points = Points::default();
        let initial_claims: [BinaryField128b; 1] =
            core::array::from_fn(|_| BinaryField128b::default());

        // This should panic due to matrix dimension mismatch
        LinCheckBuilder::new(polys, points, matrix, 2, initial_claims);
    }

    #[test]
    #[should_panic(expected = "Invalid matrix dimensions")]
    fn test_invalid_polynomial_length() {
        let polys: [MultilinearLagrangianPolynomial; 1] =
            core::array::from_fn(|_| MultilinearLagrangianPolynomial::default());
        let points = Points::default();
        let matrix = MatrixLinear::new(4, 4, vec![BinaryField128b::default(); 16]);
        let initial_claims: [BinaryField128b; 1] =
            core::array::from_fn(|_| BinaryField128b::default());

        // Force incorrect polynomial length by mismatching points and variables
        LinCheckBuilder::new(polys, points, matrix, 3, initial_claims);
    }

    #[test]
    fn test_lincheck_builder_new_with_valid_inputs() {
        // Number of input polynomials
        const N: usize = 2;
        // Number of output claims
        const M: usize = 2;
        // Total variables
        const NUM_VARS: usize = 3;
        // Active variables
        const NUM_ACTIVE_VARS: usize = 2;

        // Create valid inputs
        let polys: [MultilinearLagrangianPolynomial; N] = core::array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::from(1); 1 << NUM_VARS])
        });
        let points = Points::from(vec![BinaryField128b::from(2); NUM_VARS]);
        let matrix = MatrixLinear::new(
            N * (1 << NUM_ACTIVE_VARS),
            M * (1 << NUM_ACTIVE_VARS),
            vec![BinaryField128b::from(1); N * M * (1 << (NUM_ACTIVE_VARS * 2))],
        );
        let initial_claims: [BinaryField128b; M] =
            [BinaryField128b::from(4), BinaryField128b::from(5)];

        // Construct LinCheckBuilder
        let lincheck = LinCheckBuilder::new(
            polys,
            points.clone(),
            matrix.clone(),
            NUM_ACTIVE_VARS,
            initial_claims,
        );

        // Validate the LinCheckBuilder state
        assert_eq!(lincheck.num_vars, points.len());
        assert_eq!(lincheck.num_active_vars, NUM_ACTIVE_VARS);
        assert_eq!(lincheck.matrix, matrix);
        assert_eq!(lincheck.points, points);
        assert_eq!(lincheck.initial_claims, initial_claims);
    }

    #[test]
    #[should_panic(expected = "Invalid matrix dimensions")]
    fn test_lincheck_builder_new_with_invalid_matrix_input_size() {
        const N: usize = 2;
        const M: usize = 2;
        const NUM_VARS: usize = 3;
        const NUM_ACTIVE_VARS: usize = 2;

        // Create invalid matrix (wrong input size)
        let matrix = MatrixLinear::new(
            (N - 1) * (1 << NUM_ACTIVE_VARS), // Incorrect input size
            M * (1 << NUM_ACTIVE_VARS),
            vec![BinaryField128b::from(1); (N - 1) * M * (1 << (NUM_ACTIVE_VARS * 2))],
        );

        // Other valid inputs
        let polys: [MultilinearLagrangianPolynomial; N] = core::array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::from(1); 1 << NUM_VARS])
        });
        let points = Points::from(vec![BinaryField128b::from(2); NUM_VARS]);
        let initial_claims: [BinaryField128b; M] =
            [BinaryField128b::from(4), BinaryField128b::from(5)];

        // Attempt construction (should panic)
        LinCheckBuilder::new(polys, points, matrix, NUM_ACTIVE_VARS, initial_claims);
    }

    #[test]
    #[should_panic(expected = "Invalid matrix dimensions")]
    fn test_lincheck_builder_new_with_invalid_matrix_output_size() {
        const N: usize = 2;
        const M: usize = 2;
        const NUM_VARS: usize = 3;
        const NUM_ACTIVE_VARS: usize = 2;

        // Create invalid matrix (wrong output size)
        let matrix = MatrixLinear::new(
            N * (1 << NUM_ACTIVE_VARS),
            (M + 1) * (1 << NUM_ACTIVE_VARS), // Incorrect output size
            vec![BinaryField128b::from(1); N * (M + 1) * (1 << (NUM_ACTIVE_VARS * 2))],
        );

        // Other valid inputs
        let polys: [MultilinearLagrangianPolynomial; N] = core::array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::from(1); 1 << NUM_VARS])
        });
        let points = Points::from(vec![BinaryField128b::from(2); NUM_VARS]);
        let initial_claims: [BinaryField128b; M] =
            [BinaryField128b::from(4), BinaryField128b::from(5)];

        // Attempt construction (should panic)
        LinCheckBuilder::new(polys, points, matrix, NUM_ACTIVE_VARS, initial_claims);
    }

    #[test]
    #[should_panic(expected = "Number of variables must be >= active variables")]
    fn test_lincheck_builder_new_with_insufficient_points() {
        const N: usize = 2;
        const M: usize = 2;
        const NUM_VARS: usize = 2; // Less than NUM_ACTIVE_VARS
        const NUM_ACTIVE_VARS: usize = 3; // Active variables exceed total variables

        // Valid matrix and other inputs
        let polys: [MultilinearLagrangianPolynomial; N] = core::array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::from(1); 1 << NUM_VARS])
        });
        let points = Points::from(vec![BinaryField128b::from(2); NUM_VARS]);
        let matrix = MatrixLinear::new(
            N * (1 << NUM_ACTIVE_VARS),
            M * (1 << NUM_ACTIVE_VARS),
            vec![BinaryField128b::from(1); N * M * (1 << (NUM_ACTIVE_VARS * 2))],
        );
        let initial_claims: [BinaryField128b; M] =
            [BinaryField128b::from(4), BinaryField128b::from(5)];

        // Attempt construction (should panic)
        LinCheckBuilder::new(polys, points, matrix, NUM_ACTIVE_VARS, initial_claims);
    }

    #[test]
    #[should_panic(expected = "Polynomial size mismatch")]
    fn test_lincheck_builder_new_with_invalid_polynomial_length() {
        const N: usize = 2;
        const M: usize = 2;
        const NUM_VARS: usize = 3; // Total variables
        const NUM_ACTIVE_VARS: usize = 2;

        // Invalid polynomials (wrong length)
        let polys: [MultilinearLagrangianPolynomial; N] = core::array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![
                BinaryField128b::from(1);
                (1 << NUM_VARS) - 1
            ]) // Incorrect size
        });

        // Valid points, matrix, and initial claims
        let points = Points::from(vec![BinaryField128b::from(2); NUM_VARS]);
        let matrix = MatrixLinear::new(
            N * (1 << NUM_ACTIVE_VARS),
            M * (1 << NUM_ACTIVE_VARS),
            vec![BinaryField128b::from(1); N * M * (1 << (NUM_ACTIVE_VARS * 2))],
        );
        let initial_claims: [BinaryField128b; M] =
            [BinaryField128b::from(4), BinaryField128b::from(5)];

        // Attempt construction (should panic)
        LinCheckBuilder::new(polys, points, matrix, NUM_ACTIVE_VARS, initial_claims);
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_lincheck_builder_build() {
        // Number of input polynomials
        const N: usize = 2;
        // Number of output claims
        const M: usize = 2;
        // Total variables
        const NUM_VARS: usize = 3;
        // Active variables
        const NUM_ACTIVE_VARS: usize = 2;

        // Create valid inputs
        let polys: [MultilinearLagrangianPolynomial; N] = [
            vec![BinaryField128b::from(5678); 1 << NUM_VARS].into(),
            vec![BinaryField128b::from(910); 1 << NUM_VARS].into(),
        ];

        let points = Points::from(vec![BinaryField128b::from(2); NUM_VARS]);
        let matrix = MatrixLinear::new(
            N * (1 << NUM_ACTIVE_VARS),
            M * (1 << NUM_ACTIVE_VARS),
            vec![BinaryField128b::from(18); N * M * (1 << (NUM_ACTIVE_VARS * 2))],
        );
        let initial_claims: [BinaryField128b; M] =
            [BinaryField128b::from(4), BinaryField128b::from(5)];

        // Construct LinCheckBuilder
        let lincheck = LinCheckBuilder::new(polys, points, matrix, NUM_ACTIVE_VARS, initial_claims);

        // Build the LinCheck prover
        let lincheck_prover = lincheck.build(&Point(BinaryField128b::from(1234)));

        // Expected ProdCheck prover
        let expected_prover = ProdCheck {
            p_polys: [
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(5678),
                    BinaryField128b::from(5678),
                    BinaryField128b::from(5678),
                    BinaryField128b::from(5678),
                ]),
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(910),
                    BinaryField128b::from(910),
                    BinaryField128b::from(910),
                    BinaryField128b::from(910),
                ]),
            ],
            q_polys: [
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(18692268690725379462709786785192376188),
                    BinaryField128b::from(18692268690725379462709786785192376188),
                    BinaryField128b::from(18692268690725379462709786785192376188),
                    BinaryField128b::from(18692268690725379462709786785192376188),
                ]),
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(18692268690725379462709786785192376188),
                    BinaryField128b::from(18692268690725379462709786785192376188),
                    BinaryField128b::from(18692268690725379462709786785192376188),
                    BinaryField128b::from(18692268690725379462709786785192376188),
                ]),
            ],
            claim: BinaryField128b::from(258410230055561301416705741312625744282),
            challenges: Points::default(),
            num_vars: 2,
            cached_round_msg: None,
        };

        // Validate the LinCheck prover
        assert_eq!(lincheck_prover, expected_prover);
    }
}
