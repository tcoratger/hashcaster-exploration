use crate::prodcheck::ProdCheck;
use hashcaster_field::{binary_field::BinaryField128b, matrix_lin::MatrixLinear};
use hashcaster_poly::{
    multinear_lagrangian::MultilinearLagrangianPolynomial, point::Points,
    univariate::UnivariatePolynomial,
};
use num_traits::Zero;
use std::array;

/// A builder for constructing and managing the state of the LinCheck protocol.
///
/// # Generics
/// - `N`: Number of input polynomials.
/// - `M`: Number of output claims.
#[derive(Clone, Debug)]
pub struct LinCheckBuilder<const N: usize, const M: usize> {
    /// Linear transformation matrix applied to the polynomial coefficients.
    matrix: MatrixLinear,
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

impl<const N: usize, const M: usize> Default for LinCheckBuilder<N, M> {
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

impl<const N: usize, const M: usize> LinCheckBuilder<N, M> {
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
        matrix: MatrixLinear,
        num_active_vars: usize,
        initial_claims: [BinaryField128b; M],
    ) -> Self {
        // Assert matrix dimensions match expectations.
        assert!(matrix.n_in() == N * (1 << num_active_vars), "Invalid matrix dimensions");
        assert!(matrix.n_out() == M * (1 << num_active_vars), "Invalid matrix dimensions");

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
    pub fn build(self, gamma: BinaryField128b) -> ProdCheck<N> {
        // Compute chunk size based on active variables.
        // Each polynomial is divided into chunks of size `2^num_active_vars`.
        let chunk_size = 1 << self.num_active_vars;

        // Split evaluation points into active and dormant sets.
        // - Active variables (`x_0, ..., x_(a-1)`) directly affect chunk-wise computation.
        // - Dormant variables (`x_a, ..., x_(n-1)`) are used to restrict the polynomial.
        let (pt_active, pt_dormant) = self.points.split_at(self.num_active_vars);

        // Generate equality polynomial for dormant variables.
        let eq_dormant = MultilinearLagrangianPolynomial::new_eq_poly(&pt_dormant.into());

        // Initialize restricted polynomials with zero coefficients.
        // Initializes `N` restricted polynomials, each with `2^num_active_vars` coefficients.
        let mut p_polys: [_; N] = array::from_fn(|_| {
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::zero(); chunk_size])
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
        let gammas = BinaryField128b::compute_gammas_folding::<M>(gamma);

        // Generate equality polynomial for active variables.
        let eq_active = MultilinearLagrangianPolynomial::new_eq_poly(&pt_active.into());

        // Combine gamma powers and active equality polynomial.
        //
        // `gamma_eqs[i] = γ^j * eq_active[i]`, where `j` is derived from `i`.
        let gamma_eqs: Vec<_> =
            gammas.iter().flat_map(|gpow| (0..chunk_size).map(|i| *gpow * eq_active[i])).collect();

        // Apply the transposed matrix operation to compute `q`.
        // `q = Mᵀ * gamma_eqs`.
        let mut q =
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::zero(); N * chunk_size]);
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
        let claim = UnivariatePolynomial::new(self.initial_claims.to_vec()).evaluate_at(&gamma);

        // Return a `ProdCheck` object with the computed data.
        ProdCheck::new(p_polys, q_polys, claim, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_lincheck() {
        // Create a default LinCheckBuilder instance
        let lincheck: LinCheckBuilder<2, 2> = LinCheckBuilder::default();

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
}
