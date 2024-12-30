use hashcaster_field::{binary_field::BinaryField128b, matrix_lin::MatrixLinear};
use hashcaster_poly::{multinear_lagrangian::MultilinearLagrangianPolynomial, point::Points};

#[derive(Clone, Debug)]
pub struct LinCheck<const N: usize, const M: usize> {
    matrix: MatrixLinear,
    polys: [MultilinearLagrangianPolynomial; N],
    points: Points,
    num_vars: usize,
    num_active_vars: usize,
    initial_claims: [BinaryField128b; M],
}

impl<const N: usize, const M: usize> Default for LinCheck<N, M> {
    fn default() -> Self {
        Self {
            matrix: Default::default(),
            polys: core::array::from_fn(|_| MultilinearLagrangianPolynomial::default()),
            points: Default::default(),
            num_vars: 0,
            num_active_vars: 0,
            initial_claims: core::array::from_fn(|_| Default::default()),
        }
    }
}

impl<const N: usize, const M: usize> LinCheck<N, M> {
    pub fn new(
        polys: [MultilinearLagrangianPolynomial; N],
        points: Points,
        matrix: MatrixLinear,
        num_active_vars: usize,
        initial_claims: [BinaryField128b; M],
    ) -> Self {
        // Ensure the matrix has the correct input and output sizes.
        assert!(matrix.n_in() == N * (1 << num_active_vars), "Invalid matrix dimensions");
        assert!(matrix.n_out() == M * (1 << num_active_vars), "Invalid matrix dimensions");

        // Ensure the number of variables is at least the number of active variables.
        let num_vars = points.len();
        assert!(num_vars >= num_active_vars);

        // Ensure that each polynomial has the correct number of coefficients.
        polys.iter().for_each(|poly| {
            assert!(poly.len() == 1 << num_vars);
        });

        // Initialize the linear check with the provided parameters.
        Self { matrix, polys, points, num_vars, num_active_vars, initial_claims }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_lincheck() {
        // Create a default LinCheck instance
        let lincheck: LinCheck<2, 2> = LinCheck::default();

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

        // Create a new LinCheck instance
        let lincheck = LinCheck::new(polys, points.clone(), matrix.clone(), 2, initial_claims);

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
        LinCheck::new(polys, points, matrix, 2, initial_claims);
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
        LinCheck::new(polys, points, matrix, 3, initial_claims);
    }
}
