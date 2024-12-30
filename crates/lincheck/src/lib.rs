use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::{multinear_lagrangian::MultilinearLagrangianPolynomial, point::Points};

#[derive(Clone, Debug)]
pub struct LinCheck<const N: usize, const M: usize> {
    polys: [MultilinearLagrangianPolynomial; N],
    points: Points,
    num_vars: usize,
    num_active_vars: usize,
    initial_claims: [BinaryField128b; M],
}

impl<const N: usize, const M: usize> Default for LinCheck<N, M> {
    fn default() -> Self {
        Self {
            polys: core::array::from_fn(|_| MultilinearLagrangianPolynomial::default()),
            points: Default::default(),
            num_vars: 1,
            num_active_vars: 1,
            initial_claims: core::array::from_fn(|_| Default::default()),
        }
    }
}

impl<const N: usize, const M: usize> LinCheck<N, M> {
    // pub fn new(polys:[MultilinearLagrangianPolynomial; N], points: Points, matrix)
}
