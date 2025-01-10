use hashcaster_poly::{
    multinear_lagrangian::MultilinearLagrangianPolynomials,
    point::{Point, Points},
};
use hashcaster_primitives::binary_field::BinaryField128b;

pub mod linear;
pub mod matrix;
pub mod rho_pi;
pub mod theta;

pub fn main() {
    // Setup the number of variables
    const NUM_VARS: usize = 20;

    // Setup the phase switch parameter
    const PHASE_SWITCH: usize = 5;

    // Setup NUM_VARS random points
    let points: Points =
        (0..NUM_VARS).map(|_| Point(BinaryField128b::random())).collect::<Vec<_>>().into();

    // Create 5 multilinear lagrangian polynomials with `2^NUM_VARS` coefficients each
    let mut polys: MultilinearLagrangianPolynomials = Default::default();
    for _ in 0..5 {
        polys
            .push((0..1 << NUM_VARS).map(|_| BinaryField128b::random()).collect::<Vec<_>>().into());
    }

    println!("Hello, world!");
}
