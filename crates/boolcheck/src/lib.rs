use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::{
    compressed::CompressedPoly, multinear_lagrangian::MultilinearLagrangianPolynomials,
};

pub mod bool_trait;
pub mod builder;
pub mod package;

#[derive(Clone, Debug, Default)]
pub struct BoolCheck {
    pt: Vec<BinaryField128b>,
    poly: Vec<BinaryField128b>,
    polys: Vec<Vec<BinaryField128b>>,
    ext: Option<Vec<BinaryField128b>>,
    poly_coords: Option<Vec<BinaryField128b>>,
    c: usize,
    challenges: Vec<BinaryField128b>,
    bit_mapping: Vec<u16>,
    eq_sequence: MultilinearLagrangianPolynomials,
    round_polys: Vec<CompressedPoly>,
    claim: BinaryField128b,
}

#[cfg(test)]
mod tests {
    use builder::BoolCheckBuilder;
    use hashcaster_poly::multinear_lagrangian::MultilinearLagrangianPolynomial;
    use package::BooleanPackage;

    use super::*;
    use std::iter::repeat_with;

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
