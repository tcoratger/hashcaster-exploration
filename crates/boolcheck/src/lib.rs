use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::{compressed::CompressedPoly, univariate::UnivariatePolynomials};

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
    eq_sequence: UnivariatePolynomials,
    round_polys: Vec<CompressedPoly>,
    claim: BinaryField128b,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::repeat_with;

    #[test]
    fn test_new_andcheck() {
        // Set the number of variables for the test.
        let num_vars = 20;

        // Generate a vector `points` of `num_vars` random field elements in `BinaryField128b`.
        // This represents a set of random variables that will be used in the test.
        let points: Vec<_> = repeat_with(BinaryField128b::random).take(num_vars).collect();

        // Generate a vector `p` with 2^num_vars random elements in `BinaryField128b`.
        // This represents one operand (a polynomial) in the AND operation.
        let p: Vec<_> = repeat_with(BinaryField128b::random).take(1 << num_vars).collect();

        // Generate another vector `q` with 2^num_vars random elements in `BinaryField128b`.
        // This represents the second operand (a polynomial) in the AND operation.
        let q: Vec<_> = repeat_with(BinaryField128b::random).take(1 << num_vars).collect();

        // Compute the element-wise AND operation between `p` and `q`.
        // The result is stored in `p_and_q`.
        let p_and_q: Vec<_> = p.iter().zip(q.iter()).map(|(x, y)| *x & *y).collect();
    }
}
