use crate::matrix::{composition::CombinedMatrix, identity::IdentityMatrix, sum::SumMatrix};
use hashcaster_primitives::{binary_field::BinaryField128b, linear_trait::LinearOperations};
use std::ops::Deref;
use theta_ac::ThetaAC;
use theta_cd::ThetaCD;
use theta_de::ThetaDE;

pub mod theta_ac;
pub mod theta_cd;
pub mod theta_de;

/// Utility function for calculating the linear index in a flattened 3D array representation of the
/// Keccak state.
///
/// # Arguments
/// - `x`: Row index in the 5x5 grid.
/// - `y`: Column index in the 5x5 grid.
/// - `z`: Offset within the 64-bit lane.
///
/// # Returns
/// A single usize index representing the position in the flattened 1D array.
pub(crate) const fn idx(x: usize, y: usize, z: usize) -> usize {
    x * 320 + y * 64 + z
}

#[derive(Debug)]
pub struct Theta(
    SumMatrix<IdentityMatrix, CombinedMatrix<ThetaDE, CombinedMatrix<ThetaCD, ThetaAC>>>,
);

impl Default for Theta {
    fn default() -> Self {
        Self::new()
    }
}

impl Theta {
    pub fn new() -> Self {
        Self(SumMatrix::new(
            IdentityMatrix::new(1600),
            CombinedMatrix::new(ThetaDE, CombinedMatrix::new(ThetaCD, ThetaAC)),
        ))
    }
}

impl LinearOperations for Theta {
    fn n_in(&self) -> usize {
        self.0.n_in()
    }

    fn n_out(&self) -> usize {
        self.0.n_out()
    }

    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        self.0.apply(input, output);
    }

    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        self.0.apply_transposed(input, output);
    }
}

impl Deref for Theta {
    type Target =
        SumMatrix<IdentityMatrix, CombinedMatrix<ThetaDE, CombinedMatrix<ThetaCD, ThetaAC>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_primitives::binary_field::BinaryField128b;

    #[test]
    fn test_theta_apply_and_transposed() {
        // **Initialize the Theta operator**
        // - `Theta` is a composition of multiple linear transformations, including `ThetaDE`,
        //   `ThetaCD`, and `ThetaAC`.
        let theta = Theta::new();

        // **Validate dimensions**
        // - Ensure that the input and output dimensions of the `Theta` operator match expectations.
        assert_eq!(theta.n_in(), 1600, "Input size mismatch for Theta operator");
        assert_eq!(theta.n_out(), 1600, "Output size mismatch for Theta operator");

        // **Generate random input for `apply`**
        // - This represents the input vector for the forward transformation.
        let input_apply = BinaryField128b::random_vec(1600);

        // **Prepare output storage for `apply`**
        // - This will store the result of the forward transformation.
        let mut output_apply = vec![BinaryField128b::from(0); 1600];

        // **Step 5: Apply the Theta operator**
        // - Compute the forward transformation for the input.
        theta.apply(&input_apply, &mut output_apply);

        // **Generate random input for `apply_transposed`**
        // - This represents the input vector for the transposed transformation.
        let input_transposed = BinaryField128b::random_vec(1600);

        // **Prepare output storage for `apply_transposed`**
        // - This will store the result of the transposed transformation.
        let mut output_transposed = vec![BinaryField128b::from(0); 1600];

        // **Apply the transposed Theta operator**
        // - Compute the transposed transformation for the input.
        theta.apply_transposed(&input_transposed, &mut output_transposed);

        // **Compute dot product of `apply` output and `apply_transposed` input**
        // - This computes the left-hand side (lhs) of the transpose relationship: \( lhs = \sum
        //   (Theta(x) \cdot y) \)
        let lhs = output_apply
            .iter()
            .zip(input_transposed.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * *b));

        // **Compute dot product of `apply_transposed` output and `apply` input**
        // - This computes the right-hand side (rhs) of the transpose relationship: \( rhs = \sum (x
        //   \cdot Theta^T(y)) \)
        let rhs = output_transposed
            .iter()
            .zip(input_apply.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * *b));

        // **Validate the transpose property**
        // - For a valid linear operator \( T \): \( \langle T(x), y \rangle = \langle x, T^T(y)
        //   \rangle \)
        // - This ensures that the `apply` and `apply_transposed` methods satisfy the transpose
        //   relationship.
        assert_eq!(lhs, rhs, "Dot product property violated for apply and apply_transposed");
    }
}
