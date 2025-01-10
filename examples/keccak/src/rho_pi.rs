use crate::theta::idx;
use hashcaster_primitives::{binary_field::BinaryField128b, linear_trait::LinearOperations};

/// Rotation offsets for the Rho-Pi transformation in Keccak.
///
/// Each value specifies the number of bit positions to rotate a 64-bit lane.
pub const ROTATIONS: [[usize; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// Implements the combined Rho and Pi transformations for the Keccak state matrix.
///
/// It realizes B[y, 2x + 3y] = rot(A[x, y], r[x, y])
///
/// ## Overview
/// - **Rho**: Applies bitwise rotations to each 64-bit lane in the state matrix \( A[x, y] \).
/// - **Pi**: Rearranges the lanes in the state matrix by permuting the coordinates.
///
/// ## Algorithm
/// For each position \( (x, y) \) in the state matrix:
/// - Compute the rotation offset \( r[x, y] \).
/// - Rotate the lane \( A[x, y] \) by \( r[x, y] \) positions.
/// - Map the rotated value to the position \( B[y, 2x + 3y \mod 5] \).
///
/// ## Details
/// - The Rho step ensures intra-lane diffusion by rotating individual bits.
/// - The Pi step ensures inter-lane diffusion by rearranging lanes across the state matrix.
///
/// ## Context in Keccak
/// - The Rho-Pi transformations are applied in every Keccak round after the Theta step.
#[derive(Debug)]
pub struct RhoPi {}

impl LinearOperations for RhoPi {
    fn n_in(&self) -> usize {
        1600
    }

    fn n_out(&self) -> usize {
        1600
    }

    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        for x in 0..5 {
            for y in 0..5 {
                let rot = 64 - ROTATIONS[x][y];
                let id = (2 * x + 3 * y) % 5;
                for z in 0..64 {
                    output[idx(y, id, z)] += input[idx(x, y, (z + rot) % 64)];
                }
            }
        }
    }

    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        for x in 0..5 {
            for y in 0..5 {
                let rot = 64 - ROTATIONS[x][y];
                let id = (2 * x + 3 * y) % 5;
                for z in 0..64 {
                    output[idx(x, y, (z + rot) % 64)] += input[idx(y, id, z)];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_primitives::binary_field::BinaryField128b;

    #[test]
    fn test_rhopi_apply() {
        let rho_pi = RhoPi {};

        // Define input state with specific values for manual verification.
        let mut input = [BinaryField128b::from(0); 1600];
        // A -> x = 0, y = 0, z = 0
        input[idx(0, 0, 0)] = BinaryField128b::from(1);
        // A -> x = 1, y = 1, z = 1
        input[idx(1, 1, 21)] = BinaryField128b::from(2);
        // A -> x = 2, y = 2, z = 2
        input[idx(2, 2, 23)] = BinaryField128b::from(3);

        // Initialize the output array with zeros.
        let mut output = [BinaryField128b::from(0); 1600];

        // Apply the transformation.
        rho_pi.apply(&input, &mut output);

        // Manually compute expected outputs after applying Rho-Pi.
        assert_eq!(output[idx(0, 0, 0)], BinaryField128b::from(1));
        assert_eq!(output[idx(1, 0, 1)], BinaryField128b::from(2));
        assert_eq!(output[idx(2, 0, 2)], BinaryField128b::from(3));
    }

    #[test]
    fn test_rhopi_apply_transposed() {
        let rho_pi = RhoPi {};

        // Define input state with specific values for manual verification.
        let mut input = [BinaryField128b::from(0); 1600];
        input[idx(0, 0, 0)] = BinaryField128b::from(1);
        input[idx(1, 0, 1)] = BinaryField128b::from(2);
        input[idx(2, 0, 2)] = BinaryField128b::from(3);

        // Initialize the output array with zeros.
        let mut output = [BinaryField128b::from(0); 1600];

        // Apply the transposed transformation.
        rho_pi.apply_transposed(&input, &mut output);

        // Manually compute expected outputs after applying the transposed Rho-Pi.
        assert_eq!(output[idx(0, 0, 0)], BinaryField128b::from(1));
        assert_eq!(output[idx(1, 1, 21)], BinaryField128b::from(2));
        assert_eq!(output[idx(2, 2, 23)], BinaryField128b::from(3));
    }
}
