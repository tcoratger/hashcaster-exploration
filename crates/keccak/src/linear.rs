use crate::{matrix::composition::CombinedMatrix, rho_pi::RhoPi, theta::Theta};
use hashcaster_primitives::{
    binary_field::BinaryField128b, linear_trait::LinearOperations,
    poly::multinear_lagrangian::MultilinearLagrangianPolynomial,
};
use std::array;

/// A linear operator combining the RhoPi and Theta transformations for the Keccak permutation.
///
/// ## Overview
/// - `KeccakLinearUnbatched` encapsulates the RhoPi and Theta transformations as a single combined
///   linear operator.
/// - These transformations are core components of the Keccak permutation, responsible for ensuring
///   state mixing and diffusion.
///
/// ## Structure
/// - The `CombinedMatrix` struct combines the RhoPi transformation, which rearranges and rotates
///   state elements, with the Theta transformation, which applies column-wise parity adjustments.
#[derive(Debug)]
pub struct KeccakLinearUnbatched(CombinedMatrix<RhoPi, Theta>);

impl Default for KeccakLinearUnbatched {
    fn default() -> Self {
        Self::new()
    }
}

impl KeccakLinearUnbatched {
    /// Creates a new `KeccakLinearUnbatched` operator by combining RhoPi and Theta transformations.
    ///
    /// ## Returns
    /// - A new `KeccakLinearUnbatched` instance.
    pub fn new() -> Self {
        Self(CombinedMatrix::new(RhoPi {}, Theta::new()))
    }
}

impl LinearOperations for KeccakLinearUnbatched {
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

/// A batched implementation of the Keccak linear operator.
///
/// ## Overview
/// - `KeccakLinear` extends `KeccakLinearUnbatched` by introducing a structured mapping between
///   flattened input/output states and internal intermediate states.
///
/// ## Features
/// - Handles the flattened 5x1024 Keccak state with efficient transformations.
/// - Provides utility functions for creating, applying, and validating transformations.
#[derive(Debug)]
pub struct KeccakLinear(KeccakLinearUnbatched);

impl Default for KeccakLinear {
    fn default() -> Self {
        Self::new()
    }
}

impl KeccakLinear {
    /// Creates a new instance of `KeccakLinear` using the combined RhoPi and Theta transformations.
    ///
    /// ## Returns
    /// - A new `KeccakLinear` instance.
    pub fn new() -> Self {
        Self(KeccakLinearUnbatched::new())
    }
}

impl LinearOperations for KeccakLinear {
    /// Returns the number of input elements required by the transformation.
    ///
    /// ## Details
    /// - The input represents the flattened Keccak state (5 blocks of 1024 elements each).
    fn n_in(&self) -> usize {
        5 * 1024
    }

    /// Returns the number of output elements produced by the transformation.
    ///
    /// ## Details
    /// - The output has the same structure as the input.
    fn n_out(&self) -> usize {
        5 * 1024
    }

    /// Applies the combined RhoPi and Theta transformations to the input state.
    ///
    /// ## Parameters
    /// - `input`: A slice of size `n_in()` representing the input state.
    /// - `output`: A mutable slice of size `n_out()` to store the transformed state.
    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        // Create a 3x1600 intermediate state representation.
        let mut state = [[BinaryField128b::ZERO; 1600]; 3];

        // Split the input into blocks and map them to the intermediate state.
        for i in 0..5 {
            let input_ptr = &input[i * 1024..];
            for j in 0..3 {
                // Copy 320 elements for each sub-block from the input to the intermediate state.
                state[j][i * 320..(i + 1) * 320]
                    .copy_from_slice(&input_ptr[j * 320..(j + 1) * 320]);
            }
        }

        // Create a 3x1600 intermediate output state.
        let mut output_state = [[BinaryField128b::ZERO; 1600]; 3];

        // Apply the combined RhoPi and Theta transformations to each sub-state.
        state.iter().zip(output_state.iter_mut()).for_each(|(s, o)| self.0.apply(s, o));

        // Map the transformed intermediate output back to the flattened output vector.
        for i in 0..5 {
            let output_ptr = &mut output[i * 1024..];
            for j in 0..3 {
                // Copy 320 elements from each sub-block of the output state back to the flattened
                // output.
                output_ptr[j * 320..(j + 1) * 320]
                    .copy_from_slice(&output_state[j][i * 320..(i + 1) * 320]);
            }
        }
    }

    /// Applies the inverse of the combined RhoPi and Theta transformations to the input state.
    ///
    /// ## Parameters
    /// - `input`: A slice of size `n_in()` representing the input state.
    /// - `output`: A mutable slice of size `n_out()` to store the transformed state.
    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        // Create a 3x1600 intermediate state representation.
        let mut state = [[BinaryField128b::ZERO; 1600]; 3];

        // Split the input into blocks and map them to the intermediate state.
        for i in 0..5 {
            for j in 0..3 {
                // Copy 320 elements for each sub-block from the input to the intermediate state.
                state[j][i * 320..(i + 1) * 320]
                    .copy_from_slice(&input[i * 1024 + j * 320..i * 1024 + (j + 1) * 320]);
            }
        }

        // Create a 3x1600 intermediate output state.
        let mut output_state = [[BinaryField128b::ZERO; 1600]; 3];

        // Apply the inverse transformations to each sub-state.
        state.iter().zip(output_state.iter_mut()).for_each(|(s, o)| self.0.apply_transposed(s, o));

        // Map the transformed intermediate output back to the flattened output vector.
        for i in 0..5 {
            for j in 0..3 {
                // Copy 320 elements from each sub-block of the output state back to the flattened
                // output.
                output[i * 1024 + j * 320..i * 1024 + (j + 1) * 320]
                    .copy_from_slice(&output_state[j][i * 320..(i + 1) * 320]);
            }
        }
    }
}

/// Implementation of Keccak linear round witness computation.
///
/// ## Overview
/// This function processes a series of input blocks using the `KeccakLinearUnbatched` operator,
/// applying the Keccak linear round transformations batch by batch.
///
/// ## Parameters
/// - `input`: A fixed-size array of 5 slices, each representing a segment of the Keccak state.
///
/// ## Returns
/// - A fixed-size array of 5 vectors, each containing the transformed output.
///
/// ## Assumptions
/// - All input slices must have the same length (`l`), which must be a multiple of 1024.
///
/// ## Steps
/// 1. Validate input lengths and divisibility.
/// 2. Initialize the Keccak linear operator and output storage.
/// 3. Process each batch of 1024 elements, applying transformations sequentially.
/// 4. Return the transformed output vectors.
pub fn keccak_linround_witness(
    input: [&[BinaryField128b]; 5],
) -> [MultilinearLagrangianPolynomial; 5] {
    // Get the length of the first input slice.
    let l = input[0].len();

    // Validate that all input slices have the same length and that the length is a multiple of
    // 1024.
    assert!(input.iter().all(|x| x.len() == l) && l % 1024 == 0, "Invalid input length");

    // Initialize the Keccak linear operator.
    let m = KeccakLinearUnbatched::new();

    // Initialize output vectors, one for each input slice.
    let mut output =
        array::from_fn(|_| MultilinearLagrangianPolynomial::from(vec![BinaryField128b::ZERO; l]));

    // Process each batch of 1024 elements.
    (0..l / 1024).for_each(|batch_index| {
        // Prepare the input state for the current batch.
        let mut input_state = [[BinaryField128b::ZERO; 1600]; 3];
        input.iter().enumerate().for_each(|(i, block)| {
            // Map the input slice to the intermediate state representation.
            (0..3).for_each(|j| {
                input_state[j][i * 320..(i + 1) * 320].copy_from_slice(
                    &block[batch_index * 1024 + j * 320..batch_index * 1024 + (j + 1) * 320],
                );
            });
        });

        // Initialize the output state for the current batch.
        let mut output_state = [[BinaryField128b::ZERO; 1600]; 3];

        // Apply the Keccak linear operator to each part of the state.
        input_state.iter().zip(output_state.iter_mut()).for_each(|(input, output)| {
            m.apply(input, output);
        });

        // Map the transformed output state back to the flattened output vectors.
        output.iter_mut().enumerate().for_each(|(i, block)| {
            (0..3).for_each(|j| {
                block[batch_index * 1024 + j * 320..batch_index * 1024 + (j + 1) * 320]
                    .copy_from_slice(&output_state[j][i * 320..(i + 1) * 320]);
            });
        });
    });

    // Return the transformed output vectors.
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_lincheck::{builder::LinCheckBuilder, prodcheck::ProdCheckOutput};
    use hashcaster_primitives::{
        poly::{
            point::{Point, Points},
            univariate::UnivariatePolynomial,
        },
        sumcheck::{Sumcheck, SumcheckBuilder},
    };
    use num_traits::MulAdd;
    use rand::rngs::OsRng;

    #[test]
    fn test_keccak_linear() {
        let rng = &mut OsRng;

        // Create a new instance of the KeccakLinear operator.
        let keccak_linear = KeccakLinear::new();

        // **Validate dimensions**
        // - The number of input elements (`n_in`) and output elements (`n_out`) must be consistent.
        // - These represent the flattened Keccak state with 5 blocks of 1024 elements each.
        assert_eq!(keccak_linear.n_in(), 5 * 1024);
        assert_eq!(keccak_linear.n_out(), 5 * 1024);

        // **Generate random input vector `a`**
        // - This simulates a realistic input for the KeccakLinear transformation.
        let a: Vec<_> = (0..1024 * 5).map(|_| BinaryField128b::random(rng)).collect();

        // **Apply the KeccakLinear transformation**
        // - `m_a` will store the result of applying the transformation to `a`.
        let mut m_a = vec![BinaryField128b::ZERO; 1024 * 5];
        keccak_linear.apply(&a, &mut m_a);

        // **Generate another random input vector `b`**
        // - This will be used to test the transpose operation.
        let b: Vec<_> = (0..1024 * 5).map(|_| BinaryField128b::random(rng)).collect();

        // **Apply the transposed transformation**
        // - `m_b` will store the result of applying the transpose operation to `b`.
        let mut m_b = vec![BinaryField128b::ZERO; 1024 * 5];
        keccak_linear.apply_transposed(&b, &mut m_b);

        // **Compute the dot product of the forward-transformed `m_a` with `b`**
        // - This computes `lhs = sum(m_a[i] * b[i])`, where `m_a` is the result of applying the
        //   forward transformation to `a`.
        let lhs =
            m_a.iter().zip(b.iter()).fold(BinaryField128b::ZERO, |acc, (a, b)| a.mul_add(*b, acc));

        // **Compute the dot product of the transpose-transformed `m_b` with `a`**
        // - This computes `rhs = sum(m_b[i] * a[i])`, where `m_b` is the result of applying the
        //   transposed transformation to `b`.
        let rhs =
            m_b.iter().zip(a.iter()).fold(BinaryField128b::ZERO, |acc, (a, b)| a.mul_add(*b, acc));

        // **Validate the equality of `lhs` and `rhs`**
        // - **Mathematical justification**:
        //   - For a valid linear operator `A` and its transpose `A^T`:
        //     - `<A(x), y> = <x, A^T(y)>`
        //   - Here, `<., .>` represents the dot product.
        // - This test verifies that the transpose operation is correctly implemented and maintains
        //   the mathematical properties of a linear operator.
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_keccak_linround_witness_against_keccak_linear() {
        // Define the number of batches (e.g., 2) and the size of each batch (1024 elements).
        const NUM_BATCHES: usize = 2;
        const BATCH_SIZE: usize = 1024;

        let rng = &mut OsRng;

        // Create a new instance of the KeccakLinear operator.
        let keccak_linear = KeccakLinear::new();

        // Generate random input data for all 5 inputs.
        let input_data: [Vec<_>; 5] = array::from_fn(|_| {
            (0..NUM_BATCHES * BATCH_SIZE).map(|_| BinaryField128b::random(rng)).collect()
        });

        // Create slices for the input to `keccak_linround_witness`.
        let input_slices: [_; 5] = [
            input_data[0].as_slice(),
            input_data[1].as_slice(),
            input_data[2].as_slice(),
            input_data[3].as_slice(),
            input_data[4].as_slice(),
        ];

        // Call the `keccak_linround_witness` function.
        let witness_output = keccak_linround_witness(input_slices);

        // Validate against `KeccakLinear` transformation.
        for batch_index in 0..NUM_BATCHES {
            // Flatten the input for this batch.
            let mut flat_input = vec![BinaryField128b::ZERO; 5 * BATCH_SIZE];
            for i in 0..5 {
                flat_input[i * BATCH_SIZE..(i + 1) * BATCH_SIZE].copy_from_slice(
                    &input_slices[i][batch_index * BATCH_SIZE..(batch_index + 1) * BATCH_SIZE],
                );
            }

            // Apply the KeccakLinear transformation.
            let mut flat_output = vec![BinaryField128b::ZERO; 5 * BATCH_SIZE];
            keccak_linear.apply(&flat_input, &mut flat_output);

            // Validate that the flattened output matches the corresponding witness output.
            for i in 0..5 {
                assert_eq!(
                    flat_output[i * BATCH_SIZE..(i + 1) * BATCH_SIZE],
                    witness_output[i][batch_index * BATCH_SIZE..(batch_index + 1) * BATCH_SIZE]
                );
            }
        }
    }

    #[test]
    fn test_keccak_lincheck() {
        // Setup the number of variables
        const NUM_VARS: usize = 20;

        // Setup the number of active variables
        const NUM_ACTIVE_VARS: usize = 10;

        // Setup the phase switch parameter
        const PHASE_SWITCH: usize = 5;

        let rng = &mut OsRng;

        // Setup a Keccak linear matrix
        let keccak_linear = KeccakLinear::new();

        // Setup NUM_VARS random points
        let points = Points::random(NUM_VARS, rng);

        // Create 5 multilinear lagrangian polynomials with `2^NUM_VARS` coefficients each
        let polys: [MultilinearLagrangianPolynomial; 5] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::random(1 << NUM_VARS, rng));

        // Apply the Keccak linear round witness computation
        let m_p = keccak_linround_witness(array::from_fn(|i| polys[i].as_slice()));

        // Compute the initial claims
        let initial_claims: [_; 5] = array::from_fn(|i| m_p[i].evaluate_at(&points));

        // Setup the lincheck prover
        let prover_builder =
            LinCheckBuilder::new(&polys, &points, &keccak_linear, NUM_ACTIVE_VARS, initial_claims);

        // Setup a random gamma for folding
        let gamma = Point::random(rng);

        // Build the prover
        let mut prover = prover_builder.build(&gamma);

        // Claim to be updated during the main loop
        let mut claim = UnivariatePolynomial::from(initial_claims.to_vec()).evaluate_at(&gamma);

        // Empty vector to store challenges
        let mut challenges = Points::default();

        // Main loop
        for _ in 0..NUM_ACTIVE_VARS {
            // Compute the round polynomial
            let round_poly = prover.round_polynomial().coeffs(claim);

            // Generate a random challenge
            let challenge = Point::random(rng);

            // Update the claim
            claim = round_poly[0] +
                round_poly[1] * *challenge +
                round_poly[2] * *challenge * *challenge;

            // Bind the challenge
            prover.bind(&challenge);

            // Store the challenge
            challenges.push(challenge);
        }

        // Fetch the final evaluations
        let ProdCheckOutput { p_evaluations, q_evaluations } = prover.finish();

        // Compute the expected claim
        let expected_claim = p_evaluations
            .iter()
            .zip(q_evaluations.iter())
            .map(|(a, b)| *a * b)
            .fold(BinaryField128b::ZERO, |a, b| a + b);

        // Validate the claim
        assert_eq!(claim, expected_claim);

        // Extend the challenges with points beyond the active variables
        challenges.extend(points[NUM_ACTIVE_VARS..].iter().cloned());

        for i in 0..5 {
            assert_eq!(p_evaluations[i], polys[i].evaluate_at(&challenges));
        }
    }
}
