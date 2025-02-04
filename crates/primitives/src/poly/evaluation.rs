use crate::{
    binary_field::BinaryField128b, frobenius_cobasis::COBASIS_FROBENIUS_TRANSPOSE,
    sumcheck::EvaluationProvider,
};
use num_traits::MulAdd;
use rand::Rng;
use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    array,
    ops::{Deref, DerefMut},
};

/// Evaluations of a polynomial at some points.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Evaluations(pub Vec<BinaryField128b>);

impl Evaluations {
    /// Creates a new `Evaluations` instance.
    pub const fn new(evaluations: Vec<BinaryField128b>) -> Self {
        Self(evaluations)
    }

    /// Creates a new `Evaluations` instance with random evaluations.
    pub fn random<RNG: Rng>(n: usize, rng: &mut RNG) -> Self {
        Self((0..n).map(|_| BinaryField128b::random(rng)).collect())
    }

    /// Consumes the `Evaluations` instance and returns the inner vector of evaluations.
    pub fn into_inner(self) -> Vec<BinaryField128b> {
        self.0
    }

    /// Computes the evaluation of the `pi` function at a given index `i`.
    ///
    /// # Theory
    /// The `pi` function computes a linear functional in the dual basis representation of a binary
    /// field. Specifically, it evaluates the linear functional associated with the `i`-th
    /// cobasis vector on a given set of evaluations. This is mathematically equivalent to
    /// summing up the products of the cobasis elements and their corresponding evaluation
    /// points, ensuring an isomorphic mapping to the dual space.
    ///
    /// Given:
    /// - `cobasis_frobenius[j]` as the `j`-th element of the transpose of the Frobenius cobasis
    ///   matrix.
    /// - `twist[j]` as the `j`-th evaluation of the polynomial in the Frobenius orbit.
    ///
    /// The formula for `pi` is:
    /// \[
    /// \pi_i = \sum_{j=0}^{127} c_{ij} \cdot twist[j]
    /// \]
    /// Where `c_{ij}` are the elements of the Frobenius cobasis matrix transpose.
    ///
    /// This function performs this computation efficiently using an iterator and a fold operation.
    ///
    /// # Parameters
    /// - `i`: The index of the cobasis vector (0 ≤ `i` < 128).
    ///
    /// # Returns
    /// - A `BinaryField128b` instance representing the result of the `pi` function evaluation.
    pub fn pi(&self, i: usize) -> BinaryField128b {
        // Retrieve the `i`-th row of the COBASIS_FROBENIUS_TRANSPOSE matrix.
        // This row corresponds to the coefficients of the linear functional `pi_i`.
        let cobasis_frobenius = &COBASIS_FROBENIUS_TRANSPOSE[i];

        // Compute the summation:
        // Iterate over the evaluations and corresponding cobasis coefficients.
        self.iter().enumerate().fold(BinaryField128b::ZERO, |acc, (j, twist)| {
            // For each pair,
            // - multiply the twist by the cobasis coefficient
            // - add to the accumulator.
            acc + BinaryField128b::new(cobasis_frobenius[j]) * twist
        })
    }

    /// Applies the "twist" transformation to the evaluations.
    ///
    /// # Theory
    /// The twist transformation maps the evaluations of a polynomial into
    /// an inverse Frobenius orbit. This involves applying successive squarings
    /// (Frobenius map: \( x \mapsto x^2 \)) and reconstructing the evaluations
    /// using a weighted sum with a fixed basis.
    ///
    /// ## Steps
    /// 1. Perform successive squarings of all evaluations.
    /// 2. Use a basis to compute weighted sums of the squared values to form the twisted
    ///    evaluations.
    /// 3. Reverse the order of the twisted evaluations to align them with the inverse Frobenius
    ///    orbit.
    /// 4. Replace the original evaluations with the twisted evaluations.
    pub fn twist(&mut self) {
        // Ensure the evaluations can be chunked into 128-element groups.
        assert_eq!(self.len() % 128, 0, "Evaluations must be a multiple of 128.");

        // Process each chunk of 128 elements separately.
        self.par_chunks_exact_mut(128).for_each(|chunk| {
            // Create the twisted evaluations for this chunk.
            let mut twisted_evals: [_; 128] = array::from_fn(|_| {
                // Apply the Frobenius map (squaring) to all elements in the chunk.
                chunk.iter_mut().for_each(|x| *x *= *x);

                // Compute the twisted evaluation for the i-th basis index.
                (0..128).fold(BinaryField128b::ZERO, |acc, j| {
                    BinaryField128b::basis(j).mul_add(chunk[j], acc)
                })
            });

            // Reverse the twisted evaluations to align with the inverse Frobenius orbit.
            twisted_evals.reverse();

            // Replace the original chunk with the twisted evaluations.
            chunk.copy_from_slice(&twisted_evals);
        });
    }

    /// Reverts the "twist" transformation by applying the inverse.
    ///
    /// # Theory
    /// The untwist transformation reverts the evaluations back to their
    /// original space by:
    /// 1. Applying the Frobenius transformation to each element.
    /// 2. Using the `pi` function to reconstruct the original evaluations based on a dual basis
    ///    representation.
    ///
    /// ## Steps
    /// 1. Apply the Frobenius map \( x \mapsto x^{2^i} \) to align the elements.
    /// 2. Use the `pi` function to reconstruct the evaluations from the twisted form.
    pub fn untwist(&mut self) {
        // Ensure the evaluations can be chunked into 128-element groups.
        assert_eq!(self.len() % 128, 0, "Evaluations must be a multiple of 128.");

        // Process each chunk of 128 elements separately.
        self.par_chunks_exact_mut(128).for_each(|chunk| {
            // Apply the Frobenius transformation \( x \mapsto x^{2^i} \) for alignment.
            // Each element in the chunk is updated to its Frobenius-transformed value.
            #[allow(clippy::cast_possible_wrap)]
            chunk.iter_mut().enumerate().for_each(|(i, val)| {
                *val = val.frobenius(i as i32);
            });

            // Compute the untwisted evaluations
            let untwisted_chunk: [_; 128] = array::from_fn(|i| {
                COBASIS_FROBENIUS_TRANSPOSE[i]
                    .iter()
                    .enumerate()
                    .fold(BinaryField128b::ZERO, |acc, (j, coeff)| {
                        BinaryField128b::new(*coeff).mul_add(chunk[j], acc)
                    })
            });

            // Replace the current chunk with the untwisted values.
            chunk.copy_from_slice(&untwisted_chunk);
        });
    }
}

impl From<Vec<BinaryField128b>> for Evaluations {
    fn from(evaluations: Vec<BinaryField128b>) -> Self {
        Self(evaluations)
    }
}

impl From<&[BinaryField128b]> for Evaluations {
    fn from(evaluations: &[BinaryField128b]) -> Self {
        Self(evaluations.to_vec())
    }
}

impl Deref for Evaluations {
    type Target = Vec<BinaryField128b>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Evaluations {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl FromIterator<BinaryField128b> for Evaluations {
    fn from_iter<T: IntoIterator<Item = BinaryField128b>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl AsRef<[BinaryField128b]> for Evaluations {
    fn as_ref(&self) -> &[BinaryField128b] {
        &self.0
    }
}

/// Evaluations of a polynomial at some fixed points using a constant-size array.
///
/// # Generic Parameters
/// - `N`: The number of evaluations stored.
///
/// # Fields
/// - `evaluations`: An array of `BinaryField128b` elements representing the evaluations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedEvaluations<const N: usize>(pub [BinaryField128b; N]);

impl<const N: usize> Serialize for FixedEvaluations<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_seq(self.0.iter())
    }
}

impl<'de, const N: usize> Deserialize<'de> for FixedEvaluations<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize into a Vec first
        let vec = Vec::<BinaryField128b>::deserialize(deserializer)?;

        // Ensure the vector has the correct length
        if vec.len() != N {
            return Err(serde::de::Error::invalid_length(
                vec.len(),
                &format!("Expected length {N}").as_str(),
            ));
        }

        // Convert Vec into a fixed-size array
        let array: [BinaryField128b; N] = vec
            .try_into()
            .map_err(|_| serde::de::Error::custom("Failed to convert Vec to fixed-size array"))?;

        Ok(Self(array))
    }
}

impl<const N: usize> FixedEvaluations<N> {
    /// Creates a new `FixedEvaluations` instance from an array of evaluations.
    ///
    /// # Parameters
    /// - `evaluations`: A fixed-size array of `BinaryField128b` values.
    ///
    /// # Returns
    /// A `FixedEvaluations<N>` instance storing the provided evaluations.
    pub const fn new(evaluations: [BinaryField128b; N]) -> Self {
        Self(evaluations)
    }

    /// Applies the "twist" transformation to the evaluations.
    ///
    /// # Theory
    /// The twist transformation maps the evaluations of a polynomial into
    /// an inverse Frobenius orbit. This involves applying successive squarings
    /// (Frobenius map: \( x \mapsto x^2 \)) and reconstructing the evaluations
    /// using a weighted sum with a fixed basis.
    ///
    /// ## Steps
    /// 1. Perform successive squarings of all evaluations.
    /// 2. Use a basis to compute weighted sums of the squared values to form the twisted
    ///    evaluations.
    /// 3. Reverse the order of the twisted evaluations to align them with the inverse Frobenius
    ///    orbit.
    /// 4. Replace the original evaluations with the twisted evaluations.
    pub fn twist(&mut self) {
        // Ensure the evaluations can be chunked into 128-element groups.
        assert_eq!(self.len() % 128, 0, "Evaluations must be a multiple of 128.");

        // Process each chunk of 128 elements separately.
        self.0.par_chunks_exact_mut(128).for_each(|chunk| {
            // Create the twisted evaluations for this chunk.
            let mut twisted_evals: [_; 128] = array::from_fn(|_| {
                // Apply the Frobenius map (squaring) to all elements in the chunk.
                chunk.iter_mut().for_each(|x| *x *= *x);

                // Compute the twisted evaluation for the i-th basis index.
                (0..128).fold(BinaryField128b::ZERO, |acc, j| {
                    BinaryField128b::basis(j).mul_add(chunk[j], acc)
                })
            });

            // Reverse the twisted evaluations to align with the inverse Frobenius orbit.
            twisted_evals.reverse();

            // Replace the original chunk with the twisted evaluations.
            chunk.copy_from_slice(&twisted_evals);
        });
    }

    /// Reverts the "twist" transformation by applying the inverse.
    ///
    /// # Theory
    /// The untwist transformation reverts the evaluations back to their
    /// original space by:
    /// 1. Applying the Frobenius transformation to each element.
    /// 2. Using the `pi` function to reconstruct the original evaluations based on a dual basis
    ///    representation.
    ///
    /// ## Steps
    /// 1. Apply the Frobenius map \( x \mapsto x^{2^i} \) to align the elements.
    /// 2. Use the `pi` function to reconstruct the evaluations from the twisted form.
    pub fn untwist(&mut self) {
        // Ensure the evaluations can be chunked into 128-element groups.
        assert_eq!(self.len() % 128, 0, "Evaluations must be a multiple of 128.");

        // Process each chunk of 128 elements separately.
        self.0.par_chunks_exact_mut(128).for_each(|chunk| {
            // Apply the Frobenius transformation \( x \mapsto x^{2^i} \) for alignment.
            // Each element in the chunk is updated to its Frobenius-transformed value.
            #[allow(clippy::cast_possible_wrap)]
            chunk.iter_mut().enumerate().for_each(|(i, val)| {
                *val = val.frobenius(i as i32);
            });

            // Compute the untwisted evaluations
            let untwisted_chunk: [_; 128] = array::from_fn(|i| {
                COBASIS_FROBENIUS_TRANSPOSE[i]
                    .iter()
                    .enumerate()
                    .fold(BinaryField128b::ZERO, |acc, (j, coeff)| {
                        BinaryField128b::new(*coeff).mul_add(chunk[j], acc)
                    })
            });

            // Replace the current chunk with the untwisted values.
            chunk.copy_from_slice(&untwisted_chunk);
        });
    }
}

impl<const N: usize> Deref for FixedEvaluations<N> {
    type Target = [BinaryField128b; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> AsRef<[BinaryField128b]> for FixedEvaluations<N> {
    fn as_ref(&self) -> &[BinaryField128b] {
        &self.0
    }
}

impl<const N: usize> EvaluationProvider<N> for FixedEvaluations<N> {
    fn evals(self) -> Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn test_pi() {
        let rng = &mut OsRng;

        // Generate a random element `r` in the binary field.
        let mut r = BinaryField128b::random(rng);

        // Create an `Evaluations` instance representing the Frobenius orbit of `r`.
        // This orbit contains 128 successive squarings of `r`.
        let orbit: Evaluations = (0..128)
            .map(|_| {
                // Save the current value of `r`.
                let res = r;
                // Update `r` by squaring it (Frobenius map: r -> r^2).
                r *= r;
                // Return the previous value of `r`.
                res
            })
            .collect();

        // Iterate over each index `i` in the range [0, 127].
        (0..128).for_each(|i| {
            // Compute the result of the `pi` function at index `i` for the `orbit`.
            let bit = orbit.pi(i);

            // Interpret the result of `pi(i)` as either 0 or 1.
            // - If the result equals 0 in the binary field, it represents a bit value of 0.
            // - If the result equals 1 in the binary field, it represents a bit value of 1.
            let lhs = match bit {
                b if b == BinaryField128b::ZERO => 0,
                b if b == BinaryField128b::ONE => 1,
                _ => panic!(),
            };

            // Validate that the computed bit `lhs` matches the expected bit value.
            // The expected bit value is extracted from the `i`-th position of the binary
            // representation of `r` (right-shifted by `i` and modulo 2).
            assert_eq!(lhs, (r.into_inner() >> i) % 2);
        });
    }

    #[test]
    fn test_pi_all_zeroes() {
        let orbit: Evaluations = vec![BinaryField128b::ZERO; 128].into();

        for i in 0..128 {
            assert_eq!(orbit.pi(i), BinaryField128b::ZERO, "Failed for index {i}");
        }
    }

    #[test]
    fn test_pi_single_non_zero() {
        let mut orbit = vec![BinaryField128b::ZERO; 128];
        // Set a single non-zero value.
        orbit[5] = BinaryField128b::ONE;
        let orbit: Evaluations = orbit.into();

        for (i, cobasis) in COBASIS_FROBENIUS_TRANSPOSE.iter().enumerate() {
            let expected = BinaryField128b::new(cobasis[5]);
            assert_eq!(orbit.pi(i), expected, "Failed for index {i}");
        }
    }

    #[test]
    fn test_pi_alternating() {
        let orbit: Evaluations = (0..128)
            .map(|i| if i % 2 == 0 { BinaryField128b::ONE } else { BinaryField128b::ZERO })
            .collect();

        for (i, cobasis) in COBASIS_FROBENIUS_TRANSPOSE.iter().enumerate() {
            let expected = cobasis
                .iter()
                .enumerate()
                .filter(|(j, _)| j % 2 == 0) // Only include contributions from even indices.
                .map(|(_, &val)| BinaryField128b::new(val))
                .fold(BinaryField128b::ZERO, |acc, x| acc + x);

            assert_eq!(orbit.pi(i), expected, "Failed for index {i}");
        }
    }

    #[test]
    fn test_pi_random_orbit() {
        let rng = &mut OsRng;

        let orbit = Evaluations::random(128, rng);

        for (i, cobasis) in COBASIS_FROBENIUS_TRANSPOSE.iter().enumerate() {
            let expected: BinaryField128b = (0..128)
                .map(|j| BinaryField128b::new(cobasis[j]) * orbit[j])
                .fold(BinaryField128b::ZERO, |acc, x| acc + x);

            assert_eq!(orbit.pi(i), expected, "Failed for index {i}");
        }
    }

    #[test]
    fn twist_untwist() {
        let rng = &mut OsRng;

        // Generate a random set of 128 evaluations.
        let lhs = Evaluations::random(128, rng);

        // Clone `lhs` to create an independent copy for transformation.
        let mut rhs = lhs.clone();

        // Apply `twist` followed by `untwist` to `rhs`.
        rhs.twist();
        rhs.untwist();

        // Assert that `rhs` matches the original `lhs`.
        assert_eq!(lhs, rhs, "Twist followed by untwist did not restore the original evaluations.");

        // Apply `untwist` followed by `twist` to `rhs`.
        rhs.untwist();
        rhs.twist();

        // Assert that `rhs` matches the original `lhs` again.
        assert_eq!(lhs, rhs, "Untwist followed by twist did not restore the original evaluations.");
    }

    #[test]
    fn test_twist_all_zeros() {
        // All evaluations are initially zero.
        let mut evaluations: Evaluations = vec![BinaryField128b::ZERO; 128].into();

        // Apply the `twist` transformation.
        evaluations.twist();

        // Assert that all evaluations remain zero after the transformation.
        evaluations.iter().for_each(|&val| {
            assert_eq!(val, BinaryField128b::ZERO, "Twist failed for all-zero input.");
        });
    }

    #[test]
    fn test_untwist_all_zeros() {
        // All evaluations are initially zero.
        let mut evaluations: Evaluations = vec![BinaryField128b::ZERO; 128].into();

        // Apply the `untwist` transformation.
        evaluations.untwist();

        // Assert that all evaluations remain zero after the transformation.
        evaluations.iter().for_each(|&val| {
            assert_eq!(val, BinaryField128b::ZERO, "Untwist failed for all-zero input.");
        });
    }
}
