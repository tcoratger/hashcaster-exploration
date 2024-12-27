use hashcaster_field::{
    binary_field::BinaryField128b, frobenius_cobasis::COBASIS_FROBENIUS_TRANSPOSE,
};
use num_traits::Zero;
use std::ops::{Deref, DerefMut};

/// Evaluations of a polynomial at some points.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Evaluations(Vec<BinaryField128b>);

impl Evaluations {
    /// Creates a new `Evaluations` instance.
    pub const fn new(evaluations: Vec<BinaryField128b>) -> Self {
        Self(evaluations)
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
        self.iter().enumerate().fold(BinaryField128b::zero(), |acc, (j, twist)| {
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
        // Create a vector to store the twisted evaluations.
        // This will hold the new evaluations of P in the inverse Frobenius orbit.
        let mut twisted_evals = vec![];

        // Perform the twist operation for 128 basis elements.
        // Each iteration computes one twisted evaluation of P.
        for _ in 0..128 {
            // Square each element in the input evaluations (`evals`).
            // This is the Frobenius mapping applied to all elements in `evals`.
            // Perform x = x^2 for all x in evals.
            self.iter_mut().map(|x| *x *= *x).count();

            // Compute the evaluation of P for the current twist.
            // - Multiply each element of `evals` by the corresponding basis element
            // - Sum them up to get the evaluation of P in this twisted context.
            twisted_evals.push(
                (0..128)
                    .map(|i| BinaryField128b::basis(i) * self[i])
                    .fold(BinaryField128b::zero(), |a, b| a + b),
            );
        }

        // Reverse the twisted evaluations.
        // This ensures that the evaluations align with the inverse Frobenius orbit.
        twisted_evals.reverse();

        // Replace the original evaluations with the twisted evaluations.
        // The `evals` slice now contains the twisted evaluations.
        self.clone_from_slice(&twisted_evals);
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
        // Apply the Frobenius transformation \( x \mapsto x^{2^i} \) for alignment.
        for i in 0..128 {
            self[i] = self[i].frobenius(i as i32);
        }

        // Use a fixed-size array to store the untwisted evaluations.
        let mut untwisted = [BinaryField128b::zero(); 128];

        // Compute the untwisted evaluations using the `pi` function.
        for i in 0..128 {
            untwisted[i] = self.pi(i);
        }

        // Replace the current evaluations with the untwisted evaluations.
        self.copy_from_slice(&untwisted);
    }
}

impl From<Vec<BinaryField128b>> for Evaluations {
    fn from(evaluations: Vec<BinaryField128b>) -> Self {
        Self(evaluations)
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

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::One;

    #[test]
    fn test_pi() {
        // Generate a random element `r` in the binary field.
        let mut r = BinaryField128b::random();

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
            .collect::<Vec<_>>()
            .into();

        // Iterate over each index `i` in the range [0, 127].
        (0..128).for_each(|i| {
            // Compute the result of the `pi` function at index `i` for the `orbit`.
            let bit = orbit.pi(i);

            // Interpret the result of `pi(i)` as either 0 or 1.
            // - If the result equals 0 in the binary field, it represents a bit value of 0.
            // - If the result equals 1 in the binary field, it represents a bit value of 1.
            let lhs = match bit {
                b if b == BinaryField128b::zero() => 0,
                b if b == BinaryField128b::one() => 1,
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
        let orbit: Evaluations = vec![BinaryField128b::zero(); 128].into();

        for i in 0..128 {
            assert_eq!(orbit.pi(i), BinaryField128b::zero(), "Failed for index {i}");
        }
    }

    #[test]
    fn test_pi_single_non_zero() {
        let mut orbit = vec![BinaryField128b::zero(); 128];
        // Set a single non-zero value.
        orbit[5] = BinaryField128b::one();
        let orbit: Evaluations = orbit.into();

        for i in 0..128 {
            let expected = BinaryField128b::new(COBASIS_FROBENIUS_TRANSPOSE[i][5]);
            assert_eq!(orbit.pi(i), expected, "Failed for index {i}");
        }
    }

    #[test]
    fn test_pi_alternating() {
        let orbit: Evaluations = (0..128)
            .map(|i| if i % 2 == 0 { BinaryField128b::one() } else { BinaryField128b::zero() })
            .collect::<Vec<_>>()
            .into();

        for i in 0..128 {
            let expected = COBASIS_FROBENIUS_TRANSPOSE[i]
                .iter()
                .enumerate()
                .filter(|(j, _)| j % 2 == 0) // Only include contributions from even indices.
                .map(|(_, &val)| BinaryField128b::new(val))
                .fold(BinaryField128b::zero(), |acc, x| acc + x);

            assert_eq!(orbit.pi(i), expected, "Failed for index {i}");
        }
    }

    #[test]
    fn test_pi_random_orbit() {
        let orbit: Evaluations =
            (0..128).map(|_| BinaryField128b::random()).collect::<Vec<_>>().into();

        for i in 0..128 {
            let expected: BinaryField128b = (0..128)
                .map(|j| BinaryField128b::new(COBASIS_FROBENIUS_TRANSPOSE[i][j]) * orbit[j])
                .fold(BinaryField128b::zero(), |acc, x| acc + x);

            assert_eq!(orbit.pi(i), expected, "Failed for index {i}");
        }
    }

    #[test]
    fn twist_untwist() {
        // Generate a random set of 128 evaluations.
        let lhs: Evaluations =
            Evaluations::from((0..128).map(|_| BinaryField128b::random()).collect::<Vec<_>>());

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
        let mut evaluations: Evaluations = vec![BinaryField128b::zero(); 128].into();

        // Apply the `twist` transformation.
        evaluations.twist();

        // Assert that all evaluations remain zero after the transformation.
        evaluations.iter().for_each(|&val| {
            assert_eq!(val, BinaryField128b::zero(), "Twist failed for all-zero input.");
        });
    }

    #[test]
    fn test_untwist_all_zeros() {
        // All evaluations are initially zero.
        let mut evaluations: Evaluations = vec![BinaryField128b::zero(); 128].into();

        // Apply the `untwist` transformation.
        evaluations.untwist();

        // Assert that all evaluations remain zero after the transformation.
        evaluations.iter().for_each(|&val| {
            assert_eq!(val, BinaryField128b::zero(), "Untwist failed for all-zero input.");
        });
    }
}
