use std::ops::{Deref, DerefMut};

use hashcaster_field::binary_field::BinaryField128b;
use num_traits::{MulAddAssign, One, Zero};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

/// Represents a univariate polynomial with coefficients over a binary field.
///
/// # Fields
/// - `coeffs`: A vector of `BinaryField128b` elements representing the coefficients of the
///   polynomial. The coefficients are stored in increasing order of degree, where the 0th index
///   corresponds to the constant term.
///
/// # Example
/// ```rust,ignore
/// use hashcaster_field::binary_field::BinaryField128b;
/// use univariate_polynomial::UnivariatePolynomial;
///
/// let coeffs = vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];
/// let poly = UnivariatePolynomial::new(coeffs);
/// let at = BinaryField128b::from(2);
/// let result = poly.evaluate_at(&at);
/// println!("Result: {:?}", result);
/// ```
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UnivariatePolynomial {
    /// The coefficients of the polynomial, stored in increasing order of degree.
    pub coeffs: Vec<BinaryField128b>,
}

impl UnivariatePolynomial {
    /// Creates a new polynomial from a vector of coefficients.
    ///
    /// # Parameters
    /// - `coeffs`: A vector of `BinaryField128b` coefficients, with the constant term at index 0.
    ///
    /// # Returns
    /// A `UnivariatePolynomial` instance representing the polynomial with the given coefficients.
    pub const fn new(coeffs: Vec<BinaryField128b>) -> Self {
        Self { coeffs }
    }

    /// Evaluates the polynomial at a given point using Horner's method.
    ///
    /// # Parameters
    /// - `at`: The point at which to evaluate the polynomial, represented as a `BinaryField128b`.
    ///
    /// # Returns
    /// The result of evaluating the polynomial at the given point, as a `BinaryField128b`.
    ///
    /// # Methodology
    /// This method employs Horner's method for efficient polynomial evaluation:
    /// `P(x) = (((a_n \cdot x + a_{n-1}) \cdot x + a_{n-2}) \cdot x + ... + a_0)`
    pub fn evaluate_at(&self, at: &BinaryField128b) -> BinaryField128b {
        // Start with an accumulator initialized to zero.
        self.coeffs.iter().rev().fold(BinaryField128b::zero(), |mut acc, &coeff| {
            // Multiply the accumulator by `at` and add the current coefficient.
            acc.mul_add_assign(at, coeff);
            // Return the updated accumulator for the next iteration.
            acc
        })
    }
}

impl Deref for UnivariatePolynomial {
    type Target = Vec<BinaryField128b>;

    fn deref(&self) -> &Self::Target {
        &self.coeffs
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UnivariatePolynomials(Vec<UnivariatePolynomial>);

impl UnivariatePolynomials {
    /// Constructs the equality polynomial sequence for a given set of points.
    ///
    /// # Description
    /// The equality polynomial sequence is a recursive construction where each step computes
    /// a polynomial that evaluates to 1 at a specific subset of points and 0 elsewhere. This
    /// function calculates the entire sequence of such polynomials for the given points.
    ///
    /// # Formula
    /// Each polynomial in the sequence is defined recursively:
    /// `eq_k(x) = eq_{k-1}(x) \cdot (1 + m_k \cdot x)`
    /// where:
    /// - `eq_0(x) = 1` is the base case.
    /// - `m_k` is the multiplier for the \(k\)-th point.
    ///
    /// # Parameters
    /// - `points`: A vector of `BinaryField128b` elements representing the input points.
    ///
    /// # Returns
    /// - A `UnivariatePolynomials` instance containing the sequence of equality polynomials.
    pub fn new_eq_poly_sequence(points: &[BinaryField128b]) -> Self {
        // Start with the base case: eq_0(x) = 1.
        let mut polynomials = vec![UnivariatePolynomial::new(vec![BinaryField128b::one()])];

        // Iterate over the points in reverse order.
        for (i, &multiplier) in points.iter().rev().enumerate() {
            // Reference the last computed polynomial.
            let previous = &polynomials[i];

            // Allocate space for the new polynomial with twice the size of the previous one.
            let mut new_coeffs = vec![BinaryField128b::zero(); 1 << (i + 1)];

            // Compute the new polynomial using the recurrence relation.
            new_coeffs.par_chunks_exact_mut(2).zip(previous.par_iter()).for_each(
                |(chunk, &prev_coeff)| {
                    // Calculate the updated coefficients.
                    let multiplied = multiplier * prev_coeff;
                    chunk[0] = prev_coeff + multiplied; // Update the first coefficient.
                    chunk[1] = multiplied; // Update the second coefficient.
                },
            );

            // Append the new polynomial to the list.
            polynomials.push(UnivariatePolynomial::new(new_coeffs));
        }

        // Return the constructed sequence of equality polynomials.
        Self(polynomials)
    }
}

impl Deref for UnivariatePolynomials {
    type Target = Vec<UnivariatePolynomial>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for UnivariatePolynomials {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_univariate_polynomial_evaluate_at() {
        // Define the coefficients for the polynomial P(x) = 3 + 2x + x^2
        let coeffs = vec![
            BinaryField128b::from(3), // a_0
            BinaryField128b::from(2), // a_1
            BinaryField128b::from(1), // a_2
        ];

        // Create the polynomial
        let poly = UnivariatePolynomial::new(coeffs);

        // Define the point to evaluate the polynomial at
        let x = BinaryField128b::from(2);

        // Manually compute the expected value
        // P(2) = 3 + 2 * 2 + 1 * (2^2)
        let expected = BinaryField128b::from(3) +
            BinaryField128b::from(2) * BinaryField128b::from(2) +
            BinaryField128b::from(1) * BinaryField128b::from(2) * BinaryField128b::from(2);

        // Evaluate the polynomial at x
        let result = poly.evaluate_at(&x);

        // Assert that the result matches the expected value
        assert_eq!(result, expected, "Wrong evaluation result");
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_eq_poly_sequence() {
        // Define a vector of input points in the field BinaryField128b.
        let points = vec![
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
            BinaryField128b::from(4),
        ];

        // Compute the equality polynomial sequence using the eq_poly_sequence function.
        let eq_sequence = UnivariatePolynomials::new_eq_poly_sequence(&points);

        // Assert that the computed equality polynomial sequence matches the expected result.
        // This ensures the function is working as intended.
        assert_eq!(
            eq_sequence,
            UnivariatePolynomials(vec![
                UnivariatePolynomial::new(vec![BinaryField128b::from(
                    257870231182273679343338569694386847745
                )]),
                UnivariatePolynomial::new(vec![
                    BinaryField128b::from(257870231182273679343338569694386847749),
                    BinaryField128b::from(4)
                ]),
                UnivariatePolynomial::new(vec![
                    BinaryField128b::from(276728653372472173290161332362114236431),
                    BinaryField128b::from(24175334173338157438437990908848766986),
                    BinaryField128b::from(24175334173338157438437990908848766989),
                    BinaryField128b::from(24175334173338157438437990908848766985)
                ]),
                UnivariatePolynomial::new(vec![
                    BinaryField128b::from(262194116391218557050997340512544882712),
                    BinaryField128b::from(28499219382283035146096761727006801943),
                    BinaryField128b::from(36391510607255973141463116147421347857),
                    BinaryField128b::from(12382329933390930187138101121107623963),
                    BinaryField128b::from(318146307338789859576041968956220637212),
                    BinaryField128b::from(336838576029515239038751755741412982801),
                    BinaryField128b::from(323629372821402637551770173079877058582),
                    BinaryField128b::from(299454038648064480113332182171028291615)
                ]),
                UnivariatePolynomial::new(vec![
                    BinaryField128b::from(156988152385753943545944842966784802826),
                    BinaryField128b::from(238399472904936139640581627536708993042),
                    BinaryField128b::from(103112231144489255218858846742755934222),
                    BinaryField128b::from(118147824772591482324176031488095027225),
                    BinaryField128b::from(214143658130290692373901413934757314572),
                    BinaryField128b::from(247871520449118298941928385465288753181),
                    BinaryField128b::from(114993504431031574539843754968093818891),
                    BinaryField128b::from(127368045919134702485539060344707612688),
                    BinaryField128b::from(138015824183221341818021192535835148297),
                    BinaryField128b::from(181502608447665156913871231917135757333),
                    BinaryField128b::from(311457361334396773361093192980979777548),
                    BinaryField128b::from(30864341024960054608398038725843484701),
                    BinaryField128b::from(197871324294196201012813621595002109966),
                    BinaryField128b::from(137763997785582402678037463463867973656),
                    BinaryField128b::from(30950013923125879264268791915275616264),
                    BinaryField128b::from(326990117386703710211842172749841694743)
                ])
            ])
        );
    }
}
