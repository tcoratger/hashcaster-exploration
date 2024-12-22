use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::{
    compressed::CompressedPoly, multinear_lagrangian::MultilinearLagrangianPolynomials,
};
use num_traits::Zero;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub mod bool_trait;
pub mod builder;
pub mod package;

#[derive(Clone, Debug, Default)]
pub struct BoolCheck {
    points: Vec<BinaryField128b>,
    poly: Vec<BinaryField128b>,
    polys: Vec<Vec<BinaryField128b>>,
    extended_table: Vec<BinaryField128b>,
    poly_coords: Option<Vec<BinaryField128b>>,
    c: usize,
    challenges: Vec<BinaryField128b>,
    bit_mapping: Vec<u16>,
    eq_sequence: MultilinearLagrangianPolynomials,
    round_polys: Vec<CompressedPoly>,
    claim: BinaryField128b,
}

impl BoolCheck {
    /// Returns the current round of the protocol.
    ///
    /// The current round is represented by the number of challenges that have been submitted to the
    /// prover.
    pub fn current_round(&self) -> usize {
        self.challenges.len()
    }

    /// Returns the number of evaluation points for the polynomials in the protocol.
    ///
    /// For examples, the multilinear polynomial `p` can be written as:
    /// p(x) = p_0 + p_1*x_1 + p_2*x_2 + ... + p_n*x_n
    ///
    /// The number of variables in the polynomial is `n`.
    pub fn number_variables(&self) -> usize {
        self.points.len()
    }

    pub fn compute_round_polynomial(&mut self) -> CompressedPoly {
        // Compute the current round of the protocol.
        let round = self.current_round();

        // Compute the number of variables in the polynomial.
        let number_variables = self.number_variables();

        // In the sumcheck protocol, the number of rounds is limited by the number of variables.
        assert!(round < number_variables, "Protocol has reached the maximum number of rounds.");

        // If the round polynomial has already been computed, return it.
        //
        // We have all the computed polynomials up to the current round in the cache.
        if self.round_polys.len() > round {
            // Return the polynomial from the cache.
            if let Some(poly) = self.round_polys.last() {
                return poly.clone();
            }
            // This situation should never happen.
            return CompressedPoly::default();
        }

        // The protocol is defined in two phases:
        // - The first phase last `c` rounds. It operates over the quadratic polynomial `P Ʌ Q` on
        //   the subset `(0,1,∞)^{c+1} x (0,1)^{n-c-1}`
        // - After `c` phases, the prover no longer has the required data (as we have computed the
        //   extension table only on `3^{c+1} \times 2^{n-c-1}` subset) and cannot proceed further.
        //   This is when we change strategy and compute `P_i(r_0, \ldots, r_c, \mathbf{x}_{>c})`
        //   and `Q_i(r_0, \ldots, r_c, \mathbf{x}_{>c})` (using RESTRICT functionality).
        if round <= self.c {
            // Fetch the extended table that was computed during the setup phase.
            //
            // The extended table is a table of size `3^{c+1} \times 2^{n-c-1}`.
            let extended_table = &self.extended_table;

            // Fetch the equality polynomial for the current round.
            //
            // The equality polynomials for all the points are stored in cache during the setup
            // phase.
            let eq_poly_round = self.eq_sequence.poly_at(number_variables - round - 1);

            // The first phase of the protocol lasts `c` rounds.
            //
            // This is the number of remaining rounds in the first phase.
            let phase1_dimensions = self.c - round;

            // Compute `3^{remaining rounds in the first phase}`.
            let pow3 = 3usize.pow(phase1_dimensions as u32);

            let mut poly_deg2 = (0..(1 << (number_variables - self.c - 1)))
                .into_par_iter()
                .map(|i| {
                    // Pre-compute the base index and offset for the current `i`.
                    let base_index = i << phase1_dimensions;
                    let base_offset = 3 * (i * pow3);

                    // Prepare
                    let mut pd2_part = [BinaryField128b::zero(); 3];

                    (0..(1 << phase1_dimensions)).for_each(|j| {
                        let index = base_index + j;
                        let offset = base_offset + 3 * self.bit_mapping[j] as usize;
                        let multiplier = eq_poly_round[index];

                        pd2_part
                            .iter_mut()
                            .zip(&extended_table[offset..offset + 3])
                            .for_each(|(acc, value)| *acc += *value * multiplier);
                    });

                    pd2_part
                })
                .reduce(
                    || [BinaryField128b::zero(); 3],
                    |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                );

            // Cast polynomial to coefficient form.
            poly_deg2[1] += poly_deg2[0] + poly_deg2[2];
        } else {
        }

        CompressedPoly::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use builder::BoolCheckBuilder;
    use hashcaster_poly::multinear_lagrangian::MultilinearLagrangianPolynomial;
    use package::BooleanPackage;
    use std::iter::repeat_with;

    #[test]
    fn test_current_rounds() {
        // Define a sample BoolCheck instance.
        // No challenges have been submitted yet.
        let mut bool_check = BoolCheck::default();

        // Assert the initial round (no challenges yet).
        assert_eq!(bool_check.current_round(), 0);

        // Add a challenge and test the round count.
        bool_check.challenges.push(BinaryField128b::from(10));
        assert_eq!(bool_check.current_round(), 1);

        // Add more challenges and test the updated round count.
        bool_check.challenges.extend(vec![
            BinaryField128b::from(20),
            BinaryField128b::from(30),
            BinaryField128b::from(40),
        ]);
        assert_eq!(bool_check.current_round(), 4);

        // Add another challenge and verify the round count again.
        bool_check.challenges.push(BinaryField128b::from(50));
        assert_eq!(bool_check.current_round(), 5);
    }

    #[test]
    fn test_number_variables() {
        // Create a BoolCheck instance with a defined number of points (variables).
        let points =
            vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];
        let bool_check = BoolCheck { points: points.clone(), ..Default::default() };

        // Assert that the number of variables matches the length of the points vector.
        assert_eq!(bool_check.number_variables(), points.len());

        // Test with no points (empty vector).
        let empty_bool_check = BoolCheck { points: vec![], ..Default::default() };
        assert_eq!(empty_bool_check.number_variables(), 0);

        // Test with a large number of points.
        let large_points: Vec<_> = (0..1000).map(BinaryField128b::from).collect();
        let large_bool_check = BoolCheck { points: large_points.clone(), ..Default::default() };
        assert_eq!(large_bool_check.number_variables(), large_points.len());
    }

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
