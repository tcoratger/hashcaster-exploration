use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::point::Points;
use std::array;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MulticlaimBuilder<const N: usize> {
    pub polys: [Vec<BinaryField128b>; N],
    pub points: Points,
    pub openings: Vec<BinaryField128b>,
}

impl<const N: usize> Default for MulticlaimBuilder<N> {
    fn default() -> Self {
        Self {
            polys: array::from_fn(|_| Default::default()),
            points: Default::default(),
            openings: Vec::new(),
        }
    }
}

impl<const N: usize> MulticlaimBuilder<N> {
    pub fn new(
        polys: [Vec<BinaryField128b>; N],
        points: Points,
        openings: Vec<BinaryField128b>,
    ) -> Self {
        // Check that the number of openings is correct.
        assert_eq!(openings.len(), N * 128, "Invalid number of openings");

        // Check that the length of each polynomial is 2^{number of points}.
        let expected_poly_len = 1 << points.len();
        assert!(
            polys.iter().all(|poly| poly.len() == expected_poly_len),
            "Invalid polynomial length"
        );

        Self { polys, points, openings }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_field::binary_field::BinaryField128b;
    use hashcaster_poly::point::{Point, Points};

    #[test]
    fn test_multiclaim_builder_default() {
        // Create a default MulticlaimBuilder instance with N = 3.
        let builder: MulticlaimBuilder<3> = MulticlaimBuilder::default();

        // Verify that the default polys array is empty.
        for poly in builder.polys.iter() {
            assert!(poly.is_empty());
        }

        // Verify that the default points collection is empty.
        assert!(builder.points.is_empty());

        // Verify that the default openings vector is empty.
        assert!(builder.openings.is_empty());
    }

    #[test]
    fn test_multiclaim_builder_new_valid() {
        // Define valid polynomials for N = 2.
        let polys: [Vec<BinaryField128b>; 2] = [
            vec![BinaryField128b::from(1), BinaryField128b::from(2)],
            vec![BinaryField128b::from(3), BinaryField128b::from(4)],
        ];

        // Define valid points.
        let points = Points::from(vec![Point::from(BinaryField128b::from(1))]);

        // Define valid openings for N = 2 with 128 * N elements.
        let openings = vec![BinaryField128b::from(0); 2 * 128];

        // Create a new MulticlaimBuilder instance.
        let builder = MulticlaimBuilder::new(polys.clone(), points.clone(), openings.clone());

        // Verify that the builder contains the expected fields.
        assert_eq!(builder, MulticlaimBuilder { polys, points, openings });
    }

    #[test]
    #[should_panic(expected = "Invalid number of openings")]
    fn test_multiclaim_builder_new_invalid_openings() {
        // Define valid polynomials for N = 2.
        let polys: [Vec<BinaryField128b>; 2] = [
            vec![BinaryField128b::from(1), BinaryField128b::from(2)],
            vec![BinaryField128b::from(3), BinaryField128b::from(4)],
        ];

        // Define valid points.
        let points = Points::from(vec![Point::from(BinaryField128b::from(1))]);

        // Define an invalid number of openings (not N * 128).
        let openings = vec![BinaryField128b::from(0); 100];

        // Attempt to create a new MulticlaimBuilder instance (should panic).
        MulticlaimBuilder::new(polys, points, openings);
    }

    #[test]
    #[should_panic(expected = "Invalid polynomial length")]
    fn test_multiclaim_builder_new_invalid_polynomial_length() {
        // Define invalid polynomials with incorrect lengths for N = 2.
        let polys: [Vec<BinaryField128b>; 2] = [
            vec![BinaryField128b::from(1)], // Length is 1, should be 2^number of points.
            vec![BinaryField128b::from(3), BinaryField128b::from(4)],
        ];

        // Define valid points (1 point, so polynomials should have length 2).
        let points = Points::from(vec![Point::from(BinaryField128b::from(1))]);

        // Define valid openings for N = 2 with 128 * N elements.
        let openings = vec![BinaryField128b::from(0); 2 * 128];

        // Attempt to create a new MulticlaimBuilder instance (should panic).
        MulticlaimBuilder::new(polys, points, openings);
    }

    #[test]
    fn test_multiclaim_builder_new_edge_case() {
        // Define valid polynomials for N = 1.
        let polys: [Vec<BinaryField128b>; 1] = [vec![BinaryField128b::from(42)]];

        // Define an empty points collection (0 points, so polynomials should have length 1).
        let points = Points::default();

        // Define valid openings for N = 1 with 128 * N elements.
        let openings = vec![BinaryField128b::from(0); 128];

        // Create a new MulticlaimBuilder instance.
        let builder = MulticlaimBuilder::new(polys.clone(), points.clone(), openings.clone());

        // Verify that the builder contains the expected fields.
        assert_eq!(builder, MulticlaimBuilder { polys, points, openings });
    }
}
