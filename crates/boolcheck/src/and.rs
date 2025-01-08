use crate::algebraic::AlgebraicOps;
use hashcaster_primitives::binary_field::BinaryField128b;
use num_traits::Zero;
use std::ops::Index;

/// A structure that implements the behavior of the AND operations.
#[derive(Debug, Clone)]
pub struct AndPackage<const I: usize, const O: usize>;

impl<const I: usize, const O: usize> AlgebraicOps<I, O> for AndPackage<I, O> {
    fn algebraic(
        &self,
        data: [impl Index<usize, Output = BinaryField128b>; 4],
    ) -> [[BinaryField128b; O]; 3] {
        let mut ret = [[BinaryField128b::zero(); O]; 3];

        // Iterate over the 128 basis elements of the binary field.
        for i in 0..128 {
            // Calculate the basis element for the current iteration.
            let basis = BinaryField128b::basis(i);

            // Extract the elements from the data slices.
            let a = data[0][i];
            let b = data[1][i];
            let a_next = data[2][i];
            let b_next = data[3][i];

            // `Σ (ϕ_i * a * b)`
            ret[0][0] += basis * a * b;
            // `Σ (ϕ_i * a_next * b_next)`
            ret[1][0] += basis * a_next * b_next;
            // `Σ (ϕ_i * (a + a_next) * (b + b_next))`
            ret[2][0] += basis * (a + a_next) * (b + b_next);
        }

        ret
    }

    fn linear(&self, _data: &[BinaryField128b; I]) -> [BinaryField128b; O] {
        // Return a zero-initialized array as the result of the linear compression.
        [BinaryField128b::zero(); O]
    }

    fn quadratic(&self, data: &[BinaryField128b; I]) -> [BinaryField128b; O] {
        // Compute and return the bitwise AND of the two input elements.
        [data[0] & data[1]; O]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebraic::{StrideMode, StrideWrapper};

    #[test]
    fn test_exec_alg_and() {
        // Generate two random field elements as input.
        let a1 = BinaryField128b::random();
        let a2 = BinaryField128b::random();

        let a = [a1, a2];
        let b = [
            BinaryField128b::from(a1.into_inner() >> 1),
            BinaryField128b::from(a2.into_inner() >> 1),
        ];
        let c = [a1, BinaryField128b::from(a2.into_inner() >> 1)];
        let d = [BinaryField128b::from(a1.into_inner() >> 1), a2];

        let and_package = AndPackage::<2, 1>;

        // Prepare input for the algebraic implementation.
        // - Take the array of input field elements.
        // - Transform each binary field element into 128 components, representing the bitwise
        //   values of the element.
        // - Flatten the resulting array of arrays into a single array.
        let mut input_coords = a
            .iter()
            .flat_map(|x| (0..128).map(|i| BinaryField128b::from((x.into_inner() >> i) & 1 != 0)))
            .collect::<Vec<_>>();

        input_coords.push(BinaryField128b::zero());

        let rhs = and_package.algebraic([
            StrideWrapper { arr: &input_coords, start: 0, offset: 1, mode: StrideMode::Wrapper0 },
            StrideWrapper { arr: &input_coords, start: 128, offset: 1, mode: StrideMode::Wrapper0 },
            StrideWrapper { arr: &input_coords, start: 0, offset: 1, mode: StrideMode::Wrapper1 },
            StrideWrapper { arr: &input_coords, start: 128, offset: 1, mode: StrideMode::Wrapper1 },
        ]);

        let a_quad = and_package.quadratic(&a);
        let b_quad = and_package.quadratic(&b);
        let c_quad = and_package.quadratic(&c);
        let d_quad = and_package.quadratic(&d);

        assert_eq!(rhs[0], a_quad);
        assert_eq!(rhs[1], b_quad);
        assert_eq!(rhs[2], [a_quad[0] + b_quad[0] + c_quad[0] + d_quad[0]]);
    }
}
