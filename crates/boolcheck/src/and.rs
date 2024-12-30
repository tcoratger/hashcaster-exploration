use crate::algebraic::AlgebraicOps;
use hashcaster_field::binary_field::BinaryField128b;
use num_traits::Zero;

/// A structure that implements the behavior of the AND operations.
#[derive(Debug, Clone)]
pub struct AndPackage<const M: usize>;

impl<const M: usize> AlgebraicOps for AndPackage<M> {
    type AlgebraicOutput = [[BinaryField128b; M]; 3];

    type LinearCompressedOutput = [BinaryField128b; M];

    type QuadraticCompressedOutput = [BinaryField128b; M];

    fn algebraic(
        &self,
        data: &[BinaryField128b],
        idx_a: usize,
        offset: usize,
    ) -> Self::AlgebraicOutput {
        // Ensure the output size is valid for the AND package.
        assert_eq!(M, 1, "Invalid output size for AND package");

        // Initialize the indices for the first and second operands.
        let mut idx_a = idx_a * 2;
        // Calculate the index for the second operand with the specified offset.
        let mut idx_b = idx_a + offset * 128;
        let mut ret = [[BinaryField128b::zero(); M]; 3];

        // Iterate over the 128 basis elements of the binary field.
        for i in 0..128 {
            // Calculate the basis element for the current iteration.
            let basis = BinaryField128b::basis(i);

            // Extract the elements from the data slices.
            let a = data[idx_a];
            let b = data[idx_b];
            let a_next = data[idx_a + 1];
            let b_next = data[idx_b + 1];

            // `Σ (ϕ_i * a * b)`
            ret[0][0] += basis * a * b;
            // `Σ (ϕ_i * a_next * b_next)`
            ret[1][0] += basis * a_next * b_next;
            // `Σ (ϕ_i * (a + a_next) * (b + b_next))`
            ret[2][0] += basis * (a + a_next) * (b + b_next);

            // Move to the next indices
            idx_a += offset;
            idx_b += offset;
        }

        ret
    }

    fn linear_compressed(&self, data: &[BinaryField128b]) -> Self::LinearCompressedOutput {
        // Ensure the output size is valid for the AND package.
        assert_eq!(M, 1, "Invalid output size for AND package");

        // Validate the size of the input data.
        assert_eq!(data.len(), 2, "Invalid input size for AND package");

        // Return a zero-initialized array as the result of the linear compression.
        [BinaryField128b::zero(); M]
    }

    fn quadratic_compressed(&self, data: &[BinaryField128b]) -> Self::QuadraticCompressedOutput {
        // Ensure the output size is valid for the AND package.
        assert_eq!(M, 1, "Invalid output size for AND package");

        // Validate the size of the input data.
        assert_eq!(data.len(), 2, "Invalid input size for AND package");

        // Compute and return the bitwise AND of the two input elements.
        [data[0] & data[1]; M]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let and_package = AndPackage::<1>;

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

        let rhs = and_package.algebraic(&input_coords, 0, 1);

        let a_quad = and_package.quadratic_compressed(&a);
        let b_quad = and_package.quadratic_compressed(&b);
        let c_quad = and_package.quadratic_compressed(&c);
        let d_quad = and_package.quadratic_compressed(&d);

        assert_eq!(rhs[0], a_quad);
        assert_eq!(rhs[1], b_quad);
        assert_eq!(rhs[2], [a_quad[0] + b_quad[0] + c_quad[0] + d_quad[0]]);
    }
}
