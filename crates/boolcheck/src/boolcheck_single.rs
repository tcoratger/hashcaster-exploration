use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::compressed::CompressedPoly;

#[derive(Clone, Debug, Default)]
pub struct BoolCheckSingle {
    pt: Vec<BinaryField128b>,
    poly: Vec<BinaryField128b>,
    polys: Vec<Vec<BinaryField128b>>,
    ext: Option<Vec<BinaryField128b>>,
    poly_coords: Option<Vec<BinaryField128b>>,
    c: usize,
    challenges: Vec<BinaryField128b>,
    bits_to_trits_map: Vec<u16>,
    eq_sequence: Vec<Vec<BinaryField128b>>,
    round_polys: Vec<CompressedPoly>,
}

impl BoolCheckSingle {
    /// This function calculates two mappings for a ternary (base-3) representation of integers:
    /// - **Trit Mapping**: Maps every integer `i` to either a power of 3 or a scaled binary
    ///   equivalent, depending on the digits in its base-3 representation.
    /// - **Bit Mapping**: Reversely maps binary numbers to their ternary equivalents with only the
    ///   digits `0` and `1`.
    ///
    /// ### Mathematical Principle
    /// For an integer `i`:
    /// - **Trit Mapping Logic**:
    ///   - If the ternary representation of `i` contains a `2`, the mapping is determined by the
    ///     position (most significant digit) of the first `2`: `trit_mapping[i] = 3^k`, where `k`
    ///     is the index of the most significant `2`.
    ///   - If no `2` is present, the mapping is twice the binary equivalent of the ternary number:
    ///     `trit_mapping[i] = 2 * b`, where `b` is the binary equivalent of `i`.
    ///
    /// - **Bit Mapping Logic**:
    ///   - Maps binary numbers to their equivalent ternary number containing only `0` and `1`.
    ///     `bit_mapping[b] = j`, where `j` is the ternary equivalent of `b`.
    ///
    /// ### Parameters
    /// - `c`: The highest index for ternary digits. Determines the range of ternary numbers to
    ///   process.
    ///
    /// ### Returns
    /// - `(Vec<u16>, Vec<u16>)`: A tuple containing:
    ///   - `bit_mapping`: Maps binary numbers to ternary equivalents.
    ///   - `trit_mapping`: Maps ternary numbers based on their digit properties.
    pub fn trit_mapping(&self) -> (Vec<u16>, Vec<u16>) {
        let c = self.c;
        // Calculate 3^(c+1), the total number of ternary numbers to process
        let pow3 = 3usize.pow((c + 1) as u32);

        // Preallocate memory for efficiency
        let mut bit_mapping = Vec::with_capacity(1 << (c + 1));
        let mut trit_mapping = Vec::with_capacity(pow3);

        // Iterate over all possible ternary numbers from 0 to 3^(c+1)-1
        for i in 0..pow3 {
            // Initialize variables for current ternary number, binary equivalent, and most
            // significant digit (msd)
            let (mut current, mut bin_value, mut msd) = (i, 0u16, None);

            // Decompose the number `i` into base-3 and calculate mappings
            for idx in 0..=c {
                // Extract the least significant ternary digit
                let digit = (current % 3) as u16;
                current /= 3;

                // Record the position of the most significant `2` if found
                if digit == 2 {
                    msd = Some(idx);
                } else {
                    // Update the binary equivalent by masking the digit with `1`
                    bin_value |= (digit & 1) << idx;
                }
            }

            // Determine the mapping based on the presence of a `2`
            if let Some(msd) = msd {
                // If a `2` exists, map to the corresponding power of 3
                trit_mapping.push(3u16.pow(msd as u32));
            } else {
                // Otherwise, map to twice the binary equivalent
                trit_mapping.push(bin_value << 1);
                bit_mapping.push(i as u16);
            }
        }

        // Return both mappings
        (bit_mapping, trit_mapping)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_mapping_small_c() {
        // Create an instance of BoolCheckSingle with c = 1 (two ternary digits: 0, 1, 2).
        let bool_check = BoolCheckSingle { c: 1, ..Default::default() };

        // Call the trit_mapping method to compute the mappings.
        let (bit_mapping, trit_mapping) = bool_check.trit_mapping();

        // Check that the binary-to-ternary mapping matches the expected result.
        assert_eq!(bit_mapping, vec![0, 1, 3, 4]);

        // Check that the ternary-to-binary mapping matches the expected result.
        assert_eq!(trit_mapping, vec![0, 2, 1, 4, 6, 1, 3, 3, 3]);
    }

    #[test]
    fn test_trit_mapping_medium_c() {
        // Create an instance of BoolCheckSingle with c = 2 (three ternary digits: 0, 1, 2).
        let bool_check = BoolCheckSingle { c: 2, ..Default::default() };

        // Call the trit_mapping method to compute the mappings.
        let (bit_mapping, trit_mapping) = bool_check.trit_mapping();

        // Check that the binary-to-ternary mapping matches the expected result.
        assert_eq!(bit_mapping, vec![0, 1, 3, 4, 9, 10, 12, 13]);

        // Check that the ternary-to-binary mapping matches the expected result.
        assert_eq!(
            trit_mapping,
            vec![
                0, 2, 1, 4, 6, 1, 3, 3, 3, 8, 10, 1, 12, 14, 1, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9
            ]
        );
    }

    #[test]
    fn test_trit_mapping_large_c() {
        let bool_check = BoolCheckSingle { c: 4, ..Default::default() };

        let (bit_mapping, trit_mapping) = bool_check.trit_mapping();

        assert_eq!(
            bit_mapping,
            vec![
                0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31, 36, 37, 39, 40, 81, 82, 84, 85, 90, 91,
                93, 94, 108, 109, 111, 112, 117, 118, 120, 121
            ]
        );

        assert_eq!(
            trit_mapping,
            vec![
                0, 2, 1, 4, 6, 1, 3, 3, 3, 8, 10, 1, 12, 14, 1, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                16, 18, 1, 20, 22, 1, 3, 3, 3, 24, 26, 1, 28, 30, 1, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
                27, 27, 27, 27, 27, 27, 27, 27, 32, 34, 1, 36, 38, 1, 3, 3, 3, 40, 42, 1, 44, 46,
                1, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 48, 50, 1, 52, 54, 1, 3, 3, 3, 56, 58, 1,
                60, 62, 1, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27, 27,
                27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 81, 81, 81,
                81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
                81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
                81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81,
                81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81
            ]
        );
    }

    #[test]
    fn test_trit_mapping_no_c() {
        // Create an instance of BoolCheckSingle with c = 0 (single ternary digit).
        let bool_check = BoolCheckSingle { c: 0, ..Default::default() };

        // Call the trit_mapping method to compute the mappings.
        let (bit_mapping, trit_mapping) = bool_check.trit_mapping();

        // Check that the binary-to-ternary mapping matches the expected result.
        assert_eq!(bit_mapping, vec![0, 1]);

        // Check that the ternary-to-binary mapping matches the expected result.
        assert_eq!(trit_mapping, vec![0, 2, 1]);
    }
}
