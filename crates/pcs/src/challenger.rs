use hashcaster_primitives::binary_field::BinaryField128b;
use p3_challenger::{CanObserve, CanSample, HashChallenger};
use p3_keccak::Keccak256Hash;
use p3_symmetric::CryptographicHasher;

/// A challenger for the F128 field.
#[derive(Clone, Debug)]
pub struct F128Challenger<H: CryptographicHasher<u8, [u8; 32]> = Keccak256Hash>(
    HashChallenger<u8, H, 32>,
);

impl<H> F128Challenger<H>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    /// Creates a new challenger for the F128 field from:
    /// - An initial state, represented as a vector of bytes.
    /// - A cryptographic hasher.
    pub fn new(initial_state: Vec<u8>, hasher: H) -> Self {
        Self(HashChallenger::new(initial_state, hasher))
    }
}

impl F128Challenger<Keccak256Hash> {
    /// Creates a new challenger for the F128 field using the Keccak256 hash function.
    pub fn new_keccak256() -> Self {
        Self::new(Vec::new(), Keccak256Hash)
    }
}

impl<H> CanObserve<BinaryField128b> for F128Challenger<H>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    fn observe(&mut self, value: BinaryField128b) {
        self.0.observe_slice(&value.into_inner().to_be_bytes());
    }
}

impl<H> CanSample<BinaryField128b> for F128Challenger<H>
where
    H: CryptographicHasher<u8, [u8; 32]>,
{
    fn sample(&mut self) -> BinaryField128b {
        BinaryField128b::new(u128::from_be_bytes(self.0.sample_array()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple test hasher that outputs a constant array of the sum of its input repeated 32
    /// times.
    #[derive(Clone)]
    struct TestHasher;

    impl CryptographicHasher<u8, [u8; 32]> for TestHasher {
        /// Computes a hash by summing the input elements and repeating the sum 32 times in the
        /// output.
        fn hash_iter<I>(&self, input: I) -> [u8; 32]
        where
            I: IntoIterator<Item = u8>,
        {
            let sum = input.into_iter().sum();
            [sum; 32]
        }

        /// Computes a hash from slices by summing all elements and repeating the sum 32 times.
        fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
        where
            I: IntoIterator<Item = &'a [u8]>,
        {
            let sum = input.into_iter().flat_map(|slice| slice.iter()).copied().sum();
            [sum; 32]
        }
    }

    #[test]
    fn test_new_keccak256() {
        let mut challenger = F128Challenger::new_keccak256();

        // Sample from the empty challenger to validate default behavior
        let sample = challenger.sample();

        // Compute the hash of an empty input and extract the last 16 bytes
        let hash_bytes = {
            let hash_empty = Keccak256Hash.hash_iter(Vec::new());
            let mut bytes: [_; 16] = hash_empty[16..32].try_into().unwrap();
            bytes.reverse();
            bytes
        };

        // Validate that the sample matches the hash output
        assert_eq!(sample, BinaryField128b::new(u128::from_be_bytes(hash_bytes)));
    }

    #[test]
    fn test_challenger_with_empty_initial_state() {
        // Create a challenger with an empty initial state
        let mut challenger = F128Challenger::new(vec![], TestHasher {});

        // Sampling from an empty initial state
        let sample = challenger.sample();

        // Since the initial state is empty, the hash should yield a sum of 0
        assert_eq!(sample, BinaryField128b::new(u128::from_be_bytes([0; 16])));
    }

    #[test]
    fn test_sample_simple() {
        // Create a challenger with an initial state
        let initial_state = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut challenger = F128Challenger::new(initial_state, TestHasher {});

        // Sampling should reflect the initial state with the hasher applied
        let sample = challenger.sample();
        // We should expect the sum (x 16) of the initial state as the sample
        assert_eq!(sample, BinaryField128b::new(u128::from_be_bytes([55; 16])));
    }

    #[test]
    fn test_sample_after_observe() {
        let mut challenger = F128Challenger::new_keccak256();

        let value = BinaryField128b::new(6789);
        challenger.observe(value);

        let sample = challenger.sample();

        // Compute the hash of an empty input and extract the last 16 bytes
        let hash_bytes = {
            let hash_empty = Keccak256Hash.hash_iter(6789_u128.to_be_bytes());
            let mut bytes: [_; 16] = hash_empty[16..32].try_into().unwrap();
            bytes.reverse();
            bytes
        };

        // Validate that the sample matches the hash output
        assert_eq!(sample, BinaryField128b::new(u128::from_be_bytes(hash_bytes)));
    }

    #[test]
    fn test_observe_and_sample_multiple_values() {
        // Create a challenger with a predefined state
        let mut challenger = F128Challenger::new(vec![], TestHasher {});

        // Observe multiple values
        let values =
            vec![BinaryField128b::new(1), BinaryField128b::new(2), BinaryField128b::new(3)];

        for value in values {
            challenger.observe(value);
        }

        // Sample after observing multiple values
        let sample = challenger.sample();

        // Expected hash based on TestHasher logic (sum of all observed values)
        let expected_hash = {
            let combined =
                [1_u128.to_be_bytes(), 2_u128.to_be_bytes(), 3_u128.to_be_bytes()].concat();
            let sum = combined.iter().copied().sum::<u8>();
            [sum; 16]
        };

        assert_eq!(sample, BinaryField128b::new(u128::from_be_bytes(expected_hash)));
    }

    #[test]
    fn test_observe_large_binary_field() {
        // Create a challenger using Keccak256
        let mut challenger = F128Challenger::new_keccak256();

        // Observe a large value
        let value = BinaryField128b::new(u128::MAX);
        challenger.observe(value);

        // Sample the value
        let sample = challenger.sample();

        // Compute the expected Keccak256 hash
        let expected_hash = {
            let hash = Keccak256Hash.hash_iter(u128::MAX.to_be_bytes());
            let mut bytes: [_; 16] = hash[16..32].try_into().unwrap();
            bytes.reverse();
            bytes
        };

        // Validate the sample matches the Keccak256 hash
        assert_eq!(sample, BinaryField128b::new(u128::from_be_bytes(expected_hash)));
    }
}
