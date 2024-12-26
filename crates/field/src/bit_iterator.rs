/// An iterator over the set bits in a `u64`.
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BitIterator(u64);

impl BitIterator {
    /// Creates a new [`BitIterator`] with the specified value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
}

impl Iterator for BitIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }
        // Count trailing zeros
        let tz = self.0.trailing_zeros() as usize;
        // Clear the lowest set bit
        self.0 &= self.0 - 1;
        Some(tz)
    }
}

#[cfg(test)]
mod tests {
    use super::BitIterator;

    #[test]
    fn test_empty_iterator() {
        // Test with a value of 0, no bits set
        let mut iter = BitIterator::new(0);
        // No bits to iterate over
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_single_bit_set() {
        // Test with only one bit set
        let mut iter = BitIterator::new(0b0001);
        // Only the 0th bit is set
        assert_eq!(iter.next(), Some(0));
        // No more bits
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_multiple_bits_set() {
        // Test with multiple bits set
        let mut iter = BitIterator::new(0b10101);
        // First set bit is at position 0
        assert_eq!(iter.next(), Some(0));
        // Next set bit is at position 2
        assert_eq!(iter.next(), Some(2));
        // Next set bit is at position 4
        assert_eq!(iter.next(), Some(4));
        // No more bits
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_highest_bit_set() {
        // Test with the highest bit set in u64
        let mut iter = BitIterator::new(1 << 63);
        // Only the highest bit is set
        assert_eq!(iter.next(), Some(63));
        // No more bits
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_all_bits_set() {
        // Test with all bits set in u64
        let mut iter = BitIterator::new(u64::MAX);
        for i in 0..64 {
            // Each bit position from 0 to 63 should be returned
            assert_eq!(iter.next(), Some(i));
        }
        // No more bits
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_random_bit_pattern() {
        // Test with a random bit pattern
        let mut iter = BitIterator::new(0b110_0101);
        // First set bit is at position 0
        assert_eq!(iter.next(), Some(0));
        // Next set bit is at position 2
        assert_eq!(iter.next(), Some(2));
        // Next set bit is at position 5
        assert_eq!(iter.next(), Some(5));
        // Next set bit is at position 5
        assert_eq!(iter.next(), Some(6));
        // No more bits
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_trailing_zeros_optimization() {
        // Test behavior with leading zeros
        let mut iter = BitIterator::new(0b1000_0000);
        // The only set bit is at position 7
        assert_eq!(iter.next(), Some(7));
        // No more bits
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_reset_and_reuse_iterator() {
        // Test reusing the iterator with a new value
        let mut iter = BitIterator::new(0b1001);
        // First set bit is at position 0
        assert_eq!(iter.next(), Some(0));
        // Next set bit is at position 3
        assert_eq!(iter.next(), Some(3));
        // No more bits
        assert_eq!(iter.next(), None);

        // Reset the iterator with a new value
        iter = BitIterator::new(0b10);
        // First set bit is at position 1
        assert_eq!(iter.next(), Some(1));
        // No more bits
        assert_eq!(iter.next(), None);
    }
}
