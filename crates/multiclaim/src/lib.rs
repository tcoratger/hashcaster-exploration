use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_lincheck::prodcheck::ProdCheck;
use std::array;

pub mod builder;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiClaim<const N: usize> {
    pub polys: [Vec<BinaryField128b>; N],
    pub gamma: BinaryField128b,
    pub object: ProdCheck<N>,
}

impl<const N: usize> Default for MultiClaim<N> {
    fn default() -> Self {
        Self {
            polys: array::from_fn(|_| Default::default()),
            gamma: Default::default(),
            object: Default::default(),
        }
    }
}
