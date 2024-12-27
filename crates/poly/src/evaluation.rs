use hashcaster_field::binary_field::BinaryField128b;
use std::ops::{Deref, DerefMut};

/// Evaluations of a polynomial at some points.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Evaluations(Vec<BinaryField128b>);

impl Evaluations {
    /// Creates a new `Evaluations` instance.
    pub const fn new(evaluations: Vec<BinaryField128b>) -> Self {
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
