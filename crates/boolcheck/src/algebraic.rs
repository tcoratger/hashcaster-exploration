use std::ops::Index;

use hashcaster_primitives::binary_field::BinaryField128b;

/// A trait defining a generic algebraic evaluation behavior.
///
/// This trait abstracts the evaluation of bitwise operations or similar computations across
/// data slices. The specifics of the evaluation are defined by the implementor.
pub trait AlgebraicOps<const I: usize, const O: usize> {
    /// Performs an algebraic evaluation based on the provided data and indices.
    fn algebraic(
        &self,
        data: &[BinaryField128b],
        idx_a: usize,
        offset: usize,
    ) -> [[BinaryField128b; O]; 3];

    /// Executes the linear part of a boolean formulas.
    fn linear(&self, data: &[BinaryField128b; I]) -> [BinaryField128b; O];

    /// Executes the quadratic part of a boolean formula and compresses the results.
    fn quadratic(&self, data: &[BinaryField128b; I]) -> [BinaryField128b; O];
}

/// Enum to represent the mode of operation for [`StrideWrapper`].
#[derive(Debug, Clone, Copy)]
pub enum StrideMode {
    Wrapper0,
    Wrapper1,
}

/// Unified wrapper for accessing elements with stride and mode.
#[derive(Debug, Clone, Copy)]
pub struct StrideWrapper<'a, T> {
    pub arr: &'a [T],
    pub start: usize,
    pub offset: usize,
    pub mode: StrideMode,
}

impl<T: std::ops::Add<Output = T> + Copy> Index<usize> for StrideWrapper<'_, T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        match self.mode {
            StrideMode::Wrapper0 => &self.arr[self.start + i * self.offset],
            StrideMode::Wrapper1 => &self.arr[self.start + i * self.offset + 1],
        }
    }
}
