use std::ops::Index;

use hashcaster_field::binary_field::BinaryField128b;

/// A trait defining a generic algebraic evaluation behavior.
///
/// This trait abstracts the evaluation of bitwise operations or similar computations across
/// data slices. The specifics of the evaluation are defined by the implementor.
pub trait AlgebraicOps {
    /// The type of output produced by the algebraic evaluation.
    type AlgebraicOutput;

    /// The type of output produced by the linear evaluation.
    type LinearOutput;

    /// The type of output produced by the quadratic evaluation.
    type QuadraticOutput;

    /// Performs an algebraic evaluation based on the provided data and indices.
    fn algebraic(
        &self,
        data: [impl Index<usize, Output = BinaryField128b>; 4],
    ) -> Self::AlgebraicOutput;

    /// Executes the linear part of a boolean formulas.
    fn linear(&self, data: &[BinaryField128b]) -> Self::LinearOutput;

    /// Executes the quadratic part of a boolean formula and compresses the results.
    fn quadratic(&self, data: &[BinaryField128b]) -> Self::QuadraticOutput;
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
