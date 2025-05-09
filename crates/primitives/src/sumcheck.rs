use crate::{
    binary_field::BinaryField128b,
    poly::{compressed::CompressedPoly, evaluation::FixedEvaluations},
};

/// A trait that abstracts the sumcheck protocol builder.
pub trait SumcheckBuilder<const N: usize, const CP: usize> {
    /// The sumcheck protocol type.
    type Sumcheck: Sumcheck<N, CP>;

    /// Builds a new sumcheck instance.
    fn build(self, gamma: &BinaryField128b) -> Self::Sumcheck;
}

/// A trait that abstracts the sumcheck protocol methods.
pub trait Sumcheck<const N: usize, const CP: usize> {
    /// The output type of the sumcheck protocol.
    type Output: EvaluationProvider<N>;

    /// Computes the polynomial for the current round of the sumcheck protocol.
    fn round_polynomial(&mut self) -> CompressedPoly<CP>;

    /// Updates the state of the sumcheck instance by binding a new challenge.
    fn bind(&mut self, challenge: &BinaryField128b);

    /// Returns the output of the sumcheck protocol.
    fn finish(self) -> Self::Output;
}

pub trait EvaluationProvider<const N: usize> {
    /// Returns the evaluations of the sumcheck protocol output.
    fn evals(self) -> FixedEvaluations<N>;
}
