use crate::poly::{compressed::CompressedPoly, evaluation::FixedEvaluations, point::Point};

/// A trait that abstracts the sumcheck protocol builder.
pub trait SumcheckBuilder<const N: usize> {
    /// The sumcheck protocol type.
    type Sumcheck: Sumcheck<N>;

    /// Builds a new sumcheck instance.
    fn build(self, gamma: &Point) -> Self::Sumcheck;
}

/// A trait that abstracts the sumcheck protocol methods.
pub trait Sumcheck<const N: usize> {
    /// The output type of the sumcheck protocol.
    type Output: EvaluationProvider<N>;

    /// Computes the polynomial for the current round of the sumcheck protocol.
    fn round_polynomial(&mut self) -> CompressedPoly;

    /// Updates the state of the sumcheck instance by binding a new challenge.
    fn bind(&mut self, challenge: &Point);

    /// Returns the output of the sumcheck protocol.
    fn finish(&self) -> Self::Output;
}

pub trait EvaluationProvider<const N: usize> {
    /// Returns the evaluations of the sumcheck protocol output.
    fn evals(self) -> FixedEvaluations<N>;
}
