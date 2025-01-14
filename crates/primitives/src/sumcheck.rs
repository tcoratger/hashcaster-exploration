use crate::poly::{compressed::CompressedPoly, point::Point};

pub trait Sumcheck {
    /// The output type of the sumcheck protocol.
    type Output;

    /// Computes the polynomial for the current round of the sumcheck protocol.
    fn round_polynomial(&mut self) -> CompressedPoly;

    /// Updates the state of the sumcheck instance by binding a new challenge.
    fn bind(&mut self, challenge: &Point);

    /// Returns the output of the sumcheck protocol.
    fn finish(&self) -> Self::Output;
}
