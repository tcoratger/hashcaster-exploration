use hashcaster_primitives::poly::{compressed::CompressedPoly, evaluation::FixedEvaluations};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProof<const N: usize, const M: usize> {
    /// Round polynomials of the sumcheck protocol
    pub round_polys: Vec<CompressedPoly<M>>,
    /// Evaluations
    pub evals: FixedEvaluations<N>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FriPcsProof {
    pub(crate) transcript: Vec<u8>,
    pub(crate) advice: Vec<u8>,
}
