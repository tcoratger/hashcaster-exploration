use hashcaster_primitives::poly::{compressed::CompressedPoly, evaluation::Evaluations};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SumcheckProof {
    /// Round polynomials of the sumcheck protocol
    pub round_polys: Vec<CompressedPoly>,
    /// Evaluations
    pub evals: Evaluations,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FriPcsProof {
    pub(crate) transcript: Vec<u8>,
    pub(crate) advice: Vec<u8>,
}
