use hashcaster_primitives::poly::{compressed::CompressedPoly, evaluation::Evaluations};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProof {
    /// Round polynomials of the sumcheck protocol
    pub round_polys: Vec<CompressedPoly>,
    /// Evaluations
    pub evals: Evaluations,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriPcsProof {
    transcript: Vec<u8>,
    advice: Vec<u8>,
}
