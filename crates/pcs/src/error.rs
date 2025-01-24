use binius_core::poly_commit::batch_pcs;

#[derive(Debug)]
pub enum SumcheckError {
    UnmatchedSubclaim(String),
}

#[derive(Debug)]
pub enum PcsError {
    Sumcheck(SumcheckError),
    Pcs(batch_pcs::Error),
}

impl From<batch_pcs::Error> for PcsError {
    fn from(err: batch_pcs::Error) -> Self {
        Self::Pcs(err)
    }
}

impl From<SumcheckError> for PcsError {
    fn from(err: SumcheckError) -> Self {
        Self::Sumcheck(err)
    }
}
