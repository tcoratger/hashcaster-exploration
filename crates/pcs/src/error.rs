use binius_core::poly_commit::batch_pcs;

#[derive(Debug)]
pub enum PcsError {
    Pcs(batch_pcs::Error),
}

impl From<batch_pcs::Error> for PcsError {
    fn from(err: batch_pcs::Error) -> Self {
        Self::Pcs(err)
    }
}
