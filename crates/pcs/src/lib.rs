use crate::utils::iso_slice;
use binius_core::{
    fiat_shamir::HasherChallenger,
    merkle_tree_vcs::{BinaryMerkleTreeProver, BinaryMerkleTreeScheme},
    poly_commit::{batch_pcs::BatchPCS, PolyCommitScheme, FRIPCS},
    tower::{PackedTop, TowerFamily, TowerUnderlier},
    transcript::{AdviceReader, AdviceWriter, TranscriptReader, TranscriptWriter},
};
use binius_field::{
    as_packed_field::{PackScalar, PackedType},
    BinaryField128b, ExtensionField, PackedExtension, PackedField, PackedFieldIndexable,
    TowerField,
};
use binius_hal::make_portable_backend;
use binius_hash::Hasher;
use binius_math::{EvaluationDomainFactory, MultilinearExtension};
use binius_ntt::NTTOptions;
use error::PcsError;
use groestl_crypto::Groestl256;
use hashcaster_primitives::{
    binary_field::BinaryField128b as F128,
    poly::{multinear_lagrangian::MultilinearLagrangianPolynomial, point::Points},
};
use p3_symmetric::PseudoCompressionFunction;
use proof::FriPcsProof;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use utils::iso_slice_packed;

pub mod challenger;
pub mod error;
pub mod proof;
pub mod utils;

/// The cryptographic extension field used in the constraint system protocol.
pub type FExt<Tower> = <Tower as TowerFamily>::B128;

/// The evaluation domain for sumcheck protocols.
pub type FDomain<Tower> = <Tower as TowerFamily>::B8;

/// The Reed–Solomon alphabet used for FRI encoding (small field).
pub type FEncode<Tower> = <Tower as TowerFamily>::B32;

/// The packed coefficient subfield used in the PCS.
pub type PackedCoefficientSubfield<Tower, U, F> =
    <PackedType<U, FExt<Tower>> as PackedExtension<F>>::PackedSubfield;

/// The FRI-based Binius polynomial commitment scheme (PCS).
///
/// The polynomial is committed by committing the Reed-Solomon encoding of its packed values with a
/// Merkle tree as a vector commitment scheme.
pub type PCS<Tower, U, F, Digest, DomainFactory, Hash, Compress> = FRIPCS<
    F,
    FDomain<Tower>,
    FEncode<Tower>,
    PackedType<U, FExt<Tower>>,
    DomainFactory,
    BinaryMerkleTreeProver<Digest, Hash, Compress>,
    BinaryMerkleTreeScheme<Digest, Hash, Compress>,
>;

/// A batched polynomial commitment scheme for multilinear polynomials.
///
/// This type enables efficient evaluation of multiple multilinear polynomials at a single point
/// $\vec{r}$.
///
/// It combines packed coefficients, an extension field, and the FRI-based commitment scheme.
pub type BatchFRIPCS<Tower, U, F, Digest, DomainFactory, Hash, Compress> = BatchPCS<
    PackedCoefficientSubfield<Tower, U, F>,
    FExt<Tower>,
    PCS<Tower, U, F, Digest, DomainFactory, Hash, Compress>,
>;

/// The packed subfield type extracted from a scalar type `U` and a field `F`.
pub type PackedScalar<U, F> = <U as PackScalar<F>>::Packed;

/// The packed subfield extracted from the packed extension.
pub type PackedSubfield<U, F, FS> = <PackedType<U, F> as PackedExtension<FS>>::PackedSubfield;

/// The commitment inside the polynomial commitment scheme.
pub type Commitment<Tower, U, F, Digest, DomainFactory, Hash, Compress> =
    <BatchFRIPCS<Tower, U, F, Digest, DomainFactory, Hash, Compress> as PolyCommitScheme<
        PackedSubfield<U, FExt<Tower>, F>,
        FExt<Tower>,
    >>::Commitment;

/// The committed value of the polynomial commitment scheme.
pub type Committed<Tower, U, F, Digest, DomainFactory, Hash, Compress> =
    <BatchFRIPCS<Tower, U, F, Digest, DomainFactory, Hash, Compress> as PolyCommitScheme<
        PackedSubfield<U, FExt<Tower>, F>,
        FExt<Tower>,
    >>::Committed;

/// Alias for multilinear extensions over packed subfields.
pub type PackedMultilinearExtension<Tower, U> = MultilinearExtension<
    PackedSubfield<U, <Tower as TowerFamily>::B128, <Tower as TowerFamily>::B128>,
>;

/// Alias for the return type of the `commit` function.
pub type CommitResult<Tower, U, Digest, DomainFactory, Hash, Compress> = (
    Vec<PackedMultilinearExtension<Tower, U>>,
    Commitment<Tower, U, <Tower as TowerFamily>::B128, Digest, DomainFactory, Hash, Compress>,
    Committed<Tower, U, <Tower as TowerFamily>::B128, Digest, DomainFactory, Hash, Compress>,
);

/// A batched FRI polynomial commitment scheme.
///
/// This structure supports efficient commitment and evaluation of multilinear polynomials
/// using FRI over a GF(2^128) extension field.
#[allow(missing_debug_implementations)]
pub struct BatchFRIPCS128<Tower, U, Digest, DomainFactory, Hash, Compress>
where
    U: TowerUnderlier<Tower> + PackScalar<Tower::B128>,
    Tower: TowerFamily,
    Tower::B128:
        PackedTop<Tower> + ExtensionField<Tower::B128> + PackedExtension<Tower::B128> + TowerField,
    Digest: PackedField<Scalar: TowerField>,
    DomainFactory: EvaluationDomainFactory<Tower::B8>,
    Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
    Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
    PackedType<U, Tower::B128>: PackedTop<Tower> + PackedFieldIndexable,
{
    /// The underlying batched polynomial commitment scheme (PCS) leveraging FRI.
    ///
    /// This field handles commitments and openings for multilinear polynomials.
    batch_fri_pcs: BatchFRIPCS<Tower, U, Tower::B128, Digest, DomainFactory, Hash, Compress>,
}

impl<Tower, U, Digest, DomainFactory, Hash, Compress>
    BatchFRIPCS128<Tower, U, Digest, DomainFactory, Hash, Compress>
where
    U: TowerUnderlier<Tower> + PackScalar<Tower::B128>,
    Tower: TowerFamily,
    Tower::B128: PackedTop<Tower>
        + ExtensionField<Tower::B128>
        + PackedExtension<Tower::B128>
        + TowerField
        + From<BinaryField128b>,
    Digest: PackedField<Scalar: TowerField>,
    DomainFactory: EvaluationDomainFactory<Tower::B8> + Default,
    Hash: Hasher<Tower::B128, Digest = Digest> + Send + Sync,
    Compress: PseudoCompressionFunction<Digest, 2> + Default + Sync,
    PackedType<U, Tower::B128>: PackedTop<Tower> + PackedFieldIndexable,
{
    pub fn new(
        security_bits: usize,
        log_inv_rate: usize,
        num_vars: usize,
        batch_size: usize,
    ) -> Self {
        let merkle_prover = BinaryMerkleTreeProver::new(Compress::default());
        let log_n_polys = batch_size.next_power_of_two().ilog2() as usize;
        let fri_n_vars = num_vars + log_n_polys;
        let fri_pcs = FRIPCS::with_optimal_arity(
            fri_n_vars,
            log_inv_rate,
            security_bits,
            merkle_prover,
            DomainFactory::default(),
            NTTOptions::default(),
        )
        .unwrap();
        Self { batch_fri_pcs: BatchPCS::new(fri_pcs, num_vars, log_n_polys).unwrap() }
    }

    pub fn commit(
        &self,
        polys: &[MultilinearLagrangianPolynomial; 5],
    ) -> CommitResult<Tower, U, Digest, DomainFactory, Hash, Compress> {
        let polys: Vec<_> = polys
            .par_iter()
            .map(|poly| {
                MultilinearExtension::from_values(iso_slice_packed::<U, Tower::B128>(poly)).unwrap()
            })
            .collect();

        let (commitment, committed) = self.batch_fri_pcs.commit(&polys).unwrap();
        (polys, commitment, committed)
    }

    pub fn open(
        &self,
        polys: &[PackedMultilinearExtension<Tower, U>],
        committed: &Committed<Tower, U, Tower::B128, Digest, DomainFactory, Hash, Compress>,
        points: &Points,
    ) -> FriPcsProof {
        let mut transcript = TranscriptWriter::<HasherChallenger<Groestl256>>::default();
        let mut advice = AdviceWriter::default();
        self.batch_fri_pcs
            .prove_evaluation(
                &mut advice,
                &mut transcript,
                committed,
                polys,
                &iso_slice(points.as_binary_field_slice()),
                &make_portable_backend(),
            )
            .unwrap();
        FriPcsProof { transcript: transcript.finalize(), advice: advice.finalize() }
    }

    pub fn verify(
        &self,
        commitment: &Commitment<Tower, U, Tower::B128, Digest, DomainFactory, Hash, Compress>,
        proof: &FriPcsProof,
        points: &Points,
        evals: &[F128],
    ) -> Result<(), PcsError> {
        let mut transcript =
            TranscriptReader::<HasherChallenger<Groestl256>>::new(proof.transcript.clone());
        let mut advice = AdviceReader::new(proof.advice.clone());
        self.batch_fri_pcs.verify_evaluation(
            &mut advice,
            &mut transcript,
            commitment,
            &iso_slice(points.as_binary_field_slice()),
            &iso_slice(evals),
            &make_portable_backend(),
        )?;
        Ok(())
    }
}
