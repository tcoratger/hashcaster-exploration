use crate::utils::iso_slice_packed;
use binius_core::{
    merkle_tree::{
        BinaryMerkleTreeProver, BinaryMerkleTreeScheme, MerkleTreeProver, MerkleTreeScheme,
    },
    piop::{commit, make_commit_params_with_optimal_arity, CommitMeta},
    protocols::fri::{CommitOutput, FRIParams},
};
use binius_field::{BinaryField128bPolyval, PackedBinaryPolyval1x128b};
use binius_hash::compress::Groestl256ByteCompression;
use binius_math::{MLEDirectAdapter, MultilinearExtension};
use groestl_crypto::Groestl256;
use hashcaster_primitives::poly::multinear_lagrangian::MultilinearLagrangianPolynomials;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

pub mod challenger;
pub mod proof;
pub mod utils;

/// The cryptographic extension field used in the constraint system protocol.
///
/// Hashcaster exclusively uses the F128 POLYVAL basis for now.
pub type FExt = BinaryField128bPolyval;

/// The evaluation domain for sumcheck protocols.
///
/// Hashcaster uses the F128 POLYVAL basis for homogeneity.
///
/// Although Binius traditionally uses an 8-bit field for evaluation domains (sufficient for
/// handling all reasonable sumcheck constraint degrees, even with skipped rounds via the univariate
/// skip technique), we adhere to the F128 POLYVAL basis.
pub type FDomain = BinaryField128bPolyval;

/// The Reed–Solomon alphabet used for FRI encoding.
///
/// This domain is fixed to the F128 POLYVAL basis.
///
/// While Binius uses an 32-bit field, which is large enough to handle trace sizes up to 64 GiB of
/// committed data, we adhere to the F128 POLYVAL basis.
pub type FEncode = BinaryField128bPolyval;

type MerkleScheme =
    BinaryMerkleTreeScheme<BinaryField128bPolyval, Groestl256, Groestl256ByteCompression>;

type MerkleProver =
    BinaryMerkleTreeProver<BinaryField128bPolyval, Groestl256, Groestl256ByteCompression>;

type MerkleDigest = <MerkleScheme as MerkleTreeScheme<BinaryField128bPolyval>>::Digest;

type MerkleCommitted = <MerkleProver as MerkleTreeProver<BinaryField128bPolyval>>::Committed;

/// A builder for constructing a Polynomial Commitment Scheme (PCS).
#[derive(Debug)]
pub struct BatchFriPcs {
    /// The Merkle tree prover used to generate the Merkle tree for the PCS.
    merkle_prover: MerkleProver,
    /// The FRI parameters used for polynomial commitment and verification.
    fri_params: FRIParams<BinaryField128bPolyval, BinaryField128bPolyval>,
}

impl BatchFriPcs {
    /// Constructs a new [`PCS`].
    ///
    /// This function initializes the PCS setup, including the Merkle tree prover, FRI parameters,
    /// and commitment metadata for the batch of polynomials being committed.
    ///
    /// # Arguments
    ///
    /// - `security_bits`: The security level of the PCS, expressed in bits.
    /// - `log_inv_rate`: The log inverse rate used in the FRI protocol. Determines the expansion
    ///   factor for the polynomial's degree when encoding.
    /// - `num_vars`: The number of variables in each multilinear polynomial being committed.
    /// - `batch_size`: The number of multilinear polynomials in the batch.
    ///
    /// # Returns
    ///
    /// A fully initialized `PCS` containing the Merkle tree prover and FRI parameters.
    pub fn new(
        security_bits: usize,
        log_inv_rate: usize,
        num_vars: usize,
        batch_size: usize,
    ) -> Self {
        // Initialize the Merkle tree prover with Groestl256-based compression.
        let merkle_prover =
            BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);

        // Extract the Merkle tree scheme for use in FRI parameter construction.
        let merkle_scheme = merkle_prover.scheme();

        // Generate commitment metadata for a batch of `batch_size` polynomials,
        // each with `num_vars` variables.
        let commit_meta = CommitMeta::with_vars(vec![num_vars; batch_size]);

        // Configure the FRI parameters with optimal arity based on the commitment metadata,
        // security level, and log inverse rate.
        let fri_params = make_commit_params_with_optimal_arity::<_, BinaryField128bPolyval, _>(
            &commit_meta,
            merkle_scheme,
            security_bits,
            log_inv_rate,
        )
        .unwrap();

        // Return the fully constructed PCS.
        Self { merkle_prover, fri_params }
    }

    /// Commits a batch of multilinear polynomials to the PCS.
    pub fn commit(
        &self,
        polys: &MultilinearLagrangianPolynomials,
    ) -> CommitOutput<PackedBinaryPolyval1x128b, MerkleDigest, MerkleCommitted> {
        // Construct a packed version of the multilinear polynomials.
        let transparent_multilins: Vec<_> = polys
            .par_iter()
            .map(|poly| {
                MLEDirectAdapter::from(
                    MultilinearExtension::from_values(iso_slice_packed(poly)).unwrap(),
                )
            })
            .collect();

        // Commit to the packed multilinear polynomials.
        commit(&self.fri_params, &self.merkle_prover, &transparent_multilins).unwrap()
    }
}
