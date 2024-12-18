use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::compressed::CompressedPoly;

pub mod boolcheck_builder;
pub mod boolcheck_trait;
pub mod package;

#[derive(Clone, Debug, Default)]
pub struct BoolCheck {
    pt: Vec<BinaryField128b>,
    poly: Vec<BinaryField128b>,
    polys: Vec<Vec<BinaryField128b>>,
    ext: Option<Vec<BinaryField128b>>,
    poly_coords: Option<Vec<BinaryField128b>>,
    c: usize,
    challenges: Vec<BinaryField128b>,
    bit_mapping: Vec<u16>,
    eq_sequence: Vec<Vec<BinaryField128b>>,
    round_polys: Vec<CompressedPoly>,
    claim: BinaryField128b,
}
