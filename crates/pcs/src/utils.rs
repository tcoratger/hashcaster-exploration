use binius_field::{BinaryField128bPolyval, PackedBinaryPolyval1x128b, PackedField};
use hashcaster_primitives::binary_field::BinaryField128b;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

fn iso(value: &BinaryField128b) -> BinaryField128bPolyval {
    BinaryField128bPolyval::from(value.into_inner())
}

pub(crate) fn iso_slice_packed(values: &[BinaryField128b]) -> Vec<PackedBinaryPolyval1x128b> {
    values
        .par_chunks(PackedBinaryPolyval1x128b::WIDTH)
        .map(|scalars| PackedBinaryPolyval1x128b::from_scalars(scalars.iter().map(iso)))
        .collect()
}
