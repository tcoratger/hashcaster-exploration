use crate::PackedSubfield;
use binius_field::{
    as_packed_field::PackScalar, BinaryField128b, BinaryField128bPolyval, Field, PackedField,
};
use hashcaster_primitives::binary_field::BinaryField128b as F128;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

pub fn iso<F: From<BinaryField128b>>(value: &F128) -> F {
    F::from(BinaryField128b::from(BinaryField128bPolyval::from(value.into_inner())))
}

pub fn iso_slice<F: From<BinaryField128b>>(values: &[F128]) -> Vec<F> {
    values.iter().map(iso).collect()
}

pub fn iso_slice_packed<U: PackScalar<F>, F: Field + From<BinaryField128b>>(
    values: &[F128],
) -> Vec<PackedSubfield<U, F, F>> {
    values
        .par_chunks(PackedSubfield::<U, F, F>::WIDTH)
        .map(|scalars| PackedSubfield::<U, F, F>::from_scalars(scalars.iter().map(iso)))
        .collect()
}
