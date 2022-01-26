#!/bin/bash
set -eu  # Exit on error

storage_dir=$1
librispeech_dir=$storage_dir/LibriSpeech
wham_dir=$storage_dir/wham_noise/chunked_20s
librimix_outdir=$storage_dir
python_path=python3

for n_src in 2 3; do
    metadata_dir=metadata/Libri$n_src"Mix"
    $python_path scripts/create_librimix_from_metadata.py \
        --librispeech_dir $librispeech_dir \
        --wham_dir $wham_dir \
        --metadata_dir $metadata_dir \
        --librimix_outdir $librimix_outdir \
        --n_src $n_src \
        --freqs 16k \
        --types mix_both
done
