storage_dir="/media/hidde/Storage/datasets"
librispeech_dir=$storage_dir/LibriSpeech
rir_dir=$storage_dir/simulated_rirs_16k

for n_src in 2 3; do
  metadata_dir=metadata/Libri$n_src"Mix"
  python scripts/create_additional_columns.py \
    --librispeech_dir $librispeech_dir \
    --metadata_dir $metadata_dir \
    --rir_dir $rir_dir
done