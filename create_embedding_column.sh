storage_dir="/media/hidde/Storage/datasets"
librispeech_dir=$storage_dir/LibriSpeech
metadata_dir=metadata/Libri2Mix

python scripts/create_embedding_column.py \
    --librispeech_dir $librispeech_dir \
    --metadata_dir $metadata_dir