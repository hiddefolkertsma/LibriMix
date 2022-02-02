#!/bin/bash
set -eu  # Exit on error

storage_dir=$1
librispeech_dir=$storage_dir/LibriSpeech
wham_dir=$storage_dir/wham_noise

function download_librispeech() {
	if ! test -d $librispeech_dir/$1; then
		echo "Downloading LibriSpeech/$1 into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/$1.tar.gz -P $storage_dir
		tar -xzf $storage_dir/$1.tar.gz -C $storage_dir
		rm -rf $storage_dir/$1.tar.gz
	fi
}

function download_wham() {
	if ! test -d $wham_dir; then
		echo "Downloading wham_noise into $storage_dir"
		# If downloading stalls for more than 20s, relaunch from previous state.
		wget -c --tries=0 --read-timeout=20 https://storage.googleapis.com/whisper-public/wham_noise.zip -P $storage_dir
		unzip -qn $storage_dir/wham_noise.zip -d $storage_dir
		rm -rf $storage_dir/wham_noise.zip
	fi
}

download_librispeech dev-clean &
download_librispeech test-clean &
download_librispeech train-clean-100 &
download_librispeech train-clean-360 &
download_librispeech train-other-500 &
download_wham &

wait

# Path to python
python_path=python3

# Augment WHAM noise if necessary
$python_path scripts/augment_train_noise.py --wham_dir $wham_dir

# Chunk WHAM into 20s segments
wham_dir_chunked=$wham_dir/chunked_20s
if ! test -d "$wham_dir_chunked"; then
    echo "Chunked WHAM not found. Creating chunked WHAM..."
	$python_path scripts/chunk_speakers.py $wham_dir $wham_dir_chunked
	$python_path scripts/create_wham_metadata.py --wham_dir $wham_dir_chunked -z 20
else
	echo "Chunked WHAM found in $wham_dir_chunked"
fi

# Chunk LibriSpeech to 6s segments for computing embeddings
librispeech_chunked=$librispeech_dir/chunked
function chunk_librispeech() {
	if ! test -d $librispeech_chunked/$1; then
		echo "Chunking $1 into $librispeech_chunked/$1"
		$python_path scripts/chunk_speakers.py $librispeech_dir/$1 $librispeech_chunked/$1 -z 6
		echo "Done chunking $1"
	else
		echo "Chunked $1 found in $librispeech_chunked/$1"
	fi
}

chunk_librispeech dev-clean &
chunk_librispeech test-clean &
chunk_librispeech train-clean-100 &
chunk_librispeech train-clean-360 &
chunk_librispeech train-other-500 &

wait

# Create metadata
for n_src in 1 2 3; do
	echo "Generating metadata for Libri$n_src"Mix
  	$python_path scripts/create_librimix_metadata.py \
    	--librispeech_dir $librispeech_dir \
    	--librispeech_md_dir metadata/LibriSpeech \
    	--wham_dir $wham_dir_chunked \
    	--wham_md_dir $wham_dir_chunked/metadata \
    	--metadata_outdir metadata/Libri$n_src"Mix" \
    	--n_src $n_src
done
