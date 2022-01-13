import functools
import os
import argparse
import pandas as pd
import random
import tqdm.contrib.concurrent


parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, required=True,
                    help='Path to librispeech root directory')
parser.add_argument('--metadata_dir', type=str, required=True,
                    help='Path to the LibriMix metadata directory')


def main(args):
    # Get LibriSpeech root path
    librispeech_dir = args.librispeech_dir
    # Get metadata directory
    metadata_dir = args.metadata_dir
    choose_embeddings(librispeech_dir, metadata_dir)


def choose_embeddings(librispeech_dir, metadata_dir):
    """ Create embedding utterances for the LibriMix dataset """
    # Get metadata files
    md_filename_list = [
        file for file in os.listdir(metadata_dir) if 'info' not in file
    ]
    # Create all parts of librimix
    for md_filename in md_filename_list:
        print(md_filename)
        csv_path = os.path.join(metadata_dir, md_filename)
        process_metadata_file(csv_path, librispeech_dir)


def process_metadata_file(csv_path, librispeech_dir):
    """ Process a metadata generation file to choose embedding utterances """
    md_file = pd.read_csv(csv_path, engine='python')
    embedding_list = tqdm.contrib.concurrent.process_map(
        functools.partial(process_row, librispeech_dir),
        [row for _, row in md_file.iterrows()],
        chunksize=10,
    )

    md_file['embedding_path'] = embedding_list
    md_file.to_csv(csv_path, index=False)


def process_row(librispeech_dir, row):
    # Get primary speaker file
    primary = row['source_1_path']
    primary_dir, primary_file = os.path.split(primary)
    full_dir = os.path.join(librispeech_dir, primary_dir)
    # Choose a random other utterance from this speaker as the
    # embedding utterance and save its path
    utterances = [
        os.path.join(primary_dir, utt) for utt in os.listdir(full_dir)
        if os.path.isfile(os.path.join(full_dir, utt)) and utt != primary_file
        and utt.endswith('.flac')
    ]
    # TODO check if chosen utterance is not the primary utterance
    # in another row in the metadata file (slow)
    return random.choice(utterances)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)