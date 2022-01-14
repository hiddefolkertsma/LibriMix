import functools
import os
import argparse
import pandas as pd
import random
import tqdm.contrib.concurrent
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, required=True,
                    help='Path to librispeech root directory')
parser.add_argument('--metadata_dir', type=str, required=True,
                    help='Path to the LibriMix metadata directory')
parser.add_argument('--rir_dir', type=str, required=True,
                    help='Path to the RIR directory')


def main(args):
    # Get LibriSpeech root path
    librispeech_dir = args.librispeech_dir
    # Get metadata directory
    metadata_dir = args.metadata_dir
    # Get room impulse response directory
    rir_dir = args.rir_dir
    # Get metadata files
    md_filename_list = [
        file for file in os.listdir(metadata_dir) if 'info' not in file
    ]
    # Create additional columns for embedding and RIR paths
    for md_filename in md_filename_list:
        print(md_filename)
        csv_path = os.path.join(metadata_dir, md_filename)
        choose_embeddings(csv_path, librispeech_dir)
        choose_rirs(csv_path, rir_dir)


def choose_embeddings(csv_path, librispeech_dir):
    """ Process a metadata generation file to choose embedding utterances """
    md_file = pd.read_csv(csv_path, engine='python')
    all_primaries = md_file['source_1_path']
    embedding_list = tqdm.contrib.concurrent.process_map(
        functools.partial(process_row, librispeech_dir, all_primaries),
        [row for _, row in md_file.iterrows()],
        chunksize=10,
    )

    md_file['embedding_path'] = embedding_list
    md_file.to_csv(csv_path, index=False)


def process_row(librispeech_dir, all_primaries, row):
    # Get primary speaker file
    primary = row['source_1_path']
    primary_dir, primary_file = os.path.split(primary)
    full_dir = os.path.join(librispeech_dir, primary_dir)
    # Choose a random other utterance from this speaker as the
    # embedding utterance and save its path
    utterances = [
        os.path.join(primary_dir, utt) for utt in os.listdir(full_dir)
        if os.path.isfile(os.path.join(full_dir, utt)) # ignore dirs
        and utt.endswith('.flac')    # ignore the transcript .txt files
        and utt not in all_primaries # don't use any row's primary, incl. current row
    ]

    return random.choice(utterances)


def choose_rirs(csv_path, rir_dir):
    md_file = pd.read_csv(csv_path, engine='python')
    # Read all RIR paths relative to rir_dir
    rir_paths = [
        os.path.relpath(y, rir_dir) for x in os.walk(rir_dir)
        for y in glob.glob(os.path.join(x[0], '*.wav'))
    ]
    # Choose a random unique RIR for each row
    rir_paths = random.sample(rir_paths, md_file.shape[0])
    md_file['rir_path'] = rir_paths
    md_file.to_csv(csv_path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)