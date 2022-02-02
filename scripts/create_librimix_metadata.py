import argparse
import functools
import os
import random
import warnings
import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
import tqdm.contrib.concurrent

# Global parameters
# eps secures log and division
EPS = 1e-10
# max amplitude in sources and mixtures
MAX_AMP = 0.9
# In LibriSpeech all the sources are at 16K Hz
RATE = 16000
# We will randomize loudness between this range
MIN_LOUDNESS = -33
MAX_LOUDNESS = -25

# A random seed is used for reproducibility
random.seed(72)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, required=True,
                    help='Path to librispeech root directory')
parser.add_argument('--librispeech_md_dir', type=str, required=True,
                    help='Path to librispeech metadata directory')
parser.add_argument('--wham_dir', type=str, required=True,
                    help='Path to wham root directory')
parser.add_argument('--wham_md_dir', type=str, required=True,
                    help='Path to wham metadata directory')
parser.add_argument('--metadata_outdir', type=str, default=None,
                    help='Where librimix metadata files will be stored.')
parser.add_argument('--n_src', type=int, required=True,
                    help='Number of sources desired to create the mixture')


def main(args):
    librispeech_dir = args.librispeech_dir
    librispeech_md_dir = args.librispeech_md_dir
    wham_dir = args.wham_dir
    wham_md_dir = args.wham_md_dir
    n_src = args.n_src
    # Create Librimix metadata directory
    md_dir = args.metadata_outdir
    if md_dir is None:
        root = os.path.dirname(librispeech_dir)
        md_dir = os.path.join(root, f'LibriMix/metadata')
    os.makedirs(md_dir, exist_ok=True)
    create_librimix_metadata(librispeech_dir, librispeech_md_dir, wham_dir,
                             wham_md_dir, md_dir, n_src)


def create_librimix_metadata(librispeech_dir, librispeech_md_dir, wham_dir,
                             wham_md_dir, md_dir, n_src):
    """Generate LibriMix metadata using LibriSpeech and WHAM! metadata."""
    # Dataset name
    dataset = f'libri{n_src}mix'
    # List metadata files in LibriSpeech
    librispeech_md_files = os.listdir(librispeech_md_dir)
    # List metadata files in wham_noise
    wham_md_files = os.listdir(wham_md_dir)
    # If you wish to ignore some metadata files add their name here
    # Example : to_be_ignored = ['dev-other.csv']
    to_be_ignored = []

    check_already_generated(md_dir, dataset, to_be_ignored, librispeech_md_files)
    # Go through each metadata file and create metadata accordingly
    for librispeech_md_file in librispeech_md_files:
        if not librispeech_md_file.endswith('.csv'):
            print(f"{librispeech_md_file} is not a csv file, continue.")
            continue
        # Get the name of the corresponding noise md file
        try:
            wham_md_file = [f for f in wham_md_files if
                            f.startswith(librispeech_md_file.split('-')[0])][0]
        except IndexError:
            print('WHAM! metadata is missing. You can either generate the '
                  'missing WHAM! files or add the LibriSpeech metadata to '
                  'the to_be_ignored list')
            break

        # Open .csv files from LibriSpeech
        librispeech_md = pd.read_csv(os.path.join(
            librispeech_md_dir, librispeech_md_file), engine='python')
        # Open .csv files from wham_noise
        wham_md = pd.read_csv(os.path.join(
            wham_md_dir, wham_md_file), engine='python')
        # Filenames
        save_path = os.path.join(md_dir,
                                 '_'.join([dataset, librispeech_md_file]))
        info_name = '_'.join([dataset, librispeech_md_file.strip('.csv'),
                              'info']) + '.csv'
        info_save_path = os.path.join(md_dir, info_name)
        print(f"Creating {os.path.basename(save_path)} file in {md_dir}")
        # Create dataframe
        mixtures_md, mixtures_info = create_librimix_df(
            librispeech_md, librispeech_dir, wham_md, wham_dir,
            n_src)
        # Round number of files
        mixtures_md = mixtures_md[:len(mixtures_md) // 100 * 100]
        mixtures_info = mixtures_info[:len(mixtures_info) // 100 * 100]

        # Save csv files
        mixtures_md.to_csv(save_path, index=False)
        mixtures_info.to_csv(info_save_path, index=False)


def check_already_generated(md_dir, dataset, to_be_ignored,
                            librispeech_md_files):
    """Check which metadata files have already been generated."""
    already_generated = os.listdir(md_dir)
    for generated in already_generated:
        if generated.startswith(f"{dataset}") and 'info' not in generated:
            if 'train-clean-100' in generated:
                to_be_ignored.append('train-clean-100.csv')
            elif 'train-clean-360' in generated:
                to_be_ignored.append('train-clean-360.csv')
            elif 'train-other-500' in generated:
                to_be_ignored.append('train-other-500.csv')
            elif 'dev' in generated:
                to_be_ignored.append('dev-clean.csv')
            elif 'test' in generated:
                to_be_ignored.append('test-clean.csv')
            print(f"{generated} already exists in "
                  f"{md_dir} it won't be overwritten")
    for element in to_be_ignored:
        librispeech_md_files.remove(element)


def create_librimix_df(librispeech_md_file, librispeech_dir,
                       wham_md_file, wham_dir, n_src):
    """Generate LibriMix dataframe from LibriSpeech and WHAM metadata files."""

    # Generate pairs of sources and accompanying noise to mix
    pairs = generate_pairs(librispeech_md_file, wham_md_file, n_src)

    # Compute the mixture metadata for each pair
    result = tqdm.contrib.concurrent.process_map(
            functools.partial(process_pair, librispeech_md_file, librispeech_dir,
                       wham_md_file, wham_dir, n_src), pairs, chunksize=10)

    # Create mixture dataframes with the raw metadata lists
    source_infos, gain_lists, clipped = zip(*result)
    mixtures_md, mixtures_info = make_metadata_dataframe(source_infos, gain_lists, n_src)
    print(f"Among {len(mixtures_md)} mixtures, {np.sum(clipped)} clipped.")

    return mixtures_md, mixtures_info


def process_pair(librispeech_md_file, librispeech_dir,
                       wham_md_file, wham_dir, n_src, pair):
    """Process a pair of sources to mix."""
    utt_pair, noise = pair # Indices of the utterances and the noise
    # Read the utterance files and get some metadata
    source_info, source_list = read_utterances(
        librispeech_md_file, utt_pair, librispeech_dir)
    # Add the noise
    source_info, source_list = add_noise(
        wham_md_file, wham_dir, noise, source_list, source_info)
    # Compute initial loudness, randomize loudness and normalize sources
    loudness, _, source_list_norm = set_loudness(source_list)
    # Randomly place the speech clips in the mixture
    source_info, source_list_pad = randomly_pad(source_list_norm, source_info, n_src)
    # Do the mixture
    mixture = mix(source_list_pad)
    # Check the mixture for clipping and renormalize if necessary
    # (we pass source_list_norm here because we don't want the zero padding
    # to influence the loudness)
    renormalize_loudness, did_clip = check_for_clipping(mixture,
                                                        source_list_norm)
    # Compute gain
    gain_list = compute_gain(loudness, renormalize_loudness)
    
    return source_info, gain_list, did_clip


def generate_pairs(librispeech_md_file, wham_md_file, n_src):
    """Choose pairs of sources for the mixtures."""
    # Initialize list for pairs sources
    utt_pairs = []
    noises = []
    # In train sets utterances are only used once
    if 'train' in librispeech_md_file.iloc[0]['subset']:
        utt_pairs = generate_utt_pairs(librispeech_md_file, utt_pairs, n_src)
        noises = choose_noises(utt_pairs, noises, wham_md_file)
    # Otherwise we want 3000 mixtures
    else:
        while len(utt_pairs) < 3000:
            utt_pairs = generate_utt_pairs(librispeech_md_file, utt_pairs, n_src)
            noises = choose_noises(utt_pairs, noises, wham_md_file)
            utt_pairs, noises = remove_duplicates(utt_pairs, noises)
            if n_src == 1: # Can't remove duplicates with 1 source
                break
        utt_pairs = utt_pairs[:3000]
        noises = noises[:len(utt_pairs)]

    return list(zip(utt_pairs, noises))


def generate_utt_pairs(librispeech_md_file, utt_pairs, n_src):
    """Generate pairs of utterances for the mixtures."""
    # Create a dict of speakers
    utt_dict = {}
    # Maps from speaker ID to list of all utterance indices in the metadata file
    speakers = list(librispeech_md_file["speaker_ID"].unique())
    for speaker in speakers:
        utt_indices = librispeech_md_file.index[librispeech_md_file["speaker_ID"] == speaker]
        utt_dict[speaker] = list(utt_indices)
    
    while len(speakers) >= n_src:
        # Select random speakers
        selected = random.sample(speakers, n_src)
        # Select random utterance from each speaker
        utt_list = []
        for speaker in selected:
            utt = random.choice(utt_dict[speaker])
            utt_list.append(utt)
            utt_dict[speaker].remove(utt)
            if not utt_dict[speaker]: # no more utts for this speaker
                speakers.remove(speaker)
        utt_pairs.append(utt_list)

    return utt_pairs


def choose_noises(pairs, noise_pairs, wham_md_file):
    """Choose noise sources for the mixtures."""
    # Initially take not augmented data
    md = wham_md_file[wham_md_file['augmented'] == False]
    # If there are more mixtures than noises then use augmented data
    if len(pairs) > len(md):
        md = wham_md_file
    # If there are still more mixtures than noises, repeat the elements
    # in the noise list as necessary (this is ceil division)
    repeat = -(len(pairs) // -len(md))
    # For k pairs of utterances, pick k noise clips
    noise_pairs = random.sample(list(md.index) * repeat, len(pairs))

    return noise_pairs


def remove_duplicates(utt_pairs, noise_pairs):
    """"Remove duplicate pairs."""
    # Look for identical mixtures (O(n²))
    for i, (pair, pair_noise) in enumerate(zip(utt_pairs, noise_pairs)):
        for j, (du_pair, du_pair_noise) in enumerate(
                zip(utt_pairs, noise_pairs)):
            # sort because [s1,s2] = [s2,s1]
            if sorted(pair) == sorted(du_pair) and i != j:
                utt_pairs.remove(du_pair)
                noise_pairs.remove(du_pair_noise)
    return utt_pairs, noise_pairs


def read_utterances(librispeech_md_file, pair, librispeech_dir):
    """Load the utterances as audio files and get utterance metadata."""
    # Read lines corresponding to pair
    utt_md = librispeech_md_file.iloc[pair]

    utterance_list = []
    id_list = []
    for _, utt in utt_md.iterrows():
        # Read the file
        abs_path = os.path.join(librispeech_dir, utt['origin_path'])
        signal, _ = sf.read(abs_path, dtype='float32')
        utterance_list.append(signal)
        # Save its id
        id_list.append(os.path.split(utt['origin_path'])[1].strip('.flac'))
    
    # Metadata for the mixture
    utterances_info = {
        'mixture_id': "_".join(id_list),
        'speaker_id_list': list(utt_md['speaker_ID']),
        'sex_list': list(utt_md['sex']),
        'path_list': list(utt_md['origin_path'])
    }

    return utterances_info, utterance_list


def add_noise(wham_md_file, wham_dir, noise_idx, source_list, source_info):
    """Read noise file and add it to source_list."""
    # Get the row corresponding to the index
    noise = wham_md_file.iloc[noise_idx]
    # Get the noise path
    try:
        noise_path = os.path.join(wham_dir, noise['origin_path'].values[0])
    except AttributeError:
        noise_path = os.path.join(wham_dir, noise['origin_path'])
    # Read the noise
    n, _ = sf.read(noise_path, dtype='float32')
    # Keep the first channel
    if len(n.shape) > 1:
        n = n[:, 0]

    # Shorten any utterances that are longer than the noise
    for i, _ in enumerate(source_list):
        source_list[i] = source_list[i][:len(n)]
    source_list.append(n)

    # Get relative path
    try:
        source_info['noise_path'] = noise['origin_path'].values[0]
    except AttributeError:
        source_info['noise_path'] = noise['origin_path']
    return source_info, source_list


def randomly_pad(sources_list, sources_info, n_src):
    """Randomly zero-pad the sources left and right to the length of the noise clip.
    This places them somewhere in the mixture at random."""
    noise = sources_list[-1]
    length = len(noise) # clip length
    sources_padded = []
    sources_info['start_list'] = []

    # pad all speaker utts to this length
    for source in sources_list[:n_src]:
        to_pad = length - len(source) # total padding
        front = np.random.randint(0, to_pad + 1) # random padding before
        back = to_pad - front
        sources_padded.append(np.pad(source, (front, back), mode='constant'))
        sources_info['start_list'].append(front)
    
    return sources_info, sources_padded


def set_loudness(sources_list):
    """Compute original loudness and normalize them randomly."""
    # Initialize loudness
    loudness_list = []
    # In LibriSpeech all sources are at 16KHz hence the meter
    meter = pyln.Meter(RATE)
    # Randomize sources loudness
    target_loudness_list = []
    sources_list_norm = []

    # Normalize loudness
    for i in range(len(sources_list)):
        # Compute initial loudness
        loudness_list.append(meter.integrated_loudness(sources_list[i]))
        # Pick a random loudness
        target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # Noise has a different loudness
        if i == len(sources_list) - 1:
            target_loudness = random.uniform(MIN_LOUDNESS - 5,
                                             MAX_LOUDNESS - 5)
        # Normalize source to target loudness

        with warnings.catch_warnings():
            # We don't want to pollute stdout, but we don't want to ignore
            # other warnings.
            warnings.simplefilter("ignore")
            src = pyln.normalize.loudness(sources_list[i], loudness_list[i],
                                          target_loudness)
        # If source clips, renormalize
        if np.max(np.abs(src)) >= 1:
            src = sources_list[i] * MAX_AMP / np.max(np.abs(sources_list[i]))
            target_loudness = meter.integrated_loudness(src)
        # Save scaled source and loudness.
        sources_list_norm.append(src)
        target_loudness_list.append(target_loudness)
    return loudness_list, target_loudness_list, sources_list_norm


def mix(sources_list):
    """Mix the sources."""
    mixture = np.zeros_like(sources_list[0])
    for i in range(len(sources_list)):
        mixture += sources_list[i]
    return mixture


def check_for_clipping(mixture, sources_list_norm):
    """Check the mixture for clipping and re-normalize if needed."""
    # Initialize renormalized sources and loudness
    renormalize_loudness = []
    clip = False
    # Recreate the meter
    meter = pyln.Meter(RATE)
    # Check for clipping in mixtures
    if np.max(np.abs(mixture)) > MAX_AMP:
        clip = True
        weight = MAX_AMP / np.max(np.abs(mixture))
    else:
        weight = 1
    # Renormalize
    for i in range(len(sources_list_norm)):
        new_loudness = meter.integrated_loudness(sources_list_norm[i] * weight)
        renormalize_loudness.append(new_loudness)
    return renormalize_loudness, clip


def compute_gain(loudness, renormalize_loudness):
    """Compute the gain between the original and target loudness."""
    gain = []
    for i in range(len(loudness)):
        delta_loudness = renormalize_loudness[i] - loudness[i]
        gain.append(np.power(10.0, delta_loudness / 20.0))
    return gain


def make_metadata_dataframe(source_infos, gain_lists, n_src):
    """Create the Pandas dataframes with the metadata."""
    # Create a dataframe that will be used to generate sources and mixtures
    mixtures_md = pd.DataFrame()
    # Create a dataframe with additional info.
    mixtures_info = pd.DataFrame()
    # Add the metadata to the dataframe
    mixture_ids = [md['mixture_id'] for md in source_infos]
    mixtures_md['mixture_ID'] = mixture_ids
    mixtures_md['primary_speaker'] = [md['speaker_id_list'][0] for md in source_infos]
    mixtures_info['mixture_ID'] = mixture_ids
    for i in range(n_src):
        mixtures_md[f'source_{i + 1}_path'] = [md['path_list'][i] for md in source_infos]
        mixtures_md[f'source_{i + 1}_start'] = [md['start_list'][i] for md in source_infos]
        mixtures_md[f'source_{i + 1}_gain'] = [gain_list[i] for gain_list in gain_lists]
        mixtures_info[f'speaker_{i + 1}_ID'] = [md['speaker_id_list'][i] for md in source_infos]
        mixtures_info[f'speaker_{i + 1}_sex'] = [md['sex_list'][i] for md in source_infos]
    mixtures_md['noise_path'] = [md['noise_path'] for md in source_infos]
    mixtures_md['noise_gain'] = [gain_list[-1] for gain_list in gain_lists]
    
    return mixtures_md, mixtures_info


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
