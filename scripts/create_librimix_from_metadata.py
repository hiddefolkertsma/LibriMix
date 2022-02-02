import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import functools
from scipy.signal import resample_poly
import tqdm.contrib.concurrent
from vad import WebRTCVAD

# Eps secures log and division
EPS = 1e-10
# Rate of the sources in LibriSpeech
RATE = 16000
# Frame size in ms and aggressiveness for the VAD
FRAME_SIZE_MS = 30
AGGRESSIVENESS = 3
EXTENSION = 'flac'

parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir',
                    type=str,
                    required=True,
                    help='Path to librispeech root directory')
parser.add_argument('--wham_dir',
                    type=str,
                    required=True,
                    help='Path to wham_noise root directory')
parser.add_argument('--metadata_dir',
                    type=str,
                    required=True,
                    help='Path to the LibriMix metadata directory')
parser.add_argument('--librimix_outdir',
                    type=str,
                    default=None,
                    help='Path to the desired dataset root directory')
parser.add_argument('--n_src',
                    type=int,
                    required=True,
                    help='Number of sources in mixtures')
parser.add_argument('--freqs',
                    nargs='+',
                    default=['8k', '16k'],
                    help=f'Sample rates to be generated')
parser.add_argument('--types',
                    nargs='+',
                    default=['mix_clean', 'mix_both', 'mix_single'],
                    help='--types mix_clean mix_both mix_single ')


def main(args):
    # Get LibriSpeech root path
    librispeech_dir = args.librispeech_dir
    wham_dir = args.wham_dir
    # Get metadata directory
    metadata_dir = args.metadata_dir
    # Get LibriMix root path
    librimix_outdir = args.librimix_outdir
    n_src = args.n_src
    if librimix_outdir is None:
        librimix_outdir = os.path.dirname(metadata_dir)
    librimix_outdir = os.path.join(librimix_outdir, f'Libri{n_src}Mix')
    # Get the desired frequencies
    freqs = args.freqs
    freqs = [freq.lower() for freq in freqs]
    types = args.types
    types = [t.lower() for t in types]
    # Get the number of sources
    create_librimix(librispeech_dir, wham_dir, librimix_outdir, metadata_dir,
                    freqs, n_src, types)


def create_librimix(librispeech_dir, wham_dir, out_dir, metadata_dir, freqs,
                    n_src, types):
    """Generate sources mixtures and saves them in out_dir"""
    # Get metadata files
    md_filename_list = [
        file for file in os.listdir(metadata_dir) if 'info' not in file
        and file.endswith('.csv')
    ]
    # Create all parts of librimix
    for md_filename in md_filename_list:
        csv_path = os.path.join(metadata_dir, md_filename)
        process_metadata_file(csv_path, freqs, n_src, librispeech_dir,
                              wham_dir, out_dir, types)


def process_metadata_file(csv_path, freqs, n_src, librispeech_dir, wham_dir,
                          out_dir, types):
    """Process a metadata generation file to create sources and mixtures"""
    print(f'Processing {csv_path}')
    md_file = pd.read_csv(csv_path, engine='python')
    for freq in freqs:
        # Get the frequency directory path
        freq_path = os.path.join(out_dir, f"{EXTENSION}{freq}")
        # Transform freq = "16k" into 16000
        freq = int(freq.strip('k')) * 1000
        # Subset metadata path
        subset_metadata_path = os.path.join(freq_path, 'metadata')
        os.makedirs(subset_metadata_path, exist_ok=True)
        # Directory where the mixtures and sources will be stored
        dir_name = os.path.basename(csv_path).replace(
            f'libri{n_src}mix_', '').replace('-clean',
                                                '').replace('.csv', '')
        dir_path = os.path.join(freq_path, dir_name)
        # If the files already exist then continue the loop
        if os.path.isdir(dir_path):
            print(f"Directory {dir_path} already exists. "
                    f"Files won't be overwritten")
            continue

        print(f"Creating mixtures and sources from {csv_path} "
                f"in {dir_path}")
        # Create subdir
        if types == ['mix_clean']:
            subdirs = [f's{i + 1}' for i in range(n_src)
                        ] + ['mix_clean', 'embeddings', 'labels']
        else:
            subdirs = [f's{i + 1}' for i in range(n_src)
                        ] + types + ['noise', 'embeddings', 'labels']
        # Create directories accordingly
        for subdir in subdirs:
            os.makedirs(os.path.join(dir_path, subdir))
        # Go through the metadata file
        process_utterances(md_file, librispeech_dir, wham_dir, freq,
                            subdirs, dir_path, subset_metadata_path, n_src)


def process_utterances(md_file, librispeech_dir, wham_dir, freq, subdirs,
                       dir_path, subset_metadata_path, n_src):
    # Dictionary that will contain all metadata
    md_dic = {}
    # Get dir name
    dir_name = os.path.basename(dir_path)
    # Create Dataframes
    for subdir in subdirs:
        if subdir.startswith('mix'):
            md_dic[f'metrics_{dir_name}_{subdir}'] = create_empty_metrics_md(
                n_src, subdir)
            md_dic[f'mixture_{dir_name}_{subdir}'] = create_empty_mixture_md(
                n_src, subdir)

    # Go through the metadata file and generate mixtures
    for results in tqdm.contrib.concurrent.process_map(
            functools.partial(process_utterance, n_src, librispeech_dir,
                              wham_dir, freq, subdirs, dir_path),
        [row for _, row in md_file.iterrows()],
            chunksize=10,
    ):
        for mix_id, snr_list, abs_mix_path, abs_source_path_list, abs_noise_path, length, subdir in results:
            # Add line to the dataframes
            add_to_metrics_metadata(md_dic[f"metrics_{dir_name}_{subdir}"],
                                    mix_id, snr_list)
            add_to_mixture_metadata(md_dic[f'mixture_{dir_name}_{subdir}'],
                                    mix_id, abs_mix_path, abs_source_path_list,
                                    abs_noise_path, length, subdir)

    # Generate VAD labels for primary speaker utterances
    print(f'Generating VAD labels for {md_file.shape[0]} primary speaker utterances')
    mixture_ids = md_file['mixture_ID']
    labels_paths = tqdm.contrib.concurrent.process_map(functools.partial(label_vad, dir_path), mixture_ids, chunksize=100)

    # Save the metadata files
    for md_df in md_dic:
        md_dic[md_df]['labels_path'] = labels_paths
        save_path_mixture = os.path.join(subset_metadata_path, md_df + '.csv')
        md_dic[md_df].to_csv(save_path_mixture, index=False)


def process_utterance(n_src, librispeech_dir, wham_dir, freq, subdirs,
                      dir_path, row):
    res = []
    # Get sources and mixture infos
    mix_id, gain_list, start_list, sources = read_sources(row, n_src, librispeech_dir,
                                              wham_dir)
    # Transform sources
    transformed_sources = transform_sources(sources, freq, gain_list, start_list)
    # Write the sources get their paths
    abs_source_path_list = [
        write_audio(mix_id, source, dir_path, f's{idx+1}', freq)
        for idx, source in enumerate(transformed_sources[:n_src])
    ]
    abs_noise_path = write_audio(mix_id, transformed_sources[-1], dir_path,
                                 'noise', freq)

    # Mixtures are different depending on the subdir
    for subdir in subdirs:
        if subdir == 'mix_clean':
            sources_to_mix = transformed_sources[:n_src]
        elif subdir == 'mix_both':
            sources_to_mix = transformed_sources
        elif subdir == 'mix_single':  # single speaker + noise
            sources_to_mix = [transformed_sources[0], transformed_sources[-1]]
        else:
            continue

        # Mix sources
        mixture = mix(sources_to_mix)
        # Write mixture and get its path
        abs_mix_path = write_audio(mix_id, mixture, dir_path, subdir, freq)
        length = len(mixture)
        # Compute SNR
        snr_list = compute_snr_list(mixture, sources_to_mix)
        res.append((mix_id, snr_list, abs_mix_path, abs_source_path_list,
                    abs_noise_path, length, subdir))

    return res


def label_vad(dir_path, mixture_id):
    vad = WebRTCVAD(AGGRESSIVENESS, FRAME_SIZE_MS)
    # Get the path to the primary speaker's utterance
    utt_path = os.path.join(dir_path, 's1', f'{mixture_id}.{EXTENSION}')
    # Get labels and save
    labels = vad.label(utt_path)
    labels_path = os.path.join(dir_path, 'labels', f'{mixture_id}.npy')
    np.save(labels_path, labels)
    return labels_path


def create_empty_metrics_md(n_src, subdir):
    """Create the metrics dataframe."""
    metrics_dataframe = pd.DataFrame()
    metrics_dataframe['mixture_ID'] = {}
    if subdir == 'mix_clean':
        for i in range(n_src):
            metrics_dataframe[f"source_{i + 1}_SNR"] = {}
    elif subdir == 'mix_both':
        for i in range(n_src):
            metrics_dataframe[f"source_{i + 1}_SNR"] = {}
        metrics_dataframe[f"noise_SNR"] = {}
    elif subdir == 'mix_single':
        metrics_dataframe["source_1_SNR"] = {}
        metrics_dataframe[f"noise_SNR"] = {}
    return metrics_dataframe


def create_empty_mixture_md(n_src, subdir):
    """Create the mixture dataframe."""
    mixture_dataframe = pd.DataFrame()
    mixture_dataframe['mixture_ID'] = {}
    mixture_dataframe['mixture_path'] = {}
    if subdir == 'mix_clean':
        for i in range(n_src):
            mixture_dataframe[f"source_{i + 1}_path"] = {}
    elif subdir == 'mix_both':
        for i in range(n_src):
            mixture_dataframe[f"source_{i + 1}_path"] = {}
        mixture_dataframe[f"noise_path"] = {}
    elif subdir == 'mix_single':
        mixture_dataframe["source_1_path"] = {}
        mixture_dataframe[f"noise_path"] = {}
    mixture_dataframe['length'] = {}
    return mixture_dataframe


def read_sources(row, n_src, librispeech_dir, wham_dir):
    """Get sources and info to mix the sources."""
    # Get info about the mixture
    mixture_id = row['mixture_ID']
    # Get a Python list [Source_1_path, Source_2_path, ..., Source_n_path]
    sources_path_list = get_list_from_csv(row, 'source_path', n_src)
    # Same for the gains: [Source_1_gain, Source_2_gain, ..., Source_n_gain]
    gain_list = get_list_from_csv(row, 'source_gain', n_src)
    # And the starting indices
    start_list = get_list_from_csv(row, 'source_start', n_src)
    # This is a list of the actual loaded sources
    sources_list = []
    # Read the files to make the mixture
    for sources_path in sources_path_list:
        sources_path = os.path.join(librispeech_dir, sources_path)
        source, _ = sf.read(sources_path, dtype='float32')
        sources_list.append(source)
    # Read the noise
    noise_path = os.path.join(wham_dir, row['noise_path'])
    noise, _ = sf.read(noise_path, dtype='float32')
    # if noises have 2 channels take the first
    if len(noise.shape) > 1:
        noise = noise[:, 0]
    sources_list.append(noise)
    gain_list.append(row['noise_gain'])

    return mixture_id, gain_list, start_list, sources_list


def get_list_from_csv(row, column, n_src):
    """Transform a list in the .csv into Python list."""
    python_list = []
    for i in range(n_src):
        current_column = column.split('_')
        current_column.insert(1, str(i + 1))
        current_column = '_'.join(current_column)
        python_list.append(row[current_column])
    return python_list


def extend_noise(noise, max_length):
    """Concatenate noise using Hanning window."""
    noise_ex = noise
    window = np.hanning(RATE + 1)
    # Increasing window
    i_w = window[:len(window) // 2 + 1]
    # Decreasing window
    d_w = window[len(window) // 2::-1]
    # Extend until max_length is reached
    while len(noise_ex) < max_length:
        noise_ex = np.concatenate(
            (noise_ex[:len(noise_ex) - len(d_w)],
             np.multiply(noise_ex[len(noise_ex) - len(d_w):], d_w) +
             np.multiply(noise[:len(i_w)], i_w), noise[len(i_w):]))
    noise_ex = noise_ex[:max_length]
    return noise_ex


def transform_sources(sources_list, freq, gain_list, start_list):
    """Transform sources before mixing."""
    # Normalize sources (apply gain to them)
    sources_list_norm = loudness_normalize(sources_list, gain_list)
    # Resample the sources
    sources_list_resampled = resample_list(sources_list_norm, freq)
    # Pad the utterances:
    reshaped_sources = pad_lengths(sources_list_resampled, start_list)
    return reshaped_sources


def loudness_normalize(sources_list, gain_list):
    """Normalize sources loudness."""
    # Create the list of normalized sources
    normalized_list = []
    for i, source in enumerate(sources_list):
        normalized_list.append(source * gain_list[i])
    return normalized_list


def resample_list(sources_list, freq):
    """Resample the source list to the desired frequency."""
    # Create the resampled list
    resampled_list = []
    # Resample each source
    for source in sources_list:
        resampled_list.append(resample_poly(source, freq, RATE))
    return resampled_list


def pad_lengths(source_list, start_list):
    """Pad the utterances with their randomly generated padding from the metadata."""
    sources_list_padded = []
    n_src = len(start_list)
    length = len(source_list[-1]) # length of background noise

    for i, source in enumerate(source_list[:n_src]):
        front = int(start_list[i])
        back = length - front - len(source)
        if len(source) < length:
            sources_list_padded.append(np.pad(source, (front, back), mode='constant'))
        else:
            sources_list_padded.append(source[:length]) # trim
    sources_list_padded.append(source_list[-1])

    assert(len(sources_list_padded[0]) == length)
    return sources_list_padded


def write_audio(mix_id, signal, dir_path, subdir, freq):
    """Write signal to a file and return its path."""
    ex_filename = f"{mix_id}.{EXTENSION}"
    save_path = os.path.join(dir_path, subdir, ex_filename)
    abs_save_path = os.path.abspath(save_path)
    sf.write(abs_save_path, signal, freq)
    return abs_save_path


def mix(sources_list):
    """Mix the sources in `sources_list` together."""
    # Initialize mixture
    mixture = np.zeros_like(sources_list[0])
    for source in sources_list:
        mixture += source
    return mixture


def compute_snr_list(mixture, sources_list):
    """Compute the SNR on the mixture mode min."""
    snr_list = []
    # Compute SNR for min mode
    for source in sources_list:
        noise_min = mixture - source
        snr_list.append(snr_xy(source, noise_min))
    return snr_list


def snr_xy(x, y):
    return 10 * np.log10(np.mean(x**2) / (np.mean(y**2) + EPS) + EPS)


def add_to_metrics_metadata(metrics_df, mixture_id, snr_list):
    """Add a new line to `metrics_df`."""
    row_metrics = [mixture_id] + snr_list
    metrics_df.loc[len(metrics_df)] = row_metrics


def add_to_mixture_metadata(mix_df, mix_id, abs_mix_path, abs_sources_path,
                            abs_noise_path, length, subdir):
    """Add a new line to `mixture_df`."""
    sources_path = abs_sources_path
    noise_path = [abs_noise_path]
    if subdir == 'mix_clean':
        noise_path = []
    elif subdir == 'mix_single':
        sources_path = [abs_sources_path[0]]
    row_mixture = [mix_id, abs_mix_path] + sources_path + noise_path + [length]
    mix_df.loc[len(mix_df)] = row_mixture


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
