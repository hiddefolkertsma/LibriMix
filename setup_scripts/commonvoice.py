import argparse
import functools
import os
import shutil

import pandas as pd
import tqdm.contrib.concurrent


def main(args):
    filename = args.file
    folder = os.path.dirname(filename)

    # Make a folder for this split
    split = os.path.splitext(os.path.basename(filename))[0]
    if os.path.exists(os.path.join(folder, f"{split}.csv")):
        print(f"{split}.csv already exists. Skipping.")
        return

    os.makedirs(os.path.join(folder, split), exist_ok=True)

    # Read the metadata file and process it
    metadata = pd.read_table(filename, engine="python")
    result = tqdm.contrib.concurrent.process_map(
        functools.partial(process_row, folder, split),
        [row for _, row in metadata.iterrows()],
        chunksize=10,
        max_workers=os.cpu_count(),
    )

    speaker_ids, paths = zip(*result)
    # Write the LibriMix-compatible output file
    metadata_out = pd.DataFrame()
    metadata_out["speaker_ID"] = speaker_ids
    metadata_out["subset"] = split
    metadata_out["origin_path"] = paths
    metadata_out.to_csv(os.path.join(folder, f"{split}.csv"), index=False)


def process_row(folder, split, row):
    """Moves a clip to its speaker's folder and returns the new path."""
    speaker_id = row["client_id"]
    path = row["path"]
    # Prepare the output path, e.g. "train/speaker_1/clip_1.wav"
    new_path = os.path.join(split, speaker_id, path)
    # Make the speaker's folder
    os.makedirs(os.path.join(folder, split, speaker_id), exist_ok=True)
    # Move the clip to the speaker's folder
    try:
        shutil.move(os.path.join(folder, "clips", path), os.path.join(folder, new_path))
    except shutil.Error:
        # This happens if the file is already in the right place
        # or if the file doesn't exist, in which case we just ignore it
        pass
    return speaker_id, new_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="TSV file to process")
    args = parser.parse_args()
    main(args)
