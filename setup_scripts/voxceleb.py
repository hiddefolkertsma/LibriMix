import argparse
import os
from typing import Iterator, Tuple

import pandas as pd
from tqdm import tqdm

def main(args):
    path = args.path
    splits = ['test', 'dev']

    for split in splits:
        _, ext, _ = next(os.walk(os.path.join(path, split)))
        create_metadata(path, split, ext[0])

def create_metadata(path, split, ext):
    metadata = pd.DataFrame()
    src = os.path.join(path, split, ext) # e.g. <VoxCeleb root>/test/wav
    
    # Get speaker, path pairs
    paths = list(tqdm(iter_speakers(src, ext), desc='walk src'))
    speaker_ids, paths = zip(*paths)

    # Write to metadata file
    metadata['speaker_ID'] = speaker_ids
    metadata['subset'] = split
    metadata['origin_path'] = paths
    metadata.to_csv(os.path.join(path, f'{split}.csv'), index=False)

def iter_speakers(src: str, ext: str) -> Iterator[Tuple[str, str]]:
    for ent in os.scandir(src):
        speaker = ent.name
        for root, _, names in os.walk(ent.path):
            for name in names:
                if not name.endswith(ext):
                    continue
                path = os.path.join(root, name)
                yield speaker, path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to VoxCeleb1/2 folder containing `dev` and `test`")
    args = parser.parse_args()
    main(args)
