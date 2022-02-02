### About the dataset
This fork of LibriMix is an open source dataset for source separation in noisy 
environments. It is derived from LibriSpeech signals (clean subset) and WHAM noise.

## Extensions made
This is a fork from https://github.com/JorisCos/LibriMix, this version of LibriMix has the following extensions/modifications:
- The WHAM! dataset is first *chunked*. That means we concatenate all noise files and cut that into chunks of a given size (default: 20s).
- Instead of generating fully overlapped speech segments with various lengths, all our output clips have the same length (the length of the noise chunks). The speech segments are randomly placed in this clip. If they are longer than the noise clip, they are trimmed on the right side. The `min` and `max` mode don't exist in this fork.
- Various optimizations (mostly in the metadata generation).

### Generating LibriMix
To generate LibriMix, clone the repo and run the setup script. This downloads all datasets, chunks WHAM and creates metadata (if necessary). Then run `./generate_librimix.sh`.
```
git clone https://github.com/hiddefolkertsma/LibriMix
cd LibriMix
./setup.sh storage_dir
./generate_librimix.sh storage_dir
```

Make sure that SoX is installed on your machine.

For Windows :
```
conda install -c groakat sox
```

For Linux :
```
conda install -c conda-forge sox
```

You can either change `storage_dir` and `n_src` by hand in the script or use the command line. By default, LibriMix will be generated for 2 and 3 speakers, at 16Khz, for type `mix_both`.

### Related work
If you wish to implement models based on LibriMix you can checkout 
[Asteroid](https://github.com/mpariente/asteroid) and the 
[recipe](https://github.com/mpariente/asteroid/tree/master/egs/librimix/ConvTasNet)
associated to LibriMix for reproducibility.

Along with LibriMix, SparseLibriMix a dataset aiming towards more realistic, conversation-like scenarios
has been released [here](https://github.com/popcornell/SparseLibriMix).

(contributors: [@JorisCos](https://github.com/JorisCos), [@mpariente](https://github.com/mpariente) and [@popcornell](https://github.com/popcornell) )

### Citing Librimix 

```BibTex
@misc{cosentino2020librimix,
    title={LibriMix: An Open-Source Dataset for Generalizable Speech Separation},
    author={Joris Cosentino and Manuel Pariente and Samuele Cornell and Antoine Deleforge and Emmanuel Vincent},
    year={2020},
    eprint={2005.11262},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
