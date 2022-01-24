from multiprocessing.pool import ThreadPool
from queue import Queue
from typing import Any, Iterator, Tuple
import argparse
import itertools
import os
import threading

from tqdm import tqdm
import torch
import torchaudio

EXT = '.flac'
EXTENSIONS = (".wav", ".flac", ".mp3", ".sph")
STAGE_CPUS = os.cpu_count() // 2
SAMPLERATE = 16000
CHUNK_SIZE = SAMPLERATE * 10
NORM_PEAK  = 0.9

def queue_iterator(q: Queue) -> Iterator[Any]:
    while True:
        item = q.get()
        if item is StopIteration:
            break
        yield item

def dst_iterator(inp: Iterator[Any], dst: str='') -> Iterator[Tuple[str, Any]]:
    counter = None
    last_speaker = None
    for speaker, add, item in inp:
        if speaker != last_speaker:
            counter = itertools.count(start=1)
            last_speaker = speaker
        nonce = next(counter)
        yield f"{dst}/{speaker}/{nonce}{EXT}", add, item

def iter_speakers(src: str) -> Iterator[Tuple[str, str]]:
    for ent in os.scandir(src):
        speaker = ent.name
        for root, _, names in os.walk(ent.path):
            for name in names:
                if not name.endswith(EXTENSIONS):
                    continue
                path = os.path.join(root, name)
                yield speaker, path

def load_one(args: Tuple[str, str]) -> Tuple[str, torch.Tensor]:
    speaker, path = args
    samples, sr = torchaudio.load(path)
    assert(sr == SAMPLERATE)
    return (speaker, samples[0]) # only take first channel

def load_audio(src: str, q: Queue, *, workers: int=None):
    paths = list(tqdm(iter_speakers(src), desc='walk src'))
    count = len(paths)
    try:
        with ThreadPool(workers) as pool, tqdm(desc='load', total=count) as pbar:
            q.put(count)
            iterator = pool.imap(load_one, paths, chunksize=1)
            for speaker, samples in iterator:
                q.put((speaker, samples))
                pbar.update(1)
            pool.close()
            pool.join()
    finally:
        q.put(StopIteration)

def chunk_audio(inp: Queue, out: Queue, *, chunk_size: int=SAMPLERATE * 10):
    try:
        count = inp.get()
        last_speaker = None
        with tqdm(desc='chunk', total=count) as pbar, torch.no_grad():
            out.put(count)
            acc = torch.zeros(chunk_size)
            num = pos = 0
            for speaker, x in queue_iterator(inp):
                if speaker != last_speaker:
                    last_speaker = speaker
                    num = pos = 0
                    pbar.set_postfix({'speaker': speaker})
                x.mul_(NORM_PEAK / x.max().abs())
                num += 1
                while len(x) > 0:
                    need = chunk_size - pos
                    actual = min(need, len(x))
                    acc[pos:pos+actual] = x[:actual]
                    if actual == need:
                        out.put((speaker, num, acc))
                        acc = torch.zeros(chunk_size)
                        num = pos = 0
                    else:
                        pos += actual
                    x = x[actual:]
                pbar.update(1)
    finally:
        out.put(StopIteration)

def save_one(args: Tuple[str, int, torch.Tensor]):
    path, add, samples = args
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, samples.unsqueeze(0), SAMPLERATE)
    return add

def save_audio(dst: str, q: Queue, *, prefix='', workers: int=None):
    count = q.get()
    wrapped_iter = dst_iterator(queue_iterator(q), dst=dst)
    with ThreadPool(workers) as pool:
        with tqdm(desc='save', total=count) as pbar:
            for add in pool.imap_unordered(save_one, wrapped_iter, chunksize=1):
                pbar.update(add)
        pool.close()
        pool.join()

def merge_tree(src: str, dst: str, *, chunk_size: int, prefix: str=''):
    os.makedirs(dst, exist_ok=True)

    load_queue = Queue(STAGE_CPUS * 4)
    save_queue = Queue(STAGE_CPUS * 4)

    load = threading.Thread(target=load_audio, args=(src, load_queue), kwargs={'workers': STAGE_CPUS}, daemon=True)
    save = threading.Thread(target=save_audio, args=(dst, save_queue), kwargs={'workers': STAGE_CPUS}, daemon=True)
    load.start()
    save.start()

    chunk_audio(load_queue, save_queue, chunk_size=chunk_size)

    load.join()
    save.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='source directory')
    parser.add_argument('dst', type=str, help='destination directory')
    parser.add_argument('-z',  type=float, help='chunk size (seconds)', default=10.0)
    args = parser.parse_args()

    merge_tree(args.src, args.dst, chunk_size=int(SAMPLERATE * args.z))
