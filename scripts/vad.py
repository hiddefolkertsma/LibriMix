import webrtcvad
import matplotlib.pyplot as plt
import wave
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class WebRTCVAD:
    def __init__(self, aggressiveness=3, frame_size_ms=30):
        assert aggressiveness in (1, 2, 3)
        assert frame_size_ms in (10, 20, 30)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_size_ms = frame_size_ms


    def label(self, path, kernel_size=3):
        """Computes labels for a .wav file using WebRTCVAD.
        Labels are computed per frame, smoothed and returned
        as sample index ranges."""
        # Read audio and split into frames
        audio_pcm, sr = self._read_wave(path)
        frames = self._frame_generator(audio_pcm, sr)
        # Get the VAD labels for each frame
        labels = [int(self.vad.is_speech(frame.bytes, sr)) for frame in frames]
        # Convolve the labels with a window of ones to smoothen
        if kernel_size:
            labels = np.convolve(labels, np.ones(kernel_size, dtype=int), mode='same')
            labels[labels > 0] = 1
        # Convert the labels to ranges
        return self._label_to_ranges(labels, sr)


    def _label_to_ranges(self, labels, sr):
        """Converts a list of VAD labels per frame to a list of
        sample index ranges during which activity is detected.
        
        Example with frame size 10 and sample rate 1000: 
        [1, 1, 1, 0, 1, 0, 0, 1, 1] -> [[0, 29], [40, 49], [70, 89]]"""
        # Zero-pad each end
        labels = np.concatenate(([0], labels, [0]))
        # Compute diffs, absdiff is 1 where speech/nonspeech segments begin
        absdiff = np.diff(labels)
        # Ranges start and end where absdiff is nonzero.
        ranges = np.nonzero(absdiff)[0].reshape(-1, 2)
        # Frame indices to sample indices
        samples_per_frame = int(self.frame_size_ms * sr / 1000)
        ranges *= samples_per_frame
        # Example frame labels: [[0, 3], [4, 5], [7, 9]]
        # The ranges' second element is the index of the first non-speech frame.
        # We can therefore subtract 1 from it after the multiplication
        # to get the index of the last speech segment frame's last sample.
        ranges[:, -1] -= 1
        return ranges


    def _read_wave(self, path):
        """Reads a .wav file.
        Takes the path, and returns (PCM audio data, sample rate).
        """
        with wave.open(path, 'rb') as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate


    def _frame_generator(self, audio, sample_rate):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        n = int(sample_rate * (self.frame_size_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n


    def _plot_pcm(audio_pcm, sr, vad, frame_size_ms):
        """Plots the waveform and the corresponding VAD labels."""
        samples_per_frame = int(frame_size_ms * sr / 1000)
        signal = np.frombuffer(audio_pcm, "int16")
        plt.plot(signal / (2**15))
        vad = np.repeat(vad, samples_per_frame)
        vad = np.pad(vad, (0, len(signal) - len(vad)), 'constant', constant_values=0)
        plt.step(np.arange(len(signal)), vad)
        plt.show()