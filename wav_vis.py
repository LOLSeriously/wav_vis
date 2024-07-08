#!/usr/bin/env python3

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, spectrogram, hilbert
from scipy.fft import fft
import numpy as np
import os
import sys
from matplotlib.widgets import Button
import re
import argparse
import librosa
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)]
class WaveformViewer:
    def __init__(self, directory, plot_types, files_per_page):
        self.directory = directory
        self.wav_files = sorted([f for f in os.listdir(directory) if f.endswith('.wav')], key=natural_sort_key)
        self.current_page = 0
        self.waveforms_per_page = files_per_page
        self.plot_types = plot_types

        num_plots = len(plot_types)
        self.fig, self.axes = plt.subplots(files_per_page, num_plots, figsize=(6*num_plots, 2*files_per_page))
        if num_plots == 1 or files_per_page == 1:
            self.axes = self.axes.reshape(files_per_page, -1)
        self.fig.canvas.manager.set_window_title('Audio File Viewer')
        
        # Navigation buttons
        self.prev_button_ax = self.fig.add_axes([0.2, 0.0005, 0.1, 0.03])  #horiz position , width, height
        self.next_button_ax = self.fig.add_axes([0.7, 0.0005, 0.1, 0.03])

        self.prev_button = Button(self.prev_button_ax, 'Previous')
        self.next_button = Button(self.next_button_ax, 'Next')

        self.prev_button.on_clicked(self.prev_page)
        self.next_button.on_clicked(self.next_page)

        # Connect the resize event
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)

        self.update_plot()

    def plot_wav(self, filename, axes):
        try:
            sample_rate, data = wavfile.read(filename)
        except Exception as e:
            print(f"Error reading file: {filename}. Error: {str(e)}")
            return

        if len(data.shape) > 1:
            data = data[:, 0]
        
        # Ensure data is in float format and normalized
        data = data.astype(np.float32) / np.max(np.abs(data))
        
        time = np.arange(len(data)) / sample_rate

        plot_functions = {
            'waveform': self.plot_waveform,
            'spectrum': self.plot_spectrum,
            'spectrogram': self.plot_spectrogram,
            'mfcc': self.plot_mfcc,
            'phase': self.plot_phase,
            'envelope': self.plot_envelope,
            'harmonic': self.plot_harmonic
        }

        for ax, plot_type in zip(axes, self.plot_types):
            try:
                plot_functions[plot_type](ax, data, sample_rate, time, filename)
            except Exception as e:
                print(f"Error plotting {plot_type} for file {filename}: {str(e)}")
                ax.text(0.5, 0.5, f"Error plotting {plot_type}", ha='center', va='center')

    def plot_waveform(self, ax, data, sample_rate, time, filename):
        ax.plot(time, data)
        ax.set_title(f'Waveform: {os.path.basename(filename)}', fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=6)
        ax.set_ylabel('Amplitude', fontsize=6)

    def plot_spectrum(self, ax, data, sample_rate, time, filename):
        frequencies, spectrum = welch(data, sample_rate, nperseg=1024)
        ax.semilogx(frequencies, 10 * np.log10(spectrum))
        ax.set_title(f'Spectrum: {os.path.basename(filename)}', fontsize=8)
        ax.set_xlabel('Frequency (Hz)', fontsize=6)
        ax.set_ylabel('Power/Frequency (dB/Hz)', fontsize=6)
        ax.set_xlim(20, sample_rate/2)

    def plot_spectrogram(self, ax, data, sample_rate, time, filename):
        f, t, Sxx = spectrogram(data, sample_rate)
        ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        ax.set_title(f'Spectrogram: {os.path.basename(filename)}', fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=6)
        ax.set_ylabel('Frequency (Hz)', fontsize=6)
        ax.set_ylim(0, sample_rate/2)

    def plot_mfcc(self, ax, data, sample_rate, time, filename):
        mfccs = librosa.feature.mfcc(y=data.astype(float), sr=sample_rate, n_mfcc=13)
        ax.imshow(mfccs, aspect='auto', origin='lower')
        ax.set_title(f'MFCC: {os.path.basename(filename)}', fontsize=8)
        ax.set_xlabel('Time', fontsize=6)
        ax.set_ylabel('MFCC Coefficients', fontsize=6)

    def plot_phase(self, ax, data, sample_rate, time, filename):
        spectrum = fft(data)
        phase = np.angle(spectrum)
        ax.plot(np.arange(len(phase)), phase)
        ax.set_title(f'Phase: {os.path.basename(filename)}', fontsize=8)
        ax.set_xlabel('Frequency Bin', fontsize=6)
        ax.set_ylabel('Phase (radians)', fontsize=6)

    def plot_envelope(self, ax, data, sample_rate, time, filename):
        analytic_signal = hilbert(data)
        amplitude_envelope = np.abs(analytic_signal)
        ax.plot(time, amplitude_envelope)
        ax.set_title(f'Envelope: {os.path.basename(filename)}', fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=6)
        ax.set_ylabel('Amplitude', fontsize=6)

    def plot_harmonic(self, ax, data, sample_rate, time, filename):
        # Compute the spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
        
        # Plot the spectrogram
        img = ax.imshow(D, aspect='auto', origin='lower', extent=[0, time[-1], 0, sample_rate/2])
        plt.colorbar(img, ax=ax)
        
        # Estimate fundamental frequency
        n_fft = 2048
        hop_length = 512
        f0 = librosa.yin(data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), 
                        sr=sample_rate, frame_length=n_fft, hop_length=hop_length)
        
        # Create a time array for the f0 estimate
        f0_times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)
        
        # Plot the estimated fundamental frequency
        ax.plot(f0_times, f0, color='cyan', linewidth=2, alpha=0.8, label='Estimated f0')
        
        ax.set_title(f'Harmonic Structure: {os.path.basename(filename)}', fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=6)
        ax.set_ylabel('Frequency (Hz)', fontsize=6)
        ax.legend(fontsize=6)

    def update_plot(self):
        self.fig.clear()  # Clear the entire figure
        
        # Set the figure size
        self.fig.set_size_inches(6 * len(self.plot_types), 2 * self.waveforms_per_page)
        
        # Recreate subplots without figsize parameter
        num_plots = len(self.plot_types)
        self.axes = self.fig.subplots(self.waveforms_per_page, num_plots)
        if num_plots == 1 or self.waveforms_per_page == 1:
            self.axes = np.array(self.axes).reshape(self.waveforms_per_page, -1)

        start_idx = self.current_page * self.waveforms_per_page
        end_idx = min(start_idx + self.waveforms_per_page, len(self.wav_files))

        for i, filename in enumerate(self.wav_files[start_idx:end_idx]):
            axes = self.axes[i]
            self.plot_wav(os.path.join(self.directory, filename), axes)

        self.fig.suptitle(f'Page {self.current_page + 1} of {len(self.wav_files) // self.waveforms_per_page + 1}', fontsize=10)
        
        # Use tight_layout with a larger bottom margin
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        # Recreate navigation buttons
        self.prev_button_ax = self.fig.add_axes([0.2, 0.02, 0.1, 0.04])
        self.next_button_ax = self.fig.add_axes([0.7, 0.02, 0.1, 0.04])
        self.prev_button = Button(self.prev_button_ax, 'Previous')
        self.next_button = Button(self.next_button_ax, 'Next')
        self.prev_button.on_clicked(self.prev_page)
        self.next_button.on_clicked(self.next_page)

        self.fig.canvas.draw_idle()

    def next_page(self, event):
        if (self.current_page + 1) * self.waveforms_per_page < len(self.wav_files):
            self.current_page += 1
            self.update_plot()

    def prev_page(self, event):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_plot()

    def on_resize(self, event):
        self.fig.tight_layout(rect=[0, 0.05, 1, 1])  # Added rect parameter
        self.fig.canvas.draw_idle()

def main():
    parser = argparse.ArgumentParser(description="Audio File Viewer")
    parser.add_argument("directory", nargs='?', help="Directory containing WAV files")
    parser.add_argument("--plots", nargs='+', choices=['waveform', 'spectrum', 'spectrogram', 'mfcc', 'phase', 'envelope', 'harmonic'],
                        default=['waveform'], 
                        help="Choose plot types to display")
    parser.add_argument("--files", type=int, default=5, help="Number of files to display per page (max 50)")
    
    args = parser.parse_args()

    # If directory is not provided as a named argument, assume it's the last positional argument
    if args.directory is None:
        if len(sys.argv) > 1 and not sys.argv[-1].startswith('-'):
            args.directory = sys.argv[-1]
        else:
            parser.error("Directory path is required.")

    if not os.path.isdir(args.directory):
        parser.error(f"The directory '{args.directory}' does not exist.")

    # Limit the number of files per page
    files_per_page = min(max(1, args.files), 50)

    viewer = WaveformViewer(args.directory, args.plots, files_per_page)
    plt.show()

if __name__ == "__main__":
    main()
