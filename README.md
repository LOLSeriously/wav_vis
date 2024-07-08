# wav_vis
Python script to plot wav files using a variety of visualization tools

Example Usage: 

```./wav_vis.py  AKWF/AKWF_0001 --plots waveform spectrum harmonic```



<img width="1777" alt="Screenshot 2024-07-08 at 8 28 02 AM" src="https://github.com/LOLSeriously/wav_vis/assets/57334876/e62d8c4a-a194-4ee0-a2d0-a1434117c5e5">




Install required libraries: 

```pip install matplotlib scipy numpy librosa```

--plots option:
1. **waveform**: This plot shows the amplitude of the audio signal over time. It represents the raw audio data, displaying how the sound pressure level varies as the sound wave progresses. This is useful for visualizing the overall shape of the sound, including its amplitude, duration, and any obvious patterns or repetitions.

2. **spectrum**: This plot displays the frequency content of the audio signal. It shows the amplitude or power of different frequency components present in the sound. The spectrum is typically calculated using a Fourier transform and is useful for identifying dominant frequencies, harmonics, and the overall frequency distribution of the sound.

3. **spectrogram**: This would show how the frequency content of the signal changes over time, providing a visual representation of both time and frequency domains.

4. **mfcc**, Mel-frequency cepstrum: This representation is often used in speech processing and music information retrieval, showing the short-term power spectrum of a sound.

5. **phase**: This would show the phase of the signal's frequency components, which can be useful for identifying phase-related issues in audio.

6. **envelope**: This would show the overall shape or envelope of the waveform, which can be useful for analyzing dynamics and ADSR (Attack, Decay, Sustain, Release) characteristics.

7. **harmonic**: This plot could show the strengths of the fundamental frequency and its harmonics, which is particularly useful for analyzing musical tones.
   
     NOTE: Limited to 3 plots total at the same time

--files option:
The files option allows you to specify the number of audio files to display on a single page of the viewer. This controls how many waveforms (or other selected plot types) are shown simultaneously. For example, "--files 10" would display plots for 10 audio files at once. This option is useful for balancing between seeing many files at once and maintaining detail in each plot. A higher number allows for more comparisons at once, while a lower number might provide more detailed views of each file.

    NOTE: Plotting too many files at once will create a very large window vertically
