"""
Faculdade de Engenharia Industrial - FEI

Centro Universitário da Fundação Educacional Inaciana "Padre Sabóia de Medeiros" (FEI)

FEI's Stricto Sensu Graduate Program in Electrical Engineering

Concentration area: ARTIFICIAL INTELLIGENCE APPLIED TO AUTOMATION AND ROBOTICS

Master's thesis student Andre Luiz Florentino

"""

import librosa
import numpy  as np
from scipy.stats import skew, kurtosis


"""
Class to pre-process the audio files (AudioPPFE_ML = Audio Pre-Processing Feature Extractor ML)

Input : list of audio files digilitalized and list of label for each file
Output: numpy array of the features splitted according the windowing interval

"""


class audioPPFE_ML:

    def __init__(self,
                 audio: list,
                 CNN2D: bool,
                 time_length=4):

        self.rawdata = audio
        self.time_length = time_length

        self.SR = 22050
        self.frame_size = 1024
        self.hop_length = 512
        self.frames = 44
        self.n_mfcc = 13
        self.n_mels = 128
        self.n_fft = 2048
        self.bands = 60

        self.target_samples = int(self.time_length * self.SR)
        self.window_size = 512 * (self.frames - 1)

        self.features_vector = []
        self.features_array = []
        self.audio_windowed = []
        self.log_specgrams = []
        self.framesLst = []
        self.features_agg = []

        self._windowing()

        if CNN2D:
            self._logMel_extractor()
        else:
            self._feature_extractor()

    # Windowing procedure
    def _windows(self, audio):
        start = 0
        while start < len(audio):
            yield int(start), int(start + self.window_size)
            start += (self.window_size / 2)

    # Procedure to normalize the audio dataset considering 44 frames @22.050 Hz --> ~0,99s per window
    def _windowing(self):
        for audio_ in self.rawdata:
            # Pass the window method
            for (start, end) in self._windows(audio_):
                if len(audio_[start:end]) == self.window_size:
                    # Window the audio
                    signal = audio_[start:end]
                    # Appends to array
                    self.audio_windowed.append(signal)

    def _feature_extractor(self):
        for audio_w in self.audio_windowed:
            # Extract the features
            self.features_vector.append(np.mean(librosa.feature.rms(y=audio_w,
                                                                    hop_length=self.hop_length,
                                                                    frame_length=self.frame_size)))
            self.features_vector.append(np.mean(librosa.feature.zero_crossing_rate(y=audio_w,
                                                                                   hop_length=self.hop_length,
                                                                                   frame_length=self.frame_size)))
            self.features_vector.append(np.mean(librosa.feature.spectral_centroid(y=audio_w,
                                                                                  hop_length=self.hop_length,
                                                                                  n_fft=self.n_fft,
                                                                                  sr=self.SR)))
            self.features_vector.append(np.mean(librosa.feature.spectral_bandwidth(y=audio_w,
                                                                                   hop_length=self.hop_length,
                                                                                   n_fft=self.n_fft,
                                                                                   sr=self.SR)))
            self.features_vector.append(np.mean(librosa.feature.spectral_rolloff(y=audio_w,
                                                                                 hop_length=self.hop_length,
                                                                                 n_fft=self.n_fft,
                                                                                 sr=self.SR)))
            # 128 x frames matrix
            mel = librosa.feature.melspectrogram(y=audio_w,
                                                 hop_length=self.hop_length,
                                                 n_fft=self.n_fft,
                                                 sr=self.SR,
                                                 n_mels=self.n_mels)
            for i in range(0, self.n_mels):
                self.features_vector.append(np.mean(mel[i]))

            # 13 x frames matrix (mean and standard deviation)
            mfcc = librosa.feature.mfcc(y=audio_w, sr=self.SR, n_mfcc=self.n_mfcc)
            for i in range(0, self.n_mfcc):
                self.features_vector.append(np.mean(mfcc[i]))
                self.features_vector.append(np.std(mfcc[i]))
                self.features_vector.append(np.median(mfcc[i]))
                self.features_vector.append(skew(mfcc[i]))
                self.features_vector.append(kurtosis(mfcc[i]))

                delta1 = librosa.feature.delta(mfcc[i])
                delta2 = librosa.feature.delta(mfcc[i], order=2)

                self.features_vector.append(np.mean(delta1))
                self.features_vector.append(np.std(delta1))
                self.features_vector.append(np.mean(delta2))
                self.features_vector.append(np.std(delta2))

            # 7 x frames matrix
            spec_contrast = librosa.feature.spectral_contrast(y=audio_w,
                                                              hop_length=self.hop_length,
                                                              n_fft=self.n_fft,
                                                              sr=self.SR)
            spec_contrast_sp = np.shape(spec_contrast)
            for i in range(0, spec_contrast_sp[0]):
                self.features_vector.append(np.mean(spec_contrast[i]))
                self.features_vector.append(np.std(spec_contrast[i]))
                self.features_vector.append(np.median(spec_contrast[i]))
                self.features_vector.append(skew(spec_contrast[i]))
                self.features_vector.append(kurtosis(spec_contrast[i]))

            # 12 x frames matrix
            chroma = librosa.feature.chroma_stft(y=audio_w,
                                                 hop_length=self.hop_length,
                                                 n_fft=self.n_fft,
                                                 sr=self.SR)
            chroma_sp = np.shape(chroma)
            for i in range(0, chroma_sp[0]):
                self.features_vector.append(np.mean(chroma[i]))
                self.features_vector.append(np.std(chroma[i]))
                self.features_vector.append(np.median(chroma[i]))
                self.features_vector.append(skew(chroma[i]))
                self.features_vector.append(kurtosis(chroma[i]))

            # 6 x frames matrix
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_w), sr=self.SR)
            tonnetz_sp = np.shape(tonnetz)
            for i in range(0, tonnetz_sp[0]):
                self.features_vector.append(np.mean(tonnetz[i]))
                self.features_vector.append(np.std(tonnetz[i]))
                self.features_vector.append(np.median(tonnetz[i]))
                self.features_vector.append(skew(tonnetz[i]))
                self.features_vector.append(kurtosis(tonnetz[i]))

            self.features_array.append(self.features_vector)
            self.features_vector = []

    def _logMel_extractor(self):
        for audio_w in self.audio_windowed:
            # for signal in tqdm(audio_clips):
            melspec = librosa.feature.melspectrogram(y=audio_w,
                                                     n_mels=self.bands,
                                                     hop_length=self.hop_length,
                                                     n_fft=self.n_fft,
                                                     sr=self.SR)

            logspec = librosa.power_to_db(melspec)
            frames = logspec.shape[1]
            self.framesLst.append(frames)

            # Flattens the array (bands , frames) to (bands * frames , 1) E.g.: (60 , 216) --> (12.960 , 1)
            logspec = logspec.flatten()[:, np.newaxis]

            # Appends to array
            self.log_specgrams.append(logspec)

        # Reshape to audio, bands, frames and channels E.g.: (Depends on the model Ori or Aug, 60, 44, 1)
        self.log_specgrams = np.asarray(self.log_specgrams, dtype='float32').reshape(len(self.log_specgrams),
                                                                                     self.bands,
                                                                                     self.frames,
                                                                                     1)

        # Initiate zeros for the log mel spectrogram delta
        features = np.concatenate((self.log_specgrams,
                                   np.zeros(np.shape(self.log_specgrams)),
                                   np.zeros(np.shape(self.log_specgrams))), axis=3)

        # Add the delta for the log mel spectrogram as channels
        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0], order=1)
            features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 0], order=2)

        # Vertically stack up the deltas to create an aggregated structures of features
        mel, delta1, delta2 = np.split(features, 3, axis=3)
        self.features_agg = np.concatenate((mel, delta1, delta2), axis=1)

        if len(set(self.framesLst)) == 1:
            duration = "{:.2f}".format((self.framesLst[0] * self.hop_length) / self.SR)
            print(f"\nMel spectrograms created by a {duration} seconds audio. Number of frames: {self.framesLst[0]}")

