"""
Faculdade de Engenharia Industrial - FEI

Centro Universitário da Fundação Educacional Inaciana "Padre Sabóia de Medeiros" (FEI)

FEI's Stricto Sensu Graduate Program in Electrical Engineering

Concentration area: ARTIFICIAL INTELLIGENCE APPLIED TO AUTOMATION AND ROBOTICS

Master's thesis student Andre Luiz Florentino

"""

import librosa
import numpy  as np
from tqdm import tqdm

# Global variables

FRAME_SIZE = 1024
HOP_LENGTH = 512

"""
Class to pre-process the audio files (AudioPP = Audio Pre-Processing)

Input : list of path + files and list of label for each file
Output: numpy array of the pre-processed audio files and related labels

For each file:
- Remove the silence frames
- Extend the audio file to 10 seconds 
- Apply a windowing technique to normalize the audio considering 44 frames @22.050 Hz --> ~0,99s per window

"""

class audioPP:

    def __init__(self,
                 fn:list,
                 y_cat: list,
                 y_cod: list,
                 fold_nr: list,
                 time_length = 5,
                 threshold   = 60,
                 aug = True,
                 windowing = True,
                 frames = 44):
        self.fn                = fn
        self.aug               = aug
        self.windowing         = windowing

        # dB Threshold below the reference db, in this case np.max from amplitude to db resulted 80dB
        self.silence_threshold = threshold

        # Calculate the number of samples needed to fit the target time length (in seconds)
        self.SR               = 22050
        self.time_length      = time_length
        self.target_samples   = int(self.time_length * self.SR)
        self.frames           = frames
        self.window_size      = 512 * (self.frames - 1)
        self.audio_augmented  = []
        self.labels_cat_wind  = []
        self.labels_cod_wind  = []
        self.labels_fold_wind = []
        self.audio_array      = []
        self.audio_windowed   = []

        if self.aug == True:
            self.y_cat       = y_cat
            self.y_cod       = y_cod
            self.fold_nr     = fold_nr
            self.labels_cat  = []
            self.labels_cod  = []
            self.labels_fold = []

            self._augmentation()
        else:
            self.labels_cat  = y_cat
            self.labels_cod  = y_cod
            self.labels_fold = fold_nr

            for file in tqdm(self.fn):

                # Load the audio files, check for mono and sampling rate
                rawdata, sr  = librosa.load(file, sr = self.SR, mono = True)
                num_channels = rawdata.shape[0] if len(rawdata.shape) > 1 else 1
                if num_channels != 1:
                    print(f'Audio file is not mono. Fix the problem of {file}')
                if sr != self.SR:
                    print(f'Sampling rate different than {sr} Hz. Fix the problem')
                self.audio_augmented.append(rawdata)

        self._pre_processing()

        if self.windowing == True:
            self._windowing()
        else:
            if self.aug == True:
                self.audio_windowed   = self.audio_array
                self.labels_cat_wind  = self.labels_cat
                self.labels_cod_wind  = self.labels_cod
                self.labels_fold_wind = self.labels_fold
            else:
                self.audio_windowed   = self.audio_array
                self.labels_cat_wind  = y_cat
                self.labels_cod_wind  = y_cod
                self.labels_fold_wind = fold_nr

            print(f"Shape of the audio data..................: {np.shape(self.audio_windowed)}")
            print(f"Shape of the categorical label data......: {np.shape(self.labels_cat_wind)}")
            print(f"Shape of the one hot encoder label data..: {np.shape(self.labels_cod_wind)}")
            print(f"Shape of the fold data...................: {np.shape(self.labels_fold_wind)}")


    # Procedure to augment the dataset considering time shifting, time stretching and pitch shifting
    def _augmentation(self):
        for file, label_cat, label_cod, label_fold in zip(tqdm(self.fn), self.y_cat, self.y_cod, self.fold_nr):

            # Load the audio files
            rawdata, _ = librosa.load(file, sr = self.SR)
            start_     = int(np.random.uniform(-4800,4800))

            # Time shifting (randomly)
            if start_ >= 0:
                audio_time_shift = np.r_[rawdata[start_:], np.random.uniform(-0.001,0.001, start_)]
            else:
                audio_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), rawdata[:start_]]

            self.audio_augmented.append(rawdata)
            self.audio_augmented.append(audio_time_shift)
            self.labels_cat.append(label_cat)
            self.labels_cat.append(label_cat)
            self.labels_cod.append(label_cod)
            self.labels_cod.append(label_cod)
            self.labels_fold.append(label_fold)
            self.labels_fold.append(label_fold)

            # Time stretching
            self.audio_augmented.append(librosa.effects.time_stretch(rawdata, rate=0.85))
            self.audio_augmented.append(librosa.effects.time_stretch(rawdata, rate=1.15))
            self.labels_cat.append(label_cat)
            self.labels_cat.append(label_cat)
            self.labels_cod.append(label_cod)
            self.labels_cod.append(label_cod)
            self.labels_fold.append(label_fold)
            self.labels_fold.append(label_fold)

            # Pitch shifting

            self.audio_augmented.append(librosa.effects.pitch_shift(rawdata, sr = self.SR, n_steps = 4))
            self.audio_augmented.append(librosa.effects.pitch_shift(rawdata, sr = self.SR, n_steps = -4))
            self.labels_cat.append(label_cat)
            self.labels_cat.append(label_cat)
            self.labels_cod.append(label_cod)
            self.labels_cod.append(label_cod)
            self.labels_fold.append(label_fold)
            self.labels_fold.append(label_fold)

    # Procedure to remove silence frames and extend each audio file to the targeted time length
    def _pre_processing(self):
        for i, audio in tqdm(enumerate(self.audio_augmented, start=0)):

            # Split the audio into non-silent intervals
            non_silent_intervals = librosa.effects.split(audio,
                                                         top_db       = self.silence_threshold,
                                                         frame_length = FRAME_SIZE,
                                                         hop_length   = HOP_LENGTH)

            # Extract non-silent segments from the original audio data
            non_silent_audio  = []
            for interval in non_silent_intervals:
                start, end = interval
                non_silent_audio.extend(audio[start:end])

            # Convert the list back to a NumPy array
            non_silent_audio_array = np.array(non_silent_audio)
            # print(librosa.get_duration(y=non_silent_audio_array, sr=self.SR))

            # Repeat the non-silent audio array to fit the target time length
            extended_audio = np.tile(non_silent_audio_array, self.target_samples // len(non_silent_audio_array) + 1)

            # Truncate the extended audio to match the desired duration
            self.audio_array.append(extended_audio[:self.target_samples])

            # print('Raw audio file duration.....: ' , "{:0.4f} s".format(librosa.get_duration(y=rawdata)))
            # print('Trimmed audio file duration.: ' , "{:0.4f} s".format(librosa.get_duration(y=non_silent_audio_array)))
            # print('Extended audio file duration: ' , "{:0.4f} s".format(librosa.get_duration(y=self.audio_array[i])))
            # print('---------------')


    # Windowing procedure
    def _windows(self, audio):
        start = 0
        while start < len(audio):
            yield int(start), int(start + self.window_size)
            start += (self.window_size / 2)


    # Procedure to normalize the audio dataset considering 43 frames @22.050 Hz --> ~0,99s per window
    def _windowing(self):
        for audio_, label_cat_, label_cod_, label_fold_ in zip(tqdm(self.audio_array), self.labels_cat, self.labels_cod, self.labels_fold):
            # Pass the window method
            for (start, end) in self._windows(audio_):
                if len(audio_[start:end]) == self.window_size:

                    # Window the audio
                    signal  = audio_[start:end]

                    # Appends to array
                    self.audio_windowed.append(signal)
                    self.labels_cat_wind.append(label_cat_)
                    self.labels_cod_wind.append(label_cod_)
                    self.labels_fold_wind.append(label_fold_)

        print(f"Shape of the audio data..................: {np.shape(self.audio_windowed)}")
        print(f"Shape of the categorical label data......: {np.shape(self.labels_cat_wind)}")
        print(f"Shape of the one hot encoder label data..: {np.shape(self.labels_cod_wind)}")
        print(f"Shape of the fold data...................: {np.shape(self.labels_fold_wind)}")
