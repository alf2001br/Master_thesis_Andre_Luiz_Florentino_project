"""
Faculdade de Engenharia Industrial - FEI

Centro Universitário da Fundação Educacional Inaciana "Padre Sabóia de Medeiros" (FEI)

FEI's Stricto Sensu Graduate Program in Electrical Engineering

Concentration area: ARTIFICIAL INTELLIGENCE APPLIED TO AUTOMATION AND ROBOTICS

Master's thesis student Andre Luiz Florentino

"""

# Force processor to CPU instead of CPU + GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import librosa
import sys
import warnings
import time
import pickle

from scipy.stats import skew, kurtosis

import numpy                      as np
import tflite_runtime.interpreter as tflite

warnings.filterwarnings('ignore')


"""
Class to predict the raw audio (ESR_evaluation = Environmental Sound Recognition evaluation)

Input : list of audio files digilitalized
Output: real time prediction shown on the screen

"""


class ESR_evaluation_tflite:

    def __init__(self, audio: list, classifier: str, path_models: str, path_arrays: str):

        # Global paths for the saved classifiers and other vectors information
        self.path_models = path_models
        self.path_arrays = path_arrays

        # Load the categorical classes
        self.nom_classes = []
        with open(os.path.join(self.path_arrays, 'nom_classes.csv'), 'r') as file:
            for line in file:
                self.nom_classes.append(line.strip())

        # Load the mean and std of the validation set
        self.X_min = np.genfromtxt(os.path.join(self.path_arrays, 'X_train_min.csv'), delimiter=',', dtype='float32')
        self.X_max = np.genfromtxt(os.path.join(self.path_arrays, 'X_train_max.csv'), delimiter=',', dtype='float32')

        self.classifier = classifier
        self.rawdata = audio

        self.time_length = 4
        self.SR          = 22050
        self.frame_size  = 1024
        self.hop_length  = 512
        self.frames      = 44
        self.n_mfcc      = 13
        self.n_mels      = 128
        self.n_fft       = 2048
        self.bands       = 60
        self.count       = 1

        self.target_samples = int(self.time_length * self.SR)
        self.window_size    = 512 * (self.frames - 1)

        self.audio_windowed = []
        self.features_agg   = []
        self.predictions    = []
        self.totalPredTime  = []

        self.startTimer    = 0
        self.endTimer      = 0

        # Load the classifier LOGISTIC REGRESSION (LR) with the highest accuracy
        if self.classifier == 'SVC':
            with open(os.path.join(self.path_models, 'Model_SVC_norm_windowed.pkl'), 'rb') as file:
                self.model_SVC = pickle.load(file)

        # Load the classifier LOGISTIC REGRESSION (LR) with the highest accuracy
        elif self.classifier == 'LR':
            with open(os.path.join(self.path_models, 'Model_LogisticR_norm_windowed.pkl'), 'rb') as file:
                self.model_LR = pickle.load(file)

                # Load the classifier RANDOM FOREST (RF) with the highest accuracy
        elif self.classifier == 'RF':
            with open(os.path.join(self.path_models, 'Model_Forest_norm_windowed.pkl'), 'rb') as file:
                self.model_RF = pickle.load(file)

        # Load the ARTIFICIAL NEURAL NETWORK (ANN) or MULT LAYER PERCEPTRON (MLP) model with the highest accuracy
        elif self.classifier == 'ANN':
            self.interpreter_ANN    = tflite.Interpreter(os.path.join(self.path_models, 'Model_ANN_weights_0_best_norm_windowed.tflite'))
            self.input_details_ANN  = self.interpreter_ANN.get_input_details()
            self.output_details_ANN = self.interpreter_ANN.get_output_details()

            # Load the CONVOLUTIONAL NEURAL NETWORK 1D (CNN1D) model with the highest accuracy
        elif self.classifier == 'CNN1D':
            self.interpreter_CNN1D    = tflite.Interpreter(os.path.join(self.path_models, 'Model_CNN_1D_weights_0_best_norm_windowed.tflite'))
            self.input_details_CNN1D  = self.interpreter_CNN1D.get_input_details()
            self.output_details_CNN1D = self.interpreter_CNN1D.get_output_details()

        # Load the CONVOLUTIONAL NEURAL NETWORK 2D (CNN2D) model with the highest accuracy
        elif self.classifier == 'CNN2D':
            self.interpreter_CNN2D    = tflite.Interpreter(os.path.join(self.path_models, 'Model_CNN_2D_weights_0_best_windowed.tflite'))
            self.input_details_CNN2D  = self.interpreter_CNN2D.get_input_details()
            self.output_details_CNN2D = self.interpreter_CNN2D.get_output_details()

        else:
            print("Invalid classifer. System will exit.")
            sys.exit()

        self._windowingPredict()

    # Windowing procedure
    def _windows(self, audio):
        start = 0
        while start < len(audio):
            yield int(start), int(start + self.window_size)
            start += (self.window_size / 2)

    # Function to normalize the audio dataset considering 44 frames @22.050 Hz --> ~0,99s per window
    def _windowingPredict(self):
        for audio_ in self.rawdata:
            # Pass the window method
            for (start, end) in self._windows(audio_):
                if len(audio_[start:end]) == self.window_size:

                    # Start the time counter
                    self.startTimer = time.perf_counter_ns()

                    # Window the audio
                    signal = audio_[start:end]

                    if self.classifier == 'CNN2D':
                        # Call the log-mel feature extractor
                        self._logMel_extractor(signal)

                        # Stop the time counter
                        self.endTimer = time.perf_counter_ns()
                        totalPredTime = ((self.endTimer - self.startTimer) / 1000000)
                        self.totalPredTime.append(totalPredTime)
                        print(f'Total predict time:..: {totalPredTime:.4f}ms\n')

                    else:
                        # Call the ML feature extractor
                        self._feature_extractor(signal)

                        # Stop the time counter
                        self.endTimer = time.perf_counter_ns()
                        totalPredTime = ((self.endTimer - self.startTimer) / 1000000)
                        self.totalPredTime.append(totalPredTime)
                        print(f'Total predict time:..: {totalPredTime:.4f}ms\n')

    def _feature_extractor(self, signal):
        self.features_vector = []

        self.features_vector.append(np.mean(librosa.feature.rms(y=signal,
                                                                hop_length=self.hop_length,
                                                                frame_length=self.frame_size)))
        self.features_vector.append(np.mean(librosa.feature.zero_crossing_rate(y=signal,
                                                                               hop_length=self.hop_length,
                                                                               frame_length=self.frame_size)))
        self.features_vector.append(np.mean(librosa.feature.spectral_centroid(y=signal,
                                                                              hop_length=self.hop_length,
                                                                              n_fft=self.n_fft,
                                                                              sr=self.SR)))
        self.features_vector.append(np.mean(librosa.feature.spectral_bandwidth(y=signal,
                                                                               hop_length=self.hop_length,
                                                                               n_fft=self.n_fft,
                                                                               sr=self.SR)))
        self.features_vector.append(np.mean(librosa.feature.spectral_rolloff(y=signal,
                                                                             hop_length=self.hop_length,
                                                                             n_fft=self.n_fft,
                                                                             sr=self.SR)))
        # 128 x frames matrix
        mel = librosa.feature.melspectrogram(y=signal,
                                             hop_length=self.hop_length,
                                             n_fft=self.n_fft,
                                             sr=self.SR,
                                             n_mels=self.n_mels)
        for i in range(0, self.n_mels):
            self.features_vector.append(np.mean(mel[i]))

        # 13 x frames matrix (mean and standard deviation)
        mfcc = librosa.feature.mfcc(y=signal, sr=self.SR, n_mfcc=self.n_mfcc)
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
        spec_contrast = librosa.feature.spectral_contrast(y=signal,
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
        chroma = librosa.feature.chroma_stft(y=signal,
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
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=self.SR)
        tonnetz_sp = np.shape(tonnetz)
        for i in range(0, tonnetz_sp[0]):
            self.features_vector.append(np.mean(tonnetz[i]))
            self.features_vector.append(np.std(tonnetz[i]))
            self.features_vector.append(np.median(tonnetz[i]))
            self.features_vector.append(skew(tonnetz[i]))
            self.features_vector.append(kurtosis(tonnetz[i]))

        # Replace NaN values by 0
        self.features_vector = np.array(self.features_vector)
        self.features_vector[np.isnan(self.features_vector)] = 0

        # Normalize the validation set using the min and max from training
        self.features_vector = (self.features_vector - self.X_min) / (self.X_max - self.X_min)

        # Call the prediction
        self._val_predict_tflite()

    # Function to extract and aggregate the log-mel features + delta + delta delta
    def _logMel_extractor(self, signal):
        melspec = librosa.feature.melspectrogram(y=signal,
                                                 n_mels=self.bands,
                                                 hop_length=self.hop_length,
                                                 n_fft=self.n_fft,
                                                 sr=self.SR)

        logspec = librosa.power_to_db(melspec)

        # Flattens the array (bands , frames) to (bands * frames , 1) E.g.: (60 , 216) --> (12.960 , 1)
        logspec = logspec.flatten()[:, np.newaxis]

        # Appends to array
        self.log_specgrams = []
        self.log_specgrams.append(logspec)

        # Reshape to audio, bands, frames and channels E.g.: (Depends on the model Ori or Aug, 60, 44, 1)
        self.log_specgrams = np.asarray(np.array(logspec), dtype='float32').reshape(len(self.log_specgrams),
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

        # Call the prediction
        self._val_predict_tflite()

    # Function to predict
    def _val_predict_tflite(self):
        print(f'Audio clip...........: {self.count}')
        self.count = self.count + 1

        if self.classifier == 'SVC':
            predict_val = self.model_SVC.predict(np.array(self.features_vector).reshape(1, -1))
            self.predictions.append(predict_val[0])
            print(f'Prediction...........: {predict_val[0]}')
        if self.classifier == 'LR':
            predict_val = self.model_LR.predict(np.array(self.features_vector).reshape(1, -1))
            self.predictions.append(predict_val[0])
            print(f'Prediction...........: {predict_val[0]}')

        elif self.classifier == 'RF':
            predict_val = self.model_RF.predict(np.array(self.features_vector).reshape(1, -1))
            self.predictions.append(predict_val[0])
            print(f'Prediction...........: {predict_val[0]}')

        elif self.classifier == 'ANN':
            self.features_vector = self.features_vector.astype(self.input_details_ANN[0]['dtype'])
            features_vector = self.features_vector.reshape(1, -1)
            self.interpreter_ANN.allocate_tensors()
            self.interpreter_ANN.set_tensor(self.input_details_ANN[0]['index'], features_vector)
            self.interpreter_ANN.invoke()
            predict_val = np.argmax(self.interpreter_ANN.get_tensor(self.output_details_ANN[0]['index']), axis=1)
            self.predictions.append(predict_val[0])
            print(f'Prediction...........: {self.nom_classes[predict_val[0]]}')

        elif self.classifier == 'CNN1D':
            self.features_vector = self.features_vector.astype(self.input_details_CNN1D[0]['dtype'])
            features_vector = self.features_vector.reshape(1, -1,1)
            self.interpreter_CNN1D.allocate_tensors()
            self.interpreter_CNN1D.set_tensor(self.input_details_CNN1D[0]['index'], features_vector)
            self.interpreter_CNN1D.invoke()
            predict_val = np.argmax(self.interpreter_CNN1D.get_tensor(self.output_details_CNN1D[0]['index']), axis=1)
            self.predictions.append(predict_val[0])
            print(f'Prediction...........: {self.nom_classes[predict_val[0]]}')

        elif self.classifier == 'CNN2D':
            self.features_agg = self.features_agg.astype(self.input_details_CNN2D[0]['dtype'])
            features_agg = self.features_agg.reshape(1, -1,44,1)
            self.interpreter_CNN2D.allocate_tensors()
            self.interpreter_CNN2D.set_tensor(self.input_details_CNN2D[0]['index'], features_agg)
            self.interpreter_CNN2D.invoke()
            predict_val = np.argmax(self.interpreter_CNN2D.get_tensor(self.output_details_CNN2D[0]['index']), axis=1)
            self.predictions.append(predict_val[0])
            print(f'Prediction...........: {self.nom_classes[predict_val[0]]}')

