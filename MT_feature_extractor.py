"""
Faculdade de Engenharia Industrial - FEI

Centro Universitário da Fundação Educacional Inaciana "Padre Sabóia de Medeiros" (FEI)

FEI's Stricto Sensu Graduate Program in Electrical Engineering

Concentration area: ARTIFICIAL INTELLIGENCE APPLIED TO AUTOMATION AND ROBOTICS

Master's thesis student Andre Luiz Florentino

"""

from tqdm import tqdm
import pandas as pd
import numpy  as np
import librosa
import warnings
from scipy.stats import skew, kurtosis


warnings.filterwarnings('ignore')


"""
Class to extract the features for machine learning / ensemble methods algotithms and neural networks

Input : dataframe with the audio files and labels, both categorical and one hot encoder
Output: 172 features

"""

class feature_extractor:

    def __init__(self, db_aug: pd.DataFrame):
        self.DB_aug     = db_aug
        self.n_mfcc     = 13
        self.n_mels     = 128
        self.hop_length = 512
        self.frame_size = 1024
        self.n_fft      = 2048
        self.SR         = 22050

        tqdm.pandas()
        self.DB_aug     = self.DB_aug.progress_apply(self._feature_parser, axis = 1)


    # Method to parse each row and extract the defined audio feature
    def _feature_parser(self, row):

        # Extract the features
        row['RMSE']      = np.mean(librosa.feature.rms(y                = row.Audio, hop_length = self.hop_length, frame_length = self.frame_size))
        row['ZCR']       = np.mean(librosa.feature.zero_crossing_rate(y = row.Audio, hop_length = self.hop_length, frame_length = self.frame_size))
        row['CENTROIDS'] = np.mean(librosa.feature.spectral_centroid(y  = row.Audio, hop_length = self.hop_length, n_fft = self.n_fft, sr = self.SR))
        row['BANDWIDTH'] = np.mean(librosa.feature.spectral_bandwidth(y = row.Audio, hop_length = self.hop_length, n_fft = self.n_fft, sr = self.SR))
        row['ROLLOFF']   = np.mean(librosa.feature.spectral_rolloff(y   = row.Audio, hop_length = self.hop_length, n_fft = self.n_fft, sr = self.SR))

        # 128 x frames matrix
        mel              = librosa.feature.melspectrogram(y = row.Audio, hop_length = self.hop_length, n_fft = self.n_fft, sr = self.SR, n_mels = self.n_mels)
        for i in range(0, self.n_mels):
            row[f'MEL_{ i +1}'] = np.mean(mel[i])

        # 13 x frames matrix (mean and standard deviation)
        mfcc             = librosa.feature.mfcc(y = row.Audio, sr = self.SR, n_mfcc = self.n_mfcc)
        for i in range(0, self.n_mfcc):
            row[f'MFCC_{ i +1}'] = np.mean(mfcc[i])
            row[f'MFCC_std_{ i +1}'] = np.std(mfcc[i])
            row[f'MFCC_median_{ i +1}'] = np.median(mfcc[i])
            row[f'MFCC_skew_{ i +1}'] = skew(mfcc[i])
            row[f'MFCC_kurtosis_{ i +1}'] = kurtosis(mfcc[i])

            delta1 = librosa.feature.delta(mfcc[i])
            delta2 = librosa.feature.delta(mfcc[i], order=2)

            row[f'MFCC_delta1_mean_{i+1}'] = np.mean(delta1)
            row[f'MFCC_delta1_std_{i+1}']  = np.std(delta1)
            row[f'MFCC_delta2_mean_{i+1}'] = np.mean(delta2)
            row[f'MFCC_delta2_std_{i+1}']  = np.std(delta2)

        # 7 x frames matrix
        spec_contrast    = librosa.feature.spectral_contrast(y = row.Audio, hop_length = self.hop_length, n_fft = self.n_fft, sr = self.SR)
        spec_contrast_sp = np.shape(spec_contrast)
        for i in range(0, spec_contrast_sp[0]):
            row[f'CONSTRAST_{ i +1}'] = np.mean(spec_contrast[i])
            row[f'CONSTRAST_std_{ i +1}'] = np.std(spec_contrast[i])
            row[f'CONSTRAST_median_{ i +1}'] = np.median(spec_contrast[i])
            row[f'CONSTRAST_skew_{ i +1}'] = skew(spec_contrast[i])
            row[f'CONSTRAST_kurtosis_{ i +1}'] = kurtosis(spec_contrast[i])

        # 12 x frames matrix
        chroma           = librosa.feature.chroma_stft(y = row.Audio, hop_length = self.hop_length, n_fft = self.n_fft, sr = self.SR)
        chroma_sp        = np.shape(chroma)
        for i in range(0, chroma_sp[0]):
            row[f'CHROMA_{ i +1}'] = np.mean(chroma[i])
            row[f'CHROMA_std_{ i +1}'] = np.std(chroma[i])
            row[f'CHROMA_median_{ i +1}'] = np.median(chroma[i])
            row[f'CHROMA_skew_{ i +1}'] = skew(chroma[i])
            row[f'CHROMA_kurtosis_{ i +1}'] = kurtosis(chroma[i])

        # 6 x frames matrix
        tonnetz          = librosa.feature.tonnetz(y = librosa.effects.harmonic(row.Audio), sr = self.SR)
        tonnetz_sp       = np.shape(tonnetz)
        for i in range(0, tonnetz_sp[0]):
            row[f'TONNETZ_{ i +1}'] = np.mean(tonnetz[i])
            row[f'TONNETZ_std_{ i +1}'] = np.std(tonnetz[i])
            row[f'TONNETZ_median_{ i +1}'] = np.median(tonnetz[i])
            row[f'TONNETZ_skew_{ i +1}'] = skew(tonnetz[i])
            row[f'TONNETZ_kurtosis_{ i +1}'] = kurtosis(tonnetz[i])
        return row
