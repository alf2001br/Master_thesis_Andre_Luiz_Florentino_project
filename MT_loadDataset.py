"""
Centro Universitário da Fundação Educacional Inaciana "Padre Sabóia de Medeiros" (FEI)

FEI's Stricto Sensu Graduate Program in Electrical Engineering

Concentration area: ARTIFICIAL INTELLIGENCE APPLIED TO AUTOMATION AND ROBOTICS

Master's thesis student Andre Luiz Florentino

"""

import os
import glob
import mimetypes

import pandas as pd
import numpy  as np

mimetypes.init()
mimetypes.add_type('audio/ogg','.ogg')



"""
Class to load the dataseta ESC-10, BDLib2 and US8K

Input : root path where the dataset is saved with its name
Output: DataFrame

DataFrame columns:
- Class
- Fold
- File_name
- Path

"""

class loadDataset:

    def __init__(self, path:str):

        self.path        = path
        self.subfolders  = next(os.walk(self.path))[1]

        self.dict_list   = []
        self.db_B        = pd.DataFrame()
        self.csv_file    = os.path.basename(os.path.normpath(self.path)) + ".csv"

        if os.path.basename(os.path.normpath(self.path)) == "US8K_AV":
            self.db_B = pd.read_csv(os.path.join(self.path, self.csv_file))
            self.db_B = self.db_B.drop('Unnamed: 0', axis=1)
        else:
            self._readDoc()
            self._OHEV()
            self._exportCSV()

    # Procedure to read the folders, sound files and create the dataframe
    def _readDoc(self):

        for folder in self.subfolders:
            os.chdir(os.path.join(self.path, folder))

            # Setup for ESC-10
            if os.path.basename(os.path.normpath(self.path)) == "ESC-10":
                sounds = (glob.glob('*.ogg'))
                for s in sounds:
                    row_dict = {'Fold': s[0],
                                'Folder_name': folder,
                                'Class_categorical': folder[6:],
                                'File_name': s,
                                'Path': os.path.join(self.path, folder,s),
                                'Class_ID': folder[:3],
                                'Clip_ID': str(s[2:-6]),
                                'Clip_take': s[-5:-4]}
                    self.dict_list.append(row_dict)

                self.db_B = pd.DataFrame.from_dict(self.dict_list)
                self.db_B = self.db_B.astype({'Fold': 'int64', 'Class_ID': 'int64', 'Clip_ID': 'int64'})


            # Setup for BDLib2
            if os.path.basename(os.path.normpath(self.path)) == "BDLib2":
                sounds = (glob.glob('*.wav'))
                for s in sounds:
                    row_dict = {'Fold': folder,
                                'Folder_name': folder,
                                'Class_categorical': s[:-6],
                                'File_name': s,
                                'Path': os.path.join(self.path, folder,s)}
                    self.dict_list.append(row_dict)

                self.db_B = pd.DataFrame.from_dict(self.dict_list)


            # Setup for US8K
            if os.path.basename(os.path.normpath(self.path)) == "US8K":
                metadata = 'UrbanSound8K.csv'

                dtype_dict = {'slice_file_name': 'str',
                              'fsID': 'int64',
                              'start': 'float64',
                              'end': 'float64',
                              'salience': 'int64',
                              'fold': 'int64',
                              'classID': 'int64',
                              'Class': 'str'}

                db_raw = pd.read_csv(os.path.join(self.path, metadata), dtype=dtype_dict)
                db_raw['Folder_name'] = 'fold' + db_raw['fold'].astype(str)
                db_raw['Path']   = db_raw.apply(lambda x: os.path.join(self.path, x['Folder_name'], x['slice_file_name']), axis=1)

                db_raw = db_raw.rename(columns={'fold': 'Fold',
                                                'class': 'Class_categorical',
                                                'slice_file_name': 'File_name'})

                self.db_B = db_raw[['Fold',
                                    'Folder_name',
                                    'Class_categorical',
                                    'File_name',
                                    'Path',
                                    'classID',
                                    'fsID',
                                    'start',
                                    'end',
                                    'salience']]


    # Create the One Hot Encoder Vector (OHEV)
    def _OHEV(self):
        df_class  = self.db_B['Class_categorical']
        class_enc = np.array(pd.get_dummies(df_class, columns = [str], dtype=int))
        self.db_B.insert(loc = 2, column = 'Class_OHEV', value = class_enc.tolist())


    # Export the dataframe as CSV file
    def _exportCSV(self):
        os.chdir(self.path)
        self.db_B.to_csv(self.csv_file)
        print("\nCSV exported.\nCheck the folder : ",self.path)
