"""
Faculdade de Engenharia Industrial - FEI

Centro Universitário da Fundação Educacional Inaciana "Padre Sabóia de Medeiros" (FEI)

FEI's Stricto Sensu Graduate Program in Electrical Engineering

Concentration area: ARTIFICIAL INTELLIGENCE APPLIED TO AUTOMATION AND ROBOTICS

Master's thesis student Andre Luiz Florentino

"""

import psutil
import csv
import time

import pandas as pd
import numpy  as np
import sys

# Force processor to CPU instead of CPU + GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

current_path = os.getcwd()
root_path    = os.path.dirname(current_path)
sys.path.insert(1, root_path)

from MT_ESR_evaluation_tflite import ESR_evaluation_tflite
 
 
# Globals
path           = os.path.join(root_path, "US8K_AV_saved_models")
path_modelsVal = os.path.join(root_path, "_ESR", "Saved_models_fold_1_validation")
path_arrays    = os.path.join(root_path, "_ESR", "Arrays")
csv_file       = 'US8K_AV.csv'
pkl_features   = 'US8K_AV_features_original.pkl'
fold_name      = '1'

# Results from the file 11_ESR_evaluation.ipynb for confirming the predictions
saved_predict_val_SVC   = np.genfromtxt(os.path.join(path_arrays, '_saved_predict_val_SVC.csv'),   delimiter=',', dtype = str)
saved_predict_val_SVC   = np.char.strip(saved_predict_val_SVC, "'")
saved_predict_val_LR    = np.genfromtxt(os.path.join(path_arrays, '_saved_predict_val_LR.csv'),    delimiter=',', dtype = str)
saved_predict_val_LR    = np.char.strip(saved_predict_val_LR, "'")
saved_predict_val_RF    = np.genfromtxt(os.path.join(path_arrays, '_saved_predict_val_RF.csv'),    delimiter=',', dtype = str)
saved_predict_val_RF    = np.char.strip(saved_predict_val_RF, "'")
saved_predict_val_ANN   = np.genfromtxt(os.path.join(path_arrays, '_saved_predict_val_ANN_tflite.csv'),   delimiter=',', dtype = int)
saved_predict_val_CNN1D = np.genfromtxt(os.path.join(path_arrays, '_saved_predict_val_CNN1D_tflite.csv'), delimiter=',', dtype = int)
saved_predict_val_CNN2D = np.genfromtxt(os.path.join(path_arrays, '_saved_predict_val_CNN2D_tflite.csv'), delimiter=',', dtype = int)


DB_ori_from_pkl = pd.read_pickle(os.path.join(path, pkl_features))

# Separate 1 fold for validation and create a DB for the training / testing
DB_ori_from_pkl_VAL = DB_ori_from_pkl[DB_ori_from_pkl['Fold'] == fold_name].copy()
DB_ori_from_pkl_TRN = DB_ori_from_pkl[DB_ori_from_pkl['Fold'] != fold_name].copy()

print(f'Validation set...: {len(DB_ori_from_pkl_VAL)} samples')
print(f'Training set.....: {len(DB_ori_from_pkl_TRN)} samples' )
print(f'Total............: {len(DB_ori_from_pkl_VAL) + len(DB_ori_from_pkl_TRN)} samples \n')

audio_val = np.array(DB_ori_from_pkl_VAL.Audio.to_list())
y_val     = np.array(DB_ori_from_pkl_VAL.Class_OHEV.to_list())
y_val_cat = np.array(DB_ori_from_pkl_VAL.Class_categorical.to_list())

# Clear up the memory
del DB_ori_from_pkl, DB_ori_from_pkl_TRN


Classifiers = ['SVC', 'LR', 'RF', 'ANN', 'CNN1D', 'CNN2D']

# #Checking if the script is OK before running a long batch
# ESR_EVAL       = ESR_evaluation_tflite(audio_val[0:100], 'CNN2D', path_modelsVal, path_arrays)
# predictions    = np.array(ESR_EVAL.predictions)
# totalPredTime  = np.array(ESR_EVAL.totalPredTime)
# 
# print(f'Average prediction time...: {np.average(totalPredTime):.4f} ms')

batchLen  = 10
dic_evalT = {}

# Start the time counter
startTimer = time.perf_counter_ns()

for i in range(batchLen):
    print("=================================================================")
    print(f'Batch...: {i}\n')
    dic_eval  = {}
    for classifier in Classifiers:
        print(f'Classifier: {classifier}\n')
        ESR_EVAL       = ESR_evaluation_tflite(audio_val, classifier, path_modelsVal, path_arrays)
        predictions    = np.array(ESR_EVAL.predictions)
        totalPredTime  = np.array(ESR_EVAL.totalPredTime)

        dic_eval[f'predictions_{classifier}']   = predictions
        dic_eval[f'totalPredTime_{classifier}'] = totalPredTime
        
    dic_evalT[f'predictions_{i}'] = dic_eval
    
# Stop the time counter
endTimer = time.perf_counter_ns()
totalPredTime = ((endTimer - startTimer) / 1000000000/60)
print(f'\n\nTotal prediction time for the batch..: {totalPredTime:.4f} min\n')

# Save the results as numpy file
np.save(os.path.join(current_path, '_dic_evalT_tflite_raspi.npy'), dic_evalT)

for key in dic_evalT.keys():
    print(f'Classifier: CNN2D\n')
    print(key)
    PTaverage = np.average(dic_evalT[key]['totalPredTime_CNN2D'])
    PTstd     = np.std(dic_evalT[key]['totalPredTime_CNN2D'])
    print(f'Average prediction time...: {PTaverage:.4f}ms +\- {PTstd:.4f}ms\n')
    

# Write the total prediction time of each classifier in each loop

for key in dic_evalT.keys():

    with open(os.path.join(current_path, '_totalPredTime_tflite_raspi.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header SUPPORT VECTOR CLASSIFIER
        writer.writerow(['Key', 'Total Prediction Time', 'totalPredTime_SVC'])
        # Write data
        for key in dic_evalT.keys():
            total_pred_time = np.array(dic_evalT[key]['totalPredTime_SVC']).tolist()
            writer.writerow([key, total_pred_time])

        # Write header LOGISTIC REGRESSION
        writer.writerow(['Key', 'Total Prediction Time', 'totalPredTime_LR'])
        # Write data
        for key in dic_evalT.keys():
            total_pred_time = np.array(dic_evalT[key]['totalPredTime_LR']).tolist()
            writer.writerow([key, total_pred_time])
            
        # Write header RANDOM FOREST
        writer.writerow(['Key', 'Total Prediction Time', 'totalPredTime_RF'])
        # Write data
        for key in dic_evalT.keys():
            total_pred_time = np.array(dic_evalT[key]['totalPredTime_RF']).tolist()
            writer.writerow([key, total_pred_time])
            
        # Write header LOGISTIC ARTIFICIAL NEURAL NETWORK
        writer.writerow(['Key', 'Total Prediction Time', 'totalPredTime_ANN'])
        # Write data
        for key in dic_evalT.keys():
            total_pred_time = np.array(dic_evalT[key]['totalPredTime_ANN']).tolist()
            writer.writerow([key, total_pred_time])
            
        # Write header CONVOLUTIONAL NEURAL NETWORK 1D
        writer.writerow(['Key', 'Total Prediction Time', 'totalPredTime_CNN1D'])
        # Write data
        for key in dic_evalT.keys():
            total_pred_time = np.array(dic_evalT[key]['totalPredTime_CNN1D']).tolist()
            writer.writerow([key, total_pred_time])
            
        # Write header CONVOLUTIONAL NEURAL NETWORK 2D
        writer.writerow(['Key', 'Total Prediction Time', 'totalPredTime_CNN2D'])
        # Write data
        for key in dic_evalT.keys():
            total_pred_time = np.array(dic_evalT[key]['totalPredTime_CNN2D']).tolist()
            writer.writerow([key, total_pred_time])

print('Output has been written to _totalPredTime_tflite_raspi.csv')


predictionsName  = ['predictions_SVC', 
                    'predictions_LR', 
                    'predictions_RF', 
                    'predictions_ANN', 
                    'predictions_CNN1D', 
                    'predictions_CNN2D']

# Check predictions

for key in dic_evalT.keys():
    savedPredictions = [saved_predict_val_SVC,
                        saved_predict_val_LR, 
                        saved_predict_val_RF, 
                        saved_predict_val_ANN, 
                        saved_predict_val_CNN1D, 
                        saved_predict_val_CNN2D]
    print(f'\n{key}\n')
    for prediction, savedPrediction in zip(predictionsName, savedPredictions):
        print(prediction)
        checkPred = np.array(dic_evalT[key][prediction])
        print(np.array_equal(checkPred, savedPrediction))