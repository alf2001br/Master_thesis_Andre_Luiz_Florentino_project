"""
Faculdade de Engenharia Industrial - FEI

Centro Universitário da Fundação Educacional Inaciana "Padre Sabóia de Medeiros" (FEI)

FEI's Stricto Sensu Graduate Program in Electrical Engineering

Concentration area: ARTIFICIAL INTELLIGENCE APPLIED TO AUTOMATION AND ROBOTICS

Master's thesis student Andre Luiz Florentino

"""

# ATENTION

# Do not use this script to profile the prediction time. The library memory_profiler takes a heavy overhead
# in the processing time.

# Use the jupyter notebook: 11_ESR_evaluation.ipynb instead.


import os
import pandas          as pd
import numpy           as np
from memory_profiler   import profile, memory_usage

current_path = os.getcwd()

path            = os.path.join(current_path, "_dataset", "US8K_AV")
path_models     = os.path.join(current_path, "US8K_AV_saved_models")
path_modelsVal  = os.path.join(current_path, "_ESR", "Saved_models_fold_1_validation")
path_arrays     = os.path.join(current_path, "_ESR", "Arrays")

fold_name       = '1'
pkl_features    = 'US8K_AV_features_original.pkl'
DB_ori_from_pkl = pd.read_pickle(os.path.join(path_models, pkl_features))
DB_ori_from_pkl = DB_ori_from_pkl[['Audio', 'Class_categorical', 'Class_OHEV', 'Fold']]

# Separate fold "1" for validation
DB_ori_from_pkl_VAL = DB_ori_from_pkl[DB_ori_from_pkl['Fold'] == fold_name].copy()

del DB_ori_from_pkl

audio_val = np.array(DB_ori_from_pkl_VAL.Audio.to_list())
y_val     = np.array(DB_ori_from_pkl_VAL.Class_OHEV.to_list())
y_val_cat = np.array(DB_ori_from_pkl_VAL.Class_categorical.to_list())


"================================================== START ============================================="

Classifiers = ['SVC', 'LR', 'RF', 'ANN', 'CNN1D', 'CNN2D']

@profile
def func(classifier: str, interval: tuple):
    ESR_EVAL      = evaluator(audio_val[interval[0]:interval[1]], classifier, path_modelsVal, path_arrays)
    predictions   = np.array(ESR_EVAL.predictions)
    totalPredTime = np.array(ESR_EVAL.totalPredTime)

    return predictions, totalPredTime


if __name__ == '__main__':

    opc = 0
    while str(opc) not in '12':
        print()
        print("1-) Tensorflow")
        print("2-) Tensorflow lite")

        opc = input("\nSelect the model: ")
        if opc.isdigit():
            opc = int(opc)
        else:
            opc = 0

    if opc == 1:
        from MT_ESR_evaluation import ESR_evaluation

        evaluator = ESR_evaluation

    else:
        from MT_ESR_evaluation_tflite import ESR_evaluation_tflite

        evaluator = ESR_evaluation_tflite

    # Manual loop 10x on each classifier to get the prediction + all lybraries memory allocation
    # Atention: looping will mask the results unless all previous process are destroyed in the memory
    # Interval of 10 audio clips is enough to average the result. A larger interval will take heavy
    # memory overhead in the Raspberry Pi.

    func('CNN2D', (0,10))

    # mem_usage = memory_usage((func, (), {'classifier': 'CNN2D', 'interval': (0,1)}), max_usage=True)
    # print(mem_usage)

