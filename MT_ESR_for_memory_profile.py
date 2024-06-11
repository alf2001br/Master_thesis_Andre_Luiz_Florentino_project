import pandas          as pd
import numpy           as np
from memory_profiler   import profile, memory_usage

path             = "C:\\Andre_Florentino\\03_particular\\04_mestrado-FEI\\98_dataset\\US8K_AV\\"
fold_name        = '01'
pkl_features     = 'US8K_AV_features_original.pkl'
DB_ori_from_pkl  = pd.read_pickle(path + pkl_features)
DB_ori_from_pkl  = DB_ori_from_pkl[['Audio', 'Class_categorical', 'Class_OHEV', 'Fold']]

# Separate fold "1" for validation and create a DB for the training / testing
DB_ori_from_pkl_VAL = DB_ori_from_pkl[DB_ori_from_pkl['Fold'] == fold_name].copy()
del DB_ori_from_pkl

audio_val = np.array(DB_ori_from_pkl_VAL.Audio.to_list())
y_val     = np.array(DB_ori_from_pkl_VAL.Class_OHEV.to_list())
y_val_cat = np.array(DB_ori_from_pkl_VAL.Class_categorical.to_list())


"================================================== START ============================================="

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


Classifiers = ['LR', 'RF', 'ANN', 'CNN1D', 'CNN2D']

@profile
def func(classifier):
    ESR_EVAL = evaluator(audio_val[0:1], classifier)
    predictions = np.array(ESR_EVAL.predictions)
    totalPredTime = np.array(ESR_EVAL.totalPredTime)

    return predictions, totalPredTime


if __name__ == '__main__':
    func('CNN2D')
    # mem_usage = memory_usage((func, (), {'classifier': 'LR'}), max_usage=True)
    # print(mem_usage)
