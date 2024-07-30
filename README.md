# Master_thesis_Andre_Luiz_Florentino_project

### Faculdade de Engenharia Industrial - FEI

### Centro Universitário da Fundação Educacional Inaciana "Padre Sabóia de Medeiros" (FEI)


*FEI's Stricto Sensu Graduate Program in Electrical Engineering*

Concentration area: ARTIFICIAL INTELLIGENCE APPLIED TO AUTOMATION AND ROBOTICS

Master's thesis student Andre Luiz Florentino

***

The main purpose of this repository is to provide all codes in either Jupyter notebook format or native .py format, enabling interested parties to reproduce the results obtained in my master's thesis. To achieve the same results, it is essential to follow the guidelines outlined below:

1. The .py files are used in Jupyter notebooks. They will not generate meaningful results if run alone, without the appropriate context.
2. The Jupyter notebooks are intentionally numbered. It is important to run each notebook until the end before moving on to the next one, as skipping ahead may result in missing key information.
3. Some of the scripts are very time consuming, therefore, when it's recommended to run them overnight;
4. The training and testing of the models were performed on a notebook equipped with an Intel® Core™ i7-10850H CPU @2,70 GHz, 80 GB of RAM, and a Quadro T2000 graphics card with 4 GB of memory. The operational system in use was Windows 10 Enterprise. It may be necessary to adjust some of the codes if one has less computational power, and specially, less memory (RAM) capacity;
5. A minimum of 100 GB of hard disk is necessary to save all datasets, models, pictures, etc;
6. Many thanks to Valerio Velardo. Most of the charts were based on his playlist available in:
	https://www.youtube.com/@ValerioVelardoTheSoundofAI
7. The content of the folders below were intentionally ignored in this repository, except for the folders */_analysys* and */_pictures*.

![screenshot](_pictures/README_picture_1.png)
![[README_picture_1.png]]
***
### List of Jupyter notebooks

* *01_Feature_extraction_exploration.ipynb:* 

Read the datasets and perform several data exploration on features. Outputs the script (class): MT_loadDataset.py
This notebook has a lot of images related to the audio attributes and therefore it´s too big for visualization in the GitHub. Nevertheless, one can download it and run it without any further constraints.
***

* *02_PreProcessing_and_data_exploration.ipynb:* 

Pre-process the audio with augmentations and windowing techniques. Outputs the scripts (classes) for pre-processing and feature extraction: MT_audioPP.py and MT_feature_extractor.py
****

* *03_New_dataset_US8K_AV.ipynb:* 

Creates the new dataset **US8K_AV** based on the source US8K.
Creates the class SILENCE from audio files downloaded from the [Freesound.org](https://freesound.org/) project and merge it in the new dataset US8K_AV.

This jupyter notebook is the base to create the official dataset uploaded at Harvard Dataverse.

Add the link to Harvard Dataverse here.
***

* *04_ML_modeling.ipynb:* 

Machine learning modeling for the classifiers: Gaussian Naïve Bayes (GNB), Support Vector Machine (SVM), Logistic Regression (LR), K-Nearest Neighbors (k-NN), Random Forest (RF) and voting soft / hard.
***

* *05_Machine_learning_cross_validation.ipynb*

This notebook is used to run a cross-validation of the machine learning classifiers, given the k-fold specification of each dataset. The outcome is a .CSV file with the classification metrics. You can run the notebook with different models to compare results afterwards. In this dissertation, 12 combinations were utilized.

* norm_PCA_augmented
* norm_PCA_original
* norm_PCA_windowed
* norm_augmented
* norm_original
* norm_windowed
* std_PCA_augmented
* std_PCA_original
* std_PCA_windowed
* std_augmented
* std_original
* std_windowed

Additionally, one can also set the mutual information option in this script, but it will only work for the models original, augmented and windowed. In any of them, only the normalization technique will work, since mutual information does not accept negative values.

The combination "augmented_windowed" is implied for the datasets ESC-10 and BDLib2, but for US8K and US8K_AV, only the windowing technique was considered to save time during the training phase. For simplification purposes, all these models are referred to as "windowed" in the notebook.
***

* *06_Neural_network.ipynb*

This notebook covers Neural networks (ANN and CNN 1D) modeling and cross-validation of the classifiers, given the k-fold specification of each dataset. The outcome is a .CSV file with the classification metrics. You can run the notebook with different models to compare results afterwards in the same way as the previous notebook (05_Machine_learning_cross_validation.ipynb).
***

* *07_Compile_results.ipynb*

Use this notebook to compile all the results from the .CSV files into a bar plot with the standard deviation for each classifier (8x), and a box plot with the medians, quartiles, whiskers and outliers for each classifier model (12x). 
The results of the CNN 2D are also included but separated from the other plots given the model unique specification (original, augmented and windowed, all of them without normalization or PCA techniques).
***

* *08_Feature_extraction_for_CNN_2D_aggregated.ipynb*

Pre-process the audio with augmentations and windowing techniques to extract the Log-Mel spectrograms and its first and second derivatives (delta and delta delta). These features and then aggregated (stacked up) to create an "image" of 180 mel frequencies (60 + 60 + 60) x 44 frames for the windowed model. For the original or augmented model, the number of frames is related to the normalized audio duration:
* ESC-10 (5 s) = 180 x 216
* BDLib2 (10s) = 180 x 431
* US8K (4s) = 180 x 173
* US8K_AV (4s) = 180 x 173
***

* *09_CNN_2D.ipynb*

This notebook covers the Convolutional Neural Network 2D (CNN 2D) modeling and cross-validation of two reference architectures: Su et al. (2019) [@Su2020](https://www.mdpi.com/1424-8220/19/7/1733) and Luz et al. (2021) [@Luz2021](https://www.sciencedirect.com/science/article/abs/pii/S0003682X20309245) , given the k-fold specification of each dataset. The outcome is a .CSV file with the classification metrics.
It´s important to mention that both architectures were utilized as baseline but several features were changed such as regularization techniques, strides, fully connected layers, optimizers, etc... 
***

* *10_ESR_data_preparation.ipynb*

The purpose of this notebook is to review the results of the classifiers SVC, LR and RF (machine learning and ensemble method) and the models ANN, CNN 1D and CNN2 for the dataset US8K_AV, retrieving saved models from previous notebooks. 
For all of the them, the validation set was defined as the fold '1'. The trained models utilized were created from the scripts *04_ML_modeling.ipynb* and *06_Neural_network.ipynb*.
At the end, the original audios from the validation set are windowed and evaluated using the trained classifiers/models. The results are printed in the screen window by window, including the the result of the most likely class for the audio clip (7 windows) and the average total prediction time. All the results are saved as .CSV and .PKL files to be compared afterwards with the results from the Raspberry Pi.
***

* *11_ESR_evaluation.ipynb*

Use this notebook to run *n* batches of evaluation on the original audio in order to compile the total prediction time and memory profile of each classifier/model. The output is saved as .CSV and .PKL files to be compared afterwards with the results from the Raspberry Pi.
The images shown in this notebook were created in the Excel using the .CSV file *_totalPredTime.csv* and *_totalPredMem_jupyter.csv*.
***

 * RUN the Python file *MT_ESR_for_memory_profile.py* in a IDE:

To evaluate the memory allocation to load all libraries and run the predictions. I was a bit lazy here and didn't optimize the code, so one has to run the script _n_ times (suggested) 10x and specifying the the target classifier and target interval.

The results have to be compiled manually in Excel in order to produce one of the chart _memory_usage_ shown below.
***

* *12_ESR_NN_conversion_quantization.ipynb*

Use this notebook to convert the neural network saved models into TFLite format to be deployed (embedded) in the Raspberry Pi. The conversion (quantization) was set as default and the differences in the accuracy results were negligible (ANN slightly worse whereas CNN1D and CNN2D better).
***

* *13_ESR_evaluation_tflite.ipynb*

This notebook runs almost the same cells as the notebook *11_ESR_evaluation.ipynb*, the difference lies in the NN models that were converted to TFLite format. Again, some files are saved for further processing in Excel (*_totalPredTime_tflite.csv* and *_totalPredMem_jupyter_tflite.csv*) or just as documentation for comparison. 
***

* *14_Recording_real_time_audio.ipynb*

Tests and trials to record live audio from a microphone.
***

 * *15_ESR.ipynb*

This notebook performs the live prediction, capturing the sound from the microphone and passing it to a thread to run the algorithm that will frame the audio, extract its features, and finally run the prediction based on ONE specific classifier / model, in this case, it's set up as the CNN 2D.

For the evaluation flow in the Raspberry Pi, an .PY script was created to optimize the performance.
***

 ### List of scripts .PY 

In order of utilization in the Jupyter notebooks:

* MT_loadDataset.py
* MT_audioPP.py
* MT_feature_extractor.py
* MT_audioPPFE_ML.py
* MT_ESR_evaluation.py
* MT_ESR_evaluation_tflite.py
* MT_ESR_for_memory_profile.py (the only one that outputs meaningful results standalone)

This is the final file that runs live predictions from a microphone. It's based on the jupyter notebook *15_ESR.ipynb*.
* MT_ESR.py 

***

### Compilation of the results
#### Accuracy

**ESC-10**
![screenshot](_analysis/07_Compile_results_ESC-10_01.png)

![screenshot](_analysis/07_Compile_results_ESC-10_02.png)

**BDLi2**
![screenshot](_analysis/07_Compile_results_BDLib2_03.png)

![screenshot](_analysis/07_Compile_results_BDLib2_04.png)

**US8K**
![screenshot](_analysis/07_Compile_results_US8K_05.png)

![screenshot](_analysis/07_Compile_results_US8K_06.png)

**US8K_AV**
![screenshot](_analysis/07_Compile_results_US8K_AV_07.png)

![screenshot](_analysis/07_Compile_results_US8K_AV_08.png)

All picture in one....

![screenshot](_pictures/README_picture_2.png)
![[README_picture_2.png]]

Comparison in Excel for the CNN 2D (Larger pictures available in the notebook *09_CNN_2D.ipynb*)

![screenshot](_pictures/README_picture_3.png)
![[README_picture_3.png]]



***
#### Total prediction time and memory profile - Notebook


![screenshot](_pictures/README_picture_4.jpg)

![[README_picture_4.jpg]]


![screenshot](_pictures/README_picture_5.jpg)

![[README_picture_5.jpg]]


![screenshot](_pictures/README_picture_6.jpg)

![[README_picture_6.jpg]]

#### Total prediction time and memory profile - Raspberry Pi

![screenshot](_pictures/README_picture_7.jpg)

![[README_picture_7.jpg]]

![screenshot](_pictures/README_picture_8.jpg)

![[README_picture_8.jpg]]

![screenshot](_pictures/README_picture_9.jpg)

![[README_picture_9.jpg]]

