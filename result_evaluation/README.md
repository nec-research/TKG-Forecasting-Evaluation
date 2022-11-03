# TKG-Forecasting-Evaluation
## Evaluation of models based on created .pkl files

## 1. Requirements
Install:
* torch
* numpy
* os
* enum
* pickle
* json


## 2. Prerequisites
* In the folder result_evaluation create a folder for each model that should be evaluated
* ['TANGO', 'RE-GCN', 'xERTE', 'TLogic', 'CyGNet', 'RE-Net', 'Timetraveler']  
* New models have to be added in run_evaluation, dir_names, l.140
* Copy all .pkl files of interest to the respective models folder

## 3. Run Evaluation
* Run run_evaluation.py
* This creates a .json file, with one entry per .pkl file, containing scores of interest (MRR, MR, Hits@K, MRR over snapshots)
* Run parser.py to create an excel-table from this json file, with one sheet per setting.

