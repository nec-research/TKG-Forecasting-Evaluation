# TKG-Forecasting-Evaluation
TKG Forecasting Evaluation Paper

clone (including submodules with forked and modified orignal models): git clone --recursive https://github.com/nec-research/TKG-Forecasting-Evaluation.git

## Requirements
* for each model create a conda environment, with the following names: xerte, regcn, renet, titer, tango, cygnet, tlogic
* for each conda environment install the required packages as described by each method, see each repos requirements.txt
* for general evaluation: torch, numpy, os, time

## Set Up Experiments
* make sure to uncomment each model of interest in run_exp.sh and select the datasets of interest, as specified in the comments. For example:
'conda activate regcn 
 python3 run.py --gpu 1 --model 4 --num_seeds 1 --exp_name_int 0 --dataset_ids 1 3 4 5 6'
* if desired: check the desired hyperparameters and evaluation settings in run.py. multi-step and single-step setting can be set for each method with feedgt_list = [False, True]. False means multi-step, and True means single-step
* Create a folder "Results" for each model directory

## Run Experiments
* run ./run_exp.sh

## Additional Information
* Each Models datasets are stored in the respective Models folder
* Experiments might run for long time, with total runtimes of multiple weeks
* Be aware that some models have high (GPU) memory requirements, especially for the datasets GDELT, ICEWS05-15 and WIKI

## Evaluation
* See Readme in result_evaluation

## testing for xERTE 
for xERTE: 
* modify xERTE/tKGR/load_and_test.py  according to the comments (A), (B), (C) to specify dictionaries and best epochs
* run xERTE/tKGR/load_and_test.py 

## Add new model
* Copy Code to this folder or, ideally, create a git submodule of a fork of the respective repository
* Add the datasets to the Code Folder
* Create a conda environment and install all dependencies provided by the original authors in this environment
* Make sure to fulfill all items from the checklist in paper, supplementary material
* Log the scores for each test query as implemented in the other models (see git diff) during testing, to a .pkl file, with keys: querys, values: scores and gt. For logging the scores your can use the methods as provided in evaluation_utils.py
* Add the model and hyperparameters to run.py (in eval() you need to add the model to the d_dict, and add an elif model == 'newmodel':  .... ideally, you set the model args in get_arguments_list())
* Add the model and settings to run_exp.sh
For evaluation of the new model:
* Follow steps in the results_evaluation Readme
