"""/*
 *     TKG Forecasting Evaluation
 *
 *        File: run_exp.sh
 *
 *     Authors: Deleted for purposes of anonymity 
 *
 *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
 * 
 * The software and its source code contain valuable trade secrets and shall be maintained in
 * confidence and treated as confidential information. The software may only be used for 
 * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
 * license agreement or nondisclosure agreement with the proprietor of the software. 
 * Any unauthorized publication, transfer to third parties, or duplication of the object or
 * source code---either totally or in part---is strictly prohibited.
 *
 *     Copyright (c) 2021 Proprietor: Deleted for purposes of anonymity
 *     All Rights Reserved.
 *
 * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY 
 * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT 
 * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION. 
 * 
 * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
 * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE 
 * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
 * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
 * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
 * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
 * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGES.
 * 
 * For purposes of anonymity, the identity of the proprietor is not given herewith. 
 * The identity of the proprietor will be given once the review of the 
 * conference submission is completed. 
 *
 * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 */"""


import os
import time
import argparse
import logging

def get_arguments_list(dataset, model, gpu, setting, feedgt='False', runnr=0, window=None, setseed=1):
    """
    Return the args for each method, according to seetings in run_exp.sh and in run.py
    and according to hyperparameter settigns as reported in paper.
    """
    if model == 'RE-Net':
        print(runnr)
        args_list = [f'--gpu {gpu} --dropout 0.5 --n-hidden 200 --lr 1e-3 --max-epochs 20 --batch-size 1024 --runnr {runnr} ',
                    f'--gpu {gpu} --dropout 0.5 --n-hidden 200 --lr 1e-3 --max-epochs 20 --batch-size 1024 --setting {setting} --runnr {runnr} ',
                    f'--gpu {gpu} --n-hidden 200 --setting {setting} --feedgt {feedgt}  --runnr {runnr}']
        print(args_list)

    elif model == 'TLogic': # size datasets gdelt > wiki > ice0515 > ice18 > yago >ice14
        if setseed == 1:
            seed = 12 #as in code
        else:
            seed = 0 #no set seed
        p = 0 # num processes for learning rules
        p2 = 0 # num processes for rule application 
        w = 0 #window size. 0 = infinite, singlestep. -1 and -200: multistep. 
        if dataset == "ICEWS18":
            p = 15
            p2 = 1
            if feedgt == False:
                w = -1
            else:
                if window == None: w = 200
                else: w = window 
        elif dataset == "ICEWS14":
            p = 16
            p2 = 1
            if feedgt == False:
                w = -1
            else:
                if window == None: w = 0
                else: w = window 
        elif dataset == "ICEWS05-15":
            p = 15
            p2 = 1
            if feedgt == False:
                w = -1
            else:
                if window == None: w = 1000 
                else: w = window         
        elif dataset == "YAGO":
            p = 15
            p2 = 1
            if feedgt == False:
                w = -1
            else:
                if window == None: w = 0
                else: w = window              
        args_list = [f'-d {dataset} --runnr {runnr} -l 1 2 3 -n 200 -p {p} --seed {seed}',
        f'-d {dataset} -r {runnr}_r[1,2,3]_n200_exp_s{seed}_rules.json -l 1 2 3 -w {w} -p {p2}  --runnr {runnr} --seed {seed}',
        f'-d {dataset} -c {runnr}_r[1,2,3]_n200_exp_s{seed}_cands_r[1,2,3]_w{w}_score_{seed}[0.1,0.5].json']
        if dataset == "WIKI" or dataset  == "GDELT": # special rule lengths for large datasets.
            p = 16
            p2 =16
            if feedgt == False:
                w = -200
            else:
                if window == None: w = 200  #needs less memory
                else: w = window   
            args_list = [f'-d {dataset} --runnr {runnr} -l 1 2 -n 200 -p {p} --seed {seed}',
            f'-d {dataset} -r {runnr}_r[1,2]_n200_exp_s{seed}_rules.json -l 1 2  -w {w} -p {p2}  --runnr {runnr} --seed {seed}',
            f'-d {dataset} -c {runnr}_r[1,2]_n200_exp_s{seed}_cands_r[1,2,3]_w{w}_score_{seed}[0.1,0.5].json']

    elif model == 'RE-GCN':
        if dataset == 'YAGO': 
            args_list = [f'--train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 1 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --runnr {runnr}',
                        f'--train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 1 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --runnr {runnr}',
                        f'--train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 1 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --multi-step --topk 0 --runnr {runnr}']
        elif dataset == 'ICEWS14':
            args_list = [f'--train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --runnr {runnr}',
                        f'--train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --runnr {runnr}',
                        f'--train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --multi-step --topk 0 --runnr {runnr}']
        elif dataset == 'ICEWS18':
            args_list = [f'--train-history-len 6 --test-history-len 6 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --runnr {runnr}',
                        f'--train-history-len 6 --test-history-len 6 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --runnr {runnr}',
                        f'--train-history-len 6 --test-history-len 6 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --multi-step --topk 0 --runnr {runnr}']
        elif dataset == 'GDELT':
            args_list = [f'--train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --runnr {runnr}',
                        f'--train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --runnr {runnr}',
                        f'--train-history-len 1 --test-history-len 1 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --multi-step --topk 0 --runnr {runnr}']
        elif dataset == 'WIKI':
            args_list = [f'--train-history-len 2 --test-history-len 2 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --runnr {runnr}',
                        f'--train-history-len 2 --test-history-len 2 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --runnr {runnr}',
                        f'--train-history-len 2 --test-history-len 2 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --multi-step --topk 0 --runnr {runnr}']
        elif dataset  == 'ICEWS05-15':
            args_list = [f'--train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --runnr {runnr}',
                        f'--train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --runnr {runnr}',
                        f'--train-history-len 10 --test-history-len 10 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu {gpu} --test --topk 0 --runnr {runnr}']
        elif model == 'CEN':
            if dataset == 'ICEWS14':
                args_list = [f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 10 --test -1  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 10 --test 0  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 7 --test 2  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 7 --test 3  --ft_lr=0.001 --norm_weight 1 --gpu {gpu}',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 7 --test 4  --ft_lr=0.001 --norm_weight 1 --gpu {gpu}']
            elif dataset == 'ICEWS18':
                args_list = [f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 10 --test -1  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 10 --test 0  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 6 --test 2  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 6 --test 3  --ft_lr=0.001 --norm_weight 1 --gpu {gpu}',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 6 --test 4  --ft_lr=0.001 --norm_weight 1 --gpu {gpu}']
            elif dataset == 'WIKI':
                args_list = [f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 2 --train-history-len 10 --test-history-len 10 --test -1  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 2 --train-history-len 10 --test-history-len 10 --test 0  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 2 --train-history-len 10 --test-history-len 2 --test 2  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 2 --train-history-len 10 --test-history-len 2 --test 3  --ft_lr=0.001 --norm_weight 1 --gpu {gpu}',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 2 --train-history-len 10 --test-history-len 2 --test 4  --ft_lr=0.001 --norm_weight 1 --gpu {gpu}']
            elif dataset == 'GDELT': #hyperparams as for WIKI
                args_list = [f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 2 --train-history-len 10 --test-history-len 10 --test -1  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 2 --train-history-len 10 --test-history-len 10 --test 0  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 2 --train-history-len 10 --test-history-len 10 --test 2  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 2 --train-history-len 10 --test-history-len 10 --test 3  --ft_lr=0.001 --norm_weight 1 --gpu {gpu}',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 2 --train-history-len 10 --test-history-len 10 --test 4  --ft_lr=0.001 --norm_weight 1 --gpu {gpu}']
            if dataset == 'YAGO': #hyperparams as for ICEWS14
                args_list = [f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 10 --test -1  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 10 --test 0  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 3 --test 2  --ft_lr=0.001 --norm_weight 1 --gpu {gpu} ',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 3 --test 3  --ft_lr=0.001 --norm_weight 1 --gpu {gpu}',
                            f'--dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction -d {dataset} --start-history-len 3 --train-history-len 10 --test-history-len 3 --test 4  --ft_lr=0.001 --norm_weight 1 --gpu {gpu}']            
    elif model == 'CyGNet':
        if dataset == 'ICEWS18':
            args_list = [f'--entity object --time-stamp 24 -alpha 0.8 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu {gpu} --batch-size 1024 --counts 4 --valid-epoch 5 --setting {setting}',
                        f'--entity subject --time-stamp 24 -alpha 0.8 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu {gpu} --batch-size 1024 --counts 4 --valid-epoch 5 --setting {setting}']

        elif dataset == 'ICEWS14':
            args_list = [f'--entity object --time-stamp 24 -alpha 0.8 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu {gpu} --batch-size 1024 --counts 4 --valid-epoch 5 --setting {setting}',
                        f'--entity subject --time-stamp 24 -alpha 0.8 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu {gpu} --batch-size 1024 --counts 4 --valid-epoch 5 --setting {setting}']

        elif dataset == 'GDELT':
            args_list = [f'--entity object --time-stamp 15 -alpha 0.7 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu {gpu} --batch-size 1024 --counts 2 --valid-epoch 1 --setting {setting}',
                        f'--entity subject --time-stamp 15 -alpha 0.7 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu {gpu} --batch-size 1024 --counts 2 --valid-epoch 1 --setting {setting}']

        elif dataset == 'YAGO':
            args_list = [f'--entity object --time-stamp 1 -alpha 0.5 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu {gpu} --batch-size 1024 --counts 4 --valid-epoch 5 --setting {setting}',
                        f'--entity subject --time-stamp 1 -alpha 0.5 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu {gpu} --batch-size 1024 --counts 4 --valid-epoch 5 --setting {setting}']

        elif dataset == 'WIKI':
            args_list = [f'--entity object --time-stamp 1 -alpha 0.5 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu {gpu} --batch-size 1024 --counts 4 --valid-epoch 5 --setting {setting}',
                         f'--entity subject --time-stamp 1 -alpha 0.5 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu {gpu} --batch-size 1024 --counts 4 --valid-epoch 5 --setting {setting}']

    elif model == 'TANGO':
        if dataset == 'ICEWS18':  #--input_step {4} days  --core_layer 2  --jump_init 1
            if feedgt == True:
                num_test_timesteps =1 
            else:
                num_test_timesteps = 34
            embsize = 200
            core_layer = 2
            score_func = 'tucker'
            scale = 0.1
            w = 1
            input_step = 4
        elif dataset == 'ICEWS14': #--input_step {4} days --scale 0.01 --jump_init 0.01
            if feedgt == True:
                num_test_timesteps =1 
            else:
                num_test_timesteps = 31
            embsize = 200
            core_layer = 2
            score_func = 'tucker'
            scale = 0.01
            w = 0.01
            input_step = 4
        elif dataset == 'GDELT': #--input_step {4} NOT SPECIFIED! using same configurations as WIKI
            if feedgt == True:
                num_test_timesteps =1 
            else:
                num_test_timesteps = 384
            embsize = 200
            core_layer = 2
            score_func = 'distmult'
            scale = 0.1
            w = 1
            input_step = 4
        elif dataset == 'YAGO': #--input_step {4} years --embsize 300 --core_layer 3 # maybe also --initsize {300} and --hiddensize {300} --score_func distmult --jump_init 1
            if feedgt == True:
                num_test_timesteps =1 
            else:
                num_test_timesteps = 6
            embsize = 300
            core_layer = 3
            score_func = 'distmult'
            scale = 0.1
            w = 1
            input_step = 4
        elif dataset == 'WIKI': #--input_step {4} years --score_func distmult --jump_init 1
            if feedgt == True:
                num_test_timesteps =1 
            else:
                num_test_timesteps = 10
            embsize = 200
            core_layer = 2
            score_func = 'distmult'
            scale = 0.1
            w = 1
            input_step = 4
        args_list = [f'--device cuda:{gpu} --dataset {dataset} --setting {setting} --target_step {num_test_timesteps} --embsize {embsize} --core_layer {core_layer} --score_func {score_func} --scale {scale} --jump_init {w} --input_step {input_step} ', #train 
            f'--device cuda:{gpu} --resume --dataset {dataset} --setting {setting} --target_step {num_test_timesteps} --embsize {embsize} --core_layer {core_layer} --score_func {score_func} --scale {scale} --jump_init {w} --input_step {input_step} --test'] #test
    elif model == 'xERTE':
        if feedgt == True:
            singleormultistep = 'singlestep'
        else:
            singleormultistep ='multistep'
        if dataset == 'ICEWS18':
            args_list = [f'--warm_start_time 48 --emb_dim 256 128 64 32 --batch_size 128 --lr 0.0002 --dataset {dataset} --epoch 10 --sampling 3 --device {gpu}  --DP_steps 3 --DP_num_edges 15 --max_attended_edges 60 --node_score_aggregation sum --ent_score_aggregation sum --ratio_update 0.75 --setting {setting} --singleormultistep {singleormultistep}']
        elif dataset == 'ICEWS14':
            args_list = [f'--warm_start_time 48 --emb_dim 256 128 64 32 --batch_size 128 --lr 0.0002 --dataset {dataset} --epoch 10 --sampling 3 --device {gpu}  --DP_steps 3 --DP_num_edges 15 --max_attended_edges 40 --node_score_aggregation sum --ent_score_aggregation sum --setting {setting} --singleormultistep {singleormultistep}']
        elif dataset == 'YAGO':
            args_list = [f'--warm_start_time 48 --emb_dim 256 128 64 32 --batch_size 128 --lr 0.0002 --dataset {dataset} --epoch 10 --sampling 3 --device {gpu}  --DP_steps 3 --DP_num_edges 15 --max_attended_edges 60 --node_score_aggregation sum --ent_score_aggregation sum --ratio_update 0.75 --setting {setting} --singleormultistep {singleormultistep}']
        else:
            args_list = [f'--warm_start_time 48 --emb_dim 256 128 64 32 --batch_size 128 --lr 0.0002 --dataset {dataset} --epoch 10 --sampling 3 --device {gpu}  --DP_steps 3 --DP_num_edges 15 --max_attended_edges 60 --node_score_aggregation sum --ent_score_aggregation sum --ratio_update 0.75 --setting {setting} --singleormultistep {singleormultistep}'] 
            # no hyperparams specified for WIKI and GDELT. I use the hyperparams from YAGO and ICEWS18, as most similar
    elif model == "Timetraveler":
        trainflag = True #set to false if only testing
        k = 305 # to cover all train timestamps for all datasets except GDELT for mle_dirichlet.py
        print('dataset ', dataset)
        if dataset =='ICEWS14' or dataset =='ICEWS0515' or dataset =='ICEWS18':
            timespan =24
            N = 50
        elif dataset == 'GDELT':
            timespan =15
            N = 60 #because closest to WIKI
            k = 2304 # to cover all train timestamps
        else:
            timespan = 1
            if dataset == 'YAGO':
                N = 30
            else: #WIKI
                N = 60
        if feedgt == True:
            singleormultistep = 'singlestep'
        else:
            singleormultistep ='multistep'
        if gpu != -1:
            print("Timetraveler only runs on gpu 0")
            if trainflag == True:
                args_list = [f'--data_dir data/{dataset} --time_span {timespan} --k {k}',
                    f'--data_path {dataset} --cuda --do_train --do_test --reward_shaping --max_action_num {N} --ent_dim {80} --IM --time_span {timespan} --singleormultistep {singleormultistep} --setting {setting}']
            else:
                print("DO NOT TRAIN")
                args_list = [f'--data_dir data/{dataset} --time_span {timespan} --k {k}',
                    f'--data_path {dataset} --cuda --do_test --reward_shaping --max_action_num {N} --ent_dim {80} --IM --time_span {timespan} --singleormultistep {singleormultistep} --setting {setting}']
               
        else:
            args_list = [f'--data_dir data/{dataset} --time_span {timespan}',
                f'--data_path {dataset} --do_train --do_test --reward_shaping --max_action_num {N} --ent_dim {80} --IM --time_span {timespan}  --singleormultistep {singleormultistep} --setting {setting}']

    return args_list

# start preprocessing, training and testing of each method as specified in args on datasets as specified in args and 
# gpu as specified in args
# hyperparams are set in get_arguments_list, following the instructions in original papers
# other settings are set in the following loop for each method, e.g. filter settings and single-/multi-step prediction.
def eval(args):
    time.sleep(5)
    root_dir = os.getcwd()

    d_dict = {1: 'CyGNet', 2: 'xERTE', 3: 'RE-Net', 4: 'RE-GCN', 5: 'TLogic', 6:'TANGO', 7:'Timetraveler', 8:'CEN'}

    model = int(args.model)
    model = d_dict[model]
    gpu = str(args.gpu)

    exp_int = args.exp_name_int
    num_seeds = args.num_seeds
    start_index = 5*exp_int
    end_index = num_seeds+5*exp_int
    
    print('model')

    log_dir = os.path.join(root_dir, 'logs', str(model) + '.log')
    logging.basicConfig(filename=log_dir, filemode='a',
                        format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.DEBUG)

    dataset_ids = args.dataset_ids
    dataset_list = [dataset_ids ] if (type(dataset_ids ) == int) else dataset_ids 
    print("dataset_list", dataset_list)
    datasets = []
    if int(dataset_list[0]) == 0:
        datasets = ['ICEWS18','ICEWS05-15', 'ICEWS14', 'YAGO', 'WIKI', 'GDELT']
    else:
        dataset_dict = {1:'ICEWS18', 2:'ICEWS05-15', 3:'ICEWS14', 4:'YAGO', 5:'WIKI', 6:'GDELT'}
        for id in dataset_list:
            datasets.append(dataset_dict[int(id)])
    print("datasets", datasets)
    
    for run in range(start_index,end_index):
        print(datasets)
        print(run)

        logging.debug('************** START ************** Experiment No: ' + str(exp_int) + " seed: " + str(run-start_index))
        for dataset in datasets:
            model_dir = os.path.join(root_dir, model)
            os.chdir(model_dir)
            if model == 'RE-Net': # renet trains always with the same metric. 
                setting ='raw'
                feedgt_list = [False] # we only know multi-step setting for RE-Net
                logging.debug('{} {} - {} - {}'.format('_' * 30, model, dataset, setting))
                args_list = get_arguments_list(dataset, model, gpu, setting, feedgt_list[0],  run)
                print(args_list)
                # ----------------------------------------------------------Train
                os.chdir(os.path.join(model_dir, 'data', dataset))
                os.system('python3 {}'.format('get_history_graph.py'))
                os.chdir(model_dir)

                logging.debug(f'Pretraining parameters: {args_list[0]}')
                os.system(f'python3 pretrain.py -d {dataset} {args_list[0]}')
                logging.debug(f'Training parameters: {args_list[1]}')
                os.system(f'python3 train.py -d {dataset} {args_list[1]}')

                # ----------------------------------------------------------Test without ground truth                     
                for feedgt in feedgt_list: # for feedgt =False (multistep) and = True (single step) settings. TODO: implement single step
                    args_list = get_arguments_list(dataset, model, gpu, setting, feedgt, run)
                    print(args_list)
                    logging.debug(f'Testing parameters: {args_list[2]}')
                    os.system(f'python3 test.py>>renettest.txt -d {dataset} {args_list[2]}' )
                    

            elif model == 'TLogic':
                setting ='time'
                model_dir = os.path.join(root_dir, model, 'mycode')
                feedgt_list = [False, True]
                window = None
                for feedgt in feedgt_list:
                    logging.debug('{} {} - {} - {}'.format('_' * 30, model, dataset, setting))
                    args_list = get_arguments_list(dataset, model, gpu, setting, feedgt, runnr=run, window=window, setseed=args.setseed)
                    
                    os.chdir(model_dir)

                    logging.debug(f'Learning parameters: {args_list[0]}')
                    os.system(f'python3 learn.py {args_list[0]}')

                    logging.debug(f'Application parameters: {args_list[1]}')
                    os.system(f'python3 apply.py {args_list[1]}')

                    logging.debug(f'Eval parameters: {args_list[2]}')
                    os.system(f'python3 evaluate.py {args_list[2]}')

            elif model == "RE-GCN":
                feedgt_list = [False, True]
                logging.debug('{} {} - {} - {}'.format('_' * 30, model, dataset, 'Raw and Time ONLY'))
                args_list = get_arguments_list(dataset, model, gpu, setting="Raw & Time", runnr=run)
                # 0th index: training
                # 1st index: testing with ground history
                # 2nd index: testing without ground history
                # ----------------------------------------------------------Train
                if dataset in ['ICEWS14', 'ICEWS18', 'ICEWS05-15']:
                    os.chdir(os.path.join(model_dir, 'data', dataset))
                    os.system('python ent2word.py') 
                os.chdir(os.path.join(model_dir, 'src'))
                logging.debug(f'Training parameters: {args_list[0]}')
                os.system('python main.py -d {} {}'.format(dataset, args_list[0]))
                for feedgt in feedgt_list:
                    # ----------------------------------------------------------Test
                    if feedgt == True:
                        logging.debug(f'Feed the GT for testing: single step')
                        logging.debug(f'Testing parameters: {args_list[1]}')
                        os.system('python main.py>>regcntest.txt -d {} {}'.format(dataset, args_list[1]))
                    else:
                        logging.debug(f'Do NOT Feed the GT for testing: multi step')
                        logging.debug(f'Testing parameters: {args_list[2]}')
                        os.system('python main.py>>regcntest.txt -d {} {}'.format(dataset, args_list[2]))            
            elif model == "CEN":
                feedgt_list = [False, True]
                logging.debug('{} {} - {} - {}'.format('_' * 30, model, dataset, 'Raw and Time ONLY'))
                args_list = get_arguments_list(dataset, model, gpu, setting="Time")
                # 0th index: Pretrain models with the minimum length.
                # 1st index: Curriculum Training.
                # 2nd index: Evaluate the offline models
                # 3rd index: Online training data: First, train the models with timestamps in the valid set
                # 4th index: Online training data: Then, train the models with timestamps in the test set
                # ----------------------------------------------------------Train
                os.chdir(os.path.join(model_dir, 'src'))
                logging.debug(f'Pre-Training parameters: {args_list[0]}')
                os.system('python main.py -d {} {}'.format(dataset, args_list[0]))
                logging.debug(f'Curriculum Training: {args_list[1]}')
                os.system('python main.py -d {} {}'.format(dataset, args_list[1]))

                for feedgt in feedgt_list:
                    # ----------------------------------------------------------Test
                    if feedgt == False:
                        logging.debug(f'Do not Feed the GT for testing: multi step')
                        logging.debug(f'Testing parameters: {args_list[2]}')
                        os.system('python main.py>>centest_entityloss.txt -d {} {}'.format(dataset, args_list[2]))
                    else:

                        logging.debug(f'Online Training parameters: {args_list[3]}')
                        os.system('python main.py -d {} {}'.format(dataset, args_list[3]))
                    
                        logging.debug(f'Do Feed the GT for testing: online learning/ step')
                        logging.debug(f'Testing parameters: {args_list[4]}')
                        os.system('python main.py>>centest_entityloss.txt -d {} {}'.format(dataset, args_list[4]))   
            elif model == "CyGNet":
                settings = ['time', 'static', 'raw'] 
                for setting in settings:
                    feedgt_list = [False] # we only know multi-step setting 
                    logging.debug('{} {} - {} - {}'.format('_' * 30, model, dataset, setting))
                    args_list = get_arguments_list(dataset, model, gpu, setting, runnr=run)
                    print('args', args_list)

                    # ----------------------------------------------------------Train
                    os.system('python {} --dataset {} {}'.format('get_historical_vocabulary.py', dataset, args_list[0]))
                    for args in args_list:
                         logging.debug(f'Training parameters: {args}')
                         os.system('python {} --dataset {} {}'.format('train.py', dataset, args))

                    # ----------------------------------------------------------Test
                    for feedgt in feedgt_list:
                        if feedgt == True:
                            multistep = False
                        else:
                            multistep = True
                        logging.debug(f'Testing parameters:  ' + str(dataset) + '__' +  str(setting) + '__' + str(multistep) + '__' + str(run))
                        os.system('python {} --dataset {} --setting {} --multi_step {} --runnr {}'.format('test.py', dataset, setting, multistep, run))
            elif model == "TANGO":
                settings = ['time', 'static', 'raw'] 
                # ----------------------------------------------------------Dataset Preprocess
                os.chdir(os.path.join(model_dir, dataset))
                os.system('python3 {}'.format('predicate_preprocess.py'))
                os.chdir(model_dir)
                for setting in settings:
                    feedgt_list = [True] # we only know single-step setting                 
                    for feedgt in feedgt_list:
                        logging.debug('{} {} - {} - {} - {}'.format('_' * 30, model, dataset, setting, feedgt))   
                        args_list = get_arguments_list(dataset, model, gpu, setting, feedgt = feedgt, runnr=run) 
                        # ----------------------------------------------------------Train 
                        logging.debug(f'Training parameters: {args_list[0]}')
                        os.system('python {} {}'.format('TANGO.py', args_list[0])) 

                        # ----------------------------------------------------------Test                  
                        logging.debug(f'Testing parameters: {args_list[1]}')
                        os.system('python {} {}'.format('TANGO.py', args_list[1])) 

            elif model == "xERTE":
                settings = ['time'] # trains on time-aware filter but logs best epoch for each filter setting to later select it when testing
                os.chdir(os.path.join(model_dir,'tKGR'))
                for setting in settings:
                    feedgt_list = [True]
                    for feedgt in feedgt_list:
                        logging.debug('{} {} - {} - {} - {}'.format('_' * 30, model, dataset, setting, feedgt))   
                        args_list = get_arguments_list(dataset, model, gpu, setting, feedgt = feedgt, runnr=run)
                        logging.debug(f'Training and testing parameters: {args_list[0]}') 
                        os.system('python {} {}'.format('train.py', args_list[0])) 

            elif model == 'Timetraveler':
                settings = ['time'] # trains on time-aware filter 
                os.chdir(os.path.join(model_dir))
                for setting in settings:
                    feedgt_list = [True] # we only know single-step setting   
                    for feedgt in feedgt_list:
                        logging.debug('{} {} - {} - {} - {}'.format('_' * 30, model, dataset, setting, feedgt))   
                        datadirstring = '--data_dir data/' + str(dataset)
                        args_list = get_arguments_list(dataset, model, gpu, setting, feedgt = feedgt, runnr=run)
                        
                        os.system('python {} {}'.format('preprocess_data.py', datadirstring))
                    
                        logging.debug(f'Dirichlet parameters: {args_list[0]}') 
                        os.system('python {} {}'.format('mle_dirichlet.py', args_list[0]))

                        logging.debug(f'Training and testing parameters: {args_list[1]}') 
                        os.system('python {} {}'.format('main.py', args_list[1])) 


    logging.debug('************** END ************** Experiment No: ' + str(exp_int))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiments')
    parser.add_argument("--gpu", type=int, default=0, help="gpu. -1: cpu, if possible with method")
    parser.add_argument("--model", type=int, default=3, help="1: 'CyGNet', 2: 'xERTE', 3: 'RE-Net', 4: 'RE-GCN', 5: 'Tlogic', 6:'TANGO', 7:Timetraveler")
    parser.add_argument("--num_seeds", type=int, default=1, help="number of repetitions 1,...,10")
    parser.add_argument("--exp_name_int", type=int, default=0, help="experiment name. if higher than 0: will be added to the run-num for logging")
    parser.add_argument("--dataset_ids", type=int, nargs="+", default=0, help="1: 'ICEWS18', 2:'ICEWS05-15', 3:'ICEWS14', 4:'YAGO', 5:'WIKI', 6:'GDELT', 0:'all'")
    parser.add_argument("--setseed", type=int, default=0, help="if 1: we set seed manually, if 0: do not set seed")
    args = parser.parse_args()

    eval(args)


