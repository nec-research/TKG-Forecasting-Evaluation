#
#      TKG Forecasting Evaluation
# 
#         File: run.py
# 
#  
#       Authors: Julia Gastinger (julia.gastinger@neclab.eu), Timo Sztyler, Lokesh Sharma, Anett Schuelke
#  
# NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.  

#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
#        PROPRIETARY INFORMATION ---  

# SOFTWARE LICENSE AGREEMENT

# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.

# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor. 

# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).

# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.

# COPYRIGHT: The Software is owned by Licensor.  

# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.

# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.

# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.

# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.

# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.

# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.

# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.  

# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.

# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.

# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.

# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.  

# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.

# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.

# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.

# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.

# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.

# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.

#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.


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

    elif model == 'TLogic'or model == 'TRKG-Miner': # size datasets gdelt > wiki > ice0515 > ice18 > yago >ice14
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

    d_dict = {1: 'CyGNet', 2: 'xERTE', 3: 'RE-Net', 4: 'RE-GCN', 5: 'TLogic', 6:'TANGO', 7:'Timetraveler', 8:'CEN', 9:'TRKG-Miner'}

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
            elif model == 'TRKG-Miner':
                setting = 'time'
                model_dir = os.path.join(root_dir, 'TRKG-Miner', 'mycode')
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


