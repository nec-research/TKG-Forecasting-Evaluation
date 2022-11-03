# """/*
#  *     TKG Forecasting Evaluation
#  *
#  *        File: run_exp.sh
#  *
#  *     Authors: Deleted for purposes of anonymity 
#  *
#  *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
#  * 
#  * The software and its source code contain valuable trade secrets and shall be maintained in
#  * confidence and treated as confidential information. The software may only be used for 
#  * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
#  * license agreement or nondisclosure agreement with the proprietor of the software. 
#  * Any unauthorized publication, transfer to third parties, or duplication of the object or
#  * source code---either totally or in part---is strictly prohibited.
#  *
#  *     Copyright (c) 2021 Proprietor: Deleted for purposes of anonymity
#  *     All Rights Reserved.
#  *
#  * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR 
#  * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY 
#  * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT 
#  * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION. 
#  * 
#  * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
#  * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE 
#  * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
#  * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
#  * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
#  * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
#  * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
#  * THE POSSIBILITY OF SUCH DAMAGES.
#  * 
#  * For purposes of anonymity, the identity of the proprietor is not given herewith. 
#  * The identity of the proprietor will be given once the review of the 
#  * conference submission is completed. 
#  *
#  * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#  */"""

# '''
#     # parser.add_argument("--gpu", type=int, default=0, help="gpu")
#     # parser.add_argument("--model", type=int, default=3, help="1: 'CyGNet', 2: 'xERTE', 3: 'RE-Net', 4: 'RE-GCN', 5: 'Tlogic', 6:'TANGO', 7:Timetraveler")
#     # parser.add_argument("--num_seeds", type=int, default=1, help="number of repetitions 1,...,10")
#     # parser.add_argument("--exp_name_int", type=int, default=0, help="experiment name. if higher than 0: will be added to the run-num for logging")
#     # parser.add_argument("--dataset_ids", type=int, nargs="+", default=0, help="1: 'ICEWS18', 2:'ICEWS05-15', 3:'ICEWS14', 4:'YAGO', 5:'WIKI', 6:'GDELT', 0:'all'")
#     # parser.add_argument("--setseed", type=int, default=0, help="if 1: we set seed manually, if 0: do not set seed")
# # size datasets gdelt > wiki > ice0515 > ice18 > yago >ice14
# '''


eval "$(conda shell.bash hook)"
#cygnet
conda activate cygnet
python3 run.py --gpu 2 --model 1 --num_seeds 1 --exp_name_int 0 --dataset_ids 3 #1 3 4 5 6

#tlogic
conda activate tlogic - runs on cpu
python3 run.py --gpu 2 --model 5 --num_seeds 1 --exp_name_int 0 --setseed 1 --dataset_ids 3 #1 3 4 5 6

#regcn
conda activate regcn 
python3 run.py --gpu 2 --model 4 --num_seeds 1 --exp_name_int 0 --dataset_ids 3 #1 3 4 5 6

#timetraveler
conda activate titer
python3 run.py --gpu 2 --model 7 --num_seeds 1 --exp_name_int 0 --dataset_ids 3 #1 3 4 5 6

# tango
conda activate tango
python3 run.py --gpu 0 --model 6 --num_seeds 1 --exp_name_int 0 --dataset_ids 3 #1 3 4 5 6

# renet
conda activate renet # renet only runs on gpu 0 due to implementation issue.
python3 run.py --gpu 0 --model 3 --num_seeds 1 --exp_name_int 0 --dataset_ids 3 #1 3 4 5 6

# xerte
conda activate xerte
python3 run.py --gpu 0 --model 2 --num_seeds 1 --exp_name_int 0 --dataset_ids 3 #1 3 4 5 6
