# """/*
#  *     TKG Forecasting Evaluation
#  *
#  *        File: evaluation_utils.py
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

import torch
import numpy as np


def create_scores_tensor(predictions_dict, num_nodes, device=None):
    """ for given dict with key: node id, and value: score -> create a tensor with num_nodes entries, where the score 
    from dict is enetered at respective place, and all others are zeros.

    :returns: predictions  tensor with predicted scores, one per node; e.g. tensor([ 5.3042,  6....='cuda:0') torch.Size([23033])
    """
    predictions = torch.zeros(num_nodes, device=device)
    for node_id in predictions_dict.keys():
        predictions[node_id] = predictions_dict[node_id]
    return predictions

def query_name_from_quadruple_cygnet(quad, ob_pred=True):
    """ get the query namefrom the given quadruple. 
    :param quad: numpy array, len 4: [sub, rel, ob, ts]; 
    :param ob_pred: [bool] true: the object is predicted, false: the subject is predicted
    :return: 
    query_name [str]: name of the query, with xxx showing the entity of interest. e.g.'30_13_xxx_334' for 
        object prediction or 'xxx_13_18_334' for subject prediction
    test_query_ids [np array]: sub, rel, ob, ts (original rel id)
    """
    rel = quad[1]
    ts = quad[3]
    sub = quad[0]
    ob = quad[2]
    
    if ob_pred == True:
        query_name = str(sub) + '_' + str(rel) + '_' + 'xxx'+ str(ob) +'_' + str(ts)
    else:
        query_name = 'xxx'+ str(sub)+ '_' + str(rel) + '_' + str(ob) + '_'  + str(ts)

    test_query_ids = np.array([sub, rel, ob, ts])

    return query_name, test_query_ids

def query_name_from_quadruple(quad, num_rels, plus_one_flag=False):
    """ get the query namefrom the given quadruple. if they do reverse prediction with nr*rel+rel_id then we undo it here
    :param quad: numpy array, len 4: [sub, rel, ob, ts]; if rel>num_rels-1: this means inverse prediction
    :param num_rels: [int] number of relations
    :param plus_one_flag: [Bool] if the number of relations for inverse predictions is one higher than expected - the case for timetraveler:self.quadruples.append([ex[2], ex[1]+num_r+1, ex[0], ex[3]]
    :return: 
    query_name [str]: name of the query, with xxx showing the entity of interest. e.g.'30_13_xxx_334' for 
        object prediction or 'xxx_13_18_334' for subject prediction
    test_query_ids [np array]: sub, rel, ob, ts (original rel id)
    """
    rel = quad[1]
    ts = quad[3]
    if rel > (num_rels-1): #wrong direction
        
        ob_pred = False
        if plus_one_flag == False:
            rel = rel - (num_rels) 
        else:
            rel = rel - (num_rels) -1 
        sub = quad[2]
        ob = quad[0]
    else:
        ob_pred = True
        sub = quad[0]
        ob = quad[2]      
    
    if ob_pred == True:
        query_name = str(sub) + '_' + str(rel) + '_' + 'xxx'+ str(ob) +'_' + str(ts)
    else:
        query_name = 'xxx'+ str(sub)+ '_' + str(rel) + '_' + str(ob) + '_'  + str(ts)
    
    test_query_ids = np.array([sub, rel, ob, ts])
    return query_name, test_query_ids

def get_total_number(inPath, fileName):
    """ return number of nodes and number of relations
    from renet utils.py
    """
    import os
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def load_scores(directory:str, method_name:str, dataset:str, query_name:str, device:str):
    
    """ load scores from pt file from folder resultscores
    scores_dict is a dict with keys: 'predictions' 'ground_truth' and values: tensors
    for the predictions it is a tensor with a score for each node -> e.g. 23000 entries for ICEWS18
    for the ground truth it is one entry with the ground truth node
    :param directory: [str] directory, usually e.g. '/home/jgastinger/tempg/Baseline Evaluation'
    :param method_name: [str] e.g. renet
    :param dataset: [str] e.g. ICEWS18
    :param query_name: [str] e.g. "xxx_1_235_24" -> the xxx is the element in question; order: subid_relid_obid_timestep
    :returns: scores: tensor with predicted scores, one per node; gt: tensor with the id of the ground truth node
    """
    print("HI")
    dir_results = directory + '/resultscores'+ '/' + dataset
    name = method_name  + query_name +'.pt'
    location = dir_results + '/' + name
    scores_dict = torch.load(location, map_location=torch.device(device))
    scores = scores_dict['predictions']
    gt = scores_dict['ground_truth']
    return scores, gt

