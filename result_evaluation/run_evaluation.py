"""/*
 *     TKG Forecasting Evaluation
 *
 *        File: run.py
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
from enum import unique
import os
import pickle
import json
import torch
from src import utils
import numpy as np
import testfunction


# noinspection PyShadowingNames
def length_consistency(dataset_name: str, file_length: int) -> None:
    """
    Checks if the length of pkl file is equal to the twice test sample size for the respective dataset as
    mentioned in the pickle filename
    :str pickle_filename: Name of the pickle file
    :int file_length: Length of the pkl file
    """
    print(f"file_length for dataset {dataset_name} is {file_length}, withtest_samples being {test_sample_size[dataset_name]}")
    if dataset_name == 'WIKI': #wiki has less triples bec. some quadruples in the test set are duplicates
        wiki_samples = 123768
        error_msg = f'{pickle_filename} length {file_length} != (2 * {wiki_samples}) [wiki test samples in {dataset_name}]'
        assert (wiki_samples) == file_length, error_msg
    else:
        test_samples = test_sample_size[dataset_name]
        error_msg = f'{pickle_filename} length {file_length} != (2 * {test_samples}) [test samples in {dataset_name}]'
        assert (2 * test_samples) == file_length, error_msg
    

# noinspection PyShadowingNames
def restructure_pickle_file(pickle_file: dict, num_rels: int) -> list:
    """
    Restructure the pickle format to be able to use the functions in RE-GCN implementations.
    The main idea is to use them as tensors so itspeeds up the computations
    :param pickle_file:
    :param num_rels:
    :return:
    """

    test_triples, final_scores, timesteps = [], [], []
    for query, scores in pickle_file.items():
        timestep = int(query.split('_')[-1])
        timesteps.append(timestep)
    timestepsuni = np.unique(timesteps)  # list with unique timestamps

    timestepsdict_triples = {}  # dict to be filled with keys: timestep, values: list of all triples for that timestep
    timestepsdict_scores = {}  # dict to be filled with keys: timestep, values: list of all scores for that timestep

    for query, scores in pickle_file.items():
        timestep = int(query.split('_')[-1])
        triple = query.split('_')[:-1]
        triple = np.array([int(elem.replace('xxx', '')) if 'xxx' in elem else elem for elem in triple], dtype='int32')
        if query.startswith('xxx'):                 # then it was subject prediction -
            triple = triple[np.argsort([2, 1, 0])]  # so we have to turn around the order
            triple[1] = triple[1] + num_rels  # and the relation id has to be original+num_rels to indicate it was
            # other way round

        if timestep in timestepsdict_triples:
            timestepsdict_triples[timestep].append(torch.tensor(triple))
            timestepsdict_scores[timestep].append(torch.tensor(scores[0]))
        else:
            timestepsdict_triples[timestep] = [torch.tensor(triple)]
            timestepsdict_scores[timestep] = [torch.tensor(scores[0])]

    for t in np.sort(list(timestepsdict_triples.keys())):
        test_triples.append(torch.stack(timestepsdict_triples[t]))
        final_scores.append(torch.stack(timestepsdict_scores[t]))

    return timestepsuni, test_triples, final_scores


def setup(dataset_name: str, pickle_file: dict):
    """
    Fetch required dependencies to implement utils.get_total_rank() from src code
    """
    directory = None #'All'  # the consistent datasets

    data = utils.load_data(dataset_name, directory)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    # for time-aware filter:
    all_ans_list_test = utils.load_all_answers_for_time_filter(
        data.test, num_rels, num_nodes, False)  # list with one entry per test timestep for static filter:

    # for static filter:
    all_data = np.concatenate((data.train, data.valid, data.test), axis=0)
    all_ans_static = utils.load_all_answers_for_filter(
        all_data, num_rels, False)  # not time ordered -> it's not a list with one
    # dict per timestep but a dict with all triple-combis
    # (Bordes et al 2013) propose to remove all triples (except the triple of interest) that appear in the
    # train, valid, or test set from the list of corrupted triples.

    # pickle file:
    timesteps, test_triples, final_scores = restructure_pickle_file(pickle_file, num_rels)

    return timesteps, test_triples, final_scores, all_ans_list_test, all_ans_static



if __name__ == '__main__':
    dir_names = ['TANGO', 'RE-GCN', 'xERTE', 'TLogic', 'CyGNet', 'RE-Net', 'Timetraveler']  

    output_filename = 'output4.json'

    # load previous saved reuslts
    if os.path.exists(output_filename):
        with open(output_filename) as file:
            output = json.load(file)
    else:
        output = dict()

    test_sample_size = {
        'ICEWS14': 7371,
        'ICEWS18': 49545,
        'ICEWS05-15': 46159,
        'GDELT': 305241,
        'WIKI': 63110,
        'YAGO': 20026,
    }

    for directory in dir_names:
        if directory not in output:
            output[directory] = dict()
        # Each method is a key in output4.json; used for saving results
        pickle_files = os.listdir(directory)
        for pickle_filename in pickle_files:  # Iterate on each pickle file
            if pickle_filename[-4:] == '.pkl':
                dataset_name = pickle_filename.split('-')[1]
                # Fetch new score only if it does not exist in output.json
                if pickle_filename not in output[directory].keys():
                    try:
                        print(f'Loading: {pickle_filename}')
                        with open(os.path.join(directory, pickle_filename), 'rb') as file:
                            pickle_file = pickle.load(file)
                        output[directory][pickle_filename] = dict()

                        # Consistency check                        
                        length_consistency(dataset_name, len(pickle_file))
                        timesteps, test_triples, final_scores, all_ans_list_test, all_ans_static = \
                            setup(dataset_name, pickle_file)
                        
                        scores_raw, scores_t_filter, scores_s_filter = testfunction.test(timesteps, test_triples, final_scores,
                                                                            all_ans_list_test, all_ans_static)

                        # Save results as a dictionary object
                        if 'raw' not in output[directory][pickle_filename]:
                            output[directory][pickle_filename]['raw'] = scores_raw
                        if 'time' not in output[directory][pickle_filename]:
                            output[directory][pickle_filename]['time'] = scores_t_filter
                        if 'static' not in output[directory][pickle_filename]:
                            output[directory][pickle_filename]['static'] = scores_s_filter

                        with open(output_filename, 'w') as file:
                            json.dump(output, file, indent=4)
                    except:
                        print('did not work for ', pickle_filename)
                else:
                    print(f'Results for {pickle_filename} already exists.')
            else:
                print(f'Warning: Invalid file format {pickle_filename}')
                print('=' * 100)
