"""/*
 *     TKG Forecasting Evaluation
 *
 *        File: run.py
 *
#  *     Authors: Julia Gastinger (julia.gastinger@neclab.eu), Timo Sztyler, Lokesh Sharma, Anett Schuelke
#  *
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
