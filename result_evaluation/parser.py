"""/*
 *     TKG Forecasting Evaluation
 *
 *        File: parser.py
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
 */
"""

import os
import json
import pandas as pd
import numpy as np


# noinspection PyShadowingNames
def helper_function(filename: str) -> tuple:
    filename = filename.replace('.pkl', '')
    if "feedvalid" in filename:
        filename = filename.replace('feedvalid', '')
    
    filename = filename.split('-')  # Idea is to use these keywords to decide the location of values in dataframe
    _setting, _filtering, _dataset, _method = 'NA', 'NA', 'NA', 'NA'
    if 'cen' in filename:
        print(filename)
    for item in filename:
        _setting = item if item in setting else _setting
        _filtering = item if item in filtering else _filtering
        _dataset = item if item in datasets else _dataset
        # Invert the method name from small letters to a consistent format
        _method = inv_dict[item] if item in method_names.values() else _method
        if _method == 'tlogic':
            print(filename)
        if item == 'True':
            return None, None, None, None
    return _setting, _filtering, _dataset, _method


# noinspection PyShadowingNames
def normalize_sub_dicts(jsonfile: dict) -> dict:
    # normalise JSON file for redundancies in filter settings for methods with redundant values

    for method in ['TANGO', 'xERTE', 'CyGNet']:  # list of redundant methods
        sub_dict = jsonfile[method]
        list_pkl_names = sorted(list(sub_dict.keys()))  # sorting might not be required, but useful for debugging
        unique_prefixes = set(['-'.join(pkl_name.split('-')[:-1]) for pkl_name in list_pkl_names])
        assert len(unique_prefixes) * 3 == len(list_pkl_names), f'Missing pickle file for method `{method}`'

        list_unique_pkl_names = [f'{prefix}-raw.pkl' for prefix in unique_prefixes]
        # Create a normalized sub-dictionary of exactly same structure
        normalized_sub_dict = {pkl_name: {_filter: dict() for _filter in ['raw', 'static', 'time']}
                               for pkl_name in list_unique_pkl_names}

        # Idea is to keep just **raw.pkl name and replace its `time & static` values
        for unq_prefix, unq_pkl_name in zip(unique_prefixes, list_unique_pkl_names):
            raw = jsonfile[method][unq_pkl_name]['raw']
            static = jsonfile[method][f'{unq_prefix}-static.pkl']['static']
            time = jsonfile[method][f'{unq_prefix}-time.pkl']['time']

            normalized_sub_dict[unq_pkl_name]['raw'] = raw
            normalized_sub_dict[unq_pkl_name]['static'] = static
            normalized_sub_dict[unq_pkl_name]['time'] = time

        # Update the original sub_dict in jsonfile with normalized sub_dict
        jsonfile[method] = normalized_sub_dict


if __name__ == '__main__':

    ROOT = os.path.join(os.getcwd())
    with open(os.path.join(ROOT, 'output4.json'), 'r') as stream:
        jsonfile = json.load(stream)
    normalize_sub_dicts(jsonfile)

    setting = ['multistep', 'singlestep', 'singlesteponline']
    filtering = ['time', 'raw', 'static']
    metrics = ['mrr', 'hits@1', 'hits@3', 'hits@10']
    datasets = ['GDELT', 'YAGO', 'WIKI', 'ICEWS14', 'ICEWS18']
    method_names = {        
        'RE-GCN': 'regcn',
        'RE-Net': 'renet',
        'xERTE': 'xerte',
        'CyGNet': 'cygnet',
        'TLogic': 'tlogic',
        'TANGO': 'tango',
        'Timetraveler': 'titer',
        'CEN': 'cen'        
    }
    inv_dict = {value: key for key, value in method_names.items()}  # used in helper function
    assert len(method_names.keys()) == len(jsonfile.keys()), 'Reports for all methods not present in jsonfile!'

    # Initialise variables relating to the dataframe
    column_names = [f'{dataset}_{metric}' for dataset in datasets for metric in metrics]
    index_names = [f'{method_name}_{step}' for step in setting for method_name in method_names.keys()]
    raw_df = pd.DataFrame(index=index_names, columns=column_names)
    static_df = pd.DataFrame(index=index_names, columns=column_names)
    time_df = pd.DataFrame(index=index_names, columns=column_names)
    raw_df.fillna(value='NA', inplace=True)
    static_df.fillna(value='NA', inplace=True)
    time_df.fillna(value='NA', inplace=True)

    for method_name in method_names.keys():
        sub_dict = jsonfile[method_name]

        # Iterate on each sub-dict (.pkl report values)
        for pkl_name, report in sub_dict.items():
            print(pkl_name, '\n', '=' * 100)
            _setting, _, _dataset, _method = helper_function(pkl_name)
            if _setting is None:  # special constraint check that avoids pkl files with `True` in their names
                continue

            for filter, values in report.items():
                index = f'{_method}_{_setting}'
                mrr = np.round(values[1] * 100, 2)
                hits = [np.round(value * 100, 2) for value in values[2]]

                if filter == 'raw':
                    raw_df.loc[index, f'{_dataset}_mrr'] = mrr
                    raw_df.loc[index, f'{_dataset}_hits@1'] = hits[0]
                    raw_df.loc[index, f'{_dataset}_hits@3'] = hits[1]
                    raw_df.loc[index, f'{_dataset}_hits@10'] = hits[2]
                elif filter == 'static':
                    static_df.loc[index, f'{_dataset}_mrr'] = mrr
                    static_df.loc[index, f'{_dataset}_hits@1'] = hits[0]
                    static_df.loc[index, f'{_dataset}_hits@3'] = hits[1]
                    static_df.loc[index, f'{_dataset}_hits@10'] = hits[2]
                elif filter == 'time':
                    time_df.loc[index, f'{_dataset}_mrr'] = mrr
                    time_df.loc[index, f'{_dataset}_hits@1'] = hits[0]
                    time_df.loc[index, f'{_dataset}_hits@3'] = hits[1]
                    time_df.loc[index, f'{_dataset}_hits@10'] = hits[2]
                else:
                    raise Exception

    # Save the output as a .xlsx document
    writer = pd.ExcelWriter(os.path.join(ROOT, 'output4.xlsx'), engine='xlsxwriter')
    raw_df.to_excel(writer, sheet_name='raw')
    static_df.to_excel(writer, sheet_name='static')
    time_df.to_excel(writer, sheet_name='time')
    writer.save()
