#
#      TKG Forecasting Evaluation
# 
#         File: parser.py
# 
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
