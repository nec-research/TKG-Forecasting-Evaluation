"""/*
 *    Utils for Testing for TKG Forecasting
 *
    Subset of utils function from RE-GCN source code (only keeping the relevant parts)
    https://github.com/Lee-zix/RE-GCN/blob/master/rgcn/utils.py
    Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueq
 *
 *     
"""

"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import os.path

import numpy as np
import torch
import logging
import src.knowledge_graph as knwlgrh

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################


def load_data(dataset, directory, bfs_level=3, relabel=False):
    if dataset in ['ICEWS18', 'ICEWS14', "GDELT", "ICEWS14", "ICEWS05-15","YAGO",
                     "WIKI"]:
        # path = os.path.join(os.getcwd(), 'data', directory)
        path = os.path.join(os.path.realpath('..'), 'data', directory)
        return knwlgrh.load_from_local(path, dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    return all_ans_list

def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        if latest_t != t:  
            # show snapshot
            latest_t = t
            if len(snapshot):  # appends in the list lazily i.e. when new timestamp is observed
                # load the previous batch and empty the cache
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])

    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    # Loops only for sanity check
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # edges are indices of combined (src, dst)
        uniq_r = np.unique(snapshot[:, 1])
        edges = np.reshape(edges, (2, -1))  #FIXME: unused just like in RENET
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list

def stat_ranks(rank_list, method, mode, mrr_snapshot_list):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)
    mr = torch.mean(total_rank.float())
    mrr = torch.mean(1.0 / total_rank.float())
    print("MR ({}): {:.6f}".format(method, mr.item()))
    print("MRR ({}): {:.6f}".format(method, mrr.item()))

    if mode == 'test':
        logging.debug("MR ({}): {:.6f}".format(method, mr.item()))
        logging.debug("MRR ({}): {:.6f}".format(method, mrr.item()))
        # logging.debug("MRR over time ({}): {:.6f}".format(method, mrr_snapshot_list))
    hit_scores = []
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
        if mode == 'test':
            logging.debug("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
            hit_scores.append(avg_count.item())
    return (mr.item(), mrr.item(), hit_scores, mrr_snapshot_list)


def get_total_rank(test_triples, score, all_ans, all_ans_static, eval_bz, rel_predict=0):
    '''

    :param test_triples: triples with inverse relationship.
    :param score:
    :param all_ans: dict with [s,o]:rel:[o,s] or [s,o]:[o,s]:rel per timestamp.
    :param all_ans_static: dict with [s,o]:rel:[o,s] or [s,o]:[o,s]:rel, timestep independent
    :param eval_bz: evaluation batch size
    :param rel_predict: if 1 predicts relations/link prediction otherwise entity prediction.
    :return:
    '''
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_t_rank = []
    filter_s_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        # print(rel_predict)
        if rel_predict == 1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        # raw:
        rank.append(sort_and_rank(score_batch, target))

        # time aware filter
        if rel_predict == 1:
            filter_score_batch_t = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch_t = filter_score(triples_batch, score_batch, all_ans)
        filter_t_rank.append(sort_and_rank(filter_score_batch_t, target))

        # static filter
        if rel_predict:  # if rel_predict == 1
            filter_score_batch_s = filter_score_r(triples_batch, score_batch, all_ans_static)
        else:
            filter_score_batch_s = filter_score(triples_batch, score_batch, all_ans_static)
        filter_s_rank.append(sort_and_rank(filter_score_batch_s, target))


    rank = torch.cat(rank)
    filter_t_rank = torch.cat(filter_t_rank)
    filter_s_rank = torch.cat(filter_s_rank)

    rank += 1 # change to 1-indexed
    filter_t_rank += 1
    filter_s_rank += 1

    mrr = torch.mean(1.0 / rank.float())
    filter_t_mrr = torch.mean(1.0 / filter_t_rank.float())
    filter_s_mrr = torch.mean(1.0 / filter_s_rank.float())

    return filter_s_mrr.item(), filter_t_mrr.item(), mrr.item(), rank, filter_t_rank, filter_s_rank


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        # try:
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
        # except:
        #     print('KeyError in all_ans')

    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True) # with default: stable=False; pytorch docu: "If stable is True then the sorting routine becomes stable, preserving the order of equivalent elements."
    indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
    indices = indices[:, 1].view(-1)
    return indices



