"""/*
 *     Testing for TKG Forecasting
 *
    Subset of test function from RE-GCN source code (only keeping the relevant parts)
    https://github.com/Lee-zix/RE-GCN/blob/master/src/main.py  test()
    Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueq
 *
 *     
"""
from src import utils
def test(timesteps: list, test_triples: list, final_scores: list, all_ans_list_test: list, all_ans_static):
    """
    Subset of test function from RE-GCN source code (only keeping the relevant parts)
    https://github.com/Lee-zix/RE-GCN/blob/master/src/main.py  test()
    Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang and Xueqi Cheng. 
    Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning. SIGIR 2021.
    Note: there were many changes in the utils mainly related to data structures
    :param timesteps: list of timestep values in the test set
    :param test_triples: list of tensors (no.of triples, 3)
    :param final_scores: list of tensors (no. of triples, num_nodes)
    :param all_ans_list_test: some dictionary structure as per logic in source code
    :param all_ans_static: some dictionary structure as per logic in source code TODO describe
    :return:
    """
    ranks_raw, ranks_t_filter, ranks_s_filter, mrr_raw_list, mrr_t_filter_list, mrr_s_filter_list = [], [], [], [], [], []
    assert len(timesteps) == len(test_triples) == len(final_scores) == len(all_ans_list_test)
    timesteps = list(range(len(timesteps)))  # rename to match the standard of all_and_list_test
    for time_idx, test_triple, final_score in zip(timesteps, test_triples, final_scores):
        mrr_s_filter_snap, mrr_t_filter_snap, mrr_snap, rank_raw, rank_t_filter, rank_s_filter = utils.get_total_rank(
            test_triple, final_score,
            all_ans_list_test[time_idx],
            all_ans_static,
            eval_bz=1000,
            rel_predict=0)
        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_t_filter.append(rank_t_filter)
        ranks_s_filter.append(rank_s_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_t_filter_list.append(mrr_t_filter_snap)
        mrr_s_filter_list.append(mrr_s_filter_snap)

    mode = 'test'
    scores_raw = utils.stat_ranks(ranks_raw, "Entity Prediction Raw", mode, mrr_raw_list)
    scores_t_filter = utils.stat_ranks(ranks_t_filter, "Entity TimeAware Prediction Filter", mode, mrr_t_filter_list)
    scores_s_filter = utils.stat_ranks(ranks_s_filter, "Entity Static Prediction Filter", mode, mrr_s_filter_list)

    return scores_raw, scores_t_filter, scores_s_filter
