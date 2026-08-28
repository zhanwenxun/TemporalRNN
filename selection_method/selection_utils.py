import os
import time
import numpy as np
from tensorflow.keras.models import load_model

from selection_method.necov_method.neural_cov import CovRank
from selection_method.ranker import Ranker

# 优先级排序方法
from utils.utils import add_df


def get_rank_func(x_train, y_train, model, dataname, x_s, nb_classes=10):
    ranker = Ranker(model, x_s)
    rts_rank = ranker.dac_rank(nb_classes=nb_classes, dataname=dataname, x_train=x_train, y_train=y_train)

    return rts_rank




# 覆盖率排序方法
def get_cov_func(cov_name_list, model_path, cov_initer, x_s, y_s):
    cov_ranker = CovRank(cov_initer, model_path, x_s, y_s)
    func_list = []
    name_func_map = {
        "NAC": cov_ranker.cal_nac_cov,
        "NBC": cov_ranker.cal_nbc_cov,
        "SNAC": cov_ranker.cal_snac_cov,
        "TKNC": cov_ranker.cal_tknc_cov,
    }
    for cov_name in cov_name_list:
        func_list.append(name_func_map[cov_name])
    return func_list


def prepare_cov_ps(cov_name_list, model_path, x_s, save_path, cov_initer, y_s):
    print("prepare cov  ps ...")
    df = None
    func_list = get_cov_func(cov_name_list, model_path, cov_initer, x_s, y_s)
    for name, func in zip(cov_name_list, func_list):
        p = save_path.format(name)
        if os.path.exists(p):
            continue
        print(name)
        csv_data = {}
        csv_data["name"] = name
        s = time.time()
        rank_lst = func()
        # assert len(rank_lst) >= max_select_size
        e = time.time()
        csv_data["time"] = e - s
        np.save(p, rank_lst)
        df = add_df(df, csv_data)
    return df
