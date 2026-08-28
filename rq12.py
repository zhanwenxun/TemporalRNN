import math
import argparse
import numpy as np
import pandas as pd
from statics import *
# from selection_tools import get_selection_information_new
from selection_tools import ats_selection, ats_selection_rank, get_selection_information_vectorized, sc_cam_selection
import keras
import datetime
import sys
import os
import re
import tensorflow as tf
import xlsxwriter

# Specify that the first GPU is available, if there is no GPU, apply: "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   # Do not occupy all of the video memory, allocate on demand
sess = tf.compat.v1.Session(config=config)
print(f"[GPU状态] 是否检测到GPU: {tf.config.list_physical_devices('GPU')}")

tf.compat.v1.keras.backend.set_session(sess)

''' 查准率 '''
# RQ1: Bug Detection Rate on {10%, 20%} selected test set.
if __name__ == '__main__':
    parse = argparse.ArgumentParser("Calculate the bug detection rate for the selected dataset.")
    # 输入参数：模型文件位置、模型类型、数据集
    parse.add_argument('-model_type', required=True, choices=['lstm', 'blstm', 'gru'])
    parse.add_argument('-dataset', required=True, choices=['mnist', 'snips', 'fashion', 'agnews', 'svhn'])
    args = parse.parse_args()

    # 对应3种RNN模型和5种数据集
    if args.model_type == "lstm" and args.dataset == "mnist":
        # 把28x28的图片分成28次输入，每次输入一行（28个像素）
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_lstm import MnistLSTMClassifier
        dl_model = "./RNNModels/mnist_demo/models/mnist_lstm.h5"

        lstm_classifier = MnistLSTMClassifier()
        # model会返回序列
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = MnistLSTMClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/mnist_toselect"
        wrapper_path = "./RNNModels/mnist_demo/output/lstm/abst_model/wrapper_lstm_mnist_3_10.pkl"
        # 原始和增强各拿30%，总共就是6000条数据
        total_num = 6000

    elif args.model_type == "blstm" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_blstm import MnistBLSTMClassifier

        dl_model = "./RNNModels/mnist_demo/models/mnist_blstm.h5"

        lstm_classifier = MnistBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = MnistBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/mnist_toselect"
        wrapper_path = "./RNNModels/mnist_demo/output/blstm/abst_model/wrapper_blstm_mnist_3_10.pkl"
        total_num = 6000

    elif args.model_type == "gru" and args.dataset == "mnist":
        time_steps = 28
        w2v_path = ""
        from RNNModels.mnist_demo.mnist_gru import MnistGRUClassifier

        dl_model = "./RNNModels/mnist_demo/models/mnist_gru.h5"

        lstm_classifier = MnistGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = MnistGRUClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/mnist_toselect"
        wrapper_path = "./RNNModels/mnist_demo/output/gru/abst_model/wrapper_gru_mnist_3_10.pkl"
        total_num = 6000

    elif args.model_type == "blstm" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_blstm import SnipsBLSTMClassifier
        dl_model = "./RNNModels/snips_demo/models/snips_blstm.h5"

        lstm_classifier = SnipsBLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = SnipsBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/snips_toselect"
        wrapper_path = "./RNNModels/snips_demo/output/blstm/abst_model/wrapper_blstm_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2000

    elif args.model_type == "gru" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_gru import SnipsGRUClassifier
        dl_model = "./RNNModels/snips_demo/models/snips_gru.h5"

        lstm_classifier = SnipsGRUClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = SnipsGRUClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/snips_toselect"
        wrapper_path = "./RNNModels/snips_demo/output/gru/abst_model/wrapper_gru_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2000

    elif args.model_type == "lstm" and args.dataset == "snips":
        time_steps = 16
        from RNNModels.snips_demo.snips_lstm import SnipsLSTMClassifier
        dl_model = "./RNNModels/snips_demo/models/snips_lstm.h5"

        lstm_classifier = SnipsLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/snips_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/snips_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = SnipsLSTMClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/snips_toselect"
        wrapper_path = "./RNNModels/snips_demo/output/lstm/abst_model/wrapper_lstm_snips_3_10.pkl"
        w2v_path = "./RNNModels/snips_demo/save/w2v_model"
        total_num = 2000

    elif args.model_type == "lstm" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_lstm import FashionLSTMClassifier
        dl_model = "./RNNModels/fashion_demo/models/fashion_lstm.h5"

        lstm_classifier = FashionLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = FashionLSTMClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/fashion_toselect"
        wrapper_path = "./RNNModels/fashion_demo/output/lstm/abst_model/wrapper_lstm_fashion_3_10.pkl"
        total_num = 6000

    elif args.model_type == "gru" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_gru import FashionGRUClassifier
        dl_model = "./RNNModels/fashion_demo/models/fashion_gru.h5"

        lstm_classifier = FashionGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = FashionGRUClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/fashion_toselect"
        wrapper_path = "./RNNModels/fashion_demo/output/gru/abst_model/wrapper_gru_fashion_3_10.pkl"
        total_num = 6000

    elif args.model_type == "blstm" and args.dataset == "fashion":
        time_steps = 28
        w2v_path = ""
        from RNNModels.fashion_demo.fashion_blstm import FashionBLSTMClassifier
        dl_model = "./RNNModels/fashion_demo/models/fashion_blstm.h5"

        lstm_classifier = FashionBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = FashionBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/fashion_toselect"
        wrapper_path = "./RNNModels/fashion_demo/output/blstm/abst_model/wrapper_blstm_fashion_3_10.pkl"
        total_num = 6000

    elif args.model_type == "lstm" and args.dataset == "agnews":
        time_steps = 35
        from RNNModels.agnews_demo.agnews_lstm import AgnewsLSTMClassifier
        dl_model = "./RNNModels/agnews_demo/models/agnews_lstm.h5"

        lstm_classifier = AgnewsLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/agnews_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/agnews_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = AgnewsLSTMClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        w2v_path = "./RNNModels/agnews_demo/save/w2v_model"
        to_select_path = "./gen_data/agnews_toselect"
        wrapper_path = "./RNNModels/agnews_demo/output/lstm/abst_model/wrapper_lstm_agnews_3_10.pkl"
        total_num = 4560

    elif args.model_type == "blstm" and args.dataset == "agnews":
        time_steps = 35
        from RNNModels.agnews_demo.agnews_blstm import AgnewsBLSTMClassifier
        dl_model = "./RNNModels/agnews_demo/models/agnews_blstm.h5"

        lstm_classifier = AgnewsBLSTMClassifier()
        lstm_classifier.data_path = "./RNNModels/agnews_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/agnews_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = AgnewsBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        w2v_path = "./RNNModels/agnews_demo/save/w2v_model"
        to_select_path = "./gen_data/agnews_toselect"
        wrapper_path = "./RNNModels/agnews_demo/output/blstm/abst_model/wrapper_blstm_agnews_3_10.pkl"
        total_num = 4560

    elif args.model_type == "gru" and args.dataset == "agnews":
        time_steps = 35
        from RNNModels.agnews_demo.agnews_gru import AgnewsGRUClassifier
        dl_model = "./RNNModels/agnews_demo/models/agnews_gru.h5"

        lstm_classifier = AgnewsGRUClassifier()
        lstm_classifier.data_path = "./RNNModels/agnews_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = "./RNNModels/agnews_demo/save/embedding_matrix.npy"
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = AgnewsGRUClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        w2v_path = "./RNNModels/agnews_demo/save/w2v_model"
        to_select_path = "./gen_data/agnews_toselect"
        wrapper_path = "./RNNModels/agnews_demo/output/gru/abst_model/wrapper_gru_agnews_3_10.pkl"
        total_num = 4560

    elif args.model_type == "lstm" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_lstm import SvhnLSTMClassifier
        dl_model = "./RNNModels/svhn_demo/models/svhn_lstm.h5"

        lstm_classifier = SvhnLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = SvhnLSTMClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/svhn_toselect"
        wrapper_path = "./RNNModels/svhn_demo/output/lstm/abst_model/wrapper_lstm_svhn_3_10.pkl"
        total_num = 7000

    elif args.model_type == "blstm" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_blstm import SvhnBLSTMClassifier
        dl_model = "./RNNModels/svhn_demo/models/svhn_blstm.h5"

        lstm_classifier = SvhnBLSTMClassifier()
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = SvhnBLSTMClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/svhn_toselect"
        wrapper_path = "./RNNModels/svhn_demo/output/blstm/abst_model/wrapper_blstm_svhn_3_10.pkl"
        total_num = 7000

    elif args.model_type == "gru" and args.dataset == "svhn":
        time_steps = 32
        w2v_path = ""
        from RNNModels.svhn_demo.svhn_gru import SvhnGRUClassifier
        dl_model = "./RNNModels/svhn_demo/models/svhn_gru.h5"

        lstm_classifier = SvhnGRUClassifier()
        model = lstm_classifier.load_hidden_state_model(dl_model)
        dense_classifier = SvhnGRUClassifier()
        dense_model = dense_classifier.reload_dense(dl_model)

        to_select_path = "./gen_data/svhn_toselect"
        wrapper_path = "./RNNModels/svhn_demo/output/gru/abst_model/wrapper_gru_svhn_3_10.pkl"
        total_num = 7000

    else:
        print("The model and svhn_data set are incorrect.")
        sys.exit(1)

    # Bug Detection Rate 故障检测率
    # Inclusiveness 故障查全率
    # Diversity 故障多样性
    state_w_bdr, ran_bdr, deepgini_bdr, maxp_bdr, ats_bdr, deepvec_bdr = {}, {}, {}, {}, {}, {}
    bscov_bdr, btcov_bdr, hscov_bdr, sc_ctm_bdr, sc_cam_bdr = {}, {}, {}, {}, {}

    state_w_inc, ran_inc, deepgini_inc, maxp_inc, ats_inc, deepvec_inc = {}, {}, {}, {}, {}, {}
    bscov_inc, btcov_inc, hscov_inc, sc_ctm_inc, sc_cam_inc = {}, {}, {}, {}, {}

    state_w_div, ran_div, deepgini_div, maxp_div, ats_div, deepvec_div = {}, {}, {}, {}, {}, {}
    bscov_div, btcov_div, hscov_div, sc_ctm_div, sc_cam_div = {}, {}, {}, {}, {}

    # 测试预算占比：10%，20%
    pre_li = [10, 20]
    for i in pre_li:
        state_w_bdr[i], state_w_inc[i], state_w_div[i] = [], [], []
        ran_bdr[i], ran_inc[i], ran_div[i] = [], [], []
        deepgini_bdr[i], deepgini_inc[i], deepgini_div[i] = [], [], []
        maxp_bdr[i], maxp_inc[i], maxp_div[i] = [], [], []
        ats_bdr[i], ats_inc[i], ats_div[i] = [], [], []
        deepvec_bdr[i], deepvec_inc[i], deepvec_div[i] = [], [], []  # deepvec 对应 mymethod/gini_bdr

        # 白盒方法初始化
        bscov_bdr[i], bscov_inc[i], bscov_div[i] = [], [], []
        btcov_bdr[i], btcov_inc[i], btcov_div[i] = [], [], []
        hscov_bdr[i], hscov_inc[i], hscov_div[i] = [], [], []
        sc_ctm_bdr[i], sc_ctm_inc[i], sc_ctm_div[i] = [], [], []
        sc_cam_bdr[i], sc_cam_inc[i], sc_cam_div[i] = [], [], []

    fault_type_num = 0
    # 对30个数据文件进行循环（原始和增强各取30%）
    files = [f for f in os.listdir(to_select_path) if 'aug' not in f and 'ori' not in f]
    # 逻辑：提取文件名中的所有数字，取最后一个数字转为 int 进行比较，这样可以按照0、1、2、3这样的顺序循环
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    file_index = 0
    file_num = len(files)
    for file in files:
        print("")
        # 打印时间和开始选择数据
        print("time:", datetime.datetime.now())
        print("Processing file:", file)
        # 获取数据文件地址
        file_path = to_select_path + "/" + str(file)
        # ATS方法
        ats_rank = ats_selection_rank(file_path, w2v_path, model, dense_model, time_steps)
        # 获取选取测试用例所需的信息
        weight_state, trend_set, mymethod, right, fault_types, deepgini, maxp, stellar_bscov, stellar_btcov, hscov, sc_vals, sc_sets = get_selection_information_vectorized(
            file_index, args.dataset, args.model_type, file_path, model, lstm_classifier, dense_model, wrapper_path,
            w2v_path, time_steps)

        print("Deal ", file, " Complete!")

        for pre in pre_li:
            select_num = int(total_num * 0.01 * pre)
            # 数据集、模型、候选测试集序号、选取比例
            selected_index_path = "./selected_index/" + str(args.dataset) + "_" + str(args.model_type) + "/file" + str(file_index) + "_" + str(pre) + "_"

            # 存储各方法选取用例索引
            state_w_selected = selection(weight_state, trend_set, select_num)
            np.save(selected_index_path + "deepstate_selected", state_w_selected)
            random_selected = ran_selection(total_num, select_num)
            np.save(selected_index_path + "random_selected", random_selected)
            mymethod_selected = total_selection(mymethod, total_num, select_num)
            np.save(selected_index_path + "deepvec_selected", mymethod_selected)
            deepgini_selected = gini_selection(np.array(deepgini), total_num, select_num)
            np.save(selected_index_path + "deepgini_selected", deepgini_selected)
            maxp_selected = maxP_selection(np.array(maxp), total_num, select_num)
            np.save(selected_index_path + "maxp_selected", maxp_selected)
            ats_selected = ats_selection(ats_rank, total_num, select_num)
            np.save(selected_index_path + "ats_selected", ats_selected)

            # 白盒方法选择
            # BSCov (CTM: 根据覆盖率数值排序)
            bscov_selected = ctm_selection(np.array(stellar_bscov), total_num, select_num)
            np.save(selected_index_path + "bscov_selected", bscov_selected)
            # BTCov (CTM)
            btcov_selected = ctm_selection(np.array(stellar_btcov), total_num, select_num)
            np.save(selected_index_path + "btcov_selected", btcov_selected)
            # HSCov (CTM: 根据HSCov数值排序)
            hscov_selected = ctm_selection(np.array(hscov), total_num, select_num)
            np.save(selected_index_path + "hscov_selected", hscov_selected)
            # SC - CTM (根据SC覆盖率数值排序)
            sc_ctm_selected = ctm_selection(np.array(sc_vals), total_num, select_num)
            np.save(selected_index_path + "sc_ctm_selected", sc_ctm_selected)
            # SC - CAM (根据覆盖集合进行贪心选择)
            sc_cam_selected = sc_cam_selection(sc_sets, total_num, select_num)
            np.save(selected_index_path + "sc_cam_selected", sc_cam_selected)

            # 黑盒方法评估
            state_w_R, state_w_P, _, _, _ = selection_evaluate(right, state_w_selected)
            state_w_D = count_unique_elements(fault_types, state_w_selected)
            random_R, random_P, _, _, _ = selection_evaluate(right, random_selected)
            random_D = count_unique_elements(fault_types, random_selected)
            my_cam_R, my_cam_P, _, _, _ = selection_evaluate(right, mymethod_selected)
            my_cam_D = count_unique_elements(fault_types, mymethod_selected)
            gini_R, gini_P, _, _, _ = selection_evaluate(right, deepgini_selected)
            gini_D = count_unique_elements(fault_types, deepgini_selected)
            maxp_R, maxp_P, _, _, _ = selection_evaluate(right, maxp_selected)
            maxp_D = count_unique_elements(fault_types, maxp_selected)
            ats_R, ats_P, _, _, _ = selection_evaluate(right, ats_selected)
            ats_D = count_unique_elements(fault_types, ats_selected)

            # 白盒方法评估
            bscov_R, bscov_P, _, _, _ = selection_evaluate(right, bscov_selected)
            bscov_D = count_unique_elements(fault_types, bscov_selected)
            btcov_R, btcov_P, _, _, _ = selection_evaluate(right, btcov_selected)
            btcov_D = count_unique_elements(fault_types, btcov_selected)
            hscov_R, hscov_P, _, _, _ = selection_evaluate(right, hscov_selected)
            hscov_D = count_unique_elements(fault_types, hscov_selected)
            sc_ctm_R, sc_ctm_P, _, _, _ = selection_evaluate(right, sc_ctm_selected)
            sc_ctm_D = count_unique_elements(fault_types, sc_ctm_selected)
            sc_cam_R, sc_cam_P, _, _, _ = selection_evaluate(right, sc_cam_selected)
            sc_cam_D = count_unique_elements(fault_types, sc_cam_selected)

            # Bug Detection Rate对应Precision查准率
            # Inclusiveness对应Recall查全率
            # Diversity对应揭错类型多样性
            # 写入列表
            state_w_bdr[pre].append(state_w_P);
            state_w_inc[pre].append(state_w_R);
            state_w_div[pre].append(state_w_D)
            ran_bdr[pre].append(random_P);
            ran_inc[pre].append(random_R);
            ran_div[pre].append(random_D)
            deepvec_bdr[pre].append(my_cam_P);
            deepvec_inc[pre].append(my_cam_R);
            deepvec_div[pre].append(my_cam_D)
            deepgini_bdr[pre].append(gini_P);
            deepgini_inc[pre].append(gini_R);
            deepgini_div[pre].append(gini_D)
            maxp_bdr[pre].append(maxp_P);
            maxp_inc[pre].append(maxp_R);
            maxp_div[pre].append(maxp_D)
            ats_bdr[pre].append(ats_P);
            ats_inc[pre].append(ats_R);
            ats_div[pre].append(ats_D)

            # 白盒结果
            bscov_bdr[pre].append(bscov_P);
            bscov_inc[pre].append(bscov_R);
            bscov_div[pre].append(bscov_D)
            btcov_bdr[pre].append(btcov_P);
            btcov_inc[pre].append(btcov_R);
            btcov_div[pre].append(btcov_D)
            hscov_bdr[pre].append(hscov_P);
            hscov_inc[pre].append(hscov_R);
            hscov_div[pre].append(hscov_D)
            sc_ctm_bdr[pre].append(sc_ctm_P);
            sc_ctm_inc[pre].append(sc_ctm_R);
            sc_ctm_div[pre].append(sc_ctm_D)
            sc_cam_bdr[pre].append(sc_cam_P);
            sc_cam_inc[pre].append(sc_cam_R);
            sc_cam_div[pre].append(sc_cam_D)

        # 去除掉真实标签和预测标签相同的类型
        filtered_fault_type = [ft for ft in fault_types if ft[0] != ft[1]]
        fault_type_num += len(set(filtered_fault_type))
        file_index += 1

    # 方法列表
    methods = ['Random', 'BSCov', 'BTCov', 'HSCov', 'SC_CTM', 'SC_CAM', 'DeepGini', 'MaxP', 'ATS', 'DeepState', 'DeepVec']

    precision_dict = {
        'Method': methods,
        '10%': [np.mean(ran_bdr[10]), np.mean(bscov_bdr[10]), np.mean(btcov_bdr[10]),
                np.mean(hscov_bdr[10]), np.mean(sc_ctm_bdr[10]), np.mean(sc_cam_bdr[10]),
                np.mean(deepgini_bdr[10]), np.mean(maxp_bdr[10]), np.mean(ats_bdr[10]),
                np.mean(state_w_bdr[10]), np.mean(deepvec_bdr[10])],
        '20%': [np.mean(ran_bdr[20]), np.mean(bscov_bdr[20]), np.mean(btcov_bdr[20]), 
                np.mean(hscov_bdr[20]), np.mean(sc_ctm_bdr[20]), np.mean(sc_cam_bdr[20]), 
                np.mean(deepgini_bdr[20]), np.mean(maxp_bdr[20]), np.mean(ats_bdr[20]),
                np.mean(state_w_bdr[20]), np.mean(deepvec_bdr[20])]
    }
    print("\n查准率")
    df1 = pd.DataFrame(precision_dict).set_index('Method').transpose()  # 转置
    print(df1)
    os.makedirs("results", exist_ok=True)

    recall_dict = {
        'Method': methods,
        '10%': [np.mean(ran_inc[10]), np.mean(bscov_inc[10]), np.mean(btcov_inc[10]),
                np.mean(hscov_inc[10]), np.mean(sc_ctm_inc[10]), np.mean(sc_cam_inc[10]),
                np.mean(deepgini_inc[10]), np.mean(maxp_inc[10]), np.mean(ats_inc[10]),
                np.mean(state_w_inc[10]), np.mean(deepvec_inc[10])],
        '20%': [np.mean(ran_inc[20]), np.mean(bscov_inc[20]), np.mean(btcov_inc[20]), 
                np.mean(hscov_inc[20]), np.mean(sc_ctm_inc[20]), np.mean(sc_cam_inc[20]), 
                np.mean(deepgini_inc[20]), np.mean(maxp_inc[20]), np.mean(ats_inc[20]),
                np.mean(state_w_inc[20]), np.mean(deepvec_inc[20])]
    }
    print("\n查全率")
    df2 = pd.DataFrame(recall_dict).set_index('Method').transpose()  # 转置
    print(df2)

    print("\n错误类型多样性")
    average_fault_type_num = math.ceil(fault_type_num / file_num)
    print("30个文件平均错误类型数：" + str(average_fault_type_num))

    methods_div = methods + ['All']
    diversity_dict = {
        'Method': methods_div,
        '10%': [np.mean(ran_div[10]), np.mean(bscov_div[10]), np.mean(btcov_div[10]),
                np.mean(hscov_div[10]), np.mean(sc_ctm_div[10]), np.mean(sc_cam_div[10]),
                np.mean(deepgini_div[10]), np.mean(maxp_div[10]), np.mean(ats_div[10]),
                np.mean(state_w_div[10]), np.mean(deepvec_div[10]), average_fault_type_num],
        '20%': [np.mean(ran_div[20]), np.mean(bscov_div[20]), np.mean(btcov_div[20]),
                np.mean(hscov_div[20]), np.mean(sc_ctm_div[20]), np.mean(sc_cam_div[20]),
                np.mean(deepgini_div[20]), np.mean(maxp_div[20]), np.mean(ats_div[20]),
                np.mean(state_w_div[20]), np.mean(deepvec_div[20]), average_fault_type_num]
    }

    df3 = pd.DataFrame(diversity_dict).set_index('Method').transpose()  # 转置
    print(df3)

    # 存储到excel表格
    file_path = "./results/rq1_" + str(args.dataset) + "_" + str(args.model_type) + '_results.xlsx'
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # 将每个 DataFrame 写入不同的工作表并命名
        df1.to_excel(writer, sheet_name='precision', index=True)
        df2.to_excel(writer, sheet_name='recall', index=True)
        df3.to_excel(writer, sheet_name='diversity', index=True)