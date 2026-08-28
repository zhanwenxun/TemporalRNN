import time
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from data_util import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    parse = argparse.ArgumentParser("Calculate the bug detection rate for the selected dataset.")
    # 输入参数：模型、数据集、prostart、labelstart
    parse.add_argument('-model', required=True, choices=['lstm', 'blstm', 'gru'])
    parse.add_argument('-dataset', required=True, choices=['mnist', 'snips', 'fashion', 'agnews', 'svhn'])
    parse.add_argument('-prostart', required=True, choices=['0', '3', '5', '10', '15', '20'])
    parse.add_argument('-labelstart', required=True, choices=['0', '3', '5', '10', '15', '20'])
    args = parse.parse_args()

    print("Dataset: {}".format(args.dataset))
    print("Model: {}".format(args.model))

    '''获取原始训练集的预测结果'''
    # 使用10%模型未见过的训练数据（验证集）
    # train_data_path = "./rnn_output/" + args.dataset + "_" + args.model + "/" + args.dataset
    # 使用全部的训练数据
    train_data_path = "./rnn_output/" + args.dataset + "_" + args.model + "/all_train/" + args.dataset

    # 获取RNN对训练集的预测信息（预测概率向量、预测标签、信息熵、是否预测正确、隐藏状态）
    train_pros = np.load(train_data_path + "_train_pros.npy")
    train_labels = np.load(train_data_path + "_train_labels.npy")
    train_infos = np.load(train_data_path + "_train_infos.npy")
    train_right = np.load(train_data_path + "_train_right.npy")
    train_lstm = np.load(train_data_path + "_train_lstm.npy")

    start_time1 = time.time()
    # 提取训练集的特征
    print("Extract Train Dataset Feature")
    train_features = extract_features(train_pros, train_labels, train_infos, train_lstm, int(args.prostart), int(args.labelstart))

    # 获取该数据集的类别数
    cat_num = train_pros[0][0].shape[0]
    # 提取训练集是否被预测错误的信息
    train_errors = np.array(train_right == 0).astype(int)

    # 计算正负样本比例（用于处理不平衡的训练集）
    pos_count = np.sum(train_errors)
    neg_count = len(train_errors) - pos_count
    # 当二分类样本不平衡的时候，scale_pos_weight参数需要传入（负样本数量 / 正样本数量）
    scale_pos_weight = neg_count / pos_count

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        train_features, train_errors,
        test_size=0.2,
        stratify=train_errors,
        random_state=42
    )

    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)

    # 构建CatBoost模型
    print("Initialize CatBoost Model")
    catboost_params = {
        'iterations': 1000,  # 最大迭代次数
        'learning_rate': 0.05,  # 学习率
        'depth': 6,  # 树深度
        'l2_leaf_reg': 3,  # L2正则化系数
        'border_count': 128,  # 数值特征的分箱数
        'loss_function': 'Logloss',  # 二分类损失函数
        'eval_metric': 'AUC',  # 评估指标
        'random_seed': 42,  # 随机种子
        'use_best_model': True,  # 使用最佳模型
        'od_type': 'Iter',  # 过拟合检测类型
        'od_wait': 50,  # 早停等待轮数
        'verbose': 100,  # 每100轮输出一次日志
        'scale_pos_weight': scale_pos_weight,  # 处理数据集类别不平衡的情况
        'task_type': 'GPU' if device == 'cuda' else 'CPU',  # 使用GPU加速
        'devices': '0' if device == 'cuda' else None,  # 指定GPU设备
    }

    # 训练CatBoost模型
    print("Training CatBoost Model")
    model = CatBoostClassifier(**catboost_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        plot=False
    )

    end_time1 = time.time()
    start_time2 = time.time()

    # 候选测试集预测结果的路径
    test_data_path = "./rnn_output/" + args.dataset + "_" + args.model + "/" + args.dataset
    # 存储评估结果信息
    bdr_10, bdr_20 = [], []
    inc_10, inc_20 = [], []
    div_10, div_20 = [], []

    # 30个候选测试集
    for idx in range(0, 30):
        print(idx)
        print("Obatin Test Prediciton Info")
        # 获取原始模型对测试集的预测结果信息(预测概率向量、预测标签、信息熵、隐藏状态）
        test_pros = np.load(test_data_path + "_test_pros_" + str(idx) + ".npy")
        test_labels = np.load(test_data_path + "_test_labels_" + str(idx) + ".npy")
        test_infos = np.load(test_data_path + "_test_infos_" + str(idx) + ".npy")
        test_lstm = np.load(test_data_path + "_test_lstm_" + str(idx) + ".npy")
        # 获取测试集是否预测正确，及其对应的错误类型
        test_right = np.load(test_data_path + "_test_right_" + str(idx) + ".npy")
        fault_types = np.load(test_data_path + "_fault_types_" + str(idx) + ".npy")

        print("Extract Test Dataset Feature")
        # 提取特征（形状为[用例数量, 7个特征]）
        # 可能这里存在一些时间开销
        test_features = extract_features(test_pros, test_labels, test_infos, test_lstm, int(args.prostart), int(args.labelstart))
        # 获取测试集是否预测错误的信息
        test_errors = (test_right == 0).astype(int)
        # 获取候选测试集中预测错误的用例总数
        errors_num = np.sum(test_errors)
        # 不使用类别特征cat_features
        test_pool = Pool(test_features, label=test_errors)

        print("CatBoost Model Predict")
        # CatBoost模型评估候选测试集中各用例为正样本（被预测错误）的概率
        scores = model.predict_proba(test_pool)[:, 1]
        # 计算CatBoost模型在候选测试集上的AUC值
        test_auc = roc_auc_score(test_errors, scores)
        print(f"Test AUC: {test_auc:.4f}")

        # 计算各选取比例下的测试预算 test_budget
        test_num = test_labels.shape[0]
        num_10p = int(test_num * 0.1)
        num_20p = int(test_num * 0.2)


        ''' ------------ 只考虑不确定性，直接按照scores降序排列 ------------ '''
        index = np.argsort(scores)[::-1]
        # 获取按照socres降序排列后测试用例的bug和错误类型信息
        is_bug = test_errors[index]
        fault_diversity = fault_types[index]
        
        bdr_10.append(np.sum(is_bug[:num_10p]) / num_10p)
        bdr_20.append(np.sum(is_bug[:num_20p]) / num_20p)
        
        inc_10.append(np.sum(is_bug[:num_10p]) / errors_num)
        inc_20.append(np.sum(is_bug[:num_20p]) / errors_num)
        
        div_10.append(count_unique_elements(fault_diversity[:num_10p]))
        div_20.append(count_unique_elements(fault_diversity[:num_20p]))


        ''' ------------ 平衡不确定性和多样性，进一步提升测试子集的多样性 ------------ '''
        ''' 按（maxp，第二maxp）分组，依据各组所占比例，从各组按照scores降序排列后抽取 '''
#         index_10p = select_diverse_subset(scores, test_pros[:, -1, :], num_10p)
#         selected_index_10p = np.zeros(test_num, dtype=int)
        # selected_index_10p[index_10p] = 1
#         index_20p = select_diverse_subset(scores, test_pros[:, -1, :], num_20p)
#         selected_index_20p = np.zeros(test_num, dtype=int)
        # selected_index_20p[index_20p] = 1

#         # 存储所选用例索引
#         # 数据集、模型、候选测试集序号、选取比例
        # selected_index_path = "./selected_index/" + str(args.dataset) + "_" + str(args.model) + "/file" + str(idx) + "_"
        # np.save(selected_index_path + str(10) + "_temprnn_scores_selected", selected_index_10p)
        # np.save(selected_index_path + str(20) + "_temprnn_scores_selected", selected_index_20p)

#         ''' 提升多样性 提取出不同比例选取的测试子集 '''
#         is_bug_10p = test_errors[index_10p]
#         is_bug_20p = test_errors[index_20p]

#         fault_diversity_10p = fault_types[index_10p]
#         fault_diversity_20p = fault_types[index_20p]

#         bdr_10.append(np.sum(is_bug_10p[:num_10p]) / num_10p)
#         bdr_20.append(np.sum(is_bug_20p[:num_20p]) / num_20p)

#         inc_10.append(np.sum(is_bug_10p[:num_10p]) / errors_num)
#         inc_20.append(np.sum(is_bug_20p[:num_20p]) / errors_num)

#         div_10.append(count_unique_elements(fault_diversity_10p))
#         div_20.append(count_unique_elements(fault_diversity_20p))

    data_dict = {
        'Metrics': ['precision', 'recall', 'diversity'],
        '10%': [np.mean(bdr_10), np.mean(inc_10), np.mean(div_10)],
        '20%': [np.mean(bdr_20), np.mean(inc_20), np.mean(div_20)],
    }

    df = pd.DataFrame(data_dict).set_index('Metrics').transpose()  # 转置
    print(df)

    end_time2 = time.time()
    print("\nCatBoost模型训练耗时： ", str(end_time1 - start_time1), "秒")
    print("对30个候选测试集进行选择，平均耗时: ", str((end_time2 - start_time2)/30), "秒")

    df.to_csv("./results/rq1_temprnn_scores_{}_{}.csv".format(args.dataset, args.model))

if __name__ == '__main__':
    main()
