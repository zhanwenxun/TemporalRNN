import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data_util import extract_features
# from data_util import count_unique_elements, extract_features, select_test_subset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 多样性提升方法统一在这里切换。
# 可选值见 data_util.py 中的 DIVERSITY_METHODS 注释。
DIVERSITY_METHOD = "top2_group"

# 当前实验使用全部训练数据；如需切换训练集来源，只改这里即可。
TRAIN_SPLIT_DIR = "all_train"
TEST_FILE_COUNT = 30
BUDGET_RATIOS = (0.1, 0.2)


def parse_args():
    parser = argparse.ArgumentParser("Calculate the bug detection rate for the selected dataset.")
    # 输入参数：模型、数据集、概率特征起始时间步、标签特征起始时间步
    parser.add_argument("-model", required=True, choices=["lstm", "blstm", "gru"])
    parser.add_argument("-dataset", required=True, choices=["mnist", "snips", "fashion", "agnews", "svhn"])
    parser.add_argument("-prostart", required=True, choices=["0", "3", "5", "10", "15", "20"])
    parser.add_argument("-labelstart", required=True, choices=["0", "3", "5", "10", "15", "20"])
    parser.add_argument(
        "-method",
        default=DIVERSITY_METHOD,
        help="测试子集选择方法，默认使用文件顶部 DIVERSITY_METHOD。",
    )
    return parser.parse_args()


def load_train_prediction_info(dataset, model):
    """读取原始 RNN 在训练集上的多时间步预测信息。"""
    base_path = Path("./rnn_output") / f"{dataset}_{model}" / TRAIN_SPLIT_DIR / dataset
    return {
        "pros": np.load(str(base_path) + "_train_pros.npy"),
        "labels": np.load(str(base_path) + "_train_labels.npy"),
        "infos": np.load(str(base_path) + "_train_infos.npy"),
        "right": np.load(str(base_path) + "_train_right.npy"),
        "lstm": np.load(str(base_path) + "_train_lstm.npy"),
    }


def load_test_prediction_info(dataset, model, file_index):
    """读取原始 RNN 在一个候选测试集上的多时间步预测信息。"""
    base_path = Path("./rnn_output") / f"{dataset}_{model}" / dataset
    return {
        "pros": np.load(str(base_path) + f"_test_pros_{file_index}.npy"),
        "labels": np.load(str(base_path) + f"_test_labels_{file_index}.npy"),
        "infos": np.load(str(base_path) + f"_test_infos_{file_index}.npy"),
        "lstm": np.load(str(base_path) + f"_test_lstm_{file_index}.npy"),
        "right": np.load(str(base_path) + f"_test_right_{file_index}.npy"),
        "fault_types": np.load(str(base_path) + f"_fault_types_{file_index}.npy"),
    }


def build_catboost_params(scale_pos_weight):
    """CatBoost 用于学习“原模型是否预测错误”的二分类器。"""
    params = {
        "iterations": 1000,  # 最大迭代次数
        "learning_rate": 0.05,  # 学习率
        "depth": 6,  # 树深度
        "l2_leaf_reg": 3,  # L2 正则化系数
        "border_count": 128,  # 数值特征分箱数
        "loss_function": "Logloss",  # 二分类损失函数
        "eval_metric": "AUC",  # 评估指标
        "random_seed": 42,  # 随机种子
        "use_best_model": True,  # 使用验证集上的最佳模型
        "od_type": "Iter",  # 早停方式
        "od_wait": 50,  # 早停等待轮数
        "verbose": 100,
        "scale_pos_weight": scale_pos_weight,  # 处理预测错误样本较少的问题
        "task_type": "GPU" if DEVICE == "cuda" else "CPU",
    }
    if DEVICE == "cuda":
        params["devices"] = "0"
    return params


def train_error_detector(train_data, pro_start, label_start):
    """训练二级模型：输入时序特征，输出原始 RNN 预测错误的概率。"""
    print("Extract Train Dataset Feature")
    train_features = extract_features(
        train_data["pros"],
        train_data["labels"],
        train_data["infos"],
        train_data["lstm"],
        pro_start,
        label_start,
    )
    train_errors = (train_data["right"] == 0).astype(int)

    pos_count = np.sum(train_errors)
    neg_count = len(train_errors) - pos_count
    if pos_count == 0:
        raise ValueError("训练集中没有预测错误样本，无法训练错误检测器。")

    X_train, X_val, y_train, y_val = train_test_split(
        train_features,
        train_errors,
        test_size=0.2,
        stratify=train_errors,
        random_state=42,
    )

    print("Initialize CatBoost Model")
    model = CatBoostClassifier(**build_catboost_params(neg_count / pos_count))

    print("Training CatBoost Model")
    model.fit(Pool(X_train, label=y_train), eval_set=Pool(X_val, label=y_val), plot=False)
    return model, train_features


# def evaluate_selected_subset(selected_indices, test_errors, fault_types, errors_num):
#     """统计一个测试预算下的 precision、recall、diversity。"""
#     selected_errors = test_errors[selected_indices]
#     selected_fault_types = fault_types[selected_indices]
#
#     precision = np.sum(selected_errors) / len(selected_indices) if len(selected_indices) else 0.0
#     recall = np.sum(selected_errors) / errors_num if errors_num else 0.0
#     diversity = count_unique_elements(selected_fault_types)
#     return precision, recall, diversity
#
#
# def save_selected_mask(dataset, model, file_index, ratio, selected_indices, test_num, method):
#     """保存选中测试用例的 0/1 mask，便于后续复现实验。"""
#     output_dir = Path("./selected_index") / f"{dataset}_{model}"
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     selected_mask = np.zeros(test_num, dtype=int)
#     selected_mask[selected_indices] = 1
#     output_path = output_dir / f"file{file_index}_{int(ratio * 100)}_{method}_selected.npy"
#     np.save(output_path, selected_mask)


# def run_one_test_file(args, model, cat_num, file_index):
#     """处理一个候选测试集：提特征、打分、按预算选样本、计算指标。"""
#     print(file_index)
#     print("Obtain Test Prediction Info")
#     test_data = load_test_prediction_info(args.dataset, args.model, file_index)
#
#     print("Extract Test Dataset Feature")
#     test_features = extract_features(
#         test_data["pros"],
#         test_data["labels"],
#         test_data["infos"],
#         test_data["lstm"],
#         int(args.prostart),
#         int(args.labelstart),
#     )
#     test_errors = (test_data["right"] == 0).astype(int)
#     errors_num = np.sum(test_errors)
#
#     print("CatBoost Model Predict")
#     scores = model.predict_proba(Pool(test_features, label=test_errors))[:, 1]
#     if len(np.unique(test_errors)) == 2:
#         print(f"Test AUC: {roc_auc_score(test_errors, scores):.4f}")
#     else:
#         print("Test AUC: skipped because this candidate set has only one class.")
#
#     metrics_by_ratio = {}
#     test_num = test_data["labels"].shape[0]
#     for ratio in BUDGET_RATIOS:
#         budget = int(test_num * ratio)
#         selected_indices = select_test_subset(
#             method=args.method,
#             scores=scores,
#             budget=budget,
#             cat_num=cat_num,
#             test_features=test_features,
#             test_pros=test_data["pros"],
#             test_labels=test_data["labels"],
#             test_lstm=test_data["lstm"],
#         )
#         save_selected_mask(args.dataset, args.model, file_index, ratio, selected_indices, test_num, args.method)
#         metrics_by_ratio[ratio] = evaluate_selected_subset(
#             selected_indices,
#             test_errors,
#             test_data["fault_types"],
#             errors_num,
#         )
#
#     return metrics_by_ratio

# def summarize_metrics(all_metrics):
#     """汇总 30 个候选测试集上的平均 precision、recall、diversity。"""
#     rows = {}
#     for ratio in BUDGET_RATIOS:
#         values = np.array([metrics[ratio] for metrics in all_metrics], dtype=float)
#         rows[f"{int(ratio * 100)}%"] = np.mean(values, axis=0)
#
#     df = pd.DataFrame.from_dict(
#         rows,
#         orient="index",
#         columns=["precision", "recall", "diversity"],
#     )
#     print(df)
#     return df


# def main():
#     args = parse_args()
#     print(f"Dataset: {args.dataset}")
#     print(f"Model: {args.model}")
#     print(f"Diversity method: {args.method}")
#
#     train_data = load_train_prediction_info(args.dataset, args.model)
#     cat_num = train_data["pros"][0][0].shape[0]
#
#     train_start = time.time()
#     model, _ = train_error_detector(train_data, int(args.prostart), int(args.labelstart))
#     train_end = time.time()
#
#     select_start = time.time()
#     all_metrics = [
#         run_one_test_file(args, model, cat_num, file_index)
#         for file_index in range(TEST_FILE_COUNT)
#     ]
#     result_df = summarize_metrics(all_metrics)
#     select_end = time.time()
#
#     print("\nCatBoost模型训练耗时：", train_end - train_start, "秒")
#     print("对30个候选测试集进行选择，平均耗时：", (select_end - select_start) / TEST_FILE_COUNT, "秒")
#
#     output_dir = Path("./catboost_results")
#     output_dir.mkdir(parents=True, exist_ok=True)
#     result_path = output_dir / f"catboost_{args.method}_{args.dataset}_{args.model}.csv"
#     result_df.to_csv(result_path)


# if __name__ == "__main__":
#     main()
