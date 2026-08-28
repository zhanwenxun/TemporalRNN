import argparse
import numpy as np
import pandas as pd
import datetime
import sys
import os
import tensorflow as tf
import xlsxwriter
import math

# 尝试导入 HDBSCAN，兼容 sklearn 新版或独立库
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        import hdbscan as HDBSCAN
    except ImportError:
        print("请安装 hdbscan 或 更新 scikit-learn >= 1.3 以使用 HDBSCAN 聚类功能。")
        # sys.exit(1) # 如果为了演示可以注释掉，下面有兜底逻辑

# 导入您现有的工具函数
from statics import cacl_change_rate, selection_evaluate, count_unique_elements
from selection_tools import process_snips_data, process_agnews_data, mnist_input_preprocess, svhn_input_preprocess

# GPU 配置
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

def get_rnntcs_rank(model, dense_model, X, time_steps, dataset):
    """
    实现 RNNtcs 的核心选择逻辑：
    1. 获取多时间步预测概率
    2. HDBSCAN 聚类发现离群点
    3. 过滤高置信度样本
    4. 基于 CR 指标排序
    """
    print("正在进行 RNNtcs 特征提取与聚类...")
    
    # 1. 数据预处理 (保持与 selection_tools 一致)
    if dataset == 'svhn':
        X_in = np.array([svhn_input_preprocess(x) for x in X])
    elif dataset in ['mnist', 'fashion']:
        X_in = np.array([mnist_input_preprocess(x) for x in X])
    else:
        X_in = np.array(X)

    # 2. 获取多时间步的预测概率分布 (特征提取)
    # 获取 LSTM 层的隐藏状态输出 (N, Time, Hidden)
    lstm_outs = model.predict(X_in, verbose=0)[1]
    
    n_samples = len(X)
    n_classes = dense_model.output_shape[-1]
    test_pros = np.zeros((n_samples, time_steps, n_classes))
    
    # 通过 Dense 层获取每个时间步的概率 (N, Time, Classes)
    # 为了加速，可以尝试 batch 预测，这里保持逻辑清晰使用循环
    for t in range(time_steps):
        test_pros[:, t, :] = dense_model.predict(lstm_outs[:, t, :], verbose=0)

    # 3. HDBSCAN 聚类
    # 特征向量：将 (Time, Classes) 展平为一维向量
    features = test_pros.reshape(n_samples, -1)
    
    # 超参数 min_cluster_size：论文通常针对离群点检测设为较小值，如 5
    try:
        # 优先使用 sklearn 的实现 (如果不传参，默认 min_cluster_size=5)
        clusterer = HDBSCAN(min_cluster_size=5)
        if hasattr(clusterer, 'fit_predict'):
            cluster_labels = clusterer.fit_predict(features)
        else:
             # 兼容旧版 hdbscan 库
            clusterer = HDBSCAN.HDBSCAN(min_cluster_size=5)
            cluster_labels = clusterer.fit_predict(features)
            
    except NameError:
        print("警告: 未找到 HDBSCAN，将跳过聚类步骤（所有点视为离群点）。")
        cluster_labels = -1 * np.ones(n_samples) # 全设为 -1 (离群点)

    # 4. 计算置信度与 CR 指标
    # 置信度：最后一个时间步的最大概率
    final_probs = test_pros[:, -1, :]
    confidences = np.max(final_probs, axis=1)
    
    # CR 指标：状态变化率 (使用 statics.py 中的函数)
    label_seqs = np.argmax(test_pros, axis=2)
    cr_scores = np.array([cacl_change_rate(seq) for seq in label_seqs])
    
    # 5. 排序策略
    # 阈值：论文中常见的高置信度阈值为 0.9
    CONF_THRESH = 0.9
    
    indices = np.arange(n_samples)
    
    # 逻辑判断掩码
    is_outlier = (cluster_labels == -1)   # 离群点 (Cluster ID为-1)
    is_low_conf = (confidences <= CONF_THRESH) # 低置信度 (非高确信度)
    
    # 优先级分组
    # Group A: 离群点 且 低置信度 (最有可能出错)
    mask_A = is_outlier & is_low_conf
    
    # Group B: 非离群点 且 低置信度 (次优)
    mask_B = (~is_outlier) & is_low_conf
    
    # Group C: 高置信度 (兜底用，论文说剔除，但为了填满预算需要保留在最后)
    mask_C = ~is_low_conf 

    def get_sorted_indices(mask):
        if np.sum(mask) == 0:
            return np.array([], dtype=int)
        idxs = indices[mask]
        scores = cr_scores[mask]
        # 按 CR 指标降序排列
        sorted_args = np.argsort(scores)[::-1]
        return idxs[sorted_args]
        
    rank_A = get_sorted_indices(mask_A)
    rank_B = get_sorted_indices(mask_B)
    rank_C = get_sorted_indices(mask_C) # 高置信度也按 CR 排一下
    
    # 合并所有排名
    final_rank = np.concatenate([rank_A, rank_B, rank_C])
    
    print(f"筛选统计: 离群点&低置信(Group A): {len(rank_A)}, 正常点&低置信(Group B): {len(rank_B)}, 高置信(Group C): {len(rank_C)}")
    
    return final_rank

def rnntcs_selection(rank, total_num, select_num):
    selected = np.zeros(total_num)
    # 选取排名前 k 个
    # 注意：如果 select_num 大于 rank 长度（极少情况），取全部
    limit = min(len(rank), select_num)
    if limit > 0:
        selected[rank[:limit]] = 1
    return selected

if __name__ == '__main__':
    parse = argparse.ArgumentParser("RNNtcs Test Case Selection")
    parse.add_argument('-model_type', required=True, choices=['lstm', 'blstm', 'gru'])
    parse.add_argument('-dataset', required=True, choices=['mnist', 'snips', 'fashion', 'agnews', 'svhn'])
    args = parse.parse_args()

    # --- 模型与路径配置 (参考 rq1&2.py) ---
    dl_model_path = ""
    dataset_name = args.dataset
    model_type = args.model_type
    
    # 路径映射逻辑
    base_path = f"./RNNModels/{dataset_name}_demo"
    if dataset_name == 'mnist':
        time_steps = 28
        w2v_path = ""
        dl_model_path = args.dl_model if 'dl_model' in args else f"{base_path}/model/{dataset_name}_{model_type}.h5" # 假设路径
        if model_type == 'lstm':
            from RNNModels.mnist_demo.mnist_lstm import MnistLSTMClassifier as Classifier
        elif model_type == 'blstm':
            from RNNModels.mnist_demo.mnist_blstm import MnistBLSTMClassifier as Classifier
        elif model_type == 'gru':
            from RNNModels.mnist_demo.mnist_gru import MnistGRUClassifier as Classifier
        to_select_path = "./gen_data/mnist_toselect"
        total_num = 6000
        
    elif dataset_name == 'snips':
        time_steps = 16
        w2v_path = f"{base_path}/save/w2v_model"
        if model_type == 'lstm':
            from RNNModels.snips_demo.snips_lstm import SnipsLSTMClassifier as Classifier
        elif model_type == 'blstm':
            from RNNModels.snips_demo.snips_blstm import SnipsBLSTMClassifier as Classifier
        elif model_type == 'gru':
            from RNNModels.snips_demo.snips_gru import SnipsGRUClassifier as Classifier
        to_select_path = "./gen_data/snips_toselect"
        total_num = 2000
        
    elif dataset_name == 'fashion':
        time_steps = 28
        w2v_path = ""
        if model_type == 'lstm':
            from RNNModels.fashion_demo.fashion_lstm import FashionLSTMClassifier as Classifier
        elif model_type == 'blstm':
            from RNNModels.fashion_demo.fashion_blstm import FashionBLSTMClassifier as Classifier
        elif model_type == 'gru':
            from RNNModels.fashion_demo.fashion_gru import FashionGRUClassifier as Classifier
        to_select_path = "./gen_data/fashion_toselect"
        total_num = 6000

    elif dataset_name == 'agnews':
        time_steps = 35
        w2v_path = f"{base_path}/save/w2v_model"
        if model_type == 'lstm':
            from RNNModels.agnews_demo.agnews_lstm import AGNewsLSTMClassifier as Classifier
        elif model_type == 'blstm':
            from RNNModels.agnews_demo.agnews_blstm import AgnewsBLSTMClassifier as Classifier
        elif model_type == 'gru':
            from RNNModels.agnews_demo.agnews_gru import AgnewsGRUClassifier as Classifier
        to_select_path = "./gen_data/agnews_toselect"
        total_num = 4560
        
    elif dataset_name == 'svhn':
        time_steps = 32
        w2v_path = ""
        if model_type == 'lstm':
            from RNNModels.svhn_demo.svhn_lstm import SvhnLSTMClassifier as Classifier
        elif model_type == 'blstm':
            from RNNModels.svhn_demo.svhn_blstm import SvhnBLSTMClassifier as Classifier
        elif model_type == 'gru':
            from RNNModels.svhn_demo.svhn_gru import SvhnGRUClassifier as Classifier
        to_select_path = "./gen_data/svhn_toselect"
        total_num = 7000

    # 加载模型 (尝试自动推断路径，如果运行报错请手动指定路径)
    # 注意：这里假设您的模型路径结构统一，如果不统一请修改 dl_model_path
    if not os.path.exists(dl_model_path):
        # 尝试常见的命名格式
        dl_model_path = f"./RNNModels/{dataset_name}_demo/output/{model_type}/abst_model/model_{dataset_name}_{model_type}.h5"
        if not os.path.exists(dl_model_path):
             # 再次尝试一种格式，参考 rq1.py
             dl_model_path = f"./RNNModels/{dataset_name}_demo/model/{dataset_name}_{model_type}.h5"
    
    print(f"Loading model from: {dl_model_path}")
    
    try:
        classifier_instance = Classifier()
        if dataset_name == 'snips' or dataset_name == 'agnews':
             # 某些数据集类需要设置路径属性
             classifier_instance.data_path = f"{base_path}/save/standard_data.npz"
             classifier_instance.embedding_path = f"{base_path}/save/embedding_matrix.npy"
        
        model = classifier_instance.load_hidden_state_model(dl_model_path)
        dense_classifier = Classifier()
        dense_model = dense_classifier.reload_dense(dl_model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查路径或在代码中硬编码正确的 dl_model_path")
        sys.exit(1)

    # --- 结果存储初始化 ---
    rnntcs_bdr, rnntcs_inc, rnntcs_div = {}, {}, {}
    pre_li = [10, 20]
    for i in pre_li:
        rnntcs_bdr[i], rnntcs_inc[i], rnntcs_div[i] = [], [], []

    # 获取文件列表
    files = [f for f in os.listdir(to_select_path) if 'aug' not in f and 'ori' not in f]
    import re
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0)
    
    fault_type_num = 0
    file_num = len(files)

    for file in files:
        print(f"\nProcessing file: {file} at {datetime.datetime.now()}")
        file_path = to_select_path + "/" + file
        
        # 加载数据 (逻辑复用 selection_tools)
        if file_path.endswith(".npz"):
            with np.load(file_path, allow_pickle=True) as f:
                X, Y = f['X'], f['Y']
        elif file_path.endswith(".csv"):
            if "snips" in file_path:
                X, Y = process_snips_data(file_path, w2v_path)
            elif "agnews" in file_path:
                X, Y = process_agnews_data(file_path, w2v_path)
        
        # 获取真实标签 (Ground Truth)
        if "svhn" in file_path or "mnist" in file_path or "fashion" in file_path:
             # Y 可能是 scalar，转 one-hot 或者直接用
             pass 
        
        # 预处理数据用于预测 (Accuracy Check)
        if dataset_name == 'svhn':
             X_pred = np.array([svhn_input_preprocess(x) for x in X])
             Y_labels = Y # SVHN Y通常是scalar
        elif dataset_name in ['mnist', 'fashion']:
             X_pred = np.array([mnist_input_preprocess(x) for x in X])
             Y_labels = Y 
        else:
             X_pred = np.array(X)
             Y_labels = np.argmax(Y, axis=1) if len(Y.shape) > 1 else Y

        # 计算 Ground Truth (Right/Wrong)
        # 获取最终预测
        lstm_out_all = model.predict(X_pred, verbose=0)[1]
        final_lstm = lstm_out_all[:, -1, :]
        final_preds = dense_model.predict(final_lstm, verbose=0)
        final_labels = np.argmax(final_preds, axis=1)
        
        # 兼容 Y 的格式
        if len(Y.shape) > 1: # one-hot
            true_labels = np.argmax(Y, axis=1)
        else:
            true_labels = Y
            
        right = list((final_labels == true_labels).astype(int))
        fault_types = list(zip(true_labels, final_labels))
        
        # --- 执行 RNNtcs 算法 ---
        # 传入原始 X (get_rnntcs_rank 内部会处理预处理)
        rank = get_rnntcs_rank(model, dense_model, X, time_steps, dataset_name)
        
        for pre in pre_li:
            select_num = int(total_num * 0.01 * pre)
            
            # 选择
            selected_mask = rnntcs_selection(rank, total_num, select_num)
            
            # 评估
            # selection_evaluate 返回: (Recall_rate, Precision_rate, ...)
            # 注意: rq1.py 中接收顺序可能是: R, P, ... 请核对 statics.py
            # 根据您提供的代码片段: random_R, random_P, ... = selection_evaluate(...)
            # 所以第一个是 Recall (查全), 第二个是 Precision (查准/BDR)
            recall, precision, _, _, _ = selection_evaluate(right, selected_mask)
            diversity = count_unique_elements(fault_types, selected_mask)
            
            rnntcs_bdr[pre].append(precision)
            rnntcs_inc[pre].append(recall)
            rnntcs_div[pre].append(diversity)
            
        # 更新总错误类型数
        filtered_fault_type = [ft for ft in fault_types if ft[0] != ft[1]]
        fault_type_num += len(set(filtered_fault_type))
        
    # --- 结果输出 ---
    methods = ['RNNtcs']
    
    # 查准率 (Precision)
    precision_dict = {
        'Method': methods,
        '10%': [np.mean(rnntcs_bdr[10])],
        '20%': [np.mean(rnntcs_bdr[20])]
    }
    
    # 查全率 (Recall)
    recall_dict = {
        'Method': methods,
        '10%': [np.mean(rnntcs_inc[10])],
        '20%': [np.mean(rnntcs_inc[20])]
    }
    
    # 多样性 (Diversity)
    average_fault_type_num = math.ceil(fault_type_num / file_num)
    diversity_dict = {
        'Method': methods + ['All'],
        '10%': [np.mean(rnntcs_div[10]), average_fault_type_num],
        '20%': [np.mean(rnntcs_div[20]), average_fault_type_num]
    }
    
    print("\n=== RNNtcs 实验结果 ===")
    df1 = pd.DataFrame(precision_dict)
    df2 = pd.DataFrame(recall_dict)
    df3 = pd.DataFrame(diversity_dict)
    print("查准率 (Precision):\n", df1)
    print("查全率 (Recall):\n", df2)
    print("多样性 (Diversity):\n", df3)
    
    # 保存至 Excel
    os.makedirs("./results", exist_ok=True)
    file_path = f"./results/rnntcs_{args.dataset}_{args.model_type}_results.xlsx"
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='precision', index=False)
        df2.to_excel(writer, sheet_name='recall', index=False)
        df3.to_excel(writer, sheet_name='diversity', index=False)
        
    print(f"\n结果已保存至: {file_path}")