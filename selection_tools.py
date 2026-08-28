from gensim.models import Word2Vec
import re
import sys

from tensorflow import keras
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

from statics import *

import os
import pickle

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.append('..')

intent_dic = {"PlayMusic": 0, "AddToPlaylist": 1, "RateBook": 2, "SearchScreeningEvent": 3,
              "BookRestaurant": 4, "GetWeather": 5, "SearchCreativeWork": 6}

def process_agnews_data(data_path, w2v_path):
    data = pd.read_csv(data_path)
    w2v_model = Word2Vec.load(w2v_path)
    sentences_ = list(data["news"])
    intent_ = list(data["label"])
    intent = [i - 1 for i in intent_]

    sentences = []
    for s in sentences_:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        # stemming
        sentences.append([i.lower() for i in w])

    # 取得所有单词
    vocab_list = list(w2v_model.wv.vocab.keys())
    # 每个词语对应的索引
    word_index = {word: index for index, word in enumerate(vocab_list)}

    # 序列化
    def get_index(sentence):
        sequence = []
        for word in sentence:
            try:
                sequence.append(word_index[word])
            except KeyError:
                pass
        return sequence

    X_data = list(map(get_index, sentences))

    maxlen = 35  # 截长补短
    X_pad = pad_sequences(X_data, maxlen=maxlen)
    Y = keras.utils.to_categorical(intent, num_classes=4)
    return X_pad, Y

def process_snips_data(data_path, w2v_path):
    data = pd.read_csv(data_path)
    w2v_model = Word2Vec.load(w2v_path)
    sentences_ = list(data["text"])
    intent_ = list(data["intent"])
    intent = [intent_dic[i] for i in intent_]

    sentences = []
    for s in sentences_:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        # stemming
        sentences.append([i.lower() for i in w])

    # 取得所有单词
    vocab_list = list(w2v_model.wv.vocab.keys())
    # 每个词语对应的索引
    word_index = {word: index for index, word in enumerate(vocab_list)}

    # 序列化
    def get_index(sentence):
        sequence = []
        for word in sentence:
            try:
                sequence.append(word_index[word])
            except KeyError:
                pass
        return sequence

    X_data = list(map(get_index, sentences))

    maxlen = 16  # 截长补短
    X_pad = pad_sequences(X_data, maxlen=maxlen)
    Y = keras.utils.to_categorical(intent, num_classes=7)
    return X_pad, Y

def get_val_data(file_path, w2v_path):
    X_val, Y_val = [], []
    if file_path.split(".")[-1] == "npz":
        with np.load(file_path, allow_pickle=True) as f:
            X, Y = f['X'], f['Y']
        for x in X:
            X_val.append(x[0])
        return np.array(X_val), Y

    elif file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        X, Y = process_snips_data(file_path, w2v_path)
        for x in X:
            X_val.append(x)
        return np.array(X_val), Y

    elif file_path.split(".")[-1] == "csv" and "agnews" in file_path.split(".")[-2]:
        X, Y = process_agnews_data(file_path, w2v_path)
        for x in X:
            X_val.append(x)
        return np.array(X_val), Y

from ATS.ATS import ATS

# 这个代码直接传入真实标签，原ATS里是传入预测标签
def ats_selection_rank(file_path, w2v_path, model, dense_model, time_steps):
    ats = ATS()
    # 图像数据集
    if file_path.split(".")[-1] == "npz":
        with np.load(file_path, allow_pickle=True) as f:
            X, Y = f['X'], f['Y']
            # print(Y.shape) # (6000,)
    if file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        X, Y = process_snips_data(file_path, w2v_path)
        # Y = np.argmax(Y, axis=1)
    elif file_path.split(".")[-1] == "csv" and "agnews" in file_path.split(".")[-2]:
        X, Y = process_agnews_data(file_path, w2v_path)
        # print(Y.shape) # (4560, 4)
        # Y = np.argmax(Y, axis=1)
        # print(Y.shape) # (4560,)

    # 预处理数据集
    if "svhn" in file_path:
        X_processed = np.array([svhn_input_preprocess(x)[0] for x in X])
        # Y_processed = keras.utils.to_categorical(Y, num_classes=10)
    elif "mnist" in file_path or "fashion" in file_path:
        X_processed = np.array([mnist_input_preprocess(x)[0] for x in X])
        # Y_processed = keras.utils.to_categorical(Y, num_classes=10)
    else:
        X_processed = np.array(X)
        # Y_processed = Y

    # 获取最终时间步的预测标签
    # 批量获取LSTM输出，只取最后一个时间步
    lstm_last = model.predict(X_processed, verbose=0)[1][:, -1, :]
    # 直接预测最后一个时间步
    dense_output = dense_model.predict(lstm_last, verbose=0)
    Y_psedu = np.argmax(dense_output, axis=1)

    # 预测标签频数统计
    unique, counts = np.unique(Y_psedu, return_counts=True)
    # 将唯一值和对应的频数组合起来
    freq = dict(zip(unique, counts))
    # print(freq)

    ''' ATS实现存在问题： '''
    ''' 1、n按照类别数应该是10 10 7 4 '''
    # 8 8 6 4
    ''' 2、这里直接传入真实标签，但ATS应该是传入预测标签（伪标签） '''
    # SVHN图像数据集
    if file_path.split(".")[-1] == "npz" and "svhn" in file_path.split(".")[-2]:
        div_rank, _, _ = ats.get_priority_sequence(X, Y_psedu, 10, model, dense_model, th=0.001, dataset='svhn')
    # MNIST或Fashion数据集
    elif file_path.split(".")[-1] == "npz" and "svhn" not in file_path.split(".")[-2]:
        div_rank, _, _ = ats.get_priority_sequence(X, Y_psedu, 10, model, dense_model, th=0.001, dataset='mnist')
    # Snips数据集
    elif file_path.split(".")[-1] == "csv" and "snips" in file_path.split(".")[-2]:
        div_rank, _, _ = ats.get_priority_sequence(X, Y_psedu, 7, model, dense_model, th=0.001, dataset='snips')
    # AgNews数据集
    else:
        div_rank, _, _ = ats.get_priority_sequence(X, Y_psedu, 4, model, dense_model, th=0.001, dataset='agnews')

    return div_rank

def ats_selection(rank, length, selected_num):
    selected = np.zeros(length)
    selected[rank[:selected_num]] = 1
    return selected

# SC-CAM 的贪心选择策略 (基于集合覆盖)
def sc_cam_selection(sc_sets, total_num, select_num):
    selected_indices = []
    covered_elements = set()
    mask = np.ones(total_num, dtype=bool)

    for _ in range(select_num):
        best_idx = -1
        max_gain = -1

        # 寻找能覆盖最多新元素的样本
        # 优化：仅在未选择的样本中搜索
        candidates = np.where(mask)[0]
        for idx in candidates:
            gain = len(sc_sets[idx] - covered_elements)
            if gain > max_gain:
                max_gain = gain
                best_idx = idx

        if best_idx != -1 and max_gain > 0:
            selected_indices.append(best_idx)
            covered_elements.update(sc_sets[best_idx])
            mask[best_idx] = False
        else:
            # 如果没有增益，随机填充剩余数量
            remaining = np.where(mask)[0]
            np.random.shuffle(remaining)
            needed = select_num - len(selected_indices)
            selected_indices.extend(remaining[:needed])
            break

    selected = np.zeros(total_num)
    selected[selected_indices] = 1
    return selected

# 需要实现的向量化辅助函数
def calculate_info_entropy_batch(probs):
    """批量计算信息熵"""
    log_probs = np.log2(np.clip(probs, 1e-10, 1.0))
    return -np.sum(probs * log_probs, axis=1)

def calculate_gini_batch(probs):
    """批量计算Gini系数"""
    return 1 - np.sum(probs ** 2, axis=1)

def cal_distance_batch(probs):
    """批量计算预测向量的L2范数（模）"""
    return np.sqrt(np.sum(probs**2, axis=1))

# 计算DU指标（向量化版本）
def cal_weight_dis_vectorized(dis_seq, time_weights):
    """
    向量化计算加权平均距离
    dis_seq: (n_samples, time_steps) 每个样本的时间步距离序列
    time_weights: (time_steps,) 时间步权重向量
    """
    # 计算加权和
    weighted_sum = np.sum(dis_seq * time_weights, axis=1)
    # 计算权重总和
    total_weight = np.sum(time_weights)
    return weighted_sum / total_weight

# 向量化（减少运行时间开销）
def get_selection_information_vectorized(file_index, dataset, model_type, file_path, model, lstm_classifier, dense_model, wrapper_path,
                                         w2v_path, time_steps):
    # 加载数据
    if file_path.endswith(".npz"):
        with np.load(file_path, allow_pickle=True) as f:
            # print("走了图像数据预处理")
            X, Y = f['X'], f['Y']
    elif file_path.endswith(".csv"):
        if "snips" in file_path:
            # print("走了snips数据预处理")
            X, Y = process_snips_data(file_path, w2v_path)
        elif "agnews" in file_path:
            # print("走了agnews数据预处理")
            X, Y = process_agnews_data(file_path, w2v_path)

    # 预处理整个数据集
    if "svhn" in file_path:
        X_processed = np.array([svhn_input_preprocess(x)[0] for x in X])
        Y_processed = keras.utils.to_categorical(Y, num_classes=10)
    elif "mnist" in file_path or "fashion" in file_path:
        X_processed = np.array([mnist_input_preprocess(x)[0] for x in X])
        Y_processed = keras.utils.to_categorical(Y, num_classes=10)
    else:
        X_processed = np.array(X)
        Y_processed = Y

    # 批量获取LSTM输出（将 LSTM 作为特征提取器）
    lstm_outs = model.predict(X_processed, verbose=0)[1]

    if dataset in ('mnist', 'fashion', 'svhn'):
        classes_num = 10
    elif dataset == 'agnews':
        classes_num = 4
    elif dataset == 'snips':
        classes_num = 7
    else:
        print("Wrong dataset!")
        return 0

    # 加载 DeepStellar Wrapper 模型
    # with open(wrapper_path, 'rb') as f:
    #     abst_model = pickle.load(f)

    # 初始化存储数组
    n_samples = len(X)
    test_pros = np.zeros((n_samples, time_steps, classes_num))
    test_labels = np.zeros((n_samples, time_steps), dtype=int)
    test_infos = np.zeros((n_samples, time_steps))
    deepgini_vals = np.zeros((n_samples, time_steps))
    maxp_vals = np.zeros((n_samples, time_steps))

    # 初始化白盒指标容器
    stellar_bscov_vals = []
    stellar_btcov_vals = []
    hscov_vals = []
    sc_vals = []  # 用于 CTM
    sc_sets = []  # 用于 CAM

    # 向量化处理每个时间步
    weights = (np.arange(1, time_steps + 1) ** 2)  # 权重向量
    gini_distances = np.zeros((n_samples, time_steps))

    # 先进行向量化预测，再对 LSTM 输出进行 Coverage 计算
    for t in range(time_steps):
        # 获取当前时间步的LSTM输出
        lstm_t = lstm_outs[:, t, :]
        # 批量预测（将全连接层作为分类器）
        dense_output = dense_model.predict(lstm_t, verbose=0)

        # 存储预测结果
        test_pros[:, t, :] = dense_output
        test_labels[:, t] = np.argmax(dense_output, axis=1)
        test_infos[:, t] = calculate_info_entropy_batch(dense_output)

        # 计算指标
        deepgini_vals[:, t] = calculate_gini_batch(dense_output)
        maxp_vals[:, t] = np.max(dense_output, axis=1)
        gini_distances[:, t] = cal_distance_batch(test_pros[:, t, :])

    # 计算白盒覆盖率指标 (逐样本处理)
    hidden_units = lstm_outs.shape[-1]
    for i in range(n_samples):
        # 1. Stellar Metrics (BSCov, BTCov)
        # 这里的 get_stellar_cov 来自 statics.py
        bscov, btcov = get_stellar_cov(lstm_classifier, model, X[i], wrapper_path)
        stellar_bscov_vals.append(bscov)
        stellar_btcov_vals.append(btcov)

        # 2. RNNTest Metrics (SC)
        lstm_out_sample = lstm_outs[i]  # (time_steps, units)
        # 计算正向和负向激活之和
        plus_sum = np.sum(np.maximum(lstm_out_sample, 0), axis=1)
        minus_sum = np.sum(np.minimum(lstm_out_sample, 0), axis=1)
        # get_testrnn_sc 返回 (覆盖率数值, 激活时间步集合)
        sc_val, sc_set = get_testrnn_sc(plus_sum, minus_sum)
        sc_vals.append(sc_val)
        sc_sets.append(sc_set)

        # 3. HSCov (Hidden State Coverage)
        # 定义：到达最大值的隐藏层状态比例
        # 计算每个时间步中哪个神经元的值最大
        max_indices = np.argmax(lstm_out_sample, axis=1)
        # 统计有多少个唯一的神经元做过"最大值"
        unique_max_neurons = len(np.unique(max_indices))
        hscov = unique_max_neurons / hidden_units
        hscov_vals.append(hscov)

    # 计算加权指标
    weighted_deepgini = np.sum(deepgini_vals * weights, axis=1) / np.sum(weights)
    weighted_maxp = np.sum(maxp_vals * weights, axis=1) / np.sum(weights)

    # 计算DU指标 (向量化)
    mymethod = cal_weight_dis_vectorized(gini_distances, weights)
    # 计算CS指标
    diss = [deal_vecseq(test_pros[i].reshape(time_steps, -1).tolist()) for i in range(n_samples)]

    # 获取分类变化
    confident_mask = (maxp_vals >= 0.5) | (np.arange(time_steps)[None, :] == time_steps - 1)
    classify_outs = np.where(confident_mask, test_labels, -1)

    trend_set = [get_change_set(seq[seq != -1]) for seq in classify_outs]
    weight_state = [cacl_change_rate_with_weights(seq[seq != -1]) for seq in classify_outs]

    # 检查预测结果
    final_labels = test_labels[:, -1]
    true_labels = np.argmax(Y_processed, axis=1)
    right = list((final_labels == true_labels).astype(int))
    fault_types = list(zip(true_labels, final_labels))

    # 保存结果
    np.save(f"./rnn_output/{dataset}_{model_type}/{dataset}_test_pros_{file_index}.npy", test_pros)
    np.save(f"./rnn_output/{dataset}_{model_type}/{dataset}_test_labels_{file_index}.npy", test_labels)
    np.save(f"./rnn_output/{dataset}_{model_type}/{dataset}_test_infos_{file_index}.npy", test_infos)
    np.save(f"./rnn_output/{dataset}_{model_type}/{dataset}_test_right_{file_index}.npy", right)
    np.save(f"./rnn_output/{dataset}_{model_type}/{dataset}_fault_types_{file_index}.npy", fault_types)
    np.save(f"./rnn_output/{dataset}_{model_type}/{dataset}_test_lstm_{file_index}.npy", lstm_outs)

    final = [mymethod, diss]
    return weight_state, trend_set, final, right, fault_types, weighted_deepgini.tolist(), weighted_maxp.tolist(), stellar_bscov_vals, stellar_btcov_vals, hscov_vals, sc_vals, sc_sets
