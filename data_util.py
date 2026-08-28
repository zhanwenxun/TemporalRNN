import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.cluster import KMeans, MiniBatchKMeans  # 导入
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity

# 计算揭露故障类型数目时用到
def count_unique_elements(fault_types):
    unique_elements = set()

    for fault in fault_types:
        if fault[0] != fault[1]:
            unique_elements.add(tuple(fault))

    return len(unique_elements)

''' 前n-1个时间步和最终时间步在预测概率向量上的cosine距离 t+1平方加权平均 '''
def average_pro_diff(pros, pro_start):
    p = pros[:, pro_start:, :]
    target = p[:,-1,:]
    time_steps = p.shape[1]
    time_weights = np.arange(1, time_steps) ** 2
    average_distances=np.zeros(p.shape[0], dtype=float)
    for id, row in enumerate(p):
        cosine_distances = 1 - cosine_similarity(row[:-1, :], target[id].reshape(1, -1)).flatten()
        weighted_sum = np.sum(cosine_distances * time_weights)
        total_weight = np.sum(time_weights)
        average_distances[id] = weighted_sum / total_weight

    return average_distances

''' 各时间步预测概率向量的1-maxp t+1平方加权平均 '''
def average_confidence(pros, pro_start):
    p = pros[:, pro_start:, :]
    # 计算每个时间步的最大概率
    max_probs = np.max(p, axis=-1)  # 形状: (n_cases, n_timesteps)
    # 计算1-最大概率
    one_minus_max = 1 - max_probs

    # 应用时间权重
    time_weights = np.arange(1, p.shape[1] + 1) ** 2
    weighted_sum = np.sum(one_minus_max * time_weights, axis=1)
    total_weight = np.sum(time_weights)

    return weighted_sum / total_weight

''' 相邻时间步预测概率向量的cosine距离求和 '''
def neighbor_pro_distance(pros, pro_start):
    p = pros[:, pro_start:, :]
    sum_neighbor_pro_distance = np.zeros(p.shape[0], dtype=float)
    for id, row in enumerate(p):
        sum_neighbor_pro_distance[id] = np.sum(1 - np.diag(cosine_similarity(row[:-1, :], row[1:, :])))

    return sum_neighbor_pro_distance

''' 各时间步预测概率向量的信息熵 t+1平方加权平均 '''
def avg_info(infos):
    # 时间权重
    time_steps = infos.shape[1]
    time_weights = np.arange(1, time_steps + 1) ** 2
    # 计算加权平均值
    weighted_sum = np.sum(infos * time_weights, axis=1)
    total_weight = np.sum(time_weights)

    return weighted_sum / total_weight

''' 第一轮时间步信息熵-最终轮时间步信息熵 '''
def info_diff(infos):
    start_infos = infos[:, 0]
    final_infos = infos[:, -1]

    return start_infos - final_infos

''' 各时间步隐藏状态和最终轮隐藏状态的平均cosine距离 '''
def avg_lstm_diff(lstm_out, pro_start):
    lstm_out1 = lstm_out[:, pro_start:]
    target = lstm_out1[:, -1, :]
    time_steps = lstm_out1.shape[1]
    time_weights = np.arange(1, time_steps) ** 2
    average_distances = np.zeros(lstm_out1.shape[0], dtype=float)
    for id, row in enumerate(lstm_out1):
        cosine_distances = 1 - cosine_similarity(row[:-1, :], target[id].reshape(1, -1)).flatten()
        weighted_sum = np.sum(cosine_distances * time_weights)
        total_weight = np.sum(time_weights)
        average_distances[id] = weighted_sum / total_weight

    return average_distances

''' 前n-1时间步的预测标签和最终时间步预测标签逐一对比不同的情况所占比例 '''
def label_diff_ratio(labels, label_start):
    labels1 = labels[:, label_start:]
    target_labels = labels1[:, -1] # 最终轮预测标签

    # 逐一对比前n-1时间步与最终时间步的预测标签是否不同
    comparisons = labels1[:, :-1] != target_labels[:, np.newaxis]  # 形状: (n_cases, n_timesteps-1)

    # 统计标签不同的总次数
    sum_label_diff = np.sum(comparisons, axis=1)
    sum_diff = labels1.shape[1] - 1

    return sum_label_diff / sum_diff

''' 提取RNN对用例多时间步预测结果的7种时序特征 '''
# pros是预测概率向量，labels是预测标签，infos是信息熵（不确定性分数），lstm_out是隐藏状态
def extract_features(pros, labels, infos, lstm_out, pro_start, label_start):

    # 创建特征字典
    features_dict = {
        # 最终时间步和倒数第二时间步的预测标签作为分类特征
        # 'final_label': labels[:, -1].astype(int),
        # 'pen_label': labels[:, -2].astype(int),
        # 前n-1个时间步和最终时间步在预测概率向量上的cosine距离 t+1平方加权平均
        'avg_pro_diff': average_pro_diff(pros, pro_start),
        # 各时间步预测概率向量的1-maxp t+1平方加权平均
        'average_confidence': average_confidence(pros, pro_start),
        # 相邻时间步预测概率向量的cosine距离求和
        'neighbor_pro_distance': neighbor_pro_distance(pros, pro_start),
        # 各时间步预测概率向量的信息熵 t+1平方加权平均
        'avg_info': avg_info(infos),
        # 第一轮时间步预测概率向量信息熵-最终时间步预测概率向量信息熵 落差
        'info_diff': info_diff(infos),
        # 前n-1时间步和最终时间步在隐藏状态上的cosine距离 t+1平方加权平均
        'avg_lstm_diff': avg_lstm_diff(lstm_out, pro_start),
        # 统计所有轮（除最后一轮）的预测标签和最后一轮预测标签是不同标签的次数
        'sum_label_diff': label_diff_ratio(labels, label_start),
        # 计算每个用例预测标签随时间步发生变化所占的比例 CR指标
        # 'change_rate': calc_change_rate_vectorized(labels, label_start)
    }

    # 创建 DataFrame
    feature_df = pd.DataFrame(features_dict)

    # 对数值列进行归一化 [0,1] (排除标签列)
    # numerical_cols = feature_df.columns.drop(['final_label','pen_label'])
    numerical_cols = feature_df.columns
    scaler = MinMaxScaler(feature_range=(0, 1))
    # MinMax标准化
    feature_df[numerical_cols] = scaler.fit_transform(feature_df[numerical_cols])

    # 确保标签列是整数类型
    # feature_df['final_label'] = feature_df['final_label'].astype(int)
    # feature_df['pen_label'] = feature_df['pen_label'].astype(int)
    return feature_df



''' 多样性创新（衡量用例之间的相似度指标 或 衡量测试子集的多样性） '''
''' ------多样性创新方法4：按 (maxp, 2nd maxp) 分组并按比例选取------ '''
def adjustQuotas(quotas, groups, n):
    total_selected = sum(quotas.values())
    diff = n - total_selected

    if diff == 0: return quotas

    # 按组大小降序排列
    sorted_keys = sorted(groups, key=lambda k: len(groups[k]), reverse=True)

    # (要求 2) 简化调整逻辑：只给前 |diff| 个组 +/- 1
    for i in range(abs(diff)):
        if i >= len(sorted_keys):
            break  # 避免 diff > 组数
        key = sorted_keys[i]
        if diff > 0:
            quotas[key] += 1
        elif quotas[key] > 1:  # 保证配额至少为 1
            quotas[key] -= 1

    return quotas


def select_diverse_subset(scores, final_pro, n):
    total_size = len(scores)
    pool_size = min(3 * n, total_size)  # 筛选池大小

    if n == 0: return []
    if n >= total_size: return list(np.arange(total_size))

    # --- 1. 高效预筛选 (按 scores) ---
    # 使用 argpartition 在 O(C) 内找到 top N，再 O(N log N) 排序
    partition_indices = np.argpartition(scores, -pool_size)[-pool_size:]
    partition_scores = scores[partition_indices]
    sorted_order = np.argsort(partition_scores)[::-1]
    # pool_indices 是已经按分数降序排列的 3*n 个索引
    pool_indices = partition_indices[sorted_order]

    # --- 2. 向量化计算 Top 2 键 ---
    # O(N*K) 获取 3*n 个用例的概率
    pool_probs = final_pro[pool_indices]  # (N, K) 矩阵
    # O(N*K) 获取 top 2 的索引 (无序)
    top_2_unsorted = np.argpartition(pool_probs, -2, axis=1)[:, -2:]  # (N, 2)
    # O(N) 获取 top 2 的概率值
    top_2_scores = np.take_along_axis(pool_probs, top_2_unsorted, axis=1)  # (N, 2)
    # O(N) 获取内部排序 (例如 [[1, 0], [0, 1], ...])
    top_2_sort_order = np.argsort(top_2_scores, axis=1)[:, ::-1]  # (N, 2)
    # O(N) 得到最终排好序的 (maxp, 2nd maxp) 索引矩阵
    top_2_keys_matrix = np.take_along_axis(top_2_unsorted, top_2_sort_order, axis=1)  # (N, 2)

    # --- 3. 分组 ---
    groups = defaultdict(list)
    # O(N) 循环构建字典
    for idx, key_pair in zip(pool_indices, top_2_keys_matrix):
        key = (key_pair[0], key_pair[1])
        # 因为 pool_indices 已按分数排序，组内自动有序
        groups[key].append(idx)

    # --- 4. 按比例分配 ---
    total_pool_size = len(pool_indices)
    quotas = {}  # 对应 selection_counts

    for group_key, indices in groups.items():
        prop_count = int(n * len(indices) / total_pool_size)
        quotas[group_key] = max(1, prop_count)  # 保证每组至少 1 个

    quotas = adjustQuotas(quotas, groups, n)

    # --- 5. 选取 ---
    selected_indices = []
    for group_key, count in quotas.items():
        # (优化 3) 无需再次排序，组内已按分数排好
        selected_indices.extend(groups[group_key][:count])

    return selected_indices


def select_diverse_subset2(scores, test_labels, final_pro, n):
    total_size = len(scores)
    pool_size = min(3 * n, total_size)  # 筛选池大小

    if n == 0: return []
    if n >= total_size: return list(np.arange(total_size))

    # --- 1. 高效预筛选 (按 scores) ---
    # 使用 argpartition 在 O(C) 内找到 top N，再 O(N log N) 排序
    partition_indices = np.argpartition(scores, -pool_size)[-pool_size:]
    partition_scores = scores[partition_indices]
    sorted_order = np.argsort(partition_scores)[::-1]
    # pool_indices 是已经按分数降序排列的 3*n 个索引
    pool_indices = partition_indices[sorted_order]

    # --- 2. 为每个候选用例计算分组键 ---
    groups = defaultdict(list)

    for idx in pool_indices:
        seq = test_labels[idx] # 该用例在所有时间步的预测标签
        prob = final_pro[idx] # 最终时间步的预测概率分布

        # 统计标签出现的频次
        vals, counts = np.unique(seq, return_counts=True)
        max_count = np.max(counts)
        max_classes = vals[counts == max_count]

        if len(max_classes) >= 2:
            # 特殊情况1：多个标签并列出现次数最多（例如偶数时间步各占一半）
            # 使用 final_pro 的第一大和第二大概率类别作为键
            top2 = np.argpartition(prob, -2)[-2:]
            sorted_top2 = top2[np.argsort(prob[top2])[::-1]]
            first, second = sorted_top2[0], sorted_top2[1]
        elif max_count == len(seq):
            # 特殊情况2：所有时间步预测标签完全相同
            first = max_classes[0]      # 唯一的标签
            # 取 final_pro 的第二大概率类别（不排除与 first 相同的情况，实际极少发生）
            second = np.argsort(prob)[-2]   # 倒数第二个索引即为第二大
        else:
            # 普遍情况：最高频次唯一且不是全相同
            # 取频次最高的两个类别（按频次降序）
            sorted_idx = np.argsort(counts)[::-1]
            first = vals[sorted_idx[0]]
            second = vals[sorted_idx[1]]   # 至少存在两个不同类别，故索引安全

        key = (first, second)
        groups[key].append(idx)  # 由于 pool_indices 已按分数降序，组内自动有序

    # --- 3. 按比例分配配额 ---
    total_pool_size = len(pool_indices)
    quotas = {}

    for group_key, indices in groups.items():
        prop_count = int(n * len(indices) / total_pool_size)
        quotas[group_key] = max(1, prop_count)  # 每组至少选 1 个

    quotas = adjustQuotas(quotas, groups, n)  # 调整配额使总和等于 n

    # --- 4. 选取 ---
    selected_indices = []
    for group_key, count in quotas.items():
        selected_indices.extend(groups[group_key][:count])

    return selected_indices


def select_diverse_subset3(scores, label_seq, final_pro, n):
    """
        通过多阶段筛选提升测试子集多样性：
        1. 质量初选 (Scores) -> 2. 混淆度过滤 (Margin) -> 3. 行为模式分组 (RNN Sequence)
    """
    total_size = len(scores)
    if n == 0:
        return []
    if n >= total_size:
        return list(np.arange(total_size))

    # 阶段 1：预筛选 - 建立高潜质候选池 (大小为 3*n)
    ''' 试一下2*n '''
    pool_size = min(2 * n, total_size)
    # 获取分数最高的 pool_size 个用例索引（快速排序，所以内部是无序的）
    partition_indices = np.argpartition(scores, -pool_size)[-pool_size:]
    # 按分数从高到低排序，确保后续组内选取时质量优先
    sorted_order = np.argsort(scores[partition_indices])[::-1]
    pool_indices = partition_indices[sorted_order]

    # 阶段 2：不确定性过滤 - 挖掘分类边界用例
    probs_sub = final_pro[pool_indices]

    # 获取每个样本概率最高的两个类别索引（top2_idx）
    top2_idx = np.argpartition(probs_sub, -2, axis=1)[:, -2:]
    top2_vals = np.take_along_axis(probs_sub, top2_idx, axis=1)

    # 显式排序确保：第一列为最大概率 (maxp)，第二列为次大 (max2p)
    sort_order = np.argsort(top2_vals, axis=1)[:, ::-1]
    top2_idx_sorted = np.take_along_axis(top2_idx, sort_order, axis=1)
    top2_vals_sorted = np.take_along_axis(top2_vals, sort_order, axis=1)

    top1_labels = top2_idx_sorted[:, 0]
    top2_labels = top2_idx_sorted[:, 1]
    maxp = top2_vals_sorted[:, 0]
    max2p = top2_vals_sorted[:, 1]

    ''' 这里如果用最终时间步的maxp和max2p，有时候模型在序列中间非常纠结，但最后一步却表现得很自信（即使那是错的） '''
    ''' 试了倒数第5时间步的maxp和max2p，表现不行'''
    # 计算差值 (Margin/Diff)：差值越小说明模型越“混淆”
    diff = maxp - max2p

    # 按差值升序排序，保留前 1.5n 个最不确定的用例
    sorted_diff_order = np.argsort(diff)
    ''' 试一下1.3*n '''
    keep_count = min(int((4/3) * n), len(pool_indices))
    # keep_count = min(int((4/3) * n), len(pool_indices))
    keep_order = sorted_diff_order[:keep_count]

    # 提取过滤后的核心指标
    filtered_indices = pool_indices[keep_order]
    filtered_top1 = top1_labels[keep_order]
    filtered_top2 = top2_labels[keep_order]

    if len(filtered_indices) == 0:
        return []

    # 阶段 3：行为模式分组 - 基于 RNN 序列预测路径
    groups = defaultdict(list)

    for i, idx in enumerate(filtered_indices):
        seq = label_seq[idx]
        prob_top1 = filtered_top1[i]
        prob_top2 = filtered_top2[i]

        # 统计序列中各标签出现的频次
        vals, counts = np.unique(seq, return_counts=True)
        max_count = np.max(counts)
        max_classes = vals[counts == max_count]

        # 分组键 (Key) 确定逻辑：
        if len(max_classes) >= 2:
            # 情况1：多个标签并列最高频 -> 使用最终概率分布的 Top1 和 Top2
            first, second = prob_top1, prob_top2
        elif max_count == len(seq):
            # 情况2：序列预测标签完全一致 -> 使用该标签与最终步次高概率标签
            first = max_classes[0]
            second = prob_top2
        else:
            # 普遍情况：取序列中频次最高的两个标签
            sorted_idx = np.argsort(counts)[::-1]
            first = vals[sorted_idx[0]]
            second = vals[sorted_idx[1]]

        groups[(first, second)].append(idx)

    # 阶段 4：配额分配与最终选择
    total_filtered = len(filtered_indices)
    quotas = {}

    # 按各行为组在筛选池中的占比初步分配预算 n
    for group_key, indices in groups.items():
        prop_count = int(n * len(indices) / total_filtered)
        quotas[group_key] = max(1, prop_count)

    # 调用外部函数微调配额，确保总数精确等于 n
    quotas = adjustQuotas(quotas, groups, n)

    selected_indices = []
    for group_key, count in quotas.items():
        ''' 目前在每个组内直接取 [:count]（即分数最高的前几个）。由于同一组内的用例行为已经很接近，高分用例往往在特征空间上也是聚集的，这会产生冗余 '''
        # 由于 filtered_indices 内部保持了分数降序，此处直接取每组前 count 个
        selected_indices.extend(groups[group_key][:count])

    return selected_indices

def score_margin_group_sampling(
        scores,
        final_pro,
        n,
        score_multiplier=2.0,
        margin_multiplier=2.0,
        min_group_quota=1,
        random_seed=None
):
    """
    Score-margin candidate union + balanced top-2 confusion group sampling.

    The method keeps TemporalRNN score as the main precision guard, adds
    final-step boundary cases by small top1-top2 margin, then samples balanced
    final-step (top1_label, top2_label) groups. Within each group, cases are
    always ranked by failure-risk score descending.
    """
    scores = np.asarray(scores)
    final_pro = np.asarray(final_pro)

    total_size = len(scores)
    if n == 0:
        return []
    if n >= total_size:
        return list(np.arange(total_size))

    score_multiplier = max(1.0, float(score_multiplier))
    margin_multiplier = max(1.0, float(margin_multiplier))
    min_group_quota = max(1, int(min_group_quota))

    # High-risk pool by TemporalRNN failure-risk score.
    score_pool_size = min(int(np.ceil(score_multiplier * n)), total_size)
    score_partition = np.argpartition(scores, -score_pool_size)[-score_pool_size:]
    score_order = np.argsort(scores[score_partition])[::-1]
    score_pool_indices = score_partition[score_order]

    # Boundary pool by final-step margin = maxp - max2p.
    top2_unsorted = np.argpartition(final_pro, -2, axis=1)[:, -2:]
    top2_scores = np.take_along_axis(final_pro, top2_unsorted, axis=1)
    top2_order = np.argsort(top2_scores, axis=1)[:, ::-1]
    top2_labels = np.take_along_axis(top2_unsorted, top2_order, axis=1)
    top2_scores_sorted = np.take_along_axis(top2_scores, top2_order, axis=1)
    margins = top2_scores_sorted[:, 0] - top2_scores_sorted[:, 1]

    margin_pool_size = min(int(np.ceil(margin_multiplier * n)), total_size)
    margin_pool_indices = np.argsort(margins)[:margin_pool_size]

    # Merge score and margin pools, preserving score-pool priority.
    merged = []
    seen = set()
    for idx in np.concatenate([score_pool_indices, margin_pool_indices]):
        idx = int(idx)
        if idx not in seen:
            merged.append(idx)
            seen.add(idx)

    if not merged:
        return list(np.argsort(scores)[::-1][:n])

    # Group candidates by final-step top-2 confusion pattern.
    groups = defaultdict(list)
    for idx in merged:
        key = (int(top2_labels[idx, 0]), int(top2_labels[idx, 1]))
        groups[key].append(idx)

    # Within each group, score remains the ranking criterion.
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda idx: scores[idx], reverse=True)

    rng = np.random.default_rng(random_seed) if random_seed is not None else None

    def group_sort_key(key):
        group_scores = scores[groups[key]]
        tie_break = rng.random() if rng is not None else 0.0
        return (float(np.max(group_scores)), tie_break)

    group_keys = sorted(groups.keys(), key=group_sort_key, reverse=True)
    cursors = {key: 0 for key in group_keys}
    selected = []
    selected_set = set()

    # First pass: give high-priority groups a small guaranteed quota.
    for key in group_keys:
        quota = min(min_group_quota, len(groups[key]))
        for _ in range(quota):
            if len(selected) >= n:
                return selected[:n]
            idx = int(groups[key][cursors[key]])
            cursors[key] += 1
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

    # Balanced fill: take one extra case per group per pass.
    while len(selected) < n:
        updated = False
        for key in group_keys:
            if cursors[key] >= len(groups[key]):
                continue
            idx = int(groups[key][cursors[key]])
            cursors[key] += 1
            updated = True
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) >= n:
                return selected[:n]
        if not updated:
            break

    # Fallback only if the merged pool cannot fill the budget.
    if len(selected) < n:
        for idx in np.argsort(scores)[::-1]:
            idx = int(idx)
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
            if len(selected) >= n:
                break

    return selected[:n]