from collections import Counter
import numpy as np
import pandas as pd
from deepstellar.coverage import Coverage
import os

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def mnist_input_preprocess(data):
    data = data.reshape(data.shape[0], 28, 28)
    data = data.astype('float32')
    data /= 255
    return data

def svhn_input_preprocess(data):
    data = data.reshape(data.shape[0], 32, 32, 3)
    data = data.astype('float32')
    data /= 255
    data = data.mean(axis=-1)  # Convert to grayscale
    return data



# calculate change rate (without weights)
def cacl_change_rate(array):
    count = 0
    if len(array) <= 1:
        return 0
    for i in range(1, len(array)):
        if array[i] != array[i - 1]:
            count = count + 1
    return count / (len(array) - 1)


# calculate change rate (with weights)
def cacl_change_rate_with_weights(array):
    up = 0
    down = 0
    for i in range(1, len(array)):
        down = down + (i * i)
        if array[i] != array[i - 1]:
            up = up + (i * i)
    if down == 0:
        return 0
    else:
        return up / down


# get change set
def get_change_set(label_seq):
    change_set = set()
    for i in range(len(label_seq) - 1):
        tmp1 = label_seq[i]
        tmp2 = label_seq[i + 1]
        change_set.add(str(tmp1) + str(tmp2))
    return change_set


# calculate Jaccard similarity
def calc_Jaccard_sim(x, y):
    return len((x & y))/len((x | y)) if len((x | y)) != 0 else 0


def gini_sort_order(x):
    y = np.sort(x)[::-1]
    d = Counter(x)

    order = []
    for i in x:
        arg_li = np.where(y == i)[0]
        if len(arg_li) == 1:
            order.append(arg_li[0])
        elif len(arg_li) > 1:
            order.append(arg_li[d[i] - 1])
            d[i] = d[i] - 1
    return order


# The selection method of DeepState: change rate first, then compare the change trend.
def selection(change_rate_li, trend, n):
    d = Counter(change_rate_li)
    # 根据CR指标降序排列
    sorted_d = sorted(dict(d), reverse=True)  # The change rate is sorted from large to small, and count the numbers
    selected = np.zeros(len(change_rate_li))  # The selected mark is 1, and the eliminated mark is -1

    count = 0
    for value in sorted_d:
        num = dict(d)[value]  # The number of use cases corresponding to the current change rate
        if num == 1:
            place = np.where(change_rate_li == np.float64(value))[0][0]
            selected[place] = 1
            count += 1
            if count >= n:
                return selected

        elif num > 1:
            place_li = np.where(change_rate_li == np.float64(value))[0]
            for j in range(len(place_li)):
                if selected[place_li[j]] == -1 or selected[place_li[j]] == 1:
                    continue
                selected[place_li[j]] = 1
                count += 1
                if count >= n:
                    return selected

                tmp_trend1 = trend[place_li[j]]
                # print("selected case trend:", tmp_trend1)   #
                for k in range(j + 1, len(place_li)):
                    if selected[place_li[k]] == -1 or selected[place_li[k]] == 1:
                        continue
                    tmp_trend2 = trend[place_li[k]]
                    tmp_sim = calc_Jaccard_sim(tmp_trend1, tmp_trend2)  # The bigger the sim, the higher the similarity
                    # print("tmp_sim between case", place_li[j], "and", place_li[k], "is", tmp_sim)   #
                    if tmp_sim > 0.5:  # 0.2
                        selected[place_li[k]] = -1
                    # else:
                    #     selected[place_li[k]] = 1
                    #     count += 1
                    #     if count >= n:
                    #         return selected

    if count < n:
        print("selection not enough. It will full fill the other cases.")
        for p in range(len(selected)):
            if selected[p] == -1:
                selected[p] = 1
                count += 1
                if count == n:
                    return selected

def total_selection_random_uncertainty(final, length, selected_num):
    selected = np.zeros(length)
    distance = np.array(final[0])
    diss = np.array(final[1])

    # 对 distance 进行随机排序
    distance_indices = np.random.permutation(len(distance))

    selected_indices = []
    selected_set = set()
    cnt = 0

    # 然后选择不同 diss 值的元素
    for idx in distance_indices:
        if cnt >= selected_num:
            break
        if diss[idx] != 0 and diss[idx] not in selected_set:
            selected_indices.append(idx)
            selected_set.add(diss[idx])
            cnt += 1

    # 如果选择的数量不足 selected_num，补充选择剩余的元素
    if cnt < selected_num:
        for idx in distance_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
                cnt += 1
                if cnt >= selected_num:
                    break

    selected[selected_indices] = 1

    # print('selected:', selected)
    return selected


def total_selection_random_sim(final, length, selected_num):
    selected = np.zeros(length)
    distance = np.array(final[0])
    diss = np.array(final[1])

    # 对 distance 进行升序排列
    distance_indices = np.argsort(distance)

    selected_indices = []
    cnt = 0

    # 随机以相同的概率选择元素
    for idx in distance_indices:
        if cnt >= selected_num:
            break
        if np.random.rand() < 0.5:  # 以相同的概率选择
            selected_indices.append(idx)
            cnt += 1

    # 如果选择的数量不足 selected_num，按顺序补充选择剩余的元素
    if cnt < selected_num:
        remaining_indices = [i for i in distance_indices if i not in selected_indices]
        while cnt < selected_num and remaining_indices:
            idx = remaining_indices.pop(0)  # 按顺序选择
            selected_indices.append(idx)
            cnt += 1

    selected[selected_indices] = 1

    # print('selected:', selected)
    return selected

def ran_selection(length, select_num):
    x = np.zeros(length - select_num)
    y = np.ones(select_num)
    z = np.concatenate((x, y))
    np.random.shuffle(z)
    return z


# selection evaluation
def selection_evaluate(right, select):
    collections_right = Counter(right)
    collections_select = Counter(select)
    T_o = len(right)  # The size of the original sample
    # 获取被选测试子集的大小
    T_s = collections_select[1]  # The size of the selected sample
    # 获取候选测试集的bug数量
    Tf_o = collections_right[0]  # The number of bug cases in the original sample
    # 记录被选测试子集中bug数量
    Tf_s = 0  # The number of bug cases in the selected sample
    for right_value, select_value in zip(right, select):
        if right_value == 0 and select_value == 1:  # A bug case is detected and selected
            Tf_s += 1
    R = Tf_s / Tf_o if Tf_o != 0 else 0  # inclusiveness
    P = Tf_s / T_s if T_s != 0 else 0  # bug detection rate of the selected set
    O_P = Tf_o / T_o if T_o != 0 else 0  # bug detection rate of the original set

    theo_R = T_s / Tf_o if T_s < Tf_o else 1
    theo_P = Tf_o / T_s if T_s > Tf_o else 1

    return R, P, O_P, theo_R, theo_P


# check the predict result the right or wrong
def check_predict_result(predict, label, right):
    if predict == label:
        # print("predict right:", 1)
        right.append(1)
    else:
        # print("predict right:", 0)
        right.append(0)


def cam_selection(x, length, select_num):
    selected = np.zeros(length)
    original_selected_num = len(x)
    if original_selected_num >= select_num:
        final_selected = x[:select_num]
    else:  # The use case selected by cov is smaller than the expected use case, then randomly add the remaining ones
        tmp = np.setdiff1d(np.arange(length), x)
        np.random.shuffle(tmp)
        final_selected = np.append(x, tmp[:(select_num - original_selected_num)])
    # print(final_selected)
    for i in final_selected:
        selected[i] = 1
    return selected


def ctm_selection(cov, length, selected_num):
    selected = np.zeros(length)
    arg_sorted_cov = cov.argsort()[::-1]
    for i in arg_sorted_cov[:selected_num]:
        selected[i] = 1
    return selected


def nc_cam_selection(nc_cam, length, select_num):
    final_selected = np.zeros(length)
    selected_id = []

    count = 0
    for i in range(len(nc_cam)):
        if nc_cam[i] == 1:
            selected_id.append(i)
            count += 1
            if count >= select_num:
                break
    if count < select_num:
        tmp = np.setdiff1d(np.arange(length), selected_id)
        np.random.shuffle(tmp)
        selected_id = selected_id + list(tmp[:(select_num - count)])

    for i in selected_id:
        final_selected[i] = 1

    return final_selected


def get_stellar_cov(classifier, model, x, dtmc_wrapper_f):
    BSCov, BTCov = 0, 0
    stats = classifier.get_state_profile(np.array([x]), model)
    coverage_handlers = []

    for criteria, k_step in [("state", 0), ("transition", 0)]:  # , ("k-step", 3), ("k-step", 6)
        cov = Coverage(dtmc_wrapper_f, criteria, k_step)
        coverage_handlers.append(cov)

    for i, coverage_handler in enumerate(coverage_handlers):
        cov = coverage_handler.get_coverage_criteria(stats)
        total = coverage_handler.get_total()
        if i == 0:
            BSCov = len(cov) / total  # Basic State Coverage(BSCov)
        if i == 1:
            BTCov = len(cov) / total  # Basic Transition Coverage(BTCov)
    return BSCov, BTCov


def get_testrnn_sc(plus_sum, minus_sum):
    count = 0
    act_time = []
    for i in range(1, len(plus_sum)):
        delta = abs(plus_sum[i] - plus_sum[i - 1]) + abs(minus_sum[i] - minus_sum[i - 1])
        if delta >= 0.6:
            count += 1
            act_time.append(i)
    sc = count / len(plus_sum) if count != 0 else 0
    return sc, set(act_time)


def get_nc_activate(lstm_out):
    activated = np.argwhere(lstm_out[0] > 0).tolist()
    activated_li = []
    for a in activated:
        a = tuple(a)
        activated_li.append(a)
    act = set(activated_li)
    return act


# 计算范数（这里没有1-）
def cal_distance(x):
    distance = np.sqrt(np.sum(x ** 2))
    return distance

# 计算DU指标（如果越小，就越不确定，即最后应该升序排列）
def cal_weight_dis(dis_seq):
    n = len(dis_seq)
    sum = 0.0
    m = 0.0
    for i in range(1, n + 1):
        # sum += np.exp(i) * dis_seq[i - 1]
        # m += np.exp(i)
        sum += (i ** 2) * dis_seq[i - 1]
        m += (i ** 2)
    return sum / m


def distance_selection(distance, length, selected_num):
    selected = np.zeros(length)
    arg_sorted_dis = distance.argsort()
    for i in arg_sorted_dis[:selected_num]:
        selected[i] = 1
    return selected


def total_selection_sensitive(final, length, selected_num):
    selected = np.zeros(length)
    distance = np.array(final[0])
    diss = np.array(final[1])
    # 对 distance 进行升序排列
    distance_indices = np.argsort(distance)
    distance_sorted = distance[distance_indices]
    diss_sorted = diss[distance_indices]
    selected_indices = []
    i = 0
    cnt = 0
    while i < len(distance_sorted):
        if i == len(distance_sorted) - 1 or diss_sorted[i] != diss_sorted[i + 1]:
            selected_indices.append(distance_indices[i])
            cnt += 1
        elif diss_sorted[i] == diss_sorted[i + 1] and diss_sorted[i] == 0:
            selected_indices.append(distance_indices[i])
            cnt += 1
        else:
            selected_indices.append(distance_indices[i])
            cnt += 1
            i += 1
        i += 1
        if cnt >= selected_num:
            break
    selected[selected_indices] = 1
    if cnt < selected_num:
        for i in range(len(selected)):
            if selected[i] != 1:
                selected[i] = 1
                cnt += 1
                if cnt >= selected_num:
                    break

    # print('selected:', selected)
    return selected

''' DeepVec方法 '''
# final是DeepVec方法的DU和CS指标，length是候选测试集的大小，selected_num是选择测试子集的大小
def total_selection(final, length, selected_num):
    selected = np.zeros(length)
    distance = np.array(final[0])
    diss = np.array(final[1])

    # 对 distance 进行升序排列
    ''' 因为前面计算distance没有用1-，所以这里用的是升序排列 '''
    distance_indices = np.argsort(distance)

    selected_indices = []
    selected_set = set()
    cnt = 0

    # 然后选择不同 diss 值的元素
    for idx in distance_indices:
        if cnt >= selected_num:
            break
        # 判断CS指标是否在被选测试子集里出现过
        if diss[idx] != 0 and diss[idx] not in selected_set:
            selected_indices.append(idx)
            selected_set.add(diss[idx])
            cnt += 1

    # 如果选择的数量不足 selected_num，补充选择剩余的元素
    if cnt < selected_num:
        for idx in distance_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
                cnt += 1
                if cnt >= selected_num:
                    break

    selected[selected_indices] = 1

    # print('selected:', selected)
    return selected



# 计算CS指标
'''这里用的是e的t次方加权，不是t的平方加权'''
def deal_vecseq(vec_seq):
    angles = []  # 存储夹角序列
    count = 0  # 超过45度的计数
    n = len(vec_seq[0])
    base = np.ones(n)
    maxi = np.zeros(n)
    maxi[0] = 1.0
    t = 0
    for i in range(len(vec_seq) - 1):
        vec1 = vec_seq[i]
        vec2 = vec_seq[i + 1]
        vec1_normalized = vec1 / np.linalg.norm(vec1)  # 归一化第一个向量
        vec2_normalized = vec2 / np.linalg.norm(vec2)  # 归一化第二个向量
        angle = angle_between_vectors(vec1_normalized, vec2_normalized)
        angles.append(np.degrees(angle))  # 将弧度转换为角度制
        if np.degrees(angle) > np.degrees(angle_between_vectors(base, maxi)):  # 判断是否超过 tau
            ''' 改成用t平方作为权重还好一点 '''
            # count += np.exp(i + 1)  # 使用循环次数 t 的平方加权
            count += (i + 1) ** 2
        # t += np.exp(i + 1)
        t += (i + 1) ** 2
    ratio = count / t  # 超过45度的次数与总长度之比，使用循环次数 t 的平方加权
    return ratio


# 辅助函数，计算两个向量之间的夹角（弧度）
def angle_between_vectors(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos_angle = np.clip(dot_product / norm_product, -1, 1)  # 将cos_angle限制在[-1, 1]范围内
    angle = np.arccos(cos_angle)
    return angle

def count_unique_elements(fault_types, selection):
    selected_faults = [fault_types[i] for i, selected in enumerate(selection) if selected == 1]
    unique_elements = set()

    for fault in selected_faults:
        if fault[0] != fault[1]:
            unique_elements.add(fault)

    return len(unique_elements)

def deal_vecseq_sensitive(vec_seq,tau):
    angles = []  # 存储夹角序列
    count = 0  # 超过45度的计数
    n = len(vec_seq[0])
    maxi = np.zeros(n)
    maxi[0] = 1.0
    t = 0
    for i in range(len(vec_seq) - 1):
        vec1 = vec_seq[i]
        vec2 = vec_seq[i + 1]
        vec1_normalized = vec1 / np.linalg.norm(vec1)  # 归一化第一个向量
        vec2_normalized = vec2 / np.linalg.norm(vec2)  # 归一化第二个向量
        angle = angle_between_vectors(vec1_normalized, vec2_normalized)
        angles.append(np.degrees(angle))  # 将弧度转换为角度制
        if np.degrees(angle) > tau:  # 判断是否超过 tau
            count += np.exp(i + 1)  # 使用循环次数 t 的平方加权
        t += np.exp(i + 1)
    ratio = count / t  # 超过45度的次数与总长度之比，使用循环次数 t 的平方加权
    return ratio


from sklearn_extra.cluster import KMedoids

def kmedoids_selection(X, total_num, select_num):
    """
    使用 k-medoids 方法选择代表性样本

    参数：
    X: numpy.ndarray, 特征矩阵
    total_num: int, 候选集总数
    select_num: int, 要选的数量

    返回：
    numpy.ndarray, 选择的索引，选择为1，未选择为0
    """
    if len(X.shape) > 2:
        X = X.reshape((X.shape[0], -1))
    kmedoids = KMedoids(n_clusters=select_num, random_state=0).fit(X)
    selected_indices = kmedoids.medoid_indices_
    selection_array = np.zeros(total_num)
    selection_array[selected_indices] = 1
    return selection_array


# import torch

# from MMD.mmd_critic import  select_prototypes, select_criticisms
# from MMD.kernels import rbf_kernel

# def mmdcritic_selection(X,total_num, select_num):
#     # 将输入数据展平并转换为 PyTorch 张量
#     X = torch.tensor(X, dtype=torch.float).view(X.shape[0], -1)
#     # 计算 RBF 核矩阵
#     K = rbf_kernel(X)
#     # 选择原型样本
#     prototype_indices = select_prototypes(K, select_num)
#     # 创建长度为 7000 的索引数组
#     selection_array = torch.zeros(total_num, dtype=torch.long)
#     # 将选择的原型样本索引设置为 1
#     selection_array[prototype_indices] = 1
#     return selection_array.numpy()

def calculate_maxp(x):
    maxptmp = np.max(x)
    return maxptmp

def maxP_selection(maxp, length, selected_num):
    selected = np.zeros(length)
    # maxP应该是升序排列去选择用例，越小约不确定
    arg_sorted_maxp = maxp.argsort()[::1]
    for i in arg_sorted_maxp[:selected_num]:
        selected[i] = 1
    return selected

def calculate_gini(x):
    ginitmp = np.sum(x ** 2)
    return 1.0 - ginitmp

def gini_selection(gini, length, selected_num):
    selected = np.zeros(length)
    # 降序排列（DeepGini值越大，越不确定）
    arg_sorted_gini = gini.argsort()[::-1]
    for i in arg_sorted_gini[:selected_num]:
        selected[i] = 1
    return selected