import os
import keras
import numpy as np
from scipy.io import loadmat
from statics import *
import argparse
from keras.datasets import mnist
from keras.datasets import fashion_mnist


# 信息熵计算的向量化实现
def calculate_info_entropy_vectorized(probs):
    # 添加微小值防止log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log2(probs), axis=-1)


def load_all_data_for_agnews_snips(path):
    """为AgNews和SNIPS数据集加载全部训练数据"""
    with np.load(path, allow_pickle=True) as f:
        X_train, Y_train = f['X_train'], f['Y_train']

    return X_train, Y_train


def load_first10_data_for_agnews_snips(path):
    """为AgNews和SNIPS数据集加载未用于训练的前10%数据"""
    with np.load(path, allow_pickle=True) as f:
        X_val, Y_val = f['X_val'], f['Y_val']

    return X_val, Y_val


def load_svhn_data():
    """加载SVHN数据集"""
    train_data = loadmat('data/train_32x32.mat')
    test_data = loadmat('data/test_32x32.mat')
    x_train = train_data['X'].transpose((3, 0, 1, 2))
    y_train = train_data['y'].flatten() - 1
    x_test = test_data['X'].transpose((3, 0, 1, 2))
    y_test = test_data['y'].flatten() - 1
    return x_train, y_train, x_test, y_test


def get_dataset_config(dataset, model_type, sample_size):
    """根据数据集和模型类型返回配置信息"""
    configs = {
        'mnist': {
            'time_steps': 28,
            'batch_size': 32,
            'num_classes': 10,
            'data_preprocess': mnist_input_preprocess,
            'model_class': {
                'lstm': 'MnistLSTMClassifier',
                'gru': 'MnistGRUClassifier',
                'blstm': 'MnistBLSTMClassifier'
            }.get(model_type),
            'model_path': f"./RNNModels/mnist_demo/models/mnist_{model_type}.h5",
            'data_loader': lambda: mnist.load_data()[0]  # 返回(x_train, y_train)
        },
        'fashion': {
            'time_steps': 28,
            'batch_size': 32,
            'num_classes': 10,
            'data_preprocess': mnist_input_preprocess,
            'model_class': {
                'lstm': 'FashionLSTMClassifier',
                'gru': 'FashionGRUClassifier',
                'blstm': 'FashionBLSTMClassifier'
            }.get(model_type),
            'model_path': f"./RNNModels/fashion_demo/models/fashion_{model_type}.h5",
            'data_loader': lambda: fashion_mnist.load_data()[0]
        },
        'svhn': {
            'time_steps': 32,
            'batch_size': 32,
            'num_classes': 10,
            'data_preprocess': svhn_input_preprocess,
            'model_class': {
                'lstm': 'SvhnLSTMClassifier',
                'gru': 'SvhnGRUClassifier',
                'blstm': 'SvhnBLSTMClassifier'
            }.get(model_type),
            'model_path': f"./RNNModels/svhn_demo/models/svhn_{model_type}.h5",
            'data_loader': lambda: load_svhn_data()[:2]  # 返回(x_train, y_train)
        },
        'agnews': {
            'time_steps': 35,
            'batch_size': 32,
            'num_classes': 4,
            'data_preprocess': None,  # 文本数据已预处理
            'model_class': {
                'lstm': 'AgnewsLSTMClassifier',
                'gru': 'AgnewsGRUClassifier',
                'blstm': 'AgnewsBLSTMClassifier'
            }.get(model_type),
            'model_path': f"./RNNModels/agnews_demo/models/agnews_{model_type}.h5",
            'data_loader': lambda s=sample_size: (
                load_all_data_for_agnews_snips(
                    "./RNNModels/agnews_demo/save/standard_data.npz"
                ) if s == 'all' else
                load_first10_data_for_agnews_snips(
                    "./RNNModels/agnews_demo/save/standard_data.npz"
                )
            )
        },
        'snips': {
            'time_steps': 16,
            'batch_size': 32,
            'num_classes': 7,
            'data_preprocess': None,  # 文本数据已预处理
            'model_class': {
                'lstm': 'SnipsLSTMClassifier',
                'gru': 'SnipsGRUClassifier',
                'blstm': 'SnipsBLSTMClassifier'
            }.get(model_type),
            'model_path': f"./RNNModels/snips_demo/models/snips_{model_type}.h5",
            'data_loader': lambda s=sample_size: (
                load_all_data_for_agnews_snips(
                    "./RNNModels/snips_demo/save/standard_data.npz"
                ) if s == 'all' else
                load_first10_data_for_agnews_snips(
                    "./RNNModels/snips_demo/save/standard_data.npz"
                )
            )
        }
    }
    return configs.get(dataset)


def get_model_class(dataset, model_type):
    """动态导入模型类"""
    if dataset == 'mnist':
        if model_type == 'lstm':
            from RNNModels.mnist_demo.mnist_lstm import MnistLSTMClassifier
            return MnistLSTMClassifier
        elif model_type == 'gru':
            from RNNModels.mnist_demo.mnist_gru import MnistGRUClassifier
            return MnistGRUClassifier
        elif model_type == 'blstm':
            from RNNModels.mnist_demo.mnist_blstm import MnistBLSTMClassifier
            return MnistBLSTMClassifier
    elif dataset == 'fashion':
        if model_type == 'lstm':
            from RNNModels.fashion_demo.fashion_lstm import FashionLSTMClassifier
            return FashionLSTMClassifier
        elif model_type == 'gru':
            from RNNModels.fashion_demo.fashion_gru import FashionGRUClassifier
            return FashionGRUClassifier
        elif model_type == 'blstm':
            from RNNModels.fashion_demo.fashion_blstm import FashionBLSTMClassifier
            return FashionBLSTMClassifier
    elif dataset == 'svhn':
        if model_type == 'lstm':
            from RNNModels.svhn_demo.svhn_lstm import SvhnLSTMClassifier
            return SvhnLSTMClassifier
        elif model_type == 'gru':
            from RNNModels.svhn_demo.svhn_gru import SvhnGRUClassifier
            return SvhnGRUClassifier
        elif model_type == 'blstm':
            from RNNModels.svhn_demo.svhn_blstm import SvhnBLSTMClassifier
            return SvhnBLSTMClassifier
    elif dataset == 'agnews':
        if model_type == 'lstm':
            from RNNModels.agnews_demo.agnews_lstm import AgnewsLSTMClassifier
            return AgnewsLSTMClassifier
        elif model_type == 'gru':
            from RNNModels.agnews_demo.agnews_gru import AgnewsGRUClassifier
            return AgnewsGRUClassifier
        elif model_type == 'blstm':
            from RNNModels.agnews_demo.agnews_blstm import AgnewsBLSTMClassifier
            return AgnewsBLSTMClassifier
    elif dataset == 'snips':
        if model_type == 'lstm':
            from RNNModels.snips_demo.snips_lstm import SnipsLSTMClassifier
            return SnipsLSTMClassifier
        elif model_type == 'gru':
            from RNNModels.snips_demo.snips_gru import SnipsGRUClassifier
            return SnipsGRUClassifier
        elif model_type == 'blstm':
            from RNNModels.snips_demo.snips_blstm import SnipsBLSTMClassifier
            return SnipsBLSTMClassifier

    raise ValueError(f"不支持的模型类型: {model_type} 对于数据集: {dataset}")


def main(dataset, model_type, sample_size):
    # 获取配置信息
    config = get_dataset_config(dataset, model_type, sample_size)
    if not config:
        raise ValueError(f"不支持的数据集和模型组合: {dataset} + {model_type}")

    # 获取模型类
    ModelClass = get_model_class(dataset, model_type)

    # 加载模型
    lstm_classifier = ModelClass()

    # 为文本分类模型设置数据路径和嵌入路径
    if dataset in ['agnews', 'snips']:
        lstm_classifier.data_path = f"./RNNModels/{dataset}_demo/save/standard_data.npz"
        lstm_classifier.embedding_path = f"./RNNModels/{dataset}_demo/save/embedding_matrix.npy"

    model = lstm_classifier.load_hidden_state_model(config['model_path'])

    # 重新加载dense层
    dense_classifier = ModelClass()
    dense_model = dense_classifier.reload_dense(config['model_path'])

    # 加载数据
    x_train, y_train = config['data_loader']()

    # 根据sample_size参数截取数据
    if sample_size == 'first10':
        if dataset in ['mnist', 'fashion']:
            x_train = x_train[-6000:]
            y_train = y_train[-6000:]
        elif dataset == 'svhn':
            x_train = x_train[-7000:]
            y_train = y_train[-7000:]
        save_dir = f"./rnn_output/{dataset}_{model_type}/"
    else:  # 'all'
        save_dir = f"./rnn_output/{dataset}_{model_type}/all_train/"

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 数据预处理（图像数据需要预处理，文本数据不需要）
    if config['data_preprocess']:
        x_train_processed = config['data_preprocess'](x_train)
    else:
        x_train_processed = x_train

    # 预测所有样本的RNN输出（所有时间步的隐藏状态）
    lstm_outputs = model.predict(x_train_processed, batch_size=config['batch_size'], verbose=1)[1]

    # 重塑为 (样本数*时间步, 隐藏单元)
    reshaped_outputs = lstm_outputs.reshape(-1, lstm_outputs.shape[-1])

    # 批量预测所有时间步的概率
    all_probs = dense_model.predict(reshaped_outputs, batch_size=config['batch_size'] * 10, verbose=1)

    # 重塑为 (样本数, 时间步, 类别数)
    train_pros = all_probs.reshape(len(x_train_processed), config['time_steps'], -1)

    # 向量化计算预测标签和信息熵
    train_labels = np.argmax(train_pros, axis=-1)
    train_infos = calculate_info_entropy_vectorized(train_pros)

    # 计算最终预测正确性
    final_predictions = train_labels[:, -1]  # 取最后一个时间步的预测

    # 处理标签格式（AgNews和SNIPS是独热编码，需要转换）
    if dataset in ['agnews', 'snips']:
        y_train_labels = np.argmax(y_train, axis=1)
    else:
        y_train_labels = y_train

    right = (final_predictions == y_train_labels).astype(int)

    # 保存所需的预测概率向量、预测标签和预测信息熵，以及是否预测正确
    np.save(f"{save_dir}{dataset}_train_pros.npy", train_pros)
    np.save(f"{save_dir}{dataset}_train_labels.npy", train_labels)
    np.save(f"{save_dir}{dataset}_train_infos.npy", train_infos)
    np.save(f"{save_dir}{dataset}_train_right.npy", right)
    np.save(f"{save_dir}{dataset}_train_lstm.npy", lstm_outputs)

    print(f"处理完成！结果已保存到: {save_dir}")
    print(f"样本数量: {len(x_train)}")
    print(f"预测正确率: {np.mean(right):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='获取RNN模型对训练集的预测信息')
    parser.add_argument('-dataset', required=True, choices=['mnist', 'fashion', 'svhn', 'agnews', 'snips'],
                        help='数据集名称')
    parser.add_argument('-model_type', required=True, choices=['lstm', 'gru', 'blstm'],
                        help='RNN模型类型')
    parser.add_argument('-sample_size', required=True, choices=['all', 'first10'],
                        help='获取的训练样本数量')

    args = parser.parse_args()

    main(args.dataset, args.model_type, args.sample_size)
