import argparse
import hashlib
import os
import random
import sys

import numpy as np
import pandas as pd
from statics import *
from selection_tools import get_val_data
import tensorflow as tf
from tensorflow.keras import backend as K

# Specify that the first GPU is available, if there is no GPU, apply: "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

IMAGE_DATASETS = {"mnist", "fashion", "svhn"}
TEXT_DATASETS = {"snips", "agnews"}
SELECT_METHODS = [
    'random', 'deepgini', 'maxp', 'ats', 'deepstate', 'rnntcs', 'deepvec',
    'temporalrnn'
]

BUDGET_RATIO = 10
DEFAULT_SEED = 2026


def set_retrain_seed(seed):
    """Set seeds before each retraining run to reduce randomness across methods."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_w2v_path(dataset):
    if dataset == "agnews":
        return "RNNModels/agnews_demo/save/w2v_model"
    if dataset == "snips":
        return "RNNModels/snips_demo/save/w2v_model"
    return ""


def get_candidate_path(dataset, candidate_set_id):
    ext = "csv" if dataset in TEXT_DATASETS else "npz"
    return f"./gen_data/{dataset}_toselect/{dataset}_toselect_{candidate_set_id}.{ext}"


def eval_id_path(dataset, candidate_set_id):
    return os.path.join("evaluation_set", dataset, f"candidate_{candidate_set_id}_ids.npz")


def selected_index_path(dataset, model_type, candidate_set_id, budget_ratio, method):
    return f"./selected_index/{dataset}_{model_type}/file{candidate_set_id}_{budget_ratio}_{method}_selected.npy"


def _label_key(label):
    arr = np.asarray(label)
    return tuple(arr.reshape(-1).tolist())


def _image_sample_key(sample, label):
    arr = np.ascontiguousarray(np.squeeze(sample).astype(np.float32))
    digest = hashlib.sha1(arr.tobytes()).hexdigest()
    return (arr.shape, digest, _label_key(label))


def _text_columns(dataset):
    if dataset == "snips":
        return "text", "intent"
    if dataset == "agnews":
        return "news", "label"
    raise ValueError(f"Unsupported text dataset: {dataset}")


def _normalize_text_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _text_row_key(dataset, row):
    text_col, label_col = _text_columns(dataset)
    return (_normalize_text_value(row[text_col]), _normalize_text_value(row[label_col]))


def compute_key_hash(key):
    """Convert any key tuple to a deterministic hash string."""
    return hashlib.md5(str(key).encode()).hexdigest()


def load_full_test_pools(dataset, w2v_path=""):
    if dataset in IMAGE_DATASETS:
        base_dir = f"./gen_data/gen_test_dataset/dau/{dataset}_harder"
        paths = {
            "original_x": os.path.join(base_dir, "x_ori_test_0.npy"),
            "original_y": os.path.join(base_dir, "y_ori_test_0.npy"),
            "mutated_x": os.path.join(base_dir, "x_test_0.npy"),
            "mutated_y": os.path.join(base_dir, "y_test_0.npy"),
        }
        for path in paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Full test pool file not found: {path}")
        original_x = np.load(paths["original_x"], allow_pickle=True)
        original_y = np.load(paths["original_y"], allow_pickle=True)
        mutated_x = np.load(paths["mutated_x"], allow_pickle=True)
        mutated_y = np.load(paths["mutated_y"], allow_pickle=True)
        if len(original_x) != len(mutated_x):
            raise ValueError(
                f"Original/mutated pool size mismatch for {dataset}: "
                f"original={len(original_x)}, mutated={len(mutated_x)}"
            )
        return {
            "kind": "image",
            "original_x": original_x,
            "original_y": original_y,
            "mutated_x": mutated_x,
            "mutated_y": mutated_y,
            "num_base_ids": len(original_x),
        }

    if dataset in TEXT_DATASETS:
        original_path = f"./gen_data/{dataset}_toselect/{dataset}_toselect_ori.csv"
        mutated_path = f"./gen_data/{dataset}_toselect/{dataset}_toselect_aug.csv"
        for path in [original_path, mutated_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Full test pool file not found: {path}")

        original_df = pd.read_csv(original_path)
        mutated_df = pd.read_csv(mutated_path)
        if len(original_df) != len(mutated_df):
            raise ValueError(
                f"Original/mutated pool size mismatch for {dataset}: "
                f"original={len(original_df)}, mutated={len(mutated_df)}"
            )
        original_x, original_y = get_val_data(original_path, w2v_path)
        mutated_x, mutated_y = get_val_data(mutated_path, w2v_path)
        return {
            "kind": "text",
            "original_df": original_df,
            "mutated_df": mutated_df,
            "original_x": original_x,
            "original_y": original_y,
            "mutated_x": mutated_x,
            "mutated_y": mutated_y,
            "num_base_ids": len(original_df),
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def _add_base_key(mapping, key, base_id, source):
    """Allow one content key to correspond to multiple base_ids."""
    if key not in mapping:
        mapping[key] = []
    mapping[key].append(base_id)


def build_base_id_mapping(dataset, full_pools):
    mapping = {}  # key -> list of base_ids
    if full_pools["kind"] == "image":
        for base_id, (sample, label) in enumerate(zip(full_pools["original_x"], full_pools["original_y"])):
            _add_base_key(mapping, _image_sample_key(sample, label), base_id, "original")
        for base_id, (sample, label) in enumerate(zip(full_pools["mutated_x"], full_pools["mutated_y"])):
            _add_base_key(mapping, _image_sample_key(sample, label), base_id, "mutated")
    else:
        for base_id, row in full_pools["original_df"].iterrows():
            _add_base_key(mapping, _text_row_key(dataset, row), base_id, "original")
        for base_id, row in full_pools["mutated_df"].iterrows():
            _add_base_key(mapping, _text_row_key(dataset, row), base_id, "mutated")
    return mapping


def get_candidate_key_hashes(dataset, candidate_path, w2v_path):
    """Return content hashes of candidate samples in candidate row order."""
    if dataset in IMAGE_DATASETS:
        x_val, y_val = get_val_data(candidate_path, w2v_path)
        hashes = []
        for sample, label in zip(x_val, y_val):
            key = _image_sample_key(sample, label)
            hashes.append(compute_key_hash(key))
        return hashes
    else:
        df = pd.read_csv(candidate_path)
        hashes = []
        for _, row in df.iterrows():
            key = _text_row_key(dataset, row)
            hashes.append(compute_key_hash(key))
        return hashes


def _full_pool_key(dataset, full_pools, base_id, source):
    """Return content key of original/mutated sample for a given base_id."""
    if full_pools["kind"] == "image":
        if source == "original":
            return _image_sample_key(
                full_pools["original_x"][base_id],
                full_pools["original_y"][base_id]
            )
        if source == "mutated":
            return _image_sample_key(
                full_pools["mutated_x"][base_id],
                full_pools["mutated_y"][base_id]
            )
        raise ValueError(f"Unsupported source: {source}")

    if full_pools["kind"] == "text":
        if source == "original":
            return _text_row_key(dataset, full_pools["original_df"].iloc[base_id])
        if source == "mutated":
            return _text_row_key(dataset, full_pools["mutated_df"].iloc[base_id])
        raise ValueError(f"Unsupported source: {source}")

    raise ValueError(f"Unsupported full_pools kind: {full_pools['kind']}")


def prepare_eval_ids(dataset, w2v_path):
    """
    Recover candidate base_ids from existing candidate files and build base-id-level
    held-out complement evaluation sets.
    """
    full_pools = load_full_test_pools(dataset, w2v_path)
    key_to_base_ids = build_base_id_mapping(dataset, full_pools)
    all_base_ids = np.arange(full_pools["num_base_ids"], dtype=np.int64)

    output_dir = os.path.join("evaluation_set", dataset)
    os.makedirs(output_dir, exist_ok=True)

    for candidate_set_id in range(30):
        candidate_path = get_candidate_path(dataset, candidate_set_id)
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"Candidate file not found: {candidate_path}")

        candidate_base_ids_set = set()
        candidate_row_base_ids = []
        candidate_key_hashes_set = set()

        if dataset in IMAGE_DATASETS:
            x_cand_val, y_cand_val = get_val_data(candidate_path, w2v_path)
            for row_idx, (sample, label) in enumerate(zip(x_cand_val, y_cand_val)):
                key = _image_sample_key(sample, label)
                candidate_key_hashes_set.add(compute_key_hash(key))

                base_ids = key_to_base_ids.get(key, [])
                if not base_ids:
                    raise ValueError(
                        f"Candidate sample key not in full pool mapping: "
                        f"dataset={dataset}, candidate={candidate_set_id}, row={row_idx}"
                    )

                # Conservative handling: if duplicate content maps to multiple base_ids,
                # exclude all of them from the evaluation set.
                candidate_base_ids_set.update(int(bid) for bid in base_ids)
                candidate_row_base_ids.append(int(base_ids[0]))
        else:
            df = pd.read_csv(candidate_path)
            for row_idx, row in df.iterrows():
                key = _text_row_key(dataset, row)
                candidate_key_hashes_set.add(compute_key_hash(key))

                base_ids = key_to_base_ids.get(key, [])
                if not base_ids:
                    raise ValueError(
                        f"Candidate text key not in full pool mapping: "
                        f"dataset={dataset}, candidate={candidate_set_id}, row={row_idx}"
                    )

                candidate_base_ids_set.update(int(bid) for bid in base_ids)
                candidate_row_base_ids.append(int(base_ids[0]))

        candidate_base_ids = np.array(sorted(candidate_base_ids_set), dtype=np.int64)
        eval_base_ids = np.array(
            [int(bid) for bid in all_base_ids if int(bid) not in candidate_base_ids_set],
            dtype=np.int64
        )

        overlap = set(candidate_base_ids.tolist()) & set(eval_base_ids.tolist())
        if overlap:
            raise ValueError(
                f"Base-level leakage detected while preparing eval ids: "
                f"dataset={dataset}, candidate={candidate_set_id}, overlap_size={len(overlap)}"
            )

        # Keep full content-hash leakage check as an auxiliary check. Even though
        # candidate files are paired by base_id, x_eval contains both original and
        # mutated samples, so both hashes should be recorded.
        eval_key_hashes_set = set()
        for bid in eval_base_ids:
            original_key = _full_pool_key(dataset, full_pools, int(bid), "original")
            mutated_key = _full_pool_key(dataset, full_pools, int(bid), "mutated")
            eval_key_hashes_set.add(compute_key_hash(original_key))
            eval_key_hashes_set.add(compute_key_hash(mutated_key))

        if len(candidate_row_base_ids) == 0:
            raise ValueError(f"Empty candidate set: dataset={dataset}, candidate={candidate_set_id}")
        if len(eval_base_ids) == 0:
            raise ValueError(f"Empty evaluation set: dataset={dataset}, candidate={candidate_set_id}")

        save_path = eval_id_path(dataset, candidate_set_id)
        np.savez(
            save_path,
            candidate_base_ids=candidate_base_ids,
            candidate_row_base_ids=np.array(candidate_row_base_ids, dtype=np.int64),
            candidate_keys_hashes=np.array(list(candidate_key_hashes_set), dtype=object),
            eval_base_ids=eval_base_ids,
            eval_keys_hashes=np.array(list(eval_key_hashes_set), dtype=object),
        )
        print(
            f"Saved {save_path}: "
            f"candidate_rows={len(candidate_row_base_ids)}, "
            f"candidate_unique_base_ids={len(candidate_base_ids)}, "
            f"eval_base_ids={len(eval_base_ids)}, "
            f"expected_x_eval_size={2 * len(eval_base_ids)}"
        )


def build_eval_data_from_base_ids(full_pools, eval_base_ids):
    x_eval = np.concatenate([
        full_pools["original_x"][eval_base_ids],
        full_pools["mutated_x"][eval_base_ids],
    ])
    y_eval = np.concatenate([
        full_pools["original_y"][eval_base_ids],
        full_pools["mutated_y"][eval_base_ids],
    ])
    return x_eval, y_eval


def assert_no_key_leakage(selected_key_hashes, eval_key_hashes_set, context):
    intersection = set(selected_key_hashes) & eval_key_hashes_set
    if intersection:
        sample = list(intersection)[:5]
        raise ValueError(
            f"Key leakage detected for {context}. "
            f"Selected hashes intersect with eval key hashes. "
            f"Intersection size={len(intersection)}. Sample={sample}"
        )


def assert_no_base_id_leakage(selected_base_ids, eval_base_ids, context):
    selected_base_ids_set = set(np.asarray(selected_base_ids, dtype=np.int64).tolist())
    eval_base_ids_set = set(np.asarray(eval_base_ids, dtype=np.int64).tolist())
    intersection = selected_base_ids_set & eval_base_ids_set
    if intersection:
        sample = list(intersection)[:10]
        raise ValueError(
            f"Base-level leakage detected for {context}. "
            f"Selected base_ids intersect with eval base_ids. "
            f"Intersection size={len(intersection)}. Sample={sample}"
        )


def normalize_selected_to_mask(selected, candidate_size, context):
    """
    将不同格式的 selected 文件统一转换为 bool mask。

    支持三种格式：
    1. bool mask: [True, False, ...]
    2. 0/1 mask: [1, 0, 0, 1, ...]
    3. -1/1 mask: [1, -1, -1, 1, ...]，其中 1 表示 selected，-1 表示 unselected
    4. index list: [12, 35, 100, ...]
    """
    selected = np.asarray(selected).reshape(-1)

    if len(selected) == candidate_size:
        unique_vals = set(np.unique(selected).tolist())

        # bool mask
        if selected.dtype == np.bool_:
            return selected.astype(bool)

        # 0/1 mask
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            return selected.astype(bool)

        # -1/1 mask 或 -1/0/1 mask
        # 约定：大于 0 表示 selected，其余表示 unselected
        if unique_vals.issubset({-1, 0, 1, -1.0, 0.0, 1.0}):
            return selected > 0

    # 否则按 index list 处理
    try:
        indices = selected.astype(int)
    except Exception as exc:
        raise ValueError(
            f"Cannot convert selected array to mask or indices for {context}. "
            f"selected dtype={selected.dtype}, shape={selected.shape}"
        ) from exc

    if len(indices) == 0:
        return np.zeros(candidate_size, dtype=bool)

    min_idx = indices.min()
    max_idx = indices.max()
    if min_idx < 0 or max_idx >= candidate_size:
        raise ValueError(
            f"Index out of range for {context}: "
            f"min={min_idx}, max={max_idx}, candidate_size={candidate_size}. "
            f"If this is a -1/1 mask, please check normalize_selected_to_mask()."
        )

    mask = np.zeros(candidate_size, dtype=bool)
    mask[indices] = True
    return mask


def get_model_config(args):
    w2v_path = ""
    if args.model_type == "lstm" and args.dataset == "mnist":
        from RNNModels.mnist_demo.mnist_lstm import MnistLSTMClassifier
        return MnistLSTMClassifier(), "./RNNModels/mnist_demo/models_retrain/", "./RNNModels/mnist_demo/models/mnist_lstm.h5", w2v_path
    if args.model_type == "gru" and args.dataset == "mnist":
        from RNNModels.mnist_demo.mnist_gru import MnistGRUClassifier
        return MnistGRUClassifier(), "./RNNModels/mnist_demo/models_retrain/", "./RNNModels/mnist_demo/models/mnist_gru.h5", w2v_path
    if args.model_type == "blstm" and args.dataset == "mnist":
        from RNNModels.mnist_demo.mnist_blstm import MnistBLSTMClassifier
        return MnistBLSTMClassifier(), "./RNNModels/mnist_demo/models_retrain/", "./RNNModels/mnist_demo/models/mnist_blstm.h5", w2v_path
    if args.model_type == "lstm" and args.dataset == "fashion":
        from RNNModels.fashion_demo.fashion_lstm import FashionLSTMClassifier
        return FashionLSTMClassifier(), "./RNNModels/fashion_demo/models_retrain/", "./RNNModels/fashion_demo/models/fashion_lstm.h5", w2v_path
    if args.model_type == "gru" and args.dataset == "fashion":
        from RNNModels.fashion_demo.fashion_gru import FashionGRUClassifier
        return FashionGRUClassifier(), "./RNNModels/fashion_demo/models_retrain/", "./RNNModels/fashion_demo/models/fashion_gru.h5", w2v_path
    if args.model_type == "blstm" and args.dataset == "fashion":
        from RNNModels.fashion_demo.fashion_blstm import FashionBLSTMClassifier
        return FashionBLSTMClassifier(), "./RNNModels/fashion_demo/models_retrain/", "./RNNModels/fashion_demo/models/fashion_blstm.h5", w2v_path
    if args.model_type == "lstm" and args.dataset == "svhn":
        from RNNModels.svhn_demo.svhn_lstm import SvhnLSTMClassifier
        return SvhnLSTMClassifier(), "./RNNModels/svhn_demo/models_retrain/", "./RNNModels/svhn_demo/models/svhn_lstm.h5", w2v_path
    if args.model_type == "gru" and args.dataset == "svhn":
        from RNNModels.svhn_demo.svhn_gru import SvhnGRUClassifier
        return SvhnGRUClassifier(), "./RNNModels/svhn_demo/models_retrain/", "./RNNModels/svhn_demo/models/svhn_gru.h5", w2v_path
    if args.model_type == "blstm" and args.dataset == "svhn":
        from RNNModels.svhn_demo.svhn_blstm import SvhnBLSTMClassifier
        return SvhnBLSTMClassifier(), "./RNNModels/svhn_demo/models_retrain/", "./RNNModels/svhn_demo/models/svhn_blstm.h5", w2v_path
    if args.model_type == "lstm" and args.dataset == "agnews":
        from RNNModels.agnews_demo.agnews_lstm import AgnewsLSTMClassifier
        classifier = AgnewsLSTMClassifier()
        classifier.data_path = "RNNModels/agnews_demo/save/standard_data.npz"
        classifier.embedding_path = "RNNModels/agnews_demo/save/embedding_matrix.npy"
        return classifier, "./RNNModels/agnews_demo/models_retrain/", "./RNNModels/agnews_demo/models/agnews_lstm.h5", get_w2v_path("agnews")
    if args.model_type == "gru" and args.dataset == "agnews":
        from RNNModels.agnews_demo.agnews_gru import AgnewsGRUClassifier
        classifier = AgnewsGRUClassifier()
        classifier.data_path = "RNNModels/agnews_demo/save/standard_data.npz"
        classifier.embedding_path = "RNNModels/agnews_demo/save/embedding_matrix.npy"
        return classifier, "./RNNModels/agnews_demo/models_retrain/", "./RNNModels/agnews_demo/models/agnews_gru.h5", get_w2v_path("agnews")
    if args.model_type == "blstm" and args.dataset == "agnews":
        from RNNModels.agnews_demo.agnews_blstm import AgnewsBLSTMClassifier
        classifier = AgnewsBLSTMClassifier()
        classifier.data_path = "RNNModels/agnews_demo/save/standard_data.npz"
        classifier.embedding_path = "RNNModels/agnews_demo/save/embedding_matrix.npy"
        return classifier, "./RNNModels/agnews_demo/models_retrain/", "./RNNModels/agnews_demo/models/agnews_blstm.h5", get_w2v_path("agnews")
    if args.model_type == "lstm" and args.dataset == "snips":
        from RNNModels.snips_demo.snips_lstm import SnipsLSTMClassifier
        classifier = SnipsLSTMClassifier()
        classifier.data_path = "RNNModels/snips_demo/save/standard_data.npz"
        classifier.embedding_path = "RNNModels/snips_demo/save/embedding_matrix.npy"
        return classifier, "./RNNModels/snips_demo/models_retrain/", "./RNNModels/snips_demo/models/snips_lstm.h5", get_w2v_path("snips")
    if args.model_type == "gru" and args.dataset == "snips":
        from RNNModels.snips_demo.snips_gru import SnipsGRUClassifier
        classifier = SnipsGRUClassifier()
        classifier.data_path = "RNNModels/snips_demo/save/standard_data.npz"
        classifier.embedding_path = "RNNModels/snips_demo/save/embedding_matrix.npy"
        return classifier, "./RNNModels/snips_demo/models_retrain/", "./RNNModels/snips_demo/models/snips_gru.h5", get_w2v_path("snips")
    if args.model_type == "blstm" and args.dataset == "snips":
        from RNNModels.snips_demo.snips_blstm import SnipsBLSTMClassifier
        classifier = SnipsBLSTMClassifier()
        classifier.data_path = "RNNModels/snips_demo/save/standard_data.npz"
        classifier.embedding_path = "RNNModels/snips_demo/save/embedding_matrix.npy"
        return classifier, "./RNNModels/snips_demo/models_retrain/", "./RNNModels/snips_demo/models/snips_blstm.h5", get_w2v_path("snips")
    raise ValueError("The model and dataset set are incorrect.")
    


def save_intermediate_results(detail_rows, args, suffix="partial"):
    """Save intermediate results to CSV."""
    if not detail_rows:
        return
    output_dir = "./results/rq3"
    os.makedirs(output_dir, exist_ok=True)
    partial_path = os.path.join(
        output_dir,
        f"rq3_retrain_holdout_{args.dataset}_{args.model_type}_{suffix}.csv"
    )
    pd.DataFrame(detail_rows).to_csv(partial_path, index=False)
    print(f"Intermediate results saved to: {partial_path}")


# RQ3/RQ4: Retrain the RNNs with selected data and evaluate on held-out complements.
def main(args):
    if args.prepare_eval_ids:
        prepare_eval_ids(args.dataset, get_w2v_path(args.dataset))
        return

    _, retrain_base_path, ori_model_path, w2v_path = get_model_config(args)
    os.makedirs(retrain_base_path, exist_ok=True)

    full_pools = load_full_test_pools(args.dataset, w2v_path)
    detail_rows = []

    for i in range(args.max_candidates):
        print("file index: " + str(i))
        cand_path = get_candidate_path(args.dataset, i)
        if not os.path.exists(cand_path):
            raise FileNotFoundError(f"Candidate file not found: {cand_path}")

        ids_path = eval_id_path(args.dataset, i)
        if not os.path.exists(ids_path):
            raise FileNotFoundError(
                f"Evaluation id file not found: {ids_path}. "
                f"Run: python rq3.py -dataset {args.dataset} -model_type {args.model_type} --prepare_eval_ids"
            )

        x_cand_val, y_cand_val = get_val_data(cand_path, w2v_path)
        print(f"Loaded Candidate Set from {cand_path}, Size: {len(x_cand_val)}")

        ids = np.load(ids_path, allow_pickle=True)
        eval_base_ids = ids["eval_base_ids"]
        eval_keys_hashes = set(ids["eval_keys_hashes"])

        if "candidate_row_base_ids" not in ids:
            raise ValueError(
                f"'candidate_row_base_ids' not found in {ids_path}. "
                f"Please rerun --prepare_eval_ids with the updated code."
            )
        candidate_row_base_ids = ids["candidate_row_base_ids"]

        x_eval, y_eval = build_eval_data_from_base_ids(full_pools, eval_base_ids)

        candidate_key_hashes = get_candidate_key_hashes(args.dataset, cand_path, w2v_path)
        if len(candidate_key_hashes) != len(x_cand_val):
            raise ValueError(
                f"Mismatch between candidate_key_hashes and candidate data length: "
                f"hashes={len(candidate_key_hashes)}, candidate_size={len(x_cand_val)}"
            )

        if len(candidate_row_base_ids) != len(x_cand_val):
            raise ValueError(
                f"Mismatch between candidate_row_base_ids and candidate data length: "
                f"candidate_row_base_ids={len(candidate_row_base_ids)}, "
                f"candidate_size={len(x_cand_val)}, "
                f"dataset={args.dataset}, model={args.model_type}, candidate={i}"
            )

        if len(x_eval) != 2 * len(eval_base_ids):
            raise ValueError(
                f"Evaluation set size mismatch: "
                f"x_eval={len(x_eval)}, eval_base_ids={len(eval_base_ids)}, "
                f"expected={2 * len(eval_base_ids)}, "
                f"dataset={args.dataset}, model={args.model_type}, candidate={i}"
            )

        for method_idx, method in enumerate(SELECT_METHODS):
            print(str(method) + "\n")
            index_file_path = selected_index_path(args.dataset, args.model_type, i, BUDGET_RATIO, method)
            if not os.path.exists(index_file_path):
                print(f"Warning: Index file not found for method {method}: {index_file_path}")
                continue

            selected_raw = np.load(index_file_path, allow_pickle=True)
            context = (
                f"dataset={args.dataset}, model={args.model_type}, "
                f"candidate={i}, method={method}, budget={BUDGET_RATIO}"
            )
            mask = normalize_selected_to_mask(selected_raw, len(x_cand_val), context)

            X_selected_array = x_cand_val[mask]
            Y_selected_array = y_cand_val[mask]
            selected_indices = np.where(mask)[0]

            if len(selected_indices) == 0:
                raise ValueError(f"No selected samples for {context}")

            selected_base_ids = candidate_row_base_ids[selected_indices]
            assert_no_base_id_leakage(selected_base_ids, eval_base_ids, context)

            selected_key_hashes = [candidate_key_hashes[idx] for idx in selected_indices]
            assert_no_key_leakage(selected_key_hashes, eval_keys_hashes, context)

            retrained_model_path = os.path.join(
                retrain_base_path,
                f"{args.dataset}_{args.model_type}_{method}_file{i}_holdout.h5"
            )

            seed = DEFAULT_SEED + i * 100 + method_idx
            set_retrain_seed(seed)
            K.clear_session()
            lstm_classifier, _, _, _ = get_model_config(args)
            lstm_classifier.retrain(X_selected_array, Y_selected_array, retrained_model_path, ori_model_path)

            retrained_acc, imp_acc = lstm_classifier.evaluate_retrain(
                retrained_model_path, ori_model_path, x_eval, y_eval
            )

            if os.path.exists(retrained_model_path):
                os.remove(retrained_model_path)
                print(f"Deleted temporary model: {retrained_model_path}")

            original_acc = retrained_acc - imp_acc
            detail_rows.append({
                "dataset": args.dataset,
                "model": args.model_type,
                "candidate_set_id": i,
                "method": method,
                "budget_ratio": BUDGET_RATIO,
                "selected_size": int(np.sum(mask)),
                "eval_set_size": int(len(x_eval)),
                "original_accuracy_on_Ei": original_acc,
                "retrained_accuracy_on_Ei": retrained_acc,
                "accuracy_improvement": imp_acc,
            })

        save_intermediate_results(detail_rows, args, suffix=f"after_candidate_{i}")

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        summary_df = pd.DataFrame(columns=[
            "method",
            "avg_accuracy_improvement",
            "avg_original_accuracy_on_Ei",
            "avg_retrained_accuracy_on_Ei",
            "num_candidates",
        ])
    else:
        summary_df = (
            detail_df
            .groupby("method", as_index=False)
            .agg(
                avg_accuracy_improvement=("accuracy_improvement", "mean"),
                avg_original_accuracy_on_Ei=("original_accuracy_on_Ei", "mean"),
                avg_retrained_accuracy_on_Ei=("retrained_accuracy_on_Ei", "mean"),
                num_candidates=("candidate_set_id", "count"),
            )
        )

    print("\n========== Final Average Improvement (Held-out Complement Ei) ==========")
    print(summary_df)

    output_dir = "./results/rq3"
    os.makedirs(output_dir, exist_ok=True)
    save_excel_path = os.path.join(
        output_dir,
        f"rq3_retrain_holdout_{args.dataset}_{args.model_type}_results.xlsx"
    )
    with pd.ExcelWriter(save_excel_path, engine='xlsxwriter') as writer:
        detail_df.to_excel(writer, sheet_name='detail', index=False)
        summary_df.to_excel(writer, sheet_name='summary', index=False)
    print(f"Results saved to: {save_excel_path}")


if __name__ == '__main__':
    parse = argparse.ArgumentParser("Calculate retraining improvement on held-out complement evaluation sets.")
    parse.add_argument('-model_type', required=True, choices=['lstm', 'blstm', 'gru'])
    parse.add_argument('-dataset', required=True, choices=['mnist', 'snips', 'fashion', 'agnews', 'svhn'])
    parse.add_argument(
        '--prepare_eval_ids',
        action='store_true',
        help='Prepare evaluation_set/{dataset}/candidate_{i}_ids.npz and exit.'
    )
    parse.add_argument(
        '--max_candidates',
        type=int,
        default=30,
        help='Number of candidate sets to run. Default: 30. Use 5 for quick local checks.'
    )
    args = parse.parse_args()

    print(str(args.dataset) + " " + str(args.model_type))
    main(args)
