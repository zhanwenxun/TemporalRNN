import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_util import (
    count_unique_elements,
    extract_features,
    score_margin_group_sampling,
)
from temporalrnn_utils import load_test_prediction_info, load_train_prediction_info, train_error_detector


def parse_args():
    parser = argparse.ArgumentParser(
        "Run TemporalRNN experiment."
    )
    parser.add_argument("-dataset", choices=["mnist", "snips", "fashion", "agnews", "svhn"])
    parser.add_argument("-model", choices=["lstm", "blstm", "gru"])
    parser.add_argument("-prostart", choices=["0", "3", "5", "10", "15", "20"])
    parser.add_argument("-labelstart", choices=["0", "3", "5", "10", "15", "20"])
    parser.add_argument("--file-count", type=int, default=30)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--output-dir", default="./results/temporalrnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-missing", action="store_true")

    # Parameters for the TemporalRNN score-margin group sampling step.
    parser.add_argument("--score-multiplier", type=float, default=2.0)
    parser.add_argument("--margin-multiplier", type=float, default=2.0)
    parser.add_argument("--min-group-quota", type=int, default=1)

    parser.add_argument(
        "--aggregate-existing",
        action="store_true",
        help="Only aggregate existing detail_*.csv files in output-dir.",
    )
    args = parser.parse_args()

    if not args.aggregate_existing:
        missing = [
            name for name in ("dataset", "model", "prostart", "labelstart")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"missing required arguments unless --aggregate-existing is used: {missing}")

    return args


def evaluate_selection(selected_indices, test_errors, fault_types):
    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_count = len(selected_indices)
    selected_errors = test_errors[selected_indices] if selected_count else np.array([])
    selected_failed = int(np.sum(selected_errors))
    total_failed = int(np.sum(test_errors))

    selected_fault_types = fault_types[selected_indices] if selected_count else []
    diversity = int(count_unique_elements(selected_fault_types))
    total_diversity = int(count_unique_elements(fault_types))

    precision = selected_failed / selected_count if selected_count else 0.0
    recall = selected_failed / total_failed if total_failed else 0.0
    diversity_recall = diversity / total_diversity if total_diversity else 0.0
    denom = precision + diversity_recall
    combined_f1 = 2 * precision * diversity_recall / denom if denom else 0.0

    return {
        "precision": precision,
        "diversity": diversity,
        "selected_failed": selected_failed,
        "selected_count": selected_count,
        "total_failed": total_failed,
        "total_diversity": total_diversity,
        "recall_optional": recall,
        "diversity_recall_optional": diversity_recall,
        "combined_f1_optional": combined_f1,
    }


def select_temporalrnn(scores, budget, test_data, args):
    return score_margin_group_sampling(
        scores=scores,
        final_pro=test_data["pros"][:, -1, :],
        n=budget,
        score_multiplier=args.score_multiplier,
        margin_multiplier=args.margin_multiplier,
        min_group_quota=args.min_group_quota,
        random_seed=args.seed,
    )


def run_experiment(args):
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Budgets: {args.budgets}")
    print(
        "TemporalRNN params: "
        f"score_multiplier={args.score_multiplier}, "
        f"margin_multiplier={args.margin_multiplier}, "
        f"min_group_quota={args.min_group_quota}"
    )

    train_data = load_train_prediction_info(args.dataset, args.model)
    cat_num = train_data["pros"][0][0].shape[0]
    scorer, _ = train_error_detector(train_data, int(args.prostart), int(args.labelstart))

    detail_rows = []
    for file_index in range(args.file_count):
        try:
            test_data = load_test_prediction_info(args.dataset, args.model, file_index)
        except FileNotFoundError:
            if args.skip_missing:
                print(f"Skip missing candidate file {file_index}.")
                continue
            raise

        print(f"Evaluate candidate file {file_index}")
        test_features = extract_features(
            test_data["pros"],
            test_data["labels"],
            test_data["infos"],
            test_data["lstm"],
            int(args.prostart),
            int(args.labelstart),
        )
        test_errors = (test_data["right"] == 0).astype(int)
        scores = scorer.predict_proba(Pool(test_features))[:, 1]
        test_num = test_data["labels"].shape[0]

        for budget_ratio in args.budgets:
            budget_size = int(test_num * budget_ratio)
            selected_indices = select_temporalrnn(scores, budget_size, test_data, args)

            # 存储所选测试用例索引
            selected_index_dir = Path("./selected_index") / f"{args.dataset}_{args.model}"
            selected_index_dir.mkdir(parents=True, exist_ok=True)
            selected_index_path = selected_index_dir / f"file{file_index}_"
            selected_index = np.zeros(test_num, dtype=int)
            selected_index[selected_indices] = 1
            np.save(str(selected_index_path) + str(int(budget_ratio*100)) + "_temporalrnn_selected", selected_index)

            metrics = evaluate_selection(
                selected_indices,
                test_errors,
                test_data["fault_types"],
            )
            detail_rows.append({
                "dataset": args.dataset,
                "model": args.model,
                "file_index": file_index,
                "budget_ratio": budget_ratio,
                "budget_size": budget_size,
                "strategy": "temporalrnn",
                **metrics,
            })

    detail_df = pd.DataFrame(detail_rows)
    detail_path = output_dir / f"detail_{args.dataset}_{args.model}.csv"
    detail_df.to_csv(detail_path, index=False)
    write_aggregate_outputs(detail_df, output_dir, args.dataset, args.model)
    print(f"Saved detail rows: {detail_path}")
    print("\n")


def build_summary(detail_df):
    return (
        detail_df
        .groupby(["dataset", "model", "budget_ratio", "strategy"], as_index=False)
        .agg(
            avg_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            avg_diversity=("diversity", "mean"),
            std_diversity=("diversity", "std"),
            avg_selected_failed=("selected_failed", "mean"),
            avg_selected_count=("selected_count", "mean"),
            avg_total_failed=("total_failed", "mean"),
            avg_total_diversity=("total_diversity", "mean"),
            avg_recall_optional=("recall_optional", "mean"),
            avg_diversity_recall_optional=("diversity_recall_optional", "mean"),
            avg_combined_f1_optional=("combined_f1_optional", "mean"),
        )
    )


def build_delta(summary_df, baseline_strategy="temporalrnn"):
    baseline = summary_df[summary_df["strategy"] == baseline_strategy][[
        "dataset",
        "model",
        "budget_ratio",
        "avg_precision",
        "avg_diversity",
    ]].rename(columns={
        "avg_precision": "baseline_precision",
        "avg_diversity": "baseline_diversity",
    })

    delta = summary_df.merge(
        baseline,
        on=["dataset", "model", "budget_ratio"],
        how="left",
    )
    delta["baseline_strategy"] = baseline_strategy
    delta["delta_precision"] = delta["avg_precision"] - delta["baseline_precision"]
    delta["delta_diversity"] = delta["avg_diversity"] - delta["baseline_diversity"]
    delta["relative_precision_pct"] = np.where(
        delta["baseline_precision"] != 0,
        100 * delta["delta_precision"] / delta["baseline_precision"],
        np.nan,
    )
    delta["relative_diversity_pct"] = np.where(
        delta["baseline_diversity"] != 0,
        100 * delta["delta_diversity"] / delta["baseline_diversity"],
        np.nan,
    )
    delta["precision_drop_flag"] = delta["delta_precision"] < -0.01
    delta["diversity_gain_flag"] = delta["delta_diversity"] > 0
    return delta


def build_paper_table(summary_df):
    table = summary_df.copy()
    table["precision_diversity"] = table.apply(
        lambda row: f"{row['avg_precision']:.4f} / {row['avg_diversity']:.2f}",
        axis=1,
    )

    pivot = table.pivot_table(
        index=["dataset", "model", "budget_ratio"],
        columns="strategy",
        values="precision_diversity",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    metric_pivot = table.pivot_table(
        index=["dataset", "model", "budget_ratio"],
        columns="strategy",
        values=["avg_precision", "avg_diversity"],
        aggfunc="first",
    ).reset_index()
    metric_pivot.columns = [
        "_".join([str(part) for part in col if part])
        if isinstance(col, tuple) else col
        for col in metric_pivot.columns
    ]
    return pivot, metric_pivot


def build_budget_average(summary_df):
    return (
        summary_df
        .groupby(["budget_ratio", "strategy"], as_index=False)
        .agg(
            avg_precision=("avg_precision", "mean"),
            avg_diversity=("avg_diversity", "mean"),
            avg_combined_f1_optional=("avg_combined_f1_optional", "mean"),
        )
    )


def write_aggregate_outputs(detail_df, output_dir, dataset=None, model=None):
    summary = build_summary(detail_df)
    delta = build_delta(summary)
    paper_table, paper_metric_table = build_paper_table(summary)
    budget_average = build_budget_average(summary)

    suffix = f"_{dataset}_{model}" if dataset and model else "_all"
    output_dir = Path(output_dir)
    summary.to_csv(output_dir / f"summary{suffix}.csv", index=False)
    delta.to_csv(output_dir / f"delta{suffix}.csv", index=False)
    budget_average.to_csv(output_dir / f"budget_average{suffix}.csv", index=False)
    paper_table.to_csv(output_dir / f"paper_table{suffix}.csv", index=False)
    paper_metric_table.to_csv(output_dir / f"paper_metric_table{suffix}.csv", index=False)


def aggregate_existing(output_dir):
    output_dir = Path(output_dir)
    detail_files = sorted(output_dir.glob("detail_*.csv"))
    if not detail_files:
        raise FileNotFoundError(f"No detail_*.csv files found in {output_dir}")

    detail_df = pd.concat(
        [pd.read_csv(path) for path in detail_files],
        ignore_index=True,
    )
    combined_detail_path = output_dir / "detail_all.csv"
    detail_df.to_csv(combined_detail_path, index=False)
    write_aggregate_outputs(detail_df, output_dir)
    print(f"Aggregated {len(detail_files)} detail files into {combined_detail_path}")


def main():
    args = parse_args()
    if args.aggregate_existing:
        aggregate_existing(args.output_dir)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
