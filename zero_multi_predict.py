












import os
import esm
import torch
import random
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import make_model

def set_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_mut_positions_0base(wt_seq: str, mut_seq: str):
    """
    根据 wt_seq 和 mut_seq 逐位比较，返回所有突变位点（0-base）
    """
    if len(wt_seq) != len(mut_seq):
        raise ValueError(f"Sequence length mismatch: {len(wt_seq)} vs {len(mut_seq)}")
    return [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]


def build_single_mutant_seq(wt_seq: str, mut_seq: str, pos: int):
    """
    从多突变序列中抽出某一个位点，构造成单突变序列
    其余位点保持 WT，仅当前位置替换成 mut_seq 对应残基
    """
    seq_list = list(wt_seq)
    seq_list[pos] = mut_seq[pos]
    return "".join(seq_list)


def aggregate_predictions(pred_list, mode="sum_div_pow", alpha=0.5):
    """
    将一个多突变样本拆出来的多个单突变预测进行聚合
    mode:
        - sum
        - mean
        - sum_div_pow: sum(preds) / (k ** alpha)
    """
    k = len(pred_list)
    if k == 0:
        return np.nan

    pred_sum = float(np.sum(pred_list))

    if mode == "sum":
        return pred_sum
    elif mode == "mean":
        return float(np.mean(pred_list))
    elif mode == "sum_div_pow":
        return pred_sum / (k ** alpha)
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


def fit_linear_calibration(y_true, y_pred):
    """
    拟合线性校准: y_true ≈ a * y_pred + b
    注意：只能在验证集上拟合；不要在正式测试集上直接拟合再汇报
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return 1.0, 0.0

    a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)


def apply_linear_calibration(pred, a, b):
    return a * pred + b


class ZeroShotMultiMutationDataset(Dataset):
    """
    输入原始 CSV（列: wt_seq, mut_seq, pH, temp, ddG）
    输出拆开的单突变样本，用于 zero-shot 推断。
    最后按原始样本所在行聚合。
    """

    def __init__(self, csv_path, max_len=1022):
        super().__init__()
        self.csv_path = csv_path
        self.max_len = max_len
        self.df = pd.read_csv(csv_path)

        required_cols = ["wt_seq", "mut_seq", "ddG"]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing required column: {c}")

        _, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = esm2_alphabet.get_batch_converter()

        self.samples = []
        self.skipped_truncation = 0
        self.skipped_no_mut = 0
        self._build_samples()

    def _build_samples(self):
        for row_id, row in tqdm(
            self.df.iterrows(),
            total=len(self.df),
            desc="Building single-mutation samples"
        ):
            wt_seq = str(row["wt_seq"])
            mut_seq = str(row["mut_seq"])
            ddg = float(row["ddG"])

            diff_positions = find_mut_positions_0base(wt_seq, mut_seq)

            if len(diff_positions) == 0:
                self.skipped_no_mut += 1
                continue

            valid_positions = []
            for pos in diff_positions:
                # 如果突变位点超出截断长度，模型实际上看不到这个位点
                if pos >= self.max_len:
                    self.skipped_truncation += 1
                    continue
                valid_positions.append(pos)

            if len(valid_positions) == 0:
                continue

            for pos in valid_positions:
                single_mut_seq = build_single_mutant_seq(wt_seq, mut_seq, pos)

                self.samples.append(
                    {
                        "row_id": row_id,
                        "wt_seq": wt_seq[:self.max_len],
                        "single_mut_seq": single_mut_seq[:self.max_len],
                        "position_0base": pos,
                        # 模型输入位置需要 +1，因为 ESM token 前面有 BOS token
                        "position_for_model": pos + 1,
                        "true_ddg": ddg,
                        "num_mutations_total": len(diff_positions),     # 原始总突变数
                        "num_mutations_used": len(valid_positions),     # 实际进入模型的突变数
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        _, _, wt_tokens = self.batch_converter([("none", item["wt_seq"])])
        _, _, mut_tokens = self.batch_converter([("none", item["single_mut_seq"])])

        return {
            "wild_tokens": wt_tokens.squeeze(0),
            "mutant_tokens": mut_tokens.squeeze(0),
            "position": torch.tensor(item["position_for_model"], dtype=torch.long),
            "row_id": torch.tensor(item["row_id"], dtype=torch.long),
            "true_ddg": torch.tensor(item["true_ddg"], dtype=torch.float),
            "num_mutations_total": torch.tensor(item["num_mutations_total"], dtype=torch.long),
            "num_mutations_used": torch.tensor(item["num_mutations_used"], dtype=torch.long),
        }


@torch.no_grad()
def collect_single_predictions(model, testing_loader, device):
    """
    收集每个原始样本(row_id)对应的所有单突变预测
    返回:
        row_pred_list: {row_id: [pred1, pred2, ...]}
        row_num_mut_total: {row_id: 原始总突变数}
        row_num_mut_used:  {row_id: 实际用于预测的突变数}
    """
    model.eval()
    model.to(device)

    row_pred_list = {}
    row_num_mut_total = {}
    row_num_mut_used = {}

    for batch in tqdm(testing_loader, desc="Zero-shot inference", total=len(testing_loader)):
        wild_tokens = batch["wild_tokens"].to(device)
        mutant_tokens = batch["mutant_tokens"].to(device)
        positions = batch["position"].to(device)

        row_ids = batch["row_id"].cpu().numpy()
        num_mut_total = batch["num_mutations_total"].cpu().numpy()
        num_mut_used = batch["num_mutations_used"].cpu().numpy()

        outputs = model(wild_tokens, mutant_tokens, positions, is_training=False)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        preds = outputs.detach().float().view(-1).cpu().numpy()

        for i in range(len(preds)):
            rid = int(row_ids[i])
            pred_i = float(preds[i])

            if rid not in row_pred_list:
                row_pred_list[rid] = []
                row_num_mut_total[rid] = int(num_mut_total[i])
                row_num_mut_used[rid] = int(num_mut_used[i])

            row_pred_list[rid].append(pred_i)

    return row_pred_list, row_num_mut_total, row_num_mut_used


def build_final_predictions(row_pred_list, mode="sum_div_pow", alpha=0.5):
    pred_dict = {}
    for rid, pred_list in row_pred_list.items():
        pred_dict[rid] = aggregate_predictions(pred_list, mode=mode, alpha=alpha)
    return pred_dict


def compute_metrics(eval_df, group_name="All"):
    """
    对给定子集计算指标
    """
    if len(eval_df) == 0:
        print(f"{group_name}: no valid samples")
        return {
            "Group": group_name,
            "Num_samples": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Spearman": np.nan,
            "Pearson": np.nan,
        }

    labels = eval_df["ddG"].astype(float).to_numpy()
    predictions = eval_df["our"].astype(float).to_numpy()

    mse = np.mean((labels - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(labels - predictions))

    try:
        spearman_corr = stats.spearmanr(labels, predictions).correlation
    except Exception:
        spearman_corr = np.nan

    try:
        pearson_corr = pearsonr(labels, predictions)[0]
    except Exception:
        pearson_corr = np.nan

    print(group_name)
    print(f"Num evaluated samples: {len(eval_df)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Spearman: {spearman_corr:.6f}" if not np.isnan(spearman_corr) else "Spearman: nan")
    print(
        f"Pearson correlation coefficient: {pearson_corr:.6f}"
        if not np.isnan(pearson_corr) else
        "Pearson correlation coefficient: nan"
    )
    print("-" * 50)

    return {
        "Group": group_name,
        "Num_samples": len(eval_df),
        "RMSE": rmse,
        "MAE": mae,
        "Spearman": spearman_corr,
        "Pearson": pearson_corr,
    }


def evaluate_and_save(
    csv_path,
    pred_dict,
    row_num_mut_total,
    row_num_mut_used,
    do_calibration=False,
    separate_calibration_by_group=False,
):
    """
    将预测结果写回原 CSV 的 our 列，并按照
    All / Double / Triple or higher-order 分别计算指标

    没有验证集时，建议:
        do_calibration = False
    """
    df = pd.read_csv(csv_path)

    # 重新统计每条原始样本的总突变位点数
    mut_counts_total = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Counting mutation numbers"):
        wt_seq = str(row["wt_seq"])
        mut_seq = str(row["mut_seq"])
        diff_positions = find_mut_positions_0base(wt_seq, mut_seq)
        mut_counts_total.append(len(diff_positions))
    df["num_mutations_total"] = mut_counts_total

    # 写入实际使用的突变位点数
    df["num_mutations_used"] = 0
    for rid, k in row_num_mut_used.items():
        if 0 <= rid < len(df):
            df.at[rid, "num_mutations_used"] = k

    # 写入原始预测
    df["our_raw"] = np.nan
    for rid, pred in tqdm(pred_dict.items(), desc="Writing raw predictions"):
        if 0 <= rid < len(df):
            df.at[rid, "our_raw"] = pred

    # 是否校准
    if do_calibration:
        # 强提醒：没有验证集别这么干
        print("\n[Warning] You enabled calibration without a separate validation set.")
        print("[Warning] Reported metrics may be optimistic due to data leakage.\n")

        eval_df_raw = df.dropna(subset=["ddG", "our_raw"]).copy()

        if separate_calibration_by_group:
            df["our"] = df["our_raw"]

            double_df = eval_df_raw[eval_df_raw["num_mutations_total"] == 2].copy()
            if len(double_df) >= 2:
                a2, b2 = fit_linear_calibration(double_df["ddG"], double_df["our_raw"])
                idx = df["num_mutations_total"] == 2
                df.loc[idx, "our"] = df.loc[idx, "our_raw"].apply(
                    lambda x: apply_linear_calibration(x, a2, b2) if pd.notna(x) else np.nan
                )

            triple_df = eval_df_raw[eval_df_raw["num_mutations_total"] >= 3].copy()
            if len(triple_df) >= 2:
                a3, b3 = fit_linear_calibration(triple_df["ddG"], triple_df["our_raw"])
                idx = df["num_mutations_total"] >= 3
                df.loc[idx, "our"] = df.loc[idx, "our_raw"].apply(
                    lambda x: apply_linear_calibration(x, a3, b3) if pd.notna(x) else np.nan
                )
        else:
            a, b = fit_linear_calibration(eval_df_raw["ddG"], eval_df_raw["our_raw"])
            df["our"] = df["our_raw"].apply(
                lambda x: apply_linear_calibration(x, a, b) if pd.notna(x) else np.nan
            )
    else:
        df["our"] = df["our_raw"]

    # 保存回原文件
    df.to_csv(csv_path, index=False)

    # 评估
    eval_df = df.dropna(subset=["ddG", "our"]).copy()

    all_df = eval_df.copy()
    double_df = eval_df[eval_df["num_mutations_total"] == 2].copy()
    triple_plus_df = eval_df[eval_df["num_mutations_total"] >= 3].copy()

    print("\nEvaluation results")
    print("=" * 50)
    metrics_all = compute_metrics(all_df, "All")
    metrics_double = compute_metrics(double_df, "Double")
    metrics_triple_plus = compute_metrics(triple_plus_df, "Triple or higher-order")

    # metrics_df = pd.DataFrame([metrics_all, metrics_double, metrics_triple_plus])
    # metrics_save_path = os.path.splitext(csv_path)[0] + "_metrics_by_mutnum.csv"
    # metrics_df.to_csv(metrics_save_path, index=False)

    # print(f"Saved predictions to columns 'our_raw' and 'our' in: {csv_path}")
    # print(f"Saved grouped metrics to: {metrics_save_path}")


def main():
    seed = 1024
    set_seed(seed)

    batch_size = 1
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoint/best_model_rmse_our.pt"#best_model_rmse_our.pt
    test_csv = "ddgdata/M1261_multipos.csv"

    # =======================
    # 可调参数
    # =======================
    aggregation_mode = "sum_div_pow"   # "sum", "mean", "sum_div_pow"
    alpha = 0.7                       # 仅在 sum_div_pow 下生效
    do_calibration = False             # 没有验证集时建议 False
    separate_calibration_by_group = False
    # =======================

    print("Loading dataset...")
    testing_dataset = ZeroShotMultiMutationDataset(test_csv, max_len=1022)
    print(f"Number of split single-mutation samples: {len(testing_dataset)}")
    print(f"Skipped no-mutation rows: {testing_dataset.skipped_no_mut}")
    print(f"Skipped truncated mutation sites: {testing_dataset.skipped_truncation}")

    testing_loader = DataLoader(
        testing_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Building model...")
    model = make_model("GlobalLocalDualBranch")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    print("Running prediction...")
    row_pred_list, row_num_mut_total, row_num_mut_used = collect_single_predictions(
        model=model,
        testing_loader=testing_loader,
        device=device,
    )

    pred_dict = build_final_predictions(
        row_pred_list=row_pred_list,
        mode=aggregation_mode,
        alpha=alpha,
    )

    print("Saving results and evaluating...")
    evaluate_and_save(
        csv_path=test_csv,
        pred_dict=pred_dict,
        row_num_mut_total=row_num_mut_total,
        row_num_mut_used=row_num_mut_used,
        do_calibration=do_calibration,
        separate_calibration_by_group=separate_calibration_by_group,
    )


if __name__ == "__main__":
    main()


