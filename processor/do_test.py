from tqdm import tqdm
import torch
from torch.cuda import amp
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import torch
import numpy as np
import json


@torch.no_grad()
def extract_site_attention(model, input_ids, pos_0based, top_k=10, device="cuda:0"):
    """
    从 ESM2 最后一层 attention 中提取：
    突变位点 -> 全部 token 的 attention 分数。

    input_ids: [B, T]
    pos_0based: [B]，原始 residue 位置，0-based
    """

    model.eval()
    input_ids = input_ids.to(device)

    if not torch.is_tensor(pos_0based):
        pos_0based = torch.tensor(pos_0based)

    pos_0based = pos_0based.to(device).long().view(-1)

    # 0-based residue position -> ESM token position
    # 因为 token 0 是 <cls>
    pos_token = pos_0based + 1

    outputs = model.esm2.forward(
        input_ids,
        repr_layers=[33],
        need_head_weights=True
    )

    # attentions: [B, num_layers, H, T, T]
    # 取最后一层: [B, H, T, T]
    last_attn = outputs["attentions"][:, -1]

    B, H, T, _ = last_attn.shape
    batch_idx = torch.arange(B, device=device)

    # 取突变位点作为 query，对所有 token 的 attention
    # [B, H, T]
    site_attn = last_attn[batch_idx, :, pos_token, :]

    # 对 heads 求平均
    # [B, T]
    attn_mean = site_attn.mean(dim=1)

    # 屏蔽特殊 token 和 padding
    attn_mean_masked = attn_mean.clone()

    alphabet = model.esm2_alphabet
    special_ids = [
        alphabet.cls_idx,
        alphabet.eos_idx,
        alphabet.padding_idx,
    ]

    for sid in special_ids:
        if sid is not None:
            attn_mean_masked[input_ids == sid] = -float("inf")

    # 一般也建议排除突变位点自身，否则 top residue 可能总是自己
    attn_mean_masked[batch_idx, pos_token] = -float("inf")

    top_scores, top_indices = torch.topk(
        attn_mean_masked,
        k=top_k,
        dim=1,
        largest=True
    )

    return attn_mean.cpu(), top_indices.cpu(), top_scores.cpu()

def compare_wt_mut_attention(attn_wt, attn_mut, selected_indices, eps=1e-8):
    """
    attn_wt: [B, T]
    attn_mut: [B, T]
    selected_indices: [B, K]
    """

    selected_wt = torch.gather(attn_wt, dim=1, index=selected_indices)
    selected_mut = torch.gather(attn_mut, dim=1, index=selected_indices)

    delta = selected_mut - selected_wt
    ratio = delta / (selected_wt.abs() + eps)
    positive_mask = delta > 0

    return selected_wt, selected_mut, delta, ratio, positive_mask
# def do_test(model, testing_loader, device):
#
#     model.to(device)
#     labels, predictions = do_valid(model, testing_loader, device)
#     print(f'RMSE {np.sqrt(np.mean((labels-predictions)**2))} MAE {np.mean(np.abs(labels - predictions))} Correlation {stats.spearmanr(labels, predictions)}, Pearsonr {pearsonr(labels, predictions)[0]}')
#     return labels, predictions


# def do_valid(model, testing_loader, device):
#     model.eval()
#
#     eval_loss, eval_accuracy = 0, 0
#     nb_eval_examples, nb_eval_steps = 0, 0
#     eval_preds, eval_labels, eval_scores = [], [], []
#
#     with torch.no_grad():
#         for idx, batch in enumerate(testing_loader):
#
#             input_ids1, input_ids2, pos, labels, classes = batch
#             input_ids1 = input_ids1.to(device)
#             input_ids2 = input_ids2.to(device)
#             labels = labels.to(device).reshape(-1, 1)
#             logits = model(input_ids1, input_ids2, pos)
#             loss = torch.nn.functional.mse_loss(logits, labels)
#             eval_loss += loss.item()
#
#             nb_eval_steps += 1
#             nb_eval_examples += labels.size(0)
#
#             eval_labels.extend(labels.cpu().detach())
#             eval_preds.extend(logits.cpu().detach())
#
#     labels = [id.item() for id in eval_labels]
#     predictions = [id.item() for id in eval_preds]
#
#     eval_loss = eval_loss / nb_eval_steps
#     print(f"Validation Loss: {eval_loss}")
#
#     return np.array(labels), np.array(predictions)

from scipy import stats
from scipy.stats import pearsonr
import numpy as np

def do_test(model, testing_loader, device, top_k=10):

    model.to(device)

    labels, predictions, attention_records = do_valid(
        model,
        testing_loader,
        device,
        top_k=top_k
    )

    print(
        f'RMSE {np.sqrt(np.mean((labels - predictions) ** 2))} '
        f'MAE {np.mean(np.abs(labels - predictions))} '
        f'Correlation {stats.spearmanr(labels, predictions)}, '
        f'Pearsonr {pearsonr(labels, predictions)[0]}'
    )

    return labels, predictions, attention_records
def do_valid(model, testing_loader, device, top_k=10):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0

    eval_preds = []
    eval_labels = []
    attention_records = []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            input_ids1, input_ids2, pos, labels, classes = batch

            input_ids1 = input_ids1.to(device)
            input_ids2 = input_ids2.to(device)
            labels = labels.to(device).reshape(-1, 1)

            # 正常预测
            logits = model(input_ids1, input_ids2, pos)

            loss = torch.nn.functional.mse_loss(logits, labels)
            eval_loss += loss.item()
            nb_eval_steps += 1

            eval_labels.extend(labels.cpu().detach())
            eval_preds.extend(logits.cpu().detach())

            # 提取 WT attention
            attn_wt, top_idx_wt, top_score_wt = extract_site_attention(
                model=model,
                input_ids=input_ids1,
                pos_0based=pos,
                top_k=top_k,
                device=device
            )

            # 提取 Mut attention
            attn_mut, top_idx_mut, top_score_mut = extract_site_attention(
                model=model,
                input_ids=input_ids2,
                pos_0based=pos,
                top_k=top_k,
                device=device
            )

            # 用 WT 的 top attention residues 作为比较对象
            selected_indices = top_idx_wt

            selected_wt, selected_mut, delta, ratio, positive_mask = compare_wt_mut_attention(
                attn_wt=attn_wt,
                attn_mut=attn_mut,
                selected_indices=selected_indices
            )

            B = input_ids1.shape[0]

            for b in range(B):
                # token index -> residue index
                # token 0 是 <cls>，所以 residue_0based = token_index - 1
                token_indices = selected_indices[b].tolist()
                residue_indices_0based = [int(x - 1) for x in token_indices]
                residue_indices_1based = [int(x) for x in token_indices]

                record = {
                    "top_token_indices": token_indices,
                    "top_residue_indices_0based": residue_indices_0based,
                    "top_residue_indices_1based": residue_indices_1based,
                    "wt_attention": [float(x) for x in selected_wt[b].tolist()],
                    "mut_attention": [float(x) for x in selected_mut[b].tolist()],
                    "delta_attention": [float(x) for x in delta[b].tolist()],
                    "attention_ratio": [float(x) for x in ratio[b].tolist()],
                    "positive_negative": [
                        "positive" if bool(x) else "negative"
                        for x in positive_mask[b].tolist()
                    ],
                    "positive_ratio": float(positive_mask[b].float().mean().item())
                }

                attention_records.append(record)

    labels = np.array([x.item() for x in eval_labels])
    predictions = np.array([x.item() for x in eval_preds])

    eval_loss = eval_loss / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")

    return labels, predictions, attention_records