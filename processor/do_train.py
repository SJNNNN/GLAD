
from scipy import stats
from scipy.stats import pearsonr
import os
import torch
import numpy as np
from torch.cuda import amp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
# from tqdm import tqdm
# from scipy import stats
# from scipy.stats import pearsonr
# import torch
# import numpy as np
# from torch.cuda import amp


def do_train(model, optimizer, scheduler, training_loader, testing_loader, device, epochs):
    scaler = amp.GradScaler()

    # print(pocket_prob_dict)
    best_metric_spearman = float('-inf')
    best_metric_rmse = float('inf')
    best_epoch = 0
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        model.train()

        tr_loss = 0.0
        nb_tr_steps = 0

        # ===== 训练进度条 =====
        train_pbar = tqdm(
            training_loader,
            desc=f"Epoch [{epoch + 1}/{epochs}] Training",
            leave=False
        )

        for batch in train_pbar:
            optimizer.zero_grad()

            input_ids1, input_ids2, pos, labels, classes = batch
            input_ids1 = input_ids1.to(device)
            input_ids2 = input_ids2.to(device)

            labels = labels.to(device).reshape(-1, 1)



            with amp.autocast(enabled=True):
                # logitis_output = model(input_ids1, input_ids2, pos, delta_emb=delta, is_training=True)
                logitis_output = model(input_ids1, input_ids2, pos, is_training=True)
                # logitis_output = model(input_ids1, input_ids2, pos, wt_site_prob=wt_site_prob, is_training=True)
                if len(logitis_output) == 3:
                    logits_1, logits_2, logits_mean = logitis_output
                    loss = (
                            torch.nn.functional.mse_loss(logits_1, labels)
                            + torch.nn.functional.mse_loss(logits_2, labels)
                            + torch.nn.functional.mse_loss(logits_mean, labels)
                    )
                else:
                    logits_pred = logitis_output[0]
                    loss = torch.nn.functional.mse_loss(logits_pred, labels)

            tr_loss += loss.item()
            nb_tr_steps += 1

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()

            # ===== 实时显示 loss =====
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # scheduler.step(epoch)
        scheduler.step(epoch)
        epoch_loss = tr_loss / nb_tr_steps
        print(f"\nEpoch {epoch + 1} Training Loss: {epoch_loss:.4f}")

        # ===== 验证 =====
        labels, predictions = do_valid(model, testing_loader, device)

        print(
            f"RMSE {np.sqrt(np.mean((labels - predictions) ** 2)):.4f} | "
            f"MAE {np.mean(np.abs(labels - predictions)):.4f} | "
            f"Spearman {stats.spearmanr(labels, predictions)[0]:.4f} | "
            f"Pearson {pearsonr(labels, predictions)[0]:.4f}"
        )
        rmse = np.sqrt(np.mean((labels - predictions) ** 2))
        mae = np.mean(np.abs(labels - predictions))
        spearman = stats.spearmanr(labels, predictions)[0]
        pearson = pearsonr(labels, predictions)[0]
        if spearman > best_metric_spearman:
            best_metric_spearman = spearman
            torch.save(model.state_dict(), "checkpoint/best_model_spearman_1.pt")#1 ptk_r

        # 保存模型：RMSE最小
        if rmse < best_metric_rmse:
            best_metric_rmse = rmse
            torch.save(model.state_dict(), "checkpoint/best_model_rmse_1.pt")#ptk_r

        # ===== 早停检查 =====
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}, best epoch was {best_epoch}")
            break
    print(
        f"\nTraining finished. Best model from epoch {best_epoch} with RMSE {best_metric_rmse:.4f} and Spearman {best_metric_spearman:.4f}")

    # torch.save(model.state_dict(), "checkpoint/training1.pt")

# import torch
#
# def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#     """
#     返回 1 - PearsonCorr(pred, target)
#     pred/target: shape [B,1] or [B]
#     """
#     pred = pred.view(-1)
#     target = target.view(-1)
#
#     pred = pred - pred.mean()
#     target = target - target.mean()
#
#     denom = (pred.std(unbiased=False) * target.std(unbiased=False)).clamp_min(eps)
#     corr = (pred * target).mean() / denom
#     return 1.0 - corr
#
# ######our model
# def do_train(model, optimizer, scheduler, training_loader, testing_loader, device, epochs):
#     scaler = amp.GradScaler()
#
#     # ===== 新增：跟踪最佳模型 =====
#     best_metric_spearman = float('-inf')  # Spearman 越大越好
#     best_metric_rmse = float('inf')  # RMSE 越小越好
#     best_epoch = 0
#     patience = 10  # 早停耐心值
#     no_improve = 0
#
#     for epoch in range(epochs):
#         model.train()
#         tr_loss = 0.0
#
#         nb_tr_examples, nb_tr_steps = 0, 0
#
#         train_pbar = tqdm(
#             training_loader,
#             desc=f"Epoch [{epoch + 1}/{epochs}] Training",
#             leave=False
#         )
#
#         for batch in train_pbar:
#             optimizer.zero_grad()
#
#             input_ids1, input_ids2, pos, labels, classes = batch
#             input_ids1 = input_ids1.to(device)
#             input_ids2 = input_ids2.to(device)
#             labels = labels.to(device).reshape(-1, 1)
#
#             # with amp.autocast(enabled=True):
#             #     logitis_output = model(input_ids1, input_ids2, pos, is_training=True)
#             #     if len(logitis_output) == 3:
#             #         logits_1, logits_2, logits_mean = logitis_output
#             #         loss = (
#             #                 torch.nn.functional.mse_loss(logits_1, labels)
#             #                 + torch.nn.functional.mse_loss(logits_2, labels)
#             #                 + torch.nn.functional.mse_loss(logits_mean, labels)
#             #         )
#             #     else:
#             #         logits_pred = logitis_output[0]
#             #         loss = torch.nn.functional.mse_loss(logits_pred, labels)
#             # tr_loss += loss.item()
#             lambda_p = 0.1  # 轻量 Pearson 项权重，建议先 0.05/0.1/0.2 做消融
#
#             with amp.autocast(enabled=True):
#                 logitis_output = model(input_ids1, input_ids2, pos, is_training=True)
#
#                 if len(logitis_output) == 3:
#                     logits_1, logits_2, logits_mean = logitis_output
#
#                     mse = (
#                             torch.nn.functional.mse_loss(logits_1, labels)
#                             + torch.nn.functional.mse_loss(logits_2, labels)
#                             + torch.nn.functional.mse_loss(logits_mean, labels)
#                     )
#
#                     # Pearson：只加在 mean 上最稳（你也可以三项都加，但更“激进”）
#
#                     p_mean = pearson_loss(logits_mean, labels)
#
#                     loss = mse +lambda_p * p_mean
#
#                 else:
#                     logits_pred = logitis_output[0]
#                     mse = torch.nn.functional.mse_loss(logits_pred, labels)
#                     #p = pearson_loss(logits_pred, labels)
#                     loss = mse #+ lambda_p * p
#
#             tr_loss += loss.item()
#             nb_tr_steps += 1
#             nb_tr_examples += labels.size(0)
#
#             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.1)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scale = scaler.get_scale()
#             scaler.update()
#             train_pbar.set_postfix(loss=f"{loss.item():.4f}")
#
#         scheduler.step(epoch)
#         epoch_loss = tr_loss / nb_tr_steps
#         print(f"\nEpoch {epoch + 1} Training Loss: {epoch_loss:.4f}")
#
#         # ===== 验证 =====
#         labels, predictions = do_valid(model, testing_loader, device)
#
#         # 计算指标
#         rmse = np.sqrt(np.mean((labels - predictions) ** 2))
#         mae = np.mean(np.abs(labels - predictions))
#         spearman = stats.spearmanr(labels, predictions)[0]
#         pearson = pearsonr(labels, predictions)[0]
#
#         print(
#             f"RMSE {rmse:.4f} | MAE {mae:.4f} | "
#             f"Spearman {spearman:.4f} | Pearson {pearson:.4f}"
#         )
#
#         # 保存模型：Spearman最大
#         if spearman > best_metric_spearman:
#             best_metric_spearman = spearman
#             torch.save(model.state_dict(), "checkpoint/best_model_spearman2.pt")
#
#         # 保存模型：RMSE最小
#         if rmse < best_metric_rmse:
#             best_metric_rmse = rmse
#             torch.save(model.state_dict(), "checkpoint/best_model_rmse2.pt")
#
#         # ===== 早停检查 =====
#         if no_improve >= patience:
#             print(f"Early stopping at epoch {epoch + 1}, best epoch was {best_epoch}")
#             break
#
#     print(f"\nTraining finished. Best model from epoch {best_epoch} with RMSE {best_metric_rmse:.4f} and Spearman {best_metric_spearman:.4f}")
#
def do_valid(model, testing_loader, device):
    model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0
    eval_preds, eval_labels = [], []

    # ===== 验证进度条 =====
    valid_pbar = tqdm(
        testing_loader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():
        for batch in valid_pbar:
            input_ids1, input_ids2, pos, labels, classes = batch

            input_ids1 = input_ids1.to(device)
            input_ids2 = input_ids2.to(device)
            labels = labels.to(device).reshape(-1, 1)

            logits = model(input_ids1, input_ids2, pos)
            loss = torch.nn.functional.mse_loss(logits, labels)

            eval_loss += loss.item()
            nb_eval_steps += 1

            eval_labels.extend(labels.cpu().numpy())
            eval_preds.extend(logits.cpu().numpy())

            valid_pbar.set_postfix(loss=f"{loss.item():.4f}")

    eval_loss /= nb_eval_steps
    print(f"Validation Loss: {eval_loss:.4f}")

    labels = np.array(eval_labels).ravel()
    predictions = np.array(eval_preds).ravel()

    return labels, predictions





#####伪标签加真实标签
#
# def weighted_mse(pred: torch.Tensor, target: torch.Tensor, re_label: torch.Tensor,
#                  w_real: float = 0.8, w_pseudo: float = 0.2) -> torch.Tensor:
#     """
#     pred/target: [B,1]
#     re_label: [B] or [B,1]  (真实=1，伪=0)
#     """
#     if re_label.dim() == 1:
#         re_label = re_label.view(-1, 1)
#     re_label = re_label.float()
#
#     weights = torch.where(re_label > 0.5,
#                           torch.tensor(w_real, device=re_label.device),
#                           torch.tensor(w_pseudo, device=re_label.device)).view(-1, 1)
#
#     mse = torch.nn.functional.mse_loss(pred, target, reduction="none")  # [B,1]
#     # 加权平均（避免 batch 内比例变化导致 loss 规模漂）
#     loss = (weights * mse).sum() / (weights.sum() + 1e-12)
#     return loss
#
#
# # =========================
# # Train / Valid
# # =========================
#
# def do_valid(model, testing_loader, device, eval_only_real: bool = True):
#     model.eval()
#
#     eval_loss = 0.0
#     nb_eval_steps = 0
#     eval_preds, eval_labels, eval_relabels = [], [], []
#
#     pbar = tqdm(testing_loader, desc="Validation", leave=False)
#
#     with torch.no_grad():
#         for batch in pbar:
#             input_ids1, input_ids2, pos, labels, classes, re_label = batch
#             input_ids1 = input_ids1.to(device)
#             input_ids2 = input_ids2.to(device)
#             pos = pos.to(device)
#             labels = labels.to(device).reshape(-1, 1)
#             re_label = re_label.to(device)
#
#             logits = model(input_ids1, input_ids2, pos)  # 你原来的 eval forward
#             # 这里 validation loss 也按同样权重算（你也可以改成纯 mse）
#             loss = weighted_mse(logits, labels, re_label, w_real=0.8, w_pseudo=0.2)
#
#             eval_loss += loss.item()
#             nb_eval_steps += 1
#
#             eval_labels.extend(labels.detach().cpu().numpy().ravel().tolist())
#             eval_preds.extend(logits.detach().cpu().numpy().ravel().tolist())
#             eval_relabels.extend(re_label.detach().cpu().numpy().ravel().tolist())
#
#             pbar.set_postfix(loss=f"{loss.item():.4f}")
#
#     eval_loss /= max(1, nb_eval_steps)
#     print(f"Validation Loss: {eval_loss:.4f}")
#
#     labels = np.array(eval_labels)
#     preds = np.array(eval_preds)
#     relab = np.array(eval_relabels)
#
#     # 指标默认只在真实样本上算（更靠谱）
#     if eval_only_real and np.any(relab == 1):
#         mask = (relab == 1)
#         labels = labels[mask]
#         preds = preds[mask]
#
#     return labels, preds
#
#
# def do_train(model, optimizer, scheduler, training_loader, testing_loader, device, epochs,
#              w_real: float = 0.9, w_pseudo: float = 0.1,
#              eval_only_real: bool = True):
#
#     scaler = amp.GradScaler()
#
#     best_metric_spearman = float('-inf')
#     best_metric_rmse = float('inf')
#     best_epoch = 0
#     patience = 10
#     no_improve = 0
#
#     for epoch in range(epochs):
#         model.train()
#         tr_loss = 0.0
#         nb_tr_steps = 0
#
#         train_pbar = tqdm(training_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Training", leave=False)
#
#         for batch in train_pbar:
#             optimizer.zero_grad(set_to_none=True)
#
#             input_ids1, input_ids2, pos, labels, classes, re_label = batch
#             input_ids1 = input_ids1.to(device, non_blocking=True)
#             input_ids2 = input_ids2.to(device, non_blocking=True)
#             pos = pos.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True).reshape(-1, 1)
#             re_label = re_label.to(device, non_blocking=True)
#
#             with amp.autocast(enabled=True):
#                 out = model(input_ids1, input_ids2, pos, is_training=True)
#
#                 if isinstance(out, (list, tuple)) and len(out) == 3:
#                     logits_1, logits_2, logits_mean = out
#
#                     loss1 = weighted_mse(logits_1, labels, re_label, w_real=w_real, w_pseudo=w_pseudo)
#                     loss2 = weighted_mse(logits_2, labels, re_label, w_real=w_real, w_pseudo=w_pseudo)
#                     lossm = weighted_mse(logits_mean, labels, re_label, w_real=w_real, w_pseudo=w_pseudo)
#                     loss = loss1 + loss2 + lossm
#                 else:
#                     # 兼容你某些模型返回 (logits,) 或直接 logits
#                     logits_pred = out[0] if isinstance(out, (list, tuple)) else out
#                     loss = weighted_mse(logits_pred, labels, re_label, w_real=w_real, w_pseudo=w_pseudo)
#
#             tr_loss += loss.item()
#             nb_tr_steps += 1
#
#             scaler.scale(loss).backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
#             scaler.step(optimizer)
#             scaler.update()
#
#             train_pbar.set_postfix(loss=f"{loss.item():.4f}")
#
#         # scheduler（你原来是 scheduler.step(epoch)，不同 scheduler 可能要 scheduler.step()）
#         try:
#             scheduler.step(epoch)
#         except TypeError:
#             scheduler.step()
#
#         epoch_loss = tr_loss / max(1, nb_tr_steps)
#         print(f"\nEpoch {epoch + 1} Training Loss: {epoch_loss:.4f}")
#
#         # ===== 验证 =====
#         labels, predictions = do_valid(model, testing_loader, device, eval_only_real=eval_only_real)
#
#         rmse = np.sqrt(np.mean((labels - predictions) ** 2))
#         mae = np.mean(np.abs(labels - predictions))
#         spearman = stats.spearmanr(labels, predictions)[0]
#         pearson = pearsonr(labels, predictions)[0]
#
#         print(f"RMSE {rmse:.4f} | MAE {mae:.4f} | Spearman {spearman:.4f} | Pearson {pearson:.4f}")
#
#         improved = False
#
#         if spearman > best_metric_spearman:
#             best_metric_spearman = spearman
#             torch.save(model.state_dict(), "checkpoint/best_model_spearman1.pt")
#             improved = True
#
#         if rmse < best_metric_rmse:
#             best_metric_rmse = rmse
#             best_epoch = epoch + 1
#             torch.save(model.state_dict(), "checkpoint/best_model_rmse1.pt")
#             improved = True
#
#         if improved:
#             no_improve = 0
#         else:
#             no_improve += 1
#
#         if no_improve >= patience:
#             print(f"Early stopping at epoch {epoch + 1}, best epoch was {best_epoch}")
#             break
#
#     print(f"\nTraining finished. Best epoch {best_epoch} | RMSE {best_metric_rmse:.4f} | Spearman {best_metric_spearman:.4f}")
#


#对比学习

# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, features1, features2, labels):
#         """
#         features1: [B, D] (batch size, feature dimension)
#         features2: [B, D]
#         labels: [B] (1 for positive pair, 0 for negative pair)
#         """
#         # 计算欧几里得距离
#         euclidean_distance = F.pairwise_distance(features1, features2, p=2)
#
#         # 对比损失
#         loss = torch.mean((1 - labels) * torch.pow(euclidean_distance, 2) +
#                           (labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#         return loss
# def do_train(model, optimizer, scheduler, training_loader, testing_loader, device, epochs):
#     scaler = amp.GradScaler()
#
#     # ===== 跟踪最佳模型 =====
#     best_spearman = float('-inf')   # Spearman 越大越好
#     best_rmse = float('inf')        # RMSE 越小越好
#     best_epoch_spearman = 0
#     best_epoch_rmse = 0
#
#     patience = 10
#     no_improve = 0
#
#     contrastive_loss_fn = ContrastiveLoss(margin=1.0)
#
#     for epoch in range(epochs):
#         model.train()
#         tr_loss = 0.0
#         nb_tr_steps = 0
#
#         train_pbar = tqdm(
#             training_loader,
#             desc=f"Epoch [{epoch + 1}/{epochs}] Training",
#             leave=False
#         )
#
#         for batch in train_pbar:
#             optimizer.zero_grad()
#
#             input_ids1, input_ids2, pos, labels, classes = batch
#             input_ids1 = input_ids1.to(device)
#             input_ids2 = input_ids2.to(device)
#             labels = labels.to(device).reshape(-1, 1)
#             classes = classes.to(device)
#
#             with amp.autocast(enabled=True):
#                 logitis_output = model(input_ids1, input_ids2, pos, is_training=True)
#
#                 if len(logitis_output) == 3:
#                     logits_1, logits_2, logits_mean = logitis_output
#                     contrastive_loss = contrastive_loss_fn(logits_1, logits_2, classes)
#                     mse_loss = F.mse_loss(logits_mean, labels)
#                     loss = 0.2*contrastive_loss + mse_loss
#                 else:
#                     logits_pred = logitis_output[0]
#                     contrastive_loss = contrastive_loss_fn(
#                         logits_pred, logits_pred, classes
#                     )
#                     mse_loss = F.mse_loss(logits_pred, labels)
#                     loss = contrastive_loss + mse_loss
#
#             tr_loss += loss.item()
#             nb_tr_steps += 1
#
#             scaler.scale(loss).backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
#             scaler.step(optimizer)
#             scaler.update()
#
#             train_pbar.set_postfix(loss=f"{loss.item():.4f}")
#
#         scheduler.step(epoch)
#         epoch_loss = tr_loss / nb_tr_steps
#         print(f"\nEpoch {epoch + 1} Training Loss: {epoch_loss:.4f}")
#
#         # ===== 验证 =====
#         labels_np, predictions_np = do_valid(model, testing_loader, device)
#
#         rmse = np.sqrt(np.mean((labels_np - predictions_np) ** 2))
#         mae = np.mean(np.abs(labels_np - predictions_np))
#         spearman = stats.spearmanr(labels_np, predictions_np)[0]
#         pearson = pearsonr(labels_np, predictions_np)[0]
#
#         print(
#             f"RMSE {rmse:.4f} | MAE {mae:.4f} | "
#             f"Spearman {spearman:.4f} | Pearson {pearson:.4f}"
#         )
#
#         improved = False
#
#         # ===== Spearman 最优 =====
#         if spearman > best_spearman:
#             best_spearman = spearman
#             best_epoch_spearman = epoch + 1
#             torch.save(model.state_dict(), "checkpoint/best_model_spearman.pt")
#             improved = True
#
#         # ===== RMSE 最优 =====
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_epoch_rmse = epoch + 1
#             torch.save(model.state_dict(), "checkpoint/best_model_rmse.pt")
#             improved = True
#
#         # ===== 早停 =====
#         if improved:
#             no_improve = 0
#         else:
#             no_improve += 1
#
#         if no_improve >= patience:
#             print(
#                 f"Early stopping at epoch {epoch + 1}\n"
#                 f"Best Spearman: epoch {best_epoch_spearman}, value {best_spearman:.4f}\n"
#                 f"Best RMSE: epoch {best_epoch_rmse}, value {best_rmse:.4f}"
#             )
#             break
#
#     print(
#         f"\nTraining finished.\n"
#         f"Best Spearman: {best_spearman:.4f} (epoch {best_epoch_spearman})\n"
#         f"Best RMSE: {best_rmse:.4f} (epoch {best_epoch_rmse})"
#     )
#
# def do_valid(model, testing_loader, device):
#     model.eval()
#
#     eval_loss = 0.0
#     nb_eval_steps = 0
#     eval_preds, eval_labels = [], []
#
#     # ===== 验证进度条 =====
#     valid_pbar = tqdm(
#         testing_loader,
#         desc="Validation",
#         leave=False
#     )
#
#     with torch.no_grad():
#         for batch in valid_pbar:
#             input_ids1, input_ids2, pos, labels, classes = batch
#
#             input_ids1 = input_ids1.to(device)
#             input_ids2 = input_ids2.to(device)
#             labels = labels.to(device).reshape(-1, 1)
#
#             logits = model(input_ids1, input_ids2, pos)
#             loss = torch.nn.functional.mse_loss(logits, labels)
#
#             eval_loss += loss.item()
#             nb_eval_steps += 1
#
#             eval_labels.extend(labels.cpu().numpy())
#             eval_preds.extend(logits.cpu().numpy())
#
#             valid_pbar.set_postfix(loss=f"{loss.item():.4f}")
#
#     eval_loss /= nb_eval_steps
#     # print(f"Validation Loss: {eval_loss:.4f}")
#
#     labels = np.array(eval_labels).ravel()
#     predictions = np.array(eval_preds).ravel()
#
#     return labels, predictions

#分类
# from scipy import stats
# from scipy.stats import pearsonr
# import torch
# import numpy as np
# from torch.cuda import amp
# from tqdm import tqdm
#
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     matthews_corrcoef,
#     roc_auc_score,
#     confusion_matrix,
# )
#
#
# def _cls_metrics_from_ddg_logits(ddg_pred: np.ndarray, cls_true: np.ndarray, threshold: float = 0.0):
#     """
#     用预测ddg的正负号当分类：pred_cls = (ddg_pred > 0)
#     threshold=0.0 等价于 sigmoid(logit)>0.5 的分界点
#     """
#     cls_true = cls_true.astype(int).ravel()
#     ddg_pred = ddg_pred.astype(float).ravel()
#
#     # 预测类别（按符号）
#     cls_pred = (ddg_pred > threshold).astype(int)
#
#     acc = accuracy_score(cls_true, cls_pred)
#     p, r, f1, _ = precision_recall_fscore_support(cls_true, cls_pred, average="binary", zero_division=0)
#     mcc = matthews_corrcoef(cls_true, cls_pred)
#     cm = confusion_matrix(cls_true, cls_pred)
#
#     # AUC：用 ddg_pred 当 score（越大越倾向 label=1）
#     auc = None
#     try:
#         if len(np.unique(cls_true)) == 2:
#             auc = roc_auc_score(cls_true, ddg_pred)
#     except Exception:
#         auc = None
#
#     return {
#         "acc": acc,
#         "p": p,
#         "r": r,
#         "f1": f1,
#         "mcc": mcc,
#         "auc": auc,
#         "cm": cm,
#     }
#
#
# def do_train(
#     model,
#     optimizer,
#     scheduler,
#     training_loader,
#     testing_loader,
#     device,
#     epochs,
#     # ---- 新增：分类辅助loss权重（=0 表示只回归）
#     lambda_cls: float = 0.2,
#     # ---- 新增：分类loss温度，避免 ddg 数值尺度过大/过小影响 BCE
#     cls_tau: float = 1.0,
# ):
#     scaler = amp.GradScaler()
#
#     mse_loss_fn = torch.nn.MSELoss()
#     bce_loss_fn = torch.nn.BCEWithLogitsLoss()
#
#     for epoch in range(epochs):
#         model.train()
#         tr_loss = 0.0
#         nb_tr_steps = 0
#
#         train_pbar = tqdm(training_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Training", leave=False)
#
#         for batch in train_pbar:
#             optimizer.zero_grad()
#
#             input_ids1, input_ids2, pos, labels, classes = batch
#             input_ids1 = input_ids1.to(device)
#             input_ids2 = input_ids2.to(device)
#             pos = pos.to(device)  # ✅ 建议也搬到GPU
#             labels = labels.to(device).reshape(-1, 1).float()      # ddg
#             classes = classes.to(device).reshape(-1, 1).float()    # 0/1
#
#             with amp.autocast(enabled=True):
#                 logitis_output = model(input_ids1, input_ids2, pos, is_training=True)
#
#                 # 你的模型训练时返回 [logits_cls, logits_env, logits_mean]
#                 if isinstance(logitis_output, (list, tuple)) and len(logitis_output) == 3:
#                     logits_1, logits_2, logits_mean = logitis_output  # 都是 [B,1]
#
#                     # ---- 回归loss（原样保留）----
#                     loss_reg = (
#                         mse_loss_fn(logits_1, labels) +
#                         mse_loss_fn(logits_2, labels) +
#                         mse_loss_fn(logits_mean, labels)
#                     )
#
#                     # ---- 可选：分类辅助loss（用 logits_mean 的符号做二分类）----
#                     if lambda_cls > 0:
#                         cls_logits = logits_mean / float(cls_tau)
#                         loss_cls = bce_loss_fn(cls_logits, classes)
#                         loss = loss_reg + float(lambda_cls) * loss_cls
#                     else:
#                         loss = loss_reg
#
#                 else:
#                     # 如果你的 forward 返回单个tensor
#                     logits_pred = logitis_output[0] if isinstance(logitis_output, (list, tuple)) else logitis_output
#                     loss_reg = mse_loss_fn(logits_pred, labels)
#
#                     if lambda_cls > 0:
#                         cls_logits = (logits_pred / float(cls_tau))
#                         loss_cls = bce_loss_fn(cls_logits, classes)
#                         loss = loss_reg + float(lambda_cls) * loss_cls
#                     else:
#                         loss = loss_reg
#
#             tr_loss += loss.item()
#             nb_tr_steps += 1
#
#             scaler.scale(loss).backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
#             scaler.step(optimizer)
#             scaler.update()
#
#             train_pbar.set_postfix(loss=f"{loss.item():.4f}")
#
#         # scheduler
#         try:
#             scheduler.step(epoch)
#         except TypeError:
#             scheduler.step()
#
#         epoch_loss = tr_loss / max(nb_tr_steps, 1)
#         print(f"\nEpoch {epoch + 1} Training Loss: {epoch_loss:.4f}")
#
#         # ===== 验证：回归 + 分类（含 MCC）=====
#         ddg_true, ddg_pred, cls_true = do_valid(model, testing_loader, device)
#
#         print(
#             f"RMSE {np.sqrt(np.mean((ddg_true - ddg_pred) ** 2)):.4f} | "
#             f"MAE {np.mean(np.abs(ddg_true - ddg_pred)):.4f} | "
#             f"Spearman {stats.spearmanr(ddg_true, ddg_pred)[0]:.4f} | "
#             f"Pearson {pearsonr(ddg_true, ddg_pred)[0]:.4f}"
#         )
#
#         cls_metrics = _cls_metrics_from_ddg_logits(ddg_pred, cls_true, threshold=0.0)
#         msg = (
#             f"CLS Acc {cls_metrics['acc']:.4f} | "
#             f"P {cls_metrics['p']:.4f} | R {cls_metrics['r']:.4f} | "
#             f"F1 {cls_metrics['f1']:.4f} | MCC {cls_metrics['mcc']:.4f}"
#         )
#         if cls_metrics["auc"] is not None:
#             msg += f" | AUC {cls_metrics['auc']:.4f}"
#         print(msg)
#         print("Confusion Matrix:\n", cls_metrics["cm"])
#
#     print("Saving model...")
#     os.makedirs("checkpoint", exist_ok=True)
#     torch.save(model.state_dict(), "checkpoint/training1.pt")
#
#
# def do_valid(model, testing_loader, device):
#     model.eval()
#
#     eval_loss = 0.0
#     nb_eval_steps = 0
#     eval_preds, eval_labels, eval_cls = [], [], []
#
#     valid_pbar = tqdm(testing_loader, desc="Validation", leave=False)
#
#     with torch.no_grad():
#         for batch in valid_pbar:
#             input_ids1, input_ids2, pos, labels, classes = batch
#
#             input_ids1 = input_ids1.to(device)
#             input_ids2 = input_ids2.to(device)
#             pos = pos.to(device)  # ✅
#             labels = labels.to(device).reshape(-1, 1).float()
#             classes = classes.to(device).reshape(-1, 1).long()
#
#             out = model(input_ids1, input_ids2, pos)
#
#             # 你验证时 forward 返回 logits_mean（[B,1]）
#             logits = out
#             loss = torch.nn.functional.mse_loss(logits, labels)
#
#             eval_loss += loss.item()
#             nb_eval_steps += 1
#
#             eval_labels.extend(labels.cpu().numpy())
#             eval_preds.extend(logits.cpu().numpy())
#             eval_cls.extend(classes.cpu().numpy())
#
#             valid_pbar.set_postfix(loss=f"{loss.item():.4f}")
#
#     eval_loss /= max(nb_eval_steps, 1)
#     print(f"Validation Loss: {eval_loss:.4f}")
#
#     ddg_true = np.array(eval_labels).ravel()
#     ddg_pred = np.array(eval_preds).ravel()
#     cls_true = np.array(eval_cls).ravel()
#
#     return ddg_true, ddg_pred, cls_true
