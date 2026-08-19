# evidential_loss.py
import torch
import torch.nn as nn
import numpy as np


class EvidentialLossNLL(nn.Module):
    """
    证据深度学习损失函数：NLL + 正则项
    对应论文公式：L = L_NLL + λ * L_REG
    """

    def __init__(self, lambda_reg=0.2, epsilon=1e-4):
        super().__init__()
        self.lambda_reg = lambda_reg  # 论文中 λ = 0.2
        self.epsilon = epsilon

    def forward(self, gamma, nu, alpha, beta, targets):
        """
        Args:
            gamma: 预测均值 [B]
            nu: 精度参数 [B]
            alpha: 形状参数 [B]
            beta: 尺度参数 [B]
            targets: 真实标签 [B]
        """
        # 确保维度一致
        targets = targets.view(-1)
        gamma = gamma.view(-1)
        nu = nu.view(-1)
        alpha = alpha.view(-1)
        beta = beta.view(-1)

        # ========== 1. 负对数似然损失 (NLL) ==========
        # 基于学生 t 分布
        twoBlambda = 2 * beta * (1 + nu)

        nll = 0.5 * torch.log(np.pi / nu) \
              - alpha * torch.log(twoBlambda) \
              + (alpha + 0.5) * torch.log(nu * (targets - gamma) ** 2 + twoBlambda) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)

        L_NLL = nll

        # ========== 2. 正则化损失 (REG) ==========
        # 惩罚过度自信：预测错误但高置信度时损失大
        error = torch.abs(targets - gamma)
        reg = error * (2 * nu + alpha)
        L_REG = reg

        # ========== 3. 总损失 ==========
        loss = L_NLL + self.lambda_reg * (L_REG - self.epsilon)

        return loss.mean(), L_NLL.mean(), L_REG.mean()  # 返回各分量用于监控


class EvidentialLossEnsemble(nn.Module):
    """
    多分支集成的 EDL 损失
    """

    def __init__(self, lambda_reg=0.2, epsilon=1e-4):
        super().__init__()
        self.edl_loss = EvidentialLossNLL(lambda_reg, epsilon)

    def forward(self, outputs_dict, targets):
        """
        Args:
            outputs_dict: 包含 'cls', 'env', 'mean' 的字典
            targets: 真实标签
        """
        gamma_cls, nu_cls, alpha_cls, beta_cls = outputs_dict['cls']
        gamma_env, nu_env, alpha_env, beta_env = outputs_dict['env']
        gamma_mean, nu_mean, alpha_mean, beta_mean = outputs_dict['mean']

        # 三个分支的损失
        loss_cls, nll_cls, reg_cls = self.edl_loss(gamma_cls, nu_cls, alpha_cls, beta_cls, targets)
        loss_env, nll_env, reg_env = self.edl_loss(gamma_env, nu_env, alpha_env, beta_env, targets)
        loss_mean, nll_mean, reg_mean = self.edl_loss(gamma_mean, nu_mean, alpha_mean, beta_mean, targets)

        # 总损失（可调整权重）
        total_loss = loss_cls + loss_env + loss_mean

        return total_loss, {
            'cls_loss': loss_cls.item(),
            'env_loss': loss_env.item(),
            'mean_loss': loss_mean.item(),
            'total_nll': (nll_cls + nll_env + nll_mean).item(),
            'total_reg': (reg_cls + reg_env + reg_mean).item()
        }