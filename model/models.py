









import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import numpy as np
from .utils import trunc_normal_, Block


class Local(nn.Module):
    # mutual attention weight selection
    def __init__(self):
        super(Local, self).__init__()

    def forward(self, x):
        # --> x[B, H, T] --> [B, T]
        weights = x.mean(1)
        max_inx = torch.argsort(weights, dim=1, descending=True)
        return max_inx


class GlobalLocalDualBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm2, self.esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        self.Global_transform = nn.Linear(1280, 1280)
        self.Local_transform = nn.Linear(1280, 1280)

        self.Global_classifier = nn.Linear(1280, 1)
        self.Global_const1 = torch.nn.Parameter(torch.ones((1, 1280)))
        self.Global_const2 = torch.nn.Parameter(-1 * torch.ones((1, 1280)))

        self.token_select = Local()
        self.num_token = 32
        self.Local_classifier = nn.Linear(1280, 1)
        self.Local_const1 = torch.nn.Parameter(torch.ones((1, 1280)))
        self.Local_const2 = torch.nn.Parameter(-1 * torch.ones((1, 1280)))

        self.Global_token = nn.Parameter(torch.zeros(1, 1, 1280))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_token + 1, 1280))
        trunc_normal_(self.Global_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.fusion_block = Block(1280, 8)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, token_ids1, token_ids2, pos, is_training=False, return_feat=False):
        """
            token_ids1: [B, L]
            token_ids2: [B, L]
            pos: [B, ]
        """
        # --> representations, attentions
        pos = int(pos)
        outputs1 = self.esm2.forward(token_ids1, repr_layers=[33], need_head_weights=True)
        outputs2 = self.esm2.forward(token_ids2, repr_layers=[33], need_head_weights=True)
        # --> [B, L+1, 1280]
        outputs1_rep = outputs1['representations'][33]
        outputs2_rep = outputs2['representations'][33]

        outputs1_pos_attn = outputs1['attentions'][:, -1]
        outputs2_pos_attn = outputs2['attentions'][:, -1]

        outputs1_Global = self.Global_transform(outputs1_rep[:, 0, :])
        outputs2_Global = self.Global_transform(outputs2_rep[:, 0, :])

        outputs_Global = self.Global_const1 * outputs1_Global + self.Global_const2 * outputs2_Global
        logits_Global = self.Global_classifier(outputs_Global)

      
        # --> [B, L+1, 1280]
        outputs1_rep_Local = self.Local_transform(outputs1_rep)
        outputs2_rep_Local = self.Local_transform(outputs2_rep)
        # --> (B, H, T, T) -> (B, H, T)
        attn1_pos_positions = outputs1_pos_attn[np.arange(len(outputs1_pos_attn)), :, pos].squeeze(dim=2)
        attn2_pos_positions = outputs2_pos_attn[np.arange(len(outputs2_pos_attn)), :, pos].squeeze(dim=2)
        selected_inx1 = self.token_select(attn1_pos_positions)
        selected_inx2 = self.token_select(attn2_pos_positions)

        # --> (B, T)
        B = selected_inx1.shape[0]
        selected_tokens1 = [[] for i in range(B)]
        selected_tokens2 = [[] for i in range(B)]
        for i in range(B):
            selected_tokens1[i] = outputs1_rep_Local[i, selected_inx1[i, :self.num_token]]
            selected_tokens2[i] = outputs2_rep_Local[i, selected_inx2[i, :self.num_token]]
        # --> [B, num_toknes, 1280]
        selected_tokens1 = torch.stack(selected_tokens1)
        selected_tokens2 = torch.stack(selected_tokens2)

        Global_tokens = self.Global_token.expand(B, -1, -1)
        Local_tokens1 = torch.cat((Global_tokens, selected_tokens1), dim=1) + self.pos_embed
        Local_tokens2 = torch.cat((Global_tokens, selected_tokens2), dim=1) + self.pos_embed

        Local_feat1 = self.fusion_block(Local_tokens1)[:, 0]
        Local_feat2 = self.fusion_block(Local_tokens2)[:, 0]

        outputs_Local = self.Local_const1 * Local_feat1 + self.Local_const2 * Local_feat2
        logits_Local= self.Local_classifier(outputs_Local)

        logtis_mean = (logits_Global + logits_Local) / 2

        if return_feat:
            return [outputs_Global, outputs_Local]

        if is_training:
            return [logits_Global, logits_Local, logtis_mean]
        else:
            return logtis_mean



# import torch
# import torch.nn as nn
# import esm
#
# from .utils import trunc_normal_, Block
#
#
# class Local(nn.Module):
#     """
#     Mutual attention based token ranking.
#     Input:
#         x: [B, H, T]
#         valid_token_mask: [B, T] or None
#     Output:
#         sorted indices: [B, T]
#     """
#     def __init__(self):
#         super(Local, self).__init__()
#
#     def forward(self, x, valid_token_mask=None):
#         # x: [B, H, T] -> [B, T]
#         weights = x.mean(dim=1)
#
#         # mask invalid tokens so they will not be selected into top-k
#         if valid_token_mask is not None:
#             weights = weights.masked_fill(~valid_token_mask, float("-inf"))
#
#         max_inx = torch.argsort(weights, dim=1, descending=True)
#         return max_inx
#
#
# class GlobalLocalDualBranch(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.esm2, self.esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#
#         self.Global_transform = nn.Linear(1280, 1280)
#         self.Local_transform = nn.Linear(1280, 1280)
#
#         self.Global_classifier = nn.Linear(1280, 1)
#         self.Global_const1 = nn.Parameter(torch.ones((1, 1280)))
#         self.Global_const2 = nn.Parameter(-1 * torch.ones((1, 1280)))
#
#         self.token_select = Local()
#         self.num_token = 32
#         self.Local_classifier = nn.Linear(1280, 1)
#         self.Local_const1 = nn.Parameter(torch.ones((1, 1280)))
#         self.Local_const2 = nn.Parameter(-1 * torch.ones((1, 1280)))
#
#         self.Global_token = nn.Parameter(torch.zeros(1, 1, 1280))
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_token + 1, 1280))
#         trunc_normal_(self.Global_token, std=0.02)
#         trunc_normal_(self.pos_embed, std=0.02)
#
#         self.fusion_block = Block(1280, 8)
#
#         # token ids from ESM alphabet
#         self.padding_idx = self.esm2_alphabet.padding_idx
#         self.cls_idx = self.esm2_alphabet.cls_idx
#         self.eos_idx = self.esm2_alphabet.eos_idx
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {"pos_embed", "Global_token"}
#
#     def _get_valid_residue_mask(self, token_ids):
#         """
#         Return mask for real residue tokens only, excluding BOS/CLS, EOS, PAD
#         token_ids: [B, L]
#         """
#         valid_mask = (
#             (token_ids != self.padding_idx) &
#             (token_ids != self.cls_idx) &
#             (token_ids != self.eos_idx)
#         )
#         return valid_mask
#
#     def _gather_topk_tokens(self, rep, topk_inx):
#         """
#         rep: [B, L, C]
#         topk_inx: [B, K]
#         return: [B, K, C]
#         """
#         B, K = topk_inx.shape
#         C = rep.size(-1)
#
#         gather_index = topk_inx.unsqueeze(-1).expand(B, K, C)
#         selected_tokens = torch.gather(rep, dim=1, index=gather_index)
#         return selected_tokens
#
#     def forward(self, token_ids1, token_ids2, pos, is_training=False, return_feat=False):
#         """
#         token_ids1: [B, L]
#         token_ids2: [B, L]
#         pos: [B] or list/tuple
#              mutation position in ESM token index space.
#              If your pos is residue index instead of token index,
#              you likely need to convert it outside this model first.
#         """
#         device = token_ids1.device
#         B, L = token_ids1.shape
#
#         if not torch.is_tensor(pos):
#             pos = torch.tensor(pos, dtype=torch.long, device=device)
#         else:
#             pos = pos.to(device=device, dtype=torch.long)
#
#         # protect against out-of-range positions
#         pos = pos.clamp(min=0, max=L - 1)
#
#         # ===== ESM forward =====
#         outputs1 = self.esm2.forward(token_ids1, repr_layers=[33], need_head_weights=True)
#         outputs2 = self.esm2.forward(token_ids2, repr_layers=[33], need_head_weights=True)
#
#         # [B, L, 1280]
#         outputs1_rep = outputs1["representations"][33]
#         outputs2_rep = outputs2["representations"][33]
#
#         # [B, num_layers, num_heads, T, T] -> take last layer
#         outputs1_pos_attn = outputs1["attentions"][:, -1]   # [B, H, T, T]
#         outputs2_pos_attn = outputs2["attentions"][:, -1]   # [B, H, T, T]
#
#         # ===== Global branch =====
#         outputs1_Global = self.Global_transform(outputs1_rep[:, 0, :])
#         outputs2_Global = self.Global_transform(outputs2_rep[:, 0, :])
#
#         outputs_Global = self.Global_const1 * outputs1_Global + self.Global_const2 * outputs2_Global
#         logits_Global = self.Global_classifier(outputs_Global)
#
#         # ===== Local branch =====
#         outputs1_rep_Local = self.Local_transform(outputs1_rep)
#         outputs2_rep_Local = self.Local_transform(outputs2_rep)
#
#         # valid residue token mask
#         wt_residue_mask = self._get_valid_residue_mask(token_ids1)   # [B, L]
#         mut_residue_mask = self._get_valid_residue_mask(token_ids2)  # [B, L]
#
#         # get attention from mutation position to all tokens
#         batch_idx = torch.arange(B, device=device)
#         attn1_pos_positions = outputs1_pos_attn[batch_idx, :, pos, :]  # [B, H, T]
#         attn2_pos_positions = outputs2_pos_attn[batch_idx, :, pos, :]  # [B, H, T]
#
#         # rank tokens, masking invalid token positions
#         selected_inx1 = self.token_select(attn1_pos_positions, wt_residue_mask)   # [B, T]
#         selected_inx2 = self.token_select(attn2_pos_positions, mut_residue_mask)  # [B, T]
#
#         topk_inx1 = selected_inx1[:, :self.num_token]  # [B, K]
#         topk_inx2 = selected_inx2[:, :self.num_token]  # [B, K]
#
#         # gather selected token features
#         selected_tokens1 = self._gather_topk_tokens(outputs1_rep_Local, topk_inx1)  # [B, K, 1280]
#         selected_tokens2 = self._gather_topk_tokens(outputs2_rep_Local, topk_inx2)  # [B, K, 1280]
#
#         # ===== Local fusion =====
#         Global_tokens = self.Global_token.expand(B, -1, -1)
#         Local_tokens1 = torch.cat((Global_tokens, selected_tokens1), dim=1) + self.pos_embed
#         Local_tokens2 = torch.cat((Global_tokens, selected_tokens2), dim=1) + self.pos_embed
#
#         Local_feat1 = self.fusion_block(Local_tokens1)[:, 0]
#         Local_feat2 = self.fusion_block(Local_tokens2)[:, 0]
#
#         outputs_Local = self.Local_const1 * Local_feat1 + self.Local_const2 * Local_feat2
#         logits_Local = self.Local_classifier(outputs_Local)
#
#         logits_mean = (logits_Global + logits_Local) / 2
#
#         if return_feat:
#             return [outputs_Global, outputs_Local]
#
#         if is_training:
#             return [logits_Global, logits_Local, logits_mean]
#         else:
#             return logits_mean
#


# import torch
# import torch.nn as nn
# import esm
#
# from .utils import trunc_normal_, Block
#
#
# class Local(nn.Module):
#     """
#     Select top-k local tokens based on attention scores.
#     """
#     def __init__(self):
#         super(Local, self).__init__()
#
#     def forward(self, x, valid_mask=None, topk=None):
#         """
#         Args:
#             x: attention scores, shape [B, H, T]
#             valid_mask: optional bool mask, shape [B, T], True means valid
#             topk: int, number of tokens to select
#         Returns:
#             indices: selected token indices, shape [B, topk] if topk is not None,
#                      otherwise sorted indices [B, T]
#         """
#         weights = x.mean(dim=1)  # [B, T]
#
#         if valid_mask is not None:
#             weights = weights.masked_fill(~valid_mask, float("-inf"))
#
#         if topk is not None:
#             indices = torch.topk(weights, k=topk, dim=1).indices
#         else:
#             indices = torch.argsort(weights, dim=1, descending=True)
#
#         return indices
#
#
# class GlobalLocalDualBranch(nn.Module):
#     def __init__(self, num_token=32, esm_layer=33, embed_dim=1280, num_heads=8):
#         super().__init__()
#
#         # Load pretrained ESM-2 model
#         self.esm2, self.esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#
#         self.esm_layer = esm_layer
#         self.embed_dim = embed_dim
#         self.num_token = num_token
#
#         # Global / local transforms
#         self.Global_transform = nn.Linear(embed_dim, embed_dim)
#         self.Local_transform = nn.Linear(embed_dim, embed_dim)
#
#         # Optional normalization after difference features
#         self.global_norm = nn.LayerNorm(embed_dim)
#         self.local_norm = nn.LayerNorm(embed_dim)
#
#         # Classifiers
#         self.Global_classifier = nn.Linear(embed_dim, 1)
#         self.Local_classifier = nn.Linear(embed_dim, 1)
#
#         # Token selector
#         self.token_select = Local()
#
#         # Learnable global token + positional embeddings
#         self.Global_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_token + 1, embed_dim))
#
#         trunc_normal_(self.Global_token, std=0.02)
#         trunc_normal_(self.pos_embed, std=0.02)
#
#         # Fusion block
#         self.fusion_block = Block(embed_dim, num_heads)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'Global_token'}
#
#     def _build_valid_mask(self, token_ids):
#         """
#         Build valid token mask for ESM outputs.
#         Excludes BOS / EOS / PAD tokens.
#         Args:
#             token_ids: [B, L]
#         Returns:
#             valid_mask: [B, L] bool
#         """
#         pad_idx = self.esm2_alphabet.padding_idx
#         cls_idx = self.esm2_alphabet.cls_idx
#         eos_idx = self.esm2_alphabet.eos_idx
#
#         valid_mask = (token_ids != pad_idx)
#         valid_mask &= (token_ids != cls_idx)
#         valid_mask &= (token_ids != eos_idx)
#
#         return valid_mask
#
#     def _gather_tokens(self, reps, indices):
#         """
#         Gather token embeddings by indices.
#         Args:
#             reps: [B, L, C]
#             indices: [B, K]
#         Returns:
#             gathered: [B, K, C]
#         """
#         gather_idx = indices.unsqueeze(-1).expand(-1, -1, reps.size(-1))
#         return torch.gather(reps, dim=1, index=gather_idx)
#
#     def forward(self, token_ids1, token_ids2, pos, is_training=False, return_feat=False):
#         """
#         Args:
#             token_ids1: [B, L], wild-type sequence tokens
#             token_ids2: [B, L], mutant sequence tokens
#             pos: [B], mutation position in raw residue indexing (0-based)
#             is_training: whether to return branch outputs for training
#             return_feat: whether to return features
#         Returns:
#             logits_mean or [logits_Global, logits_Local, logits_mean]
#         """
#         device = token_ids1.device
#         B, L = token_ids1.shape
#
#         # -------------------------
#         # 1. ESM forward
#         # -------------------------
#         outputs1 = self.esm2(token_ids1, repr_layers=[self.esm_layer], need_head_weights=True)
#         outputs2 = self.esm2(token_ids2, repr_layers=[self.esm_layer], need_head_weights=True)
#
#         rep1 = outputs1["representations"][self.esm_layer]   # [B, L, C]
#         rep2 = outputs2["representations"][self.esm_layer]   # [B, L, C]
#
#         # attentions: [B, num_layers, num_heads, L, L]
#         attn1 = outputs1["attentions"][:, -1]  # [B, H, L, L]
#         attn2 = outputs2["attentions"][:, -1]  # [B, H, L, L]
#
#         # -------------------------
#         # 2. Global branch
#         # -------------------------
#         # ESM global token is at position 0 (<cls>)
#         global1 = self.Global_transform(rep1[:, 0, :])   # [B, C]
#         global2 = self.Global_transform(rep2[:, 0, :])   # [B, C]
#
#         outputs_Global = self.global_norm(global2 - global1)
#         logits_Global = self.Global_classifier(outputs_Global)  # [B, 1]
#
#         # -------------------------
#         # 3. Local branch
#         # -------------------------
#         rep1_local = self.Local_transform(rep1)  # [B, L, C]
#         rep2_local = self.Local_transform(rep2)  # [B, L, C]
#
#         # raw residue index -> ESM token index
#         # if pos is 0-based over residues, then +1 for BOS/CLS
#         pos_esm = pos.long() + 1
#         pos_esm = pos_esm.clamp(min=0, max=L - 1)
#
#         batch_idx = torch.arange(B, device=device)
#
#         # attention from mutation token to all tokens
#         # [B, H, L]
#         attn1_mut = attn1[batch_idx, :, pos_esm, :]
#         attn2_mut = attn2[batch_idx, :, pos_esm, :]
#
#         # shared token selection for WT and Mutant
#         joint_attn = 0.5 * (attn1_mut + attn2_mut)  # [B, H, L]
#
#         # valid positions: exclude BOS/EOS/PAD
#         valid_mask1 = self._build_valid_mask(token_ids1)
#         valid_mask2 = self._build_valid_mask(token_ids2)
#         valid_mask = valid_mask1 & valid_mask2
#
#         # make sure mutation position itself is always valid if inside range
#         valid_mask[batch_idx, pos_esm] = True
#
#         selected_idx = self.token_select(
#             joint_attn,
#             valid_mask=valid_mask,
#             topk=self.num_token
#         )  # [B, num_token]
#
#         selected_tokens1 = self._gather_tokens(rep1_local, selected_idx)  # [B, K, C]
#         selected_tokens2 = self._gather_tokens(rep2_local, selected_idx)  # [B, K, C]
#
#         # prepend learnable global token
#         global_tokens = self.Global_token.expand(B, -1, -1)  # [B, 1, C]
#
#         local_tokens1 = torch.cat((global_tokens, selected_tokens1), dim=1) + self.pos_embed
#         local_tokens2 = torch.cat((global_tokens, selected_tokens2), dim=1) + self.pos_embed
#
#         local_feat1 = self.fusion_block(local_tokens1)[:, 0, :]  # [B, C]
#         local_feat2 = self.fusion_block(local_tokens2)[:, 0, :]  # [B, C]
#
#         outputs_Local = self.local_norm(local_feat2 - local_feat1)
#         logits_Local = self.Local_classifier(outputs_Local)  # [B, 1]
#
#         # -------------------------
#         # 4. Final prediction
#         # -------------------------
#         logits_mean = (logits_Global + logits_Local) / 2.0
#
#         if return_feat:
#             return [outputs_Global, outputs_Local]
#
#         if is_training:
#             return [logits_Global, logits_Local, logits_mean]
#
#         return logits_mean
