from torch.utils.data import Dataset
import esm
import torch
import pandas as pd

class ProteinDataset(Dataset):
    def __init__(self, df):
        self.df = df
        _, esm1v_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm1v_batch_converter = esm1v_alphabet.get_batch_converter()

    def __getitem__(self, idx):
        _, _, esm1b_batch_tokens1 = self.esm1v_batch_converter([('' , ''.join(self.df.iloc[idx]['wt_seq'])[:1022])])
        _, _, esm1b_batch_tokens2 = self.esm1v_batch_converter([('' , ''.join(self.df.iloc[idx]['mut_seq'])[:1022])])
        pos = self.df.iloc[idx]['pos']
        return esm1b_batch_tokens1.squeeze(dim=0), esm1b_batch_tokens2.squeeze(dim=0), pos+1, torch.FloatTensor([self.df.iloc[idx]['ddg']])

    def __len__(self):
        return len(self.df)

def data_frame_to_list(dataframe):
    # header = ["wt_seq", "mut_seq", "ddg", "pos"]
    header = ["wt_seq", "mut_seq", "dTm", "pos"]
    lenOfDate = len(dataframe)
    listOfdate = []
    for raw in range(lenOfDate):
        listTemp = []
        for name in header:
            listTemp.append(dataframe.iloc[raw][name])
        listOfdate.append(listTemp)
    return listOfdate

class DdgData(object):
    def __init__(self, data_path):
        _, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.tokenizer = esm2_alphabet.get_batch_converter()

        dataframe = pd.read_csv(data_path)
        self.wild_tokens, self.mutant_tokens, self.positions, self.ddGs, self.cls = self.__process__(dataframe)

    def __process__(self, data):
        data_list = data_frame_to_list(data)
        wild_seqs = []
        mutation_seqs = []
        positions = []
        ddGs = []
        cls = []

        for index, (wild_seq, mutation_seq, ddg, pos) in enumerate(data_list):
            wild_seqs.append(("none", wild_seq[:1022]))
            mutation_seqs.append(("none", mutation_seq[:1022]))
            positions.append(int(pos) + 1)
            ddGs.append(ddg)
            if ddg > 0:
                cls.append(1)
            else:
                cls.append(0)
        wild_labels, wild_strs, wild_tokens = self.tokenizer(wild_seqs)
        mutation_labels, mutation_strs, mutant_tokens = self.tokenizer(mutation_seqs)
        positions = torch.tensor(positions)
        ddGs = torch.tensor(ddGs)
        cls = torch.tensor(cls)

        return wild_tokens, mutant_tokens, positions, ddGs, cls

    def __len__(self):
        return len(self.ddGs)

class ddG_Dataset(torch.utils.data.Dataset):
    def __init__(self, wild_tokens, mutant_tokens, positions, ddGs, cls):

        self.wild_tokens = wild_tokens
        self.mutant_tokens = mutant_tokens
        self.positions = positions
        self.ddGs = ddGs
        self.cls = cls

    def __len__(self):
        return len(self.ddGs)

    def __getitem__(self, index):
        return (self.wild_tokens[index], self.mutant_tokens[index], self.positions[index], self.ddGs[index].float(), self.cls[index])


#





# from torch.utils.data import Dataset
# import esm
# import torch
# import pandas as pd
#
#
# class ProteinDataset(Dataset):
#     """
#     Simple on-the-fly tokenization dataset.
#     Now returns: (wt_tokens, mut_tokens, pos+1, ddg_tensor, pdb_id)
#     """
#     def __init__(self, df: pd.DataFrame):
#         self.df = df
#         _, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#         self.esm_batch_converter = esm2_alphabet.get_batch_converter()
#
#     def __getitem__(self, idx):
#         wt_seq = ''.join(self.df.iloc[idx]['wt_seq'])[:1022]
#         mut_seq = ''.join(self.df.iloc[idx]['mut_seq'])[:1022]
#
#         _, _, wt_tokens = self.esm_batch_converter([('', wt_seq)])
#         _, _, mut_tokens = self.esm_batch_converter([('', mut_seq)])
#
#         pos = int(self.df.iloc[idx]['pos'])
#         ddg = float(self.df.iloc[idx]['ddg'])
#         pdb_id = self.df.iloc[idx]['pdb_id']
#
#         return (
#             wt_tokens.squeeze(0),
#             mut_tokens.squeeze(0),
#             pos + 1,
#             torch.FloatTensor([ddg]),
#             pdb_id
#         )
#
#     def __len__(self):
#         return len(self.df)
#
#
# def data_frame_to_list(dataframe: pd.DataFrame):
#     # ✅ add pdb_id
#     header = ["pdb_id", "wt_seq", "mut_seq", "ddg", "pos"]
#     lenOfData = len(dataframe)
#     listOfData = []
#     for raw in range(lenOfData):
#         row_items = []
#         for name in header:
#             row_items.append(dataframe.iloc[raw][name])
#         listOfData.append(row_items)
#     return listOfData
#
#
# class DdgData(object):
#     """
#     Pre-tokenize all sequences once.
#     Now also keeps pdb_ids list.
#     """
#     def __init__(self, data_path: str):
#         _, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#         self.tokenizer = esm2_alphabet.get_batch_converter()
#
#         dataframe = pd.read_csv(data_path)
#         (self.wild_tokens,
#          self.mutant_tokens,
#          self.positions,
#          self.ddGs,
#          self.cls,
#          self.pdb_ids) = self.__process__(dataframe)
#
#     def __process__(self, data: pd.DataFrame):
#         data_list = data_frame_to_list(data)
#
#         wild_seqs = []
#         mutation_seqs = []
#         positions = []
#         ddGs = []
#         cls = []
#         pdb_ids = []
#
#         for index, (pdb_id, wild_seq, mutation_seq, ddg, pos) in enumerate(data_list):
#             wild_seqs.append(("none", str(wild_seq)[:1022]))
#             mutation_seqs.append(("none", str(mutation_seq)[:1022]))
#
#             positions.append(int(pos) + 1)
#             ddGs.append(float(ddg))
#             pdb_ids.append(pdb_id)
#
#             cls.append(1 if float(ddg) > 0 else 0)
#
#         _, _, wild_tokens = self.tokenizer(wild_seqs)
#         _, _, mutant_tokens = self.tokenizer(mutation_seqs)
#
#         positions = torch.tensor(positions)
#         ddGs = torch.tensor(ddGs)
#         cls = torch.tensor(cls)
#
#         return wild_tokens, mutant_tokens, positions, ddGs, cls, pdb_ids
#
#     def __len__(self):
#         return len(self.ddGs)
#
#
# class ddG_Dataset(torch.utils.data.Dataset):
#     """
#     Now returns: (wild_tokens, mutant_tokens, position, ddG, cls, pdb_id)
#     """
#     def __init__(self, wild_tokens, mutant_tokens, positions, ddGs, cls, pdb_ids):
#         self.wild_tokens = wild_tokens
#         self.mutant_tokens = mutant_tokens
#         self.positions = positions
#         self.ddGs = ddGs
#         self.cls = cls
#         self.pdb_ids = pdb_ids
#
#     def __len__(self):
#         return len(self.ddGs)
#
#     def __getitem__(self, index):
#         return (
#             self.wild_tokens[index],
#             self.mutant_tokens[index],
#             self.positions[index],
#             self.ddGs[index].float(),
#             self.cls[index],
#             self.pdb_ids[index]
#         )
#

















#预训练
# def data_frame_to_list(dataframe: pd.DataFrame):
#     # 必须包含 re_label：真实=1，伪标签=0
#     header = ["wt_seq", "mut_seq", "ddg", "pos", "re_label"]
#     list_data = []
#     for i in range(len(dataframe)):
#         row = [dataframe.iloc[i][h] for h in header]
#         list_data.append(row)
#     return list_data
#
#
# class DdgData(object):
#     """
#     预先把全量序列都 tokenize 成 tensor（适合数据量不是特别离谱的场景）
#     """
#     def __init__(self, data_path: str):
#         _, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#         self.tokenizer = esm2_alphabet.get_batch_converter()
#
#         dataframe = pd.read_csv(data_path)
#         # 基本检查
#         need_cols = {"wt_seq", "mut_seq", "ddg", "pos", "re_label"}
#         miss = need_cols - set(dataframe.columns)
#         if miss:
#             raise ValueError(f"CSV missing columns: {miss}")
#
#         (self.wild_tokens,
#          self.mutant_tokens,
#          self.positions,
#          self.ddGs,
#          self.cls,
#          self.re_labels) = self.__process__(dataframe)
#
#     def __process__(self, data: pd.DataFrame):
#         data_list = data_frame_to_list(data)
#
#         wild_seqs = []
#         mutation_seqs = []
#         positions = []
#         ddGs = []
#         cls = []
#         re_labels = []
#
#         for (wild_seq, mutation_seq, ddg, pos, re_label) in data_list:
#             # ESM 输入最长 1022（你原来也这么做）
#             wild_seqs.append(("none", str(wild_seq)[:1022]))
#             mutation_seqs.append(("none", str(mutation_seq)[:1022]))
#
#             positions.append(int(pos) + 1)      # 你原本 pos+1
#             ddGs.append(float(ddg))
#
#             cls.append(1 if float(ddg) > 0 else 0)
#             re_labels.append(int(re_label))     # 真实=1，伪=0
#
#         _, _, wild_tokens = self.tokenizer(wild_seqs)
#         _, _, mutant_tokens = self.tokenizer(mutation_seqs)
#
#         positions = torch.tensor(positions, dtype=torch.long)
#         ddGs = torch.tensor(ddGs, dtype=torch.float32)
#         cls = torch.tensor(cls, dtype=torch.long)
#         re_labels = torch.tensor(re_labels, dtype=torch.long)
#
#         return wild_tokens, mutant_tokens, positions, ddGs, cls, re_labels
#
#     def __len__(self):
#         return len(self.ddGs)
#
#
# class ddG_Dataset(Dataset):
#     def __init__(self, wild_tokens, mutant_tokens, positions, ddGs, cls, re_labels):
#         self.wild_tokens = wild_tokens
#         self.mutant_tokens = mutant_tokens
#         self.positions = positions
#         self.ddGs = ddGs
#         self.cls = cls
#         self.re_labels = re_labels
#
#     def __len__(self):
#         return len(self.ddGs)
#
#     def __getitem__(self, index):
#         return (
#             self.wild_tokens[index],
#             self.mutant_tokens[index],
#             self.positions[index],
#             self.ddGs[index].float(),
#             self.cls[index],
#             self.re_labels[index],  # ✅ 真实(1)/伪(0)
#         )

















































# from torch.utils.data import Dataset
# import esm
# import torch
# import pandas as pd
#
# # ================
# # Helper
# # ================
# def _to_seq(x):
#     # 兼容 x 是 list/tuple 或 string
#     if isinstance(x, (list, tuple)):
#         return "".join(x)
#     return str(x)
#
# # ================
# # 方案A：ProteinDataset（按条 tokenize）
# # ================
# class ProteinDataset(Dataset):
#     def __init__(self, df, delta_emb_dict=None, delta_dim=128, strict_delta=False):
#         """
#         df: 需要包含列 wt_seq, mut_seq, pos, ddg, name
#         delta_emb_dict: torch.load("S669_S8754_delta_emb.pt") 得到的 dict
#         delta_dim: delta embedding 维度（你MPNN是128就填128）
#         strict_delta: True 时如果缺 delta 就直接报错；False 时用全0代替
#         """
#         self.df = df
#         self.delta_emb_dict = delta_emb_dict
#         self.delta_dim = int(delta_dim)
#         self.strict_delta = bool(strict_delta)
#
#         # ⚠️ 注意：esm.pretrained.esm2_t33_650M_UR50D() 会加载大模型。
#         # 你原来就是这么写的，我先保持不改逻辑。
#         _, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#         self.batch_converter = esm_alphabet.get_batch_converter()
#
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         wt_seq = _to_seq(row["wt_seq"])[:1022]
#         mut_seq = _to_seq(row["mut_seq"])[:1022]
#         pos = int(row["pos"]) + 1
#         ddg = float(row["ddg"])
#
#         # ===== ESM tokens =====
#         _, _, wt_tok = self.batch_converter([("", wt_seq)])
#         _, _, mut_tok = self.batch_converter([("", mut_seq)])
#
#         # ===== delta_emb（NEW）=====
#         if self.delta_emb_dict is not None:
#             name = row["name"]  # NEW：必须有 name 列
#             if name in self.delta_emb_dict:
#                 delta_emb = self.delta_emb_dict[name]["delta_emb"].float()
#             else:
#                 if self.strict_delta:
#                     raise KeyError(f"[ProteinDataset] delta_emb missing for name={name}")
#                 delta_emb = torch.zeros(self.delta_dim, dtype=torch.float32)
#         else:
#             delta_emb = torch.zeros(self.delta_dim, dtype=torch.float32)
#
#         return (
#             wt_tok.squeeze(0),                      # [T]
#             mut_tok.squeeze(0),                     # [T]
#             torch.tensor(pos, dtype=torch.long),    # []
#             torch.tensor(ddg, dtype=torch.float32), # []
#             delta_emb                               # [delta_dim]  NEW
#         )
#
#     def __len__(self):
#         return len(self.df)
#
#
# # ================
# # 方案B：DdgData + ddG_Dataset（一次性 tokenize）
# # ================
# def data_frame_to_list(dataframe):
#     header = ["wt_seq", "mut_seq", "ddg", "pos", "name"]  # ✅ 加 name
#     lenOfDate = len(dataframe)
#     listOfdate = []
#     for raw in range(lenOfDate):
#         listTemp = []
#         for name in header:
#             listTemp.append(dataframe.iloc[raw][name])
#         listOfdate.append(listTemp)
#     return listOfdate
#
#
# class DdgData(object):
#     def __init__(self, data_path, delta_emb_dict=None, delta_dim=128, strict_delta=False):
#         _, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#         self.tokenizer = esm2_alphabet.get_batch_converter()
#
#         self.delta_emb_dict = delta_emb_dict
#         self.delta_dim = int(delta_dim)
#         self.strict_delta = bool(strict_delta)
#
#         dataframe = pd.read_csv(data_path)
#         self.wild_tokens, self.mutant_tokens, self.positions, self.ddGs, self.cls, self.delta_embs = self.__process__(dataframe)
#
#     def __process__(self, data):
#         data_list = data_frame_to_list(data)
#         wild_seqs = []
#         mutation_seqs = []
#         positions = []
#         ddGs = []
#         cls = []
#         delta_list = []  # ✅ 新增
#
#         for index, (wild_seq, mutation_seq, ddg, pos, name) in enumerate(data_list):
#             wild_seqs.append(("none", str(wild_seq)[:1022]))
#             mutation_seqs.append(("none", str(mutation_seq)[:1022]))
#             positions.append(int(pos) + 1)
#             ddGs.append(float(ddg))
#             cls.append(1 if float(ddg) > 0 else 0)
#
#             # ✅ 取 delta_emb
#             if self.delta_emb_dict is not None:
#                 if name in self.delta_emb_dict:
#                     delta_list.append(self.delta_emb_dict[name]["delta_emb"].float())
#                 else:
#                     if self.strict_delta:
#                         raise KeyError(f"[DdgData] missing delta_emb for name={name}")
#                     delta_list.append(torch.zeros(self.delta_dim, dtype=torch.float32))
#             else:
#                 delta_list.append(torch.zeros(self.delta_dim, dtype=torch.float32))
#
#         _, _, wild_tokens = self.tokenizer(wild_seqs)
#         _, _, mutant_tokens = self.tokenizer(mutation_seqs)
#         positions = torch.tensor(positions, dtype=torch.long)
#         ddGs = torch.tensor(ddGs, dtype=torch.float32)
#         cls = torch.tensor(cls, dtype=torch.long)
#         delta_embs = torch.stack(delta_list, dim=0)  # ✅ [N, delta_dim]
#
#         return wild_tokens, mutant_tokens, positions, ddGs, cls, delta_embs
#
#     def __len__(self):
#         return len(self.ddGs)
#
#
#
# class ddG_Dataset(torch.utils.data.Dataset):
#     def __init__(self, wild_tokens, mutant_tokens, positions, ddGs, cls, delta_embs):
#         self.wild_tokens = wild_tokens
#         self.mutant_tokens = mutant_tokens
#         self.positions = positions
#         self.ddGs = ddGs
#         self.cls = cls
#         self.delta_embs = delta_embs  # ✅ 新增
#
#     def __len__(self):
#         return len(self.ddGs)
#
#     def __getitem__(self, index):
#         return (
#             self.wild_tokens[index],
#             self.mutant_tokens[index],
#             self.positions[index],
#             self.ddGs[index].float(),
#             self.cls[index],
#             self.delta_embs[index].float()  # ✅ 新增返回
#         )
#

