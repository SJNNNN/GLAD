
# ====== 必须最先执行 ======
import os
import tempfile
from pathlib import Path

def setup_tmpdir():
    user = os.environ.get("USER", "user")
    base_tmp = f"/data/SJNDATA/tmp/{user}"
    hf_home = f"{base_tmp}/hf"

    # 创建目录
    Path(base_tmp).mkdir(parents=True, exist_ok=True)
    Path(hf_home).mkdir(parents=True, exist_ok=True)

    # 系统临时目录（multiprocessing / torch 用）
    os.environ["TMPDIR"] = base_tmp
    os.environ["TMP"] = base_tmp
    os.environ["TEMP"] = base_tmp
    tempfile.tempdir = base_tmp

    # Hugging Face 缓存目录
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{hf_home}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{hf_home}/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{hf_home}/datasets"

    for p in [
        os.environ["HUGGINGFACE_HUB_CACHE"],
        os.environ["TRANSFORMERS_CACHE"],
        os.environ["HF_DATASETS_CACHE"],
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)

    print(f"[tmp] TMPDIR = {base_tmp}")
    print(f"[hf ] HF_HOME = {hf_home}")

setup_tmpdir()
# ====== 到这里为止，才允许 import torch ======

from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, make_scheduler
from processor import do_train
from  processor import do_test
from utils.set_seed import set_seed
import torch

seed = 1024
set_seed(seed)

# lr = 6e-6
lr = 6e-6
epochs = 80
batch_size = 1
device = "cuda:0"
load_checkpoint = False
# --> data
training_loader, testing_loader =make_dataloader("ddgdata/S8754_parsed.csv", "ddgdata/S669_with_name.csv", batch_size)
# --> modelll
model = make_model("GlobalLocalDualBranch")
optimizer_name = "Adam"
# --> optimizer and scheduler
optimizer =  make_optimizer(model, lr, optimizer_name=optimizer_name)
scheduler =  make_scheduler(optimizer, lr, training_loader, epochs, scheduler_name="CosineLR")
# --> load
if load_checkpoint:
    checkpoint_path = "checkpoint/best_model_rmse_final_ourmodel.pt
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
model.to(device)
# --> train
do_train(model,
         optimizer,
         scheduler,
         training_loader,
         testing_loader,
         device,
         epochs)
# do_test(model, testing_loader,device)

# import pandas as pd
# import numpy as np
#
# # 1. 读取原始数据
# df = pd.read_csv("dTm/S571_GLAD_predicted.csv")
#
# # 2. 得到预测结果
# labels, predictions = do_test(model, testing_loader, device)
#
# # 3. 安全检查（非常重要）
# if len(predictions) != len(df):
#     raise ValueError(f"长度不匹配！predictions={len(predictions)}, df={len(df)}")
#
# # 4. 写入 predicted 列
# df["predicted"] = predictions
#
# # 5. 保存
# df.to_csv("ddgdata/ddG_cleaned/S783_GLAD_predicted.csv", index=False)
#
# print("保存完成：S669_with_predicted.csv")


# import pandas as pd
# import json
#
# df = pd.read_csv("dTm/S571_GLAD_predicted.csv")
#
# labels, predictions, attention_records = do_test(
#     model,
#     testing_loader,
#     device,
#     top_k=10
# )
#
# if len(predictions) != len(df):
#     raise ValueError(f"长度不匹配！predictions={len(predictions)}, df={len(df)}")
#
# if len(attention_records) != len(df):
#     raise ValueError(f"长度不匹配！attention_records={len(attention_records)}, df={len(df)}")
#
# df["predicted"] = predictions
#
# df["top_residue_indices_0based"] = [
#     json.dumps(x["top_residue_indices_0based"])
#     for x in attention_records
# ]
#
# df["top_residue_indices_1based"] = [
#     json.dumps(x["top_residue_indices_1based"])
#     for x in attention_records
# ]
#
# df["wt_attention"] = [
#     json.dumps(x["wt_attention"])
#     for x in attention_records
# ]
#
# df["mut_attention"] = [
#     json.dumps(x["mut_attention"])
#     for x in attention_records
# ]
#
# df["delta_attention"] = [
#     json.dumps(x["delta_attention"])
#     for x in attention_records
# ]
#
# df["attention_ratio"] = [
#     json.dumps(x["attention_ratio"])
#     for x in attention_records
# ]
#
# df["positive_negative"] = [
#     json.dumps(x["positive_negative"])
#     for x in attention_records
# ]
#
# df["positive_ratio"] = [
#     x["positive_ratio"]
#     for x in attention_records
# ]
#
# df.to_csv("dTm/S571_GLAD_predicted_attention.csv", index=False)
#
# print("保存完成：ddgdata/S669_GLAD_predicted_attention.csv")
