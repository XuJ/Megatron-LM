#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/5/7 10:09
# @Author  : jiaoxu
# @File    : parse_logging_file.py
# @Software: PyCharm

import logging
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log_dir = "/home/zjlab/log"
log_file1 = "pretrain_gpt_distributed_with_mp_20210430.log"
log_file2 = "pretrain_gpt_distributed_with_mp_20210506.log"
fig_file = "test.jpg"

iterations = []
elapse_time_per_iters = []
learning_rates = []
lm_losses = []
loss_scales = []
grad_norms = []

val_iterations = []
val_losses = []
val_ppls = []

for log_file in [log_file1, log_file2]:
    with open(os.path.join(log_dir, log_file), "r", encoding="utf8") as log_fh:
        for i, line in enumerate(log_fh):
            line = line.strip()
            if line.startswith("iteration"):
                line_split_length = len(line.split("|"))
                if line_split_length not in [9, 11]:
                    print("ERROR: length of line split not equal to 9 or 11")
                    print(line_split_length, line)
                else:
                    split_line = line.split("|")
                    iteration = int(split_line[0].strip().split(" ")[-3].split("/")[0])
                    elapsed_time_per_iteration = float(split_line[2].strip().split(" ")[-1].strip())
                    learning_rate = float(split_line[3].strip().split(" ")[-1].strip())
                    if line_split_length == 9:
                        lm_loss = np.nan
                        loss_scale = float(split_line[5].strip().split(" ")[-1].strip())
                        grad_norm = np.nan
                    else:
                        lm_loss = float(split_line[5].strip().split(" ")[-1].strip())
                        loss_scale = float(split_line[6].strip().split(" ")[-1].strip())
                        grad_norm = float(split_line[7].strip().split(" ")[-1].strip())
                    iterations.append(iteration)
                    elapse_time_per_iters.append(elapsed_time_per_iteration)
                    learning_rates.append(learning_rate)
                    lm_losses.append(lm_loss)
                    loss_scales.append(loss_scale)
                    grad_norms.append(grad_norm)
            elif line.startswith("validation loss"):
                line_split_length = len(line.split("|"))
                if line_split_length != 4:
                    print("ERROR: length of line split not equal to 4")
                    print(line_split_length, line)
                else:
                    split_line = line.split("|")
                    iteration = int(split_line[0].strip().split("iteration")[-1].strip())
                    lm_loss = float(split_line[1].strip().split(":")[-1].strip())
                    lm_loss_ppl = float(split_line[2].strip().split(":")[-1].strip())
                val_iterations.append(iteration)
                val_losses.append(lm_loss)
                val_ppls.append(lm_loss_ppl)

train_df = pd.DataFrame({
    "iter": iterations,
    "time": elapse_time_per_iters,
    "lr": learning_rates,
    "train_loss": lm_losses,
    "train_loss_scale": loss_scales,
    "train_grad_norm": grad_norms
})
train_df.drop_duplicates(subset=["iter"], keep="last", inplace=True)
validation_df = pd.DataFrame({
    "iter": val_iterations,
    "val_loss": val_losses,
    "val_ppl": val_ppls
})
validation_df.drop_duplicates(subset=["iter"], keep="last", inplace=True)
# train_df = train_df[(train_df["iter"]<=10000)&(train_df["iter"]>=800)]
# validation_df = validation_df[(validation_df["iter"]<=10000)&(validation_df["iter"]>=800)]

fig = plt.figure(1, figsize=(16,9))
plt.subplot(211)
plt.plot(train_df["iter"], train_df["train_loss"], label="train")
plt.plot(validation_df["iter"], validation_df["val_loss"], label="validation")
plt.ylabel("lm_loss")
plt.legend()
plt.subplot(212)
plt.plot(validation_df["iter"], validation_df["val_ppl"])
plt.ylabel("perplexity")
plt.xlabel("iteration")
plt.savefig(os.path.join(log_dir, fig_file))
plt.close("all")