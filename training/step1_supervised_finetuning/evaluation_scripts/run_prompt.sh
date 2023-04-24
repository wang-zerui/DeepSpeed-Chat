#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py \
    --model_name_or_path_baseline /mnt/petrelfs/wangzerui/DeepSpeed/DeepSpeedExamples/applications/DeepSpeed-Chat/llama_model/7132v2 \
    --model_name_or_path_finetune /mnt/petrelfs/wangzerui/DeepSpeed/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/7b \
