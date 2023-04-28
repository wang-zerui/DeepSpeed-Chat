#!/bin/bash
#SBATCH --job-name=16card
#SBATCH --partition=nlp
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=output/actor-models/13b/multinode.log

MASTER_PORT=30123
RDV_ADDR=$(hostname)
WORLD_SIZE=$SLURM_JOB_NUM_NODES

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

srun torchrun --nproc_per_node=8 \
   --nnodes=4 \
   --rdzv_id=$SLURM_JOB_ID \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$RDV_ADDR \
    main.py \
   --multinode \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP \
   --data_split 2,4,4 \
   --model_name_or_path /mnt/petrelfs/wangzerui/DeepSpeed/DeepSpeedExamples/applications/DeepSpeed-Chat/llama_model/7132k \
   --per_device_train_batch_size 6 \
   --per_device_eval_batch_size 6 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0.1 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log



