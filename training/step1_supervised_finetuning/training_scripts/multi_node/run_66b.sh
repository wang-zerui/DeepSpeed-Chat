#!/bin/bash
#SBATCH --job-name=100B        # name
#SBATCH -p nlp
#SBATCH --nodes=4                    # nodes
#SBATCH --ntasks-per-node=8          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8           # number of cores per tasks
#SBATCH --gpus-per-task=1
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/mnt/petrelfs/wangzerui/DeepSpeed/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/66b/training.log          # output file name

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export MASTER_PORT=12340
export WORLD_SIZE=32

mkdir -p $OUTPUT

srun python main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP \
   --data_split 2,4,4 \
   --model_name_or_path /mnt/petrelfs/wangzerui/DeepSpeed/DeepSpeedExamples/applications/DeepSpeed-Chat/llama_model/100B \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3\
   --lora_dim 128 \
   --lora_module_name layers. \
   --deepspeed \
   --output_dir /mnt/petrelfs/wangzerui/DeepSpeed/DeepSpeedExamples/applications/DeepSpeed-Chat/output/actor-models/66b
