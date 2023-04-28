#!/bin/bash
#SBATCH --job-name=16card
#SBATCH --partition=nlp
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=output/step3-models/13b/multinode.log

MASTER_PORT=30123
RDV_ADDR=$(hostname)
WORLD_SIZE=$SLURM_JOB_NUM_NODES

ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6


srun torchrun --nproc_per_node=8 \
   --nnodes=4 \
   --rdzv_id=$SLURM_JOB_ID \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$RDV_ADDR \
    main.py \
   --multinode \
   --data_path Dahoas/rm-static  \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 4 \
   --per_device_mini_train_batch_size 4 \
   --generation_batch_numbers 4 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 100 \
   --offload_reference_model \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --critic_gradient_checkpointing \
   --output_dir $OUTPUT \
   

