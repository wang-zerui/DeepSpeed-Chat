srun -p nlp --gres=gpu:8 --async python train.py \
--step=2 \
--actor-model=7b \
--reward-model=13b \
--actor-zero-stage=3 \
--num-gpus=8

python train.py \
--step=1 \
--actor-model=66b \
--actor-zero-stage=3 \
--num-gpus=32 \
&> train.py.log