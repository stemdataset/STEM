#!/bin/bash

model=ViT-L/14@336px
pic_size=336
eval_split=$1
python -m src.eval_clip \
  --model_name ${model} \
  --pic_size ${pic_size} \
  --eval_split "${eval_split}" \
  --output_dir "results/clip_${model}_${eval_split}"
