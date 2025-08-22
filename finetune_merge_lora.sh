CUDA_VISIBLE_DEVICES=1 python scripts/merge_lora_weights.py --model-path checkpoints/llava-v1.5-7b-task-lora-0626-pair --model-base 'liuhaotian/llava-v1.5-7b' --save-model-path /tmp2/ycliang/LLaVA/fine_tune_llava_lora_0626-pair-new

