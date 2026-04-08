NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=2,3 \
swift sft \
    --model /mnt/QianfanOCR/  \
    --model_type internvl3_5 \
    --load_from_cache_file true \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 1e-4 \
    --gradient_checkpointing true \
    --gradient_accumulation_steps 2 \
    --save_steps 500 \
    --save_total_limit 20 \
    --logging_steps 5 \
    --max_length 32768 \
    --max_new_tokens 32768 \
    --output_dir /mnt/qianfanocr_sft \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --dataset /mnt/dataset/sft/qianfanocr_sft_dataset.jsonl
