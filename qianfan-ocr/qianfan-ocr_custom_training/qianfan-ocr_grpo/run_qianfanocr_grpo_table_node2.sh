export  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
export  NPROC_PER_NODE=8
export  NNODES=2 
export  NODE_RANK=1
export  MASTER_ADDR=10.1.57.115
export  MASTER_PORT=8004 
export NCCL_SOCKET_IFNAME=eth0 
export NCCL_IB_DISABLE=1 
export RDZV_BACKEND=c10d



swift rlhf \
    --rlhf_type grpo \
    --model /mnt/QianfanOCR/ \
    --model_type internvl3_5 \
    --external_plugins /mnt/reward/qianfanocr_table_reward_plugin_html.py \
    --reward_funcs  external_qianfanocr_table \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 10.1.97.240  \
    --vllm_server_port 8003 \
    --dataset /mnt/dataset/reward/qianfanocr_grpo_dataset.jsonl \
    --load_from_cache_file false \
    --max_length 32768 \
    --max_completion_length 32768 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --output_dir /mnt/qianfanocr_grpo \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 16 \
    --temperature 1.0 \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --async_generate true \
    --report_to tensorboard \
    --beta 0.001 \
    --max_grad_norm 0.5
