export NCCL_SOCKET_IFNAME=eth0 
export NCCL_IB_DISABLE=1  

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rollout \
    --model_type internvl3_5 \
    --model /mnt/QianfanOCR \
    --vllm_tensor_parallel_size 1 \
    --vllm_data_parallel_size 8 \
    --temperature 1.0 \
    --port 8003
