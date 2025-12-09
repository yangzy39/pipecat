export BIN="/nas-wulanchabu/miniconda3/envs/matpo/bin"
# === 修改点 1: 更新模型路径 ===
export MODEL_CKPT=/nas-wulanchabu/yitong.yzy/models/Qwen-2.5-7B-Instruct-F


MASTER_IP=$(ip addr show eth0 | grep "inet " | awk '{print $2}' | cut -d'/' -f1)
echo "Head node IP address is: ${MASTER_IP}"

echo "Starting vLLM with Ray backend..."


${BIN}/vllm serve ${MODEL_CKPT} \
    --served-model-name Qwen-2.5-7B-Instruct-F \
    --host 0.0.0.0 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --port 23547 
