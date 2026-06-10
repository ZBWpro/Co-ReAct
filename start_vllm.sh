#!/bin/bash
# DR-Tulu vLLM 模型服务启动脚本
# 同时启动 Search Agent (DR-Tulu-8B) 和 Browse Agent (Qwen3-8B)
# 参数与官方保持一致

MODEL_DIR="/root/storage/kjz/dr-tulu/models"
SEARCH_MODEL="${MODEL_DIR}/DR-Tulu-8B"
BROWSE_MODEL="${MODEL_DIR}/Qwen3-8B"

SEARCH_PORT=30001
BROWSE_PORT=30002

echo "=========================================="
echo "  DR-Tulu vLLM 模型服务启动"
echo "=========================================="
echo ""
echo "Search Agent: ${SEARCH_MODEL} → port ${SEARCH_PORT} (GPU 0)"
echo "Browse Agent: ${BROWSE_MODEL} → port ${BROWSE_PORT} (GPU 1)"
echo ""

# 启动 Search Agent (DR-Tulu-8B) - GPU 0
echo "[1/2] 启动 Search Agent (DR-Tulu-8B) on GPU 0, port ${SEARCH_PORT}..."
CUDA_VISIBLE_DEVICES=0 vllm serve "${SEARCH_MODEL}" \
    --dtype auto \
    --port ${SEARCH_PORT} \
    --max-model-len 40960 \
    > /tmp/vllm_search_${SEARCH_PORT}.log 2>&1 &
SEARCH_PID=$!
echo "  PID: ${SEARCH_PID}, 日志: /tmp/vllm_search_${SEARCH_PORT}.log"

# 启动 Browse Agent (Qwen3-8B) - GPU 1
echo "[2/2] 启动 Browse Agent (Qwen3-8B) on GPU 1, port ${BROWSE_PORT}..."
CUDA_VISIBLE_DEVICES=1 vllm serve "${BROWSE_MODEL}" \
    --dtype auto \
    --port ${BROWSE_PORT} \
    --max-model-len 40960 \
    > /tmp/vllm_browse_${BROWSE_PORT}.log 2>&1 &
BROWSE_PID=$!
echo "  PID: ${BROWSE_PID}, 日志: /tmp/vllm_browse_${BROWSE_PORT}.log"

echo ""
echo "=========================================="
echo "  等待模型加载完成（可能需要几分钟）..."
echo "=========================================="

# 等待两个服务都启动
wait_for_port() {
    local port=$1
    local name=$2
    local max_wait=300  # 最多等 5 分钟
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "✅ ${name} 已就绪 (port ${port})"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "⏳ ${name} 仍在加载中... (${elapsed}s)"
        fi
    done

    echo "❌ ${name} 启动超时"
    return 1
}

wait_for_port ${SEARCH_PORT} "Search Agent (DR-Tulu-8B)" &
WAIT_SEARCH=$!
wait_for_port ${BROWSE_PORT} "Browse Agent (Qwen3-8B)" &
WAIT_BROWSE=$!

wait $WAIT_SEARCH
SEARCH_OK=$?
wait $WAIT_BROWSE
BROWSE_OK=$?

echo ""
if [ $SEARCH_OK -eq 0 ] && [ $BROWSE_OK -eq 0 ]; then
    echo "=========================================="
    echo "  🎉 两个模型服务都已启动成功！"
    echo "=========================================="
    echo ""
    echo "接下来请在另一个终端运行："
    echo "  conda activate dr_agent"
    echo "  cd /root/storage/kjz/dr-tulu/agent"
    echo "  python workflows/auto_search_sft.py serve --port 8080"
    echo ""
    echo "然后浏览器访问 http://<服务器IP>:8080"
    echo ""
    echo "按 Ctrl+C 停止所有模型服务"
else
    echo "⚠️  部分服务启动失败，请检查日志："
    echo "  Search Agent: /tmp/vllm_search_${SEARCH_PORT}.log"
    echo "  Browse Agent: /tmp/vllm_browse_${BROWSE_PORT}.log"
fi

# 等待前台，Ctrl+C 时杀掉所有后台进程
trap "echo '正在停止模型服务...'; kill $SEARCH_PID $BROWSE_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
