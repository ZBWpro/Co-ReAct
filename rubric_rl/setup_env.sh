#!/bin/bash
# Setup environment for rubric GRPO training with verl
#
# Usage:
#   bash setup_env.sh

set -e

ENV_NAME="dr-agent-verl"
VERL_DIR="/root/storage/kjz/dr-tulu/verl"

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -n ${ENV_NAME} python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "=== Cloning verl (main branch) ==="
if [ ! -d "${VERL_DIR}" ]; then
    git clone https://github.com/volcengine/verl.git ${VERL_DIR}
else
    echo "verl already cloned at ${VERL_DIR}, pulling latest..."
    cd ${VERL_DIR} && git pull
fi

echo "=== Installing verl (--no-deps, official method) ==="
cd ${VERL_DIR}
pip install --no-deps -e .

echo "=== Installing dependencies ==="
# Core RL training deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install vllm transformers accelerate deepspeed
pip install ray[default]
# verl runtime deps
pip install codetiming hydra-core omegaconf pydantic
# Reward function deps
pip install scipy aiohttp pandas pyarrow

echo "=== Verifying installation ==="
python -c "import verl; print('verl installed successfully')"
python -c "import scipy; print('scipy installed successfully')"
python -c "import pandas; print('pandas installed successfully')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Prepare data:  conda activate ${ENV_NAME} && cd /root/storage/kjz/dr-tulu/rubric_rl && python prepare_data.py"
echo "  2. Train:         python train_grpo.py"
echo ""
echo "  Cloudsway Gemini 2.5 Pro is pre-configured (no API key needed)."
echo "  To override: export CLOUDSWAY_GEMINI_URL=... and CLOUDSWAY_API_KEY=..."
echo ""
echo "For a test run:     python train_grpo.py --test"
