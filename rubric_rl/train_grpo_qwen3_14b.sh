#!/bin/bash
# GRPO training for Qwen3-14B rubric generator using verl.
# Prompt: strongmodel_qwen14b (2-message, 4 few-shot, single-tool constraint). Data: data_qwen14b_v2 (token-truncated).
#
# Usage:
#   bash train_grpo_qwen3_14b.sh                    # full training
#   bash train_grpo_qwen3_14b.sh --test             # small test run (2 epochs)
#   bash train_grpo_qwen3_14b.sh trainer.total_epochs=5  # override any param

set -x

# Workaround for vLLM torch.compile FakeTensorMode bug with torch 2.10
export VLLM_TORCH_COMPILE_LEVEL=0

# ============================================================================
# Paths
# ============================================================================
PROJECT_DIR="/root/storage/kjz/dr-tulu/rubric_rl"
MODEL_PATH="/root/storage/kjz/dr-tulu/models/Qwen3-14B"
TRAIN_DATA="${PROJECT_DIR}/data_qwen14b_v2/train.parquet"
EVAL_DATA="${PROJECT_DIR}/data_qwen14b_v2/eval.parquet"
REWARD_FN="${PROJECT_DIR}/reward_fn.py"
OUTPUT_DIR="${PROJECT_DIR}/checkpoints_qwen3_14b_v2"

# ============================================================================
# wandb
# ============================================================================
export WANDB_API_KEY="${WANDB_API_KEY:?Please set WANDB_API_KEY in your environment or .env file}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="rubric-rl"

# ============================================================================
# Test mode
# ============================================================================
EXTRA_ARGS=""
if [[ "$1" == "--test" ]]; then
    echo "=== TEST MODE: 2 epochs, small batch ==="
    EXTRA_ARGS="trainer.total_epochs=2 data.train_batch_size=64 actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 trainer.save_freq=999 trainer.test_freq=5"
    shift
fi

# ============================================================================
# Logging to file
# ============================================================================
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_qwen3_14b_${TIMESTAMP}.log"
echo "=== Logging to ${LOG_FILE} ==="

# ============================================================================
# Launch GRPO training
# ============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${EVAL_DATA} \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.repetition_penalty=1.1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    reward.custom_reward_function.path=${REWARD_FN} \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rubric-rl' \
    trainer.experiment_name='qwen3-14b-rubric-v2' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=2 \
    trainer.save_freq=10 \
    trainer.test_freq=3 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    ${EXTRA_ARGS} "$@" 2>&1 | tee ${LOG_FILE}
