# Co-ReAct: Rubric-Guided Action Selection for Deep Research Agents

Co-ReAct extends ReAct's (Reason, Act, Observe) loop to **(Rubric, Reason, Act, Verify, Observe)** by inserting a trained rubric generator before each action and a verification step after. The rubric generator is trained with listwise GRPO using Spearman rank-correlation against multi-judge expert consensus rankings.

## Setup

```bash
cd agent/
pip install -e .
cp .env.example .env   # fill in SERPER_API_KEY, S2_API_KEY, JINA_API_KEY
```

### Start vLLM

```bash
# Search agent
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-14B --dtype auto --port 30001 --max-model-len 40960

# Rubric generator (GRPO-trained)
CUDA_VISIBLE_DEVICES=1 vllm serve <your-rubric-model> --dtype auto --port 30005 --max-model-len 8192
```

## Usage

### 1. Data Collection

Collect candidate actions at branching points and rank via multi-judge Borda count:

```bash
cd agent/
python workflows/collect_rl_data.py generate-dataset \
    --config workflows/collect_rl_data.yaml \
    --output rl_data/branching_points.jsonl \
    --max-concurrent 20
```

### 2. Train Rubric Generator (GRPO)

```bash
cd rubric_rl/
python prepare_data.py --input ../rl_data/branching_points.jsonl --output data/train.parquet
bash train_grpo_qwen3_14b.sh
```

### 3. Co-ReAct Inference & Evaluation

```bash
cd agent/

# SQA-CS-V2
python workflows/co_react_eval.py generate-dataset sqa \
    --config workflows/co_react_eval.yaml \
    --output eval_output/co_react_sqa.jsonl \
    --max-concurrent 20
python scripts/evaluate.py sqa eval_output/co_react_sqa.jsonl

# DeepResearchBench (DRB)
python workflows/co_react_eval.py generate-dataset drb \
    --config workflows/co_react_eval.yaml \
    --output eval_output/co_react_drb.jsonl \
    --max-concurrent 20
python scripts/evaluate.py drb eval_output/co_react_drb.jsonl
```

## Repository Structure

```
co-react/
├── agent/
│   ├── dr_agent/              # Core agent library
│   │   ├── rubric_generator.py    # Rubric generation + verification
│   │   ├── judges.py              # Multi-judge Borda ranking
│   │   ├── client.py              # LLM client
│   │   ├── mcp_backend/           # MCP tool server (search, browse)
│   │   └── tool_interface/        # Tool abstraction
│   ├── workflows/
│   │   ├── co_react_eval.py       # Co-ReAct inference loop
│   │   └── collect_rl_data.py     # Branching point data collection
│   ├── evaluation/                # Benchmark eval (DRB, SQA-CS-V2)
│   └── scripts/
│       ├── evaluate.py
│       └── launch_chat.py
├── rubric_rl/                     # Rubric generator GRPO training
│   ├── reward_fn.py               # Spearman-based reward
│   ├── rubric_judge.py            # Rubric-guided action ranking
│   ├── prepare_data.py            # Data preparation
│   └── train_grpo_qwen3_14b.sh   # Training script
└── start_vllm.sh
```

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@misc{kang2026co,
  title={Co-ReAct: Rubrics as Step-Level Collaborators for ReAct Agents},
  author={Kang, Jiazheng and Zhang, Bowen and Song, Zixin and Chen, Jiangwang and Yang, Xiao and Zhu, Da and Jiang, Guanjun},
  year={2026},
  eprint={2605.23590},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2605.23590},
}
```