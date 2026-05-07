**Work in Progress:** This repository is still undergoing a cleanup process for public release. The files here represent a functional core of the project, but this is not the full or final version of the codebase.

# Contextual Influence & Scoring

This repository contains the data and evaluation scripts used to measure contextual influence and token-level log-probability deltas across different text generation tasks. The project evaluates Large Language Models (LLMs) across three distinct domains: **Essays**, **LLM Reviews**, and **Reference Letters**.

## 📂 Project Structure

The repository is divided into three main domains, each containing its respective dataset and a dedicated scoring script tailored to its specific task context:
```
.
├── Essays/
│   ├── Data/
│   └── score_essays.py
├── LLM_Reviews/
│   ├── 500_Reviews/
│   │   └── {year}/
│   │       ├── Data/
│   └── {model_name}_bullet_reviews.json  # Exception: kept at the year level
│   └── score_reviews.py
└── Reference_Letters/
    ├── Data/
    └── score_letters.py
```

### Generic Instructions ($C$)
While the three scoring scripts (`score_essays.py`, `score_reviews.py`, `score_letters.py`) share a similar core engine for computing log-probabilities and extracting hint spans, they differ in how they handle context. Each script specifically accounts for differences in the **Generic Instructions ($C$)**. As defined in our corresponding paper, $C$ represents the task-specific framing injected into the context (e.g., *"You are an expert reviewer for the ICLR Conference..."*).

## ⚙️ Setup and Installation

This codebase relies on PyTorch and the Hugging Face ecosystem. To install the required dependencies, run:
```bash
pip install torch transformers scikit-learn nltk numpy tqdm
```
*(Note: If you are using the `keywords` or `extractive` hint methods, the script will automatically download necessary NLTK data during execution).*

## 🚀 Usage

### 1. Running a Single Job Locally
You can run the scoring engine directly via Python. The script extracts baseline log-probabilities and compares them against "hinted" log-probabilities using various ablation methods (e.g., `random_spans`, `surprisal_spans`, `prefix`).

Here is an example of evaluating a specific reviewer (Claude-Sonnet-3.5) from the 2019 LLM Reviews dataset using the `surprisal_spans` method:
```bash
python LLM_Reviews/score_reviews.py \
  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --input_file "LLM_Reviews/500_Reviews/2019/Data/generated_and_rewritten_reviews.json" \
  --output_file "Scores/reviewer_0_ratio0.2.json" \
  --ratio 0.2 \
  --hint_method "surprisal_spans" \
  --span_length 5 \
  --max_seq_len 25000 \
  --target_reviewer 0
```

### 2. Running Batch Experiments (SLURM)
For large-scale evaluations across multiple ablation ratios and hint methods, we provide a SLURM array script. This script dynamically calculates the target reviewer, ratio, and method based on the `$SLURM_ARRAY_TASK_ID`.

To run a massive sweep across 12 reviewers, 4 ratios, and 3 methods, you can submit an array job of 144 tasks (`0-143`):
```bash
#!/bin/bash
#SBATCH --job-name=llm_scoring
#SBATCH --array=0-143
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1

# --- CONFIGURATION ---
ratios=(0.2 0.3 0.4 0.5)
methods=("random_spans" "surprisal_spans" "prefix")
SPAN_LENGTH=5
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
BENCHMARK_FILE="LLM_Reviews/500_Reviews/2019/Data/generated_and_rewritten_reviews.json"

# Total number of reviewers to iterate over
NUM_TOTAL_REVIEWERS=12

# --- CALCULATE INDICES ---
i=$SLURM_ARRAY_TASK_ID

num_ratios=${#ratios[@]}
num_methods=${#methods[@]}

# 1. Modulo gives us the Reviewer Index (0 to 11)
reviewer_idx=$(( i % NUM_TOTAL_REVIEWERS ))
# 2. Division & Modulo gives the Ratio Index (0 to 3)
ratio_idx=$(( (i / NUM_TOTAL_REVIEWERS) % num_ratios ))
# 3. Double Division gives the Method Index (0 to 2)
method_idx=$(( (i / (NUM_TOTAL_REVIEWERS * num_ratios)) % num_methods ))

current_ratio=${ratios[$ratio_idx]}
current_method=${methods[$method_idx]}

OUT_DIR="Scores/${current_method}"
mkdir -p "$OUT_DIR"
mkdir -p logs

# Output file reflects the index so jobs don't overwrite each other
OUT_FILE="${OUT_DIR}/score_reviewer${reviewer_idx}_ratio${current_ratio}.json"

echo "Executing Task $i: Reviewer $reviewer_idx | Method: $current_method | Ratio: $current_ratio"

python LLM_Reviews/score_reviews.py \
  --model_name "$MODEL" \
  --input_file "$BENCHMARK_FILE" \
  --output_file "$OUT_FILE" \
  --ratio "$current_ratio" \
  --hint_method "$current_method" \
  --span_length "$SPAN_LENGTH" \
  --max_seq_len 25000 \
  --target_reviewer "$reviewer_idx"
```

## 📊 Output Format
The scripts generate a JSON array containing the token-level deltas for each evaluation. Each entry includes the `baseline_logprobs`, the `hinted_logprobs`, and the total `context_length` for downstream analysis and visualization.