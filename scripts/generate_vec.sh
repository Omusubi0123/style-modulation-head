gpu=${1:-0}
trait=${2:-"loser"}
# model=${3:-"Qwen/Qwen2.5-7B-Instruct"}
model=${3:-"meta-llama/Llama-3.1-8B-Instruct"}

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu uv run python eval/eval_persona.py \
    --model $model \
    --trait $trait \
    --output_path data/eval_persona_extract/$model/${trait}_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name $trait \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract \
    --max_concurrent_judges 4

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu uv run python eval/eval_persona.py \
    --model $model \
    --trait $trait \
    --output_path data/eval_persona_extract/$model/${trait}_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract \
    --max_concurrent_judges 4

PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu uv run python src/generate_vec.py \
    --model_name $model \
    --pos_path data/eval_persona_extract/$model/${trait}_pos_instruct.csv \
    --neg_path data/eval_persona_extract/$model/${trait}_neg_instruct.csv \
    --trait $trait  \
    --save_dir data/persona_vectors/$model/ \
    --threshold 50
