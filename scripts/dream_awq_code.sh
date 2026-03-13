export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='/mnt/public_models/dream-7b-instruct/'
export PYTHONPATH="$DIRPATH/lm-evaluation-harness:${PYTHONPATH}"

# AWQ Code (HumanEval) - Changed w_bit to 4
python $DIRPATH/llm-awq/entry.py \
    --model_path $MODEL_PATH \
    --w_bit 4 \
    --q_group_size 128 \
    --run_awq \
    --tasks humaneval_instruct_noprefix \
    --num_fewshot 0 \
    --batch_size 1 \
    --max_new_tokens 128 \
    --diffusion_steps 128 \
    --temperature 0.08 \
    --top_p 0.9 \
    --alg entropy \
    --alg_temp 0.0 \
    --escape_until \
    --add_bos_token \
    --apply_chat_template
