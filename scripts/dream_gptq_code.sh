export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='/mnt/public_models/dream-7b-instruct/'
export PYTHONPATH="$DIRPATH/lm-evaluation-harness:${PYTHONPATH}"

# GPTQ Code Test (HumanEval) - Removed the unrecognized flag
python $DIRPATH/AutoGPTQ/quantize.py \
    --model $MODEL_PATH \
    --wbits 4 \
    --tasks humaneval_instruct_noprefix \
    --batch_size 1 \
    --num_fewshot 0 \
    --max_new_tokens 128 \
    --diffusion_steps 128 \
    --temperature 0.08 \
    --top_p 0.9 \
    --alg entropy \
    --alg_temp 0.0 \
    --escape_until \
    --add_bos_token \
    --apply_chat_template
