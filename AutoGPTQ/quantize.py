from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging


import torch


import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--tasks",
    type=str,
    default="humaneval_instruct",
    help="The tasks to evaluate on, separated by commas.",
)
parser.add_argument(
    "--num_fewshot",
    type=int,
    default=0,
    help="The number of few-shot examples to use for each task.",
)
parser.add_argument(
    "--limit",
    type=int,
    default=-1,
    help="The number of examples to evaluate on for each task. -1 means no limit.",
)
parser.add_argument(
    "--model",
    type=str,
    default="/root/dlm/model/llada-8b-base",
    help="The model to use for evaluation. Default is 'llada_dist'.",
)
parser.add_argument(
    "--wbits",
    type=int,
    default=4,
    help="The number of bits to quantize the model to. Default is 4.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=1024,
    help="The number of steps to run the model for. Default is 128.",
)
parser.add_argument(
    "--mc_num",
    type=int,
    default=128,
    help="The number of Monte Carlo samples to use for evaluation. Default is 1.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32
)
parser.add_argument(
    "--gen_length",
    type=int,
    default=1024,
    help="The number of tokens to generate for each example. Default is 1024.",
)
parser.add_argument(
    "--block_length",
    type=int,
    default=1024,
    help="The number of tokens to generate in each block. Default is 1024.",
)

parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=768,
    help="The number of tokens to generate in each block. Default is 1024.",
)
parser.add_argument(
    "--diffusion_steps",
    type=int,
    default=512,
    help="The number of tokens to generate in each block. Default is 1024.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature for Dream diffusion generation.",
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.95,
    help="Nucleus sampling p for Dream diffusion generation.",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=None,
    help="Top-k sampling for Dream diffusion generation (optional).",
)
parser.add_argument(
    "--alg",
    type=str,
    default="entropy",
    help="Remasking strategy for Dream diffusion generation.",
)
parser.add_argument(
    "--alg_temp",
    type=float,
    default=0.0,
    help="Temperature for the remasking strategy.",
)
parser.add_argument(
    "--escape_until",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If False, post-cut generations by task stop strings.",
)
parser.add_argument(
    "--add_bos_token",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to prepend BOS token before generation.",
)
parser.add_argument(
    "--apply_chat_template",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Whether to apply chat template. Default: auto-enable for *_instruct tasks.",
)
parser.add_argument(
    "--calib_nsamples",
    type=int,
    default=128,
    help="Number of calibration samples for GPTQ.",
)
parser.add_argument(
    "--calib_seqlen",
    type=int,
    default=2048,
    help="Calibration sequence length for GPTQ.",
)
args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = args.model
quantized_model_dir = "llada-8b-base-4bit"

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    import numpy as np

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True, trust_remote_code=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=args.wbits,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, trust_remote_code=True)


traindataset, testenc = get_wikitext2(args.calib_nsamples, 0, args.calib_seqlen, pretrained_model_dir)
model.quantize(traindataset)
model = model.to("cuda")
model.eval()

from lm_eval.api.registry import get_model
class_name = model.__class__.__name__.lower()
if 'llada' in class_name:
    model_cls = get_model('llada_dist')
    model_args = dict(
        steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=0., cfg_scale=0., remasking='low_confidence', mc_num=args.mc_num, batch_size=args.batch_size
    )
    model = model_cls(model=model, model_path=pretrained_model_dir, **model_args)
elif 'dream' in class_name or 'dream' in pretrained_model_dir.lower():
    model_cls = get_model('dream_base')
    model_args = dict(
        diffusion_steps=args.diffusion_steps,
        max_new_tokens=args.max_new_tokens,
        mc_num=args.mc_num,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
        escape_until=args.escape_until,
        add_bos_token=args.add_bos_token,
    )
    model = model_cls(model=model, pretrained=pretrained_model_dir, **model_args)
else:
    raise NotImplementedError


from lm_eval import evaluator

results = {}
task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
if args.apply_chat_template is None:
    apply_chat_template = any("instruct" in task for task in task_names)
else:
    apply_chat_template = args.apply_chat_template

t_results = evaluator.simple_evaluate(
    model,
    tasks=task_names,
    num_fewshot=args.num_fewshot,
    limit=None if args.limit == -1 else args.limit,
    model_args=model_args,
    apply_chat_template=apply_chat_template,
    confirm_run_unsafe_code=True,
)
results.update(t_results)
# print(results.keys())
print(args)
print(results['results'])

# save results
import json
import os
if not os.path.exists('results'):
    os.makedirs('results')
with open(f'results/{args.model.split("/")[-1]}-{args.wbits}w.json', 'a') as f:
    # check whether could write as json
    try:
        json.dump(results['results'], f)
    except Exception as e:
        print(f"Error writing results to {f}: {e}")
