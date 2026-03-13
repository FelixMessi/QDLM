from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
# from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from torch import nn
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
# parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
# parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)

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
    default="/home/yichen/dlm/model/llada-8b-base",
    help="The model to use for evaluation. Default is 'llada_dist'.",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument(
    "--dump_fake", type=str, default=None, help="save fake-quantized model"
)
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
parser.add_argument(
    "--vila-15",
    action="store_true",
    help="quantizing vila 1.5",
)
parser.add_argument(
    "--vila-20",
    action="store_true",
    help="quantizing or smoothing vila 2.0 (NVILA)",
)
parser.add_argument(
    "--smooth_scale",
    action="store_true",
    help="generate the act scale of visiontower",
)
parser.add_argument(
    "--media_path",
    type=str,
    nargs="+",
    help="The input video to get act scale for visiontower",
)
parser.add_argument(
    "--act_scale_path",
    type=str,
    default=None,
    help="Path to save act scale",
)
parser.add_argument(
    "--mc_num",
    type=int,
    default=128,
    help="The number of Monte Carlo samples for evaluation.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=1024,
    help="The number of steps to run for evaluation.",
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
    "--awq_n_samples",
    type=int,
    default=128,
    help="Number of calibration samples used by AWQ search.",
)
parser.add_argument(
    "--awq_seqlen",
    type=int,
    default=512,
    help="Calibration sequence length used by AWQ search.",
)
args = parser.parse_args()
assert (
    args.act_scale_path is not None and len(args.media_path) > 0
) or not args.smooth_scale
vila_10_quant_mode = (
    ("llava" in args.model_path.lower() or "vila" in args.model_path.lower())
    and not args.vila_15
    and not args.vila_20
)

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer


def build_model_and_enc(model_path, dtype):
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    if vila_10_quant_mode:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False},
        )
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
        config.use_cache = False
        if "mpt" in config.__class__.__name__.lower():
            enc = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            enc = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, trust_remote_code=True
            )

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch_dtype, trust_remote_code=True
            )
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights()

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock",
                "DreamDecoderLayer",
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()
    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
        if not vila_10_quant_mode:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                model_path, config=config, trust_remote_code=True, **kwargs
            )

        model.eval()

        if args.run_awq:
            # assert args.dump_awq, "Please save the awq results with --dump_awq"
            # import ipdb; ipdb.set_trace()
            awq_results = run_awq(
                model,
                enc,
                w_bit=args.w_bit,
                q_config=q_config,
                n_samples=args.awq_n_samples,
                seqlen=args.awq_seqlen,
            )
            # import ipdb; ipdb.set_trace()
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                os.makedirs(dirpath, exist_ok=True)

                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)

        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model, awq_results)

        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert (
                    args.dump_quant is None
                ), "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_fake:
                    model.save_pretrained(args.dump_fake)
                    print("Pseudo-quantized models saved at", args.dump_fake)
            elif args.q_backend == "real":  # real quantization
                real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_quant:
                    if not args.dump_quant.endswith("v2.pt"):
                        print("[Info] Auto-change the dump_quant file name to *v2.pt")
                        args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
                    dirpath = os.path.dirname(args.dump_quant)
                    os.makedirs(dirpath, exist_ok=True)

                    print(f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock",
                "DreamDecoderLayer",
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)
    return model, enc


def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    # a hack here to auto set model group
    if args.smooth_scale and args.vila_20:
        if os.path.exists(args.act_scale_path):
            print(f"Found existing Smooth Scales {args.act_scale_path}, skip.")
        else:
            from awq.quantize import get_smooth_scale

            act_scale = get_smooth_scale(args.model_path, args.media_path)
            os.makedirs(os.path.dirname(args.act_scale_path), exist_ok=True)
            torch.save(act_scale, args.act_scale_path)
            print("Save act scales at " + str(args.act_scale_path))
            args.model_path = args.model_path + "/llm"
        if args.dump_awq is None and args.dump_quant is None:
            exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()
    model, enc = build_model_and_enc(args.model_path, args.dtype)

    from lm_eval.api.registry import get_model
    class_name = model.__class__.__name__.lower()
    model_path_lc = args.model_path.lower()
    if 'llada' in class_name or 'llada' in model_path_lc:
        model_cls = get_model('llada_dist')
        model_args = dict(
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=0., cfg_scale=0., remasking='low_confidence', mc_num=args.mc_num, batch_size=args.batch_size
        )
        model = model_cls(model=model, model_path=args.model_path, **model_args)
    elif 'dream' in class_name or 'dream' in model_path_lc:
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
        model = model_cls(model=model, pretrained=args.model_path, **model_args)
    else:
        raise NotImplementedError(f"Unsupported model class for lm_eval wrapper: {model.__class__.__name__}")

    

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
    print(args)
    print(results['results'])

    # save results
    import json
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    with open(f'results/{args.model_path.split("/")[-1]}-{args.w_bit}w.json', 'a') as f:    
        try:
            json.dump(results['results'], f)
        except Exception as e:
            print(f"Error writing results to {f}: {e}")

    


if __name__ == "__main__":
    main()
