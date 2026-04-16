import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa,
    load_docred,
    load_cord,
    load_funsd,
    load_chemprot
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
from evaluate_extraction import evaluate_extraction_task, print_evaluation_results
import time

# ── Dataset registry ──────────────────────────────────────────────
# Simple tasks (only need split kwarg)
_SIMPLE_LOADERS = {
    "gsm8k":         lambda args: load_gsm8k(split=args.split),
    "aime2024":      lambda args: load_aime2024(split="train"),
    "aime2025":      lambda args: load_aime2025(split="train"),
    "gpqa":          lambda args: load_gpqa_diamond(split="test"),
    "arc_easy":      lambda args: load_arc_easy(split="test"),
    "arc_challenge": lambda args: load_arc_challenge(split="test"),
    "mbppplus":      lambda args: load_mbppplus(split="test"),
    "humanevalplus": lambda args: load_humanevalplus(split="test"),
    "medqa":         lambda args: load_medqa(split="test"),
}

# Extraction tasks requiring doc_path
_EXTRACTION_TASKS = {"docred", "cord", "funsd", "chemprot"}


def _resolve_extraction_mode(args) -> str:
    """Determine extraction mode from args."""
    if args.extraction_mode == "full":
        return "full"
    if args.prompt == "hierarchical" and args.extraction_mode == "partitioned":
        return "partitioned"
    return "chunks"


def _load_extraction_dataset(args):
    """Load extraction dataset based on task name."""
    if args.task != "chemprot" and not args.doc_path:
        raise ValueError(f"--doc_path is required for {args.task} task")

    mode = _resolve_extraction_mode(args)
    common = dict(
        doc_path=args.doc_path,
        split=args.split,
        mode=mode,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        num_partitions=args.num_partitions,
    )

    if args.task == "docred":
        return load_docred(**common)
    elif args.task == "cord":
        return load_cord(**common, image_path=args.image_path)
    elif args.task == "funsd":
        return load_funsd(
            **common,
            image_path=args.image_path,
            annotations_dir=args.annotations_dir,
            images_dir=args.image_dir,
        )
    elif args.task == "chemprot":
        return load_chemprot(split=args.split)
    else:
        raise ValueError(f"Unknown extraction task: {args.task}")


def load_dataset_for_task(args):
    """Unified dataset loader."""
    if args.task in _SIMPLE_LOADERS:
        return _SIMPLE_LOADERS[args.task](args)
    if args.task in _EXTRACTION_TASKS:
        return _load_extraction_dataset(args)
    raise ValueError(f"Unsupported task: {args.task}")


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        # 将原始数据中的 vertex_set 透传到预测结果，供评估时索引→名字解析
        if "vertex_set" in current_batch[offset]:
            res["vertex_set"] = current_batch[offset]["vertex_set"]
        # 透传 title 和 raw_labels 用于官方评测脚本
        if "title" in current_batch[offset]:
            res["title"] = current_batch[offset]["title"]
        if "raw_labels" in current_batch[offset]:
            res["raw_labels"] = current_batch[offset]["raw_labels"]
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True,
                        help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name to use (e.g. 'Qwen/Qwen3-14B', 'Qwen/Qwen2-VL-7B-Instruct').")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa', 'docred', 'cord', 'funsd', 'chemprot'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--doc_path", type=str, default=None, help="Path to document file (for docred/cord/funsd/chemprot tasks)")
    parser.add_argument("--train_path", type=str, default="./data/train_annotated.json", help="Full path to train file for official Re-DocRED Ign-F1 evaluation")
    parser.add_argument("--extraction_mode", type=str, choices=["full", "chunks", "partitioned"], default="full", 
                        help="Document processing mode: 'full' (recommended for DocRED), 'chunks' for sequential, 'partitioned' for hierarchical")
    parser.add_argument("--chunk_size", type=int, default=3000, help="Characters per chunk for extraction tasks")
    parser.add_argument("--chunk_overlap", type=int, default=300, help="Overlap between chunks")
    parser.add_argument("--num_partitions", type=int, default=3, help="Number of partitions for hierarchical extraction")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save results JSON file (e.g., results.json)")
    
    # Multimodal arguments
    parser.add_argument("--use_vision_model", action="store_true", help="Use vision-language model (Qwen-VL) for multimodal tasks")
    parser.add_argument("--image_path", type=str, default=None, help="Path to image file for multimodal extraction (CORD/FUNSD)")
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to LoRA weights directory for fine-tuned model")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images for batch processing")
    parser.add_argument("--annotations_dir", type=str, default=None, help="Directory containing annotation files (FUNSD segm_file JSONs)")

    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--use_verifier", action="store_true", help="Run an independent verifier agent after judger (extraction tasks only)")
    parser.add_argument("--verifier_without_lora", action="store_true", help="Run verifier with base model only, without loading LoRA weights")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()
    
    # Force batch_size=1 for vision models to avoid image token mismatch
    if args.use_vision_model:
        print("[INFO] Vision model detected, forcing batch_size=1")
        args.generate_bs = 1

    # ChemProt JSON output is compact, but Qwen3 thinking mode consumes tokens before JSON.
    # 2048 gives enough room for <think>...</think> + JSON without repetition risk.
    _CHEMPROT_MAX_TOKENS = 2048
    if args.task == "chemprot" and args.max_new_tokens > _CHEMPROT_MAX_TOKENS:
        print(f"[INFO] chemprot task: capping max_new_tokens {args.max_new_tokens} → {_CHEMPROT_MAX_TOKENS} "
              f"(pass --max_new_tokens larger value to override)")
        args.max_new_tokens = _CHEMPROT_MAX_TOKENS

    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    verifier_model = model
    if args.use_verifier and args.verifier_without_lora and args.lora_weights:
        verifier_args = deepcopy(args)
        verifier_args.lora_weights = None
        verifier_model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=verifier_args)
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        # ChemProt is a short single-abstract task; non-judger analysis needs far fewer tokens.
        # Other extraction tasks may have longer documents so keep 2× headroom.
        if args.task == "chemprot":
            _text_mas_each = 256
        elif args.task in _EXTRACTION_TASKS:
            _text_mas_each = args.max_new_tokens * 2
        else:
            _text_mas_each = args.max_new_tokens
        method = TextMASMethod(
            model,
            max_new_tokens_each=_text_mas_each,
            max_new_tokens_judger=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            verifier_model=verifier_model,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            verifier_model=verifier_model,
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # dataset loading
    dataset_iter = load_dataset_for_task(args)

    if args.max_samples == -1:
        # Try to determine dataset length without materializing all data into
        # memory.  For large multimodal datasets (FUNSD/CORD with PIL Images),
        # list() would pull every sample and image into RAM at once → OOM.
        if hasattr(dataset_iter, '__len__'):
            # HuggingFace Dataset objects, plain lists, etc.
            args.max_samples = len(dataset_iter)
        else:
            # Generator / bare iterator – count by consuming, then replay.
            # This is the only path that materializes; we stream items out as
            # fast as possible and keep only the count visible.
            materialized = list(dataset_iter)
            args.max_samples = len(materialized)
            dataset_iter = iter(materialized)
            del materialized

    progress = tqdm(total=args.max_samples)

    _interrupted = False
    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            try:
                processed, preds = process_batch(
                    method,
                    batch,
                    processed,
                    preds,
                    progress,
                    args.max_samples,
                    args,
                )
            except Exception as e:
                print(f"\n⚠️  Batch processing failed at sample {processed}: {e}")
                _interrupted = True
                break
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples and not _interrupted:
        try:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                max_samples=args.max_samples,
                args=args,
            )
        except Exception as e:
            print(f"\n⚠️  Final batch failed at sample {processed}: {e}")
            _interrupted = True
    progress.close()

    # Emergency save on interruption so partial results are not lost
    if _interrupted and preds:
        emergency_path = f"{args.task}_{args.method}_emergency_{processed}.json"
        with open(emergency_path, 'w', encoding='utf-8') as f:
            json.dump({"predictions": preds, "processed": processed}, f, ensure_ascii=False, indent=2)
        print(f"🚨 Emergency save: {len(preds)} results written to {emergency_path}")
    
    total_time = time.time() - start_time

    # 根据任务类型选择评估方式
    extraction_tasks = ["docred", "cord", "funsd", "chemprot"]
    
    if args.task in extraction_tasks:
        # 文档抽取任务：使用专门的评估指标
        print("\n" + "="*70)
        print(f"📊 Evaluating {args.task.upper()} Extraction Results...")
        print("="*70)
        
        # 收集金标准
        golds = [p.get("gold", "") for p in preds]
        
        # 计算评估指标
        metrics = evaluate_extraction_task(args.task, preds, golds)
        
        # 打印详细结果
        print_evaluation_results(args.task, metrics)
        
        # DocRED 额外调用官方评测脚本（如果 evaluation.py 可用）
        if args.task == "docred":
            try:
                from evaluate_extraction import convert_to_official_format
                from evaluation import official_evaluate_benchmark
                formatted_preds = convert_to_official_format(preds)
                re_f1, evi_f1, re_f1_ign_ann, re_f1_ign, re_p, re_r, \
                    re_f1_freq, re_f1_long_tail, re_f1_intra, re_f1_inter, \
                    re_p_freq, re_r_freq, re_p_lt, re_r_lt = official_evaluate_benchmark(
                    tmp=formatted_preds,
                    path=os.path.dirname(os.path.abspath(args.doc_path)) if args.doc_path else "./data",
                    train_file=os.path.abspath(args.train_path) if args.train_path else "train_annotated.json",
                    dev_file=os.path.basename(args.doc_path) if args.doc_path else "dev.json",
                )
                print("\n" + "="*60)
                print("📊 Re-DocRED Official Evaluation Results")
                print("="*60)
                print(f"  Overall  F1 : {re_f1 * 100:.2f}%  (P: {re_p * 100:.2f}%  R: {re_r * 100:.2f}%)")
                print(f"  Ign-F1 (anno): {re_f1_ign_ann * 100:.2f}%")
                print(f"  Ign-F1 (train): {re_f1_ign * 100:.2f}%")
                print(f"  Intra-sent F1: {re_f1_intra * 100:.2f}%")
                print(f"  Inter-sent F1: {re_f1_inter * 100:.2f}%")
                print(f"  Freq  F1     : {re_f1_freq * 100:.2f}%  (P: {re_p_freq * 100:.2f}%  R: {re_r_freq * 100:.2f}%)")
                print(f"  LongTail F1  : {re_f1_long_tail * 100:.2f}%  (P: {re_p_lt * 100:.2f}%  R: {re_r_lt * 100:.2f}%)")
                print("="*60)
                metrics["official_f1"] = re_f1
                metrics["official_ign_f1_annotated"] = re_f1_ign_ann
                metrics["official_ign_f1_train"] = re_f1_ign
                metrics["official_intra_f1"] = re_f1_intra
                metrics["official_inter_f1"] = re_f1_inter
            except ImportError:
                print("\n[INFO] evaluation.py not found — skipping official Re-DocRED benchmark.")
                print("       Place evaluation.py in the project root to enable official scoring.")
            except Exception as e:
                print(f"\n[WARNING] Official evaluation failed: {e}")
        
        # 保存结果
        result_summary = {
            "method": args.method,
            "model": args.model_name,
            "task": args.task,
            "prompt": args.prompt,
            "max_samples": len(preds),
            "total_time_sec": round(total_time, 4),
            "time_per_sample_sec": round(total_time / max(len(preds), 1), 4),
            **metrics  # 包含所有评估指标
        }
    else:
        # 传统QA任务：使用准确率
        acc, correct = evaluate(preds)
        
        result_summary = {
            "method": args.method,
            "model": args.model_name,
            "split": args.split,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "accuracy": acc,
            "correct": correct,
            "total_time_sec": round(total_time, 4),
            "time_per_sample_sec": round(total_time / max(args.max_samples, 1), 4),
        }
    
    # Print results to stdout
    print(
        json.dumps(
            result_summary,
            ensure_ascii=False,
            indent=2
        )
    )
    
    # Save results to file if output_path is specified
    if args.output_path:
        output_data = {
            "summary": result_summary,
            "predictions": preds
        }
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Results saved to {args.output_path}")
    else:
        # Auto-generate output path
        auto_output_path = f"{args.task}_{args.method}_results.json"
        output_data = {
            "summary": result_summary,
            "predictions": preds
        }
        with open(auto_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Results saved to {auto_output_path}")


if __name__ == "__main__":
    main()
