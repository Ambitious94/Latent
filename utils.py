import os
import random
import re
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# ── Code execution with process pool ──────────────────────────────
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError


def _exec_code_worker(code: str):
    """Top-level worker function (must be picklable) that executes *code* in a
    fresh namespace and returns (ok, error_msg)."""
    try:
        exec(code, {})
        return True, None
    except Exception:
        return False, traceback.format_exc()


# Lazy singleton – created on first call, reused afterwards.
_code_executor: Optional[ProcessPoolExecutor] = None


def _get_code_executor() -> ProcessPoolExecutor:
    global _code_executor
    if _code_executor is None:
        # max_workers=4 keeps memory reasonable while giving good throughput
        _code_executor = ProcessPoolExecutor(max_workers=4)
    return _code_executor


def run_with_timeout(code: str, timeout: int = 10):
    """Execute *code* in a pooled worker process with a hard timeout.

    Using a persistent ``ProcessPoolExecutor`` avoids the overhead of
    spawning and tearing down a fresh ``Process`` for every single code
    snippet, which can be a significant bottleneck when evaluating
    hundreds of MBPP+ / HumanEval+ samples.
    """
    executor = _get_code_executor()
    future = executor.submit(_exec_code_worker, code)
    try:
        ok, error = future.result(timeout=timeout)
        return ok, error
    except FuturesTimeoutError:
        future.cancel()
        return False, f"TimeoutError: Execution exceeded {timeout} seconds"
    except Exception as e:
        return False, f"ExecutionError: {e}"


def evaluate_prediction(task: str, final_text: str, item: Dict, idx: int = 0) -> Dict:
    """
    Unified answer evaluation logic shared across all methods.

    Returns a dict with keys: prediction, gold, correct, error_msg
    """
    if task in ['mbppplus', 'humanevalplus']:
        pred = extract_markdown_python_block(final_text)
        gold = item.get("gold", "")

        if pred is None:
            ok = False
            error_msg = "python error: No python code block found"
        else:
            python_code_to_exe = pred + "\n" + gold
            ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)

        print(f'=========================================')
        print(f'Question {idx}')
        print(f'error_msg: {error_msg}')

    elif task in ["aime2024", "aime2025"]:
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        gold = str(item.get("gold", "")).strip()
        error_msg = None
        try:
            pred_int = int(pred)
            gold_int = int(gold)
            ok = (pred_int == gold_int)
        except (ValueError, TypeError):
            ok = False
            error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

    elif task in ['docred', 'cord', 'funsd', 'finer', 'chemprot']:
        pred = final_text.strip()
        gold = item.get("gold", "{}")
        error_msg = None
        try:
            json.loads(pred)
            ok = True
        except (json.JSONDecodeError, TypeError) as e:
            ok = False
            error_msg = f"JSON parse error: {e}"

        print(f'=========================================')
        print(f'Document {idx} - {task}')
        print(f'Extracted: {pred[:200]}...' if len(pred) > 200 else f'Extracted: {pred}')
        if error_msg:
            print(f'Error: {error_msg}')

    else:
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        gold = item.get("gold", "")
        ok = (pred == gold) if (pred and gold) else False
        error_msg = None

    return {"prediction": pred, "gold": gold, "correct": ok, "error_msg": error_msg}

