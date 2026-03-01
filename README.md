# LatentIE — Latent Multi-Agent System for Information Extraction & Reasoning

A research framework implementing **Latent Multi-Agent System (LatentMAS)** where agents communicate through latent hidden-state representations (KV-cache) instead of explicit text, enabling more efficient multi-agent collaboration.

## Architecture

The system uses a 4-agent pipeline: **Planner → Critic → Refiner → Judger**

| Method | Communication | Description |
|--------|--------------|-------------|
| `baseline` | None | Single agent direct inference |
| `text_mas` | Text | Multi-agent with text-based communication |
| `latent_mas` | Hidden States | Multi-agent with latent-space communication (core innovation) |

## Supported Tasks

| Category | Tasks |
|----------|-------|
| Math Reasoning | GSM8K, AIME2024, AIME2025 |
| Knowledge QA | GPQA, ARC-Easy/Challenge, MedQA |
| Code Generation | MBPP+, HumanEval+ |
| Document IE | DocRED, CORD, FUNSD, FinER-139 |

## Quick Start

```bash
pip install -r requirements.txt

# Baseline on GSM8K
python run.py --method baseline --model_name Qwen/Qwen3-14B --task gsm8k

# LatentMAS with latent thinking steps
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --latent_steps 5 --think

# TextMAS on DocRED extraction
python run.py --method text_mas --model_name Qwen/Qwen3-14B --task docred --doc_path data/test_docred.json

# With vLLM acceleration
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --use_vllm --latent_steps 5
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--method` | `baseline` / `text_mas` / `latent_mas` |
| `--model_name` | HuggingFace model name |
| `--task` | Task to evaluate |
| `--prompt` | `sequential` or `hierarchical` architecture |
| `--latent_steps` | Number of latent generation steps |
| `--use_vllm` | Enable vLLM backend |
| `--use_vision_model` | Enable vision-language model |
| `--lora_weights` | Path to LoRA weights |

## Project Structure

```
├── run.py                 # Main entry point
├── models.py              # ModelWrapper (HF + vLLM + Vision)
├── data.py                # Dataset loaders
├── prompts.py             # Prompt templates for all tasks
├── utils.py               # Shared utilities & evaluation
├── evaluate_extraction.py # IE-specific metrics (P/R/F1)
├── methods/
│   ├── baseline.py        # Single-agent method
│   ├── text_mas.py        # Text-based multi-agent
│   └── latent_mas.py      # Latent-space multi-agent
└── data/                  # Local datasets
```
