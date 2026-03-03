"""
LoRA微调脚本 - 针对文档信息抽取任务微调Qwen-VL模型

使用方法:
python finetune_lora.py \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --task funsd \
    --train_data /data/funsd/instances_train.json \
    --annotations_dir /data/funsd/annotations \
    --image_dir /data/funsd/imgs \
    --output_dir ./lora_weights/funsd \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
"""

import os
import json
import torch
import argparse
from typing import List, Dict
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from PIL import Image
from data import load_funsd, load_docred, load_cord, load_finer
from prompts import DOCRED_REL_MAP


class DocumentExtractionDataset(Dataset):
    """文档信息抽取数据集"""
    
    def __init__(self, data_items: List[Dict], processor, task: str):
        self.data_items = data_items
        self.processor = processor
        self.task = task
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # 构建输入消息
        question = item.get("question", "")
        gold = item.get("gold", "{}")
        image = item.get("image")
        entity_list = item.get("entity_list", "")  # DocRED实体列表
        
        # 根据任务类型构建训练prompt（与推理时格式一致）
        if self.task == "funsd":
            instruction = """Task: Extract form fields and their semantic relationships.

Identify form entities with these labels:
- question: Field labels or prompts ("Name:", "Date of Birth:", "Address:")
- answer: Filled-in values or responses
- header: Section titles or form headers
- other: Other text elements

Identify relations:
- Link questions to their corresponding answers
- Use entity text for matching

Output JSON format:
{"entities": [{"text": "Name:", "label": "question"}, {"text": "John Smith", "label": "answer"}], "relations": [{"head": "Name:", "tail": "John Smith"}]}

Example:
Form text: "Employee Name: Sarah Johnson | Department: Engineering | Salary: $85,000"
Output:
{
  "entities": [
    {"text": "Employee Name:", "label": "question"},
    {"text": "Sarah Johnson", "label": "answer"},
    {"text": "Department:", "label": "question"},
    {"text": "Engineering", "label": "answer"},
    {"text": "Salary:", "label": "question"},
    {"text": "$85,000", "label": "answer"}
  ],
  "relations": [
    {"head": "Employee Name:", "tail": "Sarah Johnson"},
    {"head": "Department:", "tail": "Engineering"},
    {"head": "Salary:", "tail": "$85,000"}
  ]
}"""
        
        elif self.task == "docred":
            # DocRED: 索引抽取 + 语义化关系名 + Evidence CoT
            # 构建可用关系列表（自然语言）
            rel_names_list = ", ".join(
                f'"{name}"' for name in DOCRED_REL_MAP.values()
            )
            
            # 将 gold 转为索引格式 + 语义关系名 + 保留 evidence
            try:
                gold_data = json.loads(gold) if isinstance(gold, str) else gold
                if isinstance(gold_data, dict) and "relations" in gold_data:
                    # 构建实体名→索引的反向映射（包含所有别名）
                    vertex_set = item.get("vertex_set", [])
                    name_to_idx = {}
                    for vidx, mentions in enumerate(vertex_set):
                        if mentions:
                            for m in mentions:
                                name_to_idx[m.get("name", "")] = vidx
                    
                    new_relations = []
                    for r in gold_data["relations"]:
                        pid = r.get("relation", "")
                        rel_name = DOCRED_REL_MAP.get(pid, pid)
                        head_name = r.get("head", "")
                        tail_name = r.get("tail", "")
                        head_id = name_to_idx.get(head_name, -1)
                        tail_id = name_to_idx.get(tail_name, -1)
                        if head_id >= 0 and tail_id >= 0:
                            new_relations.append({
                                "head_id": head_id,
                                "relation": rel_name,
                                "tail_id": tail_id
                                # evidence 已移除：句子未编号，强求 evidence 反而诱发幻觉
                            })
                    gold_data["relations"] = new_relations
                    gold = json.dumps(gold_data, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                pass
            
            instruction = f"""Task: Document-level relation extraction.

Entities in this document:
{entity_list}

Extract relations between entities using their index numbers [i] from the list above.
Use the exact natural-language relation names below.

Valid relation names: {rel_names_list}

Output JSON format:
{{"relations": [{{"head_id": 0, "relation": "country", "tail_id": 5}}]}}

Rules:
1. head_id/tail_id must be integer indices from the entity list above (e.g. 0, 1, 2...)
2. relation must be one of the valid relation names listed above (use the exact string)"""
        
        elif self.task == "cord":
            instruction = """Task: Extract receipt/invoice information from OCR text.

Extract these fields:
1. num_items: Total number of items purchased (integer)
2. subtotal_price: Price before tax/service charge (string with currency)
3. service_price: Service charge amount (string)
4. tax_price: Tax amount (string)
5. total_price: Final total amount (string)
6. etc: Additional charges or notes (string)

Important:
- Extract exact amounts as they appear (e.g., "$12.50", "25.00")
- If a field is not present, use empty string ""
- num_items should be an integer count

Output JSON format:
{"num_items": 3, "subtotal_price": "25.50", "service_price": "2.00", "tax_price": "2.48", "total_price": "29.98", "etc": ""}

Example:
Input: "Item1 $10.00\nItem2 $15.50\nSubtotal $25.50\nTax $2.48\nTotal $27.98"
Output: {"num_items": 2, "subtotal_price": "25.50", "tax_price": "2.48", "total_price": "27.98", "service_price": "", "etc": ""}"""
        
        elif self.task == "finer":
            instruction = """Task: Fine-grained financial named entity recognition.

Identify and classify financial entities in text.

Entity types:
- PER: Person names (executives, analysts, investors)
- ORG: Organizations (companies, banks, institutions)
- LOC: Locations (countries, cities, regions)
- MONEY: Monetary amounts ("$1M", "100 million dollars")
- DATE: Dates and time periods ("Q3 2023", "March 15")
- PERCENT: Percentage values ("5%", "15.5 percent")
- STOCK: Stock tickers and symbols ("AAPL", "NASDAQ:MSFT")
- METRIC: Financial metrics ("revenue", "profit margin", "EPS")
- PRODUCT: Financial products ("bonds", "derivatives", "mortgage")
- LAW: Financial regulations ("Dodd-Frank", "Basel III")

Output JSON format:
{"entities": [{"text": "Apple Inc.", "type": "ORG", "start": 0, "end": 10}, {"text": "$2.5B", "type": "MONEY", "start": 25, "end": 30}]}

Rules:
- start/end are character positions in original text (0-based)
- text is the exact entity string
- type must be one of the predefined types

Example:
Input: "Apple reported $95.3B revenue in Q1 2024, up 5%."
Output: {"entities": [{"text": "Apple", "type": "ORG", "start": 0, "end": 5}, {"text": "$95.3B", "type": "MONEY", "start": 15, "end": 22}, {"text": "Q1 2024", "type": "DATE", "start": 34, "end": 41}, {"text": "5%", "type": "PERCENT", "start": 46, "end": 48}]}"""
        else:
            instruction = "Task: Extract information"
        
        # 构建消息格式（与推理时LoRA prompts一致）
        system_msg = "You are an expert document information extraction system. Extract structured information accurately and output valid JSON only."
        
        if image:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{instruction}\n\nDocument text:\n{question}\n\nExtract and output JSON:"}
            ]
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": gold}
            ]
        else:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"{instruction}\n\nDocument text:\n{question}\n\nExtract and output JSON:"},
                {"role": "assistant", "content": gold}
            ]
        
        # 使用processor处理
        if hasattr(self.processor, 'apply_chat_template'):
            # Qwen系列模型支持chat template
            text = self.processor.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback：手动构建文本
            text = ""
            for msg in messages:
                if isinstance(msg["content"], list):
                    text += "\n".join([c["text"] for c in msg["content"] if c.get("type") == "text"])
                else:
                    text += msg["content"]
                text += "\n"
        
        if image:
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=2048,
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=2048,
            )
        
        # 创建labels：只在 assistant 回复部分计算 loss
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        labels = input_ids.clone()
        
        # Mask padding tokens
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        
        # 找到 assistant 回复的起始位置，将之前所有 token 的 label 设为 -100
        # 使用严格的完整序列匹配，避免文档内容中偶发的 "assistant" 字词干扰掩码位置
        assistant_token_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        if assistant_token_ids:
            seq_len = len(assistant_token_ids)
            match_idx = -1
            tgt = torch.tensor(assistant_token_ids, device=input_ids.device)
            # 从后往前遍历，找最后一次出现完整序列的位置
            for i in range(len(input_ids) - seq_len, -1, -1):
                if torch.equal(input_ids[i:i + seq_len], tgt):
                    match_idx = i + seq_len
                    break
            if match_idx != -1:
                # +1 跳过紧跟的换行符 \n
                start_idx = match_idx + 1
                labels[:start_idx] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # 只有VL模型才有pixel_values
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        
        return result


def load_training_data(args):
    """加载训练数据"""
    print(f"Loading training data for {args.task}...")
    
    if args.task == "funsd":
        data_iter = load_funsd(
            doc_path=args.train_data,
            split="train",
            mode="full",
            annotations_dir=args.annotations_dir,
            images_dir=args.image_dir
        )
    elif args.task == "docred":
        data_iter = load_docred(
            doc_path=args.train_data,
            split="train",
            mode="full"
        )
    elif args.task == "cord":
        data_iter = load_cord(
            doc_path=args.train_data,
            split="train",
            mode="full"
        )
    elif args.task == "finer":
        data_iter = load_finer(
            doc_path=args.train_data,
            split="train",
            mode="full"
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")
    
    data_items = list(data_iter)
    print(f"Loaded {len(data_items)} training samples")
    
    return data_items


def vl_data_collator(features: List[Dict]) -> Dict:
    """Custom data collator that correctly handles Qwen2-VL's variable-length
    pixel_values (NaViT) as well as standard text-only batches.

    Qwen2-VL encodes images of different resolutions into variable-length 1-D
    ``pixel_values`` tensors.  ``torch.stack`` would crash on shape mismatch,
    so we ``torch.cat`` them along dim-0 instead — exactly what the model's
    forward pass expects.  ``image_grid_thw`` is likewise concatenated.

    Text tokens (``input_ids``, ``attention_mask``, ``labels``) are
    right-padded to the longest sequence in the current batch (dynamic
    padding) so we never waste compute on unnecessary padding tokens.
    """
    first = features[0]
    batch: Dict = {}

    # --- text fields: dynamic padding to max length in batch ---
    for key in ("input_ids", "attention_mask", "labels"):
        if key not in first or first[key] is None:
            continue
        tensors = [f[key] for f in features]
        max_len = max(t.size(0) for t in tensors)
        # Determine padding value: -100 for labels, 0 for others
        pad_value = -100 if key == "labels" else 0
        padded = torch.stack([
            torch.cat([t, t.new_full((max_len - t.size(0),), pad_value)])
            if t.size(0) < max_len else t
            for t in tensors
        ])
        batch[key] = padded

    # --- vision fields: variable-length concat (Qwen2-VL NaViT) ---
    # pixel_values from different images may have different lengths because
    # Qwen2-VL patches images at their native resolution.  The model expects
    # them concatenated along dim-0 with image_grid_thw recording boundaries.
    if "pixel_values" in first and first["pixel_values"] is not None:
        pv_list = [f["pixel_values"] for f in features if f.get("pixel_values") is not None]
        if pv_list:
            batch["pixel_values"] = torch.cat(pv_list, dim=0)

    if "image_grid_thw" in first and first["image_grid_thw"] is not None:
        thw_list = [f["image_grid_thw"] for f in features if f.get("image_grid_thw") is not None]
        if thw_list:
            batch["image_grid_thw"] = torch.cat(thw_list, dim=0)

    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--task", type=str, required=True, choices=["funsd", "docred", "cord", "finer"])
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--annotations_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples (use all if not specified)")
    parser.add_argument("--use_vision_model", action="store_true", help="Use vision-language model (auto-detect if not specified)")
    args = parser.parse_args()
    
    # 自动检测是否应该使用VL模型
    if not args.use_vision_model:
        # FUNSD和CORD默认使用VL模型，DocRED和FinER使用文本模型
        if args.task in ["funsd", "cord"]:
            args.use_vision_model = True
            print(f"[Auto] Task {args.task} detected, using vision-language model")
        else:
            args.use_vision_model = False
            print(f"[Auto] Task {args.task} detected, using text-only model")
    
    # 加载模型和processor
    print(f"Loading model: {args.model_name}")
    
    if args.use_vision_model:
        # 加载VL模型
        from transformers import AutoProcessor
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_name)
    else:
        # 加载纯文本模型
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoTokenizer.from_pretrained(args.model_name)
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
            processor.pad_token_id = processor.eos_token_id
    
    # 配置LoRA
    print("Configuring LoRA...")
    
    # 自动查找模型中的linear层作为target_modules
    # 这样可以适配不同模型架构（Qwen2, Qwen3, Qwen-VL等）
    import re
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 提取层名称的最后一部分（如 q_proj, k_proj等）
            layer_name = name.split('.')[-1]
            if layer_name not in target_modules and not layer_name.startswith('lm_head'):
                target_modules.append(layer_name)
    
    # 如果自动检测失败，使用常见的target_modules
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    print(f"LoRA target modules: {target_modules}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 确保模型在训练模式并启用梯度
    model.train()
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    
    # 加载数据
    train_data = load_training_data(args)
    
    # 限制训练样本数量（如果指定）
    if args.max_train_samples is not None and args.max_train_samples > 0:
        original_size = len(train_data)
        train_data = train_data[:args.max_train_samples]
        print(f"Using {len(train_data)} out of {original_size} training samples")
    else:
        print(f"Using all {len(train_data)} training samples")
    
    train_dataset = DocumentExtractionDataset(train_data, processor, args.task)
    
    # 检测GPU是否支持bf16
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if use_bf16:
        print("Using bf16 training (GPU supports it)")
    else:
        print("Using fp16 training (bf16 not supported, falling back to fp16)")
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=False,  # 禁用以避免与LoRA冲突
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none"
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=vl_data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    # 保存
    print(f"Saving LoRA weights to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
