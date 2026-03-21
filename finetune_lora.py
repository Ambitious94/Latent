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
1. head_id must be the SUBJECT (the entity performing the action or being described).
2. tail_id must be the OBJECT (the target entity of the relation).
3. head_id/tail_id must be integer indices from the entity list above (e.g. 0, 1, 2...).
4. relation must be one of the valid relation names listed above (use the exact string).
Output the JSON directly with no explanation."""
        
        elif self.task == "cord":
            instruction = """Task: Extract receipt/invoice information from OCR text or image into a nested JSON structure.

You must extract information into two main sections:
1. "menu": A list of purchased items. Each item must be a dictionary containing:
    - "nm": Name of the item (string)
    - "cnt": Quantity purchased (string, e.g., "1")
    - "price": Price of the item (string)
2. "total": A dictionary containing summary amounts:
    - "total_price": The final total amount (string)
    - "cashprice": Cash given by the customer (string, optional)
    - "changeprice": Change returned (string, optional)
    - "subtotal_price": Subtotal before tax (string, optional)
    - "tax_price": Tax amount (string, optional)

Rules:
- Output valid JSON only.
- If a field or value is missing in the receipt, use an empty string "".
- If there are no menu items, output an empty list [] for "menu".

Output JSON format example:
{"menu": [{"nm": "EGG TART", "cnt": "1", "price": "13,000"}], "total": {"total_price": "13,000", "cashprice": "15,000", "changeprice": "2,000"}}"""
        
        elif self.task == "finer":
            instruction = """Task: Fine-grained financial entity recognition (FinER).

Identify and extract financial entities and their corresponding XBRL tags from the text.

Output JSON format:
{"entities": [{"text": "entity_text", "type": "XBRL_Tag_Name", "start": 0, "end": 10}]}

Rules:
- start/end are character positions in the original text (0-based).
- text is the exact string of the entity.
- type must be the exact financial XBRL tag corresponding to the entity.
- If no financial entities are found, output {"entities": []}."""
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
            # 1. 完整对话 (用于训练)
            text = self.processor.apply_chat_template(messages, tokenize=False)
            # 2. 仅 Prompt (用于计算掩码长度，add_generation_prompt=True 会自动加上 <|im_start|>assistant\n)
            prompt_messages = messages[:-1]
            prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback：手动构建文本
            text = ""
            for msg in messages:
                if isinstance(msg["content"], list):
                    text += "\n".join([c["text"] for c in msg["content"] if c.get("type") == "text"])
                else:
                    text += msg["content"]
                text += "\n"
            prompt_text = text[:text.rfind(gold)]  # 粗略截取

        # ⚠️ 提升 max_length 到 4096，防止较长文档的 JSON 标签被截断
        MAX_LEN = 4096

        if image:
            # 编码完整文本
            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt", padding=False, truncation=True, max_length=MAX_LEN
            )
            # 编码 Prompt 仅用于获取长度 (传入 image 确保视觉 token 占位准确)
            prompt_inputs = self.processor(
                text=[prompt_text], images=[image], return_tensors="pt", padding=False, truncation=True, max_length=MAX_LEN
            )
        else:
            inputs = self.processor(
                text=[text], return_tensors="pt", padding=False, truncation=True, max_length=MAX_LEN
            )
            prompt_inputs = self.processor(
                text=[prompt_text], return_tensors="pt", padding=False, truncation=True, max_length=MAX_LEN
            )

        # 获取 Prompt 的准确 Token 长度
        prompt_len = prompt_inputs["input_ids"].shape[1]

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # 【完美掩码】：将 prompt_len 之前的所有 token 的 label 设为 -100
        # 使用 min 保护防止被截断导致的越界
        mask_len = min(prompt_len, labels.shape[0])
        labels[:mask_len] = -100

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
    
    # ======== 新增：直接从 HuggingFace 内存加载 CORD 数据 ========
    if args.task == "cord":
        print("⏳ 正在从 HuggingFace 加载 CORD-v2 数据集...")
        from datasets import load_dataset
        import json
        
        dataset = load_dataset("naver-clova-ix/cord-v2", split="train")
        data_items = []
        
        for item in dataset:
            # 1. 提取 PIL 图片
            pil_image = item["image"]
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
                
            # 2. 提取 OCR 文本
            ground_truth = json.loads(item["ground_truth"])
            ocr_text = ""
            for line in ground_truth.get("valid_line", []):
                words = [w.get("text", "") for w in line.get("words", [])]
                ocr_text += " ".join(words) + "\n"
                
            # 3. 清洗并构建我们设计的【嵌套 JSON】
            raw_gt_parse = ground_truth.get("gt_parse", {})

            # --- 新增的鲁棒解析逻辑 ---
            raw_menu = raw_gt_parse.get("menu", [])
            # 如果 menu 是单个字典，把它包装成列表
            if isinstance(raw_menu, dict):
                raw_menu = [raw_menu]
            # 如果是其他奇葩类型(比如字符串)，直接置为空列表
            elif not isinstance(raw_menu, list):
                raw_menu = []

            clean_menu = []
            for m in raw_menu:
                if isinstance(m, dict):
                    # 强转为字符串，防止内部还嵌套其他奇奇怪怪的结构
                    clean_menu.append({
                        "nm": str(m.get("nm", "")),
                        "cnt": str(m.get("cnt", "")),
                        "price": str(m.get("price", ""))
                    })
            # --------------------------

            raw_total = raw_gt_parse.get("total", {}) if isinstance(raw_gt_parse.get("total"), dict) else {}
            clean_total = {
                "total_price": raw_total.get("total_price", ""),
                "cashprice": raw_total.get("cashprice", ""),
                "changeprice": raw_total.get("changeprice", ""),
                "subtotal_price": raw_total.get("subtotal_price", ""),
                "tax_price": raw_total.get("tax_price", "")
            }
            gold_json = {"menu": clean_menu, "total": clean_total}
            
            # 4. 直接喂给你的 Dataset 类
            data_items.append({
                "question": ocr_text.strip(),
                "gold": json.dumps(gold_json, ensure_ascii=False),
                "image": pil_image
            })
            
            if args.max_train_samples and len(data_items) >= args.max_train_samples:
                break
                
        print(f"Loaded {len(data_items)} CORD training samples directly from HuggingFace.")
        return data_items
    # =========================================================
    
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
        # CORD 加载已在函数开头通过 HuggingFace 直接加载，此处应不会执行
        raise RuntimeError("CORD 加载应在函数开头完成，如执行到此表示逻辑错误")
    elif args.task == "finer":
        data_iter = load_finer(
            doc_path=args.train_data,
            split="train",
            mode="full"
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    data_items = []
    import json
    for item in data_iter:
        # 仅对 finer 任务过滤空实体样本，避免影响其他任务的数据格式
        if args.task == "finer":
            try:
                gold_data = json.loads(item["gold"])
                if len(gold_data.get("entities", [])) == 0:
                    continue  # 忽略 FinER 的空样本
            except Exception:
                pass

        data_items.append(item)
        if args.max_train_samples is not None and args.max_train_samples > 0 and len(data_items) >= args.max_train_samples:
            break

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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
        dataloader_num_workers=0,
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
