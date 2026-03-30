from typing import Dict, Iterable, Optional

from datasets import load_dataset

from utils import extract_gold, normalize_answer


def load_gsm8k(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
        }


def load_aime2025(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("yentinglin/aime_2025", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_aime2024(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("HuggingFaceH4/aime_2024", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_gpqa_diamond(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_arc_easy(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(l: str) -> str:
            s = str(l).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        # Map choices
        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        # Map answers
        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_arc_challenge(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(l: str) -> str:
            s = str(l).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_winogrande(
    split: str = "validation",
    subset: str = "winogrande_debiased",
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("allenai/winogrande", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        ask_str = 'Pickout proper choice that fits the _ in the following sentence:'
        sentence = item["sentence"].strip()
        option1 = str(item["option1"]).strip()
        option2 = str(item["option2"]).strip()
        question = f"{ask_str}\n{sentence}\n1: {option1}\n2: {option2}"
        answer = str(item["answer"])
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_mbppplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("evalplus/mbppplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
Your answer will be tested on test cases like:
{item["test_list"][0]}
{item["test_list"][1]}
{item["test_list"][2]}
"""

        answer = str(item["test"])
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_humanevalplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("evalplus/humanevalplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
"""
        raw_answer = str(item["test"])
        answer = raw_answer.replace('candidate', item['entry_point'])
        answer += f'\n\ncheck({item["entry_point"]})'
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


# qa data from https://github.com/lupantech/AgentFlow/tree/main

def load_medqa(split=None, subset=None, cache_dir=None):
    import os
    data_path = os.path.join(os.path.dirname(__file__), "data", "medqa.json")
    ds = load_dataset("json", data_files=data_path, split='train')
    for item in ds:
        question = item["query"]
        raw_answer = str(item["answer"])

        choice_map = {"0":"A", "1":"B", "2":"C", "3":"D"}
        answer = None

        for idx, op in enumerate(item['options']):
            if raw_answer in op:
                answer = choice_map[str(idx)].lower()
                break

        if answer is None:
            # Fallback: use raw_answer directly if no match found
            answer = normalize_answer(raw_answer)

        gold = normalize_answer(answer)

        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


# ============= Document Extraction Datasets =============

def load_docred(
    doc_path: str,
    split: str = "train",
    mode: str = "chunks",
    chunk_size: int = 3000,
    overlap: int = 300,
    num_partitions: int = 3,
    cache_dir: Optional[str] = None
) -> Iterable[Dict]:
    """
    Load DocRED dataset for document-level relation extraction.
    Format: {"relations": [{"head": "entity1", "relation": "relation_type", "tail": "entity2"}]}
    
    Gold labels are converted from index format {h:0, t:1, r:"P17"} to name format.
    """
    import json
    import os
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"DocRED file not found: {doc_path}")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Standard DocRED extraction schema
    extract_template = {
        "relations": [
            {"head": "", "relation": "", "tail": "", "evidence": []}
        ]
    }
    
    for doc in data:
        # Reconstruct full document text
        if isinstance(doc.get("sents"), list):
            full_text = " ".join([" ".join(sent) if isinstance(sent, list) else str(sent) for sent in doc["sents"]])
        else:
            full_text = str(doc.get("text", ""))
        
        # Get vertexSet for entity name lookup
        vertex_set = doc.get("vertexSet", [])
        
        # Build entity index -> name mapping (use first mention's name)
        def get_entity_name(idx):
            if idx < len(vertex_set) and len(vertex_set[idx]) > 0:
                return vertex_set[idx][0].get("name", f"Entity_{idx}")
            return f"Entity_{idx}"
        
        # Get gold labels and convert to name format
        gold_labels_raw = doc.get("labels", [])
        gold_relations_with_names = []
        for label in gold_labels_raw:
            head_idx = label.get("h")
            tail_idx = label.get("t")
            relation = label.get("r", "")
            evidence = label.get("evidence", [])
            
            gold_relations_with_names.append({
                "head": get_entity_name(head_idx),
                "relation": relation,
                "tail": get_entity_name(tail_idx),
                "evidence": evidence  # 保留evidence字段
            })
        
        # Build entity list string for prompt
        entity_list = []
        for idx, mentions in enumerate(vertex_set):
            if mentions:
                name = mentions[0].get("name", f"Entity_{idx}")
                etype = mentions[0].get("type", "UNKNOWN")
                entity_list.append(f"[{idx}] {name} ({etype})")
        entity_list_str = "\n".join(entity_list)
        
        # Store gold labels - use clean format for training
        # 训练时只需要relations，评估时可通过vertex_set重建raw格式
        gold_output = {
            "relations": gold_relations_with_names
        }
        
        # 同时保存用于评估的原始标签（不包含在训练gold中）
        raw_labels_for_eval = gold_labels_raw
        
        if mode == "full":
            yield {
                "question": full_text,
                "entity_list": entity_list_str,
                "solution": json.dumps(gold_output, ensure_ascii=False),
                "gold": json.dumps(gold_output, ensure_ascii=False),
                "extract_template": json.dumps(extract_template, ensure_ascii=False),
                "dataset": "docred",
                "title": doc.get("title", ""),
                "vertex_set": vertex_set,
                "raw_labels": raw_labels_for_eval,  # For official evaluation
            }
        
        elif mode == "chunks":
            chunks = []
            start = 0
            while start < len(full_text):
                end = start + chunk_size
                chunk = full_text[start:end]
                chunks.append(chunk)
                start = end - overlap
            
            for i, chunk in enumerate(chunks):
                yield {
                    "question": chunk,
                    "entity_list": entity_list_str,
                    "solution": json.dumps(gold_output, ensure_ascii=False),
                    "gold": json.dumps(gold_output, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "chunk_info": f"Chunk {i+1}/{len(chunks)}",
                    "dataset": "docred",
                    "title": doc.get("title", ""),
                    "vertex_set": vertex_set,
                    "raw_labels": raw_labels_for_eval,
                }
        
        elif mode == "partitioned":
            partition_size = len(full_text) // num_partitions
            for i in range(num_partitions):
                start = i * partition_size
                end = start + partition_size if i < num_partitions - 1 else len(full_text)
                partition = full_text[start:end]
                
                yield {
                    "question": partition,
                    "entity_list": entity_list_str,
                    "solution": json.dumps(gold_output, ensure_ascii=False),
                    "gold": json.dumps(gold_output, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "partition_info": f"Partition {i+1}/{num_partitions}",
                    "dataset": "docred",
                    "title": doc.get("title", ""),
                    "vertex_set": vertex_set,
                    "raw_labels": raw_labels_for_eval if 'raw_labels_for_eval' in dir() else [],
                }


def load_cord(
    doc_path: str,
    split: str = "train",
    mode: str = "chunks",
    chunk_size: int = 2000,
    overlap: int = 200,
    num_partitions: int = 3,
    cache_dir: Optional[str] = None,
    image_path: Optional[str] = None  # New parameter for multimodal support
) -> Iterable[Dict]:
    """
    Load CORD dataset for receipt/invoice extraction.
    支持双模加载：如果传入了真实有效的本地路径，则读取本地；否则从 Hugging Face 自动拉取。
    """
    import json
    import os
    from PIL import Image
    
    # Standard CORD extraction schema (official nested format)
    extract_template = {
        "menu": [
            {"nm": "", "cnt": "", "price": ""}
        ],
        "total": {
            "total_price": "",
            "cashprice": "",
            "changeprice": "",
            "subtotal_price": "",
            "tax_price": ""
        }
    }

    # --- 内部辅助函数：统一处理三种 mode 的 yield ---
    def _yield_by_mode(text_content, gold_json, img_obj):
        if mode == "full":
            result = {
                "question": text_content,
                "solution": json.dumps(gold_json, ensure_ascii=False),
                "gold": json.dumps(gold_json, ensure_ascii=False),
                "extract_template": json.dumps(extract_template, ensure_ascii=False),
                "dataset": "cord",
            }
            if img_obj:
                result["image"] = img_obj
            yield result
        
        elif mode == "chunks":
            chunks = []
            start = 0
            while start < len(text_content):
                end = start + chunk_size
                chunks.append(text_content[start:end])
                start = end - overlap
            
            for i, chunk in enumerate(chunks):
                result = {
                    "question": chunk,
                    "solution": json.dumps(gold_json, ensure_ascii=False),
                    "gold": json.dumps(gold_json, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "chunk_info": f"Chunk {i+1}/{len(chunks)}",
                    "dataset": "cord",
                }
                if img_obj:
                    result["image"] = img_obj
                yield result
        
        elif mode == "partitioned":
            partition_size = len(text_content) // num_partitions
            for i in range(num_partitions):
                start = i * partition_size
                end = start + partition_size if i < num_partitions - 1 else len(text_content)
                
                result = {
                    "question": text_content[start:end],
                    "solution": json.dumps(gold_json, ensure_ascii=False),
                    "gold": json.dumps(gold_json, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "partition_info": f"Partition {i+1}/{num_partitions}",
                    "dataset": "cord",
                }
                if img_obj:
                    result["image"] = img_obj
                yield result

    # ==========================================
    # 模式 A: 从本地路径加载
    # ==========================================
    if doc_path and doc_path.lower() != "dummy" and os.path.exists(doc_path):
        print(f"📂 检测到本地路径，正在从本地加载 CORD 数据集: {doc_path}")
        with open(doc_path, 'r', encoding='utf-8') as f:
            if doc_path.endswith('.json'):
                data = json.load(f)
            else:
                full_text = f.read()
                data = [{"text": full_text}]
        
        if isinstance(data, dict) and "samples" in data:
            data = data["samples"]
        
        for doc in (data if isinstance(data, list) else [data]):
            image_obj = None
            doc_image_path = doc.get("filepath") or image_path
            if doc_image_path:
                if not os.path.isabs(doc_image_path):
                    base_dir = os.path.dirname(doc_path)
                    doc_image_path = os.path.join(base_dir, doc_image_path)
                if os.path.exists(doc_image_path):
                    try:
                        image_obj = Image.open(doc_image_path).convert("RGB")
                    except Exception as e:
                        print(f"[Warning] Failed to load image from {doc_image_path}: {e}")
            
            full_text = doc.get("text", "")
            raw_gt_parse = doc.get("gt_parse", {}) if isinstance(doc, dict) else {}
            has_gt_parse = isinstance(raw_gt_parse, dict) and ("menu" in raw_gt_parse or "total" in raw_gt_parse)
            
            gold = {"menu": [], "total": {
                "total_price": "", "cashprice": "", "changeprice": "", "subtotal_price": "", "tax_price": ""
            }}

            if has_gt_parse:
                clean_menu = []
                for m in raw_gt_parse.get("menu", []):
                    if isinstance(m, dict):
                        clean_menu.append({
                            "nm": str(m.get("nm", "")),
                            "cnt": str(m.get("cnt", "")),
                            "price": str(m.get("price", ""))
                        })

                raw_total = raw_gt_parse.get("total", {})
                if not isinstance(raw_total, dict):
                    raw_total = {}

                clean_total = {
                    "total_price": str(raw_total.get("total_price", "")),
                    "cashprice": str(raw_total.get("cashprice", "")),
                    "changeprice": str(raw_total.get("changeprice", "")),
                    "subtotal_price": str(raw_total.get("subtotal_price", "")),
                    "tax_price": str(raw_total.get("tax_price", ""))
                }
                gold = {"menu": clean_menu, "total": clean_total}
            
            elif "ground_truth" in doc:
                gold = doc["ground_truth"]
            
            if not full_text and image_obj:
                full_text = "[Image-based receipt]"
            
            yield from _yield_by_mode(full_text, gold, image_obj)

    # ==========================================
    # 模式 B: 从 Hugging Face 云端加载
    # ==========================================
    else:
        from datasets import load_dataset
        
        hf_split = "validation" if split == "valid" else split
        print(f"⏳ 未检测到有效本地路径，正在从 HuggingFace 加载 CORD-v2 ({hf_split} split)...")
        
        dataset = load_dataset("naver-clova-ix/cord-v2", split=hf_split, cache_dir=cache_dir)
        
        for idx, item in enumerate(dataset):
            # 1. 处理图片与缩放防爆显存
            pil_image = item["image"]
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
                
            max_pixels = 1024
            if max(pil_image.size) > max_pixels:
                ratio = max_pixels / max(pil_image.size)
                new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                
            # 2. 提取 OCR 文本用于辅助
            ground_truth = json.loads(item["ground_truth"])
            ocr_text = ""
            for line in ground_truth.get("valid_line", []):
                words = [w.get("text", "") for w in line.get("words", [])]
                ocr_text += " ".join(words) + "\n"
                
            # 3. 鲁棒清洗嵌套 JSON 格式
            raw_gt_parse = ground_truth.get("gt_parse", {})
            raw_menu = raw_gt_parse.get("menu", [])
            if isinstance(raw_menu, dict):
                raw_menu = [raw_menu]
            elif not isinstance(raw_menu, list):
                raw_menu = []
                
            clean_menu = []
            for m in raw_menu:
                if isinstance(m, dict):
                    clean_menu.append({
                        "nm": str(m.get("nm", "")),
                        "cnt": str(m.get("cnt", "")),
                        "price": str(m.get("price", ""))
                    })

            raw_total = raw_gt_parse.get("total", {}) if isinstance(raw_gt_parse.get("total"), dict) else {}
            clean_total = {
                "total_price": str(raw_total.get("total_price", "")),
                "cashprice": str(raw_total.get("cashprice", "")),
                "changeprice": str(raw_total.get("changeprice", "")),
                "subtotal_price": str(raw_total.get("subtotal_price", "")),
                "tax_price": str(raw_total.get("tax_price", ""))
            }
            gold = {"menu": clean_menu, "total": clean_total}
            
            full_text = ocr_text.strip()
            if not full_text:
                full_text = "[Image-based receipt]"

            yield from _yield_by_mode(full_text, gold, pil_image)


def load_funsd(
    doc_path: str,
    split: str = "train",
    mode: str = "chunks",
    chunk_size: int = 2500,
    overlap: int = 250,
    num_partitions: int = 3,
    cache_dir: Optional[str] = None,
    image_path: Optional[str] = None,
    annotations_dir: Optional[str] = None,
    images_dir: Optional[str] = None
) -> Iterable[Dict]:
    """
    Load FUNSD dataset for form understanding.
    支持双模加载：传入 dummy 则从 Hugging Face 加载 (支持 konfuzio/funsd_plus 关系链接)，否则从本地加载。
    """
    import json
    import os
    from PIL import Image

    extract_template = {
        "entities": [
            {"id": 0, "text": "", "label": ""}
        ],
        "relations": [
            {"head": 0, "tail": 1, "type": "linked"}
        ]
    }

    # ==========================================
    # 模式 B: 从 Hugging Face 云端加载 (支持 FUNSD+)
    # ==========================================
    if doc_path == "dummy" or not os.path.exists(doc_path):
        from datasets import load_dataset
        
        # FUNSD+ 在 HF 上的验证集名为 'test'
        hf_split = "test" if split == "valid" else split
        print(f"⏳ 未检测到有效本地路径，正在从 HuggingFace 加载 FUNSD+ (konfuzio/funsd_plus - {hf_split} split)...")
        
        dataset = load_dataset("konfuzio/funsd_plus", split=hf_split, cache_dir=cache_dir)
        
        # 尝试动态获取标签映射，失败则回退到默认的 FUNSD 标签
        try:
            int2str = dataset.features['labels'].feature.int2str
        except Exception:
            id2tag = {0: "O", 1: "B-HEADER", 2: "I-HEADER", 3: "B-QUESTION", 4: "I-QUESTION", 5: "B-ANSWER", 6: "I-ANSWER"}
            int2str = lambda x: id2tag.get(x, "O")
            
        for idx, item in enumerate(dataset):
            pil_image = item["image"]
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
                
            max_pixels = 1024
            if max(pil_image.size) > max_pixels:
                ratio = max_pixels / max(pil_image.size)
                new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                
            words = item.get("words", [])
            labels = item.get("labels", [])
            grouped_words = item.get("grouped_words", [])
            linked_groups = item.get("linked_groups", [])
            doc_id = str(item.get("id", idx))
            
            entities = []
            relations = []
            
            # 优先使用 FUNSD+ 的 grouped_words 精准划分实体
            if grouped_words:
                for group_id, word_indices in enumerate(grouped_words):
                    if not word_indices: continue
                    text = " ".join([words[i] for i in word_indices if i < len(words)])
                        
                    raw_label_val = labels[word_indices[0]] if word_indices[0] < len(labels) else 0
                    raw_label = int2str(raw_label_val) if isinstance(raw_label_val, int) else str(raw_label_val)
                    clean_label = raw_label.replace("B-", "").replace("I-", "").lower()
                    if clean_label == "o": clean_label = "other"
                    
                    entities.append({"id": group_id, "text": text, "label": clean_label})
                    
                # 解析 FUNSD+ 的关系链接
                if linked_groups:
                    for link in linked_groups:
                        if len(link) >= 2:
                            relations.append({"head": link[0], "tail": link[1], "type": "linked"})
                            
            gold = {"entities": entities, "relations": relations}
            full_text = " ".join(words)
            
            # --- 内部的分发模式逻辑 ---
            if mode == "full":
                yield {
                    "question": full_text,
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "dataset": "funsd",
                    "doc_id": doc_id,
                    "image": pil_image
                }
            elif mode == "chunks":
                yield {
                    "question": full_text,
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "chunk_info": f"Image {doc_id}",
                    "dataset": "funsd",
                    "doc_id": doc_id,
                    "image": pil_image
                }
            elif mode == "partitioned":
                yield {
                    "question": full_text,
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "partition_info": f"Image {doc_id}",
                    "dataset": "funsd",
                    "doc_id": doc_id,
                    "image": pil_image
                }
        
        # HF 分支结束，直接返回
        return  

    # ==========================================
    # 模式 A: 从本地路径加载 (保留官方代码原样)
    # ==========================================
    print(f"📂 检测到本地路径，正在从本地加载 FUNSD 数据集: {doc_path}")
    with open(doc_path, 'r', encoding='utf-8') as f:
        if doc_path.endswith('.json'):
            data = json.load(f)
        else:
            full_text = f.read()
            data = [{"text": full_text}]
    
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        base_dir = os.path.dirname(doc_path)
        ann_dir = annotations_dir or os.path.join(base_dir, "annotations")
        img_dir = images_dir or os.path.join(base_dir, "images")
        
        for img_info in data["images"]:
            file_name = img_info.get("file_name", "")
            segm_file = img_info.get("segm_file", "")
            image_id = img_info.get("id")
            
            image_obj = None
            if img_dir:
                img_path = os.path.join(img_dir, file_name)
                if os.path.exists(img_path):
                    try:
                        image_obj = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        print(f"[Warning] Failed to load image {img_path}: {e}")
            
            gold = {"entities": [], "relations": []}
            full_text = ""
            
            if segm_file and ann_dir:
                segm_path = os.path.join(ann_dir, segm_file)
                if os.path.exists(segm_path):
                    try:
                        with open(segm_path, 'r', encoding='utf-8') as sf:
                            segm_data = json.load(sf)
                            if "form" in segm_data:
                                form = segm_data["form"]
                                texts = []
                                for item in form:
                                    entity = {
                                        "id": item.get("id"),
                                        "text": item.get("text", ""),
                                        "label": item.get("label", "other")
                                    }
                                    gold["entities"].append(entity)
                                    texts.append(item.get("text", ""))
                                    
                                    for link in item.get("linking", []):
                                        gold["relations"].append({
                                            "head": link[0],
                                            "tail": link[1],
                                            "type": "linked"
                                        })
                                full_text = " ".join(texts)
                    except Exception as e:
                        pass
            
            if not full_text and image_obj: full_text = f"[Form image: {file_name}]"
            
            if mode == "full":
                yield {"question": full_text, "solution": json.dumps(gold, ensure_ascii=False), "gold": json.dumps(gold, ensure_ascii=False), "extract_template": json.dumps(extract_template, ensure_ascii=False), "dataset": "funsd", "doc_id": file_name, **({"image": image_obj} if image_obj else {})}
            elif mode == "chunks":
                yield {"question": full_text, "solution": json.dumps(gold, ensure_ascii=False), "gold": json.dumps(gold, ensure_ascii=False), "extract_template": json.dumps(extract_template, ensure_ascii=False), "chunk_info": f"Image {file_name}", "dataset": "funsd", "doc_id": file_name, **({"image": image_obj} if image_obj else {})}
            elif mode == "partitioned":
                yield {"question": full_text, "solution": json.dumps(gold, ensure_ascii=False), "gold": json.dumps(gold, ensure_ascii=False), "extract_template": json.dumps(extract_template, ensure_ascii=False), "partition_info": f"Image {file_name}", "dataset": "funsd", **({"image": image_obj} if image_obj else {})}
        return


def load_finer(
    doc_path: str = "nlpaueb/finer-139",
    split: str = "train",
    mode: str = "full",
    chunk_size: int = 3000,
    overlap: int = 300,
    num_partitions: int = 3,
    cache_dir: Optional[str] = None,
    tag2id_path: Optional[str] = None
) -> Iterable[Dict]:
    """
    Load FinER-139 dataset directly from HuggingFace Hub.
    Auto-converts IOB2 tokens to entity spans with precise character-level offsets.
    """
    import json
    from datasets import load_dataset

    # 1. 自动从 Hugging Face 加载数据集
    # 如果 doc_path 是 "nlpaueb/finer-139"，直接从云端/缓存加载
    if doc_path == "nlpaueb/finer-139":
        print(f"Loading FinER-139 from HuggingFace Hub (split: {split})...")
        ds = load_dataset("nlpaueb/finer-139", split=split, cache_dir=cache_dir, trust_remote_code=True)
        # 直接从 HF 数据集特征中提取标签映射字典
        tag_names = ds.features['ner_tags'].feature.names
        id2tag = {i: name for i, name in enumerate(tag_names)}
    else:
        # 兼容旧版的本地 JSON 文件加载逻辑
        import os
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"FinER file not found: {doc_path}")
        with open(doc_path, 'r', encoding='utf-8') as f:
            ds = json.load(f) if doc_path.endswith('.json') else [{"text": f.read()}]
        # 这里省略了本地找 tag2id 的代码，因为推荐直接用 HF 数据集
        id2tag = {} 

    def iob2_to_entities_with_char_offsets(tokens, ner_tags, id2tag):
        """将 IOB2 转换为实体列表，并精准计算字符级坐标(Character-level start/end)"""
        entities = []
        current_entity = None
        current_char_pos = 0  # 追踪字符索引
        
        for token, tag_id in zip(tokens, ner_tags):
            tag = id2tag.get(tag_id, "O") if id2tag else "O"
            token_len = len(token)
            
            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "type": tag[2:],  # Prompts 中要求输出的是 type
                    "start": current_char_pos,
                    "end": current_char_pos + token_len
                }
            elif tag.startswith("I-") and current_entity and tag[2:] == current_entity["type"]:
                current_entity["text"] += " " + token
                current_entity["end"] = current_char_pos + token_len
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                    
            # 加 1 是因为后面的 full_text 拼接时是用空格(" ")拼接的
            current_char_pos += token_len + 1
            
        if current_entity:
            entities.append(current_entity)
            
        return {"entities": entities}

    # Standard FinER extraction schema
    extract_template = {
        "entities": [
            {"text": "", "type": "", "start": 0, "end": 0}
        ]
    }
    
    for item in ds:
        if "tokens" in item and "ner_tags" in item:
            tokens = item["tokens"]
            ner_tags = item["ner_tags"]
            full_text = " ".join(tokens)
            gold_dict = iob2_to_entities_with_char_offsets(tokens, ner_tags, id2tag)
        else:
            full_text = item.get("text", "") or str(item)
            gold_dict = {"entities": item.get("entities", [])}
            
        if mode == "full":
            yield {
                "question": full_text,
                "solution": json.dumps(gold_dict, ensure_ascii=False),
                "gold": json.dumps(gold_dict, ensure_ascii=False),
                "extract_template": json.dumps(extract_template, ensure_ascii=False),
                "dataset": "finer",
            }
        
        elif mode == "chunks":
            chunks = []
            start = 0
            while start < len(full_text):
                end = start + chunk_size
                chunks.append(full_text[start:end])
                start = end - overlap
            
            for i, chunk in enumerate(chunks):
                yield {
                    "question": chunk,
                    "solution": json.dumps(gold_dict, ensure_ascii=False),
                    "gold": json.dumps(gold_dict, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "chunk_info": f"Chunk {i+1}/{len(chunks)}",
                    "dataset": "finer",
                }
        
        elif mode == "partitioned":
            partition_size = len(full_text) // num_partitions
            for i in range(num_partitions):
                start = i * partition_size
                end = start + partition_size if i < num_partitions - 1 else len(full_text)
                
                yield {
                    "question": full_text[start:end],
                    "solution": json.dumps(gold_dict, ensure_ascii=False),
                    "gold": json.dumps(gold_dict, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "partition_info": f"Partition {i+1}/{num_partitions}",
                    "dataset": "finer",
                }


def load_chemprot(split: str = "train", max_samples: Optional[int] = None) -> Iterable[Dict]:
    from datasets import load_dataset
    import json

    print("⏳ 正在从 HuggingFace 加载 ChemProt 数据集...")
    # 使用 bigbio_kb schema，它把实体和关系整理得最好
    dataset = load_dataset("bigbio/chemprot", "chemprot_bigbio_kb", split=split)

    # 仅保留官方打榜的 5 种核心关系，但键名必须是小写，以匹配 bigbio 的真实数据
    valid_relations = {
        "up-regulator": "UPREGULATOR",
        "down-regulator": "DOWNREGULATOR",
        "agonist": "AGONIST",
        "antagonist": "ANTAGONIST",
        "substrate": "SUBSTRATE"
    }

    count = 0
    for item in dataset:
        passages = item.get("passages", [])
        text = " ".join([p.get("text", [""])[0] for p in passages if isinstance(p, dict)]).strip()

        # 建立实体ID到文本的映射，用于解析关系
        id2text = {}
        for ent in item.get("entities", []):
            if isinstance(ent, dict):
                ent_id = ent.get("id")
                ent_text = ent.get("text", [""])
                ent_text = ent_text[0] if isinstance(ent_text, list) and ent_text else ""
                if ent_id:
                    id2text[ent_id] = ent_text

        relations = []
        for rel in item.get("relations", []):
            if not isinstance(rel, dict):
                continue
            rel_type = str(rel.get("type", "")).lower()
            # 仅提取有效的5大类交互关系
            if rel_type in valid_relations:
                arg1_id = rel.get("arg1_id")
                arg2_id = rel.get("arg2_id")
                head_text = id2text.get(arg1_id, "")
                tail_text = id2text.get(arg2_id, "")
                if head_text and tail_text:
                    relations.append({
                        "head": head_text,
                        "relation": valid_relations[rel_type],
                        "tail": tail_text,
                    })

        gold_json = {"relations": relations}

        yield {
            "question": text,
            "gold": json.dumps(gold_json, ensure_ascii=False),
            "solution": json.dumps(gold_json, ensure_ascii=False),
            "extract_template": '{"relations": [{"head": "", "relation": "", "tail": ""}]}',
            "dataset": "chemprot",
        }

        count += 1
        if max_samples and count >= max_samples:
            break

    print(f"Loaded {count} ChemProt samples.")
