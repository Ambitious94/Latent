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
                    "vertex_set": vertex_set,
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
    Supports official CORD format from samples.json:
    Format: {"num_items": int, "subtotal_price": str, "service_price": str, "tax_price": str, "total_price": str, "etc": str}
    Supports multimodal input with image_path parameter or filepath field in data.
    """
    import json
    import os
    from PIL import Image
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"CORD file not found: {doc_path}")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        if doc_path.endswith('.json'):
            data = json.load(f)
        else:
            # Plain text OCR output
            full_text = f.read()
            data = [{"text": full_text}]
    
    # Handle official CORD format with "samples" wrapper
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    
    # Standard CORD extraction schema (official format)
    extract_template = {
        "num_items": 0,
        "subtotal_price": "",
        "service_price": "",
        "tax_price": "",
        "total_price": "",
        "etc": ""
    }
    
    for doc in (data if isinstance(data, list) else [data]):
        # Load image from filepath field or parameter
        image_obj = None
        doc_image_path = doc.get("filepath") or image_path
        if doc_image_path:
            # Handle relative path from doc_path directory
            if not os.path.isabs(doc_image_path):
                base_dir = os.path.dirname(doc_path)
                doc_image_path = os.path.join(base_dir, doc_image_path)
            
            if os.path.exists(doc_image_path):
                try:
                    image_obj = Image.open(doc_image_path).convert("RGB")
                except Exception as e:
                    print(f"[Warning] Failed to load image from {doc_image_path}: {e}")
        
        # Extract text and ground truth
        full_text = doc.get("text", "")
        
        # Build ground truth from official CORD fields
        gold = {}
        for field in ["num_items", "subtotal_price", "service_price", "tax_price", "total_price", "etc"]:
            if field in doc:
                gold[field] = doc[field]
        
        # Fallback to custom ground_truth field if present
        if not gold and "ground_truth" in doc:
            gold = doc["ground_truth"]
        
        # If no text but has image, use placeholder
        if not full_text and image_obj:
            full_text = "[Image-based receipt]"
        
        if mode == "full":
            result = {
                "question": full_text,
                "solution": json.dumps(gold, ensure_ascii=False),
                "gold": json.dumps(gold, ensure_ascii=False),
                "extract_template": json.dumps(extract_template, ensure_ascii=False),
                "dataset": "cord",
            }
            if image_obj:
                result["image"] = image_obj  # Add image object for multimodal processing
            yield result
        
        elif mode == "chunks":
            chunks = []
            start = 0
            while start < len(full_text):
                end = start + chunk_size
                chunks.append(full_text[start:end])
                start = end - overlap
            
            for i, chunk in enumerate(chunks):
                result = {
                    "question": chunk,
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "chunk_info": f"Chunk {i+1}/{len(chunks)}",
                    "dataset": "cord",
                }
                if image_obj:
                    result["image"] = image_obj
                yield result
        
        elif mode == "partitioned":
            partition_size = len(full_text) // num_partitions
            for i in range(num_partitions):
                start = i * partition_size
                end = start + partition_size if i < num_partitions - 1 else len(full_text)
                
                result = {
                    "question": full_text[start:end],
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "partition_info": f"Partition {i+1}/{num_partitions}",
                    "dataset": "cord",
                }
                if image_obj:
                    result["image"] = image_obj
                yield result


def load_funsd(
    doc_path: str,
    split: str = "train",
    mode: str = "chunks",
    chunk_size: int = 2500,
    overlap: int = 250,
    num_partitions: int = 3,
    cache_dir: Optional[str] = None,
    image_path: Optional[str] = None,  # New parameter for multimodal support
    annotations_dir: Optional[str] = None,  # Directory containing segm_file JSONs
    images_dir: Optional[str] = None  # Directory containing form images
) -> Iterable[Dict]:
    """
    Load FUNSD dataset for form understanding.
    
    Supports two formats:
    1. Official COCO-style: instances_test.json with images, categories, annotations
       - Requires annotations_dir for segm_file JSONs containing entities/relations
       - Requires images_dir for form images
    2. Simple format: {"text": "...", "entities": [...], "relations": [...]}
    
    Output format: {"entities": [{"text": "", "label": "question/answer/header/other", "box": []}], 
                    "relations": [{"head": 0, "tail": 1, "type": "answer_to"}]}
    """
    import json
    import os
    from PIL import Image
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"FUNSD file not found: {doc_path}")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        if doc_path.endswith('.json'):
            data = json.load(f)
        else:
            full_text = f.read()
            data = [{"text": full_text}]
    
    # Standard FUNSD extraction schema
    extract_template = {
        "entities": [
            {"text": "", "label": "", "box": []}
        ],
        "relations": [
            {"head": 0, "tail": 1, "type": ""}
        ]
    }
    
    # Detect COCO-style format (official FUNSD)
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        # COCO-style format
        base_dir = os.path.dirname(doc_path)
        ann_dir = annotations_dir or os.path.join(base_dir, "annotations")
        img_dir = images_dir or os.path.join(base_dir, "images")
        
        for img_info in data["images"]:
            file_name = img_info.get("file_name", "")
            segm_file = img_info.get("segm_file", "")
            image_id = img_info.get("id")
            
            # Load image
            image_obj = None
            if img_dir:
                img_path = os.path.join(img_dir, file_name)
                if os.path.exists(img_path):
                    try:
                        image_obj = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        print(f"[Warning] Failed to load image {img_path}: {e}")
            
            # Load segm_file for entities and relations
            gold = {"entities": [], "relations": []}
            full_text = ""
            
            if segm_file and ann_dir:
                segm_path = os.path.join(ann_dir, segm_file)
                if os.path.exists(segm_path):
                    try:
                        with open(segm_path, 'r', encoding='utf-8') as sf:
                            segm_data = json.load(sf)
                            # Parse FUNSD segm format
                            if "form" in segm_data:
                                form = segm_data["form"]
                                texts = []
                                for item in form:
                                    entity = {
                                        "text": item.get("text", ""),
                                        "label": item.get("label", "other"),
                                        "box": item.get("box", []),
                                        "id": item.get("id")
                                    }
                                    gold["entities"].append(entity)
                                    texts.append(item.get("text", ""))
                                    
                                    # Extract linking relations
                                    for link in item.get("linking", []):
                                        gold["relations"].append({
                                            "head": link[0],
                                            "tail": link[1],
                                            "type": "linked"
                                        })
                                full_text = " ".join(texts)
                    except Exception as e:
                        print(f"[Warning] Failed to load segm_file {segm_path}: {e}")
            
            # If no segm_file, extract from COCO annotations
            if not full_text:
                # Get annotations for this image
                img_anns = [a for a in data["annotations"] if a.get("image_id") == image_id]
                full_text = f"[Form image: {file_name}]"
                gold = {"entities": [], "annotations_count": len(img_anns)}
            
            if not full_text and image_obj:
                full_text = f"[Form image: {file_name}]"
            
            if mode == "full":
                result = {
                    "question": full_text,
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "dataset": "funsd",
                    "doc_id": file_name,
                }
                if image_obj:
                    result["image"] = image_obj
                yield result
            
            elif mode == "chunks":
                # For COCO format with chunks mode - treat each image as one item
                result = {
                    "question": full_text,
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "chunk_info": f"Image {file_name}",
                    "dataset": "funsd",
                    "doc_id": file_name,
                }
                if image_obj:
                    result["image"] = image_obj
                yield result
            
            elif mode == "partitioned":
                # For COCO format, each image is a partition
                result = {
                    "question": full_text,
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "partition_info": f"Image {file_name}",
                    "dataset": "funsd",
                }
                if image_obj:
                    result["image"] = image_obj
                yield result
        
        return  # Exit after processing COCO format
    
    # Simple format processing
    # Load image if path provided (for multimodal)
    image_obj = None
    if image_path and os.path.exists(image_path):
        try:
            image_obj = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Failed to load image from {image_path}: {e}")
    
    for doc in (data if isinstance(data, list) else [data]):
        full_text = doc.get("text", "") or str(doc)
        gold = doc.get("annotations", {})
        
        if mode == "full":
            result = {
                "question": full_text,
                "solution": json.dumps(gold, ensure_ascii=False),
                "gold": json.dumps(gold, ensure_ascii=False),
                "extract_template": json.dumps(extract_template, ensure_ascii=False),
                "dataset": "funsd",
            }
            if image_obj:
                result["image"] = image_obj
            yield result
        
        elif mode == "chunks":
            chunks = []
            start = 0
            while start < len(full_text):
                end = start + chunk_size
                chunks.append(full_text[start:end])
                start = end - overlap
            
            for i, chunk in enumerate(chunks):
                result = {
                    "question": chunk,
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "chunk_info": f"Chunk {i+1}/{len(chunks)}",
                    "dataset": "funsd",
                }
                if image_obj:
                    result["image"] = image_obj
                yield result
        
        elif mode == "partitioned":
            partition_size = len(full_text) // num_partitions
            for i in range(num_partitions):
                start = i * partition_size
                end = start + partition_size if i < num_partitions - 1 else len(full_text)
                
                result = {
                    "question": full_text[start:end],
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "partition_info": f"Partition {i+1}/{num_partitions}",
                    "dataset": "funsd",
                }
                if image_obj:
                    result["image"] = image_obj
                yield result


def load_finer(
    doc_path: str,
    split: str = "train",
    mode: str = "chunks",
    chunk_size: int = 3000,
    overlap: int = 300,
    num_partitions: int = 3,
    cache_dir: Optional[str] = None,
    tag2id_path: Optional[str] = None  # Path to tag2id.json for IOB2 format
) -> Iterable[Dict]:
    """
    Load FinER-139 dataset for fine-grained financial entity recognition.
    Supports both formats:
    1. Already converted: {"text": "...", "entities": [{"text": "", "label": "", "start": 0, "end": 0}]}
    2. Official IOB2: {"tokens": [...], "ner_tags": [...]} - automatically converts to entities
    """
    import json
    import os
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"FinER file not found: {doc_path}")
    
    # Helper function to convert IOB2 tags to entities
    def iob2_to_entities(tokens, ner_tags, id2tag):
        """Convert IOB2 format to entity list"""
        entities = []
        current_entity = None
        
        for i, (token, tag_id) in enumerate(zip(tokens, ner_tags)):
            tag = id2tag.get(tag_id, "O")
            
            if tag.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                # Start new entity
                label = tag[2:]
                current_entity = {
                    "text": token,
                    "label": label,
                    "start": i,
                    "end": i
                }
            elif tag.startswith("I-") and current_entity:
                # Continue current entity
                current_entity["text"] += " " + token
                current_entity["end"] = i
            else:  # O tag
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Save last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        if doc_path.endswith('.json'):
            data = json.load(f)
        else:
            full_text = f.read()
            data = [{"text": full_text}]
    
    # Load tag2id mapping if IOB2 format detected
    id2tag = None
    if isinstance(data, list) and data and "ner_tags" in data[0]:
        # Detect IOB2 format, need tag2id mapping
        if tag2id_path and os.path.exists(tag2id_path):
            with open(tag2id_path, 'r', encoding='utf-8') as f:
                tag2id = json.load(f)
                id2tag = {v: k for k, v in tag2id.items()}
        else:
            # Try to find tag2id.json in same directory
            base_dir = os.path.dirname(doc_path)
            default_tag2id = os.path.join(base_dir, "tag2id.json")
            if os.path.exists(default_tag2id):
                with open(default_tag2id, 'r', encoding='utf-8') as f:
                    tag2id = json.load(f)
                    id2tag = {v: k for k, v in tag2id.items()}
            else:
                # Auto-generate id2tag from data if not found
                print(f"⚠️  tag2id.json not found, auto-generating from data...")
                all_tag_ids = set()
                for doc in data:
                    if "ner_tags" in doc:
                        all_tag_ids.update(doc["ner_tags"])
                
                # Create a default mapping (0 -> O, others -> ENTITY-{id})
                id2tag = {0: "O"}
                for tag_id in sorted(all_tag_ids):
                    if tag_id != 0 and tag_id not in id2tag:
                        id2tag[tag_id] = f"ENTITY-{tag_id}"
                
                print(f"✓ Auto-generated {len(id2tag)} tag mappings: {id2tag}")
                
                # Optionally save the mapping for future use
                try:
                    with open(default_tag2id, 'w', encoding='utf-8') as f:
                        tag2id_auto = {v: k for k, v in id2tag.items()}
                        json.dump(tag2id_auto, f, indent=2, ensure_ascii=False)
                    print(f"✓ Saved auto-generated mapping to {default_tag2id}")
                except:
                    pass
    
    # Standard FinER extraction schema
    extract_template = {
        "entities": [
            {"text": "", "label": "", "start": 0, "end": 0}
        ]
    }
    
    for doc in (data if isinstance(data, list) else [data]):
        # Handle IOB2 format
        if "tokens" in doc and "ner_tags" in doc:
            tokens = doc["tokens"]
            ner_tags = doc["ner_tags"]
            full_text = " ".join(tokens)
            gold = iob2_to_entities(tokens, ner_tags, id2tag)
        else:
            # Already converted format
            full_text = doc.get("text", "") or str(doc)
            gold = doc.get("entities", [])
        
        if mode == "full":
            yield {
                "question": full_text,
                "solution": json.dumps(gold, ensure_ascii=False),
                "gold": json.dumps(gold, ensure_ascii=False),
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
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
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
                    "solution": json.dumps(gold, ensure_ascii=False),
                    "gold": json.dumps(gold, ensure_ascii=False),
                    "extract_template": json.dumps(extract_template, ensure_ascii=False),
                    "partition_info": f"Partition {i+1}/{num_partitions}",
                    "dataset": "finer",
                }
