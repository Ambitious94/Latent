"""
Evaluation metrics for document extraction tasks
"""
import json
import re
from typing import Dict, List, Any
from collections import defaultdict
from prompts import DOCRED_REL_MAP, REL_NAME_TO_ID


def extract_json_from_text(text: str) -> dict:
    """从模型输出中提取JSON,支持多种格式"""
    if not text:
        return {}
    
    # 尝试直接解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    
    # 尝试找到JSON块
    patterns = [
        r'```json\s*(.*?)\s*```',  # markdown json块
        r'```\s*(.*?)\s*```',       # 普通代码块
        r'\{[^{}]*"relations"[^{}]*\[.*?\]\s*\}',  # relations格式
        r'\{.*\}',                  # 任意JSON对象
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    
    return {}


def normalize_entity_name(name: str) -> str:
    """标准化实体名称用于匹配"""
    if not name:
        return ""
    # 转小写，去除多余空格，去除特殊字符
    normalized = name.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def _normalize_relation(rel: str) -> str:
    """将关系字符串统一为 P-ID 格式，支持自然语言名和 P-ID 两种输入"""
    rel = rel.strip()
    # 已经是 P-ID 格式 (P17, P131, ...)
    if re.match(r'^P\d+$', rel):
        return rel
    # 尝试从自然语言名反向查找 P-ID
    pid = REL_NAME_TO_ID.get(rel.lower())
    if pid:
        return pid
    # 容错: 尝试模糊匹配 (去除冠词、介词差异)
    rel_lower = rel.lower()
    for name, pid in REL_NAME_TO_ID.items():
        if rel_lower in name or name in rel_lower:
            return pid
    # 无法映射，原样返回
    return rel


def _resolve_entity(r: dict, vertex_set: list) -> tuple:
    """从关系dict中解析出 (head_name, tail_name)，支持索引和名字两种格式"""
    # 优先使用索引格式 (head_id / tail_id)
    head_id = r.get("head_id")
    tail_id = r.get("tail_id")
    if head_id is not None and tail_id is not None:
        try:
            head_id = int(head_id)
            tail_id = int(tail_id)
        except (ValueError, TypeError):
            head_id = tail_id = None
    
    if head_id is not None and tail_id is not None and vertex_set:
        head = ""
        tail = ""
        if 0 <= head_id < len(vertex_set) and vertex_set[head_id]:
            head = vertex_set[head_id][0].get("name", "")
        if 0 <= tail_id < len(vertex_set) and vertex_set[tail_id]:
            tail = vertex_set[tail_id][0].get("name", "")
        return normalize_entity_name(head), normalize_entity_name(tail)
    
    # Fallback: 名字格式 (head / tail)
    head = normalize_entity_name(r.get("head", ""))
    tail = normalize_entity_name(r.get("tail", ""))
    return head, tail


def evaluate_docred(predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    评估 DocRED 关系抽取
    
    Metrics: Precision, Recall, F1 for relation triplets
    支持:
    1. 索引抽取 (head_id/tail_id) 和名字抽取 (head/tail) 双模式
    2. 实体名称模糊匹配(忽略大小写)
    3. 从模型输出中智能提取JSON
    4. 自然语言关系名 ↔ P-ID 双向匹配
    5. 分别统计每个关系类型的性能
    """
    pred_relations = []
    gold_relations = []
    
    # 按关系类型统计
    relation_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for pred, gold in zip(predictions, golds):
        # 获取 vertex_set 用于索引→名字解析
        vertex_set = []
        if isinstance(gold, dict):
            vertex_set = gold.get("vertex_set", [])
        if not vertex_set and isinstance(pred, dict):
            vertex_set = pred.get("vertex_set", [])
        
        # 解析预测结果
        try:
            pred_text = pred.get("prediction", "")
            pred_data = extract_json_from_text(pred_text)
            pred_rels = pred_data.get("relations", [])
            
            for r in pred_rels:
                head, tail = _resolve_entity(r, vertex_set)
                rel = _normalize_relation(r.get("relation", ""))
                if head and tail and rel:
                    pred_relations.append((head, rel, tail))
        except Exception as e:
            pass
        
        # 解析金标准
        try:
            gold_text = gold.get("gold", gold) if isinstance(gold, dict) else gold
            gold_data = json.loads(gold_text) if isinstance(gold_text, str) else gold_text
            gold_rels = gold_data.get("relations", []) if isinstance(gold_data, dict) else []
            
            for r in gold_rels:
                head, tail = _resolve_entity(r, vertex_set)
                rel = _normalize_relation(r.get("relation", ""))
                if head and tail and rel:
                    gold_relations.append((head, rel, tail))
        except Exception as e:
            pass
    
    pred_set = set(pred_relations)
    gold_set = set(gold_relations)
    
    # 计算整体指标
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 按关系类型统计
    for rel_tuple in pred_set & gold_set:
        relation_stats[rel_tuple[1]]["tp"] += 1
    for rel_tuple in pred_set - gold_set:
        relation_stats[rel_tuple[1]]["fp"] += 1
    for rel_tuple in gold_set - pred_set:
        relation_stats[rel_tuple[1]]["fn"] += 1
    
    # 计算每个关系类型的F1
    per_relation_f1 = {}
    for rel, stats in relation_stats.items():
        r_tp, r_fp, r_fn = stats["tp"], stats["fp"], stats["fn"]
        r_prec = r_tp / (r_tp + r_fp) if (r_tp + r_fp) > 0 else 0.0
        r_rec = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0 else 0.0
        r_f1 = 2 * r_prec * r_rec / (r_prec + r_rec) if (r_prec + r_rec) > 0 else 0.0
        per_relation_f1[rel] = {"precision": r_prec, "recall": r_rec, "f1": r_f1}
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "pred_count": len(pred_set),
        "gold_count": len(gold_set),
        "per_relation": per_relation_f1,
        "unique_relations_predicted": len(set(r[1] for r in pred_set)),
        "unique_relations_gold": len(set(r[1] for r in gold_set)),
    }


def evaluate_cord(predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    评估嵌套结构的 CORD 收据抽取 (支持 Menu 列表无序匹配 + Total 字典匹配)
    """
    tp = 0
    pred_count = 0
    gold_count = 0

    for pred, gold in zip(predictions, golds):
        # 解析预测结果
        try:
            pred_data = json.loads(pred.get("prediction", "{}"))
        except (json.JSONDecodeError, TypeError, ValueError):
            pred_data = {}

        # 解析金标准
        gold_data = json.loads(gold) if isinstance(gold, str) else gold
        if not isinstance(gold_data, dict):
            gold_data = {}

        # ==========================================
        # 1. 评估 Total 字典 (平铺字段对比)
        # ==========================================
        pred_total = pred_data.get("total", {}) if isinstance(pred_data.get("total"), dict) else {}
        gold_total = gold_data.get("total", {}) if isinstance(gold_data.get("total"), dict) else {}

        eval_total_keys = ["total_price", "cashprice", "changeprice", "subtotal_price", "tax_price"]
        for key in eval_total_keys:
            p_val = str(pred_total.get(key, "")).strip()
            g_val = str(gold_total.get(key, "")).strip()

            if p_val:
                pred_count += 1
            if g_val:
                gold_count += 1
            if p_val and g_val and p_val == g_val:
                tp += 1

        # ==========================================
        # 2. 评估 Menu 列表 (无序多重集对比)
        # ==========================================
        pred_menu = pred_data.get("menu", []) if isinstance(pred_data.get("menu"), list) else []
        gold_menu = gold_data.get("menu", []) if isinstance(gold_data.get("menu"), list) else []

        # 将字典转换为不可变的 Tuple，以便进行集合操作与统计
        # 格式: (nm, cnt, price)
        def menu_to_tuples(menu_list):
            res = []
            for item in menu_list:
                if isinstance(item, dict):
                    nm = str(item.get("nm", "")).strip()
                    cnt = str(item.get("cnt", "")).strip()
                    price = str(item.get("price", "")).strip()
                    # 只要有一个字段非空，就认为提取了一个有效条目
                    if nm or cnt or price:
                        res.append((nm, cnt, price))
            return res

        p_tuples = menu_to_tuples(pred_menu)
        g_tuples = menu_to_tuples(gold_menu)

        pred_count += len(p_tuples) * 3  # 每个商品算作 3 个属性的提取
        gold_count += len(g_tuples) * 3

        # 贪心匹配：寻找完全一样的元组并从候选中剔除（支持重复购买同一商品）
        g_tuples_unmatched = list(g_tuples)
        for p_tup in p_tuples:
            if p_tup in g_tuples_unmatched:
                tp += 3  # nm, cnt, price 全中
                g_tuples_unmatched.remove(p_tup)
            else:
                # 容错匹配：如果不是全对，算算对了几个字段 (部分得分)
                best_overlap = 0
                best_idx = -1
                for i, g_tup in enumerate(g_tuples_unmatched):
                    overlap = sum(1 for p, g in zip(p_tup, g_tup) if p == g and p != "")
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = i

                if best_idx != -1:
                    tp += best_overlap
                    g_tuples_unmatched.pop(best_idx)

    # 计算最终的 Precision, Recall, F1
    precision = tp / pred_count if pred_count > 0 else 0.0
    recall = tp / gold_count if gold_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "pred_count": pred_count,
        "gold_count": gold_count
    }


def evaluate_funsd(predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    评估 FUNSD 表单理解
    
    Metrics: Entity-level and Relation-level F1
    """
    # Entity evaluation
    pred_entities = []
    gold_entities = []
    
    # Relation evaluation
    pred_relations = []
    gold_relations = []
    
    for pred, gold in zip(predictions, golds):
        try:
            # Handle prediction - may be string or dict
            if isinstance(pred, dict):
                pred_str = pred.get("prediction", "{}")
            else:
                pred_str = str(pred) if pred else "{}"
            pred_data = json.loads(pred_str) if isinstance(pred_str, str) else pred_str
            
            # Handle gold - may be string or dict
            gold_data = json.loads(gold) if isinstance(gold, str) else gold
            
            # Ensure both are dicts
            if not isinstance(pred_data, dict):
                pred_data = {}
            if not isinstance(gold_data, dict):
                gold_data = {}
        except Exception as e:
            print(f"[Warning] Failed to parse prediction/gold: {e}")
            pred_data = {}
            gold_data = {}
            continue
        
        # Extract entities
        pred_ents = pred_data.get("entities", [])
        gold_ents = gold_data.get("entities", [])
        
        # 新增：建立 ID 到 Text 的映射字典，用于解析关系
        pred_id2text = {e.get("id"): str(e.get("text", "")).strip() for e in pred_ents if "id" in e}
        gold_id2text = {e.get("id"): str(e.get("text", "")).strip() for e in gold_ents if "id" in e}
        
        # 提取实体 (忽略 ID，只对比提取的文本和标签)
        pred_entities.extend([(str(e.get("text", "")).strip(), e.get("label", "")) for e in pred_ents if e.get("text")])
        gold_entities.extend([(str(e.get("text", "")).strip(), e.get("label", "")) for e in gold_ents if e.get("text")])
        
        # 提取关系并智能解析
        pred_rels = pred_data.get("relations", [])
        gold_rels = gold_data.get("relations", [])
        
        for r in pred_rels:
            head_val = r.get("head")
            tail_val = r.get("tail")
            # 如果是数字ID，则去映射表里找对应的文本；如果没找到，就回退为字符串
            head_text = pred_id2text.get(head_val, str(head_val) if head_val is not None else "")
            tail_text = pred_id2text.get(tail_val, str(tail_val) if tail_val is not None else "")
            if head_text and tail_text:
                pred_relations.append((head_text, tail_text, r.get("type", "linked")))
                
        for r in gold_rels:
            head_val = r.get("head")
            tail_val = r.get("tail")
            head_text = gold_id2text.get(head_val, str(head_val) if head_val is not None else "")
            tail_text = gold_id2text.get(tail_val, str(tail_val) if tail_val is not None else "")
            if head_text and tail_text:
                gold_relations.append((head_text, tail_text, r.get("type", "linked")))
    
    # Calculate entity metrics
    pred_ent_set = set(pred_entities)
    gold_ent_set = set(gold_entities)
    
    ent_tp = len(pred_ent_set & gold_ent_set)
    ent_fp = len(pred_ent_set - gold_ent_set)
    ent_fn = len(gold_ent_set - pred_ent_set)
    
    ent_precision = ent_tp / (ent_tp + ent_fp) if (ent_tp + ent_fp) > 0 else 0.0
    ent_recall = ent_tp / (ent_tp + ent_fn) if (ent_tp + ent_fn) > 0 else 0.0
    ent_f1 = 2 * ent_precision * ent_recall / (ent_precision + ent_recall) if (ent_precision + ent_recall) > 0 else 0.0
    
    # Calculate relation metrics
    pred_rel_set = set(pred_relations)
    gold_rel_set = set(gold_relations)
    
    rel_tp = len(pred_rel_set & gold_rel_set)
    rel_fp = len(pred_rel_set - gold_rel_set)
    rel_fn = len(gold_rel_set - pred_rel_set)
    
    rel_precision = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0.0
    rel_recall = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) > 0 else 0.0
    rel_f1 = 2 * rel_precision * rel_recall / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0.0
    
    return {
        "entity_precision": ent_precision,
        "entity_recall": ent_recall,
        "entity_f1": ent_f1,
        "relation_precision": rel_precision,
        "relation_recall": rel_recall,
        "relation_f1": rel_f1,
        "overall_f1": (ent_f1 + rel_f1) / 2
    }


def evaluate_chemprot(predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    评估 ChemProt 化学-蛋白质关系抽取
    匹配标准: (head_text, relation_type, tail_text) 三元组完全匹配
    支持实体列表吸附：若模型输出实体名与标准实体列表中某项完全匹配（忽略大小写），则替换为标准名
    """
    tp = fp = fn = 0

    for pred, gold in zip(predictions, golds):
        # 1. 解析预测结果
        try:
            pred_data = json.loads(pred.get("prediction", "{}"))
        except (json.JSONDecodeError, TypeError, ValueError):
            pred_data = {}

        # 2. 解析金标准
        gold_data = json.loads(gold) if isinstance(gold, str) else gold
        if not isinstance(gold_data, dict):
            gold_data = {}

        # 3. 提取关系列表 (加上类型检查防弹衣)
        pred_rels_raw = pred_data.get("relations", [])
        gold_rels_raw = gold_data.get("relations", [])
        if not isinstance(pred_rels_raw, list):
            pred_rels_raw = []
        if not isinstance(gold_rels_raw, list):
            gold_rels_raw = []

        # 3. 提取实体标准名集合（用于吸附）
        entities_meta = pred.get("entities_meta", [])
        canonical = {e["text"].strip().lower(): e["text"].strip() for e in entities_meta if isinstance(e, dict)}

        def snap(name: str) -> str:
            """将模型输出的实体名吸附到标准名（忽略大小写）"""
            key = name.strip().lower()
            return canonical.get(key, name.strip())

        # 4. 构建三元组集合 (转小写，去除空格，提升鲁棒性)
        pred_rels = set()
        for r in pred_rels_raw:
            if isinstance(r, dict) and "head" in r and "tail" in r and "relation" in r:
                h = snap(str(r["head"])).lower()
                t = snap(str(r["tail"])).lower()
                rel = str(r["relation"]).strip().lower()
                if h and t and rel:
                    pred_rels.add((h, rel, t))

        gold_rels = set()
        for r in gold_rels_raw:
            if isinstance(r, dict) and "head" in r and "tail" in r and "relation" in r:
                h = str(r["head"]).strip().lower()
                t = str(r["tail"]).strip().lower()
                rel = str(r["relation"]).strip().lower()
                if h and t and rel:
                    gold_rels.add((h, rel, t))

        # 5. 计算交并集
        tp += len(pred_rels & gold_rels)
        fp += len(pred_rels - gold_rels)
        fn += len(gold_rels - pred_rels)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }


def evaluate_finer(predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    评估 FinER-139 金融实体识别 (严格序列标注模式)
    
    通过将预测的实体映射回原文的 Token 序列，生成 BIO 标签，
    并调用与官方基线完全相同的 seqeval 库进行严格对齐评测。
    """
    import re
    try:
        from seqeval.metrics.sequence_labeling import precision_recall_fscore_support
    except ImportError:
        print("\n[WARNING] 请先安装 seqeval: pip install seqeval")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def align_to_bio(text: str, entities: List[Dict]) -> List[str]:
        """将 JSON 格式的实体转换为 BIO 标签序列"""
        tokens = []
        spans = []
        # 使用简单的正则分词匹配主流英文序列标注（按空格和标点切分）
        for match in re.finditer(r'\S+', text):
            tokens.append(match.group())
            spans.append((match.start(), match.end()))
            
        bio_tags = ["O"] * len(tokens)
        
        # 按实体长度降序排序，优先标注长实体，防止互相覆盖
        entities_sorted = sorted(entities, key=lambda x: len(x.get("text", "")), reverse=True)
        
        for ent in entities_sorted:
            ent_text = ent.get("text", "")
            # 兼容模型输出 label 或 type 两种 key
            ent_type = ent.get("type", ent.get("label", "")).upper()
            start = ent.get("start", -1)
            end = ent.get("end", -1)
            
            if not ent_text or not ent_type:
                continue
                
            # 容错降级：如果 LLM 没有输出或输出了错误的 start/end 位置，使用字符串查找回退
            if start == -1 or end == -1:
                start = text.find(ent_text)
                if start != -1:
                    end = start + len(ent_text)
                    
            if start == -1:
                continue  # 原文中确实找不到该实体，直接丢弃(当做FP或FN)
                
            # 将字符级 span 映射到 Token 级的 BIO 标签
            started = False
            for i, (tok_start, tok_end) in enumerate(spans):
                # 如果 Token 的区间和实体的区间有交集
                if max(start, tok_start) < min(end, tok_end):
                    if not started:
                        if bio_tags[i] == "O":  # 只有空位才写入
                            bio_tags[i] = f"B-{ent_type}"
                            started = True
                    else:
                        if bio_tags[i] == "O":
                            bio_tags[i] = f"I-{ent_type}"
                            
        return bio_tags

    y_true_all = []
    y_pred_all = []
    
    for pred, gold in zip(predictions, golds):
        # 1. 获取当前文档的原始文本 (在 run.py 传递 batch 时保留在 question 中)
        text = pred.get("question", "")
        if not text and isinstance(gold, dict):
            text = gold.get("question", "")
            
        if not text:
            continue # 没有原文无法进行序列对齐
            
        # 2. 解析预测数据
        try:
            pred_data = json.loads(pred.get("prediction", "{}"))
        except (json.JSONDecodeError, TypeError, ValueError):
            pred_data = {}
            
        # 3. 解析金标准数据
        gold_data = json.loads(gold) if isinstance(gold, str) else gold
        if not isinstance(gold_data, dict):
            gold_data = {}
            
        pred_ents = pred_data.get("entities", [])
        gold_ents = gold_data.get("entities", [])
        
        # 4. 生成 BIO 序列
        y_pred = align_to_bio(text, pred_ents)
        y_true = align_to_bio(text, gold_ents)
        
        y_pred_all.append(y_pred)
        y_true_all.append(y_true)
        
    # 5. 调用官方同款的 seqeval 指标计算函数
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true=y_true_all,
        y_pred=y_pred_all,
        average='micro',  # 官方 FinER 论文通常汇报 micro F1
        warn_for=('f-score',),
        beta=1,
        zero_division=0
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": 0,   # seqeval 不直接暴露绝对数量，置 0 占位
        "false_positives": 0,
        "false_negatives": 0
    }


def convert_to_official_format(predictions: List[Dict]) -> List[Dict]:
    """
    将 LLM 输出的预测列表转换为 Re-DocRED 官方评测脚本所需的格式。

    官方格式要求每条预测为：
        {'title': <doc_title>, 'h_idx': <int>, 't_idx': <int>, 'r': <P-ID>}

    :param predictions: run.py 收集的 preds 列表，每项包含 'prediction'/'title'/'vertex_set'
    :return: 官方格式的预测列表
    """
    official_preds = []

    for pred in predictions:
        title = pred.get("title", "")
        vertex_set = pred.get("vertex_set", [])

        pred_text = pred.get("prediction", "")
        pred_data = extract_json_from_text(pred_text)
        relations = pred_data.get("relations", []) if isinstance(pred_data, dict) else []

        for rel in relations:
            h_idx = rel.get("head_id")
            t_idx = rel.get("tail_id")
            rel_name = rel.get("relation", "")

            # 跳过缺少必要字段的项
            if h_idx is None or t_idx is None or not rel_name:
                continue

            try:
                h_idx = int(h_idx)
                t_idx = int(t_idx)
            except (ValueError, TypeError):
                continue

            # 索引越界检查
            if vertex_set and (h_idx >= len(vertex_set) or t_idx >= len(vertex_set)):
                continue

            # 自然语言关系名 → P-ID
            rel_name_lower = rel_name.lower().strip()
            p_id = REL_NAME_TO_ID.get(rel_name_lower)
            if not p_id:
                # 尝试直接当作 P-ID 使用
                import re as _re
                if _re.match(r'^P\d+$', rel_name):
                    p_id = rel_name
            if not p_id:
                continue

            official_preds.append({
                "title": title,
                "h_idx": h_idx,
                "t_idx": t_idx,
                "r": p_id,
            })

    return official_preds


def evaluate_extraction_task(task: str, predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """
    统一评估接口
    
    Args:
        task: 'docred', 'cord', 'funsd', 'chemprot'
        predictions: 预测结果列表
        golds: 金标准列表
    
    Returns:
        评估指标字典
    """
    if task == "docred":
        return evaluate_docred(predictions, golds)
    elif task == "cord":
        return evaluate_cord(predictions, golds)
    elif task == "funsd":
        return evaluate_funsd(predictions, golds)
    elif task == "chemprot":
        return evaluate_chemprot(predictions, golds)
    else:
        return {"error": f"Unknown task: {task}"}


def print_evaluation_results(task: str, metrics: Dict[str, float]):
    """
    打印评估结果
    """
    print("\n" + "="*60)
    print(f"📊 Evaluation Results for {task.upper()}")
    print("="*60)
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric:40s}: {value:6.2%}")
        else:
            print(f"  {metric:40s}: {value}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # 测试示例
    test_predictions = [
        {"prediction": '{"relations": [{"head": "Apple", "relation": "founded_by", "tail": "Steve Jobs"}]}'}
    ]
    test_golds = [
        [{"head": "Apple", "relation": "founded_by", "tail": "Steve Jobs"}]
    ]
    
    metrics = evaluate_docred(test_predictions, test_golds)
    print_evaluation_results("docred", metrics)
