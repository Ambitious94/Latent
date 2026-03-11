"""
将 Re-DocRED 数据集转换为 Swift (ms-swift) 微调所需的 JSONL 格式。

使用方法:
    python prepare_swift_data.py \
        --input  ./data/train_revised.json \
        --output ./data/swift_train_redocred.jsonl

输出格式（每行一条 JSON）:
    {"system": "...", "query": "...", "response": "{\"relations\": [...]}"}

注意:
- 金标准关系自动从 P-ID 转换为语义化自然语言名（与 finetune_lora.py 训练格式完全一致）
- 使用 head_id / tail_id 索引格式（不含 evidence，防止幻觉）
- system/query 与 build_lora_extraction_prompt 保持一致，推理时不会出现训练-推理不匹配
"""

import argparse
import json
import os

from prompts import DOCRED_REL_MAP, build_lora_extraction_prompt


def build_entity_list(vertex_set: list) -> str:
    lines = []
    for idx, mentions in enumerate(vertex_set):
        if mentions:
            name = mentions[0].get("name", f"Entity_{idx}")
            etype = mentions[0].get("type", "UNKNOWN")
            lines.append(f"[{idx}] {name} ({etype})")
    return "\n".join(lines)


def build_name_to_idx(vertex_set: list) -> dict:
    mapping = {}
    for vidx, mentions in enumerate(vertex_set):
        for m in (mentions or []):
            name = m.get("name", "")
            if name:
                mapping[name] = vidx
    return mapping


def convert_to_swift_format(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    skipped = 0
    written = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for doc in data:
            # ── 重建文档正文 ──────────────────────────────────────────────
            sents = doc.get("sents", [])
            full_text = " ".join(
                " ".join(w for w in sent) if isinstance(sent, list) else str(sent)
                for sent in sents
            )

            vertex_set = doc.get("vertexSet", [])
            entity_list_str = build_entity_list(vertex_set)
            name_to_idx = build_name_to_idx(vertex_set)

            # ── 构建 Query（与推理时格式完全一致）─────────────────────────
            item = {
                "entity_list": entity_list_str,
            }
            messages = build_lora_extraction_prompt("docred", full_text, item)
            # messages = [system, user, ...]
            system_content = messages[0]["content"]
            # user content 可能是字符串或列表（纯文本时为字符串）
            user_content = messages[1]["content"]
            if isinstance(user_content, list):
                # 多模态格式，拼接所有 text 部分
                user_content = "\n".join(
                    c["text"] for c in user_content if c.get("type") == "text"
                )

            # ── 构建 Response（P-ID → 语义关系名 + 索引格式）─────────────
            labels = doc.get("labels", [])
            new_relations = []
            for label in labels:
                h = label.get("h")
                t = label.get("t")
                pid = label.get("r", "")
                rel_name = DOCRED_REL_MAP.get(pid, pid)  # P17 → "country"

                if h is None or t is None:
                    continue
                if h >= len(vertex_set) or t >= len(vertex_set):
                    continue

                new_relations.append({
                    "head_id": int(h),
                    "relation": rel_name,
                    "tail_id": int(t),
                })

            if not new_relations:
                skipped += 1
                continue

            response = json.dumps({"relations": new_relations}, ensure_ascii=False)

            record = {
                "system": system_content,
                "query": user_content,
                "response": response,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"✅ 转换完成：{written} 条写入 {output_path}，{skipped} 条因无标签跳过")


def main():
    parser = argparse.ArgumentParser(description="Convert Re-DocRED to Swift JSONL format")
    parser.add_argument("--input",  default="./data/train_revised.json",
                        help="Re-DocRED 训练集 JSON 路径")
    parser.add_argument("--output", default="./data/swift_train_redocred.jsonl",
                        help="输出 JSONL 路径")
    args = parser.parse_args()

    print(f"读取: {args.input}")
    convert_to_swift_format(args.input, args.output)


if __name__ == "__main__":
    main()
