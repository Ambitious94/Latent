#!/usr/bin/env python
"""
将LatentMAS输出转换为DocRED官方评测格式
"""
import json
import sys

def convert_to_official_format(predictions_file, original_data_file, output_file):
    """
    转换格式: 
    From: {"prediction": '{"relations": [{"head": "name", "relation": "P17", "tail": "name"}]}'}
    To:   [{"title": "...", "h_idx": 0, "t_idx": 1, "r": "P17"}]
    """
    # 加载预测结果
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = data.get("predictions", [])
    
    # 加载原始数据获取title和vertexSet
    with open(original_data_file, 'r', encoding='utf-8') as f:
        original_docs = json.load(f)
    
    official_results = []
    
    for i, pred in enumerate(predictions):
        if i >= len(original_docs):
            break
        
        doc = original_docs[i]
        title = doc.get("title", "")
        vertex_set = doc.get("vertexSet", [])
        
        # 解析预测的关系
        try:
            pred_str = pred.get("prediction", "{}")
            pred_data = json.loads(pred_str)
            relations = pred_data.get("relations", [])
        except:
            continue
        
        # 构建实体名称到索引的映射
        entity_to_idx = {}
        for idx, entities in enumerate(vertex_set):
            for entity in entities:
                name = entity.get("name", "").strip()
                if name:
                    entity_to_idx[name] = idx
        
        # 转换每个关系
        for rel in relations:
            head_name = rel.get("head", "").strip()
            tail_name = rel.get("tail", "").strip()
            relation = rel.get("relation", "")
            
            # 查找索引
            h_idx = entity_to_idx.get(head_name)
            t_idx = entity_to_idx.get(tail_name)
            
            if h_idx is not None and t_idx is not None and relation:
                official_results.append({
                    "title": title,
                    "h_idx": h_idx,
                    "t_idx": t_idx,
                    "r": relation
                })
    
    # 保存官方格式
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(official_results, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 转换完成:")
    print(f"  输入: {predictions_file}")
    print(f"  原始数据: {original_data_file}")
    print(f"  输出: {output_file}")
    print(f"  预测关系数: {len(official_results)}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python convert_to_official.py <predictions.json> <original_dev.json> <output.json>")
        print("\n示例:")
        print('  python convert_to_official.py results/docred_dev.json "e:/Edge Download/dev.json" results/result.json')
        sys.exit(1)
    
    convert_to_official_format(sys.argv[1], sys.argv[2], sys.argv[3])
