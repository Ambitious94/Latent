"""
LoRA微调模型专用Prompts

微调后的模型已经学会了：
1. 各任务的输出格式（FUNSD/DocRED/CORD/FinER）
2. 实体关系的识别模式
3. 领域特定的理解能力

因此可以使用更简洁的prompts，减少冗余说明
"""

import json


def build_lora_extraction_prompts_sequential(dataset: str, role: str, question: str, item: dict, method=None, args=None):
    """
    LoRA模型的Sequential架构prompts（简化版）
    
    相比原版prompts：
    - 移除详细的格式说明（模型已学会）
    - 保留核心任务描述
    - 简化示例和约束
    """
    system_message = "You are a document extraction specialist. Output valid JSON only."
    
    # 任务简要说明（模型已了解细节）
    if dataset == "docred":
        task = "Extract document relations"
    elif dataset == "cord":
        task = "Extract receipt information"
    elif dataset == "funsd":
        task = "Extract form fields and relations"
    elif dataset == "finer":
        task = "Extract financial entities"
    else:
        task = "Extract information"
    
    if role == "planner":
        user_prompt = f"""Task: {task}

Document:
{question}

Extract all relevant information:
"""
    
    elif role == "critic":
        user_prompt = f"""Task: {task}

Document:
{question}

Review and refine the extraction:
"""
    
    elif role == "refiner":
        user_prompt = f"""Task: {task}

Document:
{question}

Finalize the extraction:
"""
    
    elif role == "judger":
        # Judger负责输出最终JSON
        if dataset == "docred":
            entity_list = item.get("entity_list", "")
            user_prompt = f"""Task: {task}

Entities: {entity_list}

Document:
{question}

Output JSON with relations:
"""
        else:
            user_prompt = f"""Task: {task}

Document:
{question}

Output complete JSON:
"""
    
    # 检查是否有图片
    if "image" in item and item["image"] is not None:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": user_prompt}
            ]},
        ]
    else:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]


def build_lora_extraction_prompts_hierarchical(dataset: str, role: str, question: str, item: dict, method=None, args=None):
    """
    LoRA模型的Hierarchical架构prompts（简化版）
    """
    system_message = "You are a document extraction specialist. Output valid JSON only."
    
    partition_info = item.get("partition_info", "")
    
    # 任务简要说明
    if dataset == "docred":
        task = "Extract document relations"
    elif dataset == "cord":
        task = "Extract receipt information"
    elif dataset == "funsd":
        task = "Extract form fields and relations"
    elif dataset == "finer":
        task = "Extract financial entities"
    else:
        task = "Extract information"
    
    if role == "planner":
        user_prompt = f"""Task: {task}
Partition: {partition_info}

Document:
{question}

Extract from this partition:
"""
    
    elif role == "critic":
        user_prompt = f"""Task: {task}
Partition: {partition_info}

Document:
{question}

Extract from this partition:
"""
    
    elif role == "refiner":
        user_prompt = f"""Task: {task}
Partition: {partition_info}

Document:
{question}

Extract from this partition:
"""
    
    elif role == "judger":
        # Judger整合所有分区信息
        if dataset == "docred":
            entity_list = item.get("entity_list", "")
            user_prompt = f"""Task: {task}

Entities: {entity_list}

Document:
{question}

Combine all partitions and output JSON:
"""
        else:
            user_prompt = f"""Task: {task}

Document:
{question}

Combine all partitions and output JSON:
"""
    
    # 检查是否有图片
    if "image" in item and item["image"] is not None:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": user_prompt}
            ]},
        ]
    else:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]


def should_use_lora_prompts(args) -> bool:
    """
    判断是否应该使用LoRA专用prompts
    
    条件：
    1. 指定了lora_weights参数
    2. 任务是文档抽取类（docred/cord/funsd/finer）
    """
    if not hasattr(args, 'lora_weights') or not args.lora_weights:
        return False
    
    if args.task not in ['docred', 'cord', 'funsd', 'finer']:
        return False
    
    return True


# 导出接口保持一致
def build_extraction_prompts_sequential_lora(dataset: str, role: str, question: str, item: dict, method=None, args=None):
    """兼容原有接口的包装函数"""
    return build_lora_extraction_prompts_sequential(dataset, role, question, item, method, args)


def build_extraction_prompts_hierarchical_lora(dataset: str, role: str, question: str, item: dict, method=None, args=None):
    """兼容原有接口的包装函数"""
    return build_lora_extraction_prompts_hierarchical(dataset, role, question, item, method, args)
