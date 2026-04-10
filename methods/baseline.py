from typing import Dict, List

from models import ModelWrapper
from prompts import build_agent_messages_single_agent, build_lora_extraction_prompt
from utils import evaluate_prediction


class BaselineMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        use_vllm: bool = False,
        args=None,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.use_vllm = use_vllm
        self.method_name = "baseline"
        self.args = args
        self.task = args.task

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        batch_messages = []
        for item in items:
            # 针对信息抽取任务，使用与微调绝对一致的 LoRA 专属提示词
            if self.task in ['docred', 're-docred', 're_docred', 'cord', 'funsd', 'finer', 'chemprot']:
                # 把 re-docred 映射为 docred 的模板
                prompt_task = "docred" if "docred" in self.task else self.task
                msg = build_lora_extraction_prompt(dataset=prompt_task, question=item["question"], item=item, args=self.args)
            else:
                # 兼容其他普通问答任务
                msg = build_agent_messages_single_agent(question=item["question"], args=self.args)
            batch_messages.append(msg)
        prompts, input_ids, attention_mask, tokens_batch, _ = self.model.prepare_chat_batch(
            batch_messages, add_generation_prompt=True
        )
        
        if self.use_vllm:
            generated_batch = self.model.vllm_generate_text_batch(
                prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            generated_batch, _ = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        results: List[Dict] = []
        
        for idx, item in enumerate(items):
            generated_text = generated_batch[idx]

            # ====== 新增：剥离 <think> 标签，精准提取 JSON ======
            start_idx = generated_text.find('{')
            end_idx = generated_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
                cleaned_text = generated_text[start_idx:end_idx+1]
            else:
                cleaned_text = generated_text
            # =================================================

            # ====== ChemProt 去重：模型复读时同一三元组会重复出现 ======
            if self.task == "chemprot":
                import json as _json
                try:
                    _data = _json.loads(cleaned_text)
                    _rels = _data.get("relations", [])
                    if isinstance(_rels, list):
                        _seen = set()
                        _deduped = []
                        for _r in _rels:
                            if isinstance(_r, dict):
                                _key = (
                                    str(_r.get("head", "")).strip().lower(),
                                    str(_r.get("relation", "")).strip().lower(),
                                    str(_r.get("tail", "")).strip().lower(),
                                )
                                if _key not in _seen:
                                    _seen.add(_key)
                                    _deduped.append(_r)
                        _data["relations"] = _deduped
                        cleaned_text = _json.dumps(_data, ensure_ascii=False)
                except Exception:
                    pass
            # ======================================================

            # 注意这里传入的是 cleaned_text
            eval_result = evaluate_prediction(self.task, cleaned_text, item, idx)
            pred = eval_result["prediction"]
            gold = eval_result["gold"]
            ok = eval_result["correct"]
            
            mask = attention_mask[idx].bool()
            trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
            agent_trace = {
                "name": "SingleAgent",
                "role": "singleagent",
                "input": prompts[idx],
                "input_ids": trimmed_ids,
                "input_tokens": tokens_batch[idx],
                "output": generated_text,
            }
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": generated_text,
                    "agents": [agent_trace],
                    "correct": ok,
                    "entities_meta": item.get("entities_meta", []),
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
