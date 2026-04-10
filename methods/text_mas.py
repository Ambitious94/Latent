from typing import Dict, List

from . import default_agents
from models import ModelWrapper
# from prompts import build_agent_messages, build_agent_messages_v6, build_agent_messages_v6_text_mas
from prompts import build_agent_messages_hierarchical_text_mas, build_agent_messages_sequential_text_mas
from utils import evaluate_prediction
import argparse
import json

class TextMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 256,
        max_new_tokens_judger: int = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_judger if max_new_tokens_judger is not None else max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.args = args
        self.method_name = "text_mas"
        self.task = args.task
        
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        for agent in self.agents:

            # Route to extraction prompts for document extraction datasets
            if self.args.task in ['docred', 'cord', 'funsd', 'finer', 'chemprot']:
                from prompts import build_extraction_prompts_text_mas_sequential, build_extraction_prompts_text_mas_hierarchical
                if self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_extraction_prompts_text_mas_hierarchical(
                            dataset=self.args.task,
                            role=agent.role,
                            question=item["question"],
                            context=contexts[idx],
                            item=item,
                            method=self.method_name,
                            args=self.args,
                        )
                        for idx, item in enumerate(items)
                    ]
                else:
                    batch_messages = [
                        build_extraction_prompts_text_mas_sequential(
                            dataset=self.args.task,
                            role=agent.role,
                            question=item["question"],
                            context=contexts[idx],
                            item=item,
                            method=self.method_name,
                            args=self.args,
                        )
                        for idx, item in enumerate(items)
                    ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_messages_hierarchical_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for idx, item in enumerate(items)
                ]
            else:
                batch_messages = [
                    build_agent_messages_sequential_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for idx, item in enumerate(items)
                ]

            prompts, input_ids, attention_mask, tokens_batch, _ = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            _max_tok = self.max_new_tokens_judger if agent.role == "judger" else self.max_new_tokens_each
            if self.model.use_vllm:
                generated_texts = self.model.vllm_generate_text_batch(
                    prompts,
                    max_new_tokens=_max_tok,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            else:
                generated_texts, _ = self.model.generate_text_batch(
                    input_ids,
                    attention_mask,
                    max_new_tokens=_max_tok,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

            agent_name_map_for_prompt_hierarchical = {
                "Planner": "Math Agent",
                "Critic": "Science Agent",
                "Refiner": "Code Agent",
                "Judger": "Task Summrizer",
                "planner": "Math Agent",
                "critic": "Science Agent",
                "refiner": "Code Agent",
                "judger": "Task Summrizer",
            }

            for idx in range(batch_size):

                text_out = generated_texts[idx].strip()

                if self.args.prompt == "hierarchical":
                    formatted_output = f"[{agent_name_map_for_prompt_hierarchical[agent.name]}]:\n{text_out}\n\n"
                else:
                    formatted_output = f"[{agent.name}]:\n{text_out}\n\n"

                if agent.role != "judger":

                    contexts[idx] = f"{contexts[idx]}{formatted_output}"
                    history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
                else:
                    final_texts[idx] = text_out
                mask = attention_mask[idx].bool()
                trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": tokens_batch[idx],
                        "output": text_out,
                    }
                )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]

            # ====== 新增：剥离 <think> 标签，精准提取 JSON ======
            start_idx = final_text.find('{')
            end_idx = final_text.rfind('}')
            if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
                cleaned_text = final_text[start_idx:end_idx+1]
            else:
                cleaned_text = final_text
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

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "context": history_contexts[idx],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "entities_meta": item.get("entities_meta", []),
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
