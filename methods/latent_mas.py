from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import evaluate_prediction
import torch
import argparse

try:
    from vllm import SamplingParams
    _HAS_VLLM = True
except ImportError:
    SamplingParams = None
    _HAS_VLLM = False

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        if _HAS_VLLM and SamplingParams is not None:
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=args.max_new_tokens,
                repetition_penalty=1.1,
            )
        else:
            self.sampling_params = None
        self.task = args.task

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # 检查是否使用LoRA模型
        use_lora = hasattr(self.args, 'lora_weights') and self.args.lora_weights
        
        # 注意: LoRA模型同样走完整的多Agent流程(Planner→Critic→Refiner→Judger)
        # 之前的直接推理旁路已移除,因为多Agent reasoning能显著提升LoRA模型的抽取质量

        for agent in self.agents:

            # Route to extraction prompts for document extraction datasets
            if self.args.task in ['docred', 'cord', 'funsd', 'finer', 'chemprot']:
                # 原始模型使用详细prompts
                from prompts import build_extraction_prompts_sequential, build_extraction_prompts_hierarchical
                if self.args.prompt == "sequential":
                    batch_messages = [
                        build_extraction_prompts_sequential(dataset=self.args.task, role=agent.role, question=item["question"], item=item, method=self.method_name, args=self.args)
                        for item in items
                    ]
                elif self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_extraction_prompts_hierarchical(dataset=self.args.task, role=agent.role, question=item["question"], item=item, method=self.method_name, args=self.args)
                        for item in items
                    ]
            else:
                # Original prompts for existing tasks
                if self.args.prompt == "sequential":
                    batch_messages = [
                        build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]
                elif self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]

            prompts, input_ids, attention_mask, tokens_batch, extra_inputs = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                    # For vision models, append <think> token to input_ids instead of re-tokenizing
                    if self.model.is_vision_model and extra_inputs:
                        think_token_id = self.model.tokenizer.encode("<think>", add_special_tokens=False)[0]
                        # Append think token to each sequence
                        think_ids = torch.full((input_ids.shape[0], 1), think_token_id, dtype=input_ids.dtype, device=input_ids.device)
                        wrapped_ids = torch.cat([input_ids, think_ids], dim=1)
                        wrapped_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
                    else:
                        # Text-only model: re-tokenize with <think>
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                        wrapped_encoded = self.model.tokenizer(
                            wrapped_prompts,
                            return_tensors="pt",
                            padding=True,
                            add_special_tokens=False,
                        )
                        wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                        wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                else:
                    wrapped_ids = input_ids
                    wrapped_mask = attention_mask

                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                # Prepare vision inputs if available (only for first agent to encode images)
                vision_kwargs = {}
                if extra_inputs and agent.role == "planner":
                    # Only pass pixel_values for the first agent (planner)
                    # Subsequent agents will use the KV-cache which already contains vision information
                    vision_kwargs = {
                        "pixel_values": extra_inputs.get("pixel_values"),
                        "image_grid_thw": extra_inputs.get("image_grid_thw"),
                    }
                
                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                    **vision_kwargs,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    # For vision models, use text representation of input_ids instead of wrapped_prompts
                    input_repr = prompts[idx] if not (self.model.is_vision_model and extra_inputs) else f"[Vision input with {len(trimmed_ids)} tokens]"
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": input_repr,
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:

                past_for_decoding = past_kv if self.latent_steps > 0 else None

                if self.args.think:
                    # For vision models, append <think> token to input_ids
                    if self.model.is_vision_model and extra_inputs:
                        think_token_id = self.model.tokenizer.encode("<think>", add_special_tokens=False)[0]
                        think_ids = torch.full((input_ids.shape[0], 1), think_token_id, dtype=input_ids.dtype, device=input_ids.device)
                        judger_ids = torch.cat([input_ids, think_ids], dim=1)
                        judger_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
                        judger_prompts = prompts  # Use original prompts for logging
                    else:
                        # Text-only model: re-tokenize with <think>
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                        judger_encoded = self.model.tokenizer(
                            judger_prompts,
                            return_tensors="pt",
                            padding=True,
                            add_special_tokens=False,
                        )
                        judger_ids = judger_encoded["input_ids"].to(self.model.device)
                        judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                else:
                    judger_ids = input_ids
                    judger_mask = attention_mask
                    judger_prompts = prompts  # Use original prompts for logging
                
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                    repetition_penalty=1.1,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
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

            # ====== 终极修复：坐标对齐吸附 (Coordinate Snapping) ======
            if self.task == "finer":
                import json
                try:
                    data = json.loads(cleaned_text)
                    doc_text = item.get("question", "")
                    for ent in data.get("entities", []):
                        ent_text = ent.get("text", "")
                        pred_start = ent.get("start", 0)

                        # 如果模型成功提取了实体文本，且文本确实在原文中
                        if ent_text and ent_text in doc_text:
                            starts = []
                            idx_search = doc_text.find(ent_text)
                            while idx_search != -1:
                                starts.append(idx_search)
                                idx_search = doc_text.find(ent_text, idx_search + 1)

                            if starts:
                                # 寻找距离模型预测坐标（带偏移的坐标）最近的真实坐标
                                real_start = min(starts, key=lambda x: abs(x - pred_start))
                                ent["start"] = real_start
                                ent["end"] = real_start + len(ent_text)

                    cleaned_text = json.dumps(data)
                except Exception:
                    pass
            # =================================================

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
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results
    
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        for agent in self.agents:
            
            # Route to extraction prompts for document extraction datasets
            if self.args.task in ['docred', 'cord', 'funsd', 'finer', 'chemprot']:
                from prompts import build_extraction_prompts_sequential, build_extraction_prompts_hierarchical
                if self.args.prompt == "sequential":
                    batch_messages = [
                        build_extraction_prompts_sequential(dataset=self.args.task, role=agent.role, question=item["question"], item=item, method=self.method_name, args=self.args)
                        for item in items
                    ]
                elif self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_extraction_prompts_hierarchical(dataset=self.args.task, role=agent.role, question=item["question"], item=item, method=self.method_name, args=self.args)
                        for item in items
                    ]
            else:
                # Original prompts for existing tasks
                if self.args.prompt == "sequential":
                    batch_messages = [
                        build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]
                elif self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]
                
            prompts, input_ids, attention_mask, tokens_batch, extra_inputs = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                
                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ) 
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding — keep the batch dimension intact.
                # squeeze(0) would collapse [1, L, H] → [L, H] when batch_size=1,
                # causing curr_prompt_emb[i] to index a single token vector instead
                # of the i-th sample sequence, corrupting all downstream logic.
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).to(self.vllm_device)
                
                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position    
                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    # Get the text up to and including "<|im_start|>user\n"
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                    
                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency
                    
                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                # Pass variable-length embeddings directly to vLLM;
                # vLLM natively handles variable-length inputs without padding.
                # Zero-padding would pollute short sequences with meaningless vectors.
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb_list
                ]
                
                
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]
                    
                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
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

            # ====== 终极修复：坐标对齐吸附 (Coordinate Snapping) ======
            if self.task == "finer":
                import json
                try:
                    data = json.loads(cleaned_text)
                    doc_text = item.get("question", "")
                    for ent in data.get("entities", []):
                        ent_text = ent.get("text", "")
                        pred_start = ent.get("start", 0)

                        # 如果模型成功提取了实体文本，且文本确实在原文中
                        if ent_text and ent_text in doc_text:
                            starts = []
                            idx_search = doc_text.find(ent_text)
                            while idx_search != -1:
                                starts.append(idx_search)
                                idx_search = doc_text.find(ent_text, idx_search + 1)

                            if starts:
                                # 寻找距离模型预测坐标（带偏移的坐标）最近的真实坐标
                                real_start = min(starts, key=lambda x: abs(x - pred_start))
                                ent["start"] = real_start
                                ent["end"] = real_start + len(ent_text)

                    cleaned_text = json.dumps(data)
                except Exception:
                    pass
            # =================================================

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
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
