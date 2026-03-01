import os
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import base64
from io import BytesIO

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False

try:
    from qwen_vl_utils import process_vision_info
    _HAS_QWEN_VL_UTILS = True
except ImportError:
    _HAS_QWEN_VL_UTILS = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # Set padding_side to 'left' for decoder-only models (required for correct generation)
    tokenizer.padding_side = 'left'


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args
        
        # Detect if model is vision-language model
        self.is_vision_model = "vl" in model_name.lower() or getattr(args, "use_vision_model", False)
        self.processor = None  # For vision models

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            # Vision models not yet supported with vLLM backend
            if self.is_vision_model:
                raise NotImplementedError(
                    "Vision-language models (Qwen-VL) are not yet supported with vLLM backend. "
                    "Please use HuggingFace backend by not setting --use_vllm flag."
                )
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval() 
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        if self.is_vision_model:
            # Load vision-language model (Qwen-VL series: Qwen2-VL, Qwen3-VL, etc.)
            from transformers import AutoProcessor, AutoModelForVision2Seq
            print(f"[VL Model] Loading vision-language model: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = self.processor.tokenizer
            _ensure_pad_token(self.tokenizer)  # Ensure pad token exists
            with torch.no_grad():
                # AutoModelForVision2Seq automatically handles Qwen2-VL/Qwen3-VL
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                )
                
                # Load LoRA weights if specified
                if args and hasattr(args, 'lora_weights') and args.lora_weights:
                    print(f"[LoRA] Loading LoRA weights from: {args.lora_weights}")
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, args.lora_weights)
                    print("[LoRA] LoRA weights loaded successfully")
        else:
            # Load text-only model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            _ensure_pad_token(self.tokenizer)
            with torch.no_grad():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                )
                
                # Load LoRA weights if specified
                if args and hasattr(args, 'lora_weights') and args.lora_weights:
                    print(f"[LoRA] Loading LoRA weights from: {args.lora_weights}")
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, args.lora_weights)
                    print("[LoRA] LoRA weights loaded successfully")
                    
            if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> Union[str, List[Dict]]:
        """Render chat messages. For vision models, return structured messages; for text models, return string."""
        if self.is_vision_model:
            # For Qwen-VL, keep structured format with image/text content
            return messages  # Processor will handle this
        
        # Text-only model: render to string
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            # Handle multimodal content (extract text only for text models)
            if isinstance(content, list):
                text_parts = [item["text"] for item in content if item.get("type") == "text"]
                content = " ".join(text_parts)
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[Union[str, List[Dict]], torch.Tensor, torch.Tensor, List[str]]:
        """Prepare single chat input. For vision models, also process images."""
        if self.is_vision_model and self.processor:
            # Process vision-language input
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            # Extract image info from messages for processing
            image_inputs = process_vision_info(messages) if _HAS_QWEN_VL_UTILS else []
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
                padding=True,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # Move pixel_values to device if present
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
            active_ids = input_ids[0][attention_mask[0].bool()].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
            return text_prompt, input_ids, attention_mask, tokens
        else:
            # Text-only processing
            prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
            encoded = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            active_ids = input_ids[0][attention_mask[0].bool()].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
            return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[Union[str, Dict]], torch.Tensor, torch.Tensor, List[List[str]], Optional[Dict]]:
        """
        Prepare batch chat input. For vision models, process images in batch.
        
        Returns:
            prompts: List of text prompts
            input_ids: Tensor of token IDs
            attention_mask: Tensor of attention masks
            tokens_batch: List of token lists
            extra_inputs: Dict with additional inputs for vision models (pixel_values, image_grid_thw, etc.)
        """
        if self.is_vision_model and self.processor:
            # Process batch for vision-language model
            prompts = []
            all_images = []
            
            for messages in batch_messages:
                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt
                )
                prompts.append(text_prompt)
                
                # Extract images using qwen_vl_utils if available
                if _HAS_QWEN_VL_UTILS:
                    vision_info = process_vision_info(messages)
                    # process_vision_info returns (images_list, videos_list) tuple
                    images = vision_info[0] if isinstance(vision_info, tuple) else vision_info if isinstance(vision_info, list) else []
                else:
                    # Fallback: extract manually
                    images = []
                    for message in messages:
                        if isinstance(message, dict) and 'content' in message:
                            content = message['content']
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and 'image' in item and item['image'] is not None:
                                        images.append(item['image'])
                # Always append list (empty list if no images)
                all_images.append(images)
            
            # Only pass images if at least one sample has images
            has_images = any(imgs for imgs in all_images)
            
            # Process with processor
            inputs = self.processor(
                text=prompts,
                images=all_images if has_images else None,
                return_tensors="pt",
                padding=True,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Collect extra vision inputs
            extra_inputs = {}
            if "pixel_values" in inputs:
                extra_inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
            if "image_grid_thw" in inputs:
                extra_inputs["image_grid_thw"] = inputs["image_grid_thw"].to(self.device)
            
            tokens_batch: List[List[str]] = []
            for ids_row, mask_row in zip(input_ids, attention_mask):
                active_ids = ids_row[mask_row.bool()].tolist()
                tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
            
            return prompts, input_ids, attention_mask, tokens_batch, extra_inputs if extra_inputs else None
        else:
            # Text-only batch processing
            prompts: List[str] = []
            for messages in batch_messages:
                prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            tokens_batch: List[List[str]] = []
            for ids_row, mask_row in zip(input_ids, attention_mask):
                active_ids = ids_row[mask_row.bool()].tolist()
                tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
            return prompts, input_ids, attention_mask, tokens_batch, None

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Generate latent representations for multi-agent collaboration.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask
            latent_steps: Number of latent generation steps
            past_key_values: Previous KV-cache
            pixel_values: Image tensor for vision models (optional)
            image_grid_thw: Image grid info for Qwen2-VL (optional)
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        # Build forward kwargs (add vision inputs if present)
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if self.is_vision_model and pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                forward_kwargs["image_grid_thw"] = image_grid_thw

        outputs = self.model(**forward_kwargs)
        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past
    
    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings

