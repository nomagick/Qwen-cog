# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union, List, Any, Generator

import types

import torch
import torch.utils.checkpoint
from transformers import PreTrainedTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList
import qwen_generation_utils

from transformers_stream_generator.main import (
    NewGenerationMixin,
    StreamGenerationConfig,
)
from transformers.generation import GenerationConfig


def chat_stream_raw(
    self,
    tokenizer: PreTrainedTokenizer,
    query: str,
    stop_words_ids: Optional[List[List[int]]] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs,
) -> Generator[str, Any, None]:
    generation_config = (
        generation_config if generation_config is not None else self.generation_config
    )
    if stop_words_ids is None:
        stop_words_ids = []

    max_window_size = kwargs.get("max_window_size", None)
    if max_window_size is None:
        max_window_size = generation_config.max_window_size
    raw_text, context_tokens = qwen_generation_utils.make_context(
        tokenizer,
        query,
        max_window_size=max_window_size,
        chat_format="raw",
    )

    stop_words_ids.extend(
        qwen_generation_utils.get_stop_words_ids(
            generation_config.chat_format, tokenizer
        )
    )
    if stop_words_ids is not None:
        stop_words_logits_processor = qwen_generation_utils.StopWordsLogitsProcessor(
            stop_words_ids=stop_words_ids,
            eos_token_id=generation_config.eos_token_id,
        )
        if logits_processor is None:
            logits_processor = LogitsProcessorList([stop_words_logits_processor])
        else:
            logits_processor.append(stop_words_logits_processor)
    input_ids = torch.tensor([context_tokens]).to(self.device)

    stream_config = StreamGenerationConfig(
        **generation_config.to_dict(), do_stream=True
    )

    outputs = []
    last_response = ""
    offset = len(last_response)
    for token in self.generate_stream(
        input_ids,
        return_dict_in_generate=False,
        generation_config=stream_config,
        logits_processor=logits_processor,
        seed=-1,
        **kwargs,
    ):
        outputs.append(token.item())
        last_response = tokenizer.decode(
            outputs, skip_special_tokens=True, errors="ignore"
        )
        if last_response and last_response[-1] != "�":
            chunk = last_response[offset:]
            offset = len(last_response)
            if chunk:
                yield chunk


def patch(model):
    model.generate_stream = types.MethodType(NewGenerationMixin.generate, model)
    model.sample_stream = types.MethodType(NewGenerationMixin.sample_stream, model)
    model.chat_stream_raw = types.MethodType(chat_stream_raw, model)

    return model


def _device_map(num_gpus, num_layers):
    per_gpu_layers = (num_layers + 2) / num_gpus

    device_map = {"transformer.wte": 0, "transformer.ln_f": 0, "lm_head": num_gpus - 1}

    used = 1
    gpu_target = 0
    for i in range(num_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0 if gpu_target < num_gpus - 1 else 1
        assert gpu_target < num_gpus
        device_map[f"transformer.h.{i}"] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(model_name_or_path):
    num_devices = torch.cuda.device_count()
    bf16_supported = torch.cuda.is_bf16_supported()

    if num_devices < 1:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            use_flash_attn=bf16_supported,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = patch(model).eval()
    elif num_devices == 1:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            use_flash_attn=bf16_supported,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = patch(model).cuda().eval()
    elif num_devices:
        device_map = _device_map(num_devices, 40 if "14B" in model_name_or_path else 32)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            use_flash_attn=bf16_supported,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = patch(model).eval()
        # model = dispatch_model(model, device_map=device_map)
    else:
        raise KeyError

    model.generation_config = GenerationConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    return model
