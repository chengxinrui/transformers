# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modular AutoCompressor file.

This file implements AutoCompressor by inheriting from Llama and adding
summary-vector (soft-prompt) compression.  Running
``python utils/modular_model_converter.py autocompressor`` flattens it into
``modeling_autocompressor.py`` and ``configuration_autocompressor.py``.

Reference: Chevalier et al., *Adapting Language Models to Compress Contexts* (2023).
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...modeling_outputs import CausalLMOutputWithPast
from ...utils import auto_docstring, logging
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@auto_docstring(checkpoint="princeton-nlp/AutoCompressor-Llama-2-7b-6k")
@strict
class AutoCompressorConfig(LlamaConfig):
    """Configuration class for AutoCompressor.

    Inherits everything from LlamaConfig and adds hyper-parameters for
    summary-vector compression.
    """

    model_type = "autocompressor"

    # New AutoCompressor-specific fields
    summary_length: int = 50
    segment_length: int = 2048
    summary_accumulation: bool = True
    stop_grad: bool = True

    def __init__(
        self,
        summary_length: int = 50,
        segment_length: int = 2048,
        summary_accumulation: bool = True,
        stop_grad: bool = True,
        **kwargs,
    ):
        self.summary_length = summary_length
        self.segment_length = segment_length
        self.summary_accumulation = summary_accumulation
        self.stop_grad = stop_grad
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# PreTrained base
# ---------------------------------------------------------------------------
class AutoCompressorPreTrainedModel(LlamaPreTrainedModel):
    """Base class for AutoCompressor models."""

    config_class = AutoCompressorConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]


# ---------------------------------------------------------------------------
# Backbone model (identical to LlamaModel except for the config class)
# ---------------------------------------------------------------------------
class AutoCompressorModel(LlamaModel):
    """The bare AutoCompressor model, outputting raw hidden states."""

    config_class = AutoCompressorConfig


# ---------------------------------------------------------------------------
# Output dataclass (used by the CausalLM head)
# ---------------------------------------------------------------------------
class CausalACOutputWithPast(CausalLMOutputWithPast):
    """CausalLMOutputWithPast extended with the ``soft_prompt`` field."""

    soft_prompt: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Causal LM head – the only class that really differs from Llama
# ---------------------------------------------------------------------------
class AutoCompressorForCausalLM(LlamaForCausalLM):
    r"""
    AutoCompressor extends Llama with context compression via *summary vectors*.

    During compression the model processes the input in segments.  Each segment
    is appended with learnable ``summary`` tokens; the hidden states of those
    tokens become the ``soft_prompt`` that summarises the segment.  When
    ``config.summary_accumulation`` is ``True`` (default), the soft prompts of
    all earlier segments are prepended to the next segment, yielding a single
    accumulated summary of the whole context.

    During generation, a previously computed ``soft_prompt`` can be passed as
    a prefix so the model attends to the compressed context.

    Example::

        >>> from transformers import AutoTokenizer
        >>> from transformers.models.autocompressor.modeling_autocompressor import AutoCompressorForCausalLM
        >>> model = AutoCompressorForCausalLM.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
        >>> tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
        >>> context = tokenizer("Long document ...", return_tensors="pt").input_ids
        >>> out = model(context, output_soft_prompt=True)
        >>> summary = out.soft_prompt  # compressed vectors
        >>> prompt = tokenizer("Question:", return_tensors="pt").input_ids
        >>> generated = model.generate(prompt, soft_prompt=summary, max_new_tokens=20)
    """

    config_class = AutoCompressorConfig

    def __init__(self, config):
        # Initialise the Llama graph first.
        super().__init__(config)
        # Add the learnable summary-token embedding matrix.
        # Use getattr with sensible defaults so plain Llama configs still load.
        summary_length = getattr(config, "summary_length", 50)
        if summary_length > 0:
            self.embed_summary = nn.Embedding(
                summary_length,
                self.get_input_embeddings().embedding_dim,
            )
            eos_id = getattr(config, "eos_token_id", 2)
            with torch.no_grad():
                self.embed_summary.weight.copy_(
                    self.get_input_embeddings().weight[eos_id].unsqueeze(0)
                )
        else:
            self.embed_summary = None
        self.post_init()

    # ------------------------------------------------------------------
    # Internal helper: run one segment.
    # ------------------------------------------------------------------
    def _forward_segment(
        self,
        segment_embeds: torch.FloatTensor,
        segment_attention_mask: torch.LongTensor,
        soft_prompt: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        """Run a single segment (+ optional prefix soft_prompt + summary tokens).

        Returns ``(segment_logits, new_soft_prompt, past_key_values)``.
        """
        bsz = segment_embeds.size(0)
        summary_length = getattr(self.config, "summary_length", 50)

        # Expand learnable summary embeddings to batch size.
        summary_token_embeds = self.embed_summary.weight.unsqueeze(0).expand(
            bsz, -1, -1
        )

        if soft_prompt is not None and soft_prompt.size(1) > 0:
            combined_embeds = torch.cat(
                [soft_prompt, segment_embeds, summary_token_embeds], dim=1
            )
            prefix_len = soft_prompt.size(1)
            if segment_attention_mask is not None:
                device, dtype = segment_attention_mask.device, segment_attention_mask.dtype
                prefix_mask = torch.ones(bsz, prefix_len, dtype=dtype, device=device)
                suffix_mask = torch.ones(bsz, summary_length, dtype=dtype, device=device)
                combined_mask = torch.cat(
                    [prefix_mask, segment_attention_mask, suffix_mask], dim=1
                )
            else:
                combined_mask = None
        else:
            combined_embeds = torch.cat([segment_embeds, summary_token_embeds], dim=1)
            prefix_len = 0
            if segment_attention_mask is not None:
                device, dtype = segment_attention_mask.device, segment_attention_mask.dtype
                suffix_mask = torch.ones(bsz, summary_length, dtype=dtype, device=device)
                combined_mask = torch.cat([segment_attention_mask, suffix_mask], dim=1)
            else:
                combined_mask = None

        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        new_soft_prompt = hidden_states[:, -summary_length:, :]
        text_hidden = hidden_states[:, prefix_len : hidden_states.size(1) - summary_length, :]
        logits = self.lm_head(text_hidden)

        return logits, new_soft_prompt, outputs.past_key_values

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        soft_prompt: Optional[torch.FloatTensor] = None,
        output_soft_prompt: Optional[bool] = None,
        segment_lengths: Optional[Union[int, List[int]]] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_soft_prompt = output_soft_prompt if output_soft_prompt is not None else False

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        bsz, seq_len, _ = inputs_embeds.shape

        # --------------------- Compression mode ---------------------
        if output_soft_prompt:
            default_seg = getattr(self.config, "segment_length", 2048)
            seg_len = (
                segment_lengths
                if isinstance(segment_lengths, int)
                else default_seg
            )

            accumulated_soft_prompt = soft_prompt
            all_logits = []
            start = 0
            while start < seq_len:
                end = min(start + seg_len, seq_len)
                seg_embeds = inputs_embeds[:, start:end, :]
                if attention_mask is not None:
                    seg_mask = attention_mask[:, start:end]
                else:
                    seg_mask = torch.ones(
                        bsz, end - start, dtype=torch.long, device=inputs_embeds.device
                    )

                logits, new_soft_prompt, _ = self._forward_segment(
                    seg_embeds,
                    seg_mask,
                    soft_prompt=accumulated_soft_prompt,
                    use_cache=False,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                if getattr(self.config, "summary_accumulation", True):
                    if accumulated_soft_prompt is not None:
                        accumulated_soft_prompt = torch.cat(
                            [accumulated_soft_prompt, new_soft_prompt], dim=1
                        )
                    else:
                        accumulated_soft_prompt = new_soft_prompt
                else:
                    accumulated_soft_prompt = new_soft_prompt

                all_logits.append(logits)
                start = end

            logits = all_logits[-1] if all_logits else None

            loss = None
            if labels is not None and logits is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., -logits.size(1) + 1 :].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                )

            if not return_dict:
                return (logits, None, None, accumulated_soft_prompt)

            return CausalACOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                soft_prompt=accumulated_soft_prompt,
            )

        # --------------------- Normal / generation mode ---------------------
        if soft_prompt is not None and (
            past_key_values is None
            or (isinstance(past_key_values, DynamicCache) and past_key_values.get_seq_length() == 0)
        ):
            inputs_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    bsz,
                    soft_prompt.size(1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            if position_ids is not None:
                position_ids = position_ids + soft_prompt.size(1)

        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            return (logits,) + outputs[1:]

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        model_inputs["soft_prompt"] = kwargs.get("soft_prompt", None)
        model_inputs["segment_lengths"] = kwargs.get("segment_lengths", None)
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.model._reorder_cache(past_key_values, beam_idx)


# ---------------------------------------------------------------------------
# Module-level exports (required by the modular converter)
# ---------------------------------------------------------------------------
logger = logging.get_logger(__name__)

__all__ = [
    "AutoCompressorConfig",
    "AutoCompressorPreTrainedModel",
    "AutoCompressorModel",
    "AutoCompressorForCausalLM",
    "CausalACOutputWithPast",
]
