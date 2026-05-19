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
"""AutoCompressor model configuration"""

from ...utils import logging
from ..llama.configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)


class AutoCompressorConfig(LlamaConfig):
    r"""
    Configuration for AutoCompressor, built on top of Llama.
    AutoCompressor adapts pre-trained LMs to compress long contexts into compact summary vectors (soft prompts).

    Reference: Chevalier et al., "Adapting Language Models to Compress Contexts", 2023.
    """

    model_type = "autocompressor"

    def __init__(
        self,
        summary_length: int = 50,
        segment_length: int = 2048,
        summary_accumulation: bool = True,
        stop_grad: bool = True,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.summary_length = summary_length
        self.segment_length = segment_length
        self.summary_accumulation = summary_accumulation
        self.stop_grad = stop_grad

        super().__init__(eos_token_id=eos_token_id, **kwargs)
