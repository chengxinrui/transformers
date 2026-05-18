AutoCompressor (Modular Integration in Transformers)
This project implements AutoCompressor (Chevalier et al., 2023) inside the Hugging Face transformers framework using the modular design pattern.

The implementation is based on Llama and supports:

✅ Context compression via summary vectors (soft prompts)
✅ Official checkpoint loading
✅ Compatibility with the latest (2025) Transformers version
✅ DynamicCache support
✅ Generation via model.generate()
1. Overview
AutoCompressor adapts a pretrained language model to compress long contexts into a small number of summary vectors. These vectors act as a learned soft prefix during generation.

The model operates in two modes:

1️⃣ Compression Mode
Python

summary_vectors = model(context_tokens, output_soft_prompt=True).soft_prompt
The long context is split into segments.
Learnable summary tokens are appended.
Hidden states of summary tokens become soft_prompt.
If summary_accumulation=True, summaries are accumulated across segments.
2️⃣ Generation Mode
Python

model.generate(prompt_tokens, soft_prompt=summary_vectors)
soft_prompt is prepended to the prompt embeddings.
Standard Llama self-attention attends to compressed context.
Generation proceeds using standard KV-cache logic.
2. Modular Architecture
The model is implemented using Transformers' modular mechanism.

Structure
text

AutoCompressorForCausalLM
    ├── AutoCompressorMixin
    └── LlamaForCausalLM
Key Files
File	Description
configuration_autocompressor.py	Extends LlamaConfig
modeling_autocompressor.py	Implements mixin + CausalLM head
__init__.py	Registers model in transformers
auto_mappings.py	Registers config
modeling_auto.py	Registers model class
3. Configuration
Python

class AutoCompressorConfig(LlamaConfig):
    summary_length: int = 50
    segment_length: int = 2048
    summary_accumulation: bool = True
    stop_grad: bool = True
These fields extend LlamaConfig and remain compatible with official checkpoints.

4. Compatibility with Transformers (2025)
Several internal changes in Transformers required adaptation:

✅ DynamicCache Support
New versions use DynamicCache instead of tuple-based KV cache.

Implemented compatibility:

Python

def get_past_key_values_len(self, past_key_values):
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].size(2)
✅ Updated Generation API
prepare_inputs_for_generation must use keyword arguments:

Python

super().prepare_inputs_for_generation(
    input_ids=input_ids,
    past_key_values=past_key_values,
    attention_mask=attention_mask,
    inputs_embeds=inputs_embeds,
    **kwargs,
)
This avoids conflicts with new parameters like next_sequence_length.

✅ No Manual Position ID Hacks
New Llama versions manage RoPE using cache_position.

We do not manually modify position_ids.

5. Usage Example
Python

from transformers.models.autocompressor.modeling_autocompressor import AutoCompressorForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoCompressorForCausalLM.from_pretrained(
    "princeton-nlp/AutoCompressor-Llama-2-7b-6k",
    torch_dtype=torch.bfloat16
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")

prompt = 'The first name of the current US president is "'
prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

context_tokens = tokenizer(long_context, return_tensors="pt").input_ids.cuda()

# Compression
summary_vectors = model(context_tokens, output_soft_prompt=True).soft_prompt

# Generation with compressed context
generation = model.generate(
    prompt_tokens,
    soft_prompt=summary_vectors,
    max_new_tokens=12,
)

print(tokenizer.decode(generation[0]))
6. Expected Output
With compressed context:

The first name of the current US president is "Joe" and the last name is "Biden".
Without context:


The first name of the current US president is "Donald" and the last name is "Trump".
7. Implementation Notes
Model inherits directly from LlamaForCausalLM.
Compression logic is injected via AutoCompressorMixin.
Official checkpoint weights are fully compatible.
Code follows snake_case naming:
softprompt → soft_prompt
output_softprompt → output_soft_prompt
8. Conclusion
This implementation:

✅ Fully modular
✅ Checkpoint-compatible
✅ Future-proof against Transformers API changes
✅ Maintains minimal intrusion into Llama codebase
It demonstrates how research models can be cleanly integrated into large frameworks using inheritance and mixins instead of rewriting base architectures.

