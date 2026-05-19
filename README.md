# AutoCompressor Modular Reimplementation in Transformers

## 1. 项目简介

本项目基于 HuggingFace `transformers` 仓库，对 AutoCompressor 进行 modular 方式重构。  
目标是参考 `modular_transformers` 的实现范式，将 AutoCompressor 以模块化方式集成到 Transformers 中，并能够通过官方示例完成正确推理。

AutoCompressor 的核心思想来自论文：

- Chevalier et al., **Adapting Language Models to Compress Contexts** (2023)

其主要能力是将长上下文压缩为较短的 **summary vectors（soft prompt）**，并在后续生成时作为压缩上下文使用。

---

## 2. 任务目标

本次工作的目标包括：

1. Fork HuggingFace `transformers` 仓库
2. 参考 `modular_transformers` 文档，以 modular 方式实现 `AutoCompressor`
3. 完成模型配置、模块导出和相关注册
4. 运行示例代码并得到正确输出
5. 接口命名风格与 Transformers 保持一致，例如：
   - `softprompt` → `soft_prompt`

---

## 3. 实现内容

本次实现主要围绕以下几个方面展开：

### 3.1 新增 AutoCompressor 配置类
在 `configuration_autocompressor.py` 中实现 `AutoCompressorConfig`，基于 `LlamaConfig` 扩展了以下字段：

- `summary_length`
- `segment_length`
- `summary_accumulation`
- `stop_grad`

该配置用于控制上下文压缩长度、分段长度以及 summary vector 的累积方式。

---

### 3.2 新增 modular 实现
在 `modular_autocompressor.py` 中完成 AutoCompressor 的 modular 版本实现，主要包括：

- `AutoCompressorConfig`
- `AutoCompressorPreTrainedModel`
- `AutoCompressorModel`
- `AutoCompressorForCausalLM`
- `CausalACOutputWithPast`

其中 `AutoCompressorForCausalLM` 基于 `LlamaForCausalLM` 扩展，实现了：

- 分段处理长上下文
- 通过 learnable summary tokens 生成 summary vectors
- 在生成阶段支持 `soft_prompt`
- 自定义 `prepare_inputs_for_generation`

---

### 3.3 生成扁平化实现
基于 modular 文件，生成：

- `configuration_autocompressor.py`
- `modeling_autocompressor.py`

从而与 Transformers 的现有代码组织方式保持一致。

---

### 3.4 模块导出与注册
在 `__init__.py` 中补充了 AutoCompressor 相关导出与 lazy import。  
此外，还补充了相关注册逻辑，使其能够通过 Transformers 的统一接口被正确加载。

---

## 4. 主要改动文件

| 文件 | 说明 |
|---|---|
| `src/transformers/models/autocompressor/modular_autocompressor.py` | AutoCompressor 的 modular 核心实现 |
| `src/transformers/models/autocompressor/configuration_autocompressor.py` | AutoCompressor 配置类 |
| `src/transformers/models/autocompressor/modeling_autocompressor.py` | 扁平化后的模型实现 |
| `src/transformers/models/autocompressor/__init__.py` | 模块导出与 lazy import |
| 其他注册文件 | 补充 AutoModel / 配置映射等注册 |

---

## 5. 核心实现说明

### 5.1 Summary Vector / Soft Prompt
AutoCompressor 通过在输入片段后追加 summary tokens，利用这些 token 的最终 hidden states 作为压缩后的上下文表示，即 `soft_prompt`。

### 5.2 分段压缩
对于长输入，模型将输入按 segment 切分，并逐段处理。  
每一段会生成新的 summary vectors；若启用 `summary_accumulation`，则前序段落的 summary vectors 会累积到后续段落中。

### 5.3 生成阶段使用压缩上下文
在 generation 时，`soft_prompt` 会作为前缀嵌入拼接到输入前，从而让模型在不显式输入完整长上下文的情况下利用压缩信息完成生成。

---

## 6. 运行方式

### 6.1 加载模型与 tokenizer

```python
from transformers.models.autocompressor.modeling_autocompressor import AutoCompressorForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoCompressorForCausalLM.from_pretrained(
    "princeton-nlp/AutoCompressor-Llama-2-7b-6k",
    torch_dtype=torch.bfloat16
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
