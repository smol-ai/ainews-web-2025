---
id: a81ae2a3-d6ce-49d5-897a-b8aad197bae8
title: 'DBRX: Best open model (just not most efficient)'
date: '2024-03-27T22:33:19.363427Z'
original_slug: ainews-dbrx-best-open-model-but-not-most-efficient
description: >-
  **Databricks Mosaic** has released a new open-source model called **DBRX**
  that outperforms **Grok**, **Mixtral**, and **Llama2** on evaluations while
  being about **2x more efficient** than Llama2 and Grok. The model was trained
  on **12 trillion tokens** using **3,000 H100 GPUs** over 2 months, with an
  estimated compute cost of **$10 million**. It uses OpenAI's **100k tiktoken
  tokenizer** and shows strong zero-shot code generation performance, even
  beating **GPT-4** on the Humaneval benchmark. DBRX also upstreamed work to
  **MegaBlocks** open source. Despite its scale and efficiency, DBRX's
  performance on MMLU is only slightly better than Mixtral, raising questions
  about its scaling efficiency. The focus of DBRX is on enabling users to train
  models efficiently, with MoE training being about **2x more FLOP-efficient**
  than dense models, achieving similar quality with nearly **4x less compute**
  than previous MPT models. This release is part of the ongoing competition for
  open-source AI leadership, including models like **Dolly**, **MPT**, and
  **Mistral**. *"If it activates 36B params, the model's perf should be
  equivalent to a 72B dense model or even 80B,"* says Qwen's tech lead.
companies:
  - databricks
  - hugging-face
  - mistral-ai
  - mosaicml
  - openai
models:
  - dbrx
  - grok
  - mixtral
  - llama-2
  - mpt-7b
  - gpt-4
topics:
  - mixture-of-experts
  - model-efficiency
  - tokenization
  - model-training
  - code-generation
  - model-architecture
  - open-source-models
  - benchmarking
  - fine-tuning
people: []
---

Archived full issue body is served from static HTML at /frozen-issues/24-03-27-ainews-dbrx-best-open-model-just-not-most-efficient.html.
