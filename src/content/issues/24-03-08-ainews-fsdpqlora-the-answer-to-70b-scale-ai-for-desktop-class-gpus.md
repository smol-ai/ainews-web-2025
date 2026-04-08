---
id: 0a752f76-32dd-43fc-bc5e-127df72c56cf
title: 'FSDP+QLoRA: the Answer to 70b-scale AI for desktop class GPUs'
date: '2024-03-08T23:21:13.565774Z'
original_slug: ainews-fsdpqlora-the-answer-to-70b-scale-ai-for
description: >-
  **Jeremy Howard** and collaborators released a new tool combining **FSDP**,
  **QLoRA**, and **HQQ** to enable training **70b-parameter** models on
  affordable consumer GPUs like **RTX 4090s** with only **24GB RAM**, overcoming
  traditional memory constraints that required expensive data center GPUs
  costing over $150k. The approach shards quantized models across multiple GPUs
  and uses techniques like gradient checkpointing and CPU offloading to achieve
  efficient training on desktop-class hardware. The blogpost details challenges
  and solutions integrating these methods, highlighting a significant cost
  reduction from $150k to under $2.5k for training large language models.
  Additionally, Twitter recaps mention **Inflection AI**'s **Inflection-2.5**
  model rivaling **GPT-4** in benchmarks with less compute, and **Grok**
  improving speed by 3x. **Yann LeCun** discusses multi-step reasoning training
  for LLMs.
companies:
  - answer.ai
  - hugging-face
  - meta-ai-fair
  - nvidia
  - inflectionai
models:
  - qlora
  - fsdp
  - inflection-2.5
  - gpt-4
topics:
  - model-training
  - quantization
  - memory-optimization
  - gradient-checkpointing
  - cpu-offloading
  - fine-tuning
  - model-sharding
  - reinforcement-learning
  - chain-of-thought
  - benchmarking
people:
  - jeremy_howard
  - tim_dettmers
  - yann_lecun
---

Archived full issue body is served from static HTML at /frozen-issues/24-03-08-ainews-fsdpqlora-the-answer-to-70b-scale-ai-for-desktop-class-gpus.html.
