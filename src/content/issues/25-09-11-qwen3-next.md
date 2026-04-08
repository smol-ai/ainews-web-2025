---
id: MjAyNS0w
title: 'Qwen3-Next-80B-A3B-Base: Towards Ultimate Training & Inference Efficiency'
date: '2025-09-11T05:44:39.731046Z'
description: >-
  **MoE (Mixture of Experts) models** have become essential in frontier AI
  models, with **Qwen3-Next** pushing sparsity further by activating only **3.7%
  of parameters** (3B out of 80B) using a hybrid architecture combining **Gated
  DeltaNet** and **Gated Attention**. This new design includes **512 total
  experts** (10 routed + 1 shared), **Zero-Centered RMSNorm** for stability, and
  improved MoE router initialization, resulting in **~10× cheaper training and
  10× faster inference** compared to previous models. **Alibaba's Qwen3-Next**
  reportedly outperforms **Gemini-2.5-Flash-Thinking** and approaches the
  flagship 235B model's performance, with deployments on **Hugging Face**,
  **Baseten**, and native **vLLM** support for efficient inference.
companies:
  - alibaba
  - mistral-ai
  - deepseek
  - snowflake
  - hugging-face
  - baseten
  - nvidia
models:
  - qwen3-next
  - qwen3
  - mixtral-8x7b
  - gemini-2.5-pro
topics:
  - mixture-of-experts
  - model-sparsity
  - gated-attention
  - hybrid-architecture
  - rmsnorm
  - model-stability
  - model-training
  - inference-optimization
  - multi-token-prediction
  - model-deployment
people:
  - justinlin610
  - teortaxestex
  - yuchenj_uw
---

Archived full issue body is served from static HTML at /frozen-issues/25-09-11-qwen3-next.html.
