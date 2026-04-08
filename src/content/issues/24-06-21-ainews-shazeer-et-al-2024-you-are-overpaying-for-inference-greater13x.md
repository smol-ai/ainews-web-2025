---
id: af97a688-ee5b-4774-b5ac-d3f723dd3834
title: 'Shazeer et al (2024): you are overpaying for inference >13x'
date: '2024-06-22T00:48:48.532463Z'
original_slug: ainews-shazeer-et-al-2024
description: >-
  **Noam Shazeer** explains how **Character.ai** serves **20% of Google Search
  Traffic** for LLM inference while reducing serving costs by a factor of **33**
  compared to late 2022, with leading commercial APIs costing at least **13.5X
  more**. Key memory-efficiency techniques include **MQA > GQA** reducing KV
  cache size by 8X, hybrid attention horizons, cross-layer KV-sharing, stateful
  caching with a 95% cache rate, and native int8 precision with custom kernels.
  **Anthropic** released **Claude 3.5 Sonnet**, which outperforms **Claude 3
  Opus** at twice the speed and one-fifth the cost, passing **64%** of internal
  pull request tests and introducing new features like Artifacts for real-time
  doc and code generation. Discussions on LLM architecture highlight the
  dominance of transformers, challenges in scaling and overfitting, and the
  importance of architecture work for progress.
companies:
  - character.ai
  - anthropic
models:
  - claude-3.5-sonnet
  - claude-3-opus
topics:
  - memory-efficiency
  - kv-cache
  - attention-mechanisms
  - stateful-caching
  - int8-precision
  - transformer-architecture
  - scaling
  - overfitting
  - architecture
people:
  - noam-shazeer
  - kevin-a-fischer
  - sebastien-bubeck
  - _aidan_clark_
  - andrej-karpathy
---

Archived full issue body is served from static HTML at /frozen-issues/24-06-21-ainews-shazeer-et-al-2024-you-are-overpaying-for-inference-greater13x.html.
