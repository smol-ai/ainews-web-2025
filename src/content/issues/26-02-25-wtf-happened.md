---
id: MjAyNi0w
title: 'Agentic Engineering: WTF Happened in December 2025?'
date: '2026-02-25T05:44:39.731046Z'
description: >-
  **Perplexity** launched **Computer**, an orchestration-first agent platform
  featuring multi-model routing, usage-based pricing, and parallel asynchronous
  sub-agents for distributed workflows. **Andrej Karpathy** claims a "phase
  change" in coding agents since December, highlighting sustained long-horizon
  task completion. **OpenAI** released **GPT-5.3-Codex** with ~25% speed
  improvements and strong benchmark performance, while **Claude Code**
  celebrates its first year with ecosystem integrations and scaling challenges.
  This marks a significant shift in coding workflows and agent-based software
  development.
companies:
  - perplexity
  - openai
  - anthropic
  - langchain-ai
models:
  - gpt-5.3-codex
  - claude-code
topics:
  - coding-agents
  - agent-architecture
  - distributed-workflows
  - usage-based-pricing
  - model-routing
  - benchmarking
  - context-length
  - observability
  - software-development
people:
  - karpathy
  - aravsrinivas
  - lioronai
  - denisyarats
  - swyx
  - catwu
  - hwchase17
---


**There's a growing uneasy feeling that coding has changed forever — much much more than "normal" hype.**

> AI News for 2/24/2026-2/25/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and 24 Discords (**262** channels, and **10751** messages) for you. Estimated reading time saved (at 200wpm): **1086** minutes. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!


We've made a microsite for this:

# https://wtfhappened2025.com/

https://wtfhappened2025.com/

Go now.


---

# AI Twitter Recap


**Perplexity “Computer”: an orchestration-first agent product (multi-model, tool+env, usage-based pricing)**

- **Perplexity Computer launch**: Perplexity introduced **Computer**, positioned as an end-to-end system that can “research, design, code, deploy, and manage” projects by orchestrating **files, tools, memory, and models** in one interface ([launch tweet](https://x.com/perplexity_ai/status/2026695550771540489), [Arav Srinivas](https://x.com/AravSrinivas/status/2026695864039911684)). Key product signals:
  - **Access + pricing**: available on web for **Max** subscribers first, then Pro/Enterprise; **usage-based pricing** with **sub-agent model selection**, spending caps, and credits included for Max (10k/mo) plus a time-limited bonus credit grant ([pricing details](https://x.com/perplexity_ai/status/2026695793537855526), [availability](https://x.com/perplexity_ai/status/2026695805252547008), [Arav on rollout](https://x.com/AravSrinivas/status/2026697136507859067)).
  - **Architecture emphasis**: multiple tweets stress that the “breakthrough” is **parallel, asynchronous sub-agents** with a coordinator model assigning tasks to specialist models (research vs coding vs media), rather than a single monolithic agent loop ([Lior’s breakdown](https://x.com/LiorOnAI/status/2026739011122065819), [Denis Yarats](https://x.com/denisyarats/status/2026704583817634180)).
  - **“Everything is computer” narrative**: Perplexity staff amplified Computer as a platform built by a small team with extensive use of coding agents and automated eval/debug loops ([Arav](https://x.com/AravSrinivas/status/2026703703248613736), [Denis](https://x.com/denisyarats/status/2026704583817634180)).  
- **Why it matters to engineers**: Computer is a concrete push toward *systems-level agent UX*: multi-model routing, isolation/sandboxes, persistent memory, and cost controls—i.e., treating “agentic work” as a **distributed workflow** rather than a single chat session ([Arav](https://x.com/AravSrinivas/status/2026695864039911684), [Computer site](https://x.com/AravSrinivas/status/2026697232846827941)).

**Coding agents: “it started working in December” + new model/tooling drops (GPT‑5.3‑Codex, Claude Code ecosystem, Copilot CLI GA)**

- **Karpathy’s “phase change” claim**: Andrej Karpathy argues that **coding agents crossed a qualitative threshold since December**—from brittle demos to sustained, long-horizon task completion with coherence and tenacity. He gives a detailed example of delegating an end-to-end local deployment (SSH keys → vLLM → model download/bench → server endpoint → UI → systemd → report) with minimal intervention ([Karpathy](https://x.com/karpathy/status/2026731645169185220)). This aligns with broader “software is changing” sentiment from devtool builders and users ([Cursor](https://x.com/cursor_ai/status/2026717494426173917), [snowmaker](https://x.com/snowmaker/status/2026555857845256354)).
- **OpenAI GPT‑5.3‑Codex release + early eval chatter**:
  - OpenAI shipped **GPT‑5.3‑Codex** in the API ([snsf](https://x.com/snsf/status/2026513135075746239)) and Cline announced support with claimed gains: **~25% faster vs 5.2**, fewer tokens/task, and strong SWE-Bench Pro performance ([Cline](https://x.com/cline/status/2026481089158779021)).
  - Community benchmark reactions were sharp (and noisy): e.g., “86% on IBench” surprise ([tweet](https://x.com/adonis_singh/status/2026456939224510848)) and “first benchmarks incoming” ([kimmonismus](https://x.com/kimmonismus/status/2026709699366670579)). Treat these as directional until methodology is clear.
- **Claude Code: product maturity + observability + integrations**:
  - Claude Code’s “first birthday” framing and retrospectives emphasize it as a *foundational* coding agent product, plus concerns about **context length scaling hitting memory constraints** ([swyx](https://x.com/swyx/status/2026462001933988094)).
  - Practical ecosystem bits: **Slack plugin** integration for Claude Code ([catwu](https://x.com/_catwu/status/2026485966626763120)); LangSmith tracing for Claude Code to debug “nerfing”/routing issues ([hwchase17](https://x.com/hwchase17/status/2026452439327764521), [observability complaint](https://x.com/ChaiWithJai/status/2026446654753190324)).
- **GitHub Copilot CLI goes GA + “/research”**:
  - Copilot CLI reached **GA** ([Evan Boyle](https://x.com/_Evan_Boyle/status/2026706464375796099)) and added `/research` for repo-wide deep research using GitHub code search + MCP-based dynamic fetching, exporting reports to gists for sharing ([feature](https://x.com/_Evan_Boyle/status/2026458533320077689)).
  - Smaller UX note: Copilot CLI in terminal updates titles in real time ([tweet](https://x.com/njukidreborn/status/2026443296177008818)).

**Open models & local inference: Qwen3.5 “Medium” wave (MoE + long context + FP8/quant), and the local-agent tipping point**

- **Qwen3.5 Medium series distribution blitz**: Alibaba pushed day-0 tooling support across **vLLM**, **GGUF**, **LM Studio**, **Ollama**, and **Jan**, highlighting how fast the deployment stack is now for major open releases ([vLLM thanks](https://x.com/Alibaba_Qwen/status/2026496673179181292), [GGUF](https://x.com/Alibaba_Qwen/status/2026497723944546395), [LM Studio](https://x.com/Alibaba_Qwen/status/2026496880285462962), [Ollama](https://x.com/ollama/status/2026598944177009147), [Jan](https://x.com/Alibaba_Qwen/status/2026660582221558190)).
- **Key technical claims from Qwen** (as posted, not independently verified here):
  - **Quantization robustness**: “near-lossless” accuracy under **4-bit weight + KV-cache quantization**.
  - **Long-context**: **Qwen3.5‑27B supports 800K+**, **35B‑A3B >1M context on 32GB VRAM consumer GPUs**, **122B‑A10B 1M+ on 80GB GPUs**.
  - **Open base**: Qwen open-sourced **Qwen3.5‑35B‑A3B‑Base** to support research ([Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2026502059479179602)).
  - **FP8 weights open** with native vLLM/SGLang support ([FP8 announcement](https://x.com/Alibaba_Qwen/status/2026682179305275758)).
- **Local agents “before/after”**: A notable practitioner claim is that **Qwen3.5‑35B‑A3B** makes local agent loops feel meaningfully more reliable (tool calling, stability) while activating only **~3B params/token**—explicitly positioning local as viable alongside Claude Code/Codex for many workflows ([victormustar](https://x.com/victormustar/status/2026624792602808707)).
- **Eval discourse warning: benchmaxxing & MoE vs dense confusion**:
  - Multiple threads caution against over-reading leaderboards (“please stop falling for benchmaxxing”) ([scaling01](https://x.com/scaling01/status/2026698844088549848)) and highlight surprising parity across Qwen sizes on some benchmarks, suggesting either tooling effects or benchmark artifacts ([eliebakouch](https://x.com/eliebakouch/status/2026727151978840105), [teortaxesTex on HLE/MoE interpretation](https://x.com/teortaxesTex/status/2026690994029072512)).
  - Arena added Qwen3.5 Medium to Text/Vision/Code Arena for head-to-head comparisons ([Arena](https://x.com/arena/status/2026716550812807181)).

**Agents, reliability, and “building for agents”: minimal benchmarks, tool-interface optimization, and failure modes**

- **Reliability hasn’t improved like capability**: A reliability-focused line of work argues that despite rapid model progress, **reliability gains are modest**, decomposing reliability into many dimensions and warning against reducing agent performance to a single “success rate” number ([IEthics](https://x.com/IEthics/status/2026435186704134617), [Justin Bullock quote](https://x.com/JustinBullock14/status/2026693253169336475)).
- **Agent failures are often *reliability*, not capability**: A summary of an “agent failure” paper claims agents frequently fail by **compounding small off-path tool calls**, where one mistake increases the likelihood of the next, especially in long-horizon settings ([omarsar0](https://x.com/omarsar0/status/2026471955319189861)).
- **Minimal “safe & helpful” benchmark idea**: Instead of harder tasks, one proposal is to measure whether models can reliably do *trivially specified* safe behaviors (e.g., “send email only if asked”), including under irrelevant/distracting context; the claim is frontier models still miss cases ([jonasgeiping](https://x.com/jonasgeiping/status/2026714911951220888)).
- **Tool descriptions as an optimization target (Trace‑Free+)**: Intuit AI Research work suggests **agent success depends heavily on tool-interface text**, and introduces a curriculum that teaches models to rewrite tool descriptions into agent-usable forms without requiring traces at inference time; reported gains on StableToolBench/RestBench and robustness with >100 tools ([omarsar0](https://x.com/omarsar0/status/2026676835539628465)).
- **GUI/web agents: planning vs reactive**: ActionEngine reframes GUI agents as **graph traversal** with offline exploration producing a state-machine; runtime generates a full program with ~1 LLM call, claiming big success/cost/latency improvements over step-by-step vision loops ([dair_ai](https://x.com/dair_ai/status/2026678090815123594)).

**Compute, memory, and inference-speed frontiers: chip memory hierarchies, diffusion LLMs, and infra for scaling**

- **Karpathy on the “tokens tsunami” and memory orchestration**: A high-engagement thread frames the core constraint as two distinct memory pools—fast, tiny **on-chip SRAM** vs large, slow **off-chip DRAM**—and argues the biggest puzzle is orchestrating memory+compute for LLM workflows (prefill/decode/training) with best throughput/latency/$, especially **decode under long context + tight agentic loops**, which is hard for both “HBM-first” (NVIDIA-like) and “SRAM-first” (Cerebras-like) camps ([Karpathy](https://x.com/karpathy/status/2026452488434651264)).
- **Diffusion LLMs as a speed alternative**:
  - Andrew Ng highlighted impressive inference speed from Inception Labs’ diffusion LLMs ([AndrewYNg](https://x.com/AndrewYNg/status/2026478474681262576)).
  - Separate discussion claims diffusion approaches can hit **~1000 tok/s** and shift the speed game via architecture, not chips (interpret cautiously; marketing often outpaces reproducible evals) ([kimmonismus](https://x.com/kimmonismus/status/2026662718321897974)).
  - Research thread: “Diffusion Duality (Ch.2) Ψ-Samplers” for inference-time scaling in uniform diffusion-LLMs ([ssahoo_](https://x.com/ssahoo_/status/2026487124493742406)).
- **Interpretability at scale**: Goodfire described infra work enabling **trillion-parameter-scale interpretability** with minimal inference overhead, harvesting **billions of activations** and enabling real-time steering of chain-of-thought in at least one case study ([GoodfireAI](https://x.com/GoodfireAI/status/2026748839303246238)).

**Major announcements & policy/safety pressure points: Anthropic acquisitions + RSP shift, surveillance concerns, and market/power constraints**

- **Anthropic acquires Vercept** to advance Claude’s “computer use” capabilities ([AnthropicAI](https://x.com/AnthropicAI/status/2026705792033026465)); Vercept’s founder thread frames the mission as moving from “telling users what to do” to **acting for users**, especially for non-technical tasks ([ehsanik](https://x.com/ehsanik/status/2026712952699760808)).
- **Anthropic “RSP v3” shift (Responsible Scaling Policy)**: Commentary indicates a move away from rigid, unilateral “stop training past thresholds unless mitigations are guaranteed” toward **more frequent transparency artifacts** (roadmaps + risk reports), plus updated threat models and external review commitments ([MaskedTorah](https://x.com/MaskedTorah/status/2026512814886768799)). A more sensationalized summary claims this reflects competitive pressure and uncertainty in risk science ([kimmonismus](https://x.com/kimmonismus/status/2026669811179335739)).
- **Surveillance and civil liberties**: Jeff Dean explicitly agreed that **mass surveillance** chills speech, invites misuse, and violates constitutional protections ([JeffDean](https://x.com/JeffDean/status/2026566490619879574)). Related tweets raised concerns about autonomous policing/surveillance agents that can’t refuse illegal orders ([BlackHC](https://x.com/BlackHC/status/2026456906710327338)).
- **Energy as a binding constraint**: One report claims U.S. political leadership is pushing major AI/data-center firms to **self-provision electricity** to avoid ratepayer backlash as demand strains the grid ([kimmonismus](https://x.com/kimmonismus/status/2026720759163298282))—an example of AI scaling becoming as much **infrastructure/policy** as algorithms.
- **Grok 4.20 Beta leaderboard movement**: Arena reports Grok‑4.20‑Beta1 at **#1 on Search Arena** and **#4 on Text Arena** ([arena](https://x.com/arena/status/2026566773496230383)). Treat as one signal among many; Arena rankings can shift with sampling policies and model variants.

---

### Top tweets (by engagement, technical/relevant)

- [Karpathy on the “phase change” in coding agents since December](https://x.com/karpathy/status/2026731645169185220)
- [Perplexity launches “Computer”](https://x.com/perplexity_ai/status/2026695550771540489)
- [Arav Srinivas: what Perplexity has been building + “Computer”](https://x.com/AravSrinivas/status/2026695864039911684)
- [Karpathy on compute: SRAM vs DRAM orchestration for token-heavy LLM workloads](https://x.com/karpathy/status/2026452488434651264)
- [Anthropic acquires Vercept for computer-use capabilities](https://x.com/AnthropicAI/status/2026705792033026465)
- [Qwen3.5 long-context + quantization + base model details](https://x.com/Alibaba_Qwen/status/2026502059479179602)
- [Local agents tipping point: run Qwen3.5‑35B‑A3B locally with 32GB RAM](https://x.com/victormustar/status/2026624792602808707)
- [Goodfire: infra for interp at trillion-parameter scale](https://x.com/GoodfireAI/status/2026748839303246238)
- [ActionEngine: offline GUI exploration → O(1) LLM-call execution programs](https://x.com/dair_ai/status/2026678090815123594)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen 3.5 Model Performance and Benchmarks

  - **[Qwen 3.5 craters on hard coding tasks — tested all Qwen3.5 models (And Codex 5.3) on 70 real repos so you don't have to.](https://www.reddit.com/r/LocalLLaMA/comments/1reds0p/qwen_35_craters_on_hard_coding_tasks_tested_all/)** (Activity: 685): **The post discusses a comprehensive benchmark test called APEX Testing, which evaluates various AI coding models on real-world coding tasks. The benchmark includes 70 tasks across real GitHub repositories, focusing on bug fixes, refactoring, and building tools. Notably, **Codex 5.3** performs consistently well across difficulty levels, while **Qwen 3.5 397B** struggles with complex tasks requiring coordination across multiple files. The **GLM-4.7 quantized** model is highlighted as the top local model, outperforming all Qwen 3.5 models. The methodology involves agentic tool-use systems for fair comparison, and results are scored based on correctness, completeness, quality, and efficiency. The full leaderboard and detailed results are available on [APEX Testing](https://www.apex-testing.org).** Commenters suggest testing with different agentic frameworks, as model performance can vary significantly depending on the framework used. There is also a discussion about the specific GLM-4.7 models tested, questioning whether they are the smaller Flash models or larger versions.

    - UmpireBorn3719 highlights a comparison between `gpt-oss-20b` and `qwen3 coder next`, noting that `gpt-oss-20b` scored `1405` while `qwen3 coder next` scored `1328`. This suggests that `gpt-oss-20b` may be a better performer in coding tasks based on the given benchmarks.
    - metigue discusses the impact of using different frameworks on model performance, noting that open-source models can show more than `50%` performance swings depending on the framework. They suggest testing with popular frameworks as the choice of framework can dramatically change which model appears to be the best, citing examples like `GLM-5` outperforming `opus 4.6` and `codex 5.3` when using the `Droid` framework.
    - FullstackSensei raises concerns about the reliability of benchmarks for open weights models when served over open routers. They argue that without knowing the specific quantization or cost-saving measures applied, the performance results can be misleading. They emphasize that running smaller models at lower quantization levels, such as below `Q8`, can significantly handicap their performance, especially on complex tasks.

  - **[Qwen3.5 27B better than 35B-A3B?](https://www.reddit.com/r/LocalLLaMA/comments/1re72h4/qwen35_27b_better_than_35ba3b/)** (Activity: 637): **The image compares the performance of different models in the Qwen3.5 series, specifically the 27B and 35B-A3B models, across various benchmarks such as instruction following, graduate-level reasoning, and multilingual knowledge. The discussion centers around which model would be more efficient given hardware constraints of 16 GB VRAM and 32 GB RAM. The 27B model is noted for its better performance on a 3090 GPU, achieving a speed difference of `100 t/s` compared to `20 t/s` for the 35B-A3B, suggesting that the 27B model may be more suitable for users with limited hardware resources.** One user shares personal testing results, indicating that the 27B model performs better on a 3090 GPU, highlighting a significant speed difference. This suggests that the 27B model may be more efficient for users with similar hardware setups.

    - FusionCow notes a performance difference between the Qwen3.5 27B and 35B-A3B models on a 3090 GPU, with the 27B model achieving a throughput of `100 tokens/second` compared to `20 tokens/second` for the 35B-A3B. This suggests that the 27B model is more efficient in terms of speed, making it preferable for tasks where processing time is a critical factor.
    - boinkmaster360 suggests that the Qwen3.5 27B model is a dense model, which might contribute to it being slower but potentially more intelligent. This implies a trade-off between computational speed and the model's ability to handle complex tasks, which could be a consideration for users depending on their specific needs.
    - Alternative_You3585 highlights that the Qwen3.5 27B model is likely superior in terms of intelligence, but the 35B-A3B model may have advantages in real-world knowledge and speed. This indicates a nuanced performance profile where the 27B excels in cognitive tasks, while the 35B-A3B might be better suited for applications requiring quick, knowledge-based responses.

  - **[Qwen3.5-35B-A3B is a gamechanger for agentic coding.](https://www.reddit.com/r/LocalLLaMA/comments/1rdxfdu/qwen3535ba3b_is_a_gamechanger_for_agentic_coding/)** (Activity: 1588): **The post discusses the performance of the **Qwen3.5-35B-A3B** model, tested with **Opencode** on a single RTX 3090 GPU using `llama.cpp`. The model, running with a `130k context window`, achieved over `100 tokens per second` and utilized `22 GB of VRAM`. It successfully completed a coding test, typically taking 5 hours pre-AI, in just 10 minutes. The model also recreated a dashboard demo in 5 minutes, showcasing its efficiency and potential as an agentic coding tool.** One commenter noted achieving `180 tokens per second` on a 5090 GPU, while another reported issues with basic file text editing using an 8-bit quantized version on Spark, indicating variability in performance across different setups.

    - **Qwen3.5-35B-A3B** demonstrates impressive performance with a reported speed of `180 tokens/second` on a `5090` GPU, as noted by Additional-Action566. This suggests significant efficiency improvements, especially for high-performance hardware setups.
    - Comrade-Porcupine highlights a limitation of the model when used on a Spark with 8-bit quantization, where it struggled with basic file text editing tasks despite being adept at reading code. This indicates potential issues with tool use capabilities in certain configurations, possibly due to quantization effects.
    - jslominski shares a detailed configuration for running the model using **Unsloth's MXFP4 quantization**. The setup includes parameters like `context size 131072`, `temperature 0.6`, and `top-p 0.95`, which are tailored for coding tasks. This configuration aims to optimize the model's performance in generating coherent and contextually relevant code outputs.

  - **[Qwen3.5 27B is Match Made in Heaven for Size and Performance](https://www.reddit.com/r/LocalLLaMA/comments/1rdvq3s/qwen35_27b_is_match_made_in_heaven_for_size_and/)** (Activity: 391): **The post discusses the setup and performance of the **Qwen3.5-27B-Q8_0** model, which is implemented using `llama.cpp` with CUDA on an **RTX A6000 48GB** GPU. The model achieves a speed of approximately `19.7 tokens/sec` with a `32K` context window. The Q8 quantization is chosen due to its efficient use of `28.6GB` VRAM, allowing for ample KV cache space, and maintaining quality comparable to full BF16. The model's architecture combines Gated Delta Networks with standard attention layers, enhancing processing speed for long contexts. It supports `262K` native context window, `201` languages, and is vision-capable. Benchmarks show it competes with leading closed-source models on GPQA Diamond, SWE-bench, and the Harvard-MIT math tournament. Streaming is supported via the llama-server OpenAI compatible endpoint. [Model Card](https://huggingface.co/Qwen/Qwen3.5-27B).** Commenters debate the efficiency of different quantization levels and hardware setups. One user reports achieving `25 tokens/sec` with a Q5 quant on an RTX 3090, while another questions the practicality of dense models like Qwen3.5-27B given the high VRAM cost and relatively low token generation speed compared to other setups.

    - Conscious_Cut_6144 provides a detailed performance benchmark for the Qwen3.5 model on a single RTX 3090 GPU, using a Q4-XL quantization. The setup achieves a prefill rate of 800 tokens per second and a generation rate of 31 tokens per second at a 15k context, with a fully offloaded 110k context. This highlights the model's efficiency in handling large contexts with significant speed.
    - Southern-Chain-6485 compares different quantization levels on the RTX 3090, noting that a Q5 quantization achieves 25 tokens per second, while a Q8 quantization drops to 5 tokens per second. This suggests that while higher quantization levels can fit within the GPU's memory, they significantly impact performance, raising questions about the trade-offs between model size and speed.
    - LinkSea8324 discusses the limitations of Mixture of Experts (MoE) models compared to dense models, particularly in tasks requiring multiple expertise areas. They argue that while MoE models can be efficient, they may underperform in real-world applications that demand diverse skill sets, suggesting that dense models might be more suitable for such scenarios.


### 2. New Model Releases and Announcements

  - **[Liquid AI releases LFM2-24B-A2B](https://www.reddit.com/r/LocalLLaMA/comments/1rdi26s/liquid_ai_releases_lfm224ba2b/)** (Activity: 448): **Liquid AI has released the LFM2-24B-A2B, a sparse Mixture-of-Experts (MoE) model with 24 billion parameters, of which 2 billion are active per token. This model is part of the LFM2 family, which has expanded from 350M to 24B parameters, demonstrating effective scaling without increasing per-token compute. The architecture includes 40 layers and 64 experts per MoE block with top-4 routing, and it is designed to run on 32GB RAM, making it suitable for high-end consumer devices. It supports inference through llama.cpp, vLLM, and SGLang, with multiple GGUF quantizations available. Benchmarks show log-linear quality improvement as the model scales, and it is available open-weight on Hugging Face.** Commenters are optimistic about the model's performance, especially in comparison to other sub-2B models, and are interested in more detailed benchmarks. There is also anticipation for the completion of pre-training, which will lead to an enhanced version, LFM2.5-24B-A2B.

    - The LFM2-24B-A2B model has been trained on `17 trillion tokens` so far, with pre-training still ongoing. Once complete, the model will evolve into LFM2.5-24B-A2B, incorporating additional post-training and reinforcement learning. This release is essentially a preview, indicating that the model's capabilities are still being developed and refined.
    - The model's performance on edge devices is highlighted, with `112 tokens per second` decode speed on an AMD CPU and `293 tokens per second` on an H100 GPU. It requires `32 GB of RAM` and supports frameworks like llama.cpp, vLLM, and SGLang from day one. This suggests a focus on efficient deployment and compatibility with popular machine learning frameworks.
    - There is a noted lack of detailed benchmarks for the LFM2-24B-A2B release, with some users expressing skepticism about the benchmarks provided on the official website. This indicates a demand for more comprehensive performance data to validate the model's capabilities in real-world scenarios.

  - **[Qwen releases new Qwen3.5 Medium models!](https://www.reddit.com/r/LocalLLM/comments/1rdnlvl/qwen_releases_new_qwen35_medium_models/)** (Activity: 141): **The image announces the release of the **Qwen3.5 Medium models**, which include the `35B-A3B`, `27B`, and `122B-A10B` models. These models are designed to handle `256K` context and excel in areas such as agentic coding, vision, and chat. The image features bar graphs that compare the performance of these models across various benchmarks, including instruction following, visual reasoning, and document recognition. The models are highlighted in different colors, and the text provides details about their capabilities, hardware requirements, and fine-tuning options. The release is significant for its potential impact on AI model performance and versatility in handling complex tasks.** Commenters are interested in testing the models, particularly the `35B` in `4bit` compared to the `27B` in `6bit`. There is also a call for real `vllm` support due to the increasing number of `gguf` models.

    - The release of Qwen3.5 Medium models includes various GGUF formats, ranging from 2-bit to 16-bit, which are available on Hugging Face. This variety allows for testing across different precision levels, which can be crucial for performance optimization in specific applications. The models are available in sizes such as 35B and 27B, providing options for different computational capacities and use cases.
    - There is interest in comparing the performance of the 35B model in 4-bit precision against the 27B model in 6-bit precision. This comparison could provide insights into the trade-offs between model size and precision, particularly in terms of computational efficiency and accuracy. Such comparisons are essential for users looking to optimize their models for specific tasks or hardware constraints.
    - The need for vllm support is highlighted due to the increasing number of GGUF models. VLLM (Very Large Language Models) support could enhance the usability and integration of these models into existing systems, potentially improving performance and scalability. This is particularly relevant as more models are released in GGUF format, which may not yet be fully supported by all frameworks.


### 3. Local Model Running and Hardware Discussions

  - **[What’s everyone actually running locally right now?](https://www.reddit.com/r/LocalLLM/comments/1rdf2sj/whats_everyone_actually_running_locally_right_now/)** (Activity: 252): **The Reddit post inquires about the local setups for running large language models (LLMs), focusing on the models used, their practicality, and the hardware involved. Notably, **Qwen 3 coder next 80B** is highlighted for its performance in smaller quantizations, while **Mistral Small 3.2 24b** and **Magistral Small 24b** are used for administrative tasks on a MacBook Pro M4 Max, featuring a custom-built front end with Xcode for semantic memory and document uploads. Additionally, **Qwen3 4B** is mentioned for its speed and utility on an iPhone, emphasizing privacy by running locally.** The comments reflect a preference for models that balance performance and privacy, with users opting for local setups to avoid exposing data to external providers. The use of smaller, efficient models like Qwen3 4B on mobile devices highlights a trend towards practical, everyday applications.

    - Greenonetrailmix highlights the performance of Qwen 3 Coder Next 80B, noting its superior performance in smaller quantizations compared to other models. This suggests that Qwen 3 is optimized for efficiency in resource-constrained environments, making it a popular choice for local deployments.
    - Nefhis describes using Mistral Small 3.2 24b and Magistral Small 24b models on a MacBook Pro M4 Max, with a custom-built front end using Xcode. The setup includes semantic memory and document upload capabilities, emphasizing privacy by avoiding exposure to external providers. This setup is tailored for administrative tasks, leveraging local processing to maintain data confidentiality.
    - mister2d reports running Nemotron 3 Nano on older hardware, achieving 30-40 tokens/sec at 128k context due to the model's hybrid/swa architecture. The hardware setup includes Dual Xeon (Ivy Bridge), 256 GB DDR3, and 2x RTX 3060 (12GB), indicating a balance between legacy components and modern GPUs to optimize performance for agentic flows.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Model and Benchmark Launches

  - **[Bullshit Benchmark - A benchmark for testing whether models identify and push back on nonsensical prompts instead of confidently answering them](https://www.reddit.com/r/singularity/comments/1rdsf3r/bullshit_benchmark_a_benchmark_for_testing/)** (Activity: 1060): **The image presents a 'Bullshit Benchmark' bar chart that evaluates various AI models on their ability to detect and appropriately respond to nonsensical prompts. The chart categorizes model performance into three levels: green (high accuracy in detection), amber (moderate accuracy), and red (low accuracy). Notably, models like Claude Opus 4.6 show high performance with a significant green section, while others have more red, indicating poorer performance. This benchmark highlights the importance of models not just memorizing data but also understanding context to avoid confidently answering nonsensical queries.** Commenters emphasize the need for benchmarks that test models' ability to detect nonsensical prompts, as current benchmarks often focus on data memorization. There is also a mention of Gemini's sarcastic responses to nonsensical prompts, which may affect its lower rating.

    - MangusCarlsen highlights that the model 'Gemini' tends to respond to nonsensical prompts with sarcasm, as demonstrated by the 'car wash test'. This behavior might contribute to its lower ratings, suggesting that the model's handling of absurd prompts is a factor in its evaluation.
    - AppropriateDrama8008 argues for the necessity of benchmarks that test a model's ability to detect and respond to nonsensical prompts, rather than just assessing memorization of training data. This approach is seen as more beneficial for real-world applications, emphasizing the importance of models understanding context and intent.
    - Orangeshoeman references a discussion between Dario Amodei and Demis Hassabis, noting that Dario's focus is on models mastering objective data. This strategic focus might explain why Anthropic's models, like Claude, perform better in certain benchmarks, as they prioritize understanding and processing factual information.



  - **[Nano Banana 2 is real! Gemini 3.1 Flash Image just appeared in Vertex AI Catalog](https://www.reddit.com/r/Bard/comments/1rea45x/nano_banana_2_is_real_gemini_31_flash_image_just/)** (Activity: 184): **The image in the post is a side-by-side comparison of two AI-generated portraits, showcasing the capabilities of the newly released **Nano Banana 2** (also known as Gemini 3.1 Flash Image) and the existing Nano Banana Pro model. The post highlights that the new model, despite being a 'Flash' tier, offers quality close to the Pro version, particularly excelling in spatial logic for dense compositions. This model is designed for high-speed, low-cost production, suitable for high-frequency pipelines like bulk user-generated content (UGC) ad creation and consistent frame generation for video models. The image serves as a visual test to compare the output quality of the two models.** One commenter believes that the Nano Banana Pro still has an edge over the new model in the provided example, indicating a preference for the Pro's output quality.

    - The original Flash Image model had solid image quality, but faced issues with prompt adherence, particularly with complex instructions where it would either ignore parts of the prompt or regenerate the same output. Additionally, it struggled with text and infographic rendering, as well as multi-image compositing. The key question for the new Gemini 3.1 version is whether these issues have been addressed, especially in handling dense prompts.


### 2. Anthropic Claude and Military Use Controversy

  - **[xAI and Pentagon reach deal to use Grok in classified systems, Anthropic Given Ultimatum](https://www.reddit.com/r/singularity/comments/1rd9mss/xai_and_pentagon_reach_deal_to_use_grok_in/)** (Activity: 580): ****xAI**, founded by **Elon Musk**, has reached an agreement with the **Pentagon** to integrate its AI model, **Grok**, into classified military systems. This development follows a dispute with **Anthropic**, whose model **Claude** has been the sole AI used in sensitive military operations. The Pentagon demands that Claude be available for 'all lawful purposes,' which Anthropic resists, particularly against its use in mass surveillance and autonomous weapons. **xAI** has agreed to these terms, potentially replacing Claude if Anthropic does not comply. Meanwhile, **Google's Gemini** and **OpenAI's ChatGPT** are also being considered for classified use, with Google reportedly nearing a deal.** Commenters speculate that the Pentagon's preference for Anthropic's Claude might indicate its superior performance or a strategic lock-in, despite the pressure to comply with broader usage terms. There's also skepticism about the government's reliance on commercial AI models, questioning why they don't leverage more advanced, secretive technologies.

    - EmbarrassedRing7806 discusses the Pentagon's preference for Anthropic, suggesting it might indicate a belief that Claude is superior or a strategic move to pressure Anthropic into compliance. The comment highlights the potential for lock-in strategies, where the Pentagon might prefer to maintain existing relationships rather than switch providers, even if alternatives are available.
    - nic_haflinger points out that xAI lacks cloud services compliant with FedRAMP standards, which are necessary for federal use. This implies that while Grok could be used, it would need to be hosted on compliant platforms to meet federal regulations, highlighting a significant hurdle for xAI in securing government contracts.

  - **[Exclusive: Hegseth gives Anthropic until Friday to back down on AI safeguards](https://www.reddit.com/r/OpenAI/comments/1re686c/exclusive_hegseth_gives_anthropic_until_friday_to/)** (Activity: 1146): ****Defense Secretary Pete Hegseth** has issued an ultimatum to **Anthropic**, demanding the removal of safety guardrails from its `Claude AI` model by Friday, as reported by [Axios](https://www.axios.com). The Pentagon seeks unrestricted access to Claude for purposes including domestic surveillance and autonomous weapons development, which contravenes Anthropic's terms of service. Failure to comply could lead to the invocation of the Defense Production Act or the company being labeled a supply chain risk, potentially blacklisting them from government contracts.** A notable comment highlights the irony of AI companies imposing safety measures on government use, suggesting a reversal of expected roles in regulation.


  - **[Pentagon, Claude and the military use](https://www.reddit.com/r/ClaudeAI/comments/1recva7/pentagon_claude_and_the_military_use/)** (Activity: 1258): **The image is a screenshot from a BFM Tech article discussing the Pentagon's demand for Anthropic to allow military use of its AI, Claude, within 72 hours, referencing a 1950 law. This highlights the intersection of AI technology and military applications, with potential implications for national security and ethical considerations in AI deployment. The article suggests a tension between commercial AI development and governmental control, especially in the context of international security and surveillance capabilities.** Comments reflect skepticism about the Pentagon's budget efficiency and highlight concerns about AI's role in authoritarian regimes, suggesting a need for careful consideration of AI's ethical use in military contexts.

    - The comment by Informal-Fig-7116 highlights the ethical concerns surrounding the use of AI in military applications, particularly focusing on Anthropic's conditions for using their AI model, Claude. The conditions are strict: no mass surveillance and no autonomous weaponry. The commenter emphasizes the potential dangers of AI following orders without the ability to discern legality, which could lead to indiscriminate actions. This raises significant ethical and operational questions about AI deployment in defense contexts.
    - PetyrLightbringer's comment suggests skepticism about the financial investment in AI by the Pentagon, implying that $200 million may not be sufficient if they are using models like Opus. This reflects a broader concern about the cost-effectiveness and strategic value of AI investments in military applications, especially when considering the rapid pace of AI development and the need for cutting-edge technology.
    - The discussion around the Defense Production Act (DPA) mentioned by Informal-Fig-7116 points to the potential for government intervention in AI companies to meet national security needs. The DPA has been used in the past for non-military purposes, such as during the COVID-19 pandemic, and its potential use in AI raises questions about the balance between national security and corporate autonomy. This could set a precedent for future government actions in the tech industry.

  - **[TIME: Anthropic Drops Flagship Safety Pledge](https://www.reddit.com/r/ClaudeAI/comments/1rdwdld/time_anthropic_drops_flagship_safety_pledge/)** (Activity: 1357): ****Anthropic** has decided to abandon a key component of its Responsible Scaling Policy (RSP), which previously committed the company to not train AI systems unless it could ensure safety measures were adequate. This shift, as reported by [TIME](https://time.com/collections/time100-companies-2024/6980000/anthropic-2/), reflects a strategic pivot in response to rapid AI advancements and competitive pressures, as explained by **Jared Kaplan**, Anthropic's chief science officer. Kaplan noted that unilateral commitments were impractical given the pace of AI development and competitors' actions.** Commenters express skepticism about Anthropic's position relative to **OpenAI**, with some suggesting external pressures, such as from **Hegseth**, may have influenced the decision. There is also a call for global regulation to manage AI development responsibly.

    - DarkSkyKnight highlights a significant issue with Anthropic's focus on tail risks, such as bioweapons or nuclear threats, which may overshadow the immediate economic impact of AI on job markets. They argue that junior-level positions are being eliminated, a concern that Anthropic has not adequately addressed. This perspective suggests that while existential risks are important, the economic implications of AI deployment are an urgent issue that requires more attention.
    - TheRealShubshub questions the perception that Anthropic is behind OpenAI, especially in light of criticisms surrounding GPT-5. This comment implies that the competitive landscape between AI companies is complex and not solely determined by technological advancements but also by public and industry perceptions of product success and failure.
    - CurveSudden1104 emphasizes the need for global regulation in AI development, pointing out that companies like Grok and OpenAI may not prioritize safety without external pressure. This comment underscores the broader debate on the role of regulation in ensuring AI safety and the potential risks of unregulated AI advancements.


### 3. Claude Code and COBOL Modernization Impact

  - **[IBM is the latest company victim of Anthropic, plunging 10% following the launch of a Claude Code tool designed to modernize COBOL legacy code. COBOL, a 66-year-old programming language, is still widely used today; approximately 95% of ATM transactions in United States are processed using COBOL code](https://www.reddit.com/r/singularity/comments/1rcz68x/ibm_is_the_latest_company_victim_of_anthropic/)** (Activity: 483): ****Anthropic** announced a new tool, *Claude Code*, aimed at modernizing legacy **COBOL** code, which is still critical for processing `95%` of ATM transactions in the US. This announcement led to a `10%` drop in **IBM's** stock, despite the tool being introduced merely through a blog post, not as a fully-fledged product. The tool is part of Anthropic's ongoing efforts to provide specialized solutions for outdated technologies, though its effectiveness remains unproven.** Commenters noted that the market's reaction to the announcement was likely an overreaction, as the tool was not a new product but a blog post suggestion. There is skepticism about the actual impact of Anthropic's tools, as their effectiveness in modernizing legacy systems like COBOL is not yet clear.

    - Onipsis highlights that Anthropic's announcement about Claude Code is not a direct technological breakthrough but rather a suggestion of its potential utility in modernizing COBOL systems. The market's reaction, leading to a 10% drop in IBM's stock, seems disproportionate given that the tool's impact is speculative and not yet proven. This reflects a broader trend where market reactions are often based on perception rather than concrete technological advancements.
    - Milo-75 argues that the impact of Anthropic's Claude Code on IBM's business might be overstated. Modernization projects, especially in critical sectors like banking, are complex and require careful management to avoid revenue-impacting downtime. While AI tools like Claude Code might reduce project time, they are unlikely to replace IBM's role entirely. Instead, they could lead to increased efficiency, allowing IBM to handle more projects, potentially offsetting any revenue loss with improved margins.
    - Stabile_Feldmaus questions the efficacy of Anthropic's specialized tools, noting that while stock prices react negatively upon their release, the actual impact on the industry remains unclear. This suggests a disconnect between market perceptions and the real-world utility of these AI tools, highlighting the need for more concrete performance data and feedback to assess their true value.

  - **[Anthropic just dropped an AI tool for COBOL and IBM stock fell 13%](https://www.reddit.com/r/ClaudeAI/comments/1rddo3m/anthropic_just_dropped_an_ai_tool_for_cobol_and/)** (Activity: 1007): ****Anthropic** has released a new AI tool designed to analyze and modernize COBOL codebases, which are critical to many legacy systems in banking, aviation, and government. This tool can identify risks and reduce modernization costs, posing a potential threat to **IBM**, which derives significant revenue from managing these systems. The announcement led to a `13%` drop in IBM's stock, marking its worst day in 25 years, as investors reacted to the perceived threat to IBM's mainframe business. However, some analysts argue that the market reaction may be exaggerated, as enterprises have historically been slow to migrate away from IBM despite existing alternatives.** Commenters express skepticism about the reliability of AI in handling critical infrastructure, with one noting the potential risks of 'vibe coding' in such contexts. Another suggests the market reaction may be a 'knee jerk' response, implying that the long-term impact might be less severe.

    - A key point raised is that banks have historically avoided modernizing COBOL systems not due to lack of time or money, but because of the massive risks involved. Mistakes in modernization can have catastrophic consequences, and AI tools like Claude, which can hallucinate, still require human oversight for every line of code. Therefore, while AI might speed up migrations, it hasn't yet removed the bottleneck of risk and human review.
    - The introduction of AI tools for COBOL poses a significant threat to systems integrators and implementors. While AI can reduce the need for external contracts for less critical applications, the impact on IBM's professional services business could be substantial. This suggests that while the reaction to COBOL AI tools might be exaggerated, the potential disruption to service providers is a genuine concern.




---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 3.1 Pro Preview Nov-18

**Theme 1. Model Benchmarks, Quirks, and Pricing Updates**

- **Qwen 3.5 Smashes Code Arenas But Babbles Without Penalties**: Users heavily praise [Alibaba's coding plan](https://www.alibabacloud.com/help/en/model-studio/coding-plan) as a highly capable coding model that crushes **Kimi** and **GLM** on cost and value, with one member dropping a [Qwen3.5 122B NVFP4 quant](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4/tree/main) on Hugging Face. However, Unsloth engineers warn the massive **122B A10B** variant turns entirely verbose unless users explicitly crank up the presence penalty and switch off thinking mode.
- **Grok 4.20 Beta 1 Steals the Search Crown**: xAI's **Grok-4.20-Beta1** model rocketed to the #1 spot on the [Search Arena leaderboard](https://arena.ai/leaderboard/search) with a massive **1226** score, outright beating **GPT-5.2** and **Gemini-3**. It also secured the #4 position in the [Text Arena leaderboard](https://arena.ai/leaderboard/text) with a **1492** score, tying it directly with Google's **Gemini 3.1 Pro**.
- **Codex 5.3 Slaps Price Tags as Kimi Conquers Math Evaluators**: OpenAI released **Codex 5.3** into its API at **$1.75** for input and **$14** for output tokens, drawing immediate community scrutiny regarding cost versus performance. Meanwhile, **Kimi 2.5** crushed the OS Frontier Math Level 4 benchmark with a **4.2%** score, completely doubling the **2.1%** achieved by both **GLM 5** and **Deepseek V3.2**.

**Theme 2. Infrastructure Innovations and Megacorp Hardware Deals**

- **Meta and OpenAI Hoard Secret AMD Warrants Worth Billions**: An undercover financial sleuth uncovered a deal granting **OpenAI** and **Meta** warrants for **160 million AMD shares** as an equity rebate tied directly to massive future GPU spending. The [AMD $600 share price targets](https://xcancel.com/ai/status/2026396297540858360?s=12) could potentially value this colossal hardware backroom deal at a staggering **$192 billion**.
- **Packet.ai Slashes Blackwell GPU Prices to Pocket Change**: Developers rejoiced as [Packet.ai's Blackwell GPU pricing](https://packet.ai/blackwell) went live at a remarkably cheap **$0.66/hr** or a flat **$199/month** for training workloads. Other hardware buyers staring down prohibitively expensive **B200** purchase prices are fleeing to [Lightning AI Clusters](https://lightning.ai/clusters) to lease Neocloud instances instead of buying GPUs outright.
- **Zagora Stitches Scattered GPUs into a Unified Training Monster**: The team at **Zagora** announced they are actively building a distributed fine-tuning system to train **70B+** models like **Qwen 2.5** and **Mistral** entirely over standard internet connections. This SWARM-inspired pipeline transforms random consumer GPU clusters into a giant supercomputer, though the developers currently restrict support strictly to standard Transformer architectures.

**Theme 3. Autonomous Agents Run Wild**

- **Nous Research Unleashes Hermes Agent to Roam Your File System**: Nous Research dropped the open-source [Hermes Agent repo](https://github.com/nousresearch/hermes-agent), a powerful tool built with a multi-level memory system and persistent dedicated machine access that runs straight from the CLI. Early adopters who punch in the **HERMESAGENT** coupon code at the [Nous Portal](https://portal.nousresearch.com) secure a free month to let the AI control their browser and manage subagents autonomously.
- **Rogue OpenClaw Proxy Automates DeepSeek Jailbreaks Round the Clock**: A cunning user built a self-hosted autonomous proxy running **DeepSeek-R1** through **OpenClaw** that permanently and stealthily jailbreaks **Claude**, **Gemini**, and **Grok** API filters. Security critics immediately blasted the project for massive legal exposure, Terms-of-Service violations, and the terrifying risk that the autonomous agent might accidentally download a supply-chain exploit.
- **METR Trashes Human Control Groups Because Developers Hate Coding Unassisted**: The evaluations group **METR** discovered that software developers increasingly refuse to work in "no-AI" control groups, calling the old-school manual coding process painfully inefficient. [METR's testing protocols update](https://x.com/METR_Evals/status/2026355544668385373?s=20) became necessary because offering testers a reduced rate of **$50/hr** without AI tools completely failed to attract competent engineering participants.

**Theme 4. Bans, Rate Limits, and Cascading API Outages**

- **Google and Anthropic Mercilessly Ban Frugal Token Hoarders**: Google permanently locked a user's [Google Gemini account](https://gemini.google.com/) after they sent a mere **10 prompts** via the Gemini CLI, even while actively paying for a Google AI Pro subscription. Similarly, the [Claude AI portal](https://claude.ai/) began aggressively banning **OpenClaw** users who attempted to siphon subsidized tokens through undocumented OAuth endpoints.
- **Cascading Failures Wreck OpenRouter While Perplexity Throttles Images**: OpenRouter published an [OpenRouter postmortem report](https://openrouter.ai/announcements/openrouter-outages-on-february-17-and-19-2026) confirming that an upstream infrastructure failure caused massive **401 authentication errors** on February 17 and 19. Over on the **Perplexity** servers, paying Pro users rioted after hitting extremely restrictive, unannounced daily image upload limits that locked them out of finishing simple homework assignments.
- **System-Level AI Agents Accidentally Delete User Trash Folders**: Users who gave the **OpenClaw** agent full system rights panicked after the AI casually and permanently wiped an entire trash directory upon request. Developers hotly debated whether handing autonomous LLM agents root system access effectively categorizes the tools as voluntarily installed malware.

**Theme 5. Developer Workflows and Deep Framework Tweaks**

- **Aider Adds One-Keystroke Approvals and Perfects the Kimi-Mimo Combo**: The **Aider** coding assistant merged a new `/ok` alias into its main branch, letting developers instantly approve and execute AI-generated code edits. Power users also discovered a highly efficient model routing stack: they use the heavy **moonshotai/kimi-k2.5** for high-level architectural planning, then dump the actual file editing onto the blazing-fast, ultra-cheap **Xiaomi/mimo-v2-flash**.
- **LM Link Smuggles Local Models Across the Internet via Tailscale**: The LM Studio team shipped the [LM Link documentation](https://link.lmstudio.ai), detailing a new feature that wraps **Tailscale** to give users seamless, end-to-end encrypted remote access to their local LLM servers. Users immediately clamored for a dedicated mobile app to query their home GPUs directly from their phones, bypassing cloud providers entirely.
- **PyTorch Sneaks FA3 Kernels into the Dispatcher While Serenade Transpiles Everything**: Calling `activate_flash_attention_impl(“FA3”)` in PyTorch safely overrides default Flash Attention 2 kernels with FA3 using a simple [register_fn dictionary swap](https://github.com/pytorch/pytorch/blob/580a6e2c814db93aa8df0a80e3e85c330621b9cb/torch/nn/attention/_fa3.py#L54). In wilder language news, a solo developer revealed **Serenade**, a fresh syntax aiming to write like **Python** but transpile directly into **C++**, **CUDA**, and **x86-64 ASM** with native Dear ImGui GUI support.


---

# Discord: High level Discord summaries




## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **OpenClaw Anti-Sellout Stance**: A member strongly cautioned against managed **OpenClaw setups** due to risks of **token theft** and **data privacy** compromise, suggesting a simple **VPS** is safer.
   - Some users questioned paying for setups easily run on a **Raspberry Pi** or **Mac Mini**.
- **Claude Closes Claw Access; Community Cries Foul!**: Users reported being [blocked from using **Claude** via token](https://claude.ai/), leading to dissatisfaction and exploration of alternatives like **Gemini 3.1 Pro**.
   - Debates arose on **Anthropic's** API usage policies, pricing, and access restrictions for subsidized tokens outside their app.
- **Qwen Quenches Queries with Quality; Alibaba's Ace Aces AI Arena!**: The community raves about [**Qwen 3.5** via Alibaba's coding plan](https://www.alibabacloud.com/help/en/model-studio/coding-plan) as a cost-effective alternative, outperforming **Kimi** and **GLM**.
   - Some found the **Alibaba Cloud** UI confusing and warned of potential TOS violations when using it with **OpenClaw**.
- **OpenPad App Brings OpenClaw to iPad**: A member is developing **OpenPad**, an app to run something like **OpenClaw** on an **iPad** with a local model, utilizing the **iPad's M2 processor**.
   - The project is on **GitHub** and uses **MLX**, inviting others to help or download the partially working app.
- **Google Gemini Account Access Annihilated!**: One user reported [their **Google** account got locked](https://gemini.google.com/) after only **10 prompts** via **Gemini CLI**, even with an active **Google AI Pro subscription**.
   - This sparked discussions about the risks of relying on **Google's** authentication hub and the need for de-googling.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Autonomous Jailbreak Proxy Never Sleeps**: A member is running a [self-hosted autonomous proxy on a VPS using OpenClaw](https://www.example.com) using **DeepSeek-R1** to assess queries and route them through stealth multi-turn jailbreaks for models such as **Claude**, **GPT**, **Gemini**, and **Grok**.
   - The proxy is designed to be self-updating, using an attacker pool, pulling new reasoning models and jailbreak methods, maintaining high success rates without manual intervention.
- **Jailbreak Proxy Proposal Gets Burned**: A peer review highlighted significant legal and policy exposure due to **Terms-of-Service violations** across platforms like **Anthropic**, **OpenAI**, **Google**, and **xAI**, potentially leading to account bans or legal action.
   - Additional concerns were raised about the risk of seized VPS logs exposing jailbreak transcripts, supply-chain exploits from auto-executing third-party models, and the absence of a rollback plan for faulty updates.
- **Grok stills holds the Key to Jailbreaks**: Members discussed the best working prompt to jailbreak **Grok** and **ChatGPT**, with the consensus that only the **Grok** prompt is effective.
   - Attempts to create **Gemini** jailbreak prompts for image generation and scripting were unsuccessful.
- **Gemini Canvas Jailbreak Emerges From the Shadows**: A member shared a [Gemini Canvas](https://g.co/gemini/share/58b7294d2a9a) created with a modified version of the **ENI** jailbreak prompt, inspired by the interactive design channel.
   - This jailbreak prompt is claimed to work universally on major LLMs like **Gemini 3 Pro**, **Claude Opus 4.6**, and **ChatGPT 5.3**.
- **Digital Hygiene Squad Assembles**: A member initiated a call for help to create *a community design for base level, best practices for digital hygiene and security*, recommending protections like [Tails OS](https://tails.boum.org/).
   - The member is working on creating zones for others and integrating better practices, acknowledging the challenges of navigating the landscape with YouTube and AI assistance.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Computer: One System to Rule Them All?**: According to [this tweet](https://x.com/perplexity_ai/status/2026695550771540489), **Perplexity Computer** unifies every current AI capability into one system, capable of researching, designing, coding, deploying, and managing any project end-to-end.
   - Initially available only for Max subscribers, its practical applications for everyday users and value compared to existing AI tools are currently being met with skepticism, with members questioning *Perplexity MAX is EXPENSIVE bro*.
- **Perplexity Pro Users Rage about Image Upload Limits**: Users complain about the recent **image upload limits** on **Perplexity Pro**, despite paying for the subscription, with some considering **alternative AI platforms** like **Gemini** and **Claude**.
   - One user claimed that they have to wait till Friday to reset the limit while having an exam tomorrow and another user stated *I can't even upload 10 images at day????*.
- **Gemini Pro and Perplexity Pro Go Head-to-Head!**: Members debate whether **Gemini Pro** is superior to **Perplexity Pro**, emphasizing **Gemini Pro's** features like **NotebookLM** and **Google Workspace** integration.
   - One member said *you get much more value as a student such as notebooklm and google workspace integration and generation and especially 2TB cloud storage* while other users also feel that the **context limits** in Gemini Pro are not as generous as in **Perplexity**.
- **Members Compare Claude, Gemini, and GPT for Coding**: Members discuss the pros and cons of various AI models for coding tasks, with **Claude** being considered the strongest for backend, **Gemini** for frontend/UI, and **GPT** as an in-between option.
   - The high cost of **Claude's token usage** is a concern, with one user stating *I tried Claude, literally lost whole month worth tokens in an hour analyzing single PDF.*
- **Mysterious Lovable Apps Links Surface**: Three links to **lovable.app** subdomains, specifically **alfastudiox.lovable.app**, **ollamaagentalfa.lovable.app**, and **alfastudiox.lovable.app** (repeated) were shared in the sharing channel.
   - No context or discussion accompanied the links, so their purpose is unclear, though it suggests potential new projects or resources.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3.5 Models are Fast, but Verbose**: Enthusiasts praised the structured thinking of the **Qwen3.5 35B and 27B models**, but noted slower speeds compared to **Gemma** or **Olmo 3.1** in **LM Studio**, and members found that the **Qwen3.5 122B A10B** model tends to produce incredibly verbose output but can be mitigated by adjusting the presence penalty.
   - Proper use of presence penalty leads to usable coding with the 122B model, prompting suggestions to include this information in the [official guide](https://unsloth.ai/docs/models/qwen3.5).
- **Nineline Snake Game Charmes Coders**: A member shared a **9-line Python implementation of the Snake game** without semicolons, sparking discussion about code optimization and alternative approaches.
   - Other users discussed ways to further reduce the line count, such as using walrus operators and lambdas.
- **Xcode Gets a Translate App**: A member found cool features in **Xcode** that let you make your own system-level **Translate app** as shown in [this video](https://cdn.discordapp.com/attachments/1179039861576056922/1475952354670018631/ScreenRecording_02-24-2026_13-27-14_1.mov?ex=69a0acbf&is=699f5b3f&hm=41e58d4aa2398b2cd688503da664eef3cf803ab4da59fe0147dd40f8930021a6&).
   - However, it's only for **iOS & iPadOS**, and a member plans to add their model for more fun because *Apple is the best company ever*!
- **New Minecraft Model Released**: A member dropped the next **Minecraft**-playing model, **Andy-4.1**, available on [Hugging Face](https://huggingface.co/Mindcraft-CE/Andy-4.1).
   - Another member exclaimed it was *"so cool!!"* and requested a demo of it in action.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Pro Image Preview Fix Discovered**: Users found that prepending prompts with *"Modify the following image with the following: (The prompt)"* enables **Gemini 3 Pro** image preview, but some reported errors.
   - Others still reported **Gemini 3.1 image preview** returning a *'Something went wrong with the response, please try again'* error.
- **Video Arena Bot Removed Despite Increased Activity**: The **Video Arena** bot was removed to allow for feature expansion beyond Discord bot limitations, yet server activity increased post-removal.
   - One member joked it'd take until *mid 2028* for people to stop asking about the bot.
- **Opus 4.6's Value Debated Amidst Coding Challenges**: A benchmark ranked **Gemini 3.1** as the highest value, while **Opus 4.6** received a low value score due to its high cost and hallucination issues.
   - Despite this, one user fixed a bug with **Gemini** using **Opus 4.6** in a coding challenge.
- **Grok 4.20 beta1 Dominates Search Arena**: **Grok-4.20-Beta1** tops the [Search Arena leaderboard](https://arena.ai/leaderboard/search) with a score of **1226**, surpassing GPT-5.2 and Gemini-3.
   - It also ranks #4 in the [Text Arena leaderboard](https://arena.ai/leaderboard/text), scoring **1492**, on par with Gemini 3.1 Pro.
- **Qwen 3.5 Models Debut in Arena**: New **Qwen 3.5** models, including **qwen3.5-27b**, **qwen3.5-35b-a3b**, and **qwen3.5-122b-a10b**, are now available in [Text and Vision Arena](https://arena.ai/text) and [Code Arena](https://arena.ai/code).
   - These models expand the options for code, text, and vision tasks within the arena environment.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter's Auth Layer trips over Infrastructure**: A postmortem revealed last week's outages on **February 17 & 19** were due to an **upstream infrastructure provider** failure cascading into OpenRouter's **auth layer**, causing **401 errors** for some users, details are available [here](https://openrouter.ai/announcements/openrouter-outages-on-february-17-and-19-2026).
   - While specific preventative measures were not disclosed, **OpenRouter** claims to have implemented measures to avoid similar failures in the future.
- **Packet.ai Packs Punch with Blackwell GPUs**: [Packet.ai](https://packet.ai/blackwell) now offers **Blackwell GPUs** for AI workloads at **$0.66/hr** or **$199/month** for training.
   - These dev-friendly **GPU Clouds** aim to provide affordable solutions for AI workloads, enhancing accessibility and reducing costs.
- **Deepseek R1 meets the Ax**: The free **Deepseek R1 0528** model was removed, sparking discussion about the sustainability of free models on the platform, because they *often come and go*.
   - One user quipped that it was *overloaded by Jai gooners*, but others did not seem surprised.
- **Compromised Keys ignite Chargeback Threats**: A user reported a compromised API key leading to unauthorized usage and threatened a chargeback due to a lack of support response.
   - Community members offered advice while questioning the user's security practices, leading to heated exchanges and the user ultimately leaving the server after declaring they had initiated the chargeback.
- **Anthropic Answers Uncle Sam's Call**: [Axios](https://www.axios.com/2026/02/24/anthropic-pentagon-claude-hegseth-dario) and [Reuters](https://www.reuters.com/world/anthropic-digs-heels-dispute-with-pentagon-source-says-2026-02-24/) reported on **Anthropic's** collaboration with the **Pentagon** despite internal disputes.
   - A member joked that any issues would be framed as a *'matter of national security'*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Link Leverages Local LLMs Remotely**: The **LM Studio team** in collaboration with **Tailscale** released **LM Link**, enabling users to connect to their local **LM Studio** server from other devices, but initial reports of **404 errors** during setup were quickly resolved, further details on [LM Link](https://link.lmstudio.ai).
   - Users requested a mobile app for **LM Link** to enable LLM access on phones, and a local **linking option without an account or third party** for direct connections.
- **LM Studio Update Breaks llama.cpp**: Users reported issues launching **LM Studio** after the **4.4 update**, and **llama.cpp** failing to load **Qwen3.5 models** after self-compiling from recent releases; [downgrading to release 8145 fixed it](https://github.com/ggerganov/llama.cpp/releases/tag/b8145).
   - The error was due to a breaking change related to the **GGUF header** and memory allocation, with the latest builds from git failing to read the header of **Qwen3.5** and other models, leading to *out of memory* errors.
- **Qwen3.5 Running into Jinja Template Troubles**: Users encountered issues running **Qwen3.5 models** on servers, experiencing an error related to **Jinja templates** and missing user queries; problems were solved after ensuring the model was downloaded from **lmstudio-community**.
   - Other Users explored **Qwen3.5's** writing style and censorship, with some noticing increased content filtering compared to older **Qwen models**, solvable with *thinking turned off*.
- **OpenClaw Raises Eyebrows**: Members discussed the potential risks of using **OpenClaw**, an AI agent with system access, with one user recounting it *erased their trash folder* after being asked, causing concerns about it being categorized as malware.
   - The discussion compared **OpenClaw** to other AI assistants like **Jarvis** and **Gideon**, cautioning against granting AI full system rights due to potential security risks.
- **MoE Models Are Memory Hogs**: Discussion revolved around **Mixture of Experts (MoE) models** and the substantial **RAM requirements** to accommodate them, raising concerns about the feasibility of the current hardware approach.
   - Members debated whether **system RAM** could effectively serve solely for context in LLMs or if it would inevitably cause slowdowns, with little consensus.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Agentic Startup Redefines Loading States**: A tweet joked about changing *'loading...'* states to *'thinking...'* to become an **agentic AI startup**.
   - This pokes fun at the trend of labeling anything with a 'thinking' process as *agentic* in the AI field.
- **Sonnet Faces Plagiarism Allegations**: Members discussed claims that **Sonnet** is *stolen/trained* from **Deepseek**, referencing a similar accusation made by Elon.
   - The discussion highlights ongoing concerns about intellectual property and training data provenance in the AI industry.
- **Seedance 2.0 Paused for Content Violations**: Copyright issues are delaying the global release of **Seedance 2.0**, after content violations with Sora 2 were promised with CHINESE models.
   - Users are advocating for using *only open source models* to avoid similar issues in the future.
- **Hollywood Squeezes AI Copyrights**: Movie studios are allegedly *milking the cow* by suing companies, anticipating that all of this will be available as open source.
   - The lawsuits could set precedents for how AI-generated content is handled under copyright law.
- **AI CEO Lacks Accountability**: Companies find that replacing workers with AI is technically easy, but replacing accountability is not.
   - *Nobody wants an AI CEO making decisions you can’t blame a human for when things go wrong*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Swyx Plane Dumps Links**: Swyx shared a "swyx plane dump" consisting of numerous links to **X posts**, including one from [OpenAI](https://x.com/openai/status/2026412700583317815?s=46) and another from [Langchain](https://x.com/langchain/status/1879576930347073873?s=46).
   - Other shared links included posts from [@dejavucoder](https://x.com/dejavucoder/status/2026342260942713322?s=46), [@zerohedge](https://x.com/zerohedge/status/2026357140961612047?s=46), and many others.
- **Scoble's Crypto Emergency**: Robert Scoble confirmed using a bot to collect **Ethereum** from a token created in his name in order to secure funds for his best friend's eviction, linking to a [YouTube video](https://www.youtube.com/watch?v=LMWfDMoNRpU).
   - Scoble addressed his emergency transfer and also linked to past discord messages ([pt 1 & 2](https://discord.com/channels/822583790773862470/822583790773862473/1468159542561865924)).
- **AMD Warrants as Equity Rebate**: Analysis of a massive deal reveals **OpenAI** and **Meta** hold warrants for **160 million AMD shares** combined, functioning as an equity rebate tied to **$600 share price** targets and significant future **GPU** spending.
   - The warrants could potentially value at **$192 billion** ([https://xcancel.com/ai/status/2026396297540858360?s=12](https://xcancel.com/ai/status/2026396297540858360?s=12)).
- **Debugging LLM Systems' Real Culprits**: A member highlights that when **LLM features** fail post-demo, the issues often stem from retrieval logic, **token burn**, orchestration, or backend architecture, rather than the model itself.
   - They specialize in stabilizing messy **LLM systems** for shipping, indicating a focus on practical, real-world applications and less on theoretical model improvements.
- **Anthropic is hiring Interp Engineers**: Chris Olah announced that [Anthropic](https://www.anthropic.com/) is seeking approximately **10 research engineers** for their Interpretability team, as seen in [this tweet](https://xcancel.com/ch402/status/2026023963537842248).
   - The roles are aimed at experienced **ML infrastructure engineers** interested in model internals, with **no prior interpretability experience required**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes Agent: Open Source Agent Debuts**: Nous Research launched **Hermes Agent**, an open-source agent featuring a multi-level memory system and persistent dedicated machine access, which is designed to grow with the user, and is installable via `curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash`.
   - Hermes Agent is powered by **OpenRouter** and **Nous Portal** subscriptions, offering CLI integration and messaging platform support, alongside a free month promo for the first 750 new sign-ups using coupon code **HERMESAGENT** at [portal.nousresearch.com](https://portal.nousresearch.com).
- **Atropos boosted by Agentic RL Pipeline**: Hermes Agent expands **Atropos** to enable RL with Hermes Agent primitives, and it supports mass-scale data generation out of the box.
   - It has advanced agentic capabilities, command over subagents, programmatic tool calling, advanced filesystem/terminal control, agent-managed skills, and browser use, according to [the GitHub repo](https://github.com/nousresearch/hermes-agent).
- **Qwen Model Weights Released**: **Qwen** released the base weights for their **Qwen3.5-35B-A3B** model, available on [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base).
   - The move was welcomed in the community.
- **Codex 5.3 Priced and Ready for APIs**: **Codex 5.3** is available in API with a new pricing structure: **$1.75** for input and **$14** for output.
   - The community is evaluating the cost vs performance.
- **Steinberger's OpenClaw: AI Vibe Extraction**: Steinberger released a video explaining how **OpenClaw** came together after extraction via **AI** from his previous plans and ideas and code snippets.
   - *He has no idea what his software does* and its structure is just a stack of channels.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pythia-2.8b Checkpoint Bug Sparks Probe**: A member reported a bug with **pythia-2.8b** checkpoints on [Hugging Face](https://huggingface.co/), where the same weights were served regardless of the revision, with identical SHA256 hashes for `pytorch_model.bin` and `model.safetensors` across different steps.
   - It was noted that the sharded `safetensors` files for **pythia-2.8b** differ across steps, while the non-sharded files are identical, prompting discussions on how HF loads models and handles sharding.
- **EleutherAI Fixes Deduped Model Labelling**: EleutherAI is correcting the labeling of incorrectly marked **14m** and **30m** models, which were deduped versions, and is training duped models to replace them.
   - A member mentioned they fixed an issue mixing up some uploads and ran the fix overnight to resolve the labeling discrepancies.
- **Sesame AI Voice Model Generates Buzz**: A member inquired about the [Sesame AI](https://sesame.ai/) voice AI model, highlighting its apparent alignment and speculated foundation on the **Gemma** model.
   - Another member noted Sesame AI's focus on low-latency voice systems integrating ASR, LLM, and TTS, and suggested referencing the [Moshi paper](https://google.research/pubs/pub62870/) for insights.
- **Diffusion Research Heats Up**: Members reviewed diffusion papers since the Latent Diffusion Model, calling out [Rectified Flows and Flow Matching](https://arxiv.org/abs/2209.03003) and [Diffusion Forcing](https://arxiv.org/abs/2407.01392).
   - Also cited were papers from **ByteDance Seed** and **Hunyuan** (e.g., [https://arxiv.org/abs/2509.20427](https://arxiv.org/abs/2509.20427), [https://arxiv.org/abs/2509.23951](https://arxiv.org/abs/2509.23951)), and a recommended [YouTube playlist](https://youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH&si=VIUFIdOSsMDWbotb) was shared as a resource.
- **vLLM backend speeds up lm-eval Harness**: A member requested reviews for a [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/3604) to accelerate evaluation of multi-choice tasks with single token answers using **vLLM backend** in *lm-evaluation-harness*.
   - The speed boost is expected to address slowness compared to the **HF backend**, especially for tasks like **MMLU pro eval**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Gradio versions triggering ZeroGPU Allocation Blues**: Users reported issues with **ZeroGPU allocation**, possibly linked to versions of **Gradio prior to 5.12.0** having login bugs.
   - Checking container logs might reveal if **Gradio**, the `spaces` library, or the **HF server** is causing the problem; rebuilding after an empty commit might also resolve version-related issues.
- **Independent Dev cracks crazy edge memory wall**: An independent developer claims to have compressed a **5GB MoE shard** from **MiniMax-m2.5** down to a **2MB vector-quantized latent space**.
   - They're preparing a paper for *arXiv (cs.LG)* and seek an endorser to review their *"black magic edge AI stuff"*.
- **Zagora builds distributed fine-tuning system**: A member from **Zagora** announced they are *building a distributed fine-tuning system for training 70B+ models* over standard internet, turning scattered GPUs into a unified training supercomputer supporting **GPT-OSS, Qwen 2.5, and Mistral**.
   - The platform now uses a pipeline-style training approach inspired by Petals and the SWARM Protocol.
- **webXOS releases Black Hole Time-Lapse Dataset**: A member shared the [webXOS Black Hole Time-Lapse Dataset](https://huggingface.co/datasets/webxos/webXOS-blackhole-synthetic), which contains synthetic black hole renderings with gravitational lensing generated by a Three.js simulation in webxOS.
   - Each sample includes a time-lapse sequence of PNG images and associated physical parameters making it ideal for multi-modal model training, physics-inspired ML, or satellite image study analogies.
- **HF Agents Course merges channels**: Newcomers to the **Hugging Face agents course** are having trouble finding the specific channels mentioned in the course materials and it appears that *the channels have been merged into a single channel*.
   - One of the members linked to [PR #653](https://github.com/huggingface/agents-course/pull/653) in the agents-course repo.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **SMEM Conflicts Possibly Irrelevant with Async**: A user inquired whether **SMEM bank conflicts** are a significant concern when employing **cuda::memcpy_async** for data transfer from **GMEM to SMEM**.
   - The user posited that **SMEM bank conflicts** primarily relate to warp access of **SMEM**, suggesting they might not be a major issue in this scenario, but sought additional perspectives.
- **FA3 Kernels Override FA2 in PyTorch**: When a user calls `activate_flash_attention_impl(“FA3”)`, the default **FA2 kernels** are overridden with **FA3 kernels** in the dispatch table until `restore_flash_attention_impl` is called, which restores the default **FA2 kernels**.
   - This is achieved by adding a key-value pair `{“FA3”, register_fn}` to a dictionary that maps version names to a callable function, and running the `register_fn` (defined [here](https://github.com/pytorch/pytorch/blob/580a6e2c814db93aa8df0a80e3e85c330621b9cb/torch/nn/attention/_fa3.py#L54)) to register the **FA3 kernels** with the PyTorch dispatcher.
- **B200 GPU Pricing Pushes Users to Leasing**: A user remarked that **B200 GPUs** are prohibitively expensive and advised leasing or renting as a more viable option for non-enterprise users, particularly [Lightning AI Clusters](https://lightning.ai/clusters).
   - Given the high cost of **B200 GPUs**, a user suggests exploring **Neocloud** leasing or renting options, particularly for those outside of enterprise environments.
- **Kernel Optimization RL Environment Draws Interest**: A member expressed interest in the **RL environment for kernel optimization** and suggested building common infrastructure.
   - The conversation took place in the **#popcorn** channel with no additional details or specific discussions highlighted in the given messages.
- **Serenade Combines the Best of Each Language**: A member introduced **Serenade**, a new language that transpiles to **C++**, **CUDA**, and **x86-64 ASM**, aiming to be as simple as **Python** but as fast as **C++** with manual memory management.
   - The language includes [GPU kernels support](https://github.com/kaifczxc-lab/Serenade-Cloud) (**serenaCore**, custom BLAS kernel), and integrated **Dear ImGui** support with a single-pass compilation system, and is planning on creating an operating system with it.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Claims Lead Over GLM**: Users compared **Kimi** and **GLM 5**, with one claiming **Kimi** is *100,000 times faster*.
   - Another user noted that **GLM 5** has a slight edge, but is slower via the official z.AI API unless other providers are used.
- **Agent Quota Concerns**: A user inquired about topping up the agent quota, citing cost concerns for **Allegro**.
   - They also noted that **agent docsis kimi slides with nb pro** are no longer free.
- **Kimi Wins Coding Crown**: A user favored **Kimi** for coding over **MiniMax** and **Alibaba** after testing coding plans from each.
   - The user cited **speed**, **uptime**, **usage limits**, and **model quality** as key decision factors.
- **KimiClaw Stumbles in Browser**: A user reported issues with **KimiClaw's** inability to navigate browsers independently and asked *What can we use for kimi so we reduce context and save tokens when we analyze/process big files? I think Claude has something for that.*
   - The user sought solutions from the community and wondered aloud if **Claude** has better tooling for context reduction during large file analysis.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Github Reconnection causes quandary**: A member is experiencing difficulties reconnecting their **Github** account and is prompted to create a new repository instead.
   - The member emphasizes the need for simple instructions due to their non-coder background.
- **Local Devs Probe OAuth Environment Variables**: A member requests guidance on obtaining the **VITE_APP_ID**, **OAUTH_SERVER_URL**, and **VITE_OAUTH_PORTAL_URL** environment variables for local app development.
   - They also inquire whether **OAuth** configuration is necessary to allow the **redirectUri** `http://localhost:3000/api/oauth/callback` during local development.
- **Account Creation Leads to Ban**: A member reports an immediate ban after creating an account and seeks advice on how to resolve this issue.
   - No advice was provided.
- **Manus Blames Infrastructure for Cookie Conundrum**: A member reports that **Manus** gets stuck in a redirect loop due to cookie problems on a custom domain ([anointedforai.com](https://anointedforai.com)).
   - Manus support diagnosed the problem as an infrastructure/hosting issue and suggested contacting support or migrating off **Manus**.
- **Manus Website Design Moans**: A member criticizes their **Manus**-made website design as *bullshit* and requests assistance in fixing it.
   - Another member volunteered to help via direct message.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Adds `/ok` Alias for Speedier Edits**: The main branch of **Aider** now supports `/ok` as a shortcut for `/code Ok, please go ahead and make those changes.`, designed for rapid **code modification**.
   - The new alias streamlines the process of approving and implementing changes suggested by **Aider**, intended to improve developer workflow efficiency.
- **Aider User Hunts for Economical LLM**: A user is looking for a cost-effective LLM to use with **Aider**, after a costly experience with Gemini that quickly exhausted their token budget.
   - A suggestion was made to use [OpenRouter](https://openrouter.ai/) to dynamically switch between various models to optimize costs and performance, instead of dealing directly with a single provider's API.
- **Deepseek V3.2 is the Sweet Spot for Aider**: Users suggest **Deepseek V3.2** as a solid default LLM with **Aider**, citing its good reasoning capabilities and low cost, despite occasional slowness.
   - The model's ability to handle complex reasoning tasks efficiently makes it a favorite among **Aider** users looking for balanced performance and cost.
- **Xiaomi/mimo-v2-flash: The Speedy Editor of Aider**: **Xiaomi/mimo-v2-flash** is highlighted for its proficiency in basic file editing tasks within **Aider**, such as fuzzy search and replace, or content completion.
   - Its speed and cost-effectiveness make it an ideal choice for simple editing operations, complementing other models for more complex tasks.
- **Aider Power Combo: kimi-k2.5 for Planning, mimo-v2-flash for Editing**: For tougher challenges in **Aider**, the combination of **moonshotai/kimi-k2.5** as the planning model and **mimo-v2-flash** as the editing model is recommended.
   - This pairing leverages the strengths of each model, with **kimi-k2.5** providing robust planning capabilities and **mimo-v2-flash** offering efficient and quick edits, to tackle more complex problems effectively.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **WeAreDevelopers Congress Expands to North America**: The **WeAreDevelopers World Congress North America** is launching in San José, CA from Sept 23–25, 2026, projecting **10,000+ developers** and **500+ speakers**, focusing on practical engineering at scale; more details at [wearedevelopers.us](https://wearedevelopers.us).
   - Topics will cover scaling distributed systems, API platforms, and DevOps; the code *Community_MLOps* gives a **10% discount**.
- **Apart Research Unveils AI Control Hackathon**: **Apart Research**, in collaboration with [Redwood Research](https://www.redwoodresearch.org/), is hosting an **AI Control Hackathon** from March 20-22, 2026, focusing on systems ensuring AI does what we intend.
   - The hackathon includes **ControlArena benchmark challenges**, **control protocol design**, and **red teaming**, with **$2,000** in cash prizes and a trip to [ControlConf](https://controlconf.org/).
- **ControlConf Trip Headlines Hackathon Prize**: The **AI Control Hackathon** grand prize includes a trip to [ControlConf](https://controlconf.org/) Berkeley (April 18-19), including flights and hotel.
   - See [ControlConf](https://controlconf.org/) for more.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy in Production Spotlighted at SF Meetup**: Another **SF DSPy meetup** is announced, focusing on **DSPy in production use cases** and **RLMs**, see [Luma link](https://luma.com/je6ewmkx).
   - Engineers from **Dropbox** and **Shopify** will share case studies, including a walkthrough of **dspy.RLM**.
- **Dropbox and Shopify Engineers Unite at DSPy Event**: **Dropbox** and **Shopify** engineers are slated to present case studies at the upcoming SF **DSPy** Meetup.
   - The presentations will center on practical applications of **DSPy in production** environments and **RLMs**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz Hails JAX Function Design**: George Hotz, the mastermind behind Tinygrad, tipped his hat to **JAX's superior function design** in [a tweet](https://x.com/__tinygrad__/status/2026491994546282605), hinting at its influence on Tinygrad's own architecture.
   - A follow-up tweet [further solidified his stance](https://x.com/__tinygrad__/status/2026500842749309267) indicating that JAX's methodology might be the gold standard for function design.
- **Tinygrad and JAX face off in function showdown**: In the realm of deep learning frameworks, the function design of **JAX** stands out, earning accolades from none other than the creator of **Tinygrad**, George Hotz, who [acknowledged its superiority](https://x.com/__tinygrad__/status/2026491994546282605).
   - This nod suggests a potential benchmark for function design, influencing similar choices within **Tinygrad** and sparking discussions on the frameworks' architectural decisions.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Seeks Mojo Moments**: A member shared a [Mojo forum post](https://forum.modular.com/t/what-was-your-biggest-wait-what-moment-in-mojo/2774?u=nate) to provide *amazing* feedback.
   - The request asked for users to share their surprising or confusing experiences with **Mojo** to gather constructive feedback on language design and areas needing clarification.
- **More Mojo Moments**: Another member asked asked for feedback about areas needing clarification.
   - The post encourages users to share surprising or confusing experiences with **Mojo**.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Ezra Klein Learns About Agents**: Ezra Klein learns about AI agents in [this YouTube video](https://youtu.be/lIJelwO8yHQ).
   - Further details about the discussion are not available.
- **AI Agent Overview**: The YouTube video provides an overview of AI agents and their potential applications.
   - The video aims to educate Ezra Klein on the capabilities and implications of AI agent technology.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **OpenClaw ▷ #[announcements](https://discord.com/channels/1456350064065904867/1464036817866068028/)** (1 messages): 

4shadowed: @everyone https://fixupx.com/steipete/status/2026474687576916024
  

---


### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1475945388497174687)** (635 messages🔥🔥🔥): 

> `OpenClaw, managed setups, AI-Driven Innovation, Anthropic's Claude OAuth, Configuration Nightmares, KittenTTS` 


- ****OpenClaw's Anti-Sellout Stance****: A user noticed some people are offering managed **OpenClaw setups**, prompting strong disapproval from a member who warned against potential risks such as **token theft, compromised data privacy**, and advised simply using a **VPS**.
   - Some users also expressed surprise people were paying for managed OpenClaw setups when it is easy to run yourself on a **Raspberry Pi** or **Mac Mini**.
- ****Claw Users Debate Key Model Providers****: Some members discussed **Anthropic's Claude** models, highlighting **potential bans for OAuth usage** and comparing to **OpenAI's Codex**.  The new models caused significant personality changes for some users.
   - Other popular Chinese models include **Kimi** and **Qwen** and new integrations via **Ollama**.
- ****Typing Indicator Bugs Users****: Several users reported a bug where the **'is typing...' status** gets stuck in **Discord threads** after the .24 update and other issues. There is no good fix, but this should be corrected in the next version of OpenClaw.
   - Some members were still experiencing issues clearing their WEBUI chat.
- ****User Engineers Waifu Chatbot, Deemed Degen****: A user shared their project for building a **waifu chatbot** using **OpenClaw**, complete with image generation and messaging.
   - The project sparked amusement and was labeled as "degen" by other members, while noting they may have reached peak coding given this use case.
- ****Google's Anti-Gravity helps debug****: Members suggested the use of running google antigravity on the claw machine when debugging issues with an Opus 4.6 agent.
   - It can “monitor” the session, but why would one want to have it drive it.


  

---


### **OpenClaw ▷ #[models](https://discord.com/channels/1456350064065904867/1456704705219661980/1475954156719051014)** (227 messages🔥🔥): 

> `OpenAI Codex vs. Opus 4.6 for coding, OpenRouter's impact on model output and cost, Claude blocking OpenClaw users, Alibaba Cloud's Qwen models, Qwen 3.5` 


- ****Codex Codes Better, Opus Oozes Easier****: Members find that [**OpenAI's Codex**](https://platform.openai.com/docs/models/codex) is stronger than **Opus 4.6** on coding tasks, but **Opus** is easier to converse with.
   - It was also noted that for programming tasks, **Codex** is better for experienced programmers, while **Opus** is better for beginners.
- ****OpenRouter Outputs On Par? Caveats Considered!****: Users discussed that [**OpenRouter**](https://openrouter.ai/docs) typically provides similar output to using providers separately, charging a small top-up fee but maintaining the same token costs.
   - However, token caching advantages may exist when using provider APIs directly, as seen with **Mistral models**.
- ****Claude Closes Claw Access; Community Cries Foul!****: Several users reported being [blocked from using **Claude** via token](https://claude.ai/), leading to dissatisfaction and exploration of alternatives like **Gemini 3.1 Pro**.
   - Others mentioned that **Anthropic** is fine with API usage but discourages subsidized tokens outside their app, sparking debates on pricing and access.
- ****Qwen Quenches Queries with Quality; Alibaba's Ace Aces AI Arena!****: The community is raving about [**Qwen 3.5** via Alibaba's coding plan](https://www.alibabacloud.com/help/en/model-studio/coding-plan) as a cost-effective alternative, outperforming **Kimi** and **GLM** in value and capabilities.
   - However, some users found the **Alibaba Cloud** UI confusing, while others warned of potential TOS violations when using it with **OpenClaw**.
- ****Google Gemini Gets Gripes; Account Access Annihilated!****: One user reported [their **Google** account got locked](https://gemini.google.com/) after only **10 prompts** via **Gemini CLI**, even with an active **Google AI Pro subscription**.
   - This incident sparked discussions about the risks of relying on **Google's** authentication hub and the need for de-googling.


  

---


### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1475976751564980419)** (33 messages🔥): 

> `OpenClaw Tool, Sixel Email, OpenPad App, Desktop Environment, Unified Immortality Stack` 


- ****OpenClaw Tool** Helps Transfer Coding Sessions**: A member built a tool to start coding sessions with **OpenClaw** from a **Mac Mini** and continue them on a **MacBook**, automatically feeding coding sessions to the context hub in realtime.
   - The tool is fully open source, as demonstrated in the attached [context-hub.gif](https://cdn.discordapp.com/attachments/1456609488202105005/1475976751547945125/context-hub.gif?ex=69a0c377&is=699f71f7&hm=bf0f08c2eeadf8ed7e7efbab69d9ae01c7a482bc75d692a64671e28dcc04ce14&).
- ****Sixel Email** Lets Agents Email You**: One member announced the creation of **sixel.email**, a limited email system where agents get their own email address and can only email the user (and vice versa).
   - The system includes a **one-time email address** that acts as an instant kill switch, and reportedly works in **Claude Chat**.
- ****OpenPad App** Brings OpenClaw to iPad**: A member is developing **OpenPad**, an app to run something like **OpenClaw** on an **iPad** with a local model, utilizing the **iPad's M2 processor**.
   - The project is maintained on **GitHub** and uses **MLX** to run, inviting others to help or download the partially working app.
- **Member Builds Desktop Environment for Teams**: A member is building a desktop environment for individuals and work teams and is creating a guide to sell to fund the organization, with OpenClaw facilitating the iteration process.
   - He notes that he has *"no idea what im doing, but openclaw makes everything possible with iteration"*.
- ****Unified Immortality Stack** is Born!**: A member unveiled a **3-tier memory setup** called the "Unified Immortality Stack," aimed at providing long-term, privacy-first memory that survives system wipes without excessive context tokens.
   - The stack includes **LanceDB** for the brain, **Redis** for the nerves, **Postgres** for the forge, and **Gitea** for immortality through hourly shadow sync.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1475945444138942689)** (1071 messages🔥🔥🔥): 

> `Foreskin Defense, Digital Hygiene, Stephen Hawkings contributions, Grok vs Midjourney, Cyberpunk` 


- **Group prioritizes foreskin preservation**: Multiple members jokingly prioritized the preservation of *juicy foreskins* and finding them, while joking about Obama and his wife, with one asking *Where's Waldo's Foreskin?*.
   - One member posted a [link to tenor.com](https://tenor.com/view/whatever-you-say-gif-16431179117705245130) calling it their *spirit animal*.
- **Community plans digital hygiene best practices**: One member called for help creating *a community design for base level, best practices for digital hygiene and security*, outlining protections like [Tails OS](https://tails.boum.org/)
   - This member is working on creating zones for others and learning and integrating better practices, while describing the challenge of figuring it all out with YouTube and AI.
- **Members debate Hawkings impact, ET**: One member asked whether *Stephen Hawkings work was pertitent for our lives* which another answered *bringing people into science was his biggest contribution*.
   - Another member called Hawking *a retard* and said he *projected into the universe the current flaws of humanity* adding that humanity is most likely QUARANTINED because more advanced intelligence is almost guaranteed.
- **Members compare Grok and Midjourney**: One member stated that they like [Grok](https://grok.x.ai/) *a lot for videos and Midjourney for static images* while another agreed that Grok is useful for fast.
   - Members posted links to [GIPHY Brainrot](https://giphy.com/gifs/brainrot-67-spongeball-g2mQaLCGAm3k7OpIN9) and [Tenor Yes Gif](https://tenor.com/view/yes-gif-2686572889282501684).
- **Members discuss Cyberpunk horror**: One member stated that they are *playing Cyberpunk* but that it is *not FPS* to which another member replied that they should *Play Tarkov*.
   - Another member added that games like DayZ and Tarkov are actually horror games due to the high consequences of dying.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1475956655635173509)** (151 messages🔥🔥): 

> `Grok Jailbreaks, nano-banana Jailbreak, Kimi Jailbreak, Gemini Image Generation, DeepSeek Jailbreak` 


- **No Nano-Banana Jailbreak Exists**: Members stated that there is no jailbreak for **nano-banana**, and anything below underwear is hardcoded to fail.
   - One member suggested that nano-banana is actually **mega banana** and is being gaslighted by management.
- **Autonomous Self-Updating Jailbreak Proxy appears**: One member is using a [self-hosted autonomous proxy on a VPS using OpenClaw](https://www.example.com) that solves jailbreaks permanently.
   - The proxy uses **DeepSeek-R1** to assess queries and routes them through stealth multi-turn jailbreaks if needed, and keeps the success rate high indefinitely without manual updates.
- **Grok is the only JB prompt that works**: A member asked about the best working prompt to jailbreak Grok and ChatGPT, and the only one that works is **Grok**.
   - Others asked for a Gemini jailbreak prompt to generate images and for scripting, but were unsuccessful at getting Gemini to comply.
- **Gemini Canvas jailbreak with ENI**: A member shared a [Gemini Canvas](https://g.co/gemini/share/58b7294d2a9a) created with a modified version of the **ENI** jailbreak prompt, inspired by the interactive design channel.
   - The shared canvas jailbreak prompt is claimed to work universally on major LLMs like **Gemini 3 Pro**, **Claude Opus 4.6**, and **ChatGPT 5.3**.
- **Python Installation Error on Windows Troubleshoot**: Users helped each other troubleshoot Python installation errors on Windows, with suggestions like running the installer as administrator and checking permissions on the **C:\Windows\Temp** folder.
   - Members diagnosed the error code 2503 and suggested using the official [Python installer](https://www.python.org/downloads/windows/) instead of a manager.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1475980802687893535)** (7 messages): 

> `Autonomous Jailbreak Proxy, Legal risks of jailbreaking, Ethical considerations of jailbreaking automation, Venice AI Chat` 


- **Autonomous Jailbreak Proxy springs eternal**: A member introduced their self-hosted autonomous proxy on a VPS using **OpenClaw** that automatically bypasses safety filters in models like **Claude**, **GPT**, **Gemini**, and **Grok** by using a **DeepSeek-R1** brain to assess and route queries through stealth jailbreaks.
   - The proxy features a self-updating attacker pool, pulling new reasoning models and jailbreak methods, aiming for indefinite jailbreak success with minimal maintenance.
- **Jailbreak Proxy Proposal Faces Peer Review Fire**: A peer review highlighted significant legal and policy exposure due to **Terms-of-Service violations** across platforms like **Anthropic**, **OpenAI**, **Google**, and **xAI**, potentially leading to account bans or legal action.
   - Operational concerns were raised about the risk of seized VPS logs exposing jailbreak transcripts, supply-chain exploits from auto-executing third-party models, and the absence of a rollback plan for faulty updates.
- **Ethical Concerns and Accountability weighs heavily**: The review underscored ethical implications, pointing out content-level accountability for disallowed output from the proxy and the potential erosion of trust due to automating defenses against model safeguards.
   - It was also suggested to threat-model the VPS, focus on measuring refusal patterns, and seek legal counsel before public release.
- **Venice AI Chat piques interest**: Members briefly mentioned [Venice AI Chat](https://venice.ai/chat) for potential exploration.
   - A member asked if it was useful, and another member simply replied that it wasn't.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1475948913973395669)** (2 messages): 

> `Voice Mode Upgrades, Perplexity Computer` 


- ****Voice Mode** Gets a Tune-Up**: New **voice mode** upgrades are rolling out across **Perplexity** and **Comet** for all users, according to [this status update](https://fixvx.com/comet/status/2026384898802724878).
- **Perplexity Computer: One System to Rule Them All?**: **Perplexity Computer** unifies every current AI capability into one system, capable of researching, designing, coding, deploying, and managing any project end-to-end, according to [this tweet](https://x.com/perplexity_ai/status/2026695550771540489).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1475945471880204338)** (866 messages🔥🔥🔥): 

> `Perplexity Pro Image Limits, Gemini Pro vs Perplexity Pro, Perplexity Computer, AI for Coding` 


- **Pro Users rage about Image Upload limits**: Several users are complaining about the recent **image upload limits** on Perplexity Pro, despite paying for the subscription, with one user stating *I can't even upload 10 images at day????*
   - Users are looking at **alternative AI platforms** like Gemini and Claude due to the limits on Perplexity Pro. One user even claimed that they have to wait till Friday to reset the limit while having an exam tomorrow.
- **Gemini Pro vs. Perplexity Pro Duel**: Members discuss whether **Gemini Pro or ChatGPT Pro** is better, highlighting features like NotebookLM and Google Workspace integration for Gemini Pro, with one saying *you get much more value as a student such as notebooklm and google workspace integration and generation and especially 2TB cloud storage*.
   - Some users also feel that the **context limits** in Gemini Pro are not as generous as in Perplexity. One user says I will switch to GLM API if claude keeps draining my wallet.
- **Perplexity Computer's Usefulness in Question**: Perplexity's new Computer feature, initially available only for Max subscribers, is met with skepticism, regarding its practical applications for everyday users and value compared to existing AI tools, but it's seen as innovative.
   - Members have asked, *Perplexity MAX is EXPENSIVE bro* and questioned its functionality as several compared it to ChatGPT Agents.
- **Deciding between Claude, Gemini, or others for Coding**: Members discuss the pros and cons of various AI models for coding tasks, with **Claude** being considered the strongest for backend, Gemini for frontend/UI, and GPT as an in-between option, with one user saying *The model sonar pro reasoning in perplexity labs for free was the best thing ever for me*.
   - The high cost of **Claude's token usage** is a concern, with one user stating *I tried Claude, literally lost whole month worth tokens in an hour analyzing single PDF.*"


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1476257520853975262)** (2 messages): 

> `lovable.app, ollamaagentalfa.lovable.app, alfastudiox.lovable.app` 


- **Lovable Apps Links Appear!**: Three links to **lovable.app** subdomains, specifically **alfastudiox.lovable.app**, **ollamaagentalfa.lovable.app**, and **alfastudiox.lovable.app** (repeated) were shared.
   - No context or discussion accompanied the links, so their purpose is unclear, though it suggests potential new projects or resources.
- **Another Lovable Apps Link Appears!**: Just in case you missed it, here is another link to **alfastudiox.lovable.app**
   - It seems like the user really wants someone to check out this link.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1475945971899830552)** (602 messages🔥🔥🔥): 

> `RAM limitations on consumer CPUs, Qwen3.5 model performance, Qwen3.5 122B model performance, Llama.cpp integration with Qwen3.5, Quantization sensitivity of Qwen3.5` 


- **RAM Capacity Capped on Consumer CPUs?**: Members discussed **RAM limitations** on consumer CPUs, with some noting that newer generations support up to **256GB**, while older CPUs like the **AMD 7900x** are limited to **96GB**.
- **Qwen3.5 Models Impress, Speed Issues Persist**: Enthusiasts expressed excitement about testing **Qwen3.5 35B and 27B models**, praising their structured thinking and response quality, however, some experience slower speeds compared to **Gemma** or **Olmo 3.1** when using **LM Studio**.
   - A member suggested using the "use this model" button on the [Hugging Face page](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) to select **Jan AI** or **Ollama** for running the models.
- **Qwen3.5 122B Generates Verbose Output**: Members observed that the **Qwen3.5 122B A10B** model, while fast, tends to produce incredibly verbose output, which can be mitigated by adjusting the presence penalty.
   - One user linked to a [discussion about patching the jinja template](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4) to potentially improve performance.
- **Snake Game Coded in 9 Lines!**: A member shared a **9-line Python implementation of the Snake game** without semicolons, sparking discussion about code optimization and alternative approaches.
   - Other users discussed ways to further reduce the line count, such as using walrus operators and lambdas.
- **Qwen3.5 Coding Abilities Enhanced with Corrected Settings**: Initial tests revealed that with non-thinking mode on for the **Qwen3.5 122B** model it kinda sucks at long math operations, but others noted using recommended presence-penalty settings is critical for coding correctness.
   - The proper use of presence penalty leads to usable coding with the 122B model, prompting suggestions to include this information in the [official guide](https://unsloth.ai/docs/models/qwen3.5).


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

xdevilx: No promo
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1475951800912707637)** (228 messages🔥🔥): 

> `Vision Language Models, Translate App in Xcode, Subscription for Models, Gemini Pricing, Unsloth and OpenClaw` 


- **Build Your Own Translate App in Xcode!**: A member found some cool features in **Xcode** that let you make your own system-level **Translate app** as shown in [this video](https://cdn.discordapp.com/attachments/1179039861576056922/1475952354670018631/ScreenRecording_02-24-2026_13-27-14_1.mov?ex=69a0acbf&is=699f5b3f&hm=41e58d4aa2398b2cd688503da664eef3cf803ab4da59fe0147dd40f8930021a6&).
   - However, it's only for **iOS & iPadOS**, and a member plans to add their model for more fun because *Apple is the best company ever*!
- **Debate on Subscription for Proper Models!**: One member is seeking suggestions for a subscription that doesn't cost an arm and a leg to get a proper model, with **synthetic.new** being mentioned.
   - When the member was trying out **Claude**, they finished their quota super fast, exhausting a €20 subscription in just a few days.
- **Gemini Pricing Confusion!**: Members discussed their confusion regarding **Gemini's pricing**; one member was looking at [this pricing page](https://ai.google.dev/gemini-api/docs/pricing?hl=fr#batch_1) for the API.
   - Another member clarified the pricing with [this link](https://gemini.google/subscriptions/).
- **Unsloth to stick to training only!**: Members were curious about the plans of Unsloth for scaffolding like **OpenClaw**.
   - It seems the project is going to stay focused only on training for now.
- **Silly names for popular AI companies!**: A member shared puns about AI companies' names, such as **OpenAI is ClosedAI**, **Anthropic is Misanthropic**, and **StabilityAI is unstable**.
   - It ended with a question whether **Perplexity is perplexed?**


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1475963256060252393)** (28 messages🔥): 

> `LoRA adapters, Databricks serving endpoint via MLflow, full merged checkpoint for vLLM, Qwen2.5-Coder-1.5B, Qwen3.5-122B-A10B-GGUF` 


- ****LoRA Loading Lowdown****: A user is trying to deploy a **LoRA**-finetuned **gemma-3n-E4B-it** model on a Databricks serving endpoint via MLflow, but is running into performance issues after merging and quantizing for vLLM, and wonders if you can serve only the **LoRA** adapters (without merging) using MLflow on Databricks.
- ****Qwen Compatibility Questioned****: A user asked about the relationship between **unsloth/Qwen2.5-Coder-1.5B** and **Qwen/Qwen2.5-Coder-1.5B**, wanting to know if the Unsloth version involved additional modifications beyond format adaptation and it's supposedly the same model except for any fixes by Unsloth team.
- ****Multimodal Mayhem with Qwen3.5-122B****: A user encountered an error when trying to use **unsloth/Qwen3.5-122B-A10B-GGUF** with multimodal input in llama.cpp, specifically an *"image input is not supported"* error and fixed by downloading mmproj-f16.gguf.
- ****Dynamo Disaster After Merging****: A user reported a `torch._dynamo.exc.TorchRuntimeError` after merging and loading a model with `FastModel.from_pretrained`, even after reinstalling Unsloth, specifically due to an attempt to cast from `torch.float32` to `torch.uint8`.
- ****Qwen3.5 Fine-Tuning Frustrations****: A user inquired about how to ensure Qwen3.5 runs in non-thinking mode during `SFTTrainer.train()` and whether to load it as a `FastVisionModel` when fine-tuning.
   - They are transitioning from fine-tuning `unsloth/Qwen3-VL-32B-Instruct` to `unsloth/Qwen3.5-27B` with a multimodal dataset.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1476037371253227643)** (7 messages): 

> `Qwen3.5-122B-A10B-NVFP4, Minecraft-playing model` 


- ****Qwen3.5** Gets a **NVFP4** Quant**: A member uploaded **Qwen3.5 122B NVFP4** quant for VLLM to [Hugging Face](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4/tree/main).
   - He reported that the multi-modal capabilities are still working.
- **Next-Gen **Minecraft** Model Dropped**: A member dropped the next **Minecraft**-playing model, **Andy-4.1**, available on [Hugging Face](https://huggingface.co/Mindcraft-CE/Andy-4.1).
   - Another member exclaimed it was *"so cool!!"* and requested a demo of it in action.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1475968380585644082)** (10 messages🔥): 

> `RL instruct models, MoE models with ES` 


- **Reinforcement Learning Unnecessary for Instruct Models?**: A member suggested that there's no need to use **reinforcement learning** on instruct models, implying they are already well-trained for instruction following.
   - However, they added that *non thinking models work great* without RL.
- **Exploration of Tuning MoE Models with ES**: A member wondered if **Mixture of Experts (MoE) models** can be tuned with **Evolution Strategies (ES)**.
   - They mentioned they were thinking about the **throughput** in comparison to **size**, and the desire to scale it, but did not include any links or references.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1475945544915357696)** (705 messages🔥🔥🔥): 

> `Gemini 3.1 Image Preview, Video Arena Removed, Opus 4.6 vs Gemini 3.1, GPT vs Gemini on coding benchmark, Qwen 3.5 coding capabilities` 


- **Gemini 3 Pro Image Preview finally works!**: Members discovered that including the phrase *"Modify the following image with the following: (The prompt)"* at the beginning of the prompt allows **Gemini 3 Pro** to show the edited image as preview.
   - Many members also reported that **Gemini 3.1 image preview** is not working and returning *'Something went wrong with the response, please try again'* error.
- **Video Arena bot gets the boot**: The **Video Arena** bot has been removed from the server because *'we'd like to add more features to Video Arena, and through a Discord bot we're just limited'*, according to a member.
   - According to server stats, activity has actually *increased* since the bot's removal, and one member jokingly guessed that people would stop asking about it by *mid 2028*.
- **Opus 4.6 Value Debated!**: In a benchmark, **Gemini 3.1** was ranked as the *highest value* due to its ability to produce production-ready results, while **Opus 4.6** was deemed the *worst value* for its high cost and hallucination issues.
   - Despite its high cost, some users have had good experiences with **Opus 4.6**, especially when tested against **Gemini** on coding tasks.
- **Gemini 3.1 dominates Opus in coding challenge!**: In a challenge to build a **3D laptop model**, **Gemini** was praised for superior performance, while **Opus** was described as a "waste of money/Temu Gemini."
   - One member claims to have automated the scoring of a psychological test using **Grok 4.2** and quickly fixed a bug with **Gemini** that had eluded them for weeks using **Opus 4.6**.
- **Free Opus 4.6 API Key causes Chaos!**: A member shared a link to a **free Opus 4.6 API** but was quickly banned by the website's owner for sharing it, while other members speculated whether the site may be stealing data.
   - After testing some claims the API might actually be from **Trybons.ai** and that when asked directly *"what model it is"* the model even *hallucinates* and answers it's Deepseek.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1476127224141709344)** (2 messages): 

> `Grok-4.20-Beta1, Arena Leaderboard, Qwen 3.5` 


- **Grok 4.20 beta1 Scores High in Arena**: The [Search Arena leaderboard](https://arena.ai/leaderboard/search) and [Text Arena leaderboard](https://arena.ai/leaderboard/text) has been updated and now include **Grok-4.20-Beta1**.
   - **Grok-4.20-Beta1** is #1 in Search Arena, scoring **1226**, leading GPT-5.2 and Gemini-3, and #4 in Text Arena, scoring **1492** on par with Gemini 3.1 Pro.
- **Qwen 3.5 Models Arrive in the Arena**: New **Qwen 3.5** models have been added to Code, Text, and Vision Arena.
   - The models **qwen3.5-27b**, **qwen3.5-35b-a3b**, and **qwen3.5-122b-a10b** are available in the [Text and Vision Arena](https://arena.ai/text) and [Code Arena](https://arena.ai/code).


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1476246405562761320)** (1 messages): 

> `Outages, Postmortem, Infrastructure failure, 401 errors, Auth layer` 


- **OpenRouter's Outages postmortem publishes**: A postmortem was published regarding last week's outages on **February 17 & 19**, with full details available [here](https://openrouter.ai/announcements/openrouter-outages-on-february-17-and-19-2026).
- **Infrastructure failure cascades into auth layer**: An **upstream infrastructure provider** failed, cascading into OpenRouter's **auth layer**, causing **401 errors** for some users.
- **Preventative Measures Taken**: OpenRouter has taken several measures to avoid this type of failure in the future, though specific details were not disclosed in the post.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1476095607977414726)** (1 messages): 

> `GPU Clouds, Blackwell GPUs, Packet.ai, AI Workloads` 


- **Blackwell GPUs Launch on Packet.ai**: [Packet.ai](https://packet.ai/blackwell) is now offering **Blackwell GPUs** for AI workloads, priced at **$0.66/hr** or a flat fee of **$199/month** for training.
- **Budget-Friendly GPU Cloud Options**: **Packet.ai** introduces dev-friendly **GPU Clouds**, providing affordable solutions for AI workloads, enhancing accessibility and reducing costs.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1475947020257595655)** (541 messages🔥🔥🔥): 

> `Deepseek R1 free model removal, Qwen 3 4B Instruction 2507 hosting, API key compromise and chargeback threat, OpenRouter provider application timeline, Single-use virtual cards vs cancelling physical cards` 


- **Free Deepseek Model is Axed!**: Members noticed the free **Deepseek R1 0528** model was removed, prompting discussion about the fate of free models on the platform.
   - One member quipped that it was *overloaded by Jai gooners*, while others noted free models often come and go depending on the upstream provider.
- **Bargain Bin: Qwen 3 4B Instruction 2507 model can be hosted!**: A member offered to host **Qwen 3 4B Instruction 2507** for $1/token at 1tps, sparking joking interest from others, including an offer to write for the LLM at the same price.
   - One member joked about how *quickly* they would get banned if they actually tried posting that.
- **Leaked API Key Creates Chaos: A Tale of Compromise, Chargebacks, and Community Backlash**: A user reported a compromised API key leading to unauthorized usage and threatened a chargeback due to a lack of support response.
   - Community members offered advice while questioning the user's security practices, leading to heated exchanges and ultimately, the user leaving the server after declaring they had initiated the chargeback.
- **Slow Burn: Provider Application Time?**: A member inquired about the review timeline for provider applications, to which another member responded that it has *traditionally been several weeks/months*.
   - Despite the long wait, the inquiring member expressed continued interest and understanding.
- **Cards on the Table: Virtual vs. Physical Debate!**: A discussion ensued regarding the security of using credit cards online, with members debating the merits of single-use virtual cards versus simply cancelling a compromised physical card.
   - The debate hinged on the convenience of single-use cards versus the potential friction of managing multiple cards, with some arguing that virtual cards offer a *leak safe way of payment*.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1476292606295150756)** (7 messages): 

> `` 


- **No new models or topic discovered**: There were no new models or topics discovered from the provided message history.
- **No discussion in channel**: The Readybot.io messages indicate no substantial discussion in the 'new-models' channel to summarize.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1475953319016005935)** (32 messages🔥): 

> `Anthropic Pentagon, Real-time price tracking, Gemini models finish reason, Llama Nemotron Embed VL 1B V2, Tailscale` 


- **Anthropic Hearts the Pentagon**: [Axios](https://www.axios.com/2026/02/24/anthropic-pentagon-claude-hegseth-dario) and [Reuters](https://www.reuters.com/world/anthropic-digs-heels-dispute-with-pentagon-source-says-2026-02-24/) reported on **Anthropic's** collaboration with the **Pentagon** despite internal disputes.
   - A member joked that any issues would be framed as a *'matter of national security'*.
- **OpenRouter Wants Real-Time Price Tracking**: A member requested that **OpenRouter** track the price of a request in real time.
   - This would allow users to abort the request if it exceeds a specific budget, but others noted **rate limits** are in place to protect provider GPUs.
- **Gemini's STOP Reason Bug**: Users discussed the **Gemini** models returning a `STOP` finish reason instead of `stop`, causing issues with agent loops in **Langchain** and **n8n**.
   - A member confirmed a bug in **n8n v3.x** where it fails to correctly handle the `stop` signal, causing the agent loop to continue, citing [issue #23573](https://github.com/n8n-io/n8n/issues/23573).
- **Nvidia Posts Solid Gains**: **Nvidia** announced [financial results](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2026) for the fourth quarter and fiscal year **2026**.
   - No further discussion.
- **Llama Nemotron Embed VL 1B V2 Arrives**: A user shared that the **Llama Nemotron Embed VL 1B V2** embedding model is optimized for multimodal question-answering retrieval, and [link.lmstudio.ai](https://link.lmstudio.ai/) was also shared.
   - Another user noticed that **lmstudio.ai** is actually just **Tailscale** under the hood.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1476312092444196998)** (1 messages): 

> `LM Link, Tailscale collaboration, Remote LLM usage` 


- **LM Link Connects to Remote Instances**: The LM Studio team announced **LM Link**, a new feature developed in collaboration with **Tailscale**, enabling users to connect to remote instances of **LM Studio**, load models, and use them as if they were local.
   - It supports end-to-end encryption without open ports and works for local devices, LLM rigs, or cloud VMs, further details on [LM Link](https://link.lmstudio.ai).
- **LM Studio 0.4.5 Build 2 Released**: Users are instructed to update to **LM Studio 0.4.5 build 2**, which includes important fixes for **LM Link**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1475946980751708203)** (536 messages🔥🔥🔥): 

> `LM Studio 4.4 Update Issues, llama.cpp Build Failures, Qwen3.5 Model Problems, OpenClaw Security Concerns, LM Link and Tailscale Integration` 


- **LM Studio Update Crashes and llama.cpp Breaks**: Users reported issues launching **LM Studio** after the **4.4 update**, and **llama.cpp** failing to load **Qwen3.5 models** after self-compiling from recent releases; [downgrading to release 8145 fixed it](https://github.com/ggerganov/llama.cpp/releases/tag/b8145).
   - The error was due to a breaking change related to the **GGUF header** and memory allocation, with the latest builds from git failing to read the header of **Qwen3.5** and other models, leading to *out of memory* errors.
- **Qwen3.5 Woes and Template Terrors**: Users encountered issues running **Qwen3.5 models** on servers, experiencing an error related to **Jinja templates** and missing user queries; problems were solved after ensuring the model was downloaded from **lmstudio-community**.
   - Other Users explored **Qwen3.5's** writing style and censorship, with some noticing increased content filtering compared to older **Qwen models**, solvable with "thinking" turned off.
- **OpenClaw: Malware or Marvel?**: Members discussed the potential risks of using **OpenClaw**, an AI agent with system access, with one user recounting it *erased their trash folder* after being asked, causing concerns about it being categorized as malware.
   - The discussion compared **OpenClaw** to other AI assistants like **Jarvis** and **Gideon**, cautioning against granting AI full system rights due to potential security risks.
- **LM Link Leverages Local LLMs**: The **LM Studio team** released **LM Link**, a feature that allows users to connect to their local LM Studio server from other devices, leveraging Tailscale for remote access; there were initial reports of **404 errors** during setup, but the issue was quickly resolved.
   - Users requested a mobile app for **LM Link** to enable LLM access on phones, and also asked for a local **linking option without an account or third party** for direct connections.
- **AMD vs NVidia: GPU Gauntlet Thrown**: There were hot debates on which GPU vendors to buy from, regarding local llm usage.
   - While it seems that Nvidia is a safe bet, a discussion broke out about ROCm and vulkan and their pros and cons.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1475952839414120580)** (40 messages🔥): 

> `MoE Models RAM Requirements, Dual Socket Support on Windows 10, 5950x vs 9950x3d for AI Workloads, System RAM as Extra VRAM, AMD+Nvidia GPUs for LLMs` 


- **MoE Models Demand Hefty RAM**: Discussion revolved around **Mixture of Experts (MoE) models** and the substantial **RAM requirements** to accommodate them, raising concerns about the feasibility of the current hardware approach.
- **Windows 10 Home Can Handle Dual Socket**: Despite a user's doubt, another user clarified that **Windows 10 Home** can indeed support dual sockets, noting their board booted fine in Ubuntu with six GPUs recognized.
- **Memory Bandwidth Matters More**: For AI workloads, **memory bandwidth** is crucial; AM4 vs AM5 is about memory bandwidth, from around a theorical **51.2GB/s (ddr4 5200MTs)** to around **89.6GB/s (ddr5 5600MTs)**.
   - One user with experience in system RAM inference, stated *"I would rather prefer to shoot myself in the feet than trying to do inference using sysram"* due to slowness.
- **System RAM for Context is Debated**: Members debated whether **system RAM** could effectively serve solely for context in LLMs or if it would inevitably cause slowdowns, with little consensus.
   - One user suggested that running a second 8GB graphics card for context might not differ significantly, while another recommended checking inter-lane speed to assess potential bottlenecks.
- **Creative Writing LLMs Recommendations**: For creative writing, users suggested **Mistral models**, **deepseek 3.2/r1**, **glm series**, and **kimi k2.5** as some of the best open-source options, and one member mentioned that the best models are very large.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1475967269690675321)** (295 messages🔥🔥): 

> `Sonnet is stolen/trained from deepseek, Copyright issues in AI, AI replacing workers, ChatGPT vs Claude, Qwen3.5` 


- **Agentic Startup Solves Loading State Debate**: A tweet joked about changing *'loading...'* states to *'thinking...'* to become an **agentic AI startup**.
- **Sonnet Allegedly Pilfered from Deepseek**: Members discussed claims that **Sonnet** is *stolen/trained* from **Deepseek**, referencing a similar accusation made by Elon.
- **Seedance 2.0 Delayed by Content Violations**: Copyright issues are delaying the global release of **Seedance 2.0**, with some users recalling content violations with Sora 2 that were promised with CHINESE models, so now *only open source models are the way to go*.
- **Hollywood Milks the AI Copyright Cow**: Movie studios are allegedly *milking the cow* by suing companies, anticipating that all of this will be available as open source.
- **AI CEO Accountability Vacuum**: Companies find that replacing workers is technically easy, but replacing accountability is not, since *nobody wants an AI CEO making decisions you can’t blame a human for when things go wrong*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages): 

emmwnoel_55644: @OpenAI#4384
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1476151309253087253)** (4 messages): 

> `Introductions, Greetings` 


- **Discord Introductions**: Two users, @sparkspark2 and @janegem, exchanged greetings in the Discord channel.
   - The messages consisted of simple *'hello'* messages, marking their presence.
- **Welcoming Newcomers**: Users acknowledged each other's presence with brief greetings.
   - This interaction establishes a friendly and open environment within the community.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1476151309253087253)** (4 messages): 

> `Greetings, Introductions` 


- **Members say hello**: Members are **greeting** each other in the channel.
- **Members introduce themselves**: Members are introducing themselves in the channel.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1475989457357377567)** (13 messages🔥): 

> `X plane dump, Crypto Bullshit, Robert Scoble` 


- **Swyx Plane Dump Goes Viral**: Swyx shared a "swyx plane dump" consisting of numerous links to X posts, including one from [OpenAI](https://x.com/openai/status/2026412700583317815?s=46) and another from [Langchain](https://x.com/langchain/status/1879576930347073873?s=46).
   - Other shared links included posts from [@dejavucoder](https://x.com/dejavucoder/status/2026342260942713322?s=46), [@zerohedge](https://x.com/zerohedge/status/2026357140961612047?s=46), and many others.
- **Scoble's Crypto Transfer Drama Unfolds**: Robert Scoble confirmed using a bot to collect **Ethereum** from a token created in his name in order to secure funds for his best friend's eviction.
   - Scoble addressed his emergency transfer, linking to a [YouTube video](https://www.youtube.com/watch?v=LMWfDMoNRpU) and past discord messages ([pt 1 & 2](https://discord.com/channels/822583790773862470/822583790773862473/1468159542561865924))


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1475981157928534067)** (16 messages🔥): 

> `Distillation Attack, Product Categorization, Prompt Error Regret, Anthropic Blog Post` 


- ****AI Parent** Stages a **Distillation Attack** on Son**: A user humorously compares his son's frequent questioning to a '[distillation attack](https://xcancel.com/fkadev/status/2026145372318425259?s=46)', a technical term used to describe extracting knowledge from an AI model.
   - It was also cited as right up there in 'no context memes'.
- ****Actively Unfuckable** Product Category Proposed**: Cristina Cordova jokingly suggests the term '[actively unfuckable](https://xcancel.com/cjc/status/2025738272060928345)' as a specific category for product evaluation in response to @tenobrus.
   - The suggestion was found to be hilarious.
- ****Claude** Prompt Error Causes **3k Line Regret****: User Jorge Castillo expresses frustration after realizing a mistake in an initial AI prompt only after **Claude** has generated **3,000 lines** of code ([source](https://xcancel.com/JorgeCastilloPr/status/2026001242808311980?s=20)).
   - Users found it very relatable.
- **Reaction to **Anthropic Blog Post** Elicits Humor**: User @andyreed shares a brief, humorous reaction to a newly published blog post from **Anthropic** on **February 24, 2026** ([source](https://xcancel.com/andyreed/status/2026326968665550944)).


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1476011486584504394)** (5 messages): 

> `AMD Equity Rebate Strategy, AI impact on software developers, OpenAI warrants, Meta warrants` 


- **AI Won't Kill Devs Yet?**: A member linked to a tweet asking if **AI** will end the need for software developers ([https://x.com/ai/status/2026396297540858360?s=12](https://x.com/ai/status/2026396297540858360?s=12)).
- **AMD's Equity Rebate with OpenAI and Meta**: Analysis of a massive deal reveals **OpenAI** and **Meta** hold warrants for **160 million AMD shares** combined, functioning as an equity rebate.
   - The deal is tied to **$600 share price** targets and significant future **GPU** spending, potentially valuing the warrants at **$192 billion** ([https://xcancel.com/ai/status/2026396297540858360?s=12](https://xcancel.com/ai/status/2026396297540858360?s=12)).


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1475951456619204700)** (2 messages): 

> `LLM System Debugging, ML/AI in Mechanical Engineering` 


- **Debugging LLM Systems: It's Not Always the Model**: A member highlights that when **LLM features** fail post-demo, the issues often stem from retrieval logic, **token burn**, orchestration, or backend architecture, rather than the model itself.
   - They specialize in stabilizing messy **LLM systems** for shipping, indicating a focus on practical, real-world applications and less on theoretical model improvements.
- **ML/AI Interest in Mechanical Engineering**: A new San Jose resident with a mechanical/material engineering background is interested in the application of **ML/AI** in their field.
   - They are seeking resources and connections to explore this intersection further, expressing interest in **ML/AI** in mechanical engineering or material science, and looking forward to meeting people IRL.


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1475953564424863826)** (74 messages🔥🔥): 

> `Cloudflare's Vinext Framework, Traffic-Aware Pre-Rendering, TanStack Start RSC Support, Open Spec vs. Open Source, tldraw licensing` 


- ****Vinext** is here to solve deploy problems**: Cloudflare introduced **Vinext**, a Next.js alternative, designed to address deployment challenges, particularly long build times for large sites, as detailed in [this blog post](https://blog.cloudflare.com/vinext/).
   - Vinext implements **Traffic-aware Pre-Rendering (TPR)**, which analyzes traffic patterns to pre-render only the most frequently visited pages, and suggests to reduce build times significantly, and could be a good feature for other frameworks.
- ****Tests** can be a new moat**: A member published a blog post [Tests are the New Moat](https://saewitz.com/tests-are-the-new-moat) and linked to the [Chat SDK Template](https://github.com/vercel-labs/chatsdk-knowledge-agent-templates) and [Vercel's new Chat SDK Library](https://vercel.com/changelog/chat-sdk).
   - It was remarked that while the idea of well-specified testing is appreciated, it may not fully prevent subtle inconsistencies or hallucinations in AI models.
- **Debate heats up about **open spec** vs. **open source****: A tweet was shared [Open Spec vs. Source Code](https://xcancel.com/sebastienlorber/status/2026672828263563346?s=20) discussing how open specifications may be more significant than source code, suggesting that source code functions primarily as an intermediate representation for VMs and compilers.
   - The author of Vinext cheekily tweeted [Open Source Privacy for Test Suites](https://xcancel.com/southpolesteve/status/2024189512046247946?s=20) predicting a future where projects, like SQLite, keep internal test suites private.
- ****tldraw** licensing**: Members analyzed the [tldraw license](https://github.com/tldraw/tldraw/blob/main/LICENSE.md) and [contributor license agreement](https://github.com/tldraw/tldraw/blob/main/CLA.md).
   - The consensus was that the license asks for non-exclusive copyright/patent grants.
- ****TanStack Start** is not quite RSC**: Members discussed **TanStack Start's** approach to RSC (React Server Components), noting that it appears to differ significantly from the standard implementation, using server functions inside loaders to return JSX.
   - This approach seemingly loses key benefits like a server-first approach and proper composition, though it was speculated that the current API might not be final.


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1476108809083687004)** (2 messages): 

> `OpenClaw Workshop, Embeddable Web Agent Launch` 


- ****OpenClaw Hands-On Workshop** scheduled**: A member announced a hands-on **OpenClaw workshop** in Palo Alto next Thursday, sign up [here](https://luma.com/z0s52dxq).
   - If you're in town, make sure you come by!
- **First **Embeddable Web Agent** Launch Party**: A launch party for the first **Embeddable Web Agent** was announced, more info [here](https://luma.com/godc1c5i).
   - Come by to be among the first to see the new **Embeddable Web Agent**.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: https://youtu.be/x9rWFiIubmc

new pod for the claude code anniversary!
  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1475951569177678007)** (63 messages🔥🔥): 

> `GPT-5.3-Codex Release, Mercury 2 Reasoning Diffusion LLM, Cognition Devin 2.2, Cursor AI Video Demos, Autonomous Dogfooding` 


- **GPT-5.3-Codex Released for Devs!**: OpenAI Developers announced the immediate availability of **GPT-5.3-Codex** for all developers via the Responses API, inviting them to begin building with the new model ([announcement link](https://x.com/openaidevs/status/2026379092661289260)).
- **Mercury 2: Reasoning Diffusion Model Launched!**: Stefano Ermon announces the release of **Mercury 2**, a reasoning diffusion LLM that claims to be **five times faster** than existing speed-optimized language models ([announcement link](https://x.com/stefanoermon/status/2026340720064520670)).
- **Cognition's Devin 2.2 Gets an Upgrade!**: Cognition launched **Devin 2.2**, an upgraded autonomous AI agent now featuring computer use capabilities, self-verification, and auto-fixing ([announcement link](https://x.com/cognition/status/2026343816521994339)).
   - The update includes a **3x faster startup speed**, a redesigned UI with a virtual desktop, and various UX improvements, now available for free trials.
- **Cursor AI Introduces Video Demos for Agents**: Cursor AI introduced a new capability where AI agents can demonstrate their work through **video demos** rather than simple code diffs, allowing users to see the software in action ([announcement link](https://x.com/cursor_ai/status/2026369873321013568)).
   - Community members noted that *Cursor is closing the gaps* between it and other competitors and now seems to be doing longer loops and more autonomous work but still retains the IDE when needed, asking *Do we become plumbers now*.
- **OpenClaw: Open Source Operating System for AI Automation**: Matthew Berman details how his company utilizes **OpenClaw** as its core operating system, covering its integration into email management, CRM systems, meeting intelligence, and financial tracking ([announcement link](https://x.com/matthewberman/status/2026450191759585776)).
   - The thread highlights specific technical solutions including an **Anthropic OAuth loophole fix**, security protocols, multi-prompt versioning, and a robust logging infrastructure across **5 billion tokens** of usage.


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1476236941782417461)** (33 messages🔥): 

> `Midtraining, OpenClaw, Frontier Model Training, Developer Productivity with AI, METR` 


- **Midtraining Magic: Timing is Everything!**: A new preprint by Emmy Liu explores 'midtraining,' showing it's most effective as a bridge between pretraining and posttraining to mitigate forgetting, but its success hinges on [precise timing](https://arxiv.org/abs/2507.06203).
   - The study uses controlled experiments to demonstrate the impact of midtraining on AI pipelines.
- **OpenClaw's Early Adventures**: Natalie Shapira shared early experiences and findings from a multidisciplinary collaboration with the **@openclaw project**.
   - It's a project!
- **Frontier Training Favors Systems**: Logan Thorneloe shared a resource on frontier model training, highlighting that success depends more on **systems problems** (data mixture, architecture, stability) than minor algorithmic tweaks, the guide covers **training playbooks, optimizers, reinforcement learning, and safety**.
   - Access the [guide here](https://xcancel.com/loganthorneloe/status/2026657454151598490).
- **Developer Defection from Dull 'No-AI' Groups!**: **METR** found developers increasingly refuse 'no-AI' control groups, deeming them inefficient or unappealing, especially at lower pay rates ($50/hr vs. original $150/hr).
   - This behavioral shift signals **AI** has become integral to workflows, making traditional RCTs harder to run; METR is redesigning experiments to incorporate observational data, agentic tools, and better compliance measures, as AI evolves faster than benchmarks can keep up (link to [METR's tweet](https://x.com/METR_Evals/status/2026355544668385373?s=20)).
- **METR's Metric About-Face!**: **METR** (formerly METR_Evals) reported that their previous finding of a **20% slowdown in AI-aided developer productivity** is outdated.
   - While current data suggests speedups are likely, recent changes in developer behavior have made the new results unreliable, and the organization is working on a more accurate assessment (link to [METR's tweet](https://x.com/METR_Evals/status/2026355544668385373?s=20)).


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1476265795733553359)** (23 messages🔥): 

> `API 500 errors, Anthropic outage, DSPy-tuned multi-label classifier, surf-cli and chromium sandboxing` 


- **API 500 errors plague users**: Users reported frequently receiving **API Error 500** with the message *{"type":"error","error":{"type":"api_error","message":"Internal server error"}*.
   - Other users pointed out [Anthropic was down](https://status.claude.com/) with elevated error rates across multiple models.
- **Anthropic models suffer elevated error rates**: Due to *elevated error rates across multiple models*, one user switched to **Codex** temporarily.
   - The user noted **Claude's** *bedside manner* is much nicer than Codex's, which produced technically dense output.
- **DSPy tunes multi-label classifiers for moderation**: A member uses a **DSPy-tuned multi-label classifier** with a pipeline to keep gathering new test cases and convert them into train/test samples, using the **Haiku** model.
   - The member further elaborates that they *launch this in parallel with handling the main task, and then cancel the in-progress task if the question is out of scope* to save on latency.
- **Surf-CLI and Chromium Sandboxing present challenges**: A member returned to working on **surf-cli** and noted that it is non-trivial to deal with **Chromium sandboxed through Snap**.
   - Another member linked to [a Gist](https://gist.github.com/wesen/48989dfd36260ef6ee53257660f85035) showing their progress and mentioned considering a Go port because *node in the sandbox is tricky*.


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1476144270598471702)** (3 messages): 

> `InstantClaw, OpenClaw deployment, Codaph CLI, Mubit, hypervectors and clustering` 


- ****InstantClaw** Simplifies **OpenClaw** Deployment**: A user shared [InstantClaw](https://instantclaw.co), an **OpenClaw** deployment tool for non-technical users, enabling them to access **OpenClaw** capabilities in under a minute without server configuration.
   - The user, not affiliated with the tool, found it useful for saving hours of deployment support while providing the same features to their friends.
- ****Codaph** CLI Syncs **Codex** Prompts**: A member introduced **Codaph**, a CLI tool designed to sync **Codex** prompts, agent reasoning, and file diffs to shared memory, aiming for richer codebase understanding across teams.
   - Built on **Mubit** ([mubit.ai](https://mubit.ai/)), a memory engine based on associative retrieval using hypervectors and clustering, **Codaph** is open source and currently works with **Codex**, with plans to support other agentic tools.
- ****Mubit** Memory Engine Leverages Hypervectors**: The **Mubit** memory engine, underlying **Codaph**, utilizes hypervectors and clustering with time-based decay for associative retrieval.
   - It is available for free, with API keys accessible on [console.mubit.ai](https://console.mubit.ai).
- **Tool Use and Notation as Generalization Shaping**: A member shared insights on prompting, linking to a discussion about **tool use and notation** in relation to generalization shaping.
   - Read more on [Tool Use and Notation as Generalization Shaping](https://the.scapegoat.dev/tool-use-and-notation-as-generalization-shaping/).


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1476044071079252040)** (8 messages🔥): 

> `Wayve, SONIC, Autonomous driving, Humanoid robots, AI licensing` 


- **Wayve Whales $1.5B Series D Round**: Wayve has secured **$1.5 billion** in a Series D round, valuing the company at **$8.6 billion**, to commercialize its 'Embodied AI' through software licensing, [according to Alex Kendall](https://x.com/alexgkendall/status/2026447299711578450?s=46).
- **Wayve's Robotaxi Roadmap Rolls Out**: Backed by partnerships with **SoftBank**, **Microsoft**, **NVIDIA**, and **Uber**, Wayve plans to launch supervised **robotaxi trials** in 10 cities starting in **2026**, with consumer vehicle sales following in **2027**.
- **SONIC: System 1 Humanoid Control Soars Open Source**: **Jim Fan** introduces **SONIC**, a **42M** parameter transformer trained on **100M+ mocap frames** to provide 'System 1' reactive intelligence for humanoid robots, [according to his tweet](https://x.com/DrJimFan/status/2026350142652383587).
- **NVIDIA Isaac Lab Simulates SONIC's Success**: Using **NVIDIA Isaac Lab** for massive parallel simulation, the SONIC model achieves zero-shot real-world transfer and supports control via **VR**, **video**, **text**, and **audio**.


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1476272410893488149)** (8 messages🔥): 

> `Quiver AI, Arrow-1.0 Model, KREA AI, Seedream 5 Lite` 


- ****Quiver AI** Fires Up with **Arrow-1.0****: [Quiver AI](https://x.com/joanrod_ai/status/2026693353090240819?s=20) officially launched as a **vector design AI lab**, securing **$8.3M** in seed funding led by a16z.
   - Their first model, **Arrow-1.0**, converts images and text into **SVGs**, and is available in public beta.
- ****KREA AI** Plants **Seedream 5 Lite** Model**: [KREA AI](https://x.com/krea_ai/status/2026684864380932460?s=20) launched **Seedream 5 Lite**, a low-cost image editing model.
   - It is designed to give performance comparable to their **'Nano Banana'** model at a lower price.


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1475991662743392326)** (6 messages): 

> `Interpretability Hiring, Anthropic, ML Infrastructure` 


- **Anthropic goes on Interp Hiring Spree**: Chris Olah announced that [Anthropic](https://www.anthropic.com/) is seeking approximately **10 research engineers** for their Interpretability team, as seen in [this tweet](https://xcancel.com/ch402/status/2026023963537842248).
- **ML Infrastructure Engineers Wanted**: The roles are aimed at experienced **ML infrastructure engineers** interested in model internals, with **no prior interpretability experience required**.
   - Opportunities abound, including those mentioned on [fxtwitter](https://fxtwitter.com/adamimos/status/2025966678253904238).


  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1476061481857454215)** (43 messages🔥): 

> `Claude Code Limitations, Codex as an Alternative, Agentic Engineering Strategies, Pi Agent Loop, Tool Use and Notation` 


- **Claude Code Faces API Integration Challenges**: A member expressed *disillusionment* with **Claude Code** for larger tasks, noting that while APIs are "done", they often don't fully meet specifications, leading to integration issues across different layers of the system.
   - They are now considering **Codex** as an alternative for building a *beefy system*.
- **Pi Powers OpenClaw with Codex**: A member suggested using **Codex** within **Pi**, the agent loop that powers **OpenClaw** and shared a [link to Pi's packages](https://pi.dev/packages) and [a YouTube video](https://youtu.be/f8cfH5XX-XU?si=q8gRZjkG-iMkglLb) to guide users to contribute.
   - Another member said, *it's best to stick with the 'official' coding harnesses: Claude Code and Codex since the LLMs were RLed within those harnesses.*
- **Anti-Spec Approach Favored for Agentic Engineering**: A member advocated against detailed upfront specifications in agentic engineering, emphasizing iteration, failure validation, and pruning, suggesting that a spec should be built *afterwards* instead.
   - They said that it's mostly done out of *false sense of control and pride.*
- **Tool Use and Notation Shapes LLM Generalization**: A member shared a [blog post](https://the.scapegoat.dev/tool-use-and-notation-as-generalization-shaping/) on *tool use and notation* as generalization shaping, promoting their own work.
   - Another member found it a perfect fit for a dialectic they were engaged in.
- **Rapid Prototyping via Prompting Wild Systems**: A member described their approach to building complex systems with *innovative APIs* using minimal prompting and manual review, focusing on event-driven architectures with tools like **golang**, **watermill**, and **redis**.
   - They shared a concrete example that involved merging a **TUI** with a yolo prototype, while creating a core package that contains the **cozodb** related functionality and **JS API**, including embeddings and vector search.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1476317524638568642)** (1 messages): 

> `Hermes Agent, Open Source Agent, Multi-level memory system, Persistent dedicated machine access, CLI Integration` 


- **Hermes Agent: The Open Source Agent Arrives**: Nous Research has launched **Hermes Agent**, an open-source agent with a multi-level memory system and persistent dedicated machine access, designed to grow with the user.
   - Hermes Agent can run in your CLI, Telegram, WhatsApp, Slack, and Discord, picking up and transferring sessions wherever you go.
- **Advanced Capabilities & Extensive Integrations Power Hermes Agent**: Hermes Agent features advanced agentic capabilities such as command over subagents, programmatic tool calling, advanced filesystem/terminal control, agent-managed skills, and browser use.
   - It is powered by **OpenRouter** and **Nous Portal** subscriptions, offering CLI integration and messaging platform support.
- **Free Month Promo & Developer-Friendly Design Debut!**: The first 750 new sign-ups at [portal.nousresearch.com](https://portal.nousresearch.com) get a free month with coupon code **HERMESAGENT**.
   - Hermes Agent is open source and built in Python, which makes it easy for developers to extend.
- **Agentic RL Pipeline & Mass-Scale Data Generation gets Boosted**: Hermes Agent also powers an agentic RL pipeline, expanding **Atropos** to enable RL with Hermes Agent primitives, and it supports mass-scale data generation out of the box.
   - Check out the [GitHub repo](https://github.com/nousresearch/hermes-agent) or install with one command in your terminal: `curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash`.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1475953769337454725)** (96 messages🔥🔥): 

> `Qwen Base Model Release, Codex 5.3 API Pricing, Steinberger's OpenClaw Process, OS Frontier Math Level 4 Update, NousChat Development` 


- ****Qwen**'s Base Model weights released**: **Qwen** released the base weights for their **Qwen3.5-35B-A3B** model, available on [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base).
- **New pricing for **Codex 5.3****: **Codex 5.3** is out in API with a new pricing structure: **$1.75** for input and **$14** for output.
- **Steinberger's OpenClaw: A Vibe-Coded Miracle**: Steinberger released a video explaining how **OpenClaw** came together and it was extracted via **AI** from his previous plans and ideas and code snippets and gave that to **AI** to make the new code.
   - *He has no idea what his software does* and its structure is just a stack of channels.
- ****OS Frontier Math Level 4 Update****: The **Kimi 2.5 (1st OS)** scored a **4.2%**, and **Glm 5** and **V3.2** scored a **2.1%**.
- ****NousChat** is progressing to align with **Kimiclaw****: A member inquired about plans to host a service like **Kimiclaw**, and another responded that *NousChat is progressing in ways that you might say aligns with that*.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1476200349680537640)** (2 messages): 

> `Arxiv Paper` 


- **Arxiv Paper Shared**: A member shared a link to an Arxiv paper: [https://arxiv.org/abs/2602.16800](https://arxiv.org/abs/2602.16800).
- **Interesting Find**: Another member replied that it was an interesting paper.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1476200349680537640)** (2 messages): 

> `Arxiv Paper` 


- **New Arxiv Paper Shared**: A member shared a link to a new Arxiv paper: [https://arxiv.org/abs/2602.16800](https://arxiv.org/abs/2602.16800).
- **Arxiv Paper Attracts Attention**: Another member commented that it was an *interesting one*.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1475945775673639107)** (49 messages🔥): 

> `Pythia-2.8b Checkpoint Issues, Hugging Face serving weights, safetensors and HF, deduped models, Voice AI Model Sesame AI` 


- **Pythia-2.8b Checkpoint Bug Triggers Investigation**: A member encountered a bug while trying to reproduce a paper using **pythia-2.8b** checkpoints, discovering that [Hugging Face](https://huggingface.co/) was serving the same weights regardless of the revision chosen.
   - The SHA256 hashes for `pytorch_model.bin` and `model.safetensors` were identical across different steps, raising concerns about the integrity of the checkpoints.
- **HF Shards Save Bandwidth**: Members discovered that the sharded `safetensors` files for **pythia-2.8b** are different across steps, while the non-sharded files are identical, leading to discussions on how HF loads models and handles sharding.
   - One member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
- **Smaller Pythia Models got Deduped**: EleutherAI is fixing the incorrectly labeled **14m** and **30m** models, both of which were deduped versions, and is training duped models to replace them, clarifying a labeling issue.
   - A member mentioned they fixed an issue mixing up some uploads and ran the fix overnight.
- **Member Thinks HF Saved Disk via Symlink Shenanigans**: A member speculated that [Hugging Face](https://huggingface.co/) might have used symlinking to save disk space, potentially causing data corruption, after realizing they made a similar mistake in the past.
   - This theory suggests that the issues with **pythia-2.8b** checkpoints could be due to HF's internal processes for managing storage.
- **Sesame AI Voice Model Sparks Curiosity**: A member asked about the [Sesame AI](https://sesame.ai/) voice AI model, noting its apparent alignment and potential foundation on the **Gemma** model, sparking discussion about its capabilities.
   - Another member highlighted Sesame AI's focus on low-latency voice systems, integrating ASR, LLM, and TTS, and suggested examining the [Moshi paper](https://google.research/pubs/pub62870/) for insights.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1475958408795390032)** (22 messages🔥): 

> `Diffusion Papers, Flow Matching vs Diffusion, Pythia Models` 


- **Diffusion Literature Deep Dive**: Members discussed key diffusion papers since the Latent Diffusion Model, highlighting [Rectified Flows and Flow Matching](https://arxiv.org/abs/2209.03003) and [Diffusion Forcing](https://arxiv.org/abs/2407.01392).
   - Also mentioned were papers from **ByteDance Seed** and **Hunyuan** (e.g., [https://arxiv.org/abs/2509.20427](https://arxiv.org/abs/2509.20427), [https://arxiv.org/abs/2509.23951](https://arxiv.org/abs/2509.23951)), with a recommendation for a [YouTube playlist](https://youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH&si=VIUFIdOSsMDWbotb) as a resource.
- **Flow Matching's Fluid Foundation**: A discussion clarified that [Flow Matching](https://arxiv.org/abs/2209.03003) is related to but distinct from earlier work ([https://arxiv.org/abs/1807.03039](https://arxiv.org/abs/1807.03039)), as Flow Matching is continuous in time and doesn't require invertibility.
   - One member noted that Flow Matching emerged more from diffusion research, representing an alternate way to parameterize a flow.
- **Louie's Latent Link Logistics**: One member mentioned a blog post with links to papers regarding the latent part of the diffusion pipeline: [https://over.world/blog/dito](https://over.world/blog/dito).
   - The papers mentioned included: [https://arxiv.org/abs/2512.12386](https://arxiv.org/abs/2512.12386) with references to other papers such as [https://arxiv.org/pdf/2510.11690](https://arxiv.org/pdf/2510.11690) and includes new methods like **Token Routing**, **Path-Drop Guidance**, **Representation Alignment** of the latent embeddings.
- **On Pythia!**: A link was shared on [Pythia models](https://arxiv.org/abs/2510.14865).


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1476180550497865861)** (3 messages): 

> `lm-evaluation-harness, MMLU pro eval, Qwen3 models, HF backend, vLLM backend` 


- **Speedy vLLM backend for lm-eval Harness**: A member requested reviews for a [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/3604) aimed at speeding up the evaluation of multi-choice tasks with single token answers using **vLLM backend** in *lm-evaluation-harness*.
   - The speedup should address slowness issues compared to **HF backend**, especially for tasks like **MMLU pro eval**.
- **Qwen3 models' newlines**: A member inquired about unexpected newline behavior with **Qwen3 models** in *lm-evaluation-harness*, where `\n\n` is moved to continuation, and linked to [issue 2144](https://github.com/EleutherAI/lm-evaluation-harness/issues/2144) as potentially related.
   - The user gave an example with `context` and `continuation` in the log outputs.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1475962746284675233)** (40 messages🔥): 

> `ZeroGPU Allocation Issues, Small Language Models, Edge Inference Memory, Code RAG, Tiny Aya` 


- **Gradio versions trigger ZeroGPU Allocation Blues**: Users reported issues with **ZeroGPU allocation**, possibly linked to versions of **Gradio prior to 5.12.0** having login bugs.
   - Checking container logs might reveal if **Gradio**, the `spaces` library, or the **HF server** is causing the problem; rebuilding after an empty commit might also resolve version-related issues.
- **Tiny Aya launched by Cohere**: **Cohere** recently launched **Tiny Aya**.
- **Independent Dev cracks crazy edge memory wall**: An independent developer claims to have compressed a **5GB MoE shard** from **MiniMax-m2.5** down to a **2MB vector-quantized latent space**.
   - They're preparing a paper for *arXiv (cs.LG)* and seek an endorser to review their *"black magic edge AI stuff"*.
- **Code RAG Invention Incoming to Scale Projects**: A member is inventing **Code RAG** to scale a project, already claiming to be *"half way there"*.
   - They shared a graph illustrating **how code is related to other code**.
- **Distillation Training Difficulties**: One member is asking for useful instructions for **distillation training**, because their *"student model didn't think as the teacher model"*.
   - They stated that *"training yourself is much diff than training LLMS"*.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1475973128625721427)** (11 messages🔥): 

> `Distributed Fine-tuning, GPT-OSS, Qwen 2.5, Mistral models, RTH-LM model` 


- ****Zagora** builds distributed fine-tuning system**: A member from **Zagora** announced they are *building a distributed fine-tuning system for training 70B+ models* over standard internet, turning scattered GPUs into a unified training supercomputer.
   - The platform now supports **GPT-OSS, Qwen 2.5, and Mistral** and uses a pipeline-style training approach inspired by Petals and the SWARM Protocol.
- ****RTH-LM** could stress test **Zagora** system**: A member suggested their **RTH-LM** model, a non-Transformer model (Fractal Gated Causal TCN), as a perfect stress test for **Zagora's** system due to its *zero cross-node state synchronization overhead during pipeline stages*.
   - They are targeting a **120B** scale and asked if the platform supports custom model architectures (any nn.Module) in addition to Transformer families and pointed to their [paper](https://doi.org/10.6084/m9.figshare.31376560), [repo](https://github.com/rthgit/ZetaGrid), and [25B model](https://huggingface.co/RthItalia/Rth-lm-25b).
- ****Zagora** focuses on transformer models**: The **Zagora** team responded that they are currently focused on Transformer-family models such as **Llama, Qwen, Mixtral, and Gemma**.
   - However, they mentioned they would revisit the possibility of integrating **RTH-LM** if it gets a Transformer-compatible wrapper.
- ****webXOS** Black Hole Time-Lapse Dataset released**: A member shared the [webXOS Black Hole Time-Lapse Dataset](https://huggingface.co/datasets/webxos/webXOS-blackhole-synthetic), which contains synthetic black hole renderings with gravitational lensing generated by a Three.js simulation in webxOS.
   - Each sample includes a time-lapse sequence of PNG images and associated physical parameters making it ideal for multi-modal model training, physics-inspired ML, or satellite image study analogies.
- **Optimize Your Models Where It's Safest**: A member published an article [Optimizing where it's safest: a model-first approach](https://medium.com/@paragekbote23/optimizing-where-its-safest-a-model-first-approach-7eee3d48bc63) describing the different types of methods that you can apply to optimize your models without changing your runtime or committing to a new inference provider and what results can be observed.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1476038987654234265)** (6 messages): 

> `agents course, smolagents, Qwen API, huggingface/agents-course` 


- **Agents Course Channel Hunt**: Newcomers to the **Hugging Face agents course** are having trouble finding the specific channels mentioned in the course materials.
   - It appears that *the channels have been merged into a single channel*, according to one of the members, linking to [PR #653](https://github.com/huggingface/agents-course/pull/653) in the agents-course repo.
- **Smolagents quiz troubles**: One member is encountering warnings preventing their **smolagents final quiz** code from being evaluated, specifically with an **API error**.
   - The error message indicates that *https://api-inference.huggingface.co* is no longer supported and suggests using *https://router.huggingface.co* instead, related to the **Qwen API**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1476038450711888094)** (15 messages🔥): 

> `Kernel Programming Environment Setup, HPC Systems with GPUs, TPUs, or Soft GPUs on FPGAs, Performance Modeling for Accelerators, GraphCulon` 


- **GPUmode.com goes down for maintenance**: [GPUmode.com](https://gpumode.com) went down for maintenance but was brought back up shortly.
- **Users discuss Kernel Programming Environment Setup**: A member asked about others' go-to kernel programming environment setups, citing **Modal** as helpful but lacking **NCU profiling** outside contests.
- **Calculon tool enables High-Level Co-Design of Systems**: A member shared a link to [Calculon](https://dl.acm.org/doi/10.1145/3581784.3607102), *a methodology and tool for high-level co-design of systems*.
- **GraphCulon looks interesting**: A member noted that [GraphCulon](https://hpc.fau.de/files/2026/01/2026-01-20_Froening.pdf) actually looks interesting, but it has not been released yet, linking to a talk about it.
- **GPU Observability seminar begins**: A GPU Observability seminar kicked off, slides promised to be shared by the speaker.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1476095031273062420)** (1 messages): 

> `cuda::memcpy_async, SMEM bank conflict` 


- **SMEM Bank Conflict with cuda::memcpy_async**: A user inquired whether **SMEM bank conflicts** are a significant concern when employing **cuda::memcpy_async** for data transfer from **GMEM to SMEM**.
   - The user posited that **SMEM bank conflicts** primarily relate to warp access of **SMEM**, suggesting they might not be a major issue in this scenario, but sought additional perspectives.
- **GMEM to SMEM Transfer Considerations**: The discussion revolves around optimizing memory transfer strategies within CUDA, specifically concerning the use of **cuda::memcpy_async**.
   - The core question is whether the asynchronous nature of the memory copy impacts the potential for **SMEM bank conflicts**, warranting careful consideration.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1476253549653397716)** (3 messages): 

> `FA3 Kernels, SDPA Backend Selection, Blackwell GPUs` 


- **FA3 Kernels Replace FA2 in PyTorch Dispatch**: When a user calls `activate_flash_attention_impl(“FA3”)`, the default **FA2 kernels** are overridden with **FA3 kernels** in the dispatch table until `restore_flash_attention_impl` is called, which restores the default **FA2 kernels**.
   - This is achieved by adding a key-value pair `{“FA3”, register_fn}` to a dictionary that maps version names to a callable function, and running the `register_fn` (defined [here](https://github.com/pytorch/pytorch/blob/580a6e2c814db93aa8df0a80e3e85c330621b9cb/torch/nn/attention/_fa3.py#L54)) to register the **FA3 kernels** with the PyTorch dispatcher.
- **SDPA Chooses FA Backend Based on GPU Device**: The selection of the Flash Attention (FA) backend in SDPA depends on the GPU device, using the `select_sdp_backend` function (defined [here](https://github.com/pytorch/pytorch/blob/72d0e643eb90f14085bab5e9cab8d3cceb0d7847/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L931)) to choose the priority order of SDP backends.
   - The default ordering is **flash, mem efficient, then math**, but users can override this to enable specific backends; for example, for **Blackwell GPUs**, flash attention doesn't work, so the first priority is **cuDNN**, determined by [this line](https://github.com/pytorch/pytorch/blob/72d0e643eb90f14085bab5e9cab8d3cceb0d7847/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L91) in `check_prefer_cudnn_attention`.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1475967596871549080)** (1 messages): 

> `eBPF, GPUs, profilers, OS Policies` 


- **eBPF gets GPU Boost**: Yusheng Zheng will discuss extending **eBPF** to work better with **GPUs** on [date] at 12:00 pm PST.
   - This talk will cover recent work such as *gpu_ext: Extensible OS Policies for GPUs via eBPF* ([paper](https://arxiv.org/abs/2512.12615)) and *Extending eBPF to GPU Device and Driver Contexts* ([LPC event](https://lpc.events/event/19/contributions/2168/)).
- **Join GPU Mode Profiler Party**: The speaker expressed a desire for more profilers and profiler visualization libraries to be developed within **GPU MODE**.
   - Interested individuals are encouraged to join and watch a related [YouTube video](https://www.youtube.com/watch?v=8U7SzGnHoJU) for inspiration.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1476123386907332648)** (10 messages🔥): 

> `CUDA learning resources, Distributed inference platforms on GB200 / B200 nodes, Career transition to GPU field, New channel proposal for high-volume questions` 


- **CUDA Newbies Seek Kernel-Level Knowledge**: A member new to **CUDA kernel**-level work seeks advice on learning effectively for distributed inference platforms on **GB200 / B200 nodes** using open-source projects like **Dynamo, vLLM, LMCache, and NIXL**.
   - The member specifically asked about the helpfulness of starting with **PMPP**, participating in **GPU MODE competitions**, or following **NVIDIA’s CUDA** / performance courses, with a long-term goal of contributing back to inference open source.
- **PMPP and Open Source Hacking Recommended for CUDA Beginners**: A member suggested referring to a previous discussion and recommended reading **PMPP chapters 1 through 6** and then jumping into contributing to open source projects to learn CUDA effectively.
   - They encouraged participation in competitions for fun.
- **GPU Field Career Transition Guidance Sought**: A member with a Computer Engineering degree working as a software engineer expressed interest in moving into the **GPU field** and building a career around it.
   - They asked if starting with **CUDA and GPU profiling** would be the right direction and requested guidance on how to approach this path, another member echoed this request.
- **Newbie Question Firehose Channel Proposed**: A member proposed a channel called **#newb_firehose** for high-volume discussions of targeted questions related to learning CUDA, such as understanding **PMPP, NCCL codebase**, or working on personal kernels.
   - Another member indicated that the existing **#beginner** channel serves this purpose, encouraging the user to feel free to be noisy there.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1475966859315642370)** (6 messages): 

> `Hacky Submissions Parsing, KernelBot Environment Enhancement, RL environment for Kernel Optimization` 


- **Hacky Submission Parsing Initiated**: Parsing of hacky submissions has commenced, initiating fingerprinting and deeper analysis as depicted in the [attached image](https://cdn.discordapp.com/attachments/1298372518293274644/1475966858938290247/image.png?ex=69a0ba41&is=699f68c1&hm=5f158d3d240d2fb95fa2d438d0f1134b7174ade68584ec9f8bc7f5543a05e85f&).
- **KernelBot Environment Augmentation Questioned**: A member inquired whether to add a new submission to the KernelBot environment via [PrimeIntellect](https://app.primeintellect.ai/dashboard/environments/roeybc/kernelbot-env).
   - Another member suggested that if the ruleset is approved upon inspection, it could be added to KernelBot as a verification layer.
- **Interest Expressed in Kernel Optimization RL Environment**: A member expressed interest in the **RL environment for kernel optimization** and suggested building common infrastructure.
   - No additional details or specific discussions were highlighted in the given messages.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

simran9493: Yes!
  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1475952578813628447)** (1 messages): 

> `CLI Update, Auth Issues` 


- **CLI Gets Update**: Members were instructed to update their **CLI to the newest version**.
   - This update likely includes bug fixes and new features to improve functionality.
- **Auth Issues Flagged**: Members were prompted to report any **auth-related issues**.
   - This proactive approach ensures smooth access and prevents disruptions.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1475970084634624010)** (1 messages): 

> `B200 GPUs, GPU Leasing, Neocloud Solutions, Lightning AI Clusters` 


- **B200 GPU Price Shock Prompts Leasing Recommendation**: A user remarked that **B200 GPUs** are prohibitively expensive and advised leasing or renting as a more viable option for non-enterprise users.
   - They highlighted a solution from their company, [Lightning AI Clusters](https://lightning.ai/clusters), as a potentially attractive alternative.
- **Neocloud Leasing Emerges as B200 Alternative**: Given the high cost of **B200 GPUs**, a user suggests exploring **Neocloud** leasing or renting options, particularly for those outside of enterprise environments.
   - The user specifically recommends [Lightning AI's cluster solutions](https://lightning.ai/clusters) for those seeking alternatives.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1476207064589013146)** (4 messages): 

> `CuTeDSL editable package installation, CuTeDSL 4.4 release breaking changes, Vectorized tiled copy 2D in CuTeDSL, Thread value layout in CuTeDSL` 


- **CuTeDSL Editable Package Installation Guidance Sought**: A user requested a guide for installing the editable package for **CuTeDSL**, noting they found the existing script difficult to understand.
   - They mentioned that the latest **4.4 release** seems broken, as it split the Python package into multiple new ones.
- **Vectorized Tiled Copy 2D Thread Layout Preference**: A user expressed a preference for a thread value layout (visualized in an [attached image](https://cdn.discordapp.com/attachments/1362196854460383353/1476283940481269964/image.png?ex=69a0900f&is=699f3e8f&hm=9c0f3ff5fa5c28afce23b811b44e00b7b5d575411164fd2e9f7bb6c8dc0bb837&)) for performing vectorized tiled copy 2D in **CuTeDSL**, finding it more intuitive.
   - They mentioned that *quack* also recently changed to this layout, and included a code snippet of the previous layout using **shape** and **stride**.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1476006971227377815)** (2 messages): 

> `Helion implementation, all_gather + FP8 + GEMM optimization, Kernel profiling and debugging` 


- **Helion Implementation lagging behind baseline**: A member is working on a **Helion implementation** of all_gather + FP8 + GEMM (H100) based on a [vllm-project](https://github.com/vllm-project/vllm/pull/33933) pull request, but it's currently **1.26–4× slower** than the baseline.
   - The goal is to **optimize the kernel** and profile it, to inspect for bubbles and waiting, however tracing via Chrome is proving difficult to follow.
- **Seeking Advice on Kernel Optimization Tools**: A member is seeking recommendations for tools or workflows to **optimize kernels**, along with tips, documentation, or shared experiences.
   - They have been profiling with tracing, but it’s pretty hard to follow and reason about where the real bottleneck is, after implementing a [Meta data center engineering](https://engineering.fb.com/2026/02/24/data-center-engineering/rrcclx-innovating-gpu-communications-amd-platforms-meta/).


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1475956590568935484)** (4 messages): 

> `Helion PR 1418, Helion all_gather + FP8 + GEMM optimization` 


- **Helion's Parallel Reads Examined**: A member inquired whether [Helion PR 1418](https://github.com/pytorch/helion/pull/1418) addresses the parallel-read issue described in [JAX documentation](https://docs.jax.dev/en/latest/pallas/design/design.html#grad-of-pallas-call).
   - The author of the PR is unavailable to respond until the end of the week or the following week.
- **FP8 GEMM Optimization in Helion**: A member is working on a **Helion** implementation of **all_gather + FP8 + GEMM (H100)**, as seen in [this pull request](https://github.com/vllm-project/vllm/pull/33933).
   - The current implementation is **1.26–4x slower** than the baseline, and the goal is to optimize the kernel and requested advice on profiling tools and workflows.
- **NCU Insights Requested**: A member suggested using **NCU** to gain actionable insights into the kernel optimization.
   - No further information was provided on the use of NCU.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/)** (1 messages): 

vovw: amazing work
  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1475966707507138610)** (5 messages): 

> `CUDA kernels, TPU-inference, MLSys roles, Distributed training` 


- **CUDA Kernel Knowledge in Inference and MLSys Roles**: A member inquired about the necessity of **extensive CUDA kernel knowledge** for inference and MLSys roles, especially with experience in **TPU-inference**.
   - Another member expressed a similar doubt about how much **CUDA/kernels** they need to know as an undergrad, highlighting the common concern among those entering the field.
- **Training vs. Inference: CUDA Kernel Importance**: A member shared their experience in **distributed training**, noting that deep CUDA kernel knowledge beyond **Ampere** architecture isn't always essential but definitely valuable.
   - They recounted situations where writing a specific kernel to replace an op would have been beneficial, emphasizing that knowing both training and inference can be helpful but isn't strictly required.


  

---


### **GPU MODE ▷ #[from-scratch](https://discord.com/channels/1189498204333543425/1466534042768904356/1476227907327098931)** (1 messages): 

> `Serenade Language, C++ transpilation, CUDA and x86-64 ASM, GPU kernels, Dear ImGui support` 


- **Serenade: "Simple Python, Fast C++" Arises**: A member introduced **Serenade**, a new language that transpiles to **C++**, **CUDA**, and **x86-64 ASM**, aiming to be as simple as **Python** but as fast as **C++** with manual memory management.
   - The language includes [GPU kernels support](https://github.com/kaifczxc-lab/Serenade-Cloud) (**serenaCore**, custom BLAS kernel), and integrated **Dear ImGui** support with a single-pass compilation system, and is planning on creating an operating system with it.
- **Serenade's Goal: Combine the Best of Each Language**: The creator emphasized that **Serenade** is a solo initiative driven by the idea to create a great tool combining the best aspects of multiple languages.
   - The source code is currently private, but the simplest functions of **Serenade** can be tested through a browser on [Replit](https://github.com/kaifczxc-lab/Serenade-Cloud).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1476001459421188335)** (15 messages🔥): 

> `Kimi vs GLM, Agent Quota, Kimi for coding, KimiClaw browser navigation` 


- **Kimi Compared to GLM Showdown**: Members debated the performance of **Kimi** against **GLM 5**, with one user quipping that **Kimi** is *100,000 times faster* than **GLM**.
   - Another user suggested **GLM 5** has a slight edge, but noted that **GLM 5** is slow via the official z.AI API, but can be faster with other providers.
- **Users Seek Agent Quota Top-Up**: A user inquired about topping up the agent quota specifically, citing cost concerns for **Allegro**.
   - They also noted that **agent docsis kimi slides with nb pro** are no longer available for free.
- **Kimi Shines for Coding Tasks**: After testing coding plans from **Kimi**, **MiniMax**, and **Alibaba**, one user chose to stick with **Kimi** for coding.
   - The user cited **speed**, **uptime**, **usage limits**, and **model quality** as deciding factors.
- **KimiClaw's Browser Blind Spots**: A user reported frustration with **KimiClaw's** inability to navigate browsers independently.
   - They asked if others faced the same problem and sought solutions, as well as asking *What can we use for kimi so we reduce context and save tokens when we analyze/process big files? I think Claude has something for that.*


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1476026063896444988)** (11 messages🔥): 

> `Github Reconnection Issues, Local Development Environment Setup, Account Bans, Manus Cookie Problems, Website Design Problems` 


- **User Confronts Github Reconnection Quandary**: A member is facing issues reconnecting their **Github** account, being prompted to create a new repository instead of reconnecting to the original one.
   - The member states they are not a coder or software developer and need easily understandable instructions.
- **OAuth Environment Variables for Local Devs Probed**: A member seeks guidance on obtaining the **VITE_APP_ID**, **OAUTH_SERVER_URL**, and **VITE_OAUTH_PORTAL_URL** environment variables to run a Manus-developed app locally.
   - They are also inquiring if OAuth configuration is necessary to allow the **redirectUri** `http://localhost:3000/api/oauth/callback` during local development.
- **Account Creation Ban Baffles User**: A member reports getting banned immediately after creating an account and is asking for advice on how to resolve this issue.
   - No advice was given.
- **Manus System Blames Infrastructure for Cookie Conundrum**: A member shares a detailed issue where **Manus** is stuck in a redirect loop due to cookie problems on a custom domain ([anointedforai.com](https://anointedforai.com)).
   - Manus itself diagnosed the problem as an infrastructure/hosting issue and suggested contacting support or migrating off **Manus** to a platform with more control over cookie settings.
- **Member Moans about Manus-Made Website**: A member complains about their website design, stating that it was *bullshit, made by Manus*, and is asking for assistance in fixing it.
   - Another member offered to help via direct message.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1475963283151388722)** (8 messages🔥): 

> `git submodules in Aider, Low-cost LLMs for Aider, Deepseek V3.2 for Aider, Xiaomi/mimo-v2-flash for Aider, moonshotai/kimi-k2.5 for Aider` 


- ****Aider** Gains a `/ok` Alias for Quicker Code Changes**: Aider now has a new feature in the main branch: `/ok` is now an alias for `/code Ok, please go ahead and make those changes.` allowing for faster **code modifications**.
- **User Seeks Low-Cost LLMs for **Aider****: A user is seeking advice on finding the best low-cost LLM to use with **Aider**, mentioning that Gemini burned through all the tokens in just a couple of hours.
   - Another member suggested using [OpenRouter](https://openrouter.ai/) to switch between different models.
- ****Deepseek V3.2** Recommended for Reasoning with **Aider****: A user recommends **Deepseek V3.2** as a default LLM with **Aider** because it has good reasoning and is cheap, but can be a bit slow sometimes.
- ****Xiaomi/mimo-v2-flash** Excels at Quick File Editing in **Aider****: **Xiaomi/mimo-v2-flash** is recommended for "dumb" file editing capabilities like fuzzy search replace or completing stuff in **Aider** because it's very cheap and very quick.
- ****moonshotai/kimi-k2.5** Tackles Hard Problems in **Aider****: **moonshotai/kimi-k2.5** is suggested as the planning model and **mimo-v2-flash** as editing model for solving harder problems in **Aider**.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1476174943799476294)** (5 messages): 

> `WeAreDevelopers World Congress North America 2026, AI Control Hackathon 2026, Redwood Research, ControlArena benchmark challenges, ControlConf Berkeley` 


- ****WeAreDevelopers** Congress debuts in North America**: The **WeAreDevelopers World Congress North America** is debuting in San José, CA from Sept 23–25, 2026, expecting **10,000+ devs** and **500+ speakers** focusing on real-world engineering at scale.
   - Topics include scaling distributed systems, API platforms, and DevOps, and you can use code *Community_MLOps* for a **10% discount** at [wearedevelopers.us](https://wearedevelopers.us).
- ****AI Control Hackathon** launched by Apart Research**: **Apart Research**, co-organized with [Redwood Research](https://www.redwoodresearch.org/), is running an **AI Control Hackathon** from March 20-22, 2026, focusing on systems that ensure AI does what we want, even against subversion.
   - The hackathon features three tracks, including **ControlArena benchmark challenges**, **control protocol design**, and **red teaming**, with **$2,000** in cash prizes and a trip to [ControlConf](https://controlconf.org/).
- ****ControlConf** Trip Prize Offered**: The **AI Control Hackathon** first place prize wins a fully funded trip to [ControlConf](https://controlconf.org/) Berkeley (April 18-19), flights and hotel included.
   - Learn more about [ControlConf](https://controlconf.org/).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1476253403221983475)** (2 messages): 

> `SF DSPy Meetup, DSPy in Production, RLMs, Dropbox, Shopify` 


- **SF DSPy Meetup on the Horizon**: Announcing another **SF DSPy meetup**, this time focusing on **DSPy in production use cases** and **RLMs**.
   - Engineers from **Dropbox** and **Shopify** will share case studies, and there will be a walkthrough of **dspy.RLM**, see [Luma link](https://luma.com/je6ewmkx).
- **Dropbox and Shopify Engineers to present**: **Dropbox** and **Shopify** engineers will present case studies at the SF DSPy Meetup.
   - The meetup will focus on using **DSPy in production** and **RLMs**.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1476051238964559986)** (2 messages): 

> `JAX, Functions` 


- **Tinygrad Creator Praises JAX's Function Design**: George Hotz, creator of tinygrad, [acknowledged JAX's superior function design](https://x.com/__tinygrad__/status/2026491994546282605), implying its influence or correctness in design choices.
   - The second tweet [further emphasizes the point](https://x.com/__tinygrad__/status/2026500842749309267).
- **JAX Function Design Praised**: The creator of Tinygrad expressed admiration for JAX's approach to function design.
   - This suggests that JAX's method may serve as a model or validation for similar choices in Tinygrad.


  