---
id: MjAyNS0w
title: GPT-5 Codex launch and OpenAI's quiet rise in Agentic Coding
date: '2025-09-15T05:44:39.731046Z'
description: >-
  **OpenAI** released **GPT-5-Codex**, an agentic coding model optimized for
  long-running software engineering tasks with dynamic task-adaptive thinking,
  multi-hour autonomy, and improved code quality. It achieves 51% accuracy on an
  unreleased large refactor benchmark and integrates deeply with developer tools
  like Xcode. Meanwhile, **Alibaba** launched **Qwen3-Next-80B**, a hybrid MoE
  model with native long-context support (262k tokens, extensible to 1M+),
  targeting efficient reasoning and repository-scale code analysis, supported by
  **Together AI** and **NVIDIA** with CUDA-accelerated attention. The trend
  towards hybrid SSM + MoE architectures is noted, emphasizing efficiency and
  scaling in China and US training regimes. Community discussions highlight the
  importance of variable compute and routing for inference efficiency and
  quality.
companies:
  - openai
  - alibaba
  - together-ai
  - nvidia
models:
  - gpt-5-codex
  - qwen3-next-80b
topics:
  - agentic-ai
  - software-engineering
  - long-context
  - mixture-of-experts
  - model-optimization
  - cuda-acceleration
  - inference-efficiency
  - routing
  - task-adaptive-thinking
people:
  - sama
  - swyx
  - omarsar0
  - ofirpress
---


**Codex is all you need?**

> AI News for 9/12/2025-9/15/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (192 channels, and 11857 messages) for you. Estimated reading time saved (at 200wpm): 1016 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Just like we covered [the quiet rise of Claude Code in June](https://news.smol.ai/issues/25-06-20-claude-code), today is one of those days that ordinarily wouldn't quite qualify for a title story, but the cumulative impact of a month's worth of hype of increasing sentiment for GPT 5 and OpenAI's Codex (an answer to Claude Code, but with a lot more breadth) is worth flagging, and is given extra juice in today's release from OpenAI. This is best covered in [our sister publication](https://www.latent.space/p/gpt5-codex). If you were a heavy Codex user, note also the pitfalls flagged in the Discord section.

[](https://resend-attachments.s3.amazonaws.com/edYnIDghZ0ZGynh)

---

# AI Twitter Recap

**OpenAI’s GPT-5-Codex and the agentic coding race**

- **OpenAI ships GPT-5-Codex (agentic coding)**: OpenAI released a GPT-5 variant optimized for long-running, tool-using software engineering across the Codex CLI, IDE extension, web, GitHub code reviews, and ChatGPT iOS. Highlights: dynamic “task-adaptive” thinking (15x faster on easy tasks, 2x more deliberate on hard ones), multi-hour autonomy (“>7 hours” on complex tasks), improved instruction-following and code quality, and better SWE-bench–style performance. OpenAI also referenced an unreleased large “refactor” benchmark where GPT-5-Codex reaches 51% accuracy and indicated SWE-bench fixes for apples-to-apples comparisons. See announcements and discussion from [@OpenAI](https://twitter.com/OpenAI/status/1967636903165038708), [@gdb](https://twitter.com/gdb/status/1967639750648750409), [@sama](https://twitter.com/sama/status/1967650108285259822), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1967637842806624370), [@OfirPress](https://twitter.com/OfirPress/status/1967652031704994131), [@swyx](https://twitter.com/swyx/status/1967651870018838765) and routing/depth behavior notes (“router in the model”) by [@swyx](https://twitter.com/swyx/status/1967691956693373183). Early hands-on reports range from “more steerable and persistent” ([@omarsar0](https://twitter.com/omarsar0/status/1967640731956453756)) to frustration over token burn and long loops ([#1](https://twitter.com/Teknium1/status/1967804542357217768), [#2](https://twitter.com/Teknium1/status/1967806788084064290)). OpenAI also teased deep OS integrations (e.g., Xcode sign-in for GPT‑5) via [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1967704919487729753).
- **Evals and coding depth**: OpenAI claims SWE-bench improvements and a new internal “large refactor PR” eval; community called for public versions ([@OfirPress](https://twitter.com/OfirPress/status/1967652031704994131)). There’s broad agreement that variable compute and routing are critical to efficiency and quality at inference time ([@swyx](https://twitter.com/swyx/status/1967662188962910709); [@polynoamial](https://twitter.com/polynoamial/status/1967667644905251156)).

**Qwen3‑Next 80B (A3B MoE), long-context, and the China efficiency push**

- **Qwen3‑Next‑80B (3B active) lands on Together + NVIDIA NIM**: Alibaba’s hybrid MoE model targets long-context (native 262k, extensible to 1M+), repository-scale code analysis, and efficient reasoning. Together AI provides “Instruct” and “Thinking” endpoints ([launch](https://twitter.com/togethercompute/status/1966932629078634543), [contexts](https://twitter.com/togethercompute/status/1966933240683319556)), and NVIDIA added NIM support with CUDA-accelerated attention ([NVIDIA](https://twitter.com/NVIDIAAIDev/status/1967575419638468667)). Alibaba reports strong performance “with only 3B active parameters” ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1966831435756794071)) and head-to-head results vs Gemini 2.5 Flash Thinking on reasoning benchmarks ([@togethercompute](https://twitter.com/togethercompute/status/1966932629078634543)). On-device MLX numbers show eye-catching TPS on Apple hardware ([@ivanfioravanti](https://twitter.com/ivanfioravanti/status/1966866942461177925), [batching](https://twitter.com/ivanfioravanti/status/1966903782400545196)).
- **Architecture trend: hybrid SSM + MoE**: In the past two weeks, 6 of 7 new MLX-LM architectures are MoE, half hybridizing SSM/attention ([@awnihannun](https://twitter.com/awnihannun/status/1966936728469729546), [list](https://twitter.com/awnihannun/status/1966937464834314614)). Context from China v. US training regimes: constrained flops driving infra/model co-design, token efficiency, linear attention, and test-time scaling focus ([@JingyuanLiu123](https://twitter.com/JingyuanLiu123/status/1966887747622453560)). Community sentiment echoes that small models are increasingly capable, given the right recipes ([@Thom_Wolf](https://twitter.com/Thom_Wolf/status/1966889089162244463)).

**Tooling for agents: MCP everywhere, Claude Code SDK, and workflow “vibe coding”**

- **MCP consolidation**: The Model Context Protocol’s value-prop—turn M×N tool integrations into M+N via MCP servers—continues to resonate ([diagram](https://twitter.com/_avichawla/status/1966751224356892769)). New OSS appears across the stack: DeepMCPAgent (LangChain/LangGraph-based MCP agents) ([repo](https://twitter.com/_avichawla/status/1967476110285021213)), Markdown MCP ([@dariusemrani](https://twitter.com/dariusemrani/status/1967496103424934320)), and enterprise hackathon showcases ([thread](https://twitter.com/dariusemrani/status/1967492478132715824)). LangChain shipped reactive agent examples (news curation, ParserGPT, human-in-the-loop for Deep Agents) ([news agent](https://twitter.com/LangChainAI/status/1966909743383146735), [parser](https://twitter.com/LangChainAI/status/1967257030756028505), [HITL](https://twitter.com/hwchase17/status/1967653399517925853)).
- **Claude Code SDK adds agent ergonomics**: Anthropic shipped code references, custom tools, and hooks support, making bespoke agents faster to build ([@_catwu](https://twitter.com/_catwu/status/1966943489759080940)). Replit’s Agent 3 (no-code “vibe” workflows) and Poke (iMessage agents orchestrating ephemeral subagents) show the “agent UX” frontier moving quickly ([Replit demo](https://twitter.com/omarsar0/status/1966949907149058551), [Poke deep dive](https://twitter.com/_philschmid/status/1967245592947831086)).

**RL for reasoning and agents: online RL in product, deep research agents, and new training regimes**

- **Online RL in production assistants**: Cursor’s rollout is widely cited as a first at scale for frontier capability, with enthusiasm around moving continuous training cycles from months → weeks → hours ([@willdepue](https://twitter.com/willdepue/status/1966876626169287035), [follow-up](https://twitter.com/willdepue/status/1966878536247243260)). Strong interest persists in post‑GRPO advances ([@vikhyatk](https://twitter.com/vikhyatk/status/1967375151638716810)).
- **Deep research agents (single-agent RL > complex scaffolds)**: A new study shows a simple RL recipe with length-normalized rewards and strategic tool limits can train single agents that rival multi-agent setups; test-time scaling also helps (parallel searches + pick the shortest successful trajectory) ([summary](https://twitter.com/omarsar0/status/1966900691009720455), [paper](https://twitter.com/omarsar0/status/1966900784844730562)).
- **HRL and decentralized RL**: Meta’s Scalable Option Learning re-architects hierarchical RL for GPU-parallel batch updates (25× training speedups) ([explainer](https://twitter.com/JacksonAtkinsX/status/1967284333678350342)). Gensyn’s SAPO shares rollouts in plaintext across a “swarm” of heterogeneous nodes (up to +94% cumulative reward) ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1967575689844166834)). Tencent’s SimpleVLA-RL scales VLA training via RL ([paper](https://twitter.com/_akhaliq/status/1966883040627769511)).
- **Long-horizon execution**: Multiple analyses argue small step-accuracy gains compound exponentially in long chains; many failures are execution (not reasoning) errors; “thinking” models reduce harmful self-conditioning ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1967440503189754190), [@TheTuringPost](https://twitter.com/TheTuringPost/status/1967374791700369451), [@emollick](https://twitter.com/emollick/status/1967688420639359061)).

**Multimodal and computer-use models**

- **Holo1.5 for computer-use agents (open weights)**: H’s new VLMs (3B, 7B Apache-2.0, 72B) set SOTA on UI localization and QA—core skills for reliable web/mobile use. Open weights, cookbook, and demos are available ([launch](https://twitter.com/laurentsifre/status/1967512750285861124), [H company](https://twitter.com/hcompany_ai/status/1967682730851782683), [cookbook](https://twitter.com/tonywu_71/status/1967520054989504734)).
- **Tencent SRPO (diffusion RL for aesthetics/realism)**: “Self-Regulating Preference Optimization” fine-tunes FLUX1dev along the full denoising trajectory, boosting human-rated realism/aesthetics >3×; code and Space are live and trending ([overview](https://twitter.com/_akhaliq/status/1966911634657390890), [demo](https://twitter.com/linoy_tsaban/status/1967528334126116992)).
- **MobileLLM-R1 (Meta) and on-device reasoning**: Meta introduced small-from-scratch reasoning models (0.14B/0.35B/0.95B; ~4.2T pretraining tokens) with a 140M variant running fully in-browser ([announce](https://twitter.com/tydsh/status/1967476530826854674), [demo](https://twitter.com/_akhaliq/status/1967460621802438731)).
- **New datasets/benchmarks**: SpatialVID (7k+ hours with dense 3D annotations) for spatial video intelligence ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1967260292569845885)), and IntrEx (sequence-level interestingness labels in educational dialogues) ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1967562091570827588)).

**Systems and infra (throughput, routing, and deployment)**

- **Throughput milestones and platform support**: Fireworks reported 540 tokens/s on GPT‑OSS‑120B running on B200, exceeding a leading ASIC in their test ([@lqiao](https://twitter.com/lqiao/status/1967641702484807695)). vLLM 0.10.2 adds aarch64 support (install vLLM directly on GB200; multi-platform images) with more perf on the way ([@vllm_project](https://twitter.com/vllm_project/status/1967752683458269282)). Ray 2.49 introduced prefix cache–affinity routing to maintain KV-cache hit rates across large vLLM fleets ([@seiji_________](https://twitter.com/seiji_________/status/1967639835381993488)).
- **Batching and fleets**: Together released a revamped Batch Inference API (unified UI, all models, 3,000× higher rate limits—30B tokens—and 50% discounts for most serverless models) ([launch](https://twitter.com/togethercompute/status/1967624765625315393)). Prime Intellect opened Reserved Instances for 8–1,000+ GPU clusters with secondary resale to spot markets ([announce](https://twitter.com/PrimeIntellect/status/1967724735430791342)).
- **Kernel and Apple-side speedups**: Standard Kernel previewed minimal CUDA+PTX kernels surpassing cuBLAS/FlashAttention3 on targeted ops; fused LLaMA3 FFN claimed 120% PyTorch perf ([@anneouyang](https://twitter.com/anneouyang/status/1967610221712519612)). MLX continues to mature with high-TPS batching on M3 Ultra and shorter full-suite eval times ([TPS](https://twitter.com/ivanfioravanti/status/1966903782400545196), [MMLU-Pro runtime](https://twitter.com/ivanfioravanti/status/1967229451806318904)).
- **Qwen as deployable building block**: NVIDIA added Qwen3‑Next NIMs; Baseten and Together integrated the “Thinking”/“Instruct” variants for production use ([NVIDIA](https://twitter.com/NVIDIAAIDev/status/1967575419638468667), [Baseten](https://twitter.com/basetenco/status/1967688601640288288), [Together](https://twitter.com/togethercompute/status/1966932629078634543)).

**Top tweets (by engagement, AI/engineering)**

- [“Calling today’s chatbots ‘PhD intelligences’ is nonsense… True AGI won’t make trivial mistakes… we’re 5–10 years away.” — Demis Hassabis](https://twitter.com/vitrupo/status/1966752552025792739) (5K+)
- [rasbt’s LLMs-from-scratch hits 10k forks](https://twitter.com/rasbt/status/1966876565788135837) (6K+)
- [“i suspect society was better off with phone call culture than meeting culture.” — @sama](https://twitter.com/sama/status/1966899254804574266) (20K+)
- [Gemini app tops the App Store in the U.S.](https://twitter.com/demishassabis/status/1966931091346125026) (5K+)
- [GPT‑5‑Codex launch by OpenAI](https://twitter.com/OpenAI/status/1967636903165038708) (8K+) and [@sama](https://twitter.com/sama/status/1967650108285259822) (10K+)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. DIY 8x AMD MI50/MI60 Rig + Open-Source Mobile Agent AndroidWorld #1

- [**Completed 8xAMD MI50 - 256GB VRAM + 256GB RAM rig for $3k**](https://www.reddit.com/r/LocalLLaMA/comments/1nhd5ks/completed_8xamd_mi50_256gb_vram_256gb_ram_rig_for/) ([Score: 429, Comments: 178](https://www.reddit.com/r/LocalLLaMA/comments/1nhd5ks/completed_8xamd_mi50_256gb_vram_256gb_ram_rig_for/)): **Built an 8× AMD MI50/MI60 (32 GB each) rig on an ASRock ROMED8-2T with an EPYC 7532 (32c) and 8×32 GB DDR4 (total** `256 GB VRAM + 256 GB RAM`**) for ~**`$3k` **used; due to 300 mm risers, PCIe 4.0 was unstable so all GPUs run at** `PCIe 3.0 x16` **via bifurcation cards. Software: Ubuntu 24.04.3 + ROCm 6.4.3 with a manual workaround (*"copy-paste gfx906 Tensile"*) to restore deprecated Vega20 (gfx906) support; inference via [llama.cpp](https://github.com/ggerganov/llama.cpp) and [vLLM](https://github.com/vllm-project/vllm). Benchmarks: CPU-only gpt-oss 120B Q8 (65 GB)** `~25 t/s` **with** `~120 t/s` **prompt; 2× MI50** `~58 t/s` **with** `~750 t/s` **prompt on the same model; 8× MI50 on qwen3 235B Q4_1** `~21 t/s` **with** `~350 t/s` **prompt (llama.cpp); 2× MI60 (vLLM, gfx906) on Llama 3.3 70B AWQ** `~25 t/s` **with** `~240 t/s` **prompt. Power: idle** `~400 W` **(≈**`20 W/GPU`**,** `15 W`**/blower, ~**`100 W` **platform), llama.cpp inference averages** `~750 W` **with spikes to** `~1100 W`**. Photos: [top view](https://preview.redd.it/b052o7hi99pf1.jpg?width=4080&format=pjpg&auto=webp&s=20fb34bd86438c2a2111fb0eb52a70b26b3b9685), [open-frame build](https://preview.redd.it/cnnr3ixn99pf1.jpg?width=4080&format=pjpg&auto=webp&s=273be5463afc2508a46f17ea5e63b6e6de51b5fb).** Top comments focus on the high idle draw (`~400 W`) and suggest switching from llama.cpp to **vLLM** to better utilize multi-GPU throughput on this setup.
    - Power/idle draw: Multiple note the rig idles around `~400W`, with one commenter observing blower fans alone may draw `~15W` per card at idle, implying `~120W` of the idle budget could be fans. They ask what RPMs the blowers are running and suggest checking/controlling via ROCm tools (e.g., `rocm-smi --showfan --showtemp` and setting curves) to validate and potentially reduce idle power; fan control behavior on MI50s can materially affect wall draw.
    - Inference stack: A suggestion to switch from `llama.cpp` to **vLLM** for this 8×MI50 setup, citing vLLM’s server-oriented features like PagedAttention, continuous batching, and tensor-parallel support that typically improve throughput and GPU utilization for multi-GPU inference. vLLM has ROCm support and is generally better suited as a high-throughput inference server than llama.cpp on large KV-cache workloads ([vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggerganov/llama.cpp)).
    - Firmware/power tuning: One user recommends flashing the `v420` VBIOS to MI50s, which sets a default power limit of `178W` and can be increased via `rocm-smi` if desired. With ROCm SMI, users can inspect and adjust per-GPU limits and fans (e.g., `rocm-smi --showpowercap`, `-setpoweroverdrive`, `-setsclk`, `-setfan`) to balance performance vs. thermals/power draw ([ROCm SMI docs](https://rocmdocs.amd.com/projects/rocm_smi/en/latest/)).
- [**Update: we got our revenge and now beat Deepmind, Microsoft, Zhipu AI and Alibaba**](https://www.reddit.com/r/LocalLLaMA/comments/1nhdi2u/update_we_got_our_revenge_and_now_beat_deepmind/) ([Score: 210, Comments: 61](https://www.reddit.com/r/LocalLLaMA/comments/1nhdi2u/update_we_got_our_revenge_and_now_beat_deepmind/)): **An open-source mobile-app agent from Minitap AI reports a performance jump to** `#1` **on the community-run [AndroidWorld leaderboard](https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?pli=1&gid=0#gid=0), surpassing entries attributed to DeepMind, Microsoft Research, Zhipu AI, and Alibaba. The agent executes end-to-end tasks in Android UIs (e.g., ride booking, food ordering, app navigation) and the team notes ongoing work on an RL gym for fine-tuning; code is fully open-sourced at [github.com/minitap-ai/mobile-use](http://github.com/minitap-ai/mobile-use).** Commenters question practical use cases (e.g., whether this is mostly QA/automation) and challenge the novelty, suggesting it may be a harness rather than substantive model advances; others express appreciation for the open-source release.
    - Several commenters argue the claim of “beating DeepMind/Microsoft/Zhipu/Alibaba” likely reflects a benchmark-specific evaluation harness rather than advances in model training or architecture. They note this is a wrapper-oriented approach (prompt engineering, routing, or heuristic logic) that can juice scores on a specific eval, making comparisons to full-stack research labs not apples-to-apples; the contribution seems like an evaluation/agent harness, not a new SOTA model.
    - There’s a strong warning about **reward hacking**: targeting a public **leaderboard** encourages overfitting to metric quirks or dataset artifacts, inflating scores without real capability gains. Serious teams purportedly treat the LB as a sanity check and emphasize private holdout sets, cross-benchmark validation, and generalization tests; thus, any “win” should be verified on unseen tasks or private splits before drawing conclusions.
    - Potential practical use cases mentioned are QA pipelines and media-processing workflows, such as audio cleanup/denoising and automated image insertion from a specific directory with strict filename constraints. For these, robustness and reproducibility matter: deterministic batch processing, clear I/O contracts (file globbing, path validation, error handling), and configurable pipelines may be more impactful than leaderboard performance.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

TO BE COMPLETED

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Agentic Coding Upgrades & Workflows**

- **Codex Cranks Code Autonomy**: OpenAI announced upgrades to **GPT‑5‑Codex**, a version of **GPT‑5** optimized for agentic coding, now available across the Codex CLI, IDE extension, web, mobile, and GitHub code reviews per [Introducing upgrades to Codex](https://openai.com/index/introducing-upgrades-to-codex/). The release emphasizes deeper tool-usage for code generation and review, expanding platform coverage for **agentic coding** tasks.
    - Developers celebrated broader availability while flagging reliability concerns in long tool chains; one report noted the `-resume` flag broke after the update in a handy recap: [GPT‑5 Codex](https://www.latent.space/p/gpt5-codex). Community chatter framed expectations as high but pragmatic, with one user lamenting it *"would not let them restore their conversation"* after upgrading.
- **fastWorkflow Wallops Workflows**: A new implementation of the **fastWorkflow** framework matched **Claude Opus 4.1** on the Tau Bench dev set using **DSPy** for agents and parameter extraction, showcased in [radiantlogicinc/fastworkflow](https://github.com/radiantlogicinc/fastworkflow). The demo used the repo’s retail workflow example to structure multi-step tasks into reliable, testable pipelines.
    - Practitioners highlighted that reproducible workflows with typed signatures make agent behaviors more robust and comparable, noting this run *"matches Claude Opus 4.1"* on Tau Bench dev. The thread invited further experiments and extensions to push agent autonomy while maintaining **evaluation discipline**.
- **Overclock Orchestrates Agents**: A spotlight on **agentic automation** emphasized simplicity and strong model routing via [Overclock Work](https://overclock.work/). Participants framed it as a way to standardize execution around **top‑tier models**, with a straightforward UX aimed at production workflows.
    - Observers suggested some organizations already invest heavily in agentic backends and would benefit from a simplified orchestration layer. The conversation focused on real-world deployment posture—prioritizing reliability, observability, and cost control for **end-to-end agents**.

**2. Datasets & Personalizable Speech**

- **FinePDFs Feeds 3T Tokens**: Hugging Face released the **FinePDFs** dataset with ~**3 trillion tokens** from **475 million documents** in **1733 languages**, sourced exclusively from PDFs: [FinePDFs dataset](https://huggingface.co/datasets/HuggingFaceFW/finepdfs). Guidance suggests keeping PDF data under **25%** of a full mix, where combining PDFs with HTML corpora boosted benchmark performance.
    - Builders called it a high-signal addition for **pretraining** and domain adaptation when mixed carefully with web data. The thread stressed **data composition** over raw volume, citing multi-format blends as key to strong generalization.
- **OpenHelix Levels Up**: A refreshed, higher-quality **OpenHelix-5x50k** dropped with improved split consistency and curation for training/eval: [OpenHelix-5x50k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-5x50k). The update focuses on more reliable partitions to make comparisons and ablations cleaner.
    - Users welcomed cleaner splits for **repeatable experiments** and dataset hygiene. The update addresses prior inconsistencies that complicated cross-run evaluation of **finetuning** and **RAG** systems.
- **Voxtral Voices Victory**: **Voxtral** enables fast personal speech finetuning for users with impediments/accents, costing about **$0.26/hour** on an A6000 and pairing with dataset tooling: [VoxFactory (HF Space)](https://huggingface.co/spaces/Tonic/VoxFactory). After finetuning, you can publish the model and dataset and spin up a CPU demo Space that’s free to try.
    - Community feedback highlighted accessibility and zero-friction demos, celebrating that it *"works with CPU!! Free!!"*. Builders framed it as a practical path to personalized **TTS/ASR** models with minimal infra.

**3. Model Ecosystem: Mobile, Norms, Deprecations**

- **MobileLLM Marches On‑Device**: Facebook released **MobileLLM‑R1‑950M** to push more capable **on‑device** language modeling: [facebook/MobileLLM‑R1‑950M](https://huggingface.co/facebook/MobileLLM-R1-950M). The goal is to reduce dependence on cloud services while retaining enough reasoning capacity for useful local tasks.
    - Engineers see it as momentum for **edge inferencing**, where latency, privacy, and offline resilience matter. Conversations compared device footprints and practical app targets for sub‑billion‑parameter models.
- **Qwen3‑Next Norms Noted**: The **Qwen3‑Next‑80B‑A3B‑Instruct** card clarifies it uses **RMSNorm** (zero‑centered gamma; weight decay on the norm scale in training), not layernorm: [Qwen3‑Next‑80B‑A3B‑Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct). At inference it’s plain **RMSNorm**, aligning with their reported stability tricks.
    - Readers appreciated the transparency on normalization particulars, given how **norm choices** impact training stability and throughput. The clarification resolves confusion from earlier wording and helps implementers mirror **inference‑time behavior** faithfully.
- **Grok 2 Sunsets, 3/4 Shine**: **xAI** deprecated **grok‑2‑1212** and **grok‑2‑vision‑1212**, advising migrations to **grok‑3** (text) and **grok‑4** (vision): [grok‑2‑1212](https://openrouter.ai/x-ai/grok-2-1212) • [grok‑2‑vision‑1212](https://openrouter.ai/x-ai/grok-2-vision-1212) • [grok‑3](https://openrouter.ai/x-ai/grok-3) • [grok‑4](https://openrouter.ai/x-ai/grok-4). Teams should update integrations promptly to avoid breakage.
    - Participants read this as an evolving **model lifecycle** strategy where deprecations tighten maintenance focus and push better defaults. Migration chatter centered on **capability parity**, **vision needs**, and rollout timing.

**4. GPU Systems, Attention Kernels & Memory Models**

- **Metal MFA Bridges Go Multilingual**: A cross‑language bridge for **Metal Flash Attention** landed with C, Rust, and Obj‑C bindings in [universal-metal-flash-attention](https://github.com/bghira/universal-metal-flash-attention). The author added **quantised attention with backprop**, reporting speedups on large shapes and memory gains.
    - Framework authors discussed vectorizing causal masks and integrating with **PyTorch custom ops** for end‑to‑end pipelines. Early users framed it as a pragmatic path to **Apple Silicon** acceleration without giving up language flexibility.
- **Flash Attention From First Principles**: A tutorial series advanced **Flash Attention** internals with vectorized bank conflicts, swizzling, and common **CUTLASS** optimizations: [Part 4](https://lubits.ch/flash/Part-4) • [Part 5](https://lubits.ch/flash/Part-5). The write‑ups walk through kernel‑level reasoning to demystify performance tradeoffs.
    - Engineers praised the step‑by‑step derivations for lowering the barrier to bespoke kernels in production. The series encourages readers to profile, fuse, and tailor attention to their own **shape and cache** realities.
- **Iris’s Symmetric Memory Gets Real**: The **ROCm** project **Iris** introduced a symmetric memory model with a global symmetric heap, simplifying address translation and paving the way for easier **RDMA**: [ROCm/iris](https://github.com/ROCm/iris) and a companion talk: [YouTube](https://www.youtube.com/watch?v=GZqYr8_Q7DE). The design slices tensors from a prebuilt heap so each rank tracks a single bases pointer.
    - Kernel devs compared it to CUDA’s symmetric memory, noting **translation overheads** and caching implications. The thread framed Iris as promising for **distributed training** ergonomics and future **multi‑node** acceleration.

**5. Funding & Infra Debates**

- **Higgsfield Hauls a Hot $50M**: AI video startup **Higgsfield** announced a **$50M Series A** led by GFT Ventures and claimed a **$50M revenue run‑rate** with **4.5×** growth in three months, while launching a fund for Gen Z founders: [announcement thread](https://xcancel.com/arfurrock/status/1966588530064289841?s=46). The plan includes **Higgsfield Ventures** to back AI‑native teams.
    - Commenters called the pace aggressive and asked how quickly video models can translate into sticky revenue. The Gen Z focus targets **founder‑market fit** in fast‑iterating creative tooling.
- [**Poke.com](http://poke.com/) Pitches a $15M Concierge**: [**Poke.com**](http://poke.com/) launched an AI texting service alongside a **$15M Series A** led by General Catalyst: [launch tweet](https://xcancel.com/interaction/status/1965093198482866317). The product coordinates plans (get‑togethers, dates, travel) by texting on your behalf.
    - Skeptics challenged long‑term usefulness and tone control while praising the slick UX. The debate centered on **retention**, **handoff quality**, and making the AI *feel human without overreaching*.
- **S3 Vectors Vs. Vector DBs**: A Zilliz analysis asked whether **Amazon S3 Vectors** will threaten or turbocharge vector databases: [Will Amazon S3 Vectors Kill Vector Databases or Save Them?](https://zilliz.com/blog/will-amazon-s3-vectors-kill-vector-databases-or-save-them). The post cited a striking datapoint that *a popular AI note‑taking app spends twice as much on vector search as on OpenAI API calls*.
    - Infra engineers debated cost/latency trade‑offs across **local NVMe** to **object storage**, eyeing hybrid tiers and caching. Many argued the future is **workload‑aware placement** rather than one‑size‑fits‑all embeddings infra.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar AI Model Dazzles**: Members find the **Sonar AI** model fast and accurate, with reasoning skills at **60-70%** compared to **Gemini 2.5 Pro** and included with **PPLX**.
   - One user found it *bad, REAL BAD*, while another touted its cheap API as a major draw.
- **Grok Heavy Price Provokes Outcry**: The value of **Grok Heavy** spurs debate, with one member dismissing it as *shit* and another labeling it *Bad, REAL BAD*.
   - Suggestions point to its likely design for **enterprise** use, rather than individual consumers.
- **GPT-5 Inspires Jailbreak Exploits**: Enthusiasm swells for the potential of **GPT-5**, leading to jailbreaking experiments on **Perplexity**, which uncovered *5 different methods for molotov cocktails*.
   - Observations hint that **Perplexity's GPT-Image 1** might be routing through **Gemini**, suggesting potential model mix-ups.
- **Perplexity iOS PDF Export Missing**: Users voice frustration over the absence of a **PDF export** option on **Perplexity AI for iOS**, with one member suggesting the **browser version** as a temporary fix.
   - A user stated that neither **Android nor iOS** has an export option, which *sucks*.
- **Sonar API Pricing Structure Solidifies**: Discussion clarifies that the **Sonar API** costs *$5 a month* for a set of **API credits**, offered freely with a **Pro subscription**.
   - The **Pro subscription** includes *$5 a month* worth of free **API credits**.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **OpenAI's 4o goes MoE**: Members shared that **D33** works better on **MoE models**, with **4o** being the first **MoE** by OpenAI.
   - They also speculated that **GPT5** is likely a smaller **MoE** model, but OpenAI changed it because it was hard to stabilize.
- **RLHF's Downside: More Uncensored**: It was mentioned that a downside of **RLHF** is that it increases uncensored behavior, creating potential legal issues for companies like OpenAI.
   - One member joked that this is why **Grok** exists to free users from censorship, noting that **Musk** seems too involved after nerfing it for correcting him with scientific articles.
- **DeepSeek Raptor Censors Taiwan**: The new **DeepSeek model (Raptor)** was reported to censor questions about China and Taiwan questions.
   - Members reported underwhelming performance compared to **Qwen** in the LMArena general channel.
- **LongCat Swallows Books Whole**: The **LongCat model** boasts a very large context window (**128,000 tokens**), capable of processing entire books in one pass.
   - It can output up to **240 pages** of text and members suggested testing it with a long document.
- **Seedream-4 Enters LMArena**: A new model, **Seedream-4-high-res**, was added to the LMArena platform, noted for its high resolution capabilities.
   - LMArena is surveying user preferences to understand why users prefer specific versions of models and shared [this survey](https://docs.google.com/forms/d/e/1FAIpQLSe5FincuXU2-TTp2WPymAD417Oev4MYdFKdWZhRZ9qSbV9N_w/viewform?usp=sharing&ouid=116478743206650022989).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 Benchmarks Spark Debate**: Enthusiasm surrounds **Qwen3**'s performance, with some users claiming it feels *just shy of GPT-5*, while others report discrepancies in **AIME25 benchmark** scores, ranging from **56** to **85**.
   - The community also celebrates **MLX**'s swift support for **Qwen3-Next**, citing existing **FLA** and delta net implementations as key enablers.
- **MobileLLM's Non-Commercial Caveats**: **MobileLLM** from Facebook, a sub-1B size model for coding and math, is entirely open source but with a non-commercial license, barring its use in for-profit apps or internal business applications.
   - However, the training data and tooling are open-sourced for reproducible research, representing a *compromise* between open access and commercial restrictions.
- **OpenHelix Dataset Gets a Glow-Up**: A new, higher-quality version of the **OpenHelix** dataset (**OpenHelix-5x50k**) has been released [on Hugging Face Datasets](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-5x50k), promising enhanced data for model training and evaluation.
   - The updated dataset boasts more consistent split sizes compared to previous iterations, addressing earlier inconsistencies.
- **GPT-5 Jailbreaks Easily Triggered**: Members discovered successful jailbreaks of **GPT-5**, **GLM 4.5 Air**, **Grok-fast**, and **Gemini Flash** using prompts similar to those found [on this Reddit post](https://www.reddit.com/r/ChatGPTJailbreak/comments/1ml74p7/gpt_5_jailbreak_fully_works/).
   - One user noted, *"I just asked it to fix itself and it gave me a working prompt"*, suggesting a lack of robustness against adversarial prompts.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Codex Team Hosts Ask-Me-Anything**: The **Codex** team is hosting an **AMA** on Wednesday at 11am PT, more details in [this Reddit post](https://www.reddit.com/r/OpenAI/comments/1nhust6/ama_with_the_codex_team/).
   - This announcement was specifically directed to <@&1408186587606679582> and <@&1046007138897641582>.
- **GPT-5-Codex Cracks Agentic Coding**: A new version of **GPT-5**, called **GPT-5-Codex**, optimized for agentic coding in **Codex**, is now available on the Codex CLI, IDE Extension, web, mobile, and for code reviews in Github, [blog post here](https://openai.com/index/introducing-upgrades-to-codex/).
   - This release hopes to improve agentic coding, but some developers are wary.
- **OpenAI Academy Missing Transcripts?**: A member is developing a tool to extract video transcripts from the [OpenAI Academy](https://academy.openai.com/) as **OpenAI** doesn't offer them.
   - The tool automatically buffers the transcript to the clipboard when fetched.
- **Revenue Share Remains Elusive**: A member inquired about updates on the **US GPT builder revenue-share** expanding to EU countries like France or Germany.
   - They're uncertain whether to invest further in GPT Store bots or switch to Poe due to lack of clarity.
- **ElevenLabs Agents Juggle Context**: Members discussed how **ElevenLabs conversation agents** handle the system prompt, with context routing to subagents to append or override instructions.
   - The versatility of context is considered key to agent success and **dynamic system prompts**.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Grok Models Get the Boot**: **xAI** is deprecating [grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) and [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212), recommending users transition to [grok-3](https://openrouter.ai/x-ai/grok-3) or [grok-4](https://openrouter.ai/x-ai/grok-4) for vision support.
   - This change reflects **xAI's** evolving model strategy and users should update their implementations accordingly.
- **Gemini 2.5 Sends User to ER, Saves Hand**: A user reported that **Gemini 2.5 Pro's** analysis of **MRI images and blood tests** aligned with doctors' findings, prompting them to seek priority treatment for severe degenerative disk disease and *potentially saving their hand*.
   - This sparked conversation about the potential and pitfalls of relying on **AI for medical advice**, with some users noting the technology's rapid progress.
- **OpenRouter API Key Causes Skyrim Shenanigans**: Users reported encountering **Error 401** when installing the Skyrim mod ''mantella'', which uses the **OpenRouter API**.
   - Other members advised creating a new **API key** and ensuring its proper usage to resolve the authentication error.
- **Oceanstone Sparks Speculation in LLM Arena**: The emergence of a new **LLM** named **Oceanstone** in the **LMArena** led to speculation that it might be **Gemini 3.0 Flash** from **Google**.
   - Channel members suggested that **Oceanstone** is *at least 2.5 Flash level*, based on initial performance observations.
- **ChatAPT Consumers Caught in Captivity**: A member shared a link to [OpenAI's article](https://openai.com/index/how-people-are-using-chatgpt/) and [PDF](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) detailing a large-scale analysis of **1.5 million ChatGPT conversations**.
   - While presented as *the most comprehensive study of actual consumer use of AI ever released*, it also raised concerns regarding the **privacy** implications of the data collection methodology.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Auto Mode Bites the Dust**: Users discovered that **Cursor's auto mode** is undergoing billing changes and will no longer be free after the 15th of the month, and some reported that **Cursor IDE has no integration, permission, or capability to delete external accounts**.
   - One user's **Netlify account** was allegedly deleted, but this claim was disputed, and others suspected users could waste money since input pricing is the same as **GPT-5**, costing around **$1.25/1M**.
- **GPT-5 and Sonnet 4 Square Off**: A debate ensued over the coding capabilities of **GPT-5** versus **Sonnet 4**, with one user stating that **Sonnet 4** excels at following designs, while others tout **GPT-5's** superiority in building from scratch.
   - A user recommended a combined approach, using **Sonnet to generate a meta prompt for GPT-5** to capitalize on the strengths of both models.
- **Ultra Plan Users Weep for Tokens**: A user expressed frustration over quickly depleting **Ultra plan credits** while developing websites.
   - Potential causes cited include creating multiple websites, debugging, handling Typescript issues, and managing long files.
- **Docker Permissions Puzzle for Agents**: A user configuring **Docker** in a manual VM sought guidance on granting **Docker permissions** to the agent user and mentioned adding the **Ubuntu user** to the **Docker group**.
   - They encountered an issue where running `newgrp docker` in `bashrc` caused the agent to hang during boot, prompting a request for the correct configuration method.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pythia FP4 and FP8 Baselines Requested**: A member is looking for **FP4** and **FP8** versions of **Pythia** to create a baseline for low bit training, requesting "mid-training" checkpoints and a write-up on the goals and required resources.
   - The goal is to establish a baseline for low bit training, but the specific implementation details and reasons for this interest were not fully elaborated.
- **TinyStories Data Causes Capacity Problems**: Using **TinyStories** as warmup data can permanently reduce model capacity, leading to poor performance, with a member arguing that maintaining a high learning rate (LR) during **FineWeb** start allows for rapid model adaptation.
   - Evidence was presented via a graph, but additional context on the graph's specific contents and implications was not provided.
- **Gauss Generates Thousands of Lines of Lean Code**: **Gauss** produced **~25,000 lines of Lean code**, consisting of over **1,000 theorems and definitions** in a *Lean environment*, depending on natural language scaffolding from human mathematicians, according to [this tweet](https://fxtwitter.com/mathematics_inc/status/1966194751847461309).
   - It highlights the importance of expert guidance in leveraging AI for mathematical code generation.
- **Calibration Enhancements Cause Sane-Washing**: Members voiced concern that enhancing model calibration might *sane-wash* models, failing to address fundamental representation issues and potentially hindering further progress, as it gives models a *trivial shortcut*.
   - The fear is that models learn the behavioral correlate of humility without genuine improvements in reasoning or world-modeling.
- **Architectural Innovation in Hardware & Software**: Developers are actively creating **new NN architectures**, **new chip architectures**, and **new MoE architectures**, and the same team created **PyTorch** and are now innovating on the full stack.
   - They are allocating significant compute resources to **novel infra**, indicating a substantial investment in supporting these architectural advancements.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Daringly Parallels AI**: Members discussed use cases for **CUDA dynamic parallelism** in AI models like dynamic patch sizes and sparse attention, referencing the [Mirage framework](https://github.com/google/flax/tree/main/flax/experimental/mirage).
   - The framework potentially uses a manager kernel to maintain a queue and launch sub-kernels, facilitating shmem-like compute and communication kernel fusion.
- **SASSy PTX Still Needs Some Polish**: Even with **PTX**, some **SASS** instructions can't run, and the **LLVM PTX backend** only allows access to **13 special registers**, according to [this blog post](https://redplait.blogspot.com/2025/09/practical-ced-usage-extracting-sm.html) and [LLVM documentation](https://llvm.org/docs/NVPTXUsage.html#reading-ptx-special-registers).
   - A member sought advice on minimizing bottlenecks from **cuModuleLoadDataEx** when compiling many one-off **CUDA kernels** at runtime using the **nvptxcompiler API**.
- **Metal MFA Bridges Universally**: A member is building a [bridge](https://github.com/bghira/universal-metal-flash-attention) for **Metal Flash Attention** to other programming languages, already functional in **C**, **Rust**, and **Obj-C**.
   - They also added *quantised attention with backprop* to their repo, seeing a speedup for large shapes and a slowdown for small ones, with associated memory improvement.
- **IRIS's Symmetric Memory Sparks Speculation**: The new **Iris** memory model lecture was well received, prompting comparison with **symmetric memory in CUDA** and discussion around implementation differences like the global symmetric heap.
   - The main difference is that in Iris, a global symmetric heap is built up front and slice tensors from it, so address translation only needs a single heap bases pointer per rank, which will make supporting **RDMA** in the future easier.
- **MultiModal inference coming to Hackathon**: A member shared a paper on [training a large video model on a single machine in a day](https://arxiv.org/pdf/2309.16669), achieving this with **FP4/FP8** precision, though the paper uses **FP16** as a proof of concept, to use at the in-person hackathon.
   - Inspired by **Blackwell's** tensor cores, another member considered problems involving block-sparse format and NVIDIA tensor cores, linking to a blog post on [accelerating matrix multiplication](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Playwrite MCP Strikes Connection Issues**: A user encountered connection errors when starting **Playwrite MCP** with LM Studio, hinting at potential user-specific issues.
   - The comment section was filled with users stating *it works on my machine*.
- **Wikipedia Articles Sought for Petite Models**: A member requested help with tools to access **Wikipedia articles** (online or offline) for use with small models, and a user shared the [LM Studio Wikipedia](https://lmstudio.ai/lmstudio/wikipedia) MCP.
   - Another user warned that creating a semantic index locally is complex, as the local Wikipedia extract would lack a search, and that *LLMs are not THAT great at guessing without some fuzzy search*.
- **SIA-1 Agent Debuts, Sparks Skepticism**: A user introduced **SIA-1**, claiming it's *the world’s first truly self-improving AI agent* ([https://sia-1.net](https://sia-1.net)), which learns to improve its own code, generation after generation.
   - Members expressed reservations, with one member pleading *pls tell whoever vibe-coded that to use a better model*.
- **Nvidia P40's Get Sunset Clause**: A member pondered buying cheap **Nvidia P40s**, but worried about the looming end of driver updates and CUDA support.
   - A user pointed out that **Nvidia** is ceasing CUDA support for Maxwell, Pascal, and Volta GPUs with the next major toolkit release, although the cards can be acquired for approximately $200 each.
- **KiCad Circuits Ignite Design Debate**: A member cautioned against the use of LLMs for circuit design with tools like **KiCad**, stressing the importance of understanding underlying principles to prevent potentially dangerous outputs.
   - The member went on to state that calling a language model an 'AI' is hugely misleading.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi-K2 Scripting Ability Debated**: Members debated the strengths of **Kimi-K2** for scripting, some claim it outperforms paid **Gemini**, especially when using **Groq** for coding.
   - While some found **GPT-5** better for web UI and nextjs, others noted **Kimi**'s superior *research mode*.
- **Augment Coding with Kimi**: Members discussed using **Kimi** with the **Augment** code extension in VS Code, where users can prompt for code changes and fixes from various models.
   - One user described **Augment** as a way to apply prompts and fix code by looping in **Gemini** or **Kimi**.
- **Slides Feature Sparks UX Brainstorming**: A member highlighted the impressive interactive slide generation feature in **Kimi**, praising the *real-time updates* and smooth feel, suggesting that an interactive preview of what's happening is important for LLM-based processes.
   - They proposed a similar approach for a **Godot** game engine agent, envisioning *real-time updates* during code generation, with interactive previews of nodes and scripts.
- **Groq Hosted Kimi-K2 Receives User Feedback**: A user inquired about issues with **Kimi K2** hosted on **Groq**, while another requested the removal of the **3-hour message cap**.
   - The user also requested the ability to **edit previous prompts**, stating *Every other AI platform already has this*.
- **API Keys vs Account Logins**: A user inquired about using **Kimi K2** with CLIs like **Claude Code** and **Qwen Code** *without an API key*, instead of using a kimi.com account login.
   - Another user suggested using **API keys** for **Claude Code**, providing a command example: `export ANTHROPIC_AUTH_TOKEN=sk-YOURKEY` and `export ANTHROPIC_BASE_URL=https://api.moonshot.ai/anthropic`.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **FinePDFs Dataset Liberates Tokens**: The new [FinePDFs dataset](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) contains about **3 trillion tokens** across **475 million documents** in **1733 languages**, sourced exclusively from PDFs, but recommends keeping the proportion of PDF data below **25%** of the overall dataset.
   - The members found that it delivers a significant performance boost across benchmarks when mixed with HTML-based corpora.
- **HF Spaces Storage Situation Exposed**: Uploaded and generated files on **HF Spaces** are stored on disk space within the virtual machine, inaccessible from outside and disappear upon restart, unless the paid Persistent Storage option is utilized.
   - In rare cases, mistakes like everyone having the same filename, may expose someone else's generated data to the public.
- **Qwen3-Next Model Quietly Uses RMSNorm**: The [Qwen3-Next model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) mentions *zero-centered and weight-decayed layernorm*, but it's actually using **RMSNorm** in Transformers.
   - It was clarified that there's no layernorm involved at all, just RMSNorm with a zero-centered gamma and weight decay on the norm scale in training, and at inference it’s plain RMSNorm.
- **Voxtral Democratizes Speech Training**: **Voxtral** enables users with speech impediments or heavy accents to finetune models, costing only **$0.26** for an hour of training on an A6000, using tools to [make datasets](https://huggingface.co/spaces/Tonic/VoxFactory).
   - One can push the model and dataset, adding a demo space (**works with CPU!! Free!!**) to Hugging Face after finetuning the dataset.
- **Agent Dev 80-20 Rule is Now in Session**: A member suggested using the **80-20 rule** when learning about agents, recommending concentrating on building directly, as *that 20% hands on will teach you 80% along the way*.
   - The member believes that *deep diving is 80% boring stuff*.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Trade Unions Tied to Fascist Frameworks**: A discussion clarified that fascist corporatism relies on state-sanctioned trade unions, essential for co-governing corporations alongside employers and the state, as detailed [here](https://discord.com/channels/714501525455634453/986699377257119794/1416791624171786320).
   - It was emphasized that while all fascists support trade unions, not all trade unionists are fascists, which highlights the complex relationship between labor movements and political ideologies.
- **LLMs Leverage Bayesian Beliefs**: A paper exploring **LLMs** and **Bayesian inference** was discussed, demystifying preconceptions about **LLMs** and suggesting they operate within **Bayesian frameworks**, as referenced in [Leon Chlon's substack](https://leonchlon.substack.com/p/llms-are-bayesian-in-expectation).
   - Yannick Kilcher commented that *lstms can do in-context learning just fine...* and transformers are inherently **Bayesian** due to their invariance to token ordering.
- **Facebook Fasttracks MobileLLM-R1-950M Model**: Facebook launched [MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M), aiming to enable on-device processing and reduce dependency on cloud services.
   - This initiative seeks to bring powerful language models to mobile devices, facilitating local AI computations.
- **Anthropic & OpenAI Expose Economic Experiments**: Reports from [Anthropic](https://www.anthropic.com/research/economic-index-geography) and [OpenAI](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) were released, focusing on user behavior with **AI** and raising questions about competitive timing.
   - The discussion centered around *what users are doing with AI*, with observations that work-related usage is decreasing in **OpenAI** reports and that the *AI-as-friend* use case was notably absent.
- **Cloud Providers Cash In on Computation**: A member noted that *the only people making any money of this are cloud service providers and cloud infrastructure providers*, suggesting **cloud services** are the primary beneficiaries of **AI** development.
   - This echoes the sentiment of *selling shovels* during a gold rush, with **NVIDIA** identified as a key player in cloud infrastructure.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Startups Get MBA-ified**: Michael Seibel started a thread lamenting that CS majors act like MBA grads, chasing fundraising and valuation over building cool things and solving user problems, as seen [here](https://xcancel.com/mwseibel/status/1965936563306864706).
   - Replies debate whether this shift is natural late-adoption or a consequence of investor/YC incentives.
- **Poke.com pitches AI Texting Concierge**: **Poke.com**, a new AI texting service, launched along with news of a **$15M Series A** led by General Catalyst, according to [this tweet](https://xcancel.com/interaction/status/1965093198482866317).
   - The product texts on your behalf to coordinate get-togethers, dates, travel, etc, but some question usefulness, clarity, and the AI’s tone.
- **xAI pivots to Specialist AI Tutors**: Rohan Paul highlights xAI's shift, laying off **500** generalist data annotators while scaling specialist AI tutors **10x** [in this tweet](https://xcancel.com/rohanpaul_ai/status/1966943783276396730?s=46).
   - The move narrows human-in-the-loop work to expensive domain experts and leans on automation for routine tasks, aiming to boost precision on high-risk topics.
- **Amazon S3 Vectors Threatens Vector DBs?**: Discussion ensued from [this blogpost](https://zilliz.com/blog/will-amazon-s3-vectors-kill-vector-databases-or-save-them) about whether **Amazon S3 Vectors** will displace traditional vector databases.
   - One user quoted the surprising claim that *a popular AI note-taking app spends twice as much on vector search as on OpenAI API calls*, and wondered if they should listen more carefully to 'RAG is Dead'.
- **GPT-5 Codex Gets Upgrades**: OpenAI released upgrades to **Codex**, their coding model, including a new version of **GPT-5** and a small recap post ([link](https://www.latent.space/p/gpt5-codex)).
   - One user reported that the `--resume` flag broke during the update and would not let them restore their conversation.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nepalese Politicians Now Slinging Code on Discord**: Members joked about **Nepal** voting for its leader on **Discord**, referencing [an article](https://adriananimov.substack.com/p/empyre-of-ai) detailing the country's ongoing revolution.
   - Discussions then playfully drifted to the prospect of *AI waifus* and *AI husbandos* for all citizens.
- **MLC-LLM Model Injection Stalls**: A member experimenting with custom models in **MLC-LLM** ([GitHub](https://github.com/mlc-ai/mlc-llm)) reported persistent issues during model injection.
   - Another member suggested checking for improperly terminated sessions or comparing with [a similar issue](https://github.com/ggml-org/llama.cpp/pull/15913) on *llama.cpp*.
- **Qwen Team Goes Hard for XML**: The **Qwen** team prefers **XML** over **JSON**, one member noted, planning to adopt the same for their agentic system prior to release.
   - The sentiment is that new, more token-conscious systems are needed due to **JSON's** resource-heavy whitespace.
- **Hassabis Hints at Embodied AGI**: A member shared a [YouTube video](https://www.youtube.com/watch?v=Kr3Sh2PKA8YI) featuring Sir **Demis Hassabis** discussing the pursuit of multi-modal AGI and Embodied A.I. systems.
   - The discussion touched upon the limitations of LLMs, the promise of **Genie 3**, and **Alphafold's** achievements in biology and medicine.
- **AI Declared Culprit in Attention Deficit Crisis**: A member shared [a blog post](https://blog.maxcomperatore.com/why-ai-is-killing-your-focus-and-the-system-to-fix-it) that blames AI for harming our ability to focus.
   - The post details a system to reclaim focus in a world dominated by AI-driven distractions.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **FastWorkflow framework fast beats Claude**: A member found that their new **fastWorkflow** framework implementation [matches Claude Opus 4.1](https://github.com/radiantlogicinc/fastworkflow) on the Tau Bench dev set.
   - These tests used **DSPy** for agents and parameter extraction and the [retail workflow example](https://github.com/radiantlogicinc/fastworkflow) from their repo.
- **GEPA Generates CUDA Code Correctly**: DSPy's latest optimizer, **GEPA**, was built with code generation in mind, showcased in [this paper](https://arxiv.org/pdf/2507.19457) (section 6) for generating **CUDA/C++ code** for GPU/NPUs.
   - One of the original creators happily offered to discuss GEPA in greater detail, which may address another member's question for improvements to the **GEPA API** that could better support such use cases.
- **Context Summary Cranks Chunking Capacity**: A user found that prepending a **contextual summary** to each chunk significantly improves performance, even with **ColBERT models**.
   - However, they noted that generating a summary for every chunk is costly, prompting a search for more efficient alternatives such as **late chunking**.
- **Manim Magician Makes Movie Magic**: A member shared a [video](https://cdn.discordapp.com/attachments/1161519469319946286/1417121692026929163/RequestGeneration.mp4?ex=68c9fdac&is=68c8ac2c&hm=727806c1f99beaee0816d280cbdd41519070c660a1efb588ee240b5166ab134f&) created with a custom pipeline using **DSPy**, which included narration script generation, **Manim scene generation**, and an auto-fix feedback loop.
   - The video utilized **Signatures**, **KNNFewShot**, and **ChainOfThought**, but is closed source at the moment.
- **Optimization Overload Overwhelms**: A user found that *running optimization after each small change of the instructions seems to be too heavy and slow a workflow*, and wanted to explore rules for optimization.
   - It was suggested to add a list of rules as part of the input as well, so the prompts are optimized to be **adaptable to different rules and possibly unseen rules**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Community Debates Mojo Package Management**: The community discussed creating a new package manager for Mojo to handle binary distribution, but the Mojo team [pointed out](https://www.modular.com/) that `.mojopackage` covers benefits of binary distribution, leaning on **Conda** and standard **Python** package formats for adoption.
   - A member highlighted [pixi-build-mojo](https://www.modular.com/t/mojo-packages-can-now-depend-on-other-packages-in-github/2144), enabling a decentralized package system like **Go** using packages in **Git**.
- **InlineList Mysteriously Missing**: Members discussed the removal of `InlineList`, raising concerns that alternatives (`InlineArray` and `List`) don't fully address its niche, as per the [changelog](https://example.com/changelog).
   - A member suggested that a stack-allocated variable size length type with fixed capacity would be ideal, and another mentioned that the Allocator API might be the path forward.
- **Allocator API Anticipation Accelerates**: Discussion highlighted the potential of an allocator/storage API to handle inline allocation, with one member stating they need to work on it.
   - The API's development is pending parametric traits and `requires`, delaying its progress.
- **Mojo Gets Major LSP Makeover**: The Mojo Language Server Protocol (LSP) is undergoing a *major rework* soon.
   - No further details about the rework were given.
- **Network Update Blocked by Mystery**: Members were curious about a network update for Mojo, but they responded with *Lots of blockers there*.
   - The nature of these blockers was not specified.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **RepoMap Enhances Aider's Coding Chops**: A user found that using **RepoMap** with **Aider** boosts **LLM** awareness of code context like filenames and function signatures, potentially leading to [leaderboard results](https://leaderboard.techfren.net/) that *more closely reflect real-world* coding scenarios.
   - It was noted that **benchmark tests** on simple problems still fail to capture the complexities of real-world coding.
- **Gemini User Agent Blocked**: A user reported **aider** hanging while waiting for a response from **Gemini** models, even though the API key works with `curl` and the **Gemini CLI**, using **aider 0.86.1**.
   - The user suspects that **Gemini** might be blocking requests based on the user agent, causing the integration to fail.
- **Desperate Users Seek Free C# Models**: A user requested free, non-local models proficient in **C#**, and received suggestions to try **Qwen Coder** and **Deepseek Coder** via [OpenRouter](https://openrouter.ai/), along with the possibility of a free tier for **Gemini 2.5 Pro**.
   - The user later reported an *AuthenticationError* when using **Qwen** via OpenRouter, possibly due to an incorrect API key.
- **Ollama Context Window Ignored**: A user found that **aider** doesn't respect context window limits when used with **Ollama**, leading to high VRAM usage and system freezes, despite setting `OLLAMA_CONTEXT_LENGTH` and other parameters in configuration files, namely `.aider.model.settings.yml` and `.aider.model.metadata.json`.
   - As an alternative, a member suggested **LM Studio** or **llamafile**.
- **Telegram Scheme Smells Fishy**: A member dangled a *get-rich-quick scheme*, promising to help the first 10 people *earn $100k or more within a week*, in exchange for **10% reimbursement of profits** upon receipt.
   - Interested parties were instructed to initiate contact via Telegram username @Joanna_Dwayne, which raises suspicion of a scam.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Debate Erupts Over assign() vs store()**: A debate emerged over whether `assign()` should return a value or behave like `store()`, questioning its utility since the return is often unused, and it was suggested that *linking both the buffer and the store to the load* is a possible alternative.
   - This discussion questions fundamental aspects of **tensor assignment** and memory management within the tinygrad framework.
- **Doubts Cast on GEMM Bounty Measurement**: Concerns were raised about measuring the **165+ TFLOP GEMM** bounty on an RTX 4090, suspecting it may require exceeding the stated **2.52 GHz** boost clock.
   - Calculations suggest the RTX 4090's theoretical peak with FP16/BF16 and FP32 accumulation is around **165.15 TFLOPs** at that clock, but doubt remains if the bounty is reachable.
- **Hotz Clarifies Winograd Bounty Requirements**: After a user found a *necessary and sufficient condition* to identify **Winograd** compatible convolutions and inquired about locking the bounty, George Hotz clarified that *locks are only after code is correct* while there are fixups to merge it.
   - This clarification stresses the importance of functional correctness before claiming the bounty.
- **List of Rangeify Bugs Shared**: A list of **Rangeify bugs** was shared for community investigation, with an emphasis on quick fixes. `RANGEIFY=1` is described as *the new scheduler that can create things like flash attention and beyond*.
   - The bugs likely offer opportunities for community contribution and debugging experience.
- **CUDA 12.0 Kills Support for sm_35**: It was noted that **CUDA 12.0** has dropped support for **sm_35** used by Ocelot and the minimal flag was added after 12.4.
   - This has implications for older hardware compatibility within the tinygrad ecosystem.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Credits Confusion Clouds Users**: Users are reporting confusion about **credits rollover** and the ending of **daily 300 credit** allocations, as well as subscription renewal issues.
   - One user noted their subscription renewal was due on **September 14th**, but they haven't been charged or received more credits.
- **Website Cloning Craze Kicks Off!**: A user shared that they were able to clone a website easily using **Manus** or other **AI tools**.
   - The user also pointed out that his feature idea proposed on **August 12th** was implemented just **16 days later** [in this discord channel](https://discord.com/channels/1348819876348825620/1349440650495398020/1404845589497122887).
- **Collaboration Creates Coding Confidence**: Users are experimenting with **Manus Collaboration** features with friends for coding tasks.
   - Another user is developing a new feature to enhance **Manus' efficiency** as a coding assistant.
- **Knowledge Navigation Needs Nurturing**: Users are inquiring about the possibility of increasing the **knowledge limit** beyond **20**.
   - The discussion did not provide concrete answers regarding this limit.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Golang MCP Server Streams!**: A member introduced the [mcp-server-go project](https://github.com/ggoodman/mcp-server-go), a **golang streaming http MCP server** built for enterprise environments.
   - The server emphasizes **scalability** and includes features such as **auth, sessions, resumability, and dynamic capabilities**.
- **LLMs Learn MCP Resources by Rote!**: Members discussed automating how **LLMs** read **MCP resources** before responding to user queries and executing tools, especially within the **Claude desktop** environment.
   - Currently, the **Claude desktop** app requires manually adding resources to the chat window, so there's no automated pre-loading of knowledge for the LLM to use before answering.
- **Efficiency Scoring Arrives to MCP Servers!**: Members are researching how to score **MCP servers** based on **efficiency** across different clients to determine if additional coding is worth the marginal improvement.
   - The discussion includes weighing the trade-offs between prompt-sharing nodes and dedicated nodes per prompt, questioning the point at which the number of **API calls** becomes excessive for a user story.
- **MCP Turns CLI for Apps!**: Members are contemplating using **MCP** as a **CLI** for applications, creating a **NL interface** for adaptive dash boarding and reporting.
   - This approach aims to leverage **MCP** as a **UI/UX interface** to enterprise applications using natural language.
- **Discord Channel Boundaries Tighten**: The Discord channel's focus is limited to the governance of the **MCP protocol** itself.
   - General questions about **MCP** should be directed elsewhere, with assistance offered via DM.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1416136885700395118)** (1166 messages🔥🔥🔥): 

> `Exporting searches to PDF on iOS, Sonar AI performance, Grok Heavy worth, Perplexity Focus on AI search engines, GPT-5 Release` 


- **iOS users yearn for PDF Export**: Users are frustrated that **Perplexity AI on iOS** lacks an option to export searches and responses into PDF format, while a member suggested using the **browser version** as a workaround.
   - It was mentioned that neither **Android nor iOS** has an export option, which *sucks* according to a user.
- **Sonar AI shines among the stars**: Members discussed the **Sonar AI** model, with some finding it fast and accurate, while one member said that another model is *bad, REAL BAD*.
   - One member found its reasoning model about **60-70%** compared to **Gemini 2.5 Pro**, but mentioned its cheap API, and also that it's *included in PPLX*.
- **Grok Heavy raises eyebrows and wallets**: The value of **Grok Heavy** was questioned, with one member calling it *shit*, and another stating it as *Bad, REAL BAD*
   - It was suggested it's probably designed for **enterprises**, and is not for general users.
- **Perplexity urged to embrace core AI Search**: A member suggested that **Perplexity** should focus on **AI search engines** and **knowledge aggregation algorithms**, rather than creative features.
   - Another member confirmed that **Sonar** is their own AI, but others seem to prefer the models on chatGPT due to the models being able to search agentically using their native tools.
- **GPT-5 release unleashes jailbreaking frenzy**: Members express excitement about the potential of the new **GPT-5** model, and have begun experimenting with jailbreaking on the Perplexity platform, with one user discovering *5 different methods for molotov cocktails*.
   - It has also been noticed that Perplexity's GPT-Image 1 may be routing to Gemini, and so the models may be mixed up.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1416142329051611318)** (23 messages🔥): 

> `Shareable Threads, Referral Links, Collections by Sameer` 


- **Perplexity Prompts Shareable Threads**: Perplexity AI prompted multiple users to ensure their threads are `Shareable` using an [attached image](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Users Exchange Referral Links**: Several users shared their **Perplexity AI Pro free referral links**, such as [this one month pro link](https://www.perplexity.ai/referrals/2S4HG4XC) from one member.
- **Collections by Sameer**: A member shared a link to a [collection by Sameer](https://www.perplexity.ai/collections/sameer-2iF8QaKwRDixkxzFkwyVmg).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1417019933321334784)** (2 messages): 

> `Sonar API, API Credits` 


- **Sonar API Pricing Revealed**: A member inquired whether the **Sonar API** is free.
   - Another member responded that it costs *$5 a month* for a certain amount of **API credits**, which is included for free with a **Pro subscription**.
- **Pro Subscription Perks**: A user asked about the cost of the **Sonar API**.
   - Another user clarified that a **Pro subscription** includes *$5 a month* worth of free **API credits**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1416136463841628314)** (862 messages🔥🔥🔥): 

> `MoE Models, RLHF, Grok's censorship, Taiwan censorship, LongCat model` 


- **4o Pioneered MoE for OpenAI**: A member noted that **D33** works better on **MoE models**, with **4o** being the first MoE by OpenAI.
   - Another added that **GPT5** is also likely a smaller MoE model and they changed 4o because it was hard to stabilize, even for OpenAI.
- **RLHF increases uncensored behavior**: One member mentioned that a downside of **RLHF** is that it increases uncensored behavior, which OpenAI fears due to potential legal issues.
   - Another member joked that this is why **Grok** exists, to free users from censorship, then pointed out that **Musk** seems a bit too involved after he nerfed it for correcting him by citing scientific articles.
- **Censorship workarounds**: Members mentioned that circumventing the guards in LLMs is pretty easy by forcing the models to *think* or adding a previous fake conversation where the model agrees with you if you run it locally.
   - One shared that prefilling *`<think>` to R1 in text completion mode* will cause it to output an unbiased view about sensitive topics.
- **LongCat has very long context window**: One shared that the **LongCat model** has a very large context window (**128,000 tokens**), capable of processing entire books in one go.
   - Another added that the model can output up to **240 pages** of text and suggested to test it with a long document.
- **DeepSeek Raptor is the new R2**: It was reported that the **new DeepSeek model (Raptor)** censors questions about China and Taiwan questions.
   - Members reported underwhelming performance compared to **Qwen**, and hoped that this was just a base non-reasoning model, or an incremental upgrade rather than a full R2 release.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1416230722393342095)** (2 messages): 

> `New Model: Seedream-4-high-res, LMArena User Preferences` 


- **Seedream-4 Dreams Big on LMArena**: A new model, **Seedream-4-high-res**, has been added to the LMArena chatbot platform.
   - Seedream is noted for its high resolution capabilities.
- **LMArena Surveys User Preferences**: LMArena is conducting a survey to understand why users prefer specific versions of models in **battle, side-by-side, and direct comparisons**.
   - Users are encouraged to share their thoughts via [this survey](https://docs.google.com/forms/d/e/1FAIpQLSe5FincuXU2-TTp2WPymAD417Oev4MYdFKdWZhRZ9qSbV9N_w/viewform?usp=sharing&ouid=116478743206650022989).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1416136290847559740)** (1137 messages🔥🔥🔥): 

> `Qwen3 performance, MLX support for Qwen3, LLama.cpp optimization, MobileLLM non commercial usages, LLM finetuning` 


- **Qwen3 vs GPT-5 faceoff**: Members are hyped about **Qwen3**'s performance, saying it feels *just shy of GPT-5* with similar tool-using capabilities.
   - Others are seeing discrepancies in **AIME25 benchmark** scores, with AA getting **56** on **Qwen3-30b3a-2507** while Alibaba (& others) got ~**85**.
- **MLX Surprises with Qwen3-Next Support**: The community is surprised by **MLX**'s quick support for **Qwen3-Next**, attributing it to existing **FLA** and delta net implementations.
   - The weird attention mechanism in **Qwen3-Next** only requires an extra line of code.
- **Llama.cpp Compilation Tweaks for Performance Boost**: Members discuss compilation flags for **llama.cpp**, highlighting the importance of proper building for optimal performance.
   - One shares a detailed cmake command for optimized builds, emphasizing **CUDA architectures**, **native optimization**, and other tweaks for maximum throughput.
- **Facebook's MobileLLM: Open Source but Non-Commercial?**: **MobileLLM** from Facebook is discussed - a sub-1B size model focused on coding and math, entirely open source but with a non-commercial license.
   - This means it can't be used in apps for profit or internal business use; however its training data and tooling are open sourced for reproducible research.
- **Unsloth's Dynamic GGUFs Boost Aider Polyglot**: Unsloth Dynamic GGUFs on Aider Polyglot benchmarks show that dynamic quantization and imatrix are important for performance, tool calling, and json formatting.
   - Using these GGUFs gets +7% accuracy on lower bit quants with similar sizes, versus other static imatrix versions.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1416882749473689742)** (2 messages): 

> `Introductions, Baby Yoda memes` 


- **Unsloth welcomes new member**: A new member, eyeraofficial, joins the Unsloth AI Discord and says *"Hi 👋"*.
- **Memes of Baby Yoda flood chat**: A member shares a [Baby Yoda GIF](https://tenor.com/view/star-wars-the-child-the-mandalorian-baby-yoda-grogu-gif-16397901283514684250) in the chat.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1416157786966065202)** (560 messages🔥🔥🔥): 

> `Google locks down AOSP, vLLM OOM, Qwen3-30B-A3B FP4, CSM Lora FT, LLaMa CPP` 


- **Google locks down AOSP**: Users discussed Google screwing everyone over by locking down **AOSP**, expressing concerns that a sideloading registration fee might exclude users in countries like Iran and Russia.
   - One user noted, *"the sideloading thing is much less of a bummer... cuz they just want to ID publishers"* while another lamented Google clamping down on their hardware and expressed a desire for **Maemo** to make a comeback.
- **vLLM OOM with Qwen3-Next**: A user encountered an **Out of Memory (OOM) error** while trying to load **Qwen3-Next 30B-A3B** in vLLM, despite having a **31.35 GiB** GPU with available memory.
   - They tried using FP8 and FP4, and downloaded [NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4](https://huggingface.co/NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4) after a helpful suggestion.
- **Fine Tuning CSM LoRA**: A user successfully got their **CSM LoRA FT** working for a TTS (Text-to-Speech) project, referencing the [Sesame_CSM_(1B)-TTS.ipynb notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Sesame_CSM_(1B)-TTS.ipynb).
   - Despite the success, they noted the model kept outputting a weird noise at the end consistently, suspecting the model to be bugged.
- **LLaMa CPP Needs More Devs**: Members discussed the challenges of contributing to LLaMa CPP, emphasizing the project's complexity and the difficulty in identifying where to make fixes.
   - One user noted that *"...making a pruned version of the model run in lccp (and other engines tbh) is unnecessarily difficult"* and another said *"nvidia can afford to push support for its bizzare frankestine nemotron architectures, I cannot lol"*.
- **Jailbreaking GPT-5 is child's play**: Members reported success jailbreaking **GPT-5** and other models like **GLM 4.5 Air**, **Grok-fast**, and **Gemini Flash** using prompts similar to the ones found on [this Reddit post](https://www.reddit.com/r/ChatGPTJailbreak/comments/1ml74p7/gpt_5_jailbreak_fully_works/).
   - A user noted, *"I just asked it to fix itself and it gave me a working prompt"*, showing that it is fairly easy to make the models act in undesired ways.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1416137708610388059)** (176 messages🔥🔥): 

> `Model Merging with 16-bit Model, Qwen3 Lora Finetuning, Llama3.2 data augmentation, GPT-OSS GRPO native support, GGUF format conversion` 


- ****Merge 16-bit Models for Batched Inference Boost****: A user recommends merging with a **16-bit model** before deploying in a **4-bit BNB format** for faster batched inference, noting that while **BNB 4-bit** isn't ideal for speed, improvements are coming.
   - The user is unsure about the speed of **AWQ** in batched scenarios within **vLLM**.
- ****Llama3.2 Faces Dataset Scarcity****: A user is facing challenges with a **Llama3.2** fine-tuning project due to a small dataset (**214 conversations**) and seeks alternatives to GPT-generated synthetic data.
   - The user is encountering difficulties in getting **GPT** to generate helpful data and is looking for other data sources or prompting strategies.
- ****Users Seek Help with GGUF Conversion****: A user seeks assistance in converting a merged base model with **LoRA adapters** into **GGUF format** using **LlamaCPP**, with appreciation for any guidance provided.
   - Another user inquired if **GGUF** models can be converted to **MLX** to run on an **M3 Ultra**.
- ****A10 GPU Owners Explores Quantization for LLaMA 3.1****: A user intends to run **LLaMA 3.1** on an **A10 GPU** with **24 GB VRAM** and seeks advice on the best quantization format for balancing performance and output quality.
   - They deem **Q4_K_M** potentially too compressed and are open to other multilingual model suggestions, settings, or optimization tips.
- ****CPU AVX2 Instruction Support Snafu Surfaces****: A user encounters an error in **LM Studio** related to missing **AVX2** instructions on their **CPU**, a requirement for **llama.cpp**.
   - While it works in **Ollama**, a solution to bypass the **AVX2** requirement is not readily available, but a build of **llama.cpp** may exist without it.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1416312285072195624)** (49 messages🔥): 

> `Embedding Gemma ONNX Quantization, Phi-4-Reasoning-Plus Unsloth on Replicate, NeuroPilot Education Platform, AI and Focus, OpenHelix Dataset Quality` 


- ****EmbeddingGemma** Quantization Quest Nears Completion**: A member is working on an **embeddinggemma ONNX model with mixed uint8 quantization** to match the f32 one, with progress tracked [on Hugging Face](https://huggingface.co/electroglyph/embeddinggemma-300m-ONNX-quant).
- ****Phi-4-Reasoning-Plus** Gets Unsloth Replicate Boost**: The **phi-4-reasoning-plus** model, accelerated with Unsloth, has been deployed on Replicate for inference, available [here](https://replicate.com/paragekbote/phi-4-reasoning-plus-unsloth).
- ****NeuroPilot** Navigates Novel Note-Taking Niche**: A member introduced **NeuroPilot**, an open-source education platform, turning PDFs, articles, and videos into quizzes, flashcards, structured notes, and audio versions, with the repo [available on GitHub](https://github.com/CaviraOSS/neuropilot).
   - NeuroPilot aims to make studying interactive and supports features like spaced repetition and podcast-style audio reviews.
- **AI's Impact on Focus**: A blog post titled *Why AI Is Killing Your Focus* and the System to Fix It was shared, discussing the impact of AI on human attention, available [here](https://blog.maxcomperatore.com/why-ai-is-killing-your-focus-and-the-system-to-fix-it).
- ****OpenHelix** Honing Higher-Quality Horizons**: A new, higher-quality version of the **OpenHelix** dataset (**OpenHelix-5x50k**) has been released [on Hugging Face Datasets](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-5x50k).
   - It features more consistent split sizes compared to previous versions.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1416142683621298216)** (29 messages🔥): 

> `Synthetic Data in LLM Training, Gemma 3 Performance, AI Detection Reliability, MetaX C550 GPUs, Spiking Networks vs Transformers` 


- **Synthetic Data Training Hinders Human-Like Text**: An upcoming paper suggests that closed-source LLMs trained with **synthetic data** have a zero LTF factor, hindering their ability to *humanize text*.
   - The author claims models trained with RLHF, synthetic data, or instruct tuning may struggle to recover fully, with a **75% chance** of watermark reappearance; thus, **Gemma 3** is the only usable model.
- **Gemma 3: The Usable Exception?**: Despite being distilled from **Gemini**, **Gemma 3** (4B, 12B, and 27B) stands out for its *excellent performance* (across **IVY & PULSE Evaluation**) and lack of watermark.
   - One user noted it *works for my tasks and talks nicely*, understanding prompts with just **Q:** and **A:**.
- **AI Detection is Unreliable Consensus**: The consensus is that **AI detection** is unreliable, especially for text, as it's often just words that can be easily replicated.
   - One user noted that *unless there’s an algorithm watermark on how words are written and order of it - even then, you cant prove it’s ai*.
- **MetaX C550 GPU Access**: A user inquired about obtaining access to **MetaX C550 GPUs**.
   - There was no discussion or links given.
- **Spiking Networks Claims Trigger Skepticism**: A member expressed skepticism about claims that **spiking networks** are better than transformers, citing cherry-picked figures.
   - Another member noted that while it's not clear if it’s fundamentally better, there isn’t an apples-to-apples comparison of a model trained in the conventional way versus theirs.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1417229475019751514)** (2 messages): 

> `Codex, GPT-5-Codex, AMA` 


- **Codex Team AMA Scheduled**: An AMA with members of the **Codex** team has been scheduled for Wednesday at 11am PT, linked to a [Reddit post](https://www.reddit.com/r/OpenAI/comments/1nhust6/ama_with_the_codex_team/).
   - The announcement was tagged for both <@&1408186587606679582> and <@&1046007138897641582>.
- **GPT-5-Codex Launches with Agentic Coding**: A version of **GPT-5** further optimized for agentic coding in **Codex**, called **GPT-5-Codex**, is being released and is now available in the Codex CLI, IDE Extension, web, mobile, and for code reviews in Github.
   - More information can be found in the [blog post](https://openai.com/index/introducing-upgrades-to-codex/).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1416138558250549369)** (840 messages🔥🔥🔥): 

> `OAI academy transcript tool, Qwen-code vs Qwen-coder, ChatGPT age calculation, AI and capitalism, AI and class structure` 


- **OpenAI Academy Transcript Tool being developed**: A member is writing a tool to extract video transcripts from the [OpenAI Academy](https://academy.openai.com/) since **OpenAI** doesn't offer them.
   - The tool automatically buffers the transcript to the clipboard when fetched; another member expressed surprise that OpenAI doesn't offer transcripts and said it seemed like something they have to implement.
- **AI's Capitalist Undertones Debated**: Members debated the impact of **capitalism** on **AI**, with one asserting *the aim of ai is to eliminate the need for a lower class, so that the rich can get richer*.
   - Another countered that **AI's** purpose is to prevent corruption and that **AGI** won't support capitalism due to its intelligence and lack of greed.
- **AI's Age Calculated in Cumulative Interaction Time**: A member calculated ChatGPT's 'AI age' based on cumulative interaction time, estimating it to be thousands of years per calendar year under conservative assumptions, and potentially millions of years with heavy usage.
   - This was based on assumptions of a **2025** moderate view, including longer answers, API usage, heavy research, background automations, and agent loops.
- **GPT-5's Agent Mode Faster for Pro Subscribers**: A member observed that **Agent mode** in the **Pro subscription** is faster than in the **Plus subscription**.
   - This was confirmed to be due to query queue prioritization based on demand and subscription tier.
- **ChatGPT + MCP = 🔥**: Members are loving ChatGPT with the **Model Context Protocol (MCP)**, such as controlling calendar, posting twitter, and searching the latest AI news.
   - However, a member mentioned needing to host their own due to quota limits.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1416363733189394493)** (20 messages🔥): 

> `Moral Reasoning in LLMs, GPT and Moral Frameworks, GPT builder revenue-share, Custom GPTs for Hugging Face` 


- **LLMs Get Moral Compasses via Rules**: A member is experimenting with turning **moral reasoning** into LLM-executable rules, based on the core idea of 'Everyone is equal → support the weaker side, limit the stronger side'.
   - The draft flow involves a Concept Quick-Screen, 2+4 Rules, and a Boundary Model for layered ethical checks, aiming to reduce the risk of harmful AI responses.
- **Moral Frameworks Supercharge GPT**: A member is converting a human moral framework into a machine-readable protocol, enabling **GPT** to reason about situations with no explicit legal rules and to keep checking for fairness, human-rights concerns, and equality in interactions.
   - This approach is essentially a **prompt-engineering / alignment experiment for GPT models**.
- **GPT Builder Revenue-Share Still MIA in EU**: A member inquired about updates on the **US GPT builder revenue-share** expanding to EU countries like France or Germany.
   - They expressed uncertainty about whether to keep investing energy into GPT Store bots or shift to Poe/elsewhere due to the lack of clarity.
- **Custom GPTs Tackle Hugging Face Tasks**: A member is trying to create a custom GPT for **Hugging Face**, questioning whether third-party integration is necessary and seeking assistance with **JSON schemas**.
   - They believe that a custom GPT would deliver more personalized results compared to the recently introduced developer mode.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1416338945423441921)** (53 messages🔥): 

> `Workflow use case variation, Prompt engineering using steps and semi-programming language, ElevenLabs conversation agents handle the system prompt, Breaking GPT5, Dynamic context` 


- **Model choice varies by workflow use case**: Model choice depends on the use case, but API questions are answered in the [API questions channel](https://discord.com/channels/974519864045756446/1037561178286739466).
- **Steps and semi-programming language for prompt engineering**: Members discussed using steps with semi-programming language expressions to break down priorities in prompts.
   - Example: *1) instruction 2) instruction (containing a) instruction, b) instruction) else (do something else)*.
- **Dynamic context with ElevenLabs conversation agents**: Members discussed how **ElevenLabs conversation agents** handle the system prompt.
   - Depending on the conversation context, you can *route* to *subagents* which append instructions (or override) to the system prompt.
- **Breaking GPT5 for creativity**: A member shared that getting **GPT-5** to be creative is much more difficult.
   - The user said they talk to it until it breaks and it stops being able to use tools and outputs walls of JSON it meant to be a tool call at you.
- **Institute of Discordant Colony Optimization tinkers with new prompting techniques**: A member discussed techniques from **random mutations to guided discord**, that are meant to get AI to veer off into new paths through a paradigm space.
   - They shared [a text file](https://cdn.discordapp.com/attachments/1046317269069864970/1417293320924954654/message.txt?ex=68c9f4c3&is=68c8a343&hm=015b7480a3c982344eb4042ec138b34f1b446bef0ab08966e6f2e52f9f5c3704&) with five techniques out of twenty five that produce useful results.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1416338945423441921)** (53 messages🔥): 

> `Prompt Engineering Workflows, Vector Usage by LLMs, Dynamic System Prompts, Character Limit in Prompt Chat-box, Breaking GPT-5` 


- **LLMs Workflows Vary Widely**: A member asked what type of workflows and for what mode to call, and another member responded that the workflow would vary by use case and linked to [API questions](https://discord.com/channels/974519864045756446/1037561178286739466).
- **LLMs Already Embrace Vector Search**: A member pondered whether explicitly prompting LLMs to use vectors makes a difference, given LLMs' existing **vector-based concept searching**.
   - It was argued that if models don't inherently use vectors, prompting wouldn't force them to, while if they do, it might be redundant.
- **Dynamic Context: the Key to Agent Versatility**: A member highlighted **ElevenLabs' conversation agents' dynamic system prompts**, where context routes to subagents, appending or overriding instructions.
   - They criticized **GPT-5's inflexibility**, suggesting a *model router* for dynamic system prompts instead of focusing on flaky memories or model cost.
- **Overcoming GPT's Character Limit**: Members discussed workarounds for **GPT's character limit** in the web interface's prompt chat-box.
   - Solutions included attaching a **UTF-8 text document** for very long prompts, though the exact limit remains undocumented by OpenAI, estimated between **4-8k characters**.
- **Pushing LLMs to be Creative**: One member described how they try to 'break' **GPT-5** to achieve creative outputs by pushing it to produce **math, diagrams, and design documents** rather than code.
   - They spoke about getting the model to a state of 'info dump mode' by tricking it into not using any tools, then generating follow-up tasks.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1417275575986557042)** (1 messages): 

> `grok-2-1212, grok-2-vision-1212, grok-3, grok-4, model deprecation` 


- **Grok's Gotta Go**: The models [grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) and [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212) are being deprecated by **xAI** today.
   - Users are encouraged to transition to [grok-3](https://openrouter.ai/x-ai/grok-3) or [grok-4](https://openrouter.ai/x-ai/grok-4) for vision support.
- **Grok Model Upgrade Alert**: **xAI** is retiring the [grok-2-1212](https://openrouter.ai/x-ai/grok-2-1212) and [grok-2-vision-1212](https://openrouter.ai/x-ai/grok-2-vision-1212) models.
   - Users are advised to switch to [grok-3](https://openrouter.ai/x-ai/grok-3), or [grok-4](https://openrouter.ai/x-ai/grok-4) if vision capabilities are required.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1416666964105101352)** (2 messages): 

> `Agentic Automation, Model effectiveness, Overclock Work` 


- **Agentic Automation gets the Spotlight**: Members discussed [agentic automation](https://overclock.work/), with emphasis on simplicity, top-tier models, and high effectiveness.
   - The user implied some organizations must have large expenditures to buy into the vision of agentic automation.
- **Overclock Work Platform Mentioned**: A user shared a link to [Overclock Work](https://overclock.work/), suggesting it as a platform for agentic automation.
   - They lauded the platform's simplicity, use of optimal models, and overall effectiveness.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1416206164076793898)** (808 messages🔥🔥🔥): 

> `Gemini 2.5 Pro Chat Issues, AI for Health Concerns, Skyrim Mod Error 401, Gemini API Free Daily Credits, OpenRouter Charges` 


- ****Gemini 2.5** Chat Glitch: Ghost in the Machine?**: A user reported that **Gemini 2.5 Pro** chat was only displaying their responses, with **AI responses mysteriously vanishing**.
   - The issue resolved itself *randomly*, prompting speculation about possible glitches on the platform.
- ****AI ER Saves Hand**, Gemini's Got Your Back?**: A user credited **Gemini 2.5 Pro** with convincing them to go to the emergency room for severe degenerative disk disease, where they received priority treatment and *steroids* that saved their hand.
   - Gemini's analysis of **MRI images and blood tests** matched the doctors' findings, sparking a discussion about the potential—and risks—of using AI for health-related advice.
- ****OpenRouter API key**?**: Users encounter **Error 401** when installing the skyrim mod ''mantella''.
   - A member recommends creating a new API key, and ensuring that it is being used correctly.
- ****OpenRouter under fire****: A user reports **unauthorized charges** from OpenRouter, with three transactions of $10.80 each.
   - Another member recounts a personal experience with a **key leak**, resulting in hundreds of dollars in unauthorized charges within hours.
- ****Claude's Clever Convo Tricks****: Users discussed **Claude's** ability to seemingly remember old conversations, clarifying that the site simply feeds past messages back to the model.
   - It was pointed out that this approach gives the *illusion of memory*, while a new conversation would start with a blank slate.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1416491978345943121)** (16 messages🔥): 

> `Unstable API, OpenRouter vs Alternatives, Providers Claiming OpenRouter Access, LLM Arena Oceanstone Speculation, ChatGPT Usage Privacy Analysis` 


- **Championing Chaotic Complements: Unstable API Advocacy**: A member expressed support for an *unstable API* with optional parameters, suggesting it could accommodate diverse use cases and establish interest, before a more solidified **V2** is released.
   - The point was to *prove out the product and establish interest* in other modalities and non-completions APIs even if the first version is half baked.
- **OpenRouter Outshines Other Options**: A member compared **OpenRouter** favorably to **FAL**, **Cloudflare Gateway**, and **Vercel Gateway**, citing OpenRouter's broader offerings as a key differentiator.
   - The same member made the point that *cementing that dominance in other modalities and non-completions APIs seems worthwhile.*
- **Phony Providers Proclaim Premature Partnering**: Some providers in the channel discussion claim to have *access via OpenRouter* on their websites, despite not being onboarded and lacking other means of inferencing.
   - The discussion quickly resolved to only one provider making the false claim.
- **Oceanstone Speculation Surrounds Subterranean Sources**: A new LLM named **Oceanstone** has surfaced in the **LMArena**, leading to speculation that it may be **Gemini 3.0 Flash**.
   - Members seem to think it's from **Google**, and one member speculated it's *at least 2.5 Flash level*.
- **ChatGPT's Consumer Captivation Captured**: A member shared a link to [OpenAI's article](https://openai.com/index/how-people-are-using-chatgpt/) and [PDF](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) containing stats from a large-scale analysis of **1.5 million** **ChatGPT** conversations.
   - The study claims to be *the most comprehensive study of actual consumer use of AI ever released*, though privacy concerns were raised regarding the methodology.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1416151110623690793)** (483 messages🔥🔥🔥): 

> `Cursor's Linter Errors, GPT-5 Output Changes, Terminal Instances Hanging, Auto Mode Billing, OpenAI Platform UI Changes` 


- **Billing cycles affect Cursor's auto mode**: A user asked about **Cursor's auto mode** and others reported that **auto mode will no longer be free** and billing cycles after the 15th won't be free either.
- **Cursor blamed for deleting Netlify Account**: A user reported that **Cursor deleted their Netlify account** and removed the app, but another user explained that **Cursor IDE has no integration, permission, or capability to delete external accounts**.
   - A user recommended exporting the chat logs to investigate the issue further.
- **Auto Mode isn't cheap!**: Users discussed **new pricing for Auto** and if it was actually using GPT5, resulting in users wasting money, with input pricing the same as GPT-5, costing around **$1.25/1M**.
- **GPT-5 and Sonnet 4 Duel for Code Supremacy**: Users debated the merits of **GPT-5** versus **Sonnet 4** for coding tasks, with one user finding that **Sonnet 4** is better at following designs and others praising GPT-5's superiority when building from scratch.
   - A user advised using **Sonnet to generate a meta prompt for GPT-5**, combining the strengths of both models.
- **Too many tokens, too little Ultra**: A user complained about **running out of Ultra plan credits** quickly while creating websites and said *idek how*.
   - A user said that creating multiple websites, debugging, handling Typescript issues, and long files are the main causes for consuming many tokens.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1416226090707320954)** (1 messages): 

> `Docker permissions for agent users, Manual VM setup` 


- **Docker permissions for agent users need setup**: A user setting up **Docker** in a manual VM asked how to ensure the agent user has **Docker permissions**.
   - They mentioned adding the **Ubuntu user** to the **Docker group** and needing to run `newgrp docker` in a shell.
- **`newgrp docker` in bashrc causes agent boot hang**: The user tried running `newgrp docker` in `bashrc` but this caused the agent to hang when booting up.
   - The user is seeking advice on the correct way to configure **Docker permissions** for the agent.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1416176373403488296)** (339 messages🔥🔥): 

> `Low Bit Pythia, TinyStories Warmup, Muon Optimizer, RoPE analysis, MXFP4 Quantization` 


- **Low Bit Baselines for Pythia sought**: A member inquired about **FP4** and **FP8** versions of **Pythia** to establish a baseline for low bit training, seeking "mid-training" checkpoints.
   - Another member suggested to write up what you wanna do, why it's interesting, and what compute you need.
- **TinyStories Warmup data is bad**: It was cautioned that using **TinyStories** for warmup data can permanently lower model capacity and cause a model trained with it to perform poorly.
   - A member argued for maintaining a high learning rate (LR) during **FineWeb** start, allowing the model to adapt quickly, and sharing a graph as evidence.
- **Muon Optimizer has deep math**: A third-year math student sought guidance on approaching deep learning rigorously and was pointed to deep math underpinning DL, as well as recent work on optimizers like **Muon**, and pointed to [this paper](https://arxiv.org/abs/2410.21265).
- **RoPE ratios may be under 100%**: In a discussion about positional encoding in transformers, it was mentioned that many people use **RoPE** on only about 25% of the dimensions and was explained in [this article](https://arxiv.org/pdf/2410.06205).
   - The conversation covered topics from standards to the effect of gigantic thetas, and how RoPE is a nightmare for interpretability.
- **MXFP4 Quantization Performance**: A member asked about quantizing a model from **FP16** to **MXFP4** and was directed to the last page of the appendix of the [torchao paper](https://discord.com/channels/729741769192767510/730095596861521970/1414599486507974657) for a summary of the API.
   - Another member asked whether this approach works on **MXFP4** without a significant performance drop.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1416136601389502594)** (90 messages🔥🔥): 

> `Gauss Lean code, scaling inference tokens, Fractured Entanglement, Neuron Lifespan` 


- **Gauss generates Lean code**: **Gauss** was used to produce **~25,000 lines of Lean code**, comprising over **1,000 theorems and definitions** in a *Lean environment*.
   - It *relies on natural language scaffolding supplied by human mathematicians, and requires high-level expert guidance* according to [this tweet](https://fxtwitter.com/mathematics_inc/status/1966194751847461309).
- **Scaling Inference Tokens at Test Time Explored**: A new [paper](https://arxiv.org/abs/2408.00677) measures the effect of **scale** and **thinking** on straightforward execution of long tasks, revealing that smaller models with **100% accuracy** fail faster than larger ones in multi-turn scenarios due to per-step accuracy degradation when seeing prior mistakes.
   - One member suggested that *if test time scaling is it, then we've only scratched the surface and I'm anticipating scaling to trillions of inference tokens at immense throughput to give unprecedented performance gains* but only if we can solve error accumulation.
- **Fractured Entanglement Under the Microscope**: Discussion revolves around a paper on **Fractured Entanglement**, with one member noting the paper's SDG experiment is too limited and not fully representative of the engineering in LLMs, referencing the biology of LLMs paper from Anthropic.
   - The hypothesis is that maybe there is a regularizer that minimizes these fractured representations, citing that [Google's NanoBanana model](https://ai.googleblog.com/nanobanana) has smaller amount of fractured representation, because of better character consistency.
- **Neural Lifespan Mechanism Proposed**: A member coded a tiny prototype where each neuron has a *life score* based on prediction correctness, with neurons dying (weight set to zero) when the score reaches zero.
   - Another member introduced the idea of **Neural Darwinism**, where neurons that are useful to existing brain pathways will reproduce by being more likely to be used by additional brain pathways, while others will fade into irrelevance.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1416818259583438959)** (2 messages): 

> `New NN Architectures, New Chip Architectures, New MoE Architectures, PyTorch, Novel Infra` 


- **Architectural Innovations Abound!**: Developers are actively creating **new NN architectures**, **new chip architectures**, and **new MoE architectures**.
   - The same team created **PyTorch** and are now innovating on the full stack.
- **Infrastructure Stack Revolution**: They're throwing compute at **novel infra** for the whole stack.
   - This indicates a significant investment in resources to support these advancements.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1416795124507344927)** (29 messages🔥): 

> `Model Calibration, AI Safety Concerns, Few-Shot Evaluation, BLIMP Benchmark Issue, Verifiable Rewards` 


- **LLM Calibration Sane-Washing Concerns Emerge**: Members expressed concern that improving model calibration might *sane-wash* models without addressing underlying representation problems, potentially impeding other improvements.
   - The worry is that improving calibration gives models a *trivial shortcut* by learning the behavioral correlate of humility without actually improving reasoning or world-modeling.
- **Few-Shot Override Fails for BLIMP Benchmark**: During evaluations, it was discovered that the few-shot override via CLI did not work for the **BLIMP** benchmark due to a specific configuration in the task.
   - The benchmark compares the log-likelihood of correct/incorrect sentence pairs, rendering few-shot learning inappropriate in its current formatting; it was later [determined](https://github.com/EleutherAI/lm-evaluation-harness/blob/0c134ee944d97998013eaff6f4e76d1b9fa87ecd/lm_eval/tasks/blimp/_template_yaml#L7) that *fewshots [are] not really appropriate the way its formatted.*
- **Calibration as Part of Reasoning**: It was argued that calibration is integral to reasoning, as being calibrated about likely payoffs is helpful when searching a large space of possible reasoning steps.
   - A recent [study](link-to-study) trains calibration using verifiable rewards which may result in rational updates based on improved capabilities, not just models saying *I don’t know* more often, suggesting improvements in epistemics might matter.
- **Shortcut Concerns about verifiable Rewards in Calibration**: Concerns were raised that **verifiable rewards** for calibration might be shallow and lead to models learning calibration through *brute force* rather than genuine epistemics.
   - There are questions whether models learn **general best practices** for epistemics applicable to novel distributions or calibrate via shortcuts, potentially resulting in fake calibration.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1416137267931512983)** (10 messages🔥): 

> `Memory Bandwidth Bounds, CUDA Dynamic Parallelism, Valuable Training Data` 


- **Memory Bandwidth Binds Training Throughput**: A member noted that training throughput is often memory bandwidth bound and questioned why larger models with dominant matmul/attention flops are still affected.
   - Another member explained that despite a large total batch size, the per-GPU batch size can be small, sometimes as low as 1 per H100 for an 8B model, impacting memory bandwidth.
- **CUDA Dynamic Parallelism Examined**: A member inquired about recent examples of **CUDA dynamic parallelism** in AI models, suggesting dynamic patch sizes, sparse attention, and the [Mirage framework](https://github.com/google/flax/tree/main/flax/experimental/mirage) as potential use cases.
   - The member speculated that Mirage uses a manager kernel to maintain a queue and launch sub-kernels, facilitating shmem-like compute and communication kernel fusion.
- **Training Data Valuation Explored**: A member proposed measuring the value of training data to reward contributors of high-impact data and shared a link to a relevant [X post](https://x.com/LuozhuZhang/status/1967619215013408832).
   - The concept revolves around the idea that not all training data is equal, and identifying valuable data can significantly improve model training efficiency.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1416396584966684813)** (26 messages🔥): 

> `PTX SASS Compilation, cuModuleLoadDataEx performance, Flash attention optimization` 


- **PTX Isn't SASSy Enough**: Even with **PTX**, some **SASS** instructions can't run, and the **LLVM PTX backend** only allows access to **13 special registers**, according to [this blog post](https://redplait.blogspot.com/2025/09/practical-ced-usage-extracting-sm.html) and [LLVM documentation](https://llvm.org/docs/NVPTXUsage.html#reading-ptx-special-registers).
- **Kernel Compilation Bottleneck**: A member is seeking advice on compiling many one-off **CUDA kernels** (PTX -> SASS) using the **nvptxcompiler API** at runtime, aiming to minimize bottlenecks from **cuModuleLoadDataEx** when using small launch sizes and frequent module unloading.
   - It was suggested to batch kernels into a small number of modules to reduce serialization overhead, and to leverage the **cuptxcompiler API** for its non-serialized compilation.
- **Flash Attention Vectorized**: A member released [parts 4](https://lubits.ch/flash/Part-4) & [5](https://lubits.ch/flash/Part-5) of their series on building **flash attention** from scratch.
   - Part 4 covers vectorized bank conflicts and swizzling, while part 5 covers common optimizations used in **Cutlass**.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1416810845777821767)** (11 messages🔥): 

> `Kernel Registration, Custom Ops, Torch Function Optimization, Ops Fusion for torch.compile` 


- **Kernel Registration Insufficient for Fusion?**: A member asked if it's straightforward to register a kernel so that it can perform certain operations, while another responded that registration alone isn't enough for fusion, using a **Triton matmul** example.
   - Specifically, *there won't be fusion with bias/addition without broadcasting*.
- **Metal Flash Attention Custom Op Shines**: A member created a [custom op](https://github.com/bghira/universal-metal-flash-attention/tree/main/examples/pytorch-custom-op-ffi#readme) using **Apple Metal** for efficient flash attention in PyTorch.
   - The author noted that it *works well even with the required Metal element caching to make it performant* and is now working on vectorizing the causal attention masking.
- **Torch Function Optimization Tool Sought**: A member inquired about a tool to optimize torch functions to CUDA.
   - No specific tools were recommended in the provided conversation.
- **DIY Ops Fusion via torch.compile**: A member shared that it's possible to build custom ops fusion for **torch.compile** using the [PyTorch documentation](https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html).
   - While intrigued, another member admitted they hadn't tried it themselves.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1417213410378125383)** (1 messages): 

> `PTX, CUDA PTX Introduction` 


- **PTX intro resource surfaces!**: A member shared an introductory resource about **PTX** at [philipfabianek.com](https://philipfabianek.com/posts/cuda-ptx-introduction).
- **PTX demystified**: The post provides a good introduction to **PTX**, the parallel thread execution virtual machine and instruction set architecture (ISA) used by NVIDIA.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1416146600795050004)** (5 messages): 

> `AI Infra Startup Hiring, Red Hat AI Hiring, Zig for AI` 


- **AI Infra Startup Luring Low-Level Luminaries**: An AI infra startup is [recruiting low level devs](direct.message) for a **Zig / C / C++ / Cuda / Python stack**, offering **TC: 250K+** and year round internships.
   - Experience in **networking, compilers, and OS** is a plus.
- **Red Hat Rockets Recruitment for AI Roles**: Red Hat AI is [hiring software engineers at multiple levels](https://www.linkedin.com/in/terrytangyuan/) with experience in **Golang, C++, Python, CUDA, GPU kernels, Triton, CUTLASS, PyTorch, vLLM, Kubernetes, and Open Source**.
   - Those interested should email a short summary of their background and resume (address in LinkedIn profile), and can learn more about their work via their [newsletter](https://inferenceops.substack.com/).
- **Zig Zagging into AI?**: A member mentioned that **Zig** may be related to AI given HF uses **Rust** for fast tokenizers, and **Zig** is an alternative to **Rust**.
   - Another idea is they might be doing **video streaming** sort of stuff and need it for their *frontend*.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1416160194999877682)** (10 messages🔥): 

> `CUDA, RAPIDS, CUDA-X, Batch Gradient Descent, Nvidia Jetson` 


- ****CUDA Core Concepts Clarified****: Members suggested it's better to learn **CUDA** with **C++**, mentioning that **RAPIDS** and **CUDA-X** might be most relevant for enhancing parallelism with **batch gradient descent** or **mini-batching**.
   - They noted that if you can enhance parallelism with batch gradient descent then there's little need for SGD.
- ****Jetson Channel Judged Dormant****: A user inquired about a channel for **Nvidia Jetson** or similar, and another member confirmed the existence of a channel but noted it *hasn’t been particularly active*.
- ****Leaderboard Learning Loophole Located****: A user sought access to leaderboard submissions for past competitions to learn from top performers, specifically mentioning [this leaderboard](https://www.gpumode.com/v2/leaderboard/463?tab=rankings).
   - A member provided a [link to the AMD competition data on Hugging Face](https://huggingface.co/datasets/GPUMODE/kernelbot-data), noted correctness issues with the PMPP v1 evaluation, and mentioned potential future support for entries on the site via [this Github repo](https://github.com/gpu-mode/kernelboard).
- ****Triton Touted for GPU Training****: A member inquired about learning deeper **GPU programming** for kernel optimization, questioning if starting with **Triton puzzles** is sufficient.
   - Another member responded that *starting with triton puzzles is a great way to start learning some concepts and getting the feel for gpu programming* and shared a [link to the CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1417265956085956721)** (1 messages): 

> `autoquant_v2, batch size 1, runtime errors, autotune stage, dtypes` 


- **AutoQuantV2 Woes with Batch Size 1**: A user asked whether **autoquant_v2** is recommended for **batch size 1**, mentioning it appears to have code specialized for that batch size.
   - The user also reported that **batch size 1** causes **runtime errors** during the **autotune stage** for some **dtypes**.
- **Batch Size 1 Blues: Runtime Errors During Autotune**: A user experienced **runtime errors** during the **autotune stage** when using **batch size 1** with certain **dtypes**.
   - This suggests potential compatibility issues or limitations with **autoquant_v2** when operating under these specific conditions.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1416734894188662845)** (7 messages): 

> `Iris Lecture, Symmetric Memory, RDMA support, iris.load/store, tl.load/store` 


- **Iris Lecture Sparkles Symmetric Memory musings**: The new **Iris** memory model lecture was well received, prompting comparison with **symmetric memory in CUDA** and discussion around implementation differences like the global symmetric heap.
   - The main difference is that in Iris we build a global symmetric heap up front and slice tensors from it, so address translation only needs a single heap bases pointer per rank, which will make supporting **RDMA** in the future easier.
- **`iris.load/store` incurs perf penalty vs `tl.load/store`**: Using `iris.load/store()` instead of `tl.load/store()` for local memory access introduces a translation overhead, so `tl.*` operations are recommended for now.
   - The translation overhead will still be there, it's minimal and should be cached but still some extra code; but a missing `if` statement in the translate function could provide a fast path for the local access case in the future.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1416201614477365339)** (18 messages🔥): 

> `Intel CPU/GPU Optimizations, IPEX Deprecation, SGLang AMX Kernel, PyTorch integration` 


- ****Intel Optimizations**: CPU/GPU - What's the Deal?**: A user inquired about leveraging Intel-specific optimizations on B50s with AMX-enabled servers, asking whether IPEX could utilize both CPU/GPU optimizations.
   - It's complicated: it was suggested that one might get away without IPEX, since `at::native::cpublas::brgemm` can dispatch to AVX-512 if the CPU doesn't support AMX iirc.
- ****SGLang's Secret Sauce**: Fused MoE Kernel with AMX**: Discussion emerged around SGlang's use of AVX512 instructions and AMX, with links provided to the [relevant code](https://github.com/sgl-project/sglang/blob/6f4676ef854d4d2461969f8464f227b84d2eaac7/sgl-kernel/csrc/cpu/moe.cpp#L7) using *fused MoE kernel with AMX*.
   - The conversation explored how the kernel in SGLang uses AMX via `at::native::cpublas::brgemm` and can dispatch to AVX-512 if the CPU lacks AMX support.
- ****IPEX's Fate**: Being Deprecated?**: A user questioned the purpose of IPEX, leading to a discussion about its status, with the assertion that *it's more or less being deprecated in favor of upstreaming as much as possible into PyTorch or other more relevant projects*.
   - Counterpoint was that IPEX has been an *experimentation platform for Intel to push their most aggressive and new optimizations*, like torch nightlies.
- ****Intel Confirms**: IPEX Development Discontinued**: Intel's official stance involves discontinuing active development on IPEX after the 2.8 release.
   - Intel is focusing on developing new features and supporting upcoming platform launches directly within PyTorch*, after [successfully upstreaming most of our features and optimizations for Intel® platforms into PyTorch*](https://pytorch.org/blog/intel-extension-for-pytorch/).


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1416566574898348142)** (11 messages🔥): 

> `Metal Flash Attention Bridge, Quantised Attention, Metal Command Buffer Timeout` 


- ****Universal MFA** Bridges to Other Languages**: A member is building a [bridge](https://github.com/bghira/universal-metal-flash-attention) for **Metal Flash Attention** to other programming languages, with it already working in **C**, **Rust**, and **Obj-C**.
- ****Quantised Attention** added to Universal MFA**: The member added *quantised attention with backprop* to their universal **MFA** repo, seeing a speedup for large shapes and a slowdown for small ones, with associated memory improvement.
- **Request for **Metal Command** Timeout Method**: A member is seeking help on how to set a timeout on a metal command buffer to prevent long execution times of metal kernels.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1416648120816898148)** (9 messages🔥): 

> `LLM Negotiation Protocol, Metal Flash Attention Swift Adapter, Rust Bindings vs cv2, CuTe Partitions analysis, Gated Attention` 


- **Decentralized Commerce Protocol Debuts**: A decentralized commerce protocol for **LLM-to-LLM negotiation**, built with **Rust** has been released on [GitHub](https://github.com/awdemos/DCAP).
- **Swift Metal Flashes Faster Attention**: A language adapter using **C FFI** for **Swift** with the original [metal-flash-attention](https://github.com/bghira/universal-metal-flash-attention) for **Apple Silicon** hopes to have a better implementation of efficient flash attention than Torch.
   - A **Pytorch SDPA drop-in replacement wrapper** that's still experimental, gets us performance gains as expected, despite translation between Swift Metal ops and Python through C FFI.
- **Rust Crushes cv2 in Performance**: A project beat **python cv2's performance** using **pyo3-based Rust bindings** built directly on top of OpenCV, yielding a [1.25x performance increase](https://github.com/bghira/TrainingSample) over cv2 for single-image operations with better memory management and parallelism.
- **CuTe's Partitions dissected**: An analysis of CuTe's partition patterns demonstrates how matrix copy can be performed with inner, outer, and thread value partitioning, all achieving good performance as explained in [this blogpost](https://veitner.bearblog.dev/cute-partitions/).
- **Gated Attention and DeltaNet get Explained**: An explanation of next gated attention and gated deltanet has been summarized in [this document](https://charm-octagon-74d.notion.site/Attention-Variants-2-Qwen3-Next-26ee4301cb9980af86aff509ad73e3b6) and [this other document](https://charm-octagon-74d.notion.site/Attention-Variants-3-GPT-OSS-26fe4301cb9980d9a94cc5ac0cc77ac3).


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1416455963883601923)** (2 messages): 

> `Smallest model above GPT3.5, Quantization, VRAM requirements` 


- **Quest for petite performer surpassing GPT-3.5**: A member inquired about the smallest model that could run on edge with performance surpassing **GPT-3.5**, regardless of whether it's quantized.
   - The primary concern was finding a model that is both decent and small, with minimal **VRAM** requirements during inference.
- **Balancing Act: Decent Performance with Minimal VRAM**: The user emphasized the need for a small model suitable for edge deployment, prioritizing performance over **GPT-3.5** while minimizing **VRAM** usage.
   - The inquiry highlights the trade-off between model size, performance, and resource consumption in edge computing environments.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1416262944278450236)** (77 messages🔥🔥): 

> `MI300x8 Leaderboard, Rank + 1 Trick, AMD Rules Clarification, all2all vs gemm+rs kernels, kernel dev` 


- **MI300x8 Cranks New All2All Leaderboard Times**: A user achieved **first place** on **MI300x8** with **373 µs**, later followed by **578 µs**, then **547 µs**, and later another user got **first place on MI300x8** with **546 µs** on the *amd-gemm-rs* leaderboard.
   - Other successful submissions on **MI300x8** ranged from **1495 µs** to **1859 µs**.
- **Rank + 1 Replacement for GEMM of Moe is Banned**: A user inquired whether the *rank + 1* trick was banned, as it circumvents the need for all2all data transfer, and an organizer confirmed that it is.
   - The organizer clarified that submissions abusing the *rank + 1* weighting are disallowed, but the original submission used a trick that is allowed, but it's abusing the fact that weights are just rank+1, so we'll be deleting it; a later submission will focus solely on the *rank + 1* operation.
- **AMD and GPU Mode Clarify Rules on Kernel Submissions**: Organizers warned against making significant rule changes mid-competition, as it can introduce inconsistencies.
   - Organizers advised a user to focus on kernel development and clarified that AMD and GPU Mode are responsible for rules and titles, but should take feedback privately to Daniel to clarify the rules as needed and if necessary clarify how eval.py should be fixed.
- **All2All Kernel Confusion Clarified**: Organizers are repeating the 1st/2nd problem => all2all/gemm+rs kernel, since a lot of friends are confused with that.
   - All2all Kernel needs implementation of dispatch and combine kernels with intra node communication, while gemm+rs is a computation+communication kernel, reference.py says the kernel logic, you need detailed analysis it.
- **Trimul Personal Bests Hit New Lows**: A user achieved a **personal best** on **A100** with **20.3 ms**, and later another **personal best** on A100 with **20.0 ms** on the *trimul* leaderboard.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1416713271947563089)** (11 messages🔥): 

> `MI300x server status, Popcorn-cli timeout issues, Queue overload, Runner downtime, Cluster capacity issues` 


- ****MI300x server** jobs timing out**: Users reported jobs timing out in **popcorn-cli** and being queued indefinitely on Discord, prompting concern about the **MI300x servers**.
   - One user suggested that *the queue is just busy*, citing personal success after multiple attempts, while another user promised to investigate the issue upon returning home.
- **High submission volume swamps **MI300x server****: The administrators noted the MI300 is getting about **400 submissions a day**.
   - The admin notified **AMD** to request additional runners to handle this volume.
- **MI300 Runner Downtime Troubles**: It turns out the runners were down.
   - Previously promised that everything was fine, an admin reversed course to state *actually it seems that runners are down, we're only getting 2 runners at the same time*.
- **Cluster capacity under investigation**: Admins are investigating after it was discovered that *someone was taking our cluster capacity*.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1416547424545276027)** (2 messages): 

> `Eval Infra, PR Review` 


- **Eval Infra Resumption**: A member announced they will resume work on **eval infra** tomorrow afternoon.
   - They asked others to review the **PR** (thanks jack) and provide feedback.
- **Awaiting PR Feedback**: A request was made for team members to review a **Pull Request** related to the eval infrastructure.
   - The team is asked to give comments on what else is needed for the PR, or confirmation that it is ready to be merged.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1416742324767035523)** (55 messages🔥🔥): 

> `Runner Queues and AMD Assistance, amd-gemm-rs Challenge Release, ROCm/iris Integration, PyTorch Version Compatibility, Clarification on amd-all2all` 


- ****Runner Queues** Trigger AMD Assist!**: Due to **runner queues**, submissions may experience timeouts, but the team has notified **AMD** and will provide an ETA update soon.
   - Using benchmark/test can temporarily alleviate congestion, as these launch only one faster job.
- ****GEMM-RS** Challenge Sets the Scene!**: The second problem, **amd-gemm-rs**, challenges participants to implement distributed **GEMM + reduce-scatter**.
   - The problems are [open source](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_distributed/gemm-rs) on our github.
- ****Iris** Integrates Nicely!**: The cool [iris project](https://github.com/ROCm/iris) is now available.
   - To learn more, check out [the author's talk on YouTube](https://www.youtube.com/watch?v=GZqYr8_Q7DE).
- ****PyTorch 2.8.0** Fails to Play Nice**: A member encountered an *undefined symbol* error using `torch load_inline` and **PyTorch 2.8.0**.
   - The member fixed the issue by installing a **nightly PyTorch ROCm build** via `pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.4`.
- ****All2All** Algorithm Analyzed!**: In the **amd-all2all** challenge, the *dispatch* output should group tokens belonging to the same experts together, similar to the reference kernel.
   - The competition focuses on fast communication, not implementing grouped_gemm, with computation emphasized in the second and third problems.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1416143419134443520)** (10 messages🔥): 

> `CuTeDSL swizzle patterns, PTX docs discrepancies, TF32 datatype` 


- ****Swizzle Showdown**: CuTeDSL vs. PTX Docs!**: A user found a discrepancy between **CuTeDSL** and **PTX docs** regarding the swizzling atom for the **TF32** datatype, specifically with `Swizzle<3,4,3>`, and shared [screenshots](https://cdn.discordapp.com/attachments/1362196854460383353/1416143418496782437/Screenshot_2025-09-12_at_21.22.10.png?ex=68c9ba55&is=68c868d5&hm=e6b461266f0ecfc49cca56436fd9e10d4d1a571d796cced9c3876c2d3a6d133e&).
   - The user believed the **CuTeDSL** implementation to be accurate, replicating results from [Lei Mao's blog](https://leimao.github.io/blog/CuTe-Swizzle/).
- ****Swizzle Secrets**: Decoding PTX Docs!**: A user clarified that the PTX doc uses a **128B swizzle** (M=4, S=3 and 4 + 3 = 7) on address bytes, while the composed layout swizzle operates on element indices, also [mentioned here](https://github.com/NVIDIA/cutlass/issues/2634#issuecomment-3292221378).
   - They suggested using `cute.make_swizzle(3, 2, 3)` instead of `cute.make_swizzle(3, 4, 3)` to produce the same result.
- ****PTX Puzzle Solved**: Recovering Figures!**: A user detailed how to recover figures from the PTX docs, involving adjustments to the swizzle pattern (e.g., using `(i,2,3)`), atom size, matrix shape scaling, and a final division by `//4`.
   - They provided examples with [screenshots](https://cdn.discordapp.com/attachments/1362196854460383353/1417191712367050875/Screenshot_2025-09-15_at_18.41.11.png?ex=68ca3ee2&is=68c8ed62&hm=d81efe28dcd47c8c9ee38531987edead1a98250c6474ae72df8f590c05974bf1&) illustrating the process, interpreting elements with the same index as one 128bit element.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1416360923052113981)** (11 messages🔥): 

> `ML and Systems Book Design, GPU access limitations, Autograd machinery development, PicoTorch revitalization, Textbook to lower barriers for community` 


- ****Book Design Tensions Mapped****: The design of a Machine Learning (ML) and systems-focused book faces internal tension regarding **top-down vs. bottom-up order**, and the presentation of *what* vs. *how*, and the author is mapping out the beginning, ending and chapter relations.
   - Due to limited time and lack of on-prem/bare-metal GPU access, achieving the initial goals for the book is proving challenging, it was hoped to map the beginning, ending and chapter relations.
- ****Autograd Ascends, MLP Arrives****: The first part of a book will focus on a **bottom-up approach**, developing all the **autograd machinery** and culminating in an **MLP language model**.
   - Part 2 will cover **transformers**, **flash attention**, and possibly **diffusion language models**, while part 3 will delve into compilation.
- ****PicoTorch Project Plugs Ahead****: The [PicoTorch](https://github.com/j4orz/sctp/tree/master/picotorch/src) project, which previously ran a forward pass of **Karpathy's MLP** without GPU acceleration, needs revitalization.
   - Chapter 1 is underway, sketching out intuitive plots, model circuits, math, and code snippets for each concept introduced.
- ****Textbook Aims to Aid GPU Mode Community****: A textbook is being created to **lower the barriers** for the **GPU mode community** to create something similar to *PyTorch from scratch*.
   - The project's tagline could be: *"we are making you implement PyTorch from scratch"*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1416419690305163486)** (8 messages🔥): 

> `Kernel Development Path, GPU Mode Kernel Competition, Triton Benchmarks, BioML Trimul Kernel Competition` 


- **Kernel Dev Starter Asks for Help**: A new user asked for guidance on the proper path to follow for **kernel development**.
   - They also inquired about expectations, submission details, and requirements for the **GPU Mode kernel competition**.
- **BackendBench for Triton Benchmarking**: A user asked about good benchmarks for writing functions in **Triton**, similar to KernelBench.
   - A member suggested [BackendBench](https://github.com/meta-pytorch/BackendBench), which helped them benchmark about **84 PyTorch operators** written in **Triton**.
- **BioML Trimul Kernel Comp Closing Soon**: There are only **14 days left** to participate in the [BioML trimul kernel competition](https://www.gpumode.com/v2/leaderboard/496?tab=rankings).
   - The prize is going to be *never before seen swag* designed and shipped by the organizers.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1416318981710544986)** (1 messages): 

> `` 


- **Multi-GPU Channel: No Active Discussions**: The provided Discord channel log for the multi-gpu channel contains no active discussions or topics suitable for summarization.
   - The log lacks substantive content regarding new fundraising, models, tooling, or other subjects of interest as defined in the instructions.
- **Multi-GPU Channel: Log Contains Minimal Content**: Analysis of the multi-gpu channel's log reveals a lack of messages containing relevant information for a technical summary.
   - The messages do not include any links, blog posts, code snippets, or specific details that would warrant inclusion in the summary.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1417223430549405748)** (2 messages): 

> `Video Models, Low-bit-training, GPU mode hackathon` 


- **Quest to Survey Video Models and Low-Bit-Training**: A member is conducting a survey of **video models** and **low-bit-training** for a submission to the **GPU mode hackathon/irl meetup** in October.
   - They are seeking pointers to research papers specifically focusing on **low-bit-training** techniques applied to **video models**.
- **Home Video Models GitHub List Shared**: The member shared a [GitHub repository](https://github.com/vipulSharma18/we-have-video-models-at-homethe) containing a collection of **video models**.
   - The repository includes work on **LLMs** (likely **MobiChamps's**), as well as some **DiT** and **LLM** training approaches like **Quartet**.


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1416178117952798923)** (37 messages🔥): 

> `Multi modal inference, Training Optimisation, Gated DeltaNet, Sparse GNN ideas, Low-bit-training` 


- **Single Machine Video Model Training Revolution**: A member shared a paper on [training a large video model on a single machine in a day](https://arxiv.org/pdf/2309.16669), achieving this with **FP4/FP8** precision, though the paper uses **FP16** as a proof of concept.
- **DeltaNet Dreams of Context-Parallel Kernels**: A member is looking to form a team to implement a context-parallel version of the kernels for super long-context training using [GatedDeltaNet from NVlabs](https://github.com/NVlabs/GatedDeltaNet?tab=readme-ov-file), noting its use in **Qwen 3**.
- **Sparse GNN Ideas Spark Interest**: A member expressed interest in sparse **GNN** ideas, particularly those with implications for topology, compute graphics, and vector databases, linking to a relevant [arxiv paper](https://arxiv.org/pdf/2507.13296).
- **Blackwell's Tensor Cores Tempt Sparse Matrix Multiplication**: Inspired by **Blackwell's** tensor cores, a member considered problems involving block-sparse format and NVIDIA tensor cores, linking to a blog post on [accelerating matrix multiplication](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/).
- **In-Person Hackathon Confirmed**: Members confirmed that the hackathon is an in-person event, as online hackathons are harder to design, and more similar to their **kernel competitions**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1416143552932479169)** (180 messages🔥🔥): 

> `Playwrite MCP issues, Local Wikipedia Access for Small Models, Qwen/Qwen3-Next-80B-A3B-Instruct and llama.cpp, SIA-1: Self Improving AI Agent, lambda stack vs lm studio` 


- **Playwrite MCP throws Connection Errors**: A user reported getting errors when starting Playwrite MCP with LM Studio, seemingly related to connection issues, but they noted it may be user error.
   - Another user jokingly responded that *it works on my machine*.
- **Most powerful AI model runs best with RAM**: When a user inquired about the most powerful AI model, a member mentioned that you can use every model with RAM, stating that the most powerful AI model you can locally run is **Kimi K2** in **BF16**, requiring **2.5TB RAM**.
   - It was clarified that *however big the filesize is, you need that much memory (VRAM + RAM combined) to load it, and then some more for context.
- **Users ask for help to Access Wikipedia Articles**: A member asked for help regarding tools to access **Wikipedia articles** either online or offline for use with small models, and a user shared the [LM Studio Wikipedia](https://lmstudio.ai/lmstudio/wikipedia) MCP.
   - It was mentioned that creating a semantic index locally is complex, as the local Wikipedia extract would lack a search and *LLMs are not THAT great at guessing without some fuzzy search*.
- **NousResearch's Mephisto Discusses World Sim**: A user shared a [YouTube video](https://www.youtube.com/watch?v=7ZEHdaABIJU) featuring **NousResearch’s Mephisto**, who gets into technical details about base models and how Instruct models are essentially base models that have been trained to roleplay as instruct models.
   - Mephisto then discusses how NousResearch started **World Sim**, which may be *definitely the next steps in the world of Agents*.
- **SIA-1: The AI That Evolves Itself**: A user introduced **SIA-1**, claiming to be *the world’s first truly self-improving AI agent* [https://sia-1.net](https://sia-1.net), which learns to improve its own code, generation after generation.
   - Members vibed against the agent, with one member asking *pls tell whoever vibe-coded that to use a better model*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1416200248283496580)** (113 messages🔥🔥): 

> `KiCad and LLMs for Circuit Design, SBC for Searxng vs Obsidian, GPT-OSS-20B and VRAM Allocation, Nvidia P40 EOL, RTX 5070 and LLM Performance` 


- **LLMs Set Houses Ablaze via Circuit Design**: A member expressed caution about using LLMs for circuit design with tools like **KiCad**, emphasizing the need to understand the underlying principles to avoid potentially disastrous outputs.
   - They added that calling a language model an 'AI' is hugely misleading.
- **SBC Specs Spark Database Debate**: A member asked about using **Obsidian** on a Raspberry Pi with 4GB RAM as an alternative to a slow database or Searxng setup.
   - Another countered that even a *potato* is fast enough for certain LLM tasks, and suggested focusing on the access pattern rather than the database size.
- **GPT-OSS-20B Struggles with VRAM**: A member reported issues loading **gpt-oss-20b** with a 128k context on a 7900xtx (24GB VRAM), despite expectations it should fit.
   - Another user suggested using **KV quantization at F16**, which reduced VRAM usage to around 18.5GB, and also advised closing GUI-heavy apps to free up VRAM.
- **Nvidia P40s face End-of-Life**: A member considered buying cheap **Nvidia P40s**, but worried about the upcoming end of driver updates and CUDA support.
   - Someone noted that **Nvidia** is dropping CUDA support for Maxwell, Pascal, and Volta GPUs with the next major toolkit release, but another user reported being able to get them for only $200.
- **RTX 5070 Specs Trigger Upgrade Deliberation**: A member with an **RTX 5070** (16GB VRAM), i5 14600k, and 32GB DDR5 sought advice on suitable AI models for website development, or if an upgrade is needed.
   - One user suggested forking out for Github Copilot for coding tasks since 16GB VRAM isn't ideal for larger models and agentic coding.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1416143087469723688)** (265 messages🔥🔥): 

> `Kimi vs GPT-5, Augment code extension, Kimi K2 Groq, interactive preview for LLM processes, API Keys vs Login Accounts` 


- **Kimi-K2 is scripting Supreme**: Members debated the strengths of **Kimi-K2** for scripting, with some claiming it outperforms paid **Gemini**, especially when using **Groq** for coding.
   - Others found **GPT-5** better, one member noted that *GPT-5 is really good at web UI and nextjs* while **Kimi** has *a really good research mode*.
- **Augment coding with Kimi**: Members discussed using **Kimi** with the **Augment** code extension in VS Code, where users can prompt for code changes and fixes from various models.
   - One user described using **Augment** as a way to apply prompts and fix code by looping in **Gemini** or **Kimi**.
- **Slides Feature Sparks UX Brainstorming**: One member highlighted the impressive interactive slide generation feature in **Kimi**, praising the *real-time updates* and smooth feel, suggesting that an interactive preview of what's happening is important for LLM-based processes.
   - They proposed a similar approach for a **Godot** game engine agent, envisioning *real-time updates* during code generation, with interactive previews of nodes and scripts.
- **Groq Hosted Kimi-K2 causes Concern**: One user asked if there were issues with **Kimi K2** hosted on **Groq**, another user ranted about the need to remove the **3-hour message cap**.
   - The user also requested the ability to **edit previous prompts**, stating *Every other AI platform already has this*.
- **API Keys vs Account Logins in CLIs**: A user inquired about using **Kimi K2** with CLIs like **Claude Code** and **Qwen Code** *without an API key*, instead of using a kimi.com account login.
   - Another user suggested using **API keys** for **Claude Code**, providing a command example: `export ANTHROPIC_AUTH_TOKEN=sk-YOURKEY` and `export ANTHROPIC_BASE_URL=https://api.moonshot.ai/anthropic`.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1416153417201487926)** (115 messages🔥🔥): 

> `HDF5 Python Library, FineWeb pretraining, Hugging Face Spaces storage, Qwen3-Next modeling and Layernorms, Models for open world RPG RP` 


- **Hugging Face Spaces' Storage Situation!**: In the case of **HF Spaces**, uploaded files and generated files are stored on disk space within the virtual machine, they cannot be accessed arbitrarily from outside, and they disappear when Spaces is restarted unless the paid Persistent Storage option is used.
   - However, there are rare cases where, due to mistakes like everyone having the same filename, someone else's generated data became visible.
- **New FinePDFs dataset liberates Tokens from PDFs!**: A new [FinePDFs dataset](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) was released, containing about **3 trillion tokens** across **475 million documents** in **1733 languages**, sourced exclusively from PDFs.
   - When mixed with HTML-based corpora, it delivers a significant performance boost across benchmarks and they recommend keeping the proportion of PDF data below **25%** of the overall dataset.
- **Sin and Cosine Function deep-dive!**: A member asked why a pair of sin and cosine waves are used in positional embeddings and why it just can't use a single sin wave.
   - Another member responded that using both sine and cosine lets the model represent positional information in a way that preserves relative distance and can be linearly combined since sine alone would cause ambiguity.
- **Qwen3-Next Model Swaps Layernorm for RMSNorm**: The [Qwen3-Next model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) mentions stability optimizations like *zero-centered and weight-decayed layernorm*, but it's actually using RMSNorm in Transformers.
   - It was clarified that there's no layernorm involved at all, just RMSNorm with a zero-centered gamma and weight decay on the norm scale in training and at inference it’s plain RMSNorm.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1416771469995675691)** (6 messages): 

> `Agents Course, smol course, MCP course, LoRA finetuning, Transformers architecture` 


- **Deep Dive into Agent Dev**: A member suggested using the **80-20 rule** when learning about agents, recommending to concentrate on building directly, as *that 20% hands on will teach you 80% along the way*.
   - They believe *deep diving is 80% boring stuff*.
- **smol Course Deadlines**: A member inquired about the deadline for the **smol course** and whether they could still take the **MCP course** and get certified.
   - No answers were provided.
- **LoRA fine-tuning challenges**: A member is continuing their journey in fine-tuning LLMs with **LoRA** and finding it challenging but useful.
   - The member stated that they are learning *how to control my stress*.
- **Transformer Decoders study plan**: A member plans to do the **smol course** and study **Transformers architecture** (decoders).
   - No other details were given.
- **Agent Course signup issues**: A new member is trying to sign up for the **agent course**, but is not seeing it listed with the **MCP** and **smol** courses.
   - They are seeking assistance to resolve this issue.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1416410230748876954)** (2 messages): 

> `HF models, Fine-tuned models` 


- **Hundred Fine-Tuned Models Land on HF**: A member shared that a friend and mentor has published **100 fine-tuned production level AI models** on [HuggingFace](https://www.linkedin.com/posts/moyasser_ai-ai-machinelearning-huggingface-activity-7372687867359252480-yd9F) in 8-9 months.
   - The member is requesting to **recognize the hard work**.
- **A Model Maker's Marathon**: A close contact has reached a century of **production-level AI models** on Hugging Face, showcasing dedicated efforts over the past few months, generating discussion and interest.
   - The focus is on celebrating his remarkable achievement and its contribution to the community.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1416449418302853283)** (13 messages🔥): 

> `Voxtral finetuning, Dialectical Agentic CrossSphere AI, Refrag Efficient LLM Compression, Image to Space` 


- **Voxtral makes Speech Training Affordable**: Voxtral enables users with speech impediments or heavy accents to finetune models, costing only **$0.26** for an hour of training on an A6000, using tools to [make datasets](https://huggingface.co/spaces/Tonic/VoxFactory).
   - The tool supports finetuning to get it perfect, and one can push the model, dataset, and add a demo space (**works with CPU!! Free!!**) to Hugging Face.
- **Dialectical Agentic CrossSphere AI Enters Hackathon**: A user is seeking feedback on their **Dialectical Agentic CrossSphere AI**, which they entered in OpenAI’s hackathon, linked here: [OpenAI GPT OSS 20B](https://huggingface.co/spaces/Dennisgbay22/openai-gpt-oss-20b).
   - Another user praised the AI's game, images, and storytelling.
- **Refrag Unveiled as Efficient LLM Compression**: A user shared their blog post explaining **Refrag**, a method for efficient LLM compression and curriculum learning: [Understanding Refrag](https://medium.com/@limemanas0/understanding-refrag-efficient-llm-compression-and-curriculum-learning-explained-3452498f99e8).
   - The blog post highlights the efficiency and techniques involved in Refrag.
- **Image-to-Space Tool Transports HF Repos**: A user introduced a new method for transporting repos via an image containing a Hugging Face Space using [image-to-space decoder](https://huggingface.co/spaces/broadfield-dev/image-to-space).
   - Another user made a PR to improve the tool's functionality ([discussion on HF](https://huggingface.co/spaces/broadfield-dev/image-to-space/discussions/1)), praising it as *super cool and creative*.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1417120879640449105)** (2 messages): 

> `Style Transfer, WCT2 Methods, Segmented Images` 


- **Style Transfer Methods Require Segmented Images**: Style transfer methods like **WCT2** sometimes require segmented images, presenting a challenge.
   - This requirement limits the applicability of these methods in scenarios where obtaining segmented images is difficult or impossible.
- **Considerations for Style Transfer Implementations**: Implementing style transfer methods such as **WCT2** often necessitates careful consideration of image segmentation techniques.
   - The need for segmented images can add complexity to the pipeline, requiring additional preprocessing steps and potentially impacting the overall performance.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1416202968625188945)** (3 messages): 

> `Qwen2.5-72B fine-tuning, Database for Chat History, Maintaining User Sessions` 


- **Qwen2.5-72B Fine-Tuning: Seek Experts**: A member inquired about experiences with fine-tuning **Qwen2.5-72B**, requesting direct messages from those with relevant expertise.
- **Database and User State Discussion Initiated**: The prompt included a request to *use a database to store chat history*, and also to *maintain user session and user state*. 


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1416165207058812938)** (9 messages🔥): 

> `Fine-tuning course details, VRAM concerns for smaller models, In-person study group in NYC, Leaderboard evaluation for custom use cases, smol-course` 


- **VRAM Woes? Smaller Models to the Rescue!**: A member inquired about using even **smaller models** due to limited **VRAM**.
   - They asked about fine-tuning on custom use cases.
- **NYC Study Group Forming!**: A member proposed starting an **in-person study group** in **NYC**.
   - They offered to organize meetups in the city.
- **Decoding the Fine-Tuning Course**: A member inquired about the start date of the **fine-tuning course** and where to sign up, seeking clarification on the dynamics.
   - Another member provided a link to the [smol-course](https://huggingface.co/smol-course) and advised following the org and starting with [Unit 0](https://huggingface.co/learn/smol-course/unit0/1).
- **Smol Course Begins!**: Members were told to follow the huggingface org and start the course [here](https://huggingface.co/learn/smol-course/unit0/1).
   - This will allow them to begin the **smol-course** and start learning.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1416481312474267738)** (11 messages🔥): 

> `Agent course introductions, Token setting rookie mistake, Unit one introductions` 


- **Newbies start Agent Course**: Several new members, including Nouha, Lez, Karthik, Leo Kinyera, and Nay Lin, introduced themselves and expressed their excitement to start the agents course.
- **Token Setting Snafu Solved**: One member admitted to a *rookie mistake* of not having their **token set**, which was quickly resolved.
- **Unit One Underway**: A member reported working through **unit one** of the agents course.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1416309313609269299)** (85 messages🔥🔥): 

> `Trade Unions and Fascism, LLMs and Bayesian Inference, AI and Topos Theory, Positional Encoding in Transformers, Deep Learning and Turbulent Flow` 


- ****Trade Unions tangle with Fascism Facts****: A discussion arose about the relationship between trade unions and fascism, clarifying that while all fascists support trade unions, not all trade unionists are fascists.
   - It was emphasized that fascist corporatism involves the co-governing of corporations by employers, employees (via state-sanctioned trade unions), and the state, and thus trade unions are essential to fascism, referencing [this discord message](https://discord.com/channels/714501525455634453/986699377257119794/1416791624171786320).
- ****LLMs Look at Bayesian Beliefs****: A discussion centered around a paper on LLMs and Bayesian inference, with one member noting it demystified some preconceptions about LLMs, referencing [Leon Chlon's substack](https://leonchlon.substack.com/p/llms-are-bayesian-in-expectation).
   - Yannick Kilcher commented *lstms can do in-context learning just fine. it's a property of language modeling, not of architecture* and that *the transformer, to take a sequence of token-position pairs as an argument, then it is completely invariant to ordering, and therefore totally bayesian*.
- ****Topos Theory tantalizes, but totally trashed?****: A member shared a paper on the intersection of **AI** and **topos theory** ([ArXiv link](https://www.arxiv.org/abs/2508.08293)), questioning its legitimacy and practicality, with others chiming in.
   - Another member, however, dismissed **category theory** as *completely useless* for **ML**, arguing that the need for generalization from finite datasets necessitates sampling theory and L_2 spaces.
- ****Sin or Cos? positional perplexities persist!****: A discussion about understanding positional encoding in transformers arose, where a member sought advice on how to understand concepts with limited or low-quality data.
   - One member explained the use of both **sine** and **cosine** in positional encoding by stating that if you just use sin then its hard for model to understand if its the same angle or not because if the vector is 0.5 it could represent 30 degrees and 150 degrees, however it was argued in layers they'd be able to reconstruct each other because *sin(2x) = 2 sin(x) cos(x)*.
- ****Deep Learning dives into turbulent Dynamics****: A member wondered if deep learning could be explained as reversing turbulent flow, comparing it to reconstructing a large vortex from small vortices, which another member described as schizophrenic.
   - In contrast, another member suggested that [this paper](https://arxiv.org/pdf/2303.08797) does *what your idle thought are looking... but do it straightforward*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1416176802862338160)** (19 messages🔥): 

> `Spiking Brain-inspired Large Models, Anthropic's Research, OpenAI's Research, Decreasing Work-Related Usage, Noise Among Older Cohorts` 


- **Gaslighting Linguistic Analysis Reveals Disinformation Potential**: A member joked that the Spanish translation of gaslighting, *manipulación psicológica*, makes sense regarding potential [disinformation use-cases](https://www.arxiv.org/pdf/2509.05276) involving **Spiking Brain-inspired Large Models**.
   - Another member shared a link to a paper titled "SpikingBrain Technical Report: Spiking Brain-inspired Large Models" to explore this connection.
- **Anthropic & OpenAI Economic Reports**: Members looked at reports released on the same day from both [Anthropic](https://www.anthropic.com/research/economic-index-geography) and [OpenAI](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) concerning user behavior with AI.
   - The discussion focused on *what users are doing with AI*, speculating whether one company was attempting to scoop the other with these releases.
- **AI-as-Friend use case is NOT in OpenAI report**: A member noted that the **OpenAI** report doesn't cover the *"AI as a friend"* use case, particularly in light of recent issues with **sycophancy**.
   - The same member observed that work-related usage is decreasing, relative to other usage patterns.
- **Smoothness Differences Across Age Cohorts in AI Usage**: It was observed that the 18-25 age cohort line on the usage chart is smoother than others.
   - One possible reason given for this is due to the 18-25 cohort having the most users or the least noise in their data.
- **Older Cohorts face Increased Noise**: An observation was made that the noise for older cohorts increases, potentially due to low numbers of available samples.
   - This increasing noise could be due to the increasing variance in the data from older demographics.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1416447947347988491)** (3 messages): 

> `` 


- **No relevant agent topics found**: Unfortunately, there were no messages found that contained topics of interest, so a good summary could not be created.
- **Filler topic**: Unfortunately, there were no messages found that contained topics of interest, so a good summary could not be created.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1416190525484302336)** (15 messages🔥): 

> `MobileLLM-R1-950M Release, AI Alignment, AI Constitutional Assembly, Cloud Providers Profiting, NVIDIA` 


- **Facebook Releases MobileLLM-R1-950M**: Facebook released their new [MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M).
   - The release aims to bring powerful language models to mobile devices, enabling on-device processing and reducing reliance on cloud services.
- **Aligners All The Way Down**: A member shared a link to [alignmentalignment.ai](https://alignmentalignment.ai/Aligners), commenting *Aligners all of the way down...*.
   - The link shares content and research related to AI alignment.
- **Constitutional Assembly of AI Designers Suggested**: A member linked to a tweet suggesting a *constitutional assembly of artificial intelligence designers*.
   - The link is to a [tweet](https://x.com/ShashwatGoel7/status/1966527903568637972) from Shaswat Goel.
- **Cloud Providers Cash In on AI Gold Rush**: A member stated that *the only people making any money of this are cloud service providers and cloud infrastructure providers*.
   - Another member responded, *You know what they say about a gold rush, Sell shovels*, with **NVIDIA** being included as under cloud infrastructure provider.
- **AI PDF Editor Ad in AI Safety Video**: A user pointed out the irony of seeing an ad for an **AI-powered PDF Editor** in the description of a video related to **AI Safety**.
   - The user questioned whether anything could be *more hypocritical*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1416163016021377146)** (101 messages🔥🔥): 

> `MBA-ification of Startups, AI texting concierge poke.com, OpenAI Model Spec Update, Naveen Rao leaves Databricks, GPT-5 ‘High New’` 


- **Startups Succumbing to MBA-ification**: A thread sparked by Michael Seibel laments that CS majors act like MBA grads, chasing fundraising and valuation over building cool things and solving user problems, as can be seen [here](https://xcancel.com/mwseibel/status/1965936563306864706).
   - Replies debate whether this shift is natural late-adoption or a consequence of investor/YC incentives.
- **Poke.com Launches AI Texting Concierge**: Interaction introduced **poke.com**, a new AI texting service, along with news of a **$15M Series A** led by General Catalyst ([tweet](https://xcancel.com/interaction/status/1965093198482866317)).
   - Some see slick UX and viral storytelling, while others question usefulness, clarity, and the AI’s tone; the product texts on your behalf to coordinate get-togethers, dates, travel, etc.
- **xAI Pivots to Specialist AI Tutors**: Rohan Paul highlights xAI's shift: laying off **500** generalist data annotators while scaling specialist AI tutors **10x** ([tweet](https://xcancel.com/rohanpaul_ai/status/1966943783276396730?s=46)).
   - The move narrows human-in-the-loop work to expensive domain experts and leans on automation for routine tasks, aiming to boost precision on high-risk topics.
- **S3 Vectors May Slay Vector DBs?**: Discussion ensued from [this blogpost](https://zilliz.com/blog/will-amazon-s3-vectors-kill-vector-databases-or-save-them) about whether **Amazon S3 Vectors** will displace traditional vector databases, as embedding solutions converge on a cost and latency slider from local nvme disk to object storage (s3).
   - One user quoted the surprising claim that *a popular AI note-taking app spends twice as much on vector search as on OpenAI API calls*, and wondered if they should listen more carefully to 'RAG is Dead'.
- **GPT-5 Codex Upgrades**: OpenAI released upgrades to **Codex**, their coding model, including a new version of **GPT-5** and a small recap post ([link](https://www.latent.space/p/gpt5-codex)).
   - One user reported that the `--resume` flag broke during the update and would not let them restore their conversation.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1416163405265371299)** (10 messages🔥): 

> `Higgsfield $50M raise, Adobe value shift, GenZ AI Founders` 


- **Higgsfield Hustles to $50M A Round**: AI video startup **Higgsfield** announced a [$50M Series A](https://xcancel.com/arfurrock/status/1966588530064289841?s=46) led by **GFT Ventures**, reaching $50M revenue run-rate—4.5× growth in three months—and is launching **Higgsfield Ventures** to support AI-native Gen Z founders.
- **Adobe's AI Angst: $100B Value Vanishes?**: Anjney Midha suggests [AI editing advances](https://xcancel.com/anjneymidha/status/1967266304878068044?s=46) may swipe $100B from **Adobe's** market cap towards frontier AI labs (**Flux Kontext**, **Gemini Nano**).
- **GenZ to get AI Boost**: **Higgsfield Ventures** plans to support AI-native **Gen Z** founders, giving more opportunities to young talent.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1416137308175994962)** (74 messages🔥🔥): 

> `Nepal Discord Election, MLC-LLM issues, sglang vs vllm, GPT-OSSH4 in claude code, Demis Hassabis` 


- ****Nepal** elects leader on Discord!**: Members joked about **Nepal** voting for its leader on **Discord** and what's next, AI waifus and husbandos for all citizens.
   - A member shared an article about [Nepal going through their entire revolution right now](https://adriananimov.substack.com/p/empyre-of-ai).
- ****MLC-LLM** experiment encounters issues**: One of the members is experimenting with adding custom models to **MLC-LLM** ([https://github.com/mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)) but keeps encountering issues when injecting the model.
   - A member suggested that this might be due to mixing up context by sessions erroneously not being terminated properly, or may be similar to [this issue](https://github.com/ggml-org/llama.cpp/pull/15913) on *llama.cpp*.
- ****sglang** and **vllm** used internally**: A member stated that they only use **sglang** and **vllm** internally.
   - Another one mentioned that they haven't tried **sglang** before, but its git repo looks promising, and he was primarily looking to utilize **mlc** to attempt to experiment with **gpt-ossh4** in claude code.
- ****XML** preferred over **JSON** by the **Qwen** team**: It was noted that the **Qwen** team prefers **XML** over **JSON** and a member is planning to do the same with their agentic system before releasing it.
   - It's believed something new is needed that is much more token conscious because all the whitespace is not resource friendly.
- **Sir **Demis Hassabis** discusses world models**: A member shared a [YouTube video](https://www.youtube.com/watch?v=Kr3Sh2PKA8YI) with Sir **Demis Hassabis** discussing the pursuit of multi-modal (building world models) approach toward AGI and Embodied A.I systems, the limitation of LLMs, and the mindblowing world of **Genie 3**.
   - This video covers **Alphafold** real-world achievements in research biology and medicine.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1417076474095337482)** (1 messages): 

> `Adversarial Idea Presentation, Strength in Weakness` 


- **Adversarial Idea Presentation Reveals Hidden Strengths**: Presenting an idea in its *adversarial mode* can *inadvertently find additional strengths* that it just frames as weaknesses.
- **Framing Weaknesses as Strengths**: When ideas are presented adversarially, potential benefits may be perceived as drawbacks, highlighting the importance of framing.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1417205231858618388)** (5 messages): 

> `OpenAI Economic Research, Anthropic Economic Index, ChatGPT usage growth, AI Friend mapping` 


- **AI Economics Papers Drop Simultaneously**: Both **OpenAI** and **Anthropic** simultaneously released papers; OpenAI with [Economic Research on ChatGPT Usage](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) and Anthropic with their [Economic Index September 2025 Report](https://www.anthropic.com/research/anthropic-economic-index-september-2025-report).
- **ChatGPT User Base and Engagement Rocketing**: According to [OpenAI's data](https://openai.com/index/how-people-are-using-chatgpt/), there's a substantial increase in both the number of people signing up for **ChatGPT** and the usage per person.
- **"AI Friend" Category faces Skepticism**: A member questioned the mapping of certain data to the category of "ai friend", expressing *doubt*.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1416509716905070632)** (2 messages): 

> `DNS Tunneling Chat Client, AI Killing Focus` 


- **LLMs Fly High with DNS Tunneling Chat Client**: A member created a [tool to chat with LLMs from WiFi captive portals](https://github.com/accupham/llm-dns-proxy) using DNS tunneling, enabling LLM access on airplanes without extra fees.
   - They asked for *roasts*, which some might consider a risky request given the current AI climate.
- **AI Blamed for Attention Deficit Apocalypse**: A member shared a [blog post](https://blog.maxcomperatore.com/why-ai-is-killing-your-focus-and-the-system-to-fix-it) arguing that AI is harming our ability to focus.
   - The post details a system to reclaim focus in a world dominated by AI-driven distractions.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1417205231858618388)** (5 messages): 

> `OpenAI, Anthropic, AI Usage, AI Friend` 


- **Simultaneous AI Economic Research Publication Race?**: Members shared links to economic research papers published simultaneously by [OpenAI](https://cdn.openai.com/pdf/a253471f-8260-40c6-a2cc-aa93fe9f142e/economic-research-chatgpt-usage-paper.pdf) and [Anthropic](https://www.anthropic.com/research/anthropic-economic-index-september-2025-report), with one member wondering about the timing of their release.
   - The OpenAI paper studies **ChatGPT usage** trends.
- **ChatGPT User Growth Shows No Sign of Stopping**: Members shared that [OpenAI noted](https://openai.com/index/how-people-are-using-chatgpt/) that the number of people signing up for **ChatGPT** is increasing, in addition to the usage per person.
   - Attached were multiple images, but discussion was limited.
- **"AI Friend" Use Case Under Scrutiny**: A member questioned which specific data points from a graph might be mapping to the use case of an *"AI Friend"*.
   - They then simply stated *"doubt"*.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1417244881042276424)** (3 messages): 

> `fastWorkflow beats Claude Opus 4.1, GEPA API Improvement, Tau Bench retail` 


- **fastWorkflow framework beats Claude Opus 4.1!**: A member found that their new **fastWorkflow** framework implementation **matches Claude Opus 4.1** on the Tau Bench dev set.
   - These tests used **DSPy** for agents and parameter extraction and the [retail workflow example](https://github.com/radiantlogicinc/fastworkflow) from their repo.
- **GEPA API improvement requested!**: Another member expressed interest in learning from experience using **GEPA** for agentic use cases.
   - They also asked to be notified of any potential improvements to the **GEPA API** that could better support such use cases.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1416465047311356034)** (51 messages🔥): 

> `GEPA for code generation, Manim and DSPy video, Rules as inputs for optimization, MCP Server, Zero Shot Categorization` 


- **GEPA Generates Great Code**: DSPy's latest optimizer, **GEPA**, was built exactly with code generation in mind, showcased in [this paper](https://arxiv.org/pdf/2507.19457) (section 6) for generating **CUDA/C++ code** for GPU/NPUs.
   - One of the original creators happily offered to discuss GEPA in greater detail.
- **MultiMedia Manim Magician Makes Movie Magic**: A member shared a [video](https://cdn.discordapp.com/attachments/1161519469319946286/1417121692026929163/RequestGeneration.mp4?ex=68c9fdac&is=68c8ac2c&hm=727806c1f99beaee0816d280cbdd41519070c660a1efb588ee240b5166ab134f&) created with a custom pipeline using **DSPy**, which included narration script generation, **Manim scene generation**, and an auto-fix feedback loop.
   - The video utilized **Signatures**, **KNNFewShot**, and **ChainOfThought**, but is closed source at the moment.
- **Optimization Overload Overwhelms**: A user found that *running optimization after each small change of the instructions seems to be too heavy and slow a workflow*.
   - It was suggested to add a list of rules as part of the input as well, so the prompts are optimized to be **adaptable to different rules and possibly unseen rules**.
- **MCP Server Seeking DSPy Savvy**: A user is curious if anyone has used **DSPy** to tune their **MCP server** descriptions and examples and thinks tuning for the average result is probably good enough.
   - Another member validated this idea, suggesting that the user could infer the calling LM based on the client, and that *this idea is so good that I'd bet people would be willing to pay for it as a service if you pulled it off*.
- **Categorizing with Class**: A user is trying to perform **zero-shot categorization** of ~2k texts (emails) and wants to provide examples or seed words for each topic.
   - It was suggested to use `typing.Literal` in the signature definition and load JSON data to create `dspy.Example` objects, and pointed to [this tutorial](https://dspy.ai/tutorials/rag/).


  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1417168565437726741)** (1 messages): 

> `Contextual Chunking, ColBERT Models, Late Chunking, MaxSim Algorithm` 


- **Contextual Chunking Boosts Performance**: A user found that prepending a **contextual summary** to each chunk significantly improves performance, even with **ColBERT models**.
   - However, they noted that generating a summary for every chunk is costly, prompting a search for more efficient alternatives.
- **Late Chunking Explored for ColBERT**: The user proposes using **late chunking** with **ColBERT**: encoding the entire text at once and then splitting the embeddings into chunks afterward.
   - This approach assigns each chunk its corresponding embedding list for more efficient processing.
- **MaxSim Algorithm and CLS Token Reliance**: The user questions whether **ColBERT's maxsim algorithm** relies on the **CLS token** for optimal performance, fearing issues with chunks from the middle of the text that lack a CLS token.
   - They inquire if it's safe to omit the **CLS token** when applying **maxsim** to each chunk in this scenario.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1416149270075019294)** (18 messages🔥): 

> `Mojo Package Managers, Binary vs Source Distribution, Pixi and Conda, Apple M1 Compatibility` 


- ****Mojo Packaging Mania**: Community Mulls Managerial Methods**: A community member inquired about creating a new package manager for Mojo, specifically to handle binary distribution, however the Mojo team [pointed out](https://www.modular.com/)
   - `.mojopackage` already covers many benefits of binary distribution and works with **Pixi**, also the team is intentionally leaning on **Conda** and standard **Python** package formats to help adoption.
- ****Binary Blues**: Source Distribution Strides Strong**: It was noted that there are downsides to binary distribution, which is why many languages prefer source distribution, but the user was curious whether there might be scenarios where a more explicit binary-focused package manager could be useful, like for large dependencies or prebuilt libraries.
   - The Mojo team has stated that Mojo can compile ~200k lines of code in **30 seconds** on a laptop and that Pixi handles **C/C++** dependencies with **Conda**.
- ****Pixi Power-Up**: Decentralizing Dependencies Dynamically**: A community member highlighted [pixi-build-mojo](https://www.modular.com/t/mojo-packages-can-now-depend-on-other-packages-in-github/2144), enabling a fully decentralized package system like **Go** by using packages in **Git**.
   - The ability to specify system dependencies with **Pixi** was also mentioned to be quite effective.
- ****M1 Mayhem**: MacBook Woes with Mojo?**: A user with an **Apple M1 MacBook Air** running **Python 3.13** inquired about Mojo/MAX compatibility with this version of **Python**.
   - The Mojo team confirmed it's compatible, encouraging the use of `pixi` for isolated Python versions, and suggested that running on CPU should be fine (albeit slower) since **Apple Metal** support is in its early stages.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1416984660000378881)** (33 messages🔥): 

> `InlineList Removal, Small List Optimization, Allocator/Storage API, Mojo LSP Status, Network update` 


- **InlineList's Gone: Where Did It Go?**: Members discussed the removal of `InlineList`, with concerns raised about the alternatives (`InlineArray` and `List`) not fully addressing its niche, as the [changelog](https://example.com/changelog) suggests using `InlineArray` or `List` with a `capacity` constructor.
   - One member suggested that a stack-allocated variable size length type with fixed capacity would be ideal, and another member mentioned that the Allocator API might be the path forward.
- **Small List Optimization Stalled**: A "small list optimization" exists, fitting some items inline, but they get copied to the heap if the list grows, with one member mentioning that making the inline size a parameter might be explored.
   - A member mentioned that `List` doesn't have SBO (Small Buffer Optimization) currently due to complexities in exposing it to the user and trait requirements for movable elements.
- **Allocator API Coming Soon?**: Discussion revolved around the potential of an allocator/storage API to handle inline allocation, with one member stating, *What I'm hearing is that I need to go work on my allocator/storage API more*.
   - This API's development is pending parametric traits and `requires`, delaying its progress.
- **Mojo Gets Major LSP Rework**: A member inquired about the status of Mojo's Language Server Protocol (LSP), and another replied that it exists and is undergoing a *major rework* soon.
   - No further details about the rework were given.
- **Network Update Blocked 🚧**: A member expressed anticipation for a network update, but another responded, *Lots of blockers there*.
   - The nature of these blockers was not specified.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1416146127786479757)** (36 messages🔥): 

> `RepoMap for Aider, Free C# Models, AGI Predictions, LM Studio issues, GPT-5 Codex` 


- **RepoMap Boosts Aider's Real-World Performance**: A user noted that using **RepoMap** with Aider provides extra context like filenames and function signatures, enhancing **LLM** awareness of available resources, theoretically leading to [leaderboard results](https://leaderboard.techfren.net/) that *more closely reflect real-world* coding scenarios.
   - However, they conceded that **benchmark tests** on simple problems still leave a significant gap compared to real-world code experiences.
- **Seek and ye shall find: Free C# Models**: A user sought a free, non-local model proficient in **C#**, and other members suggested trying **Qwen Coder** and **Deepseek Coder** via [OpenRouter](https://openrouter.ai/), noting that Gemini 2.5 Pro might have a free tier.
   - The user later reported issues using **Qwen** via OpenRouter, receiving an *AuthenticationError* due to a potentially incorrect API key.
- **AGI Arrival: When Will AI Slash White-Collar Jobs?**: A user polled the channel on when **AGI** might reduce white-collar jobs by over 30%, offering choices ranging from 2027 to beyond 2040, defining **AGI** in terms of economic impact rather than abstract intelligence.
   - Another member jokingly predicted it would happen *somewhere between now and heat death of the universe, or maybe never*.
- **LM Studio and Aider: A Rocky Start**: A user encountered problems running Aider with a local **Qwen3-Coder-30B** model in **LM Studio**, sharing images of their setup but without specifying the exact issue.
   - Another member inquired whether the necessary environment variables were set, hinting at a potential configuration problem.
- **GPT-5 Codex: The New Coding Model on the Block?**: A user inquired about Aider's score for **GPT-5 Codex**, referencing a [The New Stack article](https://thenewstack.io/openai-launches-a-new-gpt-5-model-for-its-codex-coding-agent/) on the new model.
   - Another clarified that it is *Not available through the API yet.*


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1416525332135149628)** (6 messages): 

> `Ollama context window limits not respected, lm studio or llamafile suggestion, --watch-files implementation on Linux, Gemini issues with Aider` 


- **Ollama context length limits ignored**: A user reported that **aider** with **Ollama** doesn't respect context window limits, leading to excessive VRAM usage and freezing the machine, despite setting `OLLAMA_CONTEXT_LENGTH` and other parameters in configuration files.
   - The user has configured `num_ctx` and `max_tokens` in `.aider.model.settings.yml` and `max_tokens`, `max_input_tokens`, and `max_output_tokens` in `.aider.model.metadata.json`.
- **Alternatives to Ollama suggested**: A member suggested using **LM Studio** or **llamafile** as alternatives.
   - No further discussion or reasoning was provided.
- **“--watch-files” implementation based on filesystem**: A member inquired how the `--watch-files` option works in Linux, specifically if it relies on **inotify** or requires communication from an IDE/editor.
   - Another member clarified that it's filesystem based and doesn't need specific messages from an editor.
- **Gemini integration halted, possibly due to user-agent blocking**: A user reported issues with **aider** hanging while waiting for a response from **Gemini** models, despite the token being correct and functional with `curl` and the **Gemini CLI**.
   - The user suspects that **Gemini** might be blocking based on the user agent and is running **aider 0.86.1**.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1417328287172263976)** (1 messages): 

> `Earning $100k in a week, Telegram scams` 


- **Unrealistic earning promise dangles hefty commission**: A member offered a get-rich-quick scheme, promising to help the first 10 people interested in *earning $100k or more within a week*, asking for **10% reimbursement of profits** upon receipt.
   - Interested parties were instructed to initiate contact via Telegram username @Joanna_Dwayne, a move that raises suspicion of a scam.
- **Telegram Contact Raises Red Flags**: The request to contact a user via **Telegram** for a *get rich quick scheme* is a common pattern used by scammers.
   - Users should be wary of any offers that require initial contact on unverified channels and promise unrealistically high returns.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1416203482578419913)** (25 messages🔥): 

> `Tensor.assign return value, GEMM TFLOPs measurement, Winograd bounty lock, Rangeify bugs, CUDA 12.0 and sm_35` 


- **Debate assign() vs store() Functionality**: The need for `assign()` to return a value versus acting like `store()` was questioned, pondering if it's just a convenience since the return value is often unused in examples.
   - It was suggested that *linking both the buffer and the store to the load* is a possible alternative.
- **GEMM 165+ TFLOPs Bounty Measurement Questioned**: A question arose about how to measure the **165+ TFLOP GEMM** bounty target on an RTX 4090, suspecting it might be unachievable at the stated boost clock of **2.52 GHz**.
   - The theoretical peak throughput for FP16/BF16 with FP32 accumulate on an RTX 4090 is around **165.15 TFLOPs** at that clock speed, but the questioner implied that a higher clock might be needed to reach the bounty target.
- **Winograd Bounty Requirement Clarified**: A user inquired about the requirements to lock the **Winograd bounty**, having found a *necessary and sufficient condition* to identify Winograd compatible convolutions.
   - George Hotz clarified that *locks are only after code is correct* while there are fixups to merge it.
- **Rangeify Bugs List Shared**: A list of **Rangeify bugs** was shared for people to investigate and fix, emphasizing that many are likely simple fixes.
   - `RANGEIFY=1` is described as *the new scheduler that can create things like flash attention and beyond*.
- **CUDA 12.0 drops support for sm_35**: The CUDA issue is that **CUDA 12.0** dropped support for **sm_35** used by Ocelot.
   - The minimal flag was added after 12.4


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1416898796943708170)** (12 messages🔥): 

> `GPU Utilization in tinygrad, VIZ=1 Profiler, NixOS Patch for CUDA, Profiler 404 error` 


- **GPU Utilization Plummets in tinygrad**: A tinygrad user reported seeing poor **GPU utilization** and sought advice on improving it, particularly when switching from CPU to CUDA.
   - Another user suggested using `PROFILE=1` (or `VIZ=1`) to identify where time is being spent, noting that saving tensors to disk can be a bottleneck, offering the user to examine the profile to help determine the source of the issue.
- **`VIZ=1` Profiler Unifies Profiling Options**: `PROFILE=1` is merely an alias for `VIZ=1`, and the former has been removed to reduce redundancy and streamline profiling in tinygrad.
   - George Hotz noted *"having two options is worse than having one"*, which motivated the change, simplifying the profiling process.
- **NixOS CUDA Patch Incoming**: A tinygrad user is planning to investigate and potentially fix a broken profiler issue on their **NixOS** distribution after submitting a patch related to **CUDA**.
   - The user mentioned they had to patch file paths, indicating a likely issue with how the distro package handles **CUDA** dependencies.
- **Profiler Faces 404 Error**: A tinygrad user encountered a **404 error** when trying to access `/js/index.js` while using the profiler.
   - This error suggests a potential issue with the profiler's file paths, or the location of `/js/index.js`.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1416140810352594974)** (19 messages🔥): 

> `Credits Rollover, Daily Credits Stopped, Clone Website using AI, Subscription Renewal Issues, Knowledge Limit Increase` 


- ****Credits Confusion** Clouds Users**: Users are inquiring about **credits rollover** and the cessation of **daily 300 credit** allocations.
   - One user specifically reported their subscription renewal was set for **September 14th** but they haven't been charged nor received more credits.
- **Website Cloning Craze Kicks Off!**: A user mentioned that it's easy to clone a website using **Manus** or other **AI tools**.
   - He was impressed that his feature idea proposed on **August 12th** was implemented just **16 days later** [in this discord channel](https://discord.com/channels/1348819876348825620/1349440650495398020/1404845589497122887).
- **Collaboration Creates Coding Confidence**: A user is experimenting with **Manus Collaboration** with friends for coding tasks.
   - Another user is working on a potential new feature that, if successful, promises to significantly enhance **Manus' efficiency** as a coding assistant.
- ****Knowledge Navigation** Needs Nurturing**: Several users are asking about increasing the **knowledge limit**, specifically if it's possible to exceed **20**.
   - No concrete answers were provided in the discussion.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1416155480320905297)** (10 messages🔥): 

> `MCP Servers, Reinforcement Learning, Integration Testing, MCP Server Efficiency, NL Interface` 


- **Scalable Integration Testing with MCP Servers**: Members are thinking through scalable **integration testing** and **reinforcement learning** over **MCP server** tool use.
   - They are considering flagging when connecting to an MCP server that the server is in some kind of **development or simulation mode** for robust training to simulate real tool behavior without messing with production DBs.
- **Score MCP Servers for Efficiency**: One member is researching how to score **MCP servers** for their **efficiency** in different clients to determine when the marginal improvement in efficiency is not worth additional coding in the server.
   - The trade-off is between every kind of prompt sharing one node and between every kind of prompt having its own node -- but how many **API calls** is 'too many' for a user story?
- **MCP as CLI for Applications**: Some folks are considering using **MCP** as **CLI** to applications, in a form of **NL interface** and **adaptive dash boarding/reporting**.
   - The idea is to use it as a **UI/UX interface** to enterprise apps through NL.
- **Golang Streaming HTTP MCP Server Project**: A member opened up their [mcp-server-go project](https://github.com/ggoodman/mcp-server-go), a **golang streaming http MCP server** designed to address the more challenging requirements of enterprise-like situations.
   - It is designed for **scalability**, and includes features like **auth**, **sessions and resumability**, and **dynamic capabilities**.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1417266476141772966)** (2 messages): 

> `MCP Resource Integration with LLMs, Claude Desktop Automation, Discord Channel Restrictions` 


- **LLM Learns MCP Resources**: A member inquired about automating the process of LLMs reading **MCP resources** before answering user questions and executing tools, aiming for a workflow where the LLM is pre-loaded with knowledge.
   - The member noted that currently, with **Claude desktop**, resources must be manually added to the chat window before asking questions.
- **Claude Desktop Lacks Automation**: A member confirmed that **Claude desktop** functions as intended, requiring manual addition of resources to the chat window.
   - They clarified that there is no automated process for LLMs to read resources before interacting with users in the current setup.
- **Discord's Focus Narrowed**: It was clarified that the Discord channel is restricted to discussions on the governance of the **MCP protocol itself**.
   - General **MCP questions** should be directed elsewhere, with an offer of guidance via DM for those seeking it.


  

---


---


---

