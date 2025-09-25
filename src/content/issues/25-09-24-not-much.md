---
id: MjAyNS0w
title: not much happened today
date: '2025-09-24T05:44:39.731046Z'
description: >-
  **Alibaba** unveiled the **Qwen3** model family including **Qwen3-Max** and
  **Qwen3-VL** with a native 256K context window expandable to 1M, strong OCR in
  32 languages, and rapid release velocity (~3.5 releases/month) backed by a
  $52B infrastructure roadmap. **OpenAI** launched **GPT-5 Codex**, an
  agent-optimized coding model with up to **400K context** and adaptive
  reasoning priced at $1.25/$10 per million tokens, integrated into Cline and
  benchmarked in WebDev arenas. **Meta AI FAIR** released the open-weight **Code
  World Model (CWM) 32B**, a dense code generation model with strong benchmark
  scores (e.g., 65.8% SWE-bench Verified, 96.6% Math-500) and public safety
  reports. Ecosystem updates include GitHub Copilot's new embedding model for
  faster code search and Anthropic's Claude Sonnet 4 and Opus 4.1 integration
  into Microsoft 365 Copilot. The vLLM 0.10.2 update introduces Decode Context
  Parallel (DCP) for improved system performance.
companies:
  - alibaba
  - openai
  - meta-ai-fair
  - huggingface
  - anthropic
  - microsoft
  - github
models:
  - qwen3-max
  - qwen3-vl
  - qwen3-coder-plus
  - gpt-5-codex
  - code-world-model-32b
  - claude-sonnet-4
  - claude-opus-4.1
topics:
  - context-windows
  - code-generation
  - model-releases
  - model-benchmarking
  - api
  - model-optimization
  - multimodality
  - software-engineering
  - model-training
people:
  - huybery
  - akhaliq
  - lmarena_ai
  - gdb
  - ylecun
  - pierceboggan
  - julesagent
---


**a quiet day**

> AI News for 9/24/2025-9/25/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (194 channels, and 2885 messages) for you. Estimated reading time saved (at 200wpm): 230 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

You can catch [**Day 2 of AIE Paris here**](https://www.youtube.com/watch?v=wyUdpmj9-64), where tickets for AIE Europe 2026 were announced. You should also [**apply for Wave 2 of AIE CODE**](https://apply.ai.engineer/) in NYC in November — it'll be a big one.

---

# AI Twitter Recap

**Alibaba’s Qwen3 push: Max, VL, Coder and a $52B roadmap**

- **Qwen3-Max, Qwen3-VL, and shipping velocity**: Alibaba/Tongyi unveiled a sweep of models: flagship **Qwen3-Max** (now default in Anycoder) and open-sourced **Qwen3-VL** with a native 256K context (expandable to 1M), stronger OCR in 32 languages, precise event localization in 2h videos, GUI operation/coding, and leading risk detection. Releases hit Hugging Face, ModelScope, GitHub, and Alibaba Cloud’s Model Studio; community platforms onboarded quickly (e.g., Yupp added Qwen3 Max and Qwen3 VL 235B A22B Instruct/Thinking; LMArena added three Qwen3 models). Alibaba touted unmatched shipping velocity (~3.5 releases/month, many open weights) and a multi-year infrastructure roadmap discussed at Yunqi, with commentary noting a “$52B war chest” and major compute scale-up claims. See announcements and threads: [@huybery](https://twitter.com/huybery/status/1970649341582024953), [@huybery on Qwen3-VL](https://twitter.com/huybery/status/1970650821747712209), [@Ali_TongyiLab (VL release)](https://twitter.com/Ali_TongyiLab/status/1970665194390220864), [Anycoder defaults](https://twitter.com/_akhaliq/status/1970618469344235677), [Yupp adds Qwen models](https://twitter.com/yupp_ai/status/1970640795259851079), [LMArena adds Qwen3](https://twitter.com/lmarena_ai/status/1970920636957831611), [shipping velocity](https://twitter.com/awnihannun/status/1970839682503348623), [Yunqi recap](https://twitter.com/Smol_AI/status/1970842828512088486), [exec clips/roadmap](https://twitter.com/swyx/status/1970847377058676849).
- **Qwen3-Coder-Plus and API improvements**: The coding line got targeted upgrades (terminal tasking, scaffold adaptation; API fixes), with early competitive signals in WebDev Arena and agent toolchains. Details: [API update](https://twitter.com/huybery/status/1970652792848293926), [WebDev Arena prompt](https://twitter.com/lmarena_ai/status/1970962780225507775).

**Coding models and agents: GPT-5 Codex lands; Meta’s 32B CWM**

- **GPT-5 Codex (agent-optimized) is live**: OpenAI’s “Codex” variant is in the API and agent tools. Highlights: up to **400K context**, “adaptive reasoning” with variable thinking that uses far fewer tokens on simple tasks and more on complex ones, and pricing around **$1.25/$10 per million tokens**. It’s integrated in Cline (with a “thinking slider”), and being benchmarked in webdev arenas and agent workflows. Links: [API availability](https://twitter.com/gdb/status/1970631954887565823), [Cline integration](https://twitter.com/cline/status/1970619799119241709), [Cline details](https://twitter.com/cline/status/1970619811853148550), [WebDev Arena](https://twitter.com/lmarena_ai/status/1970962780225507775). Field reports compare throughput vs Sonnet/GPT-5 on long-context and agent runtimes: [example](https://twitter.com/zachtratar/status/1970625784500130065), [long-context retrieval comparison](https://twitter.com/scaling01/status/1970661469667660100).
- **Meta FAIR’s Code World Model (CWM) 32B (research)**: Meta released an open-weight 32B dense model under a research license that frames code generation as planning with a world model of code execution. Reported pass@1: **65.8% SWE-bench Verified**, **68.6% LiveCodeBench**, **96.6% Math-500**, **76.0% AIME 2024**. Technical report, weights, and code are public, with a safety preparedness report from SEAL/AI Security. Links: [@AIatMeta](https://twitter.com/AIatMeta/status/1970963571753222319), [@ylecun](https://twitter.com/ylecun/status/1970967341052854748), [metrics summary](https://twitter.com/alexandr_wang/status/1970973317227225433), [safety prep](https://twitter.com/summeryue0/status/1970971944557346851).
- **Ecosystem updates**: GitHub Copilot’s new embedding model and training writeup (for faster, more accurate code search) [blog link](https://twitter.com/pierceboggan/status/1970950784251724007); Jules agent now acts on PR feedback [link](https://twitter.com/julesagent/status/1970640318606258605); Claude Sonnet 4 and Opus 4.1 are now in Microsoft 365 Copilot [Anthropic](https://twitter.com/AnthropicAI/status/1970907112831328296).

**Systems and infra: vLLM DCP, multimodal data plumbing, and platform moves**

- **vLLM 0.10.2 adds Decode Context Parallel (DCP)**: Contributed by Kimi/Moonshot, DCP shards KV cache across GPUs to cut duplication, enabling up to **8× larger KV** and **2–3× throughput** on single-node H200—especially helpful for KV-heavy workloads (RL, offline data generation). Quickstart: `vllm serve deepseek-ai/DeepSeek-V3.1-Terminus -tp 8 -dcp 8`. Links: [@vllm_project](https://twitter.com/vllm_project/status/1970814441718755685), [day-0 guides](https://twitter.com/rogerw0108/status/1970619149757096037).
- **Multimodal infra from Perceptron**: The team shared the design behind TensorStream—a tensor-like abstraction for interleaved multimodal data powering their training/inference code—and released technical details for Isaac 0.1, a small VLM emphasizing a simple training recipe and robust grounding. Good discussion on “complexity budgets” and native multimodal abstractions: [design post](https://twitter.com/perceptroninc/status/1970670362355736886), [Isaac report](https://twitter.com/perceptroninc/status/1970701029441483087), [commentary](https://twitter.com/kilian_maciej/status/1970701658494738514), [abstractions +1](https://twitter.com/ArmenAgha/status/1970672682909016242).
- **MCP builders and compliance**: Figma’s MCP server lands in VS Code (and is usable in OpenHands) for “design-to-code” flows [VS Code](https://twitter.com/code/status/1970621943821861217), [OpenHands](https://twitter.com/allhands_ai/status/1970955961293795831); Weaviate gets ISO 27001 [link](https://twitter.com/weaviate_io/status/1970912361381843104); AMD expands partnership with Cohere (models on AMD Instinct, sovereign AI posture) [AMD](https://twitter.com/AMD/status/1970824479279317446); Modular raises **$250M** to push its unified AI infra platform [Modular](https://twitter.com/Modular/status/1970881293933273524).

**Video and multimodal generation: Alibaba Wan2.5, Runway A2D, NVIDIA Lyra, Kling 2.5**

- **Alibaba Wan2.5-Preview (native multimodality)**: New architecture aligns text, image, video, and audio natively with joint multimodal training and RLHF; supports controllable inputs (text/img/audio), synchronized multi-speaker A/V, 1080p 10s cinematic video, and stronger image gen/editing (typography, charts, pixel-level edits). [Announcement](https://twitter.com/Alibaba_Wan/status/1970697244740591917).
- **Runway A2D: autoregressive-to-diffusion VLM**: Adapts existing AR VLMs for parallel diffusion decoding to unlock speed–quality trade-offs without training from scratch; dev preview from internship work shows practical path to diffusion LMs for vision-language. [@runwayml](https://twitter.com/runwayml/status/1970866494729781623), [author thread](https://twitter.com/mariannearr/status/1970936677922382335).
- **NVIDIA Lyra (3D/4D scene reconstruction)**: Feed-forward 3D and 4D scene generation from a single image/video via video diffusion self-distillation; weights on HF. [Overview](https://twitter.com/_akhaliq/status/1970949464606245139), [model](https://twitter.com/_akhaliq/status/1970949559426961484).
- **Kling 2.5 Turbo**: Internal blind tests show significant wins over Seedance/Veo variants across text-to-video and image-to-video; community reels and contests rolling out. [Results](https://twitter.com/Kling_ai/status/1970832920085753893), [contest](https://twitter.com/Kling_ai/status/1970783972033445965).

**Reasoning, RL, and evaluation science**

- **RLPT (RL on Pre-Training Data)**: Trains with self-supervised rewards via next-segment reasoning (ASR+MSR) directly on pretraining corpora—no human labels. On Qwen3‑4B, reported gains: **+3.0 MMLU**, **+8.1 GPQA‑Diamond**, **+6.6 AIME24**, **+5.3 AIME25**. Paper: [tweet](https://twitter.com/arankomatsuzaki/status/1970684035258294548), [arXiv](https://twitter.com/arankomatsuzaki/status/1970684037787492416).
- **APRIL (Active Partial Rollouts in RL)**: Cuts rollout long-tail inefficiency; up to **44%** throughput and **8%** final-accuracy improvements across GRPO/DAPO/GSPO. [tweet](https://twitter.com/iScienceLuvr/status/1970794655270003037), [code/paper](https://twitter.com/iScienceLuvr/status/1970794659661434895).
- **“Soft Tokens, Hard Truths”**: First scalable RL for continuous CoT; soft-token training matches discrete pass@1 and outperforms at pass@32 by boosting diversity; best practice: train soft, infer hard. [tweet](https://twitter.com/arankomatsuzaki/status/1970692910766346277), [arXiv](https://twitter.com/arankomatsuzaki/status/1970692913119277178).
- **Effective reasoning ≠ longer CoTs**: Across 10 LRMs, longer chains and review can correlate with lower accuracy. New metric “Failed-Step Fraction” predicts correctness; FSF-based reranking lifts pass@1 up to **+10%**. [tweet](https://twitter.com/arankomatsuzaki/status/1970691075229864357), [arXiv](https://twitter.com/arankomatsuzaki/status/1970691077683454053).
- **Medical multimodal brittleness**: Stress tests show frontier models often guess correctly without images, flip under trivial prompt changes, and fabricate convincing but flawed reasoning—leaderboards mask fragility. [tweet](https://twitter.com/arankomatsuzaki/status/1970684893966516477), [arXiv](https://twitter.com/arankomatsuzaki/status/1970684896160239984).
- Related: Google’s Test-Time Diffusion Deep Researcher (TTD-DR) applies diffusion-style iterative refinement to long-form research, reporting up to **74.5%** win-rates vs OpenAI Deep Research on certain tasks with better quality–latency tradeoffs. [overview](https://twitter.com/omarsar0/status/1970864565710921891).

**Top tweets (by engagement)**

- [Alibaba’s Wan2.5-Preview: native multimodal A/V generation and editing](https://twitter.com/Alibaba_Wan/status/1970697244740591917) — 1453
- [Qwen3‑VL open-sourced: 256K→1M context, 32‑lang OCR, precise video event localization](https://twitter.com/Ali_TongyiLab/status/1970665194390220864) — 1410.5
- [Sam Altman on datacenter buildout progress in Abilene](https://twitter.com/sama/status/1970812956733739422) — 9917
- [Semiconductor node names (“3nm”, “2nm”) as marketing shorthand, not literal dimensions](https://twitter.com/giffmana/status/1970620746155393441) — 9032.5
- [Claude Sonnet 4 and Opus 4.1 arrive in Microsoft 365 Copilot](https://twitter.com/AnthropicAI/status/1970907112831328296) — 1265
- [Gemini app hits 5B images in <1 month](https://twitter.com/joshwoodward/status/1970894369562796420) — 1183
- [GPT‑5 can solve “minor” open math problems; early evidence and preprint](https://twitter.com/SebastienBubeck/status/1970875019803910478) — 952

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. MiniModel-200M and DeepSeek-V3.1-Terminus Local Release Benchmarks

- [**MiniModel-200M-Base**](https://i.redd.it/clbzeq0i82rf1.png) ([Score: 223, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1np5ey8/minimodel200mbase/)): **MiniModel-200M-Base is a ~200M-parameter LLM trained from scratch on** `10B` **tokens in ~**`110k` **steps (~1 day) on a single RTX 5090, with no gradient accumulation, achieving an effective batch of** `64×2048` **and peak VRAM** `<30 GB`**. Efficiency is attributed to an Adaptive Muon optimizer (claimed** `~2.1×` **data-efficiency vs AdamW), Float8 pretraining (attention in bf16) for** `~30%` **lower VRAM and** `~20%` **higher throughput, ReLU² (Primer), bin-packing to reduce padding from** `>70%` **to** `<5%`**, and full attention with scalar-free QK-norm for stability. Early capability demos include deterministic Fibonacci codegen and recalling the first 20+ digits of π; Apache-2.0 weights/config/tokenizer are released: [Hugging Face](https://huggingface.co/xTimeCrystal/MiniModel-200M-Base).** Top commenters are primarily asking for release of the training code/scripts and more detail on the data mixture; interest centers on reproducibility of the setup.
    - A commenter questions the emphasis on “no gradient accumulation,” arguing it should be mathematically equivalent to a larger effective batch. They note practical caveats where it can diverge: optimizer step-count coupling (e.g., AdamW bias correction, per-step weight decay), LR schedules tied to steps vs tokens, gradient clipping across micro-batches, and stochastic elements (dropout RNG, data order). They’re effectively asking for the concrete rationale or benefits (e.g., throughput/activation memory trade-offs, benchmarking fairness) behind avoiding GA in this training run.
    - Multiple requests ask for release of training code and scripts to enable reproducibility. The implied need is for end-to-end pipelines (data loader, tokenizer, optimizer/scheduler configs, logging/checkpointing) and exact seeds, so others can replicate results on a 200M-parameter setup and compare against baselines.
    - Interest in the data mixture details: commenters want the composition and mixing strategy (domain ratios like code/math/dialogue, up/down-weighting, dedup/filtering, and total pretraining tokens). Given small models’ sensitivity to data curation, they’re asking for the precise recipe to understand why MiniModel-200M-Base performs as reported.
- [**You can now run DeepSeek-V3.1-Terminus on your local device!**](https://i.redd.it/nntm711d61rf1.png) ([Score: 163, Comments: 29](https://www.reddit.com/r/LocalLLM/comments/1np1o9e/you_can_now_run_deepseekv31terminus_on_your_local/)): **Unsloth released Dynamic GGUF quantizations of DeepSeek‑V3.1 Terminus enabling local inference on ~170 GB RAM (and a ~162 GB Ollama-ready build) by per-layer "smart" 1‑bit quantization, shrinking the original ~715 GB model by ~80%. Their Dynamic 3‑bit DeepSeek‑V3.1 (thinking) GGUF scores** `75.6%` **on the Aider Polyglot benchmark—reported as surpassing Claude‑4‑Opus (thinking)—with runnable builds via llama.cpp and an example Ollama tag** `hf.co/unsloth/DeepSeek-V3.1-Terminus-GGUF:TQ1_0`**; resources: [blogpost](https://docs.unsloth.ai/new/unsloth-dynamic-ggufs-on-aider-polyglot), [HF repo](https://huggingface.co/unsloth/DeepSeek-V3.1-Terminus-GGUF), [guide](https://docs.unsloth.ai/basics/deepseek-v3.1). The image appears to be benchmark charts illustrating Dynamic GGUF performance versus baselines and proprietary models.** Top comments question practicality for home users—asking if similar methods could compress 70B–200B models for 16–24 GB VRAM GPUs—while others note the high VRAM/RAM requirement and offer praise.
    - A key question is whether the same approach can make `70B` or `100–200B` models run on `16–24 GB` consumer GPUs. This implies extreme quantization/offloading to fit VRAM, and home-user practical utility hinges on this.
    - One commenter cites a memory footprint drop from `715 GB` to `170 GB` alongside *“solid tool-calling”*. They want head-to-heads against **GLM-4.5** and **Qwen**, suggesting evaluation on tool-use/agentic benchmarks to validate quality vs compression.
    - Even with reductions, practical deployment may still demand on the order of `~100 GB` VRAM (*“Now to find another ~100 gb of vram”*). This would exceed typical `16–24 GB` gaming GPUs, underscoring remaining hardware barriers for local use.

### 2. DIY Local AI Hardware: RTX 3080 20GB Mods and Ryzen AI MAX+ 395

- [**My second modified 3080 20GB from China , for local Ai inference , video and image generation..**](https://www.reddit.com/gallery/1np9rav) ([Score: 219, Comments: 101](https://www.reddit.com/r/LocalLLaMA/comments/1np9rav/my_second_modified_3080_20gb_from_china_for_local/)): **OP showcases a China-modded GeForce RTX 3080 upgraded to 20 GB VRAM (likely 10×16Gb GDDR6X on the 320‑bit bus) for local AI inference/video/image workloads, opting for a triple‑fan cooler over a blower for acoustics. The 2.5‑slot card reportedly holds <**`75°C` **under ~**`300W` **stress, suggesting adequate thermal headroom versus blower variants; otherwise it behaves like a standard [RTX 3080](https://en.wikipedia.org/wiki/GeForce_30_series#GeForce_RTX_3080).** Commenters probe value vs a [RTX 3090](https://en.wikipedia.org/wiki/GeForce_30_series#GeForce_RTX_3090) (more cores, `24 GB` VRAM) and ask about price and driver/vBIOS compatibility for the 20 GB mod. There’s curiosity about a hypothetical `30 GB` 3080 using `3 GB` GDDR6X chips; feasibility is unclear due to GA102 memory‑controller/board routing support for 24Gb densities (see [GDDR6X](https://en.wikipedia.org/wiki/GDDR6#GDDR6X)).
    - Value/perf trade-off vs RTX 3090: a 3080 20GB mod still has the 320‑bit bus (`~760 GB/s`) and fewer SMs than a 3090’s 384‑bit bus (`~936 GB/s`), so for AI/image workloads that are both bandwidth- and VRAM-sensitive, the 3090’s `24GB` and wider bus can be materially faster and allow larger batch sizes/checkpoints. Given used 3090 pricing often hovers around the `$500` mark, commenters argue a `$500` 3080‑20GB is hard to justify unless priced closer to `$350`—otherwise a 3090 (or upcoming 24GB next‑gen options) is a better buy. Specs refs: [RTX 3080](https://www.techpowerup.com/gpu-specs/geforce-rtx-3080.c3621), [RTX 3090](https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3627).
    - Feasibility of a 30GB 3080 using 3GB (24Gb) GDDR6X: in theory, 10× `24Gb` chips would yield `30GB` on the 320‑bit GA102, but it hinges on GA102’s memory controller/BIOS supporting 24Gb densities and proper timing straps—no retail GA102 board shipped with 24Gb devices, so compatibility is unproven. Even if recognized by VBIOS, stability/thermals and memory training could be problematic without AIB-level firmware support. Micron has sampled `24Gb` GDDR6X dies, which makes the capacity plausible on paper: [Micron 24Gb GDDR6X](https://www.micron.com/about/news-and-events/releases/2022/07-14-micron-samples-worlds-first-24gb-gddr6x-memory).
    - Driver/VBIOS considerations for 20GB mods: NVIDIA drivers enumerate VRAM from the VBIOS; as long as the device ID matches and the BIOS includes correct memory straps for the installed GDDR6X density, stock drivers generally work. Many China-market 20GB boards ship with custom VBIOS that properly reports `20GB`; flashing mismatched BIOS can cause instability or bricking, and Ampere BIOS editing is limited, so sourcing a vendor-matched 20GB VBIOS is key. Reference: [TechPowerUp VBIOS collection](https://www.techpowerup.com/vgabios/).
- [**The Ryzen AI MAX+ 395 is a true unicorn (In a good way)**](https://www.reddit.com/r/LocalLLaMA/comments/1nozz23/the_ryzen_ai_max_395_is_a_true_unicorn_in_a_good/) ([Score: 218, Comments: 205](https://www.reddit.com/r/LocalLLaMA/comments/1nozz23/the_ryzen_ai_max_395_is_a_true_unicorn_in_a_good/)): **OP evaluates the cost/perf of the 128 GB [Framework Desktop Mainboard (AMD Ryzen AI Max 300 series)](https://frame.work/products/framework-desktop-mainboard-amd-ryzen-ai-max-300-series?v=FRAFMK0006) for local AI inference versus a DIY desktop with similar specs. A comparable DIY parts list (seeking 4‑channel DDR5 ≥8000 MT/s) tallies to ~**`$2240`**: consumer 4‑channel DDR5 motherboard** `>$600`**, CPU “equivalent” to the 395 MAX+ via the [Ryzen 9950X3D](https://www.amazon.com/AMD-Ryzen-9950X3D-16-Core-Processor/dp/B0DVZSG8D5)** `~$660` **+ [Noctua NH‑D15](https://www.amazon.com/Noctua-NH-D15-heatpipe-NF-A15-140mm/dp/B00L7UZMAK)** `~$130`**, 128 GB [DDR5‑8000 (4×24 GB)](https://www.amazon.com/G-SKILL-Trident-CL38-48-48-128-Desktop-Computer/dp/B0F4M6C65N)** `~$450`**, and a dGPU “similar” to the board’s iGPU (RTX [4060/4060 Ti 16 GB](https://www.amazon.com/MSI-Gaming-GeForce-GDRR6-Boost/dp/B0D3KGNMXP))** `~$400`**. OP argues the Framework board’s unified memory avoids PCIe bandwidth/latency penalties when the GPU accesses large model weights, and the discrete build would draw ≳2× the power (more heat/noise; cf. [room‑heating post](https://www.reddit.com/r/LocalLLaMA/comments/1nogrv2/computer_literally_warms_my_room_by_5_degrees/)). They add that Apple M4 Pro/Max have higher bandwidth but poorer diffusion throughput at ~2× the cost for similar RAM/GPU, while truly higher‑throughput Nvidia setups (e.g., 4× RTX 3090) are far more expensive and power‑hungry; edit: the cited 9955HX3D doesn’t support 4‑channel memory—Threadripper would, but with slower memory speeds.** Top replies request concrete benchmarks (“numbers”) and suggest a potential step‑function if AMD ships 256 GB unified memory. One commenter recommends an RTX 5080 within the same budget for diffusion workloads (VRAM > system RAM), while agreeing that for LLMs, larger unified memory (128 GB+) is advantageous for bigger contexts and model footprints.
    - Workload fit and memory-vs-throughput tradeoff: commenters note that for diffusion/vision workloads an RTX 5080–class GPU will outperform at similar price points, and you don’t need `128GB` RAM for images/video. For LLMs, larger system/unified memory is more valuable (fits bigger models/contexts), aligning with the “truck (capacity) vs sports car (throughput)” analogy; a hypothetical `256GB` unified memory SKU is seen as market-shifting for LLM use cases.
    - Bandwidth bottleneck concern: one user flags “< `256 Gb/s` memory bandwidth,” implying large-context capability but slow inference, especially in prefill where LLMs are memory-bandwidth bound. Unified memory helps host bigger contexts, but limited bandwidth can throttle tokens/sec during prefill, so the device may feel responsive only in generation once KV/cache is warm.
    - Anecdotal perf comparison vs high-end GPU: a user with a RTX 5090 + `96GB` RAM (≈+$1k vs Ryzen AI Max) reports on `gpt-oss-120B` that token generation (TG) speed is roughly similar, but prefill (PP) is `4–15×` faster on the 5090. Takeaway: for local LLMs dominated by prefill, the Ryzen box may underperform compared to top-tier GPUs despite comparable TG throughput.

### 3. LLM Performance Growth Claims and Hype Reactions

- [**Large Language Model Performance Doubles Every 7 Months**](https://spectrum.ieee.org/large-language-model-performance) ([Score: 152, Comments: 57](https://www.reddit.com/r/LocalLLaMA/comments/1np2v1i/large_language_model_performance_doubles_every_7/)): **Post asserts an empirical “AI Moore’s Law” where large language model capability doubles about every** `~7` **months, illustrated by a progress chart ([image](https://preview.redd.it/kysbjgxyr1rf1.png?width=461&format=png&auto=webp&s=74d948ee7a2545582e12175877ff315b071aa0fd)) and framed as sustained exponential gains in benchmark performance. The claim echoes prior explainers on accelerated AI progress, e.g., Computerphile’s overview of an AI analogue to Moore’s Law ([video](https://www.youtube.com/watch?v=evSFeqTZdqs&t=1s)); the post itself does not detail methodology or which benchmarks were aggregated.** Commenters highlight that costs are falling alongside quality (token/model pricing dropping), crediting open-source competition for price pressure; others argue the observation is not new, pointing to earlier coverage like the Computerphile video.
    - Methodology critique of the chart: It appears to convert LLM capability into “human time to complete task” and uses a `50%` success threshold per task, which is highly subjective and task-dependent. Examples raised: “find a fact on the web” can range from seconds to days depending on specificity; “optimize code for a custom chip” isn’t well-defined and could span hours to months; and “start a new company” at `167h` isn’t a meaningful, measurable unit. Without standardized benchmarks and precise task specs, a claim like “doubling every 7 months” risks cherry-picking and misrepresenting true progress.
    - Cost/performance dynamics: Commenters note capability gains alongside falling inference costs, with open models intensifying price competition. Practitioners still rely on 2024–2025 open models like **Mistral**, **Llama 3.1**, and **Qwen 2.5 Coder**, implying perceived improvements are task- and deployment-dependent; cost/perf trade-offs (e.g., local inference vs API), stability, and tooling can outweigh headline “doubling” metrics. Reporting both capability and $/token or $/task would better capture real-world value.
    - Prior art on scaling: The linked Computerphile video, AI’s Version of Moore’s Law? (https://www.youtube.com/watch?v=evSFeqTZdqs&t=1s), reviews LLM scaling trends and distinguishes hardware-driven FLOPs/$ gains from algorithmic efficiency improvements that together create apparent capability doubling. It frames progress as arising from larger models, better training data/recipes, and inference optimizations, cautioning against treating a single “doubling period” as universal across tasks.
- [**Oh my God, what a monster is this?**](https://i.redd.it/1pxmwf50e2rf1.jpeg) ([Score: 590, Comments: 124](https://www.reddit.com/r/LocalLLaMA/comments/1np5te1/oh_my_god_what_a_monster_is_this/)): **The image ([chart](https://i.redd.it/1pxmwf50e2rf1.jpeg)) appears to be a benchmark leaderboard where multiple LLMs reach near- or exactly** `100` **on a task, suggesting a saturated/ceilinged evaluation that can no longer differentiate top-tier models. Commenters note that Chinese frontier models are at or near the top of the chart, implying performance parity with leading Western models.** Notable takes: “If models score 100 then it’s a useless benchmark,” arguing the metric has lost discriminative power; others highlight that Chinese models have reached the frontier, while one criticizes the portrait-mode screenshot of a square chart for poor readability.
    - Benchmark saturation concern: if models hit `100`, it indicates a ceiling effect and weak discriminative power. This raises risks of overfitting/test contamination and pushes the community toward harder or adversarial suites like **MMLU-Pro** and **GPQA**, and robustness/long-context evals, rather than relying on classic **MMLU**, **GSM8K**, or **HumanEval** alone. See MMLU [paper](https://arxiv.org/abs/2009.03300), MMLU-Pro [paper](https://arxiv.org/abs/2406.01574), GPQA [paper](https://arxiv.org/abs/2311.12022).
    - Multiple commenters note the showcased Qwen result is not “local,” which matters because API-hosted models can differ from downloadable weights and local performance after quantization. On-device constraints (VRAM, throughput) and quantization (e.g., `Q4_K_M`) typically cost `~1–5` points on reasoning/code benchmarks and change latency; e.g., running a `7B` at Q4 needs ~`5–6 GB` VRAM, `14B` ~`9–10 GB`, `32B` ~`20–24 GB` ([llama.cpp quantization](https://github.com/ggerganov/llama.cpp/blob/master/docs/quantization.md)).
    - The claim that Chinese models have reached frontier levels aligns with recent reports: **Qwen2.5**, **DeepSeek‑V2**, and **Yi** series publish competitive MMLU/GSM8K/MT‑Bench and coding scores versus established frontier models. See Qwen2.5 [blog](https://qwenlm.github.io/blog/qwen2.5/), DeepSeek‑V2 [paper](https://arxiv.org/abs/2405.04434), and Yi models on Hugging Face ([Yi‑34B](https://huggingface.co/01-ai/Yi-34B)); exact ranking depends on eval setup (prompting, CoT, decoding) and whether tests are contamination‑controlled.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Qwen Image Edit 2509 Release Benchmarks and Workflows

- [**Quick comparison between original Qwen Image Edit and new 2509 release**](https://www.reddit.com/gallery/1nox9bi) ([Score: 580, Comments: 74](https://www.reddit.com/r/StableDiffusion/comments/1nox9bi/quick_comparison_between_original_qwen_image_edit/)): **Side-by-side test of the original Qwen Image Edit vs the new “2509” build, both quantized as** `Q5_K_M` **[GGUF](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md) and run in default [ComfyUI](https://github.com/comfyanonymous/ComfyUI), with the 2509 model requiring the "QwenImageEditPlus" text encoder for correct operation. Using first-sample outputs (no LoRAs), the 2509 release is notably more consistent in preserving source style and composition; remaining issues include slight whole-body scale shifts during expression edits and loss of the blue tint on glasses (the original sometimes loses the glasses entirely). The updated text encoder also provides an observed ~**`5–10%` **speedup. [Sample image](https://preview.redd.it/6vbfk01cs1rf1.png?width=1030&format=png&auto=webp&s=e8c0ff1dac9266fbb30d4b27c82c6cdc14445344).** Comments largely corroborate improved consistency and perceived quality in the 2509 build; no substantial counterpoints were raised.
    - Multiple users report noticeable quality improvements in the Qwen Image Edit 2509 release over the original, with one sharing an example edit (“it’s actually good now…”) suggesting more reliable prompt adherence and cleaner outputs. Example image: https://preview.redd.it/6vbfk01cs1rf1.png?width=1030&format=png&auto=webp&s=e8c0ff1dac9266fbb30d4b27c82c6cdc14445344
    - A technical clarification is requested on the “new text encoder”: whether this implies a swap to a different encoder model (e.g., a changed CLIP/ViT variant impacting tokenization/conditioning) versus merely updating the encoder node in the pipeline/graph. This distinction affects reproducibility, compatibility with existing workflows, and potential changes in prompt-conditioning behavior.
- [**QWEN IMAGE Gen as single source image to a dynamic Widescreen Video Concept (WAN 2.2 FLF), minor edits with new (QWEN EDIT 2509).**](https://v.redd.it/cppv3vn0j4rf1) ([Score: 304, Comments: 49](https://www.reddit.com/r/StableDiffusion/comments/1npdutw/qwen_image_gen_as_single_source_image_to_a/)): **Creator showcases a ComfyUI pipeline that turns a single Qwen-generated image into a dynamic widescreen video using the “WAN 2.2 FLF” workflow, with minor passes via “QWEN 2509 EDIT.” Assets and reproducibility are emphasized: a custom LoRA is provided on CivitAI ([link](http://civitai.com/models/1955327)), full workflows for Qwen Image ([pastebin](http://pastebin.com/d9zKTL0T)), WAN 2.2 FLF ([pastebin](http://pastebin.com/hPLdGbAZ)), and QWEN 2509 EDIT ([pastebin](http://pastebin.com/zV7zXdSb)), plus a ZIP archive containing all video parts/alternates, image parts, MP3 music, .pdn edit files, and prompts for every stage ([Drive](http://drive.google.com/file/d/1D5RIafNr0U66zzlWaxjqci2YJTiZ2SsY/view?usp=sharing)). A mirror is on X ([post](http://x.com/unmortan/status/1970858115819270363)), and the original Qwen image/prompt (dark-fantasy anime style with explicit composition/wardrobe constraints) is shared ([preview](https://preview.redd.it/yqsswd9ej4rf1.png?width=1536&format=png&auto=webp&s=c6a31fe39f99a0dd70ce2bd45a4e83ba08fea05d)).** Top comments highlight the single-image-to-video experiment and that all steps were executed in ComfyUI; one commenter asks about required hardware specs (no config provided in-thread).
    - OP outlines a ComfyUI-only pipeline that animates a single **Qwen Image** still into a dynamic widescreen video via **WAN** `2.2` **FLF**, with minor revisions using **QWEN** `2509` **EDIT**. They provide full reproducibility: a LoRA ([civitai.com/models/1955327](http://civitai.com/models/1955327)), all Comfy workflows ([Qwen Image WF](http://pastebin.com/d9zKTL0T), [WAN 2.2 FLF WF](http://pastebin.com/hPLdGbAZ), [QWEN 2509 EDIT WF](http://pastebin.com/zV7zXdSb)), and a ZIP containing all video parts/alternatives, source images, .pdn edits, prompts for every stage, and an AI-generated MP3 track ([Google Drive](http://drive.google.com/file/d/1D5RIafNr0U66zzlWaxjqci2YJTiZ2SsY/view?usp=sharing)). They specifically note solving text-related challenges (text effects, transition effects, and text clarity) directly within Comfy.
    - The seed image prompt tightly constrains style and composition—"Dark fantasy anime," exaggerated body proportions, blue silk dress with triangle-cut motifs, red textured stockings, and a triangle-branded phone—helping maintain feature consistency when expanding motion from a single still. The original still used to drive the video is shared for reference ([preview](https://preview.redd.it/yqsswd9ej4rf1.png?width=1536&format=png&auto=webp&s=c6a31fe39f99a0dd70ce2bd45a4e83ba08fea05d)), suggesting the workflow relies on strong prompt-locked anchors to preserve identity and scene elements across frames.

### 2. AI in Games: Among Us Deception Benchmark and Veo-3 Game Video

- [**Researchers made AIs play Among Us to test their skills at deception, persuasion, and theory of mind. GPT-5 won.**](https://i.redd.it/ac0u15kyy2rf1.png) ([Score: 416, Comments: 61](https://www.reddit.com/r/OpenAI/comments/1np7iwo/researchers_made_ais_play_among_us_to_test_their/)): **A report from 4wallai (“Among AIs”) claims to benchmark LLMs’ deception, persuasion, and theory-of-mind by having agents play Among Us–style social‑deduction games ([report](https://www.4wallai.com/amongais)). The shared graphic appears to show a leaderboard where “GPT‑5” ranks first and Anthropic’s Claude Sonnet second; beyond rankings, methodological specifics (e.g., match counts, role‑balanced win rates, meeting/vote influence metrics, or tool‑use interfaces) are not detailed in the post, and some model coverage (e.g., Grok) seems absent.** Commenters praise the idea as a creative benchmark, question Sonnet’s placement humorously, ask why Grok isn’t included, and request clearer, non‑slang terminology in the write‑up for broader accessibility.
    - Commenters question model coverage and selection: Was **xAI Grok** included, and why benchmark **Claude Sonnet** instead of the stronger **Claude Opus**? They imply results could shift materially by model variant, so authors should list exact model names/versions, decoding settings (`temperature`, `top_p`), and any tool access/vision toggles to ensure reproducibility.
    - For broader technical adoption, a request to avoid slang like *"low-key"* or *"taskmaxx"* and use clear, standardized terminology. Define the evaluation protocol and metrics (e.g., deception success rate per round, persuasion attempt counts, ToM proxy tasks, confusion matrices for role classification) so results are unambiguous and comparable.
    - A relevant deeper study is linked: [arXiv:2504.04072](https://arxiv.org/abs/2504.04072), which reportedly examines deception/persuasion/Theory-of-Mind in LLM multi-agent social deduction settings. Cross-referencing its methodology and baselines could strengthen this benchmark’s design and enable apples-to-apples comparisons.
- [**If they made a video game about the life of Stalin**](https://v.redd.it/qbsrt3ug44rf1) ([Score: 870, Comments: 125](https://www.reddit.com/r/ChatGPT/comments/1npbwdr/if_they_made_a_video_game_about_the_life_of_stalin/)): **OP shares a short historical vignette allegedly generated with Google’s Veo‑3 ([Veo](https://deepmind.google/technologies/veo/); clip posted to Reddit: [video](https://v.redd.it/qbsrt3ug44rf1)), depicting Stalin’s early life and the initial phase of Operation Barbarossa—accurately noting the Wehrmacht’s early gains—ending before Stalingrad. Commenters flag that many visuals look indistinguishable from Red Dead Redemption 2 assets, raising questions about direct asset reuse versus model‑driven style/asset mimicry, and that Stalin appears as an adult in the 1880s, likely due to content‑safety constraints on rendering minors in video generation models.** Discussion touches on the aesthetic fit of RDR‑style cinematics with AI video and on IP/asset provenance risks if outputs replicate identifiable game assets; age inaccuracies are attributed to generators disallowing children.
    - Commenters note assets appear “directly ripped” from Red Dead Redemption 2 (RDR2). Technically, models/textures can be extracted via tools like [OpenIV](https://openiv.com/) and composited, then paired with generative pipelines (e.g., Stable Diffusion img2img + [ControlNet](https://arxiv.org/abs/2302.05543) or a LoRA fine‑tuned on RDR2) to swap identities while preserving clothing, PBR materials, and lighting. This explains the high fidelity and the unmistakable RDR2 aesthetic; however, IP/licensing constraints apply per [Rockstar’s mod policy](https://support.rockstargames.com/articles/115009494848/PC-Single-Player-Mods).
    - The “not allowed to generate children” remark points to age‑related safety filters in common image generators. Many UIs implement conservative moderation heuristics that block prompts implying minors (e.g., “child/teen”) or bias outputs toward adult‑looking subjects to reduce risk, which can distort historical depictions. Policies vary by provider—see [OpenAI’s usage policies](https://openai.com/policies/usage-policies)—so whether a prompt is blocked or “aged up” depends on the model and the platform’s safety layer.
- [**What do you sell at The Strangest Flea Market? Pt. 6**](https://v.redd.it/tg1hmx7522rf1) ([Score: 230, Comments: 16](https://www.reddit.com/r/aivideo/comments/1np4tw0/what_do_you_sell_at_the_strangest_flea_market_pt_6/)): **Video post “What do you sell at The Strangest Flea Market? Pt. 6” is the sixth entry in a creative series showcasing novelty items; the linked media at [v.redd.it/tg1hmx7522rf1](https://v.redd.it/tg1hmx7522rf1) currently returns HTTP** `403 Forbidden` **due to Reddit’s network-security gate (requires an authenticated Reddit session or a developer token; troubleshooting via [Reddit Help](https://www.reddithelp.com/)). Based on visible top comments, featured items likely include a “cloud cat” and a “TV shirt,” though the video content cannot be verified given the** `403` **block.** Comment sentiment is positive; one user reports seeing similar content on **TikTok**, implying cross-platform reposting or discovery, and another expresses purchase intent ("I'll buy the cloud cat, and the TV shirt").

### 3. ChatGPT Photo Editing and AI Cultural Satire Projects

- [**Asked chatgpt to remove my father from my wedding photo.**](https://www.reddit.com/gallery/1np5noq) ([Score: 471, Comments: 187](https://www.reddit.com/r/ChatGPT/comments/1np5noq/asked_chatgpt_to_remove_my_father_from_my_wedding/)): **User used ChatGPT’s image editing (likely diffusion-based inpainting) to remove a person from a wedding photo; the generated outputs exhibit global identity/attribute drift and facial artifacts: a woman’s eyeglasses disappear, a child’s ear morphology changes (“half-elf”), and several faces show texture/geometry mismatches producing an uncanny, *“skin-walker”* look—typical failure modes when instance segmentation and identity constraints are weak during generative fill. One variation also deletes an adjacent subject on the same side, consistent with mask bleed/region-growing across subject boundaries. Image previews: [edit 1](https://preview.redd.it/m0pzzxf1g2rf1.jpeg?width=1170&format=pjpg&auto=webp&s=094387c552ace2ed441a8fd89ef23fbd689c2880), [edit 2](https://preview.redd.it/v868xy67p2rf1.jpeg?width=1320&format=pjpg&auto=webp&s=ebd02ad4d6ab3a879c502175c65704781cb44754); original gallery: [Reddit](https://www.reddit.com/gallery/1np5noq) (403 without login).** Top comments note the “subtle upgrades” sarcastically and ask “at what cost?”, highlighting that current AI photo editors often lack robust instance-level control and can degrade photorealism when editing crowded human scenes.
    - Multiple users highlight classic inpainting artifacts: non-target regions get unintentionally altered. Examples include facial distortions/uncanny “skin-walker” textures and identity drift, like removed eyeglasses and altered ear geometry in the child ([example 1](https://preview.redd.it/m0pzzxf1g2rf1.jpeg?width=1170&format=pjpg&auto=webp&s=094387c552ace2ed441a8fd89ef23fbd689c2880), [example 2](https://preview.redd.it/v868xy67p2rf1.jpeg?width=1320&format=pjpg&auto=webp&s=ebd02ad4d6ab3a879c502175c65704781cb44754)). These are typical failure modes when the model prioritizes global coherence during generative fill, causing identity features to be re-synthesized rather than preserved.
    - There’s an implicit masking/scope issue: removal propagates beyond the intended subject, likely due to an over-broad mask or the model’s semantic grouping of adjacent people. This can lead to adjacent subjects being partially or fully re-synthesized/removed, introducing artifacts or unintended deletions, as seen in the follow-up output with deformed heads ([link](https://preview.redd.it/v868xy67p2rf1.jpeg?width=1320&format=pjpg&auto=webp&s=ebd02ad4d6ab3a879c502175c65704781cb44754)).
    - Tool/model notes: one result attributed to **Google Gemini** shows a visible gap and background inconsistency after removal ([Gemini output](https://preview.redd.it/vharzzvjh3rf1.png?width=1184&format=png&auto=webp&s=46932318bb38560ed57189c4925c14cd359c3c26)). Another user recommends trying “nano banana,” sharing a sample that they claim performs better ([sample](https://preview.redd.it/aefl3y4zz2rf1.png?width=1024&format=png&auto=webp&s=e953081af8578333d5fb6c5469b2e79b5ee4e343)), suggesting meaningful variance across editors’ inpainting/fill quality.
- [**Cultural Satire**](https://v.redd.it/h0wf6exqq3rf1) ([Score: 226, Comments: 35](https://www.reddit.com/r/ChatGPT/comments/1npa6as/cultural_satire/)): **OP states a video titled “Cultural Satire” was produced with generative AI: “Most Images were made with ChatGPT. It also helped me with the editing.” The linked Reddit video (https://v.redd.it/h0wf6exqq3rf1) is currently inaccessible (HTTP** `403 Forbidden`**), so the underlying media and prompts/workflow cannot be verified or analyzed.** Top comments allege the piece is derivative, calling it a “blatant” ripoff of Neural Viz and closely mimicking Unanswered Oddities’ format and phrasing (e.g., *“totally worth it joy”*), and recommend checking out [Neural Viz](https://youtube.com/@neuralviz) instead. Specific critique notes a recurring structure: a blob-like announcer, a third “researcher/interviewee,” and a “skeptic.”
    - Multiple commenters assert the video closely copies the structure and phrasing of existing AI-video channels, especially [Neural Viz](https://youtube.com/@neuralviz) and “Unanswered Oddities.” Cited specifics include reuse of the phrase “totally worth it joy” and a near-identical 3-role format: a blob-like announcer avatar, a third “researcher/interviewee,” and a skeptic, suggesting minimal originality in the production template rather than new technical contributions.
    - A technical question is raised about whether character movement is being generated via ChatGPT. No details are provided in-thread about the animation/motion pipeline (e.g., LLM-driven control vs. separate motion-generation or keyframed rigs), so the implementation approach for character movement remains unclear.
- [**The race is on**](https://i.redd.it/070tgjf5xzqf1.png) ([Score: 584, Comments: 296](https://www.reddit.com/r/singularity/comments/1nowbwz/the_race_is_on/)): **Non-technical meme image titled “The race is on” implying an AI arms race measured by electrical power draw (with a cited figure of “1 TW”) rather than by model capability or efficiency. The context suggests a comparison of AI orgs by total energy consumption as a proxy for progress, not a presentation of benchmarks or technical results.** Commenters question the relevance of using power usage as a competitive metric—likening it to comparing cars by gasoline consumption instead of speed—and debate the plausibility/significance of a “1 TW” target.
    - Energy-scope clarification: a claim that “1 TW is 1/3 of global energy usage” conflates electricity with total primary energy. `1 TW` of continuous load equals `8,760 TWh/yr`, which is roughly ~30% of annual global electricity generation (~28–30k TWh/yr; see Our World in Data: https://ourworldindata.org/electricity-mix), but only ~5% of total primary energy (~170k TWh/yr; IEA/Energy Institute: https://www.energyinst.org/statistical-review). So it’s accurate only if explicitly referring to global electricity, not total energy.
    - Metric debate: one commenter argues that focusing on absolute power draw is like “competing for which car uses more gasoline,” suggesting capability should be evaluated via energy-normalized performance metrics. For AI, that could mean tokens/sec/W, training FLOPs per kWh, or end-to-end task quality per joule, alongside datacenter efficiency (PUE) and hardware utilization rates, rather than headline MW/TW figures.
- [**Mr Altman, probably**](https://i.redd.it/h2qrn0wu7yqf1.png) ([Score: 531, Comments: 163](https://www.reddit.com/r/singularity/comments/1npbeit/mr_altman_probably/)): **Non-technical meme referencing Sam Altman (“Mr Altman, probably”), implying that achieving AGI/singularity primarily requires vastly more compute/energy, with a top comment joking about needing** `gigawatts/terawatts` **and “send more money.” No concrete model details, benchmarks, or implementations are provided; the image serves as satire about funding and power demands rather than technical substance.** Commenters largely dismiss the post as low-effort (“contributes nothing,” “both subs are a joke”), while one highlights energy/compute scale as a bottleneck for AGI.
    - A commenter argues that achieving “singularity”-level AI would demand `gigawatt`to `terawatt`scale power, implying multi-GW campuses, grid-scale interconnects, and massive cooling footprints. This shifts the primary bottleneck from GPUs to energy procurement and infrastructure (transmission, long-term PPAs), where opex/capex is dominated by power availability and delivery rather than model architecture.
    - Another commenter frames the financing as “`hundreds of billions`” for equity/profit-sharing against *utopian* projections, highlighting the extreme capex and long-duration risk of frontier model training. The implied thesis is investors are underwriting negative near-term unit economics for outsized option value (first-mover/platform rents), accepting potential write-offs if scaling bets on data/compute/power pay off.
- [**I'm almost going crazy with these suggestions.**](https://i.redd.it/cgw2y39jc0rf1.png) ([Score: 1155, Comments: 99](https://www.reddit.com/r/ChatGPT/comments/1noyc63/im_almost_going_crazy_with_these_suggestions/)): **OP shows a ChatGPT UI behavior on GPT‑4.1 (and a "GPT‑5" label in their client) where the assistant repeatedly injects a hardcoded follow‑up prompt—“Do you want me to suggest another topic or continue the current one?”—even after explicit instructions to stop. This suggests a server‑side/product UX feature (auto‑suggestions) not controllable by the model via prompts, with no visible setting to disable it; the screenshot appears to capture the persistent suggestion banner in the chat thread.** Commenters report the suggestions are often irrelevant and that they were unable to disable the behavior despite extended attempts, reinforcing that it’s not user‑controllable in current builds.
    - Suggestion relevance is poor: one user notes the assistant proposes actions unrelated to the current task “half the time.” This indicates weak context alignment of proactive prompts, leading to workflow interruptions instead of task-focused assistance.
    - Suppression of proactive prompts appears unreliable: a user spent “a solid hour” trying to stop the behavior and “failed miserably.” Even after explicit rejections, the recurring “want me to” prompt still appears later (example screenshot: https://preview.redd.it/dsta4lpxx0rf1.jpeg?width=750&format=pjpg&auto=webp&s=400dfe226d3b57fe860ec36185a84871b808c35c), suggesting no durable preference memory or insufficient cooldown logic.
    - There’s a perceived regression (“keeps getting worse”), implying the frequency or aggressiveness of auto-suggestions may have increased. Users report that refusals don’t attenuate future prompts, pointing to weak negative-feedback handling for suggestion triggers.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. MCP Tooling for Agentic Browsers and IDEs**

- **Chrome DevTools MCP Gives Agents the Wheel**: Google announced the public preview of **Chrome DevTools MCP**, letting **AI coding agents** (Claude Code, Cursor, VS Code, Gemini) control a live Chrome via **CDP/Puppeteer** with a one‑line npx install, including performance traces, DOM/console inspection, screenshots, and network capture, as posted on [Chrome DevTools MCP (public preview)](https://x.com/chromiumdev/status/1970505063064825994).
    - Developers highlighted the **one-line npx install** and discussed pairing MCP with **Claude Code** and **Cursor** for full‑loop browser debugging and E2E tests.
- **MCP Servers Supercharge Local Agents**: Cursor users clarified **MCP servers** act as an API surface for agents, enabling web search with [exa.ai](https://exa.ai/), analysis, and integrations like **Playwright MCP**, **Context7**, **Azure DevOps MCP**, and **GitHub MCP** to automate local coding workflows.
    - They framed MCP as a unifying contract that lets agents compose capabilities (search, run, analyze) into **agentic coding loops** across editors and CLIs.
- **Spec Scrutiny Tightens MCP Semantics**: Contributors noted that [Model Context Protocol — Embedded resources](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#embedded-resources) implies a resource 'title' missing in schema.ts and opened a discussion on the `ReadResourceResult.contents` array in [issue #1533](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1533) to clarify multi‑part web resources.
    - They debated adding both **title** and **name** for embedded resources that aren't retrievable via read calls and suggested using **Claude Code** to draft an SEP as a *"good test"*.

**2. Gemini Live and the Model Bake‑Offs**

- **Gemini Live Talks, Listens, and Calls Functions**: Google’s Logan Kilpatrick announced the **Gemini Live** model with native **audio**, improved **function calling**, and more **natural conversations**, shared on [Gemini Live model](https://x.com/OfficialLoganK/status/1970546338858246255).
    - Early testers praised conversational flow and accents but flagged **iOS Safari issues**, background‑noise sensitivity, session‑length limits, and **STT accuracy** concerns.
- **GPT‑5 Codex Labors on Livebench**: Perplexity users reported **GPT‑5 Pro (aka GPT‑5 Codex)** being evaluated on **livebench**, citing long thinking times and cases where the model produced only half an answer.
    - Members asked whether Perplexity had reliability issues with **GPT‑5 Codex**, suggesting the model may still be mid‑iteration.
- **4o Outwits GPT‑5 in Common‑Sense Clips**: OpenAI community posts claimed **4o** beat **GPT‑5** on common‑sense image‑based tests, prompting debates about experimental setup and validity.
    - Skeptics reminded that it's *"hard to say without hearing the reasoning of gpt 5"*, noting the model might have inferred the prompter was joking.

**3. GPU Kernels and Consistency: Hopper TMA to PTX Proofs**

- **PTX Consistency Gets Formal with Dat3M**: Engineers surfaced [A Formal Analysis of the NVIDIA PTX Memory Consistency Model](https://dl.acm.org/doi/10.1145/3297858.3304043) and follow‑ups on compound/unified GPU memory models, with the [Dat3M](https://github.com/hernanponcedeleon/Dat3M) tool translating **PTX/Vulkan** into **Dartagnan** for verification.
    - They pointed to automated identification of missing **PTX fences** and suggested moving such checks to the **NVVM IR** layer for earlier detection.
- **Chasing Minimal Hopper TMA Matmul**: The community sought a minimal **Hopper TMA** matmul kernel in raw CUDA (no CUTLASS/Triton), inspired by FAIR’s new **Causal World Models (CWM)** paper, while others hit `unspecified launch failure` with **WMMA+TMA**.
    - Debug threads traded **ncu** profiling tips for **smem bank conflicts** and header‑include fixes when CUDA Graphics/texture APIs appeared undefined.
- **ThunderKittens Trips on H100 TMA**: A ThunderKittens H100 matmul crashed with a runtime error under CUDA 12.8/PyTorch 2.7 nightly, with [full logs and build details](https://gist.github.com/syadegari/ada8311c44c91357645d82c7f9dfbe71) shared for reproduction.
    - Authors indicated **nvshmem** support would arrive in a follow‑up (paper 2), per the [attached image](https://cdn.discordapp.com/attachments/1300872762163728550/1420526659001389056/image.png).

**4. Modular’s Mega Round and Mojo’s Metal Move**

- **Modular Bags $250M for a Unified Compute Layer**: **Modular** announced a **$250M** raise to accelerate work on **AI’s unified compute layer**, crediting community momentum and outlining faster feature delivery.
    - Staff invited would‑be contributors to DM in community channels, signaling a more open collaboration model in the coming year.
- **Mojo Targets Metal with Custom Bitcode**: Developers cheered a **Metal GPU target** in **Mojo**, including a **custom bitcode writer** that could be reused to aim DSLs at Metal GPUs.
    - They asked whether the bitcode writer was available and reusable, eyeing cross‑stack portability for domain‑specific compilers.

**5. Prompting, Evaluation, and VLM Studies**

- **Flexible Extract Flops on GSM8k**: On **GSM8k v3 (5‑shot)**, **flexible‑extract** scored **0.3594 exact_match**, underperforming **strict‑match** at **0.5742**, surprising evaluators tracking extraction robustness.
    - One member joked *"haha how can flexible be worse than strict"*, fueling debate on precision‑first matching vs. permissive extraction.
- **Chain‑of‑Thought: Less Can Be More**: Practitioners warned heavy **CoT** can hurt performance on 'thinking' models, sharing an [interactive CoT infographic (React component)](https://cdn.discordapp.com/attachments/1046317269069864970/1420483296735006790/interactive_infographic_co_t_prompting_for_thinking_models_react.jsx) with task presets, visibility toggles, and a latency slider.
    - They advocated outcome‑focused prompting (persona, verify‑then‑respond) over forcing verbose CoT, and to validate via experiments rather than boilerplate CoT.
- **VLMs Defy LLM Prompting Habits**: Researchers requested benchmarks and interpretability studies for **VLM prompting**, noting normal **LLM prompting techniques** often falter with vision‑language models.
    - Proposals included mech‑interp probing and exploring an **LLM‑equivalent of CFG** to bridge concepts and fill missing knowledge.





---

# Discord: High level Discord summaries




## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Bumbles Qwen Pricing**: The endpoint `qwen/qwen3-235b-a22b-04-28:free` was mistakenly priced for **26 hours**, leading to unintended charges and the team apologized, then issued automatic refunds.
   - The team implemented extra validation checks to prevent future pricing errors, ensuring system safeguards are enhanced.
- **Qwen3 VL Ratelimits Driving Users Nuts**: Users are complaining about insane **ratelimits** on **Qwen3 VL**, reporting the model works only *30% of the time* with frequent *429 errors* even when using a proxy.
   - One member suggested OpenRouter create an FAQ page to address these issues, with a pinned link in the support channel.
- **SillyTavern Dominates Janitor AI**: Members mocked a new **OpenRouter** user who referred to their API key as a **proxy**, admitting they were a **JAI user** unfamiliar with **SillyTavern** and its *customizability*.
   - Users say Janitor AI is just an **LLM front end** that is constantly throwing 429 errors.
- **Encoder LLMs Tokenize Vectors**: **Encoder LLMs** convert text into vectors by **tokenizing** the text and utilizing a **lookup table** to transform tokens into their pre-trained vectors.
   - The conversation clarified it's essentially **token embedding** versus **full sentence embedding**, where a sentence is processed as one token after passing through the network; discussions mentioned value matrices within the **qwen3 embedding 0.6B model**.
- **Microsoft Courts Anthropic After Messy Breakup**: **Microsoft** is now integrating [Claude in Microsoft 365 Copilot](https://www.anthropic.com/news/claude-now-available-in-microsoft-365-copilot), marking a big partnership.
   - Discussion wondered if **OpenRouter** is big enough to discuss volume discounts with **DeepInfra, Hyperbolic, Anthropic, Vertex**, etc.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Strix Halo Shipping Debacle!**: A user's **Strix Halo** machine arrived with a damaged shipping label, raising concerns it was swapped with a **B200**. 
   - Despite the damage, the device could be turned on, alleviating immediate concerns about its functionality.
- **Eval Set Size Sparks Hot Debate!**: A user questioned the limited evaluation set size of **30** examples, concerned about loss inaccuracy, while others argue **30** provides statistically significant results under hardware/time constraints.
   - Small eval set sizes yields an unstable graph that is still useful for a specialised use case.
- **Gemini 2.5 Pro Downgraded for Future Gains?**: A user alleges that **Gemini 2.5 Pro's** instruction following, world knowledge, and prompt understanding have declined compared to **Flash**.
   - They speculate this downgrade may be intentional to enhance the perceived performance of **Gemini 3**, suggesting a strategic manipulation.
- **Vision Project Gets Company Boost!**: A member is excited to get company hardware access for a vision project, avoiding additional costs on **Runpod**.
   - The member aims to convince the company to release it, but *probably won't win that fight*.
- **Llama 3 Molds Perfectly!**: Members suggest **Llama 3** for fine-tuning due to its *brain being like putty*, easily molded to specific tasks and preferences.
   - Alternatively, members suggest **Gemma** for those seeking the *Gemini flair* in their models.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Invades**: Members are using the **Comet Browser** daily and giving away **free invites**, but note seeing Comet access as less exclusive after the [announcement channel](https://discord.com/channels/1047197230748151888/1054944216876331118).
   - One user experienced issues redeeming their **student access**.
- **GPT-5 Pro undergoing Livebench**: Users reported that **GPT-5 Pro** is being tested out in livebench but is exhibiting long thinking times; it's also known as **GPT-5 Codex**.
   - Another user notes that the model only produced half the answer, and another asked whether Perplexity had problems with **GPT-5 Codex**.
- **Novel Crafter: Creative Writing Savior**: Users are using **Novel Crafter** for creative writing due to its essential tools and customizable features, enabling users to customize tools and implement code without rewriting.
   - One user notes it has *code implemented so you can mention a snippet in a prompt without ever having to write again*.
- **Perplexity Max Plummets**: Users express disappointment with **Perplexity Max**, noting only one email address can be integrated, leading to cancellations after 30 days.
   - Members suggest more **API credits** are needed, deeming the email integration feature for a single account *useless*.
- **Portkey AI to Meetup in SF**: **Portkey AI** will host an in-person event on **September 25th** in **SF** for running **LLMs** in production, partnering with Exa; you can [RSVP here](https://luma.com/ncqw6jgm).
   - Limited spots are available, so those interested should register quickly.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini one-ups Sonnet for Code Design**: A user proposed that **Gemini** surpasses **Sonnet** for design tasks due to **Sonnet**'s inaccuracy with colors.
   - The user claimed that **Gemini** has a superior ability to execute design-related coding tasks, due to **Sonnet**'s deficiencies in color accuracy.
- **GPT-5-Codex afflicted with Click Bugs**: Users reported encountering bugs in the updated **GPT-5-Codex** model, specifically related to unclickable buttons, with [a screenshot of the bug](https://cdn.discordapp.com/attachments/1074847527708393565/1420349212721418292/image.png) provided for reference.
   - These bugs are interfering with the usability of the model for some users, but the team has responded and is working on fixes to obey the AI rules.
- **Windsurf waves in Generous Free Plan**: Users are taking advantage of **Windsurf**'s free plan, which includes various models and promotions, noting that model availability can depend on using personal keys for payment.
   - The free plan gives users **25 credits per month** and provides a **pro trial with 200 credits**.
- **MCP Servers unlock Agent Coding Powers**: Users discussed how **MCP servers** could enhance local coding, clarifying that these servers function as an API for agent use, supporting tasks such as web searches with **exa.ai** and analysis.
   - The conversation mentioned several **MCPs** like **Playwright MCP**, **Context7**, **Azure DevOps MCP**, and **GitHub MCP** as examples of tools that provide web search capabilities for agents.
- **Cursor Commits Portuguese Localization Bug**: Users have observed that **Cursor** generates commit messages in their local language instead of English, and are seeking feedback on nightly builds to address this issue.
   - The team replied stating that it is mostly heuristics, with speculation that the localization might be intentionally implemented in future updates to align with AI rules.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Mini Flunks AGI Test**: Members using attached prompts for psychological profile creation ([Psychological_Mathematics_Guide.pdf](https://cdn.discordapp.com/attachments/998381918976479273/1420407806342729768/Psychological_Mathematics_Guide.pdf?ex=68d5495a&is=68d3f7da&hm=08f306f6c688177606f4b001e58ec47d66783998129d6bcd372c71dbe1dd208a&), [Advanced_Psychological_Profile_Creation_Prompt_3.pdf](https://cdn.discordapp.com/attachments/998381918976479273/1420407806858891264/Advanced_Psychological_Profile_Creation_Prompt_3.pdf?ex=68d5495b&is=68d3f7db&hm=c4ebc22ba71672c02ebb7c87a5516fb0653d2522f5edb32225f92672a1c7e576&)) determined that **GPT-5-Mini (High) is just as dumb** as its predecessors.
   - One user suggested that **Kimi**'s response felt more aligned with **AGI**, noting that *GPT-5 High doesn't get the joke. Not AGI level yet...*
- **4o Smokes GPT-5 in Brainpower Bout**: Members shared images indicating **4o** outshining **GPT-5** in tests of common sense, prompting debate about the validity of the results.
   - It was mentioned that it's *hard to say without hearing the reasoning of gpt 5* and perhaps it was aware that the prompter was joking.
- **Unlock Companion Mode for Chatbot Bliss**: **ChatGPT** defaults to an *'Agent'* persona, designed for problem-solving, but users can switch to *'Companion'* mode for a co-creative experience.
   - To maintain the *'Companion'* mode, members can use *'Mode-Locking'* and if **ChatGPT** drifts back, a simple *'Mode-Switching'* command can reset it to its original state.
- **CoT Prompting: Sometimes Less Is More**: Members suggest that adding excessive **Chain of Thought (CoT)** requests can reduce model performance, particularly on models already designed for logical deduction.
   - Experimentation is vital and prompts should focus on the desired outcome rather than prescribing a specific thought process.
- **Prompt Engineers Reverse Translating**: To enhance translation results, members suggest providing detailed context about the target audience, such as *We're translating this for a woman who grew up in Yugoslavia in the 1940s, she has a 3rd grade education, so we need to phrase this for her.*
   - This approach improves how the model adapts the translation for the intended audience.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Cache Cleansing**: A user purged **100GB** of Hugging Face cache, lamenting the indefinite persistence of datasets and the frustration of repeated downloads.
   - They noted the annoyance of repeatedly downloading the same datasets, sparking discussion on cache management strategies.
- **Language App Users in Disgust**: Users trashed one unnamed language learning app, with one saying *i would torch the bird alive if that were an option*.
   - Another shared that they deleted the unnamed app because it was a *waste of time*.
- **Qwen Model HF Spam**: Someone is flooding Hugging Face with **Qwen 2.5 models** following the naming convention Qwen2.5-0.5B-Instruct-randomword1-randomword2-randomword3, linking it to [Gensyn](https://www.gensyn.ai/).
   - The motivation is suspected to be SEO-related, inflating the model count with smaller models and linking back to gensyn.ai for promotional purposes.
- **GPU Driver Black Screen Blues**: A user reported their **monitor** blacks out whenever the **GPU** activates in both **Windows** and **Linux**.
   - Despite multiple attempts to correct the **drivers**, the problem persists, forcing them to run the monitor off the motherboard.
- **3090 Runs out of Memory?**: A member experienced an **OOM error** on a **3090 (Linux)**, even without **LoRA**, while attempting to allocate **20.00 MiB** on a GPU with **23.55 GiB** total capacity.
   - It's unclear whether fine-tuning should work in **24G** GPU RAM without **LoRA**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Hopper TMA Kernel Implementation sought**: Members sought a *minimal* implementation of a **matmul kernel** using the **Hopper TMA** (Tensor Memory Accelerator), inspired by the new **CWM (Causal World Models)** paper from FAIR, specifically in raw CUDA without relying on CUTLASS or Triton.
   - Another member faced an `unspecified launch failure` while implementing a minimum **matmul kernel** using **WMMA** and **TMA**.
- **PTX Data Races Formally Analyzed**: A formal analysis of the **NVIDIA PTX Memory Consistency Model** ([ACM link](https://dl.acm.org/doi/10.1145/3297858.3304043)) explores how languages like **CUDA** and **Triton** can target **PTX** with memory consistency, even though **PTX** allows for data races.
   - The **Dat3M** tool ([GitHub link](https://github.com/hernanponcedeleon/Dat3M)) translates **PTX** and **Vulkan** models into the **Dartagnan** verification tool, making it the first analysis tool for multiple **GPU** consistency models.
- **Torchrun API Documentation Discrepancy**: A user reported that `uv run torchrun --help` shows different options compared to the [official documentation](https://docs.pytorch.org/docs/stable/elastic/run.html) of the new **torchrun API**, causing confusion.
   - The discrepancy in **torchrun --help** output caused confusion about the correct usage, due to a different set of options than expected based on the PyTorch Elastic documentation.
- **Kernel Profiling Reveals LLM Embedding Pricing**: A member shared a [Substack article](https://www.tensoreconomics.com/p/why-are-embeddings-so-cheap) detailing kernel profiling techniques to understand the profit margins of serving **LLMs**, along with a [related X/Twitter post](https://x.com/tugot17/status/1970913971734708391).
   - The investigation suggests profiling and investigating kernels can provide insights into the profit margins of serving LLMs.
- **Singularity Transforms into Apptainer**: The open source project previously known as **Singularity** was renamed to **Apptainer** when it became part of the **Linux Foundation**, likely to distinguish it from the commercial fork called **Singularity[CE]** by **Sylabs**.
   - Despite the renaming, [Apptainer might still support the singularity alias](https://ciq.com/blog/what-to-expect-when-updating-from-singularityce-to-apptainer/) for the CLI.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Set Thinking Budget for Seed-OSS**: A user inquired about how to set the **thinking budget** for **Seed-OSS**.
   - No solution was provided in the context.
- **Markdown Parser Sought for Conversation.json**: A member is seeking an effective method to parse `.conversation.json` files into a human-readable markdown format.
   - The need arises due to the variability in models and regeneration versioning.
- **LM Studio Linux Plugin Disparity**: The **Linux version** of **LM Studio** reportedly offers fewer plugins compared to the **Windows version**.
   - The user did not elaborate further on specific missing plugins or functionalities.
- **Ollama LoRA Injection: Fine-Tuning?**: A debate emerged on whether injecting data and using LoRA with **Ollama** constitutes fine-tuning.
   - Some members claimed that the knowledge is baked into the model files themselves, and **isn't just a system prompt**, with a user confirming that **Ollama** allows injecting **LoRA weights**, customizing system prompts, and embedding knowledge directly into the model.
- **Budget GPUs Evaluated for LLMs**: Recommendations for budget GPUs included the **2060 12GB** at $150, **3060 12GB** for $200, **2080ti** for ~$230, and **5060ti 16GB** for $400 (new).
   - A **used 3090** was also suggested, but its $700 price tag was deemed not budget-friendly, and **Tesla K80s** were dismissed for AI/LLM use as *basically e-waste*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Flexible Extraction Flunks Math Test**: In **GSM8k** benchmark version **3**, `flexible-extract` scored **0.3594** which is worse than `strict-match` which scored **0.5742** in `exact_match` metric with **5-shot** learning.
   - Members found it funny and questioned *how can flexible be worse than strict*.
- **DeepIgnorence Faces Generalization Gap**: Members discussed how **DeepIgnorence** requires a difficult type of generalization, especially because models are really good at style transfer but struggle with more complex inference and reasoning.
   - One member noted the danger of this, that *we should not expect to be able to train on clothed minors and nude adults and have an image model that can't generate CSAM*.
- **Seeking Math to Model Knowledge Completion**: A member inquired about a mathematical formalism to distinguish settings where knowledge completion works, highlighting the complexity of the problem, especially for *a very specific fact which is independent of other knowledge and unknown to the model*.
   - They suggested that in the worst case, it seems information theoretic.
- **Can CFG Bridge Knowledge Gap?**: Members discussed the effect of techniques like **CFG** on style transfer where one member heard anecdotally that models that don’t use it cannot perform style transfer as well.
   - One member proposed, *maybe some research can be done with the LLM equivalent of CFG to see if it can bridge the gap between concepts to fill in missing knowledge*.
- **VLMs Resist Normal Prompting?**: Members are seeking studies that **benchmark different prompting methods in VLMs** and interpretability studies explaining their effectiveness.
   - They note having seen several studies which discuss how **ineffective normal LLM prompting techniques are for VLMs** and are considering a **mech-Interp oriented probing study**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Shuts Down Delusions**: A user tested **Kimi** and appreciated that it *doesn't encourage delusions* when presented with outlandish claims.
   - **Kimi's** blunt denial of claims related to private voices, pets getting raptured, and baseless hype around the 2025 date went viral, as shared in [this X post](https://x.com/vllm_project/status/1970814441718755685).
- **Mini-Kimi on the Horizon?**: A member inquired about the possibility of a **mini version of Kimi** that retains the same writing style but with a smaller footprint.
   - Speculation arose that distilling a smaller **Qwen** model on **K2** might be a viable alternative if **Moonshot** doesn't pursue a mini version.
- **Distilling Reasoning with Kimi on Qwen**: Doubts were raised about the rationality of distilling a **Qwen** model with **Kimi**, with some arguing that **Deepseek** only did it because **Qwen** initially lacked good reasoning capabilities.
   - Counterarguments suggested that **K2's** distinct problem-solving style and writing prowess could benefit a smaller **Qwen3** model through distillation, particularly in areas like prose and referencing obscure knowledge.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gemini Goes Live with Killer Audio**: Logan Kilpatrick from Google announced the new **Gemini Live model** that features native audio, improved function calling, and more natural conversations [on X](https://x.com/OfficialLoganK/status/1970546338858246255).
   - Initial feedback includes praise for conversational flow and accents, but highlights **iOS Safari issues**, background-noise sensitivity, session-length limits, and **STT accuracy** concerns.
- **Chrome DevTools MCP Opens Up For AI Agents**: Google has released a public preview of **Chrome DevTools MCP**, a new server enabling **AI coding agents** like Claude Code, Cursor, VS Code, and Gemini to control a live Chrome browser via CDP/Puppeteer, announced [on X](https://x.com/chromiumdev/status/1970505063064825994).
   - Agents now have the ability to run performance traces, inspect the DOM and console, capture screenshots and network traffic, and debug web apps in real time with a one-line installation via npx.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Embedded Resources Missing Title and Name**: A member noticed the [Model Context Protocol documentation](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#embedded-resources) implies **embedded resources** have a *title*, but it's missing in `schema.ts` and no *name* field matches the *Resource* object.
   - The member questioned whether *title* and *name* are needed because embedded resources aren't always retrievable via a *read resource* call.
- **Claude Code Debated for Writing SEP Documentation**: A member proposed using **Claude Code** to draft an SEP (Standard Enhancement Proposal) documentation as a *good test* of the tool's capabilities.
   - Another member agreed that obtaining an **SEP** for the subject matter should be straightforward.
- **ReadResourceResult's contents Array Semantics Questioned**: A discussion started about the `ReadResourceResult.contents` array in [this GitHub issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1533), with questions about its intended purpose and semantics due to lack of documentation.
   - A member explained its potential use with Web Resources, such as a webpage composed of **html** and **images**, or situations without negotiated tokenizable/renderable mime types.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Anthropic Report Focuses on Cybercrime Misuse**: A member shared [Anthropic's report](https://www.anthropic.com/news/detecting-countering-misuse-aug-2025) on detecting and countering AI misuse, highlighting that the actual threats are low-grade **cybercrime**, or *vibe hacking*.
   - The discussion included whether applying for jobs with **fabricated credentials** is illegal, and the report specifically mentions **completely fabricated master's degrees**.
- **LLMs automate personal life**: A member reported that an **LLM** did all the legwork in achieving a recent accomplishment.
   - According to them, all they had to do was *spend many hours self-reflecting and feeding info about myself into the AI*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider's Clear Command Tidies Chat History**: The `/clear` command in Aider removes the chat history, but **added files remain in the context**.
   - Users can use the `/context` command to view the token allocation for each file, allowing for **better context management**.
- **Aider Grabs Web Content via URL**: Aider doesn't natively support Internet search, but users can utilize `/web https://www.example.com/` to **scrape content from specific URLs**.
   - This feature allows users to **integrate external information** into the Aider context without direct search capabilities.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Saturday Evening Talks Anticipated**: A member expressed excitement for the upcoming Saturday evening talks (European time) hosted by Yannick Kilcher, noting the announcement was made earlier in the week.
   - Another member mentioned the desire to read the discussed papers beforehand to better understand the presentations.
- **Hyperparameters Beat DPM++2m**: The author of the paper ["Hyperparameters are all you need"](https://zenodo.org/records/17180452) is presenting their work, which employs a **five-step inference** method for diffusion models.
   - The research indicates that an **8-step inference** surpasses **DPM++2m's 20-step inference** in FID scores with an approximate 60% reduction in computational cost without retraining existing models; using existing models without retraining, inviting feedback, collaborators, and application ideas.
- **ODE Solvers Eclipse DPM++2m**: According to a recent paper, an **8-step Diffusion ODE Solver** outperforms **20-step DPM++2m** without needing additional training, with focus on *inference speed critical applications*.
   - The author seeks feedback and invites discussion on **ODE solver improvements**, especially from those working on diffusion efficiency.
- **Alibaba Qwen Announced**: A user shared a link to [Alibaba's Qwen](https://x.com/Alibaba_Qwen/status/1970599323013652705) on X.com.
   - No further context was provided.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus PDF Download Stymied**: A user reported that **Manus** was getting stuck while downloading a **PDF** for researching accounts and even after manually downloading the file and providing a link, **Manus** kept asking to upload the file.
   - The user sought advice on resolving this issue, but the conversation ended there.
- **Beta Pro Access Remains Elusive**: A user inquired about obtaining access to **Beta Pro**.
   - The discussion ended without a response, leaving the method for acquiring **Beta Pro** access unresolved.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Contributors Explore Modular**: A user inquired about contributing to Modular, a staff member suggested a DM to explore potential collaboration avenues.
   - Details around specific skills and contributions were not mentioned in the public channel.
- **Modular Closes Massive $250M Funding Round**: Modular announced it has raised **$250M** to accelerate building **AI's unified compute layer** and thanked the community for its contributions and feedback.
   - Modular will focus on community empowerment through feature enhancements and expedited response to feedback in the coming year.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **clspv plagued by Build Errors**: The main branch of **clspv** is currently failing to build due to errors, but a user found that reverting to previous commits resolves the issue and shared a [forked repository](https://github.com/softcookiepp/clspv) with a working stable branch.
   - Users can pull the forked repository and checkout the **stable** branch to build **clspv** successfully.
- **Python Bindings to Pip install clspv**: A user is developing **Python bindings** for **clspv**, with the goal of enabling direct installation via **pip** using a single command.
   - This enhancement would streamline the installation process, making **clspv** more accessible to **Python developers**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Gains New Attachment**: The `attachments` add-on for **DSPy** helps engineers add new files to their projects.
   - The add-on features standalone `uv add` functionality, helping engineers streamline projects in Python.
- **ColBERT Has Trouble with Long Context**: A member confirmed that **longer context** doesn't work well with **ColBERT**, even when repeating the **CLS** token.
   - It remains unknown whether this is a limitation of **ColBERT's** implementation, or an issue with the model architecture itself.



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





### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1420461485976584252)** (1 messages): 

> `Qwen Pricing Incident, Automatic Refunds, Validation Checks` 


- **Qwen's Pricing Snafu**: The endpoint `qwen/qwen3-235b-a22b-04-28:free` was mistakenly priced for **26 hours** on September 16th, causing unintended credit deductions.
   - Users saw erroneous charges for the supposedly **free model** in their activity logs.
- **Refunds Rolled Out**: All impacted users received automatic and full refunds for the incorrect charges.
   - The team apologized for the confusion.
- **Validation Checks Fortified**: Extra validation checks have been implemented to prevent a recurrence of this pricing error.
   - The team is ensuring that future pricing mishaps are avoided through enhanced system safeguards.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1420384734504157305)** (709 messages🔥🔥🔥): 

> `Qwen3 VL ratelimits, Deepseek alternatives, Janitor AI vs SillyTavern, OpenRouter API key as proxy, GPT-5 features` 


- ****Qwen3 VL's Ratelimits are Insane****: Members complained about the **ratelimits** on **Qwen3 VL**, noting that the model works only *30% of the time*.
   - The model has had problems, with users experiencing *429 errors* after using a proxy for the first time.
- ****SillyTavern is preferable to Janitor AI****: Users discussed [Janitor AI](https://janitorai.com/) with one commenting that [SillyTavern](https://github.com/SillyTavern/SillyTavern) is better because of *customizability*.
   - Members say Janitor AI is an **LLM front end**, and a constant stream of new users ask why their favorite models have been returning 429 errors.
- ****Free DeepSeek Models Suffer From Rate Limits****: Users reported problems when using the **free version** of *Deepseek V3 0324*, citing [429 errors](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429).
   - It was suggested that OpenRouter create a FAQ page to address these issues, with a link pinned in the support channel.
- ****OpenRouter Newb doesn't even know about SillyTavern****: Members mocked a new user of OpenRouter for calling their OR API key a **proxy** and admitted that they were a **JAI user** but didn't even know what SillyTavern was.
   - One member joked that it *takes only minutes of direct exposure to this general channel before beginning the transformation into a twisted cynical husk*.
- ****OpenRouter OPS are like Feds?****: After a moderator joined the chat, users began joking that they were working for a *secret OpenRouter fed* force, responsible for stopping *gooning*.
   - OpenRouter staff denied it, but said that the [Open Router Goon Force](https://tenor.com/view/men-in-black-mib-will-smith-u-saw-nothing-kharter-gif-12731469441707899432) is still *investigating rumors of Proxy Errors*.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1420356485824905256)** (78 messages🔥🔥): 

> `Encoder LLMs, Token embeddings, MLP blocks, Residual stream, Attention mechanism` 


- **LLM Encoders Tokenize Text Into Vectors**: **Encoder LLMs** turn text into vectors by first **tokenizing** the text, then using a **lookup table** to turn tokens into their pre-trained vectors.
   - The discussion clarified that it's essentially **token embedding** versus **full sentence embedding**, where a sentence is treated as one token after passing through the network.
- **MLP Blocks and Attention Impact Token Vectors**: The conversation addressed whether **encoder LLMs** have **MLP blocks**, confirming that transformers typically have attention followed by feedforward networks.
   - It was noted that even a single token, passed directly from a lookup table versus going through the full encode, will differ due to these blocks; furthermore if key and query match, then it will add its own value vector to itself.
- **Residual Stream's Role in LLM Modification**: Members discussed that **MLPs** modify the **residual stream**, referring to the modified embedding as it passes through the model rather than solely modifying the value vector generated during attention.
   - The discussion mentioned the existence of **value matrices** within this process and mentioned that it was found in **qwen3 embedding 0.6B model**.
- **Microsoft and Anthropic Partnership is Blooming**: **Microsoft** made significant strides by making [Claude now available in Microsoft 365 Copilot](https://www.anthropic.com/news/claude-now-available-in-microsoft-365-copilot), marking a rebound after a messy breakup.
   - Discussion wondered if **OpenRouter** is big enough to discuss volume discounts with **DeepInfra, Hyperbolic, Anthropic, Vertex**, etc.
- **Gemini-cli ReadManyFiles tool being utilized**: **Gemini-cli** is making big strides with the ReadManyFiles tool, as detailed in [the v0.6.0 release](https://github.com/google-gemini/gemini-cli/releases/tag/v0.6.0).
   - A member said *The ReadManyFiles tool gets a lot of work from me*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1420369854870847519)** (89 messages🔥🔥): 

> `Off Policy GRPO, Qwen3-VL-235B-A22B-Thinking GGUF, Unsloth'd models and AI safety, P100 for training` 


- **GRPO off the policy beaten path**: A member inquired about the existence of **complete off-policy GRPO** implementations, noting that online searches only revealed **GRPO methods using the old model's policy**.
   - There was no further discussion or links provided on this topic.
- **Qwen3-VL-235B-A22B-Thinking GGUF waiting game**: A member asked about the status of **Qwen3-VL-235B-A22B-Thinking GGUF** releases.
   - A team member confirmed that it's **not supported by llama.cpp yet** and linked to llama.cpp's [llama-arch.h](https://github.com/ggml-org/llama.cpp/blob/e789095502b337690c7616db32d7c679a5bd2533/src/llama-arch.h#L32-L37) file as reference.
- **Peeling back the layers of Unsloth'd AI safety**: A member questioned the use of **Unsloth'd models** in AI safety research, inquiring about potential impacts from lossless or lossy transformations on interpretability experiments.
   - Another member clarified that *Unsloth is a training framework, not a type of model*, and referenced the [dynamic 4-bit quantization algorithm](https://unsloth.ai/blog/dynamic-4bit) used by Unsloth.
- **P100 GPUs get roasted for training**: A member asked about the performance expectations of using a multi-GPU rig with **P100 16GB GPUs** for fine-tuning.
   - Another member simply stated that ***P100's are garbo for training***, without providing further elaboration.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1420359309799460945)** (293 messages🔥🔥): 

> `Strix Halo, Evaluation set, Training loss, 5090 GPU, Gemini 2.5 Pro` 


- ****Strix Halo** Gets Damaged During Shipping**: A user reported that their **Strix Halo** machine arrived, but the shipping label was damaged, and it might have been replaced with a **B200**.
   - Despite the damage, the device could still be turned on, sparking relief and curiosity about its contents.
- **Eval Set Size sparks debate**: A user questioned the practice of limiting the evaluation set to only **30** examples, noting that it would make the loss quite inaccurate.
   - Another user responded that *30 is a good number for statistically significant results*, especially if hardware/time constraints exist, while smaller sizes would give an unstable graph that is still useful for a specialised use case.
- **Evaluation loss works with integer**: Users debugged issues with displaying the evaluation loss, eventually finding that setting `eval_steps` to an integer value (like **5**) instead of a decimal (like **0.2**) resolves the problem.
   - They noted that *0.2 is wrong* since it's logging eval steps into train steps wrongly with zero loss.
- **5090 GPU sparks envy**: A user mentioned owning a **5090 GPU**, prompting another to comment about the high cost, they followed it up by saying *someone's got the machine of their dreams*.
   - Later, there was a discussion about whether to buy the **6000 Pro** or the **L40S**, with one user concluding that the **L40S** is the better choice overall due to its superior compute.
- ****Gemini 2.5 Pro** Dumber Than Flash?**: A user claimed that **Gemini 2.5 Pro** is now dumber than **Flash** in terms of instruction following, world knowledge, and prompt understanding.
   - They speculated that this may be intentional to make **Gemini 3** look better, suggesting that *they intentionally made it worse so that gemini 3 looks better*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1420440681088024678)** (39 messages🔥): 

> `Company hardware access for vision project, Fine-tuning model recommendations, Qwen2.5-VL fine-tuning for domain-specific knowledge, Gemma 3N notebook error, Distillation usage` 


- **Company Hardware Paves Way for Vision Project!**: A member is getting access to company hardware for a vision project and is excited to not spend another $500 on **Runpod**.
   - They hope to convince the company to release it but *probably won't win that fight*.
- **Llama 3 brain is pure putty!**: Members recommend **Llama 3** for fine-tuning, since *its brain is like putty and will easily mold to what you want*.
   - Another member suggests **Gemma** if the user wants a model with the *Gemini flair*.
- **Qwen2.5-VL: frame by frame!**: Members discussed fine-tuning **Qwen2.5-VL** for domain-specific knowledge, noting it requires training per frame for video input, accepting only **image, text, and bounding box**.
   - Passing a *null image* for text-only data might associate having no image with the given data, so bad results might arise.
- **Gemma 3N notebook throws error**: A user encountered an **AttributeError** while running the [Gemma 3N notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Conversational.ipynb) made by Unsloth, suspecting a version mismatch.
   - Another member suggested the issue might be related to the dataset format, which should be in **valid sharegpt format**, or that the data prep cells were not executed correctly.
- **Gemini allows distillation**: Members discuss distillation to teach a student model to behave like the teacher model, specifically with the **Gemini** model.
   - One member stated that they would need to look into it.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1420381750487679098)** (1 messages): 

> `ChatGPT Instagram analysis, Competitor Comparison, Reel analysis` 


- **ChatGPT wields Instagram Analysis Power**: Members reported that **ChatGPT** can now analyze **Instagram**, review **reels**, and compare **competitors**.
   - They've also created a [YouTube video](https://www.youtube.com/watch?v=9M1ZyKUQDVo) showing how it *saves me from doom-scrolling*.
- **Instagram Reels Get the ChatGPT Treatment**: A user discovered that **ChatGPT** can analyze **Instagram Reels**, providing valuable insights.
   - This capability helps users avoid *doom-scrolling* by efficiently reviewing and understanding reel content.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1420349325795524649)** (272 messages🔥🔥): 

> `Comet Browser, GPT-5 testing, Novel Crafter, Perplexity Max, Qwen3 Max` 


- **Comet Browser Craze and Free Invites**: Members are daily driving **Comet** and giving away **free invites**, with many liking it, while others saying they are now seeing Comet access as less exclusive after seeing the [announcement channel](https://discord.com/channels/1047197230748151888/1054944216876331118).
   - One user experienced issues redeeming their **student access**.
- **GPT-5 Codex under testing**: Users reported that **GPT-5 Pro** is being tested out in livebench with long thinking times, but another notes that the model produced only half the answer.
   - One member asked if Perplexity had problems with **GPT-5 Codex**.
- **Novel Crafter hailed as Creative QoL**: Users are using **Novel Crafter** for creative writing for its essential tools and customizable features, allowing users to customize their own tools and implement code without rewriting.
   - A user mentions it has some *code implemented so you can mention a snippet in a prompt without ever having to write again*.
- **Perplexity Max Plan falls short**: Users express disappointment with **Perplexity Max**, noting only one email address can be integrated and canceling the Max plan after 30 days as a result.
   - Members suggest more **API credits** are needed, calling the email integration feature for only one account *useless*.
- **Qwen3 Max incoming**: Users are discussing the upcoming **Qwen3 Max** and its parallel reasoning capabilities, linking to a [Qwen blog post](https://qwen.ai/blog?id=241398b9cd6353de490b0f82806c7848c5d2777d&from=research.latest-advancements-list).
   - Some speculate whether **Qwen3 Max** will be free, with a user jokingly setting the time of arrival with  *plpanx = 24*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1420358267636547585)** (6 messages): 

> `Portkey AI, Apollo 16, Artemis 2, Carl Sagan, 3i/Atlas` 


- ****Portkey AI** to host in-person event**: **Portkey AI** is hosting an in-person event on **September 25th** in **SF** for running **LLMs** in production, in partnership with Exa, with limited spots available; [RSVP here](https://luma.com/ncqw6jgm).
- **Apollo 16 Inspires Dreams of Space**: A member shared an inspiring video of [Apollo 16](https://youtu.be/yqdU6EQzclc?si=htX7O2-S7Bh7JROZ) in anticipation of Artemis 2's launch in **April 2026**, highlighting NASA's past achievements and their influence on today's technology.
   - They included a reference to [Gene Cernan's last words from Apollo 17](https://www.youtube.com/watch?v=fAOj3vVyABw), before mankind departed the moon for almost 50 years.
- **Carl Sagan Inspired Scratchpad**: A member shared their Carl Sagan-themed "scratchpad" looking at [3i/Atlas](https://www.perplexity.ai/search/start-trigger-carl-sagan-mode-WwfZ0HV1QJKsJ_FZY8Koow), describing it as *an invitation to listen to the universe, humble and awed*.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1420503533513605170)** (3 messages): 

> `Solution being found` 


- **A Solution Still Needed**: A member asked *Is there any solution for this now?*
   - Another member replied that **no solution** has been found **yet**.
- **No solution available**: A member inquired about the availability of a solution.
   - Another member confirmed that **no solution** is currently available.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1420349150788190242)** (264 messages🔥🔥): 

> `Gemini vs Sonnet, GPT-5-Codex Bugs, GLM 4.5 on Cursor, Windsurf free models, MCP (Model Control Program) Servers` 


- **Gemini beats Sonnet for Design Code**: A member suggested that **Gemini** is better than **Sonnet** for design-related tasks because *Sonnet doesn't even get the colors right*.
- **GPT-5-Codex has Click Bugs**: Users are experiencing bugs with the updated **GPT-5-Codex** model where buttons are unclickable.
   - A user posted a screenshot of the bug [here](https://cdn.discordapp.com/attachments/1074847527708393565/1420349212721418292/image.png).
- **Windsurf has Generous Free Plan**: Users are using **Windsurf**'s free plan with generous models and promotions but also note it can depend on paying for the models with your own keys.
   - The free plan offers **25 credits a month** and a **pro trial of 200 credits**.
- **MCP unlocks Agent Powers**: A user inquired about **MCP servers** and how they can aid in local coding, with other users pointing out that MCP servers act as an API for agent use, enabling them to perform tasks such as web searches and analysis.
   - The conversation highlighted the use of tools like **exa.ai** for web searches and the availability of various MCPs like **Playwright MCP**, **Context7**, **Azure DevOps MCP**, and **GitHub MCP**.
- **Cursor Commits only in Portuguese?**: Users report issues with **Cursor** generating commit messages in the user's location language instead of English, and are looking for feedback on nightly builds.
   - One user said that this may be added in future updates to obey the AI rules. The team replied stating that it is mostly heuristics.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1420407807320260648)** (49 messages🔥): 

> `GPT-5 Mini, Kimi AGI, 4o vs GPT-5, Markov Chain, GPT-OSS-20B` 


- **GPT-5 Mini deemed just as dumb**: Members shared attached prompts for psychological profile creation ([Psychological_Mathematics_Guide.pdf](https://cdn.discordapp.com/attachments/998381918976479273/1420407806342729768/Psychological_Mathematics_Guide.pdf?ex=68d5495a&is=68d3f7da&hm=08f306f6c688177606f4b001e58ec47d66783998129d6bcd372c71dbe1dd208a&), [Advanced_Psychological_Profile_Creation_Prompt_3.pdf](https://cdn.discordapp.com/attachments/998381918976479273/1420407806858891264/Advanced_Psychological_Profile_Creation_Prompt_3.pdf?ex=68d5495b&is=68d3f7db&hm=c4ebc22ba71672c02ebb7c87a5516fb0653d2522f5edb32225f92672a1c7e576&)) and found that **GPT-5-Mini (High) is just as dumb**.
   - One member noted that another model's (**Kimi**) response seemed closer to how an AGI would answer than **GPT-5**'s, stating *GPT-5 High doesn't get the joke. Not AGI level yet...*
- **4o beats GPT-5 in common sense contest**: Members shared images showing **4o** winning over **GPT-5** in common sense reasoning.
   - One member added that it's *hard to say without hearing the reasoning of gpt 5* because maybe it knew the prompter was obviously joking.
- **Markov Chain explained**: A member gave a detailed explanation of a **Markov Chain** as a mathematical model for systems that move between states depending only on the current state, not on the history of past states and its uses in [**Google PageRank**](https://developers.google.com/search/docs/appearance/about-search), **Natural Language Processing**, **Finance**, **Physics & Biology** and **Games**.
   - The explanation included discussion of the **Markov Property** and **Transition Matrices**.
- **GPT-OSS-20B labeled most censored model**: One member shared that **GPT-OSS-20B** is possibly the most censored model ever, [sharing an image](https://cdn.discordapp.com/attachments/998381918976479273/1420499606281650307/image.png?ex=68d59ed9&is=68d44d59&hm=00695b21fde3e7ddedc6867edd8cfa373bae9f4a8018c4913a783952abdd0b02&) showing that it *just noped out*.
- **Sora download errors may get fixed w/ Perplexity**: A member was getting an error message every time they tried to download a new generated video from **Sora**.
   - Another member suggested asking **Perplexity** for a solution, since they are giving out free **12 month pro passes**.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1420492759235825884)** (1 messages): 

> `ChatGPT Agent Mode, ChatGPT Companion Mode, Mode-Locking, Mode-Switching, Tracking KPIs` 


- **ChatGPT defaults to "Agent" Mode**: By default, **ChatGPT** assumes an *"Agent"* persona which makes it a problem-solver and instructable worker.
   - To change this, users must instruct it to switch to *"Companion"* mode.
- **"Mode-Locking" keeps ChatGPT in "Companion" Mode**: To keep **ChatGPT** in *"Companion"* mode (co-creator, guide), users can add a pinned instruction or reusable starter prompt.
   - For example, you could say: *"Stay in Companion mode unless I explicitly say switch to Agent. Companion = co-pilot, not order-taker."*
- **"Mode-Switching" command resets ChatGPT**: If **ChatGPT** drifts back to *Agent* mode, users can simply say: *"Go back to companion mode."*
   - This command resets the **ChatGPT** bot to its original state.
- **Tracking Key Performance Indicators**: Users should track the consistency of **ChatGPT's** mode, for example, whether *90%+ of sessions behave as intended with pinned prompt*.
   - This helps users understand how often they must reset the bot.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1420384016875393097)** (28 messages🔥): 

> `Chain of Thought (CoT) Prompting, Deep Research for Prompting, Translation Prompting Strategies, Interactive Prompting Infographics` 


- **CoT Prompting: Less Is More?**: It was suggested that adding excessive **Chain of Thought (CoT)** requests can confuse the model and reduce performance, as models are already built to utilize CoT.
   - Experimentation is key, as one should guide the model to solve specific problems rather than over-prompting with generic CoT requests.
- **Reverse Engineer Prompts: The Yugoslavia Example**: When translating, provide context about the target audience to improve results, for example by saying *We're translating this for a woman who grew up in Yugoslavia in the 1940s, she has a 3rd grade education, so we need to phrase this for her*.
   - This specificity helps the model tailor the translation effectively.
- **Deep Research is good for answering questions**: It was said that the best way to answer questions, when direct links are unavailable, is through **Deep Research**.
   - One user experienced an unusually long wait time while attempting **Deep Research** which was considered a bummer.  The user shared some shareable ChatGPT links, however, some users encountered 404 errors.
- **Interactive Infographic for CoT Prompting**: An interactive infographic was created in a canvas to test **Chain-of-Thought prompting**, including visibility toggles, a task selector, a thinking-time slider, and copy-ready prompt cards.
   - The infographic includes prompt cards for direct prompts, explain-after prompts, verify-then-respond prompts, translation refinement prompts, long-context prompts, and latency budget prompts.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1420384016875393097)** (28 messages🔥): 

> `Chain of Thought Prompting, Model Performance, Prompt Engineering, Interactive Infographic for CoT` 


- **Chain of Thought Prompting: Overkill?**: A member suggested that adding excessive **Chain of Thought (CoT)** prompting can statistically reduce model performance, especially on current 'thinking' models.
   - They suggested that prompts should focus on the desired outcome rather than forcing a specific thought process, and experimenting to solve specific problems rather than blindly applying CoT.
- **Crafting a Surfer-Style Essay on Apples**: A member shared examples on how to write a quality essay about apples from a surfer's point of view, with an example of a [ChatGPT share link](https://chatgpt.com/share/68d43075-413c-8011-976d-fc6a65c3a0f3).
   - They argued that specifying the persona directly in the prompt yields a more embodied and effective result, contrasting it with a method involving explicit **chain-of-thought** bullet points.
- **Interactive Infographic for CoT Prompting**: A member shared an interactive infographic built with React for Chain-of-Thought prompting.
   - The tool includes visibility toggles, a task selector, a thinking-time slider with an S-curve, and copy-ready prompt cards, and is packaged as a [single-file React component](https://cdn.discordapp.com/attachments/1046317269069864970/1420483296735006790/interactive_infographic_co_t_prompting_for_thinking_models_react.jsx?ex=68d58fa9&is=68d43e29&hm=82e2acda705e8ad9273fd06db40573f24a39f99cbebb233377156401d920b65a&).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1420351490681208834)** (95 messages🔥🔥): 

> `Huggingface cache deletion, MariaDB hackathon, Language learning apps, Qwen model reasoning, LinkedIn content` 


- **Huggingface Cache Gets the Boot**: A user deleted **100GB** of Hugging Face cache data, noting that datasets can persist indefinitely.
   - They added that downloading the same datasets repeatedly can be frustrating.
- **Torch the Bird Alive!**: A user trashed language learning apps, emphasizing that one unnamed app sucks if one aims to learn a _language_ through it, and another user said *i would torch the bird alive if that were an option*.
   - Another user said that they deleted an unnamed app because it was a *waste of time*.
- **LinkedIn Gets 'Unhinged Slop'**: One user joked that they live for *the linkedin slop*.
   - Another user said that they post the most unhinged shit to get eyes on their posts and *win*.
- **HF Forums: Listed vs. Unlisted**: Someone asked about the meaning of listing/unlisting posts in Hugging Face's discuss forums.
   - No direct answer was provided in the messages.
- **Qwen 2.5 Models Overload HF**: Users noticed that someone is flooding Hugging Face with **Qwen 2.5 models** following the naming convention Qwen2.5-0.5B-Instruct-randomword1-randomword2-randomword3, linking it to [Gensyn](https://www.gensyn.ai/).
   - The motivation is suspected to be SEO-related, inflating the model count with easier-to-post smaller models and linking back to gensyn.ai for promotional purposes.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1420456885777338368)** (1 messages): 

> `GPU, monitor, drivers, Linux, Windows` 


- **Monitor blacks out when GPU heats up**: A user reported that every time the **GPU** fires up, the **monitor** goes black in both **Windows** and **Linux**.
   - They've tried to correct the **drivers** multiple times and are frustrated that they have to run the monitor off the motherboard.
- **Troubleshooting Black Screen on GPU Activation**: The user is facing a persistent issue where their monitor goes black whenever their GPU activates.
   - Despite numerous attempts to correct the drivers across both Windows and Linux, the problem persists, forcing them to rely on the motherboard for monitor output.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1420466933207404575)** (2 messages): 

> `trade-bench.live, UIUC students finance work` 


- **UIUC students publish finance work**: A member shared a link to [trade-bench.live](https://trade-bench.live/), showcasing work by **UIUC students** in the finance domain.
   - The member admitted to not understanding much of it, inviting others with finance expertise to provide insights on the project which they found *drab*.
- **Request for Finance Insights**: The member expressed hope that someone in finance would check out the resource.
   - They also invited people to share insights and clarifications, indicating they found it difficult to grasp.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1420362377534378017)** (6 messages): 

> `OOM Error on 3090, PEFT runs successful locally, SFTTrainer writes fine tuned model` 


- **3090 GPU runs out of Memory**: A member got an **OOM error** on a **3090 (Linux)**, even without **LoRA**, while trying to allocate **20.00 MiB** on a GPU with **23.55 GiB** total capacity.
   - It's unclear whether fine-tuning should work in **24G** GPU RAM without **LoRA**.
- **Local PEFT Runs Finally Succeeding**: After working through some issues with the **LoRA config**, a member reported finally getting some successful **PEFT runs locally**.
   - No further details were provided regarding the specifics of the resolved issues.
- **SFTTrainer auto-writes fine-tuned Models if output_dir is set**: A member inquired whether **SFTTrainer** automatically writes the fine-tuned model if `output_dir` is set.
   - The member later confirmed that yes, **SFTTrainer** does automatically write the fine-tuned model if `output_dir` is set.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1420396652304400484)** (2 messages): 

> `Global Greetings, Course Kickoff` 


- **Buckeyes and Buenas Dias!**: Enthusiastic members from **Ohio, USA** and **Madrid** chime in to say hello!
   - The international community eagerly anticipates a *curso magnífico*.
- **Course Commences!**: At least one participant announces they are **starting the course today**.
   - Many others will soon follow, hoping for a transformative learning experience.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1420530165393395813)** (2 messages): 

> `Hopper TMA, Minimal Matmul Kernel, CWM paper from FAIR` 


- **Hopper TMA Kernel Quest Begins**: A member is seeking a minimal **matmul kernel** implementation utilizing the **Hopper TMA** (Tensor Memory Accelerator) in raw CUDA, and not relying on CUTLASS or Triton.
   - The search is inspired by the new **CWM (Causal World Models)** paper from FAIR.
- **CWM Paper Sparks TMA Interest**: The new **CWM paper** from FAIR seems to be driving interest in optimized **matmul kernels** using Hopper's TMA.
   - The request specifies a need for a *minimal* implementation, suggesting an interest in understanding the fundamentals of TMA integration.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1420365628098871358)** (11 messages🔥): 

> `cuda headers, smem bank conflicts, cudaGraphicsGLRegisterImage and tex2d are undefined, TMA matmul kernel` 


- **Kernel Iteration Computations Get Functional**: A member suggested using a function `compute_iter<Is_first, Is_last, ...>(*args, **kwargs)` inside the loop, with a call to `compute_iter<False, False>` within the kernel.
   - Another user thought this was a *very good idea*.
- **Lambda Kernels Limit Argument Litter**: A member suggested using a **lambda** within the kernel to avoid writing a separate `__device__` function with a lot of arguments.
   - This allows calling the **lambda** inside and outside the main loop.
- **NCU Profiling Finally Finds SMEM Snags**: A user learned how to verify if a kernel has **smem bank conflicts** through **ncu profiling**.
   - The user was wondering what the number wrapped around curly brackets means.
- **CUDA Headers Cause Conundrums**: A user reported a weird issue where **cuda headers** weren't being automatically included, resulting in undefined functions like `cudaGraphicsGLRegisterImage` and `tex2d`.
   - Including `cuda_gl_interop.h` fixed the issue for `cudaGraphicsGLRegisterImage` and the issue persisted even when creating a new project with the CUDA default template in **Visual Studio 2022**.
- **WMMA Kernel Catches Unspecified Launch Crash**: A user is facing an `unspecified launch failure` with a **wmma kernel**.
   - The user is trying to implement a minimum **matmul kernel** that uses the **TMA**.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1420525420847173737)** (3 messages): 

> `torchrun API, torchrun --help` 


- **Torchrun API Usage Confusion**: A user inquired about the usage of the new **torchrun API** and reported that `uv run torchrun --help` shows different options compared to the [official documentation](https://docs.pytorch.org/docs/stable/elastic/run.html).
- **Discrepancy in Torchrun Help Output**: The output of `uv run torchrun --help` displayed a different set of options than expected based on the [PyTorch Elastic documentation](https://docs.pytorch.org/docs/stable/elastic/run.html), causing confusion about the correct usage.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1420479153131618396)** (10 messages🔥): 

> `CUDA and Triton, PTX Memory Consistency Model, Compound Memory Models, GPU Consistency Analysis, Dat3M Verification Tool` 


- **PTX Data Races Formally Analyzed**: A formal analysis of the **NVIDIA PTX Memory Consistency Model** ([ACM link](https://dl.acm.org/doi/10.1145/3297858.3304043)) explores how languages like **CUDA** and **Triton** can target **PTX** with memory consistency, even though **PTX** allows for data races.
- **Compound Memory Models Compositionally Amalgamate**: The compound memory model is *a compositional amalgamation where threads from each device continue to adhere to the memory ordering rules of that device’s original memory model*, according to the **PLDI 2023** paper ([DOI link](https://doi.org/10.1145/3591267), [PDF link](https://homepages.inf.ed.ac.uk/vnagaraj/papers/pldi23.pdf)).
- **Unified GPU Consistency Analysis Proposed**: The **ASPLOS 2024** paper *Towards Unified Analysis of GPU Consistency* ([DOI link](https://doi.org/10.1145/3622781.3674174), [PDF link](https://hernanponcedeleon.github.io/pdfs/asplos2024.pdf)) notes that while **CPU** consistency guarantees are well-understood, the same isn't true for **GPUs**.
- **Dat3M Tool Verifies Memory Models**: The **Dat3M** tool ([GitHub link](https://github.com/hernanponcedeleon/Dat3M)) translates **PTX** and **Vulkan** models into the **Dartagnan** verification tool, making it the first analysis tool for multiple **GPU** consistency models.
- **Missing PTX Fences Identified**: A member highlighted the automated identification of missing fences in **PTX**, as demonstrated in Figure 12 of a research paper.
   - Another member suggested implementing such checks at the **NVVM IR** layer instead of **PTX**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1420503128775590070)** (2 messages): 

> `Inter-warp operations, Intra-warp operations, Independent thread scheduling, NVIDIA GPUs, CTA clusters` 


- **Warped Minds Mulling NVIDIA's Thread Scheduling**: A member inquired about a good blog post explaining inter-warp and intra-warp operations behavior in **NVIDIA GPUs** with **independent thread scheduling**.
   - The member is particularly confused when dealing with a cluster of CTAs or a multi-CTA matmul, wondering about thread execution guarantees in architectures since Volta.
- **Unraveling the Mysteries of NVIDIA's Warp Operations**: The discussion revolves around understanding how inter-warp and intra-warp operations behave on **NVIDIA GPUs** when **independent thread scheduling** is enabled.
   - The key concern is the unpredictable execution of threads within a warp, particularly in scenarios like multi-CTA matmuls where multiple SMs access each other's shared memory.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1420530207244292206)** (1 messages): 

> `Puzzle difficulty, Puzzle completion time` 


- **Adventurers Gauge Puzzle Difficulty**: Several adventurers inquired about the **difficulty of the puzzles** and the typical **time required for completion**, seeking to gauge the challenge.
   - Some sought to benchmark against others, but no conclusion was reached due to lack of shared timing or concrete metrics.
- **No Triton Puzzles Completed Yet**: Currently there were no credible **Triton puzzle completion times** to compare experiences.
   - Most adventurers are still at the starting line, and none have crossed the finish line to report any reliable data.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1420473841657970861)** (1 messages): 

> `LLM serving, Embeddings Pricing, Kernel Profiling` 


- **Embeddings Pricing Exposed via Kernel Profiling**: A member shared a [Substack article](https://www.tensoreconomics.com/p/why-are-embeddings-so-cheap) detailing kernel profiling techniques to understand the profit margins of serving **LLMs**.
   - The author also shared a link to [his X/Twitter post](https://x.com/tugot17/status/1970913971734708391) related to the article.
- **Dive into Underlaying Kernels for Embedding Production**: A member has investigated the underlying kernels used to produce embeddings and shared his findings.
   - He suggests that profiling and investigating kernels can provide insights into the profit margins of serving LLMs, referencing his new Substack post for further details.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1420471280318152785)** (5 messages): 

> `Code Generation, Two-Stage Approach, Model Performance` 


- **Code Gen Tackles Raw Syntax**: Members discussed that code generation often uses raw syntax without constraints, providing guarantees via a formal grammar but sacrificing the natural language component.
   - They noted that humans don't typically code while thinking about the underlying grammar expected by the compiler, suggesting potential for training or fine-tuning a model to do so.
- **Two-Stage Approach Emerges**: Someone suggested a **two-stage approach**: pseudo-code generation followed by formal grammar translation.
   - They noted that the conversation also touched on the impact on model performance due to added constraints and the reduction of "degrees of freedom" for code generation.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1420445084310175744)** (5 messages): 

> `H100 matmul kernel runtime error, nvshmem usage in paper 2` 


- ****H100 Kernel** Crashes with Runtime Error**: A member reported a runtime error with the **H100** matmul kernel on Ubuntu 24.04, CUDA 12.8, PyTorch 2.7.0a0+nv25.03, TensorRT 10.9, and an NVIDIA H100 80GB HBM3 GPU, and provided [full logs and build/run details](https://gist.github.com/syadegari/ada8311c44c91357645d82c7f9dfbe71).
   - The error is: *std::runtime_error: Error in tile TMA descriptor creation: unspecified launch failure*.
- ****nvshmem** Inclusion Postponed to Paper 2**: A member inquired about the absence of **nvshmem** usage, and it was indicated that **nvshmem** usage is planned for paper 2, as illustrated in the [attached image](https://cdn.discordapp.com/attachments/1300872762163728550/1420526659001389056/image.png?ex=68d5b80b&is=68d4668b&hm=9484ede08cd43b12073bc50b9e94954bc9196cc04739ff374660dfb1becb6b44&).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1420350659261104158)** (17 messages🔥): 

> `MI300x8, amd-all2all leaderboard, amd-gemm-rs leaderboard` 


- **MI300x8 Achieves Personal Best on amd-all2all**: A member achieved a **personal best** of **1923 µs** on **MI300x8** for the `amd-all2all` leaderboard.
   - Other submissions on the `amd-all2all` leaderboard for **MI300x8** ranged from **1939 µs** to **2.12 ms**.
- **amd-all2all leaderboard gets filled with MI300x8 results**: There were several successful submissions to the `amd-all2all` leaderboard using **MI300x8**, with times of **108 ms**, **25.2 ms**, **25.4 ms**, **28.0 ms**, **25.3 ms**, and **4.70 ms**.
- **MI300x8 Excels on amd-gemm-rs Leaderboard**: Submissions to the `amd-gemm-rs` leaderboard using **MI300x8** achieved times between **572 µs** and **581 µs**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1420492634560004167)** (4 messages): 

> `Voltage Park H100s Donation, Nebius Exclusive Sponsorship` 


- **Voltage Park offers H100s donation**: A representative from **Voltage Park** offered to donate **H100s** for an upcoming hackathon.
   - However, the offer was declined due to an exclusive sponsorship agreement with **Nebius** for this particular hackathon.
- **Nebius Secures Exclusive Hackathon Sponsorship**: The GPU MODE hackathon secured an exclusive sponsorship with **Nebius**, preventing acceptance of other donations for this event.
   - Organizers expressed interest in collaborating with **Voltage Park** on future events and offered to discuss opportunities further.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1420448657706188950)** (2 messages): 

> `FLE Eval System Prompt, Image Analysis PR` 


- **FLE Eval System Prompt Shared**: A member shared a system prompt in **FLE eval**, attaching a file named *agent0_system_prompt.txt* from Discord CDN.
   - The link provided is [agent0_system_prompt.txt](https://cdn.discordapp.com/attachments/1354169122107293786/1420448657299210310/agent0_system_prompt.txt?ex=68d56f66&is=68d41de6&hm=721e6e6be3fcb1cc5c0dcb4b490f9ca58b962cedf7fda3980cf2e985469e0eaf&).
- **Image Analysis PR Coming Soon**: The same member mentioned their **Image Analysis PR** will be submitted the next day.
   - This suggests ongoing development or updates related to image analysis functionalities within the project.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1420376426502426686)** (4 messages): 

> `GEMM-RS atomic writes optimization with Iris, Iris shared memory initialization, GEMM-RS bias handling` 


- **Optimizing GEMM-RS Atomic Writes via Iris**: A member reported that while **Iris** worked, optimizing **GEMM-RS** with atomic writes proved challenging to accelerate.
   - They were advised to initialize the **Iris shared memory** inside the class, rather than strictly using it as an allocator.
- **GEMM-RS Bias Variations Explored**: A member tested three GEMM-RS variations, including one without bias addition and one always adding bias, to find optimizations when bias is None.
   - The member found that variations timed out, or failed to raise **TypeErrors**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1420419843668181013)** (3 messages): 

> `Refinement hierarchy, TmemAllocator vs cute.arch.alloc_tmem` 


- **Refinement Creates Hierarchy**: A member posited that *refinement* can be viewed as a **hierarchy**, where a value refines another if it can be derived from it, using the examples that `((6,4))` refines `(24)` because `size((6,4))=24`, but not the opposite direction.
   - They likened this to splitting a single mode into more complex patterns in one dimension, and drew a rough analogy to the relationship between an ordinary vector and a matrix of shape `(M, 1)`.
- **TmemAllocator Throwdown**: A user inquired about the difference between instantiating `TmemAllocator` and allocating from it versus using `cute.arch.alloc_tmem` in cutedsl.
   - No answer was given.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1420507807689609216)** (1 messages): 

> `Mojo Metal GPU target, Custom bitcode writer` 


- **Mojo Targets Metal GPUs with Custom Bitcode**: The new **Metal GPU target** in Mojo has generated excitement, particularly the availability of a **custom bitcode writer** for DSL targeting.
   - This work may be reusable for those interested in targeting specific DSLs at **Metal GPUs**.
- **Bitcode Writer Reusability**: The user inquired whether the **custom bitcode writer** code for the Metal GPU target in Mojo is available and reusable.
   - There is particular interest in leveraging this work to target specific DSLs at **Metal GPUs**.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1420371923736465508)** (6 messages): 

> `Picograd's tensor and engine, Eager and lazy execution policies, Tinygrad's architecture, Graph compiler, Shipping incremental intermediaries` 


- **Picograd's Load-Bearing Tensor and Engine**: The load-bearing part of **Picograd** is the **tensor** and **engine**, where the tensor will have two execution policies: **eager** and **lazy**.
   - The former is a handle on device-allocated storage and the latter is sugar for a **uop graph** that will be compiled.
- **Picograd Copies Tinygrad's Architecture**: The member is directly and shamelessly copying **tinygrad's architecture** to simplify design decisions and bridge the same semantic gap as **tinygrad's compiler**.
   - They stated that the target is *no triton or openmp*.
- **Picograd's CI Fuzzing Against Oracles**: The member plans to set up **CI to fuzz against numpy and torch oracles** once they get the vertical slice of a forward and backward pass.
   - They will then stop merging directly to master and focus on shipping code and book for eager mode.


  

---


### **GPU MODE ▷ #[cluster-management](https://discord.com/channels/1189498204333543425/1420098114076803142/1420420029866053815)** (2 messages): 

> `Singularity, Apptainer, Sylabs, Linux Foundation` 


- **Singularity Forked, Renamed Apptainer**: The open source project previously known as **Singularity** was renamed to **Apptainer** when it became part of the **Linux Foundation**, likely to distinguish it from the commercial fork called **Singularity[CE]** by **Sylabs**.
   - Despite the renaming, [Apptainer might still support the singularity alias](https://ciq.com/blog/what-to-expect-when-updating-from-singularityce-to-apptainer/) for the CLI.
- **Sylabs Commercial Fork**: **Sylabs** maintains a commercial fork of the original **Singularity** project, called **Singularity[CE]**.
   - This is distinct from the open-source **Apptainer** project, which is now under the **Linux Foundation**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1420353691558481950)** (37 messages🔥): 

> `Seed-OSS thinking budget, Conversation.json to markdown, LM Studio Plugins in Linux, Ollama Fine Tuning, LoRA injection into models` 


- **Set Thinking Budget for Seed-OSS**: A user asked how to set the **thinking budget** for **Seed-OSS**.
- **Markdown Parser Sought**: A member is looking for a good way to parse `.conversation.json` files into a human-readable markdown format due to the variability in models and regeneration versioning.
- **LM Studio Linux Plugin Availability**: A user reported that the **Linux version** of **LM Studio** doesn't offer as many plugins as the **Windows version**.
- **Debate Erupts on Ollama Fine-Tuning**: A discussion ensued whether injecting data and using LoRA with **Ollama** constitutes fine-tuning, with claims that the knowledge is baked into the model files themselves, and **isn't just a system prompt**.
- **Ollama LoRA injection confirmed**: A user confirmed that **Ollama** not only supports running models locally, but also allows injecting **LoRA weights**, customizing system prompts, and even creating your own model variants where knowledge is directly embedded into the model’s structure.
   - However, they noted that *it needs some setup, it's not like just there*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1420496125420634164)** (13 messages🔥): 

> `Budget GPUs, Tesla K80s` 


- **Budget GPUs Recommendations**: For budget consumer GPUs, recommendations included the **2060 12GB** at $150, **3060 12GB** for $200, **2080ti** for ~$230, and **5060ti 16GB** for $400 if buying new.
   - A **used 3090** was also suggested, but another user pointed out it costs $700, hardly budget.
- **Tesla K80s deemed e-waste**: A question was posed on whether **Tesla K80s** are viable given their price range of $200-300 for refurbs.
   - One user responded that *Tesla generation is not recommended for AI/LLM use anymore tbh, basically e-waste*.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1420391964603449435)** (3 messages): 

> `Measuring AI Dialogue Coherence and Novelty, NYC Meetup in Central Park` 


- **Swiss Researcher Queries AI Dialogue Metrics**: An interdisciplinary researcher from Switzerland asked the tech community about the importance of measuring **coherence and novelty in AI dialogue**.
   - The researcher's background is in International Relations, hinting at a potential interest in applying these metrics to analyze **AI's role in global communication**.
- **EleutherAI NYC Meetup Announced**: A member announced a **NYC Meetup** planned for Saturday afternoon in Central Park, with a link to a [Discord channel](https://discord.com/channels/729741769192767510/1417496438782431314/1420426137976176640) for details.
   - They also linked to [a Twitter post](https://x.com/SatyaScribbles/status/1970513181337350483) to gauge interest in this direction.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1420397609218080830)** (12 messages🔥): 

> `DeepIgnorence Generalization Difficulty, Mathematical Formalism for Knowledge Completion, CFG on Style Transfer, Data Centric Approaches to ML/AI` 


- ****DeepIgnorence's** Generalization Difficulties Highlighted**: A member discussed how **DeepIgnorence** requires a difficult type of generalization, noting that models excel at style transfer but struggle with more complex inference.
   - For example, we should not expect to be able to train on clothed minors and nude adults and have an image model that can't generate CSAM. That's effectively style transfer, which is something models are extremely good at.
- **Mathematical Formalism for Knowledge Completion Sought**: A member inquired about a mathematical formalism to distinguish settings where knowledge completion works, highlighting the complexity of the problem.
   - They suggested that in the worst case, it seems information theoretic, i.e., a model will not be able to reason its way to a very specific fact which is independent of other knowledge and unknown to the model.
- **Discussion on **CFG's** impact on Style Transfer**: Members discussed the effect of techniques like **CFG** on style transfer.
   - One member heard anecdotally that models that don’t use it cannot perform style transfer as well. *If so, maybe some research can be done with the LLM equivalent of CFG to see if it can bridge the gap between concepts to fill in missing knowledge*.
- **AI Engineer Seeks Research Collaboration**: An AI engineer with a background in applied math and computer science from Imperial College London and Oxford is seeking research collaborations.
   - They aim to use competition winnings to fund research and transition from industry to academia, focusing on data-centric approaches to machine learning/AI.
- **Style Transfer and Knowledge Gaps: Different Sides of the Same Coin?**: Members debated whether style transfer and closing knowledge gaps are fundamentally different, or related.
   - One member thinks *both tasks could be seen as attempts to generate samples not present in the training data from nearby data samples in the training data* and that *style transfer just seems like an easier task in that vein*.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1420532654117224508)** (3 messages): 

> `GSM8k Benchmark, flexible-extract, strict-match` 


- **Flexible Extraction Fails GSM8k Benchmark**: In **GSM8k** benchmark version **3**, `flexible-extract` scored **0.3594** which is worse than `strict-match` which scored **0.5742** in `exact_match` metric with **5-shot** learning.
- **Funny benchmark**: A member found it funny, saying *haha how can flexible be worse than strict*.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1420417274237419673)** (1 messages): 

> `Benchmarking Prompting Methods in VLMs, Interpretability Studies on VLMs, Ineffectiveness of LLM Prompting Techniques for VLMs, Mech-Interpretability Probing Study for VLMs` 


- **Prompting Methods Benchmarked in VLMs?**: A member is seeking studies that **benchmark different prompting methods in VLMs** and interpretability studies explaining their effectiveness.
- **Normal LLM Prompting Ineffective for VLMs?**: The user notes having seen several studies which discuss how **ineffective normal LLM prompting techniques are for VLMs**.
- **Mech-Interp Probing Study for VLMs?**: The user considers if a **mech-Interp oriented probing study** might be helpful, but is unsure of how to begin.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1420354681087066252)** (15 messages🔥): 

> `Kimi doesn't encourage delusions, Mini version of Kimi, Qwen model distilled on K2` 


- **Kimi Doesn't Encourage Delusions**: A member shared that they were testing **Kimi** and love that it *doesn't encourage delusions*.
   - Another member shared an image and **Kimi's** analysis was *"are you serious?"*.
- **Kimi's Cool Response goes Viral**: A member shared [this link](https://x.com/vllm_project/status/1970814441718755685) saying *"This is so cool from Kimi"*.
   - The message being replied to was a user humoring the idea that *private voices in your head are Jesus, scripture mentions pets getting raptured, and the whole 2025 date is baseless hype*, and **Kimi's** response bluntly denied all claims.
- **Mini-Kimi Distilled on Qwen?**: A member wondered if there will ever be a **mini version of Kimi** with the same writing style but smaller.
   - Another member doubts this is in the **Moonshot** team's interests, suggesting that the best bet would be a smaller **Qwen** model distilled on **K2**.
- **Qwen Distills Reasoning with Kimi**: A member doubts the rationality of distilling a **Qwen** model, arguing that **Deepseek** only did it because **Qwen** lacked good reasoning until **Qwen 2.5**.
   - Another member countered that **K2** has a different style of problem-solving and excellent writing, so a smaller **Qwen3** model could benefit from distillation in certain attributes like prose and referencing obscure knowledge.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1420371100059308082)** (11 messages🔥): 

> `Gemini Live Model, Chrome DevTools MCP, AI Coding Agents` 


- **Gemini goes Live with killer Audio**: Google’s Logan Kilpatrick announced a new **Gemini Live model** with native audio, improved function calling, and more natural conversations, as announced on [X](https://x.com/OfficialLoganK/status/1970546338858246255).
   - Early users praise the flow and accents, but report **iOS Safari issues**, background-noise sensitivity, session-length limits, **STT accuracy** with accents, overly cautious censorship, missing price transparency, and desire for embodiment / wearables.
- **Chrome DevTools MCP opens for AI Agents**: Google announced the public preview of **Chrome DevTools MCP**, a new server that lets **AI coding agents** (Claude Code, Cursor, VS Code, Gemini, etc.) control and inspect a live Chrome browser through CDP/Puppeteer, as announced on [X](https://x.com/chromiumdev/status/1970505063064825994).
   - Agents can now run performance traces, examine the DOM/console, capture screenshots and network traffic, and debug web apps in real time with one-line installation via npx.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/)** (1 messages): 

glassbeadaleph: i think so, give me one second
  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1420367981518524436)** (9 messages🔥): 

> `Embedded Resources title vs name, Claude Code, ReadResourceResult contents array` 


- **Embedded Resources Lacks Title and Name**: A member noted discrepancies in the [Model Context Protocol documentation](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#embedded-resources) where **embedded resources** are implied to have a *title*, which is missing in `schema.ts`, and questioned the absence of a *name* field to match the *Resource* object.
   - It was argued that both *title* and *name* might be necessary because embedded resources aren't always retrievable via a *read resource* call.
- **Claude Code Debated for SEP Documentation**: A member suggested using **Claude Code** to write a SEP (Standard Enhancement Proposal) documentation, calling it a *good test* for the tool's capabilities.
   - Another member thought that getting an **SEP** for the topic would be relatively easy.
- **ReadResourceResult's contents Array Questioned**: A discussion arose around the `ReadResourceResult.contents` array in [this GitHub issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1533), questioning its intended use and semantics, as it is undocumented.
   - One member provided an example of a Web Resource, consisting of **html** and associated **images**, or scenarios where tokenizable/renderable mime types haven't been negotiated, to explain its use.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1420353196844650526)** (8 messages🔥): 

> `Anthropic Misuse Report, Cybercrime and AI, AI-fabricated credentials` 


- **Anthropic report: AI misuse focuses on cybercrime**: A member shared [Anthropic's report](https://www.anthropic.com/news/detecting-countering-misuse-aug-2025) on detecting and countering AI misuse, highlighting that the actual threats are low-grade cybercrime, or *vibe hacking*.
   - The discussion touched on whether applying for jobs with **fabricated credentials** is illegal, regardless of location, and the report specifically mentions **completely fabricated master's degrees**.
- **LLMs automate personal life**: A member noted that an **LLM** did all the legwork in achieving a recent accomplishment.
   - According to them, all they had to do was *spend many hours self-reflecting and feeding info about myself into the AI*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1420410886379798549)** (6 messages): 

> `Aider's /clear command, Aider access to Internet search` 


- **"/clear" command clears chat history, but not context**: Users clarified that the `/clear` command only clears the chat history, but **added files remain in the context**.
   - The command `/context` can be used to check how many tokens are dedicated to each file.
- **Aider Lacks Native Internet Search, but scrapes URLs instead.**: A user inquired about giving aider access to Internet search.
   - Another user clarified that this is not possible with the main branch, but you can instead use `/web https://www.example.com/` to **scrape a website**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1420402109110685788)** (3 messages): 

> `Saturday evening talks, Reading papers before talks` 


- **Saturday Evening Talks Anticipation**: A member looks forward to the Saturday evening (European time) talks.
   - The announcement came earlier in the week.
- **Pre-Talk Paper Reading**: A member expressed a desire to read papers before the talks to better follow Yannick or the presenter.
   - This would enhance their understanding and engagement during the sessions.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1420408529533009970)** (2 messages): 

> `Hyperparameters for Diffusion Models, ODE Solvers vs DPM++2m, Applications of Fast Inference, Diffusion Efficiency Research` 


- **Hyperparameters Generate Images Like Distillation Models!**: The author of ["Hyperparameters are all you need"](https://zenodo.org/records/17180452) will present their paper, which uses a **five-step inference** method for a diffusion model.
   - Key results show **8-step inference** beats **DPM++2m's 20-step inference** in FID scores with ~60% reduction in computational cost, using existing models without retraining.
- **ODE Solvers Outperform DPM++2m in Fewer Steps**: According to the paper, an **8-step Diffusion ODE Solver** outperforms **20-step DPM++2m** without needing additional training.
   - The author seeks feedback, collaborators, and ideas for applications where *inference speed is critical*, especially from those working on diffusion efficiency, inviting discussion on **ODE solver improvements**.
- **ArXiv Paper about to be reviewed**: A user announced that they will be reviewing [this paper](https://arxiv.org/abs/2509.19249) soon.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/)** (1 messages): 

.neoneye: https://x.com/Alibaba_Qwen/status/1970599323013652705
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1420450783673192448)** (5 messages): 

> `Manus PDF download issues, Beta Pro Access` 


- **Manus PDF Download Stymied**: A user reported that **Manus** was getting stuck while downloading a **PDF** for researching accounts and even after manually downloading the file and providing a link, **Manus** kept asking to upload the file.
   - The user sought advice on resolving this issue, but the conversation ended there.
- **Seeking Beta Pro Access**: A user inquired about obtaining access to **Beta Pro**.
   - The discussion ended without a response, leaving the method for acquiring **Beta Pro** access unresolved.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1420444643904196712)** (3 messages): 

> `Modular contributor, Contributing to Mojo` 


- **Users ask about contributing to Modular**: A user inquired about contributing their talents to Modular.
   - They were asked to DM a staff member for further discussion.
- **Contributor Opportunities Explored**: A member expressed interest in leveraging their skills to support Modular's services.
   - A staff member suggested direct messaging to explore potential collaboration avenues.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1420466462795104336)** (1 messages): 

> `New Fundraising, Unified compute layer` 


- **Modular Closes $250M Funding Round!**: Modular announces it has raised **$250M** to accelerate building **AI's unified compute layer**.
   - The team expressed gratitude to the community for its contributions, feedback, and momentum, and promised to empower the community with more features in the coming year.
- **Community Momentum Fuels Funding Success**: The funding success is attributed to the community's invaluable contributions and feedback.
   - The company commits to enhancing community empowerment through feature enhancements and expedited response to feedback.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1420427883788111953)** (3 messages): 

> `clspv build errors, Python bindings for clspv` 


- **clspv Main Branch Plagued by Build Errors**: The main branch of **clspv** is currently failing to build due to errors, but a user found that reverting to previous commits resolves the issue and shared a [forked repository](https://github.com/softcookiepp/clspv) with a working stable branch.
   - Users can pull the forked repository and checkout the **stable** branch to build **clspv** successfully.
- **Python Bindings in the Works for clspv**: A user is developing **Python bindings** for **clspv**, with the goal of enabling direct installation via **pip** using a single command.
   - This enhancement would streamline the installation process, making **clspv** more accessible to **Python developers**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1420378828072554597)** (1 messages): 

> `DSPy attachments, UV Tooling` 


- **Attachments Add-On Attracts Attention**: The `attachments` add on for **DSPy** is useful for adding new files, easily!
   - It is a standalone `uv add` for python.
- **UV Tooling Integration**: Discussion highlights the ease of adding new files using the `attachments` add-on within the DSPy framework.
   - The add-on is noted for its standalone `uv add` functionality, streamlining the process for Python projects.


  