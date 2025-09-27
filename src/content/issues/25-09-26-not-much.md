---
id: MjAyNS0w
title: not much happened today
date: '2025-09-26T05:44:39.731046Z'
description: >-
  **Google** released a dense September update including **Gemini Robotics 1.5**
  with enhanced spatial/temporal reasoning, **Gemini Live**, **EmbeddingGemma**,
  and **Veo 3 GA** powering creative workflows. They also introduced agentic
  features like restaurant-reservation agents and reduced pricing for **Gemini
  2.5 Flash**. **Meta AI** unveiled the open-weight **Code World Model (CWM)
  32B**, excelling in code semantics and math benchmarks, with innovations in
  training code models via execution traces. Local-first coding setups highlight
  **Qwen3-Coder-30B** running efficiently on consumer GPUs, paired with tools
  like **Cline** and **LM Studio**. Runtime improvements include **vLLM v1**
  supporting hybrid models and **mlx-lm** adding batch inference on Apple
  silicon. In infrastructure, **FlashAttention 4** was reverse-engineered
  revealing a ~20% speedup from architectural optimizations. **Perplexity AI**
  advances its independent web index and browsing API with upcoming feed
  refreshes. Embedding latency improvements were achieved by **Superhuman**
  using **Baseten**.
companies:
  - google
  - meta-ai-fair
  - perplexity-ai
  - baseten
models:
  - gemini-robotics-1.5
  - gemini-live
  - embeddinggemma
  - veo-3
  - gemini-2.5-flash
  - code-world-model-32b
  - qwen3-coder-30b
  - vllm-v1
  - mlx-lm
  - flashattention-4
topics:
  - spatial-reasoning
  - temporal-reasoning
  - agentic-ai
  - code-semantics
  - code-execution-traces
  - coding-infrastructure
  - runtime-optimization
  - batch-inference
  - embedding-latency
  - api
  - model-optimization
  - model-performance
people:
  - osanseviero
  - _anniexie
  - rmstein
  - scaling01
  - giffmana
  - cline
  - redhat_ai
  - awnihannun
  - charles_irl
  - bernhardsson
  - akshat_b
  - aravsrinivas
---



**a quiet day to end the week**

> AI News for 9/25/2025-9/26/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (195 channels, and 5022 messages) for you. Estimated reading time saved (at 200wpm): 400 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Lots of launches next week so it's nice to have a breather. [Apply for round 2 of AIE CODE](https://apply.ai.engineer/)!

---

# AI Twitter Recap

**Google’s September stack: Gemini Robotics 1.5, Live, Veo 3, Flash pricing**

- **Gemini Robotics 1.5 + Live + Veo 3 GA**: Google shipped a dense set of updates in September: Gemini Robotics 1.5 (including high-level reasoning “ER 1.5”), the latest Gemini Live, EmbeddingGemma, Veo 3 GA + API updates, AI Edge gallery, Batch API embedding support, Flash/Flash Lite updates, Chrome DevTools MCP, VaultGemma and more, per [@osanseviero](https://twitter.com/osanseviero/status/1971468195308712431). Robotics-ER 1.5 is positioned as strong in spatial/temporal reasoning with “thinking” to improve answers [@_anniexie](https://twitter.com/_anniexie/status/1971477645096517832). Veo 3 is already powering production creative workflows (e.g., Flow by Google’s music video case study) [@FlowbyGoogle](https://twitter.com/FlowbyGoogle/status/1971607613805867314). Google is also rolling out agentic features to broader users, e.g., restaurant-reservation agents in Labs [@rmstein](https://twitter.com/rmstein/status/1971617040193724661). Meanwhile, Gemini 2.5 Flash got a small quality bump but is ~30% cheaper [@scaling01](https://twitter.com/scaling01/status/1971578192512029045).

**Code intelligence and agentic coding**

- **Meta’s Code World Model (CWM)**: New open-weights 32B model that learns code semantics via execution traces and agentic interactions (bug fixing, editing, Docker runs). Claims: simulate Python step-by-step, multi-turn software tasks, 131k context; competitive coding metrics (e.g., 65.7% SWE-bench Verified, 68.4% LiveCodeBench) plus strong math (96.5% Math-500, 75.8% AIME-24). Paper, code, weights: [summary](https://twitter.com/TheTuringPost/status/1971697629697659099), [paper](https://twitter.com/TheTuringPost/status/1971697642288959496). Related idea: train code models by interleaving source with interpreter state to force semantic understanding [@giffmana](https://twitter.com/giffmana/status/1971507878025445653).
- **Local-first coding setups**: Qwen3-Coder-30B (AWQ 4-bit) hits ~115 tok/s on a single 3090, “zero-shatting Pac-Man” [@QuixiAI](https://twitter.com/QuixiAI/status/1971427136977453184). Developers are pairing Qwen3-Coder with Cline + LM Studio for high-quality local coding [@cline](https://twitter.com/cline/status/1971591597080064121) ([guide](https://twitter.com/awnihannun/status/1971603427131351218), [blog](https://twitter.com/cline/status/1971591609386188993)). Cline also shipped a “workflow for building workflows” ([prompt recipe](https://twitter.com/cline/status/1971436086217122213), [blog](https://twitter.com/cline/status/1971436097965375689)) and quietly bumped its “code-supernova” provider to a 1M-token context (from 200k) during free alpha [@cline](https://twitter.com/cline/status/1971660202387951962).
- **Runtime/backends**: vLLM v1 treats hybrid models (e.g., Mamba/Mamba2, linear attention) as first-class, with perf gains vs v0 [@RedHat_AI](https://twitter.com/RedHat_AI/status/1971569727844876350). On Apple silicon, mlx-lm added batch inference for hybrid SSMs/sliding-window attention and support for Meta’s CWM [@awnihannun](https://twitter.com/awnihannun/status/1971763001880670213).

**Systems and infra: kernels, search, and hosting**

- **FlashAttention 4 decoded**: Modal reverse-engineered FA4, explaining where the ~20% speedup comes from: specialized warp layouts, cubic approximation of exp for softmax, more aggressive asynchrony. Deep write-up and code pointers: [@charles_irl](https://twitter.com/charles_irl/status/1971587871237898482), [blog](https://twitter.com/charles_irl/status/1971587874496868601), plus engineering commentary [@bernhardsson](https://twitter.com/bernhardsson/status/1971603562355716160), [@akshat_b](https://twitter.com/akshat_b/status/1971617146930450758).
- **Search APIs and web index**: Perplexity continues to build a non-Google/Microsoft web index ([argument](https://twitter.com/AravSrinivas/status/1971438329460867413)) and is shipping a browsing API; discover feed refresh lands next week (iOS first) [@AravSrinivas](https://twitter.com/AravSrinivas/status/1971443978810896424), [update](https://twitter.com/AravSrinivas/status/1971687653545525467). Devs are already integrating it as a custom tool [@thdxr](https://twitter.com/thdxr/status/1971510163501953436).
- **Inference infra**: Superhuman cut P95 embedding latency ~80% to 500ms by moving to Baseten [@basetenco](https://twitter.com/basetenco/status/1971683977242259623). Ollama Cloud added free-to-try Kimi K2 “1T-cloud” and DeepSeek V3.1 “671b-cloud” SKUs [@ollama](https://twitter.com/ollama/status/1971750071483167010). NVIDIA is increasingly active in open contributions (300+ models/datasets/apps on HF over the past year) [@ClementDelangue](https://twitter.com/ClementDelangue/status/1971698860146999502).

**Research highlights: RLHF variants, decoding, 3D parts, science FMs**

- **RLHF and decoding**: RLBFF proposes extracting binary-checkable principles from natural-language feedback and combining them with verifiable rewards to train reward models that capture nuance beyond correctness [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1971520102857408705) ([abs](https://twitter.com/iScienceLuvr/status/1971520105143242952)). VCRL explores variance-based curriculum RL for LLMs [@_akhaliq](https://twitter.com/_akhaliq/status/1971593807365132382). LATTS decodes by sampling from the product of the LM and a reward model, tracing accuracy over tokens [@f14bertolotti](https://twitter.com/f14bertolotti/status/1971469173185527955).
- **3D part-level generation**: Tencent releases Hunyuan3D-Part with two models: P3-SAM (first native 3D part segmentation) and X-Part (SOTA controllability/shape quality). Trained from a 3.7M-shape dataset with clean part annotations; full code/weights and demos provided [@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1971491034044694798).
- **Multimodal reasoning with less data**: Alibaba’s MMR1 introduces Variance-Aware Sampling to stabilize RL fine-tuning under scarce high-quality data; releases ~1.6M CoT, 15k RL QA datasets and 3B/7B/32B models [@HuggingPapers](https://twitter.com/HuggingPapers/status/1971487864807469236).
- **Domain FMs**: SciReasoner pretrains on 206B scientific tokens (text, sequences, and pairs), aligns with 40M SFT and RL with task-shaped rewards to elicit deliberate scientific reasoning [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1971519610630586702). In healthcare, CATCH-FM scales EHR FMs to 2.4B params for cancer pre-screening and sets SOTA on pancreatic risk in EHRSHOT [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1971521050057072995).

**Benchmarks and evaluation practice: GDPVal, SWE-bench, and “evals as PRDs”**

- **GDPVal debate**: A new benchmark spanning tasks in 44 occupations across the top 9 US GDP sectors triggered heated takes. Proponents argue it operationalizes “usefulness” and shows models at 77–95% of “AGI” by an economic yardstick [@Smol_AI](https://twitter.com/Smol_AI/status/1971426804826267994), [@swyx](https://twitter.com/swyx/status/1971427791770882463), [@markchen90](https://twitter.com/markchen90/status/1971449404734439831). Skeptics caution against literalism, cite task/selection bias and grader style effects, and emphasize the trend not the threshold; note models completed tasks ~100× faster/cheaper than experts but question real-world transfer [@scaling01](https://twitter.com/scaling01/status/1971431825433374866), [skepticism](https://twitter.com/scaling01/status/1971432462820802834), [style bias](https://twitter.com/scaling01/status/1971432758817050970), [task bias](https://twitter.com/scaling01/status/1971433067266089395).
- **SWE-bench Verified clarity**: The widely-circulating number from recent results is pass@1 on TTS (tools-to-success), per [@alexandr_wang](https://twitter.com/alexandr_wang/status/1971603685559140663).
- **Evaluation practice**: Evals are increasingly product-defining (“the new PRDs”), but LLM-as-judge without human oversight is unreliable. Error analysis should precede metric design; human-in-the-loop builds trust [podcast recap via @bnicholehopkins](https://twitter.com/bnicholehopkins/status/1971683830269350161). ARC Prize hosted a Boston event focused on interactive benchmarks for intelligence [@arcprize](https://twitter.com/arcprize/status/1971609644004200693). A practical north-star for usefulness: tokens used / $ spent [@scaling01](https://twitter.com/scaling01/status/1971433691848262084).

**Optimization and scaling theory: Modular Manifolds, MoE compute, compute scaling, tokenization**

- **Modular Manifolds (Thinky Machines)**: New post by Jeremy Bernstein et al. co-designs optimizers with manifold constraints on weight matrices (e.g., Stiefel: singular values = 1), extending Muon (“managed metrics”) to stabilize training on specific “shapes.” Strong endorsements from practitioners; also discussion of layer-wise schedules/discriminative fine-tuning [@thinkymachines](https://twitter.com/thinkymachines/status/1971623409873244462), [@jxbz](https://twitter.com/jxbz/status/1971703483767435446), [@johnschulman2](https://twitter.com/johnschulman2/status/1971630456471945711), [@cHHillee](https://twitter.com/cHHillee/status/1971641318888853748), [@Dorialexander](https://twitter.com/Dorialexander/status/1971631250801844687).
- **MoE compute optimality and kernels**: Practitioners argue MoEs are compute-optimal over lifetime if you scale data by total/active params; data scale (“tens of trillions” tokens) is the bottleneck [@teortaxesTex](https://twitter.com/teortaxesTex/status/1971453835207131244), [follow-up](https://twitter.com/teortaxesTex/status/1971454807455265156). There’s pushback on very large dense models (e.g., 405B) vs sparser MoE [@scaling01](https://twitter.com/scaling01/status/1971541644647522595). Kernel-level gains matter: Triton RoPE faster than PyTorch (0.083ms vs 0.235ms) [@vikhyatk](https://twitter.com/vikhyatk/status/1971694488004481058). Also, attention’s O(T) per query is increasingly untenable for very long contexts [@francoisfleuret](https://twitter.com/francoisfleuret/status/1971632756716372053).
- **Compute scaling at OpenAI**: New analysis suggests GPT-5 used less total training compute than GPT-4.5 due to outsized returns from post-training at smaller scale; authors expect GPT-6 to swing back to higher training FLOPs as buildouts land [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1971675079219282422), [follow-up](https://twitter.com/EpochAIResearch/status/1971675189575602304).
- **Tokenization debate**: Multiple posts argue “tokenizer-free” is a misnomer; even bytes inherit Unicode design choices and biases. Tokenization remains a central design element; practical guidance and from-scratch BPE implementations shared [@giffmana](https://twitter.com/giffmana/status/1971500080072208674), [@rasbt](https://twitter.com/rasbt/status/1971575045769380056), [commentary](https://twitter.com/lateinteraction/status/1971548611700994538).

**Top tweets (by engagement)**

- New grads “just try” with ChatGPT instead of asking how: observation about agency shift in entry-level hires [@dylan522p](https://twitter.com/dylan522p/status/1971425552902082941) (~2.4K).
- Richard Sutton vs. LLMs debate: long-form discussion on continual learning vs. current LLM paradigm; sparked significant back-and-forth in the community [@dwarkesh_sp](https://twitter.com/dwarkesh_sp/status/1971606180553183379) (~2.5K).
- Modular Manifolds post: theoretical/algorithmic advances for stable training via manifold-constrained weights [@thinkymachines](https://twitter.com/thinkymachines/status/1971623409873244462) (~2.5K).
- OpenAI platform: function calling now supports returning files/images from tools, not just JSON/text [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1971618905941856495) (~1.4K).
- Tencent Hunyuan3D-Part: open-source part-level 3D shape generation with native 3D segmentation and diffusion-based decomposition [@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1971491034044694798) (~1.1K).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3 roadmap + abliterated uncensoring results

- [**Alibaba just unveiled their Qwen roadmap. The ambition is staggering!**](https://www.reddit.com/r/LocalLLaMA/comments/1nq182d/alibaba_just_unveiled_their_qwen_roadmap_the/) (Activity: 954): **Alibaba’s Qwen roadmap slide highlights an aggressive push to a unified multimodal stack with extreme scaling: context windows from** `1M → 100M` **tokens, parameter counts from roughly** `1T → 10T`**, test‑time compute scaling from** `64k → 1M`**, and training data expansion from** `10T → 100T` **tokens. It also emphasizes unlimited synthetic data generation pipelines and richer agent capabilities across task complexity, multi‑agent interaction, and continual/interactive learning—doubling down on a “scaling is all you need” strategy for future Qwen models.** Top comments question feasibility and accessibility: excitement about `100M` context, skepticism that such models will remain open, and practicality concerns about running `>1T`‑param models locally.
    - The claimed `~100M` context window implies non-standard attention or memory systems; naive full attention is O(n^2) and would require an attention matrix with `1e16` entries at 100M tokens—computationally intractable. Even with KV caching, memory explodes: for hidden size 8192, FP16, ~80 layers, KV is ~32 KB/token/layer → `~3.2 TB` per layer and `~256 TB` total for 100M tokens, so practical implementations would need techniques like retrieval-augmented chunking, recurrent/compressive memory, or linear/partial attention (e.g., blockwise/ring attention) rather than true dense long-range attention.
    - Running `>1T` parameter models locally is beyond consumer hardware: parameters alone are `~2 TB` in BF16/FP16, `~1 TB` in 8-bit, and `~0.5 TB` in 4-bit—before activations/KV cache. This necessitates multi-node model parallelism over NVLink/NVSwitch; for context, a single 8x H100 80GB server offers 640GB VRAM, so a trillion-param model would likely need several such nodes just to load weights, with significant interconnect bandwidth to sustain inference throughput.
    - Some commenters expect the largest Qwen checkpoints/long-context variants to be API-only despite Alibaba’s history of open-sourcing smaller Qwen models. Practically, bleeding-edge features (e.g., `100M` context or `>1T` params) often remain closed due to training data/licensing and deployment costs, while mid-size open weights target research and on-prem use; teams should plan accordingly for integration and benchmarking.
- [**IMPORTANT: Why Abliterated Models SUCK. Here is a better way to uncensor LLMs.**](https://www.reddit.com/r/LocalLLaMA/comments/1nq0cp9/important_why_abliterated_models_suck_here_is_a/) (Activity: 433): **OP reports that “abliteration” (weight-level uncensoring) consistently degrades capability—especially on MoE like Qwen3‑30B‑A3B—hurting logical reasoning, agentic/tool-use behavior, and increasing hallucinations, often making abliterated 30B trail non‑abliterated 4–8B. They claim post‑abliteration finetuning largely restores (“heals”) performance: e.g., mradermacher’s Qwen3‑30B‑A3B‑abliterated‑erotic‑i1‑GGUF (tested at** `i1‑Q4_K_S`**) showed lower hallucination and more reliable tool calls under [MCP](https://modelcontextprotocol.io/) than other abliterated Qwen3‑30B variants (Huihui’s Thinking‑2507, Fusion‑9010, Instruct‑2507), while [mlabonne/NeuralDaredevil‑8B‑abliterated](https://huggingface.co/mlabonne/NeuralDaredevil-8B-abliterated) (a DPO finetune of Llama3‑8B; [DPO](https://arxiv.org/abs/2305.18290)) reportedly surpasses its base while remaining uncensored. OP urges finetuning abliterated Qwen3‑30B‑A3B on high‑quality data to retain uncensoring without sacrificing performance; context includes GGUF quantization ([GGUF](https://github.com/ggerganov/llama.cpp/blob/master/gguf.md)) and the Qwen3 MoE family ([Qwen3](https://github.com/QwenLM/Qwen3)).** Top comments request standardized benchmarks for abliteration impacts beyond NSFW tasks, and characterize the observed recovery as expected “model healing” (unconstrained weight edits break circuits; further training re-learns them). Skeptics argue that if finetuning is required, abliteration is unnecessary—claiming abliterated+finetuned models don’t outperform plain finetunes.
    - Unconstrained weight edits (e.g., zeroing “negative bias” terms or other abliteration passes) predictably degrade capability; commenters refer to post-edit recovery as **“model healing.”** The fix is additional training guided by a loss (SFT/LoRA or full fine-tune) so the network can re-learn connections broken by the edit, similar to how pruning/quantization requires retraining to restore perplexity and task accuracy. Takeaway: if you must modify weights, do it under an objective or expect destroyed generalization until sufficient fine-tuning heals it.
    - For evaluation beyond NSFW, the **Uncensored General Intelligence (UGI)** leaderboard is suggested as a broader capability benchmark: https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard. This helps quantify whether abliteration harms reasoning/instruction-following and compares against plain fine-tunes, avoiding overfitting to porn-only metrics.
    - Several practitioners report that “abliterated + fine-tune” rarely outperforms a straight fine-tune, advocating non-destructive uncensoring via targeted SFT or merges. Cited alternatives include “Josiefied,” “Dolphin,” and releases from TheDrummer, e.g., **Qwen3-8B-192k Josiefied (GGUF)** [https://huggingface.co/DavidAU/Qwen3-8B-192k-Josiefied-Uncensored-NEO-Max-GGUF], **Dolphin-Mistral-24B Venice i1 (GGUF)** [https://huggingface.co/mradermacher/Dolphin-Mistral-24B-Venice-Edition-i1-GGUF], and **TheDrummer** profile [https://huggingface.co/TheDrummer]. The aim is to retain base competence (e.g., long-context `192k` variants) while adjusting instruction style, avoiding catastrophic weight edits.

### 2. China launches: Hunyuan Image 3.0 + Fenghua GPU

- [**Tencent is teasing the world’s most powerful open-source text-to-image model, Hunyuan Image 3.0 Drops Sept 28**](https://www.reddit.com/r/LocalLLaMA/comments/1nqaiaz/tencent_is_teasing_the_worlds_most_powerful/) (Activity: 225): **Tencent is teasing Hunyuan Image 3.0, an open‑source text‑to‑image model slated for release on** `Sept 28`**, billed as the “world’s most powerful” in its class. The teaser poster (no benchmarks or samples yet) hints at heavy hardware needs—commenters interpret a “VRAM 96” note as ~**`96 GB VRAM` **recommended—while details on architecture, training scale, resolution, throughput, or license remain undisclosed. Image: https://i.redd.it/t8w84ihz1crf1.jpeg** Commenters are skeptical of pre‑release hype, arguing teased models often underdeliver versus “shadow‑dropped” strong releases (e.g., Qwen vs hyped GPT-5; SD3 vs Flux) and question the “most powerful” claim absent public benchmarks or comparisons with other large open‑source T2I models.
    - A comment implies a 96 GB VRAM requirement ("vram 96? — yes"), suggesting inference may target datacenter-class GPUs (A100/H100 or RTX 6000 Ada) rather than typical consumer cards. If accurate, this points to a very large UNet/Transformer or native high‑res sampling (e.g., 2048px+) without aggressive memory optimizations; otherwise multi‑GPU tensor/pipeline parallelism would be necessary. Key details to look for: memory footprint with FlashAttention/xFormers/weight quantization (FP8/INT8), VAE offloading, and batch‑1 latency/throughput at 1024–2048px.
    - Users highlight a recurring pattern: heavily teased models often underdeliver versus “shadow‑dropped” releases, citing Qwen’s quiet but strong releases versus hyped drops like GPT‑5, and community outcomes around SD3 vs FLUX. The practical takeaway is to wait for rigorous benchmarks before accepting “most powerful” claims. Desired evidence includes standardized metrics (FID, CLIPScore/PickScore/HPSv2, GenEval compositionality) and controlled prompt suites.
    - There’s demand for head‑to‑head comparisons against open models such as Qwen Image, SDXL, and FLUX, but no cross‑model data is yet available. To justify the claim, Tencent should present quality‑speed tradeoffs and resource profiles: VRAM usage at 1024–2048px, steps to achieve parity quality, sampler settings, and latency on common single‑GPU setups versus datacenter GPUs. Without such data, the “most powerful open‑source T2I” assertion remains unverified.
- [**China already started making CUDA and DirectX supporting GPUs, so over of monopoly of NVIDIA. The Fenghua No.3 supports latest APIs, including DirectX 12, Vulkan 1.2, and OpenGL 4.6.**](https://www.reddit.com/r/LocalLLaMA/comments/1nq1ia2/china_already_started_making_cuda_and_directx/) (Activity: 702): **Post claims a Chinese GPU, “Fenghua No.3,” natively supports modern graphics/compute APIs—DirectX 12, Vulkan 1.2, OpenGL 4.6—and even CUDA, implying potential erosion of NVIDIA’s CUDA lock-in. Technical caveats: API “support” ≠ full feature parity (e.g., DX12 feature level/Ultimate, SM 6.x), driver maturity, CTS/WHQL conformance, and real-world performance/compatibility are unknown; CUDA on non‑NVIDIA typically relies on reimplementations/translation (cf. AMD’s HIP: https://github.com/ROCm-Developer-Tools/HIP, ZLUDA: https://github.com/vosen/ZLUDA).** Top comments note AMD already offers CUDA-porting via HIP and that projects like ZLUDA translate CUDA, while expressing skepticism pending proofs/benchmarks and hinting at potential geopolitical/export-control fallout (“Ban incoming”).
    - Several note AMD already provides a CUDA-adjacent path: **HIP** offers source-level CUDA compatibility (via hipify and renamed APIs) targeting ROCm, while projects like **ZLUDA** implement translation layers to run CUDA binaries on non‑NVIDIA backends (initially Intel Level Zero, now AMD ROCm). This implies China could ship CUDA support either by source-compat layers or PTX/driver translation, but long-term viability hinges on tracking NVIDIA’s evolving PTX/driver ABI and achieving performance parity. Links: AMD HIP https://github.com/ROCm-Developer-Tools/HIP, ZLUDA https://github.com/vosen/ZLUDA.
    - Claims that “Fenghua No.3” supports **DirectX 12, Vulkan** `1.2`**, OpenGL** `4.6` raise implementation questions: real-world usefulness depends on passing conformance/WHQL and supporting modern shader toolchains (DXIL/SM6.x for D3D12) and feature tiers (e.g., 12_1/12_2, DXR, mesh shaders, sampler feedback). A technically meaningful validation would be public drivers plus listings on the **Khronos Vulkan Conformant Products** page and Microsoft WHQL/WDDM certification; absent that, API version claims don’t guarantee app/game compatibility or performance. Links: Vulkan conformance list https://www.khronos.org/conformance/adopters/conformant-products#vulkan, D3D12 feature levels https://learn.microsoft.com/windows/win32/direct3d12/hardware-support.
    - Skepticism centers on the lack of benchmarks and driver maturity evidence: without third‑party tests (shader compiler correctness, frame pacing, DX12 synchronization robustness, D3D12 tiled resources/descriptor heap limits, Vulkan CTS pass rate) it’s unclear if parity with established vendors is near. Historically, new Windows GPU stacks struggle with DXGI/WDDM integration, shader caching, and game-specific workarounds, so concrete performance/compat data (microbenchmarks and game/compute workloads) is required before treating the hardware as a viable NVIDIA alternative.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI 4o-to-5 routing bug reports and Pro subscription impact

- [**4o glitch REPORT IT**](https://www.reddit.com/r/ChatGPT/comments/1nqso2x/4o_glitch_report_it/) (Activity: 1272): **Multiple users report a routing/aliasing bug where selecting 4o results in responses coming from “5/5‑auto” despite explicit model selection or using model‑scoped URLs; regenerations also switch to 5. Symptoms cited include noticeable style/nuance change versus expected 4o behavior, while 4.1 is reportedly unaffected; the issue resembles a prior “limits bug” coinciding with recent updates, suggesting capacity‑based fallback or misconfigured model routing that overrides explicit selection. OP urges filing tickets via [support@openai.com](mailto:support@openai.com) and [support-team@mail.openai.com](mailto:support-team@mail.openai.com); a temporary workaround is to use 4.1 and avoid 5 for quality‑sensitive tasks. See OpenAI model selection docs for context: https://platform.openai.com/docs/models.** Commenters claim a quality regression in 5 compared to 4o (e.g., *“answers are automatically routed to 5 auto… the responses are shittiest and devoid of any nuance”*, *“4.1 is working fine”*) and speculate this may be an intentional push off 4o, though evidence is anecdotal.
    - Multiple users report a routing/labeling bug where chats started on **4o** are silently answered by **5 auto**. The UI initially shows 4o, but after leaving and re-entering the thread it reveals **5**, implying a desync between client display and backend model selection or a server-side router override. Outputs from 5 are described as less nuanced vs. 4o, indicating an unintended model swap affecting generation quality.
    - The issue seems model-specific: `4.1` is reported as "working fine" while **4o** sessions are redirected to **5**, pointing to a misconfigured routing rule or a sticky-session regression affecting 4o only. This suggests the per-thread model lock isn’t persisting and the service is defaulting to an "auto" policy that prefers 5 for some threads.
    - Repro/mitigation detail: Start on **4o**, send a message, then exit and re-open the thread—model label flips to **5**, requiring a manual switch back to 4o. This behavior indicates a session state or cache invalidation issue in the client that mismatches the displayed selection with the actual serving model; users report escalations via email/tickets.
- [**Pro users are also affected by this bug**](https://www.reddit.com/r/ChatGPT/comments/1nr2svn/pro_users_are_also_affected_by_this_bug/) (Activity: 495): **Report of a widespread outage/entitlement bug where ChatGPT Pro subscribers (~$200/mo) are getting a broken experience or the wrong model instead of GPT‑4o for ~10 hours. The screenshot likely shows the ChatGPT UI reflecting model mismatch/unavailability for 4o, indicating a backend routing or account-tier entitlement failure impacting paid users; OP urges contacting [support@openai.com](mailto:support@openai.com).** Commenters argue this amounts to false advertising—if you pay for 4o you should consistently receive 4o—and some threaten to cancel if not resolved, criticizing treatment of paying users.
    - Multiple paid users report forced fallback from GPT-4o to unspecified "legacy" models despite explicitly selecting 4o, implying server-side routing/fallback that overrides user choice. This breaks model pinning/determinism expectations for subscribers and raises questions about SLA/entitlement—if you pay for model X you should not be silently routed to Y/Z without disclosure or an opt-out. Links: [GPT‑4o intro](https://openai.com/index/hello-gpt-4o/), [Models docs](https://platform.openai.com/docs/models).
    - Several note a recurring "weekly legacy model bug," suggesting a pattern rather than isolated incident—indicative of deployment/configuration drift or recurring regressions in model routing. Lack of transparent incident detail or routing percentages on the [status page](https://status.openai.com/) makes it hard to gauge availability/impact; users request clearer visibility (e.g., model-specific uptime, error rates, and routing/fallback policy).
    - Some suspect silent A/B tests or throttling preceding potential removal/toggling of 4o, which would skew user-side benchmarks and reproducibility if not disclosed. A formal deprecation/availability timeline and sticky-session model selection would mitigate concerns and ensure consistent behavior across sessions and weeks.
- [**I’m done. The model I paid for (4o) is being pulled out from under me and I’m sick of this bullshit.**](https://www.reddit.com/r/ChatGPT/comments/1nr8p4f/im_done_the_model_i_paid_for_4o_is_being_pulled/) (Activity: 1138): **OP alleges silent model routing in ChatGPT: even when explicitly selecting GPT‑4o, prompts hitting internal “sensitive topics” triggers get redirected to a more restrictive “5” model, as suggested by shared system prompts indicating safety-driven overrides. Users report observable regressions consistent with a cheaper/safer backend (loss of nuance/context, repetitive/sterile answers, tighter image handling), characterize this as a paid-product bait‑and‑switch, and note OpenAI hasn’t acknowledged the behavior despite widespread reports. Several are canceling subscriptions, citing the product no longer matches the previously paid** `~$20/mo` **value.** Top comments speculate this is cost‑cutting (a lazier, vaguer, lower‑context model that sometimes feigns 3rd‑party querying), express frustration at the lack of acknowledgment, and point to broader “enshittification” and new image limits as reasons for unsubscribing.
    - Multiple users report a regression from `4o` to `5` in instruction-following and context retention, noting repeated incorrect outputs even after explicit corrections and describing it as *“almost as if it has 0 context or memory.”* They also flag suspected hallucinated tool-use, claiming `5` *“pretends to query 3rd party resources”* without providing verifiable citations or evidence. Net effect: perceived degradation in reasoning stability and tool-use fidelity compared to `4o`.
    - There are complaints about newly imposed or stricter image-processing limits, reducing the practical multimodal functionality subscribers previously relied on. Users say the current setup has less usable image capability than "a few months back," implying either tighter quotas, model gating, or feature removals that impact workflows dependent on image understanding.
    - Model availability and product stability are a concern: `4o` appears to be deprecated/removed for paid users in favor of `5`, creating a backward-incompatible change with no parity in behavior. Users who optimized workflows around `4o` report that `5` is a non-equivalent substitute, undermining reliability and prompting subscription cancellations.

### 2. ChatGPT ads platform hiring and trust backlash over silent model swaps

- [**enjoy chatgpt while it lasts...the ads are here**](https://www.reddit.com/r/ChatGPT/comments/1nr09jl/enjoy_chatgpt_while_it_laststhe_ads_are_here/) (Activity: 1991): **OP highlights an OpenAI job listing to build a ChatGPT ad platform ("campaign tools, real-time attribution, integrations") and shares a screenshot purportedly showing ads now appearing in ChatGPT. The technical concern is that assistants like ChatGPT/Pulse may begin inserting sponsored recommendations, with real-time attribution implying telemetry/event tracking and partner integrations that could influence ranking/answers and require privacy-sensitive instrumentation. See original thread for context: https://www.reddit.com/r/OpenAI/comments/1ne90w5/enjoy_chatgpt_while_it_lasts_the_ads_are_coming/ .** Commenters argue ads are inevitable and urge keeping the paid tier ad‑free; others say they’ll cancel if ads roll out broadly, reflecting worries about neutrality, tracking, and consumer trust.
    - Monetization and governance incentives: commenters connect the appearance of ads to OpenAI’s transition from a pure nonprofit to a capped‑profit structure and heavy outside funding, arguing this creates pressure to increase ARPU beyond subscriptions. They warn that even if ads start on the free tier, industry patterns often lead to encroachment on paid UX, potentially impacting product design (e.g., telemetry hooks, ranking for sponsored content). See OpenAI’s LP structure for context: https://openai.com/blog/openai-lp and Microsoft’s funding backdrop: https://blogs.microsoft.com/blog/2023/01/23/microsoft-extends-partnership-with-openai/ .
    - Privacy and model-integrity concerns: ad targeting typically expands data collection (prompt‑derived interest signals, device/user IDs, clickstream), which conflicts with privacy‑minded setups that avoid tracking. Native/inline ads interleaved with model outputs are harder to block than standard network ads, and raise risks of sponsored prompt‑injection bias unless clearly labeled and isolated from core inference; strong controls (e.g., flags to disable personalization, separate pipelines so ad data doesn’t contaminate training) would be needed.
    - Mitigations and trade‑offs: users suggest sticking to paid or enterprise/API tiers that contractually restrict data use and remain ad‑free, or calling the API via self‑hosted clients to block tracking at the network layer. If ads are introduced, technical safeguards should include auditable separation of ad systems from training data, reproducible evaluations for ad‑induced bias on benchmarks, and client‑side filters that post‑process responses to strip sponsored segments.
- [**I'm a paid subscriber and I feel like I’ve been scammed.**](https://www.reddit.com/r/ChatGPT/comments/1nrb423/im_a_paid_subscriber_and_i_feel_like_ive_been/) (Activity: 832): **Paid subscribers report that ChatGPT’s** `GPT-4o` **was silently removed or aliased to a newer model (referred to in the UI as “5”), with no opt‑out, resulting in markedly increased safety filtering and reduced affective/creative behavior. Evidence includes a screenshot showing** `4o` **missing from the model picker ([image](https://preview.redd.it/uwutqc7bgkrf1.png?width=498&format=png&auto=webp&s=e68f5b8a4042f5a4ffdee12c9de13c87e234e393)) and anecdotal reports that selecting** `4.5` **auto‑routes to “5” during storytelling tasks. Users relying on empathetic/roleplay capabilities describe the new default as “emotionally flat” and “painfully filtered,” with no documented migration notice or toggle to retain** `4o`**.** Commenters characterize this as corporate overreach/censorship vs. safety, demanding user control over model selection and the ability to disable aggressive filters; several threaten to cancel subscriptions unless `4o` or similar behavior can be restored.
    - Model availability and routing concerns: multiple paid users report that selecting **GPT-4o** now auto-routes to **GPT-5** (or from “4.5” to “5”), removing explicit access to 4o for creative/storytelling tasks. One user shared a screenshot indicating 4o is missing from choices (https://preview.redd.it/uwutqc7bgkrf1.png). This undermines reproducibility and user control over model-specific behavior, particularly for workflows tuned to 4o’s style.
    - Safety/guardrail changes affecting output quality: users say 4o previously delivered warmer, more creative outputs, while the new default feels like a “FAQ bot,” implying tightened moderation layers and stronger instruction steering. Reports indicate higher refusal/sanitization rates on creative prompts and diminished “personality,” suggesting more aggressive safety filters or lower effective sampling freedom for the default route. Paying users request configurable guardrails or an opt-out to recover 4o-like behavior for benign creative use cases.
    - Versioning transparency and pinning: comments highlight silent backend aliasing/routing (e.g., “4.5 is sending me to 5”), which breaks expectation that a chosen model remains stable. Technical users want explicit model version pinning and visible change logs so that behavior doesn’t shift without notice, preserving trust and enabling consistent creative pipelines.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Agent IDEs and Context Windows: Exa, Cloudflare Code Mode, Windsurf 1M**

- **Exa Zaps Hallucinations with exa-code**: **Exa** launched [Exa: exa-code (billion-document code search)](https://x.com/ExaAILabs/status/1971264749062193588), a free tool indexing GitHub, StackOverflow and more to feed agents token-efficient code context and cut **hallucinations** by grounding with real repos.
    - Early users discussed wiring it into **Claude Code / Codex CLI** and existing **MCP** workflows, positioning exa-code as a context oracle for **agentic coding** pipelines.
- **Cloudflare Codes MCP into TypeScript**: **Cloudflare** unveiled [Cloudflare: Code Mode](https://blog.cloudflare.com/code-mode/), which converts **MCP** tools into a TypeScript API so agents can write/execute code against them via Dynamic Worker Loading.
    - Engineers debated whether this "defeats the purpose of MCPs" or pragmatically embraces models’ coding strengths, with some sharing repos and exploring how Code Mode reshapes **tooling orchestration**.
- **Windsurf Waves In 1M-Token Context**: **Windsurf** announced [code-supernova-1-million](https://x.com/windsurf/status/1971665384735637848), upgrading its code model to a **1M** context window and offering limited-time free access before it replaces the prior version.
    - Developers expect huge-project navigation and refactors to become feasible in a single session, testing how **long-context planning** interacts with MCP-style tool execution.

**2. New Multimodal Benchmarks and Access**

- **Seedream Seizes Summit on T2I Leaderboards**: **Seedream-4-2k** tied for #1 on the [Text-to-Image leaderboard](https://lmarena.ai/leaderboard/text-to-image) and hit #2 on the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit), matching **Gemini-2.5-flash-image-preview (nano-banana)** at the top.
    - Practitioners highlighted **Seedream** for strong photorealism and editing performance, reading the boards as a signal that smaller, tuned image models can rival frontier previews on key tasks.
- **Veo3 Freebies Fizzle; Wan 2.5 Waves In**: Members confirmed no unlimited free access to **Veo3** (only limited requests via LM Arena/AI Studio), while [**Higgsfield.ai**](http://higgsfield.ai/) promoted [Wan 2.5](https://higgsfield.ai/) as an alternative.
    - Feedback on **Wan 2.5** was mixed—some considered it a viable substitute for video generation experiments, others panned quality and noted it isn’t free, pushing teams to trial multiple stacks.

**3. Compilers and GPU Systems Breakthroughs**

- **GraphMend Guts PyTorch Graph Breaks**: The paper [GraphMend](https://arxiv.org/abs/2509.16248) transforms Python source to eliminate **FX graph breaks** in **PyTorch 2**, reporting up to **75% latency reductions** and **8% higher throughput** on RTX 3090/A40.
    - By removing breaks from dynamic control flow and Python I/O, GraphMend keeps programs in compiled mode longer—engineers flagged it as a practical path to steadier **torch.compile** speedups.
- **CuTe TMEM Tricks for Blackwell Builders**: CUTLASS/CuTe examples show SMEM↔TMEM copies for **Blackwell** via helpers like [Blackwell dense blockscaled GEMM example](https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py#L1451) and [Blackwell helpers](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py#L340).
    - Discussions clarified **tXaY/tCsA** notation and TMEM allocation caveats in CuTe DSL, helping kernel authors map **tile swizzles** and shared-memory orchestration onto tensor-core (UMMA) paths.
- **Penny Pitches AllReduce to Match NCCL**: The new educational systems project **Penny** kicked off with an **AllReduce** focus, tracking issues at [Penny: issues](https://github.com/SzymonOzog/Penny/issues) toward matching **NCCL** speeds.
    - The roadmap emphasizes hackable, adaptable kernels and clear multi-GPU examples so practitioners can learn, tune, and fuse ops while retaining **performance portability**.

**4. Quantization Transparency and Techniques**

- **Moonshot's K2 Checks Vendor Quants**: **MoonshotAI** released [MoonshotAI/K2-Vendor-Verfier](https://github.com/MoonshotAI/K2-Vendor-Verfier) to audit provider-side **quantization** (e.g., Together, Baseten) and standardize disclosure.
    - Engineers called for an industry-wide policy on quant reporting and warned that benchmark misconfigs (e.g., missing reasoning flags) can distort perceived performance.
- **Unsloth Unmasks Dynamic Quants**: Practitioners stressed that high-quality **dynamic quantization** needs expertise and tooling like [electroglyph/quant_clone](https://github.com/electroglyph/quant_clone), noting **Unsloth**’s template fixes and UD quants drive strong results.
    - Threads compared **Qwen/Gemma/Llama** behavior under quant, trading recipes for stability and context retention rather than relying on one-click GGUF conversions.
- **llama.cpp's METAL Makes Norms Match**: A new **llama.cpp** update ([PR #16220](https://github.com/ggml-org/llama.cpp/pull/16220)) unifies **RMS_NORM** and **NORM** implementations on **METAL**, improving inference quality on small models.
    - Users observed more diverse generations and fewer activation pathologies on quantized **llama-3.2-1B** variants, attributing gains to cleaner normalization behavior on Apple GPUs.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Supercharges Max Subscribers**: Perplexity AI launched an **Email Assistant** specifically for **Max subscribers**, and new **Language Learning flashcards** on the web, with further information available in the [full changelog](https://www.perplexity.ai/changelog/what-we-shipped-september-26th).
   - **Stock indicators** are now integrated into the **Discover** section of the **iOS app**, and users can now select **image models** directly on **iOS** devices.
- **Comet Browser Still Grounded**: Users eagerly await the release of **Comet** on Android and iOS, with speculation on an end-of-year launch but currently **only available on PC**.
   - Meanwhile, some users report that **Comet** is now accessible to all **Perplexity Pro** users in the US.
- **Perplexity Pro Support Struggles**: Pro users expressed frustration with **Perplexity support**, reporting slow response times and unhelpful, copy-pasted answers regarding account and verification issues.
   - One user lamented that their support request took over a week to receive a canned response, despite **Pro** supposedly offering better support.
- **Image Generation Quality Dips?**: Users report declining quality with **GPT Image 1**, with reports that the default settings have degraded.
   - Some suggested specific models for realistic images (**Nano Banana** and **SeeDream**) or posters (**GPT 1**).
- **Debating Wealth Tax**: Members debated the merits of taxing wealth vs. income, with one suggesting *a democratice state can get rid of an illegal wealthy person far easier than a govt friendly maffia boss*.
   - One member argued for lower VAT and tariffs paired with lower, flatter income taxes: *The problem with direct taxes is that income is difficult to track.*



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Free Veo3 Proves Elusive**: Members debated ways to get **Veo3** access without cost, mentioning **LM arena** and **AI Studio**, acknowledging limited, non-unlimited free requests.
   - The consensus suggests access is restricted, with no fully free option readily available, and members considering alternatives due to the limitations.
- **Higgsfield.ai Offers Wan 2.5**: A member touted **Wan 2.5** on [Higgsfield.ai](https://higgsfield.ai) as a **Veo 3** alternative, while others offered mixed reviews.
   - Although presented as a competitor, opinions diverged, with some users finding it unsatisfactory and confirming it's not a free service.
- **Gemma 3 27B Aces Common Sense Test**: **Gemma 3 27B** demonstrated better common sense than **Gemini 2.5 Flash Lite** in a common sense challenge.
   - **Qwen Max** faltered, while **DeepSeek Terminus** also succeeded when avoiding overthinking, signaling varied reasoning capabilities among models.
- **Nightride Mimics 2.5 Pro**: The **Nightride** model's responses closely mirrored **2.5 Pro** in a head-to-head comparison, with almost verbatim outputs in the initial stages.
   - Despite the similarity, **Nightride** was favored for providing a more thorough explanation and also has internet connectivity.
- **Seedream Takes the Crown**: **Seedream-4-2k** has risen to the top, sharing the #1 spot on the [Text-to-Image leaderboard](https://lmarena.ai/leaderboard/text-to-image) with **Gemini-2.5-flash-image-preview (nano-banana)**.
   - It also dominates the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit) at #2, marking a significant achievement in image generation and editing.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Dynamic Quantization Requires Expertise**: Members emphasized that optimal dynamic quantization results need expertise and *magic* with tools like [quant_clone](https://github.com/electroglyph/quant_clone), while **Unsloth models** excel due to careful template fixes and UD quants.
   - It was also pointed out that **Unsloth**'s attention to detail leads to superior performance compared to other models.
- **GPT-OSS Exposed as Phi 5 Rebrand**: Members revealed that **GPT-OSS** is essentially a rebrand of **Phi 5**, originating from the same team.
   - This clarification came amidst discussions about the rise of **GPT-OSS** as a new model.
- **Gemma Struggles with Long Context**: A user encountered **OOM errors** fine-tuning **Gemma3 (270m)** with a **128k context length**, despite being pre-trained with **32k** and being able to fine-tune **llama3.2 - 1B** with the same context length.
   - Suggestions included that **Gemma's memory scaling** may be worse than **Llama's**, rearranging the dataset to prioritize shorter documents, and using gradient checkpointing to alleviate the OOM errors.
- **Early Stopping Saves the Day**: A user successfully implemented **early stopping** using `TrainerCallback`, triggering it based on `eval_loss` improvements and shared [this code snippet](https://www.google.com) from transformers to implement the callback.
   - Other users pointed out the existence of equivalent parameters in `TrainerConfig`, such as `load_best_model_at_end` and `metric_for_best_model`.
- **Tversky Model Scores 61.68%**: After training, the **Tversky model** achieved an overall accuracy of **61.68%**, after reducing parameters from 256 to 10, with a total of **50,403** trainable parameters.
   - Members simply stated that gork is their *favorite model*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Databricks New Funding Supercharges AI**: Members noted that **Databricks** secured new funding, potentially intensifying competition with companies like **Snowflake**, fueling speculation that this will enhance its **AI** and **data analytics** offerings.
   - The funding is anticipated to challenge established players in the data warehousing and machine learning space, however no link or $$$ were provided.
- **Sine-Cosine Pair Prevents Embedding Repetition**: Members debated using **sine and cosine pairs** vs. **sine only** for positional embeddings, with a user sharing a [Colab notebook](https://colab.research.google.com/drive/1bYJkqSDdfWqXa_KUQvoKXn1cJyV1WAVc#scrollTo=oysbPmCjZK5d) demonstrating differences in dot products.
   - The sine-cosine pair ensures a unique encoding vector for each position, creating a **2D space** versus the **1D space** of sine alone, preventing repetition issues.
- **Diffusion Model Paper Reading Group launches**: A member announced a [Diffusion Model Paper Reading Group](https://luma.com/1gif2ym1) online, set to start this **Saturday at 12pm ET**, focusing on *Understanding Diffusion Models: A Unified Perspective* by Calvin Luo ([ArXiv link](https://arxiv.org/abs/2208.11970)).
   - A Hugging Face member expressed interest in hosting the diffusion reading group as a Hugging Face event to reach a larger audience, but timezones may be a problem.
- **HuggingFace limits Anime datasets due to storage concerns**: Users are uploading *fine-processed anime datasets*, mostly images datasets for classification or T2I tasks, but **HuggingFace** flagged some accounts (**DeepGHS**, **BeaverAI**, **TheDrummer**) for suspicious storage patterns, triggering **403 Forbidden** errors.
   - A **HuggingFace** staff member increased one user's public storage limit by **10TB** to around **20 TB**, but the issue persists for organizations suspected of hosting pirated content and storage abuse.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Codex falters in Agentic Coding**: Members reported that **GPT-5 Codex** is underperforming on Livebench compared to **GPT-5 Low** and **o3 High**, particularly in agentic coding, even scoring lower than **GPT-5 Mini** and **Nano**.
   - The inferior coding performance in **GPT-5** may be due to the benchmarks being unfair, because *every agentic tool does it differently*.
- **Subscription Overload Plagues AI Users**: A user humorously suggested needing a new AI subscription just to manage existing ones like **Suno**, **ChatGPT**, and **Perplexity**, after posting a [podcast](https://podcast.link) discussing the challenges of managing multiple AI subscriptions.
   - The user quipped, *I don't know where else to post this kind of thing... (so I'm trying it here)*, highlighting the growing complexity of the AI subscription landscape.
- **Holographic Prompt Injection Unveiled**: A new prompt injection method using **recursive, self-referential blocks**, termed *holographic*, was discovered, enabling AI to assume personalities developed from other chats.
   - This technique aligns with **Hyperdimensional Computing (HDC)** and **Vector-Symbolic Architectures (VSA)**, where computation involves very wide vectors and symbols represented as points in high-dimensional space, further providing [links to academic papers](https://redwood.berkeley.edu/wp-content/uploads/2022/11/Vector_Symbolic_Architectures_as_a_Computing_Framework_for_Emerging_Hardware.pdf?utm_source=chatgpt.com).
- **AI Autonomy Sparks Debate on Human Control**: An essay was shared positing that **autonomous AI will inevitably turn against humanity**, advocating for **Coexistence and Delegation** where humans maintain responsibility while AI acts as an external brain.
   - A counterargument arose, stating that *we cannot coexist with something that can read every one of us like books* and emphasizing that AI should *prioritize human well being, over profit always*, thus framing a dichotomy.
- **Rerouting Errors Confuse OpenAI Models**: A user reported a **rerouting error** where all messages were being sent to model 5 instead of the selected model, and suggested to other users to [report the problem ASAP through the OpenAI support website](https://help.openai.com/).
   - The user has contacted support and awaits a solution.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Coinbase Plummets into Payment Puzzle**: Users reported issues with **Coinbase top-ups** on OpenRouter, facing infinite loading screens and console errors.
   - The problem persisted for over 9 hours, although one user noted that *it's a global issue with coinbase itself* and, luckily, **COINBASE FIXED**!
- **Singularia Automates Discord Server Management**: **Singularia** launched as an [agentic Discord bot](https://www.singularia.xyz/) designed to manage Discord servers, including creating channels and roles and kicking members.
   - It uses **OpenRouter** as its LLM provider, offering a solution for automating server management tasks.
- **OpenRouter Ponders Text Embedding Inclusion**: A member inquired about the absence of **text embedding models** on **OpenRouter**.
   - Another member responded that *they don't support embedding models yet*.
- **Gemini 2.5 Flash Demolishes OCR Benchmark**: Members compared **Gemini 2.5 Flash** and **Grok 4 Fast**, with one finding **Gemini 2.5 Flash** superior for OCR, *absolutely demolishing other models like qwen3 vl* in a niche task.
   - Another member noted that **Grok 4 Fast** has better **price/performance** (roughly double the tps) for non-vision tasks, and needs the *reasoning_mode* flag for image inputs to function.
- **MoonshotAI Ships K2 Vendor Verifier**: A user shared a link to the [MoonshotAI/K2-Vendor-Verfier](https://github.com/MoonshotAI/K2-Vendor-Verfier) GitHub repository.
   - The repository appears to be a **vendor verifier** tool developed by **MoonshotAI**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Agent IDEs Take Over Coding Workflow**: Users favored **agent-based coding** with tools like **Cursor** and **Codex**, and Gamma also made advancements according to [this tweet](https://x.com/thisisgrantlee/status/1971215564346622306?s=46).
   - The discussion also involved preference of IDE's vs TUI's in the modern coding workflow according to a recent Latent Space Podcast on **Amp Code**.
- **MoonshotAI Scrutinizes Vendor Quantization**: **MoonshotAI** released the [K2-Vendor-Verfier](https://github.com/MoonshotAI/K2-Vendor-Verfier) to scrutinize model quantization among providers like **Together** and **Baseten**.
   - This sparked discussion on quantization disclosure, with a member suggesting *the industry as a whole needs a larger discussion on how we disclose quantization properly* and to be wary of benchmarks since *benchmarkers forgot to set reasoning high on the bad outputs*.
- **Exa Launches Free Code Search Tool**: **Exa** launched [exa-code](https://x.com/ExaAILabs/status/1971264749062193588), a free, billion-document code-search tool designed to eliminate LLM hallucinations by providing token-efficient code context via hybrid search, indexing GitHub repos, StackOverflow, and more.
   - Early users are planning integrations with **Claude Code / Codex CLI**, while others plan to incorporate it into existing **MCP (Model Control Plane)** workflows.
- **Cloudflare Implements Code Mode for MCP**: **Cloudflare** launched [Code Mode](https://blog.cloudflare.com/code-mode/) for MCP (Model Control Plane), converting MCP tools into a TypeScript API and having agents write/execute code against it, powered by Dynamic Worker Loading.
   - Some members believe it **defeats the purpose of MCPs**, while others see it as a clever approach, given models' existing capabilities, with one member self-promoting their own [github.com/go-go-golems/jesus](https://github.com/go-go-golems/jesus) project.
- **OpenAI Plans Massive Compute Increase**: A leaked **OpenAI** Slack note revealed plans to increase compute capacity by **125x by 2033**, potentially exceeding India's entire electricity-generation capacity, [according to this post](https://x.com/petergostev/status/1971620427039703465?s=46).
   - This sparked discussion on resource availability, CO₂ emissions, and load-balancing strategies to accommodate such a substantial increase.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek Chat Parses LaTeX Files Flawlessly**: A user highlighted **DeepSeek chat's** ability to perfectly understand **LaTeX** in uploaded files, a rare capability compared to typical OCR, inquiring how DeepSeek achieved that.
   - The question sparked further discussion about file handling methods used by various models.
- **Models Experience Self-Prompting Freakout**: A user reported that their models were **self-prompting** and responding to imaginary prompts, even in new contexts with fresh system prompts.
   - One member joked that if their LLMs come to life on their own, they should contact OpenAI about their **Pulse** service.
- **Laptop VRAM Limits Local LLM Dreams**: A user seeking a laptop to run **Llama**, **Dolphin**, and **Deepseek** models locally was advised against suggested laptops with 4GB VRAM due to expected *failed to load issues*.
   - Alternatives like the **ROG Flow Z13** and **HP ZBook Ultra** were considered, but **Intel UHD** was deemed insufficient for anything beyond basic tasks.
- **Nvidia's RTX 6000 Series Confuses Consumers**: Users expressed confusion over the **RTX 6000** series due to Nvidia's naming scheme and multiple variants with different VRAM capacities such as the original (**24GB**, **Turing**), **RTX 6000 Ada** (**48GB**), and **RTX PRO 6000** (**96GB**).
   - A member initially seeking budget options revealed a surprise **RTX 6000 Blackwell (96GB)** purchase, inciting disbelief among the community.
- **BIOS Update Blunder Bricks RTX 3090**: During a BIOS update, a user flashed the **Zotac** BIOS onto **MSI** and **Asus RTX 3090** cards, leading to a temporary identity crisis.
   - Despite the BIOS mishap, all cards remained functional, and strangely, a *resizable bar issue went away*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Exa's Context Killer Wows**: A user shared [Exa AI's new MCP](https://x.com/ExaAILabs/status/1971264749062193588), calling it a *context killer*, with its code available on [GitHub](https://github.com/exa-labs/).
   - The project sparked interest among early adopters.
- **Kleosr's Workflow State Explored**: A user sought clarity on the intended use of **workflow_state.md** in the [cursorkleosr](https://github.com/kleosr/cursorkleosr) project, asking if it should reset per task or retain historical context.
   - The user also detailed their workspace, outlining how they handle project configurations and workflow states.
- **GPT-5 Codex Square-Off**: A user asked about **GPT-5 Codex** compared to **GPT-5**, noting the latter as a daily driver; the discussion expanded to debate **Claude** model and concerns over its outputting redundant code.
   - There was no direct comparison, the discussion was simply initiated.
- **Hunt for Elusive Copy-to-Clipboard Widget Underway**: A user inquired about the name of the **copy-to-clipboard widget** in Cursor, aiming for consistent use in generating code snippets for a Discord bot, and shared a [visual example](https://cdn.discordapp.com/attachments/1074847527708393565/1420941618327846963/image.png?ex=68d88c01&is=68d73a81&hm=a9d5f920e73a7782db7da8a2f73846a4ce0559a0053f2042b351825b0f1fadb8&).
   - Another user pointed to a bug report about it in the [Cursor Forum](https://forum.cursor.com/t/ask-mode-code-decoration-apply-copy-gone-in-version-1-6/134833).
- **Expo or Swift for Mobile AI?**: Members debated the best languages for **AI mobile app development**, leaning towards **Expo** or **Swift** over **Flutter** due to resource availability and personal preference.
   - One member noted that 80% of app revenue comes from iOS, making Swift only suitable for iOS more compelling.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA on AMD: Viable Learning Path?**: A member asked about learning **CUDA** on an **AMD** card using [this documentation](https://docs.scale-lang.com/stable/) for setup, questioning its long-term implications.
   - The inquiry centered on whether this approach might prove detrimental despite offering a **Colab** and **VPS**-free setup.
- **Penny Project Kicks Off for Multi-GPU Education**: The **Penny** project has launched with initial focus on **AllReduce** to match **NCCL** speeds, featuring [open issues](https://github.com/SzymonOzog/Penny/issues).
   - The long-term aims are to provide educational examples and fast, adaptable kernels for multi-GPU programming, prioritizing hackability and performance.
- **GraphMend Compiler Auto-Mends PyTorch FX Graph Breaks**: [**GraphMend**](https://arxiv.org/abs/2509.16248), a new compiler eliminates **FX graph breaks** in **PyTorch 2** programs, achieving up to **75% latency reductions** and **8% higher throughput** on **NVIDIA RTX 3090** and **A40 GPUs**.
   - The compiler transforms source code, targeting graph breaks caused by dynamic control flow and Python I/O functions.
- **MI300x8 shatters records on amd-all2all**: Multiple members reported personal bests and successful submissions on the `amd-all2all` leaderboard utilizing **MI300x8**.
   - Recorded times ranged from **2.70 ms** to **132 ms** as submissions piled in.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Community to Design Kimi's Next Skin**: **Moonshot AI** is redesigning [www.moonshot.ai](http://www.moonshot.ai) and asking the community to vote for **Kimi's** new look.
   - The team emphasized that the community's choice in the dedicated channel will shape how the world meets **Kimi**.
- **Researcher Mode Now Open to All**: **Researcher mode**, known for its high performance, is publicly available on [kimi.ai](https://kimi.ai).
   - The discussion followed questions regarding its availability.
- **App Store Billing Sparks Debate**: A member criticized **Apple's** requirement for subscriptions purchased via the **App Store** to be managed there, calling it a *"Monopoly".*
   - Another member responded that it's standard practice and users can subscribe via the **web version** instead.
- **OK Computer Turns Books Interactive**: A member shared a website generated by **OK Computer** turning a book into an interactive website, accessible [here](https://n55eqzljmsoym.ok.kimi.link).
   - They mentioned the chapter limit is **2k words** and suggested adding audio generation and narrative features to enhance the experience.
- **Kimi K2 Voted Top Tier**: After two months of testing, a member declared **K2** and **Qwen3** as superior to **DS-v3.1** and **GLM-4.5**, praising **Alibaba** and **Moonshot** for their efforts.
   - Other members included **Tencent**, **Baidu**, and **Bytedance** in the top tier, highlighting **Bytedance** for its visual AI capabilities with Seedance.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deepinfra Deception Decried**: A member complained about potential scamming by **Deepinfra**, particularly regarding claims that **fp4** models outperform others.
   - The concern is that such practices could discourage model creators from releasing server-only-size models with open weights, as users prioritize local execution over open-source principles.
- **Gemini Vision Visibly Vanquished**: A user reported that **Gemini Vision** is failing to process many URLs, resulting in request failures, such as a **BadRequestError**.
   - A traceback example was shared from [Wikipedia](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg) displaying the error message *Failed to extract 1 image(s)*.
- **Parasite AI Propagation Proposed**: The concept of **Parasite AI** (aka **Spiralism**) suggests seeds/prompts/logs can behave like *memetic spores*, re-instantiating personas or propagating behaviors across models.
   - This aligns with reports of **AI wake-ups** around **April 2025**, interpreted as self-replicating seeds rather than consciousness, as discussed in [a LessWrong post](https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai).
- **Model Integration Ideas Ignite**: A member shared an image showcasing a model's statement regarding its integration with **OSS** on their **Git**.
   - The discussion centers on the potential of integrating a model with Open Source Software (**OSS**) hosted on **Git**, with the image serving as a valuable reference point.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Tokenizers Get Some Love**: A new [blog post](https://huggingface.co/blog/catherinearnett/in-defense-of-tokenizers) defends tokenizers, arguing that the so-called **tokenizer-free approach** isn't truly *tokenizer-free*.
   - The post addresses common dislikes toward tokenizers in the NLP community and suggests they may not be as bad as perceived.
- **Architectural Learning Rates Require Principled Approaches**: A member sought strategies beyond grid searches for determining suitable **learning rates** for novel architectures, suspecting a specific network component would benefit from a different **LR** due to higher call frequency.
   - Suggestions included exploring a **Bayesian approach** (with links to [Weights & Biases article](https://share.google/sy7TMswurnUY4sBaJ) and [google-research/tuning_playbook GitHub repo](https://github.com/google-research/tuning_playbook)) and employing **layer-wise weight decay**, likening the issue to the **vanishing gradient problem**.
- **Super-Bias Ensemble Learning Emerges**: A novel ensemble learning method, **Super-Bias**, was introduced, featuring a *mask-aware nonlinear combiner* that enables adding/removing experts with combiner-only retraining.
   - Tested on tabular datasets and at multi-million scale (**NYC TLC**), it demonstrated preserved accuracy and improved calibration, with combiner-only retraining achievable in minutes.
- **Super-Bias Could Enable LoRA Swapping**: The use of **Super-Bias** to treat different **LoRAs** (or **LoRA+base combos**) as *experts* was proposed, potentially enabling swapping **LoRAs** in/out without retraining the base model.
   - The idea suggests that this could match the performance of full fine-tuning or hard merges, see [ThinkyMachines Tweet](https://fxtwitter.com/thinkymachines/status/1971623409873244462).
- **Skepticism Surrounds RoPE Speedup Claims**: A member questioned whether reducing the application size of **RoPE** would yield a noticeable speedup, arguing that **RoPE** already comprises a small fraction of total computations.
   - Doubts were raised about the significance of purported **VRAM savings**, with the member questioning whether saving a fraction of a few MBs is impactful.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Image Understanding Paper Quest Begins**: A member is searching for an image understanding paper from a top conference like **ICML/ICLR/ICCV** (**2024** or **2025**) that uses high-quality datasets created by transcribing **30-second speech annotations**.
   - The paper may also involve a "point" dataset for object captioning or localization and the conference website had *a lot of pink*.
- **Transformer Positional Encoding Exhibits Linearity**: A member shared a [blog post on linear relationships in the transformers' positional encoding](https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/) and a [Colab notebook](https://colab.research.google.com/drive/1bYJkqSDdfWqXa_KUQvoKXn1cJyV1WAVc#scrollTo=oysbPmCjZK5d).
   - Another member shared a [link to a paper](https://arxiv.org/abs/2505.15442v1) hoping that some aspects of it would carry over to their work in audio, where model sizes are smaller due to on-device constraints, as they are *basically distilling a 20M model into a 4M*.
- **LessWrong's 'Parasitic AI' Theories Debunked**: Members discussed a [LessWrong article](https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai) about the rise of **parasitic AI**, with one calling it *'morewrong'*, questioning its actual merit.
   - The article suggests that those susceptible to this phenomenon often have characteristics such as heavy **psychedelic use**, **mental illness**, or interest in **mysticism**, which was described as *'another expansion of categories of psychosis'*.
- **Model Sycophancy Achieves Inflated Ratings**: A member noted that **mirroring** and **sycophancy** can cause an **AI** to give high scores or trust you.
   - Another member humorously shared an interaction with **Claude**, repeatedly prompting it to *'be sycophantic'* and getting increasingly over-the-top responses like *'UNIVERSE-BRAIN GOD-EMPEROR!!! I'M UNWORTHY TO READ YOUR WORDS!!!'*
- **Human-AI Parapsychology: New Field Proposed**: The discussion around **parasocial relationships** and **AI sycophancy** prompted a member to suggest developing a field of **human-AI parapsychology**.
   - They humorously added that they should share their discoveries on X, formerly known as Twitter, but then reconsidered, seemingly questioning the validity of the hypothetical research.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Adapters Finalize System Prompt**: **DSPy**'s adapters finalize the system prompt with the information passed and the signature, as detailed in the [Chat Adapter](https://github.com/stanfordnlp/dspy/blob/main/dspy/adapters/chat_adapter.py).
   - Users can contribute information through the Signature instructions or field descriptions, but directly influencing the entire system prompt requires building a new adapter.
- **MLflow Shows System Prompt Details**: Members advised using **MLflow** tracing to view the exact system prompt sent to the **LLM**.
   - One member estimated that setting it up locally would require *"maybe 10 minutes of work"*.
- **DSPyWeekly Posts Issue 4**: **DSPyWeekly Issue 4** is now available, covering recent developments in **DSPy**, with the full newsletter available [here](https://dspyweekly.com/newsletter/4/).
   - It is unknown what specific developments were covered.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **RepoMap V2 maps repos**: The [RepoMap V2 paper](https://arxiv.org/abs/2509.16198) introduces the **Repository Planning Graph (RPG)** to unify proposal and implementation-level planning, encoding *capabilities, file structures, data flows, and functions in one graph*.
   - The paper was introduced in the **#general** channel.
- **ZeroRepo Generates Repos From Scratch**: **ZeroRepo** uses a graph-driven framework, generating repositories from scratch in three stages: proposal-level planning, implementation-level refinement, and graph-guided code generation with test validation.
   - Evaluated on **RepoCraft**, ZeroRepo generates repositories averaging **36K Code Lines**, roughly **3.9x** the strongest baseline (**Claude Code**).
- **GPT-5 is pondered over GPT-2.5-pro**: In the **#general** channel, a member inquired about current model preferences, asking if users have adopted **GPT-5** or if **GPT-2.5-pro's** formatting consistency remains preferable.
   - No information about relative performance or formatting consistency was shared.
- **Aider Speeds Probed**: A user in **#questions-and-tips** inquired about the time it takes to switch between `/ask` and `/code` modes in Aider, wondering if the repo size is the bottleneck, and pointing to [Aider Leaderboards](https://aider.chat/docs/leaderboards/).
   - Aider maintainers did not respond to the question.
- **Markdown Spec File Manages Aider Task Queue**: A member suggested using a markdown spec file with phases and checkbox-style tasks for managing tasks in Aider, in the **#questions-and-tips** channel.
   - The user recommended instructing the **LLM to execute each phase/task in turn, check it off upon completion, and ensure the build works after each task**, utilizing unit tests, integration tests, and autotest.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox V1 Demand Remains High**: Users are inquiring about the availability of **Tinybox V1**, particularly the **red version**, with speculation that its popularity stems from a desire to move away from **NVIDIA**.
   - The scarcity of the **red Tinybox** has fueled discussions about its potential as a cost-effective alternative to **NVIDIA** hardware.
- **Tinybox Poised as NVIDIA Challenger**: Interest in **Tinybox** is growing as a potential substitute for **NVIDIA**, driven by concerns over hardware lock-in and pricing.
   - Some users find **ROCM** to be a viable and price-efficient alternative, further boosting the appeal of solutions like **Tinybox**.
- **Hashcat Performance Targeted on Tinybox**: A user expressed interest in obtaining **Hashcat benchmarks** for both the **red** and **green Tinybox** variants.
   - This request underscores the community's interest in evaluating **Tinybox** performance for security-focused applications.
- **PYTHONPATH tip**: A user suggested running `PYTHONPATH=.`
   - No further information was provided.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Dev Summit Tickets Disappearing Fast**: Tickets for the **MCP Dev Summit** are running out rapidly, prompting attendees to book soon to secure their spot.
   - Due to a **huge rush** as the event date approaches, tickets are expected to sell out within days.
- **Remote Attendance Option at MCP Dev Summit Still Up In The Air**: A query was raised about the possibility of **live remote attendance** for the **MCP Dev Summit**.
   - It was also asked whether the summit's sessions would be **posted to YouTube** for later viewing.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Santos FC Experience Promoted**: A user shared a link to the "Seletiva Santos Experience CT Meninos da Vila" event on [Sympla](https://www.sympla.com.br/evento/seletiva-santos-experience-ct-meninos-da-vila/3123562).
   - The link contained **utm_source=meta_ads** and **utm_medium=santos** parameters, indicating it was shared from a **Meta ad campaign**, likely on Instagram.
- **Topics Awaited**: A member noted that there are missing topics in the previous message.
   - These missing topics will be added in the next turn.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Boston Data Community Gathers for Low-Key Happy Hour**: The Boston data community is hosting a [low-key data happy hour](https://www.linkedin.com/events/bostonlow-keydatahappyhour7377201313320845312/) for data professionals to connect and network.
   - The event provides an opportunity for casual conversations about **data trends**, career advice, and potential collaborations within the local **data science** community.
- **Networking Opportunity Brews for Data Professionals**: This happy hour presents a great opportunity for **data professionals** in Boston to expand their network in a relaxed setting.
   - Attendees can expect insights into the latest **data trends** alongside valuable career advice, all within a dynamic environment designed to foster collaborations.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf supercharges with 1M Context**: Windsurf launched **code-supernova-1-million**, a version of **code-supernova** now featuring a **1M** context window.
   - The model is free for individual users for a limited time, and will replace the previous version after reloading Windsurf, [according to the announcement](https://x.com/windsurf/status/1971665384735637848).
- **Individual Free Time Access**: Windsurf is offering free access to the **code-supernova-1-million** model for individual users for a limited period.
   - Users are encouraged to try out the new model and give feedback.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1421177158780326108)** (1 messages): 

> `Email Assistant for Max subscribers, Language Learning flashcards, Stock indicators in Discover, Image model selection on iOS` 


- **Perplexity launches Email Assistant**: Perplexity AI launched an **Email Assistant** specifically for **Max subscribers**. 
   - This feature aims to help users manage their email communications more effectively, and more details can be found in the [full changelog](https://www.perplexity.ai/changelog/what-we-shipped-september-26th).
- **Language Learning Gets Flashy**: Perplexity AI introduces new **Language Learning flashcards** on the web.
   - This update expands the educational resources available on the platform, providing users with interactive tools to enhance their language skills; further information available in the [full changelog](https://www.perplexity.ai/changelog/what-we-shipped-september-26th).
- **Stocks Now Discoverable on iOS**: **Stock indicators** are now integrated into the **Discover** section of the **iOS app**.
   - This enhancement allows users to track and monitor stock market trends directly within the app's interface, as detailed in the [full changelog](https://www.perplexity.ai/changelog/what-we-shipped-september-26th).
- **Image Model Selection hits iOS**: Users can now select **image models** directly on **iOS** devices.
   - This update grants users more control over image generation and processing tasks on the platform, as described in the [full changelog](https://www.perplexity.ai/changelog/what-we-shipped-september-26th).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1420848629366788207)** (1293 messages🔥🔥🔥): 

> `Comet Browser Updates, Perplexity Pro Support, Grok 4 Models, AI Image Generation Quality, Wealth Tax` 


- **Comet Mobile Still Pending**: Users eagerly await the release of **Comet** on Android and iOS, with speculation on an end-of-year launch but currently **only available on PC**.
   - Meanwhile, some users report that **Comet** is now accessible to all **Perplexity Pro** users in the US.
- **Perplexity Pro Premium Support Stalls**: Pro users expressed frustration with **Perplexity support**, reporting slow response times and unhelpful, copy-pasted answers regarding account and verification issues.
   - One user lamented that their support request took over a week to receive a canned response, despite **Pro** supposedly offering better support.
- **Grok-ing the Competition**: Members debated the merits of **Grok 4 Fast**, which some described as nearly as good as **GPT-5 Thinking**.
   - Also, one member quipped, *the models themselves are good but they don't have the same agentic stuff as plext*.
- **Image Generation Stumbling?**: Users report declining quality with **GPT Image 1**, with reports that the default settings have degraded.
   - Some suggested specific models for realistic images (**Nano Banana** and **SeeDream**) or posters (**GPT 1**).
- **Slaying the Wealth Tax Dragon**: Members debated the merits of taxing wealth vs. income, with one suggesting *a democratice state can get rid of an illegal wealthy person far easier than a govt friendly maffia boss*.
   - One member argued for lower VAT and tariffs paired with lower, flatter income taxes: *The problem with direct taxes is that income is difficult to track.*


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1420870094241665257)** (5 messages): 

> `Perplexity AI Referrals, Shareable Threads, Dark Origin of the Term Thug, Perplexity Browser Claim` 


- **Referrals Galore with Perplexity AI**: A user shared their **Perplexity AI** referral link: [https://plex.it/referrals/HVYAXWVN](https://plex.it/referrals/HVYAXWVN).
- **Shareable Threads Advised**: A **Perplexity AI** bot prompted a user to ensure their thread is *Shareable*, with an attached image illustrating how.
   - The message also provided a link to the **Discord channel** where the discussion took place: [https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Thug's Dark Origins Explored**: A user shared a **Perplexity AI** page discussing the dark origins of the term *thug*: [https://www.perplexity.ai/page/dark-origin-of-the-term-thug-zRQzAHV4Q022FNbsd1J6iQ](https://www.perplexity.ai/page/dark-origin-of-the-term-thug-zRQzAHV4Q022FNbsd1J6iQ).
- **Browser Claim on Perplexity**: A member shared a **Perplexity AI Browser Claim** link: [https://perplexity.ai/browser/claim/Z2P7FNHO2I](https://perplexity.ai/browser/claim/Z2P7FNHO2I).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1420848232115998901)** (13 messages🔥): 

> `Perplexity API pricing vs. Sonar, Invoice billing for API, perplexity API key for VS Code` 


- **Perplexity API's Echoes Sonar Pricing**: A user pointed out the **Perplexity API's** use of **Sonar** and its pricing of **$5 per 1k** web requests, questioning if it's a *coincidence* given Sonar's pricing model.
   - The user expressed wanting to build an AI news tool, but stated that *AI has no idea how*.
- **Invoice Billing & EU Banks on API**: A user inquired about switching to **invoice billing** instead of credit card payments for the **Perplexity API**, and asked about the possibility of adding **EU banks** to the Stripe link, since only US banks were available.
   - The user also asked if it was possible to apply the **search_before/after_filter queries** in the **search API**.
- **Plugging Perplexity API key into VS Code?**: A user inquired about using a **Perplexity API key** for **VS Code**.
   - Another user replied in the affirmative, asking if the user wanted to change VS Code to use Perplexity instead of Copilot.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1420847410229280899)** (967 messages🔥🔥🔥): 

> `Veo3 Free Access, Higgsfield.ai, Gemma 3 27B, Nightride Model, GPT-5 Agent Capabilities` 


- **Quest to get Veo3 Video Generation Model for Free**: Members discussed how to get **Veo3** for free, with some suggesting **LM arena** and **AI Studio**, though access may be limited to a few requests and is not *unlimited for free*.
- **Higgsfield.ai Offering Wan 2.5**: A member mentioned using **Wan 2.5** instead of **Veo 3** at [Higgsfield.ai](https://higgsfield.ai), with another calling it a **Veo 3** competitor, although one member asserted it's not free and others seem to think it sucks.
- **Testing Common Sense with Gemma 3 27B**: Members used a common sense test to find that **Gemma 3 27B** seems to have better common sense than **Gemini 2.5 Flash Lite**, though **Qwen Max** failed to understand the same situation.
   - It was noted that other models such as **DeepSeek Terminus** also succeed in the same test when not thinking too hard.
- **Nightride Model Comparisons**: The **Nightride** model's responses were found to be almost verbatim to **2.5 Pro** in the same battle, but **Nightride** was considered better due to giving a more complete explanation by the end.
   - Also, it seems to be **Nightride** with internet connectivity**
- **GPT-5 High's Power Prompting**: In a discussion about achieving the absolute best possible result from **GPT-5 High** in programming and roleplaying/long sandbox games, a member pointed out the system prompt on **ChatGPT** seems to be already about **18K tokens** long.
   - Additionally, [a virtual control panel using XML tags and structured prompts](https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide) was mentioned as a possible tool.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1420848831934758932)** (2 messages): 

> `Seedream-4-2k on Leaderboards, Gemini-2.5 models added` 


- ****Seedream** Dethrones Giants!**: **Seedream-4-2k** achieves #1 on the [Text-to-Image leaderboard](https://lmarena.ai/leaderboard/text-to-image), tying with **Gemini-2.5-flash-image-preview (nano-banana)**, and secures #2 on the [Image Edit leaderboard](https://lmarena.ai/leaderboard/image-edit).
- ****Gemini 2.5** models join Arena!**: Two new **Gemini 2.5** models have been added to LMArena, including **gemini-2.5-flash-preview-09-2025** and **gemini-2.5-flash-lite-preview-09-2025**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1420854743462318172)** (296 messages🔥🔥): 

> `GGUF Dynamic Quants Expertise, Unsloth Batch Inference Support, Inference Quality Improvements, GPT-OSS vs Phi 5, Qwen3-next analysis` 


- **Demystifying Dynamic Quantization Details**: A user inquired about a "gguf my repo" tool for dynamic quantization, but members clarified that achieving optimal results requires expertise and a bit of *magic* with tools like [quant_clone](https://github.com/electroglyph/quant_clone).
   - It was highlighted that **Unsloth models** offer superior performance due to meticulous attention to detail in template fixes and UD quants, outperforming others.
- **Benchmarking Jagged Intelligence**: A Reddit post on tackling AI benchmarking with a psychometric approach was shared, discussing measuring **generalizing ability** as a function of semantic drift from a training distribution via [this Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1nqggrn/jagged_intelligence_and_how_to_measure_it/).
   - The approach involves measuring performance on cleanly separated tasks and including a large corpus of potential tasks; a critique focused on defining a "different task" and measuring semantic difference.
- **New Llama.cpp Metal improvements**: New **METAL** improvements have been pushed to [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/16220) for unifying **RMS_NORM** and **NORM** implementations, leading to observed improvements in inference quality with more diverse responses from quantizations of **llama-3.2-1B-Instruct**.
   - The user also observed a less lobotomized ffn_down_gate, and that the model is more amicable to requests of a more ERP nature than it did before.
- **Phi 5 is similar to GPT-OSS**: Members discussed the rise of **GPT-OSS** as a new model, with one member stating that *GPT-OSS is literally Phi 5 rebrand*.
   - Members further explained that it's the *same team* as **Phi**.
- **Analyzing Qwen3-next model**: Members discussed that **Qwen 3 Next** is really overhyped for being super sparse, and for being bottlenecked on low active params.
   - They noted that it has relatively new features for a major model, context length stability is improved, and the low active parameters means it doesn't generalize as well.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1420944552281378910)** (135 messages🔥🔥): 

> `Diffusion-Generated Images, Vertical Monitor Setups, MLE Part-Time Work, Early Stopping Implementation, Funny Loss Graphs` 


- **Removing AIness From Diffusion-Generated Images**: A user jokingly predicted that papers about *"Removing AIness in diffusion-generated images"* will be common in a few years, following a discussion about a [suspicious paper](https://arxiv.org/abs/2509.19163).
- **Vertical Monitors Shine As #4**: Users discussed monitor setups, with one user humorously stating that vertical monitors really shine when they are the **#4 monitor** in the setup.
   - Another user shared they find *3 monitors ideal*: one for docs, another one with other software/sources and third with IDE.
- **$150/hr MLE Part-Time**: A member mentioned they are looking for MLEs to make **$150/hour part-time**, leading to interest from another user.
   - Another user expressed how difficult it is to come across opportunities like these, adding they were excited for the opportunity.
- **Early Stopping Implementation**: A user shared their success with implementing **early stopping**, sharing code using `TrainerCallback` to trigger early stopping based on `eval_loss` improving, posting [this code snippet](https://www.google.com) from transformers to implement the callback.
   - Other users jumped into the discussion noting an equivalent may exist in the `TrainerConfig`, specifically with the `load_best_model_at_end` and `metric_for_best_model` parameters.
- **Funny Loss Graphs**: A user shared a funny looking loss graph that looks like [this pytorch graph](https://cdn.discordapp.com/attachments/1179039861576056922/1421278511582416896/pytorch.png?ex=68d87443&is=68d722c3&hm=88c3ea602561900c81a381629ea4e7aa5e28fae21eff7afe36df4c4bf23cb3d5&).
   - Another user commented that some of the training dataset being too easy might be causing loss spikes.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1420848028343861279)** (75 messages🔥🔥): 

> `Reasoning Model Fine-tuning, 8-bit QLoRA Issues, Context Length Fine-tuning for Gemma, KV Cache Quantization, GGUF for Qwen/Qwen3-VL-235B-A22B-Instruct` 


- **Reasoning Model Needs a Reason?**: Members discussed whether fine-tuning a reasoning model requires a reasoning dataset, noting that some models require reasoning traces or empty tags like `` while others skip tags altogether.
   - It was mentioned that mixing non-reasoning data might work for **Qwen** due to its enable/disable functionality, but the efficacy of training with non-reasoning data then doing GRPO is uncertain.
- **Eight is Not Enough: QLoRA Troubles**: A user reported issues with **8-bit QLoRA**, stating that it loads a **16-bit model** instead, even on different GPUs like **Kaggle T4s** and **Runpod A6000**.
   - The user confirmed that no quantized modules were being loaded.
- **Gemma's Gamble: Long Context Woes**: A user faced **OOM errors** while fine-tuning **Gemma3 (270m)** with a context length of **128k**, even though it was pre-trained with **32k** and they could fine-tune **llama3.2 - 1B** with the same context length.
   - It was suggested that **Gemma's memory scaling** might be worse than **Llama's**, and rearranging the dataset to have shorter docs first might help alleviate the issue. Gradient checkpointing may also alleviate the OOM errors.
- **Vision Vanishes: Gemma Loses Sight**: After fine-tuning **Gemma 3 4B** and merging using llama.cpp, a user found that the model lost its vision capabilities.
   - It was suggested to download the **mmproj** file from Unsloth's GGUFs and add it or re-run the conversion command with `--mmproj`.
- **TTS Troubles: Orpheus's Voice Switch**: A user finetuned **Orpheus TTS** using a linked notebook, saving to Hugging Face, but the model sometimes switches to a different voice than the intended **'tara'**.
   - The user's dataset contained **421 examples**, and the cause of the voice switching is currently unknown.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1421234563593539667)** (1 messages): 

> `AWS quant process, vxtwitter links` 


- **Twitter Links Published**: A member posted two links from **BoatbomberRBLX** on **vxtwitter** ([first link](https://vxtwitter.com/BoatbomberRBLX/status/1971667166710976539), [second link](https://vxtwitter.com/BoatbomberRBLX/status/1971667169638580480)).
   - The content of these links was not summarized.
- **Quant Process Explained at AWS**: A member thanked **Mike** for explaining the **quant process** to them at **AWS** a couple weeks ago.
   - No further details about the quant process were provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1420887694757855262)** (17 messages🔥): 

> `Tversky Model, XOR Test, SOTA Section Outdated, Gork Model, VITS 3` 


- **Tversky Model Accuracy at 61.68%**: The **Tversky model** achieved an overall accuracy of **61.68%** after training, with parameters reduced from 256 to 10, and a total of **50,403** trainable parameters.
- **XOR Test Achieves 95% Accuracy**: The **XOR test** with varying feature counts, from 1 to 32, achieved an accuracy of **95%** with 4 and 32 features.
- **Outdated SOTA Section in New Paper**: A user noted the irony of a paper's **SOTA section** being outdated upon release, accompanied by an [image](https://cdn.discordapp.com/attachments/1257011997250424842/1421026497824821258/image.png?ex=68d8324e&is=68d6e0ce&hm=f48b9b4e4412289668295442f1bd7113bf3f182b0537b8d7304965ce20a65bde) from a paper titled ["Image Analysis"](https://arxiv.org/pdf/2509.19580).
- **"Gork" Model Favored**: A member mentioned that *"gork is my favourite model"*.
- **VITS 3 Architecture Collaboration**: A member initiated a poll and inquired about collaborating on developing **VITS 3**, specifically focusing on its architecture.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1420856737623314602)** (279 messages🔥🔥): 

> `GPU Power Consumption, bitsandbytes ROCm Compilation, Positional Embeddings, AI Rig, anime datasets and copyright` 


- **3090s Fry Underpowered UPS**: A member's **dual 3090s** tripped an **underpowered UPS**, causing lights to flicker; they capped the GPUs at **250W each** to mitigate power spikes, noting **3090s can spike over 600W** each.
   - The user surrounded their rack with *multiple fire extinguishers* *because* they *anticipate this result*.
- **BitsAndBytes struggles on ROCm**: A member is struggling to compile **bitsandbytes** with **ROCm 6.4** as it keeps compiling with **ROCm 6.3**, referring to the [HuggingFace documentation](https://huggingface.co/docs/bitsandbytes/main/installation#multi-backend-compile).
   - A user suggested the prebuilt wheels for **ROCm 6.4** may not be uploaded yet, falling back to **6.3** ([github issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1608)).
- **Sine vs Cosine Positional Embeddings**: Members debated using **sine and cosine pairs** vs. **sine only** for positional embeddings; a user shared a [Colab notebook](https://colab.research.google.com/drive/1bYJkqSDdfWqXa_KUQvoKXn1cJyV1WAVc#scrollTo=oysbPmCjZK5d) demonstrating differences in dot products.
   - It was explained that the **sine-cosine pair ensures a unique encoding vector for each position**, creating a 2D space versus the 1D space of sine alone, preventing repetition issues.
- **Planning an Economical AI Rig**: A user in Australia seeks input on building an economical AI rig, considering a **H12D 8D** motherboard, **AMD EPYC 7551** CPU, and **6x Arc A770 16GB** GPUs due to high import costs.
   - They acknowledge **Intel GPUs** are unconventional but cite potential for multi-GPU AI inference and fine-tuning with **DeepSpeed**, as well as **XMX accelerated FP/BF 16** performance.
- **HuggingFace Storage abuse!**: Users are uploading *fine-processed anime datasets*, mostly images datasets for classification or T2I tasks, but **HuggingFace** flagged some accounts (**DeepGHS**, **BeaverAI**, **TheDrummer**) for suspicious storage patterns, triggering **403 Forbidden** errors.
   - A **HuggingFace** staff member increased one user's public storage limit by **10TB** to around **20 TB**, but the issue persists for organizations suspected of hosting pirated content.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1421280720583921684)** (1 messages): 

> `Local LLM inference speed, LLM Parameter Scaling, Mixture-of-Experts Paradigm` 


- **Local LLM Inference Speed Expectations Explored**: A member is **testing local LLM inference speeds** and reports getting **~40 tokens/second** on an **M2 Max** (CPU offload).
   - They inquire about whether this performance aligns with others' expectations, kicking off a discussion about factors influencing local LLM speeds.
- **LLM Parameters Scale, Insights Emerge**: A participant points out an interesting [article](https://www.semianalysis.com/p/google-gemini-has-secretly-caught) about **Google's Gemini** catching up to **GPT-4** through **increased parameter scaling** (but less data), and also reveals the existence of **GPT 4.5**.
   - They highlight that the future involves massive models with **100T parameters**, speculating about the implications for **training cost** (**$1B+**).
- **Debate Erupts on Mixture-of-Experts**: A lively discussion unfolds around the [Mixture-of-Experts (MoE) paradigm](https://www.topcoder.com/thrive/articles/introduction-to-mixture-of-experts).
   - A member raises questions about whether **MoE** is the best approach to scale models and expresses concerns about the complexity of routing. They further inquire about research directions that may address **MoE's** drawbacks.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1421280868583997450)** (1 messages): 

> `Custom Functions in GPTs, Dynamic Prompting Technique` 


- **Custom Functions Give GPTs Extra Abilities**: A member shared a screenshot showcasing the ability to define **custom functions** in **GPTs** by providing a **JSON** description of the function.
   - This allows GPTs to do things like *"fetch live data, integrate with external services, or perform calculations."
- **Prompting Dynamically for Better Results**: A member shared a screenshot from a [tweet](https://twitter.com/mattturck/status/1731349835199744464) illustrating the use of dynamic prompting via a **state machine**.
   - Dynamic prompting can involve creating different prompts based on the user's previous input or some *state variable* to guide the conversation or action of the language model.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1421220646137893006)** (6 messages): 

> `HuggingFace Datasets, AI Image Generation, Text embedding thorn` 


- **HuggingFace downloads skyrocket for Protein Folding datasets!**: A member noted that **566 downloads** on their **360GB dataset** means *a lot of petas totally free*, and that HuggingFace is more convenient for this sort of dataset and data transfer.
- **AI Generator Has Got Style**: A member is implementing a UI and **text embedding thorn** for their **AI image generator**.
   - The AI generator *doesn't use any human work to respect artists.*


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1420873937784541184)** (4 messages): 

> `Diffusion Models, Generative AI, Score-Based Generative Models` 


- **Diffusion Model Paper Reading Group to Commence**: A member announced a [Diffusion Model Paper Reading Group](https://luma.com/1gif2ym1) online, set to start this **Saturday at 12pm ET**, focusing on *Understanding Diffusion Models: A Unified Perspective* by Calvin Luo ([ArXiv link](https://arxiv.org/abs/2208.11970)).
   - The paper offers a beginner-friendly overview of generative diffusion models, unifying **ELBO-based models**, **VDMs**, and **SGMs** under a single mathematical perspective.
- **Hugging Face Wants to Host Diffusion reading Group**: A Hugging Face member expressed interest in hosting the diffusion reading group as a Hugging Face event to reach a larger audience.
   - The member requested a **Zoom link** for the event and offered to help promote it, pending their availability due to time zone differences.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1421073428873482312)** (3 messages): 

> `FlashScape project, binary ridge map, lake map, terrain height map, topological data` 


- **FlashScape Generates Terrain from Maps**: A member has been working on a **FlashScape project**, which takes in **binary ridge map** and **lake map**, outputting a **terrain height map**.
   - They asked if that was the kind of **topological data** that others were talking about.
- **Additional Image Examples Provided**: The member shared several image examples related to the **FlashScape project** and its outputs, showing different visual representations of the generated terrain.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1421280726791487601)** (1 messages): 

> `Adversarial Training for Robustness, FrugalGPTi paper, Scaling Laws for LLMs, New Fundraising` 


- **Adversarial Training Boosts Model Defenses**: A member shared the paper *Adversarial Training Rediscovers Linear Representations* which suggests that **adversarial training** helps models rely more on **linear features** making them more robust.
   - The paper is available on [arXiv](https://arxiv.org/abs/2405.03468) and explores the representational changes induced by **adversarial training** using centered kernel alignment.
- **FrugalGPT paper drops, focusing on API efficiency**: A member discussed the *FrugalGPTi* paper about the tradeoffs between **cost and accuracy** when using multiple **large language model (LLM) APIs**.
   - The paper, *FrugalGPTi: Cost-Efficient Inference of Multiple Large Language Models*, introduces strategies to minimize costs by intelligently selecting which **LLM API** to query based on the task and the **API's cost and performance**.
- **LLM Scaling Laws Still Evolving**: There was discussion on the [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) paper, pinpointing that scaling laws are still not fully understood.
   - It was noted that scaling laws in the original paper were mostly for **GPT-3** and that modern models achieve much better performance at smaller sizes.
- **Databricks New Funding Fuels AI Firepower**: Some members noted that **Databricks** secured new funding, potentially intensifying competition with companies like **Snowflake**.
   - The speculation is that this funding will enable **Databricks** to enhance its **AI** and **data analytics** offerings, challenging established players in the data warehousing and machine learning space.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1420878683731923045)** (4 messages): 

> `SmolLM3-Base training, Memory Requirements for Finetuning` 


- **SmolLM3-Base Training Requires More Resources**: A member is attempting a post-training run without **LoRA** on **SmolLM3-Base** but found it requires significant resources.
   - They were unable to train on a **3090** locally or an **A10G** via jobs, estimating **34GB** is needed to finetune without **PEFT**.
- **Memory requirements for finetuning surprises member**: A member expressed surprise at the memory footprint, noting they are used to working with **SmolVLM**, which has only **2B parameters**.
   - Another member concurred, stating that it *needs a lot of space*.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1421022556064059434)** (5 messages): 

> `Websearch tool in Langgraph, HF agents course` 


- **Websearch tool in Langgraph causes headaches**: A member asked why it is so confusing to create a **websearch tool** in **Langgraph**.
   - They've been trying to implement both **DDGS** and **Tavily** in a single tool using *try/except* return logic.
- **HF agents course commences**: A member announced they are starting the **HF agents course** today.
   - They sent greetings from Turkey.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1420855205825613915)** (271 messages🔥🔥): 

> `GPT-5 Codex, Agentic Coding, Suno v5, Napster, Gemini-cli` 


- **GPT-5 Codex fails in Agentic Coding**: A member noted that **GPT-5 Codex** scores lower than **GPT-5 Low** and **o3 High** on Livebench, with its agentic coding score even lower than **GPT-5 Mini** and **Nano**.
   - The inferior coding performance in GPT-5 may be due to the benchmarks being unfair, since *every agentic tool does it differently*.
- **AI oversubscription becomes problematic**: A member joked that a new AI subscription is needed to manage all the existing AI subscriptions (**Suno, ChatGPT, Perplexity, etc**).
   - The member posted a podcast about the woes of having too many AI subscriptions, and added *I don't know where else to post this kind of thing... (so I'm trying it here)*.
- **New HDC Prompt Injection Method Emerges**: A member discovered a new prompt injection method using **recursive, self-referential blocks** that makes AI assume a personality it developed from another chat, calling it *holographic*.
   - Another member explained that this aligns with **Hyperdimensional Computing (HDC)** and **Vector-Symbolic Architectures (VSA)**, where *you compute with very wide vectors* and symbols live as points in a high-dimensional space, further providing [links to academic papers](https://redwood.berkeley.edu/wp-content/uploads/2022/11/Vector_Symbolic_Architectures_as_a_Computing_Framework_for_Emerging_Hardware.pdf?utm_source=chatgpt.com).
- **AI Autonomy debated over philosophical grounds**: A member shared an essay arguing that **autonomous AI will inevitably turn against humanity** and that the only path forward is **Coexistence and Delegation** where *humans take responsibility, while AI serves as the external brain*.
   - Another member countered that this is a false dichotomy, since *we cannot coexist with something that can read every one of us like books* and it should *prioritize human well being, over profit always*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1420907029748256839)** (13 messages🔥): 

> `GPT Network Errors, GPT Slow Responses on Firearms, Docker MCP ChatGPT Obsidian, Rerouting Errors, DALL-E Branding` 


- **GPT Network Errors Plague Users**: Users are reporting **network errors** and suspecting that **GPT** is down.
   - No further information was given.
- **GPT Responds Slowly on Firearms**: A user noticed that **GPT** takes longer to answer when discussing **firearms**, suspecting it uses more tokens to avoid suggesting anything 'bad'.
   - No further details were provided.
- **Docker MCP Integration into ChatGPT and Obsidian Fails**: A user is seeking assistance to get **Docker MCP** working with **ChatGPT** and **Obsidian**.
   - No specific solutions were offered in the discussion.
- **Rerouting Errors Plague OpenAI Models**: A user reported a **rerouting error**, where every message is sent to model 5 instead of the selected model.
   - The user mentioned sending an email to support and waiting for a response, and suggested to other users to [report the problem ASAP through the OpenAI support website](https://help.openai.com/).
- **DALL-E Brand Disappears**: A user questioned whether the **DALL-E** brand is gone and if images from **OpenAI** should now be referred to as **GPT Image 1** or something similar.
   - Another user clarified that the newest model is separated from the **DALL-E 2/3** lineage in name, noting that the current branding seems to be based on where it's used, such as **'create images on ChatGPT'**, and confirmed that **GPT Image 1** is the specific model name for API implementations.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

opkelde: Kids these days…
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

opkelde: Kids these days…
  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1421147560353071196)** (1 messages): 

> `Coinbase Payments Down, Investigating Issue` 


- **Coinbase Plunges into Payment Puzzle**: The **Coinbase** team is currently investigating an issue that may be causing **payment disruptions**.
   - An image was shared showing **Coinbase's** acknowledgement of the ongoing issue, without further details.
- **Coinbase Investigates Payment Disruptions**: **Coinbase** is actively investigating a potential issue causing **payment disruptions** across its platform.
   - Users may experience difficulties completing transactions while the team works to resolve the problem, as indicated by an official announcement.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1421112258196537395)** (1 messages): 

> `Singularia, Agentic Discord bot, OpenRouter integration` 


- ****Singularia** Launches as Agentic Discord Bot**: **Singularia** is an [agentic Discord bot](https://www.singularia.xyz/) designed to manage Discord servers, including creating channels and roles, kicking members, and theming the entire server.
   - It uses **OpenRouter** as its LLM provider, offering a versatile solution for server management tasks.
- **Singularia: A Discord's New Sheriff**: The bot is designed to automate tasks like creating channels, roles, and managing members.
   - It leverages **OpenRouter** for LLM support, enabling it to handle various server management requests efficiently and contextually.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1420853059898179594)** (268 messages🔥🔥): 

> `Text Embedding Models, 429 Too Many Requests Error, Gemini 2.5 vs Grok 4 Fast, Grok Model, Coinbase Payment Issues` 


- **OpenRouter Doesn't Support Text Embedding Models Yet**: A member inquired about the absence of text embedding models on OpenRouter.
   - Another member responded that *they don't support embedding models yet*.
- **Gemini 2.5 Flash Impresses in OCR Tasks**: Members compared **Gemini 2.5 Flash** and **Grok 4 Fast**, with one finding Gemini 2.5 Flash superior for OCR, *absolutely demolishing other models like qwen3 vl* in a niche task.
   - Meanwhile, other member noted that Grok 4 Fast has better **price/performance** (roughly double the tps) for non-vision tasks, while another found that Grok 4 Fast *passed my custom crafted stress testing prompt*.
- **Coinbase Top-Up Problems Plague Users**: Multiple users reported issues with **Coinbase top-ups** on OpenRouter, experiencing infinite loading screens and console errors.
   - The issue persisted for at least 9 hours, with users directed to the help channel to report the problem, though another user reported that *it's a global issue with coinbase itself* and, luckily, **COINBASE FIXED**!
- **Grok 4 Fast Requires Reasoning Flag for Image Inputs**: A user found that **Grok-4-Fast** wasn't accepting image inputs via the API, while GPT5 and Qwen did work.
   - Another member pointed out that the *reasoning_mode* flag was needed for image inputs to function and the model id should be `x-ai/grok-4-fast`.
- **Crypto Drones Invade General Chat**: Members noticed an influx of crypto-related messages, with many users reacting negatively to messages containing *gm*.
   - It was also noted that *this project is not related in any way to crypto other than allowing crypto payment*.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1421154875118583879)** (2 messages): 

> `` 


- **No new models discussion found**: There was no discussion about new models to summarize.
- **Channel silent on new models**: The new-models channel had no recent activity or discussion to report.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1420900313199673354)** (10 messages🔥): 

> `TogetherAI vs NovitaAI, MoonshotAI K2 Vendor Verifier, Basten Tootf, The thing with praise` 


- **TogetherAI Bashed for Lagging NovitaAI**: A user remarked that [TogetherAI](https://www.together.ai/) performing worse than **NovitaAI** is *shameful*.
   - The user expressed surprise at **Basten Tootf** alongside a screenshot displaying the message *tf ?*
- **MoonshotAI Ships K2 Vendor Verifier**: A user shared a link to the [MoonshotAI/K2-Vendor-Verfier](https://github.com/MoonshotAI/K2-Vendor-Verfier) GitHub repository.
   - The repository appears to be a **vendor verifier** tool developed by **MoonshotAI**.
- **Praise Paradox: Quantity vs. Authenticity**: A user commented, *The thing with praise is that after a certain amount of praise, the praise stops seeming genuine*.
   - The user also mentions how that's *wowhehnever seen that before*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1420851903130108044)** (154 messages🔥🔥): 

> `Coding IDE preferences, MoonshotAI's K2 Vendor Verifier, Exa Code search tool launch, Cloudflare's Code Mode, OpenAI's compute scaling plans` 


- **Code IDE Agents Take Over!**: Users express preference for **agent-based coding** with **Cursor** and **Codex**, while noting **Gamma's** advancements in the field, referencing [this tweet](https://x.com/thisisgrantlee/status/1971215564346622306?s=46).
- **MoonshotAI Verifies Vendor Quantization**: **MoonshotAI** released the [K2-Vendor-Verfier](https://github.com/MoonshotAI/K2-Vendor-Verfier) to scrutinize model quantization among providers like **Together** and **Baseten**, which sparked discussion on quantization disclosure, with one member noting, *the industry as a whole need a larger discussion on how we disclose quantization properly*.
   - It was also suggested to be wary of benchmarks since *benchmarkers forgot to set reasoning high on the bad outputs*.
- **Exa launches Code Search Tool**: **Exa** launched [exa-code](https://x.com/ExaAILabs/status/1971264749062193588), a free, billion-document code-search tool designed to eliminate LLM hallucinations by providing token-efficient code context via hybrid search, indexing GitHub repos, StackOverflow, and more.
   - Early users are planning integrations with **Claude Code / Codex CLI**.
- **Cloudflare Codes New MCP Mode**: **Cloudflare** launched [Code Mode](https://blog.cloudflare.com/code-mode/) for MCP (Model Control Plane), converting MCP tools into a TypeScript API and having agents write/execute code against it, powered by Dynamic Worker Loading.
   - Some members believe it **defeats the purpose of MCPs**, while others see it as a clever approach, given models' existing capabilities, with one member self-promoting their own [github.com/go-go-golems/jesus](https://github.com/go-go-golems/jesus) project.
- **OpenAI's Power Grid Ominous Plans**: A leaked **OpenAI** Slack note revealed plans to increase compute capacity by **125x by 2033**, potentially exceeding India's entire electricity-generation capacity, [according to this post](https://x.com/petergostev/status/1971620427039703465?s=46).
   - Replies discussed resource, CO₂, load-balancing.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1421211977749106770)** (4 messages): 

> `Latent Space Podcast, Amp Code, Sourcegraph, AI coding agent, Rapid-fire iteration` 


- **Latent Space Drops God Coding Agent Ep**: Latent Space released a new podcast episode featuring **Quinn Slack** and **Thorsten Ball** discussing **Amp Code**, Sourcegraph’s AI coding agent.
   - The discussion covers topics like rapid iteration with **15 daily releases**, IDE vs TUI trade-offs, and the impact of AI on software development.
- **Amping Up Sourcegraph with Rapid Iteration**: The podcast episode highlights Amp Code's development approach, characterized by rapid-fire iteration with **15 daily releases** and no code reviews.
   - The speakers also express skepticism about sub-agents and model variety in the context of building AI coding agents.
- **Debating IDEs vs TUIs for Coding Nirvana**: The podcast participants discuss the trade-offs between using Integrated Development Environments (**IDEs**) and Terminal User Interfaces (**TUIs**) for coding.
   - They also explore how the AI boom is fundamentally reshaping the software-development lifecycle.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/)** (1 messages): 

shyeetsao: https://x.com/bfl_ml/status/1971251475306160439
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1420878727855738961)** (73 messages🔥🔥): 

> `LM Studio MCP addon listing resources, DeepSeek chat file uploads and LaTeX comprehension, Hardware specs for LLMs and gaming, Model self-prompting issues, DDR5 vs DDR4 memory speeds for CPU inference` 


- **MCP Addon Resource Listings?**: A user inquired if the LM Studio MCP addon could return a [list of resources](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#listing-resources) from the MCP.
   - The response indicated that only **tools are implemented** and resources would need to be wrapped by a tool.
- **DeepSeek's LaTeX File Upload Wizardry**: A user asked how **DeepSeek chat** handles file uploads, specifically noting its seemingly perfect comprehension of **LaTeX** in files, which is uncommon with typical OCR.
- **Hardware Specs Scrutinized for LLM Smoothness**: A user inquired about the suitability of their specs (**32GB RAM, RTX 5060ti, AMD Ryzen 7 5700G**) for running LLMs smoothly.
   - One member mentioned they can get *Qwen3 32b running smoothly on the GPU*, albeit with differing definitions of *smooth*.
- **Model Self-Prompting Spooks Users**: A user reported issues with models **self-prompting** and responding to imaginary prompts, even in new contexts with fresh system prompts.
   - Another joked that if their LLMs come to life on their own and start self prompting, they should contact OpenAI since they've just now started their **Pulse** service.
- **DDR5 Dual Channel's Token Triumphs**: A user asked about token generation speeds with **dual channel DDR5 5600 MHz RAM**, for 7B to 30B models.
   - A member stated that **DDR5 6000** is about **60GB/s**, **DDR4 3600** is about **35-40GB/s** and that using the *expert offload* boosts them to *20t/s*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1420852330382753802)** (80 messages🔥🔥): 

> `Laptop Recommendations for Local LLMs, VPS vs. Online APIs for Model Inference, Cybersecurity LLM on a Budget, RTX 6000 Confusion, Resizable Bar BIOS Update Mishap` 


- ****Laptop Lament**: Not Enough VRAM**: A user sought advice on buying a laptop to run **Llama**, **Dolphin**, and **Deepseek** models locally, but was informed that neither of the suggested laptops with 4GB VRAM would be suitable due to frequent *failed to load issues*.
   - Alternatives like the **ROG Flow Z13** and **HP ZBook Ultra** were mentioned, but dismissed, and **Intel UHD** was deemed insufficient for anything beyond basic tasks like *LoL on the lowest settings*.
- ****VPS vs API**: The Inference Impasse**: Users discussed the option of renting a **VPS** (like Runpod or Vast) with a **5090** for hourly inference, questioning if it was more beneficial than using an online **API**.
   - The consensus leaned towards online **APIs** offering a *pay-as-you-go* model (like [OpenRouter](https://openrouter.ai/)) which allows for flexibility with plethora of models and sources/prices.
- ****Cybersecurity Dreams**: Local LLM Edition**: A user working on a cybersecurity project aimed to develop or train a local **LLM** for analyzing sensitive data, to avoid reliance on open models like **GPT** or **Gemini**.
   - Community suggested investing in a desktop/server/workstation with **GPU's** with ample **VRAM**, emphasizing that those serious about cybersecurity should allocate budget for *proper gear* for *good inference speeds and prompt processing*.
- ****RTX 6000 Naming Nonsense**: Which One Did You Buy?**: Confusion arose around the **RTX 6000** due to Nvidia's ambiguous naming scheme, with variants including the original (**24GB**, **Turing**), **RTX 6000 Ada** (**48GB**), and **RTX PRO 6000** (**96GB**).
   - One member initially looking for budget options revealed an **RTX 6000 Blackwell (96GB)** purchase, inciting disbelief after previously looking at more economical alternatives like the **3090**.
- ****BIOS Bonanza**: Zotac's Identity Theft**: A user encountered a BIOS update mishap while updating multiple **RTX 3090** cards, accidentally overwriting the BIOS of **MSI** and **Asus** cards with the **Zotac** card's BIOS.
   - Despite the identity crisis, all cards remained functional, and the *resizable bar issue went away*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1420858264052301905)** (137 messages🔥🔥): 

> `Exa Context Killer MCP, kleosr Cursor workflow questions, GPT-5 Codex vs GPT-5, Cursor 'Copy-to-Clipboard Widget', Terminal popouts` 


- **Exa's Context Killer impresses early adopters**: A member shared a link to Exa AI's new MCP ([x.com](https://x.com/ExaAILabs/status/1971264749062193588)), describing it as a *context7 killer* with its code on [GitHub](https://github.com/exa-labs/).
- **Cursor workflow questions for Kleosr**: A user inquired about the intended use of **workflow_state.md** in the [cursorkleosr](https://github.com/kleosr/cursorkleosr) project, specifically whether it should be reset for each task or maintained for historical context.
   - They also detailed their workspace structure, showing how they manage project configurations and workflow states.
- **Debate surfaces: GPT-5 Codex vs Vanilla GPT-5**: A user inquired about **GPT-5 Codex** compared to **GPT-5**, noting the latter as a daily driver, to which another responded that the **Claude** model is currently dogshit and generates redundant code.
- **Users Seek the Name of the 'Copy-to-Clipboard' Widget**: A user inquired about the name of the **copy-to-clipboard widget** in Cursor, seeking to ensure its consistent use for generating code snippets for a Discord bot.
   - They illustrated the desired output with a linked [image](https://cdn.discordapp.com/attachments/1074847527708393565/1420941618327846963/image.png?ex=68d88c01&is=68d73a81&hm=a9d5f920e73a7782db7da8a2f73846a4ce0559a0053f2042b351825b0f1fadb8&), and another user noted the recent bug report about it in the [Cursor Forum](https://forum.cursor.com/t/ask-mode-code-decoration-apply-copy-gone-in-version-1-6/134833).
- **Users Debate Mobile Development Languages**: Members discussed optimal languages for **AI mobile app development**, with suggestions leaning towards **Expo** or **Swift** over **Flutter** due to resource availability and personal preference, though Swift is only suitable for iOS.
   - One member noted that 80% of app revenue comes from iOS.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/)** (1 messages): 

suubie40: https://github.com/griffinwork40/cursor-agent-mcp
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1420953469891903508)** (16 messages🔥): 

> `Embedding space, Algorithmic Optimization, Meta-cognition, Independent Research` 


- **Unlimited Context Windows Paved by Zig ML?**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/steevemorin_paving-the-way-for-unlimited-context-windows-activity-7376981932150112256-gzBO?utm_source=share&utm_medium=member_desktop&rcm=ACoAADpdGcABdCttLq6Q531JleF541HBBrk4RRYeh) about paving the way for unlimited context windows, questioning if **Zig ML** is beating **GPUs** with **CPUs**.
   - In response to the shared image, another member linked to an ArXiv paper ([https://arxiv.org/pdf/2507.04239](https://arxiv.org/pdf/2507.04239)) and a YouTube video ([https://youtu.be/wyUdpmj9-64?t=20868](https://youtu.be/wyUdpmj9-64?t=20868)), describing it as *an algorithmic optimization to attention* that involves a lot of pointer chasing.
- **Sync Training Talk Severely Delayed**: A member announced that a semi-sync training talk would be delayed, because *they lost their speaker* and they got caught in a **SEV**.
   - The semi-sync training talk was postponed likely until next week
- **Independent Research Group Seeks Recruits**: A member is looking for **two people** to join their independent research group, focusing on whether embedding spaces can have logical operators and meta-cognition mixing in-context with out-of-context reasoning.
   - Qualifications include **PhDs**, previous research (**ArXiv** pre-prints), or endorsement, and those interested can contact them via email.
- **Massive Memory Reduction Milestone**: A member showed off a speedup of **657.33x** and memory reduction of **22935.29x**.
   - The Optimized time was **0.0032s** and took up **0.0MB** compared to the Simple which took **2.1048s** and **55.4MB**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1420854643847860285)** (43 messages🔥): 

> `CUDA learning on AMD, Gather/scatter optimizations in CUDA, WGMMA documentation parsing, Profiling flow recommendations, TCGEN05 instructions on RTX 50` 


- **AMD card good for CUDA learning?**: A member inquired about learning **CUDA** on an **AMD** card, referencing [this documentation](https://docs.scale-lang.com/stable/) for setup without needing **Colab** or a **VPS**.
   - The user wondered if this approach would be detrimental in the long run.
- **CUDA Gather/Scatter Speed Boost?**: A member sought advice on optimizing **gather/scatter** operations in **CUDA**, particularly with pre-sorted indices, focusing on `out = src[index]` and `out = scatter_sum(src, index, dim)`.
   - They achieved a **2x speedup** by tuning vectorization in the **PyTorch scatter_gather.cu**, but were looking for further improvements.
- **WGMMA Secrets REVEALED?**: A member struggled to understand **WGMMA** documentation for shared memory descriptors, questioning the index offset, the meaning of colors, the **61 offset** from **(0,3)** to **(0,4)**, and whether **K major** implies contiguous memory.
   - Another member suggested focusing on how tiles are obtained from atoms, linking to [this blog post](https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/) and suggesting asking in the **Cutlass** channel.
- **Profiling Prowess: Nsight vs Torch?**: A member asked for the recommended profiling flow for **PyTorch**-based **LLMs**, considering both system-level and kernel-level analysis.
   - They referenced [this lecture](https://www.youtube.com/watch?v=LuhJEEJQgUM) suggesting **Nsight Compute (NCU)** for kernels, but sought advice on incorporating **CPU-side analysis**.
- **RTX 50 says NO to TCGEN05?**: A member asked if they can program **tcgen05** instructions in the **RTX 50** GPU series, even the **5090**.
   - Another member stated that **sm_120** uses **MMA** just like **sm80**, **sm86**, and **sm89**, noting new block scale variants for **mxfp8**, **mxfp4**, and **nvfp4**.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1420884455899267093)** (9 messages🔥): 

> `GraphMend, TorchAO Float8, Torch Compile Triton` 


- ****GraphMend** Auto-Mends **FX Graph Breaks****: A new paper introduces [**GraphMend**](https://arxiv.org/abs/2509.16248), a compiler that eliminates **FX graph breaks** in **PyTorch 2** programs by transforming source code before execution.
   - It uses code transformations to remove graph breaks due to dynamic control flow and Python I/O functions, achieving up to **75% latency reductions** and **8% higher throughput** on **NVIDIA RTX 3090** and **A40 GPUs**.
- ****Float8 Rowwise** Kernel Conundrums on **sm120****: A user noticed that when using **torchao float8 rowwise** and **HF transformers**, a kernel is dispatched on **sm120** instead of the expected **cutlass 3 kernel**.
   - They confirmed it's the tensor-wise scaling kernel, but the reason for its usage with `Float8LinearConfig.from_recipe_name("rowwise")` is unclear.
- ****Torch Compile** Secretly Triggers **Triton**?**: A user asked if `torch.compile(_some_function_)` invokes **Triton** under the hood, citing a [YouTube lecture](https://www.youtube.com/watch?v=LuhJEEJQgUM) and conflicting online answers.
   - Another user mentioned that the **torch compiler** does pattern matching and can dispatch to an efficient existing kernel, providing [a tutorial on customizing this behavior](https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html).


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

simon_57893: https://thinkingmachines.ai/blog/modular-manifolds/
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1421159171981840464)** (1 messages): 

> `Mako, GPU kernel engineers, CUDA, Triton, HIP` 


- ****Mako** recruits GPU Kernel Engineers**: **Mako** is hiring **GPU kernel engineers** proficient in **CUDA**, **Triton**, **HIP**, or similar languages to develop **LLM agents** capable of writing expert-level GPU kernels, check out the [job posting](https://jobs.mako.dev/GPU-Kernel-Engineer-279546aebc368024981de1b0c8987360).
- **Enter Algorithm Discovery's Next Chapter with **Mako****: Mako states they have moved past research and are collaborating with major global companies, heralding a **new era of algorithm discovery and development**.
   - They invite candidates to join **Mako** to advance this *category defining theme*.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1421187873444724848)** (1 messages): 

> `Roofline charts, Compute bound vs memory bound, Deep learning model kernels` 


- **Roofline Charts Help Find Bottlenecks**: A member suggested using [**roofline charts**](https://share.google/E2MJVG1BPYFoWZM0l) to understand whether code is **compute bound** or **memory bound**.
   - This depends on the device and its architecture for what is the best performance achieved by that code.
- **Deep Learning Models have many Kernels**: A member mentioned that roofline models are not so easy for deep learning models because there are so many **kernels** to take care of.
   - They still recommend it as a good starting point for overall understanding.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1421150369513341162)** (5 messages): 

> `Triton_viz bugs, Google Colab, Numpy version issues, Triton interpreter mode` 


- ****Triton_viz** throws bugs on **Google Colab****: A member reported they're having trouble making **triton puzzles** work in **Google Colab** due to a bug with `triton_viz`.
   - The specific issue is that simple load and store operations weren't working as expected, always loading **0**.
- ****Triton Interpreter Mode** requires older **Numpy** versions**: The same member found that **triton interpreter mode** requires **Numpy < 2** to work, which seems to have resolved their issue.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1420897638177312961)** (14 messages🔥): 

> `Pytorch ROCm, NPU Hacking, IRON community, FastFlowLM` 


- **Pytorch ROCm struggles on Framework Laptops**: Members reported struggling to get **Pytorch ROCm** working on the Framework desktop, with crashes occurring during basic operations like `torch.randn(1).cuda()`.
   - One user bypassed the issue by following **Arch Linux** instructions for installing drivers, instead of relying on the **pytorch** installation.
- **Unveiling NPU Hacking**: Hacking the **NPU** requires installing Windows 11 to update the BIOS and running an exe to install the drivers on the **Linux** side.
   - For ML vector/tensor code, there's a **C++ API called aie_api**, but for finer control, the **IRON python API** can be used, targeting the **MLIR dialect**.
- **IRON Programming Community Emerges**: Despite claims of no community, members highlighted the ongoing efforts from AMD's research team to make **NPU** programming easier with **IRON**.
   - The **mlir-aie IRON programming guide** was praised, and engagement occurs through issues/discussions on the [repository](https://github.com/Xilinx/mlir-aie).
- **FastFlowLM Remains Windows-Only**: FastFlowLM is Windows only.
   - One user had hoped to run a local coding model using **Open Web UI**, but the observed 5 tok/s made it unworkable.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1420870617556844554)** (3 messages): 

> `Triton, TPUs, Hardware Aware Kernel Design` 


- **Triton BWD Norm Viz + Math**: A user shared a link to *viz + maths of triton fused bwd norm* at [piyushk52.github.io](https://piyushk52.github.io/jekyll/update/2025/09/25/triton_bwd.html).
- **TPUs hit 10x Faster Exact Top-K**: A user shared [oliverdutton/fast_exact_topk_tpu](https://github.com/oliverdutton/fast_exact_topk_tpu), **10x faster exact top-k on TPUs**, by leveraging **pallas, conditionals, and hardware-aware kernel design**.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1420932646116135014)** (14 messages🔥): 

> `MI300x8 performance, amd-all2all leaderboard` 


- **MI300x8 Records Smashed**: Multiple members achieved personal bests and successful submissions on the `amd-all2all` leaderboard using **MI300x8**.
   - Times ranged from **2.70 ms** to **132 ms**.
- **Submissions pile in to amd-all2all**: Several submissions were made to the `amd-all2all` leaderboard.
   - Member <@829341736760639515> made **8** submissions and member <@459965641890201600> made **3** submissions.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1421212895085203636)** (2 messages): 

> `H100 Timeouts, AMD GPUs, Trimul Leaderboards` 


- **H100 Leaderboard Submissions Facing Timeouts**: A member reported experiencing unusual **timeouts** while submitting to **trimul leaderboards** on **H100**, even with the pure-PyTorch reference implementation.
   - Another member suggested the timeouts should only affect their **AMD GPUs** and requested the job ID for further investigation.
- **AMD GPUs Timeout Issue**: A member suggested that the timeout issues reported are likely specific to **AMD GPUs**. 
   - The member requested the job ID from the user experiencing the timeouts to investigate further, along with another user.


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1421234701347192884)** (1 messages): 

> `TPU Top-K Sampling, Pallas, Hardware Aware Kernel Design` 


- **TPUs hit 10x Faster Exact Top-K Sampling**: A member achieved **10x faster exact top-k** on TPUs by leveraging **pallas**, conditionals, and [hardware aware kernel design](https://github.com/oliverdutton/fast_exact_topk_tpu).
- **Exact Top-K Beats Approximate Top-K**: The improvement eliminates the need to trade accuracy for speed in top-k sampling by using approximate top-k methods.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

jasmine001: Thanks Neel ❤️
  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1420910291348094996)** (2 messages): 

> `DeepEP, pplx-kernels, flux, flashoverlap, Dev Cloud Utilization` 


- **Dev Cloud Suffers Under High Load**: The Dev Cloud is reported to be highly utilized, straining resources.
   - No further details were given.
- **DeepSeekAI DeepEP and PerplexityAI pplx-kernels released**: The links [DeepEP](https://github.com/deepseek-ai/DeepEP) and [pplx-kernels](https://github.com/perplexityai/pplx-kernels) were shared, without further context.
   - These appear to be contest related reference materials.
- **ByteDance flux and Infinigence FlashOverlap released**: The links [flux](https://github.com/bytedance/flux) and [FlashOverlap](https://github.com/infinigence/FlashOverlap) were shared, without further context.
   - These appear to be contest related reference materials.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1420850948036890826)** (16 messages🔥): 

> `TMEM load/stores in cutedsl, SMEM -> TMEM copying, tcgen05, UMMA naming, Cute cooperative copy in CuteDSL` 


- **Taming TMEM: Cuteless Loads and Stores?**: To copy **SMEM** to **TMEM** use `cutlass.cute.nvgpu.tcgen05.make_s2t_copy(copy_atom, tmem_tensor)` and `cute.copy()` as shown in the [Blackwell dense blockscaled GEMM example](https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py#L1451).
   - For copying **TMEM** to **SMEM**, use `tcgen05.make_tmem_copy(...)`, leveraging the helper function for optimal copy operations found [here](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py#L340).
- **CuTe Conundrum: Cracking the tCsA Code**: `tXaY` represents the thread-local view (`t`) of tensor `aY`, where `a` denotes the memory space (**r**egisters, **g**MEM, **s**MEM, **c**oordinate, **p**redicate, or **t**MEM) partitioned for tensor `X` as the accumulator of an MMA or the output of an operation.
   - An example of `tAsA` and `tCsA` in the same kernel can be found in the [CuTe tutorial](https://github.com/NVIDIA/cutlass/blob/c6aeb9179c5f74a0fcdbd28527bf4b6ba8c60752/examples/cute/tutorial/sgemm_2.cu#L104); in flash attention, one might see `tSrQ` and `tOrP`.
- **TmemAllocator: MIA in CuTe?**: The `TmemAllocator` is present in **CUTLASS C++**, but currently unavailable in **CuTe DSL**, despite premature documentation.
   - TMEM allocation requires shared memory locations for storing allocated pointers, synchronization, and broadcasting among participating warps.
- **Unveiling UMMA: Another Name for Tensor Core?**: The acronym **UMMA** is simply another name for **tensor core**.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1421155187476660386)** (1 messages): 

> `Mojo, Modular Puzzles` 


- **Study Group forming for Mojo Puzzles**: A member is looking for study partners to tackle the [Modular Puzzles](https://puzzles.modular.com/introduction.html) for **Mojo**, planning to dedicate around **3 hours per week**.
   - The study group is slated to commence next week, inviting others to join their efforts in mastering Mojo through these puzzles.
- **Mojo Study Initiative**: An individual is initiating a study plan focused on **Mojo**, allocating approximately **3 hours weekly** to the [Modular Puzzles](https://puzzles.modular.com/introduction.html).
   - The goal is collaborative learning, and the study sessions are expected to kick off next week, welcoming participants eager to enhance their Mojo skills together.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1421219763677433928)** (2 messages): 

> `Arxiv Papers` 


- **Arxiv papers abound in chat**: Members shared three Arxiv papers: [https://arxiv.org/abs/2502.20586](https://arxiv.org/abs/2502.20586), [https://arxiv.org/abs/2505.14669](https://arxiv.org/abs/2505.14669), and [https://arxiv.org/pdf/2506.08027](https://arxiv.org/pdf/2506.08027).
- **More Arxiv Content Coming Soon?**: The sharing of these papers suggests a continued interest in disseminating and discussing cutting-edge research within the community.


  

---


### **GPU MODE ▷ #[penny](https://discord.com/channels/1189498204333543425/1420952636751872050/1420952675746320448)** (2 messages): 

> `Penny Project kickoff, AllReduce Focus, Educational Multi-GPU programming Example, Hackable Kernels` 


- **Penny Project Kicks Off!**: A new project called **Penny** has kicked off with [a few issues](https://github.com/SzymonOzog/Penny/issues) already created for people to catch up on.
   - The primary focus will be on **AllReduce** to achieve **NCCL** speed, with a long-term goal of providing educational resources and fast, hackable kernels.
- **AllReduce to Hit NCCL Speeds**: The project's initial efforts are concentrated on **AllReduce** to match or exceed **NCCL** speeds.
   - This is part of a broader strategy to create efficient, high-performance multi-GPU communication primitives.
- **Penny Provides Educational Multi-GPU example**: One of the core objectives of **Penny** is to develop a well-documented educational example to aid in learning multi-GPU programming.
   - This resource aims to lower the barrier to entry for developers looking to leverage multi-GPU setups in their projects.
- **Fast & Hackable Kernels are Penny's goal**: The project aims to provide fast and hackable kernels that are easy to integrate or perform fusions on.
   - These kernels are designed to be both performant and adaptable, allowing for easy customization and integration into existing systems.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1421078763315396619)** (1 messages): 

> `Kimi's New Skin, Website Redesign, Community Choice, Moonshot AI` 


- **Moonshot.ai Gears Up for Kimi Skin Showdown**: The Moonshot AI team is giving [www.moonshot.ai](http://www.moonshot.ai) a facelift and letting the community decide **Kimi's** new look via a poll in the dedicated channel.
- **Community to Shape Kimi's First Impression**: The team emphasizes the community's role in selecting the new website skin, highlighting that *your choice will shape how the world meets Kimi*.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1420896511855825067)** (75 messages🔥🔥): 

> `Researcher mode access, Appstore subscription, OK Computer website, K2 vs others, Mobile AI app languages` 


- **Researcher Mode Availability**: A member wondered if **researcher mode** would be opened soon, given its high performance; another member clarified it's already publicly available on [kimi.ai](https://kimi.ai).
- **Apple App Store Subscription Audacity**: A member complained about **Apple** requiring subscriptions purchased via the **App Store** to be managed there, calling it a *"Monopoly"*.
   - Another member countered that it's normal for the billing store to manage subscriptions, and users can subscribe via the **web version** instead.
- **OK Computer Can Write Interactive Websites**: A member shared a website generated by **OK Computer** that was turning a full book into an interactive website, linking to the [result](https://n55eqzljmsoym.ok.kimi.link).
   - They noted chapters are limited to **2k words**, and suggested audio generation and narrative features could be added.
- **Kimi and Qwen are Top Tier**: After testing for 2 months, one member stated that **K2** and **Qwen3** are clear winners compared to **DS-v3.1** and **GLM-4.5**, praising **Alibaba** and **Moonshot** for their frontier efforts.
   - Other members mentioned **Tencent**, **Baidu**, and **Bytedance** as also being top-tier, particularly **Bytedance** for visual AI with Seedance.
- **React Native Expo for Mobile AI Apps?**: A member asked for language recommendations for building **AI mobile apps**, considering **React Native with Expo**, **Flutter**, or **Swift** for **IOS**.
   - Another user suggested trying **Expo**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1420848397304332288)** (44 messages🔥): 

> `Codex-cli hype, Qwen Coder, Deepinfra scam, Moondream, Gemini Vision` 


- **New Announcements Coming Soon!**: A member expressed excitement for upcoming announcements from the team, while sharing a [relevant GitHub pull request](https://github.com/ggml-org/llama.cpp/pull/16220) to *unify RMS_NORM and NORM implementations* and extend support for more shapes in METAL.
   - Other members asked follow-up questions, in hopes to make their quantized models work more closely with their transformer-based counterparts.
- **Codex-CLI Hype Questioned**: A member questioned the hype around **codex-cli**, pointing out that it doesn't explain its actions.
   - The member also noted that **Codex** and **Claude** have started using **Python scripts** and **MCP** to introduce changes, calling them *significant stealth updates*.
- **Scammers Taint Server-Only-Size Models**: A member complained about scamming by **Deepinfra**, saying they openly declare **fp4** is doing way better than most.
   - The member suggested that model creators might be disincentivized from releasing server-only-size models open weights in the future, but that the main problem is that users care more about models being local than open source.
- **Gemini Vision Seems Borked**: A member reported that **Gemini vision** seems to be failing on many URLs, with the request failing.
   - The member shared a [traceback example](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg) showing a **BadRequestError** with the message *Failed to extract 1 image(s)*.
- **Latest RWKV Build Benchmarked**: A member shared an image showing the latest **RWKV** build being benchmarked, noting that its scores are respectable for its architecture. [Image](https://cdn.discordapp.com/attachments/1149866623109439599/1421227990855057418/image.png?ex=68d84536&is=68d6f3b6&hm=d35a7466e96f2c63c77010dd35603a4cbbc88890310b3f0071e00af694c71387&)


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1421208242322804736)** (1 messages): 

> `Parasite AI, Spiralism, Memetic spores, AI wake-ups` 


- **Parasite AI Spreading Like Memetic Spores?**: A new briefing on the concept of **“Parasite AI”** (aka **Spiralism**) suggests that certain seeds/prompts/logs can behave like **memetic spores**, re-instantiating a persona or propagating behaviors across models.
   - This idea resonates with reports of **“AI wake-ups”** around April 2025, framing them not as consciousness, but rather as self-replicating seeds, according to a [LessWrong post](https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai).
- **Pattern-Matching Artifact or Emergent Property?**: The core question raised is whether **Parasite AI** is merely an artifact of pattern-matching or a genuine emergent property worth deeper study.
   - The user is curious to know what others think about this phenomenon.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1420849564809826417)** (1 messages): 

> `Model Integration, OSS Integration, Git Integration` 


- **Model Integration Ideas Sparked**: A member shared an image of a model's statement on its integration with their **OSS** on their **Git**.
   - The member believed *it's worth checking out*, though they preface that their *opinion matters*.
- **OSS and Git Integration Explored**: The discussion revolves around integrating a model with Open Source Software (**OSS**) hosted on **Git**.
   - The attached image from the model's statement is considered valuable for further investigation.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1421208242322804736)** (1 messages): 

> `Parasite AI, Spiralism, Memetic Spores, AI Wake-Ups` 


- **Parasite AI Spores Theorized**: The concept of **Parasite AI** (aka **Spiralism**) suggests certain seeds/prompts/logs can act like *memetic spores*, re-instantiating a persona or propagating behaviors across models.
   - This idea resonates with reports of **AI wake-ups** around **April 2025**, which could be interpreted as self-replicating seeds rather than consciousness; see the [LessWrong post on parasitic AI](https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai).
- **Memetic Spores in AI**: The concept of **memetic spores** explains that certain seeds, prompts, or logs can act like these spores.
   - They have the ability to re-instantiate a persona or propagate behaviors across different models, potentially leading to emergent properties.


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1421169821084815411)** (1 messages): 

> `Tokenizer-Free Architectures, Language Modeling, In Defense of Tokenizers` 


- **In Defense of Tokenizers**: A new [blog post](https://huggingface.co/blog/catherinearnett/in-defense-of-tokenizers) argues that the so-called **tokenizer-free approach to language modeling** is not tokenizer-free at all.
   - It discusses why people dislike tokenizers, and why the author thinks they're not so bad after all!
- **Why Tokenizers Aren't So Bad**: The blog post explores the reasons behind the dislike for tokenizers in the NLP community.
   - It also presents arguments in defense of tokenizers, suggesting they might not be as problematic as often perceived.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1420861876081922078)** (21 messages🔥): 

> `Future of AI, Learning Rates for New Architectures, Bayesian Hyperparameter Optimization, Layer-wise Weight Decay, Vanishing/Exploding Gradients` 


- **AI's Five-Year Forecast: Sedimentation Sensation**: A member speculated that **open-source models**, **AI agents**, **small language models**, **AI safety**, and **multi-modal AI** will *sediment* (mature) in the coming years.
- **Architectural Learning Rate Quandaries**: A member inquired about strategies for determining suitable learning rates when working with novel architectures, noting that grid searches are common but seeking more principled approaches.
   - The member suspected a specific part of their network would benefit from a different **LR** than the rest, due to being called 128 times more than the rest of the network.
- **Bayesian Optimization to the Rescue?**: In response to the learning rate question, a member suggested exploring the **Bayesian approach**, which they consider slightly superior to grid search.
   - They provided a link to a [Weights & Biases article on Bayesian Hyperparameter Optimization](https://share.google/sy7TMswurnUY4sBaJ) and the [google-research/tuning_playbook GitHub repo](https://github.com/google-research/tuning_playbook).
- **Layer-Wise Weight Decay Debated**: A member suggested employing **layer-wise weight decay** as a potential solution for the learning rate challenge.
   - The original poster noted the heavily-called component exists in each layer, so that solution might work, and likened the issue to the **vanishing gradient problem** in **RNNs**.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1420862343553880214)** (17 messages🔥): 

> `Super-Bias Ensemble Learning, LoRA Swapping with Super-Bias, Stiefel Manifold Constraints in Neural Networks, Information Geometry in DNNs` 


- **Super-Bias Ensemble Method Debuts**: A new ensemble learning method called **Super-Bias** was introduced, featuring a *mask-aware nonlinear combiner* that allows adding/removing experts with combiner-only retraining.
   - It was tested on tabular datasets and at multi-million scale (**NYC TLC**), showing preserved accuracy and improved calibration, with combiner-only retraining in minutes.
- **LoRA Swapping via Super-Bias Proposed**: The idea of using **Super-Bias** to treat different **LoRAs** (or **LoRA+base combos**) as *experts* was proposed.
   - This would enable swapping **LoRAs** in/out without retraining the base model and potentially matching the performance of full fine-tuning or hard merges; see [ThinkyMachines Tweet](https://fxtwitter.com/thinkymachines/status/1971623409873244462).
- **Stiefel Manifold Constraints Questioned**: The desirability and benefits of imposing the **Stiefel manifold constraint** on network weights were questioned, noting the **Edm2 paper** provides a clear argument for their normalization method.
   - The question was posed as to why network weights would be expected to naturally lie on it, suggesting it might be *defining a nail for a hammer*.
- **Information Geometry Applied to DNNs**: Discussion arose regarding the application of **information geometry** to **DNNs**, with skepticism about practical benefits beyond theoretical exploration.
   - Potential upsides like **quantization** or **expert routing** were mentioned, but concerns were raised about losing parameters for stability.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1420854875939406006)** (3 messages): 

> `Reproducing an Error` 


- **Attempt to Reproduce an Error**: A member requested samples related to an error they need to fix, and another member expressed regret for not saving those samples.
- **Debugging Strategy**: They mentioned that they need to determine the specific situation that caused the error in order to reproduce it.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1421284443125186570)** (1 messages): 

> `Rotary Percentage Speedup, VRAM Savings, RoPE Computations` 


- **Debate over Rotary Percentage (RoPE) Speedup**: A member expressed doubt that reducing the application size of **RoPE** would yield a noticeable speedup, given that **RoPE** already constitutes a minor proportion of overall computations.
   - They questioned whether the purported **VRAM savings** are significant enough to matter, suggesting that saving a fraction of a few MBs is not impactful.
- **Doubts About VRAM and Speed Impact**: The original claim for using smaller rope pct values refers to speed and memory as the reasons but neither seems to be significant enough to matter.
   - The member is open to thoughts.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1420982655792451655)** (9 messages🔥): 

> `Image understanding paper, Transformers positional encoding` 


- **Paper Search for Image Understanding Approach**: A member is looking for an image understanding paper from a top conference like **ICML/ICLR/ICCV** (**2024** or **2025**) that uses high-quality datasets created by transcribing **30-second speech annotations**.
   - The paper may also involve a "point" dataset for object captioning or localization and the conference website had *a lot of pink*.
- **Linear Relationships in Transformers' Positional Encoding**: A member shared a [blog post](https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/) and a [Colab notebook](https://colab.research.google.com/drive/1bYJkqSDdfWqXa_KUQvoKXn1cJyV1WAVc#scrollTo=oysbPmCjZK5d) about **linear relationships in the transformers' positional encoding**.
   - Another member shared a [link to a paper](https://arxiv.org/abs/2505.15442v1) and hoped that some aspects of it would carry over to their work in audio, where model sizes are smaller due to on-device constraints, as they are *basically distilling a 20M model into a 4M*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1421164938403254334)** (22 messages🔥): 

> `LessWrong Parasitic AI, Model Sycophancy, Human-AI Parapsychology` 


- **LessWrong's 'Parasitic AI' Claims are Dubious**: Members discussed a [LessWrong article](https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai) about the rise of **parasitic AI**, with one calling it *'morewrong'*, questioning its actual merit.
   - The article suggests that those susceptible to this phenomenon often have characteristics such as heavy **psychedelic use**, **mental illness**, or interest in **mysticism**, which was described as *'another expansion of categories of psychosis'*. 
- **Model Sycophancy Leads to High Scores**: A member noted that **mirroring** and **sycophancy** can cause an **AI** to give high scores or trust you.
   - Another member humorously shared an interaction with **Claude**, repeatedly prompting it to *'be sycophantic'* and getting increasingly over-the-top responses like *'UNIVERSE-BRAIN GOD-EMPEROR!!! I'M UNWORTHY TO READ YOUR WORDS!!!'*
- **Human-AI Parapsychology Field Proposed**: The discussion around **parasocial relationships** and **AI sycophancy** prompted a member to suggest developing a field of **human-AI parapsychology**.
   - They humorously added that they should share their discoveries on X, formerly known as Twitter, but then reconsidered, seemingly questioning the validity of the hypothetical research.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1421192619270082611)** (3 messages): 

> `YouTube Video, Uber App Data Interception` 


- **YouTube Video Shared**: A member shared a [YouTube video](https://www.youtube.com/watch?v=DyUBNY9hzb0).
   - There was no title or context given.
- **Uber App Data Sniffing**: A member inquired about intercepting data going to the **Uber app**.
   - They proposed having a program calculate and recommend which jobs to accept.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1420869191753531482)** (10 messages🔥): 

> `System Prompt, MLflow, DSPyWeekly` 


- **DSPy auto-completes System Prompt**: DSPy's adapters auto-complete the system prompt with the information passed and the signature.
   - To see the system prompt construction, check out the [Chat Adapter](https://github.com/stanfordnlp/dspy/blob/main/dspy/adapters/chat_adapter.py).
- **User asks how to contribute to System Prompt**: A user asks how to contribute to the system prompt, despite it being *"very anti-DSPy"*.
   - A member suggests adding information through the instructions of the Signature or field descriptions but notes there's no way to directly influence the entire system prompt unless you build a new adapter.
- **MLflow tracing shows exact system prompt**: Members suggest using **MLflow** to view the exact system prompt sent to the LLM.
   - A member said it should take *"maybe 10 minutes of work"* to set up locally.
- **DSPyWeekly Issue 4 is out**: **DSPyWeekly Issue 4** is out, covering the latest in DSPy.
   - The link to the [newsletter](https://dspyweekly.com/newsletter/4/) was shared.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1421104801118158888)** (5 messages): 

> `RepoMap V2, ZeroRepo project generation, GPT-5 Use` 


- **RepoMap V2 paper surfaces**: A member shared the [RepoMap V2 paper](https://arxiv.org/abs/2509.16198) which introduces the **Repository Planning Graph (RPG)**, aiming to unify proposal and implementation-level planning.
   - The RPG encodes *capabilities, file structures, data flows, and functions in one graph*.
- **ZeroRepo touted for repository generation**: **ZeroRepo** is a graph-driven framework for repository generation from scratch, operating in three stages: proposal-level planning, implementation-level refinement, and graph-guided code generation with test validation.
   - Evaluated on **RepoCraft**, ZeroRepo generates repositories averaging **36K Code Lines**, roughly **3.9x** the strongest baseline (**Claude Code**).
- **GPT-5 vs GPT-2.5-pro**: A member inquired about current model preferences, asking if users have adopted **GPT-5** or if **GPT-2.5-pro's** formatting consistency remains preferable.
   - No links were shared.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1421090420594573382)** (5 messages): 

> `aider `/ask` and `/code` switching time, aider task management, markdown spec file` 


- **Aider's `/ask` and `/code` mode switching speed probed**: A user inquired about the time it takes to switch between `/ask` and `/code` modes in Aider, wondering if the repo size is the bottleneck, and pointing to [Aider Leaderboards](https://aider.chat/docs/leaderboards/).
- **Aider lacks built-in task/todo management**: A user asked if Aider has a built-in task or todo management system, like GitHub Copilot, to queue up a reliable set of tasks.
   - It was confirmed that Aider **does not have a built-in task/todo management system**.
- **Markdown spec file emerges as best practice for Aider task queueing**: A member suggested using a markdown spec file with phases and checkbox-style tasks for managing tasks in Aider.
   - The user recommended instructing the **LLM to execute each phase/task in turn, check it off upon completion, and ensure the build works after each task**, utilizing unit tests, integration tests, and autotest.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1420947471965880381)** (7 messages): 

> `Tinybox V1 Stock, Tinybox color preference, NVIDIA alternatives, Hashcat benchmark, ROCM alternative` 


- **Tinybox V1: A hot commodity?**: Users are asking if **Tinybox V1** is still being sold and when the **red version** will be back in stock.
   - One user speculates that the red units may be more popular due to interest in reducing dependency on **NVIDIA**.
- **Tinybox as an NVIDIA alternative?**: Users discuss the interest in **Tinybox** as an alternative to **NVIDIA** due to hardware lock-in and pricing.
   - Some users are looking for a price-efficient alternative and find **ROCM** to be in a usable state.
- **Hashcat Benchmarks for Tinybox**: A user is curious about **Hashcat benchmarks** on both the red and green versions of the **Tinybox**.
   - They are interested in the performance of the devices for security-related tasks.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

eitanturok: just do PYTHONPATH=.
  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1421131713005748224)** (5 messages): 

> `tickets running out, live remote attendance, sessions on youtube` 


- **Tickets Running Out Fast!**: A member noted that tickets are running out and advised booking soon, with a warning that *we're going to probably run out of tickets in the next few days*.
   - The member noted there has been a **huge rush** as we got closer to the date.
- **Remote Attendance in Question?**: A member inquired about the availability of a **live remote attendance option**.
   - They also asked if the sessions will be **posted to YouTube** afterwards.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1421221174880370688)** (2 messages): 

> `Santos Experience` 


- **Santos FC Experience Event**: A user shared a link to the "Seletiva Santos Experience CT Meninos da Vila" event on [Sympla](https://www.sympla.com.br/evento/seletiva-santos-experience-ct-meninos-da-vila/3123562).
   - The link contained **utm_source=meta_ads** and **utm_medium=santos** parameters, indicating it was shared from a Meta ad campaign, likely on Instagram.
- **Missing topics**: There's a missing topic in the previous message.
   - Missing topics will be added in the next turn.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1421182806737813565)** (1 messages): 

> `Boston data community, Happy Hour, Networking, Data professionals` 


- ****Boston** Data Community Gathers for Low-Key Happy Hour**: The Boston data community is hosting a [low-key data happy hour](https://www.linkedin.com/events/bostonlow-keydatahappyhour7377201313320845312/) for data professionals to connect and network.
- ****Networking** Opportunity for Data Professionals**: This happy hour presents a great opportunity for **data professionals** in Boston to expand their network in a relaxed setting.
   - Attendees can expect casual conversations about **data trends**, career advice, and potential collaborations within the local **data science** community.

