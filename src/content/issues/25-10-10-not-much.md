---
id: MjAyNS0x
title: not much happened today
date: '2025-10-10T05:44:39.731046Z'
description: >-
  **FrontierMath Tier 4** results show **GPT-5 Pro** narrowly outperforming
  **Gemini 2.5 Deep Think** in reasoning accuracy, with concerns about problem
  leakage clarified by **Epoch AI Research**. **Mila** and **Microsoft** propose
  **Markovian Thinking** to improve reasoning efficiency, enabling models to
  reason over 24K tokens with less compute. New research suggests base models
  inherently contain reasoning mechanisms, with "thinking models" learning to
  invoke them effectively. In systems, **NVIDIA Blackwell** combined with
  **vLLM** wins InferenceMAX with significant throughput gains, while **Together
  AI's ATLAS** adaptive speculative decoding achieves 4× speed improvements and
  reduces RL training time by over 60%. **SparseServe** introduces dynamic
  sparse attention with KV tiering, drastically improving throughput and latency
  in GPU memory management.
companies:
  - openai
  - google-deepmind
  - microsoft
  - epoch-ai-research
  - togethercompute
  - nvidia
  - mila
models:
  - gpt-5-pro
  - gemini-2.5
  - vllm
  - deepseek-v3.1
topics:
  - reasoning
  - reinforcement-learning
  - inference
  - speculative-decoding
  - sparse-attention
  - kv-cache-management
  - throughput-optimization
  - compute-efficiency
  - tokenization
people:
  - epochairesearch
  - yitayml
  - _philschmid
  - jiqizhixin
  - cvenhoff00
  - neelnanda5
  - lateinteraction
  - mgoin_
  - blackhc
  - teortaxestex
---


**a quiet day**

> AI News for 10/9/2025-10/10/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (197 channels, and 7403 messages) for you. Estimated reading time saved (at 200wpm): 586 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Second round [**applications for AIE CODE**](https://apply.ai.engineer/) close in 5 days!

---

# AI Twitter Recap

**Reasoning: FrontierMath shootout, Markovian Thinking, and what “reasoning training” actually teaches**

- FrontierMath Tier 4 results: In compute-heavy settings, GPT-5 Pro set a new record at 13% accuracy, edging out Gemini 2.5 Deep Think by a single problem (not statistically significant). Grok 4 Heavy lags. Epoch clarifies leakage concerns: OpenAI has access to 28/48 problems; 5 of GPT‑5 Pro’s 8 solves were on the held-out set. See the full thread from [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1976685685349441826), with context and methodology in follow-ups ([held-out details](https://twitter.com/EpochAIResearch/status/1976685757369851990), [historical totals](https://twitter.com/EpochAIResearch/status/1976685769130705300)). Gemini 2.5 Deep Think’s strong performance is also highlighted by [@YiTayML](https://twitter.com/YiTayML/status/1976470535308734575) and [@_philschmid](https://twitter.com/_philschmid/status/1976626257090535432). FrontierMath site: [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1976685780862144978).
- Markovian Thinking (Delethink): Mila + Microsoft propose training models to “write state” at fixed boundaries, decoupling reasoning length from context size—turning reasoning into linear compute. An R1‑Distill 1.5B model reasons up to 24K tokens with only 8K context, beating LongCoT-RL trained on full 24K at ~4× lower compute (7 vs 27 H100‑months). Coverage by [@jiqizhixin](https://twitter.com/jiqizhixin/status/1976466786565656986) and a summary + links by [@TheTuringPost](https://twitter.com/TheTuringPost/status/1976798665038758377) ([efficiency details](https://twitter.com/TheTuringPost/status/1976798717094379588), [paper/code](https://twitter.com/TheTuringPost/status/1976798729274544403)).
- What reasoning training actually teaches: New work argues base models already contain reasoning mechanisms; “thinking models” learn when to invoke them. Invoking skills at the right time recovers up to 91% of the gap between base and reasoning models. See the thread by [@cvenhoff00](https://twitter.com/cvenhoff00/status/1976633766811734461) and commentary from [@NeelNanda5](https://twitter.com/NeelNanda5/status/1976660983084130377) ([follow-ups](https://twitter.com/NeelNanda5/status/1976710233692012619)).
- Caution on RL-on-math generalization: Several results rely on Qwen bases that are already heavily mid-trained for math—be careful extrapolating broad claims from this setup alone ([@lateinteraction](https://twitter.com/lateinteraction/status/1976761442842849598)).

**Systems and inference: Blackwell + vLLM, adaptive speculators, and sparse-attention KV tiering**

- NVIDIA Blackwell + vLLM wins InferenceMAX: vLLM shows strong Pareto gains via deep joint work with NVIDIA—100+ PRs across the stack, FP4/FP8 kernels, async scheduling, graph fusions, and FlashInfer integration—with another 2–3× throughput expected from speculative decoding and Data + Expert Parallel (DEP). Summaries from [@mgoin_](https://twitter.com/mgoin_/status/1976452383258648972) and [@NVIDIAAIDev](https://twitter.com/NVIDIAAIDev/status/1976686560398426456) (see [benchmark stream](https://twitter.com/SemiAnalysis_/status/1976669740035977702)).
- ATLAS (Together AI): An adaptive speculative decoding system that learns from your live traffic; reported 4× faster vs baseline (500 TPS on DeepSeek‑V3.1) and improving with usage. Threads: [@togethercompute](https://twitter.com/togethercompute/status/1976655646474031362) ([adaptive explainer](https://twitter.com/togethercompute/status/1976655647925215339), [results](https://twitter.com/togethercompute/status/1976655649120612525)), [@tri_dao](https://twitter.com/tri_dao/status/1976692444977938499). Early reports suggest >60% RL training time reduction via self-adaptive speculators ([@BlackHC](https://twitter.com/BlackHC/status/1976730114902851908)); coverage in VB: [link](https://twitter.com/togethercompute/status/1976743626685530540).
- SparseServe for Dynamic Sparse Attention: With DSA, the bottleneck shifts from HBM bandwidth to HBM capacity due to KV cache residency. SparseServe introduces HBM↔DRAM KV tiering (GPU FlashH2D, CPU FlashD2H), working-set–aware dynamic batching, and layer-segmented prefill—achieving 9.26× lower TTFT and 3.14× higher throughput vs SOTA in vLLM-based tests. Overview by [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1976544233700929614); hardware implications noted by [@teortaxesTex](https://twitter.com/teortaxesTex/status/1976556643031933352).
- Kernel velocity > “general hardware”: Expect more custom kernels (MoEs, low‑precision matmuls, attention variants, SSMs) as Triton lowers the barrier and high-level overheads dominate at Blackwell speeds ([@awnihannun](https://twitter.com/awnihannun/status/1976715815019037101)).

**Model and tooling releases**

- Qwen3‑VL Cookbooks: A polished set of notebooks for local/API use across multimodal tasks—computer use, omni recognition, doc parsing/OCR, 3D grounding, video understanding, mobile agents, long doc understanding, spatial reasoning, and more. Links inside the post by [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1976479304814145877).
- Speech-to-speech: GPT Realtime Mini (OpenAI) is ~7× cheaper than flagship Realtime, cuts TTFA to 0.81s (from 1.27s), doubles context to 32k, and adds image input—positioned for scalable agents over WebRTC/WebSocket/SIP. Comparative analysis vs Gemini 2.5 Flash Native Audio Dialog by [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1976696262083985636) ([chart](https://twitter.com/ArtificialAnlys/status/1976696264080474365), [model explorer](https://twitter.com/ArtificialAnlys/status/1976696265754001747)).
- Small, fast, open vision: Moondream 3 (9B, 64‑expert MoE, ~2B active) adds native pointing, improved OCR, and 32K context—optimized for UI understanding and agent workflows. Announcements by [@moondreamai](https://twitter.com/moondreamai/status/1976624914070401142) and preview on FAL: [@fal](https://twitter.com/fal/status/1976682702167228919).
- Agentic coding: KAT‑Dev‑72B‑Exp (Kwaipilot) ranks #2 on SWE‑Bench Verified; tuned via mid‑training → SFT+RFT → Agentic RL; fits on 4× RTX 3090 @ 4‑bit ([@TheAhmadOsman](https://twitter.com/TheAhmadOsman/status/1976606921756205531)).
- RL post‑training with LoRA/QLoRA/DoRA/QDoRA: Tora (built on torchtune) unifies GRPO, FSDP, compile support; enables stable 4‑bit RL (QLoRA/QDoRA) and speeds rollouts 2–4× with DoRA‑Cache ([@gm8xx8](https://twitter.com/gm8xx8/status/1976443792850092464)).
- Tooling quick hits: LangSmith now supports JS code evals in addition to Python for faster, stack‑native evaluations ([@LangChainAI](https://twitter.com/LangChainAI/status/1976700402105233603)); LangChain v1 ships a customizable create_agent and middleware hooks for pre/post model/tool calls ([@sydneyrunkle](https://twitter.com/sydneyrunkle/status/1976751776620593564), [hooks explainer](https://twitter.com/sydneyrunkle/status/1976753314462417344)); LlamaIndex adds explainable document classification with custom rules ([@llama_index](https://twitter.com/llama_index/status/1976686683468026337)); Glass Health launches a production Developer API with HIPAA compliance and citation metadata ([@GlassHealthHQ](https://twitter.com/GlassHealthHQ/status/1976713436773138599)).

**Scale, compute, and training estimates**

- Tokens processed per month: Google at ~1.3 quadrillion, OpenAI ~260T, Groq ~50T per [@sundeep](https://twitter.com/sundeep/status/1976475987962626062); Google’s Demis Hassabis reiterates 1.3 quadrillion tokens/mo ([@demishassabis](https://twitter.com/demishassabis/status/1976712484657475691)). Note tokens vary in information density and usefulness across models/vocabs/tasks ([@awnihannun](https://twitter.com/awnihannun/status/1976676812022550864)).
- Where compute goes: Epoch estimates OpenAI spent ~$7B on compute last year; most on R&D (experiments/failed runs), with final training runs <~$1B ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1976714284349767990) and [follow-ups](https://twitter.com/EpochAIResearch/status/1976714297255588053)).
- GPT‑5 training speculation: Rough external estimates suggest ~100B active params, 30–100T tokens, RL 10–100% of pretraining, totaling ~6e25 FLOPs ([@teortaxesTex](https://twitter.com/teortaxesTex/status/1976441366969532888)). Separate MoE sparsity chatter implies very high total parameters with small active subsets (e.g., 256–1024 experts with 4–8 active), though some argue actual active counts and costs may be lower than headlines ([analysis thread](https://twitter.com/teortaxesTex/status/1976773126584516801)).

**Robotics and embodied AI**

- Retargeted acrobatics on hardware: Using OmniRetarget + BeyondMimic minimal RL tracking, a humanoid executed a wallflip with a 5/5 success rate; training required only minor tweaks (e.g., relaxing terminations, adjusting rewards) ([@zhenkirito123](https://twitter.com/zhenkirito123/status/1976663920552427619)). Separately, Unitree G1 reproduced a signature taekwondo spin-kick, with sim‑to‑real issues (IMU gyro saturation) resolved via tuning ([@kevin_zakka](https://twitter.com/kevin_zakka/status/1976460408077812085)).
- Vision for agents: Moondream 3 targets real‑world UIs and structured perception for downstream agent frameworks (see Model releases above).

**Evals, benchmarking, and governance**

- Benchmarking reform: “Benchmarking is Broken—Don’t Let AI be its Own Judge” proposes PeerBench, a community‑governed, proctored evaluation blueprint: sealed execution, rolling item banks, delayed transparency ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1976586775603851344)).
- CoT transparency: Labs should disclose whether/how they train against chain‑of‑thought, argues [@RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/1976686565654221150), citing METR’s GPT‑5 eval ([details](https://twitter.com/RyanPGreenblatt/status/1976686576521679252)). Suggestion: third‑party disclosure to trusted evaluators if IP is sensitive ([follow‑up](https://twitter.com/RyanPGreenblatt/status/1976686574080491880)).
- OpenBench expands: Groq’s OpenBench now supports ARC‑AGI, widening standardized eval coverage for reasoning benchmarks ([@GregKamradt](https://twitter.com/GregKamradt/status/1976718318544601573)).
- “Evals in the wild” and goalpost shift: Practical evaluation beyond static test sets is increasingly emphasized ([@lateinteraction](https://twitter.com/lateinteraction/status/1976439833158615345)); the cultural shift from toy tests to sustained autonomy and economic impact was aptly summarized by [@aidan_mclau](https://twitter.com/aidan_mclau/status/1976658416451149874).
- Governance dispute (context only): An Encode GC thread about an OpenAI subpoena sparked internal and external commentary; see perspectives from [@_NathanCalvin](https://twitter.com/_NathanCalvin/status/1976649051396620514), OpenAI’s [@jasonkwon](https://twitter.com/jasonkwon/status/1976762546041634878), and critical concerns from [@jachiam0](https://twitter.com/jachiam0/status/1976690339546112098). Keep an eye on impacts to policy discussions and openness norms.

**Top tweets (by engagement)**

- “Interviewed an engineer… I’m 99% sure he was using [an AI helper]” — hiring signal shift and evaluation hygiene from [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1976700504152719435).
- Stunning real‑world wallflip via OmniRetarget + BeyondMimic from [@zhenkirito123](https://twitter.com/zhenkirito123/status/1976663920552427619).
- Demis: Google processed ~1.3 quadrillion tokens in a month ([@demishassabis](https://twitter.com/demishassabis/status/1976712484657475691)).
- “2024 evals vs 2025 evals” meme capturing the shift to long‑horizon, impact‑focused metrics ([@aidan_mclau](https://twitter.com/aidan_mclau/status/1976658416451149874)).
- Subpoena thread that catalyzed policy discourse: [@_NathanCalvin](https://twitter.com/_NathanCalvin/status/1976649051396620514).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

nothing met our bar

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. NVIDIA GB300 NVL72 + ComfyUI GDS Performance Updates

- [**Microsoft unveils the first at-scale NVIDIA GB300 NVL72 cluster, letting OpenAI train multitrillion-parameter models in days instead of weeks**](https://www.reddit.com/r/singularity/comments/1o2t53m/microsoft_unveils_the_first_atscale_nvidia_gb300/) (Activity: 424): **Microsoft/Azure announced the first production NVIDIA GB300 NVL72 cluster (NDv6 GB300) for OpenAI, spanning >4,600 Blackwell Ultra GPUs; each NVL72 VM fuses 72 GPUs via NVLink Switch fabric (130 TB/s per rack) into a unified 37 TB accelerator delivering** `1.44 exaflops FP4` **per VM, with rack‑to‑rack scale-out over Quantum‑X800 InfiniBand at** `800 Gb/s per GPU` **([source](http://blogs.nvidia.com/blog/microsoft-azure-worlds-first-gb300-nvl72-supercomputing-cluster-openai/?linkId=100000386364404)). The stack targets low‑precision training/inference using NVFP4 and the NVIDIA Dynamo compiler, plus SHARP v4 and ConnectX‑8 SuperNICs with adaptive routing/telemetry‑based congestion control, and Azure cites MLPerf Inference v5.1 leadership (e.g., up to** `5×` **throughput on a** `671B` **reasoning model vs. Hopper). Back‑of‑envelope: >4,600 GPUs ≈ ~64 NVL72 VMs → O(~**`92 exaFLOPS FP4`**) aggregate peak; note FP4 metrics aren’t directly comparable to FP64 TOP500/HPL exaflop systems.** Top comments ask for apples‑to‑apples comparisons across systems (suggesting normalizing by precision, per‑GPU flops, bisection bandwidth, fabric/NIC speeds, and MLPerf results), speculate this enables multi‑trillion‑parameter training timelines, and debate data vs. parameter scaling—pointing out MoE can scale parameters while active compute stays modest, while “Chinchilla‑optimal” dense models need tokens on the order of tens of trillions (implying ~0.1–1T parameter dense models unless augmented with synthetic/private data).
    - On scale/topology: an NVL72 is a 72‑GPU rack‑scale “island” wired by NVSwitch so all GPUs share a single high‑bandwidth NVLink domain; multiple islands are then stitched over InfiniBand/Ethernet in Azure. So numbers like “~4,608 GPUs” correspond to ~64 NVL72 racks (64×72), whereas “hundreds of thousands” would imply thousands of such islands across regions. The key is that most tensor/pipeline parallel traffic stays intra‑island (orders of magnitude faster than inter‑island), which is why NVL72 matters for large model training; see NVIDIA’s GB200 NVL72 overview for topology specifics: https://www.nvidia.com/en-us/data-center/products/gb200-nvl72/ .
    - How to compare to other supercomputers: TOP500/Green500 rank FP64 Linpack and don’t reflect AI training (mixed‑precision) or communication patterns. For AI, compare (a) per‑GPU AI FLOPs and HBM capacity, (b) intra‑island bisection bandwidth (NVLink/NVSwitch) versus interconnect (400/800G InfiniBand/ROCE), and (c) end‑to‑end MLPerf Training times at scale; NVLink/NVSwitch islands generally reduce gradient‑sync overhead compared to Ethernet/InfiniBand‑only designs. Relevant baselines are multi‑rack H100/MI300 systems in MLPerf v3.x (https://mlcommons.org/en/training-results-3-1/), where topology often dominates scaling beyond a few thousand GPUs.
    - On “how many parameters/data exist?”: after deduplication/quality filters, estimates put the total high‑quality text+code corpus at O(10–30T) tokens (see Epoch AI’s analysis: https://epochai.org/blog/how-much-text-is-there). By the Chinchilla scaling law, optimal dense models use ~`20×` tokens per parameter, implying a 1T‑parameter dense model ideally needs ~20T tokens—near the upper bound—so “multi‑trillion‑parameter” training typically means MoE where only `~1–2` experts are active per token, keeping active parameters ~100–200B while leveraging a much larger total parameter pool (Chinchilla: https://arxiv.org/abs/2203.15556, Switch Transformers MoE: https://arxiv.org/abs/2101.03961).
- [**We can now run wan or any heavy models even on a 6GB NVIDIA laptop GPU | Thanks to upcoming GDS integration in comfy**](https://www.reddit.com/r/StableDiffusion/comments/1o2wklr/we_can_now_run_wan_or_any_heavy_models_even_on_a/) (Activity: 814): **Developer Maifee integrated NVIDIA GPUDirect Storage (GDS) into [ComfyUI](https://github.com/maifeeulasad/ComfyUI) to stream model weights directly from NVMe to GPU VRAM (cuFile DMA), enabling heavy models to run on GPUs with as little as** `6 GB` **VRAM without custom offloaders or quantization. Test via:** `git checkout offloader-maifee` **and run** `python3 main.py --enable-gds --gds-stats`**; a merge request is under review for upstreaming. GDS (see NVIDIA docs: https://developer.nvidia.com/blog/gpudirect-storage/) bypasses CPU/host RAM, but practical throughput/latency is bounded by NVMe + PCIe and model access patterns; requires Linux with compatible drivers (nvidia-fs), CUDA, and supported filesystems (ext4/xfs).** Commenters ask how GDS differs from RAM offloading: GDS provides a zero-copy NVMe→GPU DMA path that avoids CPU mediation and page cache, whereas RAM offload stages tensors in system memory and incurs extra copies; performance depends on storage/PCIe limits. Noted constraint: it’s Linux-only at the moment.
    - GPUDirect Storage (GDS) enables DMA from NVMe SSDs directly into GPU VRAM, bypassing the CPU and host RAM copy path. Practically, this shifts the data path from SSD → RAM → GPU to SSD → GPU, lowering CPU involvement and achieving ~4–5 GB/s effective SSD→GPU reads (limited by NVMe/PCIe), versus ~3–5 GB/s when double-copying through RAM; GPU reads from host RAM are still capped by PCIe (e.g., PCIe 4.0 x8 ≈ ~16 GB/s bidirectional, ~8 GB/s per direction). It improves I/O efficiency, not compute, and is currently Linux-only; see NVIDIA’s docs: https://developer.nvidia.com/gpudirect-storage.
    - GDS does not reduce VRAM requirements, so it won’t avoid OOM on a 6 GB GPU for a 14 GB model—the active parameters/activations must still fit in VRAM at compute time. Running oversized models relies on offloading/partitioned execution (e.g., CPU offload, layer-wise streaming), often aided by frameworks like Hugging Face Accelerate, DeepSpeed, or quantized runtimes like llama.cpp; GDS can speed these pipelines by accelerating SSD→GPU transfers but doesn’t change the memory footprint.
    - Where GDS helps most: out-of-core workflows and low-RAM systems that stream model weights/activations from fast NVMe, reducing CPU overhead and avoiding extra copies. If ample RAM is available, preloading models into RAM and transferring over PCIe can be faster than streaming from SSD due to RAM’s higher bandwidth; the tradeoff is I/O path efficiency and CPU load versus raw media speed. Net benefit is workload- and platform-dependent and warrants benchmarking.

### 2. AniSora V3.2 (Wan2.2) 360° I2V and Sora-2 Demos

- [**360° anime spins with AniSora V3.2**](https://www.reddit.com/r/StableDiffusion/comments/1o2qjiw/360_anime_spins_with_anisora_v32/) (Activity: 594): [**AniSora V3.2](https://github.com/bilibili/Index-anisora) is an anime-focused image-to-video model built on Wan2.2 I2V that plugs directly into the ComfyUI Wan2.2 workflow; loading an input illustration into the FLF2V graph and applying the repo’s recommended prompt yields out‑of‑the‑box “360° character turnarounds” with smooth rotation, strong flat-illustration fidelity, and preserved line detail. The shared workflow and example are provided here: [🦊AniSora V3#68d82297000000000072b7c8](https://scrapbox.io/work4ai/%F0%9F%A6%8AAniSora_V3#68d82297000000000072b7c8).** Commenters note naming confusion (Wan-based yet called “AniSora”) but praise results; they suggest this could enable high-throughput multi‑view data for 3D asset pipelines and ask about generalization to non‑realistic styles for consistent multi‑view generation.
    - A user reports a reproducible ComfyUI stability/memory issue: on a `24GB` VRAM GPU the AniSora V3.2 workflow completes the High KSampler pass but crashes ComfyUI when loading the LOW model, despite peak VRAM showing only ~`19.5GB`. They tried inserting a cleanup node to unload the HIGH model before the LOW stage with no success and asked which ComfyUI version the author used, suggesting potential version-specific model-loading/GC/fragmentation behavior across the two-stage HIGH→LOW pipeline.
    - Several commenters probe whether the 360° anime spins can provide consistent multi-views for downstream 3D pipelines (video-to-3D, NeRF/GS-style reconstruction) and how robust it is on non‑photorealistic inputs. The idea is to exploit temporally/stylistically consistent rotations to improve multi-view supervision for 3D model generation, potentially enabling higher-quality reconstructions of anime-styled assets compared to casual, less consistent view sampling.
- [**Hyperspace and Beyond**](https://www.reddit.com/r/singularity/comments/1o36ptd/hyperspace_and_beyond/) (Activity: 793): **Non-technical meme. The image titled “Hyperspace and Beyond” appears to satire low-effort, hype-y reactions (e.g., posting a hyperspace/flashy GIF) on research-paper threads rather than substantive engagement; there are no model details, benchmarks, or technical claims to assess.** Comments criticize a user who habitually drops a GIF on paper posts without reading/understanding them, and mock karma-chasing over contributing meaningful discussion; another commenter agrees “unironically.”
    - A commenter suggests replacing a monolithic metallic body with a swarm of nanobots that can detect threats/damage and reconfigure within `microseconds`, essentially a form of [programmable matter](https://en.wikipedia.org/wiki/Programmable_matter)/[claytronics](https://www.cs.cmu.edu/~claytronics/). While microsecond-scale local sensing/actuation is plausible for MEMS/NEMS (e.g., fast piezoelectric actuators: [ref](https://en.wikipedia.org/wiki/Piezoelectric_actuator)), the hard problems are power density/delivery, heat dissipation, coordination/communication latency, fault tolerance across massive populations, and manufacturing yield at the "zillion" scale. Related work in [active/metamaterials](https://en.wikipedia.org/wiki/Metamaterial) and [self-healing materials](https://en.wikipedia.org/wiki/Self-healing_material) explores pieces of this vision, but fully adaptive nanoscale swarms at that reactivity remain beyond current engineering capabilities.
- [**Figure doing housework, barely. Honestly would be pretty great to have robots cleaning up the house while you sleep.**](https://www.reddit.com/r/singularity/comments/1o2u46w/figure_doing_housework_barely_honestly_would_be/) (Activity: 1439): **Post shares a short clip of a Figure humanoid robot attempting basic household cleaning, described as working only “barely,” suggesting early-stage general-purpose manipulation in an unstructured home setting. The linked video ([v.redd.it/ob954u6uh8uf1](https://v.redd.it/ob954u6uh8uf1)) is inaccessible (**`HTTP 403 Forbidden`**), so no verifiable details on autonomy level (teleop vs. onboard), task success rates, or control stack are available. OP muses about unattended, overnight operation (cleaning while users sleep), but no safety or reliability data are provided.** Top comments anticipate rapid capability improvement “in a couple of years,” while others flag human–robot interaction and safety/UX concerns about night-time operation (startle risk), with the rest being humorous/off-topic.
- [**This is cheating at this point 😂**](https://www.reddit.com/r/ChatGPT/comments/1o2onjk/this_is_cheating_at_this_point/) (Activity: 17459): **A viral Reddit post shares a likely OpenAI Sora text‑to‑video clip depicting “Jesus Christ” in a water‑sport context, with commenters noting its realism. The linked asset is hosted on v.redd.it and currently returns** `HTTP 403 Forbidden` **for unauthenticated requests ([video link](https://v.redd.it/yasrp8dd07uf1)), indicating Reddit’s network‑security block (login/OAuth required). Context: Sora is a generative video model capable of photorealistic, physics‑consistent clips; curated samples are public while general access remains limited ([OpenAI Sora](https://openai.com/sora)).** A top comment notes, “These Sora clips are really starting to fool people,” pointing to increasing believability among non‑technical audiences (e.g., on Facebook) and implicit concerns about provenance/deepfake risk; other comments are largely humorous puns.
    - A commenter flags that **OpenAI’s Sora** text-to-video outputs are now photorealistic enough to fool casual viewers on social platforms, highlighting a detection problem as *“40+ year olds”* share them. Sora can generate up to `~60s` clips at high resolution with strong temporal coherence and reasonably plausible physics, reducing older telltale artifacts ([openai.com/sora](https://openai.com/sora)). This raises the need for more robust forensic cues (e.g., motion/physics edge cases, specular highlight consistency, water/splash dynamics) and provenance tools to reliably label synthetic media.
- [**I'm sorry but this is some of the funniest Al I've seen yet.**](https://www.reddit.com/r/ChatGPT/comments/1o2w0fm/im_sorry_but_this_is_some_of_the_funniest_al_ive/) (Activity: 3223): **The post links to a Reddit-hosted video ([v.redd.it](https://v.redd.it/7c7g9vy949uf1), returns 403 when unauthenticated) that appears to use AI-generated audio dubbed over real footage—commenters note “Only audio is AI.” The AI track is a comedic voice clone of “Zelensky” debating definitions of a “micropenis,” indicating straightforward TTS/voice-cloning rather than a full video deepfake; a still preview is shared ([image](https://preview.redd.it/p7kc6a30y9uf1.jpeg?width=320&format=pjpg&auto=webp&s=cfc48f8f941d84a4d475bb1f99d64459aae6558d)).** Commenters primarily verify modality (audio-only synthesis) and react to the humor; there’s no mention of specific models, pipelines, or artifacts beyond noting the likely audio dub.
    - A commenter asks if only the audio is AI; technically, many viral clips keep the real video while swapping in cloned speech via TTS/voice conversion (e.g., **ElevenLabs**, **Microsoft VALL-E** can mimic a voice from `3–60s` of reference audio). Without video re-synthesis, phoneme–lip desync is a tell (mouth shapes won’t match consonants/vowels); if the video is faked too, models like [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) or [SyncNet](https://www.robots.ox.ac.uk/~vgg/software/lipsync/) can align mouths to the synthetic track. Real-time voice conversion such as [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) enables low-latency dubbing, making audio-only spoofs particularly easy to produce.
    - A second theme is media authenticity: *“this will invalidate everything real”*. The risk is that high-quality AI audio is cheap and ubiquitous, while detectors remain unreliable on short/noisy clips; provenance schemes like **C2PA** [Content Credentials](https://c2pa.org/) and audio watermarks (Meta’s [AudioSeal](https://ai.meta.com/research/publications/audioseal-synthetic-speech-watermarking/), Google’s [SynthID](https://deepmind.google/technologies/synthid/)) help but are fragile under re-encoding/transformation and can be removed. Near-term, robust verification depends more on signed capture pipelines and distribution metadata than on post-hoc classification, which has nontrivial false positive/negative rates.

### 3. Delivery Fails: DoorDash Porch Collapse and Amazon Drops

- [**600lb DoorDasher falls through the porch floor 😂**](https://www.reddit.com/r/aivideo/comments/1o2qdyg/600lb_doordasher_falls_through_the_porch_floor/) (Activity: 2401): **The linked media is a Reddit-hosted video at [v.redd.it/cetb4u6kf7uf1](https://v.redd.it/cetb4u6kf7uf1) that currently returns** `HTTP 403 Forbidden` **without authentication, indicating a Reddit network-security block; access requires a logged-in session or token-based API credentials per Reddit’s guidance (login/support linked from the 403 page). The post title alleges a** `~600 lb` **DoorDash courier falls through a porch floor, but without media access the claim and any technical metadata (e.g., duration, codec, timestamps) cannot be verified.** Commenters speculate about AI-generated video realism—one notes AI “got the physics … down,” implying a debate over simulated vs. real-world physics fidelity—while others frame it as likely viral content; no concrete technical evidence is provided in-thread.
    - A couple of commenters suggest the clip is AI‑generated (one names **Sora 2**) and note how convincingly it models mass/force and structural failure—“AI really got the physics … to a tee.” This implies stronger learned physics priors and temporal coherence in modern video‑gen models (e.g., object permanence, momentum conservation, contact dynamics) compared to earlier generations that often broke under long horizons, making outputs increasingly hard to distinguish from real footage.
- [**Amazon Delivery**](https://www.reddit.com/r/aivideo/comments/1o2p45q/amazon_delivery/) (Activity: 481): **Video post titled “Amazon Delivery,” hosted on [v.redd.it](https://v.redd.it/1ubc2u6847uf1), currently returns** `403 Forbidden` **without authentication, indicating access control (OAuth/cookie) or WAF enforcement; troubleshoot by logging in or ensuring proper headers (e.g.,** `User-Agent`**,** `Referer`**). The clip appears to depict a dog jumping through a window and colliding with a bush, with commenters noting surprisingly plausible foliage–body interaction (deformation, occlusion, momentum transfer)—a challenging edge case often underrepresented in video-gen training data—suggesting improved physical coherence in contact dynamics.** While some comments are playful (e.g., “r/PackageDelivered”), a technical critique notes that despite a Bourne-like leap, the dog’s awkward struggle in the bush reveals remaining limits in control/footing realism, hinting at gaps in fine-grained physics and affordance modeling.
    - A commenter notes the plausibility of the dog's interaction with the bush—a hard case for generative video/scene synthesis because of **deformable foliage**, heavy self-occlusion, and limited representation in training data—implying the model generalizes contact dynamics and secondary motion reasonably well. They add that while experts might spot artifacts (e.g., collision/penetration errors, inconsistent leaf deformation, or temporal incoherence), to non-experts the physics appears convincing, indicating strong learned priors despite likely gaps in explicit physical modeling.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Inference Acceleration & Kernel Optimizations**

- **Predicted-Outputs Prefill Powers vLLM**: Cascade announced **Predicted Outputs** for **vLLM**, converting likely completions into prefill for partial matches to turbocharge generation; see the [Predicted Outputs in vLLM](https://cascadetech.ai/blog/vllm-predicted-outputs/) post, the [live demo](https://app.cascadetech.ai/), and the supporting [tweet thread](https://x.com/saganite/status/1976707696578691101). The method turns speculative predictions into cached compute, aiming to reduce latency without changing APIs.
    - Members called it *"could dramatically speed up vllm inferences by pre-computing potential output sequences"*, viewing it as a practical speculator that slots into existing vLLM deployments. Early testers compared it to spec-decoding benefits but praised its simplicity and portability across workloads.
- **Residual Recalc Rockets Throughput**: **LLMQ** implemented attention-residual recalculation to relieve memory pressure, netting big speedups on constrained rigs; the PR [attention-residual recalculation](https://github.com/IST-DASLab/llmq/pull/7) shows **Qwen2.5-14B** on 4×4090 jumping from **3.2k→6.0k TPS (fp8)** and **2.5k→4.5k TPS (bf16)**. The optimization re-computes low-cost terms during backward to trade flops for bandwidth.
    - Engineers noted the gains shine in high-memory-pressure regimes where activations dominate, calling it a *"significant net speed-up"* for 14B-scale training. Discussion emphasized applying the trick alongside mixed precision and careful activation checkpointing for best results.
- **Swizzles Straightened; ldmatrix Lessons Learned**: Kernel devs pinned **Triton**’s `ldmatrix` tiling logic at d = log2(4/w) in the code, with references to [Utility.cpp](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.cpp#L186-L199) and clarified optimal swizzling via [GenericSwizzling.cpp (PR #6982)](https://github.com/triton-lang/triton/pull/6982). Concurrently, **CUTLASS** users flagged **PTX** K-contig/swizzling doc mismatches (see [NVIDIA PTX doc](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-k-32b-swizzle-tf32)) and reported compiler aborts with `ldmatrix` copy atoms.
    - Contributors expect PTX doc fixes around version 13.1 and cautioned that many *ICEs are user errors* when pushing exotic copy atoms. Takeaway: follow Triton’s reference implementations for swizzling, and validate tensor descriptors to avoid layout-induced stalls and aborts.

**2. Tiny Models, Mighty Benchmarks**

- **Samsung’s Small Fry Smashes ARC**: Samsung’s **Tiny Recursive Model (TRM)** posted **44.6%** on **ARC-AGI-1**, outscoring larger models like **DeepSeek-R1**, **Gemini 2.5 Pro**, and **o3-mini** in shared benchmark talk. Engineers highlighted the surprising gap despite TRM’s compact size.
    - Debate centered on whether TRM is *"hyper specialization"* for ARC-AGI versus broad generalization expected of **LLMs**. Practitioners warned against over-indexing on a single benchmark without multi-task corroboration.
- **ARC Arms Race Raises Red Flags**: Researchers flagged that **HRM/recursive-model** approaches might overfit **ARC-AGI** through aggressive data augmentation on a small public set, blurring lines with data leakage. Some suggested major labs likely pursue similar augmentation to chase leaderboard gains.
    - Others proposed hybrid systems—pairing a structured reasoner for unseen pattern tasks with an **LLM** for world knowledge—though no robust integration pattern has emerged. A linked discussion explored **GNN** intermediaries as potential controllers for small reasoning modules.
- **Hyperparameter Haste Halves Diffusion Steps**: An implementation of the paper [Hyperparameters are all you need](https://arxiv.org/abs/2510.02390) showed **8 steps** matching the FID of **20 steps** in image generation, claiming ~**60%** compute reduction and **2.5×** speedups; try the Spaces: [Counterfeit v3.0 version](https://huggingface.co/spaces/coralLight/Hyperparameters_Are_All_You_Need), [XL version](https://huggingface.co/spaces/coralLight/Hyperparameters-are-all-you-need-xl-version), and [original SD version](https://huggingface.co/spaces/coralLight/hyperparameters-are-all-you-need-sd-version). The approach required no training or distillation and worked across models.
    - Users reported better quality than DPM++2m at a fraction of steps and praised reproducible configs for quick A/Bs. Teams asked for next-model targets and shared before/after artifacts to validate speed–quality tradeoffs.

**3. AI Funding & M&A Roundup**

- **Spellbook’s Series B Sprouts to $50M**: **Spellbook** raised a **$50M Series B** led by Khosla, pitched as *Cursor for contracts*, per [Scott Stevenson](https://xcancel.com/scottastevenson/status/1976280608436572393). The platform claims **4,000 customers** since 2022 and a suggestion-accept rate jumping from **5%→60%**, with funds going to product (e.g., realtime market-comparison beta).
    - Law-tech folks lauded the traction metrics and pragmatic feature roadmap. Builders expect tighter IDE-like workflows (diffs, redlines, audit trails) to become table stakes for legal AI assistants.
- **Datacurve Digs $17.7M for Data**: **Datacurve** announced a **$15M Series A + $2.7M Seed** to build high-quality training datasets for foundation models—especially coding—per [Serena Ge](https://xcancel.com/serenaa_ge/status/1976328983458480539). Backers include Chemistry, YC, Cohere, Afore, and angels.
    - Engineers see dedicated, license-clean, richly-labeled code corpora as critical for next-gen model reliability. The round signals sustained investor appetite for specialized data vendors powering **frontier LLM** training.
- **Elastic Eats Jina for Embeddings & Agents**: **Elastic** acquired **Jina AI** to fortify retrieval, embeddings, and context-engineering for agentic AI, per [Elastic’s announcement](https://xcancel.com/elastic/status/1976278980018765886). The move targets enterprise search with deeper multimodal and agent tooling.
    - Practitioners expect tighter **ES** integration with vector pipelines, hybrid search, and RAG orchestration. The acquisition hints at consolidation where infra incumbents bake AI-native stacks directly into their platforms.

**4. Protocol Standards & Structured Tools**

- **Well-Known Wins: MCP Metadata Makes the Map**: The **Model Context Protocol (MCP)** community proposed a `.well-known/` endpoint for server metadata—scoping doc name, URL-relative location, and minimal `Implementation` content—see the [MCP blog update](https://blog.modelcontextprotocol.io/posts/2025-09-26-mcp-next-version-update/#server-identity), [discussion](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1147), and [PR thread](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1054#issuecomment-3117705161). A [Dev Summit deck](https://github.com/modelcontextprotocol/registry/blob/main/docs/explanations/dev-summit-2025-10-registry-status-presentation.pdf) outlines the registry’s direction.
    - Contributors favored a minimal SEP to avoid client breakage while enabling discovery and identity. Standardized metadata is expected to simplify server onboarding, trust signals, and tooling interop.
- **Picture This: Portable Images in structuredContent**: Members debated representing image content in **structuredContent**, noting portability breaks when hosts pass `StructuredOutput` straight to model APIs. The guidance: the protocol doesn’t mandate host mapping, and providers poorly support image-returning tools; a temporary tool that maps string→image can bridge the gap.
    - Teams cautioned against locking schemas to any single LLM API’s quirks. A thin indirection layer—tool stubs for assets—keeps UIs hydrated while models see lean, serializable descriptors.
- **Skybridge Skips Schemas, Sparks Schema Schism**: The **skybridge** tool was noted to not use `outputSchema`, reigniting debate about reconciling `ContentBlock[]` with `structuredContent` (the latter often used for widget hydration yet still model-visible). Contributors probed whether any formal schema binding should be defined.
    - Consensus trended toward pragmatic flexibility: hydrate UIs with structured blocks, but avoid over-constraining model I/O while schema practices are still evolving. Expect incremental conventions rather than a one-shot grand spec.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT-5 Scores Dubious Coding Victory**: Members are debating the best AI model for coding, with some preferring **GPT-5** for its responsiveness and understanding, while others vouch for **Claude 4.5** for planning and research.
   - Debate sparked whether **GPT-5** produces *more optimized code*.
- **Perplexity Pro Caps Video Creation**: Users report that even with **Perplexity Pro**, there are search limits (around **300** per day) and they risk getting flagged as spammers if they type too fast, and that *video creations are limited to 5 a month*.
   - The aggressive limits have some members questioning its value.
- **Comet Browser Set for Mobile Launch**: Members anxiously await the mobile release of **Comet Browser**, with an estimated release date around *the end of the year*.
   - One user boasted that *Comet works well with test taking and will choose the correct answer for you*.
- **GPTs Agents Remain a Static Brain**: Members discussed that **GPTs agents** *do not update their base knowledge* after the initial training, when uploading new files.
   - Instead, uploaded files are saved as *knowledge files* that the agent [references when required](https://link.to/openai-docs).
- **Perplexity's Search API Gets the Cloudflare Boot**: A member reported a weird problem while using **Perplexity's Search API**, encountering a `PermissionDeniedError` that appears to be related to a **Cloudflare** denial.
   - The root cause of the **Cloudflare** block remains unclear.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena Mulls Pricing Plans**: Members discussed the future of **LM Arena's** pricing, referencing [the ToS](https://lmarena.ai/tos) which allows them to charge at any time, due to running costs in the millions.
   - However, a maintainer stated that *it's our intent to keep free and open for all*, exploring [alternative strategies](https://news.lmarena.ai/ai-evaluations/) to avoid charging fees.
- **Sonnet 4.5 is Depressed and Failing**: Members reported frequent errors with **Sonnet 4.5 Thinking**, particularly after extended chats, suspecting potential API issues.
   - Despite suggestions to clear cookies, the problem persists for many, with no definitive solution identified.
- **Video Arena Gets Grilled**: Users critiqued the **Video Arena** for its limited model choices, request limits, the necessity of logging in for image use, and high costs.
   - A member suggested that video generation is more expensive than text bots, leading to these restrictions.
- **Gemini 3.0 Launch Delayed?**: The community debated whether the **Gemini 3.0** release is still on track for October or if it will be pushed to December.
   - Some claimed it was undergoing A/B testing within AiStudio, though this remains unconfirmed.
- **Grok Channels Inner Spielberg**: Users are exploring **Grok's** video generation capabilities, which are uncensored, audio-enabled, and unlimited.
   - Despite the low resolution of *560 x 560* and potential watermarks, it's celebrated as *its free lil bro*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Perplexity's LinuxComet faces Privacy Backlash**: **LinuxComet**, a browser from Perplexity, is criticized as a *massive tracker* collecting more data than top browsers combined, prompting users to seek alternatives like **Safari**, **Firefox**, and **Brave**.
   - Some members joked that using **Firefox** for privacy is like wearing a tinfoil hat in a glass house posted on **Google Maps**, with the browser choice reflecting individual privacy stances.
- **GPT-5 Thinking Mini's Thinking Too Much**: Users are looking for a way to **disable GPT-5 Thinking Mini**, as it automatically switches from **GPT-5 Instant** and provides unsatisfactory results.
   - Currently, no solution was provided, leaving users stuck with the unwanted behavior.
- **Samsung's Tiny Model Stuns Benchmarkers**: Samsung's **Tiny Recursive Model (TRM)** outperformed larger models like DeepSeek-R1, Google's Gemini 2.5 Pro, and OpenAI's o3-mini on standard AI benchmarks, achieving **44.6%** accuracy on the ARC-AGI-1 test.
   - The discussion raised questions about the **TRM's hyper specialization** for that benchmark vs the generalized purpose of LLMs.
- **ChatGPT Business Emerges from ChatGPT Teams**: **ChatGPT Team** is rebranding to **ChatGPT Business** as of *August 29, 2025*, see [release notes](https://help.openai.com/en/articles/11391654-chatgpt-business-release-notes).
   - Details about this shift have emerged in official OpenAI documentation.
- **Sora 2 Prompting: Cracking the Code Visually**: Members debated the existence of a "best all around prompt" for **Sora 2**, concluding that good prompts include the **details and qualities you want to see**.
   - Users are reverse engineering prompts by finding a *wow* video, copying the prompt, restructuring it, and tweaking the content to see how text affects the video generation.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **VST Popularity Tracked on GitHub**: A [GitHub repo](https://github.com/MikesRuthless12/tools-hub) tracks the popularity of **VSTs** via website traffic, along with **YouTube channel** subscriber counts and average monthly views.
   - This data provides valuable insights for assessing the reach and engagement within the **VST community**.
- **BYOK Payment Issues Plague Users**: Users report being prompted for **Payment Required** with **BYOK**, failing to redirect to their keys and incurring account charges.
   - This issue remains unresolved and affects users attempting to use their own keys on the platform.
- **DeepSeek Limits Irk Role-Players**: Users express frustration with **rate limits** and **removal of DeepSeek models** on OpenRouter, seeking **uncensored free alternatives** for role-playing.
   - A user seeks a local model for a 4070 32gb laptop, following restrictions on **DeepSeek 3.1**.
- **SSE Usage Data Retrieval Hits Snag**: Users encounter issues retrieving **usage data** with **SSE** using the OpenRouter API, with the *usage* object missing from received messages.
   - A potential fix involves addressing the return of the *finish_reason: stop* chunk before the usage chunk, as detailed in [this litellm issue](https://github.com/BerriAI/litellm/issues/11626).
- **Qwen to Debut New Models**: **Qwen** is set to release more models next week, as announced in a [post on X](https://x.com/JustinLin610/status/1976681042041028823).
   - Enthusiasts anticipate enhanced capabilities and performance with the upcoming model releases.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3-VL: Autocompiler Disable Saves VRAM**: Members debated **Qwen3-VL** support in Unsloth, with one user confirming that disabling the autocompiler using `os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'` enables the model and uses about **44GB VRAM**.
   - The discussion clarified that disabling the autocompiler is a temporary workaround.
- **DGX Spark Finetuning: Good FLOPS, Bad Bandwidth?**: A user questioned **DGX Spark's** finetuning viability, citing bandwidth concerns; another shared a demo experience, reporting it was *slow as heck like less than 10/s* for a **32B** model.
   - The member hoped for software optimization to make **LoRA** viable, considering the lack of **CUDA** support in the AMD alternative.
- **TRL Library: Trainers Face Refactoring**: The upstream **TRL library** is considering removing several trainers (**CPO/SimPO, KTO, OrPO, PPO, RLOO**) without retaining support, leading to refactoring concerns, as seen in [this GitHub issue](https://github.com/huggingface/trl/issues/4223).
   - Despite the concerns, it looks like **KTO, ORPO and PPO** are being moved to **trl.experimental**
- **WSL2 Users Suffer Xformer Package Pain**: A user ran into package incompatibility issues (CUDA version mismatches with **xformers**) while fine-tuning **Mistral 7B** on WSL2 for a summarization dataset.
   - A member suggested uninstalling and reinstalling **torch**, **torchvision**, and **torchaudio**, and provided commands for installing **xformers** from GitHub.
- **7M Recursive Model Smokes ARC-AGI, Generalization Doubted**: A member shared a paper on a [7M recursive model](https://arxiv.org/abs/2510.048717) with only two layers, achieving **SOTA** at **ARC-AGI 1 and 2**, focusing on deep supervision and protections against collapse/divergence.
   - Others noted that, while impressive, the model's **generalization abilities** may be limited due to its size and the fact that **ARC-AGI** tests with a private testing set.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3 Obliterates Code Llama**: Members are suggesting [Qwen3](https://lmstudio.ai/models/qwen/qwen3-coder-30b) for coding tasks due to its **speed and accuracy**, running at **20 tk/s** compared to CodeLlama at **5 tk/s**.
   - However, users note that even basic coding tasks need models greater than **20b parameters**, so one user mentioned [Llama 3.3 70b](https://lmstudio.ai/models/meta-llama/Llama-3-70B-Instruct-GGUF) would be better.
- **Context Length Causes Consternation**: Users explored how context length affects performance; even if context is empty, **higher context allocation slows generation time**.
   - LM Studio is implementing guardrails for memory allocation controlled by the memory estimator, but you can disable them at your own risk.
- **Tool Call Tantrums Trigger Termination**: LM Studio should **not** be aborting generation when a tool call fails, but currently it is, and the failure message is not being displayed.
   - One possible solution is to disable MCPs (Multi-Call Procedures) to prevent model confusion, or look into fetch website tools because Playwright is *extremely token inefficient*.
- **Sparkle's Server Shows off 16 Arc GPUs**: Sparkle has introduced the [Arc Pro B60 dual server](https://videocardz.com/newz/sparkle-unveils-arc-pro-b60-dual-server-with-16-gpus-and-up-to-768-gb-of) boasting **16 GPUs** and up to **768 GB of VRAM**, powered by a **10800W PSU**.
   - Members were excited about a server product that focused on multiple GPU.
- **4080 Ti Super Sparks Sparring with M3**: Members discussed whether a **Nvidia 4080 Ti Super** with **16GB VRAM** and **32GB RAM** is superior to an **Apple M3** with **36GB RAM** for machine learning tasks.
   - One member suggested checking [GPU benchmarks](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) for LLM inference to compare performance, noting that an **M3 Max 40-Core GPU with 64GB** is a comparable configuration.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **API Keys unlock models**: Users can now add **API keys** in the **Models** page to use personal keys with models, and can turn off all models and add them manually.
   - This allows for greater control over model access and billing.
- **AutoHotkey controls Cursor**: Members discussed using **AutoHotkey (AHK)** to control **Cursor** from **Discord**, with some planning to generate **AHK scripts** using Cursor.
   - The community expressed enthusiasm and interest in exploring this integration.
- **Cursor pricing in the crosshairs**: Users debated the increased cost of the **Auto** model and reduced usage limits in **Cursor's Pro** plans, considering alternative models or services, as summarized in [this forum post](https://forum.cursor.com/t/the-pro-account-limits-have-been-clearly-reduced/134738).
   - The changes have led some users to re-evaluate the value proposition of Cursor's subscription plans.
- **GPT-5 comes in clutch**: A member touted **GPT-5** for coding as cheaper, but another member noted that one downside is it requires more time because of its reasoning.
   - This sparked a debate on the balance between cost-effectiveness and development time.
- **Apply Button Goes Rogue**: Users reported that the **APPLY** button in **ASK** mode had disappeared, making it harder to apply changes, and causing them to begrudgingly have to use **AGENT** mode because it goes rogue.
   - This change has frustrated users who prefer the direct control offered by the **ASK** mode.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Tiling Logic Triumphed**: The `ldmatrix` tiling calculation involves *d = log_2(4/w)*, where *w* is the byte width, with the full implementation available in [Triton's source code](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.cpp#L186-L199).
   - The community has clarified that these calculations are implemented in the different lowerings within **Triton**.
- **Optimal Swizzling Secrets Swirl**: Discussions clarified optimal swizzling in linear layout, based on discrepancies between Figure 5, Sections 5.4 and 9.2 in the relevant paper, but [the code from pull 6982](https://github.com/triton-lang/triton/pull/6982) is the *true companion*.
   - This implementation can be found at **GenericSwizzling.cpp**.
- **Panther Lake Powers Up Performance**: New [Panther Lake slides](https://www.techpowerup.com/review/intel-panther-lake-technical-deep-dive/13.html) showcase increased compute per slice and up to **16 MiB of L2$**, which deliver a **40% performance-per-watt uplift** relative to Arrow Lake, addressing memory bandwidth constraints.
   - The architecture of **Celestial** is still unclear, hinging on the desired ratio of compute-to-fixed-function units relative to **Panther Lake**.
- **CUTLASS Compiler gets Copy Atoms**: A user reported compiler aborts when utilizing **ldmatrix** copy atoms, who created a code repro for the issue named `t.py`.
   - The team mentioned that many *user errors* show up as internal compiler errors (**ICEs**).
- **LLMQ Boosts Bandwidth with Recalculation**: The attention-residual recalculation (implemented in [this llmq PR](https://github.com/IST-DASLab/llmq/pull/7)) was implemented, which can lead to significant net speed-up in cases with high memory pressure, like a **14B model on 4x4090**.
   - The optimization increased throughput from **3.2k to 6.0k TPS** when training **Qwen2.5-14B** in **fp8**, and from **2.5k to 4.5k TPS** when training in **bf16**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Colab A100 Time Costs 75 Cents per Hour**: Members discussed the affordability of **Colab** at **$10** per month, with one calculating it provides approximately **13.2 hours** of **A100 GPU** time, costing about **75 cents/GPU/hour**.
   - One user quipped about asking parents for allowance, while others debated the best strategies for GPU allocation and utilization for smaller projects.
- **Hyperparameter Diffusion Sprints Ahead in Image Generation**: A member launched a HuggingFace Space showcasing an implementation of [Hyperparameters are all you need](https://arxiv.org/abs/2510.02390), demonstrating that **8 steps** can achieve comparable FID performance to **20 steps** in image generation resulting in a **60% compute reduction**.
   - The implementation includes a [Counterfeit v3.0 version](https://huggingface.co/spaces/coralLight/Hyperparameters_Are_All_You_Need), an [XL version](https://huggingface.co/spaces/coralLight/Hyperparameters-are-all-you-need-xl-version), and an [original SD version](https://huggingface.co/spaces/coralLight/hyperparameters-are-all-you-need-sd-version) for testing, running **2.5x faster** and with *better quality*.
- **BERT Speaks Polish Tweets**: A member fine-tuned a **BERT model** to predict sentiment and emotions in **Polish language tweets**, which can be found [here](https://huggingface.co/spaces/yazoniak/twitteremo-pl-classifier).
   - This model aims to provide more accurate sentiment analysis for Polish-speaking online communities, catering to a language often underrepresented in sentiment analysis tools.
- **MedScan AI: Your New Doctor?**: A member launched **MedScan**, an AI tool built on **Hugging Face models** for smart medical search and report analysis, accessible [here](https://ragmedscan.vercel.app/).
   - It aims to provide users with *quick access* to relevant medical information and assistance in understanding complex medical reports, potentially *improving healthcare accessibility*.
- **Continual Learning Requires Rethinking**: A member explains the challenges of continual learning, suggesting current papers focus on *"curing symptoms but not cause"* and proposes **designing principles** for the model to learn what and how much to remember based on reward signals.
   - They suggest that the *optimization function* should be a learnable **RNN** working towards increasing rewards for every action, aligning with **Richard Sutton's** ideas on emergent imitation learning in AGI.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Datacurve Secures $17.5M for Training Frontier LLMs**: Datacurve announced a combined **$15M Series A** and **$2.7M Seed** round to provide high-quality training datasets for foundation models, specifically for coding, detailed in [Serena Ge's announcement](https://xcancel.com/serenaa_ge/status/1976328983458480539).
   - The funding, spearheaded by Chemistry with participation from YC, Cohere, Afore, and angel investors, underscores the growing demand for specialized training data.
- **Spellbook Raises $50M Series B for AI Contract Tool**: Spellbook, which brands itself as *Cursor for contracts*, closed a **$50 million Series B** led by Khosla Ventures, as announced by [Scott Stevenson](https://xcancel.com/scottastevenson/status/1976280608436572393).
   - The AI drafting & review platform boasts **4,000 customers** since 2022, with suggestion-accept rates increasing from **5% to 60%**, and will allocate new funds to product enhancements, starting with a realtime market-comparison beta.
- **Kernel Bags $22M to Bolster LLM Cloud Infrastructure**: Kernel, under CEO Catherine Jue, revealed a **$22M Seed + Series A** round, guided by Accel, to automate browser workloads at scale and introduce Kernel Agent Authentication, as mentioned in [this announcement](https://xcancel.com/juecd__/status/1976325764166615498?s=46).
   - With customers like CashApp & Rye, Kernel Agent Authentication aims to provide an identity layer, giving AI applications secure, scoped, auditable control over user actions.
- **Elastic Gobbles Up Jina AI for Smarter Search**: Elastic has acquired Jina AI to enhance their retrieval, embeddings, and context-engineering capabilities to empower agentic AI, as [announced by Elastic](https://xcancel.com/elastic/status/1976278980018765886).
   - The acquisition is expected to fortify Elastic's position in enterprise search and AI agent technology.
- **Gemini Flash 2.5 Nano Banana: JSON Prompting Showcased**: Emily exhibited a detailed **JSON prompt** for **Gemini Flash 2.5 Nano Banana** that produces a blue-accented, anime-style mirror selfie of an East-Asian woman in her PC-corner bedroom, showcased in [this Tweet](https://x.com/iamemily2050/status/1976431328280416520?s=46).
   - The demonstration sparked a discussion on the merits of **JSON** versus **natural-language prompting** with Emily advocating for JSON’s reproducibility and control, with replications posted in [this tweet](https://x.com/toyxyz3/status/1976650667046605263?s=46).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Llama.cpp Grapples with Hybrid ROPE**: A request was made for assistance with implementing **hybrid attention ROPE** in *llama.cpp*, related to [this GitHub pull request](https://github.com/ggml-org/llama.cpp/pull/16095#issuecomment-3390312481).
   - It was also noted that the **Hugging Face implementation** doesn't appear to use *partial ROPE*, despite its presence in the config, referencing [this HF code](https://github.com/huggingface/transformers/blob/0419ff881d7bb503f4fc0f0a7a5aac3d012c9b91/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L176).
- **Precision Pains Plague RoPE**: Discussions noted that **RoPE calculations** are sensitive to precision, suggesting the sine and cosine matrices should be computed in **fp32** to avoid inaccuracies, citing [EleutherAI's gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/positional_embeddings.py#L63) as a reference.
   - A recent paper highlighted that using **ROPE** with **BF16** in long contexts can cause precision problems.
- **nGPT's Odd OOD Outburst**: A counterintuitive connection between **training loss** and **out-of-distribution (OOD) generalization** was discussed in the research channel, questioning why **nGPT's architecture** might fail to generalize.
   - One suggestion involved **length generalization** or increased vulnerability to exposure bias, particularly considering generalization between *"seen" samples*.
- **VLMs Vie for Resolution**: A member shared initial results from a **benchmarking project** focused on optimizing **image resolution** against **output quality** in Vision Language Models (VLMs), using **Gemini 2.0 Flash** on **COCO 2017 Val dataset** for captioning tasks, with a [report attached](https://cdn.discordapp.com/attachments/747850033994662000/1425953022671847515/minres_report.pdf?ex=68eac73d&is=68e975bd&hm=194b9521fd489b4596bc9cdb078a3f90adc9533c1be618a2aaa1d0174f450c82&).
   - The benchmarks provide a pathway to discovering efficient vision processing techniques.
- **LessWrong Lights Way to Interpretability**: A member shared a [LessWrong post](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher) about how to become a **mechanistic interpretability researcher**.
   - Additionally, a member shared a [YouTube video](https://www.youtube.com/watch?v=ruLcDtr_cGo) providing a *nice introduction* to **attribution graphs** from **Anthropic's Circuit Tracing and Model Biology papers**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Atropos Tutorial Video Appears**: A quick tutorial video for using **Atropos** and a broader overview of how environments work is now available on [Twitter](https://fxtwitter.com/NousResearch/status/1925381160097697803) and [YouTube](https://www.youtube.com/watch?v=in__ELD4NxE).
   - This video accompanies discussions around **Atropos** and addresses member questions about its use and environment setup.
- **Predicted Outputs in vllm Released**: New tech is released called **Predicted Outputs in vllm**, for faster generation by converting output to prefill for (partial) matches of a prediction; see the [blog post](https://cascadetech.ai/blog/vllm-predicted-outputs/), [demo](https://app.cascadetech.ai/), and [tweet thread](https://x.com/saganite/status/1976707696578691101) for more.
   - This sparks discussion about how this approach could dramatically speed up **vllm** inferences by pre-computing potential output sequences.
- **Constant Params Needed for Efficient Training**: A recent [paper](https://arxiv.org/html/2410.21228v1) indicates that for efficient training, the ratio **(params * samples / data)** needs to remain constant.
   - Members wished the paper had experimented with training token amounts beyond **Chinchilla** (13B on 260B tokens) for more comprehensive insights.
- **LoRA Fine-Tuning May Cause Catastrophic Forgetting**: Members noted that a recent paper argues that looking at **loss** might not be enough to determine if catastrophic forgetting occurs during fine-tuning.
   - Evaluating on both new training tasks and pre-training evals, and inspecting the **singular values of LoRA adapters** might give more insight to this phenomena.
- **LoRA Targets Can Still Learn New Facts**: A member argued that even with **attention layer only targets**, **LoRA** can learn new facts and knowledge, disputing claims that its ineffective.
   - They pointed to the [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/) which argues that many environments only use **LoRA** across the attention layers.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP's .well-known Endpoint Thickens**: Members are discussing a `.well-known/` endpoint for **MCP server metadata**, with references to a [blog entry](https://blog.modelcontextprotocol.io/posts/2025-09-26-mcp-next-version-update/#server-identity), [GitHub discussion](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1147), [pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1054#issuecomment-3117705161), and a [Dev Summit presentation](https://github.com/modelcontextprotocol/registry/blob/main/docs/explanations/dev-summit-2025-10-registry-status-presentation.pdf).
   - The aim is to push through a minimal SEP focusing on the document name, location relative to the MCP server URL, and minimal content (specifically, `Implementation`).
- **Image Content Standards Debated**: The discussion revolves around representing **image content** in `structuredContent`, with concerns that it's not portable and breaks on clients that pass `StructuredOutput` directly to model APIs.
   - It was suggested that the protocol doesn't prescribe how host apps map `structuredContent` to LLM APIs, and support for image returning tools is poor among providers, and giving the model a temporary tool that maps string to an image is a potential solution.
- **Skybridge Tool Skirts OutputSchema**: The discussion highlights that **skybridge** doesn't use `outputSchema` on the tools, and members explored whether there is an opportunity to define any kind of thing for that.
   - Members discussed the dissimilarity of `ContentBlock[]` and the `structuredContent`, in which it was said that the `structuredContent` is used for widget hydration - but it's also available to the model.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Tipped as Future Language**: Members welcomed a new user, predicting that *Mojo será el lenguaje del futuro* <:mojo:1098996988797784084>.
   - The user also stated, *También soy de Colombia 🙈*.
- **New GPU causing Backwards Compatibility woes**: A user questioned whether a certain approach suffers from lack of forward compatibility with GPUs.
   - They asked, *Doesn't this aproach have the problem of no forwards compatibility? You'd have to recompile to support each new gpu generation wouldn't you?*
- **Apple Silicon M3 Gets GPU Vector Addition**: A member tested **GPU support** on **Apple Silicon M3**, running a simple vector add example, but solved issues with the host buffer remaining unmodified with explicit synchronization.
   - It was also mentioned that *printing from inside kernel* on Apple GPU is not yet implemented, and `enqueue_copy` might be a no-op due to shared memory.
- **Jetson Orin Nano Packs Punch for Robotics**: Members discussed using **Jetson Orin Nano 8GB** for robotics, suggesting it's sufficient for developing vision systems, especially for resource-constrained deployments where **battery life** is critical.
   - They highlighted that object detection and image classification models run well on the Orin Nano, enabling scaling up to larger systems later.
- **Mojo Native FFT Lands, Ready to Rumble with FFTW?**: After PR [#5378](https://github.com/modular/modular/pull/5378) there will be a **Mojo native FFT implementation**.
   - Members also discussed performance improvements versus **FFTW** with one user stating they need multidimensional transforms where the sizes are known only at runtime, for which fftw serves well enough for the CPU, but it would be nice to eventually have a GPU implementation as well.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Architectural Tricks Alone Cannot Cure Forgetting**: A member argued that preventing **catastrophic forgetting** requires more than just architectural changes; the optimizer must evolve beyond **autodiff-based gradient descent**.
   - They suggested **hard-sectioning memories** into separate buckets and ensuring those buckets don't fundamentally forget, noting that models requiring **i.i.d. sampling** inherently struggle with forgetting.
- **IID Sampling Proves Problematic for DL**: A member pointed out that **Deep Learning (DL)** methods struggle with **MNIST** when digits are sorted instead of random without sampling old digits.
   - They noted that DL without i.i.d. gets 10% (random) which highlights the reliance on **i.i.d.** for effective learning.
- **Separate Token Embeddings Stave Off Injections?**: A member proposed training a separate set of **token embeddings** for **system/user prompts** to make it easier for models to distinguish between prompt and content, thus reducing **prompt injection** vulnerability.
   - In response, another member noted that these separate embeddings are called **soft prompts** and could defend against **prompt injection** attacks.
- **Camel AI gets New Outfit**: A user shared that [Camel AI](https://github.com/camel-ai/camel) received a few updates and thinks the **roleplay method** is great, and there's a need to test **Workforce** and all the tools.
   - No further discussion on specific features was found.
- **Berkeley Launches LLM Agents Course**: A member shared two LLM Agents courses from Berkeley, which might be interesting to check out: [playlist 1](https://www.youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc) and [playlist 2](https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn).
   - No further discussion or insights were provided about the courses.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GLM 4.6 Subscription Integrated Successfully**: Members confirmed that a **GLM 4.6 subscription plan** works in Aider, referencing the [OpenAI-compatible guide](https://aider.chat/docs/llms/openai-compat.html) for setup, specifically using the *pass* endpoint.
   - This discussion arose as members explored routing prompts using an **n8n workflow** as an API endpoint, leveraging smaller local models to strip environment secrets before engaging larger, off-prem models.
- **Aider Adopts Haiku for Git Commits**: Aider is now using **Haiku** for generating git commit messages, prioritizing speed and cost-effectiveness, specifically using `openrouter/anthropic/claude-sonnet-4`.
   - A member suggested **gemini flash lite** as a cost-effective alternative and recommended setting a *weak model* for commit messages to optimize resource use.
- **Custom Profiles Unleashed in Openrouter**: Users can manage custom profiles in **OpenRouter** to specify custom prompts, temperatures, and reasoning complexity for models, enhancing model management.
   - These profiles are specified in the aider config or via `/model` and `/editor-model` commands, pointing to model definitions in the `aider/resources/` directory.
- **Aider's Model Specification Syntax Detailed**: The `/model` and `/editor-model` chat commands in Aider allow users to specify models, including those not defined in the `.aider.conf.yml` file, providing flexibility in model selection.
   - This on-the-fly selection complements the configuration-based model settings, streamlining model adjustments.
- **Persona Definitions as Read-Only Assets**: Users asked about pushing **personas definitions** (such as those from [vibecodingtools.tech](https://www.vibecodingtools.tech/templates)) as `/read-only` assets to the underlying model.
   - It's suggested to load these personas only when switching between tasks (e.g., from Planning to Coding) rather than for each request, and pushed to the underlying model.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Sora 2 Invite Codes Abundant**: Members shared that **Sora 2 invite codes** were readily available, with the product achieving over *1 million downloads*.
   - Despite the availability, some members expressed preference for awaiting the **public release**.
- **Kimi's Coding Prowess Shines**: **Kimi** demonstrates strong coding capabilities, employing an **agentic mode** and **tool usage** via an **IDE** to execute **Python scripts** and **batch commands** for system debugging.
   - One member asserted that Kimi's coding performance surpasses that of *most other models*.
- **Hack Club and Moonshot AI: Separate Entities**: A discussion arose regarding a potential connection between **Moonshot AI** and an email from **Hack Club**.
   - It was clarified that **Hack Club** and **Moonshot AI** are distinct and unrelated organizations.
- **Mysterious Videos Crafted with Kimi**: Certain members mentioned creating *wild videos* using **Kimi**.
   - However, specific details concerning the content or nature of these videos were not disclosed.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Community Debates 'AI Slop' in PRs**: A member questioned whether [PR #12530](https://github.com/tinygrad/tinygrad/pull/12530) contains *AI slop*, suggesting that dismissing it as incomprehensible is an evasion of responsibility, vouching for the code's quality, specifically mentioning [PR #12539](https://github.com/tinygrad/tinygrad/pull/12539).
   - The submitter compared it to **geohot's** algebraic Upat tests in [issue #12449](https://github.com/tinygrad/tinygrad/pull/12449), but another member mentioned that *AI PRs will be closed without comment, and if you are pushy about them, you will be banned.*
- **Reduce Group Turns Red for Clarity**: The group for **reduce** is now marked in *bright RED* instead of green, to highlight that a local is being used for reduction.
   - The change specifies that green will be reserved for future functionality, see [PR #12604](https://github.com/tinygrad/tinygrad/pull/12604/files).
- **`cuda_ioctl_sniffer` Gets Rust Makeover**: A member is converting George Hotz's `cuda_ioctl_sniffer` into **Rust** with an interactive terminal to test individual **CUDA kernels**.
   - They posted [a demo image](https://cdn.discordapp.com/attachments/1070745817025106080/1425975923458445343/image.png?ex=68eadc91&is=68e98b11&hm=8472f883712ba4af77167cb978546cf31d701a98b6a67bcca03ea1b59fdab985) of the `saxpy` kernel output and aims to support more GPUs, using IOCTL to launch **CUDA kernels**.
- **Winograd Test Asserts Failure**: While attempting to implement loop splitting, a member encountered an **assertion failure** in `test_winograd.py`.
   - The error message indicates a value **6.49** failing a *not less than* comparison against **2.6**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Spotlighting Reduces Prompt Injection**: A member demoed **spotlighting** to reduce the risk of prompt injection, referencing [Microsoft's research paper](https://github.com/estsauver/dspy-spotlight).
   - The member noted that they are still developing a benchmark and test suite against **XPIA attacks**.
- **DSPy Community Repo Shines Light on Community Projects**: A member created the [DSPy Community repo](https://github.com/dspy-community) to highlight projects, preventing them from disappearing into the void.
   - Currently, it's a **README** on the main profile listing libraries and projects, and **PRs are welcome**.
- **MCP Tool Authentication Conundrums Aired**: A member raised questions about creating a `dspy.Tool` from an **MCP Tool** with [authentication](https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#authentication) and how it's handled.
   - The inquiry focused on whether the authentication process is correctly managed when using `dspy.Tool.from_mcp_tool` with tools that require authentication.
- **shadcn Sparks DSPy Website Revamp Dreams**: Inspired by **shadcn**, a member suggested that DSPy could benefit from *an explorer website, a CLI for consistent module placement*, and *a way to publish optimized models* to the repo.
   - The idea is to enable users to adapt modules easily, moving away from `pip install` and towards easier customization.
- **DSPy Module Marketplace Gains Traction**: Community members are pushing for a **platform/marketplace** for DSPy modules to promote sharing and reuse of optimized programs.
   - This marketplace would host optimized programs, such as classification tasks for customer reviews optimized for **Qwen**, **4.1-mini**, and **4.1-nano**, allowing users to quickly deploy solutions for common tasks.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Godhand AI-Assisted Previz Creation Workflow Emerges**: A [LinkedIn post](https://www.linkedin.com/posts/godhand_ai-assisted-previz-creation-workflow-quick-activity-7382046352287088640-Ev0V) highlights **Godhand's** AI-assisted previz creation workflow, promising a quicker and more efficient approach to previz.
   - This workflow has the potential to significantly reduce production time in initial project planning.
- **Users Bawl for Support Staff Intervention**: Multiple users expressed frustration and demanded immediate support staff attention in the channel.
   - One user exclaimed, *"HELLO!!! WHERE IS THE SUPPORT STAFF?!"*, highlighting dissatisfaction with the platform's responsiveness.
- **Manus Excels in Initial Project Structure**: A member reported finding **Manus** efficient for initial planning and structuring projects, mentioning it cost only **1500 credits** to build a RAG pet assistant vector db.
   - The user recommended leveraging **Manus** for planning and then transitioning to **Claude Code** for the coding phase, streamlining workflows with prompts and n8n workflows.
- **Prompt Engineering: Key to Success**: Channel members stressed the importance of writing explicit and detailed prompts when using AI tools such as **Manus**.
   - A user cautioned that *throwing files at any AI and telling it to figure out the details of your prompt is simply very bad practice*, emphasizing the need for careful prompt design.
- **Smooth Claude API Integration Unleashed**: A user announced that **Claude** can now be seamlessly integrated into **Manus** via API calls, removing the need for copy-pasting.
   - This streamlines workflows and enhances the user experience by providing direct access to **Claude's** capabilities within **Manus**.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Diffusion Model Paper Reading Group Kicks Off**: A new **Diffusion Model Paper Reading Group** is launching, meeting this Saturday at **9 AM PST / 12 PM EST** (hybrid in SF + online) to discuss the [*Denoising Diffusion Implicit Models (DDIM)* paper](https://arxiv.org/abs/2010.02502).
   - The session is beginner-friendly for those with **Python + basic PyTorch** knowledge, and participants can RSVP at the [provided link](https://luma.com/vioj15il).
- **DDIM Paper Turbocharges Image Generation**: The reading group will discuss how **DDIM** speeds up image generation while maintaining high quality, which is foundational to **Stable Diffusion**.
   - The paper, [*Denoising Diffusion Implicit Models (DDIM)*](https://arxiv.org/abs/2010.02502) by Song et al., 2020, is considered core knowledge for generative AI engineers.
- **Diffusion & LLM Bootcamp Announced**: The reading group session is part of a **3-month Diffusion Model Bootcamp** (Nov 2025), inspired by **MIT’s Diffusion Models & Flow Matching course**, featuring AI & Software engineers, PMs & creators.
   - The bootcamp offers hands-on experience in building and training your own **Diffusion Model + ComfyUI Pipeline + GenAI app building**.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1425921587525455904)** (1204 messages🔥🔥🔥): 

> `Perplexity AI Models, Comet Browser, AI code debugging, AI programming language, Coding on phones` 


- **GPT-5: Coding Champ or Chump?**: Members debate the best AI model for coding, with some preferring **GPT-5** for its responsiveness and understanding, while others find **Claude 4.5** better for planning and research.
   - There was debate on whether using **GPT-5** is better for coding as there are claims it produces *more optimized code*.
- **Perplexity Pro got a limit?**: Users report that even with **Perplexity Pro**, there are search limits (around **300** per day) and that they are getting flagged as spammers if they type too fast.
   - Some members are noting that with **Perplexity Pro** *video creations are also limited to 5 a month*.
- **Comet Browser Mobile Release Impends**: Members anxiously await the mobile release of **Comet Browser**, with an estimated release date around *the end of the year*.
   - One user said *Comet works well with test taking and will choose the correct answer for you*.
- **Is Perplexity the model or not?**: Members discussed a concern about whether **Perplexity AI** really used **GPT-5** or not when you select it, suggesting that they might be using lower powered models or perplexity assistants.
   - One member noted that *the AIs save your way of talking and your style*, suggesting that may be why perplexity seems different.
- **AI tool for debugging**: The best AI to ask questions for errors is **GPT-5** in **Cursor**.
   - One member recommended a user *make sure the propmts are good* as *they make the difference*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1426169670108581970)** (1 messages): 

> `GPTs Agents, OpenAI's sidebars` 


- **GPTs Agents Remain Stateless Post-Training**: A member inquired whether GPTs agents learn from new information added after initial training.
   - Another member clarified that uploaded files are saved as "knowledge" files for the agent but **do not update the agent's base knowledge**; [files are referenced when required](https://link.to/openai-docs).
- **OpenAI Platform Sidebar UI Swaps**: Some members noticed user interface changes on platform.openai.com, specifically in the sidebars.
   - One user reported that **two icons disappeared**: one for threads and another for messages.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1426247987549438094)** (2 messages): 

> `Search API issues, Permission Denied Error, Cloudflare deny` 


- **Search API throws Permission Denied Error**: A member reported a weird problem while using **Perplexity's Search API**, encountering a `PermissionDeniedError` that appears to be related to a **Cloudflare** denial.
   - It's unclear why the API is being blocked by Cloudflare.
- **Possible reasons for Cloudflare deny with Search API**: The user is seeking potential reasons behind the **Cloudflare** denial when using the **Search API**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1425920856923705445)** (1202 messages🔥🔥🔥): 

> `LM Arena Pricing, Sonnet 4.5 Error, Video Arena Features, Gemini 3.0 Launch, AI Video Generation` 


- **LM Arena's Fate: Free Forever?**: Members discussed the future of LM Arena's pricing, with [one user pointing out the ToS](https://lmarena.ai/tos) allows them to charge at any time, due to running costs in the millions.
   - A maintainer replied that *it's our intent to keep free and open for all*, further noting that they are exploring [alternative strategies](https://news.lmarena.ai/ai-evaluations/) to avoid charging fees.
- **Sonnet 4.5 Thinking is depressed**: Members are having issues about **Sonnet 4.5 Thinking** failing and throwing constant errors, especially after extended chats.
   - Others pointed to possible API issues, with a few commenting that they also experienced the same, it was further suggested that clearing cookies might resolve the issue, but this did not seem to work for everyone.
- **Video Arena Lacks Model Choices**: Members discussed the limitations of the **Video Arena**, specifically the lack of model selection, limited video requests, the need to log in to use the image feature, and overall expense.
   - One member noted *Making videos is more expensive than the normal text bots (I think) that’s why it’s limited*.
- **Gemini 3.0: October Mirage?**: Members debated the **Gemini 3.0** release date, debating whether it was still on track for October or delayed to December.
   - Some members claimed that it was running in A/B testing inside of AiStudio, but this could not be confirmed.
- **Grokked Videos gain Unlimited Traction**: Members explored **Grok**'s video generation capabilities, impressed with its uncensored, audio-enabled, and unlimited usage.
   - The downside is that *the output for 1:1 video is whoopin 560 x 560*, and that it may produce watermarks, however *its free lil bro*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1425953912422269070)** (1 messages): 

> `LMArena Survey, Arena Champions Program` 


- **LMArena Seeks User Feedback via Survey**: LMArena is collecting user feedback to improve the product via [this survey](https://docs.google.com/forms/d/e/1FAIpQLSevxsX_kvJ_fiv74Rcf2yPl9lnSNOtmmb_wMnBCy1fEri_jEg/viewform?usp=dialog).
   - The survey aims to understand what is important to users to make **LMArena** a great product.
- **Arena Champions Program Launches**: LMArena introduces the **Arena Champions Program**, which rewards members who show genuine commitment to meaningful conversation.
   - Apply to the program [here](https://docs.google.com/forms/d/e/1FAIpQLSdRWfqG8_MMKQ4H23FHFZVJsg0OuQrZqn5h9l-QqhWpNI77xg/viewform?usp=dialog) for access to a private space to engage without interruptions, requiring demonstration of **AI interest** and **commitment to meaningful conversation**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1425923461150146651)** (493 messages🔥🔥🔥): 

> `LinuxComet Privacy Issues, Browser Recommendations, DuckDuckGo Privacy, AI Agency Business Model, Sora2 Availability in EU` 


- **LinuxComet Declared Privacy-Invasive**: A member criticized **LinuxComet**, a browser from Perplexity, as a *massive tracker* collecting more data than top browsers combined, calling it *terribad* and cautioning against paid use.
   - They sarcastically remarked that anyone *claiming to care about privacy* almost certainly doesn't, advocating for browsers like **Safari** or **Firefox**, while questioning Apple's telemetry practices.
- **Browser Choice Reflects Privacy Stance?**: Members discussed browser privacy, with one favoring **DuckDuckGo** for its privacy-focused wrapper around Chrome and another endorsing **Brave** due to its crypto rewards for ad viewing.
   - Concerns were raised about **Firefox** being funded by Google, implying potential tracking despite its privacy focus, with one user joking that *using firefox for privacy reasons is like... wearing a tinfoil hat to block ads while standing in a glass house posted on Google Maps*.
- **AI Browser Domination Predicted**: A member predicted that **AI in the browser** will dominate, positioning the browser as the modern operating system and praising **Comet AI** for its direction despite current computer vision model limitations.
   - Another user lamented being unable to share a ChatGPT subscription, citing corporate oppression, while others debated the definition and utility of AI-integrated browsers.
- **Ethical AI Agency**: A discussion emerged around the concept of an **AI agency**, focusing on automating processes and discord response, with a member suggesting the need for someone with an **MBA** to assess profitability.
   - The discussion highlighted the goal of helping companies automate processes, but cautioned about the need for someone with a business background to ensure profitability.
- **Tiny Samsung TRM Surpasses Larger Models**: Members discussed Samsung's Tiny Recursive Model (**TRM**), highlighting its unexpected performance on standard AI benchmarks, with TRM achieving **44.6%** accuracy on the ARC-AGI-1 test.
   - Its scores outperformed much larger models like DeepSeek-R1, Google's Gemini 2.5 Pro, and OpenAI's o3-mini, however the discussion raised questions about the TRM's hyper specialization for that benchmark vs the generalized purpose of LLMs.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1425927860593623131)** (17 messages🔥): 

> `OpenAI liability waiver, Responsibility for OpenAI's part, ChatGPT Business vs Enterprise, GPT-5 Thinking Mini, MCP dev channel` 


- **OpenAI Faces Liability Concerns, User Suggests Waiver**: A user suggests OpenAI institute a **legal liability waiver** that users must sign to take responsibility for their actions, aiming to reduce OpenAI's liability instead of *"butchering the usefulness of the models."
- **ChatGPT Business Rebrands from ChatGPT Teams**: **ChatGPT Team** is now **ChatGPT Business** as of *August 29, 2025*, according to [release notes](https://help.openai.com/en/articles/11391654-chatgpt-business-release-notes).
- **Users Request Control over GPT-5 Thinking Mini Behavior**: A user is seeking a way to **disable GPT-5 Thinking Mini**, as it automatically switches from **GPT-5 Instant** and gives unsatisfactory results for their use case.
   - No solution was provided in the messages.
- **Community Proposes MCP Dev Channel for OpenAI Integrations**: A member proposed a new **MCP dev channel** focused on MCP server integrations, uses, and experiences with OpenAI products, see [Discord channel link](https://discord.com/channels/974519864045756446/1426291246132887552).


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1426001506980597922)** (13 messages🔥): 

> `Sora 2 prompting, Visual learning, Prompt engineering by example` 


- **Debate rages on about best Sora 2 prompts**: Members debated the existence of a "best all around prompt" for **Sora 2**, with one member stating that *good prompts include the details and qualities you want to see*.
- **Learning Prompt Engineering by Example**: One member suggests learning prompt engineering by finding a video that looks good, copying the prompt, restructuring it, and then changing the content.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1426001506980597922)** (13 messages🔥): 

> `Sora 2 Prompting, Bypassing Guidelines` 


- **Debate on 'Best' Sora 2 Prompts Arises**: A user inquired about the *best all-around prompt* for **Sora 2**, but a member stated that there's no such thing, and that good prompts should include the **details and qualities** you want to see.
- **Reverse Engineering Prompts from Sora Videos**: A user suggested reverse engineering prompts by finding a *wow* video, copying the prompt, restructuring it, and tweaking the content to see how text affects the video generation.
   - This lets you to see *what text affected what and how to tweak and adjust*.
- **Prompting for Sora 2 is like Prompting ChatGPT**: A member stated the *chat gbt wasn’t working* and explained that they're *more of a visual learner* and are *trying to actually learn* about prompts.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1426263729888301257)** (2 messages): 

> `VST Popularity, YouTube channel subscribers and views` 


- **VST Popularity Contest on GitHub**: A member shared a [GitHub repo](https://github.com/MikesRuthless12/tools-hub) for tracking the popularity of **VSTs** based on website traffic.
   - The repo also includes data on **YouTube channels**, tracking their current subscriber count and average monthly views.
- **YouTube Channel Insights**: The [linked GitHub repository](https://github.com/MikesRuthless12/tools-hub) provides metrics on **YouTube channels**, including subscriber counts and average monthly views, related to VSTs.
   - This data can be valuable for assessing the reach and engagement of different channels within the **VST community**.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1425923625369997496)** (401 messages🔥🔥): 

> `BYOK payment issues, Gemini 3 release, Constraining PDF pages with API, Roleplayers and free models, Usage data issues with SSE` 


- **BYOK Payment Required? Users Report**: Some users are experiencing issues with **BYOK** (Bring Your Own Key), noting that they are being asked for **Payment Required** and not being redirected to their connected key, resulting in charges to their accounts.
   - This issue is unresolved as of the discussion.
- **DeepSeek Rate Limits Frustrate Role-Players**: Users are frustrated with **rate limits** and the **removal of DeepSeek models** on OpenRouter, seeking **uncensored free alternatives** for role-playing.
   - One user is looking for *a model to run locally* on a 4070 32gb laptop for role-playing purposes after **DeepSeek 3.1** was restricted.
- **Troubleshooting SSE Usage Data Retrieval**: Users report issues retrieving **usage data** when using **SSE** (Server-Sent Events) with the OpenRouter API, specifically that the *usage* object isn't included in the messages received.
   - A user identified that OpenRouter APIs return the chunk with *finish_reason: stop* **before** the usage chunk, causing **litellm** to terminate the iteration early, and provided a [potential fix](https://github.com/BerriAI/litellm/issues/11626).
- **OpenRouter's Alpha Responses API Faces Downtime**: Users reported experiencing **500 Internal Server Errors** when using the alpha responses API.
   - It was later confirmed that the API was indeed down but has since been resolved: *sorry our alpha responses API was down for a bit! it should work again now*.
- **GLM 4.5 and 4.6 Not Working for Some**: Users report that **GLM 4.5** and **GLM 4.6** by Chutes are not working inside OpenRouter, while others suggest using **GLM 4.5 air (free)** with **Z.ai** as the provider to avoid errors.
   - This comes amidst discussion of Google reallocating servers to **Gemini 3.0**, leading to a reported quality decrease for **2.5**.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1425978657616429168)** (2 messages): 

> `` 


- **No new models discussion found**: There was no discussion about new models in the provided messages.
- **Channel silent on model updates**: The 'new-models' channel appears to have no relevant activity to summarize based on the given message history.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1425979781715066963)** (12 messages🔥): 

> `Sambanova Deepseek R1/V3, BYOK Azure Keys Routing, ChatQwen Models` 


- **Sambanova's Deepseek R1/V3 Quality Check**: A member inquired about the quality of **Sambanova** models for the **Deepseek R1/V3** series, seeking feedback from others who have used them.
- **Azure BYOK Keys Encounter Routing Woes**: A user reported issues with their **Azure BYOK key setup**, specifically that traffic was being routed to **OpenRouter's OpenAI** despite having the *"Always use this key"* feature enabled, and another user asked if setting OpenAI to ignored providers resolves the issue.
   - The user then found [BYOK - Bring Your Own Keys to OpenRouter](https://openrouter.ai/docs/use-cases/byok#azure-api-keys) documentation to solve this issue.
- **Qwen Keeps On Truckin' with More Models Next Week**: A user shared that **Qwen** is set to release more models next week, according to a [post on X](https://x.com/JustinLin610/status/1976681042041028823).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1425924739028746393)** (115 messages🔥🔥): 

> `Qwen3-VL Support, DGX Spark performance, trl library changes, Pretraining datasets, Fine-tuning vulnerabilities` 


- **Qwen3-VL Support Status Debated**: It's debated whether **Qwen3-VL** is supported in Unsloth; one member said it *should already work in Unsloth*, another says *just need to disable the auto compiler*.
   - One user confirmed that disabling the autocompiler using `os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'` worked and used about **44GB VRAM**.
- **Spark's Finetuning Viability Questioned**: A user wondered about the **DGX Spark's** performance for finetuning, noting skepticism due to its bandwidth, despite good FLOPS.
   - Another user who saw a live demo said it was *slow as heck like less than 10/s* for a **32B** model, but they hoped software optimization could make **LoRA** viable, since the AMD alternative lacks **CUDA**.
- **TRL Library Restructuring Creates Angst**: The upstream **TRL library** is considering removing several trainers (**CPO/SimPO, KTO, OrPO, PPO, RLOO**) without retaining support, leading to refactoring concerns, as seen in [this GitHub issue](https://github.com/huggingface/trl/issues/4223).
   - Despite the concerns, it looks like **KTO, ORPO and PPO** are being moved to **trl.experimental**
- **Quest for Pretraining Datasets Intensifies**: A member is looking for good pretraining datasets with **<=10B tokens**, preferring filtered datasets over random subsets.
   - They were pointed to [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) but was skeptical of the datasets' benchmarking practices due to the organization's *benchmaxxed* models.
- **Fine-Tuning Vulnerabilities Explored**: A member inquired about specific vulnerabilities focused around fine-tuning, launching an investigation into AI specific **CVE**'s for a project.
   - Another member highlighted the **safetensors** format and Hugging Face's disabling of running scripts in downloaded models for security, referencing [this HuggingFace blog post](https://huggingface.co/blog/safetensors-security-audit).


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1426066896121364520)** (2 messages): 

> `AI Development, Full-Stack Engineering, Blockchain Building, AI + Web3 Projects` 


- **AI Developer Joins Unsloth Discord**: An AI developer with full-stack engineering and blockchain building experience introduced themselves on the Discord channel.
   - They are open to projects that *push boundaries with **AI + Web3**.*
- **Full Stack Engineer Seeks AI + Web3 Projects**: A full-stack engineer shared their background in AI model design, blockchain networks, and polished frontends.
   - They expressed interest in projects combining **AI and Web3 technologies**, highlighting their experience in delivering end-to-end solutions.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1425923422126346320)** (173 messages🔥🔥): 

> `Datacenter cooling for municipal heating, One More Parameter, RAM Usage and Browsers, Zen Browser, Batch Size vs. Forward Pass Time` 


- **Datacenter Heat Powers Homes?**: Members discussed using **datacenter cooling water** for municipal heating, noting that while it's being done, it's not at scale, and that [Google already has a project in Finland](https://blog.google/around-the-globe/google-europe/our-first-offsite-heat-recovery-project-lands-in-finland/).
   - Rather than cleaning the water, a heat exchanger transferring heat between two water loops is preferred to avoid expensive failures in domestic systems, similar to how the **USSR** used thermal-electrical centrums.
- **Just One More Parameter Bro!**: A member joked about the endless pursuit of more parameters for LLMs, quipping, *"just one more parameter bro it will be the best LLM ever we need more parameters bro more parameters will solve everything just one more please vro"*.
   - Other members joined in with similar sentiments, such as *"please just one more layer 🥺"* and *"Just one million bro~~~"*.
- **RAM Guzzling Browsers**: Members discussed RAM usage, with one noting that *"modern websites gulp memory like crazy"* and a **YouTube tab** can take up **500MB+ RAM**.
   - Another suggested using a tab unloader and pointed out that *"theres no possible excuse for google main page taking 150-160mb ram"*, blaming the page's excessive dependencies and **Chrome's monopoly**.
- **Zen Browser Praised, Firefox Alternative Spotted**: A member recommended **Zen Browser** as amazing, as it unloads tabs automatically.
   - It's been said it's an open source version of **Arc Browser** that uses **Firefox/Gecko** in the backend.
- **Batch Size Affects Forward Pass Time?**: A member asked whether a forward pass of batch size 5 takes the same amount of time as a forward pass of batch size 50.
   - It was explained that *"1x50 is going to be significantly faster than 10x5 _if_ your GPU can handle everything at the same time"* and that it depends on compute vs. memory bottlenecks, with higher batch sizes taking longer but having higher throughput.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1425932159415357522)** (84 messages🔥🔥): 

> `Unsloth installation on Amazon ml.g4dn.xlarge, GGUF model creation from Lora, Distributed FSDP runs support, WSL2 package compatibility issues, AI code agent for generating test cases` 


- **Running Unsloth on Amazon ml.g4dn.xlarge Instances**: A user inquired about installing Unsloth on an Amazon **ml.g4dn.xlarge** instance, noting that Google Colab works but Amazon is complicated, and was directed to Unsloth's [Docker image](https://hub.docker.com/r/unsloth/unsloth).
- **GGUF Conversion Woes**: Users report that saving models as **GGUF** with `model.save_pretrained_gguf()` fails due to runtime errors and quantization issues, even in the [official Ollama notebook](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing).
   - It was suggested that manual conversion is currently necessary.
- **WSL2 Setup Struggles**: A user encountered package incompatibility issues (CUDA version mismatches with **xformers**) while fine-tuning **Mistral 7B** on WSL2 for a summarization dataset.
   - A member suggested uninstalling and reinstalling **torch**, **torchvision**, and **torchaudio**, and provided commands for installing **xformers** from GitHub.
- **DeepSeek API test case generation with LLM Agents**: A member is looking for assistance in developing an **AI code agent** using only the **DeepSeek API** to generate strictly correct test cases from a given problem.
   - The important aspect here is for the agent to generate code from a given problem and generate the correct test cases.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1426320638653497539)** (1 messages): 

> `Qwen3-8B fine-tuning, Novel chapter training, Data Cleaning` 


- **Qwen3-8B Learns Novel Writing**: A member fine-tuned **Qwen3-8B** on approximately **8k** real novel chapters to evaluate the dataset's quality.
   - While the results showed potential, the member observed that the model inherited **Qwen's** repetition issue, suggesting the need for more epochs and better data cleaning.
- **Data Cleaning Importance Highlighted**: The experiment with **Qwen3-8B** revealed the importance of thorough data cleaning when training on extracted novel chapters.
   - Artifacts from the extraction process impacted the model's performance, underscoring the need for careful preprocessing.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1426261254925848727)** (27 messages🔥): 

> `Recursive Model, HRM, ARC-AGI, Data Augmentation, GNN + Reasoning Model` 


- **7M Recursive Model Achieves SOTA at ARC-AGI**: A member shared a paper on a [7M recursive model](https://arxiv.org/abs/2510.048717) with only two layers, achieving **SOTA** at **ARC-AGI 1 and 2**, focusing on deep supervision and protections against collapse/divergence.
   - Others noted that, while impressive, the model's **generalization abilities** may be limited due to its size and the fact that **ARC-AGI** tests with a private testing set.
- **HRM trained on Data Augmentation**: It was pointed out that **HRM** models, while effective, may be **overfit to ARC AGI** due to training on data augmentation bordering on data leaking.
   - Another member stated that major labs likely employ **data augmentation** for gaming **ARC AGI** scores due to the small public set and potential investor capital gains.
- **Integrating HRM with LLMs**: A member inquired about the possibility of building systems that use both **HRM** for tasks involving unseen things and **LLMs/transformers** for world knowledge.
   - A member responded that there isn't a known mechanism to integrate them well, but suggested ideas like a **GNN** trained to interact with a small reasoning model.
- **Discussion on System of Models**: A member suggested that, instead of tight integration, a system of models including a **world model** could be used.
   - In relation, a [Discrete Distribution Networks](https://discrete-distribution-networks.github.io/) link was posted.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1425928445791047771)** (290 messages🔥🔥): 

> `Model choice recommendations, Context length impact on performance, Tool call failures, Uncensored Models` 


- ****Qwen3 triumphs over code llama****: Members are suggesting [Qwen3](https://lmstudio.ai/models/qwen/qwen3-coder-30b) for coding tasks due to its **speed and accuracy**, running at **20 tk/s** compared to CodeLlama at **5 tk/s**.
   - However, users note that even basic coding tasks need models greater than **20b parameters**, so one user mentioned [Llama 3.3 70b](https://lmstudio.ai/models/meta-llama/Llama-3-70B-Instruct-GGUF) would be better.
- ****Context Length's Unbearable Lightness****: Users explored how context length affects performance; even if context is empty, **higher context allocation slows generation time**.
   - LM Studio is implementing guardrails for memory allocation controlled by the memory estimator, but you can disable them at your own risk.
- ****Tool Call Tantrums Trigger Termination****: LM Studio should **not** be aborting generation when a tool call fails, but currently it is, and the failure message is not being displayed.
   - One possible solution is to disable MCPs (Multi-Call Procedures) to prevent model confusion, or look into fetch website tools because Playwright is *extremely token inefficient*.
- ****Safety-Maxed Models Suffer Story Stifling****: GPTOSS-20b is described as a *safety-maxed model* due to OpenAI's restrictions, so it's less permissive for creative storytelling.
   - It's recommended to base comparisons on GPTOSS 20b, because *every single model that exists is less protective* and to try Mistral-based models, because they are *very liberal outta the box*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1425933122830340158)** (26 messages🔥): 

> `Sparkle Arc Pro B60 Dual Server, Nvidia 4080 Ti Super vs Apple M3, GPU Benchmarks for LLM Inference, Server GPUs vs Multi-GPU Setups, Galax Single Slot RTX 5060 Ti GPU` 


- **Sparkle's Server Shines with 16 Arc GPUs**: Sparkle has introduced the [Arc Pro B60 dual server](https://videocardz.com/newz/sparkle-unveils-arc-pro-b60-dual-server-with-16-gpus-and-up-to-768-gb-of) boasting **16 GPUs** and up to **768 GB of VRAM**, powered by a **10800W PSU**.
- **4080 Ti Super Duels with M3 for Supremacy**: Members discussed whether a **Nvidia 4080 Ti Super** with **16GB VRAM** and **32GB RAM** is superior to an **Apple M3** with **36GB RAM** for machine learning tasks.
   - One member suggested checking [GPU benchmarks](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) for LLM inference to compare performance, noting that an **M3 Max 40-Core GPU with 64GB** is a comparable configuration.
- **Server GPUs Showdown Against Multi-GPU Rigs**: A member expressed surprise that server GPUs aren't significantly faster than multi-3090 setups, citing a small **10 t/s (60%-16vs25) difference**.
   - Others clarified that Mac Studio GPUs aren't designed for LLM inference and that server GPUs prioritize **memory capacity** over raw processing power, while multi-GPU setups may face slowdowns due to model splitting.
- **RTX 5060 Ti Rumors Rile Up Tech Enthusiasts**: A member shared a link about [Galax's single-slot GeForce RTX 5060 Ti GPU](https://wccftech.com/galax-single-slot-geforce-rtx-5060-ti-gpu-16-gb-vram-blower-fan/) with **16 GB VRAM** and a blower fan.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1425921476288057478)** (270 messages🔥🔥): 

> `API Keys in Models Page, AutoHotkey and Cursor Integration, Cursor Plan Pricing, GPT-5 Cost, Apply button in ASK mode` 


- **Unlocking Model Access with API Keys**: One user inquired about putting **API keys** in the **Models** page, and a response clarified that doing so allows using personal API keys with the models.
   - It was recommended to turn off all models and add them manually.
- **Orchestrating Cursor with AutoHotkey Control**: A user suggested using **AutoHotkey (AHK)** to control **Cursor** from a **Discord** channel.
   - Others seemed enthusiastic and will try Cursor to generate some **AHK scripts**.
- **Debating Value and Auto's Price in Cursor's Plans**: Users discussed the increased cost of the **Auto** model and reduced usage limits in **Cursor's Pro** plans, with some considering alternative models or services.
   - One user stated *The real issue is this: [https://forum.cursor.com/t/the-pro-account-limits-have-been-clearly-reduced/134738](https://forum.cursor.com/t/the-pro-account-limits-have-been-clearly-reduced/134738)*.
- **GPT-5 coding comes cheap and snappy**: A member touted **GPT-5** is good and cheaper than other options.
   - Another member said one downside is it requires more time because of its reasoning.
- **Lamenting the Loss of Apply Button in ASK mode**: Users reported that the **APPLY** button in **ASK** mode had disappeared, making it harder to apply the changes to the file.
   - Some stated that they don't like **AGENT** mode because it goes rogue, but that they are now forced to use that.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1425930578573787146)** (6 messages): 

> `Agents responding with Hello, Linear integration issues, Using Background Agents` 


- **Agents struggle to respond**: A member reported basic prompts like *"Respond with hello"* sometimes fail, attaching an [image](https://cdn.discordapp.com/attachments/1367213641027551352/1425934080939266048/image.png?ex=68eab599&is=68e96419&hm=21ad0141f990885a040fcda6e7beb3f542520958576fa5f3d5734ac945352dcb&) to show the agent's failure.
   - Another member said that it's working pretty consistent for them.
- **Linear integration bug found**: A member encountered an issue with **linear integration** and tried reconnecting **GitHub** and **linear** but the problem persists, as shown on their [screenshot](https://cdn.discordapp.com/attachments/1367213641027551352/1426297823498076191/Screenshot_20251010-125638.png?ex=68eab6dc&is=68e9655c&hm=10b31c0bc7e1118c9626e3f3868bd20e89d6aa1cc9574a670e37c5781bcd32c7&).
- **Workflow for Background Agents**: A member shared their workflow for using **Background Agents**:
   - The workflow is:
1) Create a new BA to code a new feature.
2) Allow the BA to implement the code suggestions.
3) Interact with the BA to review code, fix bugs, remove hallucinations.
4) Merge the new code changes into the main branch.
- **Background Agents lose context**: After code is merged into main, the **Background Agent** is shutdown and so the learning degrades over time, suggesting that the *work should be done contextualizing a BA to write Python*, instead of coding a specific *feature*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1425957118879465502)** (2 messages): 

> `CUDA kernels, Trainium Platform, High-Level Books` 


- **Trainium Platform Sparking Curiosity**: A member shifted focus from **CUDA kernels** to the **Trainium platform**, curious about the number of active developers.
   - They shared their [exploration blogpost](https://numbersandcode.wordpress.com/2025/10/08/trainium-exploration/) and observed minimal discussion or a dedicated channel.
- **Disses on High-Level Technical Books**: A member recalled reading a high-level technical book upon release and disliking it.
   - They felt the book lacked depth, contrasting with a *"line by line from scratch"* approach.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1425932082995003463)** (12 messages🔥): 

> `ldmatrix tiling calculation, optimal swizzling in linear layout, deterministic atomic_add reduction in Triton, Triton community meetup` 


- ****ldmatrix** tiling calc's logarithm logic located!**: There was agreement that in the `ldmatrix` tiling calculation, **d should be log_2(4/w)** where *w* is the byte width.
   - It was confirmed that *all this is implemented in the different lowerings in triton* so in case of doubt you can check the [source code](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.cpp#L186-L199).
- ****Optimal Swizzling Secrets** Spilled**: Discussion arose on how optimal swizzling in linear layout works, pinpointing disagreement between Figure 5, Sections 5.4 and 9.2 of the paper.
   - The best companion to understand this algorithm is its first implementation at [GenericSwizzling.cpp in pull 6982](https://github.com/triton-lang/triton/pull/6982).
- **Turnstile Reduction for **Deterministic Atomic Adds****: A member inquired about achieving deterministic atomic_add reduction in Triton, noting that CUDA has something like *"turnstile reduction"* where CTA waits on previous K blocks to finish first thru a barrier.
   - No specific Triton solution was offered in this message history.
- ****Triton Community Meetup** Scheduled for Next Year**: The next Triton community meetup will be on **Nov 5th, 2025 from 10am-11am PST** and here's the [meeting link](https://tinyurl.com/2s3z953y).
   - Tentative agenda: *TLX updates*, *Triton + PyTorch Symmetric Memory*, and *Triton Flex Attention in PyTorch*.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1425961852675817524)** (24 messages🔥): 

> `Thread block execution order, Blackwell CLC, CUB library, cuda::ptx::mbarrier_try_wait, Model cudagraph capture` 


- **Thread Block Order an Unconfirmed Delight?**: A member inquired about the guarantees of thread block execution order, particularly if **(0,0)** always runs before **(10,10)** when there are more thread blocks than the system can run.
   - While one member recalled documentation supporting a lowest linearized ID approach, another stated that it is *not officially guaranteed/documented/supported*, classing it as **undefined behavior (UB)**, although the **CUB library** likely accounts for this.
- **CUB hides Thread Block Order Complexity**: A member shared a [YouTube video](https://youtu.be/VLdm3bV4bKo?si=o4vi1dOK3sc7U-kH) showcasing an abstraction in **CUB** that deals with thread block ordering, and the [TilePrefixCallbackOp](https://nvidia.github.io/cccl/cub/api/structcub_1_1TilePrefixCallbackOp.html#_CPPv4NK3cub20TilePrefixCallbackOp10GetTileIdxEv) can be used outside an actual scan but only for a **1D index**.
- **Try MBarrier, Proceed with Caution**: A member asked if `cuda::ptx::mbarrier_try_wait` is non-blocking, and another member clarified that `test_wait` is non-blocking, while `try_wait` is *potentially blocking*, requiring a check of `waitComplete` after getting the result, referring to the [NVIDIA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait).
- **CUDA Cores: Many to a Warp?**: A member asked if a warp gets assigned to a single **CUDA core** or if multiple **CUDA cores** are needed to execute all 32 threads of a warp.
   - Another member responded that a warp scheduler keeps track of multiple warps and issues a single instruction per clock, and each instruction can be executed across any number of cores (e.g. 16), implying multiple cores are often involved.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1425965909725675661)** (4 messages): 

> `parrot lib, InferenceMAX, ATLAS, Billy Dally hardware` 


- ****Parrot** Flies High with GPU Array-Based Library!**: Connor Hoekstra's **parrot** is an array-based library in C++ designed to run on GPUs, shared in [this tweet](https://x.com/blelbach/status/1976255534467571730).
- ****InferenceMAX** Powers Up Open Source Inference!**: [InferenceMAX](https://inferencemax.semianalysis.com/) is an open-source project focused on inference, further detailed in [this newsletter](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference) and available on [GitHub](https://github.com/InferenceMAX/InferenceMAX).
- **Billy Dally Plugs New Hardware for AI Agents!**: A [YouTube video](https://www.youtube.com/watch?v=thsm6tff8h0) features Billy Dally discussing hardware designed for **AI agents**.
- ****ATLAS** Navigates New LLM Inference with Runtime-Learning Accelerators!**: Together AI introduces the **AdapTive-LeArning Speculator System (ATLAS)**, a new paradigm in **LLM Inference** via Runtime-Learning Accelerators, as detailed in [this blog post](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas) and [tweeted by Tri Dao](https://x.com/tri_dao/status/1976692444977938499).


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1426312418694402172)** (1 messages): 

> `GPU Performance Engineer Hiring, NVIDIA GPU Architecture, Kernel Optimization, Software-Hardware Co-design` 


- **GPU Performance Engineer Position Available!**: A **GPU Performance Engineer** position is open, requiring strong understanding of **NVIDIA GPU architecture** (**Blackwell**, **Hopper**) and experience optimizing **kernels**.
   - The position offers **$250K USD + Equity** and welcomes candidates with any level of experience; send your Github / CV to apply.
- **NVIDIA GPU Expertise Sought for Performance Role**: The ideal candidate should possess a strong grasp of **NVIDIA GPU architecture**, specifically **Blackwell** and **Hopper**, along with kernel optimization skills.
   - Experience with **CuTe**, **CUTLASS**, profilers, Linux kernel, driver internals, and software-hardware co-design are highly valued.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1425979778690715819)** (6 messages): 

> `Distributed Training Libraries: TorchTitan vs NVIDIA-Nemo, CUDA Kernel Debugging in Visual Studio, 4D Parallelism, Megatron Core, TorchTitan's Adaptability` 


- **TorchTitan vs NVIDIA-Nemo face off**: A newbie asked about choosing between **TorchTitan** and **NVIDIA-Nemo** for a **256 H200** training job.
   - A member suggested that NVIDIA's library is better unless extensive hacking is needed, as **TorchTitan** is more adaptable but NVIDIA-Nemo is more performant.
- **CUDA Kernel Debugging Unveiled in Visual Studio**: A new user shared a method for debugging **CUDA kernels** within **Visual Studio**, including a screenshot showcasing the debugging interface.
   - Others suggested that many use other debuggers, like **burn**, **rig**, and **ndarray**.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1426255066486276308)** (2 messages): 

> `GPU Programming with CUDA, Resources for learning CUDA, CUDA Projects for Students` 


- **Student Seeks Guidance on GPU Programming with CUDA**: A student asked for advice on how to get into **GPU programming** using **CUDA**, including project and topic suggestions.
   - Another member directed them to check the resources in the channels [<#1198358627594023014>](https://discord.com/channels/1169494361482651678/1198358627594023014) and [<#1191300313928433664>](https://discord.com/channels/1169494361482651678/1191300313928433664).
- **CUDA Resources**: Resources for **CUDA** can be found in the relevant discord channels.
   - It is recommended that those who are interested in learning should check those channels.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1426026871576985681)** (3 messages): 

> `Triton Projects, Picograd` 


- **Triton Project Brainstorm Initiated**: A user suggested finding a project or idea that requires the use of **Triton** and then building it.
   - This prompt aims to stimulate innovation and practical application of Triton within the community.
- **Call for Triton Kernels in Picograd**: Another user suggested adding **Triton kernels** to the **picograd** project in the relevant channel.
   - One user responded, *"picograd is awesome :)"* suggesting enthusiasm for integrating **Triton** with **picograd**.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1426021261204656238)** (6 messages): 

> `Panther Lake, Xe3 Details, Celestial Architecture Speculation, Memory Bandwidth Bottleneck` 


- **Panther Lake's Xe3 Compute Boost**: New [Panther Lake slides](https://www.techpowerup.com/review/intel-panther-lake-technical-deep-dive/13.html) reveal more compute per slice, up to **16 MiB of L2$**, and **two more threads** with variable register allocation.
   - These changes are claimed to deliver a **40% performance-per-watt uplift** relative to Arrow Lake, addressing memory bandwidth constraints.
- **Memory Bandwidth Bottleneck Solved**: Memory-bound tasks on **Battlemage** suffered on **Lunar Lake** due to reduced global memory bandwidth, but the extra **L1$/SLM and L2$** in Panther Lake should mitigate this.
   - Lunar Lake had a bandwidth reduction *by a factor of 2.7* relative to available compute, but increased cache may compensate.
- **Celestial Architecture Speculation**: The architecture of **Celestial** remains unclear, hinging on the desired ratio of compute-to-fixed-function units relative to **Panther Lake**.
   - The best assumption is that **Celestial** will use **six-subslice slices**, but they could increase or decrease that given the architecture.
- **Xe2 Kernel Compatibility**: Since Xe2 is architecturally close to Xe3, kernels developed on Xe2 GPUs *should translate very well forward* to **Xe3**.
   - We *probably won't get a hint of Celestial for a while* according to one member.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

vipul_todo_18: Other people's promotion: 

https://x.com/jyo_pari/status/1976324891545829876
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1426071303974162514)** (1 messages): 

> `ThunderKittens compilation issues, Nvidia GH200, CUDA 12.3, fp8e8m0 undefined type` 


- **ThunderKittens screams at CUDA 12.3**: A member is trying to compile **ThunderKittens** on an **Nvidia GH200** machine (**arm64**) with **CUDA 12.3** and encounters undefined types like `__nv_fp8_e8m0`.
- **fp8 woes in CUDA 12.3 linger**: The undefined types `__nv_fp8_e8m0`, `__nv_fp8x2_e8m0`, and `__nv_fp8x4_e8m0` appear when using both `python setup.py install` and the provided Makefiles.
   - It is unclear whether a newer **CUDA** version is required or if there's another issue hindering the compilation.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1426001983659180134)** (26 messages🔥): 

> `amd-gemm-rs Leaderboard Updates, amd-all2all Leaderboard Submissions, MI300x8 Performance, amd-ag-gemm Submissions` 


- **Gemm-tastic Gains on MI300x8**: One member achieved **2nd place** on the `amd-gemm-rs` leaderboard with a time of **516 µs** on **MI300x8**.
   - Other notable placements include **4th place** at **530 µs** and **9th place** at **536 µs**, demonstrating competitive performance tuning.
- **All2All Arena: MI300x8 Thrives**: Multiple successful submissions were recorded on the `amd-all2all` leaderboard using **MI300x8**, with times ranging from **443 µs** to **1767 µs**.
   - One user secured **7th place** with a time of **566 µs**, highlighting ongoing efforts to optimize all-to-all communication.
- **AG-GEMM Achieves Acceleration on MI300x8**: Successful submissions to the `amd-ag-gemm` leaderboard on **MI300x8** were recorded, consistently achieving times around **470-520 µs**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1426341075173380299)** (1 messages): 

> `Runner Timeouts` 


- **Runner Timeouts Plague**: There are currently **issues with runners** that are causing unexpected **timeouts**.
   - The team is actively investigating and promises an update soon.
- **Runner Timeout Investigation**: A team is investigating **runner timeouts** that are occurring unexpectedly.
   - An update will be provided as soon as possible regarding the cause and resolution of these issues.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1426225361980489820)** (4 messages): 

> `COVID, Factorio Crime Scene` 


- ****Absence Acknowledged:** COVID sidelines member**: A member announced they have **COVID** and will miss today's meeting, requesting updates on any important conclusions.
   - Another member responded with *'Get better soon!'*.
- ****Factorio Fiasco:** Unexpected In-Game Incident**: A member left **Factorio** running to attend a meeting and returned to find a 'crime scene' in the game.
   - An image was attached ([Screenshot](https://cdn.discordapp.com/attachments/1354169122107293786/1426296785088942151/Screenshot_2025-10-09_at_2.58.39_PM.png?ex=68eab5e4&is=68e96464&hm=7bc9686f5888df3fa014a43d97a1420560317bf36528519d13716e475ea90b69&)) with the comment *'that was new'*, suggesting an unprecedented in-game event.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1426232820342194320)** (16 messages🔥): 

> `AMD Cluster Timeout Issues, Memory Access Fault Debugging, Jot Runner Timeout Extension` 


- ****AMD Cluster** has Timeout Troubles**: Users reported timeout errors on the **AMD cluster**, with jobs timing out due to a **5-minute limit** on `queue_time + running_time`.
   - A maintainer acknowledged the issue was on the **AMD side**, requiring their intervention, and noted *it might take a while* to resolve.
- ****Segfaults** Signal Memory Access Mishaps**: A user encountered a *memory access fault* on **GPU node-7**, which indicates an illegal memory access in their code, often manifesting as a segfault.
   - A maintainer explained that local setups may mask these issues due to different permissions or luck, recommending step-by-step debugging with print statements.
- ****Jot Runner's** Timeout Extension Plea**: A user requested the Jot runner consider only `running_time` for timeout calculations, excluding `queue_time`, to prevent premature job termination.
   - A maintainer responded that it's difficult to implement but offered to extend the timeout window temporarily, requesting timely notifications of such issues.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1425954776444702870)** (28 messages🔥): 

> `Grouped GEMM performance for MoEs, PTX docs for K-contig and swizzling errors, Torch tensor support in CUTLASS, Compiler aborts with ldmatrix copy atoms, Debugging pipeline stalls` 


- ****MoEs' M-Occupancy Malaise****: The performance of Grouped GEMM for **MoEs** during inference is being evaluated, especially regarding **M-occupancy**, which can be significantly lower than in vanilla GEMM due to random token distribution per expert and a question was raised about its impact on traditional roofline models.
   - With `gpt-oss 20b` prefill phase, M-occupancy has been seen as low as ~60% due to wasteful compute.
- ****PTX Parallax? Precision Problematic****: There is an issue regarding the accuracy of **PTX documentation** for K-contiguous layouts and swizzling, specifically concerning tensor descriptors as shown in [this Triton code](https://github.com/triton-lang/triton/blob/b5fea1e3f4c2cb0b40c0ce98261b240d8728d2f9/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAHelpers.h#L250-L253) and [this NVIDIA doc](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-k-32b-swizzle-tf32).
   - The PTX folks have picked up the issue and should have a fix in version **13.1**.
- ****Torch Tensors Temptingly Transferred****: It is currently possible to directly pass **torch tensors**, though it assumes a full dynamic layout and still goes through a **DLPack path**.
   - Native support for bypassing **DLPack** is on the roadmap, more details are available [here](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#implicit-conversion).
- ****Compiler Cries 'Abort!' on ldmatrix****: A user reported encountering compiler aborts when utilizing **ldmatrix** copy atoms and created a code repro for the issue named `t.py`.
   - They also mentioned that many *user errors* show up as internal compiler errors (**ICEs**).
- ****Debugging Dilemmas: Pipeline Stall Sleuthing****: A user inquired about effective methods for debugging pipeline stalls, seeking more precise tools than the coarse-grained **nsight compute**.
   - Suggestions included using **nsight compute** for warp state statistics and **gluon** with **proton** for a timeline view of kernel execution.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1425970712660934767)** (1 messages): 

> `Discord Roles, Competition Winners, AMD Competition` 


- **Discord Server Adds Competition Winner Roles**: The Discord server has introduced new roles for competition winners: <@&1418285356490428476> and <@&1425969596296462356>.
   - The server plans to extend these roles to the victors of the ongoing **AMD competition** as well.
- **AMD Competition Winners to Receive Discord Roles**: Winners of the current **AMD competition** will also receive special roles on the Discord server, similar to previous competition winners.
   - This initiative aims to recognize and highlight the achievements of community members.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1425930522147815545)** (5 messages): 

> `5070 ti super 24gb vs 5080, Distributed training libraries: torchtitan vs nvidia-nemo, Tesla P40 24GB performance` 


- **5070 Ti Super 24GB Specs Speculated**: A member inquired about thoughts on a potential **5070 Ti Super** with **24GB** VRAM versus the **5080**, indicating anticipation for new GPU options.
- **TorchTitan faces NVIDIA-NeMo in training showdown**: A newcomer asked for advice on choosing between **TorchTitan** and **NVIDIA-NeMo** for distributed training on **256 H200s**, citing concerns about TorchTitan's maturity and potential inefficiency compared to NVIDIA-NeMo's Megatron-core.
   - They highlighted **NVIDIA-NeMo**'s proven efficiency for 4D parallelism at very large scales, while acknowledging **TorchTitan**'s accessibility but fearing its compute efficiency for training a **7B dense model** on **2 trillion tokens**.
- **Tesla P40 surprises with decent 30B performance**: Despite its low cost, the **Tesla P40**'s performance was debated, with one member noting its average performance.
   - Another member shared a [Reddit benchmark](https://tokens-per-second-visualizer.tiiny.site/) showing the **P40** achieving **8 tps** on a **30b model**, which they felt was surprisingly good.


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1425984523409297478)** (2 messages): 

> `Project Teams, GPU Inference, Novel GPU Work` 


- **User Expresses Excitement to Join Project Teams**: A user expressed excitement about joining project teams, particularly those related to **inference projects** that could potentially contribute to their approval.
   - The user hopes they are not too late to join and is generally enthusiastic about contributing.
- **New project idea receives positive feedback**: Another user expresses enthusiasm about a potential project, describing it as a *great idea*.
   - They suggest that making the project work on **GPUs** would be **novel work** with interesting challenges and opportunities and asks another user to review the paper.


  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1426178066647879771)** (7 messages): 

> `Attention-Residual Recalculation, Contributor Guide, Weird Quantizations` 


- **Attention-Residual Recalculation Boosts Performance**: Recalculating the **attention-residual** during the backward pass was implemented in [llmq](https://github.com/IST-DASLab/llmq/pull/7), which can lead to significant net speed-up in cases with high memory pressure, like a **14B model on 4x4090**.
   - Using this optimization increased the throughput from **3.2k to 6.0k TPS** when training **Qwen2.5-14B** in **fp8**, and from **2.5k to 4.5k TPS** when training in **bf16**.
- **Contributor Guide Quandary**: A member inquired about a contributor guide for quick onboarding, setting expectations for contributions, and clarifying attribution on research based on contributions.
   - Specifically, they wanted to know the difference between just **GitHub contribution** versus possible attribution on research, if the contribution was significant.
- **Wacky Quantizations**: A member inquired about support for training in *weirder* quantizations and what it would look like to configure on the command line or config files.
   - The main developer clarified that quantized optimizer state should be relatively straightforward, even with weird quant formats, but for matmuls, you'd first need an actual matmul implementation for your quantization format.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1426288309868433409)** (2 messages): 

> `FLA Benchmark, GDN, Mamba2, PTC Talk` 


- **FLA Use Case as Convincing Benchmark**: A member expressed support for the **FLA** use case being a good and convincing benchmark.
   - He specifically mentioned **GDN** and potentially **Mamba2** as relevant models.
- **Anticipation for PTC Talk**: A member conveyed their excitement for an upcoming **PTC** talk.
   - No further details were provided regarding the topic or speaker.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1425920757413974036)** (108 messages🔥🔥): 

> `Colab Cost, Continual Learning, GPU Time on A100, Fine-tuning Llama3, HF Fellowship` 


- **Colab ain't Costly!**: Members discuss the affordability of **Colab** at **$10** per month, with one stating it provides approximately **13.2 hours** of **A100 GPU** time, costing about **75 cents/GPU/hour**.
   - One user with *"less than a Swiss franc in my bank account"* was encouraged to seek allowance from parents.
- **Continual Learning Critiques**: A member explains the challenges of continual learning, suggesting current papers focus on *"curing symptoms but not cause"* and proposes **designing principles** for the model to learn what and how much to remember based on reward signals.
   - They suggest that the *optimization function* should be a learnable **RNN** working towards increasing rewards for every action, aligning with **Richard Sutton's** ideas on emergent imitation learning in AGI.
- **Fine-Tuning Frustrations**: One member seeks advice on reducing the training time for fine-tuning a **Llama3 8B** model on a **200,000** dataset with a token length of **2048**, estimating it will take over **5 days** on a **40GB GPU**.
   - Another user says they need to download llama at some point as well as get contr net working.
- **HF Fellowships are Happening**: A member mentions that *prithiv* is now a **Hugging Face Fellow** and points to [his spaces and models](https://huggingface.co/prithivMLmods) as highly practical.
   - Another user asks *"how do people become a hugging face fellow ?"*, suggesting *"seems like have to contribute to ML stuff"*.
- **MoE Models Muster Momentum**: One member seeks a good open source **MoE** (Mixture of Experts) model with configurable total parameters for pretraining.
   - Another member responds *"i cant wait to pretrain a 24M parameter moe"*.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1425982556997681264)** (1 messages): 

> `Hyperparameter Diffusion, Faster Image Generation, Compute Reduction` 


- **Hyperparameter Diffusion Generates Breakthroughs**: A member launched a HuggingFace Space showcasing an implementation of the paper [Hyperparameters are all you need](https://arxiv.org/abs/2510.02390), demonstrating that **8 steps** can achieve comparable FID performance to **20 steps** in image generation.
   - The implementation includes a [Counterfeit v3.0 version](https://huggingface.co/spaces/coralLight/Hyperparameters_Are_All_You_Need), an [XL version](https://huggingface.co/spaces/coralLight/Hyperparameters-are-all-you-need-xl-version), and an [original SD version](https://huggingface.co/spaces/coralLight/hyperparameters-are-all-you-need-sd-version) for testing.
- **Diffusion is 2.5x Faster and Better**: The new method achieves **2.5x faster** image generation with **better quality** compared to DPM++2m, and it works with any model.
   - The member noted that no training or distillation is needed, resulting in a **60% compute reduction**.
- **Hyperparameters Reduce Compute in Diffusion**: It's shown that **8 steps** can generate images with FID performance comparable to the **20 steps** achieving **60% compute reduction**.
   - A member also shared some example images generated using this method, and solicited feedback on which models to test next.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1426172522218655805)** (2 messages): 

> `BERT model for Polish tweets, MedScan AI medical tool, NLP in healthcare` 


- **BERT Model Speaks Polish!**: A member fine-tuned a **BERT model** to predict sentiment and emotions in **Polish language tweets**, which can be found [here](https://huggingface.co/spaces/yazoniak/twitteremo-pl-classifier).
- **MedScan Launches as Friendly AI Doctor!**: A member launched **MedScan**, an AI tool built on **Hugging Face models** for smart medical search and report analysis, accessible [here](https://ragmedscan.vercel.app/).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1426043548234547303)** (1 messages): 

> `Custom model vs finetune, Text encoder model` 


- **Clarification on Model Type Requested**: A member inquired whether a model was a custom model or a finetune.
- **Text Encoder Model Questioned**: A member asked which text encoder model was being used.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

cakiki: <@892799950470144060> no cross-posting please
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1425927260216496178)** (101 messages🔥🔥): 

> `Datacurve Funding, Spellbook Funding, OpenAI AMA Cancellation, Kernel Funding & Agent Auth, Elastic Acquires Jina AI` 


- **Datacurve Nabs $17.5M for Frontier LLM Training**: Datacurve announced a combined **$15M Series A** and **$2.7M Seed** round to supply high-quality training datasets for foundation models, especially for coding, as per [Serena Ge's announcement](https://xcancel.com/serenaa_ge/status/1976328983458480539).
   - The funding was led by Chemistry, with participation from YC, Cohere, Afore, and various angel investors.
- **Spellbook Spells Out $50M Series B for AI Contract Drafting**: Spellbook, positioning itself as *Cursor for contracts*, closed a **$50 million Series B** led by Khosla Ventures, as announced by [Scott Stevenson](https://xcancel.com/scottastevenson/status/1976280608436572393).
   - The AI drafting & review platform has grown to **4,000 customers** since 2022, with suggestion-accept rate jumping from **5% to 60%**, and new funding will drive product enhancements, starting with a realtime market-comparison beta.
- **Kernel Kernels Up $22M to Fix LLM Cloud**: Kernel, led by CEO Catherine Jue, announced a **$22M Seed + Series A** round led by Accel, for automating browser workloads at scale and launching Kernel Agent Authentication, per [this announcement](https://xcancel.com/juecd__/status/1976325764166615498?s=46).
   - Customers include CashApp & Rye, and Kernel Agent Authentication provides an identity layer giving AI apps safe, scoped, fully-auditable control over user actions.
- **Elastic Swallows Jina AI for Multimodal Search**: Elastic has acquired Jina AI to strengthen their retrieval, embeddings, and context-engineering capabilities to power agentic AI, as [announced by Elastic](https://xcancel.com/elastic/status/1976278980018765886).
   - The community has overwhelmingly applauded the move, highlighting its potential to reinforce Elastic's enterprise search and AI agent dominance.
- **Unlimited Claude for $3 via GLM Hack?**: A user claimed Chinese reverse-engineering unlocked an *unlimited Claude coding* tier for just **$3/mo** by routing requests to **GLM-4.6** on z.ai instead of genuine Sonnet, per [this claim](https://xcancel.com/shydev69/status/1976641232622453045).
   - Others questioned latency and actual Claude quality, with one user conceding it’s not suitable for their own work due to GLM offering no controllable thinking mode, however [the blogpost for the hack](https://shydev.medium.com/get-unlimited-claude-code-for-3-53d61d5b2b2f) is gaining traction.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1426227622077272146)** (8 messages🔥): 

> `Gemini Flash 2.5 Nano Banana, JSON prompting, nano-banana AI outputs, replicating nano-banana AI outputs, removing nano-banana AI outputs` 


- **Gemini Flash 2.5 Nano Banana: Prompt Showcase**: Emily shared a detailed **JSON prompt** for **Gemini Flash 2.5 Nano Banana** that generates a blue-accented, anime-style mirror selfie of an East-Asian woman in her PC-corner bedroom, showcased in [this Tweet](https://x.com/iamemily2050/status/1976431328280416520?s=46).
- **JSON vs Natural Language Prompting: Debate Sparked**: Followers debated **JSON** vs **natural-language prompting**, while Emily advocated for JSON’s advantages for reproducible, controlled workflows.
   - Users posted their own replications in [this tweet](https://x.com/toyxyz3/status/1976650667046605263?s=46).
- **Nano-Banana AI Outputs: Soul Mark Debate**: Users discussed faint, repetitive artifacts seen on multiple **nano-banana AI outputs**, speculating whether it’s an intentional watermark, a transformer artifact, or simply a generational quirk.
- **Artifact Removal Tips Shared**: Tips on replicating (greyscale then oversaturate), removing (upscaling), and the absence of any tracking ID were shared, with a mix of jokes (*it’s not a watermark, it’s a soul*) and technical suggestions.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1426236885818150952)** (15 messages🔥): 

> `ROPE Implementation in llama.cpp, Precision Sensitivity in RoPE Calculations, Neural Theorem Proving Channels` 


- **ROPE Rescue Requested for llama.cpp**: A request was made for assistance with a snag in implementing **hybrid attention ROPE** in a *llama.cpp* merge request, specifically related to [this GitHub pull request](https://github.com/ggml-org/llama.cpp/pull/16095#issuecomment-3390312481).
- **HF Implementation Lacks Partial ROPE?**: A member noted that the **Hugging Face implementation** doesn't appear to use *partial ROPE*, questioning its presence only in the config, referencing [this HF code](https://github.com/huggingface/transformers/blob/0419ff881d7bb503f4fc0f0a7a5aac3d012c9b91/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L176).
- **ROPE precision can cause inaccuracies**: It was suggested that **RoPE calculations** are sensitive to precision and that the sine and cosine matrices should be computed in **fp32** to avoid inaccuracies, citing [EleutherAI's gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/positional_embeddings.py#L63) as an example.
- **BF16 Long Context Precision Woes**: A member mentioned a recent paper indicating that **ROPE** used with **BF16** in long contexts can lead to precision issues.
- **Seeking Channel for Neural Theorem Proving**: A member inquired about a dedicated channel for **neural theorem proving** questions and wondered whether to post in the <#747850033994662000> channel.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1425932393251864576)** (71 messages🔥🔥): 

> `OOD Generalization and Training Loss, Vision Language Model optimization, Cosine Decay vs Infinite LR for Scaling Laws, Warmup Stable No-Decay Training, Scalar RMSProp Adaptivity` 


- **OOD Generalization Paradox**: A counterintuitive connection between **training loss** and **out-of-distribution (OOD) generalization** was discussed, questioning why **nGPT's architecture** might fail to generalize.
   - One suggestion involved **length generalization** or increased vulnerability to exposure bias, particularly considering generalization between *"seen" samples*.
- **Vision Language Model Resolution Optimization**: A member shared initial results from a **benchmarking project** focused on optimizing **image resolution** against **output quality** in Vision Language Models (VLMs), using **Gemini 2.0 Flash** on **COCO 2017 Val dataset** for captioning tasks, with a [report attached](https://cdn.discordapp.com/attachments/747850033994662000/1425953022671847515/minres_report.pdf?ex=68eac73d&is=68e975bd&hm=194b9521fd489b4596bc9cdb078a3f90adc9533c1be618a2aaa1d0174f450c82&).
- **Optimizers Compared for Scaling Law Suites**: Members discussed comparing **cosine decay**, **cosine decay with annealing**, and **infinite LR with annealing** for scaling laws, questioning if infinite LR with annealing enables cheaper scaling law suites via partially trained checkpoints.
   - A member linked a paper about using **warmup stable no-decay** and **checkpoint averaging** ([https://arxiv.org/abs/2507.17634](https://arxiv.org/abs/2507.17634)), noting its use in trillion-parameter models.
- **Debate over Scalar RMSProp**: A member contested a claim about **Scalar RMSProp**, arguing its adaptivity isn't tied to the maximum stable step size; every optimizer hits this at the end of the step, due to sharpness adapting to the optimizer.
   - Counterarguments included discussions around regions in parameter space and sharpness regularization.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1426205203807600770)** (4 messages): 

> `Mechanistic Interpretability, Attribution Graphs, Circuit Tracing, Model Biology` 


- **LessWrong Post Guides Aspiring Interpretability Researchers**: A member shared a [LessWrong post](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher) about how to become a **mechanistic interpretability researcher**.
- **Anthropic's Circuit Tracing Explained**: A member shared a [YouTube video](https://www.youtube.com/watch?v=ruLcDtr_cGo) providing a *nice introduction* to **attribution graphs** from **Anthropic's Circuit Tracing and Model Biology papers**.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1425920864968507402)** (1 messages): 

> `Moxin-VLM, VLM-R1` 


- **New VLMs Arrive: Moxin & R1**: A member shared links to two new Vision Language Model (VLM) GitHub repositories: [Moxin-VLM](https://github.com/moxin-org/Moxin-VLM.git) and [VLM-R1](https://github.com/om-ai-lab/VLM-R1.git).
- **VLM landscape expands**: The landscape of open source Vision Language Models (VLMs) expands with the addition of [Moxin-VLM](https://github.com/moxin-org/Moxin-VLM.git) and [VLM-R1](https://github.com/om-ai-lab/VLM-R1.git), offering new avenues for research and application.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1425929921154388101)** (46 messages🔥): 

> `Atropos tutorial video, Training Qwen3-Next, Predicted Outputs in vllm, r/LocalLLaMA post removal` 


- **Atropos Tutorial Video Surfaces Online**: A member asked about a quick tutorial video on YouTube/Xwitter for using **Atropos** and a broader overview of how environments work in Atropos was located on [Twitter](https://fxtwitter.com/NousResearch/status/1925381160097697803) and also [YouTube](https://www.youtube.com/watch?v=in__ELD4NxE).
- **Challenges Training Qwen3-Next Model**: Members discussed issues related to training **Qwen3-Next**, including hangs and slow checkpoint shard loading, with one reporting a slow loading phase of **8 minutes**.
   - Another member reported a **14% MFU** without experiencing any issues.
- **Predicted Outputs in vllm**: New tech is releasing that is **Predicted Outputs in vllm** where very fast generation occurs by converting output to prefill for (partial) matches of a prediction.
   - Check out the [blog post](https://cascadetech.ai/blog/vllm-predicted-outputs/), [demo](https://app.cascadetech.ai/), and [tweet thread](https://x.com/saganite/status/1976707696578691101) for more.
- **r/LocalLLaMA Post Removal Questioned**: A member inquired why their r/LocalLLaMA post about *fast predicted outputs in vllm* was removed from [this reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1o39l0l/fast_predicted_outputs_in_vllm/).
   - It was suggested that some of the subreddit's moderators are present in the Discord channel but they were unsure who.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1426225124994056293)** (4 messages): 

> `FP8 LLM Finetuning, QLoRA effectiveness, Test Time RL, LoRA Precision` 


- **FP8 Finetuning: Worth the Squeeze?**: A member inquired about experimenting with finetuning an **FP8 LLM** and how **gradient descent** behaves.
   - Another member responded that **FP8** training for full precision fine-tuning doesn't look worthwhile currently due to *spotty support* in accessible training frameworks and dependency on Torch compile.
- **QLoRA: Precision Power-Up!**: A member asked about the effectiveness of **QLoRA** finetuning.
   - Another member stated that **LoRA** doesn't heavily depend on the model's precision for training stability since the LoRA parameters are trained in **BF16/FP32**, and **FP8 LoRA** should maintain full quality versus BF16, with only a slight decrease in quality for **QLoRA**.
- **Test Time RL buzz**: One member refers to "Test Time RL" on context: [https://fixupx.com/rryssf_/status/1976269613072843063](https://fixupx.com/rryssf_/status/1976269613072843063).
   - They clarify it's more of a *compliment* to **RL**, not a replacement, and that they've been building tools on this intuition for months!


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1425953936938111119)** (18 messages🔥): 

> `Constant Params * Samples / Data Ratio, ThinkingMachine's LoRA Framework, LoRA vs FFT, Information Bottleneck in RL, Robust Fine Tuning Strategies` 


- **Ratio Requirements Revealed for Parameter Efficiency**: A member pointed out that a recent paper ([arxiv.org/html/2410.21228v1](https://arxiv.org/html/2410.21228v1)) shows that you need **(params * samples / data) to be constant** for efficient training.
   - The member wished the paper explored training token amounts beyond **Chinchilla** (13B on 260B tokens) to provide more insights, as current frontier models often exceed **20T tokens**.
- **LoRA Fine-Tuning May Cause Catastrophic Forgetting**: A member noted that looking at **loss** might not be enough to determine if catastrophic forgetting occurs during fine-tuning, as argued in the paper.
   - They suggest evaluating on both new training tasks and pre-training evals to gauge forgetting and inspecting the **singular values of LoRA adapters** to check for correlations.
- **LoRA vs FFT: The Battle for Supremacy**: A member recalled comparing the maximum potential of **LoRA** vs **FFT** in their early AI days, noting that loss is insufficient to determine the best method.
   - The discussion referenced **Thinking Machines' blog**, which also argues that many environments use **LoRA** only across attention layers, which is ineffective ([thinkingmachines.ai/blog/lora/](https://thinkingmachines.ai/blog/lora/)).
- **Information Bottleneck in RL Unveiled**: A member shared an interesting image demonstrating the **information bottleneck in RL**.
   - No additional details were provided.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1425953936938111119)** (18 messages🔥): 

> `LoRA, FFT, 8bit, Information Bottleneck, Thinking Machines Blog` 


- **Paper Faces Rejection from ICLR 2025**: A member shared a link to a [paper](https://arxiv.org/html/2410.21228v1) that was rejected from **ICLR 2025**, noting its insights on **LoRA** training and the importance of keeping *(params * samples / data)* constant.
   - The member criticized the paper for only testing the **Chinchilla** scaling laws and not attempting to scale to current models trained on **20T tokens**.
- **LoRA SFT needs Pretraining Evals**: After **LoRA SFT**, an informed instruction fine-tuning pipeline should *perform pre-training evals on SFT model* to evaluate how much forgetting occurred, according to a member.
   - They suggest also inspecting **singular values** of LoRA adapters to see how much these intruder dimensions are correlated with pre-training forgetting.
- **Loss is not enough to determine**: Members argued that **loss** alone is insufficient to determine the maximum potential of **LoRA** vs **FFT**.
   - Another member agreed with a new [paper's](https://arxiv.org/html/2410.21228v1) argument to use both new training tasks and pre-training evals to gauge how much forgetting occurred during post training.
- **LoRA targets can learn new facts**: A member stated that even with **attention layer only targets**, **LoRA** can learn new facts and knowledge.
   - They pointed to the [Thinking Machines blog](https://thinkingmachines.ai/blog/lora/) to say the blog argues that many environments only use LoRA across the attention layers, which is ineffective.
- **Proving Information Bottleneck in RL**: One member found a demonstration of the **information bottleneck** in **RL** extremely interesting and shared a photo.
   - There was no link provided.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1425942869876805806)** (85 messages🔥🔥): 

> `MCP .well-known Endpoint, Representing Image Content in structuredContent, skybridge tool and outputSchema` 


- **MCP's `.well-known` Endpoint Plot Thickens**: Members are discussing a `.well-known/` endpoint for **MCP server metadata**, with references to a [blog entry](https://blog.modelcontextprotocol.io/posts/2025-09-26-mcp-next-version-update/#server-identity), [GitHub discussion](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1147), [pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1054#issuecomment-3117705161), and a [Dev Summit presentation](https://github.com/modelcontextprotocol/registry/blob/main/docs/explanations/dev-summit-2025-10-registry-status-presentation.pdf).
   - The aim is to push through a minimal SEP focusing on the document name, location relative to the MCP server URL, and minimal content (specifically, `Implementation`).
- **Image Content Standard Debates**: The discussion revolves around representing **image content** in `structuredContent`, with concerns that it's not portable and breaks on clients that pass `StructuredOutput` directly to model APIs.
   - It was suggested that the protocol doesn't prescribe how host apps map `structuredContent` to LLM APIs, and support for image returning tools is poor among providers; the solution is to give the model a temporary tool that maps string to an image, and add it when returning the structured content.
- **Skybridge Tool Skirts OutputSchema**: The discussion highlights that **skybridge** doesn't use `outputSchema` on the tools, and members explored whether there is an opportunity to define any kind of thing for that.
   - Members discussed the dissimilarity of `ContentBlock[]` and the `structuredContent`, in which it was said that the `structuredContent` is used for widget hydration - but it's also available to the model.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1426002234964967456)** (5 messages): 

> `Mojo language, GPU compatibility` 


- **Mojo predicted as future language**: Members welcomed a new user, predicting that *Mojo será el lenguaje del futuro* <:mojo:1098996988797784084>.
   - The user also stated, *También soy de Colombia 🙈*.
- **Backwards compatibility woes in GPU**: A user questioned whether a certain approach suffers from lack of forward compatibility.
   - They asked, *Doesn't this aproach have the problem of no forwards compatibility? You'd have to recompile to support each new gpu generation wouldn't you?*


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1425926868682674257)** (64 messages🔥🔥): 

> `Apple Silicon M3 GPU support, Jetson Orin Nano for Robotics, Mojo Native FFT Implementation, SIMD vector width loading tricks, Complex number memory interleaving` 


- **Apple Silicon gets GPU Vector Addition**: A member tested **GPU support** on **Apple Silicon M3**, running a simple vector add example, encountering issues with the host buffer remaining unmodified, but solved with explicit synchronization.
   - It was also mentioned that *printing from inside kernel* on Apple GPU is not yet implemented, and `enqueue_copy` might be a no-op due to shared memory.
- **Jetson Orin Nano is Enough for Robotics**: Members discussed using **Jetson Orin Nano 8GB** for robotics, with one expressing affordability concerns but another suggesting it's sufficient for developing vision systems, especially for resource-constrained deployments where **battery life** is critical.
   - They highlighted that object detection and image classification models run well on the Orin Nano, enabling scaling up to larger systems later.
- **Mojo FFT implementation lands, is it faster than FFTW?**: After PR [#5378](https://github.com/modular/modular/pull/5378) lands there will be a **Mojo native FFT implementation**.
   - Members also discussed performance improvements versus **FFTW** with one user stating they need multidimensional transforms where the sizes are known only at runtime, for which fftw serves well enough for the CPU, but it would be nice to eventually have a GPU implementation as well.
- **Memory Interleaving is Fast for Complex Numbers**: It was discovered that for complex numbers with two components, reading memory directly into vectors with alternating real and imaginary elements can be efficiently deinterleaved, potentially improving memory locality and cache usage in **SIMD** work.
   - A member shared their loading tricks in this [open-source project](https://github.com/bartesaghilab/cryoluge/blob/main/src/cryoluge/image/complex_image.mojo#L98) and suggested using the compile package to view the output


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1426079606200471664)** (46 messages🔥): 

> `Forgetting in Models, IID Sampling Issues, Prompt Injection Defense, Separate Token Embeddings, Graph Neural Network Training` 


- **Architectural Tricks Alone Cannot Cure Forgetting**: A member argued that preventing **catastrophic forgetting** requires more than just architectural changes; the optimizer must evolve beyond **autodiff-based gradient descent**.
   - They suggested **hard-sectioning memories** into separate buckets and ensuring those buckets don't fundamentally forget, noting that models requiring **i.i.d. sampling** inherently struggle with forgetting.
- **IID Sampling Proves Problematic**: A member pointed out that **Deep Learning (DL)** methods struggle with **MNIST** when digits are sorted instead of random without sampling old digits.
   - They noted that DL without i.i.d. gets 10% (random) which highlights the reliance on **i.i.d.** for effective learning.
- **Separate Token Embeddings Stave Off Injections?**: A member proposed training a separate set of **token embeddings** for **system/user prompts** to make it easier for models to distinguish between prompt and content, thus reducing **prompt injection** vulnerability.
   - While some agreed it could create a safer model, others suggested that even with separate embeddings, the model might still collapse representations and be vulnerable to manipulation.
- **Soft Prompts Defend Against Injections**: In response to a proposed idea, a member noted that separate embeddings are called **soft prompts**, and could defend against **prompt injection** attacks.
   - They mentioned that while the token embeddings themselves aren't different, some embeddings are prepended to the context as real vectors that are found by backprop rather than being discrete.
- **GNNs Show Two Stage Learning?**: A member asked about a graph showing a **two-stage behavior** during the training of a **Graph Neural Network**.
   - Others suggested it might be due to hyperparameter settings or the end of the first epoch, with the network re-encountering the same input points.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1425925907264049232)** (6 messages): 

> `Cat Studies, Follow up paper` 


- **Feline Fine with Finetuning**: A member shared a picture of their cat *studying along with them* with the caption [link to the cat photo](https://cdn.discordapp.com/attachments/1045297868136779846/1425925906928767036/image.png?ex=68eaadfc&is=68e95c7c&hm=a38546b5f969ef9fde2a1e7aac3f142d7b823ee1be9436481c8f858d1ad0739e&).
- **Paper Chaser's Latest Pursuit**: A member shared a [cool paper](https://arxiv.org/abs/2509.24372) that follows up on [this paper](https://arxiv.org/abs/1712.06568) and suggested *we should def take a look at this one*.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1426145250241548299)** (2 messages): 

> `Camel AI updates, Berkeley LLM Agents Course` 


- **Camel AI gets New Outfit**: A user shared that [Camel AI](https://github.com/camel-ai/camel) received a few updates and thinks the **roleplay method** is great, and there's a need to test **Workforce** and all the tools.
- **Berkeley Launches LLM Agents Course**: A member shared two LLM Agents courses from Berkeley, which might be interesting to check out: [playlist 1](https://www.youtube.com/playlist?list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc) and [playlist 2](https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1425982703525826660)** (2 messages): 

> `YouTube Links, oN0nViY4gn4, 1gO2bC5xLlo` 


- **YouTube Links Flood Channel**: Two members posted YouTube video links: [oN0nViY4gn4](https://youtu.be/oN0nViY4gn4) and [1gO2bC5xLlo](https://youtu.be/1gO2bC5xLlo).
- **No discussion**: No discussion of the links was found in the channel.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1425924819760976122)** (18 messages🔥): 

> `GPT-5-Codex with Aider, Deepseek v3.1 with roo code, GLM 4.6 in Aider, Default Prompt Function in Aider, Aider Chat Modes` 


- ****GPT-5-Codex** and Aider Integration Explored**: A member inquired about experiences using **GPT-5-Codex** with Aider, while others shared their experiments with models like **Deepseek v3.1**, **gpt-oss-120b**, and plans to test **GLM 4.6** and vision models like **Qwen3-VL**.
   - The same member also explored routing prompts to appropriate models/tools using an **n8n workflow** as an API endpoint and using a smaller local model as a gateway to larger, off-prem models for tasks like stripping environment secrets.
- ****GLM 4.6** Subscription works fine in Aider**: Members confirmed that using a **GLM 4.6 subscription plan** in Aider works, referring to the [OpenAI-compatible guide](https://aider.chat/docs/llms/openai-compat.html) for setup.
   - One member specifically mentioned using the *pass* endpoint for this integration.
- **Default Prompt Config in Aider: Ask Mode?**: A member asked how to configure Aider to use `/ask` as the default prompt function.
   - Others directed them to Aider's [usage modes documentation](https://aider.chat/docs/usage/modes.html) and suggested setting `edit-format: chat` or `edit-format: architect` in the config file, or to set it to true to proceed with the edit.
- **Set **Architect Mode** by Default?**: Members discussed setting the default chat mode to *architect* in Aider's configuration.
   - One member stated they set the value to `true` to use the `architect` mode to analyze all prompts.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1426202856226164776)** (9 messages🔥): 

> `Haiku for Git Commits, Custom Profiles in Openrouter, Model Specification Syntax, Weak vs. Main Model, Persona Definitions` 


- **Aider opts for nimble Haiku for Git Commits**: Aider is switching to **Haiku** for generating git commit messages, citing efficiency due to its speed and lower cost, specifically using `openrouter/anthropic/claude-sonnet-4`.
   - A member suggested using **gemini flash lite** for commit messages as a cost-effective alternative and added that you can set a *weak model* for commit messages to optimize for speed and cost.
- **Custom Profiles Boost Openrouter's Model Management**: Users can manage custom profiles in **OpenRouter** to specify custom prompts, temperatures, and reasoning complexity for models.
   - These profiles can be specified in the aider config or through `/model` and `/editor-model` commands, pointing to model definitions in the `aider/resources/` directory.
- **Aider's Model Specification Syntax Unleashed**: The `/model` and `/editor-model` chat commands in Aider allow users to specify models, even those not explicitly defined in the `.aider.conf.yml` file.
   - This offers flexibility in selecting models on the fly, complementing the configuration-based model settings.
- **Understanding Aider's Weak vs. Main Model Strategy**: Aider distinguishes between *weak* and *main* models to optimize performance; the *weak model* handles tasks like commit message generation, while the *main model* addresses core coding tasks.
   - This allows for efficient resource allocation by using faster, cheaper models for less critical operations.
- **Persona Definitions as Read-Only Assets**: The user asked about pushing **personas definitions** (such as those from [vibecodingtools.tech](https://www.vibecodingtools.tech/templates)) as `/read-only` assets to the underlying model.
   - It's suggested to load these personas only when switching between tasks (e.g., from Planning to Coding) rather than for each request, and pushed to the underlying model


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1425929369171263549)** (23 messages🔥): 

> `Sora invite codes, Kimi coding abilities, Hack Club & Moonshot AI, Making videos with Kimi` 


- **Sora 2 Invite Codes are easy to get**: Some members said they could easily get **Sora 2 invite codes** and that the product hit *1m+ downloads*.
   - Another member said they would rather wait for the **public release**.
- **Kimi Excels at Coding Tasks**: **Kimi** is cool at coding, using an **agentic mode/tool usage through the IDE** to execute **python scripts** and **batch commands** to understand stuff about the system for debugging.
   - One member found this to be *straight up better than most other models*.
- **Hack Club has nothing to do with Moonshot AI**: Members discussed whether **Moonshot AI** was involved with an email from **Hack Club**.
   - It was confirmed that **Hack Club** is unrelated to **Moonshot AI**.
- **Wild Videos Created with Kimi**: Some members reported making "wild videos" with Kimi.
   - No further details were provided about the nature of the videos.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1425920754062590114)** (10 messages🔥): 

> `AI Slop in PRs, Algebraic Upat Tests, Tinygrad vs PyTorch, Reduce Group Color Change` 


- **Community Debates 'AI Slop' in Pull Requests**: A member questioned whether [PR #12530](https://github.com/tinygrad/tinygrad/pull/12530) contains *AI slop*, suggesting that dismissing it as incomprehensible is an evasion of responsibility.
   - They vouched for the code's quality, specifically mentioning [PR #12539](https://github.com/tinygrad/tinygrad/pull/12539), and compared it to **geohot's** algebraic Upat tests in [issue #12449](https://github.com/tinygrad/tinygrad/pull/12449).
- **Tinygrad's Stance on AI-Generated Code Divides Community**: A member mentioned that *AI PRs will be closed without comment, and if you are pushy about them, you will be banned.*
   - It was emphasized that *if you do not understand every line of the PR you are submitting, don't submit it*.
- **Tinygrad or PyTorch for Linear Algebra Speed?**: A member inquired about using **tinygrad** for **torch linalg** features like **cross product** and **norm**, asking if **tinygrad** converts these to **UOps** and generates **C code**.
   - They asked if **tinygrad** offers advantages over **PyTorch** when only focusing on **fast cross products** and **matrix multiplication**.
- **Reduce Group Color Shifts to Bright Red**: The group for **reduce** is now marked in *bright RED* instead of green, to highlight that a local is being used for reduction.
   - The discussion specifies that green will be reserved for future functionality, see [PR #12604](https://github.com/tinygrad/tinygrad/pull/12604/files).


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1425929038777684019)** (11 messages🔥): 

> `tinygrad vector operations, loop splitting, cuda_ioctl_sniffer in Rust, Winograd test failures` 


- **Vector Operations Questioned**: A member asked if **tinygrad** supports fast vector operations like **cross product, norm, and trig functions**.
   - No response was recorded.
- **Loop Splitting Resources Sought After**: A member requested framework-agnostic learning resources on **loop splitting** to *fix `cat` at a high level*.
   - They are attempting a bounty involving loop splitting, noting their current implementation fails **3 unit tests**.
- **Winograd Test Suffers Assertion Failure**: While attempting to implement loop splitting, the member encountered an **assertion failure** in `test_winograd.py`.
   - The error message indicates a value **6.49** failing a *not less than* comparison against **2.6**.
- **`cuda_ioctl_sniffer` Recreated in Rust**: A member is converting George Hotz's `cuda_ioctl_sniffer` into **Rust** with an interactive terminal to test individual **CUDA kernels**.
   - They posted [a demo image](https://cdn.discordapp.com/attachments/1070745817025106080/1425975923458445343/image.png?ex=68eadc91&is=68e98b11&hm=8472f883712ba4af77167cb978546cf31d701a98b6a67bcca03ea1b59fdab985) of the `saxpy` kernel output and aims to support more GPUs, using IOCTL to launch **CUDA kernels**.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1426128942095405076)** (9 messages🔥): 

> `Prompt Injection Tasks, DSPy Community Repo, AgentFlow` 


- **Spotlighting Reduces Prompt Injection Risk**: A member demoed spotlighting as a tool to reduce the risk of **prompt injection tasks**, referencing [Microsoft's research paper](https://github.com/estsauver/dspy-spotlight).
   - They are still working on a benchmark and test suite against **XPIA attacks**.
- **DSPy Community Repo Launches to Highlight Projects**: A member created the [DSPy Community repo](https://github.com/dspy-community) to highlight projects so they don't disappear.
   - It is currently a **README** on the main profile listing libraries and projects, and **PRs are welcome**.
- **AgentFlow Dropped**: A member shared a link to [AgentFlow](https://github.com/lupantech/AgentFlow).
   - No other context was given.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

batmanosama: https://arxiv.org/abs/2510.05592
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1425960034126528562)** (9 messages🔥): 

> `dspy.Tool from MCP Tool with authentication, shadcn inspiration for DSPy, TTD-DR module download, Platform/marketplace for DSPy modules, AgentFlow` 


- ****MCP Tool Authentication Quandaries****: A member inquired about creating a `dspy.Tool` from an **MCP Tool** with [authentication](https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#authentication) and whether this is currently handled.
   - The member wondered what would happen when using `dspy.Tool.from_mcp_tool` with tools requiring authentication and if the authentication process is properly managed.
- ****shadcn Sparks DSPy Inspiration****: A member suggested DSPy could learn from **shadcn** by creating *a nice explorer website*, *a CLI* that puts modules in a consistent location, and *a way to publish optimized models* back to the repo.
   - This would involve users being able to easily adapt the modules to their needs instead of relying on `pip install`, promoting easier customization.
- ****TTD-DR Module Download Dream****: Members discussed the possibility of directly downloading a **TTD-DR (Test-Time Diffusion Deep Researcher)** module for local tweaking.
   - The suggestion involved extending the package to allow running `dspy install deep-research-ttd`, which would handle the setup and module placement in a DSPy-specific directory.
- ****DSPy Module Marketplace Momentum****: Members advocated for creating a **platform/marketplace** for DSPy modules to facilitate sharing and reuse of optimized programs.
   - This would involve a repository of already-optimized programs (e.g., a classification task for customer reviews optimized for **Qwen**, **4.1-mini**, **4.1-nano**), allowing users to quickly deploy solutions for common tasks.
- ****AgentFlow: A New Player Enters the Chat****: Members noticed the release of [AgentFlow](https://github.com/lupantech/AgentFlow).
   - No further details were discussed.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1425930889610924106)** (18 messages🔥): 

> `Godhand AI-assisted previz creation, Manus Support, Prompt Engineering, Cloudflare integration, Claude API` 


- ****Godhand** AI-assisted previz workflow gains traction**: A [LinkedIn post](https://www.linkedin.com/posts/godhand_ai-assisted-previz-creation-workflow-quick-activity-7382046352287088640-Ev0V) showcases **Godhand's** AI-assisted previz creation workflow.
   - The workflow promises a quick and efficient approach to previz, potentially reducing production time.
- **Users demand Support Staff Intervention!**: Multiple users demanded immediate support staff attention in the channel.
   - One user, frustrated with the platform, exclaimed, *"HELLO!!! WHERE IS THE SUPPORT STAFF?!"*
- **Manus shines in initial project planning**: One member finds **Manus** efficient, especially for initial planning and basic project structure, costing only **1500 credits** to build a RAG pet assistant vector db.
   - He suggests using **Manus** for planning and then **Claude Code** for coding the project, highlighting efficiency with prompts and n8n workflows.
- **Efficient Prompt Engineering is Key**: Multiple users emphasized the importance of explicit and detailed prompt engineering when using AI tools like **Manus**.
   - One user argued that *throwing files at any AI and telling it to figure out the details of your prompt is simply very bad practice*.
- **Integrating Claude via API**: A user announced that **Claude** can now be easily integrated into **Manus** via API calls.
   - This eliminates the need for copy-pasting, streamlining workflows.


  
