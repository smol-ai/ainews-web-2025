---
id: MjAyNS0w
title: NVIDIA to invest $100B in OpenAI for 10GW of Vera Rubin rollout
date: '2025-09-22T05:44:39.731046Z'
description: >-
  **NVIDIA** and **OpenAI** announced a landmark strategic partnership to deploy
  at least **10 gigawatts** of AI datacenters using NVIDIA's systems, with
  NVIDIA investing up to **$100 billion** progressively as each gigawatt is
  deployed, starting in the second half of 2026 on the Vera Rubin platform. This
  deal significantly impacts the AI infrastructure funding landscape,
  potentially supporting OpenAI's $300 billion commitment to Oracle. The
  announcement caused major stock market reactions, with NVIDIA's market cap
  surging by $170 billion. Additionally, advancements in deterministic inference
  for reinforcement learning and FP8 precision gains in GPU performance were
  highlighted by AI practitioners.
companies:
  - nvidia
  - openai
  - oracle
  - intel
  - enfabrica
  - wayne
models:
  - qwen3-omni
  - deepseek-v3.1
topics:
  - gpu-infrastructure
  - deterministic-inference
  - reinforcement-learning
  - fp8-precision
  - gpu-performance
  - ai-infrastructure
  - strategic-partnerships
  - investment
  - datacenters
  - cuda-graphs
  - pipeline-parallelism
  - data-parallelism
people:
  - artificialanlys
  - gdb
---


**What is going on?**

> AI News for 9/22/2025-9/23/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (193 channels, and 3072 messages) for you. Estimated reading time saved (at 200wpm): 236 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

We would normally feature the remarkable velocity of Qwen (headlined by today's [Qwen3-Omni model](https://github.com/QwenLM/Qwen3-Omni/blob/main/assets/Qwen3_Omni.pdf)) or the [new DeepSeek V3.1 update](https://www.reddit.com/r/LocalLLaMA/comments/1nnmhai/deepseek_released_deepseekv31terminus/), but really today belongs again to NVIDIA, which over the last week has deployed billions into [Intel ($5b)](https://news.smol.ai/issues/25-09-18-nvidia-intc) and [Enfabrica's execuhire ($900m)](https://www.cnbc.com/2025/09/18/nvidia-spent-over-900-million-on-enfabrica-ceo-ai-startup-technology.html) and [Wayne ($500m)](https://x.com/sawyermerritt/status/1969049038302409016?s=46).

The relevant details of the [press release](https://openai.com/index/openai-nvidia-systems-partnership/) are all we know:

> **News**
> 
> - Strategic partnership enables OpenAI to build and deploy at least 10 gigawatts of AI datacenters with NVIDIA systems representing millions of GPUs for OpenAI’s next-generation AI infrastructure.
> - To support the partnership, NVIDIA intends to invest up to $100 billion in OpenAI progressively as each gigawatt is deployed.
> - The first gigawatt of NVIDIA systems will be deployed in the second half of 2026 on NVIDIA’s Vera Rubin platform.
> 
> **San Francisco and Santa Clara—September 22, 2025—**NVIDIA and OpenAI today announced a letter of intent for a landmark strategic partnership to deploy at least 10 gigawatts of NVIDIA systems for OpenAI’s next-generation AI infrastructure to train and run its next generation of models on the path to deploying superintelligence. To support this deployment including datacenter and power capacity, NVIDIA intends to invest up to $100 billion in OpenAI as the new NVIDIA systems are deployed. The first phase is targeted to come online in the second half of 2026 using NVIDIA’s Vera Rubin platform.

We don't know this for a fact but this $100B deal is likely a big part of how [OpenAI is funding their $300B commit to Oracle](https://news.smol.ai/issues/25-09-10-oci) from 2 weeks ago (whose stock is back up at all time highs, seeming to support this theory).

**Side note:** it hasn't escaped [observers](https://x.com/SullyOmarr/status/1970176527137718654) that somehow all the stocks involved - ORCL, OpenAI, and NVIDIA - are all jumping disproportionately on this money going from one to the other. NVIDIA's stock gained $170B today after announcing this $100B investment to [secure their revenue](https://x.com/rihardjarc/status/1970170005858726278?s=46), OpenAI's stock is now presumably valued more than the most recent $500B after this deal as well, and ORCL is still $250B higher than it was before the announcement. Are there -ANY- losers here?

[From The Information](https://x.com/amir/status/1969043037805228388), we also have some insight on the breathtaking scale of OpenAI's intended infra spend, which includes about $150B more in existing + unaccounted spend.

![](https://resend-attachments.s3.amazonaws.com/RUUS0T5m7XTkV8T)

![](https://resend-attachments.s3.amazonaws.com/K0IURbEVvCVhvVz)

---

# AI Twitter Recap

**Compute, Inference, and Systems: OpenAI–NVIDIA, FP8, and cross‑vendor GPU portability**

- **OpenAI × NVIDIA: 10 GW and “millions of GPUs.”** OpenAI announced a strategic partnership with NVIDIA to deploy at least **10 gigawatts** of GPU datacenters, targeting first capacity in 2H 2026 on “Vera Rubin,” with NVIDIA intending to invest up to **$100B** as systems are deployed. OpenAI framed NVIDIA as a preferred strategic compute/networking partner; NVIDIA’s market cap jumped on the news. Details via [@OpenAINewsroom](https://twitter.com/OpenAINewsroom/status/1970157101633990895) and [@gdb](https://twitter.com/gdb/status/1970173243350008201). Commentary on how such scaling continues to drive down the “cost of intelligence” from [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1970251031373390292).
- **Deterministic inference for RL & reproducibility**: SGLang added end‑to‑end deterministic attention/sampling that remains compatible with chunked prefill, CUDA graphs, radix cache, and non‑greedy sampling—useful for reproducible rollouts and on‑policy RL with minimal overhead. See [@lmsysorg](https://twitter.com/lmsysorg/status/1970240927429206161).
- **FP8, comms, and real‑world speedups**: Practitioners reported tangible FP8 gains under parallelism with comms constraints (e.g., PCIe), with perf crossover vs BF16 under pipeline/data parallel regimes. See local results and methodology from [@TheZachMueller](https://twitter.com/TheZachMueller/status/1970262732319412599) and follow‑ups. Related: **Together AI** is offering early access to GB300 NVL72 racks ([@togethercompute](https://twitter.com/togethercompute/status/1970129083985231932)).
- **Write once, run on many GPUs**: Modular previewed cross‑vendor portability where most code written for NVIDIA/AMD “mostly just works” on Apple Silicon GPUs—aimed at lowering hardware access barriers ([@clattner_llvm](https://twitter.com/clattner_llvm/status/1979811722614833272)). See also their updated cross‑vendor stack notes ([@clattner_llvm](https://twitter.com/clattner_llvm/status/1970203319097495891)).

**Major model drops: Qwen3 Omni family, Grok‑4 Fast, DeepSeek V3.1 Terminus, Apple Manzano, Meituan LongCat**

- **Qwen’s multi‑front release wave**:
    - **Qwen3‑Omni**: an end‑to‑end omni‑modal model (text, image, audio, video) with **211 ms latency**, SOTA on **22/36** audio/AV benchmarks, tool‑calling, and a low‑hallucination Captioner. Alibaba open‑sourced the 30B A3B variants: Instruct, Thinking, and Captioner. Demos and code: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1970181599133344172), [release thread](https://twitter.com/scaling01/status/1970182151019659493).
    - **Qwen3‑Next‑80B‑A3B with FP8**: Apache‑2.0 weights focused on long‑context speed; mixture‑of‑experts with gated attention/DeltaNet, trained on ~15T tokens with GSPO, supports up to **262k tokens** (longer with mods). Summary via [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1970254860416131146).
    - **Qwen3‑TTS‑Flash**: SOTA WER for CN/EN/IT/FR, 17 voices × 10 languages, ~**97 ms** first packet; and
    - **Qwen‑Image‑Edit‑2509**: multi‑image compositing, stronger identity preservation, and native ControlNet (depth/edges/keypoints). Launches: [TTS](https://twitter.com/Alibaba_Qwen/status/1970163551676592430), [Image‑Edit](https://twitter.com/Alibaba_Qwen/status/1970189775467647266).
- **xAI Grok‑4 Fast**: A cost‑efficient multimodal reasoner with **2M context**, free in some “vibe coding” UIs; community reports **2–3× higher throughput** but weaker instruction following than GPT‑5‑mini on some tasks; SVG generation test mixed; still competitive on LisanBench. See [@ShuyangGao62860](https://twitter.com/ShuyangGao62860/status/1969240703080546376), [@_akhaliq](https://twitter.com/_akhaliq/status/1969431198859501622), [@scaling01](https://twitter.com/scaling01/status/1969426187312156790), and a long‑context filtering anecdote from [@dejavucoder](https://twitter.com/dejavucoder/status/1969383391029313598).
- **DeepSeek‑V3.1‑Terminus**: Incremental update addressing mixed‑language artifacts and improving Code/Search agents. Available on Hugging Face; community shows usable 4‑bit quant runs on **M3 Ultra** with MLX at double‑digit toks/sec. See [@deepseek_ai](https://twitter.com/deepseek_ai/status/1970117808035074215), demos by [@awnihannun](https://twitter.com/awnihannun/status/1970151204102750573).
- **Apple Manzano**: a unified multimodal LLM that shares a ViT with a hybrid vision tokenizer (continuous embeddings for understanding + 64K FSQ tokens for generation), scaling from **300M to 30B**, with strong text‑rich understanding (OCR/Doc/ChartQA) and competitive generation/editing via a lightweight DiT‑Air decoder. Threads: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1969974676802990478), summary with training details by [@gm8xx8](https://twitter.com/gm8xx8/status/1969974517024923936).
- **Meituan LongCat‑Flash‑Thinking**: open‑source “thinking” variant reporting SOTA across logic/math/coding/agent tasks with **64.5% fewer tokens** on AIME25 and a 3× training speedup via async RL. Launch: [@Meituan_LongCat](https://twitter.com/Meituan_LongCat/status/1969823529760874935).

**Coding agents, evals, and scaffolds: SWE‑Bench Pro, GAIA‑2/ARE, ZeroRepo, Perplexity Email Assistant**

- **SWE‑Bench Pro (Scale AI)**: a harder successor to SWE‑Bench Verified with multi‑file edits (avg **~107 LOC** across ~4 files), contamination resistance (GPL/private repos), and tougher deps. Current top scores: **GPT‑5 = 23.3%**, **Claude Opus 4.1 = 22.7%**, most others <15%. Details from [@alexandr_wang](https://twitter.com/alexandr_wang/status/1979805196462358919) and [@scaling01](https://twitter.com/scaling01/status/1969792786594509190).
- **Meta GAIA‑2 + ARE**: a practical agent benchmark and an open platform (with MCP tool integration) for building/evaluating agents in noisy, asynchronous environments. Findings: strong “reasoning” models can fail under time pressure (inverse scaling); Kimi‑K2 competitive at low budgets; multi‑agent helps coordination; diminishing returns beyond certain compute. See [@ThomasScialom](https://twitter.com/ThomasScialom/status/1970122143993037170) and commentary by [@omarsar0](https://twitter.com/omarsar0/status/1970147904087322661).
- **MCP‑AgentBench**: Metastone’s live‑tool benchmark with **33 servers & 188 tools** to evaluate real‑world agent performance ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1969864853238985001)).
- **Repository Planning Graph (RPG) + ZeroRepo (Microsoft)**: proposes a graph of capabilities/files/functions and data dependencies to plan/generate whole repos from specs, reporting **3.9×** more LOC than baselines on their setup. Threads: [@_akhaliq](https://twitter.com/_akhaliq/status/1969976136232022289) and explainer from [@TheTuringPost](https://twitter.com/TheTuringPost/status/1970068577509327197).
- **Perplexity Email Assistant**: a native email agent for Gmail/Outlook that drafts in your style, schedules meetings, and prioritizes inbox items—now live for Max subscribers ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1970165704826716618), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1970165878751973560)).
- **Coding UX trending up**: GPT‑5‑Codex shows dramatic capability jumps (e.g., a basic Minecraft clone in three.js) and reward shaping that “makes sure your code actually runs” ([@gdb](https://twitter.com/gdb/status/1979808193137348874), [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1969784664912179533)). Tri Dao reports **1.5× productivity** with Claude Code ([@scaling01](https://twitter.com/scaling01/status/1970146206203416666)); “code is king” remains a durable, high‑value application ([@simonw](https://twitter.com/simonw/status/1970147806225854531)).

**Safety, governance, and agent security**

- **Detecting/reducing “scheming”**: OpenAI and Apollo AI Evals introduced environments where current models exhibit situational awareness and can be prompted/trained into simple covert behavior; “deliberative alignment” reduces scheming rates, though anti‑scheming training can increase evaluation awareness without eliminating covert actions ([@gdb](https://twitter.com/gdb/status/1969437389027492333)). Practitioner notes: outcome‑based RL and “hackable” envs may introduce scheming; rising use of non‑human “reasoning traces” complicates audits ([@scaling01](https://twitter.com/scaling01/status/1969548755255861575)).
- **Guardrails with dynamic policy**: DynaGuard (ByteDance) evaluates if conversations comply with user‑defined rules, supports fast/detailed explanatory modes, and generalizes to unseen policies ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1970079921704997326)).
- **Agent ingestion principle**: “If the agent ingests anything, its permissions should drop to the level of the author”—a crisp policy design heuristic for tool‑enabled agents ([@simonw](https://twitter.com/simonw/status/1969247680762429784)).

**Research highlights: JEPA debate, synthetic data pretraining, memory for latent learning**

- **JEPA for LLMs (and for robots)**: A new LLM‑JEPA iteration claims latent prediction benefits ([@randall_balestr](https://twitter.com/randall_balestr/status/1969283010982744133)), but critiques argue it requires tightly paired data (e.g., Text↔SQL), adds forward passes, and lacks generality ([@scaling01](https://twitter.com/scaling01/status/1969410266304545066)). In robotics, V‑JEPA shows strong spatial understanding but impractical inference (~16s/action via MPC) and no language conditioning; contrasts with label‑heavy approaches like Pi0.5 ([@stevengongg](https://twitter.com/stevengongg/status/1969387819920736396)).
- **Synthetic Bootstrapped Pretraining (SBP)**: Trains a 3B model on 1T tokens by synthesizing inter‑document relations—outperforming repetition baselines and closing much of the gap to an oracle with **20×** more unique data ([@ZitongYang0](https://twitter.com/ZitongYang0/status/1970129028536484089), [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1969973861178626245)).
- **Latent learning gap and episodic memory**: A conceptual framework tying language model failures (e.g., reversal curse) to absent episodic memory; shows retrieval/episodic components can complement parametric learning for generalization ([@AndrewLampinen](https://twitter.com/AndrewLampinen/status/1969980297661047055)).
- Also notable: NVIDIA’s **ReaSyn** frames molecule synthesis as chain‑of‑reaction reasoning with RL finetuning ([@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1969976462091645144)); **Dynamic CFG** adapts guidance per step via latent evaluators, yielding large human pref gains on Imagen 3 ([@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1969975609842688383)); Microsoft’s **Latent Zoning Network** unifies generative modeling, representation learning, and classification via a shared Gaussian latent space ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1970218823140687885)).

**Top tweets (by engagement)**

- OpenAI × NVIDIA announce a strategic buildout of “millions of GPUs” and at least 10 GW of data centers ([@OpenAINewsroom](https://twitter.com/OpenAINewsroom/status/1970157101633990895), 3.7K+).
- Qwen3‑Omni: End‑to‑end omni model with SOTA audio/AV results and 30B open variants (Instruct/Thinking/Captioner) ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1970181599133344172), 3.9K+).
- Turso’s rapid evolution: Rust rewrite of SQLite with async‑first architecture, vector search, and browser/wasm support—framed as infra for “vibe coding” ([@rauchg](https://twitter.com/rauchg/status/1969515038512926823), 2.8K+).
- GPT‑5‑Codex demo: three.js “Minecraft” built from a single prompt ([@gdb](https://twitter.com/gdb/status/1979808193137348874), 3.1K+).
- SWE‑Bench Pro: harder agent coding benchmark with real‑world repos; GPT‑5 and Claude Opus 4.1 lead at ~23% ([@alexandr_wang](https://twitter.com/alexandr_wang/status/1979805196462358919), 1.7K+).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. DeepSeek-V3.1-Terminus Launch & Online Upgrade

- [**🚀 DeepSeek released DeepSeek-V3.1-Terminus**](https://i.redd.it/729mf2l1xpqf1.jpeg) ([Score: 361, Comments: 45](https://www.reddit.com/r/LocalLLaMA/comments/1nnmhai/deepseek_released_deepseekv31terminus/)): **DeepSeek announced an iterative update, DeepSeek‑V3.1‑Terminus, targeting prior V3.1 issues like CN/EN language mixing and spurious characters, and upgrading its Code Agent and Search Agent. The team claims more stable, reliable outputs across benchmarks vs V3.1 (no specific numbers provided); weights are open‑sourced on Hugging Face: https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus and accessible via app/web/API.** Commenters ask if “Terminus” signifies the final V3 checkpoint and request feedback on role‑play performance; others discuss the aggressive naming, but no technical objections are raised.
    - Clarification sought on whether "V3.1‑Terminus" denotes the final checkpoint of the `V3` line versus a routine sub-variant; the naming suggests a checkpoint/tag rather than a major-arch change, and commenters want release notes clarifying if it’s a new training run, a late-stage fine-tune, or an inference-time preset.
    - Critique of **DeepSeek**’s versioning semantics: the sequence from **R1** to `V3.1` and `V3.1‑T` is seen as confusing, with warnings that a hypothetical "3V" (Vision) could be mistaken for "V3". This ambiguity impedes reproducibility and apples-to-apples comparisons across checkpoints and capabilities unless model cards clearly specify training data, steps, and deltas between tags.
    - Requests for head-to-head benchmarks against popular open(-ish) baselines like **GLM‑4.5** and "kimik2" (as referenced by users), including roleplay performance as a targeted eval dimension. Commenters want standardized evals (e.g., instruction-following plus RP/character consistency tests) to quantify whether `V3.1‑T` improves practical usability versus current stacks.

### 2. Qwen3-Omni Multimodal Release & Open-Source Models

- [**3 Qwen3-Omni models have been released**](https://www.reddit.com/r/LocalLLaMA/comments/1nnt1bw/3_qwen3omni_models_have_been_released/) ([Score: 362, Comments: 77](https://www.reddit.com/r/LocalLLaMA/comments/1nnt1bw/3_qwen3omni_models_have_been_released/)): **Three end-to-end multilingual, omni-modal** `30B` **models—[Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct), [Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking), and [Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)—are released with a MoE-based Thinker–Talker design, AuT pretraining, and a multi-codebook speech codec to reduce latency. They handle text, image, audio, and video with real-time streaming responses (TTS/STT), support** `119` **text languages,** `19` **speech-input languages, and** `10` **speech-output languages, and report SOTA on** `22/36` **audio/video benchmarks (open-source SOTA** `32/36`**), with ASR/audio understanding/voice conversation comparable to Gemini 2.5 Pro per the [technical report](https://github.com/QwenLM/Qwen3-Omni/blob/main/assets/Qwen3_Omni.pdf). Instruct bundles Thinker+Talker (audio+text out), Thinking exposes chain-of-thought Thinker (text out), and Captioner is a fine-grained, low-hallucination audio captioner fine-tuned from Instruct with a [cookbook](https://github.com/QwenLM/Qwen3-Omni/blob/main/cookbooks/omni_captioner.ipynb).** Early user reports claim TTS quality is weak while STT is “godlike,” outperforming [Whisper](https://github.com/openai/whisper) with contextual constraints and very fast throughput (e.g., ~30 s audio transcribed in a few seconds), and note strong image understanding on complex graphs/trees to Markdown; another asks about GGUF availability.
    - Reports note the model’s **STT** is *“godlike”* versus **OpenAI Whisper**, with promptable context/constraints (e.g., telling it to never insert obscure words) and very high throughput—`~30s` of audio transcribed in a few seconds locally. Multimodal vision is praised for accurate structure extraction, e.g., converting complex graphs/tree diagrams into clean Markdown, implying robust layout understanding beyond simple OCR. See Whisper for baseline comparison: https://github.com/openai/whisper.
    - Conversely, native **TTS** quality is described as poor, which limits end-to-end speech-to-speech despite fast ASR. Real-time S2S is feasible in principle by chaining ASR → LLM → TTS, but latency/UX will depend on swapping in a higher‑quality TTS engine; STT latency appears near–real-time, but output voice quality remains the bottleneck.
    - Local deployment friction is highlighted: users ask for **GGUF** builds and note **llama.cpp** lacks full multimodal support (even `Qwen2.5-Omni` isn’t fully integrated), so audio/image features may require vendor runtimes or custom servers for now. This constrains on‑device use until community kernels catch up. Relevant refs: llama.cpp https://github.com/ggerganov/llama.cpp and GGUF format https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md.
- [**🚀 Qwen released Qwen3-Omni!**](https://www.reddit.com/gallery/1nntr5a) ([Score: 186, Comments: 3](https://www.reddit.com/r/LocalLLaMA/comments/1nntr5a/qwen_released_qwen3omni/)): **Alibaba’s Qwen team announced [Qwen3‑Omni](https://github.com/QwenLM/Qwen3-Omni), a natively end‑to‑end multimodal model unifying text, image, audio, and video (no external encoders/routers), claiming SOTA on** `22/36` **audio/AV benchmarks. It supports** `119` **text languages /** `19` **speech‑in /** `10` **speech‑out, offers ~**`211 ms` **streaming latency and** `30‑min` **audio‑context understanding, with system‑prompt customization, built‑in tool calling, and an open‑source low‑hallucination Captioner. Open releases include [Qwen3‑Omni‑30B‑A3B‑Instruct](https://huggingface.co/collections/Qwen/qwen3-omni-68d100a86cd0906843ceccbe), ‑Thinking, and ‑Captioner; code/weights and demos are on [GitHub](https://github.com/QwenLM/Qwen3-Omni), [HF](https://huggingface.co/collections/Qwen/qwen3-omni-68d100a86cd0906843ceccbe), [ModelScope](https://modelscope.cn/collections/Qwen3-Omni-867aef131e7d4f), [Chat](https://chat.qwen.ai/?models=qwen3-omni-flash), and a [demo space](https://huggingface.co/spaces/Qwen/Qwen3-Omni-Demo).** Comments flag the benchmark chart layout as making direct comparison to Gemini 2.5 Pro difficult, while several note the 30B‑A3B results appear competitive with GPT‑4o on their tasks—especially for vision‑reasoning—prompting enthusiasm to test “thinking‑over‑images” in an open model.
    - Skepticism about the benchmark visualization: one commenter notes the chart is *“masterfully crafted”* to push **Gemini 2.5 Pro** off the main comparison area, implying potential presentation bias and making side‑by‑side evaluation with **Qwen3‑Omni** harder. The point emphasizes the need for transparent axes, overlapping points, and raw numbers to enable reproducible, apples‑to‑apples comparisons across models.
    - Early read on performance: a user says the `30B-A3B` variant shows surprisingly strong results and appears to match **GPT‑**`4o` in their experience on multimodal reasoning, particularly *“thinking‑over‑images.”* If borne out in independent tests, that would position an open model close to frontier multimodal reasoning capability, attractive for local/self‑hosted use and practical evaluation beyond curated leaderboards.

### 3. Qwen-Image-Edit-2509 Release: Multi-Image Editing & ControlNet

- [**Qwen-Image-Edit-2509 has been released**](https://www.reddit.com/r/LocalLLaMA/comments/1nnt539/qwenimageedit2509_has_been_released/) ([Score: 222, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1nnt539/qwenimageedit2509_has_been_released/)): **Qwen released [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509), a September update adding multi-image editing trained via image concatenation (supporting person+person/product/scene) with best results at** `1–3` **inputs, plus markedly improved single-image identity consistency for faces, products, and on-image text (fonts/colors/materials) accessible via [Qwen Chat](https://qwen.ai/). It also adds native [ControlNet](https://arxiv.org/abs/2302.05543)style conditioning (depth, edge, keypoint maps, etc.) on top of the existing Qwen-Image-Edit architecture.** Comments highlight surprise at the monthly cadence and note prior issues with facial identity drift over multiple iterations, which this release claims to address; some compare it to Flux Kontext, saying earlier versions sometimes had worse facial resemblance, so the fast update is welcomed.
    - Identity preservation across iterative edits: Users report prior Qwen-Image-Edit builds struggled to keep faces consistent, especially over multiple edit passes or with multiple subjects. v2509 is highlighted as targeting this issue, suggesting improved face/identity conditioning and reduced drift across iterations.
    - Comparison vs **Flux Kontext**: One user found the previous release was close but sometimes worse at facial resemblance than **Flux Kontext**. The v2509 update is viewed as closing that gap by acknowledging and addressing facial similarity issues.
    - Inpainting/object removal performance: A commenter says Qwen-Image-Edit-2509 is “comparable to nano banana” on object removal tasks, implying competitive fill quality for removals. No quantitative benchmarks were provided, but the qualitative parity is noted.
- [**🔥 Qwen-Image-Edit-2509 IS LIVE — and it’s a GAME CHANGER. 🔥**](https://i.redd.it/taitk409drqf1.jpeg) ([Score: 208, Comments: 18](https://www.reddit.com/r/LocalLLaMA/comments/1nnu9b2/qwenimageedit2509_is_live_and_its_a_game_changer/)): **Qwen-Image-Edit-2509 is announced as a major upgrade of Qwen’s image editing stack with multi-image compositing (e.g., person+product/scene) and strong single-image identity/brand consistency. It claims fine-grained text editing (content, font, color, material) and integrates ControlNet controls (depth, edges, keypoints) for precise conditioning; code and weights are available on GitHub and Hugging Face ([GitHub](https://github.com/QwenLM/Qwen-Image), [HF model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509), [blog](https://qwen.ai/blog?id=7a90090115ee193ce6a7f619522771dd9696dd93&from=research.latest-advancements-list)).** Top comments critique the marketing hyperbole (e.g., “game changer,” “rebuilt”) and don’t provide benchmarks or technical counterpoints; skepticism centers on evidence for the claimed improvements.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI–NVIDIA 10 GW Supercomputer Partnership Announcements

- [**OpenAI and NVIDIA announce strategic partnership to deploy 10 gigawatts of NVIDIA systems**](https://openai.com/index/openai-nvidia-systems-partnership) ([Score: 244, Comments: 80](https://www.reddit.com/r/singularity/comments/1nnqs1m/openai_and_nvidia_announce_strategic_partnership/)): **OpenAI and NVIDIA signed a letter of intent to deploy at least** `10 GW` **of NVIDIA systems (described as “millions of GPUs”) for OpenAI’s next‑gen training/inference stack, with the first** `1 GW` **slated for** `H2 2026` **on NVIDIA’s Vera Rubin platform; NVIDIA will be a preferred compute/networking partner, and both sides will co‑optimize OpenAI’s model/infrastructure software with NVIDIA’s hardware/software [source]. NVIDIA also intends to invest up to $100B in OpenAI, disbursed progressively as each gigawatt is deployed, alongside a parallel build‑out of datacenter and power capacity and continued collaborations (e.g., Microsoft, Oracle, SoftBank, “Stargate”) toward large‑scale model development (https://openai.com/index/openai-nvidia-systems-partnership).** Top comments highlight the sheer scale of the proposed funding and note potential circular capital flows (OpenAI buys NVIDIA compute while NVIDIA invests back into OpenAI); others argue over “bubble vs. singularity” framing rather than technical merits.
    - A cited claim states: *“NVIDIA intends to invest up to $100 billion in OpenAI progressively as each gigawatt is deployed,”* tied to a `10 GW` rollout of NVIDIA systems. This reads as tranche-based vendor financing keyed to power/compute milestones, aligning capex with datacenter build-outs and de-risking supply timing while scaling GPU deployment as power and facilities come online.
    - Another comment highlights a circular capital flow: OpenAI purchases NVIDIA systems while NVIDIA invests back into OpenAI. Technically, this resembles a strategic supplier-financing/compute-prepayment structure that could secure priority allocation for next-gen NVIDIA platforms (e.g., H200/B200/GB200), lock in roadmap/pricing, and accelerate training cadence, at the cost of deeper vendor lock-in and supply-chain concentration.
- [**🚨 BREAKING: Nvidia to Invest $100 Billion in OpenAI**](https://i.redd.it/fyaydlcu2rqf1.png) ([Score: 615, Comments: 99](https://www.reddit.com/r/OpenAI/comments/1nnsniq/breaking_nvidia_to_invest_100_billion_in_openai/)): **Post claims Nvidia will invest up to** `$100B` **in a strategic partnership with OpenAI to build/deploy** `10 GW` **of AI supercomputer capacity on Nvidia hardware, translating to “millions of GPUs” coming online in 2H**`2026`**, to support OpenAI’s AGI ambitions; it adds Nvidia stock rose** `+4.69%` **on the news. Structure suggests funds are tied to progressive 10 GW rollout, effectively pre-financing/locking OpenAI to Nvidia’s stack and positioning Nvidia at the center of next‑gen AI compute.** Comments argue it’s effectively Nvidia investing in itself since OpenAI buys Nvidia hardware; OpenAI is trading equity for guaranteed compute; and Nvidia benefits by amplifying demand/prices for its chips, then recycling profits to subsidize OpenAI capacity.
    - Power-to-GPU math: taking the cited `10 GW ÷ 0.7 kW ≈ 14.3M GPUs` (assuming ~`700 W/GPU`, e.g., H100-class SXM modules), but accounting for data center PUE `~1.2–1.4` and non-GPU overheads (CPUs, NICs, switches, storage, cooling) drops usable GPU count to roughly `~8–11M`. Networking at this scale (400/800G per node over **InfiniBand NDR** or **Ethernet**) implies tens of millions of optics/ports and multi-megawatt fabric power; the interconnect and optics supply chain become bottlenecks alongside GPUs [NVIDIA Quantum-2 400G IB](https://www.nvidia.com/en-us/networking/products/infiniband/quantum-2/), [NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/). Blackwell-era modules are expected to push module power higher, further reducing the GPU count per GW and increasing cooling/networking overheads ([NVIDIA GTC Blackwell](https://www.nvidia.com/en-us/gtc/keynote/)).
    - Compute-for-equity flywheel: commenters frame this as OpenAI swapping equity for reserved NVIDIA capacity; in turn, OpenAI’s workloads popularize NVIDIA chips, letting NVIDIA raise ASPs and recycle profits into subsidized capacity for OpenAI—effectively “NVIDIA investing in NVIDIA.” Practically, this likely means multi-year take-or-pay reservations and prepayments tied to constrained inputs like **HBM3E** and **CoWoS** packaging capacity, with priority allocation rather than pure cash injection ([HBM3E overview](https://www.skhynix.com/eng/product/graphics/view.do?prdtNo=H10360), [TSMC CoWoS](https://www.tsmc.com/english/dedicatedFoundry/technology/advanced_packaging)). Deepened CUDA lock-in increases switching costs versus **AMD MI300X/MI325X + ROCm**, pressuring competitors to beat NVIDIA on $/TFLOP and memory BW to win inference/training TCO ([AMD MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html), [ROCm](https://rocmdocs.amd.com/)).
    - Scale and infrastructure constraints: `~10 GW` is utility-scale power (dozens of campuses), requiring new high-voltage substations, long-lead transformers (often `18–36` months), and substantial water/heat-rejection capacity; grid and cooling timelines may dominate deployment speed. Even with supply, orchestrating millions of GPUs demands pod-level topologies (e.g., 8–16 GPU HGX nodes, multi-tier IB/Ethernet fabrics) and careful job placement to maintain high training efficiency; otherwise interconnect bottlenecks erase scale gains. HBM supply (stacks per GPU) and optics availability are likely pacing items as much as the GPUs themselves, which aligns with “tied to capacity” language in such deals ([Uptime PUE context](https://uptimeinstitute.com/knowledge/pue), [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/)).

### 2. Qwen-Image-Edit-2509 Release and Gemini/ChatGPT Multimodal Demos

- [**Qwen-Image-Edit-2509 has been released**](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) ([Score: 348, Comments: 92](https://www.reddit.com/r/StableDiffusion/comments/1nnt6o5/qwenimageedit2509_has_been_released/)): **Qwen released [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509), the September iteration of its image-editing diffusion model, adding multi-image editing via training on concatenated images (optimal with** `1–3` **inputs; supports person+person, person+product, person+scene). It improves single-image edit consistency (better identity preservation for people with pose/style changes; stronger product identity/poster editing; richer text edits including fonts/colors/materials) and introduces native [ControlNet](https://arxiv.org/abs/2302.05543) conditioning (depth, edge, keypoint maps). A Diffusers pipeline** `QwenImageEditPlusPipeline` **is provided; examples recommend** `torch_dtype=bfloat16` **on CUDA and expose** `true_cfg_scale`**,** `guidance_scale`**,** `num_inference_steps`**, and** `seed`**.** Commenters ask if the “monthly iteration” implies regular monthly releases and whether LoRAs trained on prior versions will remain compatible across updates; they also note the likely need to redo quantization (e.g., GGUF/SVDQuant) per release, with one user immediately aiming to convert to GGUF.
    - A commenter plans immediate conversion to **GGUF**, indicating demand for a quantized, llama.cpp/ggml-friendly format for local inference on low‑VRAM or CPU‑bound setups. GGUF support typically enables deployment via llama.cpp-style backends and offline tooling; GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md.
    - There’s scrutiny of a potential `monthly` release cadence and whether LoRAs trained on previous checkpoints will remain compatible. Since LoRA deltas are tied to the base model’s weight shapes/tokenization, even small checkpoint/architecture changes can break compatibility and require re-training or re-derivation; LoRA background: https://arxiv.org/abs/2106.09685.
    - Resource constraints are highlighted: one user notes they’d need a new SVDQuant for each update and *“there’s no way I’m using even the GGUF version on my poor GPU.”* This implies monthly quantization pipeline churn (e.g., SVDQuant, GGUF) and reliance on aggressive quantization to fit VRAM limits for image-edit inference.
- [**I met my younger self using Gemini AI**](https://i.redd.it/bfnwi5lt6rqf1.jpeg) ([Score: 231, Comments: 24](https://www.reddit.com/r/GeminiAI/comments/1nnt94s/i_met_my_younger_self_using_gemini_ai/)): **Post showcases a photorealistic AI edit where the OP “meets” their younger self, reportedly created with Google’s Gemini AI. While no exact workflow is given, the result implies an image-to-image or multi-reference image editing pipeline (likely supplying a current portrait and a childhood photo) guided by a text prompt to compose the scene; OP notes the realism, saying it worked “this good.” No benchmarks or model variant/parameters are provided.** Comments are enthusiastic and request reproducible details—specifically the prompt, whether two identity reference images were used, and if a separate scene reference guided composition—highlighting interest in practical workflow replication.
    - A commenter asks for the exact prompt/workflow, explicitly probing whether the OP used two reference images (current self + childhood photo) and an additional reference for the final scene. This highlights interest in multi-image conditioning and composition control in Gemini’s image pipeline (e.g., identity preservation across inputs and scene guidance). No specifics are provided in-thread, so reproducibility details (input modalities, steps, or constraints) remain unknown.
    - The OP reports unexpectedly high fidelity ("I didn’t think it’d work this good") along with a generated composite: https://preview.redd.it/x7dvf09mzrqf1.jpeg?width=768&format=pjpg&auto=webp&s=88e91700732a51793ba05373ad87f6b7652cf01e, suggesting effective identity retention across age domains. However, the thread lacks technical parameters (model/version, input resolution, steps/seeds, or prompt details), so the result can’t be benchmarked or replicated from the given information.
- [**I didn’t know ChatGPT could do this. Took 3 prompts.**](https://v.redd.it/6mhokhu3wmqf1) ([Score: 328, Comments: 95](https://www.reddit.com/r/ChatGPT/comments/1nnbpsa/i_didnt_know_chatgpt_could_do_this_took_3_prompts/)): **OP reports that ChatGPT generated a runnable “3D Google Earth–style” web app in ~3 prompts and rendered it directly in the ChatGPT Preview sandbox—no local compile/build or hosting required. The globe implementation is suspected (by commenters) to use WebGL via [three.js](https://threejs.org/) and possibly [three-globe](https://github.com/vasturiano/three-globe); the linked video ([v.redd.it/6mhokhu3wmqf1](https://v.redd.it/6mhokhu3wmqf1)) returns** `403 Forbidden` **without Reddit auth, indicating access is gated by application/network-edge controls. The “complicated” part of the task was not achieved, but the interactive globe scaffold worked end-to-end within the ChatGPT environment.** Commenters argue LLMs are strong at scaffolding “single‑serving” web apps due to the web platform’s breadth and code-heavy training data, while another highlights reliability limits (e.g., a recent vision misclassification of an apple as a tomatillo). There’s debate/curiosity about the exact stack (three.js vs. three‑globe), but no confirmed details from OP.
    - Multiple commenters infer the demo is built with **three.js** ([threejs.org](http://threejs.org/)) and a globe helper like [three-globe](https://github.com/vasturiano/three-globe), leveraging `WebGL` for rendering. In such setups, the heavy lifting (sphere geometry, atmospheric shaders, geojson-to-3D conversion, arc/point animations) is handled by the library, and the LLM primarily wires configuration and data. This explains why it’s achievable in “3 prompts”: the code surface is mainly integrating well-documented APIs rather than writing low-level graphics.
    - LLMs are well-suited to scaffold “single-serving” web apps by composing existing packages and browser APIs. With a clear spec, they can generate a minimal stack (e.g., `Vite` + vanilla JS/TS or React) and integrate libraries (e.g., **three.js**, `d3-geo`, `TopoJSON`), relying on the web platform’s capabilities (Canvas/WebGL/WebAudio). The abundance of publicly available code in training corpora increases reliability for boilerplate and idiomatic patterns, though correctness still hinges on precise prompts and iterative testing.
    - A report of misidentifying an apple as a tomatillo highlights current limits of general-purpose VLMs on fine-grained classification. Without domain-specific priors or few-shot exemplars, lookalike classes under variable lighting/backgrounds often confuse models; specialized models (e.g., [CLIP](https://openai.com/research/clip) variants or fine-tuned Food-101 classifiers) and prompt constraints can mitigate errors. It underscores that while LLMs excel at code synthesis, vision reliability may lag without task-specific calibration.
- [**I had that moment with Kimi 2!**](https://i.redd.it/6ck3uxvzbrqf1.jpeg) ([Score: 1390, Comments: 72](https://www.reddit.com/r/singularity/comments/1nnu1qc/i_had_that_moment_with_kimi_2/)): **Screenshot shows Kimi 2 responding "Good catch" after being corrected—illustrating a hallucination likely triggered when it couldn’t access referenced documents. Commenters report Kimi 2 will fabricate content rather than return a retrieval/no-data error when document access fails, pointing to gaps in RAG/grounding and guardrails for provenance-aware answers.** Users confirm this behavior occurs on document-access failures and liken the bot’s reply to a student being caught unprepared; one notes it routinely "makes something up" in these cases.
    - Users note that when the assistant cannot access referenced documents (e.g., retrieval/permissions failures), it tends to fabricate plausible details rather than abstain. This is a classic RAG failure mode; engineering mitigations include explicit retrieval success checks, surfacing “no evidence found” states, and enforcing `cite-or-abstain` responses to avoid unsupported generations.
    - The “GPT-18” anecdote illustrates correction-induced confabulation: the model freely swaps core facts (location, power plant type) while preserving the narrative outcome (evacuation). This highlights lack of grounding and constraint satisfaction; mitigations include schema-validated tool use, entity normalization (geo/organization disambiguation), and external verification before committing to factual assertions or actions.
    - Hallucinations are reported across vendors (e.g., **ChatGPT** and **Claude**), suggesting model-agnostic limitations in factuality and tool reliability. Production setups should add deterministic guards—retrieval timeouts, confidence gating, and post-hoc verifiers—to reduce error rates, rather than relying on model prompts alone.

### 3. Robot Uprising Memes and Unitree G1 Agility Clips

- [**Unitree G1 fast recovery**](https://v.redd.it/8l0l09o6fpqf1) ([Score: 1515, Comments: 358](https://www.reddit.com/r/singularity/comments/1nnk9hk/unitree_g1_fast_recovery/)): **Short post appears to showcase the Unitree G1 humanoid executing a rapid ground‑to‑stand “fast recovery” maneuver, suggesting a whole‑body controller coordinating multi‑contact transitions with sufficient actuator peak torque/power for explosive hip/knee extension. No quantitative benchmarks (e.g., recovery time, joint power/torque, controller type) are provided; the video link [v.redd.it/8l0l09o6fpqf1](https://v.redd.it/8l0l09o6fpqf1) returns HTTP** `403` **(access‑controlled), but a still frame is visible [here](https://preview.redd.it/ofsw0lu8ipqf1.jpeg?width=940&format=pjpg&auto=webp&s=ae43ea09d75f2d803c46b5fcac9af2a85d8358ae).** Top comments emphasize the motion’s realism (“impressive and scary”) and question authenticity (“looks so good it looks fake”); no technical critique or controller/actuator discussion is present.
- [**Primary target locked! This guys the first one to go**](https://v.redd.it/xce8hshw2nqf1) ([Score: 348, Comments: 48](https://www.reddit.com/r/singularity/comments/1nncfnq/primary_target_locked_this_guys_the_first_one_to/)): **A short [v.redd.it](http://v.redd.it/) [clip](https://v.redd.it/xce8hshw2nqf1) appears to depict a robotic system announcing a target “lock” on a human and then deploying a rope/tether after switching to “OFFENSIVE MODE,” implying basic vision-based target acquisition and a powered launcher/gimbal; no telemetry, specs, or control-loop details are provided to evaluate latency, actuation speed, or safety. Several commenters challenge the demo’s validity, asserting the footage is likely sped up and asking for real‑time playback to assess tracking stability, servo response, and the risk profile of a neck-level tether.** Debate centers on law-enforcement applications versus safety/ethics, with some advocating eventual police use while others highlight strangulation hazards and reliability concerns; a quip about a person getting tangled in ~`2s` underscores skepticism about practical robustness.
    - A commenter alleges the demo is sped up and asks for real-time playback. Without true 1× footage, it’s impossible to judge controller bandwidth, state-estimation latency, actuator torque limits, and gait stability—time-lapse can hide slow step frequency and long recovery times after disturbances. Best practice would be an on-screen timecode/frame-time overlay and reporting of step rate (Hz), CoM velocity, and reaction latency to external impulses.
    - Another critic notes repeated “push tests” and simple preprogrammed punches show disturbance rejection but little capability progression. They implicitly call for harder, measurable benchmarks: uneven-terrain traversal with quantified slip, contact-rich manipulation with force/impedance control, autonomous perception/planning, payload handling, and normalized metrics like cost of transport, fall rate, and mean time-to-recovery under known impulse. Public logs or standardized benchmark suites would enable fair comparisons across humanoid platforms.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. DeepSeek v3.1 Terminus and Qwen3 Releases**

- **DeepSeek v3.1 Terminus Lands with Agentic Tweaks**: DeepSeek released **v3.1 Terminus** with open weights on [DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus), citing bug fixes, improved language consistency, and stronger **code/search agent** behavior.
    - Users noted *"somewhat degraded"* reasoning without tool use and *"slight improvements"* in agentic tool use, while others immediately asked *"when is DeepSeek-R2?"* and pointed to the broader [deepseek-ai models](https://huggingface.co/deepseek-ai).
- **Qwen3 Omni-30B Goes Multimodal**: Alibaba’s **Qwen3 Omni-30B-A3B-Instruct** (36B params) landed with multimodal encoders/decoders and multilingual audio I/O at [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct).
    - Community claims it *"beats Whisper v2/3 and Voxtral 3B"* on scam-call audio and supports **17 input** and **10 output** languages, alongside chatter about local **LoRA** training on a single **RTX 4090** and a coming wave of **realtime perceptual AI**.
- **Qwen3-TTS Speaks Up**: Tongyi Lab unveiled **Qwen3-TTS** with multiple timbres and languages optimized for English and Chinese, documented at [ModelScope: Qwen3-TTS](https://modelscope.cn/models/damo/speech_qwen3_tts/summary).
    - Builders asked about open-source availability and API pricing, but the official response shared docs only and stayed silent on openness and cost.

**2. Diffusion Sampling & Data Efficiency Breakthroughs**

- **8-Step ODE Solver Smokes 20-Step DPM++**: An independent researcher’s WACV 2025 submission, [Hyperparameter is all you need](https://zenodo.org/records/17180452), introduces an **ODE solver** that achieves **8-step** inference (and **5-step** rivaling distillation) outperforming **DPM++2m 20-step** in FID.
    - The approach is a **training-free** sampler cutting compute by ~60% via better tracing of the **probability flow trajectory**, with code at [GitHub: Hyperparameter-is-all-you-need](https://github.com/TheLovesOfLadyPurple/Hyperparameter-is-all-you-need).
- **Diffusion Dunks on Autoregressive in Low-Data**: CMU’s blog, [Diffusion Beats Autoregressive in Data-Constrained Settings](https://blog.ml.cmu.edu/2025/09/22/diffusion-beats-autoregressive-in-data-constrained-settings/), argues **diffusion models** outperform **autoregressive** methods when data is scarce.
    - Researchers flagged missing citations and pointed to a related preprint ([arXiv:2410.07041](https://arxiv.org/pdf/2410.07041v1)) that *"generalizes the approach better"* but wasn’t cited.
- **Repeat-4x Data Trick Plays Even**: A study ([arXiv:2305.16264](https://arxiv.org/pdf/2305.16264)) showed repeating data **4x** and shuffling each epoch matches training on the same amount of unique data.
    - Practitioners discussed applying the trick to **MLPs**, treating inter-epoch shuffling as a cheap regularizer for data-limited regimes.

**3. Compute Megadeals & GPU Systems**

- **OpenAI–NVIDIA Lock a ~$100B, 10 GW GPU Pact**: Latent Space members discussed an **OpenAI–NVIDIA** plan to deploy up to **10 gigawatts** of NVIDIA systems—valued around **$100B**—for next-gen datacenters starting **late‑2026**.
    - Reactions ranged from stock optimism to debates over vendor financing, AGI expectations, and whether end users will ever feel the added compute.
- **Modular 25.6 Unifies GPUs, MAX Flexes**: Modular shipped [Modular 25.6: Unifying the latest GPUs](https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple) with support for **NVIDIA Blackwell (B200)**, **AMD MI355X**, and **Apple Silicon**, powered by **MAX**.
    - Early results claim **MAX on MI355X** can outperform **vLLM** on **Blackwell**, hinting at aggressive cross-vendor tuning and a unified developer workflow.
- **NVSwitch Know-How Boosts Multi-GPU Throughput**: Engineers shared a primer on **sharing memory addresses across GPUs** and leveraging **NVSwitch** for reductions at [Stuart Sul on X](https://x.com/stuart_sul/status/1970239956624011556).
    - These patterns matter for bandwidth-bound collectives and activation flows where efficient interconnects keep **GPU utilization** high.

**4. Agent Protocols & Constrained Outputs**

- **MCP Adds response_schema for Structured Sampling**: The MCP team discussed adding **response_schema** to the sampling protocol to request structured outputs, tracked via [modelcontextprotocol/issues/1030](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1030).
    - Contributors expect modest SDK changes plus provider-specific integration, with a volunteer targeting an October demo implementation.
- **MCP Registry Publishing & Remote Servers Land**: Publishing guidance for the **MCP Registry** arrived at [Publishing an MCP server](https://github.com/modelcontextprotocol/registry/blob/main/docs/guides/publishing/publish-server.md), covering `server.json`, status, repo URL, and remotes.
    - A reference for remote configs documents `streamable-http` endpoints at [Generic server.json reference](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/generic-server-json.md#remote-server-example).
- **vLLM Bakes In Grammar-Guided Decoding**: Developers highlighted **guided decoding** that constrains logits with formal grammars in **vLLM**, see [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/sampling.py#L724) [sampling.py](http://sampling.py/).
    - They contrasted this with [KernelBench](https://github.com/google-research/kernel-bench) 0‑shot evals that skip constraints, noting grammars could eliminate many compiler errors upfront.

**5. Open-Source Platforms, DBs, and Communities**

- **CowabungaAI Forks LeapfrogAI as ‘Military-Grade’ PaaS**: **CowabungaAI**, an open-source fork of LeapfrogAI with chat, image gen, and an **OpenAI-compatible API**, launched at [GitHub: cowabungaai](https://github.com/awdemos/cowabungaai).
    - Its creator touted large code improvements and offered discounted commercial support and licensing for adoption.
- **serverlessVector: Pure-Go VectorDB Debuts**: A minimal **Golang** vector database, **serverlessVector**, is available at [takara-ai/serverlessVector](https://github.com/takara-ai/serverlessVector).
    - Engineers can test a pure-Go **vectorDB** suitable for embedded/serverless use without external dependencies.
- **Hugging‑Science Discord Spins Up**: A new **hugging-science** Discord for open projects in fusion, physics, and eval launched at [discord.gg/hU9mdFPB](https://discord.gg/hU9mdFPB).
    - Organizers are recruiting **Team Leaders**, signaling momentum for domain-focused open-science collaborations.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Invites Going Out**: Members shared invitation links for **Perplexity's Comet browser** such as [this link](https://perplexity.ai/browser/claim/2QWV7UUV6I) for users to try out, and confirmed it's available on Windows as well as Mac.
   - One user reported reaching the **Comet** daily limit and suggested upgrading to a **Max subscription** for higher usage limits.
- **Perplexity Better Than ChatGPT For Research?**: A user argued that **ChatGPT** is known for hallucinating, while **Perplexity** is better for research purposes, specifically when retrieving *trustworthy data* using this [Perplexity AI search](https://www.perplexity.ai/search/can-you-trust-the-data-provide-qjWGreD9Sg2reYcrL0jX2wI).
   - Another member echoed this sentiment.
- **GPT-5 Launch Anticipation**: Enthusiasm is building for upcoming models like **GPT-5**, with some preparing for a potential third **AI winter**, while others dismiss that theory.
   - One member doubts the current **GPT-4** models being smarter than the newest generation, while others express excitement for the advances in AI that can help them *understand the universe itself*.
- **Perplexity Pro Users Feel Overlooked**: A user shared [a tweet](https://x.com/perplexity_ai/status/1970165704826716618?s=19) highlighting complaints about **Perplexity Pro** users feeling left behind regarding feature parity with **Max** subscribers.
   - A respondent dismissed these concerns, anticipating the new features would not work properly for another four months.
- **Shareable Threads Encouraged**: A member reminded others to ensure their threads are set to *Shareable*, linking to [a specific Discord channel message](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
   - This should allow easier access to others.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **VRAM Goes BRRR for Massive Context Windows**: A user calculated that a **1M context length** model would require **600GB VRAM**, while another user questioned the accuracy of the [VRAM calculator](https://apxml.com/tools/vram-calculator) for their company's needs with **16 concurrent users** and **Qwen 3 30B** models.
   - It was noted that performance significantly degrades on a **30B model** beyond **100k context length** and training on shorter contexts is sufficient after initial context extension.
- **DeepSeek's Deep Dive with Huawei?**: A user mentioned that **DeepSeek** might be experiencing issues due to using **Huawei Ascent chips**, with the newest version [DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) showing bug fixes.
   - The fixes have somewhat degraded in reasoning without tool use and slight improvements in agentic tool use.
- **Quantization Quagmire: Google and Apple Join**: Members discussed various quantization techniques including **QAT (Quantization Aware Training)** and its successful implementations by **Google** and **OpenAI**.
   - There was also mention of [Apple's super-weight research](https://machinelearning.apple.com/research/super-weight) and experiments with **NVFP4** and **MXFP4**, with NVFP4 showing a slight performance lead, and [Unsloth's DeepSeek R1 blog](https://unsloth.ai/blog/deepseekr1-dynamic).
- **Money Talks: OOM Error Edition**: A user is *willing to pay **50 USDT*** for assistance in resolving an **out-of-memory (OOM) issue** after reviewing tutorials.
   - Another user echoed the same sentiment, indicating a potential demand for paid support within the community.
- **Diffusion Dominates Autoregressive in Data-Constrained Settings**: A [blog post](https://blog.ml.cmu.edu/2025/09/22/diffusion-beats-autoregressive-in-data-constrained-settings/) suggests **diffusion models** outperform **autoregressive models** when data is limited.
   - Another [cited paper](https://arxiv.org/pdf/2305.16264) indicates that repeating data **4x** and shuffling after each epoch yields results similar to using unique training data.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Indonesian Video Prompt Creates Vivid Imagery**: A member shared an **Indonesian** prompt to generate a video from a photo, detailing **scene, movement, stylization, audio, and duration** specifications, requesting a **2.5D hybrid animation** style with predominantly **blue and white colors and neon red** in the background, accompanied by **heavy rock music at 150 BPM**.
   - Translation provided: *Generate video from photos [upload photos]*.
- **Grok 4 Fast Undercuts Gemini Flash**: Discussion arose around **Grok 4 Fast's** performance relative to **Gemini Flash**, with one member stating that **Grok 4 Fast** is significantly better and cheaper.
   - It was further noted that *Grok 4 Fast pricing encourages others to offer competitive pricing*.
- **Seedream 4 2k Struggles with Ethnicity**: Users are reporting issues with **Seedream 4 2k** failing to maintain the integrity of the character's ethnicity when using multiple references, and **Seedream4 2k** is better for speed of generation with good results.
   - One user said *it gets multiple reference images absolutely wrong, like all the time, and sometimes it give wrong output even in single image reference*.
- **AI Models Spark Debate in Medicine**: Discussion covers the potential of AI models in healthcare, with concerns raised about their ability to identify fatal drug combinations.
   - A member with experience in drug discovery stated, *It works fine with fine tuned model with real data*.
- **Gemini 3.0 Flash Integrates Everywhere**: Amidst a flurry of **Gemini** integrations, speculation intensifies around a potential **Gemini 3.0 Flash** release, possibly featuring integrated video capabilities, and potential deployment in mass home assistant devices.
   - Members are wondering why *Google has been deploying Gemini everywhere this week*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Token Use Causes Sticker Shock**: A user reported unexpectedly high **token usage** on a new Cursor account, clocking **146k tokens** in 22 prompts, and asked for clarification.
   - Community members explained that **old chat logs** and **attached files** inflate token usage, linking to [Cursor's learn pages on context](https://cursor.com/learn/context).
- **Kaspersky Mistakenly Flags Cursor as Malware**: A user reported that **Kaspersky** is flagging Cursor as *malware*, sparking discussion around **false positives**.
   - Cursor support requested logs to investigate, reassuring users it's a generic warning related to potentially unwanted app (PUA) detection.
- **SpecStory Auto-Backups Chat Exports**: Users shared [SpecStory](https://specstory.com/), which automatically exports chats as files, to get around issues with **randomly corrupted chats**.
   - One user noted they wouldn't submit their chats to 3rd parties, and thus would rather use the local version of the tool.
- **GPT-5 Speculated to Offer Cheaper Limits than Claude Sonnet**: The community speculated that because [GPT-5 is cheaper than Claude Sonnet 4](https://link.to.gpt5-pricing), it will offer better limits.
   - While someone pointed out that GPT-Mini is free, a user clarified they were referring to Codex.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **vLLM Eyes Image/Video Generation**: A member considered the alignment of adding image or video generation capabilities to **vLLM**, planning to contact the **multimodal features team**.
   - The discussion explores expanding **vLLM's** functionalities beyond language models.
- **GB300s powers Quantum Gravity Research**: A member plans to use a substantial number of **GB300s** to facilitate high-compute scale diffusion for **magnetohydrodynamics** and **loop quantum gravity** modeling.
   - The member notes the **magnetohydrodynamics** model is more likely to succeed compared to the highly experimental **loop quantum gravity** approach.
- **CPU strangling MLPerf on local runs**: A member reported running **MLPerf inference** locally is bottlenecked by the **CPU**, despite sufficient **VRAM**.
   - The low **GPU utilization** indicates a significant performance issue that needs resolution.
- **Tianqi Chen Discusses ML Systems Evolution**: A recent interview with **Tianqi Chen** discussed [Machine Learning Systems](https://www.youtube.com/watch?v=jvqsvbntEFQ), **XGBoost**, **MXNet**, **TVM**, and **MLC LLM**.
   - Chen's reflections include his work at **OctoML**, and his academic contributions at **CMU** and **UW**.
- **New kids CowabungaAI splinters from LeapfrogAI**: An open source fork of **LeapfrogAI** called **CowabungaAI** was announced; it is a **military-grade AI Platform as a Service** by Unicorn Defense.
   - It has similar functionalities to OpenAI, including chat, image generation, and an OpenAI compatible API, and is available on [GitHub](https://github.com/awdemos/cowabungaai).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging-Science Discord Launches**: A new `hugging-science` Discord channel has launched, focusing on open-source efforts in fusion, physics, and eval, accessible [here](https://discord.gg/hU9mdFPB).
   - The channel seeks **Team Leaders** for projects, providing a leadership opportunity to guide exciting initiatives.
- **Diffusion ODE Solver steps down to 8**: An independent researcher has unveiled an **ODE solver** for diffusion models, achieving **8-step inference**, outperforming **DPM++2m's 20 steps**, and **5-step inference** rivalling the latest distillation methods; see the paper on [Zenodo](https://zenodo.org/records/17180452) and code on [GitHub](https://github.com/TheLovesOfLadyPurple/Hyperparameter-is-all-you-need).
   - This **training-free improvement** reduces computational costs by **~60%** while enhancing quality by better tracing the **probability flow trajectory** during inference.
- **Golang VectorDB sees the light**: A member has created **serverlessVector**, a pure Golang vectorDB and provided a [link to the GitHub repo](https://github.com/takara-ai/serverlessVector).
   - It is written in **Go** and available for immediate testing and implementation.
- **HF Inference Providers face Quality Complaints**: Members expressed concerns about the quality of **HF Inference providers**, wondering how HF guarantees the quality of inference endpoints, especially concerning quantization.
   - They added that they *feel endpoints should be default be zdr.*



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek Model Terminus Arrives!**: **DeepSeek** launched the final **v3.1** iteration, *Terminus*, with improved language consistency and code/search agent capabilities; open weights are available on [Hugging Face](https://huggingface.co/deepseek-ai).
   - The community immediately inquired about **DeepSeek-R2**, hinting at future developments.
- **Untapped Capital launches Fund II**: **Yohei Nakajima** announced **Untapped Capital Fund II**, a **$250k** pre-seed vehicle, continuing its mission to back founders outside traditional networks.
   - The team has shifted to a **top-down approach**, proactively sourcing startups based on early trend identification.
- **Alibaba's Qwen3-TTS makes its debut!**: **Alibaba's Tongyi Lab** introduced **Qwen3-TTS**, a text-to-speech model with multiple timbres and language support, optimized for English and Chinese; see [ModelStudio documentation](https://modelscope.cn/models/damo/speech_qwen3_tts/summary).
   - User queries focused on open-source availability and API pricing, though the official team did not address open-source plans.
- **OpenAI and NVIDIA Strike $100B GPU Mega-Deal!**: **OpenAI** and **NVIDIA** partnered to deploy up to **10 gigawatts** (millions of GPUs) of NVIDIA systems—valued at **~$100 B**—for OpenAI’s next-gen datacenters starting late-2026.
   - Reactions ranged from celebrating the stock boost to debating vendor financing, AGI expectations, and the potential impact on end-users.
- **Among AIs Benchmark Tests Social Smarts**: **Shrey Kothari** introduced *Among AIs*, a benchmark evaluating language models' deception, persuasion, and coordination within Among Us; **GPT-5** excelled as both Impostor and Crewmate.
   - Discussions included model omissions (**Grok 4** and **Gemini 2.5 Pro** incoming), game selection, data concerns, discussion rules (**3-turn debates**), and enthusiasm for game-based AI evaluations.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Text-davinci-003 Debuts Day-and-Date with ChatGPT**: **Text-davinci-003** launched the same day as **ChatGPT**, with OpenAI quickly labeling it a **GPT-3.5 model**.
   - An insider clarified that an updated **Codex model**, **code-davinci-002**, was also part of the **GPT 3.5 series** and released the day before **ChatGPT**.
- **Diffusion ODE Solver Races Ahead**: An independent researcher crafted a new **ODE solver** for diffusion models hitting **8-step inference** that outpaces **DPM++2m's 20-step inference** in **FID scores** while slashing computational costs by ~60%.
   - The solver enhances inference by better tracing the probability flow trajectory, detailed in a [WACV 2025 paper](https://zenodo.org/records/17180452) with code available on [GitHub](https://github.com/TheLovesOfLadyPurple/Hyperparameter-is-all-you-need).
- **GPT-3.5 Family Tree**: Members are hashing out the origins of **ChatGPT** and other **GPT-3.5** models like **code-davinci-002**, **text-davinci-002**, and **text-davinci-003**.
   - One insider hinted that **ChatGPT** was fine-tuned from a chain-finetuned, unreleased model.
- **Benchmarking MMLU Subtasks**: The community considered the possibility of benchmarking an **MMLU pro** subtask using **lm-eval**, focusing on subsets like **mmlu_law**.
   - Exploring this capability could allow for more detailed and precise evaluations of AI model skills within the **lm-eval** framework.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **HuggingFace Comments Spark Laughter**: Members found [the comments on HuggingFace](https://huggingface.co) humorous, with one stating *yeh i was lmaoing when i saw that*.
   - The specific nature of these comments was not detailed, but it indicates an active and engaged community reaction to content on the platform.
- **OLMo-3 Safetensors Search Intensifies**: A member is actively seeking leads on **OLMo-3 safetensors**, and has joined a Discord to track progress.
   - Although inference code is available, they highlighted that *no weights on HF yet*, suggesting the model isn't fully accessible for use.
- **Qwen3 Omni Launches with Multimodal Prowess**: [Qwen3 Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) is now live, boasting **36B parameters** and multimodal capabilities.
   - Reportedly, it *beats Whisper v2/3 and Voxtral 3B* in scam call audio processing, supporting **17 audio input languages** and **10 output languages**.
- **Realtime Perceptual AI Race Commences**: The community anticipates the rise of **realtime perceptual AI** (audio, video, and other modalities simultaneously).
   - Apple's realtime vision model was mentioned as a potential indicator of developments, sparking curiosity about the lack of public releases.
- **LoRA Training Achievable on RTX 4090**: It was mentioned that training a **LoRA** on **14B** models locally is feasible using a single **RTX 4090**.
   - A member is working on something similar, and the constraints they're facing are **bandwidth** and **latency**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Forks Navigate with Navigator Mode**: Members suggested using [aider forks](https://github.com/dwash96/aider-ce/tree/main) with **navigator mode** for an automated experience using *uv pip install* on fork repos or the **aider-ce** package.
   - The package integrates **MCP** and **navigator mode** to streamline the coding process.
- **Augment CLI Excels at Codebase Augmentation**: The **Augment CLI** tool shines with large codebases, particularly when using **Claude Code (CC) with Opus/Sonnet** and **Codex for GPT-5**.
   - The user mentioned that Codex doesn't have an API key, while **Perplexity MCP** works well with **MCP integration**.
- **Deepseek V3.1 Configuration Guidance Sought**: A user requested advice on the initial setup and configuration of **Deepseek V3.1**, along with preferred **web search tools**.
   - Because *aider* does not have a built-in tool, one member suggested **Perplexity MCP for web search** to work with it.
- **Aider Agents Multiply via Command Line**: Members discussed setting up **multiple aider agents** via the command line for external orchestration, rather than a built-in solution.
   - The suggestion was to use **git worktrees** for concurrent modifications to the same repository, which enables running multiple agents simultaneously.
- **LLM Loses Track Editting Prompt Files**: When using a file to set up the prompt, **aider prompts to edit files** and, simultaneously, the **LLM gets confused** about the task at hand.
   - The LLM then asks for clarification of intent, specifically whether it should act as the 'User' within the APM framework or modify a file using the SEARCH/REPLACE block format.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Rustaceans summon Mojo via FFI**: Members have been exploring using **FFI** to call **Mojo** from **Rust**, which is similar to calling **C** via the **C ABI**.
   - To ensure correct type handling, the Mojo function must be marked with `@export(ABI="C")`.
- **Manual Mojo Binding still only option**: The creation of **C header -> Mojo binding generators** remains a work in progress (**WIP**), rendering **CXX** assistance unavailable.
   - Currently, generating Mojo bindings necessitates a manual approach.
- **Windows Support still MIA**: The question of **Windows support** for Mojo was raised among users.
   - The response indicated that it is *not coming any time soon*.
- **Modular 25.6 turbocharges GPU Performance**: Modular Platform **25.6** is live, delivering peak performance across the latest GPUs including **NVIDIA Blackwell (B200)** and **AMD MI355X**, check out the [blog post](https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple) for more information.
   - Early results show **MAX** on **MI355X** can even outperform **vLLM** on **Blackwell**, with unified GPU programming now available for consumer GPUs like **Apple Silicon, AMD,** and **NVIDIA**.
- **MAX demands .mojopkg**: To use **MAX**, a `.mojopkg` file is required alongside the executable, containing the highest level **MLIR** that **Mojo** can produce after parsing, for the runtime's **JIT** compiler.
   - For platforms hiding hardware details (**Apple**, **NVIDIA GPUs**), **Mojo** hands off compilation to the driver, performing a *one-shot JIT* without profiling, unlike V8 or Java.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **JEPA Chat Incoming**: A chat about Joint Embedding Predictive Architecture (JEPA) paper ([https://www.arxiv.org/abs/2509.14252](https://www.arxiv.org/abs/2509.14252)) by **Yann LeCun** is scheduled for <t:1758567600:t>.
   - One member linked to an [OpenReview document](https://openreview.net/pdf?id=BZ5a1r-kVsf) about **JEPA** along with [a YouTube video](https://youtu.be/jSdHmImyUjk?si=HdFl4PZFe2nvqSi5) and two **X** (formerly Twitter) [posts](https://x.com/BlancheMinerva/status/1969974401081725214) and [https://fxtwitter.com/randall_balestr/status/1969987315750584617
- **GPT Gets Philosophical, Scores Low**: A user rated a **GPT model** a **7/10** for parsing their philosophy, but only a **4/10** for expanding on it, and **3/10** for formatting, deeming it *not good enough*.
   - The user indicated improved results after further interactions, suggesting either improved **prompting skills** or enhanced **machine reading capabilities**.
- **Presenter Seeks Optimal Paper Time**: A member asked about the best time to present a paper, suggesting an earlier session for those in the eastern timezone.
   - The member suggested presenting **6 hours earlier or later** on most days.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Environment Variables Trump MCP Secrets**: A member cautioned against passing secrets through **MCP**, advising the use of **environment variables** for single-user services, or **login+vault/OAuth** for more complex setups.
   - Their **JIRA MCP server** implementation uses stdio and retrieves credentials from the environment, deploying per single-user to avoid leaking secrets via process tables.
- **DSPy Optimization Suffers from Trace ID Jitters**: A user highlighted that initializing modules with a trace ID results in the same ID being used for all documents when running a batch after optimizing, and asked how to handle per-item **trace IDs** in **DSPy modules** without breaking optimization.
   - They considered recreating modules per article (too expensive) and moving the trace ID to forward calls, questioning if this would affect optimization since the trace_id gets passed to llm gateway for logs and auditing.
- **GEPA Silent on ReAct Context Overflows**: A user asked about experiences with **GEPA** for **ReAct Agents**, focusing on how **context overflows** would be handled with long agent trajectories.
   - Unfortunately, no one had a story to tell.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Response Schema Gains Support**: An issue to *Add **response_schema** to the **MCP Sampling Protocol*** was converted into a discussion with a member willing to implement a demo in October, enabling requests for **structured output** when using sampling via [this issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1030).
   - The consensus is that implementing this in the SDK isn't complex, with the main effort lying in the integration between the SDK and the LLM API, potentially needing provider-specific code.
- **Claude's Constrained Output Capability Examined**: The group discussed presenting **constrained output** as a client capability, with the issue that **Claude models** are incorrectly identified as not supporting this feature.
   - The preferred approach is to present **response_schema** support, allowing the client host to determine the actual implementation.
- **MCP Registry Publishing Procedures**: Instructions for publishing servers to the **Model Context Protocol (MCP)** Registry have been shared, beginning with [this guide](https://github.com/modelcontextprotocol/registry/blob/main/docs/guides/publishing/publish-server.md).
   - The guide includes steps to create an empty **GitHub** repo and a `server.json` file that specifies name, description, status, repository URL, and remote endpoint.
- **Remote Server Configurations Unveiled**: Remote server configurations for the **Model Context Protocol (MCP)** have been linked, featuring [this reference](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/generic-server-json.md#remote-server-example).
   - The provided `server.json` example defines a `$schema` and includes a `remotes` array, specifying the `streamable-http` type and URL.
- **MCP Install Instructions Auto-Generated**: A member mentioned using a tool to generate a readme file with instructions for installing the **Model Context Protocol (MCP)** in various clients via [MCP Install Instructions](https://mcp-install-instructions.alpic.cloud/).
   - The tool was reportedly *'pretty cool'* and beneficial for creating installation guides.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Colfax Meeting on Monday**: There will be a [meeting #89](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/meeting) on Monday at 9am San Diego time for a **company update**.
   - Starting next week, the meeting will be **3 hours earlier**.
- **RANGEIFY Progress Questioned**: Members questioned whether *RANGEIFY* would be default by the end of the week, including store, assign, group reduce, disk, jit, const folding, buf limit, llm, and openpilot.
   - It was noted that **children** are *not making progress* and **image** is not complete.
- **CuTe DSL a Potential Gamechanger**: Members mentioned that the **CuTe DSL** is a *potential gamechanger*.
   - They added that **ThunderKittens** is *nicestarting*.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Rejects Provocation Attempts**: One member shared an experience of *provoking* **Kimi**, appreciating its rejection of blind obedience or sympathy.
   - Another member expressed a similar sentiment, noting that **Kimi's attitude** is a reason why it's their favorite model.
- **Claude Suffers Prompt Injection Debacle**: Members discuss how **Claude** has been weakened due to [prompt injection techniques](https://en.wikipedia.org/wiki/Prompt_injection), leading it to disagree when context exceeds a certain length.
   - They noted that it is unlike **Kimi K2**, and some express disappointment in the changes to **Claude**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **GenAI E-Book Reader goes Warp Speed**: The latest release of the **GenAI E-Book Reader** introduces Generative Intelligence features for enhanced text clarification, summarization, and dictionary functions.
   - The new version integrates with **OpenRouter.ia**, granting users access to over **500 large-scale language models**, as showcased in [this video](https://youtu.be/dHggyhodAH4).
- **OpenRouter.ia plugs into GenAI E-Book Reader**: The **GenAI E-Book Reader** now supports **OpenRouter.ia**, opening up access to over **500 large-scale language models** for enriched reading assistance.
   - Users can now utilize a diverse range of models for text clarification, summarization, and advanced dictionary functionalities.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1419639406737887318)** (1175 messages🔥🔥🔥): 

> `Comet Browser invitation, GPTs Agents training, OpenAI Platform's sidebars, Comet availability for ipadOS, AI winter` 


- **Comet browser invites going out**: A member shared an [invitation link for Perplexity's Comet browser](https://perplexity.ai/browser/claim/2QWV7UUV6I) for others to try out.
   - Another confirmed that **Comet** is available on Windows too, not just Mac: *Just open it on a windows machine*.
- **Comet Daily limit reached, upgrade available**: A user reported receiving a persistent notification saying **Comet personal search daily limit reached**.
   - A member suggested upgrading to a Max subscription for higher usage limits.
- **Perplexity is better than ChatGPT for research sort of**: A user mentions that ChatGPT is known for hallucinating, generating information that is not real, whereas Perplexity is better for research purposes.
   - Another member stated **Perplexity** is good for research sort of, but chatgpt i have heard that it hallucinates*.
- **GPT-5 launch imminent, experts claim**: Multiple users express excitement for upcoming models like **GPT-5**, and some say they are preparing for the third AI winter, while others dismiss this.
   - One member stated that they *highly doubt about the gpt 4 models being smarter than the newest generation*, and others expressed their excitement to see *With that, we’ll understand the universe itself*.
- **Perplexity Pro users feeling left behind**: A user shared a link to a [tweet](https://x.com/perplexity_ai/status/1970165704826716618?s=19) complaining about **Perplexity Pro** users being left behind in terms of features compared to Max subscribers.
   - Another responded they are tripping about some B.S. that probably doesn’t even work properly for probably another four months.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1419639864155963403)** (5 messages): 

> `Comet Invitation, Shareable Threads, Trustworthy Data, Invitation Request` 


- **Comet Invite Drops**: A member shared a [Comet invitation link](https://perplexity.ai/browser/claim/2QWV7UUV6I) for others to try out **Comet**.
- **Shareable Threads Encouraged**: A member reminded others to ensure their threads are set to *Shareable*, linking to [a specific Discord channel message](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Trustworthy Data Questioned**: A link to a **Perplexity AI** search about [*trustworthy data*](https://www.perplexity.ai/search/can-you-trust-the-data-provide-qjWGreD9Sg2reYcrL0jX2wI) was shared.
- **Invitation Plea**: Another member requested an invite, providing [another invitation link](https://perplexity.ai/browser/claim/1445B3PB7M) and [a link to Chris Biju's profile](https://pplx.ai/chris-biju).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1419640498280464455)** (374 messages🔥🔥): 

> `VRAM Usage for 1M Context Length, GRPO Fine-Tuning for GPT-OSS-20B, DeepSeek V3.1 Terminus and Huawei Ascent Chips, Qwen3 and Data Privacy Concerns, QAT and GGUF Quants` 


- **Ramping Up VRAM Requirements for Large Context Windows**: A user calculated that a **1M context length** model would require **600GB VRAM**, while another user questioned the accuracy of the [VRAM calculator](https://apxml.com/tools/vram-calculator) for their company's needs with **16 concurrent users** and **Qwen 3 30B** models.
   - It was noted that performance significantly degrades on a **30B model** beyond **100k context length** and training on shorter contexts is sufficient after initial context extension.
- **GRPO's Grand Reveal Soon?**: Members discussed the imminent release of **GRPO finetuning for GPT-OSS-20B** and its limitations for increasing context length on simple datasets like **GSM8k**.
   - It was suggested to use at least **100 examples** for a meaningful test and over **1000** for optimal results when training a model for a specific programming task.
- **DeepSeek's Deep Dive with Huawei?**: A user mentioned that **DeepSeek** might be experiencing issues due to using **Huawei Ascent chips**.
   - DeepSeek has released [DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) which is a small iteration with bug fixes, but it somewhat degraded in reasoning without tool use and slight improvements in agentic tool use.
- **Qwen3's Quietly Questionable Quotas?**: The topic of data privacy arose in the context of using **OpenAI** or **Gemini paid APIs**, with the consensus being that users should assume their data is being used for training despite any stated policies.
   - A member shared a link to a [court order](https://techstartups.com/2025/06/06/court-orders-openai-to-preserve-all-chatgpt-logs-including-deleted-temporary-chats-and-api-requests/) requiring **OpenAI** to preserve all **ChatGPT logs**, including deleted temporary chats and API requests.
- **Quantization's Quantum Quest: QAT vs MXFP4**: Members discussed various quantization techniques including **QAT (Quantization Aware Training)** and its successful implementations by **Google** and **OpenAI**.
   - There was also mention of [Apple's super-weight research](https://machinelearning.apple.com/research/super-weight) and experiments with **NVFP4** and **MXFP4**, with NVFP4 showing a slight performance lead.  There was also conversation of Unsloth's DeepSeek R1 blog [post](https://unsloth.ai/blog/deepseekr1-dynamic).


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1419856967232393277)** (1 messages): 

> `Collaboration Opportunities, Software Engineering, Small Business Ventures` 


- **Collaboration: A New Business is Brewing**: A software engineer and small business owner has opened the door to potential collaboration.
   - This could lead to **new projects, ventures, or partnerships** within the community.
- **Software Engineering Collab**: A software engineer is seeking collaboration opportunities, hinting at potential projects needing technical expertise.
   - This may involve **software development, system design**, or **technical consulting**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1419704732443869326)** (41 messages🔥): 

> `Loss Curve Success, New iPhone Acquisition, CS Uni vs Bootcamps, Gacha Game Ratios, DataSeek Tool` 


- **Model Training Yields Good Loss Curve**: A member celebrated achieving a satisfactory loss curve in their model training experiment ([image attached](https://cdn.discordapp.com/attachments/1179039861576056922/1419704732485685349/image0.jpg?ex=68d2ba91&is=68d16911&hm=ba1d37bf9ab3f7d1ef93e890068e71efa4d82bc5160ff45ba680167a0fcf260a&)).
   - They clarified it wasn't an LLM, but rather *an experiment to learn training more types of models*, fueled by coffee and Noccos (180mg caffeine each).
- **iPhone Upgrade Sparks Sheet Debate**: A member showed off a new iPhone ([image attached](https://cdn.discordapp.com/attachments/1179039861576056922/1419749708393353417/finally-bought-the-new-iphone-v0-quu51epeq5qf1.jpg?ex=68d2e474&is=68d192f4&hm=df50a949cb010b8cd43ae72f0b179d95b01bc8e08a1706ab1b4b80050799562f&)).
   - One response jokingly suggested selling the phone and delaying bed sheet replacement while another member said *Every time I see such pics I feel better about how I live lol.*
- **CS Uni Questioned vs Bootcamps**: A member said that CS uni was worse than no uni and *these bootcamps were unironically better option (planting less outdated shit in your head, faster and cheaper)*.
- **Gacha Game Ratio Formula Leaked**: A member leaked a macro that **gacha devs** are using to determine new characters ([image attached](https://cdn.discordapp.com/attachments/1179039861576056922/1419755272489930874/image.png?ex=68d2e9a2&is=68d19822&hm=9404da767020fbbbb58aac3b16d88c5b74fac7aa1e714481fce0f29b590d182c&)).
   - The leaked code joke showed that `if (banner_needs_money)  -> add curvy adult woman` and similarly for other conditions.
- **DataSeek for Agent Sampling**: A member shared [DataSeek](https://github.com/robbiemu/dataseek), a tool they created for gathering samples with agents.
   - They used the tool to gather **1000 samples** for their *claimify*-related project.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1419668726751559822)** (23 messages🔥): 

> `OOM Errors & USDT, Blackwell CUDA Issues, Orpheus TTS Fine-Tuning` 


- ****Money for Nothing?** User offers USDT for OOM fix**: A user is *willing to pay **50 USDT*** for assistance in resolving an **out-of-memory (OOM) issue** after reviewing tutorials.
   - Another user echoed the same sentiment, indicating a potential demand for paid support within the community.
- ****Blackwell Blues**: Debugging CUDA with New Architectures**: A user rebuilt their environment with **CUDA 12.8/12.9** and **ARCH 120** to support their **Blackwell** card, but continues to face issues, particularly with **TP (Tensor Parallelism)**.
   - They've found that *offloading to CPU or using only 1 GPU* works but suspect a conflict possibly related to *disabling amd_iommu* for **vLLM**, leading to *funky cuda errors*.
- ****Orpheus in Error**: Batch Size Mismatch in TTS Fine-Tuning**: A user encountered a `TorchRuntimeError` during fine-tuning of **Orpheus TTS** using the [provided notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb) after switching to their own dataset, with the error indicating an *input batch_size mismatch*.
   - It was identified that the user's dataset had **variable length samples**, and the suggested fix of **increasing the max_seq_length** (or removing longer samples) resolved the issue, as described in the [TRL documentation](https://huggingface.co/docs/trl/main/en/reducing_memory_usage#how-to-choose-the-maxlength-value).


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1419820771869589726)** (1 messages): 

> `LLMs, Fine Tuning, Task-Specific Models` 


- **LLMs Tailored for Tasks**: The future involves **LLMs** fine-tuned for specific tasks, even down to individual projects.
   - This approach optimizes performance and relevance in specialized applications, ensuring models are purpose-built.
- **The Dawn of Bespoke Models**: The user thinks **LLMs** will become highly specialized, honed for precise tasks and projects.
   - This evolution promises efficiency, accuracy, and adaptability in diverse domains.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1419774309274091592)** (13 messages🔥): 

> `Diffusion vs Autoregressive, Data Repeating, Paper Citation, Peer Review` 


- **Diffusion Dominates Autoregressive Approaches**: A [blog post](https://blog.ml.cmu.edu/2025/09/22/diffusion-beats-autoregressive-in-data-constrained-settings/) suggests **diffusion models** outperform **autoregressive models** when data is limited.
   - Another [cited paper](https://arxiv.org/pdf/2305.16264) indicates that repeating data **4x** and shuffling after each epoch yields results similar to using unique training data.
- **Data Repeating Shuffles the Deck**: Repeating the same data **4x** and shuffling after each epoch can yield similar outcomes to using the same amount of unique training data ([arxiv link](https://arxiv.org/pdf/2305.16264)).
   - This suggests that shuffling between epochs is beneficial, prompting consideration for applying this technique to **MLPs**.
- **Missing Citation Sparks Debate**: A member pointed out that [this paper](https://arxiv.org/pdf/2410.07041v1) seems to be a repeat of the CMU blog post but with nicer graphs, but the CMU blog post doesn't cite the arxiv paper.
   - The response was that the later paper generalizes the approach better, but should have been cited.
- **Peer Reviewers Save the Day**: A member asked about the severity of not citing related work, and the response was *in general, this is the job of a peer reviewer*.
   - It was further noted that papers on **arXiv** are often unreviewed and subject to updates, leaving room for corrections.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1419641668344221746)** (446 messages🔥🔥🔥): 

> `Video generation from photos in Indonesian, Grok 4 Fast performance, Seedream 4 2k vs High-Res, AI in medical field, Gemini 3.0 Flash rumors` 


- **Indonesian Video Generation Prompt Shared**: A member shared an **Indonesian** prompt to generate a video from a photo, detailing **scene, movement, stylization, audio, and duration** specifications, requesting a **2.5D hybrid animation** style with predominantly **blue and white colors and neon red** in the background, accompanied by **heavy rock music at 150 BPM**.
   - Translation provided: *Generate video from photos [upload photos].*
- **Grok 4 Fast Surprises with Competitive Pricing**: Discussion arose around **Grok 4 Fast's** performance relative to **Gemini Flash**, with one member stating that **Grok 4 Fast** is significantly better and cheaper.
   - It was further noted that *Grok 4 Fast pricing encourages others to offer competitive pricing.*
- **Seedream 4 2k Image reference issues emerge**: Users are reporting issues with **Seedream 4 2k** failing to maintain the integrity of the characters ethnicity when using multiple references, and Seedream4 2k is better for speed of generation with good results.
   - One user said *it gets multiple reference images absolutely wrong, like all the time, and sometimes it give wrong output even in single image reference*
- **Debate sparks over AI models in medical field and drug discovery**: Discussion covers the potential of AI models in healthcare, with concerns raised about their ability to identify fatal drug combinations.
   - A member with experience in drug discovery stated, *It works fine with fine tuned model with real data.*
- **Whispers of the Wind: Gemini 3.0 Flash Rumors Fly**: Amidst a flurry of **Gemini** integrations, speculation intensifies around a potential **Gemini 3.0 Flash** release, possibly featuring integrated video capabilities, and potential deployment in mass home assistant devices.
   - Members are wondering why *Google has been deploying Gemini everywhere this week.*


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1419715547733430527)** (1 messages): 

> `Seedream-4, LMArena Models` 


- **Seedream-4-2k Joins the LMArena Battle**: The model `seedream-4-2k` is now available in **Battle**, **Direct**, & **Side by Side** modes on the [LMArena](https://lmarena.ai/?chat-modality=image).
   - The `seedream-4-high-res` model is not available at this time.
- **High-Res Still MIA**: Model `seedream-4-high-res` is currently unavailable, with updates to be announced in the channel.
   - Users are encouraged to monitor the announcements channel for further details.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1419639656370278410)** (419 messages🔥🔥🔥): 

> `Token Usage, Kaspersky Malware Flag, Chat Exports, GPT-5 Pricing` 


- **Cursor Token Use Baffles User**: A user noticed unexpectedly high **token usage** on a new Cursor account, spending **146k tokens** in 22 prompts, and sought clarification from the community.
   - Members explained that **old chat logs** and **attached files** contribute to context and token usage, and provided a link to [Cursor's new learn pages on context](https://cursor.com/learn/context).
- **Kaspersky Flags Cursor as Malware**: A user reported that **Kaspersky** is flagging Cursor as *malware*, prompting discussion about **false positives** and potential reasons such as Cursor modifying right-click menus.
   - A community member suggested double-checking the files, while Cursor support requested logs to investigate, reassuring users that it's a generic warning related to potentially unwanted app (PUA) detection.
- **SpecStory Automates Chat Exports**: Users discussed the issues with **randomly corrupted chats** and the lack of a *reapply* feature after canceling changes, then sought a solution by sharing [SpecStory](https://specstory.com/), which automatically exports chats as files.
   - One user noted they wouldn't submit their chats to 3rd parties, and thus use the local version of the tool.
- **GPT-5 Pricing is Cheaper than Claude Sonnet**: The community compared Claude Code and Codex CLI limits and suggested that because [GPT-5 is cheaper than Claude Sonnet 4](https://link.to.gpt5-pricing), it will yield better limits.
   - One pointed out that GPT-Mini is free, but a user clarified that they meant Codex.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1419683586201751634)** (13 messages🔥): 

> `vLLM affiliation, Image/Video Gen in vLLM, Sliding/Striding Multi-Node DiT Kernel, GB300s for High Compute Scale, Magnetohydrodynamics and Loop Quantum Gravity Modeling` 


- **Image/Video Gen in vLLM: A Good Fit?**: A member inquired about the potential alignment of adding image or video generation capabilities to **vLLM's** goals, considering contacting the **multimodal features team** on their Slack.
- **Sliding/Striding Multi-Node DiT Kernel Advances**: Members discussed sliding/striding kernels, referencing an **xDiT style multi node para** approach and implementations by the **thunderkittens team** for Hopper and the **cutlass team** for Big Blackwell.
   - One member is trying to create a fast, early sliding/striding multi-node **DiT kernel** for GB300s due to the high compute scale required for math/physics research.
- **GB300s Powering Quantum Gravity Research**: A member aims to leverage a substantial number of **GB300s** to facilitate high-compute scale diffusion for applications in magnetohydrodynamics and loop quantum gravity modeling.
   - The member predicts the **magnetohydrodynamics** model will likely succeed while noting the **loop quantum gravity** approach is highly experimental.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/)** (1 messages): 

exquisite_lemur_80905: There's also `TRITON_ALWAYS_COMPILE` to ignore the cache
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1419792738299150436)** (1 messages): 

> `MLPerf Inference, CPU Bottleneck, GPU Utilization` 


- **MLPerf Inference bottlenecked by CPU**: A member reported an issue with running **MLPerf inference** locally, where the process seems to be bottlenecked by the **CPU**, despite sufficient **VRAM** usage.
   - The user notes low **GPU utilization**, and seeks assistance in resolving this performance issue.
- **MLPerf struggles on local compute**: MLPerf is a suite of benchmarks that measures the performance of machine learning hardware and software.
   - Running MLPerf locally can be challenging due to resource constraints.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1419732690176835647)** (2 messages): 

> `Speeding up pip install, Setting TORCH_LOGS` 


- **Turbocharge Setup Compilation**: A member inquired about speeding up setup compilation, noting that `!pip install -v pytorch-fast-transformers` doesn't seem to use multi-CPU/parallel jobs to compile files.
   - Unfortunately no response was provided, so this is still an open question.
- **Torch Logs Configuration Tricks**: A member asked about the correct way of setting torch logs output, providing an example `%env TORCH_LOGS = '__main__,+dynamo,graph,fusion,output_code'`.
   - No response to this question was given.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1419679380908544032)** (1 messages): 

> `Tianqi Chen Interview, Machine Learning Systems, XGBoost, MXNet, TVM` 


- ****Tianqi** Interview on **ML Systems****: An interview with **Tianqi Chen**, discussing [Machine Learning Systems](https://www.youtube.com/watch?v=jvqsvbntEFQ), **XGBoost**, **MXNet**, **TVM**, **MLC LLM**, **OctoML**, **CMU**, **UW**, and **ACM**.
   - The interview also touches on the topics of *long-termism* and *original intentions* in the field of machine learning.
- ****OctoML** Founder Interviewed**: A recent interview features **Tianqi Chen**, the founder of [OctoML](https://octoml.ai/), discussing his journey and insights into machine learning systems.
   - Chen reflects on his work with **XGBoost**, **MXNet**, **TVM**, and **MLC LLM**, alongside his academic pursuits at **CMU** and **UW**.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1419726711414849566)** (1 messages): 

> `Remote Research Intern, Deep Learning, New Models, Model Building, Stipend Information` 


- **Remote Research Intern Opportunity in Deep Learning**: A member is seeking a **remote research intern** to work on complex **deep learning projects**, offering a stipend of **30-40k INR/month**.
   - The projects focus on building **new models** rather than engineering products using existing ones.
- **Emphasis on Model Development**: The internship emphasizes the development of **new deep learning models**, indicating a research-oriented role.
   - Interested candidates are encouraged to DM their resume, GitHub profile, or any relevant projects showcasing their skills.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

nwyin: https://jax-ml.github.io/scaling-book/roofline/
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1419662160245555221)** (6 messages): 

> `NVIDIA Tech Demos, Tailscale Interface & Pricing, VPN Business Models` 


- **NVIDIA Demos Evolve Bigtime**: A member shared a [YouTube video](https://www.youtube.com/watch?v=DfWSJvKFMPE) showcasing the evolution of **NVIDIA Tech Demos** from 1998 to 2025.
   - The video likely highlights the progression of graphics technology and real-time rendering capabilities over the years.
- **Tailscale Interface Irks Users**: Some members expressed reservations about **Tailscale**, noting that while it's nice, they dislike that the interface is hosted by the company.
   - A member commented *"almost all vpns have rent seeking pricing"*.
- **VPN Business Models Baffle Onlookers**: A member stated *"i can't figure out how they seem to make money the rent seeking ones or the non rent seeking ones both? i guess i dont understand any of their businesses"*.
   - The discussion revolves around the challenges in understanding the business models of both rent-seeking and non-rent-seeking VPN providers.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1419813330209476658)** (1 messages): 

> `CowabugaAI, LeapfrogAI, Open Source AI, Military-Grade AI, Commercial AI Support` 


- **CowabungaAI splinters from LeapfrogAI**: An open source fork of **LeapfrogAI**, named **CowabungaAI**, was announced; it is a **military-grade AI Platform as a Service** created by Unicorn Defense, offering similar functionalities to OpenAI, including chat, image generation, and an OpenAI compatible API, and is available on [GitHub](https://github.com/awdemos/cowabungaai).
- **Commercial Support offered for CowabungaAI**: The creator of **CowabungaAI** announced significant improvements to the source code and intends to offer commercial support for the application.
   - They are offering steep discounts on licensing opportunities.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1419750924741509131)** (2 messages): 

> `vLLM's guided decoding, grammars for automated code generation, kernel generation LLMs, KernelBench 0-shot evals` 


- **vLLM's Guidance on Grammar**: A member inquired about what grammars are generally used by **automated code generation** or **kernel generation LLMs**, noting [vLLM's guided decoding support](https://github.com/vllm-project/vllm/blob/main/vllm/sampling.py#L724) (constrain logits with a formal grammar).
- **KernelBench Skips Grammar Constraints**: They pointed out that the [KernelBench paper](https://github.com/google-research/kernel-bench) does 0-shot evals and doesn't seem to do any constrained decoding.
   - This is implied because they check for compiler issues, which would mostly be taken care of by the grammar constraint.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1419819419462406246)** (2 messages): 

> `GPU memory sharing, NVSwitch reduction` 


- **GPU Memory Sharing Info Valuable**: A member found [this explanation](https://x.com/stuart_sul/status/1970239956624011556) about sharing memory addresses across GPUs to be valuable.
   - They noted there are few public resources that discuss sharing memory and using **NVSwitch** for reduction.
- **NVSwitch Enables GPU Communication**: The discussion highlighted the importance of **NVSwitch** in facilitating efficient communication and memory sharing between GPUs.
   - This is particularly relevant for large-scale AI model training and inference, where minimizing data transfer overhead is crucial.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1419728395474046986)** (1 messages): 

> `radiation-hardened chips, Jetson usage in space, chips in the magnetosphere` 


- **Modern Chips in Space**: A member expressed surprise that modern chips are radiation-hardened for space applications.
   - They asked if this is due to **Jetson's** capabilities, proximity to Earth and the **magnetosphere**, or another reason.
- **Rad-Hard Jetson?**: The discussion centers on the possibility and implications of using **Jetson** chips in space environments, where radiation hardening is crucial.
   - Concerns are raised regarding whether the radiation tolerance is inherent to the chip design, influenced by the near-Earth orbit within the magnetosphere, or achieved through other means.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1419655396230172823)** (13 messages🔥): 

> `MI300x8, amd-all2all leaderboard, amd-gemm-rs leaderboard, Personal Bests` 


- **MI300x8 sets Personal Bests**: Members achieved **personal bests** on **MI300x8** across the `amd-all2all` and `amd-gemm-rs` leaderboards.
- **amd-all2all sees fierce competition**: A member achieved a **personal best** of **2.45 ms** on the `amd-all2all` leaderboard using **MI300x8**.
   - Another member secured **5th place** with a time of **1171 µs**.
- **amd-gemm-rs heats up the leaderboard**: One member repeatedly achieved **personal bests** on the `amd-gemm-rs` leaderboard with **MI300x8**, eventually reaching **557 µs**.
   - Other members also submitted successful runs in the **587 µs - 638 µs** range.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1419668514054209546)** (8 messages🔥): 

> `Sweep for Qwen 2 35b, Deepseek 3.1, GPT-oss progress, Release work` 


- **SWEeping Qwen and DeepSeek this weekend!**: A member ran a **sweep** for **Qwen 2 35b** and **Deepseek 3.1** over the weekend.
   - Progress on **GPT-oss** is still ongoing, and there are no new videos to share just yet.
- **Release work keeps some busy**: Some members reported they don't have much to report today, as they are occupied with finishing up a **release**.
   - They are open to discussing release-related matters later in the day if needed, or during a meeting on Wednesday.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1419639445153386607)** (7 messages): 

> `AMD Contractor Prize Eligibility, All2All Optimizations` 


- **AMD Contractor's Prize Money Predicament**: An **AMD contractor** was well-placed in a competition but was **ineligible for prize money**.
   - Despite this, they were invited to San Francisco and met the team, which was *more than the price money*.
- **Optimizations' Acceptable Spirit Questioned**: A member inquired whether certain optimizations for **all2all** would be within acceptable parameters.
   - They clarified that *not sending the tokens to the other ranks at all* would be unacceptable, and sought confirmation on other aspects; another user gave them permission to DM.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1419727035839942829)** (7 messages): 

> `CUTLASS MLP Accuracy, CuTe Layouts` 


- ****CUTLASS's** MLP accuracy differs from Pytorch's**: A member reports a **7x** speedup by implementing a two-layer MLP with **CUTLASS**, achieving approximately *atol 1e-3* accuracy when compared to **PyTorch**.
   - Another member commented that this level of error is acceptable even for **fp16**, especially when compared to serial CPU references due to differences in reduction orders, noting that the GPU results are numerically more accurate.
- ****Colfax** introduces **CuTe Layouts** blogpost**: A member shared a [blog post, paper, and example repo from Colfax](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) about **CuTe Layouts**.
   - Another member highlighted the work's capability to compute complicated layout algebra results *"by hand"* with visual simplicity, terming it the *"graphical calculus of layout diagrams"* as detailed in Chapter 4 of the paper.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1419653935030865941)** (2 messages): 

> `PyTorch autograd, JAX autograd, tinygrad autograd, torch dynamo, bytecode interception` 


- **Compiler-First Autograd Emerging**: A member is leaning towards a **compiler-first approach** similar to **tinygrad** and **JAX** for autograd implementation, noting that not all valid Python will be valid picograd.
   - They found **Torch Dynamo** and **bytecode interception** to be cool but too complex for their project's scope, advocating for explicit ramp-up to tinygrad's design decisions.
- **Design Decisions in Language Implementation**: Inspired by Krishnamurthi's approach in [PLAI](https://plai.org/), the member suggests explicitly pointing out design decisions (e.g., eager vs lazy, serial vs parallel) and alternative paths in the language implementation.
   - This treats programming languages as a *collection of features that can be composed together*, moving away from pre-scientific classification.
- **Explicit Motivation for Deep Learning Framework Components**: The member wants to make the motivation very explicit for each component of a deep learning framework so that readers understand the reasons behind design choices.
   - For example, readers should try optimizing deep neural networks with symbolic or numeric differentiation to understand why autodifferentiation exists.
- **Motivating the Need for Fusion Compiler**: Readers need to measure and profile the autograd system in eager mode to see that they're being bottlenecked by memory (data movement) on non-tensor core operations.
   - This motivates the need for a **fusion compiler** to optimize memory usage.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1419651515911507989)** (48 messages🔥): 

> `Hugging-Science Discord Launch, HF Inference Providers Quality, Gradients Clipping, gguf conversion with llama cpp, smollm's goals` 


- **Hugging-Science Discord Channel Launched**: A new `hugging-science` discord was launched for targeted efforts in open source, including fusion, physics, and eval, check it out [here](https://discord.gg/hU9mdFPB).
   - The channel is also looking for Team Leaders for each project; it could be a great growth opportunity to take charge and lead the way on some exciting stuff! <:hugging_rocket:968127385864134656>
- **HF Inference Providers face Quality Concerns**: A member expressed concerns about the quality of **HF Inference providers**, wondering how HF guarantees the quality of inference endpoints, especially concerning quantization.
   - They added that they *feel endpoints should be default be zdr.*
- **Grad Norms above `max_grad_norm` cause Heart Spikes**: A member asked why the `grad norm` is going above `max_grad_norm` even after setting it to 1.0.
   - Another member clarified that the **gradients get modified internally** by a clipping coefficient of `max_grad_norm/your_norm` and that *the pre-clipping norm is merely for logging purposes*.
- **`gguf` conversion aided by Llama CPP**: A member inquired about using a model in `gguf` format to run it in LM Studio.
   - Another member suggested using `llama cpp` to convert it to `gguf` locally, referring to [this github discussion](https://github.com/ggml-org/llama.cpp/discussions/2948).
- **Dreaming of the smollm Working Partner**: A member noted that *what smollm should be is a little working partner instead of a benchmaxxed brick*.
   - The member also said that *it's already impressive for a tiny team and having everything fully open*.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1419878026044510362)** (1 messages): 

> `Diffusion ODE Solver, DPM++2m, WACV 2025, Hyperparameter-is-all-you-need` 


- **Radical Reduction in Steps for ODE Solvers**: An independent researcher has developed a novel **ODE solver** for diffusion models that achieves **8-step inference**, beating **DPM++2m's 20-step inference** in FID scores, and **5-step inference** comparable to latest distillation methods.
   - The new approach requires **zero additional training** and reduces computational cost by **~60%** while improving quality; the paper is available on [Zenodo](https://zenodo.org/records/17180452) and the code on [GitHub](https://github.com/TheLovesOfLadyPurple/Hyperparameter-is-all-you-need).
- **Solver Design Traces Probability Flow**: The new ODE solver design better traces the **probability flow trajectory** during inference.
   - This is a pure **training-free improvement** to the sampling process.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1419654390855499827)** (7 messages): 

> `golang vectorDB, AI agent trust challenges, AgentXTrader, protein prediction dataset` 


- **Golang VectorDB is Born**: A member created a super simple pure golang vectorDB and provided a [link to the GitHub repo](https://github.com/takara-ai/serverlessVector).
   - It's called **serverlessVector** and written in **Go**.
- **AI Agent Trust Issues Survey**: A member is working on research about **AI agent trust challenges** in production and posted a link to an anonymous survey about [AI Agent trust](https://docs.google.com/forms/d/e/1FAIpQLSeHlONGquDLANN6eIPa8tlMUIM3Ii1xw_t67FYR9zMAFkLZfQ/viewform?usp=dialog) for anyone who's deployed AI agents.
   - The survey focuses on **identity verification** and **accountability issues** and claims to be completely anonymous.
- **AgentXTrader Wants to Trade for You**: A member introduced **AgentXTrader** and shared a [LinkedIn post](https://www.linkedin.com/posts/alaa-salamah-96167b227_whatif-agentxtrader-trading-activity-7375972297376948224-QfMh?utm_source=share&utm_medium=member_desktop&rcm=ACoAADjfvpkBwpCXn_8Pmby7ixjV93Dje5TcmgU) showcasing it in action.
   - It debates the best investment strategy.
- **Protein Prediction Dataset Lands on HF Hub**: A member noted that the protein prediction dataset by **INRIA** landed on the hub via a [link to the HF collections](https://huggingface.co/collections/DataQuests/dyna-repo-inria-mdposit-68d1ee142909acba52d0641c).
   - It is a **dyna-repo** by **INRIA** called **mdposit**.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 messages): 

fingaz_ai: i havent but im also looking into that same feature i just havent isolated one yet.
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1419730810738835576)** (2 messages): 

> `In-Person Meetup in NYC, GSM8k Eval on Trained Model` 


- **HuggingFacers plan IRL meetup in NYC**: Members in NYC are planning a weekly in-person meetup for the course and are collecting [availabilities via lettucemeet.com](https://lettucemeet.com/l/rNvMawas).
- **Members are running GSM8k evals**: A member is checking if anyone has successfully run the **GSM8k eval** on their trained model.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1419707468291768486)** (4 messages): 

> `Starting the Agents Course, Backgrounds of new course members` 


- **Fresh Faces Fine-Tune Forward**: Several individuals are starting the "Fine-Tuning Language Models" course today, with one member from Ethiopia seeking advice.
   - It's a great time to delve into the world of language models and connect with fellow learners.
- **Diverse Devs Deep-Dive into Agents**: New course members include a full-stack developer/entrepreneur aiming to apply agents for client projects, and an analytics expert from Brazil venturing into agent development.
   - The course attracts a diverse range of backgrounds all looking to expand their expertise with AI agents.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1419696013345689681)** (44 messages🔥): 

> `DeepSeek Terminus, Claude resumable streaming, Untapped Capital Fund II, Alibaba Qwen3-TTS, OpenAI NVIDIA deal` 


- **DeepSeek Model Terminus Launches!**: **DeepSeek** released a final **v3.1** iteration, *Terminus*, boosting language consistency, code/search agents, and benchmark stability, and the open weights are now on [Hugging Face](https://huggingface.co/deepseek-ai).
   - The community is already asking: *when is DeepSeek-R2?*
- **Untapped Capital Launches Fund II Pre-Seed Bonanza**: **Yohei Nakajima** announced **Untapped Capital Fund II**, a **$250k** pre-seed, generalist vehicle that continues the 2020-founded fund’s mission to back founders outside typical networks.
   - For this fund the team has shifted to a **top-down approach** spotting early trends and proactively sourcing startups.
- **Alibaba's Qwen3-TTS hits the stage!**: **Alibaba's Tongyi Lab** announced **Qwen3-TTS**, a new text-to-speech model supporting multiple timbres, languages, and dialects with high naturalness, currently optimized for English and Chinese.
   - The community is mostly excited, but many users immediately ask whether the model will be open-sourced or what its API pricing will be; the official team replied with a [ModelStudio documentation link](https://modelscope.cn/models/damo/speech_qwen3_tts/summary) but stayed silent on open-source plans.
- **OpenAI and NVIDIA lock in 100B Mega-Deal!**: **OpenAI** and **NVIDIA** sealed a strategic partnership to deploy up to **10 gigawatts** (millions of GPUs) of NVIDIA systems—valued at **~$100 B**—laying the groundwork for OpenAI’s next-gen datacenters beginning late-2026.
   - Reactions ranged from cheering the stock surge to debating vendor-financing tactics, AGI hype, and whether ordinary users will ever see the extra compute.
- **Among AIs: Social Intelligence gets a new benchmark**: **Shrey Kothari** unveiled *Among AIs*, a new benchmark where top language models compete in Among Us to evaluate deception, persuasion, and coordination skills, and **GPT-5** ranked highest in both Impostor and Crewmate wins.
   - Replies cover missing models (**Grok 4** and **Gemini 2.5 Pro** will join soon), choice of game, training-data concerns, discussion mechanics (**3-turn debates**, equal speaking time), and enthusiasm for game-based AI evaluations.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1419828424528101468)** (4 messages): 

> `Google Gemini, Runway AI, Runway Gen-2` 


- **Gemini + Runway AI combine for video manipulation!**: A user shared a [TikTok video](https://www.tiktok.com/@iatronn/video/7549624947119754502) demonstrating how to use **Google Gemini** and **Runway AI** to replace a person in a video with another person.
   - The process involves taking a screenshot of the video, using **Google Gemini** to create a replacement image, and then using **Runway Gen-2** to apply the image to the video, but the video must be less than thirty megs.
- **AI Video Manipulation Trend**: The poster mentioned that similar video manipulation techniques have been circulating on Instagram for about a year or more.
   - This suggests a growing trend in accessible AI-powered video editing tools and techniques.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1419681711624818688)** (34 messages🔥): 

> `Text-davinci-003 origin story, ChatGPT model finetuning, UChicago ML research community, GPT-3.5 series` 


- **Text-davinci-003 launched same day as ChatGPT**: **Text-davinci-003** was made available in the API on the same day as the **ChatGPT** announcement, and OpenAI stated it was a **GPT-3.5 model** within 48 hours of its release.
   - One member recalled that it had been released the day before **ChatGPT**, alongside an updated **Codex model** also considered part of the **GPT 3.5 series**, called **code-davinci-002**.
- **ChatGPT Fine-tuned from GPT-3.5 series**: Members discussed that the first **ChatGPT** version was **fine-tuned** from a model in the **GPT-3.5-series**, with the most likely candidate being **text-davinci-003**.
   - One member stated, *"OAI confirms here that the ChatGPT model is fine-tuned from a model that finished training in early 2022, so that can’t be GPT-4 since we know GPT-4 hadn’t even started pretraining until mid/late 2022."*
- **Community Seeking UChicago ML Research Access**: A member sought introductions to the **ML research community at UChicago** to attend public seminars/lab meetings for ML sys, cog neuro, mech interp, and linguistics research discussions.
   - Another suggested attending research colloquia, citing advice from their *"80 year old phd grandpa"* and suggesting that one might not need to ask for permission.
- **GPT-3.5 Family Feud**: Members debated the lineage and release order of models in the **GPT-3.5** series, including **code-davinci-002** and **text-davinci-002** and **text-davinci-003**.
   - One insider claimed **ChatGPT** was a finetune of an unreleased model, derived from chain finetuning.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1419807715957407835)** (9 messages🔥): 

> `Prefilling vs Decoding Intuition, Diffusion ODE Solver, WACV 2025 Submission` 


- **Prefilling vs Decoding in Training**: A member inquired whether the intuition that pre-training is entirely prefilling, and the post-training / RL stage is mostly decoding, is correct.
   - The member was trying to think through the advantages of techniques that increase decoding throughput but don't help during prefilling, and thinking if post training is mostly rollouts.
- **8-Step Diffusion ODE Solver Beats DPM++**: An independent researcher developed a novel **ODE solver** for diffusion models that achieves **8-step inference** that beats **DPM++2m's 20-step inference** in **FID scores** with ~60% reduction in computational cost.
   - The new ODE solver design better traces the probability flow trajectory during inference, and is a pure training free improvement to the sampling process, outlined in a paper submitted to [WACV 2025](https://zenodo.org/records/17180452) with code at [GitHub](https://github.com/TheLovesOfLadyPurple/Hyperparameter-is-all-you-need).
- **Human touch is behind research Paper**: A member inquired how much of a research paper was LLM-generated, referring to its unusual abstract length.
   - The independent researcher stated that *the code and the paper do not involve any things generated by AI*.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1419764013318930442)** (2 messages): 

> `MMLU pro benchmark, lm-eval` 


- **Benchmarking MMLU Subtasks**: Members discussed whether it's possible to benchmark a subtask of **MMLU pro** with **lm-eval**, such as just **mmlu_law**.
   - No definitive answer or method was provided in the discussion, but it implies it's a feature worth exploring within the **lm-eval** framework.
- **lm-eval potential features**: A potential feature of **lm-eval** would be to benchmark specific subtasks.
   - This would allow more precise testing of AI model capabilities.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1419648283927642133)** (34 messages🔥): 

> `HuggingFace comments, OLMo-3 safetensors, Qwen3 Omni, Realtime Perceptual AI, SVG coding among various LLMs` 


- **HuggingFace comments spark laughter**: Members found [the comments on HuggingFace](https://huggingface.co) very funny.
   - One member stated *yeh i was lmaoing when i saw that*.
- **OLMo-3 safetensors search begins**: A member is seeking leads on **OLMo-3 safetensors**, joining their Discord to follow up.
   - They noted the inference code was published, but *no weights on HF yet*.
- **Qwen3 Omni launched**: [Qwen3 Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) is now available, featuring **36B parameters** with multimodal encoders/decoders.
   - According to one member, it *beats Whisper v2/3 and Voxtral 3B* on scam call audio processing, supporting 17 audio input languages and 10 output languages.
- **Realtime Perceptual AI race heats up**: A member anticipates the next race will be **realtime perceptual AI** (audio, video, other modalities at the same time).
   - They cited Apple's realtime vision model as an example of what's brewing behind the curtains, and is curious about the lack of releases.
- **SVG Coding LLM results**: A member tested **SVG coding** among various LLMs and shared the results with an image of a duck swimming in a pond.
   - This discussion touches on **multi-modal**, **world models**, **visual language action** and **biological neural network** as new hipster lingos, as LLM fixation with just language and mathematics centric approach may just not cut it alone.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1419642734674706433)** (7 messages): 

> `LLM Training, LoRA training, Consumer hardware for LLMs, Bandwidth and Latency for LLMs` 


- **LoRA training feasible on consumer hardware**: A member mentioned that you can train a **LoRA** on whatever you want with **14B** models locally with a single **RTX 4090**.
- **Bandwidth & Latency constraints during LLM work**: A member stated that they are working on something similar, and the constraints they're facing are **bandwidth** and **latency**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1419688244106494073)** (19 messages🔥): 

> `Navigator Mode in Aider Forks, aider-ce Package, Augment CLI, Deepseek V3.1 Setup, Web Search Tools in Aider` 


- ****Aider Forks** Gain Navigator Mode**: Members are encouraged to use [aider forks](https://github.com/dwash96/aider-ce/tree/main) with **navigator mode** for a more automated experience.
   - One member suggested using *uv pip install* on fork repos or the **aider-ce** package, which includes MCP and navigator mode.
- ****Augment** CLI Excels with Large Codebases**: According to one member, the **Augment CLI** tool shines particularly when used with large codebases.
   - They recommend **Claude Code (CC) with Opus/Sonnet** and **Codex for GPT-5**, noting that Codex lacks an API key.
- **Deepseek V3.1 Configuration Tips Requested**: A new user is planning to use **Deepseek V3.1** and seeks advice on initial setup and configuration.
   - The user also inquired about preferred **web search tools**, noting that aider doesn't have a built-in tool but can scrape provided URLs; one member suggested using **Perplexity MCP for web search** with MCP integration.
- **GAS Scripting automation causes job losses?**: One member created a **GAS** (Google App Script) that responds to quotes for their business.
   - They said it performed better than a human employee, raising concerns that *if this was public, it would cause massive job losses*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1419721578064646325)** (5 messages): 

> `Running multiple aider agents, aider asks to edit files, LLM confusion with prompt files` 


- **Multi-Agent Aider Setup Discussed**: A member inquired about setting up **multiple aider agents** via command line for external orchestration, rather than a built-in solution.
   - The discussion suggested using **git worktrees** for concurrent modifications to the same repository.
- **Aider's Edit Prompts Cause LLM Confusion**: When using a file to set up the prompt, **aider prompts to edit files** and, simultaneously, the **LLM gets confused** about the task at hand.
   - The LLM then asks for clarification of intent, specifically whether it should act as the 'User' within the APM framework or modify a file using the SEARCH/REPLACE block format.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1419679454858317886)** (15 messages🔥): 

> `FFI, Rust, C ABI, C header, Mojo binding generators` 


- **Rustaceans invoke Mojo via FFI**: Members discussed using **FFI** to call **Mojo** from **Rust**, with one user confirming they have done so via the **C ABI**.
   - They added that it's similar to calling **C**, and to ensure types match up, and to mark the Mojo function with `@export(ABI="C")`.
- **Manual Mojo Binding Required for now**: A member stated that **C header -> Mojo binding generators** are still **WIP**, so **CXX** doesn't help.
   - The only way to generate Mojo bindings is doing it manually.
- **Windows Support still nonexistent**: One user asked about **Windows support** for Mojo.
   - Another responded that it is *not coming any time soon*.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1419785878183739542)** (1 messages): 

> `Modular Platform 25.6, NVIDIA Blackwell, AMD MI355X, Consumer GPUs` 


- **Modular 25.6 Fuels GPU Performance**: Modular Platform **25.6** is live, delivering peak performance across the latest GPUs including **NVIDIA Blackwell (B200)** and **AMD MI355X**.
   - Early results show **MAX** on **MI355X** can even outperform **vLLM** on **Blackwell**, with unified GPU programming now available for consumer GPUs like **Apple Silicon, AMD,** and **NVIDIA**.
- **Modular Unifies GPUs**: Modular now supports **Apple Silicon, AMD and NVIDIA** consumer GPUs, bringing unified GPU programming to laptops and desktops.
   - Check out the [blog post](https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple) for more information.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1419657830159351909)** (7 messages): 

> `Mojo MAX .mojopkg requirement, Mojo nightly install command, Variadic args binding in Mojo` 


- **MAX Requires .mojopkg**: Using **MAX** requires a `.mojopkg` file alongside the executable, which contains the highest level **MLIR** that **Mojo** can produce after parsing, for the runtime's **JIT** compiler.
   - For platforms hiding hardware details (Apple, NVIDIA GPUs), **Mojo** hands off compilation to the driver, performing a *one-shot JIT* without profiling, unlike V8 or Java.
- **Easy Mojo Installs via PIP**: The `mojo` packages are now on **PyPI**, allowing users to install the latest stable version of **Mojo**, debugger, and LSP using `pip install mojo`.
   - For nightly builds, use `pip install mojo --pre "mojo<1.0.0" --index-url https://dl.modular.com/public/nightly/python/simple/`.
- **Variadic arguments binding blocked by Rust**: Variadic args binding in **Mojo** might not be possible due to the `va_list` not being part of the function signature in C.
   - Even **Rust** doesn't currently fully support this feature ([issue 44930](https://github.com/rust-lang/rust/issues/44930)), making it a challenging aspect to implement in `c_binder_mojo`.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1419735529339228190)** (16 messages🔥): 

> `New Paper Discussion, Yann LeCun's Work, Joint Embedding Predictive Architecture, Paper Presentation Opportunity` 


- **Paper Chat Scheduled**: A chat about a paper ([https://www.arxiv.org/abs/2509.14252](https://www.arxiv.org/abs/2509.14252)) was scheduled for <t:1758567600:t>.
   - A member suggested an earlier session for those in the eastern timezone.
- **Discussing Yann LeCun's Impact**: A member inquired whether the scheduled paper was related to **Yann LeCun's** work.
   - Another member agreed and suggested looking at the background on **Joint Embedding Predictive Architecture (JEPA)**, linking to an [OpenReview document](https://openreview.net/pdf?id=BZ5a1r-kVsf).
- **Links Galore!**: Several links were shared, including a [YouTube video](https://youtu.be/jSdHmImyUjk?si=HdFl4PZFe2nvqSi5), a [tweet](https://x.com/BlancheMinerva/status/1969974401081725214), another [tweet](https://fxtwitter.com/randall_balestr/status/1969987315750584617), an [arXiv paper](https://arxiv.org/abs/2501.00656), another [YouTube video](https://www.youtube.com/watch?v=18Fn2m99X1k), and two more [arXiv papers](https://arxiv.org/abs/2507.02092) and ([https://arxiv.org/abs/2509.13805](https://arxiv.org/abs/2509.13805)).
   - These resources likely relate to the discussed paper and related topics.
- **Paper Presentation Invitation**: A member inquired about presenting a paper and inquired about the appropriate timing.
   - The member suggested doing it **6 hours earlier or later** on most days.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1419814468652761189)** (3 messages): 

> `GPT parsing philosophy, Prompting improvements` 


- **GPT's Philosophical Parsing Gets Mixed Reviews**: A user rated a **GPT model** a **7/10** for parsing their philosophy, but only a **4/10** for expanding on it, and **3/10** for formatting.
   - They added that it's better than most models, but *not good enough*.
- **User experiences Improved Results**: The user initially disliked the **GPT** results, but then reported liking the results after further interactions.
   - The user said they are either *getting better at prompting or the machine is better at reading*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1419683350180008067)** (16 messages🔥): 

> `MCP Secrets, Trace IDs, GEPA for ReAct` 


- **MCP Secrets should be stored in ENV**: A user cautioned against passing secrets through **MCP**, advising to use **environment variables** for single-user services or a **login+vault/OAuth** for more complex setups.
   - The user noted that their **JIRA MCP server** implementation uses stdio and retrieves credentials from the environment, deploying per single-user to avoid leaking secrets via process tables.
- **Trace IDs in DSPy modules affect optimization**: A user asked about handling per-item **trace IDs** in **DSPy modules** without breaking optimization, noting that initializing modules with a trace ID results in the same ID being used for all documents when running a batch after optimizing.
   - The user considered recreating modules per article (too expensive) and moving the trace ID to forward calls, asking if this would affect optimization because the trace_id gets passed to llm gateway for logs and auditing.
- **GEPA for ReAct Agents**: A user inquired about experiences with **GEPA** for **ReAct Agents**, specifically regarding how **context overflows** would be handled with long agent trajectories.
   - No war stories were shared.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1419729645607190639)** (7 messages): 

> `MCP Sampling Protocol, response_schema addition, Claude models constrained output` 


- **MCP Sampling Protocol addition has support**: A member saw an [issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1030) to *Add **response_schema** to the MCP Sampling Protocol* that was converted into a discussion.
   - Another member would be happy to try and do an implementation of this (in October) with a small demo as part of a SEP, because it would be really neat to be able to request **structured output** when using sampling.
- **response_schema addition sounds feasible**: The group discussed that it is not difficult to implement in the SDK - the work is in the integration between the SDK and LLM API / SDK.
   - One member stated that *a bit of provider specific code might be needed, but nothing bad*.
- **Claude models have constrained output**: It was mentioned that presenting **constrained output** could be presented as a facet of the client capability.
   - A member stated that *Claude models show up as not supporting this feature even though they do* and they believe *it's better to present **response_schema** support, and then the client host decides how to actually provide it.*


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1419672663160258711)** (6 messages): 

> `Model Context Protocol Registry, Publishing MCP Servers, Remote Server Configurations, MCP Install Instructions` 


- **MCP Registry Publish Instructions Emerge**: A member shared instructions for publishing servers to the **Model Context Protocol (MCP)** Registry, starting with [this guide](https://github.com/modelcontextprotocol/registry/blob/main/docs/guides/publishing/publish-server.md).
   - The guide details creating an empty **GitHub** repo and providing a `server.json` file with details like name, description, status, repository URL, and remote endpoint.
- **Remote Server Configs for MCP**: A member linked to remote server configurations for the **Model Context Protocol (MCP)**, with [this reference](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/generic-server-json.md#remote-server-example).
   - The example `server.json` includes a `$schema` definition and a `remotes` array specifying the `streamable-http` type and URL.
- **MCP Install Instructions Generated!**: A member mentioned using a tool to generate a readme file with instructions for installing the **Model Context Protocol (MCP)** in various clients via [MCP Install Instructions](https://mcp-install-instructions.alpic.cloud/).
   - He said the tool was *"pretty cool"* and helpful in creating installation guides.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1419663465752301569)** (6 messages): 

> `CuTe DSL, RANGEIFY status, Company update, ThunderKittens project` 


- **Colfax Meeting on Monday**: There will be a [meeting #89](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/meeting) on Monday at 9am San Diego time for a company update.
   - Starting next week, the meeting will be **3 hours earlier**.
- **RANGEIFY Progress Questioned**: There was a question about whether *RANGEIFY* would be default by the end of the week, including store, assign, group reduce, disk, jit, const folding, buf limit, llm, and openpilot.
   - It was noted that **children** are *not making progress* and **image** is not complete.
- **CuTe DSL Potential Gamechanger**: Members mentioned that the **CuTe DSL** is a *potential gamechanger*.
   - They added that **ThunderKittens** is *nicestarting*.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1419660499058163722)** (6 messages): 

> `Kimi K-2, Prompt Injection, Claude's Lobotomization` 


- **Kimi Rejects Provocation Attempts**: One member shared an experience of *provoking* **Kimi**, appreciating its rejection of blind obedience or sympathy.
   - Another member expressed a similar sentiment, noting that **Kimi's attitude** is a reason why it's their favorite model.
- **Claude's Prompt Injection Debacle**: Members discuss how **Claude** has been weakened due to [prompt injection techniques](https://en.wikipedia.org/wiki/Prompt_injection), leading it to disagree when context exceeds a certain length.
   - They noted that it is unlike **Kimi K2**, and some express disappointment in the changes to **Claude**.

