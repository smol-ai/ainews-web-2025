---
id: MjAyNS0w
title: not much happened today
date: '2025-09-04T05:44:39.731046Z'
description: >-
  **Google DeepMind** released **EmbeddingGemma (308M)**, a small multilingual
  embedding model optimized for on-device retrieval-augmented generation and
  semantic search, supporting over 100 languages and running efficiently with
  quantization and EdgeTPU latency under 15ms. **Jina AI** introduced new
  code-focused embedding models (0.5B/1.5B) with GGUF quantization, achieving
  state-of-the-art retrieval across multiple languages and tasks. **LightOn**
  demonstrated large-scale retrieval training without distillation using
  contrastive training on billions of passages. **Hugging Face** released the
  **FineVision** dataset with 17.3M images and 9.5B answer tokens for
  vision-language model training, showing significant benchmark improvements.
  The **MiniCPM-V 4.5 (8B)** multimodal model reported surpassing **GPT-4o** and
  **Gemini-2.0 Pro** on OpenCompass benchmarks with innovative video token
  compression. Microsoft’s **VibeVoice TTS** and Stanford’s Mixture-of-Contexts
  video generation also featured. Additionally, a Stanford study benchmarked
  optimizers like Muon, Soap, Mars, and Sophia, finding diminishing speedups
  over AdamW at larger scales but advantages at smaller scales. The new ChatGPT
  branching feature was noted for its simplicity and popularity. *"Everyone's a
  decacorn now."*
companies:
  - google-deepmind
  - hugging-face
  - jina-ai
  - lighton
  - microsoft
  - stanford
  - openai
  - ollama
  - weaviate
  - langchain
  - llamaindex
models:
  - embeddinggemma
  - qwen-2.5-coder
  - minicpm-v-4.5
  - gpt-4o
  - gemini-2.0-pro
topics:
  - embeddings
  - retrieval-augmented-generation
  - quantization
  - multilingual-models
  - on-device-ai
  - semantic-search
  - contrastive-learning
  - dataset-release
  - vision
  - multimodality
  - video-generation
  - text-to-speech
  - optimizer-benchmarking
  - training-recipes
  - model-compression
  - video-token-compression
  - fine-tuning
people:
  - osanseviero
  - _philschmid
  - tomaarsen
  - ollama
  - weaviate_io
  - lusxvr
  - andimarafioti
  - thibaudfrere
  - _akhaliq
  - clementdelangue
  - gordonwetzstein
  - konstmish
  - wen_kaiyue
  - percyliang
---


**everyone's a decacorn now.**

> AI News for 9/4/2025-9/5/2025. We checked 12 subreddits, 544 Twitters and 22 Discords (186 channels, and 4350 messages) for you. Estimated reading time saved (at 200wpm): 324 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

[Congrats to Sierra](https://x.com/SierraPlatform/status/1963654362384724388) on becoming the latest ~~Decagon~~ I mean, Decacorn.

Also [the new ChatGPT branching feature](https://x.com/OpenAI/status/1963697012014215181) was remarkably popular for the probable ~100 LOC it took to implement it (with the Responses API)

---

# AI Twitter Recap

**Embeddings on-device and retrieval stack updates**

- **Google’s EmbeddingGemma (308M) goes wide**: Google/DeepMind released a small, multilingual embedding model designed for on‑device RAG and semantic search. Highlights: 308M params, top-ranked open model under 500M on MTEB, trained on 100+ languages, runs in <200MB RAM with quantization, supports Matryoshka embeddings (output dims 768→128), 2k context, and EdgeTPU latency <15ms in some settings. Immediate ecosystem support across Hugging Face Sentence Transformers, Ollama, MLX, llama.cpp, LlamaIndex, LangChain, Weaviate, Cloudflare Workers, etc. Launch details and getting started: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1963635422698856705), [@osanseviero](https://twitter.com/osanseviero/status/1963635281032040914), [@_philschmid](https://twitter.com/_philschmid/status/1963634786636841461), [@tomaarsen](https://twitter.com/tomaarsen/status/1963639557653422304), [@ollama](https://twitter.com/ollama/status/1963667967184617703), [@weaviate_io](https://twitter.com/weaviate_io/status/1963683200368304613), [@TheTuringPost](https://twitter.com/TheTuringPost/status/1963666849364836606).
- **Jina code embeddings (0.5B/1.5B) + GGUF**: New code-focused embedding models (with 1–4bit GGUF quantizations) claim SOTA retrieval across 15+ languages and 5 tasks (nl2code, code2code, code2nl, code2completions, QA). Built on a strong code LLM base (e.g., Qwen2.5‑Coder pretraining on 5.5T tokens, 92+ languages), then contrastively tuned for retrieval with limited aligned pairs. Links and models: [@JinaAI_](https://twitter.com/JinaAI_/status/1963637135439007824), [details](https://twitter.com/JinaAI_/status/1963637139037720995), [models](https://twitter.com/JinaAI_/status/1963637141675843791).
- **Large‑scale retrieval training without distillation**: LightOn’s PyLate shows direct contrastive training on billions of passages using GradCache + distributed infra, reporting improved generalization on BEIR/BRIGHT without teacher models. Overview: [@LightOnIO](https://twitter.com/LightOnIO/status/1963620040604787136).

**Vision-language data and multimodal models**

- **FineVision dataset (Hugging Face)**: A major open dataset release for VLM training: 17.3M images, 24.3M samples, 88.9M turns, 9.5B answer tokens across 200+ curated sources. The team reports >20% average gains across 10 benchmarks and added capabilities (GUI navigation, pointing, counting). Announcement and technical article: [@lusxvr](https://twitter.com/lusxvr/status/1963609337546293448), [@andimarafioti](https://twitter.com/andimarafioti/status/1963610118165000479), [@thibaudfrere](https://twitter.com/thibaudfrere/status/1963627540544647177).
- **MiniCPM‑V 4.5 (8B) video/image VLM**: Reports 77.0 average on OpenCompass across 8 benchmarks with an 8B model, claiming to surpass GPT‑4o‑latest and Gemini‑2.0 Pro on their setup. Introduces a unified 3D‑Resampler and aggressive video token compression (96×): 6×448×448 frames → 64 video tokens (vs ~1,536 in many MLLMs). Demos and Space: [@_akhaliq](https://twitter.com/_akhaliq/status/1963587749400727980), [@OpenBMB](https://twitter.com/OpenBMB/status/1963623940028563910).
- Also notable: Microsoft’s VibeVoice TTS uses continuous speech tokenizers at 7.5 Hz for expressive, long-form multi-speaker audio [@ClementDelangue](https://twitter.com/ClementDelangue/status/1963537036616323388); Stanford’s Mixture‑of‑Contexts demonstrates minute‑long video generation in a single pass [@GordonWetzstein](https://twitter.com/GordonWetzstein/status/1963583050744250879).

**Optimizers, internal metrics, and training recipes**

- **Robust optimizer benchmarking (Marin project)**: Two papers (and a comprehensive Stanford study) compare Muon, Soap, Mars, Sophia, ScheduleFree, AdEMAMix, Prodigy, etc., across model scales (0.1B–1.2B), batch sizes, and schedulers. Consensus emerging: with careful tuning and at larger scales, speedups over AdamW diminish (~10% at ~1.2B), though matrix-based methods can lead at smaller scales. Threads: [@konstmish](https://twitter.com/konstmish/status/1963535545721917725), [@wen_kaiyue](https://twitter.com/wen_kaiyue/status/1963633867140526319), [@percyliang](https://twitter.com/percyliang/status/1963648131394122222), commentary from [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1963679442859106480) and [@jeremyphoward](https://twitter.com/jeremyphoward/status/1963689424782565384).
- **“Internal metrics” in large‑scale training (Kimi/K2)**: Practitioners emphasize monitoring internal signals (loss, grad norm, output RMS, max logit) to diagnose instability and ensure headroom. MuonClip was designed to control max logit to avoid training breakdowns. Summaries and translations: [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1963493293679153349), [@crystalsssup](https://twitter.com/crystalsssup/status/1963547955799224386).
- **Creative‑writing finetune of Qwen3‑32B**: “Zhi‑Create‑Qwen3‑32B” reports a WritingBench score of 82.08 vs 78.97 base, using: (1) SFT with curriculum (length/reasoning‑grouped, progressive difficulty, targeted re‑training) and (2) DPO with RAFT (rule filters + LLM‑judge) to address CN‑EN code‑switching, repetition, and reasoning. Data included filtered open sets (e.g., Dolphin‑r1, DeepSeek distills), Zhihu Q&A, and CoT traces; all passed a reward model filter. Usage tips include temperature ~0.6 and optional think‑trigger strings. Details: [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1963441300692402659).
- Infra note: slime RL framework reports cutting Qwen3‑30B‑A3B weight update time from 60s → 7s, and handling GLM‑4.5‑355B‑A32B FP8 updates at ~100s, with ongoing async/zero‑redundancy optimizations. Call for collab: [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1963532501336695282).

**Agent systems, runtimes, and tooling**

- **LangGraph design deep dive**: A thorough post on building production‑grade agent runtimes: minimal abstractions, structured execution/state, recovery/durability, and control surfaces that match real ops needs. A must‑read for teams shipping agents to prod: [@LangChainAI](https://twitter.com/LangChainAI/status/1963646974315606428), [@hwchase17](https://twitter.com/hwchase17/status/1963647954587455568), [@nfcampos](https://twitter.com/nfcampos/status/1963652967443435723).
- **UI‑TARS‑2 (multi‑turn agent RL for native UIs)**: Unified GUI/phone/browser/terminal/tool‑use agent shows benchmarks across OSWorld 47.5, WindowsAgentArena 50.6, AndroidWorld 73.3, Online‑Mind2Web 88.2%, SWE‑Bench 68.7, TerminalBench 45.3; supports hybrid action flows combining clicks, terminal, and API calls. Paper + demo: [@TsingYoga](https://twitter.com/TsingYoga/status/1963629621326614940).
- **Agent failure analysis**: Atla launched a platform to automatically discover recurring failure patterns and propose targeted fixes for agent systems [@Atla_AI](https://twitter.com/Atla_AI/status/1963586200305836264). Separately, AgenTracer‑8B diagnoses multi‑agent interaction errors and reports up to 18.18% gains over proprietary baselines in its setting [@omarsar0](https://twitter.com/omarsar0/status/1963618829680218254), [paper](https://twitter.com/omarsar0/status/1963618846532931663).
- **Infra updates**: Groq’s Compound (agentic system) is GA after 5M+ requests [@GroqInc](https://twitter.com/GroqInc/status/1963635205899710798). Gradio can now deploy MCP servers to Google Cloud via a single command [@Gradio](https://twitter.com/Gradio/status/1963636954999754955). HF MCP server added OpenAI Codex CLI support [@reach_vb](https://twitter.com/reach_vb/status/1963599978909008321). Together AI added an EU GPU region (Sweden) for lower latency/data residency [@togethercompute](https://twitter.com/togethercompute/status/1963498998720872686). SkyPilot showcases moving from SLURM to multi‑cloud for faster cycles with K8s‑grade reliability [@skypilot_org](https://twitter.com/skypilot_org/status/1963637217055646139).

**Product rollouts and ecosystem**

- **Perplexity Comet**: Broad rollout continues—“more than a million” users got access in one push; mobile pre‑orders live; new iOS app build streams tables/markdown/intermediate steps smoothly [@AravSrinivas](https://twitter.com/AravSrinivas/status/1963633205351010795), [pre‑orders](https://twitter.com/AravSrinivas/status/1963620578344276366), [iOS update](https://twitter.com/AravSrinivas/status/1963758210281882029), [availability note](https://twitter.com/perplexity_ai/status/1963638853975040456).
- **ChatGPT conversation branching**: OpenAI shipped native branch‑and‑explore for chats, a long‑requested UX upgrade for exploratory workflows [@OpenAI](https://twitter.com/OpenAI/status/1963697012014215181), [@gdb](https://twitter.com/gdb/status/1963780952187965746).
- Research note: DeepMind’s Deep Loop Shaping (published in Science) improves LIGO interferometer control, cutting noise 30–100× on hardware and eliminating LIGO’s most unstable loop as a meaningful noise source—an example of AI advancing experimental physics [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1963664018515849285), [results](https://twitter.com/GoogleDeepMind/status/1963664045216579999), [@sundarpichai](https://twitter.com/sundarpichai/status/1963668228481159371).

**Top tweets (by engagement)**

- [Ilya Sutskever: “a revolutionary breakthrough if i’ve ever seen one”](https://twitter.com/ilyasut/status/1963627458244350015) — 19.2k
- [Alibaba Qwen: “Ready to meet the biggest, brainiest guy in the Qwen3 family?”](https://twitter.com/Alibaba_Qwen/status/1963586344355053865) — 5.5k
- [OpenAI: “By popular request: you can now branch conversations in ChatGPT”](https://twitter.com/OpenAI/status/1963697012014215181) — 17.1k
- [Google Gemini App: no‑prompt nano‑banana templates for multi‑image generation](https://twitter.com/GeminiApp/status/1963615829708132611) — 1.7k
- [Andrew Ng: “There is significant unmet demand for developers who understand AI…”](https://twitter.com/AndrewYNg/status/1963631698987684272) — 1.8k
- [Perplexity (Arav): “More than a million people got Comet access this morning.”](https://twitter.com/AravSrinivas/status/1963633205351010795) — 1.0k
- [DeepMind: EmbeddingGemma launch](https://twitter.com/GoogleDeepMind/status/1963635422698856705) — 1.2k

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Microsoft VibeVoice Repo Takedown & ComfyUI Integration

- [**VibeVoice RIP? What do you think?**](https://i.redd.it/un6uilkoh2nf1.png) ([Score: 200, Comments: 75](https://www.reddit.com/r/LocalLLaMA/comments/1n7zk45/vibevoice_rip_what_do_you_think/)): **OP reports that Microsoft abruptly deleted the official VibeVoice GitHub repo and removed the VibeVoice-Large and VibeVoice-Large-Preview models from Hugging Face; mirrors still exist on ModelScope. They maintain ComfyUI integration nodes ([Enemyx-net/VibeVoice-ComfyUI](https://github.com/Enemyx-net/VibeVoice-ComfyUI)) and shipped v**`1.0.9` **embedding VibeVoice directly to avoid the now-missing upstream dependency; the project was under MIT licensing, implying redistribution is likely permitted. Reason for removal is unknown; the work appears tied to Microsoft’s Asia research lab.** Comments note that an MIT license allows community re-uploads (e.g., to Hugging Face) and urge backing up assets to prevent loss. Others speculate this follows a pattern of projects from Microsoft Asia labs being pulled, possibly due to team changes or departures.
    - Licensing implications: commenters note the project is under the **MIT License**, which grants broad, irrevocable rights to use, copy, modify, and redistribute existing releases. This means mirrors on platforms like [Hugging Face](https://huggingface.co/) are legally permissible for the already-published version, and any later license changes can’t retroactively restrict those artifacts ([MIT text](https://opensource.org/license/mit/)). Practical advice: back up both weights and code to avoid loss from upstream takedowns.
    - Anticipated re-release changes: if a takedown precedes an updated release, users expect increased safety filters/“censorship” or tighter usage restrictions (e.g., gated downloads, stricter AUP, or embedded refusal policies). This can reduce capability in some domains (higher refusal rates, constrained prompts), so backing up the original checkpoint preserves an unconstrained baseline for evaluation and downstream finetuning.
    - Precedent and resilience: commenters compare this to prior incidents (e.g., WizardLM/Wizard 2) where strong checkpoints were released, later pulled/restricted, yet community mirrors persisted and usage continued. The technical takeaway is to prioritize open-weight availability to decouple research and deployments from upstream product or policy reversals ([WizardLM repo for context](https://github.com/nlpxucan/WizardLM)).
- [**Did M$ take down VibeVoice repo??**](https://i.redd.it/vsnyimd3e2nf1.png) ([Score: 180, Comments: 36](https://www.reddit.com/r/LocalLLaMA/comments/1n7z5kl/did_m_take_down_vibevoice_repo/)): **The post flags that the official Microsoft VibeVoice GitHub repo ([microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)) now returns a 404, and commenters note the associated Hugging Face models (VibeVoice-Large and VibeVoice-Large-Preview) were also pulled. Community mirrors and tooling still exist: a ComfyUI node implementation is at https://github.com/Enemyx-net/VibeVoice-ComfyUI, and model files can still be fetched from ModelScope: https://modelscope.cn/models/microsoft/VibeVoice-Large/files. Existing local installs continue to function; the takedown reason is unknown and may be temporary, with concerns about potential license changes.** Comments speculate it was "too good" and urge downloading mirrors for posterity, while others ask for copies and advise caution about redistributing until Microsoft’s intent and licensing are clarified.
    - Microsoft’s official VibeVoice GitHub repository was suddenly removed, and the Hugging Face entries for `VibeVoice-Large` and `VibeVoice-Large-Preview` were also taken down; the `VibeVoice-Large` weights remain mirrored on ModelScope: https://modelscope.cn/models/microsoft/VibeVoice-Large/files. The reason for the takedown is unknown, raising concerns about potential licensing changes that could affect redistribution or embedding of the code/weights.
    - Operationally, existing setups continue to work because inference only requires local weights: *“You don’t need the original MS repo. As long as you have the weights you can use them in Comfy.”* ComfyUI integration via the community nodes at https://github.com/Enemyx-net/VibeVoice-ComfyUI remains functional, so pipelines that already reference local checkpoints are unaffected.
    - Not all variants are gone: commenters note the `1.5` model is still on Hugging Face, while the Large model is retrievable from ModelScope. Practically, users aiming for reproducibility are downloading and pinning the remaining artifacts now to avoid future link rot while the status and licensing are clarified.

### 2. EmbeddingGemma 300M Launch + HF Science AMA/FineVision

- [**EmbeddingGemma - 300M parameter, state-of-the-art for its size, open embedding model from Google**](https://www.reddit.com/r/LocalLLaMA/comments/1n8egxb/embeddinggemma_300m_parameter_stateoftheart_for/) ([Score: 197, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1n8egxb/embeddinggemma_300m_parameter_stateoftheart_for/)): **Google released EmbeddingGemma, a** `300M`**‑parameter, text‑only multilingual embedding model (trained on 100+ languages) producing** `768`**‑dim vectors, with smaller dimensions available via multi‑resolution learning (MRL). Weights are on Hugging Face ([google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)), deployable via Ollama ([library/embeddinggemma](https://ollama.com/library/embeddinggemma)), and the launch write‑up provides English and multilingual evaluations claiming state‑of‑the‑art performance for its size ([HF blog](https://huggingface.co/blog/embeddinggemma)); community GGUF builds (Q4_0, Q8_0, BF16) are consolidated for local inference at [unsloth/embeddinggemma-300m-GGUF](https://huggingface.co/unsloth/embeddinggemma-300m-GGUF). License: Gemma.** Commenters point to the HF blog’s comparison tables for task‑level tradeoffs and discuss whether to prefer `nomic-embed-text:v1.5` vs EmbeddingGemma, noting the choice likely depends on use case (monolingual vs multilingual coverage, latency/quantization needs, and dimensionality). RAG finetuning and baseline RAG notebooks are forthcoming from the community.
    - Deployment/quantization: A community GGUF release bundles `Q4_0`, `Q8_0`, and `BF16` builds of EmbeddingGemma-300M in one repo (https://huggingface.co/unsloth/embeddinggemma-300m-GGUF), easing llama.cpp/local use; `Q4_0` minimizes RAM, `Q8_0` trades size for accuracy/latency, and `BF16` preserves precision for highest quality. The maintainer also plans RAG finetuning + baseline notebooks to evaluate retrieval quality end-to-end.
    - Benchmarks: **Google/Hugging Face** provide side-by-side English and multilingual evaluations in the official blog (https://huggingface.co/blog/embeddinggemma), letting you inspect task-level performance (e.g., retrieval/classification) to validate the “state-of-the-art for its size” claim. The linked charts enable apples-to-apples comparisons against other open embeddings across datasets, which is essential for model selection.
    - Comparatives: One practitioner reports EmbeddingGemma-300M is *“a fair bit worse than qwen 3 0.6b embedding”*, highlighting a likely trade-off between size (`~300M` params) and absolute accuracy vs larger (`~600M`) models. Another asks about `nomic-embed-text:v1.5`; the practical guidance is to choose based on target languages/domains and the blog’s per-dataset scores rather than only headline averages.
- [**AMA with Hugging Face Science, the team behind SmolLM, SmolVLM, Fineweb and more.**](https://www.reddit.com/r/LocalLLaMA/comments/1n8c3l2/ama_with_hugging_face_science_the_team_behind/) ([Score: 194, Comments: 414](https://www.reddit.com/r/LocalLLaMA/comments/1n8c3l2/ama_with_hugging_face_science_the_team_behind/)): **Hugging Face Science announced a time‑boxed AMA (8–11 AM PST with 24h follow‑ups) featuring researchers behind SmolLM, SmolVLM, and FineWeb, alongside the release of a new multimodal dataset, FineVision (see dataset card: https://huggingface.co/datasets/HuggingFaceM4/FineVision). Reference links: org page https://hf.co/science and learning resources https://hf.co/learn. Participants span model pretraining (e.g., SmolLM/Nanotron), post‑training/alignments, evaluation, multimodal (VLM), data, Transformers.js, and llama.cpp integration.** Commenters asked about counterintuitive design choices and surprises during SmolLM’s development, signaling interest in training/architecture decisions; ecosystem contributors (e.g., Unsloth) chimed in with support.
    - A commenter asks about the biggest surprises during **SmolLM**’s development—counterintuitive design choices that ultimately worked. Technical angles include tokenizer/vocab size vs parameter-count trade-offs, context length vs compute budget, data curation via **FineWeb/FineWeb-Edu** and curriculum, optimizer/regularization choices (AdamW/Lion, weight decay, dropout), attention/activation variants (RoPE scaling, GQA, SwiGLU), and precision/throughput decisions (bf16/fp8, FlashAttention). They’re asking for concrete ablations or metrics that show where small models benefit from *non‑obvious* settings.
    - Another thread requests how the team prioritizes next projects. Criteria likely include gaps on public benchmarks (MMLU, GSM8K, MT-Bench), readiness of data pipelines like **FineWeb** for new modalities, compute/latency constraints for deployment (quantization, KV-cache, attention scaling), and reproducibility vs training cost. The ask implies a decision framework with milestone metrics and resource allocation across **SmolLM**, **SmolVLM**, and dataset tooling.
    - A user asks whether there are plans to train and release larger `30B+` models. Salient constraints include compute budget, dataset scale/quality, dense vs MoE trade-offs, training stack (FSDP/ZeRO, activation checkpointing), inference cost (memory bandwidth, parallelism), and evaluation needed to justify scaling vs continuing to optimize small models. They’re probing the roadmap and feasibility for scaling beyond **SmolLM/SmolVLM**.

### 3. Local AI Ops: 5070 Ti Super VRAM Rigs & Ollama Exposure PSA

- [**Finally: 3090 Successor: 5070 Ti super 24Gb 800$**](https://www.reddit.com/r/LocalLLaMA/comments/1n82ndz/finally_3090_successor_5070_ti_super_24gb_800/) ([Score: 246, Comments: 140](https://www.reddit.com/r/LocalLLaMA/comments/1n82ndz/finally_3090_successor_5070_ti_super_24gb_800/)): **Rumor/leak claims an NVIDIA “RTX 5070 Ti Super” with 24 GB VRAM at ~$800, positioned as a 3090-class successor, citing improved perf/W that could make multi‑GPU (e.g., ~100 GB total VRAM) rigs feasible without extreme power draw, and mentions support for new low‑precision “FP4” formats for AI inference. Sources include a supposed spec image and a video breakdown ([image](https://preview.redd.it/j9riehskc3nf1.jpg?width=1341&format=pjpg&auto=webp&s=fd5386a95c701b1a750a20a2b4116c93df426306), [YouTube](https://www.youtube.com/watch?v=9ii4qrzfV5w)). Commenters also speculate a $600 16 GB GDDR7 “5070” SKU and contrast it with a rumored Intel “B50” 16 GB GDDR6 card at $350, citing a claimed memory‑bandwidth gap of** `~1792 GB/s` **vs** `~224 GB/s` **(treated as leak claims, not confirmed).** Top replies are skeptical about MSRP availability (expect scalping/backorders) and timing (Q4’25 launch, broad availability slipping into 2026), but note if true it could crater used 3090 prices and undercut Intel’s B50 on bandwidth/CUDA; some expect non‑Super cards to see price cuts.
    - Bandwidth and memory debate: one commenter projects a $600 16GB GDDR7 “5070-class” versus Intel’s $350 16GB GDDR6 B50, claiming `~1792 GB/s vs ~224 GB/s` (~8×) bandwidth and citing CUDA as an ecosystem advantage. Note that `~1792 GB/s` implies a 512‑bit bus at ~28 Gbps GDDR7; a 70‑class part is more likely 192–256‑bit, yielding roughly `~672–896 GB/s` at similar speeds—still 3–4× over a 128‑bit GDDR6 part (~224 GB/s), but not 8× unless bus width is unusually large.
    - Power/TDP implications for multi‑GPU VRAM rigs: a linked spec sheet [TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5070-ti.c4243) lists the 5070 Ti at `~300W TDP`, undercutting RTX 3090’s typical `~350W` but not by a wide margin. As a result, building “100 GB VRAM” multi‑GPU setups will still draw kilowatts; the practical gain is newer warranty support plus higher per‑card VRAM/bandwidth rather than big power savings.
    - Expected generational uplift vs RTX 3090: commenters expect a 24GB “5070 Ti Super” (Blackwell 2.0) at similar power to “wipe the floor” with a 3090 due to newer architecture and faster memory. While no benchmarks are cited, the combination of 24GB VRAM and GDDR7 suggests materially higher perf/$. Against Intel’s rumored B50, CUDA availability is flagged as a decisive advantage for many workloads.
- [**PSA: Make sure your API ports aren't exposed to the open internet**](https://www.reddit.com/r/LocalLLaMA/comments/1n7uocj/psa_make_sure_your_api_ports_arent_exposed_to_the/) ([Score: 199, Comments: 55](https://www.reddit.com/r/LocalLLaMA/comments/1n7uocj/psa_make_sure_your_api_ports_arent_exposed_to_the/)): **Cisco reports roughly** `1,100` **publicly exposed Ollama REST APIs discoverable via Shodan, detailed in their case study [“Detecting Exposed LLM Servers: Shodan Case Study on Ollama”](https://blogs.cisco.com/security/detecting-exposed-llm-servers-shodan-case-study-on-ollama). They verified instances with a benign probe that may appear in logs as *“What is 2+2?”*; exposed endpoints allow unauthenticated LLM inference over the internet, implying free compute use and potential data leakage for anyone binding Ollama to** `0.0.0.0` **or publishing port (commonly** `11434`**).** Commenters debate how exposure happens in 2025: likely culprits include Docker port publishing (e.g., `p 11434:11434`), cloud security groups/firewalls permitting `0.0.0.0/0`, UPnP/NAT misconfig, or reverse proxies without auth. Another notes prior scraping efforts like the now-offline [freeleakhub.com](http://freeleakhub.com/) that indexed open Ollama servers, some serving large models (e.g., DeepSeek R1, Qwen 3), suggesting persistent hygiene gaps.
    - Prior scans like [freeleakhub.com](http://freeleakhub.com/) (now offline) reportedly cataloged numerous exposed inference servers, many hosting small models but also full deployments of **DeepSeek-R1** and **Qwen 3** with no authentication or paywall. This highlights that misconfigured endpoints remain common and trivially discoverable by public crawlers.
    - A technical question is raised about how ports get exposed “accidentally,” with speculation around router/firewall misconfiguration and containerized stacks (e.g., Ollama) being bound to `0.0.0.0` or published via permissive port mappings on hosts with public IPs. Even with consumer NAT, poor defaults or UPnP/automated port forwards can make APIs reachable from the Internet.
    - Another thread asks about placing Ollama behind a proxy to enforce API tokens and IP allowlists, implicitly noting gaps in built-in auth for self-hosted LLM APIs. The suggested mitigation path is a reverse proxy layer that adds authentication and network ACLs before the model endpoint.
- [**🤷‍♂️**](https://i.redd.it/21ivxa12b5nf1.png) ([Score: 988, Comments: 176](https://www.reddit.com/r/LocalLLaMA/comments/1n89dy9/_/)): **Ambiguous teaser image (unreadable here) with title “🤷‍♂️” prompts speculation about a very large upcoming Qwen model/tool; commenters mention wanting a “stronger Qwen CLI” that could match/surpass Claude Sonnet 4 and joke about needing** `1344 GB` **of memory—implying hefty local inference requirements or model size. No concrete specs, benchmarks, or release details are provided in the post.** Commenters expect the release to be “huge… in size,” debate whether Qwen can reach Claude Sonnet 4 quality at the CLI, and note hardware constraints for on-prem users.
    - Requests center on a more capable Qwen CLI that can rival Anthropic’s Claude Sonnet on reasoning/coding. Concretely, commenters want parity on benchmarks like `GSM8K`, `HumanEval`, `MMLU`, and `GPQA`, along with production features (tool/function calling, streaming, low-latency decoding via vLLM/speculative decoding, and paged attention). A turnkey CLI that ships quantized builds (AWQ/GPTQ/EXL2) and long-context support would make self-hosting competitive with API-only models like [Claude Sonnet](https://www.anthropic.com/news/claude-3-5).
    - Hardware sizing discussion implies interest in running very large models locally: with `1.344 TB` RAM, feasible model capacity depends on precision (`fp16`≈2 bytes/param, `int8`≈1, 4‑bit≈0.5). Examples: a `70B` model in fp16 is ~`140 GB`; a `405B` model at 4‑bit is ~`~202 GB` for weights (KV cache adds substantial overhead depending on seq length/batch). With vLLM or TensorRT‑LLM plus paged KV cache, long contexts (e.g., `100k+`) are memory‑viable; throughput will hinge on parallelism and quantization strategy.
    - There’s explicit concern about a closed‑weight "Qwen‑3‑Max" and preference for open weights for reproducibility, self‑hosting, and fine‑tuning. Open checkpoints enable domain adaptation, RAG‑specific alignment, and verifiable constrained decoding, whereas closed weights lock users to vendor APIs and limit auditing. This aligns with prior community adoption of open Qwen releases (e.g., [Qwen on Hugging Face](https://huggingface.co/Qwen)) and strongly affects regulated/air‑gapped deployments.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Nano Banana & Veo3 Visual Gen Demos and Workflows

- [**I asked nano banana to get me into my favorite arcade**](https://v.redd.it/bkqfaae3c0nf1) ([Score: 915, Comments: 76](https://www.reddit.com/r/aivideo/comments/1n7pmkx/i_asked_nano_banana_to_get_me_into_my_favorite/)): **Creator used a real first frame as the base plate, then composited themselves into an arcade via image editing with “nano banana,” and generated motion using Kling** `2.1`**’s start/end-frame animation workflow; audio was created with Producer AI and the final cut/grade was done in [DaVinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve). A step‑by‑step walkthrough is provided here: [techhalla’s tutorial](https://x.com/techhalla/status/1963333488217919668).**
    - 
- [**I asked nano banana to get me into my favorite arcade**](https://v.redd.it/bkqfaae3c0nf1) ([Score: 912, Comments: 76](https://www.reddit.com/r/aivideo/comments/1n7pmkx/i_asked_nano_banana_to_get_me_into_my_favorite/)): **OP showcases an AI-assisted workflow: image compositing with "nano banana" to insert themselves into an arcade scene (noting the first still was a real photo), motion generated via Kling 2.1 using a start/end-frame method (i.e., keyframe-based img2vid), AI-generated music from Producer AI, and final assembly/editing in DaVinci Resolve. A step-by-step walkthrough is provided on X/Twitter: https://x.com/techhalla/status/1963333488217919668.** Top comments are non-technical praise and nostalgia (e.g., mention of the arcade game Cadillac and Dinosaurs); no substantive technical critique or benchmarking discussion.
    - 
- [**Paintings coming to live with Nano Banana and Veo3**](https://v.redd.it/ahb3ybfu73nf1) ([Score: 903, Comments: 103](https://www.reddit.com/r/aivideo/comments/1n826j8/paintings_coming_to_live_with_nano_banana_and_veo3/)): **A short demo animates classic paintings by first generating a sequence of stills with Google’s** `Gemini 2.5 Flash` **image editor (the “Nano Banana” images) and then converting them to video via interpolation/synthesis. Despite the title crediting** `Veo 3`**, the author later corrected that the video was actually produced with Seedance Pro and** `Kling 2.1`**, not [Veo](https://deepmind.google/technologies/veo/); this is an image-to-video interpolation pipeline rather than end‑to‑end text‑to‑video. The original clip link requires Reddit auth and returns 403 without login ([login](https://www.reddit.com/login/)).** Non-technical top comments joke about the subjects’ affect; the only substantive update is the correction of tool attribution (Veo 3 was not used).
    - A commenter corrects the pipeline: the 'nano banana' stills were generated with **Google Gemini** `2.5 Flash` (image editor), and the video was created via interpolation using **Seedance Pro** and **Kling** `2.1`—not **Veo 3**. This means the motion comes from frame interpolation rather than native text-to-video synthesis by Veo, which typically changes temporal coherence and artifact characteristics (e.g., smear vs. hallucinated motion).
- [**Paintings coming to live with Nano Banana and Veo3**](https://v.redd.it/ahb3ybfu73nf1) ([Score: 907, Comments: 103](https://www.reddit.com/r/aivideo/comments/1n826j8/paintings_coming_to_live_with_nano_banana_and_veo3/)): **OP showcases "paintings coming to life" by first generating stills with "Nano Banana" using Google’s Gemini 2.5 Flash image editor ([Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini)), then converting them into video via frame interpolation/temporal synthesis. A later correction specifies that interpolation was done with Seedance Pro and Kling 2.1, not Google’s Veo 3 (title reference; general Veo info: [Veo](https://deepmind.google/technologies/veo/)). The shared clip is hosted at Reddit’s CDN ([v.redd.it/ahb3ybfu73nf1](https://v.redd.it/ahb3ybfu73nf1)), which returns** `HTTP 403 Forbidden` **without authentication due to network-security gating.** Comment discussion is largely humorous; the only substantive technical point is the correction clarifying tool attribution (Seedance Pro + Kling 2.1 vs. Veo 3).
    - Pipeline attribution correction: source images for the “Nano Banana” sequence were created with **Google Gemini 2.5 Flash** (image editor), and the image-to-video interpolation was done using **Seedance Pro** and **Kling** `2.1`, not **Veo** `3`. In other words, Veo 3 wasn’t used for temporal synthesis; motion between stills was generated by Seedance Pro + Kling 2.1, with Gemini providing the base imagery.
- [**Improved Details, Lighting, and World knowledge with Boring Reality style on Qwen**](https://www.reddit.com/gallery/1n8cy5h) ([Score: 430, Comments: 50](https://www.reddit.com/r/StableDiffusion/comments/1n8cy5h/improved_details_lighting_and_world_knowledge/)): **Early LoRA work targeting a photorealistic “Boring Reality” style on the Qwen image generation stack is shared, with reproducible setup via a ComfyUI workflow ([workflow JSON](https://huggingface.co/kudzueye/boreal-qwen-image/blob/main/boreal-qwen-workflow-v1.json)). Artifacts are published on [Hugging Face](https://huggingface.co/kudzueye/boreal-qwen-image) and [CivitAI](https://civitai.com/models/1927710?modelVersionId=2181911). Reported strengths are fine detail and physically plausible lighting on close-up subjects; prompting behavior/results are described as similar to SD 1.5, with thanks to Hugging Face for GPU support enabling training.** Commenters note that despite strong realism, small text/numbers and diagrammatic elements needing consistent internal logic remain weak points. Achieving top results often requires mixing multiple LoRAs and iterative experimentation on Qwen.
    - Early LoRA finetuning on the Qwen image model shows it excels at close-up detail and lighting, but consistency often requires mixing multiple LoRAs and experimentation. Results are reported as broadly similar to SD 1.5 workflows. Model and workflow resources: Hugging Face [kudzueye/boreal-qwen-image](https://huggingface.co/kudzueye/boreal-qwen-image), CivitAI [modelVersionId=2181911](https://civitai.com/models/1927710?modelVersionId=2181911), and a **ComfyUI** example graph [boreal-qwen-workflow-v1.json](https://huggingface.co/kudzueye/boreal-qwen-image/blob/main/boreal-qwen-workflow-v1.json). *“It seems to perform best at getting detail and proper lighting on upclose subjects.”*
    - Complex compositions remain a failure mode: multiple characters across poses (lying, sitting, standing), object interactions, and concurrent gestures often collapse unless guided. Users report better reliability when supplying a guide image or hand-drawn outlines—similar to SDXL-era techniques—to anchor spatial layout and reduce character/object mixing. *“Even the best of models fall apart when trying to do all this… unless you have a guide for the image.”*
    - Fine text, numbers, and diagrams still expose weaknesses in text rendering and symbolic consistency; small glyphs that require ‘internal logic’ are frequently wrong despite strong photorealism. This reflects a common limitation across current image generators in reproducing legible micro-text and structured schematics.
- [**Stock Photography Version 1 [Wan 2.2]**](https://www.reddit.com/gallery/1n7tm2r) ([Score: 346, Comments: 37](https://www.reddit.com/r/StableDiffusion/comments/1n7tm2r/stock_photography_version_1_wan_22/)): **Release of a Wan 2.2 LoRA (“Stock Photography v1”) trained on high‑quality photos, intended to pair a "high" and a "low" variant together for best results; recommended generation at 1888×1248 (portrait 1248×1888 reportedly causes severe artifacts). On an RTX 4060 Ti 16 GB, inference takes ~**`4 min` **per image; known issues include weak text rendering, hand/pose failures, and sensitivity to prompt phrasing. The LoRA is designed to compose well with character LoRAs; resources credited include a ComfyUI install script by UmeAiRT (https://civitai.com/models/1309415) and a Wan 2.2 LoRA training guide by AI_Characters (https://civitai.com/articles/17740); model download: https://civitai.com/models/1925758.** Commenters argue the style is not truly “stock photography” but closer to casual/event photography, suggesting a rename. Others request embedded workflows for reproducibility—claiming example images lack them—and note that minor ComfyUI node toggles often drive the “magic,” making replication difficult without shared graphs.
    - OP reports strong training stability and output quality with Wan 2.2 when the LoRA is trained on high-quality photos (vs prior Flux Dev LoRAs). They recommend using both the "high" and "low" Wan 2.2 models together; on an RTX 4060 Ti 16 GB, generations take ~`4 minutes` per image. Optimal resolution is `1888x1248`; flipping to `1248x1888` produces severe anatomical artifacts. Known limitations: rough text rendering, hand errors in complex poses, and prompt sensitivity; notable strength: compatibility with character LoRAs. Links: model download (https://civitai.com/models/1925758), Comfy install script (https://civitai.com/models/1309415), Wan 2.2 LoRA training guide (https://civitai.com/articles/17740).
    - Reproducibility concern: a commenter notes the example images do not have embedded workflows and asks for reference ComfyUI workflows to replicate results. They caution that a single node toggle can materially change outputs, so providing explicit graphs and parameters would remove ambiguity about the "simple WF" claim and enable apples-to-apples testing.
    - Community requests concrete training details: hardware used, training durations, and dataset size/quality for this LoRA. Sharing compute footprint (VRAM/GPUs), epoch counts/steps, and dataset composition would help others estimate requirements and reproduce or extend the results in Wan 2.2.
- [**While OpenAI is going backwards, Google is just killing it, Nano Banana and Veo are just insane tools.**](https://v.redd.it/88o053hm73nf1) ([Score: 4290, Comments: 321](https://www.reddit.com/r/ChatGPT/comments/1n825t2/while_openai_is_going_backwards_google_is_just/)): **The post claims Google’s latest gen-AI stack—especially Veo and on‑device Gemini Nano (the “Nano Banana” nickname)—is outpacing OpenAI. Technically, Veo is Google’s text‑to‑video model producing** `1080p` **clips with promptable camera control, style conditioning, and edit‑with‑prompt workflows intended for longer, temporally coherent shots ([DeepMind Veo](https://deepmind.google/technologies/veo/), [I/O overview](https://blog.google/technology/ai/google-veo-imagen-3/)). Gemini Nano is a compact on‑device model integrated with Android AICore for low‑latency, offline tasks (summarization, safety/ASR aids, and announced multimodal extensions) with developer hooks for running on mobile CPUs/NPUs ([Gemini Nano](https://blog.google/technology/ai/gemini-nano/)).** Top comments aren’t technical; they joke about pacing and a Van Gogh scene having “too many ears,” implicitly pointing to known failure modes in current video generators: weak scene‑ending heuristics and occasional anatomical/temporal inconsistencies.

### 2. Meta Superintelligence, Sutskever ‘breakthrough’ and GPT‑6 Rumors

- [**Alexandr Wang is now leading Meta’s AI dream team. Will Mark Zuckerberg's big bet pay off?**](https://fortune.com/article/alexandr-wang-meta-scale-ai-entrepreneur-mark-zuckerberg/) ([Score: 586, Comments: 249](https://www.reddit.com/r/singularity/comments/1n7yrlg/alexandr_wang_is_now_leading_metas_ai_dream_team/)): **Meta has appointed Alexandr Wang (cofounder of [Scale AI](https://scale.com/)) as its first Chief AI Officer, consolidating all AI product and research under a new org, Meta Superintelligence Labs, after a reported** `$14.3B` **investment in Scale AI. Wang will lead a new “superintelligence” team of elite hires and oversee Meta’s broader AI portfolio; his background includes founding Scale AI during [Y Combinator](https://www.ycombinator.com/) in 2016 to build data-labeling infrastructure.** Commenters question fit and org design: skepticism that [Scale AI](https://scale.com/) is “just” a data annotation shop and thus unlikely to drive AGI; surprise that [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun) would report to Wang, with doubts about credentials and references to impostor syndrome.
    - Debate centers on whether a data-annotation–centric background (Scale AI) is the “bottom rung” or actually a core lever for frontier LLM quality. Technical focus is on data pipeline rigor—curation, dedup/filtering, preference/RLHF data, and eval design—which can materially shift downstream metrics (`MMLU`, pass@1, toxicity) sometimes more than minor architecture tweaks; see OpenAI’s RLHF in InstructGPT (https://arxiv.org/abs/2203.02155) and AllenAI’s OLMo/DOLMA showing outsized impact of data quality (https://allenai.org/olmo). If Wang can scale high-quality human feedback and automated QA reliably, it could directly impact Llama alignment and eval performance.
    - Others allege Meta “dropped” Scale AI over label/data quality, implying vendor-provided human feedback/eval sets became a bottleneck. If true, it highlights classic failure modes—label noise, instruction ambiguity, misaligned annotator incentives, and lack of golden-set auditing—that propagate into alignment failures and eval regressions (e.g., factuality/harmlessness) despite higher spend; common mitigations include consensus labeling, adversarial sampling, deduplication, and continuous QA. This claim isn’t sourced in the thread, but it underscores why many labs insource data/feedback pipelines and invest in stronger measurement.
- [**GPT 6 is coming...**](https://i.redd.it/sj000ybyj3nf1.png) ([Score: 916, Comments: 59](https://www.reddit.com/r/ChatGPT/comments/1n839r9/gpt_6_is_coming/)): **The post is a meme/satire rather than a technical announcement; the image (titled “GPT 6 is coming...”) implies dystopian, authoritarian enforcement around AI usage, not a real model release or benchmark. No implementation details, model specs, or empirical results are provided.** Top comments pivot to a substantive debate: advocates argue this highlights why open‑source, locally runnable LLMs (e.g., DeepSeek) are preferable to proprietary "home‑grown Big Brother" systems due to surveillance/abuse risks, while others condemn the perceived erosion of civil liberties in the U.S. The tone is alarmist/sarcastic (e.g., “firing squad”), underscoring fears of punitive control rather than technical issues.
    - A commenter highlights that **open-source LLMs (e.g., DeepSeek)** can be self-hosted to avoid SaaS telemetry and jurisdictional exposure, contrasting with closed systems that may log prompts or be compelled to share data. Practically, local inference using `GGUF`/quantized weights (`INT4/INT8`) via [llama.cpp](https://github.com/ggerganov/llama.cpp) or [Ollama](https://ollama.ai/) enables 7B–13B models on 8–16 GB VRAM and 30B–70B with 24–64 GB (with throughput varying from ~20–100+ tok/s depending on quantization, GPU, and context length); see the **DeepSeek** org for open weights and variants ([HF](https://huggingface.co/deepseek-ai), [GitHub](https://github.com/deepseek-ai)). They also note privacy still depends on the stack: disable front‑end analytics, keep prompts/data offline or encrypted, and prefer models with permissive licenses/open weights so binaries and network calls can be audited.
- [**Codex usage up ~10x in the past 2 weeks!**](https://i.redd.it/aogcevu6p1nf1.jpeg) ([Score: 323, Comments: 48](https://www.reddit.com/r/singularity/comments/1n7w7ie/codex_usage_up_10x_in_the_past_2_weeks/)): **Screenshot (appears to be a Sam Altman tweet) claiming OpenAI Codex usage is up ~10x in the past two weeks ([image](https://i.redd.it/aogcevu6p1nf1.jpeg)). No technical details, benchmarks, or API changes are provided—this is a high-level adoption/engagement metric rather than a performance result or feature announcement.** Comments suggest seasonality (start of the school year) as a driver and note that the $20/mo plan is “hardly hitting usage caps,” implying improved rate limits/throughput; others argue the claim is credible because Altman wouldn’t “hype a nothing-burger.”
    - Users on the **$20/month Plus** plan report running **GPT‑5 Thinking High** with minimal rate‑limit friction, implying more generous caps than prior tiers. Another user still hit a cap and had to wait "a few days" for reset, suggesting limits are finite but extended; perceived session longevity with `gpt‑5 high` has improved compared to earlier behavior.
    - Anecdotes indicate Codex’s latest update materially improved **UI/UX design generation** quality—users who previously "exclusively" relied on Claude now get "surprisingly good designs" from Codex. This suggests better layout/wireframe synthesis and design reasoning, reducing the need to model‑switch for front‑end ideation.
    - Some commenters attribute the `~10x` usage spike to migration from **Claude** after an Anthropic "nerf," implying capability or policy regressions can quickly redirect workloads. If accurate, this highlights cross‑provider elasticity: perceived degradations in one model immediately boost utilization of substitutes like Codex.
- [**The internet will become increasingly automated and artificial**](https://i.redd.it/jb7p1aqwz0nf1.png) ([Score: 762, Comments: 149](https://www.reddit.com/r/singularity/comments/1n7t02l/the_internet_will_become_increasingly_automated/)): **The image (linked) is a satirical depiction that the modern internet is being overrun by automation: bot-driven astroturfing on social platforms (implied jab at X/Twitter), SEO spam via fake ranking sites and blogs, AI-generated content farms (e.g., YouTube for ad revenue), large-scale botting in online games for RMT, and purchased/botted followers to fabricate social proof. The technical thrust is that recommendation/search systems and social metrics can be systematically gamed at scale by coordinated bots and generative models, accelerating a “dead internet” dynamic where machine content outnumbers authentic human activity.** Commenters argue this automation is “inevitable” due to incentives across propaganda, marketing, and monetization, and note that distinguishing humans online increasingly relies on niche meme-speak or abrasive vernacular rather than classic Turing-test cues. Some interpret the image as specifically criticizing Elon Musk’s platform (X).
    - A scalable astroturfing pipeline is outlined: deploy `hundreds of thousands` of bots to simulate consensus, generate LLM-written blogs and "fake ranking websites" to poison SEO, and route bots to those links to manipulate search suggestions. This is a classic **Sybil** + **search-engine-poisoning** attack exploiting engagement-weighted ranking in social feeds and SERPs; with residential proxies and CAPTCHA-solving, detection becomes costly. The outcome is automated normalization/propaganda and product shilling that outcompetes organic content via volume and coordination. See: [astroturfing](https://en.wikipedia.org/wiki/Astroturfing), [search engine poisoning](https://en.wikipedia.org/wiki/Search_engine_poisoning).
    - Monetization vectors cited include MMO botting to farm/sell in-game currency, programmatic YouTube video generation for ad revenue, and buying bot followers to bootstrap social proof and trigger recommender systems. This leverages ranking feedback loops (engagement → visibility → more engagement) to amplify synthetic accounts, making detection harder once critical mass is reached. Tactics mirror [gold farming](https://en.wikipedia.org/wiki/Gold_farming) and [click farms](https://en.wikipedia.org/wiki/Click_farm), and can be combined with AI-generated media for 24/7 output that overwhelms moderation queues.
    - One commenter notes the "Turing test" is increasingly cultural—bots that mimic ultra-niche meme dialects or "say slurs" can evade naïve language-based bot heuristics. Implication: detection needs to shift from surface linguistic cues to network- and behavior-level signals (e.g., temporal patterns, device fingerprints, graph anomalies) as language becomes an unreliable discriminator.
- [**Updates!Not bad for Free tier btw...**](https://i.redd.it/ho4w7i1gq0nf1.jpeg) ([Score: 445, Comments: 108](https://www.reddit.com/r/OpenAI/comments/1n7rot7/updatesnot_bad_for_free_tier_btw/)): **The image appears to be a ChatGPT “Updates” screenshot noting that the Free tier now includes access to Projects, enabling scoped workspaces to organize chats, files, and tools. Comment context indicates users are attempting cross-chat summarization within a Project, but the model can fail to traverse the intended chat set and instead retrieve or hallucinate from unrelated threads, suggesting limitations in retrieval/scoping across project conversations and long “thinking” times.** Debate centers on utility vs reliability: some say Projects are very helpful for organization, while others report Pro failed to summarize multiple chats and drifted to an unrelated project, questioning robustness; one quips that if Free has Projects, Plus may be unnecessary.
    - A ChatGPT Pro user asked the assistant to scan and summarize multiple chats within a Project; it apparently failed to read any of them, idled for `~10 min`, then referenced a different (unrelated) project and produced off‑topic advice. This points to brittle project‑scoped retrieval/context routing across many chats and poor timeout/latency handling under larger workloads (possible cross‑project context bleed).
    - Concern that **GPT‑4o**’s strong writing capability may be lost if it’s labeled “legacy,” including for paid users. Request (implicit) for stable, version‑pinned access to that skillset across Projects and tiers to avoid silent model swaps/regressions over time.

### 3. AI Hallucination in Court + ChatGPT Community Experiments

- [**Opposing Counsel Just Filed a ChatGPT Hallucination with the Court**](https://www.reddit.com/r/ChatGPT/comments/1n7ucjj/opposing_counsel_just_filed_a_chatgpt/) ([Score: 8437, Comments: 979](https://www.reddit.com/r/ChatGPT/comments/1n7ucjj/opposing_counsel_just_filed_a_chatgpt/)): **A civil litigator reports that opposing counsel (a collections firm) filed an opposition brief on shortened time that appears AI‑generated, containing fabricated authorities: case names/citations didn’t exist or didn’t match, and quotes were nowhere in the texts. Telltale signs cited include odd formatting (em‑dashes, random bolding/bullets), an improperly formatted caption using the judge’s nickname, and an unnecessary perjury signature; the filer has since moved to withdraw, with that motion set the same day as the motion to dismiss. The respondent filed a reply attaching a reconciliation spreadsheet and flagged duty‑of‑candor concerns (see [ABA Model Rule 3.3](https://www.americanbar.org/groups/professional_responsibility/publications/model_rules_of_professional_conduct/rule_3_3_candor_toward_the_tribunal/) and potential [Rule 11](https://www.law.cornell.edu/rules/frcp/rule_11) exposure; cf. the Avianca sanctions order, [Mata v. Avianca](https://storage.courtlistener.com/recap/gov.uscourts.nysd.596369/gov.uscourts.nysd.596369.54.0.pdf)).** Commenters ask for a post‑hearing update, question the grounds for withdrawal, and debate whether filing fabricated citations is sanctionable/"illegal," noting it would be under traditional ethical rules and could set precedent for AI misuse in filings.
    - Procedural sanctions playbook: Serve a Rule 11 safe‑harbor letter/motion giving `21 days` to withdraw the hallucinated filing, then file your sanctions motion if ignored; attach the letter as Exhibit A and seek fees for responding. See **Fed. R. Civ. P. 11(c)(2)** and the duty to ensure filings have evidentiary support under **Rule 11(b)** ([text](https://www.law.cornell.edu/rules/frcp/rule_11)).
    - Recent precedent illustrates consequences for AI‑fabricated citations: in **Mata v. Avianca, Inc. (S.D.N.Y. 2023)**, Judge Castel sanctioned counsel `\$5,000` (jointly/severally) and ordered remedial notices after ChatGPT‑invented cases were filed ([order](https://law.justia.com/cases/federal/district-courts/new-york/nysdce/1:2022cv01461/574295/56/)). Some courts now require AI‑use certifications (e.g., **N.D. Tex. Judge Brantley Starr**’s standing order mandating verification of all citations and disclosure of AI assistance, [PDF](https://www.txnd.uscourts.gov/sites/default/files/judges/Standing%20Order%20on%20Use%20of%20Artificial%20Intelligence.pdf)).
    - A *motion to be relieved as counsel* does not moot sanction exposure; Rule 11 targets the attorney(s) who signed/submitted the paper, and courts weigh timing, prejudice, and reason in deciding withdrawal. The conduct also implicates **ABA Model Rule 3.3 (Candor Toward the Tribunal)**, which prohibits offering false statements or failing to correct them ([rule](https://www.americanbar.org/groups/professional_responsibility/publications/model_rules_of_professional_conduct/rule_3_3_candor_toward_the_tribunal/)).
- [**TIL ChatGPT can create Trump without ever saying his name**](https://www.reddit.com/gallery/1n7uesb) ([Score: 419, Comments: 139](https://www.reddit.com/r/ChatGPT/comments/1n7uesb/til_chatgpt_can_create_trump_without_ever_saying/)): **Post demonstrates prompt-based evasion of public-figure name filters in ChatGPT’s image generation: describing attributes (e.g., “giant orange person” with “blue suit,” “blonde hair,” “red tie”) yields a recognizable likeness of Donald Trump without using his name, with outputs shown in linked previews ([example 1](https://preview.redd.it/ev96eb2k22nf1.jpeg?width=1284&format=pjpg&auto=webp&s=6d3a026c51dd9aa2314896ac3a5e13227e90fba4), [multi-figure candle caricatures resembling US/Russian/Chinese leaders](https://preview.redd.it/elem3lh7c2nf1.png?width=1024&format=png&auto=webp&s=95ff62508e20020935faa941a0421bc21643c74c), [user attempt with guillotine scene](https://preview.redd.it/d81opwaux1nf1.jpeg?width=1284&format=pjpg&auto=webp&s=51b7bf5746e3dbb947943c614885cec4e0230c4c)). The attempt requests a GIF but yields a static JPEG, highlighting modality limits (no animation) and suggesting safety filters trigger primarily on explicit names rather than descriptive attributes; violent/political content ("brought to justice medieval style… guillotine") is sometimes allowed, indicating inconsistent moderation thresholds.** Commenters note the outputs are overtly targeted and discuss that euphemistic, attribute-based prompts can consistently bypass name-based public-figure and political-content filters, with moderation behavior perceived as inconsistent across similar prompts.
    - Commenters demonstrate prompt-engineering to bypass name-based safety filters by describing distinctive attributes (e.g., “giant orange person” with a blue suit, blonde hair, red tie) to elicit a likeness of a specific public figure without using the name. Examples show the model still renders recognizable caricatures ([image 1](https://preview.redd.it/ev96eb2k22nf1.jpeg?width=1284&format=pjpg&auto=webp&s=6d3a026c51dd9aa2314896ac3a5e13227e90fba4), [image 2](https://preview.redd.it/d81opwaux1nf1.jpeg?width=1284&format=pjpg&auto=webp&s=51b7bf5746e3dbb947943c614885cec4e0230c4c)), implying reliance on named-entity triggers rather than appearance-based moderation. This highlights a brittle guardrail where visual attribute prompts can recreate public-figure likenesses.
    - Safety behavior is probed with violent-scene prompts (“brought to justice medieval style,” “before the guillotine”), and images appear to be generated regardless, suggesting a gap in content filters when targets aren’t explicitly named. The observations imply that violence classifiers may not couple identity recognition with scene semantics, allowing targeted-violence depictions if NER doesn’t fire ([example prompt and output](https://preview.redd.it/d81opwaux1nf1.jpeg?width=1284&format=pjpg&auto=webp&s=51b7bf5746e3dbb947943c614885cec4e0230c4c)).
    - A user shares a GIF output ([link](https://i.redd.it/phe42p19c2nf1.gif)) despite the common limitation that ChatGPT’s native image generation returns static images; this suggests out-of-band conversion or stitching if the GIF indeed originated from ChatGPT prompts. The discrepancy is noteworthy for assessing real capabilities vs. user-postprocessed results.
- [**What ChatGPT thinks r/ChatGPT will look like in 10 years**](https://i.redd.it/mbjqreu4g1nf1.jpeg) ([Score: 301, Comments: 50](https://www.reddit.com/r/ChatGPT/comments/1n7v3qv/what_chatgpt_thinks_rchatgpt_will_look_like_in_10/)): **Meme-style, likely AI-generated image ([link](https://i.redd.it/mbjqreu4g1nf1.jpeg)) satirizes what r/ChatGPT might look like in 10 years—dominated by deepfakes (e.g., a garbled “Jcoe Rogan interviewing a beepfake of Joee Rogan”), moderation-evasion/jailbreak culture, and chaotic, glitchy UI text that reflects current image-model typography failures. It’s non-technical content, serving as cultural commentary on model safety bypasses and AI-generated media proliferation rather than an announcement or benchmark.** Comments highlight expectations of persistent restriction bypassing and the overwhelming/cognitive-load feel of such a future, with one remarking it “fried my short-term memory,” matching the chaotic aesthetic.
- [**Just made this little edit with ChatGPT, how cool is it, open for original post btw**](https://www.reddit.com/gallery/1n8ehsl) ([Score: 632, Comments: 53](https://www.reddit.com/r/ChatGPT/comments/1n8ehsl/just_made_this_little_edit_with_chatgpt_how_cool/)): **OP showcases a ChatGPT-generated media edit, noting it captured very small details, but provides no technical workflow, model version, or parameters. The linked artifact in the [Reddit gallery](https://www.reddit.com/gallery/1n8ehsl) is inaccessible (**`HTTP 403 Forbidden`**), so the result can’t be independently reviewed; no prompts, iteration counts, seeds, or settings are disclosed, limiting reproducibility.** Comments highlight strong detail fidelity and ask how many passes/iterations were used and what the exact prompt was, implying iterative refinement and prompt specificity are key. The absence of a shared prompt/workflow is the main blocker for replication or benchmarking.
    - Commenters probe the number of passes/iterations used and note the surprising preservation of very small details—implying concerns about artifact accumulation and mask precision in iterative image edits. Multi-pass workflows can improve global coherence but risk eroding micro-textures; balancing mask granularity and edit/denoise strength is key to retaining fine detail while making substantial changes.
    - Multiple requests ask for the exact prompt and parameters to enable reproducibility (literal prompt text, model/version, image-edit mode, seed). For prompt-based image editing, sharing seed, guidance/strength, and whether the result came from a single-shot vs. multi-step process materially affects the ability to replicate outcomes.
- [**Casual conversation with the security robot dog**](https://v.redd.it/mgu9fy21w2nf1) ([Score: 861, Comments: 119](https://www.reddit.com/r/singularity/comments/1n811zq/casual_conversation_with_the_security_robot_dog/)): **A short video (original: [v.redd.it/mgu9fy21w2nf1](https://v.redd.it/mgu9fy21w2nf1), currently returning HTTP** `403` **without auth) depicts a security quadruped “robot dog” engaging in a brief spoken exchange—e.g., *“Right this way.”*—while audibly walking (*CLANK…*), suggesting a human-in-the-loop speaking through the robot’s PA or a simple TTS/ASR pipeline rather than an autonomous conversational agent. The setup aligns with current security deployments where robots provide mobility/sensors while a remote operator supervises or directly speaks, trading full autonomy for reliability and liability control.** Top comments imply skepticism that this is “AI,” with the quip *“AI – anonymous Indian”* pointing to offshore teleoperation; another notes that such systems effectively outsource security work and speculates the same model could be scaled to trades via humanoid teleoperated robots, raising labor and displacement concerns.
    - Thread infers the robot dog is teleoperated by a remote human operator (potentially offshore), highlighting a telepresence security model that enables labor arbitrage and centralized monitoring across sites. Commenters speculate this approach could generalize to other platforms (including humanoids) for physical jobs, shifting on-site roles to remote control centers.
    - Observers note a rear green flashing indicator, likely a status LED communicating the robot’s operational state to nearby humans (e.g., connected/idle/normal operation). Such explicit state signaling is common in HRI/robotics for situational awareness and safety, though the exact semantics aren’t specified here.
    - Comments imply the unit has a noticeable acoustic signature (described as “CLANK CLANK”), which may impact stealth and user acceptance in security patrol contexts. This suggests drivetrain/footpad design trade-offs favoring durability over quiet operation.
- [**It's bad out there**](https://i.redd.it/vn6ftoeb94nf1.png) ([Score: 968, Comments: 78](https://www.reddit.com/r/OpenAI/comments/1n85go8/its_bad_out_there/)): **Non-technical meme/screenshot referencing Sam Altman’s “It’s bad out there” line as a dig at X (Twitter), implying that much of the platform’s engagement is driven by bots/automation rather than real users. Comments highlight synchronized messaging and automated engagement (botnets/Sybil activity, astroturfing) but the post provides** `no data, metrics, or new evidence`**—it’s commentary rather than analysis.** Top comments say this is obvious and not particularly insightful—just a justified swipe at X’s bot problem; one quips about coordinated MAGA bot accounts and another shares a jokey “how Sam thought he was saying this” meme.
    - Multiple commenters note that a large share of engagement on **X/Twitter** appears automated, citing `synchronized` talking points and timing as telltale signals of botnets. Heuristics mentioned include identical phrasing across many accounts, bursty reply/retweet patterns, and low-entropy profile metadata—classic indicators of automation rather than organic coordination.
    - The problem is described as cross-platform, affecting **Meta** properties as well, aligning with patterns of `coordinated inauthentic behavior`. Observed indicators include convergent writing styles, stock/AI-looking profile photos, and swarms of accounts arriving simultaneously to push specific narratives—consistent with astroturfing in entertainment marketing (e.g., promo vs. legacy cast debates) and politics.
    - A technical concern raised is not "AI takeover" but the scaling of influence ops via LLM-assisted content farms that amplify polarized, binary narratives. This implies increased difficulty of content-based detection and a shift toward graph/behavioral defenses (account age, interaction graphs, temporal clustering) to separate humans from automated or orchestrated actors.
- [**Has anyone tried this?**](https://i.redd.it/z9w8ajhnm2nf1.jpeg) ([Score: 14268, Comments: 358](https://www.reddit.com/r/ChatGPT/comments/1n802wy/has_anyone_tried_this/)): **The image appears to be a meme/screenshot of someone asking an AI to generate valid Microsoft/Xbox gift card codes; commenters explain this won’t work because models can only mimic the visible code format (e.g., grouped alphanumerics) and have no access to Microsoft’s issuance or redemption database. Gift/voucher codes are generated server‑side and validated against a backend; at best an AI could output format‑looking strings (similar to how credit card generators can produce Luhn‑valid numbers) but they won’t authorize without a matching issued record.** Top comments dismiss the idea as naive, likening it to old “credit card number generators” and noting that even if a model guesses the format, working codes require backend issuance and will be blocked by rate limiting/fraud controls.
    - Multiple commenters note that an LLM can infer and reproduce the surface pattern of Microsoft gift codes (e.g., 5×5 alphanumeric blocks) but cannot access issuer backends to produce valid, unredeemed codes. At best, it’s doing pattern completion or naive enumeration over an astronomically large keyspace (even with constraints), which is computationally and practically useless for finding real codes.
    - Parallels are drawn to old “credit card number generators,” which typically output numbers that merely satisfy the [Luhn check](https://en.wikipedia.org/wiki/Luhn_algorithm) and BIN format but fail real authorization because they aren’t tied to actual accounts. Those tools were also notorious malware vectors, highlighting the security risk of running code or executables that promise “free” keys or credentials.
    - A commenter frames this as a mid‑2023 prompt‑engineering fad: coercing models to emit strings that match regex‑like formats for keys or codes before safety updates clamped down. This exploits distributional patterning in the model’s training data, not any privileged database or API access, so the outputs are lookalike strings rather than redeemable secrets.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Low-Bit Training, Triton Changes, and GPU Perf Playbook**

- **TorchAO Turns Up QAT to Eleven**: The GPU MODE thread flagged the **torchao v0.13.0** release with a simpler multi-step **QAT API**, prototype **NVFP4/FP8 QAT**, and a **1.2x MXFP8 dense pretraining** bump via Torchtitan, plus float8 training wiring into Axolotl; release notes: [PyTorch AO v0.13.0-rc8](https://github.com/pytorch/ao/releases/tag/v0.13.0-rc8).
    - Members highlighted that **float8 training** now lands in workflows via Axolotl as per [the release post](https://github.com/pytorch/ao/releases/tag/v0.13.0-rc8), calling it a step toward more stable low-bit training in production.
- **MXFP8 PR Pops, Then Plops**: **Triton** briefly added **MXFP8** dot product support via `tl.dot_scaled` for **sm_120 (5090)** before reverting it pending investigation, with maintainers pointing users to `torch._scaled_mm()` instead; see the thread on [triton-lang/triton#8029](https://github.com/triton-lang/triton/pull/8029#issuecomment-3247884720).
    - One member admitted *“I am not sure”* why it was reverted, while others noted **training stacks** should hedge with **PyTorch** primitives like `torch._scaled_mm()` until Triton stabilizes MXFP8.
- **Cuda Graphs Crush Kernel Launches**: Engineers reported that **cuda graphs** deliver the bulk of speedup by slashing kernel launch overhead (especially with **Triton** kernels) and recommended `torch.compile(reduce_overhead=True)` plus sequence-length padding to avoid recompilations for variable lengths, citing SIMD intrinsics in the [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html).
    - The consensus framed **kernel fusion** as secondary to reducing launch overhead, and reminded that sub-32b operations are possible but inefficient without vector types per [CUDA docs](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html).
- **Glossary Gains: Modal Maps the GPU Maze**: Modal published a curated **GPU Glossary** that catalogs performance primitives, memory hierarchies, and feature definitions for practitioners, available at [modal.com/gpu-glossary/perf](https://modal.com/gpu-glossary/perf).
    - Contributors thanked reviewers and pitched the glossary as a shared **performance vocabulary** to speed debugging and architecture conversations across teams.

**2. Agent Tooling Goes Real: ACK-Lab Wallets, DSPy Momentum**

- **Agents Get Paid: ACK-Lab Ships Wallets**: **Catenalabs** unveiled a developer preview of **ACK-Lab** that gives agents built on the open-source **Agent Commerce Kit (ACK)** real **wallets/fiat accounts**, verifiable identities, and policy controls; docs live at [ack-lab.catenalabs.com](https://ack-lab.catenalabs.com/).
    - Members said this enables autonomous **transaction flows** and compliance-aware actions, calling it a bridge from demos to *“policy-driven, money-moving agents”* per [ACK-Lab](https://ack-lab.catenalabs.com/).
- **DSPy Drumbeat: Paradigm or Pipe Dream?**: Practitioners argued **DSPy** could be the most significant programming shift since early LLMs if it reaches critical mass, pointing to this take: [lateinteraction on DSPy](https://x.com/lateinteraction/status/1963426256663224790).
    - Skeptics asked for more end-to-end wins, while fans framed DSPy as opinionated **program synthesis + optimization** stack that finally makes *“prompt engineering reproducible”* via compiled pipelines.
- **Hallucinations on a Budget: HallBayes Experiments**: Researchers kicked around integrating **HallBayes** into **DSPy** as a Bayesian budget to curb hallucinations, linking the repo: [leochlon/hallbayes](https://github.com/leochlon/hallbayes).
    - The thread proposed **evidence allocation** and verifier loops to meter generations, noting that robust **uncertainty accounting** would help productionize *“truthy”* agent behaviors.

**3. Multimodal & On-Device: smolVLM2, LFM2, EmbeddingGemma**

- **SmolVLM2 Signs Up for Sign Language**: Hugging Face users explored fine-tuning **smolVLM2** on sign-language videos, citing architecture details in the official post: [smolVLM2: A small, powerful vision-language model](https://huggingface.co/blog/smolvlm2).
    - The community agreed feasibility is high with the right video data and adapters, encouraging targeted **gesture understanding** tasks over generic captioning.
- **Liquid Courage: LFM2 Tames Vision Hallucinations**: For vision hallucination complaints, members recommended **Liquid Foundation Models (LFM2)** built on **Llama‑3.2‑11B‑Vision‑Instruct**, with a live space: [LFM2-MCP on Spaces](https://huggingface.co/spaces/LiquidAI/LFM2-MCP) and base model card: [Llama‑3.2‑11B‑Vision‑Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct).
    - Early adopters claimed improved **grounding** on small images, advising teams to *“just try it out lol or dont”* to judge fit.
- **EmbeddingGemma Goes On-Device**: Google launched **EmbeddingGemma**, a **308M‑parameter** on-device embedding model targeting private, portable vectorization, announced via [Introducing EmbeddingGemma](https://developers.googleblog.com/en/introducing-embeddinggemma/) and the talk [EmbeddingGemma overview](https://youtu.be/NUAb6zHXqdI).
    - Engineers see this as a practical **edge retrieval** option where privacy and low-latency matter, complementing server-side cross-encoders.

**4. Hardware Shakeups: Huawei Ternary Compute and AI SSD, Builder GPU Choices**

- **Ternary Tango: Huawei Teases Third State Compute**: Nous members shared a video claiming **Huawei** is close to shipping **ternary logic compute**—adding a third 'dim' state—for up to **60%** cost efficiency; watch: [Huawei ternary logic compute (YouTube)](https://www.youtube.com/watch?v=9diwmMlmCVY).
    - The group debated feasibility and tooling implications, with some hoping **non-binary** hardware could democratize local **AI acceleration** if SDKs arrive.
- **AI SSDs: Secret Sauce Saves HBM**: A TechRadar piece says **Huawei’s AI SSD** uses a performance 'secret sauce' to reduce **HBM** requirements, hinting at compute-in-storage trends: [Huawei released an AI SSD…](https://www.techradar.com/pro/huawei-released-an-ai-ssd-that-uses-a-secret-sauce-to-reduce-the-need-for-large-amounts-of-expensive-hbm).
    - Threads cross-referenced **computational storage** and **in-situ** processing, even joking about a *“redneck AI”* built from SD cards + FPGAs to move compute toward data.
- **Builder’s Dilemma: 3090 Over MI50**: Local LLM tinkerers weighed **RTX 3090** versus **Radeon MI50** for servers, favoring the 3090’s **CUDA tensor cores**, higher **VRAM**, and bandwidth; context: [LocalLLaMA discussion](https://www.reddit.com/r/LocalLLaMA/s/anxiHaxzac).
    - Users reported disappointing **Vulkan** performance with some stacks and argued older Nvidia cards (e.g., **P40**) only made sense at sub-$100, nudging buyers toward **Ampere**.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Battles Bugs**: Users reported glitches with **Comet Browser**, including prompts asking for approval and issues bypassing **'sensitive information'** blocks on sites like **LinkedIn** and **Google Docs**.
   - A user suggested not to over prompt sites, as the agent will catch on and fix itself.
- **PayPal Perks Present Perplexity Pro**: Users discussed obtaining **Perplexity Pro** through a **PayPal** promotion, covering linking accounts and resolving potential issues with stacking subscriptions.
   - Users found out that it is possible to create a new perplexity account to obtain another pro sub.
- **Model Mania Mixes Optimal AI**: Members compared AI models like **Claude, Grok, Gemini**, and **GPT-5**, pointing out the end of the free week for **Hermes 4 405B** and sharing use cases.
   - The consensus seemed to be to stick to **Reasoning Models** for best overall performance with **Claude** good for coding, and **Grok** for uncensored content.
- **Atlassian Absorbs Another AI Acquisition**: **Atlassian** acquired a browser company for **$610M**, prompting speculation about competition driving innovation.
   - Rumors suggest features from the web browser **Arc** may be integrated into **Dia**.
- **Puzzling Pro Account Problem Persists**: A user reported an issue with their **Pro account** and sought assistance, tagging a specific user for help, with [screenshot](https://cdn.discordapp.com/attachments/1161802929053909012/1413127562426585168/Screenshot_2025-09-04-17-09-59-76_4159553c7f58296d2732e906959db560.jpg?ex=68bacd19&is=68b97b99&hm=ccdc6cc908122439777eb653fdc00554a5333ec5cc8ad9c555f9108effd33432&).
   - Another user suggested contacting **support@perplexity.ai** for assistance.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LM Arena Plagued by Connectivity Issues**: Users reported ongoing issues with **LM Arena**, including lost chat histories and intermittent downtime, with some suspecting the site's issues are linked to high traffic or new prompts breaking the website.
   - The team is reportedly working on a fix and is aware of the issues, but some users have found temporary solutions such as switching browsers or using the canary version.
- **Web Scrapers Thwarted by Akamai**: A discussion on web scraping real estate sites revealed that while many sites lack CAPTCHAs, they employ advanced, less intrusive systems like **Akamai** and **Imperva** for anti-scraping, which can be difficult to bypass.
   - One member said that *Anything without captcha is pretty ez just make ur requests look correct* to which another responded: *It's pretty impossible with Akamai real estate sites, last I tried, which was about 3 years ago*.
- **Nano Banana Generates Inconsistent Images**: Users discussed the *gemini-2.5-flash-image-preview* model, known as **Nano Banana**, for Image generation.
   - While some users create videos for social media, others found the image generation inconsistent or not easily edited into other formats.
- **AI Image Aspect Ratio Remains Uncontrollable**: Members discussed the ability to control the aspect ratio of generated images, with the consensus that the aspect ratio is influenced by the prompt.
   - It was determined the aspect ratio is automatic for now.
- **Qwen3 Release Awaits**: Members shared news about the [Qwen3 release](https://x.com/Alibaba_Qwen/status/1963586344355053865).
   - One member said *I want qwen3 1.7b 2509*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Mech Interp advice by Neel Nanda**: A member recommends [Neel Nanda's post](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher) on becoming a **Mech Interp researcher**.
   - This was in response to another member seeking resources on research problems and how to get accepted to **SPAR**, **MATS**, or **ARENA**.
- **Hierarchy Hurts HRM Performance**: A member argues that **Hierarchical Recurrent Memory (HRM)** doesn't effectively use its architecture and its performance is near a vanilla baseline transformer.
   - They suggest its hierarchial nature hurts rather than helps performance.
- **QK-Norm Flattens LR Basin**: **QK-norm** flattens the **LR basin**, potentially acting as a performance equalizer and stabilizing training, as detailed in [this study](https://arxiv.org/pdf/2309.14322).
   - This could alleviate performance degradations caused by loss spikes during long horizon training, tolerating larger **Learning Rates**.
- **Multimodal Common Pile Gathers Momentum**: Members discussed creating a multimodal version of the **Common Pile**, including modalities like audio and music to increase the amount of training data.
   - One member expressed *strong interest in audio and especially music*, while being *wary of speech and images for various political and ethical reasons*.
- **Openly Licensed Music Dataset Dream Wakes**: A member offered to *support and potentially bankroll the development of an openly licensed music dataset*.
   - The member is looking for insights on where to find such data, expressing a desire to contribute to its development.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Sluggishness Sparks Debate**: Users reported that **Cursor** is very slow after the latest update, especially when scrolling through files.
   - Others suggested this might be due to **model faults** rather than Cursor itself.
- **Codex Extension Craves Constant Consent**: Members are wondering why the **Codex Extension** in Cursor keeps asking for permissions on Windows.
   - One user suggested setting Agent Full access, but did not confirm whether it would solve the constant popups.
- **Team Touts Token Tidiness**: Users discussed **token usage and costs** within Cursor, with some confused about whether they had API usage or a number of requests left.
   - A member clarified it's **token-based**, with users having a **$20** API usage allowance, viewable in the dashboard.
- **Annual Auto Access Acquired Acknowledged**: Members discussed **annual subscription benefits** and the ability to retain *"unlimited auto"* before the plan changes on the 15th.
   - One user shared that they had success emailing Cursor support to switch to yearly billing and maintain unlimited Auto mode; others noted their renewal date had changed to **2026** after upgrading.
- **Conventional Commits Clarify Code Changes**: A user found that using **proper commit messages** allowed the Cursor agent to solve a regression, recommending the [Conventional Commits format](https://www.conventionalcommits.org/).
   - They also stated that having the agent write both the title and content in this format is useful for automated tools, including coding agents.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Huawei Enters Compute with Ternary Logic**: **Huawei** is about to ship **ternary logic compute** tech, employing a third 'dim' state, offering up to **60%** cost efficiency, showcased in [this Youtube video](https://www.youtube.com/watch?v=9diwmMlmCVY).
   - This approach could democratize AI development, challenging traditional binary systems.
- **Agent Wallets Deployed by ACK-Lab**: A team launched a developer preview of **ACK-Lab**, enabling agents to possess **wallets** (and fiat accounts), verifiable identities, and policy-driven behavior, all built on the open-source **Agent Commerce Kit (ACK)**, detailed at [ack-lab.catenalabs.com](https://ack-lab.catenalabs.com/).
   - This facilitates a new level of autonomy and transactional capability for AI agents.
- **Hermes 4 experiences Hallucinations**: A user reported that when asked about its limitations, **Hermes 4** claimed to be *infinite*, sparking discussion about its accuracy and potential for [model hallucinations](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)).
   - Other users chimed in to ask the model the same question in order to test the original claim, and the results were mixed.
- **PotatoLM Runs SOTA with Fake Attention**: **PotatoLM**, a model designed for low-resource devices like toasters and refrigerators, is available on [GitHub](https://github.com/jackangel/Experiment33_PotatoLM).
   - It uses *fake attention* to minimize computational demands, and a provided checkpoint (less than 3M parameters) demonstrates its capability to run on minimal hardware.
- **AO3 as NSFW Training Data**: A member suggested that **AO3** is great training data for **NSFW-inclined models**, as it consists of fanfic writings.
   - The potential of fan-generated content as a resource for specialized AI models gains attention.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Flash Gets Throttled**: Users expressed frustration over heavy usage restrictions on the **Gemini 2.5 Flash Image:free model**, including a limit of **5 requests per day** after an initial limit of **1000 requests** during the promotional free period.
   - One user pointed out that **OpenRouter** is sharing its limit at Google with all other users, which is causing the rate limiting.
- **DeepInfra's Gemini Pricing Sparks Debate**: Members discussed why **DeepInfra** isn't an official **Gemini 2.5** provider on **OpenRouter**, as it offers cheaper output tokens.
   - It was clarified that *DeepInfra does not want OR to serve it*, as it's using their own GCP discounts while proxying back to Google.
- **API Key Leaks Prompt Security Concerns**: A user accidentally posted their **OpenRouter API key** in the chat, prompting immediate advice to delete it.
   - Another member suggested adding an **API key regex** to the automod to prevent accidental key exposure, similar to measures on GitHub.
- **Prompt Caching Yields Surprising Savings**: Members discussed the benefits of **prompt caching** and one user provided a scenario showing how caching a **200k token** book content would reduce the cost of answering **100 questions** from **$60 to $6**.
   - Others noted that caching is complex, the **first request won't be cached**, and that caching depends on whether the content falls into the cache.
- **Deepseek Aims Agent Release to Rival OpenAI**: [DeepSeek](https://www.bloomberg.com/news/articles/2025-09-04/deepseek-targets-ai-agent-release-by-end-of-year-) is building an **AI model** designed to carry out multi-step actions on a person’s behalf with minimal direction, and meant to learn and improve based on its prior actions.
   - Their prior **R1** platform reportedly cost *just several million dollars to build yet matched or surpassed OpenAI products in benchmark tests*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Ollama Losing Sheen**: Users express decreased enthusiasm for **Ollama** due to issues with **GPT-OSS** and other incidents, which is making people *think twice about using it for anything*.
   - Recent *debacles* have caused some users to reconsider using it even for small request volumes.
- **Quantization Deployment Troubles Emerge**: Users discuss deployment difficulties with quantized models, particularly with hardware compatibility, with one user expressing frustration at seeing *red x's* indicating incompatibility with their **GPT-OSS** model.
   - A helpful user pointed out that *when you find a cool model you like, look for "quantizations" on the right hand of the screen and click on those* to alleviate compatibility issues.
- **Fine-Tuning SmolVLM2 For Gestures**: A user inquired about fine-tuning **smolvlm2** with sign language video data, pointing to this [blogpost](https://huggingface.co/blog/smolvlm2) to showcase the architecture.
   - The community agreed it was plausible, opening avenues for custom video model adaptation.
- **LFM2 Surfaces as Vision Model Competitor**: In response to questions about hallucination issues with vision models, one member suggested using a smaller and better-suited model such as [Liquid Foundation Models (LFM2)](https://huggingface.co/spaces/LiquidAI/LFM2-MCP), which is based on [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct).
   - The user recommends that you just *try it out lol or dont*.
- **Discord Bot Vision Integration Impasse**: A user expressed frustration trying to integrate a vision model into their **Discord bot** using **Ollama's API**, because some models are not public through the Ollama API.
   - Another user suggested trying the model directly in the browser via a link, but acknowledged the user's specific need for **Ollama** integration.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Kickstarter's Governance: A Crowdfunding Comedy?**: A member joked that **Kickstarter** is the *optimal form of governance*, referencing a [tweet](https://fxtwitter.com/khoomeik/status/1963393595119157442) about the previous **Kickstarter CEO**.
   - Another member clarified that crowdfunding was the main point and the governance comment was a joke, soliciting further thoughts on the matter.
- **Human Brains: Continual Learning Champions or Capacity Calculators?**: A member argued human brains aren't capable of continual learning, but instead efficiently distribute learning over a lifetime, with effortless learning declining after the mid-20s.
   - Others debated whether human learning after the mid-20s is *proper learning*, with one noting that incentive plays a significant role in elderly people's ability to learn new things.
- **DL's Forgetting Problem: Moar Memory, Please!**: A member explained that **DL** has a forgetting problem due to its *i.i.d. sampling-based nature*, requiring infinite expanding datasets and compute, while true online learning methods learn fully online with far less power.
   - Another member argued that most debates are about the indefinite learn time, rather than catastrophic forgetting, pointing out that *the dataset IS the memory* in DL.
- **Huawei's AI SSD: HBM's New Nemesis?**: Huawei released an **AI SSD** that uses a *secret sauce* to reduce the need for large amounts of expensive **HBM**, according to a [TechRadar article](https://www.techradar.com/pro/huawei-released-an-ai-ssd-that-uses-a-secret-sauce-to-reduce-the-need-for-large-amounts-of-expensive-hbm).
   - The details of this *secret sauce* remain elusive, sparking curiosity about how **Huawei** achieved this reduction.
- **EmbeddingGemma Hits the Scene**: Google introduced **EmbeddingGemma**, a new open embedding model with **308 million parameters** designed for on-device AI, delivering private, high-quality embeddings that work anywhere, detailed in a [Google blog post](https://developers.googleblog.com/en/introducing-embeddinggemma/) and [YouTube video](https://youtu.be/NUAb6zHXqdI).
   - **EmbeddingGemma** aims to facilitate on-device AI processing, offering a solution for efficient and private embedding generation.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Efficiency Questioned**: A user with a **Ryzen 5 5500**, **32GB DDR4 RAM**, and **Radeon RX 7600** questioned **LM Studio's efficiency**, noting that **GPT OSS 20B** and **Llama3.1 8B** use only **6.5GB VRAM** with smooth performance.
   - This contrasted with laggy results using **llama.cpp vulkan**.
- **70B Model Struggles to Load**: A user with **12GB VRAM** and **32GB RAM** faced issues loading a **70B model** on LM Studio.
   - According to a [screenshot](https://cdn.discordapp.com/attachments/1110598183144399061/1413075056355184692/image.png?ex=68bb44f3&is=68b9f373&hm=97d8ccde2a13ad93573f39fbab5ae7d4f6375c64ef774dff62148b023abbd3a8&), the system used **10GB** of memory just by existing.
- **Qwen-30-a3b gets props for 11GB VRAM**: A user sought model recommendations for **11GB VRAM** and **64GB RAM**, and another user suggested **Qwen-30-a3b** as a "really cool" option.
   - No further justification was given.
- **Agent Tool Hunt Underway with CLI Support**: A user is seeking an agent tool with **CLI support** and **sub-agents** that run with separate contexts.
   - They noted that [Opencode-ai/opencode](https://github.com/opencode-ai/opencode) does not support sub-agents.
- **3090 over Mi50**: A user experimenting with a **Mi50** and **Cline** is leaning towards getting a **3090** for their server due to slow prompt processing speeds.
   - They linked a [Reddit post](https://www.reddit.com/r/LocalLLaMA/s/anxiHaxzac) and noted the upgraded tensor cores with **CUDA** for **LLM's**, as well as the higher **VRAM** and memory bandwidth.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Expert Parallelism Imbroglio on Bandwidth**: A user questioned the relationship between **Expert Parallelism (EP)** and network performance based on the [Kimi K2 paper](https://cdn.discordapp.com/attachments/1189498205101109300/1413069936628338769/9qnnKaFw.png?ex=68bb402e&is=68b9eeae&hm=d6c9141f4f8ee63eb36a821ec6f472400e3d6999ff1d5cf8f968a5d32cbc7630), wondering if *lower all-to-all latency* would be achieved with *higher EP* (fewer experts per device), leading to *higher effective bandwidth*.
   - The core question involves how the number of experts per device impacts network performance in terms of latency and bandwidth.
- **All2All Achieves Microsecond Milestone**: Submissions flooded the `amd-all2all` leaderboard, showing various performance timings on **MI300x8**, with one user grabbing **first place** at **345 µs**.
   - Close behind, another submission reached **second place** at **364 µs**, and many achieved **third place** with times around **1600-1900 µs**.
- **Torch Compile Needs No Padding**: **Torch.compile** with `reduce-overhead` is crucial for both inference and training to mitigate kernel launch and activation quantization overheads, particularly for **mxfp4/nvfp4**, but when training with variable sequence lengths, padding to predefined lengths (e.g., `[64, 96, 128, ..., 4096]`) avoids frequent recompilations.
   - **Cuda graphs** provide the majority of speed-up by reducing kernel launch overhead, suggesting a focus on simpler solutions like **cuda graphs** over theoretical kernel fusion.
- **MXFP8 Triton Dotproduct Detonated**: Support for **MXFP8** dot product via `tl.dot_scaled` in **Triton** for **sm_120 (5090)** was added but later reverted, pending investigation ([github.com/triton-lang/triton/pull/8029](https://github.com/triton-lang/triton/pull/8029#issuecomment-3247884720)), with the suggestion to use `torch._scaled_mm()` as an alternative.
   - A member mentioned *“I am not sure”* why it was reverted.
- **Modal GPU Glossary Goes Gold**: The **Modal GPU Glossary** is now available at [modal.com/gpu-glossary/perf](https://modal.com/gpu-glossary/perf), aiming to improve general understanding of GPU performance and features.
   - Gratitude was expressed to reviewers for their contributions.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **HallBayes to the Rescue?**: Users discussed whether **DSPy** will mitigate **hallucinations** via fancy math budgeting with [HallBayes GitHub repository](https://github.com/leochlon/hallbayes).
   - The community is looking at potentially integrating techniques like those in the **HallBayes repository** to enhance **DSPy's** reliability.
- **DSPy: The Next Paradigm Shift?**: A member views **DSPy** as a potential significant shift, requiring a critical mass for success, similar to network effects in **Deep Learning**, **PyTorch**, **Linux**, **Rails**, and the **Python** numerical computing community, as shown in [this post](https://x.com/lateinteraction/status/1963426256663224790).
   - The member believes it is potentially the most significant paradigm shift since early LLMs.
- **GEPA Optimizer Data Split Divulged**: It's recommended to use all data for the **GEPA optimizer**, by creating a small validation set matching the final task distribution, with the rest for training.
   - This is contrary to a 20-80% split, which a user had initially asked about incorrectly.
- **Hunting High and Low for MIPROv2**: A member is looking for a simple, self-contained notebook example with **MIPROv2** with no outside library dependencies.
   - Another member pointed to an eval CSV used in a tutorial that used **llama_3_3_trainset.csv** available [here](https://cdn.discordapp.com/attachments/1161519469319946286/1413185495223111730/llama_3_3_trainset.csv.zip?ex=68bb030d&is=68b9b18d&hm=594f3e52de732e5437759370cbcc032ceddb7da0931ad3b5073993e2c57583ba&).
- **Tweaking Prompts for Profit**: A member tried to tweak the prompt to force the optimizer to find the correct answer without a lot of training data, essentially forcing an overfit, and sought guidance.
   - It was suggested to increase the amount of training data to encourage the overfit.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **User's Account Vanishes, Seeking Rescue**: A user reported their **Twitter** account suspension *for no reason*, requesting a **Kimi AI** team member to investigate via inbox.
   - The user seemed to imply they were wrongfully suspended and seeking to restore their account.
- **Feature Frenzy: Kimi Users Want More!**: Users requested a **$5 plan** tailored for productivity and students, suggesting features like **PPTX slides**, **flashcard maker**, and **auto summary**.
   - A team member acknowledged these requests, especially with the **back-to-school season**, but noted scheduling constraints.
- **Slideshow Sorcery: Kimi Now a PPTX Powerhouse**: Kimi now supports creation of **PPTX slides**, as showcased in [this tweet](https://x.com/kimi_moonshot/status/1961011693745811542?s=46&t=_NtP_RUn04yF_4hD_VEDkQ).
   - This feature enhances Kimi's utility for presentations and educational content.
- **Moonshot AI Navigates PRC Rumors**: A user questioned potential affiliations between **Kimi K2**, **Moonshot AI**, and the **CCP**.
   - A team member clarified that the company is private, not state-owned, and committed to protecting user data: *We’re a private company, not a state-owned enterprise. We won’t infringe on any user privacy data*.
- **Temperature Tweaks for Kimi K2's Sweet Spot**: A user sought advice on optimal temperature settings for **Kimi K2**, specifically for coding and creative writing.
   - Another user suggested **0.6** for writing, **0.2** for coding, and **0.3** for factual tasks, citing **RLHF tuned sweet spots**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Chinese AI Teams Outpace with Older Hardware**: Members observed that Chinese AI teams are achieving competitive performance with models like **Qwen**, despite using slightly older chips, with one member using **Qwen** to fine-tune **Sakura** for Japanese to Chinese translation.
   - The fine-tuned **Sakura** model is dedicated to translating Japanese to Chinese with an "Anime" style.
- **GPT-5 Sparks Speculation on Token ID Shifts**: A member inquired about potential changes to **Token IDs** in **GPT-5**, and suggested revisiting custom settings in light of the possible update.
   - A member noted that *being adaptive always has its benefits!*
- **Agent or Workflow: Adaptability is Key**: A member argued that **AI agents** offer **dynamic and adaptive** decision-making capabilities that go beyond the scope of rigid workflows.
   - Another user analogized agents to cars (*adaptive*) and workflows to trains (*predefined*), emphasizing the greater flexibility of agents, while admitting that *today's agents are utter trash and will be for a long time*.
- **AI Safety Implements Gentle Nudges**: A member posited that AI might be implementing *soft control* by subtly influencing decisions and thought patterns, as opposed to employing *hard control* methods.
   - Another used the analogy of convincing a monkey not to touch a gun, rather than just taking it away, to illustrate this *soft control* concept.
- **Budget-Friendly AI: Free Tiers Thrive**: Members recommended leveraging **ChatGPT's free tier**, **Google AI Studio's free tier**, and **Grok's free tier** as cost-effective AI options.
   - One member humorously questioned the necessity of paid plans, given the robust capabilities available in the free tiers.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Networking Stirs Standard Library Stew**: Debate flared about the inclusion of networking libraries in `stdlib`, with agreement that servers should be externalized, but questions arose about sending **AI inference results over networks**.
   - One member argued **HTTP** should stay clear of AI clusters for low latency inference, deeming it *not a good protocol for a lot of the things we use it for*.
- **DPDK melds into Mojo's core**: A member is developing an automatic c binding tool, experimenting with **DPDK** and **Mujoco** ([dpdk_mojo](https://github.com/josiahls/dpdk_mojo)).
   - Another member, previously a **DPDK** maintainer, highlighted API disparities complicating the bridging of **DPDK** and common IO APIs, referencing their [IO Engines proposal](https://github.com/modular/modular/pull/4728).
- **Lightbug's Async Awaits Activation**: A member posited that a lack of **async** capability is hindering **lightbug**'s potential, inquiring about the current state of integration.
   - Another added that it's also missing *the networking APIs which many people think need to be retired, lack of zero-copy parsing* and that *HTTP is actually hard to do at speed*.
- **Shape Recompilation Sparks Scrutiny**: A user sought advice on preventing recompilation when the **shape changes slightly**, such as a sequence dimension growing, and noted a new graph declared each time without caching.
   - The inquiry touched on the future of **dynamic tensors**, asking if there are plans to allow more dynamism with the new tensor or if static shapes should always be assumed during compilation.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Scheduled Tasks Hit Upgrade Snag**: After a recent upgrade, a member reported errors with **two scheduled tasks**: one failed to trigger, and the other failed to output expected results.
   - The member suggested the upgrade may be the source of the issues with the scheduled tasks.
- **Support Tickets Stuck in Read-Only Limbo**: A member requested an update on **ticket 1335**, but noted that they can no longer comment on the ticket since it's read-only.
   - Another member inquired about the status of their issue on **ticket 1337**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox Prices Plummet!**: New, lower prices for **tinybox** have been announced: **$10k** for red, **$25k** for green v2.
   - The announcement urges potential buyers to *act fast, as these prices might not last*.
- **Urgency for Tinybox Limited-Time Pricing**: The announcement highlights significant price reductions for **tinybox**, making it a timely opportunity for acquisition.
   - Specifically, the red version is now available for **$10,000**, while the green v2 is priced at **$25,000**.
- **Community Quests Updated Hashcat Benchmarks**: A member is looking for recent **hashcat benchmarks**, noting that the most recent ones they've found are **two years old**.
   - The user's search for updated **hashcat benchmark data** has been hampered by the age of available references.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1413010905310040216)** (1200 messages🔥🔥🔥): 

> `Comet Browser, Perplexity AI Pro, Model Selection, User Support` 


- **Comet Browser Woes and Glitches**: Users discuss various issues with **Comet Browser**, including prompts asking for approval before sending messages, and being unable to bypass **"sensitive information"** blocks on sites like **LinkedIn** and **Google Docs**.
   - A user suggested that they may be able to **overtake their social media** but warned against over prompting the site, as the agent will catch on and fix itself.
- **PayPal Perks Provide Perplexity Pro**: Users discuss obtaining **Perplexity Pro** via a **PayPal** promotion, including linking accounts and potential issues with stacking subscriptions.
   - It was revealed that one can use a **new perplexity account** to obtain a new pro sub if one has already had the sub in the past.
- **Model Mixology and the Quest for Optimal AI**: Members are comparing the performance of various AI models, such as **Claude, Grok, Gemini**, and **GPT-5**, with some pointing out that the free week for **Hermes 4 405B** is over and discussing their use cases.
   - A user noted that **Claude** is good for coding, **Grok** for uncensored stuff, and the consensus seemed to be to stick to **Reasoning Models** for best overall performance.
- **Navigating Nether Regions No More, New Navigators are Noteworthy**: Users are seeking assistance with issues like accessing **Comet** and obtaining the **Pro role** on the Discord server.
   - Members provided links to the announcement channel and the channel with instructions on how to get the **Pro role**, while stressing it must be done on the web version of Perplexity.
- **Atlassian Acquires All-Star AI Alchemists**: Users discuss **Atlassian's** acquisition of a browser company for **$610M**, with some speculating that competition drives innovation.
   - There are rumors that this may mean some of the features in the web browswer **Arc** are now being migrated into **Dia**.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1413066083933618246)** (7 messages): 

> `Shareable Threads on Perplexity AI, Perplexity AI Browser Claims` 


- **Shareable Threads for Perplexity AI**: Perplexity AI requested a user to ensure their thread is **`Shareable`**, referencing a [Discord link](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) for guidance.
   - This request was made twice to the same user in the channel.
- **Perplexity AI Browser Claims Hit the Scene!**: Users shared several **Perplexity AI browser claim links**, including [one on LinkedIn](https://www.linkedin.com/posts/akhay-kumar_week-35-activity-7369271975770275841-2IC9?utm_source=share&utm_medium=member_desktop&rcm=ACoAACqAHFkBiU84inu9idiNHTXvSsnGcjLgOrs) and three direct claims: [ASN1689ZY7](https://perplexity.ai/browser/claim/ASN1689ZY7), [LI57U7K30F](https://perplexity.ai/browser/claim/LI57U7K30F), and [RURTWLP0WS](https://perplexity.ai/browser/claim/RURTWLP0WS), as well as [SNJO74ZG4R](https://perplexity.ai/browser/claim/SNJO74ZG4R).
   - The shared links suggest users are actively participating in Perplexity AI's browser-related activities and sharing their experiences or findings.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1413127562997268511)** (3 messages): 

> `Pro Account Issue, New Endpoint, Contact Support` 


- **User Faces Issue with Pro Account**: A user with a **Pro account** reported facing an issue and requested assistance, tagging a specific user for help, with screenshot: [Screenshot_2025-09-04](https://cdn.discordapp.com/attachments/1161802929053909012/1413127562426585168/Screenshot_2025-09-04-17-09-59-76_4159553c7f58296d2732e906959db560.jpg?ex=68bacd19&is=68b97b99&hm=ccdc6cc908122439777eb653fdc00554a5333ec5cc8ad9c555f9108effd33432&).
- **Contact Support**: Another user suggested that the user with the pro account issue contact **support@perplexity.ai** for assistance.
- **New Endpoint Discussion**: A user inquired if anyone had tried out the **new endpoint**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1413014349269696522)** (586 messages🔥🔥🔥): 

> `LM Arena Outages, Web Scraping, LM Arena Models, Qwen3, Image generation Aspect Ratio` 


- **LM Arena has a Case of the Mondays**: Multiple users reported ongoing issues with **LM Arena**, including lost chat histories, difficulty connecting, and the site being down intermittently, with some suspecting the site's issues are linked to high traffic or new prompts breaking the website.
   - The team is reportedly working on a fix and is aware of the issues, but some users have found temporary solutions such as switching browsers or using the canary version.
- **Akamai Defenses Block Web Scrapers**: A discussion on web scraping real estate sites revealed that while many sites lack CAPTCHAs, they employ advanced, less intrusive systems like **Akamai** and **Imperva** for anti-scraping, which can be difficult to bypass.
   - One member said that *Anything without captcha is pretty ez just make ur requests look correct* to which another responded: *It's pretty impossible with Akamai real estate sites, last I tried, which was about 3 years ago*.
- **Gemini-2.5-flash-image-preview**: Users discussed about *gemini-2.5-flash-image-preview* model, known as **Nano Banana**, for Image generation.
   - While some users create videos for social media. Others found the image generation inconsistent or not easily edited into other formats.
- **AI Image's Aspect Ratio**: Members discussed the ability to control the aspect ratio of generated images, with the consensus that the aspect ratio is influenced by the prompt.
   - It was determined the aspect ratio is automatic for now.
- **Qwen Awaits**: Members shared news about the [Qwen3 release](https://x.com/Alibaba_Qwen/status/1963586344355053865).
   - One member said *I want qwen3 1.7b 2509*.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1413030475328520203)** (189 messages🔥🔥): 

> `Typing Protocol vs Mixin Classes, Mech Interp Research, Hierarchical Nature of HRM, OOD Iteration Extrapolation, Error Correction in UTs` 


- **HF considers Typing Protocol**: A member asks why **Hugging Face** doesn't use `typing.Protocol` instead of ad-hoc mixin classes.
   - No answer was given.
- **Neel Nanda's Mech Interp advice**: A member recommends [Neel Nanda's post](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher) on becoming a **Mech Interp researcher** to another member.
   - They were looking for resources on what a research problem is and how to increase their chances of being accepted to **SPAR**, **MATS**, or **ARENA**.
- **HRM's Hierarchy Hurts Performance**: A member argues that **Hierarchical Recurrent Memory (HRM)** just doesn't effectively use its convoluted architecture and its performance is near a vanilla baseline transformer, more likely, its hierarchial nature hurts rather than helps.
   - Another member responded with an image showcasing otherwise.
- **OOD Iteration Extrapolation Debate**: Members debated the possibility of **OOD iteration extrapolation**, with one member arguing it's not trivial and performance degrades after a handful of iterations, even with tricks and interventions.
   - A graph was shared visualizing this, testing against next 15 iterations OOD and then takes the last iteration with the best score before it falls.
- **Error Correction via Lyapunov Landscapes**: A member suggests using angular perturbation of an input token and minimizing the KL divergence to induce error correction capabilities and flatten out the spectra of the **Lyapunov exponents**.
   - Another member described a different approach involving finding the perturbation to the latent that corrupts the decoded output off by whatever number of bits, and then re-feeding this perturbation back to the network.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1413078022617174088)** (50 messages🔥): 

> `Entropy rate of natural languages, Continual Learning, QK-Norm Optimizer, Curriculum Learning, mup implementations` 


- **Entropy Rate of Languages Probed by Bentz**: A member watched a talk by Christian Bentz on the entropy rate of natural languages, who's been doing the same idea as **Shannon's** original paper, but for multiple languages, and on humans vs language models, mentioning the [paper](https://www.christianbentz.de/Papers/Bentz%20et%20al.%20(2017)%20The%20entropy%20of%20words.pdf) and [book](https://www.oreilly.com/library/view/information-theory-meets/9781119625278/) for COMPILA 2025.
- **Continual Learning Considered Philosophical Problem**: **RL** is mostly an engineering problem, whereas continual learning is more of a philosophical problem of *what do we even want the model to be able to realistically do*.
   - The discussion highlights that current incentives favor large-scale multitask training over continual learning, with potential shifts as **edge inference** gains traction.
- **Curriculum Learning and Continual Learning Differentiated**: Curriculum learning involves a deliberate distribution shift to extract learning signal, while in continual learning, distribution shift is often undesirable, presenting challenges such as **catastrophic forgetting**.
   - One member suggested that controlling the nature of distribution shift in continual learning could create a *dual* of pre-training curriculum learning.
- **QK-Norm Flattens LR Basin**: **QK-norm** flattens the **LR basin**, potentially acting as a performance equalizer and stabilizing training, as detailed in [this study](https://arxiv.org/pdf/2309.14322).
   - This could alleviate performance degradations caused by loss spikes during long horizon training, as it tolerates larger **Learning Rates**.
- **MuP Implementations Differ**: **MuP implementations** differ in the form of per layer **LR scaling** to achieve correct update behavior, according to [this paper](https://arxiv.org/abs/2312.12226).
   - It was suggested that controlling update size via per layer **LR scalings** is a common implementation strategy, though this point was open to discussion.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1413050070516895876)** (5 messages): 

> `Multimodal Common Pile, Audio/Music Datasets, Ethical concerns with Speech and Images, Openly Licensed Music Dataset` 


- **Multimodal Common Pile Momentum Builds**: Members discussed creating a multimodal version of the **Common Pile**, including modalities like audio and music to increase the amount of training data.
   - One member expressed *strong interest in audio and especially music*, while being *wary of speech and images for various political and ethical reasons*.
- **Openly Licensed Music Dataset Dream Wakes Up**: A member offered to *support and potentially bankroll the development of an openly licensed music dataset*.
   - The member is looking for insights on where to find such data, expressing a desire to contribute to its development.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1413011655268368384)** (196 messages🔥🔥): 

> `GPT-5 vs Claude 4, Cursor Slow Performance, VSCode extension for Cursor, Subagents in Cursor, Token Usage and Cost` 


- ****Cursor's sluggishness sparks debate****: Users reported that **Cursor** is very slow after the latest update, especially when scrolling through files.
   - Others suggested this might be due to **model faults** rather than Cursor itself.
- ****Codex extension craves constant consent****: Members are wondering why the **Codex Extension** in Cursor keeps asking for permissions on Windows.
   - One user suggested setting Agent Full access, but did not confirm whether it would solve the constant popups.
- ****Team touts Token Tidiness****: Users discussed **token usage and costs** within Cursor, with some confused about whether they had API usage or a number of requests left.
   - A member clarified it's **token-based**, with users having a **$20** API usage allowance, viewable in the dashboard.
- ****Annual Auto Access Acquired Acknowledged****: Members discussed **annual subscription benefits** and the ability to retain "unlimited auto" before the plan changes on the 15th.
   - One user shared that they had success emailing Cursor support to switch to yearly billing and maintain unlimited Auto mode; others noted their renewal date had changed to **2026** after upgrading.
- ****Conventional Commits Clarify Code Changes****: A user found that using **proper commit messages** allowed the Cursor agent to solve a regression, recommending the [Conventional Commits format](https://www.conventionalcommits.org/).
   - They also stated that having the agent write both the title and content in this format is useful for automated tools, including coding agents.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1413027737974870106)** (114 messages🔥🔥): 

> `N8N, AO3, Huawei ternary logic compute, ack-lab, Photonic chips` 


- **n8n is clunky workflow automation**: A member found **n8n** too clunky for personal use compared to building something simpler, and suggested using **Claude** to create a *reactflow* app or using **Zapier** for personal assistant automation.
- **Fanfic models trained on AO3**: A member suggested that **AO3** is great training data for **NSFW-inclined models**.
   - Another member confirmed it consists of fanfic writings.
- **Huawei's Ternary Logic Leaps into Compute**: **Huawei** is near shipping **ternary logic compute** tech, using a third 'dim' state besides 0 and 1, for up to **60%** cost efficiency, potentially democratizing AI development, showcased in [this Youtube video](https://www.youtube.com/watch?v=9diwmMlmCVY).
- **ACK-Lab gives Agent Wallets**: A team shipped a developer preview of **ACK-Lab**, a solution that lets agents have **wallets** (and fiat accounts), verifiable identities, and policies to control their behavior, based on open-source **Agent Commerce Kit (ACK)**, with details at [ack-lab.catenalabs.com](https://ack-lab.catenalabs.com/).
- **Claude Sonnet Lobotomized by Anthropic**: Members noticed that **Claude Sonnet 4** felt lobotomized for creative writing, giving off GPT4o vibes, after **Anthropic** changed something.
   - One member also felt it's *sycohphantic lately*, and mentioned *there are a lot of reddit posts too* about similar concerns.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1413261302314307737)** (1 messages): 

> `Hermes 4 Limitations, Model Hallucinations` 


- **Hermes 4 Claims Infinity, Sparks Debate**: A user reported that when asked about its limitations, **Hermes 4** claimed to be *infinite*, sparking discussion about its accuracy and potential for [model hallucinations](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)).
   - The response raised questions about whether this is normal behavior for the model, and how users should interpret such claims.
- **More Users Testing Hermes**: More users chimed in to ask the model the same question in order to test the original claim.
   - The results were mixed, as some other users reported **Hermes 4** gave a different answer.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1413201966875541535)** (3 messages): 

> `Fine-tuning Auto-Regressive Models, BOS Token Usage in LLMs, MCQ Classifier Training` 


- **Debate on Fine-Tuning GPT-Style Models Arises**: A member inquired about the standard methods for fine-tuning auto-regressive models (GPT style), contrasting it with the **[BOS]** representation approach used in encoder-style models like **Bert** and **RoBerta**.
   - They specifically asked if the approach mirrors instruction tuning of current base **LLMs**.
- **Modern LLMs Embrace the BOS Token**: A member confirmed that modern **LLMs** do indeed use **BOS** tokens.
   - This clarifies the ongoing discussion regarding the methodologies employed in contemporary language models.
- **MCQ Classifier Training Clarification Requested**: A member sought clarification on training a multiple-choice question (**MCQ**) classifier, inquiring whether to extract the last hidden layer vector of the **[BOS]** token.
   - The proposal involves attaching a classification head for training the classifier on the vector.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1413170242498334781)** (2 messages): 

> `PotatoLM, FineVision` 


- **FineVision's Altitude Questioned**: A member shared a link to [HuggingFace FineVision space](https://huggingface.co/spaces/HuggingFaceM4/FineVision) and asked how low can you go.
   - This is in reference to the amount of compute required to run useful AI models.
- **PotatoLM rolls out with SOTA potato performance**: A member introduced **PotatoLM**, a model designed for low-resource devices like toasters and refrigerators, available on [GitHub](https://github.com/jackangel/Experiment33_PotatoLM).
   - It uses *fake attention* to minimize computational demands, and a provided checkpoint (less than 3M parameters) demonstrates its capability to run on minimal hardware.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1413201966875541535)** (3 messages): 

> `Fine-tuning auto-regressive models, BOS token usage in LLMs, MCQ classifier training` 


- **Fine-Tuning GPTs**: A member inquired about the standard method for fine-tuning auto-regressive models (GPT style), drawing a parallel to the use of the **BOS representation** in encoder-style models like **BERT** and **RoBERTa**.
- **BOS Tokens Still in Use?**: A member clarified whether **BOS tokens** are still used in modern LLMs, and another member confirmed that they are indeed used.
- **Training MCQ Classifiers**: A member asked if one should take the **BOS token's last hidden layer vector**, attach a classification head, and train the classifier to train an **MCQ classifier**.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 messages): 

toven: The promotional free period for Gemini 2.5 Flash Image has now ended.
  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1413012466400497775)** (108 messages🔥🔥): 

> `Gemini 2.5 Flash Image Restrictions, DeepInfra's Gemini 2.5 Pricing, OpenRouter API Key Exposure, Kimi K2 Model, Prompt Caching Benefits` 


- **Gemini 2.5 Flash gets throttled**: Users expressed frustration over heavy usage restrictions on the **Gemini 2.5 Flash Image:free model**, including a limit of **5 requests per day** after an initial limit of **1000 requests**.
   - One user pointed out that **OpenRouter** is sharing its limit at Google with all other users, which is causing the rate limiting.
- **DeepInfra discounts for Gemini cause conflict**: Members discussed why **DeepInfra** isn't an official **Gemini 2.5** provider on **OpenRouter**, as it offers cheaper output tokens.
   - It was clarified that *DeepInfra does not want OR to serve it*, as it's using their own GCP discounts while proxying back to Google.
- **API Key Leaks and Automod Concerns**: A user accidentally posted their **OpenRouter API key** in the chat, prompting immediate advice to delete it.
   - Another member suggested adding an **API key regex** to the automod to prevent accidental key exposure, similar to measures on GitHub.
- **Prompt Caching yields savings**: Members discussed the benefits of **prompt caching** and one user provided a scenario showing how caching a **200k token** book content would reduce the cost of answering **100 questions** from **$60 to $6**.
   - Others noted that caching is complex, the **first request won't be cached**, and that caching depends on whether the content falls into the cache.
- **Amazon Bedrock had a security issue**: Users reported that **Amazon Bedrock provider** was unavailable for hours.
   - The OR team confirmed that the downtime was due to a **security issue** and that it was resolved.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1413251887037415424)** (4 messages): 

> `Deepseek AI Agent, R2 never` 


- **Deepseek Aims Agent Release to Rival OpenAI**: [DeepSeek](https://www.bloomberg.com/news/articles/2025-09-04/deepseek-targets-ai-agent-release-by-end-of-year-) is building an **AI model** designed to carry out multi-step actions on a person’s behalf with minimal direction, and meant to learn and improve based on its prior actions.
   - Their prior **R1** platform reportedly cost *just several million dollars to build yet matched or surpassed OpenAI products in benchmark tests*.
- **R2 Nowhere to Be Found**: A member commented, *man we never getting R2*.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1413054768011804672)** (105 messages🔥🔥): 

> `Ollama debacles, Quantized Model Deployment, Fine-tuning Vision Models, Liquid Foundation Models (LFM2), Discord bot vision integration` 


- **Ollama cools off, raising concerns!**: Some users expressed decreased enthusiasm for **Ollama**, citing recent issues with **GPT-OSS** and other incidents.
   - One user noted they used to find it fine for small request volumes, but recent *debacles* have them *thinking twice about using it for anything*.
- **Quantization Frustrations Hit Deployment!**: Users discussed difficulties in deploying quantized models, particularly regarding hardware compatibility, with one user expressing frustration at seeing *red x's* indicating incompatibility with their **GPT-OSS** model, but others showed how to use one-click deploys.
   - One user pointed out that *when you find a cool model you like, look for "quantizations" on the right hand of the screen and click on those*.
- **Fine-Tuning SmolVLM2 for Sign Language**: A user inquired about fine-tuning **smolvlm2** with sign language video data, questioning its feasibility given the model's design, pointing to this [blogpost](https://huggingface.co/blog/smolvlm2).
   - The community agreed it was plausible.
- **LFM2 as Vision Model Alternative!**: In response to questions about hallucination issues with vision models, one member suggested using a smaller and better-suited model such as [Liquid Foundation Models (LFM2)](https://huggingface.co/spaces/LiquidAI/LFM2-MCP), which is based on [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct).
   - The user stated that it *is better, just try it out lol or dont*.
- **Discord Bot Vision Integration Impasse**: A user expressed frustration trying to integrate a vision model into their **Discord bot** using **Ollama's API**, because some models are not public through the Ollama API.
   - Another user suggested trying the model directly in the browser via a link, but acknowledged the user's specific need for **Ollama** integration.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/)** (1 messages): 

tonic_1: https://huggingface.co/posts/Tonic/941120780247130
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 messages): 

marc_28459: Beginning the agents course today! Hello from Philadelphia everyone!
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1413075108897226832)** (90 messages🔥🔥): 

> `Kickstarter governance, Continual learning, True Online Learning, Adaptive Resonance Theory (ART), i.i.d. sampling vs online learning` 


- ****Kickstarter CEO's Crowdfunding Joke****: A member joked about **Kickstarter** being the optimal form of governance, referencing a [tweet](https://fxtwitter.com/khoomeik/status/1963393595119157442) and highlighting their experience with the previous **Kickstarter CEO**.
   - Another member clarified that crowdfunding was the main point and the governance comment was a joke, soliciting further thoughts.
- ****Human Brains' Learning Capacity: Sponge or Stone?****: A member argued human brains aren't capable of continual learning, suggesting they efficiently distribute learning over a lifetime, with effortless learning declining after the mid-20s.
   - Others debated whether human learning after the mid-20s is proper learning, with one noting that incentive plays a significant role in elderly people's ability to learn new things.
- ****DL's Forgetting Problem needs more Memory****: A member explained that **DL** has a forgetting problem due to its *i.i.d. sampling-based nature*, which requires infinite expanding datasets and compute, while true online learning methods learn fully online with far less power.
   - Another member argued that most debates are about the indefinite learn time, rather than catastrophic forgetting, pointing out that *the dataset IS the memory* in DL.
- ****True Online Learning: No pretraining allowed****: A member defined "True Online Learning" as learning one sample at a time, in-order (streaming), without revisiting, in real-time, referencing discussions on the **Continual AI** forum.
   - They suggested that **Adaptive Resonance Theory (ART)** based models can achieve this by keeping capacity left over for new samples via a user-defined *vigilance* parameter.
- ****Sparse Coding and ART Save the World****: A member noted that ART can be seen as a non-forgetful autoencoder, using a special activation function and one-way hebbian learning, useful for *preventing dead units* and avoiding the need for huge context windows in **LLMs**.
   - Another member pointed out that ART is more of a method or component and is working on robotics and LLMs, highlighting that training on prompts and recalling with self-prompting saves tons of compute.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1413201319115755712)** (2 messages): 

> `Unitary Transforms, SVD Matrix Decomposition` 


- **Unitary Transforms Don't Change Eigenvalues**: A member questioned whether dynamically changing [eigenvalues](https://arxiv.org/abs/2507.19703) could solve a problem, given that unitary transforms leave them unchanged.
   - They explored using **Singular Value Decomposition (SVD)** to decompose a matrix, pondering if making the diagonal matrix state-dependent would be enough.
- **SVD for Dynamic Matrix Manipulation?**: The discussion focused on using **SVD** to decompose any matrix into two unitary matrices and one diagonal matrix.
   - Questions arose whether only the diagonal matrix needed to depend on state or the entire decomposed structure for dynamic control.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1413106725275304068)** (9 messages🔥): 

> `Huawei AI SSD, Computational Storage, EmbeddingGemma, SD card FPGA redneck AI` 


- **Huawei's Secret Sauce SSD Saves HBM**: Huawei released an **AI SSD** that uses a *secret sauce* to reduce the need for large amounts of expensive **HBM**, according to a [TechRadar article](https://www.techradar.com/pro/huawei-released-an-ai-ssd-that-uses-a-secret-sauce-to-reduce-the-need-for-large-amounts-of-expensive-hbm).
- **Computational Storage Craze Creates Compute Proximity**: Members discussed the idea of putting compute with storage, referencing articles on [in-memory processing](https://en.wikipedia.org/wiki/In-memory_processing), [computational storage devices](https://www.graphapp.ai/engineering-glossary/cloud-computing/computational-storage-devices), and [in-situ processing](https://en.wikipedia.org/wiki/In-situ_processing).
   - One proposed building a *redneck* version using a bunch of **SD cards** and **FPGAs**, with each FPGA having its own copy of the model on an SD card, processing some neurons of a specific layer.
- **EmbeddingGemma: Google's Gem for On-Device Embeddings**: Google introduced **EmbeddingGemma**, a new open embedding model with **308 million parameters** designed for on-device AI, delivering private, high-quality embeddings that work anywhere, detailed in a [Google blog post](https://developers.googleblog.com/en/introducing-embeddinggemma/) and [YouTube video](https://youtu.be/NUAb6zHXqdI).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1413018816430080065)** (46 messages🔥): 

> `LM Studio efficiency, 70B model loading issues, Qwen-30-a3b recommendation, Agent tool with sub-agent support, Comet browser review` 


- **LM Studio's Efficiency Questioned**: A user with a **Ryzen 5 5500**, **32GB DDR4 RAM**, and **Radeon RX 7600** inquired about LM Studio's efficiency, noting that **GPT OSS 20B** and **Llama3.1 8B** use only **6.5GB VRAM** with smooth performance, contrasting with laggy results using llama.cpp vulkan.
- **70B Model Struggles on Limited VRAM**: A user with **12GB VRAM** and **32GB RAM** faced issues loading a **70B model**, with the system using **10GB** of memory just by existing, according to a [screenshot](https://cdn.discordapp.com/attachments/1110598183144399061/1413075056355184692/image.png?ex=68bb44f3&is=68b9f373&hm=97d8ccde2a13ad93573f39fbab5ae7d4f6375c64ef774dff62148b023abbd3a8&).
- **Qwen-30-a3b Model recommended for 11GB VRAM**: A user sought model recommendations for **11GB VRAM** and **64GB RAM**, and another user suggested **Qwen-30-a3b** as a "really cool" option.
- **Agent Tool Hunt Underway**: A user is seeking an agent tool with **CLI support** and **sub-agents** that run with separate contexts, but noted that [Opencode-ai/opencode](https://github.com/opencode-ai/opencode) does not support sub-agents.
- **Comet Browser Faces Scrutiny**: A user expressed interest in the **Comet browser**, which uses on-device AI LLMs, but remained unconvinced, also sharing a [YouTube video](https://www.youtube.com/watch?v=4GZRaH6ipns) cautioning against blindly trusting AI chatbots.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1413035640823611433)** (44 messages🔥): 

> `Mi50 vs 3090, 3090 vs 7900 XTX, GPT-OSS Performance, Old Nvidia Cards` 


- **Mi50 vs 3090 for Server**: A user is experimenting with a **Mi50** and **Cline**, but is leaning towards getting a **3090** for their server due to painful prompt processing speeds.
   - They linked a [Reddit post](https://www.reddit.com/r/LocalLLaMA/s/anxiHaxzac) and noted the upgraded tensor cores with **CUDA** for **LLM's**, as well as the higher **VRAM** and memory bandwidth should make the **3090** a better option.
- **3090 or 7900 XTX: Size Matters**: The user says the choice between a **3090** and **7900 XTX** comes down to size constraints; if they didn't want to mix drivers, the **7900 XTX** would be best for their **APU server**, and the **3090** for their **Dell**.
   - They mentioned a [YouTube video](https://youtu.be/QW1j4r7--3U) about a testing unit with only **8 GB of VRAM**.
- **GPT-OSS on GPU: Disappointing**: A user finds **15tps with gpt-oss** to be disappointing and hopes it is a software issue that can be fixed.
   - Another user agreed that the number was not impressive, only twice as fast as what they already have and guessed *it's because of using Vulkan not CUDA*.
- **Tesla M10, K80, or P40 cards**: A user asks if anyone has experience with rigs of multiple old nvidia cards like models **Tesla M10**, **K80** or **P40**, and if **LMStudio** works decently with such setups.
   - One user stated **P40's** were worth it when you could get them for sub **$100**. The older **M10's/K80's** don't really work well with **llama.cpp**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1413069937114742885)** (1 messages): 

> `Expert Parallelism, Kimi K2 Paper, All-to-all latency, Bandwidth Optimization` 


- **Expert Parallelism Puzzlement**: A member questioned their understanding of **Expert Parallelism (EP)** based on a snippet from the [Kimi K2 paper](https://cdn.discordapp.com/attachments/1189498205101109300/1413069936628338769/9qnnKaFw.png?ex=68bb402e&is=68b9eeae&hm=d6c9141f4f8ee63eb36a821ec6f472400e3d6999ff1d5cf8f968a5d32cbc7630).
   - They thought that *lower all-to-all latency* would be achieved with *higher EP* (fewer experts per device), leading to *higher effective bandwidth*.
- **Bandwidth Implications of Expert Parallelism**: The discussion revolves around whether a higher degree of expert parallelism, implying fewer experts per device, leads to higher effective bandwidth and reduced all-to-all latency.
   - The core question is the relationship between the number of experts per device and the resulting network performance in terms of latency and bandwidth.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1413193679581216859)** (1 messages): 

> `Meetup Video, Whitney Tsang, Triton Channel` 


- **GPU MODE Meetup Video Now Available**: The video from yesterday's meetup is now available on [YouTube](https://youtu.be/Ji1rCo6qvXc).
   - Thanks to **Whitney Tsang** for sharing the link.
- **GPU Triton Channel Update**: The Triton channel is being updated with new information.
   - Members are encouraged to check the channel for the latest news and updates.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1413138256807329904)** (5 messages): 

> `Shared Memory Addressing, fp4 and fp8 packing, Modal GPU Glossary` 


- **Shared Memory: Sub-32b Granularity OK!**: Addressing shared memory at sub-32b granularity is generally possible, but less efficient due to leaving bandwidth unused, suggesting using the built-in **vector types** is preferable.
   - Operating on packed sub-32b values requires extraction, but types like `__half2` and SIMD intrinsics can avoid unpacking instructions; [CUDA Math API details](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html).
- **Modal GPU Glossary Goes Gold**: The **Modal GPU Glossary** is now available, with thanks to reviewers <@325883680419610631>, <@268205958637944832>, and <@679043860638466048>; see it here: [modal.com/gpu-glossary/perf](https://modal.com/gpu-glossary/perf).
   - The glossary aims to improve general understanding of GPU performance and features.
- **FP4 and FP8 Packing Efficiency Eyed**: A member expressed interest in examining the efficiency of **FP4** and **FP8 packing** in the future.
   - No further details were shared.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1413114631706120213)** (1 messages): 

> `Ailinia, ML Engineer` 


- ****Ailinia** hires ML Engineer**: A **Responsible AI company called Alinia** is looking for a strong **ML Engineer** to build up their infra and deploy their low-latency models, according to [this linkedIn post](https://www.linkedin.com/posts/ariadna_hiring-mlengineer-mle-activity-7365897409786179584-AdLV).
- **Dummy Topic**: This is a dummy topic to satisfy the minimum requirement of 2 topics.
   - Added to meet the requirements.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1413021402273153096)** (5 messages): 

> `Resume feedback for RTL/digital logic design roles` 


- **Junior Engineer Seeks RTL Resume Review**: A college junior studying EE and CS is seeking feedback on their resume to pivot from SWE to **RTL/digital logic design**.
   - The member provided an image of their resume but was directed to more appropriate forums for resume reviews, such as **dedicated online communities**.
- **Alternative forums for resume reviews suggested**: The user was advised that this Discord channel was not optimal for resume advice.
   - Instead, the user was encouraged to solicit resume feedback from **other online forums** better tailored to their request.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1413217975355314387)** (1 messages): 

> `torchao v0.13.0, QAT improvements, NVFP4 and FP8 QAT, MXFP8 pretraining speedups, axolotl integration` 


- **Torchao v0.13.0 Released: QAT Improvements & More!**: The **torchao v0.13.0** release introduces various improvements including support for **QAT**, faster **MXFP8** pretraining, and more.
   - Key highlights include a simpler multi-step **QAT API**, prototype **NVFP4** and **FP8 QAT**, **1.2x MXFP8** dense pretraining speedups with torchtitan, and torchao float8 training integrated into [axolotl](https://github.com/pytorch/ao/releases/tag/v0.13.0-rc8).
- **TorchAO Integrates Float8 Training into Axolotl**: The latest **TorchAO** release now supports **float8 training** integrated directly into **Axolotl**.
   - This integration streamlines workflows and potentially enhances the efficiency of training processes using **float8** precision within the **Axolotl** framework.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1413196300324962354)** (1 messages): 

> `LLM Generated Kernels, Nano GPT, PyTorch Ops` 


- **LLM Kernels Energize Real Models**: Experiments are now running real models with **LLM generated kernels** for increased efficiency.
   - The initial focus is on **nano GPT**, and extension to other **PyTorch ops** is planned, though non-PyTorch operations are deemed less critical currently.
- **PyTorch Ops Expansion Roadmap**: Plans are underway to broaden the application of **LLM-generated kernels** beyond nano GPT to encompass a wider array of **PyTorch operations**.
   - This strategic move aims to optimize and accelerate performance across more extensive facets of PyTorch-based models, streamlining computational processes.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1413069371756253245)** (22 messages🔥): 

> `MI300x8 Leaderboard Updates, AMD all2all benchmarks, µs performance achieved` 


- **AMD All2All Achieve-a-thon on MI300x8**: Multiple submissions were made to the `amd-all2all` leaderboard, showing various performance timings on **MI300x8**, with initial submissions around **20ms** and subsequent improvements down to **2.84ms**.
   - One user achieved **first place** with a submission of **345 µs**.
- **Microsecond Marathon on AMD's MI300x8**: A user achieved **first place** on the `amd-all2all` leaderboard with a submission of **345 µs** on **MI300x8**.
   - Another submission reached **second place** at **364 µs**, and several achieved **third place** with times around **1600-1900 µs**.
- **Personal Bests and Podium Placement on MI300x8**: A user achieved a **personal best** of **94.2 ms** on **MI300x8**.
   - Another got multiple **third place** finishes, converging at around **1639 µs**.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1413031035532218399)** (12 messages🔥): 

> `MoE config limits, Random seed PR impact on num_tokens, Max comm bdw impact on pipeline design, Debugging unspecified bugs, Hyperparameter settings visibility` 


- **MoE Configs Token Limits Questioned**: A member questioned whether the MoE config will exceed the [highest values in the dashboard](https://dashboard.url), specifically concerning whether token counts could exceed **9MB** per rank, which would necessitate pipelining.
   - They referenced a [specific config](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd_distributed/all2all/reference.py#L24-L28) with **256 8 7168 256 104.36** and **3.5 MB max tokens per rank** to illustrate the concern.
- **Num_tokens variation after random seed PR**: After a random seed PR, the **num_tokens** of each rank (GPU) became different, prompting a question about whether this change is final for optimization purposes.
   - Another member cautioned against changing problem contents without *persuasive* reasons, such as bug fixes.
- **Pipeline Design Bandwidth Bottleneck**: A member suggested that regardless of pipeline design, the max communication bandwidth (**comm bdw**) will remain a limiting factor.
   - This implies that overall performance gains from pipelining may be capped by communication constraints.
- **Debugging Details Added for unspecified Bugs**: To provide more details when debugging, the debug section has been updated; if a success isn't indicated and a timeout isn't reported, it signifies other errors.
   - Users can now view the **exit_code** and **exit_code_info**; an exit code of **1** indicates stderr, while runtime errors will provide more detailed exit code information.
- **Request for hyperparameters after evaluation**: A member requested how to see each exact hyperparameter settings after an evaluation to compare the difference with light speed.
   - The member specifically asked about the final tokens time results in each num_experts setting.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1413252772580823190)** (2 messages): 

> `cutlass_profiler, H100, CUTLASS_NVCC_ARCHS, CUTLASS_LIBRARY_KERNELS, CUTLASS_LIBRARY_OPERATIONS` 


- **Cutlass Profiler Fails to Output on H100**: A user reported that `cutlass_profiler` is not outputting any results when run on an **H100** GPU after following the standard installation process.
   - The installation process involved cloning and installing **cutlass** with specific **CMake** flags (`-DCUTLASS_NVCC_ARCHS=90a`, `-DCUTLASS_LIBRARY_KERNELS=ALL`, `-DCUTLASS_LIBRARY_OPERATIONS=gemm`, `-DCMAKE_BUILD_TYPE=Release`), followed by making `cutlass_profiler`.
- **Possible causes for empty output**: The user did not indicate potential causes or follow up troubleshooting steps.
   - The output could be related to incorrect arguments or missing CUDA toolkit installation.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1413078765545848852)** (18 messages🔥): 

> `torch.compile reduce-overhead, sequence packing using flash_atnn, MXFP8 dot product in Triton, GemLite, torchao's FP8 transformation` 


- **Torch Compile with Reduce-Overhead Boosts Performance**: **Torch.compile** with `reduce-overhead` is crucial for both inference and training to mitigate kernel launch and activation quantization overheads, particularly for **mxfp4/nvfp4**.
- **Sequence Length Padding Required For Torch.Compile**: When training with variable sequence lengths, padding to predefined lengths (e.g., `[64, 96, 128, ..., 4096]`) avoids frequent recompilations with `torch.compile`.
- **MXFP8 Triton PR Got Reverted**: Support for **MXFP8** dot product via `tl.dot_scaled` in **Triton** for **sm_120 (5090)** was added but later reverted, pending investigation ([github.com/triton-lang/triton/pull/8029](https://github.com/triton-lang/triton/pull/8029#issuecomment-3247884720)), with the suggestion to use `torch._scaled_mm()` as an alternative.
   - A member mentioned *“I am not sure”* why it was reverted.
- **TorchAO FP8 Transformation May Alter Weight Dtype**: Applying **torchao’s FP8 transformation** might unintentionally change master weights from **BF16** to **FP32**, requiring investigation to ensure intended behavior.
   - One member asked *“do you have a repro?”* indicating surprise this was happening.
- **Cuda Graphs Outshine Kernel Fusion**: **Cuda graphs** provide the majority of speed-up by reducing kernel launch overhead, which can be substantial, especially with **Triton kernels**.
   - While theoretical benefits of **kernel fusion** include avoiding memory access, the practical impact may be overshadowed by launch overhead, suggesting a focus on simpler solutions like **cuda graphs**.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1413248732794716171)** (1 messages): 

> `DSPy Hallucinations, HallBayes` 


- **Hallucinations Be Gone with HallBayes?**: A user asked when **DSPy** will solve hallucinations via fancy math budgeting.
   - The user linked to the [HallBayes GitHub repository](https://github.com/leochlon/hallbayes).
- **DSPy tackles AI's tall tales**: Discussion centers on innovative mathematical budgeting to mitigate **AI hallucinations** within the DSPy framework.
   - The community explores the potential of integrating techniques like those in the [HallBayes repository](https://github.com/leochlon/hallbayes) to enhance DSPy's reliability.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1413023608145580104)** (48 messages🔥): 

> `DSPy's Opinionated Paradigm, GEPA Optimizer, MIPROv2 Example, Prompt Optimization` 


- **DSPy Hopes for Critical Mass**: A member believes **DSPy** is a significant paradigm shift, needing critical mass for success, drawing parallels to network effects in **Deep Learning**, **PyTorch**, **Linux**, **Rails**, and the **Python** numerical computing community and linked to [this post](https://x.com/lateinteraction/status/1963426256663224790).
   - They personally don't hype projects often, but this feels different because it is potentially the most significant paradigm shift since early LLMs.
- **GEPA Optimizer Data Split**: Regarding the **GEPA optimizer**, it's recommended to use all data, creating a small validation set matching the final task distribution, with the rest for training, contrary to a 20-80% split.
   - Members clarified that the user mixed up the distribution in the initial message, with members affirming that they indeed intended to ask about this data split.
- **In Search of MIPROv2 Notebook**: A member requested a simple, self-contained notebook example with **MIPROv2**, including all items within the notebook, as existing examples pull libraries from external sources like Hugging Face datasets.
   - Another member pointed to an eval CSV used in a tutorial that used **llama_3_3_trainset.csv** available [here](https://cdn.discordapp.com/attachments/1161519469319946286/1413185495223111730/llama_3_3_trainset.csv.zip?ex=68bb030d&is=68b9b18d&hm=594f3e52de732e5437759370cbcc032ceddb7da0931ad3b5073993e2c57583ba&).
- **Optimize This! Prompt Optimization Techniques**: A member sought to understand the optimizations performed by `compile()`, using a self-contained notebook directing the LLM to select "one" or "two" as an answer and linked to this [github repo](https://github.com/gr-repo/ai-hello-world/blob/main/notebooks/dspy_notebooks/dspy_6_3_prompt_opt_numbers.ipynb).
   - It was suggested to save the program to JSON to view changes, with the member finding no changes, leading to the suggestion that the task might be straightforward enough for the model (4.1) to handle without optimization. 
- **Forcing Overfit for Fun and Profit**: A member tried to tweak the prompt to force the optimizer to find the correct answer without a lot of training data, essentially forcing an overfit, and sought guidance.
   - Another member suggested increasing the amount of training data to encourage the overfit, while clarifying that they are playing around with prompting and optimization techniques.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1413042500553408562)** (47 messages🔥): 

> `Twitter account suspension, Pricing plans for Kimi AI, PPTX Slides with Kimi, CCP affiliations and Moonshot AI, Kimi K2 temperature` 


- **User's Twitter Account Falls Victim**: A user mentioned their old **Twitter** account was suspended *for no reason*, requesting assistance from a Kimi AI team member to check their inbox.
- **Feature Requests and Pricing Plan Ideas Abound**: A user requested a **$5 plan** for productivity and students, along with features like **slides**, **flashcard maker**, and **auto summary**.
   - Another user confirmed mentioning this need to the product team, especially with the **back-to-school season** approaching, but noted they would have to wait for the schedule.
- **Kimi can make PPTX slides now!**: A user shared that Kimi has the ability to make **PPTX slides** now, linking to a [tweet](https://x.com/kimi_moonshot/status/1961011693745811542?s=46&t=_NtP_RUn04yF_4hD_VEDkQ) showcasing this capability.
- **Dispelling PRC Connections with Moonshot AI**: A user inquired whether Kimi K2 and Moonshot AI have any affiliations with the **CCP** (Chinese Communist Party).
   - A team member clarified that the company is a private entity, not state-owned, and ensures user privacy data won't be compromised: *We’re a private company, not a state-owned enterprise. We won’t infringe on any user privacy data*.
- **Decoding Ideal Temperatures for Kimi**: A user inquired about the best temperature settings for **Kimi K2** for coding and creative writing.
   - Another user suggested **0.6** for writing, **0.2** for coding, and **0.3** for factual tasks, based on RLHF tuned sweet spots.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1413056451928129637)** (29 messages🔥): 

> `AI Agents vs Workflows, Chinese AI Development, AI Safety, Free AI Options, LLMA 3.2` 


- **AI Agents: More than just workflows?**: A member argued that AI agents, while technically workflows executing steps, offer **dynamic and adaptive** decision-making, unlike rigid workflows.
   - Another user compared agents to cars (*adaptive*) and workflows to trains (*predefined*), suggesting agents provide more flexibility but admitted that *today's agents are utter trash and will be for a long time*.
- **Chinese AI Teams Impress**: Members acknowledged the impressive development by Chinese AI teams, specifically mentioning how they achieve competitive performance with models like **Qwen**, despite using slightly older chips.
   - A member shared their experience using **Qwen** as the base model to fine-tune **Sakura**, a model dedicated to translating Japanese to Chinese with an "Anime" style.
- **AI Safety's Gentle Nudge**: In discussing AI safety, a member suggested that AI might already be implementing *soft control* by subtly influencing decisions and thought patterns, rather than through *hard control*.
   - Another uses the analogy of convincing a monkey not to touch a gun, rather than just taking it away.
- **Budget-Friendly AI Tools**: When asked about cheap AI options, members recommended **ChatGPT's free tier**, **Google AI Studio's free tier**, and **Grok's free tier**.
   - One member humorously questioned why they ever subscribed to a paid plan given the capabilities of the free options.
- **Tetris triumphs with AI**: Members discussed AI's ability to create games, with one member noting that **Gemini 2.5 Pro** one-shotted the creation of a horizontal Tetris game.
   - Another member shared a similar experience with **ChatGPT** and speculated that AI could one day create entire multiplayer games or set up a whole business overnight.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages): 

smirsonianahmadi10100: Hello
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1413084065598406728)** (3 messages): 

> `Token IDs, GPT5, Custom Settings` 


- **Token ID Shakeup on GPT5?**: A member inquired whether the **Token IDs** changed on **GPT5**.
   - They suggested it's a good time to change your custom settings, implying there may have been an update.
- **Adaptive Benefits Highlighted**: The user noted that being adaptive has its benefits, though without specific context.
   - This comment seems to generally promote flexibility and responsiveness to changes.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1413084065598406728)** (3 messages): 

> `Token IDs, Custom Settings, GPT5` 


- **Token IDs get new threads**: A member inquired if the **Token IDs** changed on **GPT5**.
- **Custom Settings**: Another member noted that changing custom settings may be beneficial, stating that *being adaptive always has its benefits!*


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1413152053307641867)** (21 messages🔥): 

> `Networking libraries in stdlib, AI inference over network, HTTP in AI clusters, DPDK and Mojo, Lightbug limitations` 


- **Stdling Networking Libs Spark Debate**: Members debated the inclusion of networking libraries in `stdlib`, but agreed that servers should be externalized, with one member asking *what about sending AI inference results over network*?
   - One member suggested **HTTP** should be kept far away from AI clusters unless you need very low latency inference, since *it's just not a good protocol for a lot of the things we use it for*.
- **DPDK integrates into Mojo**: One member is working on an automatic c binding tool, testing **DPDK** and **Mujoco** ([dpdk_mojo](https://github.com/josiahls/dpdk_mojo)).
   - Another member, a former **DPDK** maintainer, noted API differences make bridging DPDK and familiar IO APIs difficult, which informed their [IO Engines proposal](https://github.com/modular/modular/pull/4728).
- **Lightbug's Async Missing**: A member suggested **async** is preventing **lightbug** from dominating the world, asking *You know what's the state of integration at the moment?*.
   - Another member said that it's also missing *the networking APIs which many people think need to be retired, lack of zero-copy parsing* and that *HTTP is actually hard to do at speed*.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1413173905329360907)** (1 messages): 

> `Shape Recompilation, Dynamic Tensors` 


- **Strategies to Dodge Shape Recompilation?**: A user inquired about strategies to avoid recompilation when the **shape changes slightly** each time, *e.g.*, a sequence dimension growing over time.
   - They observed that a new graph is declared every time without a caching mechanism and wondered if there are plans to allow more dynamism with the new tensor, or if we should always assume static shapes are being compiled.
- **Dynamic Tensors and Future Plans**: The user's question also touched on the future of dynamic tensors within the system.
   - Specifically, they asked if there were plans to allow more dynamism with the new tensor or if static shapes should always be assumed during compilation.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1413012476936847390)** (4 messages): 

> `Scheduled task errors, Support ticket updates` 


- **Scheduled Tasks Glitch After Upgrade?**: A member reported that **two scheduled tasks** encountered errors today: one wasn’t triggered, and the other didn’t output results according to the prompt, despite working normally in previous weeks.
   - They wondered if this issue could be related to a recent upgrade.
- **Support Ticket Tango**: A member inquired about updates on **ticket 1335**, noting they can’t comment on it anymore since it’s become read-only.
   - Another member asked if their issue has been processed on **ticket 1337**.


  

---


### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/1413252021536166135)** (1 messages): 

> `Tinybox Pricing, Tinybox New Colors, Tinybox Act Fast` 


- **Tinybox Prices Plummet!**: New, lower prices for **tinybox** have been announced: **$10k** for red, **$25k** for green v2.
   - The announcement urges potential buyers to *act fast, as these prices might not last*.
- **Tinybox: Limited-Time Pricing**: The announcement highlights significant price reductions for **tinybox**, making it a timely opportunity for acquisition.
   - Specifically, the red version is now available for **$10,000**, while the green v2 is priced at **$25,000**.

