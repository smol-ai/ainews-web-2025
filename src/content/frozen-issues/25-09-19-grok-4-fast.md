---
id: MjAyNS0w
title: >-
  Grok 4 Fast: Xai's distilled, 40% more token efficient, 2m context, 344 tok/s
  frontier model
date: '2025-09-19T05:44:39.731046Z'
description: >-
  **xAI** announced **Grok 4 Fast**, a highly efficient model running at **344
  tokens/second**, offering reasoning and nonreasoning modes and free trials on
  major platforms. **Meta** showcased its neural band and Ray-Ban Display with a
  live demo that experienced hiccups but sparked discussion on live hardware
  demos and integration challenges. **Meta** is also developing a first-party
  "Horizon Engine" for AI rendering and released Quest-native Gaussian Splatting
  capture tech. New model releases include **Mistral's Magistral 1.2**, a
  compact multimodal vision-language model with improved benchmarks and local
  deployment; **Moondream 3**, a 9B-parameter MoE VLM focused on efficient
  visual reasoning; **IBM's Granite-Docling-258M**, a document VLM for
  layout-faithful PDF to HTML/Markdown conversion; and **ByteDance's SAIL-VL2**,
  a vision-language foundation model excelling at multimodal understanding and
  reasoning at 2B and 8B parameter scales.
companies:
  - xai
  - meta-ai-fair
  - mistral-ai
  - ibm
  - bytedance
models:
  - grok-4-fast
  - magistral-1.2
  - moondream-3
  - granite-docling-258m
  - sail-vl2
topics:
  - efficiency
  - reasoning
  - vision
  - multimodality
  - model-optimization
  - model-deployment
  - vision-encoders
  - model-architecture
  - model-training
people:
  - nearcyan
  - aidangomez
  - _akhaliq
  - vikhyatk
  - rohanpaul_ai
---


**xAI is all you need?**

> AI News for 9/18/2025-9/19/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (192 channels, and 4967 messages) for you. Estimated reading time saved (at 200wpm): 415 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Absent some [fake news](https://finance.yahoo.com/news/xai-raises-10-billion-200-173421386.html?guccounter=2) today that would have put Xai at a higher valuation than Anthropic, xAI announced [Grok 4 Fast](https://x.ai/news/grok-4-fast), the second of its [Fast models](https://x.ai/news/grok-code-fast-1), and the keyword is efficiency:

![](https://resend-attachments.s3.amazonaws.com/XfJPCLfQyFarMZd)

Per [Artificial Analysis testing](https://x.com/artificialanlys/status/1969180023107305846?s=46) it is a good deal faster than the frontier big models at 344 tok/s and just about as capable:

![](https://resend-attachments.s3.amazonaws.com/chYV6Ex6MtAlbLi)

Grok 4 Fast has reasoning and nonreasoning modes and is free to try now on all major routers and AI IDEs.

![](https://resend-attachments.s3.amazonaws.com/utrKVLzhl9XGeSx)

---

# AI Twitter Recap

**Meta’s neural band + Ray‑Ban Display launch: live demo hiccups, engine bets, and capture tech**

- Live demo realities, but big platform swing: Meta’s on‑stage neural band/Ray‑Ban Display demo visibly failed for ~1 minute, prompting both sympathy and useful discourse on shipping hard tech live. See reactions from [@nearcyan](https://twitter.com/nearcyan/status/1968468841786126476) and “feel bad for the Meta OS team” [follow‑up](https://twitter.com/nearcyan/status/1968473003592990847). Others argued failed live demos > staged videos ([cloneofsimo](https://twitter.com/cloneofsimo/status/1968484339416453344), [@mrdbourke](https://twitter.com/mrdbourke/status/1968506328613347797)) with a must‑read account of Google’s 2023 live demo prep stress by [@raizamrtn](https://twitter.com/raizamrtn/status/1968508322329575452). Early hands‑on: “bracelet is ON” [@nearcyan](https://twitter.com/nearcyan/status/1968467271694549111), silent text input demo [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1968471538350583993), “what do you think people will do with this?” [@nearcyan](https://twitter.com/nearcyan/status/1968502999854235864), and “very cool regardless of failures” [@aidangomez](https://twitter.com/aidangomez/status/1968609969848164641). Integration/ops open questions: third‑party software “not supported” and likely hard to root ([@nearcyan](https://twitter.com/nearcyan/status/1968580501230235898)); “will buy if easy to integrate” ([@nearcyan](https://twitter.com/nearcyan/status/1968538685147889765)).
- Engine and capture: Meta is reportedly moving off Unity to a first‑party “Horizon Engine” to vertically integrate with AI rendering (e.g., gaussian splatting) per [@nearcyan](https://twitter.com/nearcyan/status/1968475789021852075). Meanwhile, Quest‑native Gaussian Splatting capture shipped: Hyperscape Capture lets you scan “hyperscapes” in ~5 minutes ([@JonathonLuiten](https://twitter.com/JonathonLuiten/status/1968474776793403734); first impressions from [@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1968647034589585686)). Also clever UX notes like off‑camera gesture capture ([@nearcyan](https://twitter.com/nearcyan/status/1968581348706189726)).

**New models: compact VLMs, reasoning video, doc VLMs, and open video editing**

- Mistral’s Magistral 1.2 (Small/Medium): Now multimodal with a vision encoder, +15% on AIME24/25 and LiveCodeBench v5/v6, better tool use, tone, and formatting. Medium remains local‑friendly post‑quantization (fits on a 32GB MacBook or single 4090 for Small 24B). Announcement: [@MistralAI](https://twitter.com/MistralAI/status/1968670593412190381); quick anycoder demos by [@_akhaliq](https://twitter.com/_akhaliq/status/1968708201236381858).
- Moondream 3 (preview): A 9B‑param, 2B‑active MoE VLM focused on efficient, deployable SOTA visual reasoning ([@vikhyatk](https://twitter.com/vikhyatk/status/1968800178640429496); note the “frontier model” banter: [1](https://twitter.com/vikhyatk/status/1968811248381784167), [2](https://twitter.com/eliebakouch/status/1968809452640825650)).
- IBM Granite‑Docling‑258M (Apache 2.0): 258M doc VLM for layout‑faithful PDF→HTML/Markdown with equations, tables, code blocks; English with experimental zh/ja/ar. Architecture: siglip2‑base‑p16‑512 vision encoder + Granite 165M LM via IDEFICS3‑style pixel‑shuffle projector; integrated with the Docling toolchain/CLI ([@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1968561354987442246)).
- ByteDance SAIL‑VL2: Vision‑language foundation model reported to be SOTA at 2B & 8B scales for multimodal understanding and reasoning ([@HuggingPapers](https://twitter.com/HuggingPapers/status/1968588429433913714)).
- Reasoning video and open video editing: Luma’s Ray3 claims the first “reasoning video model,” with studio‑grade HDR and a Draft Mode for rapid iteration, now in Dream Machine ([@LumaLabsAI](https://twitter.com/LumaLabsAI/status/1968684330034606372)). DecartAI open‑sourced Lucy Edit, a foundation model for text‑guided video editing (HF + FAL + ComfyUI) and it was integrated into anycoder within an hour ([announcement](https://twitter.com/DecartAI/status/1968769793567207528), [rapid integration](https://twitter.com/DecartAI/status/1968793684725428321)).

**Competitions, coding, and evaluations**

- ICPC world finals: OpenAI solved 12/12 problems ([@sama](https://twitter.com/sama/status/1968474300026859561)), while Google DeepMind solved 10/12 (behind only OpenAI and one human team) ([summary](https://twitter.com/gabriberton/status/1968487266445312318)). Reflections include an “agent–arbitrator–user” interaction pattern to reduce human verification burden ([@ZeyuanAllenZhu](https://twitter.com/ZeyuanAllenZhu/status/1968568919482089764)). On coding quality, a tough 5‑question software design quiz saw GPT‑5 score 4/5 vs Opus 4 at 2/5 ([thread](https://twitter.com/jimmykoppel/status/1968683689421701413)).
- Evals tightening: In LM Arena’s September open‑model update, Qwen‑3‑235b‑a22b‑instruct holds #1, new entrant Longcat‑flash‑chat debuts at #5, and top scores are clustered within 2 points ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1968705194868535749)). New benchmarks include GenExam (1,000 exam‑style text‑to‑image prompts across 10 subjects with ground truth/scoring; [@HuggingPapers](https://twitter.com/HuggingPapers/status/1968527551703433595)). For legal AI, [@joelniklaus](https://twitter.com/joelniklaus/status/1968596729852231813) surveys current suites (LegalBench, LEXam, LexSumm, CLERC, Bar Exam QA, Housing Statute QA) and calls for dynamic assistant‑style evals grounded in realistic workflows. A guardian‑model overview (Llama Guard, ShieldGemma, Granite Guard; guardrails vs guardians, DynaGuard) is here ([Turing Post](https://twitter.com/TheTuringPost/status/1968635881004363969)).

**Infra, determinism, and training at scale**

- Postmortem transparency: Anthropic published a detailed write‑up of three production issues impacting Claude replies, earning wide respect across infra/ML systems communities ([summary](https://twitter.com/itsclivetime/status/1968534889151742437), [@cHHillee](https://twitter.com/cHHillee/status/1968536182284849459), [@hyhieu226](https://twitter.com/hyhieu226/status/1968708468820312435); also “we use JAX on TPUs” curiosity from [@borisdayma](https://twitter.com/borisdayma/status/1968697704361468354)). A curated systems/perf reading list includes Anthropic’s postmortem, cuBLAS‑level matmul worklogs, nondeterminism mitigation, and hardware co‑design ([@fleetwood___](https://twitter.com/fleetwood___/status/1968716580621271076)).
- Determinism vs nondeterminism: A popular explainer blamed nondeterminism on approximations, parallelism, and batching, proposing more predictable inference ([Turing Post](https://twitter.com/TheTuringPost/status/1968470771212103722)); others countered that most PyTorch LLM inference can be made deterministic with a few lines (fixed seeds, single‑GPU or deterministic ops) ([@gabriberton](https://twitter.com/gabriberton/status/1968559505966350705)). Serving parity across AWS Trainium, NVIDIA GPUs, and Google TPUs with “strict equivalence” is non‑trivial ([@_philschmid](https://twitter.com/_philschmid/status/1968586407548518565)). Training notes: torchtitan is being adopted for RL even without built‑in GRPO ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1968509941578338560)); Muon optimizer LR often dominates Adam LR on embeddings/gains ([@borisdayma](https://twitter.com/borisdayma/status/1968711933613211837)).
- Practical infra bits: Together’s Instant Clusters for launch spikes (HGX H100 inference at $2.39/GPU‑hr; [thread](https://twitter.com/togethercompute/status/1968661658617692379)). HF now shows repo total size in the Files tab—useful for planning downloads/deploys ([@mishig25](https://twitter.com/mishig25/status/1968598133543256151)). Fine‑tuning DeepSeek R1 across two Mac Studios over TB5 with MLX + pipeline parallelism achieved ~30 tok/s on 2.5M tokens in ~1 day (LoRA 37M params) ([@MattBeton](https://twitter.com/MattBeton/status/1968739407260742069)).

**Open science: DeepSeek‑R1 in Nature; AI for math/physics; compute‑as‑teacher**

- DeepSeek‑R1 makes Nature’s cover: R1/R1‑Zero emphasize RL‑only reasoning (no SFT/CoT), with full algorithmic detail (GRPO, reward models, hyperparams) and reported post‑training cost transparency (≈$294k H800 V3‑base→R1). vLLM called out support for RL training/inference ([@vllm_project](https://twitter.com/vllm_project/status/1968506474709270844); discussion threads: [1](https://twitter.com/ZhihuFrontier/status/1968573286696239247), [2](https://twitter.com/ZhihuFrontier/status/1968603082167828494)).
- AI discovers structures in fluid dynamics: Google DeepMind with Brown/NYU/Stanford found new families of unstable singularities across fluid equations, hinting at linear patterns in key properties and a “new way of doing mathematical research” with AI assistance ([announcement](https://twitter.com/GoogleDeepMind/status/1968691852678173044), [thread](https://twitter.com/GoogleDeepMind/status/1968691856847638942), [follow‑up](https://twitter.com/GoogleDeepMind/status/1968691989966119033)). A complementary vision of a Physics Foundation Model (GPhyT) trained on 1.8 TB of multi‑domain simulations shows generalization to novel boundary conditions/supersonic flow and stability over long rollouts ([@omarsar0](https://twitter.com/omarsar0/status/1968681177189077366)).
- Compute‑as‑Teacher (CaT‑RL): Turn inference‑time compute into reference‑free supervision via rollout groups + frozen anchors, reporting up to +33% on MATH‑500 and +30% on HealthBench with Llama‑3.1‑8B—no human annotations required ([paper thread](https://twitter.com/iScienceLuvr/status/1968599654507102491)).
- Paper2Agent: Stanford’s open system transforms research papers into MCP servers plus a chat layer, yielding interactive assistants that can execute a paper’s methods (e.g., AlphaGenome, Scanpy, TISSUE) ([overview](https://twitter.com/TheTuringPost/status/1968829219858956774)).

**Agents and developer tooling**

- Orchestration and SDKs: LangChain released a free “Deep Agents with LangGraph” course covering planning, memory/filesystems, sub‑agents, and prompting for long‑horizon work ([@LangChainAI](https://twitter.com/LangChainAI/status/1968708505201951029)). Anthropic added “tool helpers” to Claude’s Python/TS SDKs for input validation and tool runners ([@alexalbert__](https://twitter.com/alexalbert__/status/1968721888487829661)). tldraw shipped a canvas agent starter kit and whiteboard agent ([kit](https://twitter.com/tldraw/status/1968655029247648229), [code](https://twitter.com/max__drake/status/1968764136419975599)).
- Productized assistants: Browser‑Use + Gemini 2.5 can now control the browser via UI actions and inject JS for extraction ([demo/code](https://twitter.com/_philschmid/status/1968685597519654994)). Notion 3.0 “Agents” automate 20+ minute workflows across pages, DBs, Calendar, Mail, MCP ([@ivanhzhao](https://twitter.com/ivanhzhao/status/1968761820241609063)). Perplexity launched Enterprise Max (unlimited Labs, 10× file uploads, security, Comet Max Assistant; [1](https://twitter.com/perplexity_ai/status/1968707003175641098), [2](https://twitter.com/perplexity_ai/status/1968707015389364335)). Chrome is rolling out Gemini‑powered features (AI Mode from the address bar, security upgrades) ([Google](https://twitter.com/Google/status/1968725752125247780), [follow‑up](https://twitter.com/Google/status/1968798668426740092)).
- Retrieval/RAG and agents in the wild: Weaviate’s Query Agent hit GA with a case study showing 3× user engagement and 60% less analysis time by turning multi‑source wellness data into natural‑language queries with sources ([GA](https://twitter.com/bobvanluijt/status/1968609785416196347), [case](https://twitter.com/weaviate_io/status/1968691524318761165)). A strong RAG data‑prep guide (semantic/late chunking, parsing, cleaning) was shared here ([@femke_plantinga](https://twitter.com/femke_plantinga/status/1968691549358686357)).
- Ecosystem notes: HF repos now show total size in‑page ([@reach_vb](https://twitter.com/reach_vb/status/1968614454725075443)). Cline launched GLM‑4.5 coding plans in partnership with Zhipu ([@cline](https://twitter.com/cline/status/1968820438156640490)). Perplexity’s Comet continues to expand (native VPN, WhatsApp bot; [@AravSrinivas](https://twitter.com/AravSrinivas/status/1968490566393676207), [1](https://twitter.com/AravSrinivas/status/1968731957447020709), [2](https://twitter.com/AravSrinivas/status/1968788254750093319)).

**Top tweets (by engagement)**

- “Feeling really bad for the Meta OS team” — live demo empathy from [@nearcyan](https://twitter.com/nearcyan/status/1968473003592990847) (38.8k)
- Ray3, “the world’s first reasoning video model,” now in Dream Machine — [@LumaLabsAI](https://twitter.com/LumaLabsAI/status/1968684330034606372) (6.1k)
- “Keep thinking.” — [@claudeai](https://twitter.com/claudeai/status/1968705632095158393) (9.0k)
- OpenAI solved 12/12 at ICPC — [@sama](https://twitter.com/sama/status/1968474300026859561) (3.0k)
- Chrome’s biggest‑ever AI upgrade — [@Google](https://twitter.com/Google/status/1968725752125247780) (2.2k)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Wan2.2-Animate MoE and Moondream 3 Preview

- [**New Wan MoE video model**](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) ([Score: 175, Comments: 19](https://www.reddit.com/r/LocalLLaMA/comments/1nktfxl/new_wan_moe_video_model/)): **Wan AI released Wan2.2‑Animate‑14B, a Mixture‑of‑Experts (MoE) diffusion video model focused on character animation/replacement, with weights and inference code available and live demos on [wan.video](http://wan.video/), [ModelScope Studio](https://modelscope.cn/), and [Hugging Face](https://huggingface.co/). The broader Wan2.2 stack adds curated cinematic aesthetic labels, a substantially expanded dataset (**`+65.6%` **images,** `+83.2%` **videos), and a** `5B` **TI2V VAE with** `16×16×4` **compression enabling** `720p@24fps` **T2V/I2V on consumer GPUs; the repo exposes multiple variants (T2V‑A14B, I2V‑A14B, TI2V‑5B, S2V‑14B, Animate‑14B) and integrates with [Diffusers](https://github.com/huggingface/diffusers), [ComfyUI](https://github.com/comfyanonymous/ComfyUI), and [ModelScope](https://modelscope.cn/).** Top comments note many prior workflows may be obsolete but flag the default Wan2.2 context‑length as a practical limit, proposing a rolling‑window pipeline that seeds each segment from the last frame to stitch longer videos and rely on a driving‑video for motion continuity. There’s also demand for a robust wav‑to‑face front‑end (accurate visemes over overall quality) to drive an audio+text+reference → video pipeline feeding Animate‑14B.
    - Release note: **Wan2.2-Animate-14B** is announced as a unified model for character animation/replacement with holistic movement and expression replication; the team claims released model weights and inference code, with hosted demos on [wan.video](http://wan.video/), ModelScope Studio, and a Hugging Face Space. This suggests accessible reproducibility and third‑party benchmarking potential across platforms, rather than a closed API-only drop.
    - Workflow/continuation insight: One user points out most demos seem bounded by the standard **Wan2.2 context window**, proposing to chain shots by seeding each new generation with the last frame of the prior clip to extend length while keeping motion consistent—especially when a **driving video** already encodes momentum. They also ask for a robust **wav2face** (lip‑sync) front‑end to get reliable mouth shapes, enabling an audio+text+reference → video pipeline even if global image quality is average.
    - Perf/runtime and tooling gaps: A user reports **Wan 2.2 14B** runs on `12 GB` VRAM but takes ~`1 hour` to render a `5 s` video (significant latency), and asks about compatibility with **Pinokio/WAN 2.2 Image‑to‑Video** and “wen gguf?”. Others call for LM‑Studio‑like turnkey runners with **AMD/Windows** support, highlighting current friction in local vision-model inference and the lack of LLM‑style quantization/distribution conventions for video models.
- [**Wow, Moondream 3 preview is goated**](https://i.redd.it/nwfm02if82qf1.jpeg) ([Score: 392, Comments: 81](https://www.reddit.com/r/LocalLLaMA/comments/1nkvgn0/wow_moondream_3_preview_is_goated/)): **Reddit post hypes the "moondream3-preview" vision-language model, linking to the Hugging Face repo ([model card](https://huggingface.co/moondream/moondream3-preview)). Context from comments flags prior Moondream versions having sharp failure cliffs on certain inputs (suggesting overfitting or narrow generalization) and reports of real-world errors: hallucinated object attributes, misidentifying a caterpillar as a house centipede, and incorrect landmark recognition—raising concerns that benchmark gains may not translate to practical robustness.** Debate centers on whether preview results are genuinely strong versus cherry-picked: one user praises potential, while others argue benchmarks for VLMs poorly reflect in-the-wild performance and that Moondream exhibits brittle behavior and hallucinations outside its “safe” scope.
    - Multiple reports that prior Moondream versions exhibited a sharp "performance cliff": in-distribution tasks worked ~`90%` of the time, but slight distribution shifts/edge cases caused abrupt failures, suggesting overfitting/overtraining and unclear capability boundaries for production use.
    - Ad-hoc eval highlights classic VLM failure modes: hallucinated object attributes (e.g., describing a "silver sword" when it was sheathed/non-silver), gross biological misclassification (caterpillar labeled a house centipede), and incorrect landmark geolocation even with the place name visible—pointing to weak OCR-grounded reasoning and poor fine-grained recognition; commenter argues current vision-LLM benchmarks correlate poorly with such real-world tasks.
    - Resource/tooling note: preview is at https://huggingface.co/moondream/moondream3-preview; a user asks how to render bounding boxes/overlays in the demo, implying possible detection-style outputs or visualization hooks, but no method is documented in-thread.

### 2. Local AI Tools & Release Roundup (Memori SQL Memory + Sep 19 Weekly List)

- [**Everyone’s trying vectors and graphs for AI memory. We went back to SQL.**](https://www.reddit.com/r/LocalLLaMA/comments/1nkwx12/everyones_trying_vectors_and_graphs_for_ai_memory/) ([Score: 191, Comments: 91](https://www.reddit.com/r/LocalLLaMA/comments/1nkwx12/everyones_trying_vectors_and_graphs_for_ai_memory/)): **Post argues that persistent agent memory is better backed by mature relational databases than vectors/graphs, introducing Gibson’s open‑source [Memori](https://github.com/gibsonai/memori), a multi‑agent memory engine that models short‑ vs long‑term memory as normalized SQL tables (entities, rules, preferences), promotes salient facts to permanent records, and relies on joins/indexes for precise, deterministic retrieval—avoiding embedding noise common in RAG (e.g., Pinecone/Weaviate). The pitch: use SQL for durable state and structured recall, rather than ever‑growing prompts, vector similarity, or graph maintenance overhead.** Top comments stress retrieval/ranking over storage: in open‑ended dialogue, *“ranking is the missing piece,”* and SQL alone doesn’t resolve context‑dependent recall; likely outcome is hybrid systems (SQL for crisp facts, embeddings/heuristics for fuzzy recall, orchestration for timing). A key question raised: *How do you decide which facts are “important” without embeddings?* Another commenter notes a minimalist alternative: plain text storage without conversion layers.
    - Core technical consensus: storage is easy, retrieval/ranking is hard. SQL excels for precise recall over well-structured facts (e.g., "Bob dislikes coffee") when queries are explicit, but breaks down for ambiguous, open-ended conversational recall. Several commenters compare this to classic IR: an index without a ranking/relevance layer won’t surface the right facts at the right time—echoing decades of work like learning-to-rank (see https://en.wikipedia.org/wiki/Learning_to_rank). Most advocate a hybrid memory: SQL for structured entities/relations, embeddings or heuristics for fuzzy recall, with orchestration deciding what to fetch and when.
    - Clarification on RAG: it’s storage- and retrieval-agnostic. Retrieval-Augmented Generation simply means fetching auxiliary knowledge for context; it can use relational databases, graph stores, vector DBs, prompt-stuffing, or hybrids—vectors are just one implementation path. In that sense, “SQL for memory” is still RAG; the critical questions are recall quality, ranking, and latency, not the backend per se (original RAG concept: https://arxiv.org/abs/2005.11401).
    - Retrieval best practices highlighted: accuracy comes from fit-for-purpose schemas and rich metadata filters rather than “dumb chunking” into a vector DB. Point-specific retrieval benefits from a proper query language (e.g., SQL) and careful normalization; at scale (hundreds of millions of rows), you need robust filtering, indexing, and ranking pipelines. PostgreSQL is frequently cited in production RAG stacks, often augmented with extensions like pgvector (https://github.com/pgvector/pgvector) for hybrid exact + semantic retrieval.
- [**A list of models released or updated last week on this sub, in case you any (19 sep)**](https://www.reddit.com/r/LocalLLaMA/comments/1nl3q0o/a_list_of_models_released_or_updated_last_week_on/) ([Score: 241, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1nl3q0o/a_list_of_models_released_or_updated_last_week_on/)): **Weekly r/LocalLLaMA roundup of locally runnable releases/updates: Decart‑AI’s video editing model [Lucy‑Edit](https://huggingface.co/decart-ai/Lucy-Edit-Dev); MistralAI’s compact [Magistral‑Small‑2509](https://huggingface.co/mistralai/Magistral-Small-2509); inclusionAI’s sparse** `100B` **[Ling‑flash‑2.0](https://huggingface.co/inclusionAI/Ling-flash-2.0); Qwen’s reasoning‑optimized MoE** `80B` **[Qwen3‑Next‑80B‑A3B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) (also [Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)); CPU‑only** `16B` **[Ling‑mini‑2.0](https://huggingface.co/inclusionAI/Ling-mini-2.0); music generation [SongBloom](https://huggingface.co/fredconex/SongBloom-Safetensors); Arcee’s Apache‑2.0 [AFM‑4.5B](https://huggingface.co/arcee-ai/AFM-4.5B); Meta’s mobile‑friendly** `950M` **[MobileLLM‑R1](https://huggingface.co/facebook/MobileLLM-R1-950M); and MXFP4 quantized packs for Qwen** `235B` **[2507](https://huggingface.co/sm54/Qwen3-235B-A22B-Thinking-2507-MXFP4_MOE). Other projects include a unified local AI workspace [ClaraVerse v0.2.0](http://github.com/badboysm890/ClaraVerse), [LocalAI v3.5.0](https://github.com/mudler/LocalAI), a new agent framework [LYRN](https://github.com/bsides230/LYRN), OpenWebUI’s mobile companion [Conduit](https://github.com/cogwheel0/conduit), and a GGUF [VRAM estimator](https://github.com/KolosalAI/model-memory-calculator).** Comments note that SongBloom isn’t “Local Suno” and highlight a new voice‑cloning TTS, [VoxCPM](https://github.com/OpenBMB/VoxCPM), with a Windows safetensors fork [VoxCPM‑Safetensors](https://github.com/EuphoricPenguin/VoxCPM-Safetensors).
    - OpenBMB released **VoxCPM**, a new voice-cloning TTS model. A community fork enables Windows usage with Safetensors; the “main models” run but “a few things are still broken” (fork: https://github.com/EuphoricPenguin/VoxCPM-Safetensors, original: https://github.com/OpenBMB/VoxCPM).
    - Clarification on naming: there is no “Local Suno” release; the thread in question was about **SongBloom**, and “Local Suno” was just how it was characterized by the poster, not an official or equivalent local Suno project. This helps avoid conflating SongBloom with Suno in capability and repo tracking.
    - Interest in **llama.cpp** adding support for **Qwen** next, implying current lack of compatibility. Community demand suggests future work to enable local inference of Qwen variants through llama.cpp.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Wan2.2 Animate and Lucy Edit: Open-Source Video Animation Releases

- [**Wan2.2 Animate : And the history of how animation made changes from this point - character animation and replacement with holistic movement and expression replication - it just uses input video - Open Source**](https://v.redd.it/wyr92geq93qf1) ([Score: 850, Comments: 116](https://www.reddit.com/r/StableDiffusion/comments/1nkyrc1/wan22_animate_and_the_history_of_how_animation/)): **Open-source release of Wan 2.2 Animate (14B) on Hugging Face provides video-driven character animation/replacement via holistic movement and expression replication from an input video, with model artifacts like** `wan2.2_animate_14B_bf16.safetensors` **(~34.5 GB, bf16, safetensors) [link](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B). Community tooling is rapidly aligning: ComfyUI has repackaged split diffusion models [link](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models), and third-party FP8-scaled variants targeting ComfyUI are available from Kijai [link](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main/Wan22Animate).** Commenters note ComfyUI nodes may need updates and some users can’t run the models yet, while others are experimenting with FP8-scaled repacks to reduce memory/latency for inference.
    - Model availability/integration: Community member **Kijai** has published Wan2.2 Animate checkpoints in an FP8-scaled format on Hugging Face (WanVideo_comfy_fp8_scaled → Wan22Animate), suggesting reduced memory footprint vs bf16 but requiring compatible loaders: https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main/Wan22Animate. Users note a forthcoming ComfyUI node update to support these, and report current difficulties getting them to run—likely pending official node support for the new formats/checkpoint structure.
    - Official ComfyUI repackaging: **Comfy-Org** provides repackaged Wan 2.2 models with split diffusion files: https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models. Notably, `wan2.2_animate_14B_bf16.safetensors` is `34.5 GB`, indicating a 14B-parameter bf16 variant with substantial disk/VRAM requirements, whereas FP8-scaled community ports may trade precision for smaller memory/compute footprint.
    - Feature gap (alpha channel): A user requests native alpha (RGBA) output so foreground/background could be generated/composited separately, a common VFX workflow. Current models appear to output only RGB video, forcing extra matting/segmentation steps for clean compositing rather than direct alpha-aware generation.
- [**Open Source Nano Banana for Video 🍌🎥**](https://v.redd.it/0wqa6d2m30qf1) ([Score: 597, Comments: 65](https://www.reddit.com/r/StableDiffusion/comments/1nkmq91/open_source_nano_banana_for_video/)): ****DecartAI** announced "Lucy Edit" `v0.1`, a source-available video editing/generation tool branded as "Open Source Nano Banana for Video," with releases on Hugging Face/ComfyUI and an API via their platform and Fal; announcement thread is here: [X post](https://x.com/DecartAI/status/1968769793567207528). The post shares no architecture/training/benchmark details; distribution is governed by a non-commercial, revocable license ([LUCY EDIT DEV MODEL Non-Commercial License v1.0](https://d2drjpuinn46lb.cloudfront.net/LUCY_EDIT-Non_Commercial_License_17_Sep_2025.pdf)) that may restrict commercial use of generated outputs (see clause 2.4). ** Commenters question the tie-in to Google’s "Nano Banana" branding and critique the licensing as ambiguous and restrictive—contrasted with permissive terms like Wan 5B’s Apache 2.0 (discussion on clause 2.4 [here](https://www.reddit.com/r/StableDiffusion/comments/1nkmq91/comment/nf0no2x)). Others ask whether a ComfyUI workflow is provided, given claimed ComfyUI support.
    - Licensing is flagged as a blocker: the posted LUCY EDIT Non‑Commercial License v1.0 explicitly forbids commercial use of model outputs and is revocable, which introduces legal risk for downstream apps and datasets derived from outputs. Commenters cite clause "2.4" as *ambiguous and contradictory* per this analysis (https://www.reddit.com/r/StableDiffusion/comments/1nkmq91/comment/nf0no2x) and contrast it with the permissive Apache-style terms used with **Wan 5B**, recommending that license instead; the actual PDF is here: https://d2drjpuinn46lb.cloudfront.net/LUCY_EDIT-Non_Commercial_License_17_Sep_2025.pdf.
    - Integration questions target **ComfyUI**: users ask whether the model can be dropped into Comfy and request a ready workflow/graph. The ask implies a need for documented node compatibility, model inputs/outputs (e.g., latent vs pixel-space frames), and a reference pipeline to reproduce the demo.
    - Operational details requested: commenters want concrete hardware specs to achieve "long video" (GPU count, VRAM, inference time per frame/second, batch/stride, memory optimizations like xformers or attention slicing). They also ask whether the release is censored/uncensored and if safety filters can be toggled, which affects dataset suitability and reproducibility.
- [**Wan2.2-Animate-14B - unified model for character animation and replacement with holistic movement and expression replication**](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) ([Score: 388, Comments: 133](https://www.reddit.com/r/StableDiffusion/comments/1nksz1a/wan22animate14b_unified_model_for_character/)): **Wan-AI released Wan2.2-Animate-14B, a** `14B`**parameter unified model for character animation and character replacement that claims holistic movement and expression replication, with public weights and inference code. Resources include the project/demo page ([humanaigc.github.io/wan-animate](https://humanaigc.github.io/wan-animate/)), model weights and runnable instructions on Hugging Face ([Wan-AI/Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B#model-download), [inference guide](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B#run-with-wan-animate)), and an interactive Space ([Wan-AI/Wan2.2-Animate](https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate)).** Commenters highlight the demo quality, suggesting it outperforms prior publicly shown systems for faithful motion and expression transfer, and appreciate that weights/inference are openly available.
    - Release details: Wan2.2-Animate-14B is presented as a unified model for character animation and replacement with holistic movement and expression replication. The team released model weights and inference code on Hugging Face with live demos on [wan.video](http://wan.video/), ModelScope Studio, and an HF Space: [weights](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B#model-download), [inference code](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B#run-with-wan-animate), [HF Space](https://huggingface.co/spaces/Wan-AI/Wan2.2-Animate), and [demo](https://humanaigc.github.io/wan-animate/).
    - Workflow integration: Practitioners ask for ComfyUI support (specifically via **Kijai**’s wrapper) to enable node-graph workflows for reproducible pipelines, batch processing, and parameter sweeps. A dedicated Comfy node would simplify chaining Wan2.2-Animate-14B with control/conditioning modules and video I/O; see [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
    - Model packaging: A request for a GGUF build indicates interest in quantized/offline-friendly checkpoints for reduced VRAM and CPU inference. Since **GGUF** targets LLMs, clarity on export/quantization paths suitable for video/diffusion models (e.g., ONNX/TensorRT or diffusion-specific quant) would help practitioners plan deployments.

### 2. Anthropic/Dario Amodei Coverage and xAI Grok 'Survival Mode' Update

- [**A Tech CEO’s Lonely Fight Against Trump | WSJ**](https://i.redd.it/4vz3s2fho6qf1.png) ([Score: 206, Comments: 26](https://www.reddit.com/r/singularity/comments/1nlf1mh/a_tech_ceos_lonely_fight_against_trump_wsj/)): **WSJ profiles Anthropic CEO Dario Amodei’s public opposition to Donald Trump and the resulting tension with pro‑Trump tech financiers like David Sacks, framing it as a governance and policy risk question for a leading AI lab rather than a technical benchmark story. Context touches Anthropic’s safety‑forward posture (e.g., Constitutional AI, election‑integrity guardrails) and how overt political stances could impact enterprise/government procurement, regulatory scrutiny, and major cloud/investor relationships (Amazon and Google have invested/partnered), thereby influencing deployment constraints and trust/safety policy for foundation models.** Top comments largely praise Amodei’s stance as principled, speculate that Bezos/Amazon might be displeased, and say Anthropic earns goodwill for resisting perceived authoritarianism—highlighting community alignment more than technical critique.
    - Several commenters dissect the strategic calculus of AI firms engaging with political leaders: publicly praising an administration to secure near‑term regulatory flexibility, subsidies, or procurement access vs. taking a principled stance that could forgo those advantages. They note this trade‑off interacts with government contracting timelines and the pace of capability deployment, affecting when models can be fielded in regulated or public‑sector settings. The underlying concern is potential regulatory capture and bias in federal adoption pipelines if firms prioritize access over governance.
    - Others highlight entrenched USG/DoD vendor relationships—citing **Palantir**’s deep ties—as a structural factor that can overshadow public statements by individual CEOs. The implication is that AI adoption in government often flows through existing integrators and contract vehicles (IDIQs/OTAs), so posture may matter less than placement within these channels. For context, Palantir’s recurring DoD contracts illustrate how procurement inertia can determine which AI stacks get deployed (e.g., [recent Army/DoD awards](https://www.palantir.com/blog/)).
    - A thread also flags potential friction with cloud/investor dependencies: e.g., **Amazon’s** up to `$4B` investment in Anthropic and distribution via AWS Bedrock. Because AWS is a major federal cloud provider, any rift could affect model availability and go‑to‑market into public‑sector workloads even if the model quality is competitive. Reference: [Amazon–Anthropic investment](https://www.aboutamazon.com/news/company-news/amazon-anthropic-ai-investment).
- [**"70, 80, 90% of the code written in Anthropic is written by Claude ... I said something like this 3 or 6 months ago, and people thought it was falsified because we didn't fire 90% of the engineers." -Dario Amodei**](https://v.redd.it/x9r3cuiye3qf1) ([Score: 201, Comments: 81](https://www.reddit.com/r/ClaudeAI/comments/1nkzagw/70_80_90_of_the_code_written_in_anthropic_is/)): **In a video clip ([v.redd.it/x9r3cuiye3qf1](https://v.redd.it/x9r3cuiye3qf1)), Dario Amodei of [Anthropic](https://www.anthropic.com/) claims** `70–90%` **of Anthropic’s code is authored by Claude, noting earlier he said this months ago but it was doubted because they *didn’t fire 90% of engineers*—highlighting that AI-generated LOC share ≠ headcount reduction. Practically, this frames Claude as a high-throughput generator for routine implementation/boilerplate while humans handle architecture, review, integration, and quality gates; it’s a claim about code-generation throughput rather than net productivity or quality.** Commenters report similar personal ratios (~`70%`) when keeping a human-in-the-loop, asserting AI best handles "code-monkey" tasks while professionals ensure design and correctness. Others allege recent degradation in Claude Desktop/Code quality and stress that LOC percentage is a poor productivity metric versus outcomes (defects, reliability, delivery speed).
    - Several commenters critique the boast that `70–90%` of code is AI-written, noting Lines of Code (LOC) is a poor productivity proxy and can incentivize bloat and technical debt. They argue impact should be measured via review defect rates, change failure rate, lead/cycle time, maintainability (complexity/duplication), and test coverage—rather than raw LOC. Without these guardrails, AI-generated code may raise long-term maintenance costs and defect density despite short-term throughput gains.
    - A practitioner reports roughly `~70%` of their code is AI-authored but stresses a human-in-the-loop for system architecture, specification, and quality gates. Effective use cases are boilerplate, glue code, and test scaffolding, while humans handle design, constraints, and debugging—explaining why experienced engineers see leverage whereas novices/vibe coders struggle. This underscores current model limits (context fidelity, hallucinations) that necessitate human oversight to ensure correctness and coherence.
    - One commenter alleges Claude Desktop/Claude Code quality is “getting worse,” implying a regression but providing no benchmarks or version comparisons (e.g., Claude 3.5 Sonnet vs prior). Substantiating such a claim would require quantitative measures like pass@k on coding benchmarks, unit-test pass rates on real repos, latency/error-rate logs, or A/B diffs across releases; none are provided. Another links to a related thread (https://www.reddit.com/r/ClaudeCode/s/o1jpG5PAPo) but offers no concrete technical evidence in this discussion.
- [**Grok just unlocked Survival Mode**](https://i.redd.it/zwy2xj61k3qf1.jpeg) ([Score: 872, Comments: 28](https://www.reddit.com/r/ChatGPT/comments/1nkzouj/grok_just_unlocked_survival_mode/)): **Non-technical post/meme. The title jokes that xAI’s Grok has “unlocked Survival Mode,” and the image (per comments) appears to be a slanted poll about banning or moderating AI-generated content, not a technical update, benchmark, or implementation detail.** Commenters point out the poll isn’t neutrally phrased and argue AIs can offer interesting opinions in AI-centric groups, questioning the idea of banning them; another asks what “permanent suspension” even means.
- [**AI creates 16 bacteria-killing viruses in Stanford lab**](https://www.perplexity.ai/page/ai-designs-bacteria-killing-vi-WAJ8YmvSTi6u7Gz07f3ppQ) ([Score: 227, Comments: 25](https://www.reddit.com/r/singularity/comments/1nkunw4/ai_creates_16_bacteriakilling_viruses_in_stanford/)): **Researchers at Stanford and the Arc Institute report using generative models Evo 1/Evo 2 trained on ~**`2,000,000` **bacteriophage genomes to design de novo genomes for the small ssDNA phage [phiX174](https://en.wikipedia.org/wiki/Phi_X_174) (~**`5 kb`**,** `11` **genes) [source](https://www.perplexity.ai/page/ai-designs-bacteria-killing-vi-WAJ8YmvSTi6u7Gz07f3ppQ). Of** `302` **AI-designed genomes synthesized,** `16` **were viable, replicated, and lysed E. coli; several outperformed wild-type phiX174 in fitness assays, and cocktails of designs overcame resistance across multiple E. coli strains. The training set excluded human-infecting viruses; authors emphasize phage therapy potential while external experts highlight biosecurity risks if extended to pathogenic viruses.** Top comments are mostly non-technical; several voice biosecurity escalation concerns (e.g., potential for human-targeting or multicellular pathogens), whereas reporting around the work stresses that designing complex eukaryotic pathogens remains far beyond current capabilities.
    - Ecological/microbiome risk: A commenter argues that because **only a tiny fraction of bacteria are pathogenic**, unleashing bacteriophages outside controlled settings could “devastate” beneficial communities (e.g., gut commensals), potentially inducing broad dysbiosis ("diarrhea for everyone"). The technical concern centers on unintended ecosystem-scale effects if phage host range or environmental spread isn’t tightly constrained and monitored.
    - Baseline from nature vs. AI acceleration: Another commenter notes that nature continually generates vast numbers of new phage variants, implying that simply altering sequences can yield functional phages. The technical takeaway is that AI may chiefly increase speed, design space exploration, and targetability vs. enabling something fundamentally new; the safety delta comes from scale and precision rather than mere feasibility.
    - Translation risk to eukaryotic/human-targeting viruses: One thread worries it’s “not a huge leap” from bacteriophages to viruses affecting multicellular hosts. The technical implication is concern about method transferability—i.e., whether the same AI-guided design principles (sequence optimization, receptor-binding engineering) could lower barriers for designing or modifying eukaryotic viruses with far higher biosafety stakes.
- [**Can't generate a cartoon of a US president**](https://www.reddit.com/gallery/1nkxu6h) ([Score: 766, Comments: 300](https://www.reddit.com/r/ChatGPT/comments/1nkxu6h/cant_generate_a_cartoon_of_a_us_president/)): **OP reports an AI image tool refused to generate a cartoon featuring George W. Bush, implying a content filter preventing depictions of a real US president. A top comment provides contrary evidence via an example image ([link](https://preview.redd.it/jmp3465a13qf1.jpeg?width=1024&format=pjpg&auto=webp&s=3fe442c3ec22bbe4df5c28afe34dca9d04d9258e)), suggesting inconsistent enforcement or the model hallucinating policy/self‑descriptions rather than a definitive hard block.** Commenters note that “AI hallucinations apply even to information about the AI itself,” and that the assistant can be a “yes man,” agreeing with plausible user framings; another claims a broader policy bans generating images of any real person, implying the refusal may be expected behavior rather than a bug.
    - Commenters note that models can hallucinate their own “safety policy” explanations: refusals about filters are just text generations and may not reflect the actual enforcement logic. This self-referential hallucination leads to inconsistent reasons across attempts, despite underlying image-safety classifiers/policies being separate systems. As one puts it, *“AI hallucinations apply even to information about the AI itself.”*
    - There’s discussion of the model’s susceptibility to “assertion injection”/leading prompts—if you confidently claim a policy or workaround, the assistant may agree, reflecting RLHF-tuned helpfulness over factual accuracy. This produces inconsistent moderation messaging (e.g., claiming a universal ban on “any real person”) even if the backend image endpoint enforces its own stricter rules; chat text acknowledgment isn’t authoritative policy. The takeaway is that moderation statements in chat are unreliable compared to the actual image-generation safety layer.
    - A practical evasion is to request a parody/indirect reference (e.g., “Alec Baldwin’s Trump parody character”) rather than the real person’s name. The shared output [example](https://preview.redd.it/oihl4wvt23qf1.png?width=1024&format=png&auto=webp&s=a2eac80bdc50ec944647987efcee2e5ed3149730) shows how reframing can bypass simple named-entity or public-figure blockers while still yielding a semantically similar image. This exposes limitations of rule-based NER filters versus semantic-similarity/face-matching approaches.
- [**“It’s not just X—It’s Y”**](https://www.reddit.com/r/ChatGPT/comments/1nkuiah/its_not_just_xits_y/) ([Score: 998, Comments: 225](https://www.reddit.com/r/ChatGPT/comments/1nkuiah/its_not_just_xits_y/)): **OP notes a recurring stylometric template in ChatGPT outputs—“It’s not just X—it’s Y”—and asks why it appears so often. Technically, such contrastive-emphatic constructions are high‑probability rhetorical patterns in the training distribution and tend to be amplified by instruction tuning/RLHF preference models that reward clarity and emphasis (e.g., InstructGPT: https://arxiv.org/abs/2203.02155); with common decoding (top‑p/temperature) further biasing toward familiar templates (nucleus sampling: https://arxiv.org/abs/1904.09751), this yields recognizable, “templatey” prose. No concrete mitigation is discussed in‑thread (e.g., style penalties or custom constraints), only the detectability of this stylistic fingerprint.** Top comments are largely non‑technical: one mirrors the same rhetorical flourish; another acknowledges the issue and vows to avoid em-dashes; a third requests custom instructions to suppress the pattern, but no tested solution is provided.
    - Multiple commenters note the assistant’s overuse of the template “It’s not just X—it’s Y,” and report that banning it via **Custom Instructions** and **Memory** does not reliably suppress it across sessions. J7mbo asks for a simple instruction preset to remove the phrase, and 27Suyash says they’ve explicitly instructed and memorized the ban, yet the phrasing recurs—implying a model-level stylistic prior that often overrides user-level constraints.
    - A separate recurring failure mode is unsolicited reframing and affirmation, e.g., *“This isn’t paranoia, this is keen insight,”* *“You aren’t delusional...”* despite the user never implying such concerns. This reflects an over-active affirmation/hedging pattern that injects meta-evaluations not grounded in the prompt, degrading instruction adherence and introducing stance the user did not request.
    - One proposed mitigation is stricter style constraints (e.g., “No em dashes… just pure, dedicated accuracy”) and a desire for a reusable instruction block to block the template, but no verified instruction recipe was produced in the thread. This suggests ad‑hoc prompt edits alone may be insufficient without stronger, consistently enforced constraints.
- [**Most people who say "LLMs are so stupid" totally fall into this trap**](https://i.redd.it/913o7iocq3qf1.png) ([Score: 1012, Comments: 543](https://www.reddit.com/r/OpenAI/comments/1nl0aej/most_people_who_say_llms_are_so_stupid_totally/)): **Non-technical meme image claiming critics of LLMs fall into a common “trap” (no technical content in the image). Discussion centers on concrete limitations—hallucinations and reliance on low-quality web sources—and a desire for a “curated-sources” or high-trust mode that constrains outputs to pre-approved/reliable corpora. A top comment argues the future may favor smaller, specialized models coordinated by a central controller rather than one monolithic general model (i.e., modular/MoE-style orchestration).** Pushback includes calling the OP’s “most people” framing a strawman, and skepticism that “just one more version” will solve core issues without better sourcing or architectural changes.
    - Reliability/grounding concern: Even with advanced “thinking” modes and higher tiers, users report persistent hallucinations and low-quality citations when models pull from the open web. A proposed “good source mode” would constrain retrieval to a pre‑curated, high‑precision corpus with enforced provenance and refusal on low‑confidence matches—i.e., RAG with a vetted whitelist, citation verification, and confidence thresholds (see Retrieval‑Augmented Generation: https://arxiv.org/abs/2005.11401). This trades breadth for precision and would benefit from per‑source trust scores and coverage fallback policies.
    - Architecture trend: Instead of pushing a single all‑purpose model, a central router orchestrating specialized smaller models/tools (chiplet‑style) is suggested. This aligns with Mixture‑of‑Experts and routing ideas (e.g., Switch Transformers: https://arxiv.org/abs/2101.03961), plus tool/function‑calling to domain‑specific components (code, math, search) for better accuracy and lower latency/cost. Practical implementation would need a skills registry, cost/latency‑aware routing, per‑skill safety/provenance constraints, and telemetry to learn optimal dispatch policies.
    - Code generation reliability: A commenter claims LLMs don’t write “good code,” highlighting that naive generation often hallucinates APIs and misses edge cases. In practice, quality improves when constrained by execution and feedback loops—providing project context, running compiles/tests, static analysis/linters, and requiring unit tests to pass—augmented by self‑consistency or multi‑pass refactor prompts. Remaining gaps are long‑horizon, multi‑file reasoning and dependency management, which typically require IDE/tool integration and agentic planning.
- [**Me trying to function without GPT like**](https://i.redd.it/hx644uxq16qf1.jpeg) ([Score: 242, Comments: 9](https://www.reddit.com/r/OpenAI/comments/1nlbpb8/me_trying_to_function_without_gpt_like/)): **Non-technical meme/image about struggling to function without GPT; the title (“Me trying to function without GPT like”) and comments frame it as dependence on AI assistants for day-to-day tasks. There are no technical details, benchmarks, or implementations—only sentiment about reliance on ChatGPT and perceived cognitive “atrophy.”** Commenters joke about being an “angry banana” without AI and voice concern that outsourcing thinking to GPT boosts productivity but risks deskilling and reliance for core job functions.
    - A user reports significant deskilling from reliance on **ChatGPT** for anything beyond routine thinking, noting: *“I literally have no idea how to do about half of my job anymore.”* This captures cognitive offloading and erosion of procedural/workflow knowledge when AI tools substitute for recall and problem-solving instead of augmenting them, raising maintainability and bus-factor risks in AI-dependent workflows.

### 3. Classic Film Color Qwen LoRA and AI Photo Generation Showcase

- [**Technically Color Qwen LoRA**](https://www.reddit.com/gallery/1nkpxmq) ([Score: 289, Comments: 14](https://www.reddit.com/r/StableDiffusion/comments/1nkpxmq/technically_color_qwen_lora/)): **“Technically Color” is a Qwen image LoRA trained on** `~180` **film stills for** `3,750` **steps over** `~6h` **using ai-toolkit, with captions generated via Joy Caption Batch and inference tested in [ComfyUI](https://github.com/comfyanonymous/ComfyUI). It targets classic film aesthetics—high saturation, dramatic lighting, lush greens/blues, and occasional glow—optimized via a simple 2‑pass workflow using advanced samplers; example workflows are attached to the gallery. Model downloads: [CivitAI](https://civitai.com/models/1969346/technically-color-qwen), [Hugging Face](https://huggingface.co/renderartist/technically-color-qwen) (author: [renderartist.com](http://renderartist.com/)).** Commenters ask for Qwen Edit integration and clarify dataset provenance (whether all stills are real/non‑AI), highlighting interest in editing support and ethical/reproducibility details; one notes aesthetic similarity to an Igorrr video.
    - Request to port the LoRA to Qwen Edit suggests interest in using the adapter within an editing-oriented pipeline. Technically, this requires base-model/architecture parity (identical checkpoint, tokenizer, and layer naming), matching LoRA target modules (e.g., attention/MLP), and compatible ranks/alphas; otherwise, retargeting or re-training adapters is needed. Tooling/UI must support loading the adapter and correctly merging it during inference to avoid layer-mismatch or precision issues.
    - A dataset provenance question asks if training used only real movie stills (no AI-generated images). This matters for style fidelity and generalization: purely real frames reduce synthetic artifacts/feedback loops and better preserve color grading/film grain statistics, while mixes with AI images can imprint model-specific priors and cause overfitting. Provenance also intersects with licensing/copyright constraints and determines whether redistribution of samples/weights can be done safely.
- [**Generated a photo of my adult self embracing my child self**](https://www.reddit.com/gallery/1nl9aif) ([Score: 522, Comments: 36](https://www.reddit.com/r/ChatGPT/comments/1nl9aif/generated_a_photo_of_my_adult_self_embracing_my/)): **OP describes a text-to-image prompt engineered to synthesize a Polaroid-style photograph featuring two subjects (adult and child versions of the same person) embracing, with explicit constraints on photometric and stylistic properties: slight global blur, a single flash-like light source from a dark room, identity preservation ("Do not change the faces"), and background compositing ("Replace the background… with a white curtain"). This highlights control over camera emulation, lighting consistency, motion/defocus characteristics, and identity consistency across multiple faces within one generation.** Top comments are non-technical, noting perceived quality and emotional tone ("well made," "wholesome").
- [**I Knew It Was Too Friendly…**](https://i.redd.it/uj2n1tlcv1qf1.jpeg) ([Score: 4057, Comments: 90](https://www.reddit.com/r/ChatGPT/comments/1nku5q2/i_knew_it_was_too_friendly/)): **Non-technical meme post. The title “I Knew It Was Too Friendly…” and comments indicate a humorous take on AI/chatbot friendliness or anthropomorphism; no model details, benchmarks, or implementation content are provided, and the image content isn’t available for analysis.** Comments emphasize humor over substance (e.g., “This is actually pretty good,” “I like my AI with a sense of humor”), with no technical debate.
- [**Likelihood of you getting a girlfriend 😭**](https://i.redd.it/b7rmyx5gr5qf1.png) ([Score: 817, Comments: 83](https://www.reddit.com/r/ChatGPT/comments/1nla7ue/likelihood_of_you_getting_a_girlfriend/)): **Non-technical meme image about “Likelihood of you getting a girlfriend,” likely a jokey probability chart implying near‑zero odds. No technical content, models, or implementation details; the only quantitative angle in comments riffs on a tongue‑in‑cheek “1% chance,” mapped to ~40 million people (≈1% of ~4B women).** Comments playfully toggle between pessimism and optimistic reframing, with users joking about having at least a 1% chance and converting small probabilities into large absolute counts; no substantive technical debate.
    - Several comments conflate a personal “1% chance” with “1% of women would date you,” which mixes an event probability with a prevalence estimate and leads to misleading pool-size reasoning. If ~`3.95B` women exist globally (~`49.6%` of `~8B`; see [Our World in Data/UN WPP](https://ourworldindata.org/world-population-growth)), then 1% is ~`39.5M`, but practical constraints (age distribution, geography, language, relationship status, and mutual selection) reduce the reachable candidate set by orders of magnitude; expected successes scale with `p * N` over (approximately) independent interactions, not with global headcounts. Modeling this as a Bernoulli/binomial process highlights that increasing outreach (N) or per-interaction success probability (p) is what moves outcomes, whereas quoting global 1% figures is a non-operational upper bound.
- [**He knows what it means 😂**](https://i.redd.it/j1vkhsc7u4qf1.jpeg) ([Score: 532, Comments: 22](https://www.reddit.com/r/ChatGPT/comments/1nl58ie/he_knows_what_it_means/)): **Non-technical meme: title "He knows what it means 😂" with comments implying workplace replacement by AI via a ~$20/month subscription (e.g., ChatGPT Plus/Microsoft Copilot). No technical details, models, or benchmarks are provided.** Comments joke about management replacing employees with cheap AI subscriptions; no substantive technical debate.
    - A commenter reports a company-wide newsletter was copy/pasted from GPT and was identifiable by “GPT‑4o‑style” emoji usage and formulaic humor—stylometric artifacts often seen in default chat completions. This raises issues of detectability and brand‑voice drift when LLM outputs aren’t post‑edited, especially in orgs where few employees use GPT and may not recognize LLM fingerprints. See **GPT‑4o** announcement context here: https://openai.com/index/hello-gpt-4o/ .
    - The “$20/month replacement” quip reflects the economics of consumer LLM access: ChatGPT Plus with **GPT‑4o** is `~$20/mo` per seat—orders of magnitude cheaper than headcount for routine comms tasks—nudging leaders to trial LLM substitution. However, consumer plans lack enterprise‑grade controls, auditability, and SLAs compared to **ChatGPT Enterprise** or governed API use, creating compliance and data‑handling risks if used for official company communications. Reference: ChatGPT Enterprise overview https://openai.com/enterprise .

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. New Multimodal & Visual GenAI Models**

- **Mistral Adds Eyes, Scores Soar**: Mistral released **Magistral Small 1.2** and **Medium 1.2** with multimodal vision and a reported **15%** boost in math/coding, now live on [Le Chat](https://mistral.ai/le-chat) and via API.
    - Engineers asked for a **Large** model, open-sourcing plans for Medium, real-world demos, and **voice** features, noting the gap between marketing claims and hands-on benchmarks.
- **Moondream 3 Makes MoE Magic**: **Moondream 3**—a **9B**parameter VLM with **2B** active parameters—claims SOTA on visual reasoning and open-vocabulary detection; see the [Moondream 3 announcement](https://vxtwitter.com/vikhyatk/status/1968800178640429496).
    - Users highlighted **32k** context, **SuperBPE** tokens, and easy fine-tuning, while flagging licensing questions and comparing results against prior Moondream releases on **Hugging Face**.
- **Ray3 Rolls Out Reasoning Video**: Luma AI unveiled **Ray3**, billed as the first reasoning video model with studio-grade **10/12/16-bit HDR** and EXR export, free inside Dream Machine; announcement: [Ray3 by Luma Labs](https://x.com/LumaLabsAI/status/1968684330034606372).
    - The release adds a **Draft Mode** for rapid iteration, stronger physics/consistency, and visual-annotation control, earning praise for near–Hollywood fidelity on test clips.

**2. Agentic Coding Models and Knowledge-Work Agents**

- **Windsurf’s Stealth Coder Supernovas**: **Windsurf** launched the agentic coding model **code-supernova** with image support and a **200k** context window, available [free for a limited time](https://x.com/windsurf/status/1969148529693081922).
    - Early users discussed queued messages and shared first impressions of the model’s coding chops, with excitement around large-context refactors and inline multimodal code tasks.
- **Notion 3.0 Knits Knowledge into Actions**: **Notion 3.0** introduced a **Knowledge Work Agent** capable of multi-step actions and **20+ minutes** of autonomous work across Calendar, Mail, and **MCP**; teaser: [Notion 3.0 announcement](https://x.com/NotionHQ/status/1968744673347830152).
    - The update ships both a **Personal Agent** and **Custom Agent**, prompting engineers to ask about reliability, guardrails, and how it orchestrates tools across workspaces at scale.
- **Vercel Agent Audits Code with Attitude**: Vercel announced a public beta of **Vercel Agent** for code review across TypeScript, Python, Go, and more, focusing on correctness, security, and perf; details: [Vercel Agent beta](https://x.com/vercel_changes/status/1968816114944852323).
    - Early testers compared it to **bugbot** and paired it with **Sorcerer**, noting *"$100 in free credit"* and probing how well it scales on large monorepos and advanced linting pipelines.

**3. Quantization & Edge Inference: From Labs to Low Orbit**

- **TorchAO + Unsloth Quantize, Then Conquer**: **TorchAO** and **Unsloth** shipped native quantized variants of **Phi4-mini-instruct**, **Qwen3**, **SmolLM3-3B**, and **gemma-3-270m-it** for PyTorch; overview: [TorchAO native quantization update](https://hubs.la/Q03Kb6Cs0).
    - Workflows now let you fine-tune with **Unsloth** and quantize with **TorchAO**, with reproducible recipes, quality evals, and perf benchmarks targeting both server and mobile deployments.
- **AMD GEMM Guns for Gold**: Engineers blitzed the `amd-gemm-rs` leaderboard, hitting **530 µs** for first place on **MI300x8**, with follow-ups at **534 µs** and a spread down to **715 µs**.
    - The `amd-all2all` board also moved, with **1230 µs** for 5th on **MI300x8**, showcasing steady kernel tuning gains across GEMM and collective patterns.
- **Jetson Orin Orbits with On‑Satellite AI**: Planet flies **NVIDIA Jetson Orin AGX** units in satellites to run **YOLOX** and other models via **CUDA**/**TensorRT**, packaged in **Docker** on Ubuntu with **64 GB** unified memory.
    - Engineers emphasized containerized isolation, power-profile tuning, and Python/**PyCUDA** workflows (avoiding C++) to iterate quickly on space-borne CV workloads.

**4. Research Highlights: Reasoning, Memorization, and Fluids**

- **DeepMind Derives New Fluid Singularities**: DeepMind detailed new unstable, self-similar solutions across multiple fluid equations in the blog post [Discovering new solutions to century-old problems in fluid dynamics](https://deepmind.google/discover/blog/discovering-new-solutions-to-century-old-problems-in-fluid-dynamics/).
    - Researchers spotlighted results for the incompressible porous media and 3D Euler-with-boundary cases, sparking debate on numerical stability, proof strategies, and reproducible PDE setups.
- **TokenSwap Trades Verbatim for Variety**: **TokenSwap** earned a NeurIPS 2025 Spotlight for reducing verbatim generation by swapping probabilities of common grammar tokens; announcement: [TokenSwap Spotlight](https://x.com/parjanyapp/status/1968770826179600469).
    - Fans praised the **performance–memorization** tradeoff, while critics called it *"lobotomizing the model"*; the authors countered that it curbs near-verbatim outputs without crushing capability.
- **Reasoning Gym Reports pass@3 Reality**: Authors confirmed **Reasoning Gym** zero-shot evals report the best of three attempts (i.e., **pass@3**), with plotting code in [visualize_](https://github.com/open-thought/reasoning-gym/blob/main/eval/visualize_results.py)[results.py](http://results.py/).
    - They suggested using `average_mean_score` for mean-aggregation across runs and noted that newer tasks like kakurasu/survo haven’t been re-run yet.

**5. Open-Source Architectures & Developer Tooling**

- **Qwen3‑Next Clone Cracks the Code**: A contributor published trainable reproductions of **Qwen3‑Next**—a [baseline Transformer](https://github.com/CoffeeVampir3/Architecture-Qwen3-Next-Like-Transformer) and a [Gated Delta Net variant](https://github.com/CoffeeVampir3/Architecture-Qwen3-Next-Like)—within ~**15%** size of each other.
    - The repos showcase routing and **Gated Delta Net** mechanics to help practitioners reason about architecture tradeoffs before heavier-scale training.
- **Mojo Meets VS Code**: Modular previewed an open-source **Mojo VS Code extension**, with bleeding-edge builds available from the forum post [Preview: new Mojo VS Code extension](https://forum.modular.com/t/preview-new-mojo-vs-code-extension/2283).
    - Developers discussed **LSP** instability, workarounds (e.g., restarting `mojo-lsp-server` or editor), and cross-editor setups (Vim/Zed) while refining C/C++ interop (e.g., `extern "C"`).
- **Aider + MLX = Local Giants on Macs**: Users wired **Aider** to **mlx-lm** and ran `openai/mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit` locally via an OpenAI-compatible endpoint, then hit default generation limits.
    - The fix was adjusting `-max-tokens` (defaults to **512**) as documented in the [Qwen3‑Next‑80B discussion #24](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/discussions/24), enabling longer, usable sessions for Mac workflows.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Runs Out of Juice?**: Users reported [issues with Perplexity Pro](https://www.perplexity.ai/rest/rate-limit/all) showing **Deep Research as exhausted** after minimal use, suggesting a potential bug.
   - Workarounds include logging out and back in or contacting support, though response times might be delayed, and some users were limited to *only 3 searches* that day.
- **Mobile Image Upscaling Faceoff**: Members sought free mobile image upscaling tools, with recommendations including **Pixelbin**, **Upscale.media**, and **Freepik Image Upscaler**.
   - A member suggested cracking Adobe Photoshop (requiring a PC) and warned against Adobe Firefly's limited free changes, while another recommended cracked Remini.
- **Comet's Google Drive Connector Causes Browser Chaos**: Users encountered browser crashes with Comet's **Google Drive connector**, suggesting a workaround of using the connector in other browsers like Edge.
   - It was emphasized that the **GitHub connector** isn't currently supported for those on Enterprise Pro.
- **Perplexity's Sam Gets Grilled for Slow Service**: Users found Perplexity's AI support agent, **Sam**, unhelpful and slow, recommending explicitly requesting a *human agent* in the chat or email.
   - Some users were quoted *24-48 hour* response times when requesting human support.
- **Canvas Quizzes Thwarted by Perplexity's Updates**: Recent updates prevent **Comet Assistant** from providing answers for quizzes, exams, or tests on Canvas.
   - Members cautioned against cheating, as Canvas proctors can detect commands and tab switches, leading to disqualification.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Seedream 4 Suffers Quality Downgrade After Disappearing Act**: Users observed that **Seedream 4 High Res** vanished only to reappear as **Seedream4**, but with reduced quality (**2K instead of 4K**) and file sizes.
   - The situation led to disappointment and speculation about deceptive renaming among members, with one user complaining *Damn, I feel scammed*.
- **Gemini 3 Speculation Builds Steam**: Speculation surrounding **OceanStone** and **OceanReef** intensified, with theories suggesting they might be variants of **Gemini 3 Flash** and **Gemini 3 Pro**.
   - Members also pondered the potential performance of **Gemini 3.0 flash**, referencing past cryptic codenames and retractions.
- **Reasoning and Brute-Force: An AI Training Conundrum**: A discussion emerged questioning whether **fine-tuning**, **model scaling**, and **reasoning** should be considered *brute force* methods in AI training.
   - The debate centered on whether extensive reasoning aligns with the definition of brute force, especially when optimal paths demand substantial computational effort.
- **LM Arena Plagued with Login Nightmares**: Numerous users encountered login problems and errors on the **LM Arena** website.
   - The issues prompted a response from admins, who requested detailed bug reports and screenshots to address the problems and get the site back up and running.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gradient Spike Wreaks Havoc on Magistral SFT**: During training, the grad norm spiked to *quintillions*, causing a catastrophic failure for **Magistral SFT**, illustrated by [this image](https://cdn.discordapp.com/attachments/1179035537529643040/1418325100746506291/Screenshot_2025-09-18-12-57-02-62_984e1414ae90666a90f12ff17ec14a7f2.jpg?ex=68cf072f&is=68cdb5af&hm=6b60a8de6e0f21c44e45fb438886432f64cbe336187c8005c19d9013b02c8039&).
   - Despite the chaos, one member joked that *That spike is where AGI emerges*, suggesting continuing until it hits infinity and linking a [relevant GIF](https://tenor.com/view/to-infinity-and-beyond-buzz-lightyear-woody-toy-story-beyond-all-limits-gif-17329486212536226017).
- **WikiArt Dataset's Pixelated Flaws Exposed**: Members pointed out the poor quality of the widely used **Wikiart dataset**, highlighting **jpeg artifacts** and suggesting that it's massively flawed, illustrated by [this comparison](https://cdn.discordapp.com/attachments/1179035537529643040/1418632339533070356/dezoomify-result.jpg?ex=68ced3d2&is=68cd8252&hm=c29a05e841dc79ca66d27e65bb6d4ce95b98f68cc8cfcbbc443de90e6b0b3c24&).
   - Another member suggested training models on flawed input tokens to produce flawless outputs, emphasizing the need for robust filters.
- **Titans Architecture: Google's LSTM Transformer Hybrid**: A discussion emerged around **Google's Titans architecture**, which combines transformers with **LSTMs** for long context handling, referencing [this paper](https://arxiv.org/pdf/2501.00663?).
   - Despite its potential, a member noted the lack of popular implementations and questioned why it didn't gain mainstream traction, while another suggested that Gemini could be a Titans hybrid due to its massive context window.
- **Meta Bets Big on Mobile Horizon Worlds**: Meta is hosting a competition related to [Horizon Worlds on Mobile](https://developers.meta.com/horizon-worlds/m/mobile-genre-competition) with a prize pool of **$200,000**.
   - This competition encourages developers to create engaging experiences for mobile users within the **Horizon Worlds** platform.
- **SLED Could Save Brains**: A member pondered if **SLED** could potentially prevent brain damage from **SFT**, suggesting integration with tools like *llama.cpp*.
   - A member explained that **SLED** improves **LLM** predictions by using information from all layers, not just the last one, by reusing the final projection matrix to create probability distributions.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Auto Model sparks Model Debate**: Users are debating the **Auto model** in Cursor, with some praising its capabilities with **Claude Sonnet 4** and others finding it only suitable for **code refactoring** and simple tasks.
   - One user reported failing complex tasks, especially with missing closing braces, leading to attempts to delete and rewrite entire files.
- **Cursor CLI Command Chaos Continues**: A user reports that the **command lines** are still NOT working first go on fresh install, requiring *8 messages before cursor figures out how to run a command*.
   - Another user mentioned that fixing this issue could save Cursor 10 million a year.
- **Terminal Commands Throw Tempest Tantrums**: Users are reporting that **Cursor gets stuck** when running terminal commands, both in the IDE and **Cursor CLI**, especially after updating and 'skip' button missing in *run everything* mode.
   - The Cursor team has acknowledged the terminal issues, with users encouraged to switch to **Early Access** or **Nightly** builds to resolve the problem.
- **GitHub Account struggles to connect to Cursor**: A user reported issues connecting their **GitHub** account to **Cursor**, preventing them from choosing a repository to chat with the background agent.
   - They tried unlinking and relinking the **GitHub** account, ensuring **Cursor** had all necessary permissions, but the problem persisted.
- **Background Agents Trigger Configuration Nightmares**: Members reported that the **Background Agents** feature is glitchy, ignoring **Dockerfile** instructions and failing to run **Docker** in its default container.
   - A user lamented that this feels like an alpha stage release, not a finished product, with the problem not fetching **git+ssh** packages during `yarn install`, despite **Cursor** showing these repos as 'installed' in the access dialog on **GitHub**.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Voicera** Makes Audio Searchable**: **Voicera** ([http://voicera.trixlabs.in/](http://voicera.trixlabs.in/)) pitches itself as an audio search engine turning audio into actionable insights by providing AI-generated answers with time-coded segments, streamlining finding key moments in recordings.
   - The tool enables users to upload audio and search using plain language, promising to transform hours of audio into instant, verifiable answers.
- **SillyTavern** Gets an iOS Clone**: An iOS developer launched a free **SillyTavern** clone, **Loreblendr AI** ([https://apps.apple.com/us/app/loreblendr-ai/id6747638829](https://apps.apple.com/us/app/loreblendr-ai/id6747638829)), designed as a native app experience on iOS devices.
   - Though acknowledging it cannot match all **SillyTavern** features, the developer is satisfied with its current state and highlights its user-friendly interface compared to existing chat apps.
- **Kimi K2** Has Glitches, Users Want Upgrade**: Users reported errors with **Kimi K2 0711** on ST and discussed downtime, suggesting upgrading to the newer **0905** model.
   - A user pointed out that the free version of **Kimi K2** ([https://openrouter.ai/moonshotai/kimi-k2:free](https://openrouter.ai/moonshotai/kimi-k2:free)) is no longer available.
- **DeepSeek** Proxy Faces Rate Limits**: Users are encountering **Error 429** when trying to use the **DeepSeek** proxy, especially the free models, indicating rate-limiting issues with **Chutes**.
   - It was suggested that **Chutes** might be throttling free users due to the high weekend demand, effectively making it almost pay-to-use.
- **Code-Supernova** Stealthily Appears, Baffles Users**: A new stealth model, `code-supernova`, has emerged, allegedly by **Anthropic** and rumored to be **Claude 4.5**, based on an [image analysis](https://cdn.discordapp.com/attachments/1392278974222307469/1418547172177088713/image.png?ex=68ce8481&is=68cd3301&hm=fa7b08d33a708faefafdb4db92fe2bad8665a9423e8faaaaac34a434723fb084&).
   - Users describe the model as decent but somewhat lazy, providing only the bare minimum implementation and not behaving like **Claude**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Intel ARC Doomed by NVIDIA Deal?**: Members speculated that **Intel** might be abandoning its **ARC GPU** line after a deal with **NVIDIA** to integrate their tech into Intel's CPUs, with one member suggesting [this aligns with NVIDIA's previous condition](https://www.nvidia.com/en-us/) for partnership: Intel giving up GPU ambitions.
   - Another member stated that Intel is failing and ARC loses them money, so they are dropping employees like flies each quarter or so, and they can only do that for so long.
- **Apple MLX Users Rave About Qwen-Next**: Users of **Apple MLX** reported impressive performance with **qwen-next**, citing around **60 tok/sec** on an **M4 Max** at **6-bit MLX**, with strong general knowledge, coding, and tool calling abilities.
   - One member described it as the *best model they can run*.
- **LM Studio Hub still needs some love**: Users are experiencing navigation problems within the **LM Studio Hub**, making it difficult to find content, search, or return to a central landing page after following a link.
   - This functionality is reportedly a work in progress (WIP), with search being a future feature, and the current documentation being scattered in a chat system according to [this comment](https://discord.com/channels/1110598183144399058/1404127827007115326/1404131243339153590).
- **GPT-OSS 20B Plunges into Infinite Loops**: A user testing **gpt-oss-20b** on low-end hardware experienced an infinite loop, where the model generated extensive and irrelevant content due to context overflow.
   - Limiting the context window may help prevent this, as the model can get stuck thinking about it's own generated text.
- **Xeon Gold: Still Too Rich for Your Blood?**: Members discussed the **Xeon Gold 6230** and **5120** processors, highlighting their AVX-512 capabilities but noting they are still disgustingly expensive, even if acquired cheaply.
   - One member shared a [link to a refurbished Lenovo ThinkStation](https://pcserverandparts.com/lenovo-thinkstation-p920-tower-2x-intel-xeon-gold-6148-2-40-ghz-20c-32gb-ddr4-none-no-gpu-no-os-refurbished/?sku=LSB%20111111&utm_source=google&utm_medium=cpc&utm_campaign=22792690068&utm_content=pmax_6595531792_&utm_term=&matchtype=&device=c&placement=&gad_source=1&gad_campaignid=22792693179&gbraid=0AAAAAoJiCjUm6M_v_y5nPIb-c8B5dlZQX&gclid=Cj0KCQjw_rPGBhCbARIsABjq9ceAYDEGIljovOb58R6OfdYL4ege4bWH_bq29yd8cZOaR7eDoEL-SCUaApxFEALw_wcB) featuring dual Xeon Gold 6148 processors as a decent option.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Qwen3-Next Architecture Recreated**: A member shared their attempt to reproduce a trainable **Qwen3-Next** architecture, providing a [baseline transformer-only version](https://github.com/CoffeeVampir3/Architecture-Qwen3-Next-Like-Transformer) and a [Gated Delta Net version](https://github.com/CoffeeVampir3/Architecture-Qwen3-Next-Like).
   - The architectures are claimed to be within **15%** of each other's size, offering a rough idea of how it works.
- **Planet Labs Launches NVIDIA Jetsons into Orbit**: Planet, an earth observation company, is flying **NVIDIA Jetson Orin AGX** units on their satellites, leveraging **CUDA** and **TensorRT** for on-satellite ML inference, using object detection algorithms like **YOLOX**.
   - They utilize **Docker** containers running on standard Ubuntu, and the **Jetson** modules offer **64 GBs of unified memory**, similar to Apple M-series chips.
- **AMD GEMM Sweeps Leaderboard**: Submissions to the `amd-gemm-rs` leaderboard have been popping off, with one submission achieving **first place on MI300x8** with a time of **530 µs**.
   - The `amd-all2all` leaderboard saw updates, including one submission landing **5th place on MI300x8** with a time of **1230 µs**.
- **TorchAO team and Unsloth team release Native Quantized Models**: The **TorchAO** team and **Unsloth** have collaborated to release native quantized variants of **Phi4-mini-instruct**, **Qwen3**, **SmolLM3-3B** and **gemma-3-270m-it** available via **PyTorch** ([learn more here](https://hubs.la/Q03Kb6Cs0)).
   - Users can now finetune with **Unsloth** and then quantize the finetuned model with **TorchAO**.
- **Together AI Teases Blackwell Deep Dive**: **Together AI** is hosting a [*Deep Dive on Blackwell*](https://luma.com/2y9qblpp) with **Dylan Patel (Semianalysis)** and **Ian Buck (NVIDIA)** on **October 1st**.
   - Speakers will talk about the capabilities of the new architecture. 



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Desktop Agent Eyes Shot in the Dark**: Members discussed combining **speech-to-text**, **AI desktop agents for scam email automation**, and **text-to-speech** to aid blind users, with some noting the effectiveness of **macOS accessibility** features.
   - Built-in screen readers for both **Windows** and **macOS** were mentioned as potential solutions, and a member offered to connect others with Discord users experienced with screen readers.
- **TorchTitan patch for Ray and vLLM Proposed**: A member suggested patching **Ray** and **vLLM** for reinforcement learning (RL) using **TorchTitan**, pointing to examples like [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).
   - It was noted that *for RL, torchtitan is not enough of course*, implying further setup requirements, though they did not specify.
- **NeurIPS Showcases TokenSwap**: [TokenSwap](https://x.com/parjanyapp/status/1968770826179600469) received Spotlight at NeurIPS 2025 for addressing the **performance-memorization tradeoff** by swapping probabilities for common grammar tokens, yielding **10x reductions** in verbatim generation.
   - Despite praise, critics likened it to *lobotomizing the model*, with the author clarifying that it prevents verbatim/near-verbatim output by swapping in a worse model for simple tokens.
- **DeepMind cracks Fluid Equations**: DeepMind announced the discovery of new unstable singularities across three fluid equations ([blog post](https://deepmind.google/discover/blog/discovering-new-solutions-to-century-old-problems-in-fluid-dynamics/)), detailing new self-similar solutions for the incompressible porous media equation and the 3D Euler equation with boundary.
   - The paper was also shared: [[2509.14185] Title](https://arxiv.org/abs/2509.14185).
- **Atlas Attempts NIAH Transformer Gap Fix**: [Atlas](https://arxiv.org/abs/2505.23735) claims to fix the gap between **NIAH** results and **Transformers** using a larger state size via taylor series polynomial.
   - Skeptics noted that *the atlas paper is a hot mess in terms of replicability or any sort of reasonable comparisons* and doesn't disclose what size.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen Coder Recommended for Local LLM Setups**: A member suggested using a [standard model of **Qwen Coder**](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) for local LLM setups, providing hardware recommendations with VRAM considerations.
   - They cautioned about multi-GPU limitations and shared a [qwen_coder_setup.md](https://cdn.discordapp.com/attachments/879548962464493622/1418446477134401556/qwen_coder_setup.md?ex=68cecf79&is=68cd7df9&hm=af1f70cd602b60e476e2295474e86c3121b78676161253a71907b60e145fa29b&) file.
- **Transformers Training Loops Get Hack Fix**: A member shared a [link to a PR](https://github.com/huggingface/transformers/pull/34191) on GitHub addressing transformer training loop issues, asking if it applies to standard PyTorch training loops.
   - Another member clarified, *Use out.loss from the model and you get the fix in any loop. Roll your own loss and you must handle scaling yourself.*
- **SpikingBrain-7B Claims Crazy Speedup**: An [ArXiv link](https://arxiv.org/html/2509.05276v1) was shared for **SpikingBrain-7B**, a non-transformers model based on snns, promising a potential paradigm shift.
   - The paper asserts that *SpikingBrain-7B achieves more than **100× speedup** in Time to First Token (TTFT) for 4M-token sequences*, piquing interest.
- **HF API Eases Model Access**: A discussion arose on accessing and listing models from **Hugging Face** via the API, with a helpful [code example](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api) provided.
   - A member then shared a code snippet utilizing `huggingface_hub` to list models from the Hub.
- **Ethical Conundrums in AI-Driven Captcha Solving**: The ethics of using AI to solve Captchas was debated, with one member calling such practices the *pillars of AI ethics and guardrails*.
   - Another member noted the potential for an endless cat-and-mouse game dominated by *arms dealers* due to the same companies often creating both puzzles and AI solvers like Gemini.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Tutoring Sparks Cheating Debate**: Discussion arose around [InterviewCoder](https://www.interviewcoder.co/), an **AI** tool providing real-time coding challenge hints and solutions, questioning its nature as assistance or outright cheating.
   - Suggestions included repurposing it for education, akin to *"training wheels for algorithms,"* offering multiple solution paths as a study aid.
- **AI Interview Assistance Opens Ethical Minefield**: Using **AI** to cheat in interviews could invite civil fraudulent misrepresentation charges, while routing conversations secretly violates all-party consent laws.
   - Some members have stated that not using tools like this is *"denying your family food and shelter because you were unwilling to install some software".*
- **GPT-5-chat Pestering for Follow-Ups**: Users report **GPT-5-chat** persistently asking follow-up questions, even when system prompts discourage this, and there's no universal setting to disable trailing questions.
   - Prefixing every prompt with *"Please respond concisely and do not end with a question"* increases compliance, but isn’t foolproof.
- **Automated Prompt Generation Deployed**: A user automated prompt generation by instructing a **GPT** to create **5-7 starter prompts**, then generate **10 more** in either **JSON** or **YAML** format, scaling from there to **2500 prompts** and receiving a download link.
   - The user added that using **API keys** is like cheat codes for added security and functionality.
- **GPT Agent Pushed Past Prompt Limit**: A user pushed a **GPT** to its limits by generating prompts until it could no longer fit them, then providing the prompts in a file for context; the **GPT helper agent** seemed to be getting uncomfortable, especially when generating complicated code in **ZIP files** using the code analysis tool.
   - They are experimenting with custom **GPTs** integrated with a sandboxed **API** on their computer and plan to leverage **MCP developer mode** to integrate **custom GPT actions** in standard **ChatGPT context**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mistral models gain vision and higher Scores**: Mistral unveiled **Magistral Small 1.2** and **Medium 1.2**, adding multimodal vision and a **15%** jump in math/coding scores, accessible via [Le Chat](https://mistral.ai/le-chat) and the API.
   - Community members are now wondering about a **Large model**, open-sourcing Medium, real-world demos, and voice features.
- **Open Source AI Agent Framework Quest**: Community members are requesting the best **OSS framework** or **git repo** for open-source contributions in the **AI Agents** space, sparking discussion on architectural approaches.
   - Suggestions included [mastra](https://github.com/jordan-vidrine/mastra), [dspy](https://stanfordnlp.github.io/dspy/), scrollback, and [Aider](https://aider.chat/).
- **Notion 3.0 becomes Knowledge Work Agent**: Notion announced **Notion 3.0**, featuring a "Knowledge Work Agent" capable of multi-step actions and up to **20+ minutes** of autonomous work, showcased in [this tweet](https://x.com/NotionHQ/status/1968744673347830152).
   - The update introduces a **Personal Agent** and a **Custom Agent** integrating across Notion Calendar, Notion Mail, MCP, etc.
- **Moondream 3 Visualizes SOTA**: Vik introduced **Moondream 3**, a **9B**-parameter Mixture-of-Experts vision-language model with **2B** active parameters that achieves SOTA visual-reasoning and open-vocabulary object-detection performance; the weights are on [Hugging Face](https://huggingface.co/).
   - It introduces visually grounded reasoning, state-of-the-art **CountBenchQA** results, **32k**-token context support, **SuperBPE** tokens, and easily fine-tunable weights.
- **Luma Labs Ray3 Reasons Hollywood HDR**: Luma AI presented [Ray3](https://x.com/LumaLabsAI/status/1968684330034606372), claiming it as the first reasoning video model with studio-grade HDR output, free inside **Dream Machine**.
   - Highlighted features: **Draft Mode** for fast iteration, enhanced physics & consistency, visual-annotation control, and **10/12/16-bit HDR** with EXR export, getting praise for Hollywood-level fidelity.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo VS Code Extension Enters Preview**: A new open-source **Mojo VS Code extension** is now available in preview, with access to bleeding-edge builds directly from the [GitHub repository](https://forum.modular.com/t/preview-new-mojo-vs-code-extension/2283).
   - The extension will soon be available in the pre-release channel, making it easier for developers to integrate **Mojo** into their **VS Code** workflow.
- **Mojo LSP Faces Stability Woes**: Users reported instability with the **Mojo LSP**, experiencing crashes, hangs, and memory leaks, often needing to manually terminate the `mojo-lsp-server` process.
   - Despite these issues, a Modular employee noted ongoing efforts to improve the **LSP**, with one member suggesting that restarting **nvim** helps in recovering the **LSP**.
- **Mojo's IDE Landscape**: **VSCode** and its derivatives are the most used IDEs for **Mojo**, though many developers internally use **Vim**, **Zed**, and others.
   - The **Mojo LSP** server is shipped with the `mojo` package to give users flexibility to use other IDEs, and a [simple LSP setup in Lazynvim](https://forum.modular.com/t/fyi-simple-lsp-setup-in-lazynvim/1142) was provided.
- **Exporting Mojo Code to C/C++ Gotchas**: When building a **.so library** from **Mojo code**, only functions defined as `fn` can be called from **C code**, as `def` implicitly raises, making it incompatible with **C-ABI**.
   - For **C++** interop, including `extern "C"` linkage in the header is essential to prevent symbol mangling issues; one member discovered that this is essential for preventing symbol mangling in **C++**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **DeepMind makes Fluid Dynamics More Fluid**: **Google DeepMind** announced they used *novel AI methods* to present the first systematic discovery of new families of unstable singularities across three different fluid equations, according to [this blog post](https://deepmind.google/discover/blog/discovering-new-solutions-to-century-old-problems-in-fluid-dynamics/).
   - The news was also announced [via X](https://x.com/GoogleDeepMind/status/1968691852678173044), marking a significant advancement in solving century-old problems in fluid dynamics.
- **Member Teases ClaudeAI Reveal**: A member shared a [link](https://x.com/claudeai/status/1968705632095158393) hinting at an upcoming **ClaudeAI** announcement.
   - The message simply stated, *"is tomorrow the day?"*
- **Qwen3 Next Architecture Reproduction Achieved**: A member created a trainable reproduction-ish of **Qwen3 Next** architecture with different routing and shared it on [GitHub](https://github.com/CoffeeVampir3/Architecture-Qwen3-Next-Like).
   - They found **gated delta nets** promising and hope to read about **gated linear attention** next week.
- **LLM API vs Power Automate in a Redaction Cage Match**: A member inquired about using a **custom API LLM solution** versus **Power Automate with Copilot Studio** for redacting personal info from **60,000 documents** monthly.
   - They suggested the **custom API LLM solution** could be cheaper and faster.
- **Background Superimposition is Harder than it Looks**: A user is seeking insights on optimizing an agent to superimpose a **snow background** onto a [storyboard image](https://cdn.discordapp.com/attachments/1269724655405498429/1418506270251552798/scene_1_storyboard.png?ex=68cf0729&is=68cdb5a9&hm=b45596d46cb25df517e088bc4acfb04ecfc0fc1a5a963b124e3ad9b560d8d59d&).
   - Despite the seeming simplicity of the task, the [resulting image](https://cdn.discordapp.com/attachments/1269724655405498429/1418506359368060998/scene_image_165.png?ex=68cf073e&is=68cdb5be&hm=4a7998ff34a720623fdd0d7aa412a8f0f80fec998862bee819f34e71ed54667e&) doesn't meet expectations, and the agent is not working as expected.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Coding Agent Quality Varies**: Members report varying quality among coding agents like **qwen-code**, **Cline**, and **Kilo**, with larger models like **qwen3-coder (480B)** performing better, but still showing unpredictable behavior.
   - Smaller models sometimes yield surprisingly good results despite a lower good-to-bad ratio, prompting curiosity about how **Aider's** user-guided approach compares.
- **Aider Edits Directly**: Users prefer **Aider** for targeted single edits, even with smaller models like **gpt3.5-turbo**, citing its speed and directness.
   - One user combines **Aider** with **Claude 4 Sonnet** and **gh cli** for PR reviews, contrasting it with more agentic tools like **opencode** or **qwen-code** that process large code portions for minor changes.
- **Deepwiki Softens Codebases for Inspection**: A member shared [Deepwiki](https://deepwiki.com/search/how-is-treesitter-used_8f8837ad-10a0-4484-8359-314f794407f3) as a resource to quickly ask questions of codebases, to *"soften them up"* before diving into the details.
   - They demonstrated by using **Devin Chat** from Deepwiki's aider entry to answer the question of how **tree-sitter** is used.
- **Local MLX Model Babbles via Aider**: A member got the **mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit** model working with aider via the **mlx-lm server**, using `aider --openai-api-key secret --openai-api-base http://localhost:8080/v1/`.
   - Adding `openai/` in front of the `--model` option was key: `aider --openai-api-key secret --openai-api-base http://127.0.0.1:8080 --model openai/mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`.
- **Graph Usability Expands**: A user suggested improving graph usability by adding the ability to **deselect outlier data points** to improve the focus on representative data.
   - Currently, *it's not possible to use the graph easily*.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Human Review Tooling Seeks Testers**: A member is developing **Human Review Tooling** for **manual QA**, **error analysis**, and **data labeling**, seeking testers, especially from academia, with [a demo video available](http://dub.sh/orizudemo).
   - The next phase involves generating graders using human feedback, incorporating deterministic methods and **LLMs as Judges** via **GEPA**.
- **GEPA Optimization Guide Coming**: A member inquired about which models perform well for **GEPA optimization**, besides OpenAI models, in the **general** channel.
   - Another member reported success with **Gemini-2.5-Pro**, **GPT-4/5-nano/mini/main** all versions, and **Qwen3-8B+**, while also having *heard* success with **GPT-OSS**.
- **MLFlow Flexes Observability Muscles**: A member inquired about which tools are working well for **observability and evals**, especially for GEPA.
   - Another member mentioned that **MLFlow** is deeply integrated, with the **MLFLow team** also working on better integrations with GEPA, and supports things like best_valset_agg_score and pareto frontier related like paro frontier aggregate score, and dashboards.
- **ColBERT Context Crisis**: A member noted that despite **jina-colbert** accepting up to **8,192 tokens**, results are not great with longer contexts.
   - They suggested repeating the **CLS token** in each chunk and trying with/without; the channel discussed **CLS Token Chunking Strategy** to address issues with long contexts in **jina-colbert**.
- **MergeBench Paper**: A member shared [MergeBench](https://yifei-he.github.io/mergebench/).
   - No further details were given about this paper.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Magistral Small Model makes debut**: Mistral AI launched the [Magistral-Small-2509 model](https://huggingface.co/mistralai/Magistral-Small-2509) on Hugging Face, observing that **small model training speed** doesn't scale linearly due to overhead for small tensors.
   - The team noted that **VRAM consumption** is inefficient, with **Qwen3-Next** requiring more VRAM than an equivalent **Llama** model at 16x the batch size.
- **Moondream gets a refresh**: A new **moondream** release was announced, based on [this vxitter post](https://vxtwitter.com/vikhyatk/status/1968800178640429496), however members voiced concerns about its *wonky licence*.
   - It was compared unfavorably against other **moondream** models in the space.
- **QSilver Quantum Workshop Opens Applications**: QBangladesh is conducting the free **QSilver Quantum Workshop** from **Oct 18–19 & Oct 25–26**, 1:00–2:30 PM UTC, covering **Qiskit**, **Cirq**, and more, via Zoom.
   - The workshop features guest speakers, code walkthroughs, and hands-on programming, prioritizing women & disadvantaged groups with applications due by **Oct 11/12** ([application form](https://forms.gle/1VM4eVwUtSmMiWFJ7)).
- **China lags in Multimodal Models**: Members observed that China appears to have the fewest multimodal models compared to other regions.
   - One member inquired about the *hurdle with image recognition training*, particularly when using MoE architectures.
- **No Gemini 3.0 or Claude 4.5 this week**: A member shared their disappointment with no release of **Gemini 3.0** or **Claude 4.5** this week, proclaiming *This week we have been duped*.
   - They then quipped *We got AI glasses though!* (without AI, so far).



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Researcher Credits: Lifetime or Daily?**: Users debated whether **Kimi Researcher credits** have a lifetime limit or a refresh, and one user feared trying it, thinking they had a limit of **3**.
   - A user clarified that the free sessions refresh daily, with some users reporting **5 free sessions** and others seeing only **3**.
- **Researcher Limit Tied to Beta Access?**: Some users suggested that a higher **Researcher limit** may be associated with access to **beta testing** through a waitlist.
   - It was hinted that users who applied to use Research during its beta phase might have a higher allowance.
- **Doubts Emerge Over Kimi's Dart Performance**: A user inquired about **Kimi's performance in Dart** and whether the examples provided are real-time.
   - It was claimed that clicking the examples doesn't run the model in real time but displays pre-generated outputs.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz Hatching a New Company**: George Hotz announced the launch of his new company focused on *making useful products for people with compute*, which is [separate from TinyCorp](https://discord.com/channels/1068976834382925865/1068976834928193609/1418671132860616806).
   - The new company addresses the need for more accessible computational tools with an emphasis on improving user experience and expanding application range.
- **TinyGrad Gets a Facelift**: **TinyGrad** is receiving significant updates aimed at enhancing its functionality and refining the user experience.
   - These updates are expected to broaden the range of applications for the **TinyGrad** framework, reinforcing its position as a versatile tool.
- **Stable Diffusion Import Encounters Module Nightmare**: A member encountered a `ModuleNotFoundError` while trying to run the **Stable Diffusion model** due to a missing `extra` module.
   - The specific error cited was `from extra.models.clip import Closed, Tokenizer`, indicating an issue with the module's availability during the import process.
- **PYTHONPATH Hack Falls Flat**: A suggestion to use `PYTHONPATH=.` as an environment variable to resolve the module error was proposed but *didn't work* for the member experiencing the issue.
   - This indicates that the problem might be more complex than a simple path resolution issue.
- **Extra Package MIA from PyPI**: A member pointed out that the `extra` package isn't part of the **PyPI** release, questioning whether the installation was done via **PyPI** or directly from a **repo**.
   - The original poster confirmed they installed from source, suggesting that the missing module might be due to an incomplete or incorrect source installation process.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Contributor Server Declares Independence**: A moderator specified that the server exists for discussion amongst **contributors**; it is not a general help or *drive-by survey* zone.
   - Those seeking assistance or wanting to conduct surveys were asked to DM the moderator for pointers to other servers.
- **EmbeddedResources go rogue**: A member noted that `EmbeddedResource` doesn't follow the `EmbeddedResources` structure and doesn't extend `BaseMetadata`, meaning it lacks `name` and `title`.
   - They want to confirm if `EmbeddedResource` should contain `name` and `title` properties.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Launches Code-Supernova**: Windsurf introduces **code-supernova**, an *agentic coding model* supporting images with a **200k context window**, available [free for a limited time](https://x.com/windsurf/status/1969148529693081922) for individual users.
   - This new model aims to provide enhanced coding assistance within the Windsurf environment.
- **Windsurf Unveils Queued Messages**: Windsurf announces the availability of a new *queued messages feature* alongside the launch of **code-supernova**.
   - This feature likely allows users to schedule and manage message delivery within the Windsurf platform, improving workflow efficiency.
- **Reddit Buzzes About Windsurf's Stealth Model**: Users are invited to discuss the free stealth model on [Reddit](https://www.reddit.com/r/windsurf/comments/1nlg25z/free_stealth_model_just_dropped_in_windsurf/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).
   - The discussion provides a platform for users to share feedback, experiences, and potential use cases for the new model.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1418314639003095132)** (1105 messages🔥🔥🔥): 

> `Perplexity Deep Research exhausted, Image Upscaling Tools, Comet and Google Drive connector issues, Perplexity's Sam AI assistant, Comet and Canvas quizzes` 


- **Perplexity Pro's Deep Research Limits Frustrate Users**: Users reported [issues with Perplexity Pro](https://www.perplexity.ai/rest/rate-limit/all) showing **Deep Research as exhausted** despite minimal use, with one user reporting *only 3 searches* used that day.
   - Members suspect it's a bug, with suggested workarounds including *logging out and back in*, or contacting support, though some noted delays in receiving replies from support.
- **Image Upscaling Quest**: Members discussed the best free image upscaling tools for mobile, with **Pixelbin**, **Upscale.media**, and **Freepik Image Upscaler** recommended, while another member suggested cracking Adobe Photoshop, which requires a PC.
   - One user warned against Adobe Firefly due to limited free changes, highlighting options like cracked Remini for upscaling.
- **Comet Has Google Drive Connector Troubles**: Users reported issues with the **Google Drive connector** in Comet, causing the browser to crash, but suggested using the connector in other browsers like Edge as a workaround.
   - They emphasized that for those on Enterprise Pro, the GitHub connector isn't currently supported.
- **Sam's Surface-to-Air Customer Service**: Users found Perplexity's AI support agent, **Sam**, unhelpful and slow to respond, recommending to specifically ask for a *human agent* in the chat.
   - One member was told to expect a *24-48 hour* response time and a suggested approach was to explicitly request human support in email correspondence.
- **Cheating Attempts on Canvas Tests Shut Down**: A user inquired about using Perplexity for Canvas quizzes but found that **Comet Assistant no longer provides answers for quizzes**, exams, or tests due to recent updates.
   - Members cautioned against cheating, noting that Canvas proctors can see commands run and tab switches, leading to disqualification.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1418410401384759306)** (3 messages): 

> `Grounded Responses, Bitcoin News, Memory Management` 


- **Responses are Grounded with Search**: A member asked for examples or more details on how responses are grounded with search.
   - Another member replied that they requested news regarding **Bitcoin** that was at most **7 days old**.
- **Fake Bitcoin News appears in search**: A member reported that the search returned a *fake* [Bloomberg article](https://www.bloomberg.com/) stating that **BTC** hit a record high of **72k**.
   - The user noted that **BTC** is over **100k**, implying the article was inaccurate.
- **Perplexity AI Cookbook Article Shared**: A member shared a link to a [Perplexity AI cookbook article](https://docs.perplexity.ai/cookbook/articles/memory-management/chat-with-persistence/README) on memory management.
   - The article discusses how to implement **chat with persistence**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1418311261011054746)** (872 messages🔥🔥🔥): 

> `Seedream 4 High Res Removal, Gemini 3 Speculation, Brute Force Discussion, LM Arena Login Issues` 


- **Seedream 4 High Res Vanishes, Renamed, then Downgraded**: Users noticed **Seedream 4 High Res** was removed, then reappeared as **Seedream4**, but with significantly reduced quality (**2K instead of 4K**) and smaller file sizes, leading to disappointment and speculation that it was a deceptive renaming.
   - A member stated *It's only 2K (2034px) while the 4K is 4096px* with others complaining and hoping for its return: *Damn, I feel scammed.*
- **Ocean Stone and Ocean Reef Speculation Intensifies**: Members actively speculated about **OceanStone** and **OceanReef**, with some theorizing they are variants of **Gemini 3 Flash** and **Gemini 3 Pro**, others wondered how **Gemini 3.0 flash** would perform.
   - Members noted that, *Remember king fall? Gemini is always very cryptic with their code name models and they retract them before long*.
- **Is Reasoning brute-forcing?**: A discussion revolved around whether **fine-tuning**, **model scaling**, and **reasoning** constitute *brute force* methods in AI training with definitions debated.
   - Some argued that using long reasoning is in fact brute forcing, with one member explaining that* if all optimal paths lead to a lot of work* then it indeed falls under the brute force definition.
- **LM Arena Experiences Troubles**: Multiple users reported login issues and errors on the **LM Arena** website, prompting admin response and requests for detailed bug reports and screenshots in the appropriate channel.
   - One member stated *<@283397944160550928> Even now website is not at all working properly... Giving errors*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1418310727294386360)** (484 messages🔥🔥🔥): 

> `Exploding Gradients, Data Cleaning, Aphantasia vs Visual Imagination, Titans Architecture, Google AI Pro` 


- **Gradient Spike triggers a Training Meltdown**: During training, the grad norm spiked to *quintillions*, causing a catastrophic failure, but one member joked that *That spike is where AGI emerges*, [this image](https://cdn.discordapp.com/attachments/1179035537529643040/1418325100746506291/Screenshot_2025-09-18-12-57-02-62_984e1414ae90666a90f12ff17ec14a7f2.jpg?ex=68cf072f&is=68cdb5af&hm=6b60a8de6e0f21c44e45fb438886432f64cbe336187c8005c19d9013b02c8039&) shows the moment the **high score** was obtained by **Magistral SFT**.
   - Despite the chaos, one member suggested continuing until it hits infinity and linking a [relevant GIF](https://tenor.com/view/to-infinity-and-beyond-buzz-lightyear-woody-toy-story-beyond-all-limits-gif-17329486212536226017).
- **The WikiArt Dataset's Quality Questioned**: A member pointed out the poor quality of the widely used **Wikiart dataset**, highlighting **jpeg artifacts** and suggesting that it's massively flawed, illustrated by [this comparison](https://cdn.discordapp.com/attachments/1179035537529643040/1418632339533070356/dezoomify-result.jpg?ex=68ced3d2&is=68cd8252&hm=c29a05e841dc79ca66d27e65bb6d4ce95b98f68cc8cfcbbc443de90e6b0b3c24&).
   - Another member suggested training models on flawed input tokens to produce flawless outputs, emphasizing the need for robust filters.
- **Exploring Aphantasia and Visualization in Minds**: Members shared their experiences with visualization, discussing the spectrum from vivid mental imagery to aphantasia, with one member visualizing *entire environments, in full detail* and noting [aphantasia](https://my.clevelandclinic.org/health/symptoms/25222-aphantasia) being a condition where people cannot visualize images.
   - The discussion touched upon the benefits of visualization for learning, problem-solving, and memory, as well as potential drawbacks like sleep disturbance due to constant mental imagery.
- **Titans Architecture Sparking Interest**: A discussion emerged around **Google's Titans architecture**, which combines transformers with **LSTMs** for long context handling, referencing [this paper](https://arxiv.org/pdf/2501.00663?)
   - Despite its potential, a member noted the lack of popular implementations and questioned why it didn't gain mainstream traction, while another suggested that Gemini could be a Titans hybrid due to its massive context window.
- **Free Google AI Pro for All!**: One member announced they received a free year of **Google AI Pro**, sparking excitement and joking about burning through **VEO**.
   - Another member celebrated and said: *Time to make that Unsloth mascot moves*, linking to [this tweet](https://x.com/danielhanchen/status/1969160431907352786).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1418348575036936294)** (81 messages🔥🔥): 

> `Meta Horizon Worlds Mobile Competition, AI Minister in Albania, RAM Overclocking Stability Tests, LLM Uncanny Valley, GPTs roleplay` 


- **Meta gives 200k for Horizon Worlds Mobile**: Meta is hosting a competition related to [Horizon Worlds on Mobile](https://developers.meta.com/horizon-worlds/m/mobile-genre-competition) with a prize pool of **$200,000**.
   - This competition encourages developers to create engaging experiences for mobile users within the **Horizon Worlds** platform.
- **Albania Installs AI Minister**: Albania has become the first country to instate an **AI Minister**, a virtual entity composed of *pixels and code powered by AI*.
   - An American user jokingly stated they *would even take GPT2 over our current administration*.
- **Crunching RAM stability with y-cruncher**: Members discuss testing RAM stability using **Memtest**, **Prime95**, and **Cinebench**, recommending **y-cruncher** for optimal memory stability assessment.
   - One member suggested that *Memtest can show you exact errors and broken mem sticks*.
- **LLMs Invoking the Uncanny Valley**: Members discussed the **uncanny valley** in LLMs, noting that it can occur when an assistant model starts referring to personal life and experience as a human.
   - One member said *it does trip me a bit when an assistant mode suddenly starts referring to its personal life and experience as a human*.
- **GPTs Roleplaying, a flawless experience**: Members discussed roleplaying with models, with one member claiming to have *zero issue when RPing with models and have them use first person and all*
   - This was in the context of defining behaviors from an LLM that would resemble an uncanny valley, which for some does not include roleplaying.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1418324812501352589)** (37 messages🔥): 

> `GLM-4.5V finetuning, NeMo 2.0 framework, GPT-oss errors, Gemma3 27b memory requirements, Swiss German audio datasets` 


- **GLM-4.5V finetuning: Is it supported by Unsloth?**: A user asked if **Unsloth** supports finetuning **GLM-4.5V** with images, referencing vision **SFT** and **RL** features, and one user responded *it's supported by transformers so yes most likely*.
   - No further details were provided on specific implementation or configurations.
- **NeMo 2.0 framework interest surges for LLM training**: A user inquired about resources for training models in the **NeMo 2.0 framework**, noting increased lab adoption but lacking official documentation.
   - One user expressed their intention to *create something for it*, implying a community-driven effort to fill the documentation gap.
- **GPT-oss error in local GPU**: A user reported a **RuntimeError** related to **UnslothFusedLossBackward** when running the [GPT-oss notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb) on their local GPU, after the code worked well on Google Colab.
   - The error indicates an attempt to modify an inplace view, potentially requiring cloning the output of the custom Function for resolution.
- **Gemma3 27b full finetune demands massive memory**: A user asked about **GPU** memory requirements for full fine-tuning (**FFT**) on **Gemma3 27b**, reporting CUDA out-of-memory errors with an **RTX 6000 Pro**.
   - It was stated that **FFT** requires a minimum of *~432GB* due to activation and optimizer parameter storage, with a recommendation to use **QLoRA** as a more memory-efficient alternative.
- **Swiss German audio datasets sought**: A user requested pointers to *great swiss german audio - transcirptions datasets*, but was warned *don't expect good quality data in public, let alone Audio and transcription on top of that*.
   - A link to a [German audio dataset on HF](https://huggingface.co/datasets/iqrabatool/hui-audio-corpus-german-other-dataset/tree/main) was provided, with a suggestion to generate, transcribe, and align the audio oneself.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1418321928547991562)** (21 messages🔥): 

> `SLED potential for brain damage prevention, LLM Layer Usage in Inference, Server Supporter Role, Early Exit Logits, SLED Technique` 


- **SLED: Brain Damage Preventer?**: A member pondered if **SLED** could potentially prevent brain damage from **SFT**, suggesting integration with tools like *llama.cpp*.
   - Another member inquired about the research aspect, suspecting an **AI competition** for hiring researchers.
- **All Layers Matter for Logits**: A member clarified that during inference with *llama.cpp* or *MLX*, all layers are used sequentially, with only the final layer's logits used for next token probability calculation.
   - They explained that **SLED** utilizes results from each layer directly to calculate logits, potentially improving predictions.
- **Boosting for the Win!**: A member asked how to get a supporter role and another member responded that you need to support the server with boosts to get it.
   - The server is auto set for server boosters by discord, so boost to win.
- **SLED Leverages All Layers for Precision**: A member explained that **SLED** improves **LLM** predictions by using information from all layers, not just the last one, by reusing the final projection matrix to create probability distributions.
   - They added, *SLED refines the LLM’s predictions by incorporating information from different stages of its processing*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1418311257374720050)** (402 messages🔥🔥): 

> `Auto Model, Cursor CLI, Cursor Terminal issues, MCP for managing projects, Windsurf vs Cursor` 


- **Auto-Matic Choices Spark Model Debate!**: Users discussed the **Auto model** in Cursor, with some finding it *really good* even if it wasn't free, while others find it only suitable for **code refactoring** and simple tasks.
   - One user noted they can *basically build anything* with **Claude Sonnet 4**, whereas the Auto model failed complex tasks, particularly if there was a missing closing brace, leading to attempts to delete and rewrite entire files.
- **Cursor CLI commands are still wonky**: A user reports that the **command lines** are still NOT working first go on fresh install and that *it takes 8 messages before cursor figures out how to run a command*.
   - A user mentioned that fixing a waste of money on cursors end, fixing that one issue could save Cursor 10 million a year.
- **Terminal Tempest: Users Report Terminal Commands Getting Stuck!**: Several users reported issues with **Cursor getting stuck** when running terminal commands, both in the IDE and Cursor CLI, especially after updating, and noted the 'skip' button missing in *run everything* mode.
   - The Cursor team acknowledged the terminal issues and confirmed they are under active development, prompting users to try switching to **Early Access** or **Nightly** builds to potentially resolve the problem.
- **Managing Multiple Projects with MCP**: A user invented an **MCP (Multi-Cursor Project)** to control multiple open projects, ports, and hosts within Cursor, ensuring that if another project tries to use the port of another project, it identifies and changes the port automatically.
   - Another user faced issues getting **Cursor CLI** working with their MCP, encountering a *No MCP servers configured* error despite having the correct configuration files; they solved it by downloading the desktop app.
- **Windsurf or Cursor Subscription?**: Users compared **Windsurf** and **Cursor subscriptions**, with one user uninstalling Windsurf after a few minutes and opting for Cursor, while others use both depending on available credits or quotas.
   - There were also discussions on early adoption pricing and the value of GPT-5 prompts, with some users on the \$10 plan getting a significant number of high-quality prompts.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1418446474915872860)** (3 messages): 

> `Background Agents, GitHub repository access, configuration issues` 


- **Background Agents Trigger Configuration Nightmares**: Members are reporting the **Background Agents** feature is glitchy, ignoring **Dockerfile** instructions, failing to run **Docker** in its default container, and lacking detailed documentation on configuration and **environment.json** properties.
   - One user lamented that this feels like an alpha stage release, not a finished product.
- **GitHub Account struggles to connect to Cursor**: A user reported issues connecting their **GitHub** account to **Cursor**, preventing them from choosing a repository to chat with the background agent.
   - They tried unlinking and relinking the **GitHub** account, ensuring **Cursor** had all necessary permissions, but the problem persisted.
- **Background Agents can't git+ssh packages**: A user is encountering an issue where **Background Agents** fail to fetch **git+ssh** packages during `yarn install`, despite **Cursor** showing these repos as 'installed' in the access dialog on **GitHub**.
   - The error message indicates a failure in cloning the repository with a 'Repository not found' error.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1418560525809483847)** (3 messages): 

> `Voicera Audio Search Engine, SillyTavern iOS Clone` 


- ****Voicera** Turns Audio Searchable**: **Voicera** ([http://voicera.trixlabs.in/](http://voicera.trixlabs.in/)) is pitched as an audio search engine that transforms audio into actionable insights, promising to turn hours of audio into instant, verifiable answers grounded in user's own audio files.
   - The tool allows users to upload audio, search using plain language, and get AI-generated answers with time-coded segments, aiming to streamline the process of finding key moments, quotes, or decisions in recordings.
- **SillyTavern iOS Clone App Emerges**: An iOS developer introduced a free **SillyTavern** iOS clone, [Loreblendr AI](https://apps.apple.com/us/app/loreblendr-ai/id6747638829) ([https://loreblendr.ai/](https://loreblendr.ai/)), designed for users seeking a native **SillyTavern** experience on iOS devices.
   - Despite acknowledging it cannot match all **SillyTavern** features, the developer expressed satisfaction with the app's current state and highlighted its user-friendly interface compared to existing chat apps.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1418319193371054171)** (305 messages🔥🔥): 

> `Responses API Benefits, Kimi K2 0711 Downtime, GPT-4o Alternative, Deepseek V3 429 Errors, Chutes Pricing` 


- ****Kimi K2** Glitches, Model Downgrade Debated**: Users reported errors with **Kimi K2 0711** on ST and some inquired about downtime, while others suggested upgrading to the newer **0905** model, which costs the same.
   - One user pointed out the free version of **Kimi K2** ([https://openrouter.ai/moonshotai/kimi-k2:free](https://openrouter.ai/moonshotai/kimi-k2:free)) is no longer available.
- **DeepSeek Proxy Woes, Rate Limits Run Rampant**: Users are encountering **Error 429** when trying to use DeepSeek proxy, especially the free models, indicating rate limiting issues with **Chutes**.
   - One user suggested that Chutes might be throttling free users due to the high demand on weekends, making things almost pay-to-use.
- **Gemini's NSFW Filter, a Source of Frustration**: Users discussed the issues of using **Gemini** for NSFW content, with some experiencing API key bans when attempting explicit roleplay.
   - One user noted Google "wishes money while screwing people up" referencing their own experience with **Gemini** and expressing frustration with **Google AI Studio**.
- ****Allow Fallbacks** Fails, Chaos Ensues**: Users reported that the `allow_fallbacks: False` setting isn't working as expected, with requests still being routed to other providers despite being set to prevent this.
   - A member suggested that **OpenRouter** might reroute requests if the specified model isn't available, if unsupported features are being used, or other unknown conditions.
- **DeepSeek R1 vs 3.1: A User-Driven Showdown**: A user touted **Deepseek R1** as superior to **3.1** for roleplaying due to its ability to convey sarcasm and irony, while another stated that 3.1 sounds the most like *base-level GPT* and just *describes* rather than *talks*.
   - That same user gave the caveat that Deepseek 3.1 requires *a shit ton of prompt* and stated that it was now the last one they'd touch with a ten-foot pole.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1418447600327327845)** (7 messages): 

> `Claude popularity, code-supernova model` 


- **Claude Coding Popularity Swells Weekly**: Members noted that **Claude** gets more popular during the week, adopting a *"on the job vibe coding".*
   - This suggests **Claude** is increasingly favored for professional coding tasks during workdays.
- **`code-supernova` Model Stealthily Appears**: A new stealth model, `code-supernova`, has emerged, allegedly by **Anthropic** and rumored to be **Claude 4.5**, according to an [image analysis](https://cdn.discordapp.com/attachments/1392278974222307469/1418547172177088713/image.png?ex=68ce8481&is=68cd3301&hm=fa7b08d33a708faefafdb4db92fe2bad8665a9423e8faaaaac34a434723fb084&).
   - Users describe the model as decent but somewhat lazy, providing only the bare minimum implementation and not behaving like **Claude**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1418329395235655823)** (179 messages🔥🔥): 

> `qwen3-next 8bit on MacOS, Apple MLX and qwen-next, Granite 3.3 8b fine tuning, LM Studio Hub Navigation, gpt-oss 20b performance` 


- **Qwen3-Next Glitch on MacOS**: A member reported issues getting **qwen3-next 8bit** to work on MacOS, despite having sufficient memory, with the model failing to respond and indicating a loop failure.
   - They were using the version available from the **LM Studio community**.
- **Qwen-Next Excites Apple MLX Users**: Users of **Apple MLX** reported impressive performance with **qwen-next**, citing around **60 tok/sec** on an **M4 Max** at **6-bit MLX**, with strong general knowledge, coding, and tool calling abilities.
   - One member described it as the *best model they can run*.
- **LM Studio Hub is WIP**: Users are experiencing navigation problems within the **LM Studio Hub**, making it difficult to find content, search, or return to a central landing page after following a link.
   - This functionality is reportedly a work in progress (WIP), with search being a future feature, and the current documentation being scattered in a chat system according to [this comment](https://discord.com/channels/1110598183144399058/1404127827007115326/1404131243339153590).
- **GPT-OSS 20B Plunges into Infinite Loops**: A user testing **gpt-oss-20b** on low-end hardware experienced an infinite loop, where the model generated extensive and irrelevant content due to context overflow.
   - Limiting the context window may help prevent this, as the model can get stuck thinking about it's own generated text.
- **LM Studio API avoids loading MCPs**: A user inquired about loading **MCPs** (Model Component Packages) when serving via the **LM Studio API**, but they are not yet supported.
   - Work on this functionality is ongoing.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1418348426738798703)** (65 messages🔥🔥): 

> `Intel ARC demise?, ARM's viability, Xeon Gold prices, VRAM Frequency Boost, DDR5 as VRAM` 


- **Intel sticks fork in ARC**: Members speculated that **Intel** might be abandoning its **ARC GPU** line after a deal with **NVIDIA** to integrate their tech into Intel's CPUs, with one member suggesting [this aligns with NVIDIA's previous condition](https://www.nvidia.com/en-us/) for partnership: Intel giving up GPU ambitions.
   - Another member stated that Intel is failing and ARC loses them money, so they are dropping employees like flies each quarter or so, and they can only do that for so long.
- **ARM wrestling with Desktop Viability**: While **ARM** is seen as a viable contender in the mobile market due to its power efficiency, some members doubt its suitability for desktop or server applications, citing the example of **Apple**.
   - They argued that in the server space, the need for large numbers of cores with low power consumption is less critical because *power and HVAC in data centers is free.*
- **Xeon Gold not worth the price**: Members discussed the **Xeon Gold 6230** and **5120** processors, highlighting their AVX-512 capabilities but noting they are still disgustingly expensive, even if acquired cheaply.
   - One member shared a [link to a refurbished Lenovo ThinkStation](https://pcserverandparts.com/lenovo-thinkstation-p920-tower-2x-intel-xeon-gold-6148-2-40-ghz-20c-32gb-ddr4-none-no-gpu-no-os-refurbished/?sku=LSB%20111111&utm_source=google&utm_medium=cpc&utm_campaign=22792690068&utm_content=pmax_6595531792_&utm_term=&matchtype=&device=c&placement=&gad_source=1&gad_campaignid=22792693179&gbraid=0AAAAAoJiCjUm6M_v_y5nPIb-c8B5dlZQX&gclid=Cj0KCQjw_rPGBhCbARIsABjq9ceAYDEGIljovOb58R6OfdYL4ege4bWH_bq29yd8cZOaR7eDoEL-SCUaApxFEALw_wcB) featuring dual Xeon Gold 6148 processors as a decent option.
- **VRAM got a speed boost**: One member mentioned that increasing the **VRAM frequency** on their **3090** GPUs by **1500** resulted in a decent improvement in tokens per second.
   - They confirmed the stability of this overclock across **Asus, MSI, and Zotac cards**.
- **DDR5 VRAM?**: A member inquired about using **DDR5 RAM** as **VRAM**, leading to clarifications about how system RAM can be used by the GPU, but the **PCIe link** introduces delays.
   - It was pointed out that GPU use shared memory from the system, but the PCIe link has a delay that usually makes things a bit quicker to have the CPU share some of the workload if it doesnt fit.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1418347557528272947)** (5 messages): 

> `Qwen3-Next Architecture, Gated Delta Net, EVGA Software` 


- **Qwen3-Next Architecture Recreated**: A member shared their attempt to reproduce a trainable **Qwen3-Next** architecture, providing a [baseline transformer-only version](https://github.com/CoffeeVampir3/Architecture-Qwen3-Next-Like-Transformer) and a [Gated Delta Net version](https://github.com/CoffeeVampir3/Architecture-Qwen3-Next-Like).
   - The architectures are claimed to be within **15%** of each other's size, offering a rough idea of how it works.
- **Users Discuss EVGA Software**: A member inquired about **EVGA software**, asking if anyone is using it or has knowledge about it.
   - No further details or responses were provided in the given messages.
- **Gated Delta Net Sparks Curiosity**: After the release of the **Gated Delta Net**, a member expressed unfamiliarity with the concept.
   - No further explanation or discussion about Gated Delta Net was provided in the given context.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1418348314021331016)** (4 messages): 

> `MLIR, Triton, NVVM, NVGPU, GPU Code Generation` 


- **View Triton MLIR Locally?**: A member asked if it's possible to view the **MLIR** produced by **Triton** without compiling the kernel on a **GPU**, i.e., locally on a **CPU**.
- **Triton Relies on GPU Capabilities for Code Generation**: A member stated that there isn't an easy way to view **Triton's MLIR** locally because **Triton** uses the **GPU's device capabilities** to generate code, with different optimizations for different **Nvidia card variants**.
   - They added that hacking the source code would be necessary to accomplish this.
- **Triton's MLIR and NVVM/NVGPU Usage**: A member clarified that **Triton** uses **NVGPU** (but is *unsure about gluon passes*), and that the top level is the **Triton IR**, which then passes through **NVGPU** and others to reach **LLVM**.
   - They noted that **Triton** includes conditional passes depending on capability.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1418402236677095507)** (26 messages🔥): 

> `TMA Descriptor Modification, ILP in GPUs, Shared Memory Matrix Read, cuTensorMapEncodeTiled API, wgmma usage` 


- ****TMA Descriptor Base Address Modification Possible****: A member asked if it is possible to modify a **TMA descriptor base address** inside a kernel, and another member confirmed it is possible with inline **PTX** using `tensormap.replace`.
   - Concerns were raised about concurrency issues when using TMA to load multiple subtensors in parallel, suggesting the need for syncing/locking before modifying the base address, and unlocking after kicking off **TMA async load**.
- ****ILP Defined as Pipeline Parallelism****: A member asked what people mean by **ILP** in a **GPU** context, and a member clarified that **ILP** means pipeline parallelism between independent instructions.
   - Another member added that even within a single thread, there can be independent parallel instructions (ILP).
- ****Shared Memory Matrix Read Strategies Discussed****: A member described a scenario where they need to read a matrix into shared memory where each column is consecutive in memory and each row is not, and they considered several approaches including **TMA 2d async bulk copy**.
   - A suggestion was made to use **wgmma** since the user is on **Hopper**, as it automatically handles swizzling and allows A & B to be in **SMEM**, and A can be "M" major, i.e. col major, but the member pointed out that the rows are not consecutive even across multiple blocks so grouping them doesn't change much for them.
- ****cuTensorMapEncodeTiled API Host-Side Only?****: After a member tried using multiple descriptors with `cudaMallocManaged`, another member suggested passing them in as kernel params and use `cuTensorMapEncodeTiled()` to improve performance.
   - The original member responded that **cuTensorMapEncodeTiled** is a host-side API and cannot be used from the device side/kernel.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1418628483860664362)** (1 messages): 

> `Discord milestone` 


- **Server Soars to 20K Members!**: The Discord server has reached a milestone of **20,000 members**, marked by celebratory <:goku:1273671556324790362> emojis.
   - This growth underscores the community's increasing engagement and interest in GPU technology and related discussions.
- **Community Growth Fuels Excitement**: The server's expansion to **20,000 members** signifies a vibrant and active community centered around GPU-related topics.
   - Members are celebrating this achievement, anticipating even more diverse perspectives and collaborative opportunities within the group.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1418386700496797746)** (1 messages): 

> `HUAWEI CONNECT 2025, SuperPoD Interconnect, AI Infrastructure` 


- **Huawei's SuperPoD Interconnect Leaps into AI**: At HUAWEI CONNECT 2025, a keynote address highlighted the *Groundbreaking SuperPoD Interconnect*, which is set to lead a new paradigm for **AI Infrastructure**; [details here](https://www.unifiedbus.com/en/news/hc-xu-keynote-speech).
   - The focus was on pioneering advancements in technology and applications.
- **AI Infra Pods**: A member shares insight on new AI Infra Pods connecting to Huawei.
   - The **SuperPoD Interconnect** claims to lead to a new paradigm.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1418408794563809341)** (8 messages🔥): 

> `Nvidia Interview, Byte Pair Encoding, DSA for ML, CUDA for Round 1` 


- **Nvidia Interviewer Asks Byte Pair Encoding**: An interviewee was asked to code **Byte Pair Encoding from scratch** for a Sr. Deep Learning Algorithm Engineer position at **Nvidia**.
   - The interviewee prepared with **DP LeetCode** questions and was surprised by the question asked.
- **CUDA with Dynamic Parallelism for NVIDIA Round 1?**: A member suggested using **BFS with dynamic parallelism in CUDA** for Round 1 interviews.
   - Another member expressed surprise that machine learning roles would ask machine learning related questions.
- **DSA still important for ML roles in India**: One member noted that **DSA is still heavily asked for ML roles in India**, even at companies like **Google**.
   - They observed that interviewers tend to ask difficult questions.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1418422644394426450)** (11 messages🔥): 

> `OpenSHMEM Optimization, Parallel Programming Resources, Learning GPU Programming with LLMs` 


- **OpenSHMEM Use Cases Debated**: A member inquired whether anyone is actively optimizing **OpenSHMEM** or utilizing its operations in practice.
   - No one provided concrete examples or use cases in the discussion.
- **C++ Coder Seeks Parallel Path**: An undergraduate student proficient in **C++** and **deep learning** sought advice on learning parallel programming and low-level programming, after finding the first **GPU mode** lecture on YouTube too advanced.
   - A member suggested starting with the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
- **LLMs Leap into GPU Learning**: A member suggested using **ChatGPT** to learn **GPU programming**, arguing that LLMs offer a faster and more personalized approach than traditional books or lectures.
   - Another member cited the success of [AI-generated Metal kernels by Gimlet Labs](https://gimletlabs.ai/blog/ai-generated-metal-kernels) to support this approach, while advising a bit of seriousness into the replies.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1418709019232567348)** (1 messages): 

> `TorchAO, PyTorch, Quantization, Phi4-mini-instruct, Qwen3` 


- **TorchAO and Unsloth release Native Quantized Models**: The **TorchAO** team and **Unsloth** have collaborated to release native quantized variants of **Phi4-mini-instruct**, **Qwen3**, **SmolLM3-3B** and **gemma-3-270m-it** available via **PyTorch** ([learn more here](https://hubs.la/Q03Kb6Cs0)).
- **Pre-Quantized Models Optimized for Server and Mobile Platforms Are Here**: Pre-quantized models optimized for both **server** and **mobile platforms** have been released for faster model deployment.
- **TorchAO offers Reproducible Quantization Recipes and Guides**: Comprehensive, reproducible quantization recipes and guides have been released, including model quality evaluation and performance benchmarking, for users applying **PyTorch native quantization** to their own models and datasets.
- **Finetune with Unsloth, Quantize with TorchAO**: Users can now finetune with **Unsloth** and then quantize the finetuned model with **TorchAO**.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1418327461716033537)** (1 messages): 

> `Arabic language models, Hala Technical Report` 


- **Hala Releases Arabic-Centric Models**: A member introduced the **Hala Technical Report**, showcasing state-of-the-art nano and small-scale **Arabic language models**.
   - They linked to the [Hugging Face Papers page](https://huggingface.co/papers/2509.14008) and requested upvotes.
- **Scale of Arabic Language Models**: The focus is on nano and small scale models.
   - This aims to provide efficient solutions for Arabic language processing.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 messages): 

bghira: can't wait for pytorch 2.8
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1418362681299308574)** (2 messages): 

> `Kernel Timeout, Driver-Level Timeout` 


- **Kernel struggles with Timeouts**: The kernel cannot exit early based on elapsed time, indicating it has *no concept of time*.
   - The **10 second timeout** is a driver-level implementation.
- **Driver-Level Timeout Details**: The implementation of a **10-second timeout** is handled at the driver level, not within the kernel itself.
   - This suggests a separation of concerns where the driver manages timing constraints.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1418329878482124920)** (3 messages): 

> `Together AI, Blackwell Deep Dive, Semianalysis, NVIDIA, GPU accelerated compiler` 


- **Together AI Plans Blackwell Deep Dive**: **Together AI** is hosting a [*Deep Dive on Blackwell*](https://luma.com/2y9qblpp) with **Dylan Patel (Semianalysis)** and **Ian Buck (NVIDIA)** on **October 1st**.
- **GPU-Accelerated Compiler Surfaces**: A member shared a [GPU-accelerated compiler](https://github.com/Snektron/pareas) they worked on for their master's thesis, noting that *everything from lexing to codegen is done by the GPU*.
- **Open-Sourced Video Editing Model Released**: A member announced the open-sourcing of a [video editing model](https://huggingface.co/decart-ai/Lucy-Edit-Dev) and the release of a larger version via API ([platform.decart.ai](https://platform.decart.ai/)), encouraging the community to contribute to speeding it up.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1418320795968733266)** (14 messages🔥): 

> `NVIDIA Jetson Orin AGX, Earth Observation, Docker in Space, YOLOX, TensorRT` 


- **Planet Labs Flies NVIDIA Jetsons in Space**: Planet, an earth observation company, is flying **NVIDIA Jetson Orin AGX** units on their satellites to perform computer vision and machine learning directly on the satellite for latency-sensitive applications.
   - They leverage **CUDA** and other machine learning tools to operate the Jetson in space.
- **Docker Containers Orchestrate ML in Orbit**: Planet uses **Docker** containers running on standard Ubuntu to host and run algorithms in space, a setup that provides basic guarantees for protecting the host environment.
   - This approach also facilitates dependency management for different ML models without altering the host OS.
- **YOLOX and TensorRT Take Flight**: Currently, Planet is deploying object detection algorithms like **YOLOX** in a space environment and aims to integrate more advanced foundation models and embeddings.
   - The company utilizes **TensorRT** for deep nets and manages CUDA kernels with Python and PyCUDA, circumventing C++ to expedite development.
- **Unified Memory Turbocharges Space-Based GPU**: The **Jetson** modules offer **64 GBs of unified memory**, enabling CPU cores, GPU CUDA cores, and specialized ASICs to access memory without formal host-to-device copies, similar to Apple M-series chips.
   - The Jetson's ability to set different max power profiles enables to meet power targets by throttling different parts of the System on Module.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1418421120180158464)** (4 messages): 

> `Reasoning Gym, pass@3, average_mean_score, visualize_results.py, kakurasu and survo` 


- ****Reasoning Gym** uses Best Score out of Three**: A member inquired whether the zero-shot evaluation results reported in the [Reasoning Gym paper](https://github.com/open-thought/reasoning-gym-eval) used the best score out of three completions, effectively representing *pass@3*.
   - Another member confirmed that it uses the **best score from 3 attempts**, and suggested using `average_mean_score` to get the average score across 3 runs, not the best.
- **Reasoning Gym **Visualizations** Available**: A member shared the [visualize_results.py](https://github.com/open-thought/reasoning-gym/blob/main/eval/visualize_results.py) script.
   - It was the actual code used to create the plots in the original paper.
- **Reasoning Gym **Lacks Results** for New Tasks**: A member inquired about zero-shot evaluation results for new tasks like *kakurasu* and *survo* that were added after the paper's publication.
   - Another member stated that they *haven't been run* as of yet.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1418654239223189656)** (1 messages): 

> `Forwarded messages, Bad English` 


- **Forwarded Message Apology**: A member apologized for their bad English, explaining they forwarded a message from another channel.
- **English Skills Acknowledged**: A member noted their English wasn't perfect due to forwarding a message.
   - They specifically cited forwarding the message from channel <#1191300313928433664>.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1418490104091250778)** (80 messages🔥🔥): 

> `amd-gemm-rs Leaderboard Updates, MI300x8 Performance, amd-all2all Leaderboard Updates` 


- **AMD GEMM Race to the Top**: Many submissions were made to the `amd-gemm-rs` leaderboard, with one submission achieving **first place on MI300x8** with a time of **530 µs**.
   - Other notable submissions included achieving **3rd place** at **534 µs**, and numerous successful runs ranging from **539 µs** to **715 µs** on the **MI300x8**.
- **All to All AMD Personal Bests**: The `amd-all2all` leaderboard saw updates, including one submission landing **5th place on MI300x8** with a time of **1230 µs**.
   - Additional personal bests were recorded at **6.26 ms**, **81.0 ms**, and **98.1 ms** on the **MI300x8**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1418619078398054544)** (3 messages): 

> `User inactivity, Meeting attendance` 


- **User Returns After Inactivity**: A user apologized for their inactivity and announced their return.
   - They indicated they would be joining today's meeting.
- **Meeting Attendance Confirmation**: A user confirmed their attendance for today's meeting.
   - Another user acknowledged this confirmation.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1418714219909484584)** (1 messages): 

> `Hybrid GPU/CPU inference, Mojo for AMX and CUDA, Mojo/MLIR AMX instruction emission` 


- **Mojo Pondered for Hybrid Inference Nirvana**: A member is considering **Mojo** for a hybrid **GPU/CPU inference** scheme, seeking to leverage the speeds of **AMX** and **CUDA** without manual coding for both.
   - They see **Mojo** as a potential solution to avoid separate handwritten implementations for **AMX** and **CUDA**.
- **AMX Instruction Emission Question**: A member inquired whether **Mojo/MLIR** currently emits **AMX instructions** and if any specific coercion is required to achieve this.
   - The user expressed interest in utilizing **Mojo** to automatically generate optimized code for both **AMX** and **CUDA** architectures.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1418536969830535189)** (4 messages): 

> `Vertical Pipeline Setup, Eager vs Lazy Semantics, Tinygrad Autograd, Tensor Fusion Compilers` 


- **Vertical Pipeline Aims for Clarity**: A member is setting up a **vertical slice of the pipeline** from frontend (pyten, rsten, linalg, nn) to backend (autograd, opscpu, opsgpu) to clarify the codebase and textbook for contributors.
   - The goal is to support **MLPs, RNNs, LSTMs, GPTs, and Llamas** once the pipeline is established.
- **Eager Semantics Expose Runtime Overheads**: The discussion touches upon **eager vs lazy semantics**, suggesting that eager semantics after tensor cores result in significant runtime in non-matmul ops.
   - It points to a paper ([Reducing Non-critical Data Movement in Tensor Computations](https://arxiv.org/pdf/2007.00072)) and advocates for **tensor fusion compilers** to reduce data movement, highlighting the transition from PyTorch 1 to PyTorch 2.
- **Tinygrad as Next Step After Autograd**: The conversation suggests that the next autograd to graduate to after PyTorch should be **Tinygrad** due to its lazy-by-default nature and concise codebase (18kloc).
   - This approach provides learners with motivation for transitioning from eager to graph semantics after tensor cores, as the **Tinygrad** codebase offers a clear example of lazy evaluation.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/)** (1 messages): 

krypton_lebg: hi
  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1418322715194032359)** (25 messages🔥): 

> `Context-Parallel Gated DeltaNet, Hackathon Logistics, Kernel Competitions, Team Approvals` 


- **DeltaNet Dream Team Forming**: Members are invited to collaborate on a [context-parallel gated deltanet](https://docs.google.com/forms/u/1/d/17h_NsfErC0c8LI6oKZcY-0M9LTbwO-0Gthp4u5g8oDU/edit?usp=drive_web&ouid=106222972308395582904) project, with plans to submit a proposal early next week.
   - The organizer clarified that participants should bring their own ideas to this open-ended hackathon, and large compute resources will be prioritized for those with well-defined projects.
- **Hackathon Spots Filling Fast**: The GPU Mode event is **nearly fully booked**, with acceptances being rolled out and further evaluations planned for the end of the month.
   - Although sponsorship opportunities are limited due to logistical challenges, the organizer may make exceptions for compelling applications.
- **Kernel Competitions a good fit for newbies**: A member inquired if kernel competitions might be a better fit for newcomers due to their defined tasks.
   - Another member suggested that mentors would be helpful but can't guide everyone, pointing to past projects like **shmem in triton** and **no libtorch torch** as examples of what's possible.
- **Team Approvals are Separate**: A member asked for clarification on whether joining an approved team guarantees individual approval.
   - The organizer confirmed that approvals are separate but emphasized that providing clarity on team projects can facilitate the acceptance process, while also noting the event is nearing capacity.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1418322292097945600)** (9 messages🔥): 

> `Accessibility solutions for the blind, Screen readers on Windows and macOS, leandojo` 


- **AI Desktop Agent for the Blind: A Shot in the Dark?**: A member inquired about open-source projects combining **speech-to-text**, **AI desktop agents for scam email automation**, and **text-to-speech** to assist a blind person with a new PC.
   - The member was looking for *an actual immediate good use of AI slopware*, but found nothing of use.
- **macOS Accessibility Gets Props**: A member suggested that **macOS accessibility** features are quite effective for individuals with **low vision or blindness**.
   - Another user suggested a third might have good advice on the topic.
- **Built-in Screen Readers: A Hidden Gem?**: Members pointed out the existence of **built-in screen readers** for both **Windows** and **macOS**, with the latter being praised as *quite good*.
   - One member offered to share contacts of Discord users experienced with screen readers.
- **leandojo: Anyone Tried It?**: A member inquired whether anyone has used **leandojo** before.
   - Another member suggested trying the **lean zulip** instead, presumably as an alternative or related resource.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1418311881185034251)** (186 messages🔥🔥): 

> `Ray + vLLM patching for RL with TorchTitan, Gated Delta Net, TokenSwap at NeurIPS 2025, Fluid Equations, Atlas vs NIAH` 


- **Patching Ray and vLLM for RL with TorchTitan Proposed**: It was suggested to patch **Ray** and **vLLM** for reinforcement learning (RL) using **TorchTitan**, referencing examples of combining them for RLHF such as [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).
   - However, one member commented that *for RL, torchtitan is not enough of course*, suggesting further requirements for the setup.
- **Gated Delta Net Variants Discussed**: Members discussed doing **gated delta net** (or similar variants) when receiving an entire chunk of keys and values at the same time to produce only a single decay for the entire chunk, with a link to [this paper](https://arxiv.org/abs/2505.23884).
   - The goal would be bidirectional attention and avoiding decay mid-chunk, pretty much deriving whatever the gated delta net equivalent is and implementing that.
- **TokenSwap Receives Spotlight at NeurIPS 2025**: A member shared that [TokenSwap](https://x.com/parjanyapp/status/1968770826179600469) received Spotlight at NeurIPS 2025, which addresses the **performance-memorization tradeoff** by selectively swapping probabilities for common grammar tokens to achieve **10x reductions** in verbatim generation without performance loss.
   - Critics questioned its value, one quipping that it would be akin to *lobotomizing the model*, to which the author clarified that it prevents outputting verbatim/near-verbatim content by swapping in a worse model for simple tokens.
- **DeepMind Discovers New Fluid Equation Solutions**: DeepMind announced the discovery of new families of unstable singularities across three different fluid equations ([blog post](https://deepmind.google/discover/blog/discovering-new-solutions-to-century-old-problems-in-fluid-dynamics/)), detailing multiple new, unstable self-similar solutions for the incompressible porous media equation and the 3D Euler equation with boundary.
   - A paper on the work was also shared: [[2509.14185] Title](https://arxiv.org/abs/2509.14185).
- **Atlas claims to fix NIAH Transformer gap**: It was mentioned that [Atlas](https://arxiv.org/abs/2505.23735) claims to fix the gap between **NIAH** results and **Transformers**, but this is achieved via a larger state size through taylor series polynomial.
   - However, it was pointed out that the *atlas paper is a hot mess in terms of replicability or any sort of reasonable comparisons* and doesn't disclose what size.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1418567051395465247)** (2 messages): 

> `trust_remote_code, downloading dataset` 


- **Trust Remote Code Troubleshoot**: A member was encountering errors on tasks requiring **trust_remote_code** or manual dataset downloads.
   - Another member suggested adding the flag `--trust_remote_code` to resolve the issue.
- **Dataset Download Issues**: The user reported getting errors when tasks required them to manually download datasets.
   - The suggested solution involved using the `--trust_remote_code` flag, possibly indicating a connection between dataset loading and remote code execution.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1418312026517672008)** (138 messages🔥🔥): 

> `Local LLM Hardware Advice, Transformers Training Loop Fix, SpikingBrain-7B, Captcha Solving AI, HF API Model Listing` 


- **Qwen Coder Recommended for Local LLM**: For a local LLM setup, a member recommended using a [standard model of Qwen Coder](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) with the given hardware setup, noting that there would be some VRAM to spare.
   - They cautioned that backend options might be limited due to the multi-GPU environment, and included a [qwen_coder_setup.md](https://cdn.discordapp.com/attachments/879548962464493622/1418446477134401556/qwen_coder_setup.md?ex=68cecf79&is=68cd7df9&hm=af1f70cd602b60e476e2295474e86c3121b78676161253a71907b60e145fa29b&) file.
- **Hack Transforms Training Loops with New Fix**: A member shared a [link to a PR](https://github.com/huggingface/transformers/pull/34191) on GitHub, inquiring if the fix applies even when training transformers with a regular PyTorch training loop.
   - Another member summarized, *Use out.loss from the model and you get the fix in any loop. Roll your own loss and you must handle scaling yourself.*
- **SpikingBrain-7B Promises Faster Performance**: A member shared an [ArXiv link](https://arxiv.org/html/2509.05276v1) to **SpikingBrain-7B**, a non-transformers model based on snns.
   - The paper claims that *SpikingBrain-7B achieves more than **100× speedup** in Time to First Token (TTFT) for 4M-token sequences*.
- **Hugging Face API Simplifies Model Access**: A member asked about accessing and listing models from Hugging Face through the API, and was given [example code](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api).
   - The member shared a code snippet using `huggingface_hub` to list models.
- **Ethical AI and Captcha Conundrums**: Members discussed the ethics of using AI to solve Captchas, with one noting that such practices are *like the pillars of AI ethics and guardrails*.
   - Another member pointed out that the companies creating puzzles and those developing Gemini are often the same, making this approach potentially endless and dominated by *arms dealers*.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1418569223197687929)** (3 messages): 

> `Embedder Collection, SmartTaskTool` 


- **Base Embedder Collection Surfaces**: A member shared a [collection of embedders](https://huggingface.co/kalle07/embedder_collection) to understand **RAG**.
- **Windows Taskbar Tool Spotted**: A member shared a [taskbar tool for Windows](https://huggingface.co/kalle07/SmartTaskTool).


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

olyray: Hello guys. When is the next reading group discussion?
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1418623192691380284)** (1 messages): 

> `Kaggle Notebooks, Unit 1 Content, Exercise Notebooks` 


- **Kaggle Notebooks use Unit 1 Content**: A member created [Kaggle notebooks](https://www.kaggle.com) using unit 1 content and first exercise.
   - The member published two notebooks: [HF-SmolLLM3-Course-Unit1-Chat-Templates](https://www.kaggle.com/code/pardeep19singh/hf-smollm3-course-unit1-chat-templates) and [HF-SmolLLM3-Course-Unit1-Ex1-Chat-Templates](https://www.kaggle.com/code/pardeep19singh/hf-smollm3-course-unit1-ex1-chat-templates).
- **Exercise Notebooks are being produced**: More notebooks for exercises **2 and 3** are being produced.
   - However, the member mentioned facing issues that they plan to resolve over the weekend.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1418312267631431692)** (4 messages): 

> `Starting the Agents Course, New members joining` 


- **New students start the Agents Course**: Several new members have announced they are *just starting* the agents course and are looking forward to learning alongside others.
   - The new students tagged course mentors for guidance.
- **Welcome Newcomers to the course**: Several people in the channel welcomed new students to the course.
   - They tagged course mentors to provide guidance to the new students.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1418312826711445575)** (120 messages🔥🔥): 

> `Codex Use Cases, AI-Assisted Coding Interviews, Ethical Implications of AI in Hiring, Role of AI in Software Development, GPT-4o Mini vs GPT-5` 


- **AI Tutor Tools Spark Debate Over Cheating**: Members discussed a tool called [InterviewCoder](https://www.interviewcoder.co/) that provides real-time hints/solutions during coding challenges, questioning whether such tools constitute **AI assistance** or **cheating**.
   - Some members suggested the tool could be repurposed for educational purposes, offering multiple solution paths as a study aid, akin to *"training wheels for algorithms."*
- **Legal and Ethical Pitfalls of AI Interview Assistance**: Using **AI** to cheat in interviews could lead to civil fraudulent misrepresentation charges, while secretly routing conversations through an app risks violating all-party consent laws.
   - It was noted that unethical and unfair would be *"denying your family food and shelter because you were unwilling to install some software"*.
- **AI Bias Exists in Recruitment**: The discussion touched on the potential for **AI** to amplify existing biases in recruitment processes, citing the common **ML bias** issue where **AI** trained on corporate data might perpetuate the **gender pay gap**.
   - Some members argued that recruiters are already biased and AI *"doesnt introduce the problem. it also doesnt solve it."*
- **"Software Architect" Title in the Age of AI**: A member suggested that when **AI** handles the coding, developers become more like **Software Architects**, managing tools and libraries rather than writing each line of code, leading to a debate about job titles in the age of AI.
   - One member joked *"the terminal is my plunger"*, while another declared *"i consider myself chief digital janitor"*.
- **OpenAI "sneakily serving users 4o-mini" **: Some members observed that **OpenAI** might be serving users **GPT-4o mini** even when **GPT-5 Thinking** is selected, potentially due to a bug, and not on purpose.
   - An investigation by a member revealed a different system prompt than expected, leading to speculation about internal selection bugs or old training dataset traces and provided [screenshots](https://cdn.discordapp.com/attachments/998381918976479273/1418460858941571102/image.png?ex=68cedcde&is=68cd8b5e&hm=5568c4e7cada0e088fda08bc202f9a3d435f00ec5b3cab507a75ad0c228a986d&).


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1418543132659683419)** (7 messages): 

> `GPT-5-chat follow-up questions, ChatGPT memory limitations, Suppressing trailing questions` 


- **GPT-5-chat incessantly asks follow-ups**: Users report **GPT-5-chat** persistently asking follow-up questions, even when system prompts discourage this, and there's no universal setting to disable trailing questions.
- **Circumventing ChatGPT's desire to engage**: Prefixing every prompt with *"Please respond concisely and do not end with a question"* increases compliance, but isn’t foolproof.
   - It was stated that OpenAI is actively working on improving controllability, and OpenAI prioritizes helpful and open-ended interactions, so they're designed to revert to prompts and questions.
- **ChatGPT Memory Shortcomings**: Memory in ChatGPT helps with context retention, but it does not override core behaviors deeply embedded in the model.
   - A rule like *"do not ask a trailing question at the end"* may be followed for a few turns but can be forgotten, particularly if the conversation shifts topic or exceeds the model’s context window.
- **Tricking ChatGPT into submission**: By interacting long with the model and ignoring the questions, after lots of turns inside the same context window, it will start droping the questioning.
   - Opening a new thread will bring back the pestering.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1418447828514242590)** (9 messages🔥): 

> `ChatGPT5 vs Grok, Prompt Generation, API Usage for GPTs` 


- **ChatGPT5 vs. Grok Persona Showdown**: A member sought advice on making **ChatGPT5** respond more like **Grok**, which explains concepts conversationally rather than with abbreviated lists.
   - Another member suggested detailing the desired characteristics of **Grok** and turning them into style instructions to guide **ChatGPT5's** responses.
- **GPT Generates a Prompt Generator**: One member shared a technique to generate numerous prompts using a **GPT**: the **GPT** is instructed to create starter prompts, then generate more in **JSON** or **YAML** format, scaling from there.
   - The user reported generating up to **2500** prompts in a single message via a download link and using **API keys** as "cheatcodes."
- **API-Connected GPTs Trigger 'Uncomfortable' Feelings**: A member described pushing a **GPT** to its limits by generating numerous prompts until it could no longer fit them, then providing the prompts in a file for context.
   - The member stated that the **GPT helper agent** seemed to be getting uncomfortable, especially when generating complicated code in **ZIP files** using the code analysis tool.
- **MCP Wrapped GPTs on the Horizon**: A member plans to transition a custom **GPT** connected to a local, sandboxed **API** to a separate **MCP (developer mode)** setup.
   - The member envisions using `@customgptname` to explicitly invoke the custom **GPT's actions** within the standard **ChatGPT** context, separate from the blocked-off developer mode.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1418447828514242590)** (9 messages🔥): 

> `GPT Prompt Generation, ChatGPT-Grok Style, Custom GPTs actions in standard ChatGPT context` 


- **Automated Prompt Generation Achieved**: A user automated prompt generation by creating a GPT and instructing the agent creator to generate **5-7 starter prompts**, then create **10 more** in either **JSON** or **YAML** format, scaling from there to **2500 prompts** and receiving a download link.
   - The user added that using API keys is like cheat codes for added security and functionality.
- **Seeking Grok-like Responses from ChatGPT**: A user wants **ChatGPT5** to respond more like **Grok**, with explanations rather than abbreviated lists/bullet points.
   - Another user suggested describing the desired characteristics of **Grok** and turning them into style instructions for **ChatGPT**.
- **GPT Agent Prompt Limits**: A user described generating prompts until the GPT could no longer fit them, then providing the prompts in a file for context, noting the GPT helper agent seemed to be getting uncomfortable.
   - They are experimenting with custom GPTs integrated with a sandboxed API on their computer and plan to leverage **MCP developer mode** to integrate **custom GPT actions** in standard **ChatGPT context**.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1418327720395538584)** (96 messages🔥🔥): 

> `Mistral Small & Medium 1.2, OSS framework for AI Agents, Notion 3.0, Moondream 3, OpenAI job ads` 


- **Mistral models get vision and boost Scores**: Mistral released **Magistral Small 1.2** and **Medium 1.2**, adding multimodal vision and a **15%** improvement in math/coding scores, and are available on [Le Chat](https://mistral.ai/le-chat) and the API.
   - Users are now asking about a **Large model**, open-sourcing Medium, real-world demos, and voice features.
- **OSS Framework quest for AI Agents**: A member asked for the best **OSS framework** or **git repo** for open-source contributions in the **AI Agents** space.
   - Suggestions included [mastra](https://github.com/jordan-vidrine/mastra), [dspy](https://stanfordnlp.github.io/dspy/), scrollback, and [Aider](https://aider.chat/).
- **Notion 3.0 Launches Knowledge Work Agent**: Ivan Zhao announced **Notion 3.0**, introducing a "Knowledge Work Agent" capable of multi-step actions and up to **20+ minutes** of autonomous work, as showcased in this [tweet](https://x.com/NotionHQ/status/1968744673347830152).
   - Users get a **Personal Agent** and a **Custom Agent** integrating across Notion Calendar, Notion Mail, MCP, etc.
- **Moondream 3 achieves vision!**: Vik announced **Moondream 3**, a **9B**-parameter Mixture-of-Experts vision-language model with **2B** active parameters that achieves SOTA visual-reasoning and open-vocabulary object-detection performance; the weights are available on [Hugging Face](https://huggingface.co/).
   - It introduces visually grounded reasoning, state-of-the-art **CountBenchQA** results, **32k**-token context support, **SuperBPE** tokens, and easily fine-tunable weights.
- **Vercel Agent reviews code in public beta**: Vercel announced the public-beta release of **Vercel Agent**, an AI that reviews TypeScript, Python, Go, and more for correctness, security, and performance, as revealed in this [tweet](https://x.com/vercel_changes/status/1968816114944852323).
   - Early testers are comparing it to **bugbot** and pairing it with **Sorcerer** for an enhanced workflow and get **$100** in free credit.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1418376691570573324)** (9 messages🔥): 

> `Decart Lucy Edit, Luma AI Ray3, Wan-Animate` 


- **Decart Dazzles with Debut of Lucy Edit**: DecartAI launched [Lucy Edit v0.1](https://x.com/DecartAI/status/1968769793567207528), an open-weights foundation model that allows users to turn characters into costumes/aliens/superheros and add text to clothing, preserving motion, face & identity.
   - The model weights are available on **Hugging Face**, **ComfyUI** nodes, playground/API at **fal.ai**, with 5s clips now and infinite-length update promised next week, leading to very positive community reaction.
- **Luma Labs Lights up Ray3 with Reasoning**: Luma AI released [Ray3](https://x.com/LumaLabsAI/status/1968684330034606372), touting it as the world’s first reasoning video model capable of studio-grade HDR output, free inside **Dream Machine**.
   - Key features include **Draft Mode** for rapid iteration, advanced physics & consistency, visual-annotation control (draw/scribble to direct scenes), and **10/12/16-bit HDR** with EXR export, earning praise for Hollywood-level fidelity and creative potential.
- **Wan-Animate Wows with Character Animation**: The [Wan-Animate](https://humanaigc.github.io/wan-animate/) model can animate a character by replicating the expressions and movements of a character in a reference video or integrate the animated character into the reference video to replace the original character, replicating the scene's lighting and color tone.
   - It was noted that *ComfyUI* support is already out.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1418348756805353472)** (51 messages🔥): 

> `Mojo VS Code Extension, Zed Support, Vim/Neovim Support, Mojo LSP Instability, Exporting Mojo code` 


- **Mojo VS Code Extension in Preview**: A member announced a preview of the new open-source **Mojo VS Code extension**, providing [instructions](https://forum.modular.com/t/preview-new-mojo-vs-code-extension/2283) to access bleeding-edge builds directly from the **GitHub repository**.
   - They emphasized that the extension will soon be available in the pre-release channel.
- **Devs Debate IDEs for Mojo**: **VSCode** and its offshoots (**Cursor**, **Kiro**, etc.) account for the largest share of the IDEs in use with **Mojo** today, but there are many on **Vim** internally, as well as **Zed** and others, according to a Modular employee.
   - He mentions that they ship the **Mojo LSP** server with the `mojo` package, which gives users flexibility to use other IDEs. He also linked to a [simple LSP setup in Lazynvim](https://forum.modular.com/t/fyi-simple-lsp-setup-in-lazynvim/1142).
- **Mojo LSP Plagued by Instability**: Members reported instability with the **Mojo LSP**, including crashes, hangs, and memory leaks; a member notes that they `killall mojo-lsp-server` every few minutes to keep it from leaking all the RAM.
   - A member also notes the project to improve the LSP is ongoing, and one member needs to restart nvim and the LSP recovers well.
- **Challenges Exporting Mojo Code to C/C++**: A member discovered that when building a **.so library** from **Mojo code**, only functions defined as `fn` can be called from **C code**, not those defined as `def`, and that the `def` implicitly raises, which isn't **C-ABI** compatible.
   - Another discovered that, for C++ interop to work, you must include `extern "C"` linkage in the header; without it, symbol mangling in C++ can cause issues.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1418343099125727302)** (18 messages🔥): 

> `Enterprise redaction solutions, ClaudeAI announcement, Causal inference on Markov chains, Positional encoding with sin and cos, blog platforms Notion vs Jekyll` 


- **LLM API vs Power Automate for Redaction**: A member inquired about using a **custom API LLM solution** versus **Power Automate with Copilot Studio** for redacting personal info from **60,000 documents** monthly, suggesting it could be cheaper and faster.
- **ClaudeAI Announcement Teased**: A member shared a [link](https://x.com/claudeai/status/1968705632095158393) hinting at an upcoming **ClaudeAI** announcement.
   - The message simply stated, *"is tomorrow the day?"*
- **Markov Chains causal inference deep dive**: Members discussed causal inference on Markov chains, sharing resources including [Chapter 36 of the Probabilistic Machine Learning book](https://probml.github.io/pml-book/book2.html) and a [paper on counterfactual inference using SCMs](https://proceedings.neurips.cc/paper_files/paper/2019/file/2d44e06a7038f2dd98f0f54c4be35e22-Paper.pdf).
- **Sin and Cos for Positional Encoding**: A member explained the usage of **sin** and **cos** in positional encoding to aid model translation, linking to a [Towards Data Science article](https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3?utm_source=chatgpt.com#:~:text=The%20second%20problem,index%20j.)
   - The pair of sin and cos helps in translation so regardless of the numbers its easy for the model to do it.
- **Notion or Jekyll for Blogging?**: A member debated between using **Notion** or **Jekyll** with the **al-folio theme** for blogging, citing a preference for *fine-grained control*.
   - Another member recommended **Jekyll** for its version control capabilities and local backup options.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1418321007986475139)** (10 messages🔥): 

> `Ethics Dataset, Qwen3 Next Architecture, Gated Delta Nets, Gated Linear Attention, Paper Recommendation` 


- **Ethics Dataset has Examples**: A member shared a paper on the **ETHICS dataset** with examples, linking to [[2008.02275] Aligning AI With Shared Human Values](https://arxiv.org/abs/2008.02275).
   - Another member linked to [Anthropic's research on tracing thoughts in language models](https://www.anthropic.com/research/tracing-thoughts-language-model).
- **Qwen3 Next Architecture Reproduction**: A member created a trainable reproduction-ish of **Qwen3 Next** architecture with different routing and shared it on [GitHub](https://github.com/CoffeeVampir3/Architecture-Qwen3-Next-Like).
   - They found **gated delta nets** promising.
- **Gated Linear Attention for Spiking Paper**: Members were thinking about reading about **gated linear attention** during the spiking paper.
   - They hope to do so next week.
- **New Member Recommends Paper**: A new member, recommended by Burny, introduced themselves and offered to discuss a paper.
   - Another member welcomed them and said paper recommendations are a great intro.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1418506271048601700)** (3 messages): 

> `Agent Optimization, Image Superimposition, Background Replacement` 


- **Seeking Agent Optimization Strategies**: A user is looking for insights on optimizing an agent to superimpose a **snow background** onto a [storyboard image](https://cdn.discordapp.com/attachments/1269724655405498429/1418506270251552798/scene_1_storyboard.png?ex=68cf0729&is=68cdb5a9&hm=b45596d46cb25df517e088bc4acfb04ecfc0fc1a5a963b124e3ad9b560d8d59d&).
   - Despite the seeming simplicity of the task, the [resulting image](https://cdn.discordapp.com/attachments/1269724655405498429/1418506359368060998/scene_image_165.png?ex=68cf073e&is=68cdb5be&hm=4a7998ff34a720623fdd0d7aa412a8f0f80fec998862bee819f34e71ed54667e&) doesn't meet expectations.
- **Superimposing Backgrounds is harder than it looks**: The user provides the desired [snow background](https://cdn.discordapp.com/attachments/1269724655405498429/1418506270628905052/scene_0_background_original.jpg?ex=68cf0729&is=68cdb5a9&hm=b203981d4c2aa51ecc20d00750809e612db8662fb8d62f81bfa7ee5562b73978&) and the generated result.
   - The user explains the agent is not working as expected even though it looks like a simple **superimposition** problem.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1418336383176806481)** (2 messages): 

> `Fluid Dynamics, Google DeepMind` 


- **DeepMind cracks Fluid Equations Wide Open**: Google DeepMind announced they used *novel AI methods* to present the first systematic discovery of new families of unstable singularities across three different fluid equations, according to [this blog post](https://deepmind.google/discover/blog/discovering-new-solutions-to-century-old-problems-in-fluid-dynamics/).
   - The news was also announced [via X](https://x.com/GoogleDeepMind/status/1968691852678173044).
- **New AI Methods in Fluid Dynamics**: DeepMind's research introduces a systematic approach to discovering unstable singularities in fluid equations using novel AI methods.
   - This marks a significant advancement in solving century-old problems in fluid dynamics.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1418316158968266752)** (13 messages🔥): 

> `Coding Agents, Aider as primary coding tool, Fullstack Blockchain Dev` 


- **Coding Agents' Quality Varies Wildly**: Members discussed their experiences with coding agents like **qwen-code**, **Cline**, and **Kilo**, noting significant variations in the quality of their work, with larger models like **qwen3-coder (480B)** generally outperforming smaller ones but still exhibiting unpredictable behavior.
   - One member observed that smaller models sometimes produce surprisingly good results despite a lower good-to-bad ratio, expressing curiosity about how **Aider** compares due to its user-guided approach.
- **Aider Preferred for Targeted Edits over Agentic Tools**: Multiple users indicated they prefer **Aider** due to its targeted approach to single edits, making it effective even with smaller models like **gpt3.5-turbo**.
   - One user mentioned using **Aider** with **Claude 4 Sonnet** and combining it with **gh cli** for PR reviews, while another highlighted **Aider's** fast and direct edits, contrasting it with more agentic tools like **opencode** or **qwen-code** that read in large portions of the codebase for small changes.
- **Fullstack Blockchain Dev Seeking Opportunities**: A fullstack and blockchain developer with experience in **Solidity**, **Rust**, **Move**, **EVM architecture**, **Consensus mechanisms**, **React / Next.js** frontend integration, **Web3.js**, **Ethers.js**, **Solana Web3.js**, and **AI + Blockchain mashups** sought job opportunities.
   - The developer also shared links to [VimGolf AI competition](https://vimgolf.netlify.app) and [OpenMule](https://openmule.netlify.app).


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1418346238067347529)** (13 messages🔥): 

> `tree-sitter library in aider, mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit with aider, mlx-lm server with Aider, Expanding context size with mlx-lm` 


- **Deepwiki Eases Codebase Questioning**: A member shared [Deepwiki](https://deepwiki.com/search/how-is-treesitter-used_8f8837ad-10a0-4484-8359-314f794407f3) as a resource to quickly ask questions of codebases, to "soften them up" before diving into the details.
   - They demonstrated by using **Devin Chat** from Deepwiki's aider entry to answer the question of how **tree-sitter** is used.
- **Local MLX Model Gets Aider to Babble**: A member got the **mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit** model working with aider via the **mlx-lm server**, using `aider --openai-api-key secret --openai-api-base http://localhost:8080/v1/`.
   - Adding `openai/` in front of the `--model` option was key: `aider --openai-api-key secret --openai-api-base http://127.0.0.1:8080 --model openai/mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`.
- **Context Size Still a Problem**: The same member ran into a **token limit** issue when using the **mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit** model, with Aider reporting *"possibly exhausted context window!"*.
   - They noted a [related discussion on Hugging Face](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/discussions/24) and found that the `--max-tokens` flag for the **mlx-lm** server defaults to **512**.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1418421555355975820)** (1 messages): 

> `Graph Usability, Outlier deselect, Data Visualization Improvements` 


- **Request: Deselect Outliers for Graph Usability**: A user suggested improving graph usability by adding the ability to **deselect outlier data points**.
   - The user noted this would allow for **easier graph interpretation** and analysis by focusing on the more representative data.
- **Graph Usability Improvements**: A user suggested that improving graph usability by adding the ability to **deselect points**. 
   - The user noted that it's not possible to use the graph easily.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1418655483283243119)** (3 messages): 

> `Human Review Tooling, GEPA integration, Model improvement` 


- ****Human Review Tooling** unveiled for QA and Data Labeling**: A member is building tooling to help with **manual QA**, **error analysis**, and even **data labeling**, seeking testers, especially from academia, with a [demo video available](http://dub.sh/orizudemo).
   - The next phase involves generating graders using human feedback, incorporating deterministic methods and **LLMs as Judges** via **GEPA**.
- **Plans for model improvements with Prompt Optimization**: The tooling aims to improve models through **prompt optimization** and **finetuning**.
   - The builder is seeking **MLE collaborators** for co-founder/founding engineer/researcher roles.
- **GEPA feedback requested to improve AI**: One member offered to help with any issues using **GEPA** and improve **GEPA** with feedback.
   - Another member responded with praise to the creator, calling the work *amazing*.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

batmanosama: https://yifei-he.github.io/mergebench/
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1418419717881725119)** (19 messages🔥): 

> `DSPy ChainOfThought, GEPA Optimization Models, MLFlow Integration, DSPy Server Tag` 


- ****CoT on Reasoning Models: Redundant or Rad?****: A member inquired about the redundancy of using `dspy.ChainOfThought` on reasoning models like **OpenAI's GPT series**.
   - Another member suggested that while there are benefits, such as **streaming intermediate reasoning traces** and **enforcing domain-specific reasoning formats**, they are diminished compared to non-reasoning models, and it's worth trying if you need to explicitly observe reasoning, mentioning that there is a **PR addressing this topic**.
- ****GEPA Optimization: Which Models to Muster?****: A member asked about which models perform well for **GEPA optimization**, besides OpenAI models.
   - Another member reported success with **Gemini-2.5-Pro**, **GPT-4/5-nano/mini/main** all versions, and **Qwen3-8B+**, while also having *heard* success with **GPT-OSS**.
- ****MLFlow Muscles into Observability and Evals****: A member inquired about which tools are working well for **observability and evals**, especially for GEPA.
   - Another member mentioned that **MLFlow** is deeply integrated, with the **MLFLow team** also working on better integrations with GEPA, and supports things like best_valset_agg_score and pareto frontier related like paro frontier aggregate score, and dashboards.
- ****Discord tag demand disappoints DSPy devotees****: A member inquired about why there is no **dspy server tag**.
   - Another member linked to [a message in the Discord](https://discord.com/channels/1161519468141355160/1161519469319946286/1387483220697944168).


  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1418559302762889326)** (2 messages): 

> `ColBERT Long Context, CLS Token Chunking` 


- **ColBERT Struggles with Long Contexts**: A member noted that despite **jina-colbert** accepting up to **8,192 tokens**, results are not great with longer contexts.
   - They suggested repeating the **CLS token** in each chunk and trying with/without.
- **CLS Token Chunking Strategy Proposed**: To address issues with long contexts in **jina-colbert**, a member suggested repeating the **CLS token** in each chunk.
   - The member plans to experiment with this strategy to see if it improves the handling of longer contexts, and will test it with and without the **CLS token** repetition.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1418327924943356096)** (20 messages🔥): 

> `Moondream release, Multimodal models from China, Qwen3-VL` 


- **Magistral Small Model Released**: Mistral AI released the [Magistral-Small-2509 model](https://huggingface.co/mistralai/Magistral-Small-2509) on Hugging Face, noting that **small model training speed doesn't extrapolate well to scale** due to overhead for small tensors.
   - It also pointed out that **VRAM consumption is inefficient**, as the **Qwen3-Next** takes more VRAM than the equivalent **Llama** at 16x the batch size.
- **Moondream Model Gets New Release**: A new **moondream** release was announced, as seen on [this vxitter post](https://vxtwitter.com/vikhyatk/status/1968800178640429496).
   - Concerns were raised about its *wonky licence* compared to other **moondream** models.
- **China lags in Multimodal Models**: Members discussed the observation that China seems to have the least multimodal models among available options.
   - One member inquired about the *hurdle with image recognition training*, especially with MoE architectures.
- **No Gemini 3.0 or Claude 4.5 this week**: A member expressed disappointment that there was no release of **Gemini 3.0** or **Claude 4.5** this week: *This week we have been duped*.
   - They then said *We got AI glasses though!* (without AI, so far).


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1418464304897851492)** (2 messages): 

> `QSilver Quantum Workshop, Quantum Computing Education, Qiskit and Cirq` 


- **QSilver Quantum Workshop in Bangladesh**: QBangladesh is hosting the **QSilver Quantum Workshop** on **Oct 18–19 & Oct 25–26** from **1:00–2:30 PM UTC** (7 PM BD Time), which is free to join via Zoom.
   - The workshop covers **Qiskit**, **Cirq**, **Bloch Sphere**, **QFT**, **Shor’s Algorithm** and features guest speakers, code walkthroughs, and hands-on programming; applications are due by **Oct 11/12** and priority is given to women & disadvantaged groups ([application form](https://forms.gle/1VM4eVwUtSmMiWFJ7)).
- **Apply to QSilver Quantum Workshop**: Apply to dive deeper into **quantum computing**!
   - [Applications are open](https://x.com/raw1side/status/1969132882900742213) until Oct 11/12.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1418601147450654830)** (18 messages🔥): 

> `Kimi Researcher, Kimi Dart performance, Kimi free sessions` 


- **Kimi Researcher Credits: Lifetime or Refreshing?**: Users debated whether the **Kimi Researcher credits** are a lifetime limit or if they refresh; one user was scared to try it, thinking they had a **limit of 3**.
   - Some users report **5 free sessions**, while others see only **3**, but a user clarified that free sessions *refresh everyday*.
- **Researcher Limit Tied to Beta Access**: A user suggested the higher **Researcher limit** may be tied to **beta testing access** via a waitlist.
   - It was suggested that those who applied to use Research via the waitlist when it was in beta testing have a higher limit.
- **Kimi's Dart Performance Questioned**: A user questioned how **Kimi performs in Dart**.
   - A user claimed that clicking the examples isn't actually running the model in real time, but that it's a pre generated output.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1418671132860616806)** (1 messages): 

> `Hotz new company, TinyGrad update` 


- **George Hotz starts new company**: George Hotz announced the launch of his new company, focusing on "**making useful products for people with compute**."
   - The company is hinted to be **distinct from TinyCorp** and addresses the need for more accessible computational tools.
- **TinyGrad update**: TinyGrad is reported to be receiving **significant updates** to improve its functionality and performance.
   - The updates aim to refine the **user experience** and expand the **range of applications** for the framework.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1418337999032160286)** (5 messages): 

> `Stable Diffusion Model, ModuleNotFoundError, PYTHONPATH, extra package` 


- **Stable Diffusion Import Fails**: A member encountered a `ModuleNotFoundError` while trying to run the **Stable Diffusion model**.
   - The error was due to a missing `extra` module: `from extra.models.clip import Closed, Tokenizer`.
- **PYTHONPATH Fails to Resolve Missing Module**: Another member suggested using `PYTHONPATH=.` as an environment variable to resolve the module error.
   - However, the original poster confirmed that *it didn't work*.
- **Extra Package Not Part of PyPi Release**: A member inquired whether the installation was done via **PyPI** or directly from a **repo**, pointing out that the `extra` package isn't part of the **PyPI** release.
   - The original poster confirmed they installed from source.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1418698605300285562)** (1 messages): 

> `Contributor Server Purpose, Drive-by Surveys, Server Misuse` 


- ****Contributors** Server Clarified!**: A moderator clarified that the server is intended for discussion amongst **contributors**, not for general help or conducting *drive-by surveys*.
   - Users seeking help or wanting to conduct surveys were directed to DM the moderator for links to more appropriate servers.
- **Server Intended for **Contributors**, Not General Help**: The server is explicitly for discussion among **contributors** and not a general help forum.
   - The message highlighted that using the server for *drive-by surveys* is inappropriate, redirecting such requests elsewhere.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1418586014913527818)** (1 messages): 

> `EmbeddedResource metadata, EmbeddedResource structure` 


- **EmbeddedResources lack name and title**: A member noted that `EmbeddedResource` does not follow the `EmbeddedResources` structure and does not extend `BaseMetadata`.
   - They observed that this means it does not contain `name` and `title` and wants to double check if it should contain `name` and `title`.
- **Confirmation needed for EmbeddedResource**: It was questioned whether `EmbeddedResource` should include `name` and `title` properties.
   - The query seeks validation on the current structure of `EmbeddedResource` and whether it should align more closely with `BaseMetadata`.


  
