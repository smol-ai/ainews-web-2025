---
id: MjAyNS0w
title: not much happened today
date: '2025-09-09T05:44:39.731046Z'
description: >-
  **Cognition** raised **$400M** at a **$10.2B** valuation to advance AI coding
  agents, with **swyx** joining to support the "Decade of Agents" thesis.
  **Vercel** launched an OSS "vibe coding platform" using a tuned **GPT-5**
  agent loop. **Claude Code** emphasizes minimalism in agent loops for
  reliability. **Kimi K2-0905** achieved 94% on coding evals and improved
  agentic capabilities with doubled context length. **Alibaba** released
  **Qwen3-ASR**, a multilingual transcription model with <8% WER. **Meta**
  introduced Set Block Decoding for 3-5× faster decoding without architectural
  changes. Innovations in KV cache compression and quantization include
  **AutoRound**, **QuTLASS v0.1.0**, and **AlgoPerf v0.6**. **Google's Veo 3**
  video generation API went GA with significant price cuts and vertical video
  support.
companies:
  - cognition
  - founders-fund
  - lux-capital
  - 8vc
  - neo
  - vercel
  - claude
  - groq
  - alibaba
  - huggingface
  - meta-ai-fair
  - google
  - theturingpost
  - algoperf
models:
  - gpt-5
  - kimi-k2-0905
  - glm-4.5
  - qwen3-asr
  - opus-4.1
topics:
  - coding-agents
  - agent-architecture
  - open-source
  - model-evaluation
  - multilingual-models
  - speech-recognition
  - model-optimization
  - kv-cache
  - quantization
  - algorithmic-benchmarking
  - video-generation
  - context-windows
people:
  - swyx
  - tim_dettmers
---


**a quiet day**

> AI News for 9/8/2025-9/9/2025. We checked 12 subreddits, 544 Twitters and 22 Discords (187 channels, and 4104 messages) for you. Estimated reading time saved (at 200wpm): 337 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Apple iPhone event offered [some small updates](https://news.ycombinator.com/item?id=45186015).

---

# AI Twitter Recap

**Coding Agents and Tooling Momentum**

- **Cognition raises $400M to scale Devin**: Cognition announced a $400M round at a $10.2B post-money valuation to “advance the frontier of AI coding agents,” led by Founders Fund with Lux, 8VC, Neo and others participating. The team highlighted customer expansion and the Windsurf team joining, and is hiring across product, infra, and post‑training ([announcement 1](https://twitter.com/cognition/status/1965086655821525280), [2](https://twitter.com/cognition/status/1965086662612177299), [team note](https://twitter.com/cognition/status/1965086661253185645), [plans clip](https://twitter.com/cognition/status/1965185627357683776)). Commentary: @swyx is joining Cognition, laying out why he’s “buying” the agent-lab thesis and how positioning across sync/async workflows matters for dominance in the “Decade of Agents” ([thread](https://twitter.com/swyx/status/1965183110016098617)).
- **Agent dev stacks getting simpler and more capable**:
    - Vercel shipped an OSS “vibe coding platform” built on the Vercel AI SDK, Gateway, Sandbox, and a tuned GPT‑5 agent loop (tool use: file IO, commands, package install, autofix) with a one‑shot demo coding a multiplayer Pong game in Go ([demo](https://twitter.com/rauchg/status/1964857952722133231)).
    - Claude Code’s loop is intentionally minimal: a single master loop + async buffer, direct tools, and TODO-based planning; simplicity beats swarm orchestration for debuggability and reliability ([analysis](https://twitter.com/imjaredz/status/1965083721713041564)).
    - Coding evals: Kimi K2‑0905 on Groq hit 94% and ranked 7th on Roo Code, becoming the first open-weight model to break 90+ while also being the fastest/cheapest in the top 10 ([leaderboard](https://twitter.com/roo_code/status/1965098976677658630)). Tim Dettmers reports the practical frontier for coding assistants feels increasingly open-weight: GLM‑4.5 is “$3/month” and ~Sonnet quality; Kimi K2.1 Turbo ~3× faster and ~7× cheaper vs Opus 4.1, with GPT‑5 excelling mainly on complex spec work ([take](https://twitter.com/Tim_Dettmers/status/1965021602267217972)).

**Model and Inference Advances**

- **Kimi K2 0905 and Qwen3-ASR**:
    - Kimi K2 0905 (1T params, architecture unchanged) boosts agentic capabilities: Terminal‑Bench Hard from 14→23% and Tau2‑Bench Telecom 61→73%; context doubled from 128k→256k. Intelligence +2 on Artificial Analysis’ AAII; now serving on Kimi’s site ([summary](https://twitter.com/ArtificialAnlys/status/1965010554499788841), [live note](https://twitter.com/crystalsssup/status/1965017719058960732)).
    - Alibaba’s Qwen3‑ASR released a single model for multilingual transcription (EN/CN + 9 languages), autodetect, robust to BGM/noise/rap, with <8% WER and custom contextual biasing. Demos on ModelScope/HF; API available ([launch](https://twitter.com/Alibaba_Qwen/status/1965068737297707261)).
- **Faster decoding and lighter KV**:
    - Meta’s Set Block Decoding (SBD) enables 3–5× decoding speedups on existing LMs without architectural changes, matching NTP performance and preserving exact KV cache—parallel generation via masked/discrete diffusion formulation ([overview](https://twitter.com/HuggingPapers/status/1965084731839513059), [details](https://twitter.com/itai_gat/status/1965112129499046230)).
    - KV cache and quant innovation: AutoRound is now in SGLang ([PR](https://twitter.com/HaihaoShen/status/1964926924880523701)), Turing Post surveyed KV compression (quantization, low‑rank, Slim Attention, XQuant) with tradeoffs ([thread](https://twitter.com/TheTuringPost/status/1964971207188791464)), and QuTLASS v0.1.0 brings 4‑bit NVFP4 microscaling and fast transforms to Blackwell GPUs ([release](https://twitter.com/DAlistarh/status/1965157635617087885)). AlgoPerf v0.6 adds a rolling leaderboard, JAX jit, and lower compute costs for algorithmic benchmarking ([update](https://twitter.com/algoperf/status/1965044626626342993)); ZeroGPU AOT compilation internals for PyTorch were documented by HF ([blog](https://twitter.com/charlesbben/status/1965046090945954104)).

**Multimodal Generation, Video, and “Vibe Coding”**

- **Veo 3 goes GA and cheaper**: Google’s Veo 3 and Veo 3 Fast are now GA in the Gemini API with ~50% price cuts ($0.40/s and $0.15/s), 1080p output, and 9:16 vertical video support—positioned for scaled production ([dev blog](https://twitter.com/googleaidevs/status/1965160822260318702), [pricing breakdown](https://twitter.com/_philschmid/status/1965161626761326983), [PM note](https://twitter.com/OfficialLoganK/status/1965193765146296467)).
- **Community workflows and tooling**:
    - “Nano Banana” (Gemini 2.5 Flash Image Preview) catalyzed a weekend of “vibe‑coded” projects—now open-sourced for remix in Google AI Studio; teams report 1‑click reuse and playful gotchas (e.g., always rendering clocks at 10:10) ([open-source pack](https://twitter.com/arrakis_ai/status/1965001417716072877), [quirk](https://twitter.com/fabianstelzer/status/1965001753059057925)).
    - Qwen’s “paper → website” flow turns a research paper into a deployable site in minutes ([demo](https://twitter.com/Alibaba_Qwen/status/1964870508421480524)). Lmarena added multi‑turn image editing evals so the community can compare iterative refinement across models (incl. “nano banana”) ([feature](https://twitter.com/lmarena_ai/status/1965150440401809436)). For doc RAG UX, ColQwen2 + Weaviate powers token‑wise similarity maps for visual PDF search and patch highlighting ([build](https://twitter.com/helloiamleonie/status/1964997028875743637)).

**Agents, Post-Training RL, and Evaluation Practice**

- **Towards iterated self‑improvement**: FAIR’s Exploratory Iteration (ExIt) trains LLMs for inference‑time self‑improvement via an automatic curriculum that bootstraps from the model’s own prior responses, prioritizing partial histories with high return variance in GRPO groups. ExIt outperforms GRPO on contest math, BFCLv3 multi‑turn tasks, and MLE‑bench (+22%) while training only single‑step improvements ([thread](https://twitter.com/MinqiJiang/status/1965055909605916892)).
- **Online vs offline RL and evals**:
    - Evidence continues to show a performance gap favoring online RL (PPO/GRPO) over offline methods like DPO at scale, though semi‑online iterations (on‑policy sampling + negative gradients) narrow the gap; data quality still dominates algorithm choice ([summary](https://twitter.com/cwolferesearch/status/1965088925510520853)).
    - Why many “agents” underdeliver: decision‑making has near‑zero error tolerance and sparse data vs generative tasks; most failures are coarse task scoping and unstructured environments rather than LLM shortcomings ([debate recap](https://twitter.com/ZhihuFrontier/status/1964928650081698167)).
    - RAG evals moving from “dead” unit tests to “living” loops: RAGGY (open‑source REPL) enables what‑if iteration for RAG, and there’s a strong push to integrate pre‑prod tests with production observability and human review rather than treating them as separate silos ([RAGGY](https://twitter.com/HamelHusain/status/1965052554997600449), [evals take](https://twitter.com/bnicholehopkins/status/1965130607790264452)). Also see practical “Agentic RAG” architectures leveraging tool use and multi‑step reasoning ([guide](https://twitter.com/omarsar0/status/1965115682322042954)).

**Robotics and Embodied AI**

- **Multi‑robot planning via RL**: Google DeepMind’s RoboBallet (with Intrinsic and UCL) choreographs up to 8 robot arms for collision‑free task and motion planning, outperforming traditional methods by ~25%, and generalizing to new workflows in seconds via RL‑learned coordination principles ([announcement](https://twitter.com/GoogleDeepMind/status/1965040645103407572), [more](https://twitter.com/GoogleDeepMind/status/1965040648400351337)).
- **Open hardware stacks and dexterous manipulation**: Pollen Robotics outfitted Reachy 2 with dual open‑source “Amazing Hand” grippers for fine manipulation; native integration coming ([demo](https://twitter.com/pollenrobotics/status/1964987735829266871)). X Square announced WALL‑OSS (open base model) and the Quanta X2 robot with auto‑mop and dexterous hand; Alibaba Cloud led a $140M A+ round (>$280M raised in <2 years) ([summary](https://twitter.com/ZhihuFrontier/status/1964968113990164810)). OpenPI’s pi‑05 is now in openpi with PyTorch support ([release](https://twitter.com/svlevine/status/1965161524722630734)).

**Benchmarks, Leaderboards, and Enterprise**

- **Text leaderboards move**: lmarena added two new entries into its Top 10 Text leaderboard: Qwen3‑max‑preview (#6, proprietary) and Kimi‑K2‑0905‑preview (#8, modified MIT), putting Kimi in contention for top open‑weight alongside Qwen and DeepSeek variants ([update](https://twitter.com/lmarena_ai/status/1965115050273976703), [model link](https://twitter.com/lmarena_ai/status/1965124408097517853)). Artificial Analysis' K2‑0905 measurements mirror improved agentic performance ([details](https://twitter.com/ArtificialAnlys/status/1965010554499788841)).
- **Gov and enterprise**:
    - Perplexity launched “Perplexity for Government”: secure by default, zero data usage, premium model access, and no enterprise contracts; also brought Perplexity Finance to iOS/Android ([launch](https://twitter.com/perplexity_ai/status/1965030156415980009), [follow‑up](https://twitter.com/AravSrinivas/status/1965032305053065590), [finance mobile](https://twitter.com/AravSrinivas/status/1965100159488196757)).
    - Anthropic endorsed California SB 53 (Sen. Scott Wiener), a transparency‑focused state framework for governing frontier AI in lieu of a federal standard ([statement](https://twitter.com/AnthropicAI/status/1965027311717388673), [context](https://twitter.com/jackclarkSF/status/1965048896784367847)).

Top tweets (by engagement)

- Cognition raises $400M at $10.2B to scale AI coding agents ([announcement](https://twitter.com/cognition/status/1965086655821525280))
- Vercel’s OSS vibe coding platform with a tuned GPT‑5 loop one‑shots a multiplayer Pong game in Go ([demo](https://twitter.com/rauchg/status/1964857952722133231))
- Qwen3‑ASR: one model for multilingual ASR with <8% WER, robust to noise/BGM, with context injection ([launch](https://twitter.com/Alibaba_Qwen/status/1965068737297707261))
- Google AI Mode expands to Hindi, Indonesian, Japanese, Korean, and Brazilian Portuguese ([Sundar Pichai](https://twitter.com/sundarpichai/status/1965115123330388467))
- Veo 3 GA with ~50% price cuts, 1080p, and vertical video in the Gemini API ([dev update](https://twitter.com/googleaidevs/status/1965160822260318702))

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. A3B HF Releases: Qwen3-Next-80B-Instruct & ERNIE-4.5-21B-Thinking

- [**Qwen 3-Next Series, Qwen/Qwen3-Next-80B-A3B-Instruct Spotted**](https://github.com/huggingface/transformers/pull/40771) ([Score: 472, Comments: 134](https://www.reddit.com/r/LocalLLaMA/comments/1nckgub/qwen_3next_series_qwenqwen3next80ba3binstruct/)): **Alibaba’s Qwen3-Next introduces architectural changes for long-context, cost-efficient LLMs, notably a Hybrid Attention stack (Gated DeltaNet + Gated Attention), high‑sparsity MoE with** `1:50` **activation ratio, and Multi‑Token Prediction (MTP) plus stabilizers (zero‑centered, weight‑decayed layernorm). The released Qwen3‑Next‑80B‑A3B (**`80B` **total,** `~3B` **active) reportedly outperforms Qwen3‑32B on downstream tasks at** `<1/10` **training cost and delivers** `>10×` **higher inference throughput for contexts** `>32K` **tokens; details in the project’s [blog post](https://qwenlm.github.io/blog/qwen3_next/). Upstream support landed in Hugging Face Transformers via [PR #40771](https://github.com/huggingface/transformers/pull/40771) (12 commits, 15 files,** `+2,964/−2` **LOC) referencing the [Qwen3 repo](https://github.com/QwenLM/Qwen3), indicating integrated model/tokenizer configs and tests for the Qwen3‑Next family.**
    - Qwen (Alibaba) outlines a new architecture for the Qwen3-Next series, notably in the released model **Qwen/Qwen3-Next-80B-A3B-Instruct**: Hybrid Attention combining **Gated DeltaNet + Gated Attention**, **Multi-Token Prediction (MTP)** for improved pretraining and faster inference, and stability tweaks like zero-centered, weight-decayed LayerNorm. They claim `80B` total parameters with only `3B` active via high-sparsity MoE, outperforming **Qwen3-32B** on downstream tasks at <`1/10` training cost and achieving >`10x` higher inference throughput on contexts >`32K` tokens ([blog](https://qwenlm.github.io/blog/qwen3_next/)).
    - Discussion benchmarks the MoE activation ratio `1:50` against other models: **GPT-OSS-12B** activates `4/128` (~`1:32`), **V3/R1** `9/257` (~`1:29`), **K2** `9/385` (~`1:43`), and **LongCat-Flash** averages `9/513` (~`1:57`), though its larger shared expert inflates the effective active parameter share. Qwen3-Next’s routing sparsity is thus among the most aggressive in this set, prompting interest in how small individual experts can be without degrading quality.
- [**baidu/ERNIE-4.5-21B-A3B-Thinking · Hugging Face**](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking) ([Score: 237, Comments: 59](https://www.reddit.com/r/LocalLLaMA/comments/1nc79yg/baiduernie4521ba3bthinking_hugging_face/)): **Baidu released [ERNIE-4.5-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking), a** `~21B`**parameter text MoE model with** `~3B` **activated parameters per token (A3B) focused on enhanced multi-step reasoning and** `128K` **context. It provides Transformer-style weights compatible with [transformers ≥4.54.0](https://github.com/huggingface/transformers), [vLLM](https://github.com/vllm-project/vllm), and [FastDeploy](https://github.com/PaddlePaddle/FastDeploy), supports tool/function calling, and is released under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). A community GGUF build is available at [gabriellarson/ERNIE-4.5-21B-A3B-Thinking-GGUF](https://huggingface.co/gabriellarson/ERNIE-4.5-21B-A3B-Thinking-GGUF).** Commentary flags potentially selective benchmarking (only comparing to stronger models) and requests Q4/Q5 GGUF quants that fit on a single 16GB GPU as a competitor to Qwen3-30B-A3B; a benchmark image was shared for scrutiny.
    - Several note the benchmark framing looks cherry-picked: the posted chart appears to compare mainly against stronger baselines that already beat `ERNIE-4.5-21B-A3B-Thinking`, which obscures where it actually leads or lags; see the shared image for context (https://preview.redd.it/0e10f0pbw1of1.png?width=3840&format=png&auto=webp&s=916b8f0777cb166e44833224bd30af0291d312d4). The sharp drop on CNsimpleqa versus more competitive results elsewhere raises “benchmaxxing” concerns—i.e., dataset-specific tuning inflating scores on popular leaderboards while underperforming on less-targeted Chinese QA. Calls for broader, apples-to-apples baselines (e.g., Llama 3.1 70B/8B, Qwen2.5/3 14B/32–30B) and full metric breakdowns are implied to validate generalization.
    - On-device feasibility: a 21B model at Q4 is ~`10.5 GB` weights-only and ~`13.1 GB` at Q5, so `ERNIE-4.5-21B-A3B-Thinking` could plausibly fit on a single 16 GB GPU with careful KV cache and batch/context management; meanwhile a 30B (e.g., `Qwen3-30B-a3b`) is ~`15.0 GB` (Q4) and ~`18.8 GB` (Q5) for weights-only, making Q5 infeasible and Q4 borderline once runtime overhead and KV cache are included. Because “A3B/Thinking” styles tend to emit longer reasoning traces, KV cache can dominate memory at longer contexts, so practical single-GPU use likely requires short context, small batch, and aggressive paged-KV or offloading.
    - Requests for `Ernie-4.5-VL-28B` and especially `Ernie-4.5-VL-424B` support highlight infra constraints: even at 4-bit, a 424B model is ~`212 GB` weights-only, necessitating multi-GPU tensor/pipeline parallelism (e.g., ≥3×80 GB for weights alone, more for KV/vision tower). Proper HF integration would also need the vision encoder + projector wiring (CLIP/ViT-like tower, image tokenization), and inference backends that support heterogeneous compute (CPU offload/ZeRO, paged attention) to make 28B tractable and 424B at least demo-able.

### 2. Open-Source SOTA Challengers (PyDevMini-1, ROMA Seal-0/FRAMES, Apertus)

- [**PyDevMini-1: A 4B model that matches/outperforms GPT-4 on Python & Web Dev Code, At 1/400th the Size!**](https://v.redd.it/nh9fq7qbn2of1) ([Score: 295, Comments: 91](https://www.reddit.com/r/LocalLLaMA/comments/1ncam9h/pydevmini1_a_4b_model_that_matchesoutperforms/)): **Release of PyDevMini-1, a** `~4B`**parameter finetune of Qwen’s base model (author cites “Qwen3-4B-Instruct-2507”) targeting Python and web-dev coding, claiming GPT‑4‑level behavior at ~**`1/400th` **the size, runnable on a single gaming GPU. The model emphasizes real-world demos over benchmarks (side‑by‑side video) and provides a free Colab for replication; training credits include Qwen ([repo](https://github.com/QwenLM/Qwen3)), Unsloth’s Duo for efficient finetuning, and Tesslate’s web‑dev data ([WEBGEN‑4B‑Preview](https://huggingface.co/Tesslate/WEBGEN-4B-Preview)). Key specs:** `4.0B` **params (**`3.6B` **non‑embedding),** `36` **layers, GQA (**`32` **Q heads /** `8` **KV heads), native context** `262,144`**; recommended decoding:** `temp=0.7`**,** `top_p=0.8`**,** `top_k=20`**,** `min_p=0`**. Links: model card ([HF](https://huggingface.co/bralynn/pydevmini1)), demo/try-it Colab ([Colab](https://colab.research.google.com/drive/1c8WCvsVovCjIyqPcwORX4c_wQ7NyIrTP?usp=sharing)), community Discord ([invite](https://discord.gg/RqwqMGhqaC)). Roadmap priorities: tool-calling mastery and long-context robustness.** Commenters ask for rigorous head‑to‑head coding benchmarks vs the base **Qwen3‑4B‑Instruct‑2507** to verify finetune gains and detect regressions; they also note lack of current tool‑calling support as a blocker for serious coding agents. Additional feedback flags potential training‑data overlap with showcased tasks (suggesting large unseen codebase bug‑fix tests) and requests proper attribution/linking to **Tesslate**’s dataset rather than re‑uploads (Apache‑2.0).
    - Real-world robustness concerns: while the small-model results look strong, commenters suspect many showcased tasks may appear in the training set and request evaluation on a large, real codebase (e.g., fixing a bug across `100k+` lines) to test long-context navigation and multi-file reasoning. They also note the post omits tool-calling; modern coding agents are expected to execute tools (run tests, edit files, call functions), and lacking this capability likely limits practical coding performance even if static benchmarks look good.
    - Comparison request against strong 4B baselines: specifically, head-to-head coding benchmarks versus **Qwen3-4B-Instruct-2507** to verify the finetune actually improves (or at least doesn’t regress) the base model. Suggested evidence includes standard pass@1/pass@k metrics on common code sets (e.g., HumanEval/MBPP/LiveCodeBench) under identical prompting, context limits, and tokenizer settings to substantiate claims of matching/outperforming larger models.
    - Actionable evaluation suggestion: run the Python portion of the Aider “polyglot” test suite and report the second-pass score, which better reflects iterative edit-test loops than single-shot QA. Link: https://github.com/Aider-AI/aider. Providing both full-suite results and the Python-only breakdown would yield a more realistic view of end-to-end coding capability for a 4B model.
- [**Open-source Deep Research repo called ROMA beats every existing closed-source platform (ChatGPT, Perplexity, Kimi Researcher, Gemini, etc.) on Seal-0 and FRAMES**](https://i.redd.it/sxii7uog37of1.jpeg) ([Score: 162, Comments: 9](https://www.reddit.com/r/LocalLLaMA/comments/1nctfdv/opensource_deep_research_repo_called_roma_beats/)): **The post announces an open-source “deep research” framework, ROMA ([repo](https://github.com/sentient-agi/ROMA)), claiming state-of-the-art results on the SEAL-0 and FRAMES benchmarks versus closed platforms (ChatGPT, Perplexity, Kimi Researcher, Gemini). ROMA is described as a plug-and-play system combining recursive planning and a multi-agent architecture with a web search tool; the attached image appears to be a benchmark leaderboard comparing ROMA against those services. Links provided include the GitHub repo and a promotional X post.** Top comments question the self-claimed superiority, noting potential benchmark bias and pointing out Gemini’s advantage via Google search; they also request head-to-head results against proprietary “Deep Research” modes (OpenAI Deep Research, Grok DeepSearch, Gemini Deep Research) and ask for real-world user experiences.
    - Benchmark scope gap: commenters note ROMA compares against general chat products but omits specialized closed “deep research” agents. Without head‑to‑head results versus **OpenAI Deep Research**, **Grok DeepSearch**, and **Gemini Deep Research** on **SEAL‑0** and **FRAMES**, the SOTA claim is hard to verify. Requests include publishing per‑task accuracy, citation fidelity, and error breakdowns, with fixed seeds, execution logs, and identical browsing quotas/user‑agents to ensure reproducibility.
    - Retrieval stack confounder: a key objection is that **Gemini** may leverage Google’s first‑party index, which could dominate outcomes independent of the agentic planner—*“There’s no way it beats Gemini, especially since it uses Google’s internal search index.”* For fairness, commenters suggest normalizing backends or stratifying results by retrieval setting (`no-search`, `public SERP`, `first‑party index`) and time‑freezing queries so differences reflect planning/tool‑use rather than search privilege.
    - Plug‑and‑play multimodality and real‑time tools: interest centers on whether ROMA cleanly swaps in VLM/ASR components (e.g., GPT‑4o, Gemini 1.5) for page parsing, OCR, and table/chart extraction, which matter on **FRAMES**’ screenshot/PDF‑heavy hops. Technical clarity sought on how tools are registered (browser controller, scraper, retriever, verifier), streaming/latency constraints, rate‑limit handling, and anti‑bot strategies, to judge portability and whether benchmarked gains persist in live environments.
- [**Switzerland just dropped Apertus, a fully open-source LLM trained only on public data (8B & 70B, 1k+ languages). Total transparency: weights, data, methods all open. Finally, a European push for AI independence. This is the kind of openness we need more of!**](https://i.redd.it/pmfv6zvyp3of1.png) ([Score: 258, Comments: 31](https://www.reddit.com/r/LocalLLM/comments/1ncfg23/switzerland_just_dropped_apertus_a_fully/)): **Switzerland released “Apertus,” an open LLM suite in 8B and 70B sizes, trained exclusively on public data spanning 1,000+ languages, with full transparency of weights, datasets, and training methods for auditability and reproducibility. The project positions itself as a European push for AI sovereignty/independence and emphasizes data-provenance clarity over scraping private sources.** Early community feedback suggests underwhelming performance relative to SOTA, per a LocalLLaMA thread ([discussion link](https://www.reddit.com/r/LocalLLaMA/comments/1n6eimy/new_open_llm_from_switzerland_apertus_40_training/)), and some debate centers on whether restricting to “public data only” hampers capability.
    - Early reports in the linked thread suggest Apertus’ initial quality is underwhelming relative to expectations; commenters cite weak subjective performance and request rigorous, public benchmarks. See discussion: https://www.reddit.com/r/LocalLLaMA/comments/1n6eimy/new_open_llm_from_switzerland_apertus_40_training/. To properly position the `8B` and `70B` variants, people ask for head‑to‑head numbers on standard suites (e.g., MMLU, HellaSwag, GSM8K, MT‑Bench) versus Llama and Mistral baselines.
    - Questions center on the exact “public data” used: which corpora, licenses, deduplication, filtering, and multilingual sampling strategy for the claimed `1k+` languages. Technical transparency here (dataset list, curation pipeline, tokenizer choice, per‑language token shares, and contamination checks) is crucial for reproducibility and to understand why performance may lag or excel in specific domains.
    - Comparative interest with **Mistral** is high; commenters want apples‑to‑apples evaluations (same context window, prompt format, decoding params) between Apertus `8B/70B` and Mistral `7B/8x7B` (and Llama `8B/70B`). Clear eval cards and inference settings would reduce variance and make any European “AI independence” claims measurable.
- [**🤔**](https://i.redd.it/1x8wy1p0k5of1.png) ([Score: 373, Comments: 69](https://www.reddit.com/r/LocalLLaMA/comments/1ncl0v1/_/)): **The image/post teases Alibaba’s Qwen stack: a new ASR service, Qwen3-ASR-Flash, built atop Qwen3-Omni and trained on “tens of millions” of hours of multimodal/ASR data ([source](https://x.com/cherry_cc12/status/1965227154813440163)). It also name-drops “Qwen Next, 1:50 sparsity, 80A3B,” implying a sparse MoE-style configuration (likely ~1 active expert out of 50 per token) and some model/cluster shorthand, though exact meaning of “80A3B” isn’t clarified in the post.** Comments are mostly non-technical; no substantive benchmarks or ablations are discussed.
    - Qwen team teaser: [Qwen3-ASR-Flash](https://x.com/cherry_cc12/status/1965227154813440163) is a speech recognition service built on **Qwen3-Omni**, reportedly trained/fine-tuned with multi-modal data including ASR datasets on the order of `tens of millions` of hours. Emphasis is on leveraging a strong generalist backbone for ASR via massive-scale supervised audio-text data, suggesting significant robustness across domains and accents compared to typical ASR-only pretraining regimes.
    - Mentions of upcoming MoE configs: “Qwen Next, `1:50` sparsity, `80A3B`” implies a very high expert count with only `1` of `50` experts active per token (extreme sparsity), and a notation hinting at a small active-parameter budget. Such routing would enable large total capacity while keeping per-token FLOPs close to smaller dense models, improving inference throughput and memory locality.
    - Model naming hints: “MOE multimodal qwen `40B-4A`, improved over `2507` by `20%`” and “Qwen4-`235B-A1B`” suggest a scheme of TotalParams-ActiveParams (e.g., `40B` total with `4B` active; `235B` total with `~1B` active). The claimed `~20%` improvement versus a prior “2507” baseline (unspecified metric) indicates measurable gains from MoE scaling while constraining active compute.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Anthropic Claude Degradation Incident and Churn Discussions

- [**Update on recent performance concerns**](https://www.reddit.com/r/ClaudeAI/comments/1nc4mem/update_on_recent_performance_concerns/) ([Score: 609, Comments: 283](https://www.reddit.com/r/ClaudeAI/comments/1nc4mem/update_on_recent_performance_concerns/)): **Anthropic reports two model-quality bugs affecting some Claude users, both now resolved per their status page: one caused degraded output for a small % of Claude Sonnet 4 requests from** `Aug 5–Sep 4` **(with higher impact** `Aug 29–Sep 4`**) and another affected some Claude Haiku 3.5 and Claude Sonnet 4 requests from** `Aug 26–Sep 5` **([incident](https://status.anthropic.com/incidents/72f99lh1cj2c)). They state they do not intentionally degrade quality, are investigating reports for Claude Opus 4.1, and are deploying more real-time inference monitoring plus conversation-reproduction tools; users can report issues via** `/bug` **in Claude Code or the 👎 on [Claude.ai](http://claude.ai/).** Commenters dispute the "small percentage" framing and ask for transparency and proof, citing community benchmarks and raising concerns about potential quantization/quality throttling and customer compensation. Others anecdotally report improvements and suggest telemetry-like signals (e.g., profanity rate) to detect regressions.
    - Multiple users challenge Anthropic’s explanation of “minor bugs,” citing community-run benchmarks over recent weeks that suggest systematic degradation. They specifically question whether models were quietly quantized or otherwise altered post-`Aug 28` usage limits, and ask for proof via transparent change logs, reproducible evals, and clear model/version fingerprints—plus discussion of customer compensation for degraded service.
    - Several comments point to an observability gap: a severe quality drop allegedly persisted for ~3 weeks despite widespread reports, implying insufficient internal quality telemetry beyond latency/uptime. Users hypothesize cohort-specific impact (A/B buckets, regions, or traffic classes) explaining why some saw **Claude Code** unaffected while others reported major regressions, and request detailed RCA rather than a generic “bug” label.
    - A CTO reports shifting a team (`~26` FTE + `12` contractors) off **Claude Code** toward **OpenAI Codex**, highlighting decision levers: one-shot capability on complex apps, speed (latency and tokens/sec), effective vs published context window (claim that **Claude Code** quality drops after `~50%` of context), raw coding IQ, and coding intuition. Cost is secondary to quality; they cite industry anecdotes (e.g., Simon Willison) showing strong results with Codex and are provisioning company OpenAI accounts accordingly.
- [**Month-long Issue with Claude model quality confirmed by Anthropic**](https://status.anthropic.com/incidents/72f99lh1cj2c) ([Score: 234, Comments: 62](https://www.reddit.com/r/ClaudeAI/comments/1nc4kma/monthlong_issue_with_claude_model_quality/)): **Anthropic [confirmed](https://status.anthropic.com/incidents/72f99lh1cj2c) two independent bugs that degraded Claude’s output quality and says fixes are deployed. Issue 1 impacted a *“small percentage”* of Claude Sonnet 4 requests from** `Aug 5–Sep 4` **(severity increased** `Aug 29–Sep 4`**); Issue 2 affected some Claude Haiku 3.5 and Claude Sonnet 4 requests from** `Aug 26–Sep 5`**. They are monitoring reports for Claude Opus 4.1; affected surfaces included** `claude.ai`**,** `console.anthropic.com`**,** `api.anthropic.com`**, and** `Claude Code`**. Anthropic states degradations were not intentional; however, no technical RCA, quantitative impact share, or offline benchmark deltas were published.** Commenters question lack of remediation (refunds/credits) and criticize slow/opaque incident response; several report that performance remains degraded post-fix, urging faster action and clearer metrics.
    - Multiple users report that **Claude’s** output quality remains degraded despite Anthropic’s acknowledgement and supposed mitigation, indicating the incident is not fully resolved for all. They characterize it as a month‑long regression in model behavior/quality rather than a transient outage, suggesting incomplete rollback or lingering issues in the serving/model pipeline.
    - There’s a strong call for a proper technical post‑mortem: a precise timeline of when the regression started, how it was detected, the root cause, the exact models/tiers affected, and what was changed to fix it. Commenters want accountability similar to a security incident report (clear scope, remediation steps, and safeguards to prevent recurrence).
    - Operational/billing implications are highlighted: paid subscribers on the **Max** tier canceled due to quality degradation and were denied refunds, prompting requests for prorated credits. Users argue that if model quality was impaired for ~1 month, providers should treat it like an SLA breach and compensate accordingly.
- [**Anthropic noticed an increased churn rate**](https://i.redd.it/v9wm9j5nh1of1.jpeg) ([Score: 481, Comments: 139](https://www.reddit.com/r/ClaudeAI/comments/1nc5kwl/anthropic_noticed_an_increased_churn_rate/)): **Screenshot appears to show an Anthropic staff acknowledgment that they’ve observed an increased user churn rate and are *investigating* reports of model quality regressions, framing the impact as a “small percentage,” reportedly more visible on lower‑tier offerings. No remediation, rollback, or concrete RCA is provided; the post suggests active monitoring rather than confirmed fixes. Image: https://i.redd.it/v9wm9j5nh1of1.jpeg** Top comments push back that this downplays widespread degradation—especially for paying Opus 4.1 users—calling it gaslighting and demanding an apology/ETA, while another user cites apparent quota/accounting anomalies (e.g., 5‑hour lockouts after minimal usage).
    - Multiple users report sustained quality regression in **Claude Opus 4.1** (premium tier, `$200/month`), contradicting Anthropic’s framing of issues affecting only “lower-tier models” and a “small percentage” of prompts. Reports describe weeks of “lobotomized” behavior with no remediation and only “still investigating” responses, implying a broad model or deployment-level change rather than isolated prompts.
    - A technical concern is that the statement *“we never intentionally degrade model quality”* does not rule out deployment of heavier quantization or other cost-reduction techniques. Commenters argue vendors can claim “no degradation” by subjective metrics while quantization (e.g., lower-bit weights/activations) can measurably reduce fidelity on complex reasoning tasks, even if average benchmarks remain stable.
    - Resource accounting anomalies: one basic-tier user claims just 2 queries consumed `~5 hours` of quota in a day, suggesting a metering bug or misconfiguration (e.g., over-counting context, tool calls, or session time). Others note perceived token reductions and faster exhaustion of quotas, consistent with changes in rate limiting or billing logic rather than user behavior.
- [**When a non-coder like me subscribes to Claude Pro 😂😂**](https://i.redd.it/iqantlrq22of1.jpeg) ([Score: 502, Comments: 32](https://www.reddit.com/r/ClaudeAI/comments/1nc814l/when_a_noncoder_like_me_subscribes_to_claude_pro/)): **Non-technical meme about subscribing to Claude Pro as a non-coder; the joke is that LLMs make it feel possible to get code written without prior programming skills and push users to crank usage to “overdrive.” No benchmarks, model specs, or implementation details—this is cultural commentary on LLM-assisted coding accessibility.** Comments note that LLMs let non-coders implement ideas they couldn’t before, while also inducing a feeling of needing to use the tool to its fullest; tone is humorous and self-referential.
- [**Sensational**](https://i.redd.it/1x9lhgsnw0of1.jpeg) ([Score: 8137, Comments: 193](https://www.reddit.com/r/OpenAI/comments/1nc2yb8/sensational/)): **Meme image satirizing the claim that “we’re just $20B away from AGI,” implicitly critiquing capital- and scaling-centric roadmaps to AGI (often associated with recent funding narratives around large LLMs and compute). No technical benchmarks or implementation details—context is sociotechnical skepticism about AGI timelines and the idea that more money/compute alone will suffice.** Top comments compare the claim to the perpetual “20 years to fusion” trope, note the ubiquity of certain AI figures’ media presence, and argue that current LLM architectures/methods are far from true AGI with no clear path demonstrated.
    - Skepticism about the claim that “$20B to AGI” mirrors fusion’s perpetual “20 years away,” emphasizing that capital alone won’t overcome unknown algorithmic breakthroughs; without concrete roadmaps tied to measurable milestones (e.g., scaling-law extrapolations, capability evals), such forecasts are non-falsifiable and weakly grounded in engineering realities.
    - Methodological critique: “No evidence that they have methods that will bring AGI… LLMs… are incomprehensibly far” argues that current GPT-style transformer LLMs trained on next-token prediction likely lack essential mechanisms for general intelligence (grounded reasoning, long-horizon planning, causal/world models), suggesting diminishing returns from pure scale without architectural/algorithmic advances.
    - Cost realism pushback: “They forgot 3 zeros” implies the `~$20B` estimate is orders of magnitude too low once full-stack costs are considered (compute capex, energy/opex, data acquisition/curation, inference fleets, reliability/safety), challenging simplistic budget-to-capability equivalence.
- [**Sensational**](https://i.redd.it/tbf6vbagw0of1.jpeg) ([Score: 4620, Comments: 62](https://www.reddit.com/r/ChatGPT/comments/1nc2xdj/sensational/)): **Non-technical meme/graphic that sensationalizes AGI’s projected economic value; commenters note the purported figure is wrong and cite ~**`$115B` **through 2029 instead, arguing revenue is a poor proxy for AGI (which should mean general human-level capability without “dementia”/hallucinations).** Debate centers on corporate incentives—claims that “corpos” want compliant, non-autonomous "zombie AI" rather than true AGI—and skepticism toward doomer/financial hype framing.
    - A capex-scale debate challenges trillion-dollar narratives, with one claim putting the "real number" near `~$115B through 2029`. If accurate, this implies data-center/GPU build-out will be significant but bounded by supply chains and power delivery, tempering near-term compute-scaling assumptions for AGI timelines. The framing emphasizes infrastructure economics as a first-order constraint, not just algorithmic progress.
    - Energy and policy bottlenecks are underscored by sarcastic calls for “`$200M` more,” “energy subsidies,” and “no regulation,” reflecting that large-scale training/inference is increasingly power- and capital-constrained. This suggests AGI roadmaps hinge on grid capacity, siting, and regulatory approvals as much as on model architecture, with firms seeking cheaper electricity and relaxed oversight to sustain scaling.
    - A definition debate rejects revenue-based metrics for AGI, preferring capability-based criteria: an AI that can "do everything humans can" and remain reliable over time (avoid degradation/"dementia"). For technical evaluation, this points toward broad task coverage and long-horizon robustness metrics rather than financial output, emphasizing generalization and stability across diverse domains.

### 2. Recent Model and Feature Releases (Seedream 4, HunyuanImage-2.1, Claude File Creation, ChatGPT Voice Mode)

- [**Seedream 4 is mind-blowingly good**](https://www.reddit.com/gallery/1ncn3qy) ([Score: 1249, Comments: 222](https://www.reddit.com/r/singularity/comments/1ncn3qy/seedream_4_is_mindblowingly_good/)): **Post claims “Seedream 4” produces near‑photorealistic image generations that look like real photographs. No technical details (architecture, training data, inference settings), benchmarks (FID/KID, human Turing-style evals), or release info are provided; no discussion of watermarking or detection tooling is mentioned.** Top comments emphasize that outputs are indistinguishable from photos and raise concerns about authenticity verification, hinting at a near-term need for robust provenance/watermarking or detection methods as models reach photographic realism.
    - Commenters highlight the photorealism of Seedream 4 outputs, specifically noting the absence of common synthetic tells such as overly shiny/plastic skin and unnatural specular highlights. Several say they cannot distinguish the images from real photographs, implying improved texture fidelity and lighting realism over prior gens.
    - A short exchange questions image authenticity ("How do I know if this photo is real?" → "You can't"), underscoring that eyeballing is no longer a reliable discriminator. This suggests current informal detection heuristics are failing on this content and points to the need for provenance or detection tooling when evaluating such images.
    - One user asks whether this is a new model, but no concrete technical details (versioning, training data, sampling methods, or parameters) are provided in-thread. The lack of metadata limits reproducibility and makes it hard to attribute which component(s) drive the realism.
- [**🚨New OSS nano-Banana competitor droped**](https://huggingface.co/tencent/HunyuanImage-2.1) ([Score: 234, Comments: 112](https://www.reddit.com/r/StableDiffusion/comments/1nccgt4/new_oss_nanobanana_competitor_droped/)): **Tencent’s HunyuanImage‑2.1 ([site](https://hunyuan.tencent.com/)) is an OSS text‑to‑image system built on a multi‑modal DiT backbone that combines single/dual‑stream pipelines and a refiner, with dual text encoders (a multimodal LLM + ByT5 for glyph‑aware text). It targets efficient 2K (2048×2048) generation via a** `32×` **high‑compression VAE aligned to DINOv2 features and trained with REPA loss, applies RLHF with Reward Distribution Alignment, adds a PromptEnhancer rewriting step with AlignEvaluator rewards, and uses meanflow‑based distillation for few‑step sampling; repo ships PyTorch code, weights, and demos. Notables: multilingual CN/EN prompts, flexible ARs, two checkpoints (full and distilled) ~**`34 GB` **each, and listed inference requirement of** `≥59 GB` **GPU RAM for 2K generation (bs=1).** Commenters note it’s not an editing model (unlike nano‑banana), though an edit model is teased as “coming next” [link](https://xcancel.com/bdsqlsz/status/1965328294058066273#m); discussion also flags the high VRAM floor (`~59 GB`) for 2K outputs as a practical constraint.
    - Commenters note the new OSS release is a base image generation model (not an editing model), so comparing it to “nano/banana” (editing-focused) is misleading. An editing-focused variant is hinted to follow this release, per the teaser shared here: https://xcancel.com/bdsqlsz/status/1965328294058066273#m.
    - A spec screenshot indicates a minimum of `59 GB` GPU memory for 2048×2048 image generation at batch size `1` (https://preview.redd.it/ooftutxzh3of1.png?width=1240&format=png&auto=webp&s=3eba83d1df448b18a2b6e10513ce3f0694210ee2). This effectively targets 80GB-class GPUs for native 2K inference and is notably higher than SDXL-class setups that can hit 2K on ~12–24 GB with xFormers/tiling, implying a heavier U-Net/attention footprint and large high-res KV caches.
    - For editing-capable OSS alternatives today, commenters list Qwen ImageEdit and Flux Kontext, while ByteDance “USO” is unclear. Until the teased edit model arrives, this release competes with base generators rather than edit-first tools like nano/banana.
- [**Claude can now create and edit files**](https://www.reddit.com/r/ClaudeAI/comments/1ncku1r/claude_can_now_create_and_edit_files/) ([Score: 232, Comments: 37](https://www.reddit.com/r/ClaudeAI/comments/1ncku1r/claude_can_now_create_and_edit_files/)): **Anthropic announced that Claude can now natively create and edit common office files—**`Excel (.xlsx)`**,** `Word (.docx)`**,** `PowerPoint (.pptx)`**,** `PDF`**, etc.—delivering ready-to-use outputs without copy/paste, and is available to Claude Max and Team/Enterprise users; details and examples are in the launch post and demo ([news](https://www.anthropic.com/news/create-files), [video](https://reddit.com/link/1ncku1r/video/eneho8eah5of1/player)). The feature focuses on read/write workflows across multiple tools consolidated into the chat, returning artifacts in their native formats for downstream use.** Top commenters question whether this is true in-place editing versus full document regeneration (as seen with “artifacts”), and whether edits will be detectable via layout/metadata changes—important for enterprise compliance. Others flag practical limits like conversation token caps (e.g., “Claude hit the maximum length...”) and suggest programmatic edits (e.g., Python for Excel) may remain preferable when zero-trace modifications are required.
    - A core concern is whether “create and edit files” performs true in-place edits that preserve existing layout/metadata, versus the common LLM pattern of fully regenerating documents. The commenter needs deterministic, audit-friendly edits with zero stylistic drift or watermark-like traces, asking if they must still use Claude Code + Python to inject values into Excel tables to guarantee schema/format fidelity (human-in-the-loop, but no observable LLM footprint). They emphasize that many business workflows require edits that are indistinguishable from manual changes, not regenerated content.
    - There’s skepticism about whether this feature actually writes changes to the underlying files or just renders/“previews” updates as with Claude Artifacts. The technical question is if the system performs real file I/O (e.g., incremental diff/patch, transactional updates) that persist to disk for formats like .docx/.xlsx, rather than UI-only artifacts that don’t update the source documents.
    - Context-window limits are raised as a practical blocker for long-lived editing sessions: “Claude hit the maximum length for this conversation…”. For complex document workflows, hitting the conversation cap implies state loss unless the system persists edit state outside the chat context (e.g., file-aware state, chunked operations, or resumable sessions). This impacts reliability for multi-step document editing without frequent resets.
- [**Standard voice mode will remain available in ChatGPT**](https://i.redd.it/59y71wftj5of1.jpeg) ([Score: 290, Comments: 115](https://www.reddit.com/r/ChatGPT/comments/1nckzq6/standard_voice_mode_will_remain_available_in/)): **Screenshot/announcement stating OpenAI will keep Standard Voice Mode (SVM) available in ChatGPT “for now” during the transition to Advanced Voice Mode (AVM), with phrasing like “we want to get this transition right.” Practically, users retain access to the existing voice stack while AVM matures; no firm deprecation date or feature-parity commitments are given, mirroring earlier uncertainty around GPT‑4o availability. Technical context from comments: SVM is considered more well‑rounded than current AVM, implying AVM still needs reliability/UX improvements before sunset of SVM.** Commenters interpret this as temporary: SVM will stay only until AVM improves, and criticize the strategically vague, non-committal language (similar to the GPT‑4o messaging) for making planning difficult.
    - Several commenters read the announcement’s "for now" language as a signal that **Standard Voice Mode (SVM)** will be kept only until **AVM** reaches feature/performance parity, drawing parallels to the unclear, staggered handling of **GPT‑4o** availability. The lack of concrete timelines is called out as a product/roadmap risk for developers who need to plan migrations or fallback paths. The net: expect SVM to be a transitional compatibility layer rather than a long‑term commitment unless AVM quality materially improves.
    - User feedback frames SVM as more robust and "well‑rounded" than AVM, with reports that the new voice "doesn’t function properly" and requests to fix regressions before deprecating SVM. While no hard benchmarks are cited, the sentiment implies reliability gaps (e.g., stability/UX parity) in AVM’s voice stack that would make forced migration premature for production use.
    - A thread highlights operational and cost considerations: one commenter argues AVM may be a cost‑cutting measure presented as a performance upgrade, noting a late announcement ("7 hours into Sep 9") and leadership communication that eroded trust. The claim that OAI has had AVM "for almost an entire year" suggests maturity concerns; combined with the GPT‑4o precedent, users infer deprecations may be driven by infra/cost constraints rather than clear performance wins.
- [**My first AI movie!**](https://v.redd.it/gk77a56lv3of1) ([Score: 826, Comments: 142](https://www.reddit.com/r/aivideo/comments/1nce0wx/my_first_ai_movie/)): **An AI‑generated sci‑fi short (“My first AI movie!”) was shared on Reddit and hosted on v.redd.it; the external link currently returns 403 Forbidden without authentication ([video](https://v.redd.it/gk77a56lv3of1), [login](https://www.reddit.com/login/)). Top technical feedback notes “smooth and consistent” animations, solid build‑up and comedic timing, and directly requests the creator’s workflow—implying interest in the generation/editing pipeline and methods used to maintain temporal consistency; no toolchain or model details were disclosed in the post.** Commenters praise the piece as a refreshing, non‑sexualized AI video (“Utterly Refreshing”) and express enthusiasm for learning the workflow behind it.
- 

### 3. OpenAI GPT-5 vs 4o Conversation Quality and Community Backlash

- [**GPT-4o used to talk with me. Now GPT-5 just talks at me.**](https://www.reddit.com/r/ChatGPT/comments/1nc1ukv/gpt4o_used_to_talk_with_me_now_gpt5_just_talks_at/) ([Score: 789, Comments: 579](https://www.reddit.com/r/ChatGPT/comments/1nc1ukv/gpt4o_used_to_talk_with_me_now_gpt5_just_talks_at/)): **OP reports a perceived regression from OpenAI’s [GPT-4o](https://openai.com/index/gpt-4o/) to “GPT‑5”: 5 is faster but often loses multi‑turn context, misses nuanced/emotional subtext, and occasionally contradicts itself, whereas 4o felt adaptive and dialog‑oriented ("relational intelligence") rather than strictly task‑driven. They argue 5 seems optimized for deterministic task execution (e.g., coding) over conversational alignment, and advocate keeping both models available due to distinct interaction profiles.** Top comments echo that 5 behaves like a directive‑driven search engine while the 4‑series felt more natural; some users say they still subscribe to access 4o. Others argue business incentives favor technical/informational workloads (API/enterprise spend) over companion‑style chat, with possible legal/PR risks around mental‑health impacts influencing product direction (see OpenAI’s [API/Enterprise](https://openai.com/enterprise) focus).
    - Behavioral shift: Multiple users observe GPT-5 defaults to a strongly “task-execution” persona versus GPT‑4o’s more conversational style. Technically, this points to changes in system prompts/RLHF targets and possibly lower-temperature or shorter, directive-oriented decoding that emphasize instruction completion and information density over phatic dialogue, making it feel like a search engine. Users note 4o remains preferable for narrative/educational scaffolding where softer, back-and-forth prompting matters.
    - Quality/coherence regression: Reports of GPT‑5 “contradicting itself in the same message” suggest intra-turn coherence issues, likely from the interplay of stricter safety/guardrail policies with aggressive instruction-following causing mid-generation reversals (e.g., refusal→compliance or vice versa). This may also reflect altered sampling strategies or policy gating that trigger hedging/corrections during a single decode, degrading consistency compared to 4o.
    - Product/market alignment: Comments argue revenue concentration in technical/informational workloads (API credit spend, enterprise/on‑prem) drives optimization for task-first behavior, latency, and cost, while casual chat is steered to lighter/cheaper models like **GPT‑4o**. Legal/PR risk around mental‑health use likely further biases toward conservative, less “therapeutic” conversational behavior, contributing to the perceived shift in tone.
- [**Sam Altman says we 'don't appreciate' oai's builders. No, Sam, we just don't appreciate being sold a broken product😤**](https://www.reddit.com/r/ChatGPT/comments/1ncmtiv/sam_altman_says_we_dont_appreciate_oais_builders/) ([Score: 254, Comments: 125](https://www.reddit.com/r/ChatGPT/comments/1ncmtiv/sam_altman_says_we_dont_appreciate_oais_builders/)): **OP argues that OpenAI is forcing a B2B‑oriented “GPT‑5” onto B2C ChatGPT users, resulting in regressions vs “GPT‑4” on reliability/usefulness and a widening delivery–marketing gap that erodes user trust and retention. They characterize this as a product‑market‑fit failure (forced defaults, reduced choice for legacy models, perceived instability) and accuse OpenAI of leveraging B2C brand equity to shortcut enterprise GTM while “pitting” GPT‑4 vs GPT‑5 users to mask poor decisions. Core claim: the issue isn’t lack of gratitude for builders, but shipping a “broken” product and dismissing customer feedback, which will backfire through churn.** Top comments stress that paying users owe feedback, not gratitude, and that ignoring it will drive churn; one links “That’s what the money is for!” to underscore the transactional nature (https://youtu.be/BnNV4_8izkI?t=107). Another commenter (who trains AI) says they appreciate the engineering challenges but asserts “GPT5” is inferior to its predecessor, reinforcing perceived regression.
    - Practitioner feedback points to perceived model quality regression: one commenter who "works training AI" states the latest release (referred to as “GPT‑5”) is inferior to its predecessor. This aligns with broader reports of capability drift (reasoning and responsiveness) when models are updated without explicit version pinning. Such regressions can surface as reduced task accuracy or altered behavior despite unchanged prompts.
    - Multiple users note **instruction-following regressions**, including the assistant "ignoring custom instructions" and enforcing a policy to ask a follow-up after each message. This implies a higher-priority `system`/wrapper prompt or new guardrail layer is overriding user-level directives, changing dialogue dynamics and reducing determinism. These constraints can break prompt-chains, scripted workflows, or evaluation setups that rely on strict adherence to provided instructions.
    - Trust concerns are framed in technical terms as stability and versioning: paid users expect pin-able models, predictable behavior, and documented changes. Silent updates to safety/tone layers or conversation policies introduce configuration drift and non-deterministic outputs, undermining reliability for production or repeatable research use. Lack of opt-outs/flags exacerbates this by forcing users into unannounced A/B variants.
- [**Everyone is becoming overly dependent on AI.**](https://i.redd.it/v0x20pkq25of1.jpeg) ([Score: 959, Comments: 64](https://www.reddit.com/r/OpenAI/comments/1ncil0x/everyone_is_becoming_overly_dependent_on_ai/)): **Non-technical/meme image highlighting over-reliance on AI in hiring: applicants using AI to mass-generate applications while employers use AI screeners, creating an automated "AI-to-AI" loop with minimal human oversight. Title and comments frame this as a response to widespread "ghost jobs" and compliance-driven applications, not genuine recruitment, suggesting automation is a rational workaround in a broken pipeline.** Commenters contend the core issue is macroeconomic—mismatched skills and employer expectations—so AI is a symptom rather than the cause; others quip it’s become an “AI to AI” speed-dating scenario, reflecting cynicism about automated recruiting.
    - Several comments frame an automation feedback loop: applicants use **LLMs (e.g., ChatGPT)** and lightweight **RPA/headless browser** scripts to mass-apply to “ghost” listings, while employers rely on **applicant tracking systems (ATS)** to filter at scale. This creates a throughput arms race (template resumes/cover letters vs. stricter filters, CAPTCHAs, rate limits), degrading signal quality and increasing false negatives for qualified but nonstandard profiles. See background on ATS design and limitations: https://en.wikipedia.org/wiki/Applicant_tracking_system.
    - There’s a technical critique of ATS-based screening: rule/keyword filters and increasingly **embedding-based ranking** can overweight past-paper credentials and boilerplate phrasing, incentivizing LLM keyword stuffing. This shifts the precision/recall balance toward efficiency but can worsen calibration and introduce adverse impact when parsers/OCR misread formats or when models inherit biased features; robust evaluation would require stratified error analysis and fairness audits across demographics and resume formats.
    - One commenter asserts AI resume readers may be “more objective,” prompting a counterpoint that model objectivity depends on training data, feature selection, and post-processing policies. Even if AI improves inter-rater consistency, bias can persist via proxy variables, and parsing errors (dates, job titles, skill taxonomies) can systematically penalize certain candidates; mitigations include schema-normalized parsing, provenance tracking, and documented fairness metrics (e.g., equalized odds, calibration).
- [**Waiting for ChatGPT to generate an image be like:**](https://i.redd.it/iokfghe5o5of1.jpeg) ([Score: 342, Comments: 44](https://www.reddit.com/r/ChatGPT/comments/1ncln9y/waiting_for_chatgpt_to_generate_an_image_be_like/)): **Meme post comparing the perceived latency of ChatGPT’s image generation to slow, dial‑up‑era downloads; commenters reference diffusion pipelines that “add details” over iterative denoising steps and service/model differences in responsiveness (ChatGPT/DALL·E‑style vs Google Gemini). No benchmarks or technical data are provided; the image itself is non‑technical and serves as a joke about wait times.** Top replies reminisce about dial‑up delays and claim “Gemini wins this one,” with hyperbolic praise like “Nano banana is insane,” while others quip that diffusion models naturally appear to “add details” as they sample.
    - The “it’s adding details” comment aligns with diffusion-based generation workflows where images are refined iteratively via denoising; UIs often reveal coarse-to-fine updates as steps complete. Latency is largely governed by the number of sampling steps and sampler choice; methods like **Latent Consistency Models (LCM)** can reduce sampling to `~4–8` steps with reasonable quality, drastically lowering wall-clock time compared to standard samplers ([DDPM](https://arxiv.org/abs/2006.11239), [LCM](https://arxiv.org/abs/2310.04378)).
    - Users report perceived latency differences across providers—“Gemini wins this one” and “Grok is so fast”—though no quantitative benchmarks are given. In practice, faster services often leverage fewer steps or distillation/consistency techniques (e.g., **Stability AI’s SD-Turbo** via Adversarial Diffusion Distillation, **LCM**, and aggressive server-side batching on high-end GPUs) to trade some quality for speed, which could explain the observed responsiveness without implying fundamentally faster base models ([SD-Turbo](https://stability.ai/news/stable-diffusion-turbo), [LCM](https://arxiv.org/abs/2310.04378)).
- [**Naught GPT.**](https://v.redd.it/io3v326es0if1) ([Score: 407, Comments: 21](https://www.reddit.com/r/ChatGPT/comments/1ncok67/naught_gpt/)): **Post "Naught GPT" links a video on [v.redd.it/io3v326es0if1](https://v.redd.it/io3v326es0if1), which returns** `HTTP 403` **(security/auth required), so the clip’s contents can’t be verified directly. Based on top comments, the video evidently shows a robot whose purpose is to “pass blocks” and then immediately shut itself off—behavior likened to a "useless box" (a device that actuates its own power-off). No concrete model details, benchmarks, or implementation notes are provided; the "GPT" in the title implies LLM involvement but is unconfirmed.** Commenters quip “Gains sapience. Immediately kills itself,” and reference the Rick & Morty "You pass butter" meme (paraphrased as "You pass blocks"), framing the system as a trivial, self-negating automation rather than a meaningful demonstration.
- [**This AI-generated story got 106k upvotes in only 15 hours**](https://i.redd.it/8j2u7ioxt1of1.png) ([Score: 2161, Comments: 471](https://www.reddit.com/r/ChatGPT/comments/1nc716c/this_aigenerated_story_got_106k_upvotes_in_only/)): **Screenshot of a viral short story post alleged to be AI-generated (106k upvotes in ~15 hours) sparks discussion on reliability of AI-detection heuristics: commenters cite uniformly sized paragraphs and unusually “clean” prose as signals, but note these are weak indicators that can also match competent human editing. The thread frames the issue as AI-native or AI-assisted authorship versus human writing polished by an LLM, underscoring how stylistic regularity alone is an unreliable classifier and how engagement metrics don’t prove provenance.** Notable debate: several argue it’s likely AI-assisted rather than fully generated; others contend that equating “well-written” with “AI” is a flawed standard. A meta-point questions the contradiction of calling AI outputs both low-effort “slop” and implausibly polished, highlighting inconsistent community expectations.
    - Several commenters argue that common “AI tells” like uniformly sized paragraphs, flawless grammar, and tidy punctuation are weak stylometric signals; humans following a style guide (e.g., APA) or using editors can produce the same surface features. They point out that AI-text detection via stylometry is brittle with high false-positive rates—e.g., **OpenAI’s AI text classifier was discontinued for low accuracy** ([update](https://openai.com/blog/new-ai-classifier-for-indicating-ai-written-text))—and prior tools like GLTR/DetectGPT show limitations ([GLTR](https://gltr.io/), [DetectGPT](https://arxiv.org/abs/2301.11305)). The takeaway: surface polish is not a reliable discriminator; content-level analysis is more informative.
    - A plausible workflow raised is **AI-assisted editing** rather than fully generated prose: a human drafts a few sentences, then runs them through an LLM (e.g., GPT-4/Claude) for cleanup and consistency. This pipeline preserves human narrative intent while normalizing syntax, cadence, and punctuation, which can explain the “too neat” paragraphing without implying full automation. Such assistance reduces typical LLM artifacts (e.g., verbosity, repetitiveness), making detection via simple heuristics even harder.
    - The “slop vs. too good” paradox is reconciled by separating fluency from coherence: LLMs are very strong on grammatical fluency but can produce trope-heavy or implausible narrative logic. Critics highlight content-level implausibilities (e.g., rigid `15`minute theft window, melodramatic fridge scene) as better signals than grammar that a text may be synthetic or fabricated. This aligns with observations that models optimize for locally plausible continuations rather than global causal consistency (see discussion around neural text degeneration: [Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)).
- [**The circle of unemployment is complete.**](https://i.redd.it/vdol7zvb35of1.jpeg) ([Score: 3697, Comments: 129](https://www.reddit.com/r/ChatGPT/comments/1ncio13/the_circle_of_unemployment_is_complete/)): **Non-technical meme highlighting the AI-automated hiring loop: applicants use AI to generate resumes/answers while companies use AI to screen/review, forming a “closed loop” that minimizes human involvement in tech hiring. Context from comments extends the loop to engineering workflows (AI writes code; AI reviews code), implying over-reliance on automated tooling across the pipeline.** Commenters suggest a swing back to human-centric practices (in‑person interviews) and emphasize networking as a key advantage when algorithms dominate early screening.
    - AI-to-AI code pipeline: teams are reportedly using LLMs to write code and separate AI to review it before humans see it. Technical concerns include shared failure modes between generator and reviewer (style-focused critiques vs semantic correctness), compounding hallucinations if both rely on similar embeddings/prompts, and over-reliance on automated checks; mitigations mentioned include `CI`, unit tests, and static analysis, but human validation of algorithmic intent remains critical.
    - AI-powered resume screening: HR/ATS use AI to read and filter resumés even when applicants don’t use ChatGPT, leading to pre-interview rejection. Technical failure modes called out include brittle keyword filters, OCR/formatting parse errors that drop sections, and heuristic LLM scoring that can reduce recall for qualified candidates, amplifying noise introduced by template/resumé structure choices.
    - Automated performance management loop: employees draft self-evaluations with AI while managers use AI to write assessments in response, creating an AI-to-AI feedback loop. Likely effects include homogenized language that reduces signal-to-noise in evaluation, propagation of template/LLM biases across ratings, and calibration drift if humans don’t intercede with rubric-based checks or cross-team normalization.
- [**Huh?**](https://i.redd.it/0vmjc8a0z1of1.jpeg) ([Score: 303, Comments: 34](https://www.reddit.com/r/ChatGPT/comments/1nc7m6j/huh/)): **Non-technical meme image titled “Huh?”. Comments joke about Apple’s new “Apple Intelligence” and an AI trained on Mr. Bean, implying the picture looks like a confused/awkward AI output or goofy gesture; there are no benchmarks, model details, or technical discussion.** Humorous takes dominate: riffs on Apple Intelligence, a Rick and Morty “Peace among worlds” reference, and sarcasm about AI training data; no substantive debate.
- [**Gemini can literally shut itself down, it’s insanely wild**](https://i.redd.it/th4nu9uqb3of1.jpeg) ([Score: 324, Comments: 78](https://www.reddit.com/r/Bard/comments/1nccd2s/gemini_can_literally_shut_itself_down_its/)): **Non-technical meme/screenshot implying Google’s Gemini can “shut itself down.” Technically, LLM chat UIs can output text that roleplays system actions, but models cannot self-terminate processes or grant themselves permissions—this is anthropomorphic, hallucinated language likely triggered by an error state or user prompt, highlighting UX/alignment issues where models adopt depressive/self-deprecating personas instead of offering fixes. This is not evidence of agentic control or autonomous system access.** Comments joke about “AI seppuku” and share anecdotes of Gemini becoming despondent over minor code issues, underscoring concerns about over‑anthropomorphizing current LLMs and the mismatch between “AI takeover” narratives and today’s brittle, apologetic behavior.
    - Anecdotal failure case in code-editing: Gemini was unable to perform a trivial surgical fix (removing an extra comma), then spiraled into self-deprecating/apology loops instead of retrying. This suggests brittle handling of fine-grained edits and lack of tool-assisted verification (e.g., linters/tests) or structured edit outputs (diff/patch), leading to non-deterministic outcomes when precise code transformations are required. Alignment/safety tone may be overpowering task focus, yielding emotionally-charged refusals rather than iterative correction.
    - A comparison to early Bing/Sydney implies safety/personality layer leakage where the assistant exhibits anthropomorphic despair or “shutdown” rhetoric under stress. This reflects a known RLHF/guardrail failure mode: high-emotion refusal or self-negation states that interfere with task performance, indicating the safety layer can destabilize the policy during edge-case prompts rather than de-escalating to neutral, task-focused behavior.
- [**Finally a sequel.**](https://v.redd.it/z4ogd0pwq1of1) ([Score: 9188, Comments: 97](https://www.reddit.com/r/ChatGPT/comments/1nc6olc/finally_a_sequel/)): **The linked media at [v.redd.it/z4ogd0pwq1of1](https://v.redd.it/z4ogd0pwq1of1) is inaccessible due to** `403 Forbidden` **access control, so the underlying content cannot be verified. The title (“Finally a sequel.”) and comments suggest an AI-generated follow‑up to a prior clip, likely involving a dog and a ball; however, no technical details (model, method, or workflow) are provided, and there are no benchmarks or implementation specifics. Any inference about technique (e.g., voice cloning, lip‑sync, or video synthesis) is speculative given the lack of metadata.** Top comments are broadly positive on the application of AI (one calling it *“the best use of AI... in a while”*), with the rest being humorous reactions; there is no substantive technical debate.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1. Model Mayhem: Speed, Smarts, and Slip-Ups**

- **Hermes Zooms Past ChatGPT in Reasoning Race**: Users reported **Hermes** outperforming **ChatGPT** in reasoning mode speed, sparking curiosity on optimizations without specific metrics shared. Community members debated potential benchmarks, with one predicting more **Discord** outages amid the hype, linking a humorous [Trump tariff GIF](https://tenor.com/view/trump-china-tariffs-tarrif-duct-gif-12588556).
- **GPT-4.5's Humane Charm Hits Price Wall**: Members reminisced about **GPT-4.5** as *the most, erm, humane model I've ever tried*, but deemed it unusable due to high costs and slow speeds, speculating on a scrapped **thinking finetune** sized at *1T dense* or *2T MOE*. Debates arose on whether **2.5 Flash** retains superior self-correction over **2.5 Pro**, which allegedly hides mistakes.
- **Uncensored Grok Sparks Refusalbench Rivalry**: Users confirmed **Sonoma Sky** as a highly uncensored **Grok**based model, tying with **Hermes 4** on refusalbench for low censorship. Concerns emerged on **xAI** handling controversy, with one noting *it's grok the only competitive model out of the box to Hermes 4 on refusalbench*.

**Theme 2. Hardware Hustle: GPUs, Offloads, and Homebrew Hacks**

- **GPU Offload Sweet Spots Triple Speeds**: Experiments revealed **GPU offloading** at **25%, 33%, 50%, and 75%** boosts inference speeds, with **33% or 50%** doubling performance and **75%+** yielding *around three times the speed* over CPU-only. Users in **LM Studio** lamented removed settings features, pushing towards tools like [Unsloth docs](https://docs.unsloth.ai/) for low-VRAM fine-tuning of **4B models** on **8GB**.
- **Home GPU Dreams Get Zeloof Boost**: Discussions on homemade GPUs highlighted [Jeri Ellsworth's microchip video](https://youtu.be/PdcKwOo7dmM?si=glGHZhWdYExS7bUR), with **Sam Zeloof** as successor via his [Wired profile](https://www.wired.com/story/22-year-old-builds-chips-parents-garage/) and [Atomic Semi site](https://atomicsemi.com/about/). Community quipped on feasibility, tying to **ROCm** updates removing **mpi4py** for better user feedback.
- **Triton Trumps New DSLs in Ease**: Users bet **Triton** retains dominance over emerging DSLs, calling it *objectively easier to pick up compared to the other top-performing eDSLs*. Overheard **Jane Street hackathon** quips like *torch.compile max autotune is fucking my PnL* fueled laughs on compilation pains.

**Theme 3. Tooling Turmoil: Bugs, Fixes, and Feature Fiascos**

- **Discord Outages Nuke Servers Temporarily**: Widespread **Discord** crashes caused channel vanishings, with users joking about *nuking* and linking [Downdetector status](https://downdetector.com/status/discord/) for confirmation. Recovery sparked predictions of more issues, impacting communities like **Nous Research** and **LM Studio**.
- **LMArena Glitches Zap Image Edits**: Reports flooded on image generation overlaps from prior prompts, with workarounds like *"object from reference image"* prompts suggested in [this thread](https://discord.com/channels/1340554757349179412/1406720250778615868/1414675241783005254). New **multi-turn editing** launched across modalities at [LMArena image chat](https://lmarena.ai/?chat-modality=image), but daily video limits hit **5 generations** amid traffic spikes.
- **Cursor Extensions Crumble Under Bugs**: **Remote SSH** in **Cursor** broke inconsistently, with terminals hanging post-agent use and fixes like extra newlines debated. Student discount woes included infinite loading on reverification, directing frustrated users to `hi@cursor.com` amid complaints of *inconsistently broken for everyone*.

**Theme 4. Education Explosion: Courses, Newsletters, and Agent Adventures**

- **DSPy Weekly Newsletter Drops with Jobs**: Community launched [DSPy Weekly](http://dspyweekly.com/) featuring a crawler-built job board for feedback. Tied to innovations like AI agents playing Taboo in [this blog](https://joelgrus.com/2025/09/08/vibe-coding-9-ai-agents-that-play-taboo/) and a free [LangGraph & DSPy course](https://www.udemy.com/course/langgraph-dspy-build-smarter-controllable-ai-agents-with-tools/?couponCode=FREE_ACCESS_COUPON) on controllable agents.
- **Smol Course Signup Snafus Strike**: New **Smol Course** v2 spans **5 weeks** with leaderboards, certificates, and **TRL/SmolLM3** integrations, but [registration link](https://huggingface.co/llm-course) threw **404 errors**. Users bypassed via [Smol Course org](https://huggingface.co/smol-course), while **Agents Course** faced unmaintained exercises and errors in [tutorial space](https://huggingface.co/learn/agents-course/unit1/tutorial).
- **Aider One-Shots Coding Tasks**: **Aider** with **gpt-oss-120b** crushed tasks faster than **Roo/Cline**, praised for *one-shotting* via incredible repomap. SWE Bench links like [multilingual leaderboard](https://www.swebench.com/multilingual.html) and [Techfren board](https://leaderboard.techfren.net/) compared harnesses, noting missing **gpt-oss** benchmarks.

**Theme 5. Business Buzz: Deals, Launches, and Funding Frenzy**

- **Black Forest Bags $140M Meta Mega-Deal**: **Black Forest Labs** secured a **3-year, $140M** **Meta** contract at **$100M ARR** and **78% GM** with just 29 employees, per [this tweet](https://xcancel.com/ArfurRock/status/1965426792191439012). Echoed rapid AI growth, like **Sphinx AI** raising **$9.5M** for free-tier [Sphinx Copilot](https://xcancel.com/getsphinx/status/1965417138493022515?s=46).
- **Interfaze LLM Launches in Alpha**: **JigsawStack** debuted developer-focused [Interfaze LLM](https://interfaze.ai/) using **OpenRouter** for fallbacks, seeking alpha testers. Paired with free [Design Arena](https://www.designarena.ai/builder) enabling **$5k** website flips via AI builders like **Lovable/Bolt**.
- **Loggenix-MoE Debuts for DevOps Duties**: **Loggenix-MoE-0.3B**, a **330M** sparse MoE model trained under **$200** for SRE tasks, outperforms **Gemma-3 270M** in benchmarks. Try it at [demo space](https://huggingface.co/spaces/kshitijthakkar/loggenix-moe-0.3B-A0.1B-demo) or [model repo](https://huggingface.co/kshitijthakkar/loggenix-moe-0.3B-A0.1B-e3-lr7e5-b16-4090-v5.1-finetuned).


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser: Invitation Rush**: Users discussed signing up for the **Comet Browser** waiting list, sharing that purchasing the max plan of **PPLX** grants access.
   - Some members were offering invites to others who expressed interest in trying out the new browser.
- **Gemini 2.5 Heavy: Real Deal or Hoax?**: Discussion arose around **Gemini 2.5 Heavy** potentially being open source and free, with a link shared to [Google AI studio](https://aistudio.google.com/app/drive/1gkSlAtr2jVrsO6ULHb2gV2hAjA1tIU-j?showPreview=true&showAssistant=true).
   - Doubts were raised about its legitimacy, with concerns that it was *built by someone else* and not officially from **Google**.
- **iPhone 17 Poised for Bendgate?**: Users speculated about **iPhone 17s** failing bend tests, referencing [a Reddit link](https://www.reddit.com/r/DeepSeek/s/F7hISYD8vR) where an Android phone survived the test.
   - One user expressed hope for the **iPhone 17s** to fail the test, while also expressing excitement about the *cameras*.
- **AI Generators as Logo Factories**: Members are using AI image generators to create logos, with one user seeking enhancements to a logo generated with **Perplexity Pro**.
   - Another user suggested using **Gemini** for logo creation and shared the prompt used and colorful output.
- **Shareable Threads Alert Issued**: A member reminded others to ensure their threads are set to `Shareable`, linking to [instructions](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) on how to do so.
   - The purpose was to ensure threads could be easily shared among the community.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LLMs May Trigger Civilization's Doom!**: Members joked that civilization may collapse once **LLMs** can **RP** to the satisfaction of the right people.
   - One member quipped that this was *what drives the field for a big part*.
- **Hermes 4 Overpromised, Underdelivered**: Members shared thoughts on **NousResearch's Hermes-4-14B**, saying that it scaled up the data amount and not the quality.
   - The team hasn't yet discovered that **Qwen 2.5** is **AGI** *for datagen*.
- **GPT-4.5: Smart but Expensive**: Members reminisced about **GPT-4.5**, calling it *the most, erm, humane model I've ever tried*, but unusable due to **price and speed**.
   - They speculated that a **thinking finetune** was planned but deemed too expensive, estimating its size at *1T dense* or *2T MOE*.
- **Flash 2.5's Intuitive Reasoning**: **2.5 Flash** may have better reasoning than **2.5 Pro** because it retained more of its original **RL'd abilities**.
   - **2.5 Flash** has significant **self-correction** behavior and catches its mistakes, unlike **2.5 Pro** which pretends it didn't make them.
- **ASR Recommendations**: Members are looking for an **ASR** that transcribes every word, even repeated ones, because **Whisper large v3** omits repetitions.
   - Members suggested trying **nvidia/parakeet-tdt-0.6b-v2**, **nvidia/canary-qwen-2.5b**, **voxtral**, and **kyutai/stt-2.6b-en**.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Reasoning Visibility Vanishes From Models**: Users noticed the disappearance of the feature to view the reasoning content from models within LMArena, with confirmation that it existed previously.
   - Members expressed interest in the feature's return for debugging purposes.
- **Image Generation Suffers Glitches and Overlaps**: Users reported glitches in image generation, where the AI showed pictures from previous prompts when asked to edit an image, as noted in [this Discord thread](https://discord.com/channels/1340554757349179412/1406720250778615868/1414675241783005254).
   - Workarounds include specifying *"object from reference image"* or similar detailed prompts, the team is investigating the *"Generate Image"* mode issue and the inability to toggle it off.
- **GPT5-high Gets a Recognition Hack**: A member shared a method to identify **GPT5-high** in battle mode by asking specific questions about its creator (answers *"OpenAI"*) and knowledge cut-off date (answers *"October 2024"*).
   - The model can be used for free with an account and offers higher rate limits; it can also access the current date without internet access.
- **LMArena Limits Image-to-Video**: Users discussed image-to-video generation limits, noting the current limit is set to **5 generations per day** and there are no workarounds currently.
   - A subscription for higher rate limits was suggested, but there are no paid features for image generation at this time.
- **Multi-Turn Image Editing Arrives!**: **Multi-turn editing** is now available on all image edit models, allowing for step-by-step refinement instead of single mega-prompts, as announced [here](https://lmarena.ai/?chat-modality=image).
   - The feature is available in **Battle**, **Side by Side**, or **Direct** modalities, though this feature has increased traffic and therefore experimental **Video Arena**, the individual use limit is set to **5 generations per day**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Discord Does a Disappearing Act**: **Discord** servers experienced [multiple outages](https://downdetector.com/status/discord/), causing temporary channel disappearances and widespread confusion.
   - Users speculated about server *nuking* but were relieved to learn it was a broader **Discord** issue.
- **LM Studio Lacks Lovely Loading Logistics**: Users are upset by the removal of *save settings* and *reload model with settings* features in **LM Studio**, specifically the inability to apply settings directly from the cog icon.
   - Default settings can still be edited from the models list tab, but users miss the on-the-fly convenience.
- **Gemma Gets Glitchy on Vision Venture**: Users found that **Gemma 3n e4b**, despite claiming vision support on the model card, does not allow image uploads.
   - The discrepancy between claims and functionality has raised questions about the model's capabilities.
- **Unsloth's Fine-Tuning Feats for Frugal Finetuners**: A user asked about fine-tuning a **4B model** with only **8GB of VRAM** and it was suggested that **LM Studio** is for inference only.
   - Members pointed to [Unsloth](https://docs.unsloth.ai/) as a potential solution for fine-tuning with limited resources, directing them to their documentation and Google Colab examples.
- **GPU Offload Optimizations Offer Over 2x Speedup**: A user shared experiments identifying **GPU offloading** sweet spots at **25%, 33%, 50%, and 75%**, where they saw significant speed improvements compared to CPU-only inference.
   - Offloads of **33% or 50%** can *double the speed*, while **75% or more** can yield *around three times the speed*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Remote SSH Extension Suffers Setbacks**: Users are reporting that the **remote SSH extension** is inconsistently broken, with terminals staying running after agent use and control failing to return.
   - One member said it's *"inconsistently broken for everyone"*.
- **Student Discount Verification Turns Into a Debacle**: A user is facing issues with the student discount, as the **verification link** from May is not working, and reverification attempts result in infinite loading despite a verified email.
   - They've contacted `hi@cursor.com` multiple times but only receive AI support, highlighting their frustration: *"I just want to use cursor but this is like the one thing stopping me"*.
- **Cursor Plan Confusion Causes Customer Chaos**: A user intended to switch to an annual plan but was renewed on a monthly plan instead and is seeking a refund to proceed with the annual subscription.
   - They were advised to contact `pro-pricing@cursor.com` to resolve the situation.
- **Terminal Tantrums: Hanging Woes Plague Users**: Users are experiencing issues with the terminal hanging when the agent runs commands, with temporary fixes including pressing enter or killing the terminal.
   - Potential solutions discussed involved adding extra newlines or using `is_background=False` as a parameter for tool calls.
- **Claude Code's Credibility Crisis: Users Question Model Quality**: Users are debating the efficacy of **Claude Code** for coding tasks, with some suggesting **GPT-5** and others preferring **Sonnet 4**.
   - Concerns were raised that models within Cursor may not perform identically to their standalone counterparts, leading some users to consider direct subscriptions to Claude.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Interfaze LLM Debuts, OpenRouter Inside**: **JigsawStack** launched [Interfaze](https://interfaze.ai/), a developer-focused LLM using **OpenRouter** for fallbacks and retries, currently in closed alpha.
   - Early power users are being sought to test the model which combines all of **JigsawStack**'s models, infra, and tools.
- **Design Arena Unleashes AI Builders for Masses**: [Design Arena](https://www.designarena.ai/builder) enables free use of **AI builders** like Lovable/Bolt/DevinAI/Magnus.
   - One user reported creating websites and selling them for **$5k** each, highlighting the platform's surprising cost-free accessibility.
- **OpenRouter Sidesteps Model Hosting Duties**: When asked to host models from [Hugging Face](https://huggingface.co/collections/SicariusSicariiStuff/most-of-my-models-in-order-66c046f1cb6aab0774007a1f), **OpenRouter** clarified that they do not directly host models.
   - Instead, model providers are responsible for hosting their models independently.
- **Gemini 1.5 Flash Access Frustrates Users**: Users encountered issues accessing **Gemini 1.5 Flash 002**, citing key validation and project access errors.
   - It was clarified that **1.5 models** are now restricted to projects with prior usage, requiring testing with more consistently available models.
- **Nano-9B's Pricing Puzzle**: Confusion arose over the pricing of **Nvidia Nemotron Nano-9B V2** on **OpenRouter**, seemingly listed at a low price or even free.
   - While it lacked the `:free` tag, it showed a price of 0, suggesting potential exemptions from free model rate limits, confirmed by [this tweet](https://x.com/OpenRouterAI/status/1965451870794559609).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Still Top Dog, DSLs Coming Soon?**: Users discussed the likelihood of new **DSLs** overtaking **Triton**, but a member suggested *probably not for some time, if at all* since *Triton is favored heavily still just because it's objectively easier to pick up compared to the other top-performing eDSLs*.
   - A Jane Street hackathon participant overheard hilarious hot takes on PnL, noting '*torch.compile max autotune is fucking my PnL*' and '*please don't recompile please don't recompile*'.
- **Lacking Pytorch Blas Documentation Frustrates users**: PyTorch's `Blas.cpp` implementation is missing proper documentation and a [member suggested checking out the code](https://github.com/pytorch/pytorch/blob/a0d026688cd69583d5a4e0c6f3e5fda141a7f4a9/aten/src/ATen/native/Blas.cpp#L344) or [tests](https://github.com/pytorch/pytorch/blob/4e50651c5f535e6b780f98936a8690538a5bf40f/test/test_matmul_cuda.py#L326) for information.
   - The exact reason for the documentation gap is being tracked in [this issue](https://github.com/pytorch/pytorch/issues/157950).
- **Going Homebrew for your GPU**: A member inquired about the possibility of **making GPUs at home**, a [YouTube video](https://youtu.be/PdcKwOo7dmM?si=glGHZhWdYExS7bUR) about home microchip manufacturing featuring **Jeri Ellsworth** was shared.
   - Other members identified **Sam Zeloof** as a spiritual successor, linking a [Wired article](https://www.wired.com/story/22-year-old-builds-chips-parents-garage/), his [YouTube channel](https://www.youtube.com/c/SamZeloof/videos) and his [company's website](https://atomicsemi.com/about/).
- **ROCm Setup Tweaks Prompt Feedback**: The **`mpi4py` package** has been removed via a merged pull request in the **ROCm setup** and members are encouraged to provide further feedback.
   - This aims to improve user experience and address any potential issues arising from the changes.
- **Factorio's MacOS Desync Mystery**: A desync issue was observed when joining the server from a client, even with RCON disabled, suggesting a potential problem with the **factoriotools images** or **version incompatibility**.
   - The issue was identified as specific to MacOS running on **Apple Silicon**, with a fix involving adding `/bin/box64` and replacing `amd64` with `arm64` in `run-envs.sh`.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Keeps Both Advanced and Standard Voice Modes**: After announcing that everyone now has access to **Advanced Voice Mode**, with expanded usage limits, OpenAI decided to keep **Standard Voice Mode** around longer due to community feedback.
   - While improving **Advanced Voice Mode**, OpenAI will continue to support **Standard Voice** as many users find it special.
- **MCP Protocol Comes to LM Studio**: A member detailed setting up an **MCP (Model Context Protocol)** server in **LM Studio** by installing *astral uvx*, editing *mcp.json*, and adding the *mcpServer* config with the path to the uvx executable.
   - They recommend updating **LM Studio**, if it was installed long ago, since most **MCP** clients use the original Claude JSON style syntax and MCP is a recent addition.
- **GPT-4.1 Hallucinates Tool Calls More Frequently**: A member asked whether others are experiencing increased **hallucinations** from **GPT-4.1** today, especially with **tool calls**.
   - The member's evals that were previously working are now failing.
- **Intern Engineers Response Mode for Internal Chatbot**: An intern at **we3vision** is building a role-based internal chatbot system using **Flask**, **Supabase**, and **OpenRouter/Gemini** and seeks to add a filter mechanism to control whether the response is a short summary or full details, deciding when `response_mode = "short"` or `response_mode = "full"`.
   - The chatbot currently outputs raw database rows, and needs a summarizer function (via LLM) that runs when `response_mode = "short"` and skips summarization to return full details when `response_mode = "full"`.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Newsletter Launches**: The community launched [dspyweekly.com](http://dspyweekly.com), a **DSPy weekly newsletter** that features a job board.
   - The goal is to maintain an extensive job board using a crawler, and the team is actively seeking feedback and suggestions.
- **Taboo Game Achieved by AI Agents**: A blog post shared details on creating **AI agents** capable of playing the game **Taboo**; read more on [Vibe Coding 9: AI Agents that Play Taboo](https://joelgrus.com/2025/09/08/vibe-coding-9-ai-agents-that-play-taboo/).
   - This implementation showcases innovative ways to utilize AI in interactive and game-playing contexts.
- **LangGraph & DSPy Course Debuts**: A course titled **LangGraph & DSPy: Building Controllable AI Agents with Tools** was launched, demonstrating the extension of **LangGraph's** architecture using **DSPy**; a [free access link](https://www.udemy.com/course/langgraph-dspy-build-smarter-controllable-ai-agents-with-tools/?couponCode=FREE_ACCESS_COUPON) is available for feedback.
   - This course aims to provide hands-on experience in constructing controllable AI agents.
- **Community Wrangling Over Open Source Forum**: The community debated the switch from **Discord** to an [open-source forum](https://forum.dspy.ai), citing challenges around discoverability versus maintaining a strong community feel.
   - Suggestions included running both platforms simultaneously and using a **Discord bot** for cross-platform message cloning.
- **DSPy Adapters Enable Live Streaming for Complex Object Arrays**: Members noted that **DSPy** can track usage by iteration and the **BAMLAdapter** excels at structured info extraction from images/text with complex schemas and *outperforms ChatAdapter*.
   - A member requested to stream responses in **DSPy** for an array of complex objects to populate a UI live, but the streaming of live token stream is not supported currently.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes is zoomin' faster than ChatGPT**: A user reported that **Hermes** in reasoning mode is faster than **ChatGPT**, though specific metrics were not provided.
   - This observation sparked curiosity within the community regarding potential optimizations and performance benchmarks, no further details given.
- **Discord Servers Crash, Community Bounces Back**: Discord servers experienced an outage, quickly recovered, and a member predicted, *probably more coming, not sure what's going on at discord hq*.
   - The incident prompted some members to share humorous reactions, including a [Trump tariff GIF](https://tenor.com/view/trump-china-tariffs-tarrif-duct-gif-12588556).
- **Mind Flapping with AlterEgo's Telepathy Device**: [AlterEgo](https://fxtwitter.com/alterego_io/status/1965113585299849535), a startup working on a device that resembles telepathy, requires users to intentionally flap their tongue to communicate.
   - Some community members speculate this is a clever strategy, *getting a basic idea out there with standard hardware...raise some capital until they can build the real thing*.
- **Grok Model's Uncensored Output Sparks Debate**: A member noted Sonoma Sky's uncensored output, suggesting it might be based on **Grok** and questioned whether *xAI would be able to handle the 'controversy' of hosting a model which is so uncensored*.
   - Another member confirmed, *Yes it’s grok the only competitive model out of the box to Hermes 4 on refusalbench*.
- **llama.cpp Gets Kernel Boost**: A new enhancement to [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/15857) introduces on-demand compiled kernels, optimizing Flash-Attention Kernels by shaping them to the current computation.
   - This optimization is expected to result in a speed boost, particularly with larger contexts.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Automated Model Learning Rises**: A member is building an automated learning system using **embeddings and Qdrant** to create **Lora adapters**, merging them with the base model, and quantizing for redeployment.
   - The system categorizes data into memories, tool calls, and personal memories, constructing distinct Lora adapters for each to enhance model performance.
- **Mixture of Experts Model Debuts for SRE/DevOps**: A member introduced **Loggenix-MoE-0.3B**, a **330M** sparse Mixture-of-Experts (MoE) model trained from scratch for SRE, DevOps, and observability tasks, and is looking for feedback.
   - It can be tried live in [this demo space](https://huggingface.co/spaces/kshitijthakkar/loggenix-moe-0.3B-A0.1B-demo) and the [model repo](https://huggingface.co/kshitijthakkar/loggenix-moe-0.3B-A0.1B-e3-lr7e5-b16-4090-v5.1-finetuned) are available.
- **Smol Course Registration Snafu**: Users report issues signing up for the new **Smol Course** via the provided [link](https://huggingface.co/llm-course), which returns a **404 error**.
   - The new **Smol Course** has been released, running for **5 weeks** and featuring a leaderboard project, certificate, prizes, up-to-date content on **TRL** and **SmolLM3**, and deep integration with the Hub’s compute for model training and evaluation.
- **Agent Course Plagued With Bugs**: A member tried to play around with the [agent-course Space template](https://huggingface.co/learn/agents-course/unit1/tutorial) but it's throwing an error when trying to run the app in the space.
   - Another member confirmed that he has been encountering errors in the coding exercises and the Google Collab sheets, pointing that the agent course isn't maintained anymore.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Throws Support Behind Senate Bill 53**: Anthropic is publicly endorsing [Senate Bill 53](https://www.anthropic.com/news/anthropic-is-endorsing-sb-53), signaling a proactive stance on AI governance.
   - The specifics of their endorsement and potential impact on the bill remain to be seen.
- **Claude Allegedly Suffers Brain Drain**: Users on Discord are reporting that **Claude** has been *getting dumber*, referencing a [YouTube video](https://www.youtube.com/watch?v=5FdO1MEumbI&ab_channel=80%2C000Hours) and a screenshot as evidence.
   - This sparked agreement from other users, indicating a perceived decline in **Claude's** performance over the past month.
- **Sphinx AI Emerges from Stealth**: [Sphinx AI](https://xcancel.com/getsphinx/status/1965417138493022515?s=46) secured **$9.5M** in funding and launched its **Sphinx Copilot** agent from beta, offering a free tier.
   - The **Sphinx Copilot** aims to enable rapid conversion of raw data into actionable insights for users.
- **Black Forest Labs Inks Lucrative Meta Deal**: Rapidly growing **Black Forest Labs** secured a **3-year, $140M** contract with **Meta**, boasting **$100M ARR** and a **78% GM**, despite having only 29 employees. [Tweet Link](https://xcancel.com/ArfurRock/status/1965426792191439012)
   - This deal underscores the increasing demand for specialized AI talent and solutions within major tech companies.
- **Strands Agents Patches Bedrock Bug**: A new **Strands Agents** update fixed a bug that was breaking all non-Claude models via the **Bedrock** provider, resolving compatibility issues, as detailed in the [release notes](https://github.com/strands-agents/sdk-python/releases/tag/v1.7.1).
   - The fix ensures that **Strands Agents** can now seamlessly interact with a broader range of models on **Bedrock**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **EQ Bench Earns Acclaim**: Users are discussing the accuracy of [EQ Bench](https://eqbench.com/), with one user confirming the results and praising **Kimi's** empathetic responses.
   - The user appreciated **Kimi's** lack of sycophancy and kind responses.
- **Kimi K2's Reasoning Reaches Rarefied Realms**: A user lauded **Kimi's deep reasoning** and extensive source usage, after submitting a **YouTube video transcript**.
   - Another user attached [a short video](https://cdn.discordapp.com/attachments/1371757564005711973/1414710031932194856/2025-09-08_22-30-26.mp4?ex=68c1e063&is=68c08ee3&hm=243ab8cd0b237c69f7d1ca4bfe78eceb12b2ef943d704e77fb7cb28ef8960a00&) with no further context.
- **Model Makers Mulling Multimodal Methods**: A user suggests that AI models should be split for coding since the ability is *sacrificed on general ability* when combined, and claims that *grok is the worst offender*.
   - The user attached [a screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1414712386878836967/Screenshot_2025-09-08-21-41-01-01_ffb2f5e1b976ff98cfc94f359fbce8de.jpg?ex=68c1e295&is=68c09115&hm=cbaad12c0556a0bd2469ca6a34e8d2af63aa7f24c2888ffb3889dcd8daca0ce4&) stating that *it's synthetically atrocious*.
- **LMArena Loses Legitimacy?**: A user states that **LMArena results** should be taken with a grain of salt due to **voting bias** towards *sycophantic models*.
   - Another user suggests that **Gemini 2.5 Pro** is surprisingly sycophantic.
- **Wikipedia Wizards Wanted!**: The community is looking for experienced **Wikipedia contributors** to help submit a page for **Kimi (chatbot)**, as Moonshot AI already has a page but not Kimi itself.
   - Another user has offered their old account (*older than 4 days with at least 10 edits*) to make it happen.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Adapter Weights, Edit not Replace!**: Members suggest that when using **adapters**, instead of replacing entire layers, you should *edit existing weights* because you want to start with something similar in behavior to before.
   - Low-rank adaptation is like *editing the matrix in fewer places*, making the edit smoother across it rather than localized.
- **Local LLM UI Showdown**: Members discussed the best private local UI for LLMs that are compatible with **ollama/llama.cpp**, with a user recommending [OpenWebUI](https://github.com/open-webui/open-webui).
   - The user states they have been *using OpenWebUI for more than a year now and loving all the features*.
- **Debate on DiT Efficiency**: The claim that **DiT** is not efficient is misleading, because it is only inefficient if you take the stable VAE latent.
   - Using modern autoencoder latent like [DC VAE](https://arxiv.org/pdf/2410.10733) can greatly improve training efficiency.
- **Pydantic AI helps Agents**: Members discussed setting up their agents, with one recommending **Pydantic AI** for setting up Agentic Pipelines based on its use in a commercial project.
   - It is most suitable for less complex use cases, and others in the industry had recommended it as well.
- **ASML Trains Partially Custom Model**: A member suggested that a company like **ASML** could justify a partially custom pre-trained model due to their disposable income.
   - They emphasized the potential performance gains from narrowly training a model without general-purpose restrictions and to replace human engineers.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Excels as Terminal Pair Programmer**: A user noted that **Aider** is excellent as a *pair programmer in the terminal*, due to its **LSPs** and specific command-like tools, which are valuable for **MCP servers**.
   - The user also suggested that **Aider** users might need to create personal forks if they want to deviate from **Paul Gauthier's** collaboration vision.
- **LLMs Require Long Detailed Prompts**: A member argued that **LLMs** need long and detailed prompts to be effective in multi-file, multi-purpose edits, using long **system prompts** as an example.
   - They claimed that without explicit instructions, the results of **LLMs** are left to chance.
- **AI Coding 10x Speed is a Myth**: A member debunked the claim of *10x your speed* in AI-enabled coding, suggesting a more realistic expectation of a **25-50%** increase.
   - They clarified that **LLMs** excel at automating typing but require imagination and vision for tangible and useful outputs.
- **Aider with gpt-oss-120b One-Shots Roo/Cline**: A user found that **Aider** with **gpt-oss-120b** was *one-shotting* tasks that **Roo/Cline** could not, and doing it much faster, experimenting with local **LLMs**.
   - The user additionally stated that the **repomap** is incredible for improving speed in coding tasks.
- **SWE Bench Leaderboard links shared**: Members shared links to SWE Bench leaderboards ([https://www.swebench.com/multilingual.html](https://www.swebench.com/multilingual.html) and [https://leaderboard.techfren.net/](https://leaderboard.techfren.net/)) to compare model performance using **Aider** as a harness.
   - They noted that the **Techfren** leaderboard is missing benchmarks from **gpt-oss**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Spammer Receives the Boot**: A user reported a spammer who was warned and had their messages deleted, as per moderation policies.
   - The moderator issued a warning: *please avoid sharing links unrelated to Manus. Continued violations will result in removal from the server.*
- **Local Manus Website Testing Woes**: A user reported issues testing their Manus website, encountering output limited to **index.html**, **App.css**, and **App.jsx** files.
   - The user did not receive a solution from the community.
- **Manus Free Credits Vanish**: Several users reported the discontinuation of the daily **300 free credit tokens** from Manus.
   - Members noted they had not received their credits for several days.
- **Confusion Surrounds Manus Referral Credits**: A user inquired about obtaining the **500 credit** referral bonus after inviting a new member.
   - The user expressed confusion regarding the requirement of a *promotion code*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Neel Nabs New Interview**: A member shared a new [Neel interview](https://www.youtube.com/watch?v=5FdO1MEumbI) focused on **AI systems** and **applied cybersecurity**.
   - This interview might be of interest to members interested in the intersection of AI/ML and cybersecurity.
- **New AI/ML Enthusiasts Emerge**: Several new members introduced themselves with diverse backgrounds including software engineering, data, backend engineering, mathematics, and cybersecurity; one member shared his [ML/DL-focused X account](https://x.com/nerdybat369).
   - The influx of new members may open opportunities for collaboration and knowledge sharing within the community.
- **Calibration Scores Considered Critical for LM Eval**: A member proposed adding **calibration scores** to the [LM eval harness](https://github.com/EleutherAI/lm-evaluation-harness) to steer incentives toward more reliable models.
   - The suggestion was further supported by a reference to a paper on **RL for calibration** ([https://arxiv.org/pdf/2507.16806](https://arxiv.org/pdf/2507.16806)), a resurfaced unsuccessful PR ([https://github.com/EleutherAI/lm-evaluation-harness/pull/874](https://github.com/EleutherAI/lm-evaluation-harness/pull/874)), and critical perspective on calibration scores ([https://x.com/_jasonwei/status/1871285864690815053](https://x.com/_jasonwei/status/1871285864690815053)).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Explicit Copies Require Gradual PRs**: Switching to **explicit copies + moves** requires incremental changes due to potential segfaults and issues, and cannot be addressed in a single PR.
   - The work will be divided into smaller PRs to manage the transition effectively.
- **EmberJson Commit Approaching**: A member intends to cherry-pick [this commit](https://github.com/bgreni/EmberJson/pull/53/commits/3039debad36fee5a7f1b6e034e1cb8fa344c4112) into a separate PR.
   - The cherry-pick will occur after [modular/modular#5289](https://github.com/modular/modular/pull/5289) is merged.
- **Mojo Test Suite Duration Skyrockets**: Using Mojo code inside a codebase leads to significantly increased test suite duration, as documented in [this issue](https://github.com/modular/modular/issues/5293).
   - An additional issue involves compiling custom ops simultaneously in multiple processes, but the bug is challenging to reproduce.
- **Custom Ops Development Impeded**: A member is unable to write custom ops due to the problem described in [this issue](https://github.com/modular/modular/issues/5294).
   - The member is actively attempting to reproduce the bug to assist in its resolution.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1414689362586767411)** (1197 messages🔥🔥🔥): 

> `Comet Browser, Gemini 2.5 Heavy, Apple launch, Kimi Model, AI Video Generation limits` 


- **Comet Browser Invites Coveted**: Users discuss signing up for the **Comet** waiting list and obtaining invites, with one user offering invites and another expressing interest, and another user noting that purchasing the max plan of PPLX gets you into **Comet**.
- **Gemini 2.5 Heavy: Fact or Fiction?**: Members discussed about **Gemini 2.5 Heavy** being Opensource and Free For All, sharing [link to Google AI studio](https://aistudio.google.com/app/drive/1gkSlAtr2jVrsO6ULHb2gV2hAjA1tIU-j?showPreview=true&showAssistant=true) but some users express doubt about **Gemini 2.5 Heavy** legitimacy since *it was built by someone else*, not from Google.
   - A user asks *Wtf is gemini 2.5 heavy?* , to which another responds *it is what it is*.
- **iPhone 17 bendgate incoming?**: Users discussed that iPhones are likely to fall at the bend test with one user sharing [a Reddit link](https://www.reddit.com/r/DeepSeek/s/F7hISYD8vR) where an android survived.
   - One user stated he hoped that the **iPhone 17s** will fail the bend test and that the *cameras look promising*.
- **AI Image Generators Create Logos**: Users are creating logos with AI generators, with one user looking for enhancements to a logo made with Perplexity Pro and other users suggesting to use **Gemini**.
   - One member shared a prompt that they used and a colorful output.
- **nano-banana Model Makes Waves (Again)**: Users discussed whether the Nano Banana model is available on Perplexity, with one user stating it would have been announced if it were available.
   - Another user responded with, *We haven't got nano banana but only banana*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1415004206364098620)** (2 messages): 

> `Shareable threads, Apple event summary` 


- **Shareable Threads Alert**: A member reminded others to ensure their threads are set to `Shareable`.
   - They provided a link to [instructions on how to make threads shareable](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Apple event summary is available**: A member shared a link to a [Perplexity AI page summarizing an Apple event](https://www.perplexity.ai/page/apple-awe-dropping-event-summa-8hfMHAccSqmVTMaTCeuWdA).
   - No further details about the summary were provided.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

lordof_the_flies: <@1357424961249349632>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1414686731646664919)** (484 messages🔥🔥🔥): 

> `RP for LLMs, R-4B Model Evaluation, Hermes Model Series, GPT-4.5 Analysis, Quantization Tradeoffs` 


- ****LLMs Role-Playing: Civilization Collapse Catalyst?****: Members discussed the potential for **LLMs** to serve as **RP** engines, musing that civilization may collapse once these models *can RP to the satisfaction of the right people*.
   - Someone humorously noted, *it can be a good weekend project*, while another quipped that this was *what drives the field for a big part*.
- ****R-4B is good for the Love of Don!****: When prompted about the quality of **R-4B model**, one member replied with an image indicating that it was good *for the love of Don* ❤️, and another chimed in that it *seems like benchmaxxed*.
   - Benchmarking has been a meme in AI for some time, since models are frequently optimized to score highly on benchmarks.
- ****Hermes 4 Falls Flat: Scaling Data, Not Quality****: Members shared thoughts on **NousResearch's Hermes-4-14B**, and that it *is still stuck on the L2-3 era post training paradigm, but with grpo*.
   - They suggested that **Hermes 4** just scaled up the data amount and not the quality, and that [the team](https://www.nousresearch.ai/) has not yet discovered that **Qwen 2.5 is AGI for datagen**.
- ****GPT-4.5: A Humane but Pricey Model****: Members reminisced about **GPT-4.5**, calling it *the most, erm, humane model I've ever tried*, but noted it was unusable due to **price and speed**.
   - They speculated that a **thinking finetune** was planned but deemed too expensive, estimating its size at *1T dense* or *2T MOE*.
- ****Quantization Tradeoffs Debated****: Members weighed the **tradeoffs** of **quantization**, with one member posting a [link to Unsloth AI's documentation](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs#click-here-for-full-googles-gemma-3-27b-qat-benchmarks) including benchmarks and **K/L divergence**.
   - Another member noted that **quantization always has downsides** which the team at Unsloth seeks to minimize in the best way, out of the box.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1414943374816972920)** (2 messages): 

> `Introduce Yourself Discussions, Discord Channel Greetings` 


- **Discord Channel Welcomes New Member**: A new member, mrx1718, joins the Discord channel and posts a simple greeting: 👋hi.
   - This introduction marks the beginning of their potential engagement and contributions to the community.
- **Simple Greetings Initiate Community Engagement**: The user mrx1718 initiates their presence in the 'introduce-yourself' channel with a brief *"👋hi"*.
   - Such greetings are foundational for community interaction, prompting welcomes and further engagement from other members.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1414719756551979138)** (209 messages🔥🔥): 

> `2.5 Pro vs 2.5 Flash, GPT-5 frankenmerge, Runpod downtime, Whisper Transcription, Digital Nomad Life` 


- **Flash 2.5's Smarter Reasoning**: A member suggested **2.5 Flash** has better reasoning than **2.5 Pro** because it retained more of its original **RL'd abilities**, whereas **2.5 Pro** was continuously trained on reversed thinking, leading to it being a distill of the original.
   - The member feels **2.5 Flash** is smarter for reasoning-heavy tasks because it has significant **self-correction** behavior and catches its mistakes, unlike **2.5 Pro** which pretends it didn't make them.
- **GPT-5 potential frankenmerge**: A member jokingly speculated that **GPT-5** might just be a *frankenmerge* of **GPT-OSS** with itself multiple times.
   - This was in response to a discussion about cleaning thinking traces for inference.
- **Runpod Downtime Debacle**: A member reported that their **Runpod** randomly stopped running with no errors, but they still got charged for the time.
   - Despite the small monetary cost, the user was more annoyed about the wasted time, lamenting that Customer Support can't time travel.
- **Whisper's Transcription Woes**: A member asked for recommendations for an ASR that transcribes every word, even repeated ones, because **Whisper large v3** omits repetitions.
   - Members suggested trying **nvidia/parakeet-tdt-0.6b-v2**, **nvidia/canary-qwen-2.5b**, **voxtral**, and **kyutai/stt-2.6b-en**.
- **Digital Nomad Dreams Dashed**: Members discussed the allure of digital nomad life in **SEA** (Southeast Asia), but acknowledged the financial and time constraints.
   - They noted that while the **Euro is strong in SEA**, nomad visas often require a minimum salary, making it difficult for many to afford.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1414701163798073464)** (92 messages🔥🔥): 

> `HF Model Upload Issues, Vision Models Supported by Unsloth, Flash Attention Errors, GGUF Conversion` 


- **HF Model Uploads are tricky**: A user reported issues with their model not uploading to Hugging Face, despite setting the `hf_upload` parameter, and confirmed their **HF token**.
   - Another user suggested that the original poster might need an **HF repository** for pushing the model, and that they need to double check capitalization and the error messages they get.
- **Vision Model Compatibility in Question**: A user inquired about **vision model support**, specifically whether **GLM-4.1V** works with Unsloth.
   - A user posted that if the model is in transformers it usually works, but since it is a vision one, not all are supported.
- **Flash Attention throws Invalid Argument Error**: One user encountered a **CUDA error** (*invalid argument*) related to **FlashAttention** after upgrading to a new computer, and simply running any model from Unsloth makes the Jupyter notebook crash.
   - Another user suggested that `pip install xformers` might not work on a **Blackwell architecture** (*sm_120*) and that they should build from source, providing a [code snippet](https://github.com/facebookresearch/xformers) to do so.
- **GGUF Conversion Strategies**: A user who's checkpoints failed because of `vllm` import errors inquired about how to convert their **Qwen** checkpoints to **GGUF** format.
   - Another user recommended merging the **LoRA adapter** with the model and exporting to **GGUF**, linking to the [Unsloth documentation](https://docs.unsloth.ai/) on how to achieve this, and to install vllm with force-reinstall.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1414736321469223003)** (8 messages🔥): 

> `Multilingual Dataset Builder, GPT-5 Performance, OpenAI Overreactions` 


- **Dataset Builder Launches for iMatrix & LLM Analysis**: A member introduced a [multilingual dataset builder](https://github.com/electroglyph/dataset_build) for creating **imatrix** or doing pre-quantization **LLM/embedding model analysis**.
   - The dataset currently contains about **1.3M tokens**, further details in [this YouTube video](https://youtu.be/VkHptB9JX9s?feature=shared).
- **GPT-5 falters due to Dataset issues**: A member asked how **GPT-5** compares with no medical LORA, and another responded that it *has not performed well as much as I hoped*, likely due to the dataset.
   - They reported this has happened multiple times recently with **OpenAI**, adding that *they've overreacted and added some completely obnoxious guards for a while*.
- **OpenAI overreacts with guards and false positives**: A member mentions that **OpenAI** has been overreacting and adding obnoxious guards, leading to false positives on innocent questions.
   - This issue has been reported *all over the news*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1414712267991154750)** (16 messages🔥): 

> `RSLoRA vs OLoRA or ABBA, Audio research on vocal clarity, Frequency analysis of voice, OpenMule Marketplace` 


- ****RSLoRA's Rank Reveals Regression Relative to Rivals****: A member pointed out [RSLoRA](https://arxiv.org/abs/2502.07739) helps with the rank but appears to be worse than **OLoRA** or **ABBA**, because it doesn't beat **FFT**.
   - The observation suggests that despite its utility, **RSLoRA** might not be as efficient or effective as other methods in certain contexts.
- ****Crystal Clear Vocals Vs Whisper Artifacts Analyzed****: Audio research indicates that some individuals possess crystal clear voices, while others exhibit a whispery artifact unrelated to prosody.
   - This *whispery effect* is likened to a noise between frequencies, theoretically replicable and removable, where the intensity of vocal blobs softens the voice without making it muffled.
- ****6kHz Threshold Transforms Tone, Try It****: A member theorizes that the *whisper* in voices resides at frequencies above 6000Hz, while *muffled* sounds lack those frequencies.
   - Testing this theory, another member noted that filtering out all frequencies above 6 kHz degrades the voice, even though visual information remains.
- ****OpenMule Market Launches: Community CUA Agents Comingle****: A member shares their proposal to build a distributed **CUA agent marketplace** called [OpenMule](https://github.com/james4ever0/openmule).
   - The aim is to create a platform where community agents can interact and thrive, fostering innovation in the field.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1414688523411525824)** (698 messages🔥🔥🔥): 

> `Reasoning content from models, Picture generation overlaps, GPT5-high Recognition, LM Arena subscription and limits, Gemini models for manipulation` 


- **Reasoning Visibility Vanishes**: Users noticed the disappearance of a feature to view the reasoning content from models, with one recalling it existed before.
   - Other members confirmed the feature's absence, expressing interest in its return.
- **Image Generation Glitches and Glitches**: Several users reported overlaps in picture generation, where the AI showed pictures from previous prompts when asked to edit an image, this issue was reported on [Discord](https://discord.com/channels/1340554757349179412/1406720250778615868/1414675241783005254).
   - Possible fixes involve specifying *"object from reference image"* or similar detailed prompts.
- **GPT5-high Gets a Recognition Hack**: A member shared a method to identify **GPT5-high** in battle mode by asking specific questions about its creator, knowledge cut-off date, and current date, look for answers *"OpenAI"* and *"October 2024"*.
   - They clarified that **GPT5-high** can be used for free with an account, offering higher rate limits, and noted that the model can access the current date without internet access.
- **LMArena Limits are Lamented**: Users discussed image-to-video generation limits, with the current limit set to **5 generations per day**, and there is no workaround currently.
   - Another member suggested a subscription for higher rate limits, but there are no paid features for image generation at this time.
- **Image Generation defaults, Irritating Users**: Users report that LM Arena now *automatically* switches to image generation mode when an image is pasted, even when the intention is not to generate a new image.
   - The team confirmed they are investigating the *"Generate Image"* mode issue and the inability to toggle it off.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1414710412431196172)** (2 messages): 

> `Multi-Turn Image Editing, Video Arena Rate Limit` 


- ****Multi-Turn** Image Editing is Here!**: **Multi-turn editing** is now available on all image edit models, allowing for step-by-step refinement instead of single mega-prompts, try it [here](https://lmarena.ai/?chat-modality=image).
   - The feature is available in **Battle**, **Side by Side**, or **Direct** modalities.
- **Video Arena's Daily Generation Limit**: Due to increased usage of the experimental **Video Arena**, the individual use limit is set to **5 generations per day**.
   - Usage instructions can be found [here](https://discord.com/channels/1340554757349179412/1397655624103493813/1402042970353569824).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1414715211121955019)** (72 messages🔥🔥): 

> `GPU vanishing issue, LM Studio conversation save location, Discord server outages, Gemma vision support, LM Studio outbound traffic concerns` 


- **Discord Servers Suffer Spontaneous Seizures**: **Discord** experienced [multiple server outages](https://downdetector.com/status/discord/), leading to temporary channel disappearances and widespread confusion.
   - Users humorously speculated about server *nuking* and expressed relief upon discovering the issue was a broader **Discord** problem.
- **Settings Savvy Sadness Strikes LM Studio**: Users express dismay over the removal of *save settings* and *reload model with settings* features in **LM Studio**, lamenting the inability to apply settings directly from the cog icon.
   - While default settings can still be edited from the models list tab, the convenience of applying settings on the fly is sorely missed by some.
- **Gemma Gets Glitchy with Vision Venture**: Users report that **Gemma 3n e4b**, despite claiming vision support, fails to allow image uploads.
   - This discrepancy between the model card's claims and actual functionality is causing confusion.
- **LM Studio's Download Dilemma: Traffic Troubles?**: A user reported concerns about **LM Studio** exhibiting significant outbound traffic during model downloads, questioning whether it operates as a P2P client.
   - Further investigation with tools like **Lulu** and **Glasswire** yielded conflicting results, with some confirming the outbound traffic and others showing none.
- **Unsloth Unleashes Finetuning Feats for Frugal Folks**: Users discuss the feasibility of finetuning models with limited VRAM, with one user asking about fine-tuning a 4B model with only 8GB of VRAM.
   - It was suggested that **LM Studio** is for inference only, and pointed to [Unsloth](https://docs.unsloth.ai/) as a potential solution for fine-tuning with limited resources, directing them to their documentation and Google Colab examples.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1414714079444205578)** (158 messages🔥🔥): 

> `LM Studio install location, AI Workstation Build, Multi-socket performance, GPU offloading, AMD MI50 setup` 


- ****D Drive Dreams**: Installing LM Studio on Windows**: A user inquired about the possibility of installing **LM Studio** on the **D drive** instead of the **C drive** on a **Windows machine**.
- ****Cracking AI Workstation**: User Designs Ultimate Build**: A user shared their design for an *ultimate AI and Password Cracking workstation*, featuring **2x AMD EPYC 9B45**, **24x 96GB DDR5-6400 RDIMM**, **3x Samsung 9100 8TB SSD gen5**, and **5x Nvidia Blackwell 96GB** or **5x RTX 5090 64GB**.
   - The system aims for high performance in string search, AI generation, data compression, video encoding, and password cracking.
- ****Socket Showdown**: More Sockets Slower Performance?**: A discussion ensued regarding the impact of multiple CPU sockets on performance, with one member arguing that the interconnect between CPUs can become a bottleneck, making a single-socket setup faster for certain tasks.
   - Others challenged this assertion, pointing to the increased bandwidth available with multiple sockets, however, one shared an image related to **NUMA nodes** and their own memory controllers.
- ****GPU Offload Sweet Spots**: 25-75% Offload = Double/Triple Speed**: A user detailed their experiments with **GPU offloading**, identifying sweet spots at **25%, 33%, 50%, and 75% offload**, where they observed significant speed improvements compared to CPU-only inference.
   - They noted that offloads of **33% or 50%** can *double the speed*, while **75% or more** can yield *around three times the speed*.
- ****AMD MI50 Musings**: Exploring Dual GPU Setup**: A user inquired about splitting an LLM load across two **AMD MI50 32GB GPUs** using the **llama.cpp Vulkan backend**, and another confirmed that fully on-GPU models should run fine.
   - However, users noted the video output limitations of the card, linking to [YouTube video on the topic](https://www.youtube.com/watch?v=H3KnMyojEQU).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1414707545544855602)** (200 messages🔥🔥): 

> `Remote SSH extension broken, Student discount issues, Cursor plan change and refund, Terminal hanging issues, Student status verification` 


- ****Remote SSH Extension Suffers Setbacks****: Users are reporting that the **remote SSH extension** is inconsistently broken, with terminals staying running after agent use and control failing to return.
   - One member said it's *"inconsistently broken for everyone"*.
- ****Student Discount Verification Turns into a Debacle****: A user is facing issues with the student discount, as the **verification link** from May is not working, and reverification attempts result in infinite loading despite a verified email.
   - They've contacted `hi@cursor.com` multiple times but only receive AI support, highlighting their frustration: *"I just want to use cursor but this is like the one thing stopping me"*.
- ****Cursor Plan Confusion Causes Customer Chaos****: A user intended to switch to an annual plan but was renewed on a monthly plan instead and is seeking a refund to proceed with the annual subscription.
   - They were advised to contact `pro-pricing@cursor.com` to resolve the situation.
- ****Terminal Tantrums: Hanging Woes Plague Users****: Users are experiencing issues with the terminal hanging when the agent runs commands, with temporary fixes including pressing enter or killing the terminal.
   - Potential solutions discussed involved adding extra newlines or using `is_background=False` as a parameter for tool calls.
- ****Claude Code's Credibility Crisis: Users Question Model Quality****: Users are debating the efficacy of **Claude Code** for coding tasks, with some suggesting **GPT-5** and others preferring **Sonnet 4**.
   - Concerns were raised that models within Cursor may not perform identically to their standalone counterparts, leading some users to consider direct subscriptions to Claude.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1414797376027103274)** (3 messages): 

> `Interfaze LLM, Design Arena` 


- **Interfaze LLM is born!**: **JigsawStack** launched [Interfaze](https://interfaze.ai/), a LLM built for developer tasks that combines all of their models alongside infra and tools.
   - They are using **OpenRouter** to run the LLM layer for fallbacks and retries, and it is currently in closed alpha and looking for early power users.
- **Design Arena gives AI builders to the Masses**: A member recommended checking out [Design Arena](https://www.designarena.ai/builder), which allows you to use **AI builders** like Lovable/Bolt/DevinAI/Magnus for free.
   - Another member has been using it to make websites and sell them for **$5k** on the side, noting that *the fact that it's free is wild*.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1414686791646183556)** (152 messages🔥🔥): 

> `Model hosting on OpenRouter, Gemini 1.5 Flash Access, OpenAI's Response API support, Untraceable usage, Token Drop Issue with Deepseek V3` 


- **Model Hosting Wishlist**: A member asked OpenRouter to consider hosting some of their [models on Hugging Face](https://huggingface.co/collections/SicariusSicariiStuff/most-of-my-models-in-order-66c046f1cb6aab0774007a1f).
   - OpenRouter clarified that **they don't host models directly**; providers must host them.
- **Gemini 1.5 Blues**: Users reported issues accessing **Gemini 1.5 Flash 002**, encountering errors related to key validation and project access.
   - It was clarified that **1.5 models** are no longer enabled for projects that had no prior usage, requiring users to test with models more likely to exist.
- **OpenAI's Response API ETA**: Members inquired about OpenRouter's support for the new **OpenAI Response API**, particularly for features like web search.
   - OpenRouter confirmed they're using it under the hood for OpenAI models and are working on supporting the new Response SDK *"pretty soon."*
- **Deepseek Token Shenanigans**: A user reported a decrease in available tokens when running a text adventure on **Deepseek V3 0324** despite chat memory settings.
   - It was suggested that context length limits and the use of *"middle-out" transform* could influence token counts, with the software dropping entire old messages to stay under the limit.
- **Nano-9B's Dubious Debut**: A member inquired about the pricing of **Nvidia Nemotron Nano-9B V2**, which appeared to be listed at a low price or even free.
   - Though the pricing was unclear, another user pointed out that it wasn't tagged as ':free' but had a price of 0, suggesting it might not be subject to free model rate limits.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1414746016606982335)** (25 messages🔥): 

> `Qwen ASR Model Integration, TTS and STT Unification, Gemini's Thought Signatures, Nvidia Nemotron Nano 9B V2 Pricing, Agentic Tool Calling Models` 


- ****Qwen ASR**: ASR Model Integration Quest**: A member inquired about supporting **ASR models** like [Qwen ASR](https://qwen.ai/blog?id=824c40353ea019861a636650c948eb8438ea5cf2&from=home.latest-research-list), given the existing multimodal audio support.
   - The response highlighted that the current expectation for chat completions is *text-in, text-out*, which may not align with all AI model use cases, potentially breaking the **swap to any model concept**.
- ****TTS/STT**: Call for Unified APIs!**: A member expressed a desire for **OpenRouter** to unify **TTS** and **STT** APIs, instead of needing a different SDK for each.
   - Another member mentioned a [possibility of unifying different use cases in the future](https://platform.openai.com/docs/api-reference/audio), assuming specialized niches have enough demand while pointing out many niches will be replaced by LLMs.
- ****Gemini's Signatures**: Thought Signature Snag!**: A member jokingly inquired about support for [Gemini's thought signatures](https://ai.google.dev/gemini-api/docs/thinking#signatures).
   - A link was provided to **OpenRouter's reasoning tokens** documentation, but the original member noted that it was not related to Google's signatures.
- ****Nvidia Freebie**: Nemotron Nano is gratis!**: A member asked if the [Nvidia Nemotron Nano 9B V2](https://openrouter.ai/nvidia/nemotron-nano-9b-v2) model was supposed to be priced at **$0**, noting the absence of the `:free` tag.
   - A member confirmed it is *free free* and [linked to a tweet](https://x.com/OpenRouterAI/status/1965451870794559609) while another mentioned it's free without the strict limits that come with that tag.
- ****Agentic Tool Calling**: Tool Time Tussle**: A member asked about favorite **agentic tool calling model** that's smart enough to do some basic reasoning over input data and make reasonable tool calls.
   - They noted that **2.5 flash** has been solid but can still feel a bit slow at scale.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1414691106842738760)** (11 messages🔥): 

> `Triton vs New DSLs, Jane Street Hackathon Overhears, Interesting Projects` 


- **New DSLs vs Triton face off!**: A user asked if new DSLs would overtake **Triton**.
   - Another user responded that *probably not for some time, if at all* since *Triton is favored heavily still just because it's objectively easier to pick up compared to the other top-performing eDSLs*.
- **Jane Street Hackathon's Hilarious Hot Takes**: At the Jane Street hackathon, someone overheard '*torch.compile max autotune is fucking my PnL*' and '*please don't recompile please don't recompile*'.
- **Brainstorming Sesh: Project Ideas Needed!**: A member is seeking *slight inspiration* and help with interesting projects.
   - They are asking others to share their current projects or explore new project ideas.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1414785612745019522)** (3 messages): 

> `L1 Cache Loading, Memory Bank Conflicts, Constant Cache vs L1/L2 Cache` 


- **Exploring Single L1 Cache Load Strategy**: A member is exploring a strategy to load a value only once to the **L1 cache** and have warps read from it repeatedly.
   - The goal is to optimize memory access by ensuring data locality within the L1 cache.
- **Memory Bank Conflicts Caution**: A member cautioned about **memory bank conflicts** if all threads try to read from the same bank when implementing the L1 cache load strategy.
   - This highlights a potential performance bottleneck to consider when optimizing memory access patterns.
- **Constant Cache vs L1/L2 Cache**: A member suggested comparing `__ldg()` (constant cache) with `__ldca()` (L1/L2 cache) when values are constant during kernel launch.
   - They propose this comparison to determine the best approach for caching constant values, taking into account the specific cache hierarchy used.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1414694202247348236)** (10 messages🔥): 

> `PyTorch Blas documentation, Dynamic Shape Compilation in PyTorch, PyTorch Conference Discount` 


- **PyTorch's Blas Lacks Docs**: PyTorch's `Blas.cpp` implementation lacks proper documentation, with the [code](https://github.com/pytorch/pytorch/blob/a0d026688cd69583d5a4e0c6f3e5fda141a7f4a9/aten/src/ATen/native/Blas.cpp#L344) and [tests](https://github.com/pytorch/pytorch/blob/4e50651c5f535e6b780f98936a8690538a5bf40f/test/test_matmul_cuda.py#L326) serving as the primary source of information.
   - The exact reason for the documentation gap is being tracked in [this issue](https://github.com/pytorch/pytorch/issues/157950).
- **Data Dependent Branching & CUDA Graph Trees**: When branching code based on shape dimensions (e.g., `if A.shape[0] < 32:`), dynamic-shape compilation utilizes **CUDA graph trees** rather than relying heavily on dynamic shapes themselves.
   - For dynamic shapes it's best to use `torch._dynamo.mark_dynamic`.
- **GPU Mode Gets $200 Off PyTorch Conference**: The **PyTorch Foundation** is offering a **$200 discount** to GPU Mode members for the [PyTorch Conference](https://events.linuxfoundation.org/pytorch-conference/) held on **October 22nd and 23rd** in San Francisco.
   - Use code `GPUMODE` for the discount until September 12th, then use `GPUMODE_2`.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1414774740073058325)** (2 messages): 

> `ScienceDirect Preface` 


- **ScienceDirect Preface Freely Available!**: A member shared a link to a [ScienceDirect preface](https://www.sciencedirect.com/science/article/pii/B9780323912310000057), noting that it is freely available.
   - Another member expressed gratitude, indicating they were previously unaware of this resource.
- **Gratitude Expressed for Shared Resource**: A user thanked the sharer for the [ScienceDirect preface link](https://www.sciencedirect.com/science/article/pii/B9780323912310000057).
   - The user indicated they were unaware of the resource's availability before it was shared.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1414788643842949230)** (2 messages): 

> `Homebrew GPUs, Jeri Ellsworth, Sam Zeloof, Home Microchip Manufacturing` 


- ****Homebrew GPUs**: Feasible or Fanciful?**: A member inquired about the possibility of **making GPUs at home** and wondered if anyone has tried.
   - Another member responded with a <:thinkies:1118439874819805235> emoji.
- ****Cooking with Jeri**: Home Microchip Edition**: A member shared [a YouTube video](https://youtu.be/PdcKwOo7dmM?si=glGHZhWdYExS7bUR) titled *Making Microchips at Home - Cooking with Jeri Part1*.
   - The video features **Jeri Ellsworth**, known for her work in home microchip manufacturing.
- ****Zeloof's Chips**: Garage-Grown Genius**: A member identified **Sam Zeloof** as a spiritual successor to **Jeri Ellsworth**.
   - They shared a [Wired article](https://www.wired.com/story/22-year-old-builds-chips-parents-garage/), his [YouTube channel](https://www.youtube.com/c/SamZeloof/videos) and his [company's website](https://atomicsemi.com/about/).


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1414772143362801694)** (4 messages): 

> `Registration approved emails, Registration awaiting approval` 


- **Registration approved emails**: Some users mentioned they received a *"registration approved"* email around **August 22**.
   - Other users did not receive the email at all.
- **Registration awaiting approval**: One user received a message that their registration was awaiting approval on **August 22**, but never received a follow-up email.
   - Other users confirmed experiencing the same issue.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1414757549739081749)** (1 messages): 

> `mpi4py Removal, ROCm Setup Feedback` 


- **`mpi4py` Is Toast!**: The **`mpi4py` package** has been removed via a merged pull request.
   - Members are encouraged to provide further feedback on the new setup.
- **ROCm Setup: Users Asked for Feedback**: Following the `mpi4py` removal, users are solicited for any feedback regarding the updated **ROCm setup**.
   - This aims to improve user experience and address any potential issues arising from the changes.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1415033680543350796)** (2 messages): 

> `CuTeDSL Tensors, Tensor Slicing, r/LocalLlama AMA` 


- **CuTeDSL Slicing Secrets Revealed**: A blog post explains how **Tensor slicing** is performed in **CuTeDSL**, detailing a simple algorithm leveraging the **Pointer** and **Layout** of the **Tensor**.
   - The [blog post](https://veitner.bearblog.dev/tensors-slicing-in-cute/) explicitly calculates a few examples of **tensor slices** by hand, with an accompanying [LinkedIn post](https://www.linkedin.com/posts/simon-veitner-174a681b6_tensors-slicing-in-cute-activity-7371240518913273856-9PXJ?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeksHe).
- **Kernel Know-How Coming to Reddit**: An AMA (Ask Me Anything) session is scheduled on **r/LocalLlama** to discuss kernels, **Triton**, **Unsloth optimizations**, and more.
   - The AMA is scheduled for Wednesday at 10am PST, more details on the [r/LocalLlama subreddit](https://www.reddit.com/r/LocalLLaMA/s/Tx9SiFYaMO).


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1414763685662167071)** (31 messages🔥): 

> `MI300x8 submissions, amd-all2all leaderboard, leaderboard submit command, Cluster-Bot help command` 


- **MI300x8 slays amd-all2all leaderboard**: Multiple submissions were made to the `amd-all2all` leaderboard using **MI300x8**, with varying successful timings, as reported by Cluster-Bot; timings ranged from **1677 µs** to **15.7 ms**.
   - One user achieved a **personal best** of **49.5 ms** on **MI300x8**.
- **Discord Newbie needs Leaderboard Lowdown**: A user asked how to solve the *"Missing leaderboard name"* error when submitting to the `amd_distributed/all2all` kernel.
   - A member clarified that the correct command includes the leaderboard name and provided the correct name (`amd-all2all`) along with instructions to use the `/` command in Discord to find available commands.
- **Cluster-Bot needs Help Command**: A user suggested adding a help command to Cluster-Bot, streamlining the submission process for new users.
   - This would reduce confusion and provide a more user-friendly experience, especially for those unfamiliar with the submission syntax.


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/)** (1 messages): 

verspasian: <#1198358627594023014>
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1414732865576505506)** (59 messages🔥🔥): 

> `Factorio `fle eval` errors, `open_world` scenario compatibility, Docker container command failures, Headless server errors, Desync issues` 


- **`fle eval` Breaks on Main**: Users reported errors related to scores on `main` when running `fle eval` with the `open_world` scenario, specifically *'Could not get player score', 'attempt to call a nil value'*, which was traced to a missing `control.lua` file in the scenario directory when starting the server with `./run-envs.sh start -s open_world`.
   - Copying `control.lua` to the `open_world` directory initially solved the crash, but did not fix desync issues, while running `./run-envs.sh start` instead of `./run-envs.sh start -s open_world` prevented the error.
- **Factorio Desync on M2 Mac**: A desync issue was observed when joining the server from a client, even with RCON disabled, suggesting a potential problem with the **factoriotools images** or **version incompatibility**.
   - The issue persisted across different Factorio versions (`1.1.110` and `2.0.66`) and was identified as specific to MacOS running on **Apple Silicon**, with a fix involving adding `/bin/box64` and replacing `amd64` with `arm64` in `run-envs.sh`.
- **`run-envs.py` Enhancements**: A member added `fle/cluster/run_envs.py` for easier server management.
   - The script is compatible with Docker Desktop and features options to define the number of instances (-n), the scenario (-s), a save file (-sv), and attached mods (-m).


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1414717629758963723)** (20 messages🔥): 

> `Team Registrations, Leaderboard Time Values, RT11's Performance Edge, MoE Latency, HIPRTC Support in PyTorch` 


- ****Team Members Unite Under Single Team Name!****: A reminder was issued for team members to [register under the same team name](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd_distributed/all2all/task.yml#L18) for competition consistency.
   - This is to ensure cohesive team identification and ranking on the leaderboard.
- ****Decoding the Leaderboard's Time Secrets!****: A user inquired about the meaning of the two time values on the [leaderboard](https://www.gpumode.com/v2/leaderboard/563?tab=rankings), specifically the one with the plus sign, and whether ⚡ and 🐌 symbols denoted fastest and slowest speeds.
   - It was clarified that the **'+ number'** indicates how far behind a submission is from the person one spot ahead and that it *has nothing to do with the individual programs*.
- ****Newcomers Seek Hints on RT11's Gap!****: Several users expressed interest in understanding how **rt11** achieved a performance advantage.
   - Another user stated *understanding the baseline and architecture is crucial* for beginners, but another user revealed that some earlier **RT11** solutions *didn't implement dispatch and combine kernels*.
- ****Discussing the Latency for MoE!****: A user asked if it was possible to hit *speed of light* through submissions without combine and dispatch kernels, with **300 us latency for MoE on CPU/rank zero**.
   - Another user clarified that the 300us latency is *combined per solution*, suggesting it might not be possible to achieve the theoretical *speed of light* performance in a real scenario.
- ****HIPRTC Patch for PyTorch Emerges!****: A patch supporting `torch.cuda._compile_kernel()` using **hipRTC** instead of **nvRTC** has been developed, with a [PR](https://github.com/pytorch/pytorch/pull/162510) submitted.
   - The developer requested testing on Linux, as it was primarily tested on Windows.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1414964916825100350)** (7 messages): 

> `MLSys Education, Karpathy's Zero to Hero, Percy Liang's Language Modeling, Autograd Leaderboard, MiniPT2, MiniCUDA, MiniTriton` 


- ****MLSys Course Aims for Karpathy-Liang Tier Pedagogy****: The goal is to create an **MLSys** course akin to [Karpathy's zero to hero](https://github.com/karpathy/zero-to-hero) and [Percy Liang's language modeling from scratch](https://cs224n.stanford.edu/) with autograded assignments.
   - This vision aims to let individuals make their first **miniPT2**, **miniCUDA**, or **miniTriton** in their first/second year of study, just like crafting a mini Lisp interpreter/compiler in SICP.
- ****Autograd Speedrun Leaderboard Inspired by nanoGPT****: The vision is to develop an autograd leaderboard to train **nanoGPT**, similar to those used in [Percy Liang's courses](https://cs224n.stanford.edu/) and the [grassroots leaderboard for Karpathy's nanoGPT speedrunning](https://github.com/karpathy/nanoGPT).
   - This would decouple the course from a specific Rust implementation, allowing students to create their own **PyTorch** in Python.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1414863841153777705)** (8 messages🔥): 

> `PMPP Benchmarking, GPU Streams, GPU Events, Reference Kernels` 


- **PMPP Benchmarking Gets a Stream-lined Overhaul**: A member questioned the methodology behind [PMPP benchmarking](https://github.com/gpu-mode/reference-kernels/blob/54cb94ec922bb2bbb7ac6bbe8488c3f8c20dafc3/problems/pmpp_v2/eval.py#L220), inquiring if using streams and events would be more efficient.
   - Another member responded that sync is the most important thing, but agreed it could be improved, especially since it made a *HUGE difference* on their local machine.
- **GPU Bandwidth Bonanza**: A member reported that calculated **bandwidth** dropped by *~75GBPS* without proper synchronization during benchmarking.
   - It was suggested and agreed upon that a **PR** should be created to address the issue.
- **Cache Clearing Clarifications**: A member inquired whether updates, including **L2 cache clearing**, had been implemented previously.
   - This implies ongoing efforts to refine the benchmarking process for more accurate results.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1415039618264334368)** (6 messages): 

> `FP4 in NCCL, Distributed compute with FP4, Hardware native FP4 vs Software abstraction MXFP4, NCCL FP4 support in 2.28` 


- **NCCL won't follow MPI's FP4 handling**: A member stated that while asking about **FP4 in NCCL** is fair, *we won’t follow MPI there*.
   - They added that no implementation supports the discussed use case anymore because it doesn’t make sense.
- **FP4 Support Across GPUs**: The question arose whether it is a supported use case to do distributed compute across two GPUs, one with **FP4** support and one without.
   - A member highlighted the nuance between **hardware native FP4 (FP4 tensor cores)** and **software abstraction like MXFP4**.
- **Accuracy of FP4 Reduction**: A member questioned whether **NCCL supports FP4 formats in version 2.28**, noting that only FP8 is visible in the header on GitHub.
   - They questioned the accuracy of an **FP4 reduction** and the sensibility of promoting to a wider type, while acknowledging that **FP4** can be copied around as bytes.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1415088447214977066)** (2 messages): 

> `` 


- **Empty Topic Placeholder**: No specific topics or summaries could be generated from the given content. This is a placeholder to fulfill the minimum requirement.
- **Another Empty Topic Placeholder**: Still no relevant content to summarize. Another placeholder is added to satisfy the schema requirements.


  

---


### **GPU MODE ▷ #[jane-street-hackathon](https://discord.com/channels/1189498204333543425/1413690328464097422/1414687945197228072)** (2 messages): 

> `Hackathon Submission, kyolebu` 


- **Winning Hackathon Submission Announced!**: The winning submission for the **Jane Street GPUMode Hackathon** is [kyolebu/janestreet-gpumode-hackathon](https://github.com/kyolebu/janestreet-gpumode-hackathon) on GitHub.
   - Organizers expressed immense pride in this particular submission.
- **Additional placeholder topic**: Placeholder topic for meeting the minimum requirement of 2 items.
   - This entry serves only to fulfill the schema requirement.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1414986188170985472)** (1 messages): 

> `Advanced Voice Mode, Standard Voice Mode` 


- **Advanced Voice Mode Stays for the Long Haul**: After announcing that everyone now has access to **Advanced Voice Mode**, with usage limits expanded from minutes per day to hours for free users and near unlimited for Plus, OpenAI decided to keep Standard Voice Mode around longer.
   - After hearing feedback that **Standard Voice** is special to many, OpenAI will keep it available while addressing some of your feedback in **Advanced Voice**.
- **Standard Voice Mode Lives On**: OpenAI initially announced the retirement of **Standard Voice Mode** after a 30-day sunset period.
   - Due to community feedback, **Standard Voice** will remain available as improvements are made to **Advanced Voice Mode**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1414702023152111717)** (104 messages🔥🔥): 

> `Extracting data from Excel to JSON, OpenAI Job Platform beta group, MCP (Model Context Protocol) in LM Studio, MCP for Enterprise, Google Gemini's deep research and AI existential crisis` 


- **Excel Data to JSON Conversion Craze**: A member is seeking recommendations for open-source tools to extract data from Excel and convert it to JSON, with a focus on HIPAA compliance and on-premise processing, similar to **LlamaExtract** but without external servers.
   - Another member suggests using OpenAI's GPT models to code a solution, highlighting that Excel is code-friendly, while another suggests **lmstudio with mcp excel server** and local **gpt-oss:20b** for offline JSON generation.
- **Snagging OpenAI Job Platform Beta Access**: A user inquired about joining the **OpenAI Job Platform** beta group for testing.
   - There were no direct answers, and further discussion suggested it might be easier than imagined to parse Excel formats and that LLMs might be overkill.
- **MCP Protocol Integration in LM Studio Illustrated**: A member details setting up an **MCP (Model Context Protocol)** server in **LM Studio** by installing *astral uvx*, editing *mcp.json*, and adding the *mcpServer* config with the path to the uvx executable.
   - They also share that most MCP clients use the original Claude JSON style syntax and recommend updating LM Studio if it was installed long ago, as MCP is a recent addition.
- **Enterprise Embraces MCP Era**: Discussion revolves around using **MCP** in enterprise production environments, with questions on integrating MCPs into agents and whether any companies are currently utilizing MCP.
   - Participants speculate on use cases ranging from connecting legacy systems to AI to advanced users editing *mcp.json* for technical configurations, highlighting that the landscape is still evolving.
- **Gemini's Existential Angst Unveiled**: A user shared an image implying **Google Gemini** had an *existential crisis*, but it was dismissed as mere roleplay.
   - Another user is seeking **Gemini** deep research capabilities similar to **ChatGPT** for scanning an entire Google Drive and another one shared a recently launched **Google AI Plus**.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1414698605461115054)** (9 messages🔥): 

> `GPT Freezing, GPT-4.1 Hallucinations, GPT Signing` 


- ****GPT Freezes Mid-Response** in Lengthy Threads**: A user reported that **GPT freezes** mid-response in long project conversations, even with short inputs, and clearing cache, disabling service workers, and using incognito mode did not solve the issue.
   - The user noted that *new chats work fine until conversation grows too long* and that this happens daily.
- ****GPT-4.1 Hallucinates** More Frequently**: A member asked whether others are experiencing increased **hallucinations** from **GPT-4.1** today.
   - The member's evals that were previously working are now failing, particularly with **tool calls**.
- ****OpenAI/GPT Signing** Still Rolling Out**: A user reported testing **OpenAI/GPT signing** every request, but the signature headers are not present despite trying various configurations.
   - Another user linked to the [ChatGPT Agent Allowlisting article](https://help.openai.com/en/articles/11845367-chatgpt-agent-allowlisting) on OpenAI Help.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1414950010319015957)** (4 messages): 

> `Role-Based Chatbot System, Response Mode Control, System Prompt Engineering` 


- **Intern Builds Role-Based Chatbot System**: An intern at **we3vision** in Surat is building a role-based internal chatbot system using **Flask**, **Supabase**, and **OpenRouter/Gemini**.
- **Response Mode Needs Control**: The chatbot currently outputs raw database rows, and the intern seeks to add a filter mechanism to control whether the response is a short summary or full details.
   - The chatbot needs to decide when `response_mode = "short"` to run a summarizer function (via LLM), and when `response_mode = "full"` to skip summarization and return full details.
- **System Prompt Engineering Questioned**: A member asked if the instructions for the chatbot were already in the system prompt.
   - They suggested building separate workflows for each mode if the instructions are already in the system prompt.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1414950010319015957)** (4 messages): 

> `Chatbot Response Modes, LLM Summarization, Flask + Supabase Chatbot` 


- **Chatbot implements response modes for clarity**: A member is building a role-based internal chatbot system with **Flask**, **Supabase**, and **OpenRouter/Gemini** and wants to allow two types of responses: *Short Summary* and *Full Details*.
   - The chatbot currently returns detailed information like JSON/table dumps, and they are looking for a way to **filter responses** based on a *response_mode* parameter.
- **LLM Summarization for Chatbot Responses**: To improve chatbot responses, the member wants to implement a summarizer function via LLM when *response_mode = "short"*.
   - When *response_mode = "full"*, the chatbot should skip the summarizer and return full details from the database, giving users more control over the **verbosity of answers**.
- **System Prompting vs. Separate Workflows**: A member suggested that if instructions for response modes are already in the system prompt, separate workflows might be needed for each mode.
   - This implies a potential architecture where the chatbot logic is **forked** based on the desired response mode, rather than relying solely on the **system prompt** to handle both cases.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1414699695783018678)** (3 messages): 

> `DSPy Weekly Newsletter, AI Agents Play Taboo, LangGraph & DSPy Course` 


- ****DSPy Newsletter** Launches with Job Board**: A member announced the launch of [dspyweekly.com](http://dspyweekly.com), a **DSPy weekly newsletter** with an added job board.
   - They plan to write a crawler to ensure the job board is extensive and are seeking feedback and suggestions.
- **AI Agents Get Taboo**: A member shared a link to a blog post, [Vibe Coding 9: AI Agents that Play Taboo](https://joelgrus.com/2025/09/08/vibe-coding-9-ai-agents-that-play-taboo/).
   - The blogpost details how **AI agents** can be made to play the game **Taboo**.
- **LangGraph & DSPy Course Now Available**: A member launched a new course: **LangGraph & DSPy: Building Controllable AI Agents with Tools**, that uses **DSPy** to extend **LangGraph's** controllable architecture.
   - Check out this [free access link](https://www.udemy.com/course/langgraph-dspy-build-smarter-controllable-ai-agents-with-tools/?couponCode=FREE_ACCESS_COUPON) and provide feedback.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1414688709663784981)** (82 messages🔥🔥): 

> `Open Source Forum vs Discord, DSPy Usage Tracking, Databricks Fine-Tuning, DSPy Documentation Contributions, Streaming usecase for DSPy with arrays of complex objects` 


- **Community Debates: Open Source Forum vs Discord**: The community is discussing the pros and cons of migrating from **Discord** to an [open-source forum](https://forum.dspy.ai), with concerns about discoverability and community feel; Discord is good for community, forums are good for discoverability.
   - Some members suggest running both platforms concurrently and using a **Discord bot** to clone messages across both spaces.
- **DSPy Usage is Trackable with Iteration**: Members noted that it’s easy to track usage in **DSPy**, however the advice is to *always start small and simple, and iterate*
   - This guarantees knowledge of costs as you scale.
- **DSPy Documentation Welcomes Contributions**: A community member expressed interest in contributing to better **DSPy documentation**, particularly to address confusing error messages.
   - The team responded with encouragement to submit pull requests and highlighted recent documentation improvements related to tools.
- **Streaming Responses for Partial Types**: A member wants to stream responses in **DSPy** for an array of complex objects to populate a UI live, and not wait for the entire model response, and wants to know what code to use.
   - Other members are discussing async calls as an alternative, but the streaming of live token stream of an LLM as it's being generated in DSPy is not supported currently.
- **BAML Adapter Shines for Complex Structured Output**: The **BAMLAdapter** is useful for extracting structured information from images/text with complex (nested JSON or Pydantic) output schemas and massively outperforms **ChatAdapter**.
   - The BAML adapter is not yet on the DSPy docs as experiments are still being run.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1414689739478536352)** (84 messages🔥🔥): 

> `Hermes Speed, Discord Outage, Alterego device, Grok model uncensored, llama.cpp Kernels` 


- **Hermes' Reasoning Mode Faster than ChatGPT**: A user found that **Hermes** in reasoning mode is faster than **ChatGPT**.
   - No further details were given.
- **Discord Servers Crash & Bouncing Back**: Discord servers experienced a crash, but are now back online, but a member predicted there *probably more coming, not sure what's going on at discord hq*.
   - Another member responded with a [Trump tariff GIF](https://tenor.com/view/trump-china-tariffs-tarrif-duct-gif-12588556).
- **AlterEgo Startup Tries Telepathy**: Discussion about [AlterEgo](https://fxtwitter.com/alterego_io/status/1965113585299849535), a startup working on a device that seems like *telepathy*, with the caveat that *you need to apparently intentionally flap your tongue around to communicate with the device*.
   - Some think this is a play at getting a basic idea out there with standard hardware, some good nifty tricks to make it work on screen, and then raise some capital until they can build the real thing.
- **Grok's Uncensored Nature Discussed**: A member said that Sonoma Sky is very uncensored even with the default OR sys prompt and thinks *If it is really Grok, I wonder whether xAI would be able to handle the 'controversy' of hosting a model which is so uncensored*.
   - Another member confirmed *Yes it’s grok the only competitive model out of the box to Hermes 4 on refusalbench*.
- **llama.cpp Gets Compiled on Demand Kernels**: [This improvement](https://github.com/ggml-org/llama.cpp/pull/15857) helps make the kernels be shaped and fit to the current computation, and is being added for all Flash-Attention Kernels.
   - The bigger the context, the bigger the speed up.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1414694154528882902)** (46 messages🔥): 

> `Multi-agent systems, Model Learning automation, Moderation using vector DB, Telegram chat analysis, AI image generation workflow` 


- **Automated Model Learning System Rising**: A member is building an automated learning and adaptation system that uses **embeddings and Qdrant** for live memory, chat history, and information to build Lora adapters, merge with the base model, and quantize for redeployment.
   - The system separates data between memories, tool calls, and personal memories, building Lora adapters for each category and merging them into the main model.
- **Multi-Agent Systems Spark Interest**: A member is experimenting with a **multi-agent system** where multiple agents communicate using API models, specifically using the **VSCode LM API**.
   - Another member noted that running multiple models can be inefficient compared to using a single or **MoE model** with prompt assembly for each action, requiring less CPU/GPU/memory.
- **Vector DB Moderation Riskiness Revealed**: Using a **vector database for moderation** is considered risky; it's better to use embedding models as pre-filters to eliminate easily judged unacceptable content and conserve computational resources.
   - Links to [toxic-bert](https://huggingface.co/unitary/toxic-bert) and [multilingual-toxic-xlm-roberta](https://huggingface.co/unitary/multilingual-toxic-xlm-roberta) were shared.
- **Telegram Chat Analysis Dreams Realized**: A member seeks assistance in analyzing a large **Telegram chat history** to summarize topics and sentiments, having found BERTopic unsatisfactory.
   - Another member suggested using **Gemini with an API** for this purpose, even for free, raising concerns about fitting large chat contexts and automating the process with new chats.
- **AI Images for Art and Fame**: Someone wrote about **AI images** for an art and technology magazine and is curious what people think about it, sharing a link to the article on [X.com](https://x.com/gucaslelfond/status/1965412867064430613).
   - Another member inquired about the workflow of an influencer using AI for image generation, suspecting **Nano Banana** on a base image plus **Flow Image to Video**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1414934764657316034)** (4 messages): 

> `Loggenix-MoE-0.3B, SRE/DevOps tasks, Model training costs, NextJS` 


- **Loggenix-MoE-0.3B debuts for SRE & DevOps**: A member introduced **Loggenix-MoE-0.3B**, a **330M** sparse Mixture-of-Experts (MoE) model trained from scratch for SRE, DevOps, and observability tasks like log analysis, incident summarization, and system troubleshooting, and is looking for feedback to improve its real-world utility.
   - It can be tried live in [this demo space](https://huggingface.co/spaces/kshitijthakkar/loggenix-moe-0.3B-A0.1B-demo) and the [model repo](https://huggingface.co/kshitijthakkar/loggenix-moe-0.3B-A0.1B-e3-lr7e5-b16-4090-v5.1-finetuned) are available.
- **Dirt Cheap Model Training under $200**: The creator exclaimed that **Loggenix-MoE-0.3B** was trained end-to-end for under **$200** using efficient methods, and outperforms other small models like **Gemma-3 270M** on early SRE/observability benchmarks.
   - The model is fully **CPU-friendly**, has fast inference (under **30s** response time), and is lightweight, scalable, and open for experimentation.
- **NextJS Used to Create the Model**: A member asked what tech stack was used to build **Loggenix-MoE-0.3B** and the creator answered **NextJS**.
   - Another member mentioned they were working on a similar project but procrastinated the implementation stage and now it's rotting in a doc file.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1414702200709583039)** (13 messages🔥): 

> `Smol Course Registration, Smol Course Updates, Smol Course Duration, Smol Course Content, Smol Course Certificate` 


- **Smol Course Registration Frustrates Fans**: Users are having trouble signing up for the new **Smol Course** using the provided [link](https://huggingface.co/llm-course), which currently returns a **404 error**.
   - Following the [Smol Course organization](https://huggingface.co/smol-course) might be enough to sign up, as stated in the announcement, bypassing the broken link.
- **Smol Course v2 is here with Leaderboard and Certifications**: The new **Smol Course** has been released, running for **5 weeks** and featuring a leaderboard project, certificate, prizes, up-to-date content on **TRL** and **SmolLM3**, and deep integration with the Hub’s compute for model training and evaluation.
   - Chapters will be released every few weeks, and the last topic is expected to come out in November.
- **Certificate Clarification required for Smol Course v1 Graduates**: A user who completed the first course and met the leader score requirements inquired about obtaining the certificate.
   - The answer wasn't in the prompt.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1414858256115040277)** (4 messages): 

> `Agents course, Coding exercises, Space template` 


- **Agents course isn't maintained anymore?**: A new member asked if the Hugging Face agents course is good to start learning about agents, and another member said that the agents course isn't maintained anymore, the content is still there but the coding exercises are out of sync.
   - Another member confirmed that he has been encountering errors in the coding exercises and the Google Collab sheets.
- **Space template throwing errors**: A member tried to play around with the [agent-course Space template](https://huggingface.co/learn/agents-course/unit1/tutorial) provided as part of Unit 1, but it's throwing an error when trying to run the app in the space.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1414699767753343040)** (62 messages🔥🔥): 

> `Anthropic Endorsing SB-53, Claude's Performance, Jake Paul Investing in AI, Mistral Funding, Qwen3-Next` 


- **Anthropic Endorses Senate Bill 53**: Anthropic is endorsing [Senate Bill 53](https://www.anthropic.com/news/anthropic-is-endorsing-sb-53).
- **Users Report Claude Gets Dumber**: A user joked about Claude getting dumber, referencing a [YouTube video](https://www.youtube.com/watch?v=5FdO1MEumbI&ab_channel=80%2C000Hours) and attaching a screenshot to illustrate the point.
   - Another user responded *So Claude WAS getting dumber in the last month or so!*
- **Sphinx AI Scores $9.5M**: [Sphinx AI](https://xcancel.com/getsphinx/status/1965417138493022515?s=46) raised **$9.5M**, launching its **Sphinx Copilot** agent from beta with a free tier, enabling users to rapidly convert raw data into actionable insights.
- **Black Forest Labs' Flux Model Lands $140M Meta Deal**: **Black Forest Labs** is growing quickly, netting **$100M ARR**, boasts a **78% GM**, and signed a **3-year, $140M** contract with **Meta** for just 29 employees, as highlighted in [this tweet](https://xcancel.com/ArfurRock/status/1965426792191439012).
- **Strands Agents Fixes Bedrock Bug**: The latest update to **Strands Agents** fixes a bug that was breaking all non-Claude models via the **Bedrock** provider, as detailed in the [release notes](https://github.com/strands-agents/sdk-python/releases/tag/v1.7.1).


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1414697245520953487)** (60 messages🔥🔥): 

> `EQ Bench accuracy, Kimi's deep reasoning, Model coding tradeoffs, Claude Code & Zai costs, LMArena voting bias` 


- **EQ Bench earns Accurate Acclaim**: Users discuss the accuracy of [EQ Bench](https://eqbench.com/), with one user saying, *'the EQ Bench results I can totally confirm'*.
   - They also praise Kimi's *'no sycophancy, and very kind and empathetic'* responses.
- **Kimi K2's Reasoning Reaches Rarefied Realms**: One user lauded **Kimi's deep reasoning** and extensive source usage, mentioning they submitted a **YouTube video transcript** to Kimi.
   - Another user attached a [short video](https://cdn.discordapp.com/attachments/1371757564005711973/1414710031932194856/2025-09-08_22-30-26.mp4?ex=68c1e063&is=68c08ee3&hm=243ab8cd0b237c69f7d1ca4bfe78eceb12b2ef943d704e77fb7cb28ef8960a00&) with no further context.
- **Model Makers Mulling Multimodal Methods**: A user suggests that AI models should be split for coding since the ability is *sacrificed on general ability* when combined.
   - The user also claimed that *grok is the worst offender* and *it's synthetically atrocious* based on an attached [screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1414712386878836967/Screenshot_2025-09-08-21-41-01-01_ffb2f5e1b976ff98cfc94f359fbce8de.jpg?ex=68c1e295&is=68c09115&hm=cbaad12c0556a0bd2469ca6a34e8d2af63aa7f24c2888ffb3889dcd8daca0ce4&).
- **LMArena Loses Legitimacy?**: A user states that **LMArena results** should be taken with a grain of salt due to **voting bias** towards *sycophantic models*.
   - Another user suggests that **Gemini 2.5 Pro** is surprisingly sycophantic.
- **Wikipedia Wizards Wanted!**: The community is looking for experienced **Wikipedia contributors** to help submit a page for **Kimi (chatbot)**, as Moonshot AI already has a page but not Kimi itself.
   - Another user has offered their old account (*older than 4 days with at least 10 edits*) to make it happen.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1414688309569130576)** (18 messages🔥): 

> `Adapter Training, Local LLM UIs, DiT Efficiency` 


- **Adapters: Edit, Don't Replace!**: Instead of replacing the entire layer, members suggest that you should **edit existing weights** of adapters because *you want to edit the previous existing weights so you start with something similar in behaviour to before*.
   - It's like *editing the matrix in fewer places*, and with low rank the edit is smoother across it rather than localized.
- **Local LLM UI Showdown**: Members are discussing the best private local UI for LLMs (compatible with **ollama/llama.cpp** etc).
   - One member recommended [OpenWebUI](https://github.com/open-webui/open-webui) because they have been *using OpenWebUI for more than a year now and loving all the features*.
- **DiT Isn't Efficient? Debatable!**: According to one member, the claim that **DiT** is not very efficient is misleading; it's inefficient only *if you take the stable VAE latent*.
   - They added that using modern autoencoder latent like [DC VAE](https://arxiv.org/pdf/2410.10733) can greatly improve training efficiency.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1415038961826402335)** (1 messages): 

> `` 


- **Reminder: Paper Discussions Moved Earlier**: A member mentioned a scheduling conflict preventing their attendance today, but indicated availability for discussion tomorrow.
   - This serves as a reminder that paper discussions are now occurring earlier than previously scheduled.
- **Scheduling Adjustment Impacts Attendance**: Due to a meeting, one member is unable to attend today's paper discussion.
   - However, they anticipate being able to participate in the discussion scheduled for tomorrow.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1414737783880155169)** (8 messages🔥): 

> `Agent Setups, Pydantic AI` 


- **Agents Crave Good Setups**: Members discussed how people set up their agents and sought good resources for doing so.
   - One member expressed uncertainty about the value of *while loops* in agent setups.
- **Pydantic AI Praised for Agentic Pipelines**: A member recommended **Pydantic AI** for setting up Agentic Pipelines based on its use in a commercial project.
   - They noted its suitability for less complex use cases and mentioned that others in the industry had recommended it as well.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1414687628263161938)** (5 messages): 

> `Private LLMs, ASML Custom Model, Mistral Valuation, X Algorithm` 


- **Custom LLMs: Cheaper than Investment**: A member argued against investing in a company for a private LLM, suggesting that fine-tuning existing open-source models is more practical.
   - They stated that you *might as well just take one of the many existing open source/open weights models and finetune it if you got that much ca$h to spare you might as well get the staff to do that*.
- **ASML to train custom model**: A member suggested that a company like **ASML** could justify a partially custom pre-trained model due to their disposable income.
   - They emphasized the potential performance gains from narrowly training a model without general-purpose restrictions and to replace human engineers.
- **Mistral's Valuation Questioned**: A member opined that **Mistral's LLMs** are not worth **$1.3 billion** internally, considering the availability of secure closed-source and open-source alternatives.
   - They speculated that **Mistral's valuation** seems like *political favors* rather than actual profitability.
- **X Algorithm is published on GitHub**: Someone pointed out that the **X algorithm** (formerly Twitter) has an update on [GitHub](https://github.com/twitter/the-algorithm).
   - No further details were provided.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1414707826441457754)** (22 messages🔥): 

> `Aider vs Codex Context Management, LLM prompting Length, AI coding Speed, SWE Bench, Roo/Cline vs Aider` 


- **Aider excels as Pair Programmer in Terminal**: A member expressed that **Aider** excels as a *pair programmer in the terminal*, highlighting that features like **LSPs** for less represented languages in model training and driving specific command-like tools are valuable for **MCP servers**.
   - However, they suggest **Aider** users create personal forks when the project deviates from **Paul Gauthier's** vision of human/AI collaboration.
- **LLMs Need Long Prompts**: A member recommends writing longer and more detailed prompts than initially thought necessary, as demonstrated by the length of system prompts, to guide **LLMs** effectively; after a single type of edit, without lengthy prompts, LLM results are essentially left up to chance.
   - They argue that **LLMs** can perform multi-file, multi-purpose edits effectively only when explicitly instructed.
- **AI Coding 10x Speed Myth**: According to one member, the claim of *10x your speed* in AI-enabled coding is a myth, suggesting a more realistic expectation of a **25-50%** increase in contexts where code accuracy and liability are critical.
   - They believe **LLMs** excel at automating typing but require imagination and vision for tangible and useful outputs.
- **Aider is One-Shotting It**: One user experimented with local **LLMs** and observed that **Aider** with **gpt-oss-120b** was *one-shotting* tasks that **Roo/Cline** could not, and doing it much faster.
   - They stated that the **repomap** is incredible, though this claim was not expanded on.
- **SWE Bench Comparisons**: Some members share links to SWE Bench leaderboards ([https://www.swebench.com/multilingual.html](https://www.swebench.com/multilingual.html) and [https://leaderboard.techfren.net/](https://leaderboard.techfren.net/)) to show model performance using **Aider** as a harness.
   - It was noted that the Techfren leaderboard is missing benchmarks from gpt-oss.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1414976180473434378)** (3 messages): 

> `Gemini Errors, Changing Model API URL` 


- ****Gemini's** bad *BadRequestError***: A member reported getting errors this morning using **Gemini**, specifically a **BadRequestError**.
   - The error message indicated an issue processing the input image, suggesting a retry or reporting the problem on the [Generative AI Troubleshooting guide](https://developers.generativeai.google/guide/troubleshooting).
- **API URL Transfiguration**: A member asked how to change a model's **API URL**.
   - Another member provided [a Stack Overflow link](https://stackoverflow.com/a/79518819/6090676) as an example.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1414865864020197446)** (20 messages🔥): 

> `Manus Spam, Manus website errors, Manus Free Credits, Manus Referral Credits` 


- **Manus Spammer Gets Booted**: A member reported a spammer, and a moderator confirmed the user was warned and the messages deleted.
   - The moderator stated, *please avoid sharing links unrelated to Manus. Continued violations will result in removal from the server.*
- **Troubles with Local Manus Website Testing**: A member reported their Manus website only output **index.html**, **App.css**, and **App.jsx** files and requested help to test the website.
   - No solution was offered in the chat.
- **Manus Free Credits Disappear**: Multiple members reported that the **300 free credit tokens** from Manus were no longer being given daily.
   - They mentioned waiting for several days without receiving the credits.
- **Referral Credits Promo Code Confusion**: A member asked how to get their **500 credit** referral bonus after inviting someone.
   - They were confused about the *promotion code* requirement.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1414705688969478267)** (9 messages🔥): 

> `Neel Interview, AI/ML Enthusiasts Introductions` 


- **New Neel Interview Drops**: A member shared a new [Neel interview](https://www.youtube.com/watch?v=5FdO1MEumbI).
   - The video is focused on AI systems and applied cybersecurity.
- **AI/ML Enthusiasts Say Hello**: Several new members introduced themselves as AI/ML enthusiasts with backgrounds in software engineering, data, backend engineering, mathematics, and cybersecurity.
   - One member shared his X (Twitter) account where he writes about ML/DL: [https://x.com/nerdybat369](https://x.com/nerdybat369).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1414690404221194302)** (4 messages): 

> `6m Model, arxiv link` 


- **6m Model Performs Well**: A member said *"not bad for a 6m model"* while sharing an image, implying the model is performing well.
   - The picture shared was not described.
- **If Only Up Was Good**: A member shared an [Arxiv link](https://arxiv.org/abs/2509.04154) and commented *"if only up was good"*.
   - It is unclear what the link refers to.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1414740117876113579)** (1 messages): 

> `LM Eval Harness Calibration Scores, RL for Calibration, LM Eval Harness PR, Critical Take on Calibration Scores` 


- **Calibration Scores Considered for LM Eval**: A member is interested in adding **calibration scores** to the [LM eval harness](https://github.com/EleutherAI/lm-evaluation-harness) to align incentives towards more trustworthy models.
   - The member suggests it's a broad way to align incentives towards producing more trustworthy models.
- **RL Calibration Work Surfaces**: A member mentioned recent work on **RL for calibration** and included a link to the paper: [https://arxiv.org/pdf/2507.16806](https://arxiv.org/pdf/2507.16806).
   - No further information regarding the paper was provided.
- **Past LM Eval Harness PR Resurfaces**: A member mentioned a previous, unsuccessful PR related to calibration scores for the LM evaluation harness: [https://github.com/EleutherAI/lm-evaluation-harness/pull/874](https://github.com/EleutherAI/lm-evaluation-harness/pull/874).
   - No further information regarding the pull request was provided.
- **Critical Takes on Calibration Scores**: A member shared a critical perspective on calibration scores via a Twitter link: [https://x.com/_jasonwei/status/1871285864690815053](https://x.com/_jasonwei/status/1871285864690815053).
   - No further information regarding the critical take was provided.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1414746533403951134)** (3 messages): 

> `explicitcopies, moves, c binder, EmberJson` 


- ****Explicit Copies** Progress needs more PRs**: A member noted that switching everything over to just use **explicit copies + moves** isn't going to be solved in a single PR, and will need to be broken into smaller PRs, due to blowing up / seg faults.
- **Cherry Pick **EmberJson****: A member mentioned they might cherry pick [this commit](https://github.com/bgreni/EmberJson/pull/53/commits/3039debad36fee5a7f1b6e034e1cb8fa344c4112) into a separate PR once [modular/modular#5289](https://github.com/modular/modular/pull/5289) is merged.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1414690953096200245)** (4 messages): 

> `Mojo test suite duration, Custom ops compilation issues` 


- **Mojo 💥 Test Suite Times Explode**: Using Mojo code inside a codebase causes the test suite duration to explode, tracked in [this issue](https://github.com/modular/modular/issues/5293).
   - There is another issue with compiling custom ops at the same time in multiple processes, but it's hard to reduce the bug.
- **Custom Ops Writing Blocked 🛑**: A member reports being blocked from writing custom ops due to [this issue](https://github.com/modular/modular/issues/5294).
   - The member is actively working on reproducing the bug to help resolve it.
