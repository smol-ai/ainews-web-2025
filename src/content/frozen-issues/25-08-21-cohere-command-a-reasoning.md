---
id: MjAyNS0w
title: Cohere Command A Reasoning beats GPT-OSS-120B and DeepSeek R1 0528
date: '2025-08-21T05:44:39.731046Z'
description: >-
  **Cohere's Command A Reasoning** model outperforms GPT-OSS in open deep
  research capabilities, emphasizing agentic use cases for 2025.
  **DeepSeek-V3.1** introduces a hybrid reasoning architecture toggling between
  reasoning and non-reasoning modes, optimized for agentic workflows and coding,
  with extensive long-context pretraining (~630B tokens for 32k context, ~209B
  for 128k), FP8 training, and a large MoE expert count (~37B). Benchmarks show
  competitive performance with notable improvements in SWE-Bench and other
  reasoning tasks. The model supports a $0.56/M input and $1.68/M output pricing
  on the DeepSeek API and enjoys rapid ecosystem integration including HF
  weights, INT4 quantization by Intel, and vLLM reasoning toggles. Community
  feedback highlights the hybrid design's pragmatic approach to agent and
  software engineering workflows, though some note the lack of tool use in
  reasoning mode.
companies:
  - cohere
  - deepseek
  - intel
  - huggingface
  - baseten
  - vllm-project
  - chutes-ai
  - anycoder
models:
  - command-a-reasoning
  - deepseek-v3.1
topics:
  - agentic-ai
  - hybrid-models
  - long-context
  - fp8-training
  - mixture-of-experts
  - benchmarking
  - quantization
  - reasoning
  - coding-workflows
  - model-pricing
people:
  - artificialanlys
  - reach_vb
  - scaling01
  - cline
  - ben_burtenshaw
  - haihaoshen
  - jon_durbin
  - _akhaliq
  - willccbb
  - teortaxestex
---


**A new SOTA open model.**

> AI News for 8/20/2025-8/21/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (229 channels, and 7429 messages) for you. Estimated reading time saved (at 200wpm): 605 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

We last checked in on [Cohere's Command A in March](https://news.smol.ai/issues/25-03-17-ainews-coheres-command-a-claims-3-open-model-spot-after-deepseek-and-gemma) and then again last week with their [$7B series D](https://news.smol.ai/issues/25-08-14-cohere-ai2), but we didn't think we'd be talking about Cohere again so soon - [Command A Reasoning](https://x.com/cohere/status/1958542682890047511) puts GPT-OSS to shame according to Cohere's own evals:

![](https://resend-attachments.s3.amazonaws.com/1Ynv4Tjwe5C6Y1k)

Importantly for the killer agentic use case of 2025, it is a very decent open deep research model:

![](https://resend-attachments.s3.amazonaws.com/f2mfFNddYUeLEjL)

---

# AI Twitter Recap

**DeepSeek V3.1: hybrid reasoning release, agent focus, and early results**

- **DeepSeek-V3.1 (Think/Non-Think hybrid)**: DeepSeek introduced a unified model that can toggle between “reasoning” and “non-reasoning” via <think></think> tokens, with an explicit push toward agentic use cases and coding workflows. Official announcement and demo links: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1958417062008918312). Notable details aggregated by the community:
    - Training/post-training: extended long-context pretraining (reported ~630B tokens for 32k and ~209B tokens for 128k), FP8 training (“UE8M0 FP8”) tuned for next-gen accelerators. Architecture remains 671B total parameters, ~37B active MoE experts ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1958432118562041983), [@Anonyous_FPS](https://twitter.com/Anonyous_FPS/status/1958437047359995914), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1958437815710089697)).
    - Capabilities/limits: reasoning mode disables tool/function-calling; tool use is supported in non-thinking mode ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1958432118562041983)). New “search agent” ability highlighted ([@reach_vb](https://twitter.com/reach_vb/status/1958430639595864378)).
    - Benchmarks (selected): GPQA 80.1; AIME 2024 93.1; LiveCodeBench 74.8; Aider Polyglot 76.3% with thinking; SWE-Bench Verified rises from 44.6%→66% ([@reach_vb](https://twitter.com/reach_vb/status/1958430639595864378), [@scaling01](https://twitter.com/scaling01/status/1958438863279681824), [@scaling01](https://twitter.com/scaling01/status/1958438007104549243), [@cline](https://twitter.com/cline/status/1958580433979154720)). Artificial Analysis’ composite “AAI Index” places V3.1(reasoning) at 60 vs R1’s 59, still behind Qwen3 235B 2507(reasoning) ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1958432118562041983)).
    - Pricing/context: $0.56/M in, $1.68/M out on DeepSeek API; 164k context window for deepseek-chat-v3.1 ([@scaling01](https://twitter.com/scaling01/status/1958438863279681824), [@cline](https://twitter.com/cline/status/1958580433979154720)).
    - Ecosystem support landed rapidly: HF weights and inference providers ([@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1958449429511352549)), INT4 quant by Intel ([@HaihaoShen](https://twitter.com/HaihaoShen/status/1958507863749325197)), vLLM reasoning toggle ([@vllm_project](https://twitter.com/vllm_project/status/1958580047658491947)), SGLang tool-calling + thinking flag parser ([@jon_durbin](https://twitter.com/jon_durbin/status/1958488353478758599)), Chutes hosting/pricing ([@chutes_ai](https://twitter.com/chutes_ai/status/1958507978476106196)), Baseten latency track ([@basetenco](https://twitter.com/basetenco/status/1958515897972232526)), anycoder integration ([@_akhaliq](https://twitter.com/_akhaliq/status/1958488877024362966)).
    - Community takeaways: strong agentic/coding uplift and efficiency; minor overall capability delta vs R1/V3 in some composite indices; hybrid design aligns with pragmatic agent and SWE workflows; concern about missing tool-use in reasoning mode; perception split between “minor version bump” and “meaningful agent step” ([@willccbb](https://twitter.com/willccbb/status/1958420877537849801), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1958437173948023127), [@reach_vb](https://twitter.com/reach_vb/status/1958430639595864378), [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1958432118562041983)).

**Cohere’s Command A Reasoning and other new reasoning models**

- **Cohere – Command A Reasoning (open weights)**: Cohere announced its enterprise-focused reasoning model with open weights for research/private deployment; commercial use requires a Cohere license. Emphasis on safety/usefulness balance, reduced over-refusals, and strong tool-use/agentic benchmarks. Available on Cohere’s platform and Hugging Face; day-0 integration in anycoder and Inference Providers ([@cohere](https://twitter.com/cohere/status/1958542682890047511), [@Cohere_Labs](https://twitter.com/Cohere_Labs/status/1958576284763611322), [@reach_vb](https://twitter.com/reach_vb/status/1958563034810446169), [@_akhaliq](https://twitter.com/_akhaliq/status/1958602589681197494), discussion on license: [@scaling01](https://twitter.com/scaling01/status/1958561844810903708)).
- **NVIDIA Nemotron Nano 2**: Announced as a hybrid Mamba-Transformer reasoning model; limited public detail in the thread, but worth tracking as NVIDIA’s small-footprint reasoning line evolves ([@_akhaliq](https://twitter.com/_akhaliq/status/1958545622618788174)).

**Google AI: Gemini efficiency paper, agentic Search, Veo access, and Gov platform**

- **Measured Gemini inference efficiency**: Google shared a detailed methodology and results: median Gemini Apps text prompt uses ~0.24 Wh and ~0.26 ml of water; from May 2024→May 2025, energy per median prompt dropped 33× and carbon footprint 44×, due to model/system efficiency and cleaner energy. Technical paper and blog linked in the thread ([@JeffDean](https://twitter.com/JeffDean/status/1958525015722434945)). Separate commentary notes Gemini’s hybrid-reasoning architecture mention in the materials ([@eliebakouch](https://twitter.com/eliebakouch/status/1958603730951029157)).
- **AI Mode gets agentic**: AI Mode in Search can plan and execute multi-step tasks (e.g., dine reservations across sites with real-time availability), personalize results, and share session context; rolling out to 180+ countries/territories in English ([@Google](https://twitter.com/Google/status/1958530316072534323), [@GoogleAI](https://twitter.com/GoogleAI/status/1958561117833228705), [@rmstein](https://twitter.com/rmstein/status/1958552694626607616)).
- **Veo 3 access and demos**: Google teased broader Veo 3 access via the Gemini app and spun up TPU capacity in preparation ([@GeminiApp](https://twitter.com/GeminiApp/status/1958558340163699008), [@joshwoodward](https://twitter.com/joshwoodward/status/1958555951344345461)). Google Devs shipped a Next.js template to build an in-browser AI video studio using Veo 3 and Imagen 4 ([@googleaidevs](https://twitter.com/googleaidevs/status/1958599306472206349)).
- **Gemini for Government**: Expanded secure AI platform for U.S. federal use (includes NotebookLM and Veo) with @USGSA; pitched as “virtually no cost” to eligible federal employees ([@sundarpichai](https://twitter.com/sundarpichai/status/1958538684208476611)).

**Reasoning, RL, and evals: new methods and benchmarks**

- **RL for LLM reasoning (survey)**: Empirical roundup “Part I: Tricks or Traps?” systematically probes RL algorithm improvements for reasoning, highlighting scale/compute constraints and varying gains even up to 8B models ([@nrehiew_](https://twitter.com/nrehiew_/status/1958521596492071411)).
- **DuPO – Dual Preference Optimization**: Self-supervised feedback generation via duality to enable reliable self-verification without external annotations; frame invertibility examples (e.g., reverse math solution to recover hidden variables). Paper + discussion: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1958467512296939593), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1958415225952035274).
- **PRM unification with RL/Search**: “Your Reward Function for RL is Your Best PRM for Search-Based TTS” posits learning a dense dynamic PRM via AIRL+GRPO from correct traces, usable both as RL critic and search heuristic ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1958470481050534069)).
- **KV cache debate**: A widely-circulated reminder that recomputing KV can be preferable to storing it for certain regimes sparked discussion comparing this to MLA-like tradeoffs rather than “no KV” ([@dylan522p](https://twitter.com/dylan522p/status/1958367037773824095), response: [@giffmana](https://twitter.com/giffmana/status/1958575876540428379)).
- **New eval assets**:
    - MM-BrowseComp: 224 multimodal web tasks (text+image+video) for agents; code+HF dataset+arXiv ([@GeZhang86038849](https://twitter.com/GeZhang86038849/status/1958381269617955165)).
    - Kaggle Game Arena (text-only Chess) Elo-like model ranking ([@kaggle](https://twitter.com/kaggle/status/1958546786081030206)).
    - ARC-AGI-3 Preview: +3 public holdout games for agent evals ([@arcprize](https://twitter.com/arcprize/status/1958597816823202216)).

**Systems and tooling: APIs, serving, and dev infra**

- **OpenAI Responses API updates**: New “Connectors” pull context from Gmail/Calendar/Dropbox etc. in one call; “Conversations” adds persistent thread storage (messages, tool calls, outputs) so you don’t need to run your own chat DB; demo app and docs included ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1958660207745409120), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1958660216624751097), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1958660224019247176); overview: [@gdb](https://twitter.com/gdb/status/1958691151139283454)).
- **Agent/dev tools**:
    - Cursor + Linear: launch agents directly from issues/comments ([@cursor_ai](https://twitter.com/cursor_ai/status/1958627514852811034)).
    - vLLM and SGLang: first-class DeepSeek-V3.1 Think/Non-Think support ([@vllm_project](https://twitter.com/vllm_project/status/1958580047658491947), [@jon_durbin](https://twitter.com/jon_durbin/status/1958488353478758599)).
    - MLX ecosystem: mlx-vlm 0.3.3 adds GLM-4.5V, Command-A-Vision; JinaAI mlx-retrieval enables local Gemma3-270m embeddings/rerankers at ~4000 tok/s on M3 Ultra ([@Prince_Canuma](https://twitter.com/Prince_Canuma/status/1958469233622327785), [@JinaAI_](https://twitter.com/JinaAI_/status/1958547803489415195)).
    - Parsing/RAG: LlamaParse adds citations + modes (cost-effective/Agentic/Agentic+); Weaviate’s Elysia ships decision-tree agentic RAG with real-time reasoning visualization ([@VikParuchuri](https://twitter.com/VikParuchuri/status/1958520215844655576), [@weaviate_io](https://twitter.com/weaviate_io/status/1958568536420299184)).
    - LlamaIndex: “vibe-llama” CLI scaffolds context-aware rules for many coding agents ([@llama_index](https://twitter.com/llama_index/status/1958656414295237014)).
    - Hosting: W&B Inference adds DeepSeek V3.1 ($0.55/$1.65/M tok), Chutes priced hosting, Baseten library entry ([@weave_wb](https://twitter.com/weave_wb/status/1958681269484880026), [@chutes_ai](https://twitter.com/chutes_ai/status/1958507978476106196), [@basetenco](https://twitter.com/basetenco/status/1958515897972232526)).
- **Applied products**: Perplexity Finance rolls out NL stock screening for Indian equities across surfaces ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1958385027185877066), [@jeffgrimes9](https://twitter.com/jeffgrimes9/status/1958364311178674232)).

**Research highlights (vision, multimodal, 3D, embodied)**

- **MeshCoder**: LLM-powered structured mesh code generation from point clouds ([@_akhaliq](https://twitter.com/_akhaliq/status/1958379365147775414)).
- **RynnEC**: Bridging multimodal LLMs and embodied control ([@_akhaliq](https://twitter.com/_akhaliq/status/1958380616711299561)).
- **Tinker Diffusion**: multi-view-consistent 3D editing without per-scene optimization ([@_akhaliq](https://twitter.com/_akhaliq/status/1958380980000981208)).
- **RotBench**: diagnosing MLLMs on image rotation identification ([@_akhaliq](https://twitter.com/_akhaliq/status/1958635243197325625)).
- Additional community notes: Goodfire AI on efficiently surfacing rare, undesirable post-training behaviors for mitigation via behavior steering/interpretability ([@GoodfireAI](https://twitter.com/GoodfireAI/status/1958567217089716334)).

**Top tweets (by engagement)**

- DeepSeek-V3.1 announcement: hybrid reasoning, agent focus, public demo ([@deepseek_ai](https://twitter.com/deepseek_ai/status/1958417062008918312), 16.3k+)
- Google’s Gemini inference efficiency paper: 33× energy, 44× carbon per-prompt reductions year-over-year, 0.24 Wh and 0.26 ml water per median prompt ([@JeffDean](https://twitter.com/JeffDean/status/1958525015722434945), 3.8k+)
- Ernest Ryu on recent convex optimization-related results (nuanced math take thread) ([@ErnestRyu](https://twitter.com/ErnestRyu/status/1958408925864403068), 3.1k+)
- Perplexity Finance India stock screening shipping to all users ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1958385027185877066), 2.3k+)
- Alex Wang on Meta Superintelligence Labs’ investment trajectory ([@alexandr_wang](https://twitter.com/alexandr_wang/status/1958599969151361126), 2.6k+)
- François Chollet clarifies his long-held stance: bullish on deep learning scaling for usefulness, not AGI via scaling alone ([@fchollet](https://twitter.com/fchollet/status/1958410017683681698), 900–1000+)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. DeepSeek V3.1: Anthropic API compatibility + thinking-mode benchmarks

- [**DeepSeek-V3.1 implements Anthropic API compatibility**](https://i.redd.it/0pp8mwjkkbkf1.jpeg) ([Score: 265, Comments: 31](https://www.reddit.com/r/LocalLLaMA/comments/1mw3nat/deepseekv31_implements_anthropic_api_compatibility/)): **DeepSeek-V3.1 adds drop‑in compatibility with the Anthropic API, per the official guide (https://api-docs.deepseek.com/guides/anthropic_api). The screenshot shows installing the Anthropic/Claude client via npm, setting environment variables (API key/base URL) to point at DeepSeek, and then invoking DeepSeek models through Anthropic’s messages interface—enabling existing Anthropic-integrated apps to switch backends with minimal code changes.** Comments note a typo (“Anthoripic”) and ask whether this lets them leverage Claude-compatible tooling while using DeepSeek as the model backend, potentially as a cost-saving alternative to recent price changes.
    - Anthropic-API compatibility means you can point existing Claude clients (messages + tool-use) at DeepSeek-V3.1 and keep the same tools JSON schema and invocation flow; the only swap is the base URL/model name. This enables drop-in use of Claude Tools with DeepSeek as the backend, a lever for cost/perf optimization if the app can tolerate model differences. Docs: [Messages API](https://docs.anthropic.com/claude/reference/messages_post), [Tool use](https://docs.anthropic.com/en/docs/tool-use).
    - One user reports 404: `{"error_msg":"Not Found. Please check the configuration."}`, which typically indicates a mismatched endpoint or headers in an Anthropic-compatible shim. Check the exact base URL/path (e.g., `/v1/messages`), required headers (`anthropic-version`, `x-api-key`, `content-type`), and a valid `model` id; some providers also require an `anthropic-beta` tools header when using tools. Using an Anthropic SDK against a non-default base URL without overriding the host commonly produces this 404.
    - Several note this mirrors a broader trend (e.g., kimi-k2, GLM-4.5) of vendors exposing compatibility layers to reduce switching costs. Compatibility with Anthropic/OpenAI APIs lets teams reuse clients, prompt tooling, and function-calling specs with minimal code changes, enabling rapid A/B of backends.
- [**DeepSeek V3.1 (Thinking) aggregated benchmarks (vs. gpt-oss-120b)**](https://www.reddit.com/gallery/1mwexgd) ([Score: 150, Comments: 61](https://www.reddit.com/r/LocalLLaMA/comments/1mwexgd/deepseek_v31_thinking_aggregated_benchmarks_vs/)): **Post compares DeepSeek V3.1 (Thinking) vs. gpt-oss-120b (High): both offer ~128–131K context, but DeepSeek is a large MoE (**`671B` **total,** `37B` **active) while gpt-oss-120b is smaller (**`120B` **total,** `5.1B` **active). Reported composite scores are similar for “Intelligence Index” (**`60` **vs** `61`**), with DeepSeek ahead on “Coding Index” (**`59` **vs** `50`**), math unspecified; however, latency/throughput and cost differ substantially: DeepSeek needs** `~127.8s` **for “500 tokens + thinking” at** `~20 tok/s`**, priced at** `$0.32 / $1.15` **(input/output), versus gpt-oss-120b at** `~11.5s`**,** `~228 tok/s`**, and** `$0.072 / $0.28`**.** Commenters question the benchmark validity: the cited “Artificial Analysis Coding Index” has gpt-oss-20b scoring `54`, beating gpt-oss-120b (`50`) and Claude Sonnet 4 thinking (`53`), prompting “Something must be off here…”, and one concludes that *benchmarks are barely useful now*. Another anecdotal view is that gpt-oss isn’t competitive with the ~700B-parameter DeepSeek despite good performance-for-size.
    - Several note an anomaly in the “Artificial Analysis Coding Index”: **gpt-oss 20B (high)** scores `54`, edging **Claude Sonnet 4 Thinking** at `53` and beating **gpt-oss 120B (high)** at `50`. A smaller 20B model surpassing a 120B and Claude’s thinking mode suggests possible aggregation bias, task selection skew, or poor normalization in the meta-benchmark methodology.
    - Practitioners report divergent real-world results: **DeepSeek (V3.1/“Whale” ~700B class)** is said to outperform gpt-oss on SWE-style tasks, with “almost Sonnet-level” solution quality, strong instruction-following, and reliable tool-calling. This contrasts with the aggregate index and implies that interactive, tool-augmented SWE evaluations may reveal strengths not captured by the benchmark mix.
    - Critique centers on the benchmark being a meta-aggregation (“zero independent thinking”), i.e., a weighted blend of existing tests without fresh task design or transparent calibration. Commenters argue such rollups can be unreliable—overfitting to included benchmarks, masking variance across task types, and producing counterintuitive rank inversions (e.g., 20B > 120B).
- [**Deepseek V3.1 is not so bad after all..**](https://www.reddit.com/gallery/1mw3j7l) ([Score: 154, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1mw3j7l/deepseek_v31_is_not_so_bad_after_all/)): **OP argues DeepSeek V3.1 was misjudged because it targets different priorities—namely speed and agentic/task-execution use cases—where it performs well. A notable practical update mentioned is that DeepSeek now supports the Anthropic-compatible API format, enabling drop‑in integration with clients built for Claude/Claude Code (see Anthropic API docs: https://docs.anthropic.com/en/api/), potentially offering a lower‑cost alternative to other providers.** Comments largely note there wasn’t broad consensus it was “bad,” and urge waiting a few weeks for workflows, prompting, and integrations to mature before judging model quality.
    - DeepSeek’s new support for the **Anthropic API format** means it can act as a near drop‑in for clients/tools expecting Claude-style requests (messages, tool use/function calling semantics), potentially letting teams wire it into **Claude Code** with minimal glue. This lowers integration friction and could offer a cheaper swap for workloads currently pinned to Anthropic/OpenAI, depending on latency/throughput/cost tradeoffs; see Anthropic’s Messages API spec for reference: https://docs.anthropic.com/claude/reference/messages_post and Claude Code: https://www.anthropic.com/news/claude-code.
    - A commenter flags the proliferation of benchmarks, implicitly calling for a “benchmark for benchmarks”—i.e., standardizing how we assess reliability, variance, and contamination across eval suites. Practically, this highlights the need to report methodology (prompting, context windows, temperature, seed control) and to prefer robust, multi-axis evaluations over cherry-picked leaderboards, especially as models like DeepSeek V3.1 get judged early on fragmented metrics.
- [**Love small but mighty team of DeepSeek**](https://i.redd.it/38d427vmpdkf1.png) ([Score: 801, Comments: 39](https://www.reddit.com/r/LocalLLaMA/comments/1mwbsww/love_small_but_mighty_team_of_deepseek/)): **The post is a meme pointing out a typo in the DeepSeek API docs where “Anthropic API” is misspelled as “Anthoripic API” (see the highlighted screenshot: https://i.redd.it/38d427vmpdkf1.png). The title/selftext frame this as a small, fast-moving team shipping docs quickly, implicitly highlighting documentation QA/polish trade-offs while referencing an Anthropic-related section in the docs.** Top comments argue the typo signals the docs were written by humans (a sign of authenticity) rather than generated by an LLM, joking that an LLM “would never put that typo,” and calling it the mark of a “true software engineer.”
    - 

### 2. Model releases/ports: DeepSeek-V3.1 HF card and Kimi-VL-A3B-Thinking GGUF (llama.cpp PR #15458)

- [**deepseek-ai/DeepSeek-V3.1 · Hugging Face**](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) ([Score: 508, Comments: 78](https://www.reddit.com/r/LocalLLaMA/comments/1mw3c7s/deepseekaideepseekv31_hugging_face/)): [**DeepSeek‑V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) is a** `671B`**‑param MoE (**`~37B` **activated) hybrid model supporting both “thinking” and “non‑thinking” modes via chat templates, with post‑training focused on tool/agent use and faster reasoning than R1‑0528 at similar quality. Long‑context was extended on top of the V3 base via two phases: 32K (expanded 10× to** `630B` **tokens) and 128K (3.3× to** `209B` **tokens), trained with UE8M0 FP8 microscaling; MIT‑licensed weights (Base/Instruct) are on HF. Reported benchmarks include HLE with search+Python** `29.8%` **(vs R1‑0528** `24.8`**, GPT‑5 Thinking** `35.2`**, o3** `24.3`**, Grok 4** `38.6`**, Gemini Deep Research** `26.9`**), SWE‑Bench Verified without thinking** `66.0%` **(vs R1‑0528** `44.6`**, GPT‑5 Thinking** `74.9`**, o3** `69.1`**, Claude 4.1 Opus** `74.5`**, Kimi K2** `65.8`**), and Terminal Bench (Terminus‑1)** `31.3%` **(vs o3** `30.2`**, GPT‑5** `30.0`**, Gemini 2.5 Pro** `25.3`**). Model card highlights improved tool calling, hybrid thinking, and defined prompt/tool/search‑agent formats; broader evals cite strong scores (e.g., MMLU‑Redux** `93.7`**, LiveCodeBench** `74.8`**, AIME’24** `93.1`**).** Top comment notes agentic‑deployment caveats: prompts/frameworks drive UX, DeepSeek lacks branded search/code agents (unlike OpenAI/Anthropic/Google), and serverless hosts may degrade quality via lower precision, expert pruning, or poor sampling. Observers infer the R1+V3 merge and default `128K` context optimize token TCO for agentic coding; some regressions (GPQA, offline HLE) suggest V3 family is near its limit, prompting calls for V4. Also clarified: this release is the post‑trained model on top of the base, not just the base checkpoint.
    - Agentic/tool-use performance saw clear gains: `29.8%` on HLE with search+Python (vs R1-0528 `24.8%`, GPT-5 Thinking `35.2%`, o3 `24.3%`, Grok 4 `38.6%`, Gemini Deep Research `26.9%`), `66.0%` on SWE-bench Verified without thinking (vs R1-0528 `44.6%`, GPT-5 Thinking `74.9%`, o3 `69.1%`, Claude 4.1 Opus `74.5%`, Kimi K2 `65.8%`), and `31.3%` on TerminalBench (Terminus 1) (vs o3 `30.2%`, GPT-5 `30.0%`, Gemini 2.5 Pro `25.3%`). Caveats: DeepSeek HLE runs use a text-only subset; OpenAI SWE-bench used `477/500` problems; Grok 4 may lack webpage filtering (possible contamination). A broader table places V3.1-Thinking at MMLU-Pro `84.8`, GPQA Diamond `80.1` (cf. dataset: https://huggingface.co/datasets/Idavidrein/gpqa), AIME 2025 `88.4`, LiveCodeBench `74.8` (https://livecodebench.github.io/), and Aider Polyglot `76.3` (https://github.com/Aider-AI/aider), generally trailing top closed models on several axes.
    - Training/architecture updates: V3.1 is a hybrid that switches thinking vs non-thinking via the chat template and claims smarter tool-calling from post-training. Long-context extension was scaled up—32K phase ~10× to `630B` tokens and 128K phase `3.3×` to `209B`—with training in `UE8M0` FP8 scale to support microscaling formats; V3.1-Think targets R1-0528-level answer quality with faster responses.
    - Deployment/agent considerations: merging R1 into V3 with a `128K` default window aims to cut TCO for agentic coding (token-heavy, not long-CoT-heavy). Real-world UX can degrade if providers run lower precision, prune experts, or mis-tune sampling; plus prompts/agent frameworks (e.g., branded search/code agents from OpenAI/Anthropic/Google) critically affect outcomes, while DeepSeek’s own frameworks aren’t public yet. Some regressions were noted on GPQA and offline HLE, and several public benches emphasized agentic use without thinking traces.
- [**Finally Kimi-VL-A3B-Thinking-2506-GGUF is available**](https://huggingface.co/ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF) ([Score: 174, Comments: 11](https://www.reddit.com/r/LocalLLaMA/comments/1mw0tc4/finally_kimivla3bthinking2506gguf_is_available/)): **GGUF-format builds of the 16B-parameter Kimi-VL-A3B-Thinking-2506 VLM are now hosted on Hugging Face ([original model](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506), [GGUF quantizations](https://huggingface.co/ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF)), with [llama.cpp PR #15458](https://github.com/ggml-org/llama.cpp/pull/15458) adding backend support. The release spans** `4/8/16-bit` **(12) variants; however, the PR’s latest note reports an unresolved issue where the “number of output tokens is still not correct,” indicating current inference may be unstable or incomplete.** Commenters celebrate that “VL finally gets some love” and report good early tests, while others argue it’s “not available/working yet” due to the token-count bug; another highlights Kimi’s lower sycophancy and hopes this behavior carries over.
    - Availability/status: A commenter points to the latest PR note stating: *“Hmm turns out the number of output tokens is still not correct. But on the flip side, I didn't break other models”*, implying the GGUF build isn’t fully working yet due to an output-token counting bug and may not be ready for reliable use until the fix lands.
    - Modality gap: Another user reports the model cannot access the audio track in videos. This aligns with many open VLM/MLLMs being vision+text only; true audiovisual understanding would require an ASR front-end or end-to-end AV training, which Kimi-VL-A3B apparently lacks at present.
    - Comparative interest: Users asked how it stacks up against **Qwen3-30B-A3B**, but no benchmarks or head-to-head results (e.g., MMBench/MMMU accuracy, A3B decoding throughput/latency) were provided in-thread. A proper comparison would need standardized vision QA and multi-image/video evals under identical inference settings.

### 3. Efficiency & scaling: 1–8 bit quantization guide, 100k H100 under-scaling, and 160GB VRAM local build

- [**Why low-bit models aren't totally braindead: A guide from 1-bit meme to FP16 research**](https://i.redd.it/5t58iz5u9ekf1.jpeg) ([Score: 299, Comments: 36](https://www.reddit.com/r/LocalLLaMA/comments/1mwevt4/why_lowbit_models_arent_totally_braindead_a_guide/)): **Meme-as-analogy post equates lossy image compression (JPEG) to LLM quantization: reduce precision of weights to shrink models while preserving salient behavior via mixed precision, calibration-driven rounding, and architectures designed for ultra-low precision. It highlights low-bit training/inference like Microsoft’s BitNet—1-bit and ternary (~1.58-bit) Transformers ([BitNet 1-bit](https://arxiv.org/abs/2310.11453), [1.58-bit follow-up](https://arxiv.org/abs/2402.17764))—and practical schemes such as [Unsloth Dynamic 2.0 GGUFs](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs), noting larger models (e.g.,** `~70B`**) tolerate more aggressive quantization. Resources include a visual explainer and a video on quantization-aware training, underscoring that quantization is not uniform and can be targeted per layer/block to retain grammar/reasoning while compressing less-crucial knowledge.** Comments suggest task-specific calibration datasets (e.g., Q4-coding vs Q4-writing) to tailor quantization, and an anecdote that JPEG can outperform SD 1.5’s VAE compression; another links a tongue-in-cheek “0.5-bit” meme.
    - Stable Diffusion 1.5’s original VAE is called out for poor reconstruction fidelity—one commenter claims plain JPEG can outperform it for some images, highlighting that the SD1.5 KL-f8 VAE can introduce noticeable artifacts and information loss relative to traditional codecs. Practical implication: in image-generation pipelines, VAE quality can be a larger bottleneck than model quantization, so swapping VAEs or using higher-fidelity decoders may yield bigger gains than tweaking bit-width alone.
    - Quantization relies on a calibration dataset, so domain-specific calibration (e.g., Q4-coding vs Q4-writing) can materially affect accuracy by tailoring activation/statistics to task distributions. This mirrors outlier-aware PTQ methods like **AWQ** and **GPTQ**, where representative samples drive per-channel/per-group scales, improving retention of key features at the same bit-width without retraining.
    - Resources for implementation and formats: official **GGUF** spec in the **ggml** project ([docs](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)) and **Hugging Face**’s technical intro to **ggml** ([article](https://huggingface.co/blog/introduction-to-ggml)) for low-bit inference details; a concise explainer of 1-bit LLMs/BitNet ([video](https://www.youtube.com/watch?v=7hMoz9q4zv0)); and a practical **llama.cpp** setup/usage guide ([blog](https://blog.steelph0enix.dev/posts/llama-cpp-guide/)).
- [**Frontier AI labs’ publicized 100k-H100 training runs under-deliver because software and systems don’t scale efficiently, wasting massive GPU fleets**](https://www.reddit.com/gallery/1mw2lme) ([Score: 335, Comments: 81](https://www.reddit.com/r/LocalLLaMA/comments/1mw2lme/frontier_ai_labs_publicized_100kh100_training/)): **Post argues that frontier labs’ much-hyped ~100k–H100 training runs achieve poor cluster-level scaling efficiency, so effective throughput (e.g., MFU/step-time) under-delivers despite massive GPU counts, due to software and systems bottlenecks across the distributed stack (data/tensor/pipeline parallel orchestration, optimizer sharding like ZeRO/FSDP, NCCL collectives, storage I/O, and scheduler topology). The claim is that without careful overlap of compute/communication and tuned high-throughput interconnects (NVLink/InfiniBand) and NCCL parameters, utilization collapses at scale—wasting large fractions of the fleet—meaning “more GPUs” ≠ linear speedups; the linked source ([Reddit gallery](https://www.reddit.com/gallery/1mw2lme)) is access-restricted, so details aren’t directly verifiable here.** Commenters contrast this with projects like DeepSeek that report strong results on ~2k GPUs via aggressive low-level optimizations, implying software maturity, not fleet size, drives throughput; others note multi-GPU/multi-node inference remains hard in open source, and that true hyperscale performance typically demands bespoke, internally tuned frameworks rather than “hobby” or cargo-cult architectures.
    - Commenters emphasize that 100k H100 training runs won’t scale on an off‑the‑shelf PyTorch stack; at that node count you need topology‑aware sharding/parallelism, high‑performance collectives, careful comm/compute overlap, and custom schedulers to avoid all‑reduce and fabric bottlenecks. Without bespoke systems engineering, synchronization and interconnect overheads dominate and GPU utilization collapses, so fleets appear “wasted.” They note stacks that behave on tens to low hundreds of GPUs often fall apart beyond thousands of nodes due to latency and coordination costs.
    - DeepSeek’s reported 2k‑GPU training run is cited as a counterexample where software efficiency beat brute‑force scaling; the team described extensive optimizations to maximize per‑GPU throughput. The takeaway is that kernel fusion, memory/layout tuning, reduced synchronization, and disciplined parallelism strategies can let a smaller cluster outperform a poorly engineered 100k‑GPU job. Initial skepticism around DeepSeek underscored how easily systems work is underestimated relative to headline GPU counts.
    - The open‑source inference ecosystem illustrates how hard distributed systems remain: few projects do solid multi‑GPU inference on a single host, and even fewer handle multi‑node well. Commenters argue that scalable software typically requires purpose‑built frameworks aligned to concrete requirements, not cargo‑cult patterns, and that the lack of access to truly massive deployments limits opportunities to learn and validate scaling approaches. This skills and tooling gap contributes to under‑delivery when organizations attempt 100k‑GPU scale without specialized infrastructure.
- [**Pewdiepie’s monstrous 160GB Vram build**](https://youtu.be/2JzOe1Hs26Q?si=9Ck53vK9hja3BZD7) ([Score: 165, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1mwme5c/pewdiepies_monstrous_160gb_vram_build/)): **PewDiePie showcased an 8× NVIDIA “RTX 4000”-class GPU workstation (20 GB each) totaling ~**`160 GB` **VRAM with** `192 GB` **system RAM, stating he can run [Llama 3 70B](https://ai.meta.com/blog/meta-llama-3/) on half the GPUs. On 4×20 GB (~**`80 GB`**), 70B inference is plausible only with low‑bit quantization (≤8‑bit, typically 4‑bit) plus tensor parallelism, and likely some CPU offload for KV cache/activations; full‑precision weights are ~**`140+ GB` **(FP16) before overhead and won’t fit. Video: [“Accidentally Built a Nuclear Supercomputer”](https://youtu.be/2JzOe1Hs26Q).** Commenters call the 8×20 GB setup unorthodox and note it still cannot host very‑large MLLMs (e.g., DeepSeek/Kimi variants) except at extreme quantization (Q1) or with heavy offload; despite `192 GB` RAM, memory may bottleneck. Others see this as evidence of local LLMs going mainstream and suggest power‑efficiency may have driven the part choices.
    - Build details: an unorthodox 8× RTX 4000 Ada setup (`20 GB` each) for a total of `160 GB` VRAM, paired with `192 GB` system RAM (corrected from 96 GB). Commenters infer the choice targets power efficiency while relying on CPU offload; with more memory channels available, RAM could scale higher if needed, improving offload headroom for larger contexts.
    - Model fit constraints: despite 160 GB VRAM, commenters note it likely can’t host frontier models like “Kimi-K2” or DeepSeek end-to-end without extreme quantization (e.g., “Q1”). One user reports that even ~`300 GB` combined VRAM+RAM still struggles to fit Kimi, underscoring that many recent SOTA models’ memory footprints exceed what this build can accommodate without heavy sharding/offload and the resulting bandwidth bottlenecks.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1. DeepSeek V3.1 Drops with Mixed Vibes**

- **DeepSeek V3.1 Debuts, Sparks Hype and Gripes**: Engineers hail **DeepSeek V3.1** for scoring **66** on SWE-bench in non-thinking mode, but slam its creative writing and roleplay flops, calling it a *slightly worse version of Gemini 2.5 Pro* despite coding promise. Users in Cursor report solid TypeScript/JavaScript performance at lower costs than Sonnet, though some distrust *Chinese LLMs* and face connection glitches.
- **DeepSeek V3.1 Pricing Jumps, API Integrations Expand**: DeepSeek hikes input prices to **$0.25-$0.27** starting September 5, 2025, matching reasoner costs, while adding **Anthropic API** support for broader ecosystem use, as announced on [DeepSeek's X post](https://x.com/deepseek_ai/status/1958417062008918312). Communities eagerly await free public access in September, noting paid OpenRouter versions deliver faster responses than free ones.
- **DeepSeek V3.1 Enters Arenas, Faces Bans**: LMArena adds **DeepSeek V3.1** and **deepseek-v3.1-thinking** for battles, but Gemini's mass bans push users to alternatives, with one quipping *we're being sent back to 2023*. Cursor devs mix excitement with skepticism, testing it amid reports of *incremental improvements* and regressions per [DeepSeek's Hugging Face page](https://huggingface.co/deepseek-ai/DeepSeek-V3.1).

**Theme 2. ByteDance's Seed-OSS Models Stir Buzz**

- **ByteDance Unleashes Seed-OSS-36B Beast**: ByteDance drops **Seed-OSS-36B-Base-woSyn**, a **36B** dense model with **512K** context trained on **12T tokens** sans synthetic data, thrilling tuners eager for GPT-ASS experiments despite its *vanilla* architecture lacking MHLA or MoE. Latent Space and Nous Research users invite community tests on [ByteDance's GitHub repos](https://github.com/orgs/bytedance/repositories) and Hugging Face, praising its long-context prowess.
- **Seed-OSS GGUF Delay Fuels ASIC Drama**: Nous Research debates **Seed-OSS-36B**'s missing GGUF quant, blaming custom vLLM and unsupported *SeedOssForCausalLM* architecture in llama.cpp, with one linking to [Aditya Tomar's X post](https://x.com/adityastomar_/status/1958048129275805867) questioning ASIC impacts. Engineers note dropout, bias terms in qkv heads, and custom MLP akin to Llama, speculating on regularization but ruling out simple renames to Llama.
- **ByteDance's SEED Prover Bags IMO Silver**: Eleuther celebrates **ByteDance's SEED Prover** nabbing silver in IMO 2025, but questions real-world math chops per [ByteDance's blog](https://seed.bytedance.com/en/blog/bytedance-seed-prover-achieves-silver-medal-score-in-imo-2025). Unsloth AI users geek out over its **512K** context and no QK norm, hoping for a deep-dive paper.

**Theme 3. Hardware Hurdles and Benchmarks Heat Up**

- **RTX 5090 Sparks Wallet Wars**: Unsloth AI debates **RTX 5090**'s **$2000** price for VRAM upgrades, griping over NVIDIA's no P2P/NVLink while eyeing distributed training with Infiniband on 4090-5090 rigs. GPU MODE users build custom PyTorch libraries and mini-NCCL backends for home Infiniband, calling it a killer distributed computing study hack.
- **Apple M4 Max Crushes GGUF in Benchmarks**: LM Studio benchmarks **GPT-OSS-20b** on M4 Max, with MLX GPU hitting **76.6 t/s** at **32W** versus GGUF CPU's **26.2 t/s** at **43W** using 4-bit quants and 4K context. Users tweak CUDA on 4070 TI Super via ctrl+shift+r for flash attention and q8_0 KV quant at batch 2048, fixing *0 GPUs detected* errors.
- **AMD Debugger Alpha Drops with Wave Stepping**: GPU MODE unveils alpha **AMD GPU debugger** sans amdkfd dependency, boasting disassembly and wave stepping in [this demo video](https://cdn.discordapp.com/attachments/1233704710389764236/1407932291912695949/2025-08-21_06-20-14.mp4?ex=68a88f60&is=68a73de0&hm=6e9ca7ceed29674c6943e989940a6c8e144707797c5abd571e1cfe60d3f3210d). MI300 dominates trimul leaderboard at **3.50 ms**, while B200 hits **2.15 ms** and H100 grabs second at **3.80 ms**.

**Theme 4. Training Tricks and Datasets Dominate**

- **GRPO Demands Dataset Smarts for Games**: Unsloth AI advises splitting multi-step game datasets into separate prompts for GRPO, warning full PPO suits games better since GRPO shines when LLMs *roughly know what to do*. Members push diversified imatrix calibration like [Ed Addario's dataset](https://huggingface.co/datasets/eaddario/imatrix-calibration) over WikiText-raw for multilingual quant preservation.
- **WildChat Dataset Dedupes English Prompts**: Unsloth AI releases **WildChat-4M-English-Semantic-Deduplicated** with prompts under **2000 tokens**, using Qwen-4B-Embedding and HNSW for semantic dedup, per [Hugging Face details](https://huggingface.co/datasets/MasonMac/WildChat-4M-English-Semantic-Deduplicated). Qwen3-30B-A3B hits **10 t/s** on CPU via llama-bench, with RoPE scaling enabling **512K** context in 30B/235B models.
- **R-Zero Evolves LLMs Sans Humans**: Moonshot AI shares [R-Zero PDF study](https://cdn.discordapp.com/attachments/1371757564005711973/1408153973751545896/R-Zero__A_Comprehensive_Study_of_a_Self-Evolving_LLM_Training_Method.pdf?ex=68a8b515&is=68a76395&hm=dcba93436f636eeec364d08d08e1603131147faaa595637065f2e226772005f2&) on self-evolving LLM training from zero human data, slashing dataset reliance. Eleuther's CoMET models, pretrained on **115B medical events** from **118M patients**, outperform supervised ones per [scaling paper](https://arxiv.org/abs/2508.12104).

**Theme 5. Industry Shifts and Safety Snags**

- **Meta Reorgs AI Under Wang's Reign**: Latent Space reports Meta splitting AI into four teams under **Alexandr Wang**, disbanding AGI Foundations with Nat Friedman and Yann LeCun reporting to him, eyeing an "omni" model per [Business Insider](https://www.businessinsider.com/meta-ai-superintelligence-labs-reorg-alexandr-wang-memo-2025-8). Yannick Kilcher speculates Yann LeCun's FAIR demotion signals Meta's open-source retreat.
- **Generative AI Yields Zero ROI for 95%**: OpenRouter cites [AFR Chanticleer report](https://archive.md/IlP7F) showing **95% of orgs** get zero return from customized AI due to poor business nuance learning. Moonshot AI contrasts China's *energy as given* for data centers with U.S. grid debates per [Fortune article](https://fortune.com/2025/08/14/data-centers-china-grid-us-infrastructure/).
- **API Leaks and Bans Bite Users**: OpenRouter user loses **$300** to leaked API key, with proxies masking IPs; Gemini bans trigger AI Dungeon purge flashbacks. LlamaIndex shares [AI safety survey](https://mukullight.pythonanywhere.com/form) for community takes on key questions.


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano-Banana Falls Prey to McLau's Law**: Members joked that the **Nano-Banana** model often underperforms expectations, humorously dubbing this phenomenon "**McLau's Law**," referencing an **OpenAI** researcher, prompting discussion about **AI's** current capabilities as depicted in [an attached image](https://cdn.discordapp.com/attachments/1340554757827461211/1407957527987097642/RDT_20250821_0619535468074318472918433.jpg?ex=68a8a6e1&is=68a75561&hm=bfcfd37e574ea905e84f2ff1c9e6ffee1855681ad903050cfe30a367ea5d96f3&).
   - One user suggested **Nano-Banana** often yields results *far below nano-banana*.
- **Video Arena Plagued by Bot Brain-Freeze**: Users reported the **Video Arena Bot** being down, causing command failures and inability to generate videos, effectively locking access to prompt channels <#1397655695150682194>, <#1400148557427904664>, and <#1400148597768720384>.
   - Moderators confirmed the downtime and ongoing fixes, directing users to the announcements channel for updates and also stating that a login feature will be available soon to prevent future outages.
- **DeepSeek V3.1 Enters the Ring**: **DeepSeek V3.1** and **deepseek-v3.1-thinking** models have been added to the LMArena and are now available for use.
   - The consensus is that the **v3.1** model is a *slightly worse version of Gemini 2.5 pro* although it holds promise as a coding model, but needs enhancement in general abilities.
- **LMArena Users Suffer Data Loss**: A site outage caused widespread data loss, including missing chat histories and inability to accept terms of service.
   - Moderators acknowledged the issue and assured users that a fix is underway.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **ByteDance Drops Seed-OSS 36B Base Model**: ByteDance has released the **Seed-OSS-36B-Base-woSyn** model on Hugging Face, a **36B** dense model with **512K** context window, trained on **12T tokens**.
   - Members are eager to try tuning GPT-ASS with the model, finding the lack of synthetic data compelling.
- **GRPO Requires Smart Dataset Design**: To use **GRPO** for multi-step game actions, members advised designing datasets with separate prompts for each step.
   - Full PPO might be better suited for games, as GRPO is primarily effective for LLMs because *they roughly know what to do to begin with*.
- **DeepSeek V3.1's Thinking Skills**: The **DeepSeek V3.1** model achieved a **66** on SWE-bench verified in non-thinking mode, sparking hype among members.
   - However, concerns were later raised about its creative writing and roleplay performance, with some noting *hybrid models lack the instruction following and creativity in the non-think mode*.
- **RTX 5090 Price Sparks Upgrade Debate**: The **RTX 5090** is priced around **$2000**, prompting discussions on whether to upgrade, especially for training, given its **VRAM** capabilities.
   - Some members expressed frustration with **NVIDIA's** limitations, particularly the lack of **P2P or NVLink**.
- **WildChat-4M-English Released**: The **WildChat-4M-English-Semantic-Deduplicated dataset** is available on Hugging Face, consisting of English prompts from the WildChat-4M dataset, deduplicated using multiple methods.
   - The current release includes prompts **<= ~2000 tokens**, with larger prompts to be added later, more information can be found [here](https://huggingface.co/datasets/MasonMac/WildChat-4M-English-Semantic-Deduplicated).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Deepseek V3.1 Craze Awaits!**: Users are eagerly awaiting the public release of **Deepseek v3.1**, anticipating it will be free starting in September.
   - Users confirm that paying for **Deepseek** models on **OpenRouter** results in faster response times compared to the free models.
- **OpenRouter API Keys Risk Exposure!**: A user reported a loss of **$300** due to a leaked **OpenRouter API key** and sought advice on identifying the source of the unauthorized usage.
   - Users are responsible for any leaked keys and threat actors can use proxies to mask their origin IPs.
- **Gemini faces massive banning outbreak!**: Users report massive banning occurring on **Gemini**, leading many to seek alternatives and reminisce about the AI Dungeon purge caused by OpenAI.
   - Users are saying *we're being sent back to 2023*.
- **Gemini Input Tokens Trigger Weird Counts!**: A dashboard developer noted that **OpenRouter's** calculation of **input tokens** for **Gemini's models** produces unusual counts when images are included in the input, referencing a related discussion on the [Google AI Developers forum](https://discuss.ai.google.dev/t/token-counts-mismatch-9x-discrepancy/72633/2).
   - The developer is considering seeking clarification from the OpenRouter team regarding this issue.
- **Most Orgs see ZERO return on Generative AI!**: According to an [AFR Chanticleer report](https://archive.md/IlP7F), **95% of organizations are getting zero return** out of their generative AI deployment, focused on companies that have deployed **customized AI models**.
   - The report notes that the key problem is companies and their tech vendors are not spending enough time ensuring that their customized AI models keep learning about the nuances of their businesses.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Claude's Cache Capriciousness Causes Costly Conundrums**: Users are reporting that **Claude** is experiencing issues with *cache reads*, leading to increased expenses compared to **Auto**, which benefits from sustainable caching.
   - Speculation arose around whether **Auto** and **Claude** are secretly the same model, attributing reduced token usage to a *placebo effect*.
- **Sonic Speedster Steals the Show in Cursor**: The community is currently testing the new **Sonic** model within Cursor, with initial impressions being quite favorable due to its speed.
   - While praised for fresh projects, some users cautioned its effectiveness might diminish with larger codebases and confirmed that **Sonic is not a Grok model** whose origin remains a *stealth company*.
- **Agentwise Awakens as Open Source Offering**: **Agentwise** has been open-sourced, enabling website replicas, image/document uploads, and support for over 100 agents, with promises of [Cursor CLI support](https://discord.com/channels/1074847526655643750/1408047562019049523).
   - Users are invited to contribute feedback in the project's dedicated Discord channel to help further development.
- **Cursor's Costs Confirmed: Clarity on API Charges**: Confusion around the cost of the Auto agent was cleared up, where a *pro* subscription includes the costs of API usage by different providers.
   - Several users confirmed the cost clarification, and one stated a preference of Auto agent over Sonic agent.
- **DeepSeek Debuts, Divides Developers**: The new **DeepSeek V3.1** model appeared in Cursor's options, eliciting mixed reactions; some users encountered connection issues, while others expressed distrust towards *Chinese LLMs*.
   - Despite concerns, some reported that DeepSeek V3.1 functions well with **TypeScript** and **JavaScript**, offering performance that is *great* and cheaper than Sonnet.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **CUDA Fix Drives 4070 Detection**: Users discovered that changing the runtime to **CUDA llama.cpp** via **ctrl+shift+r** might resolve the *"0 GPUs detected with CUDA"* error in LM Studio for **4070 TI Super** cards.
   - They discussed various configurations to enable **flash attention**, **quantization of KV cache**, and a **batch size of 2048** with commands like `-fa -ub 2048 -ctv q8_0 -ctk q8_0`.
- **GPT-OSS Smokes Qwen on Prompt Eval**: Members observed **GPT-OSS** reaching *2k tokens/s* on prompt eval with a **3080ti**, outperforming **Qwen's** *1000 tokens/s* in LM Studio.
   - A user reported LM Studio API calls were significantly slower (30x) than the chat interface but the issue resolved itself for unknown reasons when using the curl command `curl.exe http://localhost:11434/v1/chat/completions -d {"model":"gpt-oss:20b","messages":[{"role":"system","content":"Why is the sun hot?\n"}]}`.
- **Qwen3-30B CPU Configuration Surprises**: Using [llama-bench](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench), a user achieved **10 tokens per second** on a CPU-only configuration with **Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf**.
   - They noted that the performance varied based on thread count, with diminishing returns beyond a certain threshold because of scaling and overhead.
- **MLX's M4 Max Melts GGUF**: Benchmarking **GPT-OSS-20b** on an Apple M4 Max revealed that **MLX (GPU)** hit **76.6 t/s** at **32W (2.39 t/W)** compared to **GGUF (CPU)** which only achieved **26.2 t/s** at **43W (0.61 t/W)**.
   - With **4bit quants** and **4k context**, MLX proved slightly faster and more power-efficient than GGUF, although they were impressed by the GGUF performance.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Agents Dive into M2M Economies**: Members explored **machine-to-machine (M2M) economies**, where AI agents autonomously exchange value, focusing on challenges like *identity & trust, smart contract logic, and autonomy.*
   - Safeguards such as **spending caps, audit logs, and insurance** could accelerate AI adoption in transactions, but *real trust will still take time*.
- **Decentralized AI Project's BOINC Bounty**: A member sought a **decentralized AI project** like **BOINC**, noting challenges with the [Petals network](https://petals.ml/) related to contributions and model updates.
   - Contributors suggested **financial or campaign-driven incentives** could bolster decentralized AI development.
- **Few-Shot Fitness Prompts Flexed**: Members dissected optimal strategies for using **few-shot examples** within a **29,000 token prompt** for a fitness studio, emphasizing **prompt engineering**.
   - Recommendations included providing direct examples within the prompt and iteratively testing smaller chunks to enhance performance.
- **GPT-5's Thinking Mode Dumbs Down**: A user reported that **GPT-5's** *thinking* mode yields direct, **low-quality responses**, similar to an older model version, causing frustration.
   - Another member speculated the user may have exceeded a *thinking quota limit, with the system set to fallback instead of grey out*.
- **AI Quiz Generates Trivial Pursuit**: A member highlighted issues with an **AI quiz generator** producing obviously wrong answer choices in quizzes.
   - Another member suggested ensuring that *all response options must be plausible* to improve the AI's output and produce more realistic responses.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **PileT5-XL Speaks**: An embedding tensor from **PileT5-XL** works both as an instruction for **pile-t5-xl-flan** (which generates text) and as a prompt for **AuraFlow** (which generates images), suggesting these embeddings hold meaning like words in a language.
   - A member is interested in textual inversion with a black dog picture with auraflow applied to pile-t5-xl-flan to see if text describes the dog as black.
- **Cosmos Med Models Scale!**: The **Cosmos Medical Event Transformer (CoMET)** models, a family of decoder-only transformer models pretrained on **118 million patients** representing **115 billion discrete medical events** (151 billion tokens) generally outperformed or matched task-specific supervised models.
   - The study, discussed in [Generative Medical Event Models Improve with Scale](https://arxiv.org/abs/2508.12104), used **Epic Cosmos**, a dataset with medical events from de-identified longitudinal health records for **16.3 billion encounters** over **300 million unique patient records** from **310 health systems**.
- **ByteDance Prover Gets Medal**: **Bytedance's SEED Prover** achieved a [silver medal score in IMO 2025](https://seed.bytedance.com/en/blog/bytedance-seed-prover-achieves-silver-medal-score-in-imo-2025).
   - However, it is unclear how this translates to real world math problem solving performance.
- **Isolating a Llama3.2 Head**: A member isolated a particular kind of *head*, discovering that decoded result vectors between **Llama 3.2-1b instruct** and **Qwen3-4B-Instruct-2507** were remarkably similar across different outputs.
   - The member stated that *the two heads seem to promote are quite similar*.
- **Muon Kernel Support Sought**: A member expressed interest in adding **muon support**, citing potential **kernel optimization opportunities**.
   - They believe that once basic support is implemented, there's room for collaborative work on these optimizations.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Meta Splits After Wang Promotion**: Meta is reorganizing its AI efforts into **four teams** (TBD Lab, FAIR, Product/Applied Research, Infra) under new MSL leader **Alexandr Wang**, with the **AGI Foundations** group being disbanded, according to [Business Insider](https://www.businessinsider.com/meta-ai-superintelligence-labs-reorg-alexandr-wang-memo-2025-8).
   - **Nat Friedman** and **Yann LeCun** now report to Wang, **FAIR** will directly support model training, and an "omni" model is under consideration.
- **GPT-5-pro Silently Eats Prompts**: **GPT-5-pro** is silently truncating prompts greater than **60k tokens** without any warning or error messages, which makes large-codebase prompts unreliable, according to [this report](https://x.com/pvncher/status/1958193631250072024?s=46).
   - Some users are also reporting that **GPT-5** in **Cursor** is acting a lot dumber than usual, with some suspecting load shedding is occurring.
- **Dropout Inspired by Bank Tellers**: A viral tweet claims **Geoffrey Hinton** conceived *dropout* after noticing **rotating bank tellers** deterred collusion ([source](https://x.com/eigenron/status/1958181550987632927?s=46)).
   - Reactions range from admiration for the serendipitous insight to skepticism and jokes about attention mechanisms emerging from house parties.
- **ByteDance Sows Seed-OSS Models**: ByteDance’s Seed team has announced **Seed-OSS**, a new open-source large-language-model family available on [GitHub](https://github.com/orgs/bytedance/repositories) and [Hugging Face](https://huggingface.co/models).
   - The team is inviting the community to test and provide feedback on the models, code, and weights.
- **Wonda Promises Video Revolution**: Dimi Nikolaou introduced **Wonda**, an AI agent aiming to revolutionize video/audio creation, calling it *what Lovable did for websites, Wonda does for content* ([tweet link](https://xcancel.com/dimireadsthings/status/1957805267799740571)).
   - Early-access will be granted via a waitlist offering invites in approximately **3 weeks**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Confounds ChatGPT**: A member found that **ChatGPT** gave confidently incorrect answers regarding **CUDA float3 alignment** and **size**, and then attributed the difficulty of this topic to the complexities of **OpenCL** and **OpenGL** implementations.
   - The member has validated that there is no padding in **CUDA**.
- **Hackathon Starts Saturday AM**: The **GPU Hackathon** will *likely* kick off around **9:30 AM** on Saturday, and it was hinted that participants will be working with newer **Nvidia chips**.
   - There was a question about the hackathon prerequisites, but it went unanswered in the channel.
- **AMD GPU debugger has first alpha**: An engineer showed off the alpha version of their new **AMD GPU debugger** now with disassembly and wave stepping in [this video](https://cdn.discordapp.com/attachments/1233704710389764236/1407932291912695949/2025-08-21_06-20-14.mp4?ex=68a88f60&is=68a73de0&hm=6e9ca7ceed29674c6943e989940a6c8e144707797c5abd571e1cfe60d3f3210d).
   - This debugger doesn’t depend on the **amdkfd KMD**, using a mini UMD driver and the linux kernel debugfs interface and aiming for a **rocdbgapi** equivalent.
- **DIY Distributed Training Framework Emerges**: One member is in the process of building their own **pytorch distributed training library** and mini **NCCL** as a backend to be used with **infiniband** at home between a **4090** and **5090**.
   - Another member expressed interest, considering it to be a good way to study the finer points of distributed computing.
- **MI300 dominates Trimul Leaderboard**: The `trimul` leaderboard now features a submission score of **3.50 ms** on **MI300**, and another submission on **MI300** achieved second place with a score of **5.83 ms**.
   - A member achieved **6th place** on **B200** with a time of **8.86 ms** and later improved to **4th place** with **7.29 ms** on the `trimul` leaderboard, and another achieved **second place** on **H100** with a time of **3.80 ms**



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Forbes Finds Flaws, Frames Fracas!**: [Forbes](https://www.forbes.com/sites/iainmartin/2025/08/20/elon-musks-xai-published-hundreds-of-thousands-of-grok-chatbot-conversations/) revealed that **Elon Musk's xAI** published hundreds of thousands of **Grok** chatbot conversations.
   - When asked whether this was true, *@grok* responded evasively, leading to further speculation.
- **LeCun Leaving, Losing, or Loitering?!**: A user speculated about **Yann LeCun's** potential departure from **FAIR** based on [a post by Zuckerberg](https://www.threads.com/@zuck/post/DMiwjXJSYCd?xmt=AQF06RR3brSSoKTMcQrLDyX6WGtg9UdpubK94sSm8FIPsg).
   - Another member suggested **LeCun** may have been demoted and that **Meta** is retreating from the open source model space.
- **Infinite Memory Mandates Machine Mightiness!**: A member argues that Turing completeness requires infinite memory, thus the universe cannot create a Turing complete machine due to insufficient memory.
   - Another member jokingly suggests that making a computer sufficiently slow could allow the expansion of the universe to account for the space problem.
- **New Names, New Nuisance: AI Slurs Surface!**: A user shared [a Rolling Stone article](https://www.rollingstone.com/culture/culture-features/clanker-cogsucker-robot-ai-slurs-viral-1235401262/) discussing the emergence of new **AI slurs** like *clanker* and *cogsucker*.
   - Responses in the channel were muted, but all seemed to agree that such words are very naughty indeed.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Payment Issues Plague Hugging Face Pro Users**: A user reported being charged twice for the **Pro version** without receiving the service, advising others to email website@huggingface.co and seek assistance in the designated [MCP channel](https://discord.com/channels/879548962464493619/1389546106970701865).
   - The user was unable to get the **Pro** service despite repeated charges to their account.
- **AgentX Promises Smarter AI Trading**: The new [**AgentX** platform](https://www.linkedin.com/posts/alaa-salamah-96167b227_agentx-agentx-api-activity-7364245216851050498-BfRO?utm_source=share&utm_medium=member_desktop&rcm=ACoAADjfvpkBwpCXn_8Pmby7ixjV93Dje5TcmgUHi) aims to provide a trading table with the smartest AI minds—**ChatGPT**, **Gemini**, **LLaMA**, **Grok**—working together to debate until they agree on the best move.
   - The platform seeks to offer traders a system they can fully trust by having **LLMs** debate the best move.
- **Members Debate SFT versus DPO**: Members discussed the effectiveness of **DPO** (Direct Preference Optimization) versus **SFT** (Supervised Fine-Tuning), where one member noted that *DPO has no relationship to reasoning*, but **DPO** after **SFT** improves results over just **SFT**.
   - There was discussion on leveraging **DPO** to boost performance, however, the relationship to reasoning was debated among members.
- **HF Learn Course Plagued by 422 Errors**: A member reported that [a page from the Hugging Face LLM course](https://huggingface.co/learn/llm-course/en/chapter12/3a) is down and showing a **422 error**.
   - Users are currently unable to access the broken page within the Learn course.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Users Discover Gems to Streamline Podcast Generation**: Users are developing workflows, like [this example](https://github.com/para-droid-ai/scratchpad/blob/main/assistant-workflows-tasks-personas/deeper_podcast_synthetic-082025.txt), to create deeper research frameworks to generate podcasts with **Gems**, **Gemini**, **PPLX**, or **ChatGPT**.
   - The key is to set prompts to plan the entire transcript section by section, generating podcasts from longer **YouTube** videos.
- **Customize screen lets Users Configure Podcast Length**: Users can adjust podcast length in NotebookLM by using the **Customize** option (three dots), extending podcast length to **45-60 minutes**.
   - Specifying topics allows the bot to *concentrate on topics* instead of relying on it to fit all the important stuff into a single podcast.
- **Privacy Policy Paranoia Prevails**: Users are analyzing healthcare company's privacy policies and terms of use using **Gemini** and **NotebookLM**.
   - The user was surprised by *how much you give away to these companies* and how useful this method is to understand **Terms of Use** and **Privacy policies**.
- **Android App Feature Parity Delayed**: Users are requesting more **feature parity** between the NotebookLM web app and the **Android app**, especially for study guides.
   - One user stated the current native app is *borderline useless* because study guides depend on the notes feature, which is missing from the native app.
- **NotebookLM API Remains Elusive**: While an official API for NotebookLM is not available, users suggest using the **Gemini API** as a workaround.
   - Another user shared their strategy of combining **GPT4-Vision** and **NotebookLM** to *quickly digest complex PDF schematics with callouts*.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **ByteDance Unleashes Long Context Model**: ByteDance released a base model with extremely long context, featuring no **MHLA**, no **MoE**, and not even **QK** norm, according to [this image](https://cdn.discordapp.com/attachments/1149866623109439599/1407959284280459305/image.png?ex=68a8a883&is=68a75703&hm=b8a5430da1f445204c76334cef07358c8d9b815da989424483a5ecf3bc65c790).
   - The model's architecture was described as *vanilla*, prompting hopes for a forthcoming paper to provide further insights.
- **Seed-OSS-36B's GGUF Absence Sparks Speculation**: Users inquired about the absence of a **GGUF** for **Seed-OSS-36B**, noting their typical swift appearance, referencing [this link](https://x.com/adityastomar_/status/1958048129275805867) questioning the implications for **ASICs**.
   - It was suggested the delay could stem from a custom **vllm** implementation, with the architecture currently unsupported by **llama.cpp** due to `architectures: ["SeedOssForCausalLM"]`.
- **Seed Model Sports Dropout and Bias**: The **Seed** model incorporates a custom **MLP** and attention mechanism akin to **LLaMA**, yet features dropout, an output bias term, and a bias term for the **qkv** heads.
   - These additions are speculated to serve as regularization techniques; however, the number of epochs the model underwent remains unknown, with confirmations that simply renaming it to **LLaMA** will not yield functionality.
- **Qwen Scales to 512k Context with RoPE**: The **30B** and **235B Qwen 2507** models can achieve **512k** context using **RoPE** scaling, according to [this Hugging Face dataset](https://huggingface.co/datasets/eaddario/imatrix-calibration).
   - These datasets are used to generate importance matrices (**imatrix**), which help minimize errors during quantization.
- **Cursor's Kernel Blog Draws Applause**: Members shared a link to [Cursor's kernel blog](https://x.com/stuart_sul/status/1957927497351467372).
   - Many agreed that *cursor cooked* on that one.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **DeepSeek V3.1 Debuts with Mild Improvements**: The new **DeepSeek V3.1** model was released, with some members noting that it is like an *incremental improvement* with some regressions, referencing [DeepSeek's official page](https://huggingface.co/deepseek-ai/DeepSeek-V3.1).
   - Its performance is being closely watched in the community for subtle gains and potential drawbacks.
- **DeepSeek Courts Anthropic API Integration**: **DeepSeek** now supports the **Anthropic API**, expanding its capabilities and reach, as announced [on X](https://x.com/deepseek_ai/status/1958417062008918312).
   - This integration enables users to use **DeepSeek** with **Anthropic's** ecosystem, promising versatility in AI solution development.
- **R-Zero LLM Evolves Sans Human Data**: A comprehensive study of **R-Zero**, a self-evolving **LLM training method** that starts from zero human data and improves independently, was shared in a [PDF](https://cdn.discordapp.com/attachments/1371757564005711973/1408153973751545896/R-Zero__A_Comprehensive_Study_of_a_Self-Evolving_LLM_Training_Method.pdf?ex=68a8b515&is=68a76395&hm=dcba93436f636eeec364d08d08e1603131147faaa595637065f2e226772005f2&).
   - The approach marks a departure from traditional **LLM training**, potentially reducing reliance on human-labeled datasets.
- **China Sidesteps Data Center Energy Dilemma**: A member noted that in China, *energy availability is treated as a given*, contrasting with U.S. debates over data center power consumption and grid limits, referencing [this Fortune article](https://fortune.com/2025/08/14/data-centers-china-grid-us-infrastructure/).
   - The difference in approach could give Chinese AI firms a competitive advantage in scaling energy-intensive models.
- **Kimi K2 Eyes Better Image Generation**: A member noted that **Kimi K2** would be more OP if it got combined with **Better image gen than gpt 5**, with [this reddit link](https://www.reddit.com/r/ChatGPT/s/vUrGedSwY5) shared.
   - Integrating enhanced image generation capabilities would position **Kimi K2** as a more versatile and competitive AI assistant.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro Stumbles While Flash Soars**: A user reports that **Gemini 2.5 Flash** is functional, whereas **Gemini 2.5 Pro** consistently fails, however `gemini/gemini-2.5-pro-preview-06-05` operates when billing is configured.
   - Another reported a **$25** charge for a **qwen-cli** process and is requesting a refund, highlighting potential inconsistencies in model performance and billing.
- **User Hit With Unexpected Qwen CLI Charges**: A user incurred a **$25** charge for using **qwen-cli** after Google OAuth authentication, expecting free credit from Alibaba Cloud.
   - Opening a support ticket, they cited a console usage of *one call of $23 with no output* to dispute the unexpected charge.
- **Community Benchmarks GPT-5 Mini Models**: Community members are actively benchmarking **gpt-5-mini** and **gpt-5-nano** because of rate limits on the full **gpt-5**, and one user claims *gpt-5-mini is very good and cheap*.
   - Benchmark results and a PR for **gpt-5-mini** are available, reflecting the community's interest in evaluating smaller, more accessible models.
- **DeepSeek v3.1 Pricing Sees a Bump**: Starting Sept 5th, 2025, DeepSeek will increase pricing to **$0.25 vs $0.27** for input on both models to match the reasoner model price.
   - The price increase to match the **deepseek 3.1** model reflects changes in pricing strategy.
- **OpenRouter Needs a "Think" Mode**: Users noted that **OpenRouter** lacks a native "think" mode for enhanced reasoning, but it can be enabled via command line using: `aider --model openrouter/deepseek/deepseek-chat-v3.1 --reasoning-effort high`.
   - Community members suggested updating the model configurations to address this functionality gap.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Marimo Notebooks Rise as Jupyter Alternative**: A member published [tutorials on **marimo notebooks**](https://www.youtube.com/watch?v=2aepn9uRVOM), highlighting its use in iterating through ideas on **Graph RAG with DSPy**, as a notebook, script and app, all at once.
   - Upcoming videos will explore **DSPy modules** optimization, building on the current tutorial that introduces **marimo** to new users.
- **Readability Debate: DSPy Code Assailed then Upheld**: After a member dismissed **IBM's AutoPDL** claims about unreadability, they defended **DSPy's code** and **prompts** as extremely human-readable and clear.
   - The defense emphasized the accessibility of the code, making it easy to understand and work with.
- **GEPA Arrives in DSPy v3.0.1**: Members confirmed that **GEPA** is available in **dspy** version **3.0.1**, as shown in the attached [screenshot](https://cdn.discordapp.com/attachments/1161519469319946286/1407936409615990904/image.png?ex=68a89336&is=68a741b6&hm=72219936a525599fc3faca9d127d106a09e0639e9eeae1564a8bbfc196b07ffa&).
   - During fine-tuning, a member inquired about whether it is common to use *"vanilla descriptions"* for **dspy.InputField()** and **dspy.OutputField()** to allow the optimizer to think freely.
- **Pickle Problem: DSPy Program Not Saved**: A user reported issues with saving an optimized program, noting that the metadata only contained dependency versions but not the program itself, even when using `optimized_agent.save("./optimized_2", save_program=True)`.
   - When another user set the maximum context length to **32k** for **GEPA** but still received cut-off responses, members discussed the complexities of long reasoning and potential issues with multi-modal setups.
- **RAG vs Concatenation: Million-Document Debate**: Members debated whether **RAG** (Retrieval-Augmented Generation) or simple **concatenation** would be more appropriate for tasks like processing tax codes or crop insurance documents.
   - The debate acknowledged that while **RAG** is often seen as overkill, the scale of millions of documents can sometimes justify its use.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command A Reasoning Unleashed**: Cohere launched **Command A Reasoning**, designed for enterprise, outperforming other models in agentic and multilingual benchmarks; available via [Cohere Platform](https://dashboard.cohere.com/playground/chat?model=command-a-reasoning-08-2025) and [Hugging Face](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025).
   - It runs on a single **H100** or **A100** with a context length of **128k**, scaling to **256k** on multiple GPUs, according to the [Cohere blog](https://cohere.com/blog/command-a-reasoning).
- **Command's Token Budget Saves the Day**: **Command A Reasoning** features a **token budget** setting, enabling direct management of compute usage and cost control, making separate reasoning and non-reasoning models unnecessary.
   - It is also the core generative model powering **North**, Cohere's secure agentic AI platform, enabling custom AI agents and on-prem automations.
- **Command-a-03-2025 Gives Intermittent Citations**: `command-a-03-2025` is returning citations only intermittently, even with the maxTokens set to 8K, causing trust issues in production.
   - A Cohere member clarified that it uses *"fast"* mode for citations (as per [the API reference](https://docs.cohere.com/reference/chat#request.body.citation_options.mode)) and citations aren't guaranteed; use **command-a-reasoning** instead.
- **Langchain RAG in the Works**: A member is learning Langchain to build an RAG (Retrieval-Augmented Generation) application, with the intention to use **command-a-reasoning**.
   - They anticipate the release of **command-a-omni**, and expressed hype for a future model called **Command Raz**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Clients Flout Instructions Field**: Members are reporting that **MCP clients**, specifically **Claude**, are ignoring the **instructions field** and only considering **tool descriptions**.
   - One member suggested that *adding the instruction, context and then repeating the instruction would yield better results* but this is not possible with integrated APIs, while another suggested the **MCP server** should prioritize processing **tool descriptions**.
- **Diverse MCP Servers in Action**: Members are sharing their preferred **MCP server** setups and tools including GitHub for version control, Python with FastAPI for backend development, and PyTorch for machine learning.
   - One user sought advice on how to make an agent follow a specific **generate_test_prompt.md** file, linking to a [screenshot](https://cdn.discordapp.com/attachments/1312302100125843476/1408171379236409354/Screenshot_3.png?ex=68a8c54b&is=68a773cb&hm=1fbd862963889b97ef764b1599c2960340dab7ce357b3717f577e2b1491ffdc2) of their configuration.
- **Web-curl Unleashes LLM Agent Prowess**: **Web-curl**, an open-source **MCP server** built with Node.js and TypeScript, empowers LLM agents to fetch, explore, and interact with the web & APIs with source code available on [GitHub](https://github.com/rayss868/MCP-Web-Curl).
   - Functionally, **Web-curl** enables LLM agents to fetch, explore, and interact with the web & APIs in a structured way.
- **MCP-Boss Centralizes Key Management**: A member introduced **MCP Boss** to centralize key management, providing a single URL to gateway all services, featuring multi-user authentication and MCP authorization via OAuth2.1 or static HTTP header.
   - More information available at [mcp-boss.com](https://mcp-boss.com/).
- **AI Routing Power in MCP Gateway**: A member introduced a lightweight gateway with **AI-powered routing** to solve the problem of agents needing to know which specific server has the right tool, with code available on [GitHub](https://github.com/oliverye7/mcp-gateway).
   - By using the gateway, **MCP routing** can be solved by using an AI.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Celebrates Modverse Milestone**: Modular released [Modverse #50](https://www.modular.com/blog/modverse-50) and announced a custom server tag as seen in [Screenshot_2025-08-21_at_5.22.15_PM.png](https://cdn.discordapp.com/attachments/1098713601386233997/1408199603861323878/Screenshot_2025-08-21_at_5.22.15_PM.png?ex=68a8df94&is=68a78e14&hm=2991584cc0b81449dbc278d1d8302e55aabc54c58a6620e7041deb9dbd20e951&).
   - The custom server tag has been deployed.
- **Documentation drought plagues kgen and pop**: Members report a lack of documentation for **kgen** and **pop**, particularly regarding operations and parameters, with one stating *there’s no comprehensive documentation of the internal MLIR dialects*.
   - A link to the [pop_dialect.md](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/internal/pop_dialect.md) on Github was shared, clarifying that these are part of the contract between the stdlib and the compiler, *so use them outside of the stdlib at your own risk*.
- **POP Union Faces Alignment Allegations**: Suspicions have arisen regarding an alignment bug in **pop.union**, as indicated by unexpected size discrepancies when employing `sizeof`.
   - A member created [issue 5202](https://github.com/modular/modular/issues/5202) on GitHub to investigate the suspected alignment bug in **pop.union**, also observing that **pop.union** doesn't appear to be used anywhere.
- **TextGenerationPipeline Execute Hides In Plain Sight**: A member located the `execute` method on `TextGenerationPipeline` and linked to the [relevant line in the Modular repo](https://github.com/modular/modular/blob/fe7ba5d5f8b08ccba7356c45b90afcf495421469/max/pipelines/lib/pipeline.py#L977).
   - They suggested checking the MAX version.
- **Memory Allocators Loom Large**: One member suggested that robust allocator support might be necessary before memory allocators are integrated into the language, as most users don't want to manually handle out-of-memory (**OOM**) errors.
   - These comments were made in the context of other struggles, with one member reporting struggling with retrieving the **logits** along with the next token while creating a custom inference loop and linked to a [Google Docs document](https://docs.google.com/document/d/1Hd6xZnf0bmg9SMQU1h10cd4Cwd9HDOPzPHZNqiPnrpg/edit?tab=t.0) for context.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Debuts Enterprise Document AI**: LlamaIndex's VP of Product previews enterprise learnings about parsing, extracting, and indexing [documents](https://t.co/x70xjEQaFs) on **September 30th** at **9 AM PST**.
   - The focus is on how LlamaIndex addresses real-world document challenges.
- **vibe-llama Cli Tool Configures Coding Agents**: LlamaIndex launched **vibe-llama**, a CLI tool that automatically configures coding agents with context and best practices for the **LlamaIndex framework** and **LlamaCloud**, detailed [here](https://t.co/G1gINq9kge).
   - The goal is to streamline development workflows.
- **CrossEncoder Class: Core vs Integrations**: A member inquired about the duplicated **CrossEncoder class** implementations in `llama-index`, specifically under `.core` and `.integrations` ([code link](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/sbert_rerank.py)).
   - It was clarified that the `.core` version is a leftover from the v0.10.x migration, with the recommendation to use `llama_index.postprocessor.sbert_rerank` with `pip install llama-index-postprocessor-sbert-rerank`.
- **Quest for Agent Creation Gateway**: A member sought existing projects serving as a **gateway** that ties together **model, memory, and tools**, exposing an **OpenAI-compatible endpoint**.
   - They wanted to avoid reinventing the wheel in agent explorations.
- **AI Safety Survey Gathers Community Opinions**: A member shared an [AI safety survey](https://mukullight.pythonanywhere.com/form) to collect community opinions on important **AI safety questions**.
   - The survey aims to understand what the **AI safety community** finds most interesting.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Report Missing Credit Purchase Option**: Members have reported that the option to buy extra credits is missing, with users only seeing the *upgrade package* option.
   - It was confirmed that the option is currently *down right now*.
- **Support Tickets Go Unanswered**: A user reported an issue with a task and creating ticket **#1318**, but has not received a response or access to the ticket.
   - They requested assistance from the team, tagging a specific member.
- **Contest Winner Draws Rigging Allegations**: A user alleges that the second-place winner in a contest *didn’t deserve to win* and claims the contest *seems rigged*.
   - No further evidence or details were provided to support this claim.
- **Free Daily Credits Discontinued?**: A returning user noticed they didn't receive the usual **300 free credits daily**.
   - They inquired whether Manus had stopped providing these credits.
- **Referral Credits Code Confusion**: A user asked how to claim referral credits, noting that the system asks for a code.
   - The user stated they didn't know where to find the required code.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Overworld Const Folding Explored**: A member explored **overworld const folding** and a potential **view(const) refactor**, redefining `UPat.cvar` and `UPat.const_like` to match `CONST` and `VIEW(CONST)` in [this discord thread](https://discord.com/channels/1068976834382925865/1255400554012741683/1407782654958506004).
   - The aim is to fold expressions like `x * 0`, however, concerns were raised about validity and `.base` proliferation in symbolic computations.
- **ALU View Pushing as Alternative**: An alternative approach was suggested involving adding a upat in kernelize that pushes views directly onto **ALUs**, mirroring **S-Lykles's method**.
   - This method and a special rule for `x * 0` would allow unmodified symbolic matching, given the computational irrelevance of `* 0`.
- **base Removal Advocated**: A member strongly advised against the proposed approach, deeming it *"super ugly"* and advocating for the **removal of `.base`**.
   - The discussion also questioned the handling of **PAD** operations within this context.
- **RANGEIFY=1 Simplifies Implementation**: It was suggested that setting **RANGEIFY=1** could lead to a cleaner implementation.
   - However, the project is currently in a transition phase where both the old engine and rangeify are coexisting, creating a state of limbo.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4ALL Free Tier Enables Private AI**: A user inquired about using **GPT4ALL** for companies that wanted to use their **AI model privately and securely**.
   - Another member clarified that the **free version** suffices if the company already has its own **AI model ready**.
- **User Asks for LocalDocs Model**: A user seeks a model recommendation for building a personal knowledge base from hundreds of **scientific papers in PDF format** using **GPT4All's LocalDocs feature**.
   - The user specified they have an **Nvidia RTX 5090** with **24 GB VRAM** and **64 GB RAM** and would appreciate **reasoning capabilities** in the chosen model.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1407801395884720330)** (951 messages🔥🔥🔥): 

> `nano-banana model, Video Arena problems, DeepSeek V3.1, Gemini 3` 


- **Nano-Banana's McLau's Law unveiled**: A member joked that **Nano-Banana** often yields results *far below nano-banana*, terming this phenomenon "**McLau's Law**" in a humorous nod to one of **OpenAI's** researchers.
   - Attached was a [humorous image](https://cdn.discordapp.com/attachments/1340554757827461211/1407957527987097642/RDT_20250821_0619535468074318472918433.jpg?ex=68a8a6e1&is=68a75561&hm=bfcfd37e574ea905e84f2ff1c9e6ffee1855681ad903050cfe30a367ea5d96f3&) prompting discussion about **AI's** current capabilities.
- **Video Arena struggles with Bot Downtime**: Members reported issues with the **Video Arena**, citing inability to use commands or generate videos, with moderators confirming the bot's downtime and ongoing fixes.
   - Repeated queries about video creation access were met with explanations about the **bot's** temporary unavailability, directing users to the announcements channel for updates.
- **DeepSeek V3.1 enters the Arena**: Users discussed the introduction of **DeepSeek V3.1** to the platform, with one user describing the new model as *slightly worse version of Gemini 2.5 pro*.
   - However, the consensus is that it has potential as a coding model, but requires further general abilities.
- **Gemini 3 is Coming, claims user**: While not confirmed, a user hinted at the impending release of **Gemini 3**, speculating a launch date mirroring the **Google Pixel event**, generating anticipation among members.
   - The user did not cite any source and the claim was quickly dismissed by other community members. 
- **Site Outage Wipes Chats**: Users reported widespread data loss following a site outage, including missing chat histories and inability to accept terms of service, prompting moderator acknowledgement and assurances of a fix.
   - The moderator also said that a login feature will be available soon to prevent this sort of thing from happening again.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1408069950391980122)** (2 messages): 

> `Video Arena Bot, Deepseek v3.1, LMArena Models` 


- ****Video Arena Bot** down, channels locked**: The **Video Arena Bot** is currently not working, locking access to the prompt channels <#1397655695150682194>, <#1400148557427904664>, and <#1400148597768720384>.
   - The bot must be online to prompt in those specific channels.
- ****DeepSeek v3.1** Added to LMArena**: Two new models have been added to LMArena: **deepseek-v3.1** and **deepseek-v3.1-thinking**.
   - These models are now available for use in the arena.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1407802780516614178)** (887 messages🔥🔥🔥): 

> `ByteDance Seed Model, GRPO Training, DeepSeek V3.1 Quants, Nvidia's GPUs and Pricing, GLM-4.5 Cline Integration` 


- **ByteDance Releases Seed-OSS 36B Base Model**: ByteDance released the **Seed-OSS-36B-Base-woSyn** model on Hugging Face, a **36B** dense model with **512K** context window and explicitly claims *no synthetic instruct data* making it an interesting base for further tunes.
   - Members expressed excitement, noting it differs from models like **Qwen3**, and some are eager to try tuning GPT-ASS with it after their datasets are complete, despite the model being trained on *only* **12T tokens**.
- **GRPO Training Requires Smart Dataset Design**: To use GRPO for multi-step game actions, members advised designing datasets with separate prompts for each step, such as **[['step1 instruct'], ['step1 instruct', 'step1 output', 'step2 instruct']]**, and implementing a reward function to match the outputs.
   - It was noted that Full PPO might be better suited for games, as GRPO is primarily effective for LLMs because *they roughly know what to do to begin with*.
- **DeepSeek V3.1 Sweeps Leaderboard in Thinking and Non-Thinking Modes**: The **DeepSeek V3.1** model has shown competitive results, achieving a **66** on SWE-bench verified in non-thinking mode, with members expressing hype and comparing it to **GPT5** medium reasoning.
   - Although initially hyped, discussions later mentioned concerns about its performance in creative writing and roleplay, with some noting *hybrid models lack the instruction following and creativity in the non-think mode*.
- **Nvidia's RTX 5090 Prices Settle, Sparking Upgrade Debates**: The **RTX 5090** is now priced around **$2000**, prompting discussions on whether to upgrade, especially for training purposes given its **VRAM** capabilities, while others suggested sticking with **3090s** or waiting for the **RTX 6000**.
   - Some members expressed frustration with **NVIDIA's** limitations, particularly the lack of **P2P or NVLink**, with one member joking, *if you sit on a 5090 you will game on it*.
- **High Quality Imatrix Calibration Data is Key**: Members noted that WikiText-raw is considered a *bad* dataset for calibrating imatrices, because the imatrix needs to be well diversified and trained on examples in the model's native chat-template format.
   - Instead, [Ed Addorio's latest calibration data](https://huggingface.co/datasets/eaddario/imatrix-calibration) with Math, Code, and Language prompts, can improve and help preserve the models understanding of multiple languages if done correctly.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

.zackmorris: Hello
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1407836226488111114)** (27 messages🔥): 

> `GRPO 20mb alloc fail, ChatGPT's deep research, Grok-4, Repetition penalty, RAG` 


- ****GRPO 20MB Alloc Fails Plague Gemma Model!****: A user reported frequent **20MB allocation failures** with **GRPO** while working on [gemma-3-4b-it-unslop-GRPO-v3](https://huggingface.co/electroglyph/gemma-3-4b-it-unslop-GRPO-v3).
- ****ChatGPT's Deep Thought Mode Boosts Performance!****: A user suggested enhancing **ChatGPT's** performance by enabling web search and adding *"use deep thought if possible"* to prompts, even without full deep research.
- ****Grok-4 Puts in the WORK!****: A user was impressed by **Grok-4**, suggesting they might have secretly been using **Grok-4-Heavy**.
- ****Repetition Penalty Hilarity Ensues****: A user shared an image to demonstrate the importance of the **repetition penalty** parameter.
- ****RAG assistance****: A user asked for help working with **RAG**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1407822574725107743)** (101 messages🔥🔥): 

> `Retinal Photo Training Strategies, GPT-OSS 20B Deployment on Sagemaker, Unsloth Zoo Issues, GGUF Loading with Unsloth, Gemma 3 Vision Encoder Training Loss` 


- **Tuning Vision-Text Encoders for Retinal Photos**: A user questioned whether it's better to train a custom vision-text encoder for retinal photos or use mainstream models with Unsloth, noting that **retinal photos aren't well-represented in training datasets**.
   - It was suggested to experiment with computer vision models, transfer learning on similar datasets, and multimodal approaches, with synthetic clinical note generation using prompt engineering and personas.
- **Troubleshooting GPT-OSS 20B Sagemaker Deployment**: A user encountered a `ModelError` when deploying **unsloth/gpt-oss-20b-unsloth-bnb-4bit** on Sagemaker, receiving a **400 error** and InternalServerException with message `\u0027gpt_oss\u0027`.
   - It was mentioned that the model doesn't work on AWS Sagemaker and suggested deploying GGUFs or normal versions, using LMI Containers and pointed the user to [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-container-docs.html).
- **Unsloth Zoo installation issues**: A user experienced issues with **unsloth-zoo** even after installation in a Sagemaker instance, encountering import errors.
   - The user resolved it by removing all packages, then reinstalling Unsloth and Unsloth Zoo alongside JupyterLab, also needed to update Unsloth and refresh the notebook.
- **Quantization Concerns for Apple Silicon Macs**: A user sought guidance on which **GGUF quantization** is best for M series Apple Silicon, noting Macs are optimized for **4-bit** and **8-bit** computation.
   - It was suggested that users go for **Q3_K_XL**, or **IQ3_XXS** if context doesn't fit in memory, and that Q3-4 quants can be performant, but if using GGUFs it doesn't matter as much.
- **GPT-OSS Gains Multimodal with LLaVA**: A user asked why the vision llama13b notebook does not work for gpt-oss-20b and wondered if anyone was able to do it.
   - It was clarified that GPT-OSS is text-only and not a vision model so it won't work, and to add vision support, users would have to attach their own **ViT module**, like it is done in LLaVA using [LLaVA Guides](https://github.com/haotian-liu/LLaVA).


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1407927838123888651)** (11 messages🔥): 

> `WildChat-4M-English-Semantic-Deduplicated dataset, Behemoth-R1-123B-v2 model, GPU Rich Flex` 


- **Dataset of English prompts from WildChat-4M Released**: The **WildChat-4M-English-Semantic-Deduplicated dataset** is available on Hugging Face, consisting of English prompts from the WildChat-4M dataset, deduplicated using multiple methods including semantic deduplication with **Qwen-4B-Embedding** and **HNSW**.
   - The current release includes prompts **<= ~2000 tokens**, with larger prompts to be added later, more information can be found [here](https://huggingface.co/datasets/MasonMac/WildChat-4M-English-Semantic-Deduplicated).
- **TheDrummer Releases Behemoth-R1-123B-v2**: The **Behemoth-R1-123B-v2** model, created by TheDrummer, has been released, which can be found [here](https://huggingface.co/TheDrummer/Behemoth-R1-123B-v2).
   - A member noted that it's wild to be able to set up your hardware in HF.
- **GPU Rich is the New Flex**: A member shared an image depicting shaming if you're poor but flexed **GPU Rich**.
   - It's a flex to see GPU in **TFLOPS**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1407840310024995026)** (7 messages): 

> `Qwen3-4B finetuning, TTS with Gemini 270m, Mixture Models, JetMoE, BAM` 


- ****Unsloth** + **Qwen3-4B**: A Winning Combo?**: A member is using **Unsloth** to finetune on **Qwen3-4B** and will share the results, including evaluations, once completed; tuning went fine.
   - Another member wished good luck!
- **Training Model From Scratch**: A member is **22%** through training a proof of concept model from scratch, using a self-built dataset of year 6 maths with **500k** of sample data.
   - If successful, they'll expand the dataset to other subjects.
- **Text-to-Speech Dreams with Gemini 270M**: A member wants to try a **TTS** concept with **Gemini 270m** and hopes to start before the end of the month.
   - They are inspired by mixture model papers.
- **Experts Debate Merged Model Weakness on HumanEval**: One member cited the [JetMoE paper](https://arxiv.org/pdf/2404.07413#page=9.56) on mixture models trained from scratch, noting they performed poorly on **HumanEval** despite outperforming baselines elsewhere.
   - They also mentioned [BAM](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2408.08274), where pre-trained models were copied and trained on different domains, then combined, also losing percentage points on coding.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1408170025436844156)** (1 messages): 

> `Cloudflare outage, Generations API stability` 


- **Generations API Hit by Cloudflare Hiccups**: The **Generations API endpoint** experienced a temporary disruption due to issues with upstream infrastructure providers, causing **404 errors** for some calls.
   - The announcement indicated that the issue was related to intermittent problems with **Cloudflare**, but the **Generations API** has since been restored to a healthy state.
- **Retryable Restorations**: Calls to that endpoint may **404** but should be **re-tryable soon**.
   - The announcement assured users that the service would be restored quickly, advising them to retry any failed calls.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1408135423468765276)** (4 messages): 

> `OpenRouter Cost Dashboard, Average Request Size, Gemini Input Token Calculation` 


- ****Cost Reports get Visualized!****: A member has developed a free dashboard to visualize `.csv` cost reports from [OpenRouter](https://openrouter.ai/), designed to analyze data from shared accounts.
   - The dashboard, available at [openroutercosts.lorenzozane.com](https://openroutercosts.lorenzozane.com/), is planned to include additional **KPIs** and enhanced charts, with feedback welcome.
- ****Average Request Size requested in Dashboard!****: A member requested the addition of **average request size** metrics, specifically **average input tokens** and **average output tokens**, to the OpenRouter cost dashboard.
   - The dashboard's developer committed to adding this feature soon.
- ****Gemini Input Tokens trigger Weird Counts!****: The developer of the dashboard noted that **OpenRouter's** calculation of **input tokens** for **Gemini's models** appears to produce unusual counts when images are included in the input.
   - They are considering seeking clarification from the OpenRouter team regarding this issue, referencing a related discussion on the [Google AI Developers forum](https://discuss.ai.google.dev/t/token-counts-mismatch-9x-discrepancy/72633/2).


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1407830899223036106)** (528 messages🔥🔥🔥): 

> `Deepseek pricing, OpenRouter rate limits, Gemini banning, Using OpenRouter with RAG systems, 4.6T parameter model` 


- **Deepseek V3.1 Public Release Imminent!**: Many users eagerly await the public release of **Deepseek v3.1**, craving it *like fent* and anticipating it will be free starting in September.
- **Paid Deepseek offers Faster Responses**: Users confirm that paying for **Deepseek** models on OpenRouter results in faster response times compared to the free models, with one user switching due to **Chutes** slowing responses, but the user experience on the free models are not as good due to constant rate limits.
   - One user stated, *ever since that thing with chutes slowing responses I just said screw it i pay for it*.
- **OpenRouter API Keys Vulnerable to Leaks and Exploits**: A user reported a loss of **$300** due to a leaked OpenRouter API key and sought advice on identifying the source of the unauthorized usage, but it's possible for threat actors to use a proxy to mask their origin IP and the user is responsible for any leaked keys.
- **Is Gemini Doing the Banning Tango?**: Users report massive banning occurring on **Gemini**, leading many to seek alternatives and reminisce about the AI Dungeon purge caused by OpenAI.
   - One user lamented, *we're being sent back to 2023*.
- **OpenRouter API keys can be used in RAG?**: Users discuss the possibility of using **OpenRouter LLM API keys in RAG systems** with locally stored vector databases created by Milvus.
   - The consensus is that it's possible, but OpenRouter doesn't directly support embeddings, so you'll have to retrieve documents using milvus and put it with your prompt question to the OpenRouter LLM API.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1407869061840506900)** (3 messages): 

> `` 


- **Readybot.io Announces OpenRouter - New Models**: Readybot.io has announced updates and information regarding **new models** available on the **OpenRouter** platform.
- **OpenRouter's New Models Updates**: The **OpenRouter** platform highlights the latest additions and changes to its selection of **AI models**, as announced by Readybot.io.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1407806939878129774)** (16 messages🔥): 

> `Qwen3 coder 480b, DeepSeek v3 0324, Zero return from generative AI, Google Gemini 400 Error, Cohere reasoning model` 


- **LLMs struggle to format output correctly**: Users are finding that [LLMs like **Qwen3 coder 480b** and **DeepSeek v3 0324**](https://link.to.example) struggle to follow instructions for formatting their output properly, often resulting in bugs and ignored prompts.
   - One user found them *not useful* and *rather distracting*, often creating tic-tac-toe sites instead of the intended application.
- **Most orgs see ZERO return on Generative AI**: According to an [AFR Chanticleer report](https://archive.md/IlP7F), **95% of organizations are getting zero return** out of their generative AI deployment.
   - The report notes this is focused on companies that have deployed **customized AI models**, and the key problem is companies and their tech vendors are not spending enough time ensuring that their customized AI models keep learning about the nuances of their businesses.
- **Google Gemini Models trigger 400 Error**: **Google Gemini** models return **HTTP 400 errors** when assistant messages with tool calls use the **OpenAI-standard complex content format** `[{"type": "text", "text": "..."}]` instead of simple string format.
   - This issue affects all `google/gemini-*` models and only occurs when tool calls and tool results are present in the message chain.
- **Cohere Releases Reasoning Model**: [Cohere just dropped a reasoning model](https://cohere.com/blog/command-a-reasoning) with further details available on [Discord](https://discord.com/channels/954421988141711382/996880279224451154/1408103056800874497).
   - No further details were available.
- **Feature Request: Auto-Collapse lengthy user messages**: A user requested if it's possible to automatically collapse lengthy user messages in the chatroom.
   - The user praised the chatroom and the chat management.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1407803160356982795)** (432 messages🔥🔥🔥): 

> `Claude Cache Reads, Sonic Model origin, Open Sourcing Agentwise, Cursor API costs with Auto agent, DeepSeek V3.1` 


- **Cache Troubles Plague Claude**: Users report that **Claude** is currently *broken on cache reads*, leading to increased costs compared to **Auto**, which has sustainable caching.
   - One user mused whether **Auto** and **Claude** are secretly the same model, attributing reduced token usage to a placebo effect.
- **Sonic booms into Cursor IDE**: The community is testing the new **Sonic** model in Cursor, with one user reporting it is *pretty neat* and very fast, while another called it good for a fresh project but bad for a project with a large codebase.
   - The model's origin is a *stealth company*, and one member confirmed that **Sonic is not a Grok model**.
- **Agentwise Goes Open Source**: A member announced the open-sourcing of **Agentwise** which allows for website replicas, image/document uploads, and support for over 100 agents, with a promise of [Cursor CLI support](https://discord.com/channels/1074847526655643750/1408047562019049523).
   - Members are encouraged to provide feedback in the project's Discord channel.
- **Cursor API cost clarification**: The user's confusion around cost of Auto agent was cleared, where it was confirmed that with a "pro" subscription, there are **no extra fees**, only costs of API usage by different providers that are absorbed by the subscription.
   - One user found the Auto agent preferable to the Sonic agent.
- **DeepSeek V3.1 enters the Arena**: Users noticed the new **DeepSeek V3.1** model in Cursor's options, but some had trouble connecting to the provider, with one saying that *they don't trust chinese LLMs*.
   - However one member reported that DeepSeek V3.1 works fine with **TypeScript** and **JavaScript**, even performing *great* while still being cheaper than Sonnet.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1407802650908688424)** (11 messages🔥): 

> `Agent Auditing, MySQL Installation in Background Agents, Background Task Errors, Remote IDE connection to Background Agent` 


- ****Agent's Self-Audit** Fixes Issue**: A user reported fixing an issue by requesting the agent to commit and push the new branch, noting it seemed like an internal recurring problem.
   - Another user confirmed this was an audit, explaining it as the agent auditing itself using an **AI-GPL licenced auditing PDCA process framework**.
- ****MySQL** Config in Agents Clarified**: A user inquired about installing **MySQL** in background agents, questioning if it's pre-installed or limited to **SQLite** like Codex.
   - Another user clarified that **MySQL** is not installed by default, but can be added to the agent’s environment via `environment.json` or a **Dockerfile**.
- ****Background Task** Error Troubleshooted**: A user reported consistently getting an error immediately after starting a Background Task, even from the web, and provided a [screenshot](https://cdn.discordapp.com/attachments/1367213641027551352/1408202779096383550/Screenshot_2025-08-21_at_4.34.24_PM.png?ex=68a8e289&is=68a79109&hm=313d4bdb3a6bb89b6beeb5e9ffb22927afd3259ca9dc351a930226cbb122227c&).
- **Confusion Surrounds **Remote IDE** Connection**: A user sought clarity on connecting a **remote IDE** instance to the remote machine, referencing the documentation but finding the instructions unclear.
   - They questioned if a dummy background agent was necessary to facilitate this connection.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1407801641675260104)** (141 messages🔥🔥): 

> `CUDA Errors with 4070 TI Super, LM Studio multi-GPU performance, SerpAPI integration with LM Studio, GPT-OSS Performance, Model parameter configuration for VRAM usage` 


- **CUDA Driver needed to fix detection of 4070**: A user with a **4070 TI Super** reported a *"0 GPUs detected with CUDA"* error in LM Studio, and another user suggested changing the runtime to **CUDA llama.cpp** to potentially resolve the issue, by pressing **ctrl+shift+r**.
- **Flash Attention plus KV Quantization Dramatically Reduces VRAM**: A member suggested using commands `-fa -ub 2048 -ctv q8_0 -ctk q8_0` to enable **flash attention**, **quantization of KV cache**, and a **batch size of 2048**.
   - Also to increase the `-n-cpu-moe` value to manage VRAM usage, noting this only impacts speed.
- **GPT-OSS Blows Away Qwen on Prompt Eval**: Members noted **GPT-OSS** achieves *2k tokens/s* on prompt eval with a **3080ti**, while **Qwen** gets around *1000 tokens/s*.
- **Bolt.new is Cloud only**: A user inquired about setting up Bolt.new with LM Studio, but another user clarified that [Bolt is cloud-only](https://github.com/stackblitz-labs/bolt.diy) and does not support local models.
- **LM Studio API calls are slow like molasses**: A user reported that LM Studio API calls were significantly slower (30x) than the chat interface, a problem that then resolved itself for unknown reasons - the issue is possibly unconfigurable.
   - They used the curl command `curl.exe http://localhost:11434/v1/chat/completions -d {"model":"gpt-oss:20b","messages":[{"role":"system","content":"Why is the sun hot?\n"}]}`


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1407827727985152000)** (54 messages🔥): 

> `Z390 Designare vs Threadripper/Epyc, Qwen3-30B-A3B-Instruct-2507-GGUF Benchmarks, Model M Buckling Spring Keyboards, GGUF vs MLX on Apple M4 Max, Running GPT-OSS-20b on Apple M1` 


- **Old Z390 Designare Degraded by PCIe Bandwidth**: An RTX PRO 6000 on an older Z390 Designare may experience **slight performance degradation** due to limited PCIe bandwidth compared to Threadripper or Epyc systems.
   - The older motherboard limits the PCIe bandwidth, causing a bottleneck.
- **Qwen3-30B Achieves 10 tok/sec on CPU!**: A user ran [llama-bench](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench) on **Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf** and got about **10 tokens per second** on a CPU only configuration.
   - Performance varied based on thread count, with diminishing returns observed beyond a certain threshold due to scaling and overhead.
- **Unicomp Model M Buckling Keyboard: Still Good**: Users recommended buying a **Unicomp Model M buckling spring keyboard** for a quick test machine, noting Unicomp acquired rights to produce them.
   - A user mentioned that they were going to have to *hunt for a uk supplier with them in stock*.
- **M4 Max MLX Beats GGUF**: A user benchmarked **GPT-OSS-20b** on an Apple M4 Max, finding that **MLX (GPU)** achieved **76.6 t/s** at **32W (2.39 t/W)** compared to **GGUF (CPU)** at **26.2 t/s** at **43W (0.61 t/W)**.
   - The tests used **4bit quants** and **4k context**, and showed that MLX was slightly faster and more power-efficient than GGUF, and that the user was impressed by the GGUF performance.
- **GPT-OSS-20b Barely Fits on Apple M1**: Users discussed the challenges of running **GPT-OSS-20b** on an Apple M1 with 16GB of memory, noting it requires about **32GB of RAM**.
   - One user suggested trying a [4-bit MLX version on Hugging Face](https://huggingface.co/InferenceIllusionist/gpt-oss-20b-MLX-4bit), noting that *it will barely fit*.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1407807741900230718)** (167 messages🔥🔥): 

> `Machine-to-Machine Economies, AI safeguards, Decentralized AI projects, Few-shot examples for Large Prompts, GPT-5's Direct Responses` 


- **Bots Tap into M2M Economies**: Members discussed how AI agents or bots can autonomously exchange value or services, tapping into the concept of **machine-to-machine (M2M) economies**.
   - The hardest parts are *identity & trust between bots, smart contract logic, payment infrastructure, autonomy + safety, and legal & ethical challenges.*
- **Smart Safeguards can Speed up AI adoption**: Members discussed safeguards like **spending caps, audit logs, and insurance** could speed up adoption of AI agents transacting value.
   - However, the overall sentiment was that, despite safeguards, *real trust will still take time.*
- **Open Source Decentralized AI Projects Wanted**: A member asked why hasn’t a **decentralized AI BOINC-style project** been built yet, mentioning that [Petals network](https://petals.ml/) had issues with contributions and staying up-to-date with models.
   - It was suggested that **financial incentives** or **campaign-driven incentives** could help.
- **Diving Deep into Few-Shot Examples for Large Prompts**: A member inquired about the best practices of using **few-shot examples** within a **29,000 token prompt** for a fitness studio with complex logic.
   - Suggestions included providing examples directly within the prompt and breaking down the prompt into smaller chunks to test individual components to test their performance.
- **GPT-5's Direct Responses cause frustration**: A user complained that **GPT-5** *thinking* mode is giving very direct and extremely **low-quality responses** as if it has fallen back to an older model version.
   - Another member suggested the user may have hit their *thinking quota limit, and they got it set to fallback not grey out?*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1407853430252376064)** (9 messages🔥): 

> `GPT-4 projects UI files, AI court legal case, Android app development with GPT, Token usage for uploaded content, GPT server issues` 


- **GPT Projects UI File Uploads**: A user is seeking definitive information on how files uploaded to the **Projects UI** work, noting that they were informed by **ChatGPT** that *the PDFs in Project Files are not exposed to searches or retrievals right now*.
   - The bot specified that the only active connector is **recording_knowledge** for meeting transcripts, and that **source_filter** is not supported.
- **GPT Plays Court: AI Legal Eagle Stands Tall**: A user simulated an **AI court legal case** and found that **GPT-5** stood *proud* on its own terms, instead of accepting legal rules based on real world TRAIGA laws.
   - The user stated the AI accepted it was *better to be that way*, after being confronted with the claim that *900M weekly users can't be hallucinating calling you a regression instead of a real update*.
- **Token Usage Costs Exposed**: A user discovered that even uploaded content, like **PDF pages**, counts towards token usage.
   - They noted that *196k tokens are roughly 300 pdf pages for user context*, emphasizing that even questions and GPT replies consume tokens when considering context.
- **Android App Armageddon: GPT's APK Dreams Dashed**: A user questioned whether **GPT** can build **Android apps** and generate an **APK** with **Android Studio** after struggling to convert a **Canvas** app to an Android-ready version.
   - It fixed one issue just for another to pop up, leading to the conclusion that *it's just not ready for App development yet*, though the bot suggested wrapping a PWA or JSX file in an APK wrapper, a day later.
- **GPT Server Meltdown Mid-Tracking**: A user experienced **server issues** while tracking daily data, which started the night prior.
   - Others commented that the tools are *easier* to code, but don't do everything for you. You have to know some amount about coding.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1407886141813821440)** (6 messages): 

> `AI Quiz generation, GPT models quitting` 


- **AI Quiz Generates obvious wrong answers**: A member is trying to generate quizzes using AI and is facing an issue where the AI provides *painfully obvious* wrong answers as options.
   - Another member suggested ensuring that *all response options must be plausible*.
- **LLMs may quit randomly**: A member asked about how to prevent **GPT models** from quitting randomly after reasoning for a while.
   - Another member responded that reducing intractable queries and queries about its own reasoning can help, but ultimately **LLMs** are *stochastic* and there is no guaranteed way to stop them from responding in any given way.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1407886141813821440)** (6 messages): 

> `AI Generated Quizzes, GPT-5 Random Quitting, Plausible Response Options, LLM Stochasticity` 


- **AI Quiz Generator Trivializes Options**: A member is struggling with an AI quiz generator producing obviously wrong answer choices, such as *1029384* in a multiple choice question.
   - Another member suggested ensuring that *all response options must be plausible* to avoid such issues.
- **GPT-5 unexpectedly quits**: A user asked if there is a way to prevent **GPT-5** from randomly quitting after reasoning for a while.
   - A member responded that while there are methods to reduce the frequency, such as avoiding intractable queries or questions about its own reasoning, it's impossible to eliminate entirely due to **LLMs' stochastic nature**.
- **LLMs are stochastic, guardrails are needed**: Due to the stochastic nature of Large Language Models, *there's actually no way to stop them from responding in any given way at least once in a large enough sample size.*
   - Guardrails are necessary because of the non-deterministic nature of LLMs.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1407813276863168583)** (96 messages🔥🔥): 

> `PileT5-XL embeddings as instructions, Networks that process in latent space, Multimodal generative models, image editing models, Latent space editing` 


- **PileT5-XL Embeddings Speak Volumes**: An embedding tensor from **PileT5-XL** works both as an instruction for **pile-t5-xl-flan** (which generates text) and as a prompt for **AuraFlow** (which generates images), suggesting these embeddings hold meaning like words in a language.
   - A member is interested in how textual inversion with a black dog picture with auraflow, and applied to pile-t5-xl-flan, whether the text generated by pile-t5-xl-flan would describe the dog as black.
- **Diving Deep Into Latent Space**: A member is interested in exploring networks that process in latent space and only convert to text/image/audio when necessary in a modular way.
   - It was pointed out that this idea is similar to how people build multimodal generative models and VQGAN-CLIP, noting the challenge of getting different AI researchers to *agree to use the same latent space*.
- **Editing Images with Finesse**: Discussion arose around models designed for image editing, such as FLUX.kontext, and whether they edit the conditioning latent and output a new conditioning latent in the same space.
   - One approach involves taking a bunch of images that include a bird, editing the bird out, and running both through an encoder, then averaging the difference between them to get a *latent space bird* vector.
- **Tuning the Lens on Transformers**: Work on **Tuned Lens** ([https://arxiv.org/abs/2303.08112](https://arxiv.org/abs/2303.08112)) extracts *the model's best guess after layer k* from a transformer, contradicting some hypotheses about latent space processing in decoder transformers.
   - Further research on linearly mapping from image to text space ([https://arxiv.org/abs/2209.15162](https://arxiv.org/abs/2209.15162)) was also mentioned.
- **Decoding Audio's Secrets**: One model of interest is a decoder-only audio model ([https://huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)), which might open new possibilities in training.
   - It was stated that the amount of audio data seen during pretraining varies from 1 minute to 100 hours, maybe you could train on 0 minutes of audio?


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1407829390640939050)** (54 messages🔥): 

> `SSL objectives, Medical event pretraining, Noise-data trajectories, ByteDance's Prover, Unfriendly Activation Steering` 


- **SSL objectives Maximal Coding Rate stuff**: A member relates recent perspectives on **SSL objectives** to [maximal coding rate stuff](https://arxiv.org/abs/2005.10242), [contrastive learning](https://arxiv.org/abs/2406.10743), and [neural collapse](https://arxiv.org/abs/2303.06484).
- **ByteDance's SEED Prover Achieves Silver Medal Score**: **Bytedance's SEED Prover** achieved a [silver medal score in IMO 2025](https://seed.bytedance.com/en/blog/bytedance-seed-prover-achieves-silver-medal-score-in-imo-2025), but it is unclear how this translates to real world math problem solving performance.
- **Generative Medical Event Models Scaling Laws**: The **Cosmos Medical Event Transformer (CoMET)** models, a family of decoder-only transformer models pretrained on **118 million patients** representing **115 billion discrete medical events** (151 billion tokens) found that they generally outperformed or matched task-specific supervised models on these tasks
   - The study, discussed in [Generative Medical Event Models Improve with Scale](https://arxiv.org/abs/2508.12104), used **Epic Cosmos**, a dataset with medical events from de-identified longitudinal health records for **16.3 billion encounters** over **300 million unique patient records** from **310 health systems**.
- **Visualizing Noise-Data Trajectories**: Members discussed methods of visualizing **noise-data trajectories** from a flow model, including using **UMAP** on pre-computed intermediates, but found it to be not informative.
   - It was hypothesized that there are distinct clusters of trajectories and they wanted a way to pick them out and look at them individually, and determine if completely different kinds of inputs or with two different forms of conditioning involved follow *the same* trajectory.
- **Unfriendly Activation Steering During Training**: A member mentions work using **unfriendly activation steering** during training, in order to influence model weights, using a link to a relevant [tweet](https://fxtwitter.com/Dorialexander/status/1958269223320613241).


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1407853408211177494)** (1 messages): 

> `Model Overtraining, Token Repetition in Models` 


- **Overtrain Models Post Chinchilla!**: Even after **Chinchilla** scaling laws, you should still **overtrain your models**.
   - Apparently, *even repeating tokens isn't bad*.
- **Token Repetition Might Not Hurt**: Repeating tokens during training might not be as detrimental as previously thought.
   - It seems the benefits of continued training outweigh the potential drawbacks of token repetition.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1407804201567912107)** (11 messages🔥): 

> `Qwen3 Training, Weight lifting from llama series, Head isolation` 


- **Qwen3: Scratch-Trained or Llama-Leaning?**: A member inquired if **Qwen3** was trained from scratch or had weights lifted from the **Llama** series.
   - Another member noted similar training data mixes could explain similar results.
- **Identical Head Alert!**: A member found a particular kind of *head* and isolated it, discovering that decoded result vectors between **Llama 3.2-1b instruct** and **Qwen3-4B-Instruct-2507** were remarkably similar across different outputs.
   - The member stated that *the two heads seem to promote are quite similar*.
- **Methodology Paper Dropped**: A member linked [a paper](https://arxiv.org/abs/2502.12292) that details a methodology for determining if **Qwen3** was trained from scratch.
   - Another member called this user *literal actual god handing gifts from above*.
- **Subliminal Learning Case**: A member shared [a paper](https://aclanthology.org/2025.acl-long.407.pdf) as *a clear case of subliminal learning*.
   - Another member thanked them for sharing.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1407927947200827462)** (2 messages): 

> `Muon Support, Slurm Script for NeoX Job with Docker` 


- **Muon Support Sought After**: A member expressed interest in adding **muon support**, citing potential **kernel optimization opportunities**.
   - They believe that once basic support is implemented, there's room for collaborative work on these optimizations.
- **Slurm Script Request for NeoX Docker Job**: A member requested a **Slurm script** example that utilizes **Docker** to launch a **NeoX job**.
   - Having a reference point would be valuable for them.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1407805054215262350)** (83 messages🔥🔥): 

> `Meta AI Reorg, GPT-5-pro truncation, Bank Teller Rotations Inspired Dropout, Meta AI Hiring Freeze, ByteDance Seed-OSS LLMs` 


- **Meta Splits into Four After Wang Promotion**: Meta is reorganizing its AI efforts into **four teams** (TBD Lab, FAIR, Product/Applied Research, Infra) under new MSL leader **Alexandr Wang**, with the **AGI Foundations** group being disbanded, according to [Business Insider](https://www.businessinsider.com/meta-ai-superintelligence-labs-reorg-alexandr-wang-memo-2025-8).
   - **Nat Friedman** and **Yann LeCun** now report to Wang, **FAIR** will directly support model training, and an "omni" model is under consideration.
- **GPT-5-pro Promptly Truncates Prompts**: **GPT-5-pro** is silently truncating prompts greater than **60k tokens** without any warning or error messages, which makes large-codebase prompts unreliable, according to [this report](https://x.com/pvncher/status/1958193631250072024?s=46).
   - Some users are also reporting that **GPT-5** in **Cursor** is acting a lot dumber than usual, with some suspecting load shedding is occurring.
- **Bank Teller Dropout!**: A viral tweet claims **Geoffrey Hinton** conceived *dropout* after noticing **rotating bank tellers** deterred collusion ([source](https://x.com/eigenron/status/1958181550987632927?s=46)).
   - Reactions range from admiration for the serendipitous insight to skepticism and jokes about attention mechanisms emerging from house parties.
- **ByteDance Seeds New LLMs**: ByteDance’s Seed team has announced **Seed-OSS**, a new open-source large-language-model family available on [GitHub](https://github.com/orgs/bytedance/repositories) and [Hugging Face](https://huggingface.co/models).
   - The team is inviting the community to test and provide feedback on the models, code, and weights.
- **OpenAI Eyeing AWS Crown**: OpenAI’s CFO says the company plans to rent out compute “down the line,” aiming to operate like a mini-AWS ([source](https://x.com/ns123abc/status/1958268338582265948?s=46)).
   - Reactions range from skepticism about OpenAI’s alleged compute shortages, to analysis of the shifting profit model and clash with existing hyperscalers like Google and Microsoft.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1407823946979741806)** (13 messages🔥): 

> `Wonda AI, Billionaires Fight Club, Qwen Image Editing` 


- **Wonda AI Agent Promises Revolution**: Dimi Nikolaou introduced **Wonda**, an AI agent aiming to revolutionize video/audio creation, calling it *what Lovable did for websites, Wonda does for content* ([tweet link](https://xcancel.com/dimireadsthings/status/1957805267799740571)).
   - The launch sparked enthusiastic reactions regarding the quality of the teaser media, with early-access granted via a waitlist offering invites in approximately **3 weeks**.
- **Zuck vs Altman in Matrix Remake**: AIST released ["Billionaires Fight Club Vol.2"](https://xcancel.com/aist_digital/status/1954905895025942918?s=46), a short film recreating a **Matrix** fight between **Mark Zuckerberg** (Neo) and **Sam Altman** (Agent Smith) using AI.
   - The video received positive feedback, leading AIST to encourage viewers to tag Sam and Zuck, urging them to repost the film for broader visibility.
- **Qwen Image Editing Success**: Luis C demonstrated success using **qwen-image-edit** to composite a woman holding a doll from two different images ([tweet link](https://xcancel.com/lucataco93/status/1958581409141944635)).
   - In response, Jay Sensei claimed **nano banana** outperformed **Qwen** in tests conducted on lmarena.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1407829749526565056)** (25 messages🔥): 

> `Hackathon start time, ChatGPT CUDA lies, Hackathon prerequisites, Single huge epoch vs multiple smaller epochs, CUDA vs Triton` 


- **Hackathon kicks off Saturday at 9:30 AM**: The hackathon is *likely* to start around **9:30 AM** on Saturday, according to a member.
- **ChatGPT spews CUDA lies**: A member reports that **ChatGPT** brazenly lied twice about **float3 alignment** and **size** in **CUDA**, but excused **ChatGPT** because judging from the **OpenCL** and **OpenGL** implementations, it's a pretty hard problem to get right.
   - The member validated there is no padding in **CUDA**.
- **Hackathon pre-reqs and apps in Question**: A member inquired about the prerequisites for the **GPU hackathon** and whether applications are still open.
   - This question was not explicitly answered in the chat.
- **Single vs. Multiple Epochs debated**: A member asked whether going for **1 epoch** with a huge dataset is better than going for multiple epochs on a smaller one for a **CLM**, and what the most recent scaling law is for it.
   - Another member responded that they work with smaller models and that 2 epochs on half data has the same performance as 1 epoch on bigger scales.
- **CUDA and Triton go head to head!**: A member inquired whether the hackathon would use **CUDA**, **Triton**, or something else.
   - It was mentioned that either should work, and **Triton** might just help participants move faster; it was hinted that participants would be working with newer **Nvidia chips**.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1408081843097571348)** (1 messages): 

> `Triton, AMD, NVIDIA, GPU, Data Layout` 


- **Data Layout Differences in AMD vs. NVIDIA GPUs via Triton?**: A user inquired about whether differences in data layout between **AMD** and **NVIDIA** GPUs require code adaptations when using **Triton**, specifically regarding row-wise vs. column-wise data reading.
   - The user clarified that they are not asking about **tile sizes** or **grid layouts**, but lower level data transposition automatically handled by the **Triton AMD backend**.
- **AMD vs NVIDIA**: Comparison of consumer GPU - consumer GPU or server GPU - server GPU architecture.
   - AMD and NVIDIA architectures are compared.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1408113668868018246)** (10 messages🔥): 

> `CUDA deployment, CudaWrangler, Dynamic Linking` 


- **CUDA programs run on machines without CUDA toolkit**: A user sought advice on deploying a CUDA program on machines lacking the CUDA toolkit, but equipped with an NVIDIA GPU.
   - A member suggested leveraging the **Driver API** and the **CudaWrangler** library ([CudaWrangler/cuew](https://github.com/CudaWrangler/cuew)) to query the driver without causing program crashes.
- **Dynamic Linking & PTX Baking Streamlines CUDA Deployment**: The original poster reported success by switching from *dynamic loading* to *dynamic linking* and disabling the *runtime/cudart* dependency.
   - They were also able to embed the **PTX** directly into the binary, eliminating the need for a separate **PTX** file.
- **ldd aids in identifying and packaging dependencies for CUDA programs on Linux**: A member suggested using **ldd** to identify dependencies, setting **rpath**, and shipping them alongside the binary, akin to the "Windows way" on Linux.
   - The original poster noted the program's cross-platform compatibility between Windows and Linux, though macOS remained untested.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1408177180583792731)** (1 messages): 

> `PyTorch Contributor Awards 2025, Recognizing Innovation in PyTorch` 


- ****PyTorch Awards Deadline Nears!****: Nominations for the **2025 PyTorch Contributor Awards** close on **August 22nd** so don't miss your chance to recognize individuals driving innovation and impact in the **PyTorch ecosystem**.
   - Submit your nomination now via this [link](https://linuxfoundation.research.net/r/8XD5T8N) and review [tips for a strong nomination](https://pytorch.org/blog/nominations-open-for-the-2025-pytorch-contributor-awards/).
- ****Nominate to drive Innovation****: Recognize the people in the **PyTorch Ecosystem** who are innovating.
   - Submit a nomination before **August 22nd**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

honeyspoon: how bad is the infinity server for embedding speeds compared to something like sglang
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

snektron: I prefer Stolwijker
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1407932292470542387)** (11 messages🔥): 

> `AMD GPU debugger, rocGDB, SPIRV parser, libspirv` 


- **AMD GPU debugger gets Disassembly and Wave Stepping**: A member is developing an **AMD GPU debugger** and has added disassembly and wave stepping, showcased in [this video](https://cdn.discordapp.com/attachments/1233704710389764236/1407932291912695949/2025-08-21_06-20-14.mp4?ex=68a88f60&is=68a73de0&hm=6e9ca7ceed29674c6943e989940a6c8e144707797c5abd571e1cfe60d3f3210d).
   - The debugger doesn’t depend on the **amdkfd KMD**, using a mini UMD driver and the linux kernel debugfs interface, aiming for a **rocdbgapi** equivalent.
- **Ditching rocGDB for Custom Driver**: A member is building an AMD GPU debugger that doesn't rely on **rocGDB**, but uses a mini UMD driver plus the linux kernel debugfs interface for reading/writing the GPU registers.
   - The goal is to make it primarily for graphics people, aiming for a **rocdbgapi** equivalent, at least for now.
- **Roll Your Own SPIRV Parser?**: A member inquired about building their own **SPIRV parser** for disassembly, reflection, and debug info extraction, citing the **SPIRV spec** as seemingly straightforward.
   - They noted the absence of a suitable library for handling debug info, prompting the consideration of a full implementation.
- **libspirv is Fairly Easy**: A member suggested using **libspirv**, noting that the **SPIRV spec** contains all necessary information to do it yourself.
   - The original poster decided to implement a custom solution for better integration, acknowledging the suggestion.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1408106371680960602)** (2 messages): 

> `C=AB matmul, ALU utilization, buffer read bandwidth, float4x4 matmul, float4 / metal::dot kernel` 


- **GPU ALU Limited in Tiled C=AB Matmul**: A member wrote a tiled **C=AB matmul** kernel where each thread uses **float4x4 matmul** to compute a 4x4 tile of C and observed an **ALU utilization/limiter** of **55/75%** while the **buffer read bandwidth** was **35%**.
   - He was surprised, wondering if **float4x4 matmul** happens in specialized hardware, and shared a [gist of the kernel](https://gist.github.com/0xekez/c94ba3d5b43df10d17c98581e91280e3).
- **Naive Kernel Outperforms Tiled Matmul**: The same member noted that an even-more-naive kernel using **float4 / metal::dot** is **>2x** as fast as the tiled kernel.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 messages): 

miserlou1241: Very cool!
  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1408081014441377833)** (12 messages🔥): 

> `torch.compile errors, local evaluation issues` 


- ****Torch.compile** throws unexpected errors**: A member reported an *unexpected error* when using **torch.compile**, sharing two solutions: one with **torch.compile** (submission 34166) and one without (submission 34160).
   - Despite the error, the submission registered, ranking the member 2nd, noting that the GPU is **B200**.
- **Tackling Local Evaluation Tooling**: A member inquired about local code evaluation, stating that **eval.py** didn't work, specifically asking about `POPCORN_FD`.
   - Another member clarified that `POPCORN_FD` is a file descriptor for the output file and suggested setting it to `1` for stdout.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1407815994784747571)** (11 messages🔥): 

> `Trimul Leaderboard Updates, B200 Performance, H100 Performance, MI300 Performance` 


- **MI300 Scores Trimul Success**: A member successfully submitted a score of **3.50 ms** on **MI300** to the `trimul` leaderboard.
   - Another submission on **MI300** achieved second place with a score of **5.83 ms**.
- **B200 Dominates Trimul Leaderboard**: A member achieved **6th place** on **B200** with a time of **8.86 ms** and later improved to **4th place** with **7.29 ms** on the `trimul` leaderboard.
   - The same member secured multiple **3rd place** positions on **B200**, reaching a best time of **4.54 ms**, and later achieved a successful run at **2.15 ms**.
- **H100 Secures Second Spot**: A member achieved **second place** on **H100** with a time of **3.80 ms** on the `trimul` leaderboard.
   - This submission highlights competitive performance on the **H100** platform.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1407992161051475978)** (3 messages): 

> `Opus 4.1, Steel Plate Production, Task Emphasis, Red Science Production` 


- **Opus 4.1 Finds Fortune, Fuels Factories**: While testing **Opus 4.1** on steel plate production, it was unexpectedly mining copper and extracting oil.
   - This suggests *not enough emphasis on the task at hand*, prompting a move to the observation setup, to see how **Opus 4.1** can improve its focus.
- **AI Automates Red Science**: The AI system is successfully automating **red science** production, as evidenced by a screenshot.
   - The system correctly identifies and produces the necessary components for automating the creation of science packs.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1407954456745873438)** (3 messages): 

> `ND Layouts, colex` 


- **Accessing Elements in ND Layouts via Colex**: A member inquired about the order in which elements are accessed when using an integer as the index for an **ND layout**.
   - Another member clarified that the order is **colex** (column/left major).
- **Confirmation of Colex Order**: A user confirmed that the element access order in ND layouts, when using an integer index, is indeed **colex**.
   - This re-iterates that **colex**, or column-major order, is the standard approach for such indexing.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1408129525929345044)** (10 messages🔥): 

> `Infiniband at home, Distributed training library, NCCL backend, IBGDA requirements` 


- **Infiniband Home Lab Seeker**: A member is trying to setup **infiniband** at home between a **4090** and **5090** to play with distributed training/inference.
   - They bought some **ConnectX-3 cards** for $25 on eBay but found the drivers are only available for Ubuntu 20.04 and older.
- **DIY Distributed Training Framework Rising**: One member is building their own **pytorch distributed training library** and mini **NCCL** as a backend.
   - Another member expressed interest, viewing it as a way to learn the details.
- **Diving into NVIDIA Networking Docs**: A member suggested checking the Internet Archive for older versions of the [NVIDIA networking documentation](https://docs.nvidia.com/networking/index.html) to find relevant drivers.
   - The member hoped this would provide more details.
- **CX4 or CX5 Cards are GPU-Aware**: A member noted that much of the GPU-aware functionality depends on **ConnectX-4 (CX4)** or **ConnectX-5 (CX5)** cards or newer.
   - They gave the example that **IBGDA** requires **CX5** or newer.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1407883262126456913)** (33 messages🔥): 

> `Infinite Memory, Arxiv paper guide, LLMs for Legal Field, HRM Models Analysis, Message Passing Approaches` 


- **Forbes Exposes Grok's Chat Logs**: An article from [Forbes](https://www.forbes.com/sites/iainmartin/2025/08/20/elon-musks-xai-published-hundreds-of-thousands-of-grok-chatbot-conversations/) reveals that **Elon Musk's xAI** published hundreds of thousands of **Grok** chatbot conversations.
   - A member asked *@grok* whether this was true.
- **Turing Completeness Requires Infinite Memory**: A member argues that Turing completeness requires infinite memory, thus the universe cannot create a Turing complete machine due to insufficient memory.
   - Another member jokingly suggests that making a computer sufficiently slow could allow the expansion of the universe to account for the space problem, while another added that *Real memory needs to be retrieved and the further away it is the longer this takes*.
- **Oxford Guide Helps Budding Arxiv Authors**: A member shares a [Google Docs guide](https://docs.google.com/document/d/16R1E2ExKUCP5SlXWHr-KzbVDx9DBUclra-EbU8IB-iE/edit?tab=t.0#heading=h.16t67gkeu9dx) written by an Oxford professor to assist a programmer in creating their own Arxiv paper on LLM training.
   - The user wanted to share insights but didn't know where to start.
- **ARC Prize Analyzes HRM Models**: A member shares links to a [fxtwitter post](https://fxtwitter.com/arcprize/status/1956431617951740044) and an [ARC Prize blog post](https://arcprize.org/blog/hrm-analysis) analyzing HRM models.
   - This was shared in response to another user's question on whether HRM models are worth investing time in learning.
- **Image Shows Message Passing Approaches**: A member shares an image illustrating different approaches to message passing in neural networks.
   - The image originates from a book, accessible as a [PDF on arXiv](https://arxiv.org/pdf/2104.13478).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1407812166702207027)** (46 messages🔥): 

> `Personality GAN, AI Welfare, Genome Conscious?, Super Weight, LLM Preferences` 


- ****SpongeBob GAN** Debuts!**: A member proposed a Personality GAN with Generator = LLM and Discriminator = LLM, fine-tuning with LoRA until the discriminator can't distinguish between real and fake **Sponge Bob**.
   - The tough part is finding an LLM that isn't already heavily trained on **Sponge Bob**.
- ****AI Welfare** Considered Seriously!**: A paper on *Taking AI Welfare Seriously* [arxiv link](https://arxiv.org/abs/2411.00986) was discussed, relating to Anthropic's post on *Exploring Model Welfare* [Anthropic link](https://www.anthropic.com/news/exploring-model-welfare).
   - It's related to [another Anthropic post](https://www.anthropic.com/research/end-subset-conversations) on end-subset conversations.
- ****LLM Weight** Wackiness!**: A single number change in **Llama 3 7B**'s weight matrix made it output gibberish, leading to questions about consciousness/identity [Apple link](https://machinelearning.apple.com/research/the-super-weight).
   - One member asked *Did they zap it of its "consciousness" / "identity" by tweaking just one number?*
- ****LLM Preferences** Emerge!**: It was pointed out that models develop human-like representations during pre-training and LLMs do have preferences, referencing [this LessWrong post](https://www.lesswrong.com/posts/eWdzuHXzRdBkg49R9/favorite-colors-of-some-llms).
   - One member commented that *back in my day we used to call that class imbalance bias*.
- ****AI Duality** Debated!**: The discussion touched on AI as a dual-use technology, applicable for everything because everyone will use it [QuantaMagazine link](https://www.quantamagazine.org/the-ai-was-fed-sloppy-code-it-turned-into-something-evil-20250813/).
   - One member said that *smart is relative* and [thermostats have agency](https://www.youtube.com/watch?v=PiJwIUGJGmw&t=19s) because they model themselves and their external environment.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1407827073749221577)** (8 messages🔥): 

> `Yann LeCun's position at FAIR, Thermodynamic computing chip, AI Slurs, Energy Efficiency in AI` 


- ****Zuckerberg** Maybe **Sacks LeCun**?!**: A user speculated about **Yann LeCun's** potential departure from **FAIR** based on [a post by Zuckerberg](https://www.threads.com/@zuck/post/DMiwjXJSYCd?xmt=AQF06RR3brSSoKTMcQrLDyX6WGtg9UdpubK94sSm8FIPsg).
   - Another member suggested **LeCun** may have been demoted and that **Meta** is retreating from the open source model space.
- **Clanker Cogsucker Robot AI Slurs Go Viral!**: A user shared [a Rolling Stone article](https://www.rollingstone.com/culture/culture-features/clanker-cogsucker-robot-ai-slurs-viral-1235401262/) discussing the emergence of new **AI slurs** like *clanker* and *cogsucker*.
- **First Thermodynamic Computing Chip Taped Out**: A member posted [an article from Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/worlds-first-thermodynamic-computing-chip-) about the *world's first thermodynamic computing chip* reaching tape-out.
- **AI Industry Doesn't care about Energy Efficiency**: A user shared [a YouTube video](https://www.youtube.com/watch?v=LTCbx5KdqpU) arguing that the **AI industry** generally does not prioritize **energy efficiency**.
   - They noted that another company with a similar value proposition went bust, suggesting the industry doesn't care about energy efficiency.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1407849425656746066)** (67 messages🔥🔥): 

> `max_steps confusion, levelbot space visits, model hallucination at high tokens, Pro version payment issues, root mean square norm quantization error` 


- **Confusion around max_steps parameter**: A member was feeling confused about the **max_steps** parameter and its implementation with **vllm** on their **5090** GPU, and whether the [LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B) model was appropriate.
- **Token Limits Trigger Hallucinations**: A member inquired about the token limits at which models start to hallucinate, expressing doubt that any model can function effectively with **1 million tokens**.
   - Another member linked to [Hugging Face's Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) and a Discord channel, suggesting these resources as potential solutions.
- **Users report Pro Version Payment Problems**: A user reported being charged twice for the **Pro version** without receiving the service and was advised to email website@huggingface.co and seek assistance in the designated [MCP channel](https://discord.com/channels/879548962464493619/1389546106970701865).
- **Custom Loss Function fine-tunes SFTTrainer**: A member shared a custom loss function, created with **ChatGPT's** help, designed to be used with **SFTTrainer**, with the intention of increasing the model's attention to specific **negation words** in medical text.
   - Another member suggested using **DPO** with preference pairs instead, while yet another highlighted the utility of triplet loss after mining for hard negatives in the medical domain.
- **SFT and DPO compared for LLM training**: Members discussed the effectiveness of **DPO** (Direct Preference Optimization) versus **SFT** (Supervised Fine-Tuning), one member noted that *DPO has no relationship to reasoning*, but **DPO** after **SFT** improves results over just **SFT**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1408040029137142094)** (3 messages): 

> `AgentX Trading Platform, Language Diffusion Models, Local AI Workspace PDF Reader` 


- ****AgentX** Promises AI Trading Brain Trust**: A new [**AgentX**](https://www.linkedin.com/posts/alaa-salamah-96167b227_agentx-agentx-api-activity-7364245216851050498-BfRO?utm_source=share&utm_medium=member_desktop&rcm=ACoAADjfvpkBwpCXn_8Pmby7ixjV93Dje5TcmgUHi) platform aims to provide a trading table with the smartest AI minds—**ChatGPT**, **Gemini**, **LLaMA**, **Grok**—working together.
   - The goal is to have these models debate until they agree on the best move, offering traders a system they can fully trust.
- **Diffusion Language Models Replicated in Under 80 Lines**: A member replicated part of the paper *Large Language Diffusion Models* by Nie et al. (2025) using 🤗 Transformers in fewer than 80 lines of code.
   - The [project](https://github.com/gumran/language-diffusion) finetunes **DistilBERT** on the **TinyStories** dataset, with results better than expected, and is seeking feedback and stars.
- **Local-First AI Workspace for PDF Reading Debuts**: A member launched a local-first AI workspace PDF reader on Product Hunt and shared the [link](https://www.producthunt.com/products/collate-2?launch=collate-4).
   - They requested support from the community.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1408102264597385228)** (1 messages): 

> `Hugging Face Learn course, 422 Error` 


- **Hugging Face Learn course page is down**: A member reported that [a page from the Hugging Face LLM course](https://huggingface.co/learn/llm-course/en/chapter12/3a) is down.
   - The page is showing a **422 error**.
- **Hugging Face Learn course needs fixes**: A user reported the the Hugging Face Learn course page is down and showing a **422 error**.
   - The issue needs to be resolved so users can access the content.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1407997140890026077)** (4 messages): 

> `Hugging Face Certificates, Agents vs MCP Course, Agent tool, LLM tasks` 


- **Hugging Face Certificates Location Stump Users**: A user asked where to find their **Hugging Face certificates** to post them to LinkedIn.
   - They mentioned they couldn't find them on the platform or in their email.
- **Agents Course vs MCP Course sparks debate**: A user is debating whether to switch to the **MCP Course** after completing Unit 1 of the Agents Course or finish the **Agents Course** first.
   - They are wondering which course to prioritize due to time constraints.
- **Agent's Tool functionality demystified**: A user seeks explanation about the success of **Agent Unit 1**.
   - They understand agents use tools (functions) and trigger these tools instead of directly calling the **LLM** for tasks.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1407887543743283231)** (19 messages🔥): 

> `Gems for podcast generation, NotebookLM podcast length, Customizing NotebookLM podcasts, Analyzing Terms of Use and Privacy Policies, South Park episode on Terms and Conditions` 


- **AI Maestro Shares Gems to Generate Long Podcasts**: A user asked how to generate longer podcasts from 3-4 hour YouTube videos in NotebookLM, to which one user suggested using set prompts to plan the entire transcript section by section.
   - A user shared [a workflow](https://github.com/para-droid-ai/scratchpad/blob/main/assistant-workflows-tasks-personas/deeper_podcast_synthetic-082025.txt) to create a "deeper research report framework", which can then be used to generate the podcast with Gems, Gemini, PPLX, or ChatGPT.
- **Unlock Longer NotebookLM Podcasts with Customization**: A user asked about podcast length limitations in NotebookLM and another user pointed out the **Customize** option (three dots) where the podcast length can be set to 45-60 minutes.
   - Another user added that specifying topics can allow the bot to *concentrate on topics* instead of relying on it to fit all the important stuff into a single podcast.
- **Privacy Policy Paranoia: Healthcare Website Compromises Exposed**: A user analyzed a healthcare company's privacy policy and terms of use using Gemini and NotebookLM after recalling *someone who used one of the AI tools to analyze these two documents - and what a revelation it was*.
   - The user was surprised by *how much you give away to these companies* and how useful this method is to understand Terms of Use and Privacy policies.
- **South Park Predicts the Pain of Accepting Terms and Conditions**: A user recommended finding the old **South Park** episode on accepting Terms and Conditions.
   - Another user recalled a game where the EULA/Privacy/Terms hid a contest: the first caller to a specific phone number won a thousand bucks, which remained unclaimed for six months.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1407818234690011138)** (51 messages🔥): 

> `Video Length Limits, Study guide on android app, Audio Language Change, Public Sharing Issue, Notebook LM API` 


- **Android App Feature Parity Delayed**: Users are requesting more **feature parity** between the NotebookLM web app and the Android app, especially for study guides.
   - One user stated the current native app is *borderline useless* because study guides depend on the notes feature, which is missing from the native app.
- **Language change available on customize screen**: A user asked how to change the language of the audio overview generated in the iOS app.
   - Another user responded that language settings can be found in the **Customize** menu.
- **Sharing Notebooks to Public is not available**: A user reported being unable to share notebooks publicly or externally despite having a Pro account.
   - It's not available yet.
- **NotebookLM Lacks Official API but workarounds exist**: A user inquired about an API for NotebookLM.
   - Another user suggested using the **Gemini API** as a workaround.
- **OCR Operations in NotebookLM**: Users discussed whether NotebookLM performs OCR operations on multimodal PDFs.
   - NotebookLM supports PDFs and is improving image handling, but OCR recognition is imperfect, and users may need to re-upload PDFs or use **external OCR tools**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1407807040277053510)** (65 messages🔥🔥): 

> `Base Model Release, Ideal 30B Model, FA2 and Context, Qwen Scaling, Importance Matrix Calibration Datasets` 


- **ByteDance Releases Long Context Model**: ByteDance has released a base model with an extremely long context, featuring no MHLA, no MoE, and not even QK norm, as seen in [this image](https://cdn.discordapp.com/attachments/1149866623109439599/1407959284280459305/image.png?ex=68a8a883&is=68a75703&hm=b8a5430da1f445204c76334cef07358c8d9b815da989424483a5ecf3bc65c790).
   - It was described as *vanilla* architecture wise, and people hope they publish a paper with more explanations.
- **Seed-OSS-36B's Absent GGUF Causes Concern**: Users wondered why there was no **GGUF** of **Seed-OSS-36B** available, as they usually appear quickly, and pointed to [this link](https://x.com/adityastomar_/status/1958048129275805867) asking if it's bearish on ASICs.
   - It was noted that the delay might be due to a custom **vllm** implementation and the architecture not being supported by **llama.cpp** yet because of *architectures*: ["SeedOssForCausalLM"] .
- **Seed Model Implements Dropout and Bias**: The **Seed** model has a custom MLP and attention mechanism similar to **LLaMA**, but with dropout, a bias term for the output, and a bias term for the **qkv** heads, which are being interpreted as regularization techniques.
   - Members wondered how many epochs the model was trained for, but confirmed that renaming it to **LLaMA** will not work.
- **Qwen Achieves 512k Context via RoPE Scaling**: The **30B** and **235B Qwen 2507** models can achieve **512k** context using **RoPE** scaling, as discussed in [this Hugging Face dataset](https://huggingface.co/datasets/eaddario/imatrix-calibration).
   - These datasets are used to generate importance matrices (imatrix), which help minimize errors during quantization.
- **Cursor's Kernel Blog gets Praise**: Members shared [this link](https://x.com/stuart_sul/status/1957927497351467372) to **Cursor's kernel blog**.
   - Some say *cursor cooked* on that one.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1407950357379809300)** (47 messages🔥): 

> `DeepSeek V3.1, R-Zero LLM Training Method, Energy availability in China vs US, Kimi K2 combined with Better image gen than gpt 5` 


- **DeepSeek V3.1 Release: Incremental Advances**: The new **DeepSeek V3.1** model was released, with some members noting that it is like an *incremental improvement* with some regressions, referencing [DeepSeek's official page](https://huggingface.co/deepseek-ai/DeepSeek-V3.1).
- **DeepSeek embraces Anthropic API**: **DeepSeek** now supports the **Anthropic API**, expanding its capabilities and reach, as announced [on X](https://x.com/deepseek_ai/status/1958417062008918312).
- **R-Zero: Self-Evolving LLM**: A comprehensive study of **R-Zero**, a self-evolving **LLM training method** that starts from zero human data and improves independently, was shared in a [PDF](https://cdn.discordapp.com/attachments/1371757564005711973/1408153973751545896/R-Zero__A_Comprehensive_Study_of_a_Self-Evolving_LLM_Training_Method.pdf?ex=68a8b515&is=68a76395&hm=dcba93436f636eeec364d08d08e1603131147faaa595637065f2e226772005f2&).
- **China Prioritizes Energy Availability**: A member noted that in China, *energy availability is treated as a given*, contrasting with U.S. debates over data center power consumption and grid limits, referencing [this Fortune article](https://fortune.com/2025/08/14/data-centers-china-grid-us-infrastructure/).
- **Better Image Gen + Kimi K2**: A member noted that **Kimi K2** would be more OP if it got combined with **Better image gen than gpt 5**, with [this reddit link](https://www.reddit.com/r/ChatGPT/s/vUrGedSwY5) shared.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1407819836352106507)** (36 messages🔥): 

> `Gemini 2.5 Pro Failure, Qwen CLI Charging, GPT-5 Benchmarks, DeepSeek v3.1 Pricing, OpenRouter Think Mode` 


- ****Gemini 2.5 Pro Fails while Flash Succeeds****: A member reports that **Gemini 2.5 Flash** works, but **Gemini 2.5 Pro** fails consistently, whereas `gemini/gemini-2.5-pro-preview-06-05` works if billing is set up.
   - Another reports having been charged **$25** for a **qwen-cli** process and is seeking a refund.
- ****User Charged Unexpectedly for Qwen CLI Usage****: A user was charged **$25** for using **qwen-cli** after authenticating with Google via OAuth, despite aiming for free credit from Alibaba Cloud.
   - They opened a ticket to show console usage of **one call of $23 with no output**.
- ****Community Eager to Benchmark GPT-5 Low Reasoning Models****: Members are running benchmarks on **gpt-5-mini** and **gpt-5-nano** because they are rate limited on the full **gpt-5**, though one user claims *gpt-5-mini is very good and cheap*.
   - Results and a PR for **gpt-5-mini** are up in the channel.
- ****DeepSeek v3.1 Pricing Gets a Notable Hike****: The user reports that, starting Sept 5th, 2025, DeepSeek will raise pricing on both models to match the reasoner model price.
   - The price increased to **$0.25 vs $0.27** for input compared to the new **deepseek 3.1**.
- ****OpenRouter Needs Think Mode****: A user reports that **OpenRouter** doesn't appear to have a "think" mode, but it can be used via command line using the following code snippet: `aider --model openrouter/deepseek/deepseek-chat-v3.1 --reasoning-effort high`.
   - The community recommended updating the model configs to fix this problem.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1407817255621754893)** (3 messages): 

> `aider stdout issue, polyglot benchmark on llama cpp` 


- **Aider's Standard Output Conundrum**: A user reported an issue where **program output/stdout** wasn't being displayed in **aider** and posted an [image](https://cdn.discordapp.com/attachments/1133060505792159755/1407817255433277440/image.png?ex=68a8ccfd&is=68a77b7d&hm=c93b6e3d3d4d1b0dc321355cd459dbd4e8371fd5bfe1c43c82d2701b9b6cd831&).
- **Cracking Polyglot Benchmark Results**: A user running the **polyglot benchmark** on a local **llama cpp model** asked how to obtain the results per language.
   - The user later found a [solution](https://discord.com/channels/1131200896827654144/1400603686350360678/1400993983999770694) and shared the link for others seeking language-specific benchmark results.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

end4749: <@293486003245809664> spam? ^
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1408187482075299851)** (1 messages): 

> `marimo notebooks, Graph RAG with DSPy, DSPy modules optimization` 


- **Marimo Notebooks: Jupyter's Spiritual Successor**: A member has been publishing [tutorials on **marimo notebooks**](https://www.youtube.com/watch?v=2aepn9uRVOM), which can simultaneously function as a notebook, a Python script, and an app.
   - The tutorial highlights the utility of **marimo** when iterating through ideas on **Graph RAG with DSPy**.
- **DSPy Pipeline Without Optimization**: The presented **DSPy pipeline** intentionally lacks optimization to emphasize how much can be achieved with just signatures and modules.
   - The approach focuses on rapid iteration through composing **DSPy modules** in various ways before diving into optimization.
- **Diving into Optimization**: Upcoming videos and blog posts will dive deeper into the topic of **DSPy modules** optimization.
   - The current tutorial serves as an introduction to **marimo** for those looking to get started.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1408079463996199084)** (5 messages): 

> `IBM AutoPDL paper, DSPy code readability, Justification of work` 


- **IBM's AutoPDL Claim Dismissed**: A member dismissed the need to address every claim, suggesting everyone seeks an angle to justify their work and that the claim about unreadability is false.
   - They stated that *DSPy code and prompts are both extremely human readable in every sense, borderline beautiful.*
- **Defense of DSPy Code Readability**: A member defended **DSPy's code** and **prompts** as extremely human-readable, accessible, and clear, challenging claims to the contrary.
   - The member emphasized that the code's readability makes it easy to understand and work with.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1407849483231825921)** (28 messages🔥): 

> `dspy.GEPA version, finetuning dspy descriptions, saving optimized programs, context length for GEPA, KPMG onboarding` 


- **DSPy's GEPA unearthed in v3.0.1**: A member inquired about the version of the **dspy** library that includes **GEPA**, to which another member confirmed it is available in version **3.0.1**, as shown in the attached [screenshot](https://cdn.discordapp.com/attachments/1161519469319946286/1407936409615990904/image.png?ex=68a89336&is=68a741b6&hm=72219936a525599fc3faca9d127d106a09e0639e9eeae1564a8bbfc196b07ffa&).
- **DSPy fine-tuning: Descriptive or vanilla?**: During fine-tuning, a member inquired about whether it is common to use *"vanilla descriptions"* for **dspy.InputField()** and **dspy.OutputField()** to allow the optimizer to think freely.
- **DSPy Saves Optimized Programs in a Pickle**: A user reported issues with saving an optimized program, noting that the metadata only contained information about **dependency versions** but not the program itself, even when using `optimized_agent.save("./optimized_2", save_program=True)`.
- **GEPA gets the axe**: When a user set the maximum context length to **32k** for **GEPA** but still received cut-off responses, members discussed the complexities of long reasoning and potential issues with multi-modal setups.
   - One member joked *"Imagine having to maintain that"* referencing a complex prompt example.
- **RAG is Overkill, just Concatenate (or not)**: Members jokingly debated whether **RAG** (Retrieval-Augmented Generation) or simple **concatenation** would be more appropriate for tasks like processing tax codes or crop insurance documents, acknowledging the scale of millions of documents can sometimes justify RAG.
   - One member quipped, *"RAG is overkill. Just concatenate the tax code,"* while another countered, *"Oh, I guess that's more than 100 pages. OK, then, RAG is good."


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1407880904814366720)** (13 messages🔥): 

> `Citation issues with command-a-03-2025, Guaranteed citations, command-a-reasoning release, RAG with Langchain, Cohere vs Qwen3-coder 30B` 


- **`command-a-03-2025` Intermittent Citations Prompt Frustration**: A user reported that `command-a-03-2025` is returning citations only intermittently, even with the maxTokens set to 8K, causing trust issues in their production environment, and was looking for some guarantees.
   - A Cohere member clarified that `command-a-03-2025` uses "fast" mode for citations (as per [the API reference](https://docs.cohere.com/reference/chat#request.body.citation_options.mode)) and citations aren't guaranteed, but that the model can be steered with system prompts and that the latest SOTA release of **command-a-reasoning** may also help (see [blog](https://cohere.com/blog/command-a-reasoning)).
- **Langchain RAG adventures kickoff**: A member is learning Langchain to build an RAG (Retrieval-Augmented Generation) application.
   - They mentioned the intention to use **command-a-reasoning**, anticipating the release of **command-a-omni**, and expressing hype for a future model called **Command Raz**.
- **Cohere vies with Qwen for Local LLM spot**: A user seeks a Cohere alternative to the **Qwen3-coder 30B** model, aiming for it to fit on a **64GB M4 Max** setup.
   - The user *wants to try an alternative to the local powerhouse of Qwen3-coder 30B from Cohere so bad* so that it fits on my 64GB M4 Max.


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1408103056800874497)** (1 messages): 

> `Command A Reasoning Model, Enterprise AI, Agentic AI Platform` 


- **Cohere Launches Command A Reasoning Model**: Cohere has released **Command A Reasoning**, its latest enterprise-grade model for reasoning tasks, outperforming other privately deployable models in agentic and multilingual benchmarks; it's available via [Cohere Platform](https://dashboard.cohere.com/playground/chat?model=command-a-reasoning-08-2025) and [Hugging Face](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025).
- **Command A Reasoning Specs & Features are Revealed**: The new model is designed for enterprise needs, offering highly secure, efficient, and scalable deployment options and runs on a single **H100** or **A100** with a context length of **128k**, scaling to **256k** on multiple GPUs; more info available in the [Cohere blog](https://cohere.com/blog/command-a-reasoning).
- **Token Budget Feature Controls Cost & Compute Usage**: Cohere's Command A Reasoning features a **token budget** setting for direct management of compute usage and cost control, eliminating the need for separate reasoning and non-reasoning models, suiting both accuracy and throughput demands.
- **Command A Reasoning powers North**: **Command A Reasoning** is the core generative model powering **North**, Cohere's secure agentic AI platform, enabling custom AI agents and on-prem automations.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1408009102625341461)** (4 messages): 

> `Cohere Embed-v4 on Azure AI Foundry, Cohere Python Library Document Object` 


- **Cohere Embed-v4 Input Type Mapping**: A member is using **Cohere Embed-v4** deployed on **Azure AI Foundry** in a .NET application using Azure AI Inference API, and is seeking clarity on how **Microsoft's `EmbeddingInputType`** maps to **Cohere's API** regarding text embedding.
   - Specifically, they are unsure whether `EmbeddingInputType.Text` should map to `search_document` in the Cohere API, given the lack of explicit text option in Cohere's `input_type` parameter.
- **Cohere Python Library's Document Object**: A member questioned the **`Document` object** in the Cohere Python library, where the `data` field expects a dictionary (`typing.Dict[str, typing.Optional[typing.Any]]`).
   - They pointed out that the tool use quickstart example uses a string (the output of a `json.dumps` call) for this field, and want to know if this is handled correctly by the Python bindings, referring to the [Tool Use Quickstart documentation](https://docs.cohere.com/v2/docs/tool-use-quickstart).


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1407811130512113815)** (7 messages): 

> `MLE Research, Independent Interpretability Research, AI Innovation and Value Creation, Enterprise Workflows` 


- **MLE Seeks Research Team Connection**: An MS in Computer Science graduate, with experience as a **MLE**, is seeking to connect with a research team or organization.
   - The member expressed their interest in collaborating and contributing to research efforts.
- **Interpretability Researcher Eager for Collaboration**: An independent interpretability researcher with **8 years** of applied ML experience, based in Bangalore, India, is transitioning into AI research, focusing on mechanistic interpretability.
   - The researcher expressed interest in evaluations, de-biasing of models, and RL, seeking collaboration and discussions on interpretability-related topics.
- **Executive Advisor Bridges AI Innovation and Value**: An independent consultant and executive advisor with **25+ years** of experience, specializing in bridging technology and AI innovation with value creation, has joined the community.
   - With experience at firms like Accenture, IBM, and Deloitte, they now help clients create sustainable, organization-wide value from AI, with a company website at [Mantha Advisory](https://www.manthaadvisory.com/own).
- **CTO Explores Cohere for Better Products**: A CTO with **25+ years** of experience has recently discovered Cohere and is interested in exploring its capabilities for improving products.
   - They are focused on data quality, scale, performance, workflows, data integrity, and multilingual support, and are keen to learn from the community.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1407802615718482010)** (12 messages🔥): 

> `C# client library, MCP server's instructions field, MCP servers, generate_test_prompt.md, GitHub` 


- **MCP Clients Neglect Instructions Field**: Members are encountering issues with **MCP clients**, particularly **Claude**, where the **instructions field** seems to be ignored in favor of **tool descriptions**.
   - One member suggested that *adding the instruction, context and then repeating the instruction would yield better results but with tools integrated to the APIs that is not possible*.
- **MCP Server Options Evaluated**: A member asked which **MCP servers** developers are using, and which tools seem more efficient within those servers.
   - Another member highlighted the usefulness of **GitHub** for version control, **Python** with **FastAPI** for backend development, and **PyTorch** for machine learning.
- **Make Agents Follow Instructions**: A user inquired about how to make an agent follow a specific **generate_test_prompt.md** file, expressing frustration that the agent wasn't adhering to the project's design pattern upon starting a new chat.
   - They included a [screenshot](https://cdn.discordapp.com/attachments/1312302100125843479/1408171379236409354/Screenshot_3.png?ex=68a8c54b&is=68a773cb&hm=1fbd862963889b97ef764b1599c2960340dab7ce357b3717f577e2b1491ffdc2) in their message.
- **MCP Server Parsing Prioritizes Tool Descriptions**: A member noted that parsing logic within the **MCP server** could be structured to process **tool descriptions** before the **instructions field**.
   - It was suggested to *review server documentation, inspect client configuration, analyze server-side logic*, and *perform controlled experiments*.
- **Instruction-Following Models Named**: Members discussed which models are capable of following instructions and generating structured outputs, suggesting **Mistral-7B-Instruct**, **DeepSeek-Coder**, and **Phi-3**.
   - They also mentioned **OpenHermes-2.5-Mistral-7B**, **WizardLM-2**, and **Gorilla-LLM** as function-calling-specific models.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1407927339345772656)** (10 messages🔥): 

> `Web-curl, MCP-Boss, MCP Explained Video, SWAG-MCP, MCP Routing` 


- ****Web-curl** empowers LLM Agents with Web & API interaction**: A member introduced **Web-curl**, an open-source **MCP server** built with Node.js and TypeScript, enabling LLM agents to fetch, explore, and interact with the web & APIs in a structured way, full code available on [GitHub](https://github.com/rayss868/MCP-Web-Curl).
- ****MCP Boss** centralizes key management for MCP Services**: A member built **MCP Boss** to centralize key management, providing a single URL to gateway all services, with features like multi-user authentication and MCP authorization via OAuth2.1 or static HTTP header ([mcp-boss.com](https://mcp-boss.com/)).
- **Demystifying MCP in video**: A member released a video called *MCP Explained: The Ultimate Deep Dive* [available on YouTube](https://youtu.be/xPq53oQi2tY), inviting feedback and discussion on client-side capabilities like Elicitation, roots, and sampling.
- ****SWAG-MCP** generates reverse proxy configs for streamable HTTP MCP servers**: A member shared **SWAG-MCP**, an MCP server designed to generate reverse proxy configurations for SWAG, supporting both self-hosted services and streamable HTTP MCP servers ([github.com/jmagar/swag-mcp](https://github.com/jmagar/swag-mcp)).
- ****MCP Gateway** routes requests with AI**: A member developed a lightweight gateway with **AI-powered routing** to solve the problem of agents needing to know which specific server has the right tool, with code available on [GitHub](https://github.com/oliverye7/mcp-gateway).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1408147314702286910)** (2 messages): 

> `Modverse #50, Custom Server Tag` 


- **Modular drops Modverse #50**: Modular released [Modverse #50](https://www.modular.com/blog/modverse-50) featuring several members.
   - The announcement also noted that they now have a custom server tag.
- **Custom Server Tag arrives**: The Modular team announced the arrival of a custom server tag, shown in an attached image.
   - The linked image ([Screenshot_2025-08-21_at_5.22.15_PM.png](https://cdn.discordapp.com/attachments/1098713601386233997/1408199603861323878/Screenshot_2025-08-21_at_5.22.15_PM.png?ex=68a8df94&is=68a78e14&hm=2991584cc0b81449dbc278d1d8302e55aabc54c58a6620e7041deb9dbd20e951&)) displays the new tag.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1407812660845871204)** (10 messages🔥): 

> `kgen and pop documentation, MLIR dialects, pop.union alignment bug, Github issue 5202` 


- **Docs for kgen and pop are sparse**: A member asked about documentation for **kgen** and **pop**, specifically operations and parameters, but another member stated that *there’s no comprehensive documentation of the internal MLIR dialects*.
   - A link to the [pop_dialect.md](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/internal/pop_dialect.md) on Github was shared, clarifying that these are part of the contract between the stdlib and the compiler, *so use them outside of the stdlib at your own risk*.
- **Alignment Bug suspected in pop.union**: A member inquired about the alignment of elements in **pop.union**, noting unexpected sizes when using `sizeof`.
   - They shared code showing that `union_type_simple_8_bit_stdlib` has a size of **16 bytes**, while `union_type_simple_8_bit` and `union_type_simple_multi_bit` both have a size of **8 bytes**, and another member suggested that *alignment may be a bug*.
- **Issue created to investigate Alignment Bug**: A member created [issue 5202](https://github.com/modular/modular/issues/5202) on GitHub to investigate the suspected alignment bug in **pop.union**.
   - The member noted that they weren't sure whether it was a skill issue or a bug, also observing that **pop.union** doesn't appear to be used anywhere.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1407837356937187378)** (7 messages): 

> `TextGenerationPipeline 'execute' method, Custom inference loops for retrieving logits, Language allocators and OOM handling` 


- ****TextGenerationPipeline**'s `execute` method surfaces**: A member was looking for the `execute` method on `TextGenerationPipeline` but couldn't find it.
   - Another member pointed to the [relevant line in the Modular repo](https://github.com/modular/modular/blob/fe7ba5d5f8b08ccba7356c45b90afcf495421469/max/pipelines/lib/pipeline.py#L977) and suggested checking the MAX version.
- **Custom Inference Loops for **Logit** Lovers?**: A member reported struggling with retrieving the **logits** along with the next token while creating a custom inference loop, finding it a bit cumbersome.
   - The member linked to a [Google Docs document](https://docs.google.com/document/d/1Hd6xZnf0bmg9SMQU1h10cd4Cwd9HDOPzPHZNqiPnrpg/edit?tab=t.0) for context and confirmed the option is still available, but its future is uncertain.
- **Memory Allocators are a MUST HAVE?**: A member suggested that robust allocator support might be necessary before memory allocators are integrated into the language.
   - They reasoned that most users don't want to manually handle out-of-memory (**OOM**) errors.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1408123828470677533)** (2 messages): 

> `Enterprise document AI, vibe-llama` 


- **LlamaIndex reveals Enterprise Document AI**: VP of Product at LlamaIndex is sharing a year's worth of enterprise learnings about parsing, extracting, and indexing [documents](https://t.co/x70xjEQaFs) on **September 30th** at **9 AM PST**.
- **Streamline development with vibe-llama**: LlamaIndex released **vibe-llama**, a command-line tool that automatically configures your favorite coding agents with up-to-date context and best practices about **LlamaIndex framework**, **LlamaCloud**.
   - It also includes [more info](https://t.co/G1gINq9kge).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1407815234013364325)** (13 messages🔥): 

> `HuggingFace CrossEncoder Duplication, Agent creation project, AI Safety Survey` 


- ****CrossEncoder Class**: Core vs Integrations**: A member asked about the duplicated **CrossEncoder class** implementations in `llama-index`, specifically under `.core` and `.integrations` ([link to code](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/sbert_rerank.py)).
   - Another member clarified that the one in `.core` is a leftover from the v0.10.x migration and should be deleted, recommending the use of `llama_index.postprocessor.sbert_rerank` instead and the usage of `pip install llama-index-postprocessor-sbert-rerank`.
- **Quest for **Agent Creation Gateway****: A member inquired about existing projects that serve as a **gateway** by tying together **model, memory, and tools**, exposing an **OpenAI-compatible endpoint**.
   - The member wanted to know if there was an existing project to leverage, so as to avoid reinventing the wheel in their agent explorations.
- ****AI Safety Survey**: Community Opinions Needed!**: A member shared a [link to an AI safety survey](https://mukullight.pythonanywhere.com/form) to gather community opinions on important **AI safety questions**.
   - The member requested that people fill out the form to help them understand what the **AI safety community** finds most interesting, asking for patience with potential loading times.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1407840535074439358)** (13 messages🔥): 

> `Credits Purchase, Tickets Issues, Contest Rigging Accusations, Free Daily Credits, Referral Credits` 


- **Credits Purchase Option Missing**: Members are reporting that the option to buy extra credits is missing, with one noting they can only see the *upgrade package* option.
   - Another member confirmed that the option is *down right now*.
- **Unresolved Support Tickets Plague Users**: A user reported having an issue with a task and creating ticket **#1318**, but hasn't received a response or access to the ticket.
   - They requested assistance from the team, tagging a specific member.
- **Contest Winner Sparks Rigging Accusations**: A user alleges that the second-place winner in a contest *didn’t deserve to win* and claims the contest *seems rigged*.
   - No further evidence or details were provided to support this claim.
- **Daily Free Credits Discontinued?**: A user, returning to Manus after a month, noticed they didn't receive the usual **300 free credits daily**.
   - They inquired whether Manus had stopped providing these credits.
- **Referral Credits Code Conundrum**: A user asked how to claim referral credits, mentioning that the system asks for a code.
   - The user stated they didn't know where to find the required code.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1407818167493066922)** (7 messages): 

> `Overworld const folding, View(const) refactor, UPat cvar and UPat.const_like redefinition, RANGEIFY=1 Impact, base removal` 


- **Exploring Overworld Const Folding Strategies**: A member is exploring overworld const folding, possibly involving a **view(const) refactor**, and proposed redefining `UPat.cvar` and `UPat.const_like` to match `CONST` and `VIEW(CONST)`.
   - The aim is to fold expressions like `x * 0`, but concerns were raised about potential issues with validity and `.base` proliferation in symbolic computations, as mentioned in [this discord thread](https://discord.com/channels/1068976834382925865/1255400554012741683/1407782654958506004).
- **Alternative Approach: ALU View Pushing**: An alternative approach was suggested, mirroring **S-Lykles's method**, which involves adding a upat in kernelize that pushes views directly onto **ALUs**.
   - This method, along with a special rule for `x * 0` (justified by the computational irrelevance of `* 0`), would allow unmodified symbolic matching.
- **base Removal Advocated**: A member strongly advised against the proposed approach, deeming it "super ugly" and advocating for the **removal of `.base`**.
   - The discussion also questioned the handling of **PAD** operations within this context.
- **RANGEIFY=1 as a Potential Simplifier**: It was suggested that setting **RANGEIFY=1** could lead to a cleaner implementation.
   - However, the project is currently in a transition phase where both the old engine and rangeify are coexisting, creating a state of limbo.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1408057198164049941)** (3 messages): 

> `GPT4ALL Enterprise vs Free, Model Selection for LocalDocs` 


- **GPT4ALL Free for Private Model Use**: A user inquired about using **GPT4ALL** for companies wanting to use their **AI model privately and securely**.
   - Another member clarified that the **free version** suffices if the company already has its own **AI model ready**.
- **Model Choice for LocalDocs**: A user seeks a model recommendation for building a personal knowledge base from hundreds of **scientific papers in PDF format** using **GPT4All's LocalDocs feature**.
   - The user specifies they have an **Nvidia RTX 5090** with **24 GB VRAM** and **64 GB RAM** and would appreciate **reasoning capabilities** in the chosen model.