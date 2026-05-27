---
id: MjAyNS0x
title: not much happened today
date: '2026-05-26T05:44:39.731046Z'
description: >-
  **Harness engineering** is emerging as the key differentiator for coding
  agents, emphasizing the stack of **model + harness + eval loop** over just
  stronger base models. **DeepSeek** is building a harness team to optimize
  interaction and verification loops, while **Google's Gemini Managed Agents**
  and **LangChain** formalize harness concepts like context governance and
  dynamic skill routing. New benchmarks like **DeepSWE** align closely with real
  developer experience, with **Qwen3.7 Max** and **Claude Opus 4.6** showing
  strong agentic coding performance. **Anthropic** introduced a
  security-guidance plugin for **Claude Code** reducing security PR comments by
  30–40%, and **OpenAI** highlighted **GPT-5.5** in Codex for improved document
  parsing. In research, **Claude Mythos** solved Erdős problem #90 with a
  cleaner proof path than previous models, showing latent capabilities unlocked
  by appropriate harnesses. The paper "Language Models Need Sleep" proposes a
  sleep-like consolidation phase for long-horizon memory, addressing bottlenecks
  in persistent context storage. Open research agents like **QUEST** (2B–35B
  parameters) advance long-horizon fact-seeking and citation grounding, while
  the **CUSP benchmark** from Sakana/Stanford/Oxford/AI2 evaluates current model
  capabilities in science.
companies:
  - deepseek
  - google-deepmind
  - langchain-ai
  - anthropic
  - openai
  - alibaba
  - sakana-ai
  - stanford
  - oxford
  - ai2
models:
  - qwen-3.7
  - claude-opus-4.6
  - gpt-5.5
  - mythos
  - quest-2b-35b
topics:
  - harness-engineering
  - agent-infrastructure
  - coding-benchmarks
  - security-guidance
  - long-horizon-memory
  - context-compression
  - sleep-phase
  - math-problem-solving
  - fact-seeking
  - citation-grounding
  - science-evaluation
people:
  - sebastienbubeck
---




**a quiet day.**

> AI News for 5/23/2026-5/26/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap



**Agent Harnesses, Coding Benchmarks, and the Shift Beyond “Just the Model”**

- **Harness engineering is becoming the main differentiator for coding agents**: Several posts converged on the same thesis: the winning stack is now **model + harness + eval loop**, not just a stronger base model. A long Zhihu summary argued that [DeepSeek is explicitly building a harness team](https://x.com/ZhihuFrontier/status/2059180748637376843) to close the loop between model outputs, runtime feedback, validation, and correction, with a claimed cached-input cost advantage that would support tighter interaction/verification loops. In parallel, [Google’s Gemini Managed Agents guide](https://x.com/_philschmid/status/2059263980913229989) framed agent infra as a single API call to a managed harness with sandboxing, persistence, and mounts, while [LangChain’s updated `create_agent` docs](https://x.com/sydneyrunkle/status/2059280878694531280) and [dair.ai’s “harness” paper summary](https://x.com/dair_ai/status/2059294269698199929) formalized the same stack: **context governance, trustworthy memory, dynamic skill routing**.
- **Benchmarks are getting closer to real developer experience**: [DeepSWE](https://x.com/serenaa_ge/status/2059308218564890875), introduced as a new benchmark for agentic coding, got strong endorsement from practitioners; [@theo called it](https://x.com/theo/status/2059352130289651925) “the first code bench that actually aligns with how it feels to use these models coding.” It also created more separation at the top end than public SWE leaderboards often show. Related benchmark signals: [Qwen3.7 Max debuted at #4 on Code Arena: Frontend](https://x.com/arena/status/2059297720079393107), roughly on par with **Claude Opus 4.6** on agentic webdev tasks, and [Alibaba amplified the result](https://x.com/AlibabaGroup/status/2059317802935423028). Across the tooling stack, [Anthropic shipped a security-guidance plugin for Claude Code](https://x.com/ClaudeDevs/status/2059385239781384341) and reported a **30–40% reduction** in security-related PR comments in internal use, while [OpenAI highlighted GPT-5.5 in Codex at Databricks](https://x.com/OpenAIDevs/status/2059353117934899289) for more reliable document parsing.

**Research Agents, Long-Horizon Reasoning, and “Sleep” for Context Compression**

- **Math/science agents showed more evidence of capability overhang—conditional on the right harness**: The strongest cluster of tweets was around models tackling old open problems. A mathematician reported [Claude Mythos solving Erdős problem #90](https://x.com/__alpoge__/status/2059298565093196012), with follow-up detail that the model often converged to a **different, cleaner proof path** than OpenAI’s earlier route. This was echoed by [@_sholtodouglas](https://x.com/_sholtodouglas/status/2059303540150137244), [@kimmonismus](https://x.com/kimmonismus/status/2059311386820289013), and then sharpened by [Sébastien Bubeck](https://x.com/SebastienBubeck/status/2059343132991623186): with an **appropriate harness**, both **Mythos** and **GPT-5.5** can reproduce what an internal model had done one-shot, implying a large amount of latent capability not exposed by vanilla chat UX.
- **Long-horizon memory is resurfacing as a core bottleneck**: The paper [“Language Models Need Sleep”](https://x.com/iScienceLuvr/status/2059221770075562113) got notable attention. The mechanism is a **sleep-like consolidation phase** where recent context is converted into persistent fast weights before clearing the KV cache, moving compute into an offline pass while preserving wake-time latency. [dair.ai’s summary](https://x.com/dair_ai/status/2059333792775745619) emphasized the systems angle: this is an alternative to ever-growing KV caches for agents with long trajectories. This theme connected neatly with ongoing discussion about memory systems in agents, including [Omar’s pointer to Anthropic’s memory talk and Dream feature](https://x.com/omarsar0/status/2059285935376765214).
- **Open deep-research agents and science forecasting also advanced**: [QUEST](https://x.com/iScienceLuvr/status/2059223911011930606), a family of open **2B–35B** models for long-horizon fact-seeking, citation grounding, and report synthesis, was released as a general-purpose deep research agent. On the science-evals side, Sakana/Stanford/Oxford/AI2’s [CUSP benchmark](https://x.com/SakanaAILabs/status/2059166749761872342) found current models can often identify promising research directions but struggle much more with **whether** and **when** breakthroughs materialize.

**Model, Optimizer, and Architecture Updates**

- **Optimizer work remains lively, especially around Muon variants and schedule-free training**: [AMUSE](https://x.com/jueunkim_0525/status/2059127584601055426) proposes **Anytime MUon with Stable gradient Evaluation**, combining Muon with schedule-free-style gradient evaluation for stable anytime training without LR decay, reporting gains at **124M / 720M / 1B** scale and on ViT/ImageNet fine-tuning. Related implementation discussion came from [ClashLuke’s SFMuon snippet](https://x.com/Clashluke/status/2059187617997197553) and [kellerjordan’s Modded-NanoGPT result on Newton-Muon](https://x.com/kellerjordan0/status/2059353883881976044).
- **Sparse attention design space continues to diversify**: [MiniMax teased M3 as open source](https://x.com/MiniMax_AI/status/2059286515155599595), and follow-on technical commentary suggested a new **block-sparse two-stage attention** path. [@kimmonismus summarized the reported speedups](https://x.com/kimmonismus/status/2059302121489486335): **9.7× prefilling** and **15.6× decoding** at **1M tokens** versus M2. [@eliebakouch added](https://x.com/eliebakouch/status/2059321928205156568) that M3 appears to move back to **GQA-based** sparse attention with block selection on real KV, distinct from DeepSeek’s compressed-attention variants.
- **Vision/open model releases and ranking updates**: [PrismML released Bonsai Image 4B](https://x.com/PrismML/status/2059339157600969199), including **1-bit and ternary** variants intended to run locally on laptops and phones; a follow-up noted browser-local execution was possible at ~3GB footprint. On the closed side, [Microsoft’s MAI-Image-2.5](https://x.com/MicrosoftAI/status/2059344061358563838) debuted at **#3 on the Image Arena**, breaking a top-5 club previously dominated by OpenAI and Google, with [Arena reporting a 1,254 score](https://x.com/arena/status/2059346024632820146). Meanwhile, [Artificial Analysis measured Gemini 3.5 Flash](https://x.com/ArtificialAnlys/status/2059316050391634302) at up to **~280 output tok/s** with materially stronger agentic performance, but at **~5×** the cost of Gemini 3 Flash.

**Infra, Systems, and the Semiconductor Stack**

- **Huawei’s “τ scaling” paper was read mostly as an engineering roadmap, not a new law**: A very detailed thread argued [Huawei’s “A Time Scaling Theory for Multi-Layer Electronic Systems”](https://x.com/ZhihuFrontier/status/2059118295580852374) should be interpreted as a **strategic manifesto / white paper**. The core proposal is to treat **time constant τ**, not process node, as the unifying metric across device, chip, and datacenter scales. The most concrete claims concerned **LogicFolding** on a future Kirin design, including **+55% density**, **+41% energy efficiency**, and **+13% frequency** at fixed node, plus packaging/network ideas like a **Unified Bus** and **Hi-ONE optical I/O**. The same thread was careful to note missing validation artifacts—die photos, SEMs, workload details, yield curves—and to interpret the most eye-catching numbers as promising but **unverified**. Follow-up reactions also stressed that Huawei’s path may rely more on packaging and architecture than lithographic catch-up, e.g. [@josiah_leee citing Jensen’s point](https://x.com/josiah_leee/status/2059297861745963099) that most of Hopper→Blackwell’s gains came from non-node optimizations.
- **Datacenter power and inference supply constraints are becoming first-order concerns**: [SemiAnalysis published on the 800VDC transition](https://x.com/SemiAnalysis_/status/2059253624249696658), and [John Carmack recommended it](https://x.com/ID_AA_Carmack/status/2059382254191652896), highlighting crossovers from EV power electronics into datacenter design, including high-voltage SiC parts. Separately, [Epoch AI estimated a possible inference compute crunch](https://x.com/EpochAIResearch/status/2059372951338909717): demand appears to be growing faster than serving capacity, especially for long-context workloads. Their rough model suggested that while current global Blackwell supply could serve today’s demand under favorable assumptions, throughput degrades sharply with longer contexts and demand growth may already be outrunning supply.

**Production Tooling and Developer Infrastructure**

- **Serving/inference stacks got meaningful performance and observability updates**: [vLLM merged a Rust frontend](https://x.com/vllm_project/status/2059344804295942513) as a drop-in alternative to the Python API server, with early numbers showing **~837 req/s vs ~162 req/s** on a preprocess-heavy workload in a single process. [W&B launched an MCP server](https://x.com/wandb/status/2059384552725025226) to let coding agents inspect experiments and training runs, with a schema-first redesign aimed at avoiding context-window blowups. [Unsloth added support for running GPT, Claude, and other APIs inside its local UI](https://x.com/UnslothAI/status/2059277719633101291), including prompt caching and code execution.
- **Cloudflare, OpenRouter, and vector/retrieval vendors pushed the “productionization” layer**: [OpenRouter announced a $113M Series B](https://x.com/OpenRouter/status/2059277623629664758) and said weekly volume had grown from **5T to 25T tokens** over six months. [Cloudflare relaunched its startups program](https://x.com/kristianfreeman/status/2059188629780545973) with up to **$350k** in credits, while separate posts around **Think** and agent ergonomics emphasized durable turns, reconnects, stale-state handling, and recovery as key practical differentiators. On retrieval infra, [Booking.com discussed scaling to 100M+ embeddings](https://x.com/weaviate_io/status/2059227285639581729), including filtered vector search, reads-during-writes, concurrency, and human-in-the-loop evals for partner messaging agents.

**Top tweets (by engagement)**

- **Codex / agentic coding in practice**: The highest-signal product-use tweet was [@bunkaich showing Codex help reverse-engineer and patch firmware on a cheap MP3 player](https://x.com/bunkaich/status/2059178996126900703), with the workflow spanning chip inspection, OS extraction, binary analysis, and flashing a modified image.
- **DeepSWE benchmark launch**: [@serenaa_ge’s DeepSWE announcement](https://x.com/serenaa_ge/status/2059308218564890875) became the main reference point for “does this match real coding experience?” discussion.
- **Claude Code security plugin**: [@ClaudeDevs’ release](https://x.com/ClaudeDevs/status/2059385239781384341) stood out because it paired a concrete product launch with an internal metric: **30–40% fewer** security-related PR comments.
- **OpenRouter financing + production token growth**: [@OpenRouter’s $113M Series B](https://x.com/OpenRouter/status/2059277623629664758) is one of the clearer market signals that routing and multi-model infra are now seen as durable platform layers.
- **vLLM Rust frontend**: [@vllm_project’s merge announcement](https://x.com/vllm_project/status/2059344804295942513) mattered for anyone hitting CPU/API-server bottlenecks in high-throughput serving.

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen 3.7 Launch and Qwen 3.6 Local Performance

  - **[Waiting for Qwen 3.7 open weight... The new King has arrived...](https://www.reddit.com/r/LocalLLaMA/comments/1tjvz6l/waiting_for_qwen_37_open_weight_the_new_king_has/)** (Activity: 1217): **The [image](https://i.redd.it/j8qkty82qj2h1.png) is a benchmark/marketing comparison from the [Qwen3.7 blog](https://qwen.ai/blog?id=qwen3.7) positioning **Qwen3.7-Max** as a leading frontier model across agentic coding, software engineering, MCP/tool-use, reasoning, and knowledge evaluations versus **Qwen3.6-Plus**, **DS-V4-Pro Max**, **GLM-5.1**, **Kimi K2.6**, and **Claude Opus-4.6 Max**. The technical significance is that the slide frames Qwen3.7-Max as highly competitive with or ahead of Claude-class models on many benchmarks, though **Claude Opus-4.6 Max** still appears to lead on some tasks such as `ClawEval` and `CoWorkBench`. Commenters note that this is the **Max** model, not necessarily representative of smaller/open-weight releases, and speculate about a potential `3.7-122B-A17B` `MXFP4` model with `512k` context for local hardware such as Strix Halo.** The main debate is skepticism around open weights: commenters point out that **Qwen has historically not open-weighted the Max series**, so the title’s “waiting for open weight” framing may be unrealistic. Others caution not to expect a hypothetical `27B` model to match the shown Max-tier benchmark results.

    - Several commenters distinguish **Qwen Max** from likely open-weight releases, noting that *“Qwen has never open-weighted the Max series”* and warning not to expect a smaller `27B` variant to match Max-level benchmark performance. The implied technical takeaway is that any public/open-weight Qwen 3.7 release may use a different architecture/scale than the benchmarked flagship model.
    - One technical wishlist centers on a hypothetical **Qwen 3.7 `122B-A17B` MTP MXFP4** model with `512k` context, which commenters argue would be well-suited to **Strix Halo**-class local hardware. Another user references **Qwen 3.5 `397B-A17B` NVFP4**, claiming it fits on `4x RTX 6000 Pro` GPUs with enough memory headroom for roughly `10` concurrent `200k`-token sessions, positioning it as a potential “Opus at home” if Qwen 3.7 matches reported benchmarks.
    - A commenter argues that open-weight frontier releases may be less likely because highly capable local models can undermine provider monetization. They claim Qwen’s strategy has shifted from disruption toward monetized frontier competition, which could affect whether large MoE models like `397B-A17B` are released openly.

  - **[Qwen3.6 35Ba3 has changed my workflows and even how I use my computer](https://www.reddit.com/r/LocalLLaMA/comments/1tjwrp7/qwen36_35ba3_has_changed_my_workflows_and_even/)** (Activity: 567): **The post describes a local-agent workflow using **Qwen3.6 35B a3** via `pi`, where the user converts repeatable procedures into “skills” generated/documented by Codex, then reuses them for VPS DevOps, `docling` PDF→EPUB conversion, Playwright testing, code tickets, and OS-level shell tasks. A concrete example: WhatsApp audio → transcription in AnythingLLM → `content.md` → locally generated landing page, then a `plan.md` ticket queue executed by a “manager” `pi` process spawning fresh-context sub-agents with `pi -p @plan.md "Check the first Ticket with Status UNDONE and do it"`, marking tickets `DONE`, committing via git, and finally deploying via a VPS skill.** Commenters focused on operational concerns: what hardware can run this setup, whether the agent is sandboxed/trustworthy with OS access, and how hard `pi` is to adopt compared with other agentic tools such as Hermes.

    - A user reports running `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` via **Unsloth Studio** on an **MS-02** with a **24GB RTX Pro 4000 Blackwell SFF GPU**, consistently seeing **`>100 tokens/s`**. They compare performance to “unoptimized GGUFs” on a **Mac Studio M2**, using the MS-02 as a small remote GPU server for the Mac workstation, and note that **future MLX support in Unsloth** could improve Mac-side performance. Screenshot: [preview.redd.it](https://preview.redd.it/exwng3d4ik2h1.png?width=3966&format=png&auto=webp&s=03bf5de53b529f1b26f669c21834d9f1d69d16e0).

  - **[110 tok/s with 12GB VRAM on Qwen3.6 35B A3B and ik_llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1tjh7az/110_toks_with_12gb_vram_on_qwen36_35b_a3b_and_ik/)** (Activity: 565): **The post benchmarks **Qwen3.6-35B-A3B MTP** using byteshape’s [`IQ4_XS` `4.19 bpw` GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF) on an **RTX 4070 Super 12GB + Ryzen 7 9700X**, comparing upstream [`llama.cpp`](https://github.com/ggml-org/llama.cpp) vs [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) with `--ctx-size 131072`, `q8_0` KV cache, MTP draft max `3`, and `p_min=0.75`. Using the same [`mtp-bench.py`](https://gist.github.com/am17an/228edfb84ed082aa88e3865d6fa27090/) workload, upstream `llama.cpp` averaged **`89.76 tok/s`** with aggregate MTP accept rate **`0.9393`**, while `ik_llama.cpp` averaged **`110.24 tok/s`** over `16.64s`, a claimed **`23%` throughput gain**, despite lower aggregate accept rate **`0.8749`** in the updated results. The OP attributes practical fit to `--fit`/`--fit-margin 1664` on `ik_llama.cpp`, with OOM mitigation by raising `--fit-margin` to `1792` or `2048`, and notes that running the display on an iGPU frees essentially all `12GB` VRAM for inference.** Commenters focused on reproducibility: they requested the full upstream `llama.cpp` command and noted that several MTP-related PRs had merged recently, so benchmark timing may depend strongly on build date. One technical workaround suggested for single-GPU CachyOS/KDE users is a software-rendered Plasma Wayland session using `LIBGL_ALWAYS_SOFTWARE=1` and `GALLIUM_DRIVER=llvmpipe`, reducing idle VRAM from roughly `>1024MB` to `126MB` at the cost of slow/disabled compositor effects.

    - A CachyOS/KDE Wayland user described a VRAM-saving workaround for single-GPU systems: create a custom SDDM session that forces KDE Plasma to render via CPU using `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe`, and `KWIN_COMPOSE=Q`. They reported KDE Wayland idle VRAM dropping from **> `1024 MB`** to **~`126 MB`**, freeing nearly a gigabyte of VRAM for running the 35B model, at the cost of disabled or very slow compositor animations.
    - Several commenters focused on whether the reported `110 tok/s` comes from **ik_llama.cpp** having better MTP/speculative decoding behavior than upstream `llama.cpp`. One noted that ik_llama.cpp’s acceptance rate was reportedly **never below `0.790`**, while llama.cpp dropped as low as **`0.477`**, asking for the exact llama.cpp command/settings and noting that multiple MTP-related PRs had landed in llama.cpp within the previous 24 hours.
    - A commenter asked about the `IQ4_XS` quantization used for **Qwen3.6 35B A3B**, noting it appears to be the lowest-memory Q4 quant and requesting details on both model quality/intelligence impact and the final VRAM/RAM split. This highlights the key tradeoff for 12 GB VRAM runs: fitting the model via aggressive quantization versus maintaining reasoning quality and avoiding excessive CPU/RAM offload bottlenecks.


### 2. Open-Source AI Funding and Legal Pressure

  - **[Heretic has been served a legal notice by Meta, Inc.](https://www.reddit.com/r/LocalLLaMA/comments/1tjmvx6/heretic_has_been_served_a_legal_notice_by_meta_inc/)** (Activity: 2705): **The **Heretic Free Software Project** says it received an email legal notice from a provider representing **Meta Platforms, Inc.** and has removed derivatives of Meta’s **Llama** model weights from Heretic-controlled repositories. The project also announced an official German-hosted [Codeberg mirror](https://codeberg.org/p-e-w/heretic) and says it is working on “technological measures” to preserve access to Heretic-created models without relying on a single hosting provider; the post sarcastically cites Llama as “among the 200 best” models, “trailing only `168` other models” on the [LM Arena](https://lmarena.ai/) leaderboard.** Top comments focused on the post’s sarcasm, especially the “`168` other models” leaderboard jab, and criticized Meta’s enforcement given allegations that Meta used torrented books or copyrighted material in model training.

    - A commenter highlights the legal-response wording that contextualizes **Meta’s Llama family** against current open/model competition: it is described as ranking within the top `200` on **LM Arena**, but behind `168` models from `23` competitors. The technical implication raised is that Meta’s naming-enforcement posture is being contrasted with Llama’s relative benchmark standing and a perceived slowdown in recent model releases.

  - **[DeepSeek is pushing forward with $10.29 billion financing round, with Liang Wenfeng committing to continue developing open-source AI models rather than pursuing short-term commercialization goals](https://www.reddit.com/r/LocalLLaMA/comments/1tkfvvj/deepseek_is_pushing_forward_with_1029_billion/)** (Activity: 797): ****DeepSeek** is reportedly advancing a **`$10.29B` financing round**, with founder **Liang Wenfeng** reiterating an **AGI-oriented roadmap** and a commitment to continue releasing/opening AI models rather than prioritizing near-term commercialization, per [Bloomberg](https://www.bloomberg.com/news/articles/2026-05-22/deepseek-founder-declares-agi-goal-as-10-billion-round-advances). Commenters framed this as a strategic bet that model advantages have short half-lives and that open research can accelerate iteration faster than closed talent/model moats.** Top comments argued that local inference users are a small minority, so releasing weights would not materially hurt SaaS/API revenue for labs like OpenAI, Anthropic, Google, or Mistral; any architectural lead was estimated to have roughly a `~1 year` shelf life. Another commenter said open models are already *“good enough”* for coding assistance around **GLM 5.1**-level capability, and the next frontier is compressing similar capability into smaller, faster, more efficient models.

    - Commenters argued that model weights have a short technical/commercial shelf life: architectural advantages may last only ~`1 year`, while local inference users are a tiny minority compared with hosted API users. The claim was that **OpenAI, Anthropic, Google, Mistral, etc.** could release weights without materially harming revenue, because most users lack the hardware/interest to run even a `9B` model locally.
    - One technical thread framed current open models as reaching “good enough” capability for coding assistance, citing **GLM 5.1** as a threshold model. The remaining priority, according to the comment, is not raw intelligence but distillation/compression: preserving that coding capability in smaller, faster, and more efficient deployable models.
    - A commenter pointed to DeepSeek’s own report saying they are working on adding multimodal capabilities: [DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf). The notable technical angle was that DeepSeek is continuing model expansion despite GPU/export-sanction constraints, suggesting continued progress under limited hardware access.



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code Workflows and Anthropic Agent Training

  - **[Claude Code dropped /workflows](https://www.reddit.com/r/ClaudeCode/comments/1tkjy4u/claude_code_dropped_workflows/)** (Activity: 1074): **The image is a simple Claude-branded announcement graphic for **`/workflows`** in Claude Code, tied to the post’s claim that Anthropic briefly exposed a new workflow system in `Claude Code 2.1.147` before removing it from the changelog. The claimed technical significance is replacing an LLM-based orchestrator with a `workflow.js` code-driven controller: structured phases, parallel fan-out, conditionals/loops/budgets, retries, background execution, and reduced context-window “token tax” by passing sub-agent outputs between phases instead of through the main chat context. Image: [https://i.redd.it/6tuq1a2i3p2h1.png](https://i.redd.it/6tuq1a2i3p2h1.png).** Commenters were skeptical that this is a fundamentally new multi-agent pattern, pointing to existing Claude Code [agent teams](https://code.claude.com/docs/en/agent-teams). Others dismissed it as a low-priority feature compared with wanting a newer/better model such as “Opus 4.5.”

    - A commenter linked **Anthropic’s existing Claude Code “agent teams” docs** (https://code.claude.com/docs/en/agent-teams), noting that the described `/workflows` pattern—*“one main agent (an LLM) decides what sub-agents to spawn, holds every intermediate result, and plans the next step”*—overlaps with already documented multi-agent orchestration concepts.
    - The reported `/workflows` feature appears to have been transient: one commenter says it was visible in the changelog earlier but **Anthropic has since taken it down**, providing a screenshot mirror of the removed changelog entry (https://preview.redd.it/720w663mcp2h1.png?width=2056&format=png&auto=webp&s=d7afca73806dd159eff3141db0f61de5a37526a8).
    - One user compared the feature to their own custom orchestration stack built around **skills + YAML + a JavaScript CLI**, implying `/workflows` may formalize a pattern developers are already implementing manually for repeatable Claude Code task pipelines.

  - **[Anthropic officially launched 13+ FREE AI courses with certificates (Including Agentic AI and Claude Code!)](https://www.reddit.com/r/ClaudeAI/comments/1tjpfh8/anthropic_officially_launched_13_free_ai_courses/)** (Activity: 2547): ****Anthropic** is offering a free official training catalog via its Skilljar-based academy, reachable from [Anthropic Learn](https://www.anthropic.com/learn), with certificates for courses covering **Claude**, **Claude Code**, **Claude API**, **MCP / agentic workflows**, and deployment tracks for **Amazon Bedrock** and **Google Cloud Vertex AI**. The technically notable content called out is the MCP material, including advanced topics around `STDIO` and `StreamableHTTP` transports, plus Claude Code modules for codebase editing, test execution, and “Plan Mode.” A separate free [CodeSignal](https://codesignal.com/) track, “Developing Claude Agents,” is mentioned for interactive Python/TypeScript labs and certificates.** Commenters confirm the Skilljar courses are legitimate because they are linked from Anthropic’s official site, and one user who completed `10/15` courses specifically recommends the MCP and advanced MCP modules as *“worth the squeeze.”*

    - Several commenters confirmed the Skilljar courses are legitimate **Anthropic** training materials, noting the course portal is linked from [anthropic.com/learn](https://www.anthropic.com/learn) rather than being a third-party scam or repost.
    - One user who completed `10/15` courses specifically highlighted the **MCP** and **MCP Advanced Topics** modules as worthwhile, citing practical coverage of `STDIO` and `StreamableHTTP` transport protocols for Model Context Protocol integrations.
    - A few users noted the catalog is not newly launched and has been available for months; one commenter who completed two courses described them as *“quite basic”*, suggesting the material may be more introductory than advanced for experienced AI developers.


### 2. Z-Image 6B, Gemini 3.5 Flash and OpenAI Math Updates

  - **[Tencent released Z-Image 6B with pixel space gen. No VAE &amp; 1k Resolution.](https://www.reddit.com/r/StableDiffusion/comments/1tkipk6/tencent_released_zimage_6b_with_pixel_space_gen/)** (Activity: 899): **The [image](https://i.redd.it/69r8ttxmvo2h1.jpeg) is a sample collage for **Tencent/Z-Image 6B / L2P**, illustrating `1024px`-class **pixel-space image generation** across portraits, animals, fantasy scenes, vehicles, and stylized compositions, with the key technical claim being generation **without a VAE**. The post links the project page at [nju-pcalab.github.io/projects/L2P](https://nju-pcalab.github.io/projects/L2P/) and a commenter points to model files on Hugging Face: [zhen-nan/L2P](https://huggingface.co/zhen-nan/L2P/tree/main).** Commenters mainly focused on the architectural trend — *“Everyone going for No-VAE now huh”* — and questioned practical quality with *“Is it any good?”* rather than providing benchmarks or detailed evaluations.

    - A commenter points to the model files on Hugging Face: **zhen-nan/L2P** at [https://huggingface.co/zhen-nan/L2P/tree/main](https://huggingface.co/zhen-nan/L2P/tree/main), relevant for readers wanting to inspect/download Tencent’s **Z-Image 6B** release and its claimed **pixel-space generation / no-VAE** setup.
    - Several comments highlight the broader technical trend toward **No-VAE / pixel-space image generation**, with one user noting *“Everyone going for No-VAE now huh”*. This is notable because avoiding a VAE changes the compression/latent bottleneck tradeoff and may affect reconstruction fidelity, memory cost, and native high-resolution generation such as the post’s claimed `1k` resolution.
    - One commenter raises a comparison to **Lodestone**, asking whether Tencent’s approach learned from Lodestone’s no/low-latent direction or whether Lodestone could learn from Z-Image. The thread does not provide benchmark data, but the technical comparison suggests interest in converging open-weight architectures for direct pixel-space diffusion/flow generation.

  - **[Google's latest creation: Gemini 3.5 Flash vs all](https://www.reddit.com/r/singularity/comments/1tjoarz/googles_latest_creation_gemini_35_flash_vs_all/)** (Activity: 1503): **The post reports a simple arithmetic failure in **Google Gemini 3.5 Flash** via the Gemini app: for the prompt `300+140=460` / “Is this correct? Breakdown?”, the shared Gemini run allegedly accepts the incorrect sum, while comparison runs were linked for [Claude](https://claude.ai/share/8383747a-aaf1-4f6c-a516-0e839f46a698), [Grok](https://grok.com/share/bGVnYWN5_3c63e371-eb9d-46c3-8ba2-0c745c6795a2), and [ChatGPT](https://chatgpt.com/share/6a0f1e13-a0c8-8328-b989-1ac51b92e81c). Commenters reproduced the issue and attributed it to Gemini app inference settings: **“Standard”/default thinking behaves like minimum or no reasoning**, while **Extended thinking** or AI Studio with higher thinking settings reportedly returns the correct `300 + 140 = 440`.** The main debate is that this is less evidence about the base model’s capability and more about product-level serving configuration: commenters argue the **Gemini app is “nerfed”** relative to AI Studio, especially under default/minimum thinking settings. The OP frames the result as embarrassing given claimed SOTA/finance-agent rankings, while others suggest benchmark performance may not reflect low-effort app defaults.

    - Users reported that the apparent failure depends heavily on Gemini’s **thinking level**: switching to **Extended thinking** fixes the answer, while **Standard** was characterized as effectively *“doesn’t think at all.”* Another commenter reproduced the same output via a screenshot ([preview image](https://preview.redd.it/whzg30z8hi2h1.png?width=1557&format=png&auto=webp&s=192481783e75626c47648f50954c4c8fe8fb60a7)) and claimed the Gemini app defaults to something like **minimum thinking**, whereas **AI Studio** with even **Low** thinking avoids the mistake.
    - A technical comparison was raised around **tool-calling behavior**: one commenter argued Gemini’s weakness is not necessarily raw reasoning but **tool-routing logic**, noting that ChatGPT would likely delegate the task to **Python** rather than solve it purely in-model. This implies benchmark results may depend on whether the model is allowed to invoke tools and how reliably it decides to use them.

  - **[Math grad student friend says we're cooked](https://www.reddit.com/r/OpenAI/comments/1tkcxxi/math_grad_student_friend_says_were_cooked/)** (Activity: 825): **The [image](https://i.redd.it/l7gd5lx9in2h1.png) is a **tweet screenshot** relaying a math grad student's alarmed reaction to a claimed recent **Erdős proof**, framed by the post title *“Math grad student friend says we're cooked.”* It does **not provide technical details** of the proof, theorem statement, model, benchmark, or verification process; its significance is contextual/social: a mathematician characterizes the result as previously “completely unapproachable” and says OpenAI’s announcement was “exceedingly tacky and in bad taste.”** Comment discussion is mostly non-technical and meme-driven, pivoting to jokes about “OnlyFans but for nerds.” One commenter questions what “exceedingly tacky and in bad taste” means, but there is no substantive debate about the mathematics or AI capability claim.

    - A commenter argues that the perceived safety of “creative and intellectual” work has weakened as AI systems have begun to show capability in **mathematics, theorem proving, and research-level reasoning**. The technical takeaway is that automation risk may not correlate cleanly with whether a task is repetitive; instead, advanced reasoning benchmarks and formal proof systems are increasingly relevant to assessing AI impact.




# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.