---
id: MjAyNS0x
title: not much happened today
date: '2026-06-18T05:44:39.731046Z'
description: >-
  **GLM-5.2** from **Zhipu** emerged as a leading open-weight model with
  innovative **IndexShare** sparse-attention enabling efficient **1M-token
  inference**, praised as comparable to **GPT-5.5** and **Opus 4.8** but lacking
  vision support. Other notable open models include **Laguna M.1** by **Poolside
  AI**, a **70-layer sparse MoE** optimized for long-horizon coding, and **North
  Mini Code** by **Cohere** with **4-bit quantization** and local deployment
  support via **Ollama**. The focus is shifting from standalone models to
  integrated systems combining **model + harness + memory + SCM**, exemplified
  by **Noumena Code / ncode** addressing challenges in concurrent code agent
  workflows. Automation tools like **Codex Record & Replay**, **Cursor's
  /automate**, and **Artifacts in Claude Code** enhance teachability,
  reusability, and security in AI-assisted coding workflows.
companies:
  - zhipu
  - hugging-face
  - llama-cpp
  - unsloth
  - poolsideai
  - cohere
  - ollama
  - openai
  - cursor_ai
  - claude
  - cognition
models:
  - glm-5.2
  - opus-4.8
  - gpt-5.5
  - laguna-m.1
  - north-mini-code
  - codex
topics:
  - sparse-attention
  - 1m-token-inference
  - open-weight-models
  - model-architecture
  - long-context
  - mixture-of-experts
  - quantization
  - local-deployment
  - workflow-automation
  - code-agents
  - software-configuration-management
  - automation-primitives
  - security
  - model-harness
  - agentic-coding
people:
  - rasbt
  - jeremyphoward
  - matvelloso
  - artificialanlys
  - zixuanli_
  - _xjdr
  - gneubig
  - _catwu
---


**a quiet day.**

> AI News for 6/17/2026-6/18/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**GLM-5.2’s Breakout, Open-Weight Coding Progress, and New Open Models**

- **GLM-5.2 became the day’s consensus open-model story**: multiple practitioners independently described **Zhipu’s GLM-5.2** as the first open-weight model that feels plausibly frontier-adjacent in daily use. [@rasbt](https://x.com/rasbt/status/2067612153020838055) highlighted the architecture change: beyond **MLA** and **DSA** inherited from prior GLM/DeepSeek-style designs, GLM-5.2 adds **IndexShare**, reusing sparse-attention top-k indices across groups of layers to reduce the cost of **1M-token inference**. Community sentiment was unusually strong: [@jeremyphoward](https://x.com/jeremyphoward/status/2067757468189679764) called it “at least as good as Opus 4.8 and GPT 5.5” for his use, while noting its major gap is lack of vision support; [@matvelloso](https://x.com/matvelloso/status/2067791546335019439) said it was the first open model that cleared his “daily driver” bar; [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2067761754990686483) placed it between **GPT-5.5** and **Opus 4.8** on a new agentic knowledge-work eval. Zhipu also pushed availability aggressively: [free via Hugging Face Inference Providers for a limited window](https://x.com/Zai_org/status/2067647208451604617), [local GGUF support via llama.cpp/Unsloth](https://x.com/ZixuanLi_/status/2067626723986841765), and strong app-dev deltas from **21/70 to 48/70** internal tasks vs GLM-5.1 per [@ZixuanLi_](https://x.com/ZixuanLi_/status/2067803136283005393).  
- **Other open model releases also mattered**: [@poolsideai](https://x.com/poolsideai/status/2067623353230217448) released **Laguna M.1** weights under **Apache 2.0** with **256K context**; [@vllm_project](https://x.com/vllm_project/status/2067629972941132269) described it as a **70-layer sparse MoE**, **225B total / 23B active**, **256 experts**, **top-k=16**, optimized for long-horizon agentic coding with interleaved reasoning/tool use. Poolside later showed a **3-bit MLX build** on Apple Silicon at **~26 tok/s** and **~100 GB peak memory** on an M3 Max 128 GB machine [@poolsideai](https://x.com/poolsideai/status/2067711022115471532). On the smaller end, [@cohere](https://x.com/cohere/status/2067671125073576382) pushed **North Mini Code** accessibility with **4-bit quantization**, **Ollama** support, and free **OpenRouter** access; [@ollama](https://x.com/ollama/status/2067671359506022674) amplified support for open local deployment.

**Agent Harnesses, Workflow Automation, and Coding Tooling**

- **The center of gravity keeps moving from “model” to “model + harness + memory + SCM”**: [@_xjdr](https://x.com/_xjdr/status/2067596405162848386) published a detailed argument that traditional **git/GitHub** workflows break under dozens to hundreds of concurrently running code agents: stale worktrees, diverged review state, environment setup overhead, and poor state synchronization. His proposed replacement stack combines **virtual shallow checkouts**, **jj**, **Sapling-like commit stacks**, cloud sync, file-level ACLs, and vertical integration from model to SCM to remote runtimes, now productized via **Noumena Code / ncode** with later free access to its inference engine and model [@_xjdr](https://x.com/_xjdr/status/2067741647941832818). In the same vein, [@gneubig](https://x.com/gneubig/status/2067651018217648595) argued benchmarks should evaluate the **harness + LLM pair**, not either in isolation; his OpenHands comparison found different winners depending on model family and cost profile.  
- **Automation primitives are getting more teachable and reusable**: [@OpenAIDevs](https://x.com/OpenAIDevs/status/2067681320281723113) introduced **Codex Record & Replay**, letting users demonstrate a workflow once and turn it into an inspectable skill; [@cursor_ai](https://x.com/cursor_ai/status/2067683814516858962) launched **/automate**, where Cursor configures triggers/instructions/tools from a natural-language task, adding Slack emoji triggers, GitHub triggers, and computer-use for cloud agents. [@ClaudeDevs](https://x.com/ClaudeDevs/status/2067672094209675373) shipped **Artifacts in Claude Code**, enabling agents to turn ongoing work into shareable live pages; [@_catwu](https://x.com/_catwu/status/2067674836726694200) said this has already changed internal workflows for architecture changes and prototype sharing.  
- **Security and review are becoming first-class agent tasks**: [@cognition](https://x.com/cognition/status/2067649690921820212) added automatic **security review** to Devin Review, and [@shayanshafii](https://x.com/shayanshafii/status/2067667505905332352) framed **Devin for Security** as addressing the longstanding “finding vs fixing” split in AppSec by using agentic reasoning plus harnessing to chain lower-severity findings into confirmed severe exploits.  
- **Top tweet in tooling by engagement**: [@OpenAIDevs’ Codex Record & Replay](https://x.com/OpenAIDevs/status/2067681320281723113) was the most engaged high-signal developer-tool post in the set, reflecting strong appetite for teach-by-demonstration agent workflows.

**Benchmarks, Evaluations, and Long-Horizon Agent Measurement**

- **Artificial Analysis launched a more realistic agentic knowledge-work benchmark**: [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2067744637155226101) introduced **AA-Briefcase**, built around **multi-week projects**, thousands of fragmented inputs, Slack/email/document corpora, and deliverables like financial models and board decks. On this benchmark, **Claude Fable 5** led at **1587 Elo**, with **Opus 4.8** next at **1356**, and **GLM-5.2** at **1266** as the strongest non-Anthropic open-ish entrant mentioned. Importantly, the benchmark exposes both quality and economics: **Fable 5 averaged $31/task**, **Opus 4.8 $10.40**, **GPT-5.5 xhigh $3.68**, **GLM-5.2 $2.40**, while some weaker options were orders of magnitude cheaper. The broader lesson is not just leaderboard movement, but that **real-world long-horizon knowledge work remains hard**: the top model satisfied all rubric criteria on only **3%** of tasks.  
- **Additional benchmark work pushed in the same direction**: [@terminalbench](https://x.com/terminalbench/status/2067635273652134002) released **Terminal-Bench Challenges** for long-horizon, token-intensive single tasks; [@omarsar0](https://x.com/omarsar0/status/2067618845926510770) highlighted **SkillWeaver**, which treats agent routing as **compositional skill retrieval + DAG planning** rather than single-tool selection; [@arena](https://x.com/arena/status/2067680639068094958) described **Agent Arena’s causal tracing** approach for quantifying the value of human/AI collaboration via signals like steerability, bash recovery, and tool hallucination. There was also continued meta-critique of agent eval quality from [@isidoremiller](https://x.com/isidoremiller/status/2067633428774682697), who argued current analytics-agent benchmarks are often measuring the wrong things.

**Inference, Retrieval, and Systems Efficiency**

- **Inference and retrieval optimization remained a strong secondary theme**: [@liquidai](https://x.com/liquidai/status/2067610173024219225) released **LFM2.5-Embedding-350M** and **LFM2.5-ColBERT-350M**, multilingual retrieval models covering **11 languages** with claimed **1.5 ms** end-to-end retrieval latency on their enterprise stack. [@CoreWeave](https://x.com/CoreWeave/status/2067613387056709982) claimed **289 tok/s** serving for **Kimi K2.7 Code**, emphasizing provider-side price/perf as a differentiator. [@vllm_project](https://x.com/vllm_project/status/2067641904049885492) reported **Ray Serve LLM + vLLM** improvements of up to **4.4x throughput** on prefill-heavy workloads and **24x** on decode-heavy workloads via direct streaming, a Ray V2 executor backend, and HAProxy-based ingress routing.  
- **Vector DB / parsing economics improved materially**: [@turbopuffer](https://x.com/turbopuffer/status/2067630644243382733) cut its base plan from **$64 to $16/month**, then added **i8 vectors** for **4x lower bytes/dim** and up to **75% lower storage/query costs** when paired with quantization-aware embeddings [@turbopuffer](https://x.com/turbopuffer/status/2067701891451273615). On the document side, [@llama_index](https://x.com/llama_index/status/2067657865200824560) and [@jerryjliu0](https://x.com/jerryjliu0/status/2067679507126124858) shipped **LiteParse v2.1**, claiming the fastest open, model-free **PDF/document → markdown** pipeline, outperforming several OSS parser baselines on three benchmarks.

**Health, Medicine, and Safety/Alignment Research**

- **OpenAI had a notably health-heavy day**: [@OpenAI](https://x.com/OpenAI/status/2067625110199247353) shared a **NEJM AI** study with Boston Children’s/Harvard showing **o3 Deep Research** helped clinicians revisit previously unsolved pediatric rare-disease cases; [@gdb](https://x.com/gdb/status/2067648020934701541) summarized this as helping find **18 new diagnoses across 376 previously unsolved cases**. Separately, [@OpenAI](https://x.com/OpenAI/status/2067672740539306261) said **GPT-5.5 Instant** is now on par with frontier “Thinking” models for health-related questions, supported by feedback from **hundreds of physicians across 60 countries, 49 languages, and 26 specialties**.  
- **OpenAI also published broader alignment work**: [@OpenAI](https://x.com/OpenAI/status/2067722688165232654) introduced research on training models to be **broadly and persistently beneficial**, claiming RL on health-domain conversations reinforcing traits like truthfulness, humility, and concern for human welfare improved **44/53** internal/external alignment and benefits evals, and that even health-only beneficial-trait training improved **17/19 non-health alignment evals** including deception and coding reward hacking per [@thekaransinghal](https://x.com/thekaransinghal/status/2067726279277981829). This is early, but it is one of the clearer attempts to operationalize “generalized beneficial behavior” instead of narrow refusal-style safety.

**Top tweets (by engagement)**

- **[@narendramodi on meeting Mistral’s Arthur Mensch](https://x.com/narendramodi/status/2067600763829059760)**: mostly geopolitical rather than technical, but notable as another signal of national-level AI diplomacy and India partnership positioning.
- **[@OpenAIDevs on Codex Record & Replay](https://x.com/OpenAIDevs/status/2067681320281723113)**: the day’s biggest developer-tool post; strong validation for demonstration-based automation as a product surface.
- **[@ClaudeDevs on Enterprise-Managed Auth for MCP](https://x.com/ClaudeDevs/status/2067655887662272723)**: highly engaged enterprise infrastructure announcement; central auth for MCP connectors via IdP is important plumbing for enterprise agent deployment.
- **[@OpenAI on GPT-5.5 Instant health improvements](https://x.com/OpenAI/status/2067672740539306261)**: one of the strongest signals that mainstream product models are being tuned around domain-specific utility with physician-led eval loops.
- **[@jeremyphoward on GLM-5.2](https://x.com/jeremyphoward/status/2067757468189679764)** and **[@ollama on scaling GLM-5.2 cloud capacity](https://x.com/ollama/status/2067730812645298626)**: together capture the day’s open-model mood—GLM-5.2 wasn’t just released; it was immediately pressure-tested, praised, and operationalized.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. GLM-5.2 Local Access and Quantization

  - **[GLM-5.2 is a win for local AI](https://www.reddit.com/r/LocalLLaMA/comments/1u8ai2a/glm52_is_a_win_for_local_ai/)** (Activity: 1623): **The post argues **GLM-5.2** is significant for local AI despite its `753B` total-parameter MoE footprint (`~40B` active/token), because its **MIT license**, `28.5T`-token pretraining scale, claimed `1M` context / `131k` output support, and frontier-level coding-agent behavior could enable high-quality synthetic-data distillation into `8B`/`70B` local models. The author estimates inference memory from `~744–890GB` for FP8 down to `~176–180GB` for dynamic 1-bit quantization, with KV-cache overhead of roughly `15–20GB`, `7.5–10GB`, or `3.5–5GB` per `100k` tokens for FP16/BF16, 8-bit, or 4-bit cache respectively, while noting the table was AI-generated and approximate.** Commenters report strong API-based impressions, with one claiming GLM-5.2 and MiniMax/Mimi models have largely closed the gap to proprietary frontier models and that they would trust GLM-5.2 over Opus 4.8. Others push back on “local” practicality: some users with `512GB` Macs, GB10 clusters, or multiple `128GB` AMD AI Max systems may run it, but the hardware requirements are increasingly “unobtanium,” motivating interest in a distilled or dense `70B` variant.

    - Several commenters frame **GLM-5.2** as narrowing the gap between large open-weight/API-accessible models and frontier closed models, with one user saying that alongside **MiniMax M3 / Mimi-V2.5-Pro**, the “distance between the frontier and the big open models has mostly collapsed.” They specifically compare trust and interaction quality against **Claude Opus 4.8** and **GPT-5.5**, while acknowledging there remain “frontier problems” these models still cannot solve.
    - Hardware feasibility was debated: while `512GB` Macs, **GB10 clusters**, or multiple **AMD AI MAX 128GB** systems may technically run models at this scale, one commenter argues that **Mac Studio-class setups become impractical at large context lengths**. The cited bottleneck is poor **PP/TG** performance at `50K+` context windows—“you can run it but it’s not usable”—highlighting the distinction between fitting a model in memory and achieving acceptable generation throughput.
    - A commenter highlights the parameter-efficiency claim that **GLM-5.2** reaches roughly **Claude Opus 4.6-level capabilities** in **<800B parameters**, and speculates that smaller derivatives such as **GLM-5.2 Air** at `200B–300B` or **GLM-5.2 Flash** around `40B` could be especially compelling. They also connect this to expected next-generation open models like **Gemma 5** and **Qwen 4**, assuming continuation of prior capability gains from **Gemma 4** and **Qwen 3.5/3.6**.

  - **[unsloth GLM-5.2-GGUF , including 2bit at 238GB](https://www.reddit.com/r/LocalLLaMA/comments/1u98iig/unsloth_glm52gguf_including_2bit_at_238gb/)** (Activity: 412): ****Unsloth** appears to have published **GLM-5.2 GGUF** quantizations, with the smallest/2-bit variant still reported at roughly `238GB`, implying very high RAM/VRAM requirements even for aggressively quantized local inference. A commenter provided torrent mirrors for multiple GGUF quant formats—`UD-IQ1_S`, `UD-IQ1_M`, `UD-IQ2_XXS`, `UD-IQ2_M`, `UD-Q2_K_XL`, `UD-IQ3_XXS`, `UD-IQ3_S`, `UD-Q3_K_XL`, `UD-Q4_K_XL`, and `Q8_0`—hosted via `nostr.download`, noting they can fall back to Hugging Face web servers as webseeds; related code is on [GitHub Gist](https://gist.github.com/etemiz/c5d3e3c9b3a108b2d507714ff8ad2eed).** Commenters focused on hardware impracticality—e.g. being *“230 gb short on ram”*—and expressed hope that cheaper Chinese GPUs could make models of this scale more accessible. There was also concern about possible future availability restrictions, motivating the torrent mirrors *“in case it is banned.”*

    - A commenter mirrored multiple **GLM-5.2 GGUF quantizations** as torrents, covering `UD-IQ1_S`, `UD-IQ1_M`, `UD-IQ2_XXS`, `UD-IQ2_M`, `UD-Q2_K_XL`, `UD-IQ3_XXS`, `UD-IQ3_S`, `UD-Q3_K_XL`, `UD-Q4_K_XL`, and `Q8_0`. They note the torrent setup can fall back to **Hugging Face web servers** when there are no seeders, and shared the generation/distribution code via [this gist](https://gist.github.com/etemiz/c5d3e3c9b3a108b2d507714ff8ad2eed).
    - There was interest in evaluating the extreme low-bit release beyond size alone: one commenter specifically asked for **SWE-bench results for the `2bit` quantization**, implying concern about whether the `238GB` 2-bit GGUF preserves coding-agent performance after heavy quantization.

  - **[GLM-5.2 inference is free on Hugging Face for the next 6 hours](https://www.reddit.com/r/LocalLLaMA/comments/1u99hel/glm52_inference_is_free_on_hugging_face_for_the/)** (Activity: 445): **The image is a promotional tweet announcing a **limited-time free inference window** for **GLM-5.2** on **Hugging Face Inference Providers** for `6 hours`, accessible through providers including **Zai, Together AI, Novita, Fireworks, and DeepInfra**. The post links Hugging Face’s [Inference Providers documentation](https://huggingface.co/docs/inference-providers/index), a sample [Hugging Face Chat prompt](https://huggingface.co/chat/r/aFATtCW?leafId=ed28d5b0-d99b-40be-ba8b-315b1f450e5a), and the announcement image: [https://i.redd.it/pi7i24q2828h1.png](https://i.redd.it/pi7i24q2828h1.png).** Comments were mostly skeptical or joking: one user compared the offer to a *“drug dealer tactic,”* while another suggested the free promotion may be contributing to Hugging Face/server congestion and making the service *“basically unusable.”*

    - Users reported Hugging Face inference capacity issues during the free **GLM-5.2** window, with one commenter saying the servers had been *“basically unusable the last few days”*. The only technical signal in the thread is that the promotion may be causing high load/queueing or degraded availability for hosted inference.


### 2. Edge Local Inference Releases

  - **[Gemma 4 E2B running in-browser at 255 tok/s using WebGPU kernels written by Fable 5](https://www.reddit.com/r/LocalLLaMA/comments/1u8g3d0/gemma_4_e2b_running_inbrowser_at_255_toks_using/)** (Activity: 808): **A WebML demo releases **Gemma 4 E2B** in-browser inference via custom **WebGPU kernels** reportedly optimized with **Fable 5** before shutdown, achieving about `255 tok/s` on an **M4 Max**. The demo/kernels are on [Hugging Face Spaces](https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels), using Google’s [`gemma-4-E2B-it-qat-mobile-transformers`](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers) model; the linked Reddit video was inaccessible due to Reddit `403 Forbidden`.** Comments noted browser support limitations—specifically *“No Firefox love”*—and praised the UI, with a request to open-source it. One commenter pointed to a related Hugging Face Gemma optimization effort claiming `500 TPS` on an **A10G** with allegedly no quality loss: [dashboard](https://gemma-challenge-gemma-dashboard.hf.space/).

    - A commenter linked a related **Hugging Face Gemma Challenge dashboard** where multi-agent optimization is reportedly targeting Gemma E4B inference on an **NVIDIA A10G**, reaching around `500 TPS` with *allegedly* no quality loss: https://gemma-challenge-gemma-dashboard.hf.space/. This provides a useful native/GPU-server comparison point against the post’s in-browser WebGPU result of `255 tok/s`.
    - Several comments raised practical benchmarking and deployment questions: how the in-browser WebGPU kernels compare with **llama.cpp** or other non-browser inference stacks, whether Firefox is supported, and how the downloaded ~`2 GB` model artifact can be cleared from browser storage after use.

  - **[I released Inflect-Nano, an ultra-extreme tiny 4.63m parameter TTS model.](https://www.reddit.com/r/LocalLLaMA/comments/1u8p9s1/i_released_inflectnano_an_ultraextreme_tiny_463m/)** (Activity: 1040): **The image is a **technical promotional infographic** (not a meme) for **Inflect-Nano-v1**, a tiny local TTS model advertised at `4.63M` total inference parameters: `3.46M` acoustic model + `1.17M` vocoder, producing `24 kHz` English single-voice audio. It visualizes the core claim from the post: Inflect-Nano is dramatically smaller than other TTS systems—about `17×` smaller than Kokoro, `108×` smaller than Chatterbox, and roughly `950–1000×` smaller than Fish Audio S2 Pro—positioning it as an embedded/offline/local-assistant baseline rather than SOTA-quality speech synthesis. Image: [https://i.redd.it/qmsrjpq28x7h1.png](https://i.redd.it/qmsrjpq28x7h1.png); model: [Hugging Face](https://huggingface.co/owensong/Inflect-Nano-v1).** Comments were mostly impressed by the size/functionality tradeoff, with users asking for an architecture/build explanation and whether it could run on very constrained hardware like an ESP32 with ML acceleration. One commenter joked that “ebooks” can be larger than the model, underscoring how unusually small the parameter count is for neural TTS.

    - A technically substantive thread asks for implementation details behind making a `4.63M`-parameter TTS model work: whether the author started from TTS architecture review papers, used an existing architecture, or built a hybrid design. The key technical interest is how such a small parameter count can still produce usable speech, implying questions around architecture choice, compression, dataset curation, and inference tradeoffs.
    - Several commenters focus on deployment constraints for ultra-small TTS, including whether the model could run on embedded hardware such as an **ESP32 with ML acceleration**. Another practical deployment concern is that even if the model is tiny, the **PyTorch inference stack** can dominate distribution size and complexity, especially for accessibility tooling like NVDA screen reader addons.
    - One commenter with prior NVDA integration experience links concrete projects—[kittentts-nvda](https://github.com/fastfinge/kittentts-nvda), [supertonic-nvda](https://github.com/fastfinge/supertonic-nvda), and a writeup on [AI TTS for screenreaders](https://stuff.interfree.ca/2026/01/05/ai-tts-for-screenreaders.html)—and asks for an **ONNX export or lighter inference pipeline**. They characterize Inflect-Nano as “pretty well balanced between speed and sound,” but note that dependency weight is a major blocker for real-world screen reader integration.


### 3. Local LLM Agent Loops and Persistent Worlds

  - **[Headless screenshot loops let a local 30B agent finish a raytraced FPS demo in pure C](https://www.reddit.com/r/LocalLLaMA/comments/1u89f2q/headless_screenshot_loops_let_a_local_30b_agent/)** (Activity: 321): **The post reports that adding a **headless visual feedback harness**—keyboard/mouse injection plus frame-specific screenshot capture—to a C-only raytraced FPS demo task let both **Claude Code on Opus 4.8** and a local **Qwen3.6 27B** agent iteratively debug rendering/gameplay effects rather than rely on one-shot generation. The key mechanism was agent-controlled screenshot timing: e.g. fire a rocket, capture the impact frame, inspect particles/debris, patch the C code, rebuild, and rerun—effectively a recursive visual debugging loop. The author frames this as a prompting/tooling result rather than a pure model benchmark, noting higher token/runtime cost and disclosing the local agent is their OSS project, [`codehamr`](https://github.com/codehamr/codehamr).** Commenters were mixed on significance: one argued the task may not be very challenging for current models, while another described a similar debugging loop using a custom Python logging harness where agents tail shared logs and add instrumentation until errors are resolved. The broader takeaway from comments aligns with the post: agents improve substantially when given structured observability—screenshots or logs—rather than only compiler/runtime output.

    - The author describes a simple debugging harness where the agent is instructed to use a custom Python `Log()` function instead of `print`, with optional console output and a shared log file. Agents tail the common log, add internal instrumentation, and iterate from observed failures—effectively closing a basic autonomous debug loop that models do not reliably perform without explicit tooling.
    - A commenter reports a similar visual-feedback loop in **Godot**, using a small screenshot helper around `get_viewport().get_texture().get_image()` plus frame waits and region-cropping arguments to reduce token cost. Because their UI is code-driven, the model can verify small UI changes itself from screenshots instead of requiring manual user screenshots, mirroring the post’s “agent can directly test in realtime” approach.
    - One user mentions running the workflow on an **RTX 4090** and seeing a noticeable speed improvement, with `q4_k_m` identified as their preferred quantization tradeoff for local inference quality vs. performance. This suggests the setup is sensitive to GPU throughput and quantization choice, especially for a local `30B`-class model.

  - **[I released a local LLM-powered RPG where generated NPCs, locations, items, and quests persist as in-game objects](https://www.reddit.com/r/LocalLLaMA/comments/1u894z7/i_released_a_local_llmpowered_rpg_where_generated/)** (Activity: 369): **Developer released **InstaNTale**, a local LLM-powered experimental RPG on the [Epic Games Store](https://store.epicgames.com/p/instantale-2cfd4c), where generated **NPCs, locations, items, and quests are persisted as structured game objects** rather than transient chat text. The architecture separates LLM-driven dialogue/narration/situational interpretation/quest progression from deterministic RPG systems such as inventory, equipment, party/combat, and saves; the developer also shared a Japanese [YouTube explanation playlist](https://youtube.com/playlist?list=PLsf4oJwdjJhU8xT4oygJWKjk08I9l7Ezh&si=HB1RcMQ5G5JIzDAB). The dev reports roughly `1,800` first-week EGS sales and a `4.0` store rating, suggesting some market interest despite the prototype/rough-edge framing.** Top comments focused less on implementation and more on distribution: multiple users objected to the game being **Epic Games Store-only** and asked for a Steam release. One technical question asked which local LLM is used and whether models can be swapped, but no answer was included in the provided thread excerpt.

    - Users asked for implementation specifics around model/runtime support: whether the game can target **OpenAI-compatible text-generation endpoints**, **ComfyUI** for image generation, or requires **koboldcpp**; they also asked whether LLM/image models are swappable and which models have been tested to work well.
    - There was interest in modding/extensibility: one commenter specifically asked whether users can modify **system prompts** or write scripts to alter in-game generation behavior, which is particularly relevant given the game persists generated NPCs, locations, items, and quests as objects.





## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Open Video Generation Training and Benchmarks

  - **[Big update to the LTX Trainer: One framework, many conditioning modes](https://www.reddit.com/r/StableDiffusion/comments/1u8c5ob/big_update_to_the_ltx_trainer_one_framework_many/)** (Activity: 990): ****Lightricks** released a major update to the [LTX Trainer](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer) that unifies previously separate T2V/I2V workflows under a config-driven conditioning system, allowing mixed image/video datasets and combinations of modes including T2V, I2V, forward/backward extension, in/outpainting, T2A, audio extension/inpainting, A2V/V2A foley, and IC-LoRA adapters for V2V/A2A/AV2AV. Outputs remain standard `.safetensors` compatible with `ltx-pipelines` and ComfyUI; the default config targets a single `80GB` GPU, with low-VRAM and multi-GPU configs also supported, and docs are available [here](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer/docs). They also released a Claude Code training agent plus new IC-LoRAs in the [LTX-2.3 Creative Lab Hugging Face collection](https://huggingface.co/collections/Lightricks/ltx-23-creative-lab), covering restoration, VFX, relighting/editing, consistency, and subject edits such as colorization, decompression, deblurring, in/outpainting, water simulation, day-to-night, and reference-sheet “Ingredients” conditioning.** The main technical concern in comments was documentation/versioning ambiguity: users noted that everything is labeled **LTX-2** while the release targets **LTX-2.3**, asking whether training methodology and docs are fully applicable across all LTX-2.x models and suggesting a short compatibility disclaimer in the docs.

    - A commenter raised a documentation/versioning concern around resources being labeled **LTX-2** while users may specifically be training **LTX 2.3**, noting that *“they feel like completely different models in use and training.”* They asked whether the same training methodology applies across the `LTX2.x` line and suggested adding a short compatibility disclaimer listing applicable model versions to avoid users assuming the docs are outdated.

  - **[Same running-physics test through Seedance 2.0, Gemini Omni Flash and Kling 3.0, no clean winner](https://www.reddit.com/r/GeminiAI/comments/1u8y5or/same_runningphysics_test_through_seedance_20/)** (Activity: 986): **A user compared **Seedance 2.0**, **Gemini Omni Flash**, and **Kling 3.0 Pro** on the same side-tracking sprinter video prompt, emphasizing gait/weight/fabric realism as a “running physics” stress test. Their qualitative ranking was: **Gemini Omni Flash** best for prompt adherence and body-motion plausibility but with a slightly slow/low-FPS look; **Seedance 2.0** best visual quality/cinematic lighting but less physically accurate; **Kling 3.0 Pro** had strong lighting/frame rate but suffered from false adult-content flags, prompt misreads, and unstable body motion. The linked Reddit video (`v.redd.it/gv406yfrez7h1`) was inaccessible due to a **403 Forbidden** block, so the visual evidence could not be independently verified.** Top comments were mostly skeptical/snarky about the “physics test” framing, implying the benchmark looked more like a sexually motivated or aesthetic comparison than a rigorous evaluation.



### 2. Midjourney Medical Scanner Launch

  - **[I know its an openAI sub, but midjourney just unveiled a fucking full body scanner thats meant to replace MRIs, straight from science fiction - holy shit](https://www.reddit.com/r/OpenAI/comments/1u8uttm/i_know_its_an_openai_sub_but_midjourney_just/)** (Activity: 1127): **The image is a screenshot of a purported **“Midjourney [Hardware 1] Unveiling”** livestream showing a sci‑fi circular full-body scanner concept with blue-lit modules and an anatomical visualization in the center ([image](https://i.redd.it/9hbm5bsziy7h1.jpeg)). In context, the Reddit post claims **Midjourney**, known for image-generation AI, unveiled a device intended to replace MRIs, but the provided comments emphasize that the announcement appears to be **marketing-heavy with no cited research, validation data, regulatory pathway, imaging modality details, or clinical benchmarks**.** Commenters were highly skeptical, with one calling it a *“very, very bad sign”* that the purported medical technology was announced with hype but *“0% evidence”* supporting it. Another noted surprise/concern that it is supposedly from the same Midjourney associated with generative image AI.

    - Commenters highlighted that the announcement reads as **marketing rather than medical-device disclosure**: there are no cited studies, validation datasets, sensitivity/specificity numbers, clinical trial plans, FDA/CE regulatory discussion, or comparisons against MRI/CT/ultrasound benchmarks. One commenter argued that for a purported MRI-replacement technology, the absence of evidence makes it resemble an investor pitch rather than a credible medical imaging proposal.
    - A technically skeptical thread questioned whether **sound-wave-based scanning** can plausibly deliver the anatomical/diagnostic fidelity implied by the announcement. The concern is that ultrasound-style modalities face known limitations around tissue penetration, resolution, operator dependence, acoustic windows, and reconstruction artifacts, so claims of broad “full body scanner” capability would need rigorous validation against existing imaging standards.

  - **[Midjourney Medical](https://www.reddit.com/r/singularity/comments/1u8tjcu/midjourney_medical/)** (Activity: 1033): ****Midjourney** announced a new healthcare division, **Midjourney Medical**, proposing “Ultrasonic CT” / “full body ultrasound”: a water/sound-based whole-body imaging system claimed to complete scans in as little as `60 seconds`, with no ionizing radiation or strong magnetic fields, and ambitions to deploy `50,000` scanners within `6 years` generating `1B` scans/month. The roadmap claims a first San Francisco “Midjourney Spa” opening by end-`2027`, with `10` scanners allegedly capable of more annual body scans than all MRI scanners globally combined; no accessible technical validation, clinical study data, regulatory pathway, or device specifications were provided in the Reddit-accessible material.** Top comments were strongly skeptical, comparing the announcement to **Theranos** and questioning whether an AI image company with no apparent medical-device track record can deliver a novel whole-body ultrasound modality, obtain FDA approval, and operate wellness/imaging centers by `2027`.

    - Commenters raised skepticism about **Midjourney Medical** claiming a novel “Ultrasonic CT” system capable of **whole-body ultrasound scans in ~`60 seconds`**, with performance allegedly superior to MRI, no radiation, and no magnetic fields. The main technical concern was that Midjourney has no demonstrated medical-device, imaging-hardware, clinical-validation, or FDA-approval track record, yet is proposing deployment of `50,000` scanners within `6 years` and a first San Francisco site by `2027`.
    - Several comments compared the announcement to **Theranos**, framing it as a small/non-medical company promising a revolutionary diagnostic platform without public evidence of validation, sensitivity/specificity, regulatory pathway, or clinical utility. The “spa” positioning was viewed as especially concerning because it blends wellness marketing with diagnostic imaging claims, which commenters interpreted as a potential red flag for medical-device overclaiming.

  - **[Midjourney, The Image Generation Company, Just Built the Sequel to the MRI](https://www.reddit.com/r/singularity/comments/1u8tbob/midjourney_the_image_generation_company_just/)** (Activity: 1641): **The post claims **Midjourney** built a “sequel to MRI,” but commenters identify the demonstrated technology as **Ultrasound Tomography**, specifically a Caltech-developed sound-based body-scanning approach described in [Caltech’s article](https://www.caltech.edu/about/news/scanning-the-body-with-sound). Technically, this is not an MRI successor: MRI measures nuclear magnetic properties of hydrogen nuclei to infer tissue/chemical/water composition, while ultrasound tomography reconstructs structures from acoustic propagation/reflection/scattering caused by changes in mechanical properties such as stiffness or elasticity.** Commenters pushed back on the headline as misleading, arguing this is a distinct imaging modality rather than a replacement or sequel to MRI. One commenter highlighted a potential clinical use case—early screening for brain AVMs and cardiac abnormalities in children—based on a personal loss from an undetected AVM rupture.

    - Several commenters pushed back on the “sequel to MRI” framing, arguing the system is better described as **ultrasound tomography**: MRI measures nuclear magnetic properties of hydrogen nuclei and can infer tissue composition/water content, while sound-based tomography primarily reconstructs reflections from acoustic impedance or elasticity changes. One commenter summarized the distinction as *“a tomography version of ultrasound”* rather than a replacement for MRI.
    - A commenter identified the work as **Ultrasound Tomography developed by a Caltech team**, linking Caltech’s writeup: [“Scanning the Body with Sound”](https://www.caltech.edu/about/news/scanning-the-body-with-sound). This places the technique in an existing medical-imaging research lineage rather than purely as a Midjourney-originated image-generation advance.


### 3. Anthropic Governance and AI Market Pressure

  - **[They're demanding Fable to somehow be 100% jailbreak-proof. It's so fucking over.](https://www.reddit.com/r/ClaudeAI/comments/1u8nalg/theyre_demanding_fable_to_somehow_be_100/)** (Activity: 2305): **The image is a screenshot of a **WIRED** post claiming Trump administration officials want **Anthropic** to make **Fable 5**’s guardrails impossible to jailbreak before release, while security experts argue that `100%` jailbreak resistance is not technically achievable. In context, the Reddit title frames this as an unrealistic security requirement for an AI model; the image itself is a news/article preview, not a meme. [Image link](https://i.redd.it/tyrnlpaivw7h1.png)** Commenters compare the demand to requiring cars to cause zero injuries or operating systems to be unhackable, arguing that absolute safety/security guarantees are infeasible. One commenter speculates the requirement may be politically motivated to restrict access or preserve government advantage.

    - Several commenters framed the demand for Fable to be `100%` jailbreak-proof as an impossible security requirement: like an operating system, any sufficiently capable interactive system exposes an attack surface, and *proving the absence* of all jailbreak paths is not generally feasible.
    - One technical analogy compared AI jailbreak guarantees to requiring an automobile release to ensure `zero` injuries or deaths: the critique is that complex systems can be hardened and tested, but absolute safety claims are unrealistic compared with measurable risk reduction, red-teaming, and mitigation.

  - **[World leaders meet with top AI CEOs at G7 summit in France](https://www.reddit.com/r/singularity/comments/1u8fyg6/world_leaders_meet_with_top_ai_ceos_at_g7_summit/)** (Activity: 1247): **At a **G7 working lunch on AI in France**, world leaders met with major AI executives including **OpenAI CEO Sam Altman** and **Anthropic CEO Dario Amodei**, amid reported allied tensions over the **U.S. restricting access to Anthropic’s most advanced models**. The Reddit-provided external video link was inaccessible due to Reddit `403 Forbidden`, so no additional technical details beyond the Bloomberg-sourced post text were available.** Comments were mostly non-technical, noting **Marc Benioff/Salesforce** was also present and joking about Amodei being seated near Macron and away from Trump.


  - **[OpenAI's market share falls below 50%](https://www.reddit.com/r/OpenAI/comments/1u7vjkv/openais_market_share_falls_below_50/)** (Activity: 1555): **The linked chart ([image](https://i.redd.it/s4z9kzbypq7h1.jpeg)) claims **ChatGPT/OpenAI’s AI-chatbot market share fell from the high-80% range in May 2023 to below `50%` by May 2026**, while **Google Gemini** grew into the largest challenger. Smaller shares are attributed to **Claude, Grok, DeepSeek, Perplexity, Meta AI, Microsoft Copilot**, and others, suggesting the market has shifted from a near-monopoly to a more fragmented competitive landscape; no source or methodology is visible in the provided image description, so the exact figures should be treated cautiously.** Commenters framed the decline less as OpenAI failure and more as evidence that the overall chatbot market has expanded significantly since 2023. There was also surprise that **Claude** appears to have such a low share, alongside the general view that *“competition is good for everyone.”

    - Several commenters implicitly questioned the methodology behind the “OpenAI below `50%` market share” claim, asking for the source and noting that relative share may be misleading if the overall LLM market expanded substantially after 2024. For technical readers, the key issue is whether the metric is based on web traffic, paid subscriptions, API usage, revenue, tokens served, or active users, since each would produce very different rankings across **OpenAI**, **Google Gemini**, **Anthropic Claude**, and others.
    - One user described switching from **ChatGPT Plus** to **Gemini** because Google bundles Gemini access with additional **Drive storage** at roughly the same `$20/month` consumer price point, while perceiving little quality difference for casual tasks. The technical implication is that frontier-model differentiation may be less visible for non-coding/general use cases, allowing distribution, bundling, and ecosystem integration to affect adoption as much as benchmark performance.



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.